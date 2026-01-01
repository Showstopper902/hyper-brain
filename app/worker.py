# app/worker.py
import json
import os
import re
import shlex
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

from app.db import get_conn

# ---- existing behavior (dispatcher) ----
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "3"))
ASSIGNED_WORKER_ID = os.environ.get("ASSIGNED_WORKER_ID", "hyperbolic-pool")

# ---- autoscaler knobs ----
BACKLOG_PER_VM = int(os.environ.get("BACKLOG_PER_VM", "250"))
SCALE_CHECK_SECONDS = int(os.environ.get("SCALE_CHECK_SECONDS", "10"))
SCALE_DOWN_GRACE_SECONDS = int(os.environ.get("SCALE_DOWN_GRACE_SECONDS", "120"))  # keep VM briefly after backlog clears

# Hyperbolic requirements
HYPERBOLIC_API_KEY = os.environ.get("HYPERBOLIC_API_KEY", "")
SSH_PRIVATE_KEY = os.environ.get("SSH_PRIVATE_KEY", "")
EXECUTOR_TOKEN = os.environ.get("EXECUTOR_TOKEN", "")  # brain->executor auth token
BRAIN_URL = os.environ.get("BRAIN_URL", "https://hyper-brain.fly.dev")

# B2 secrets to write onto VM (from Fly env)
B2_BUCKET = os.environ.get("B2_BUCKET", "hyperbolic-project-data")
B2_S3_ENDPOINT = os.environ.get("B2_S3_ENDPOINT", "https://s3.us-west-004.backblazeb2.com")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
AWS_DEFAULT_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-west-004")
B2_SYNC = os.environ.get("B2_SYNC", "1")

# Where the executor loop is fetched from
EXECUTOR_LOOP_RAW_URL = os.environ.get(
    "EXECUTOR_LOOP_RAW_URL",
    "https://raw.githubusercontent.com/Showstopper902/hyperbolic_project/main/scripts/hyper_executor_loop.sh",
)

# Keep Hyperbolic CLI config in a deterministic writable place (independent of user HOME)
HYPERBOLIC_CLI_HOME = os.environ.get("HYPERBOLIC_CLI_HOME", "/tmp/hyperbolic-cli-home")


def log(msg: str) -> None:
    print(f"[brain-worker] {msg}", flush=True)


def _hyperbolic_env() -> Dict[str, str]:
    """
    Make Hyperbolic CLI authentication/config stable:
    - Do NOT rely on the shell user HOME on Fly (can differ between ssh sessions/processes).
    - Force XDG/HOME into a consistent directory.
    """
    os.makedirs(HYPERBOLIC_CLI_HOME, exist_ok=True)
    env = dict(os.environ)
    env["HOME"] = HYPERBOLIC_CLI_HOME
    env["XDG_CONFIG_HOME"] = HYPERBOLIC_CLI_HOME
    env["XDG_CACHE_HOME"] = HYPERBOLIC_CLI_HOME
    env["XDG_DATA_HOME"] = HYPERBOLIC_CLI_HOME
    return env


def run(
    cmd: List[str],
    *,
    input_text: Optional[str] = None,
    check: bool = True,
    env: Optional[Dict[str, str]] = None,
    timeout: int = 120,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
        env=env,
        timeout=timeout,
    )


# -------------------------
# DB helpers
# -------------------------
def dispatch_one_job() -> bool:
    """QUEUED -> RUNNING (one job at a time)."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                update public.jobs
                   set status = 'RUNNING',
                       assigned_worker_id = %s,
                       started_at = coalesce(started_at, now())
                 where job_id = (
                    select job_id
                      from public.jobs
                     where status = 'QUEUED'
                     order by created_at asc
                     for update skip locked
                     limit 1
                 )
                returning job_id
                """,
                (ASSIGNED_WORKER_ID,),
            )
            row = cur.fetchone()
            conn.commit()
            return bool(row)


def backlog_running_unclaimed() -> int:
    """RUNNING jobs with executor_id IS NULL (your backlog metric)."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                select count(*)::int as n
                  from public.jobs
                 where status = 'RUNNING'
                   and assigned_worker_id = %s
                   and executor_id is null
                """,
                (ASSIGNED_WORKER_ID,),
            )
            row = cur.fetchone()
            return int(row["n"])


def list_managed_instances() -> List[Dict[str, Any]]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                select instance_id, executor_id, gpu_count, status, ssh_command, created_at, last_used_at
                  from public.hyper_instances
                 where status in ('BOOTING','READY')
                 order by created_at asc
                """
            )
            return list(cur.fetchall())


def upsert_instance(instance_id: str, executor_id: str, gpu_count: int, status: str, ssh_command: str) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into public.hyper_instances (instance_id, executor_id, gpu_count, status, ssh_command)
                values (%s,%s,%s,%s,%s)
                on conflict (instance_id) do update
                  set executor_id = excluded.executor_id,
                      gpu_count = excluded.gpu_count,
                      status = excluded.status,
                      ssh_command = excluded.ssh_command
                """,
                (instance_id, executor_id, gpu_count, status, ssh_command),
            )
            conn.commit()


def mark_instance_status(instance_id: str, status: str) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "update public.hyper_instances set status=%s where instance_id=%s",
                (status, instance_id),
            )
            conn.commit()


def mark_instance_used(executor_id: str) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "update public.hyper_instances set last_used_at=now() where executor_id=%s",
                (executor_id,),
            )
            conn.commit()


def instance_has_running_job(executor_id: str) -> bool:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                select 1
                  from public.jobs
                 where status='RUNNING'
                   and executor_id=%s
                 limit 1
                """,
                (executor_id,),
            )
            return cur.fetchone() is not None


# -------------------------
# Hyperbolic CLI helpers
# -------------------------
def ensure_hyperbolic_auth() -> None:
    """
    FIX:
      hyperbolic auth set-key requires the API key as an ARGUMENT (not stdin).
    Also:
      make config location deterministic with HYPERBOLIC_CLI_HOME.
    """
    if not HYPERBOLIC_API_KEY:
        raise RuntimeError("Missing HYPERBOLIC_API_KEY in Fly secrets/env")

    env = _hyperbolic_env()

    # If already authenticated, great.
    p = run(["hyperbolic", "auth", "status"], check=False, env=env, timeout=30)
    status_txt = (p.stdout + p.stderr).lower()
    if p.returncode == 0 and "authenticated" in status_txt:
        return

    # Set key (argument-based)
    p2 = run(["hyperbolic", "auth", "set-key", HYPERBOLIC_API_KEY], check=False, env=env, timeout=30)
    if p2.returncode != 0:
        raise RuntimeError(
            "hyperbolic auth set-key failed.\n"
            f"STDOUT:\n{p2.stdout}\nSTDERR:\n{p2.stderr}"
        )

    # Verify
    p3 = run(["hyperbolic", "auth", "status"], check=False, env=env, timeout=30)
    if p3.returncode != 0 or "authenticated" not in (p3.stdout + p3.stderr).lower():
        raise RuntimeError(
            "hyperbolic auth status still not authenticated after set-key.\n"
            f"STDOUT:\n{p3.stdout}\nSTDERR:\n{p3.stderr}"
        )


def _parse_rent_output(stdout: str, stderr: str) -> Tuple[str, str]:
    """
    Hyperbolic CLI output isn't guaranteed stable, so parse defensively.
    We want:
      - instance id
      - an ssh command string (starts with 'ssh ')
    """
    blob = (stdout or "") + "\n" + (stderr or "")

    # SSH command: pick the first line that looks like an ssh command
    ssh_cmd = ""
    for line in blob.splitlines():
        s = line.strip()
        if s.startswith("ssh "):
            ssh_cmd = s
            break

    # Instance id: look for common patterns
    instance_id = ""
    m = re.search(r"(?i)\binstance\s*id\b\s*[:=]\s*([a-z0-9\-]+)", blob)
    if m:
        instance_id = m.group(1).strip()
    else:
        # fallback: sometimes they just print an id on its own line
        m2 = re.search(r"(?i)\bid\b\s*[:=]\s*([a-z0-9\-]+)", blob)
        if m2:
            instance_id = m2.group(1).strip()

    if not instance_id or not ssh_cmd:
        raise RuntimeError(
            "Could not parse hyperbolic rent output for instance_id + ssh command.\n"
            f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )

    return instance_id, ssh_cmd


def hyperbolic_rent_vm(gpu_count: int) -> Tuple[str, str]:
    """
    IMPORTANT CHANGE:
      Do NOT depend on `hyperbolic instances --json` (it has been failing / non-JSON / 404).
      Instead parse the rent output directly.
    """
    env = _hyperbolic_env()

    p = run(
        [
            "hyperbolic",
            "rent",
            "ondemand",
            "--instance-type",
            "virtual-machine",
            "--gpu-count",
            str(gpu_count),
        ],
        check=False,
        env=env,
        timeout=180,
    )
    if p.returncode != 0:
        raise RuntimeError(
            "hyperbolic rent ondemand failed.\n"
            f"STDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
        )

    return _parse_rent_output(p.stdout, p.stderr)


def hyperbolic_terminate(instance_id: str) -> None:
    env = _hyperbolic_env()
    # don't hard-fail on terminate
    run(["hyperbolic", "terminate", instance_id], check=False, env=env, timeout=60)


# -------------------------
# SSH bootstrap
# -------------------------
def write_temp_ssh_key() -> str:
    if not SSH_PRIVATE_KEY:
        raise RuntimeError("Missing SSH_PRIVATE_KEY in Fly secrets/env")
    tf = tempfile.NamedTemporaryFile(mode="w", delete=False)
    tf.write(SSH_PRIVATE_KEY.strip() + "\n")
    tf.flush()
    tf.close()
    os.chmod(tf.name, 0o600)
    return tf.name


def build_ssh_command(ssh_command_str: str, key_path: str, remote_cmd: str) -> List[str]:
    parts = shlex.split(ssh_command_str)
    if not parts or parts[0] != "ssh":
        raise RuntimeError(f"Unexpected ssh_command format: {ssh_command_str}")

    # strip any existing -i <key>
    cleaned: List[str] = []
    skip_next = False
    for tok in parts:
        if skip_next:
            skip_next = False
            continue
        if tok == "-i":
            skip_next = True
            continue
        cleaned.append(tok)

    cleaned += [
        "-i",
        key_path,
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
    ]
    cleaned.append(remote_cmd)
    return cleaned


def bootstrap_executor_vm(instance_id: str, ssh_command_str: str, executor_id: str) -> None:
    key_path = write_temp_ssh_key()

    remote_script = f"""bash -lc '
set -euo pipefail
sudo mkdir -p /data/secrets /data/bin

sudo bash -c "cat > /data/secrets/b2.env <<EOF
B2_SYNC={B2_SYNC}
B2_BUCKET={B2_BUCKET}
B2_S3_ENDPOINT={B2_S3_ENDPOINT}
AWS_ACCESS_KEY_ID={AWS_ACCESS_KEY_ID}
AWS_SECRET_ACCESS_KEY={AWS_SECRET_ACCESS_KEY}
AWS_DEFAULT_REGION={AWS_DEFAULT_REGION}
EOF"
sudo chmod 600 /data/secrets/b2.env

sudo bash -c "cat > /data/secrets/hyper_executor.env <<EOF
EXECUTOR_TOKEN=\\"{EXECUTOR_TOKEN}\\"
BRAIN_URL=\\"{BRAIN_URL}\\"
ASSIGNED_WORKER_ID=\\"{ASSIGNED_WORKER_ID}\\"
POLL_SECONDS=3
IDLE_SECONDS=3600
EXECUTOR_ID=\\"{executor_id}\\"
EOF"
sudo chmod 600 /data/secrets/hyper_executor.env

sudo curl -fsSL "{EXECUTOR_LOOP_RAW_URL}" -o /data/bin/hyper_executor_loop.sh
sudo chmod +x /data/bin/hyper_executor_loop.sh

sudo tee /etc/systemd/system/hyper-executor.service >/dev/null <<EOF
[Unit]
Description=Hyperbolic GPU Executor Loop
After=network-online.target docker.service
Wants=network-online.target

[Service]
Type=simple
EnvironmentFile=-/data/secrets/hyper_executor.env
ExecStart=/data/bin/hyper_executor_loop.sh
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now hyper-executor.service
sudo systemctl status hyper-executor.service --no-pager || true
'
"""

    cmd = build_ssh_command(ssh_command_str, key_path, remote_script)

    deadline = time.time() + 240
    last_err = ""
    while time.time() < deadline:
        p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if p.returncode == 0:
            log(f"bootstrapped VM {instance_id} executor_id={executor_id}")
            return
        last_err = (p.stdout + "\n" + p.stderr).strip()
        time.sleep(5)

    raise RuntimeError(f"Failed to bootstrap VM {instance_id} via SSH.\nLast output:\n{last_err}")


# -------------------------
# Autoscaler
# -------------------------
def desired_vm_count(backlog: int) -> int:
    if backlog <= 0:
        return 0
    return (backlog + BACKLOG_PER_VM - 1) // BACKLOG_PER_VM


def maybe_scale() -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("select pg_try_advisory_lock(hashtext('hyper_autoscaler')) as got")
            got = cur.fetchone()["got"]
            conn.commit()
    if not got:
        return

    try:
        ensure_hyperbolic_auth()

        backlog = backlog_running_unclaimed()
        want = desired_vm_count(backlog)
        managed = list_managed_instances()
        have = len(managed)

        log(f"backlog={backlog} want_vms={want} have_vms={have}")

        # Scale up
        if want > have:
            to_add = want - have
            for _ in range(to_add):
                log("renting 1x GPU VM on Hyperbolic...")
                iid, ssh = hyperbolic_rent_vm(gpu_count=1)
                exec_id = f"exec-{iid}"
                upsert_instance(iid, exec_id, 1, "BOOTING", ssh)
                bootstrap_executor_vm(iid, ssh, exec_id)
                mark_instance_status(iid, "READY")

        # Scale down (conservative)
        elif want < have:
            to_remove = have - want
            for inst in managed:
                if to_remove <= 0:
                    break

                iid = inst["instance_id"]
                exec_id = inst["executor_id"]

                if instance_has_running_job(exec_id):
                    continue

                last_used = inst.get("last_used_at")
                grace_ok = True
                if last_used is not None:
                    with get_conn() as conn:
                        with conn.cursor() as cur:
                            cur.execute("select extract(epoch from (now() - %s))::int as age", (last_used,))
                            age = int(cur.fetchone()["age"])
                            conn.commit()
                    grace_ok = age >= SCALE_DOWN_GRACE_SECONDS

                if not grace_ok:
                    continue

                log(f"terminating instance {iid} (exec_id={exec_id})")
                mark_instance_status(iid, "TERMINATING")
                hyperbolic_terminate(iid)
                mark_instance_status(iid, "TERMINATED")
                to_remove -= 1

    finally:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("select pg_advisory_unlock(hashtext('hyper_autoscaler'))")
                conn.commit()


def main() -> None:
    if not EXECUTOR_TOKEN:
        raise RuntimeError("Missing EXECUTOR_TOKEN in Fly secrets/env (brain<->executor token)")
    if not (AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY):
        raise RuntimeError("Missing AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY in Fly secrets/env")

    last_scale = 0.0

    while True:
        did = dispatch_one_job()
        if not did:
            time.sleep(POLL_SECONDS)

        now = time.time()
        if now - last_scale >= SCALE_CHECK_SECONDS:
            try:
                maybe_scale()
            except Exception as e:
                log(f"autoscaler error: {e}")
            last_scale = now


if __name__ == "__main__":
    main()
