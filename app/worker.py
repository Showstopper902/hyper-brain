# app/worker.py
import json
import os
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
SCALE_DOWN_GRACE_SECONDS = int(os.environ.get("SCALE_DOWN_GRACE_SECONDS", "120"))

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


def log(msg: str) -> None:
    print(f"[brain-worker] {msg}", flush=True)


def _hyper_env() -> Dict[str, str]:
    """
    Make Hyperbolic CLI stable in Fly:
    - ensure it can write auth/config (HOME)
    - keep env key present
    """
    env = dict(os.environ)
    env.setdefault("HOME", "/tmp")
    env.setdefault("XDG_CONFIG_HOME", "/tmp/.config")
    env.setdefault("XDG_CACHE_HOME", "/tmp/.cache")
    if HYPERBOLIC_API_KEY:
        env["HYPERBOLIC_API_KEY"] = HYPERBOLIC_API_KEY
    return env


def run(
    cmd: List[str],
    *,
    input_text: Optional[str] = None,
    check: bool = True,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
        env=env,
    )


def _one_line(s: str, limit: int = 260) -> str:
    """
    Keep diagnostics on a single log line so `grep` doesn't hide important details.
    """
    if not s:
        return ""
    s = s.replace("\r", "\\r").replace("\n", "\\n")
    if len(s) <= limit:
        return s
    return s[:limit] + "...(trunc)"


def _extract_first_json(text: str) -> Any:
    """
    Hyperbolic CLI sometimes prints non-JSON before/after JSON.
    This finds the first substring that parses as valid JSON.
    """
    t = (text or "").strip()
    if not t:
        raise ValueError("empty output")

    # Fast path
    try:
        return json.loads(t)
    except Exception:
        pass

    # Try from each potential JSON start
    starts = [i for i, ch in enumerate(t) if ch in "{["]
    for i in starts[:50]:  # avoid pathological scans
        candidate = t[i:].strip()
        try:
            return json.loads(candidate)
        except Exception:
            continue

    raise ValueError("no JSON object/array found in output")


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
    """RUNNING jobs with executor_id IS NULL (for this assigned_worker_id)."""
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
    if not HYPERBOLIC_API_KEY:
        raise RuntimeError("Missing HYPERBOLIC_API_KEY in Fly secrets/env")

    env = _hyper_env()

    # If status already OK, don't spam set-key
    p = run(["hyperbolic", "auth", "status"], check=False, env=env)
    status_text = (p.stdout + "\n" + p.stderr).lower()
    if p.returncode == 0 and ("logged" in status_text or "authenticated" in status_text):
        return

    # Non-interactive attempt: feed key on stdin
    p2 = run(["hyperbolic", "auth", "set-key"], input_text=HYPERBOLIC_API_KEY + "\n", check=False, env=env)
    if p2.returncode != 0:
        # Raise with useful info (single line)
        raise RuntimeError(
            "hyperbolic auth set-key failed "
            f"(rc={p2.returncode}) "
            f"stdout={_one_line(p2.stdout)} "
            f"stderr={_one_line(p2.stderr)}"
        )

    # Re-check
    run(["hyperbolic", "auth", "status"], check=False, env=env)


def hyperbolic_instances_json() -> List[Dict[str, Any]]:
    env = _hyper_env()
    p = run(["hyperbolic", "instances", "--json"], check=False, env=env)

    raw_stdout = p.stdout or ""
    raw_stderr = p.stderr or ""

    if p.returncode != 0:
        raise RuntimeError(
            "hyperbolic instances --json failed "
            f"(rc={p.returncode}) "
            f"stdout={_one_line(raw_stdout)} "
            f"stderr={_one_line(raw_stderr)}"
        )

    # Some CLIs print JSON to stderr or add warnings in stdout; try both and combined.
    candidates = [
        raw_stdout.strip(),
        raw_stderr.strip(),
        (raw_stdout + "\n" + raw_stderr).strip(),
    ]

    last_err = None
    for text in candidates:
        if not text:
            continue
        try:
            j = _extract_first_json(text)
            # schema could be {"instances":[...]} or [...]
            if isinstance(j, dict) and "instances" in j and isinstance(j["instances"], list):
                return list(j["instances"])
            if isinstance(j, list):
                return j
            # Sometimes returns a dict not using "instances" key; treat as empty but not fatal.
            return []
        except Exception as e:
            last_err = e

    raise RuntimeError(
        "hyperbolic instances --json returned non-JSON output. "
        f"stdout={_one_line(raw_stdout)} "
        f"stderr={_one_line(raw_stderr)} "
        f"parse_err={last_err}"
    )


def extract_id_and_ssh(instance_obj: Dict[str, Any]) -> Tuple[str, str]:
    instance_id = (
        instance_obj.get("instance_id")
        or instance_obj.get("id")
        or instance_obj.get("instanceId")
        or ""
    )
    ssh_cmd = (
        instance_obj.get("ssh_command")
        or instance_obj.get("sshCommand")
        or instance_obj.get("ssh")
        or instance_obj.get("sshConnection")
        or ""
    )
    if not instance_id:
        raise RuntimeError(f"Could not find instance id in: {instance_obj}")
    if not ssh_cmd:
        if isinstance(instance_obj.get("ssh_details"), dict):
            ssh_cmd = instance_obj["ssh_details"].get("ssh_command", "")
    if not ssh_cmd:
        raise RuntimeError(f"Could not find ssh command for instance {instance_id} in: {instance_obj}")
    return instance_id, ssh_cmd


def hyperbolic_rent_vm(gpu_count: int) -> Tuple[str, str]:
    before = {extract_id_and_ssh(x)[0] for x in hyperbolic_instances_json()}

    # Minimal flags per `hyperbolic rent ondemand --help`
    env = _hyper_env()
    p = run(
        ["hyperbolic", "rent", "ondemand", "--instance-type", "virtual-machine", "--gpu-count", str(gpu_count)],
        check=False,
        env=env,
    )
    if p.returncode != 0:
        raise RuntimeError(
            "hyperbolic rent ondemand failed "
            f"(rc={p.returncode}) "
            f"stdout={_one_line(p.stdout)} "
            f"stderr={_one_line(p.stderr)}"
        )

    # Poll for new instance to appear
    deadline = time.time() + 180
    while time.time() < deadline:
        now = hyperbolic_instances_json()
        for obj in now:
            iid, ssh = extract_id_and_ssh(obj)
            if iid not in before:
                return iid, ssh
        time.sleep(2)

    raise RuntimeError("Timed out waiting for new Hyperbolic instance to appear after rent")


def hyperbolic_terminate(instance_id: str) -> None:
    env = _hyper_env()
    run(["hyperbolic", "terminate", instance_id], check=False, env=env)


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
        "-i", key_path,
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=3",
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
'"""

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

    raise RuntimeError(f"Failed to bootstrap VM {instance_id} via SSH. Last output: {_one_line(last_err, 800)}")


# -------------------------
# Autoscaler
# -------------------------
def desired_vm_count(backlog: int) -> int:
    if backlog <= 0:
        return 0
    return (backlog + BACKLOG_PER_VM - 1) // BACKLOG_PER_VM


def maybe_scale() -> None:
    # leader lock (safe if you scale Fly worker replicas later)
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
                    grace_ok = (age >= SCALE_DOWN_GRACE_SECONDS)

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
