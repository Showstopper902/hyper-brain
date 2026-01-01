# app/worker.py
import json
import math
import os
import re
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from app.db import get_conn

# ----------------------------
# Config
# ----------------------------

ASSIGNED_WORKER_ID = os.getenv("ASSIGNED_WORKER_ID", os.getenv("WORKER_ID", "hyperbolic-pool"))
EXECUTOR_TOKEN = os.environ.get("EXECUTOR_TOKEN")  # required for executor VMs
BRAIN_URL = os.getenv("BRAIN_URL", "https://hyper-brain.fly.dev")

POLL_SECONDS = int(os.getenv("POLL_SECONDS", "3"))

# Autoscaling rule: 0 backlog => 0 instances; +1 instance per 250 unclaimed RUNNING
BACKLOG_PER_INSTANCE = int(os.getenv("BACKLOG_PER_INSTANCE", "250"))
MAX_NEW_INSTANCES_PER_LOOP = int(os.getenv("MAX_NEW_INSTANCES_PER_LOOP", "2"))  # safety

# VM preferences: 1 job per VM => prefer 1 GPU, fallback 2 then 4
GPU_COUNTS_TRY = [1, 2, 4]

# Where to pull the executor loop script from
EXECUTOR_LOOP_URL = os.getenv(
    "EXECUTOR_LOOP_URL",
    "https://raw.githubusercontent.com/Showstopper902/hyperbolic_project/main/scripts/hyper_executor_loop.sh",
)

# B2 env written onto VMs
B2_SYNC = os.getenv("B2_SYNC", "1")
B2_BUCKET = os.environ.get("B2_BUCKET")
B2_S3_ENDPOINT = os.environ.get("B2_S3_ENDPOINT")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-west-004")

# SSH private key used by Brain to SSH into Hyperbolic instances for bootstrap
SSH_PRIVATE_KEY = os.environ.get("SSH_PRIVATE_KEY")

# Hyperbolic API key for CLI auth inside Fly
HYPERBOLIC_API_KEY = os.environ.get("HYPERBOLIC_API_KEY") or os.environ.get("HYPERBOLIC_API_TOKEN")


def log(msg: str) -> None:
    print(f"[brain-worker] {msg}", flush=True)


# ----------------------------
# Subprocess helpers
# ----------------------------

def run(
    cmd: List[str],
    *,
    input_text: Optional[str] = None,
    check: bool = True,
    env_extra: Optional[Dict[str, str]] = None,
    timeout: Optional[int] = None,
) -> subprocess.CompletedProcess:
    """
    Run a command, always capture stdout/stderr.
    Key fix for Fly: force writable HOME/XDG dirs so CLIs can write config.
    """
    env = os.environ.copy()
    if env_extra:
        env.update({k: v for k, v in env_extra.items() if v is not None})

    env.setdefault("HOME", "/tmp")
    env.setdefault("XDG_CONFIG_HOME", "/tmp/.config")
    env.setdefault("XDG_CACHE_HOME", "/tmp/.cache")

    try:
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
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "Command failed: "
            + shlex.join(cmd)
            + f"\nexit_code={e.returncode}"
            + (f"\nSTDOUT:\n{e.stdout}" if e.stdout else "")
            + (f"\nSTDERR:\n{e.stderr}" if e.stderr else "")
        ) from None


def ensure_hyperbolic_auth() -> None:
    """
    Non-interactive Hyperbolic CLI auth inside Fly.
    We pass key via env + attempt set-key in multiple non-interactive styles.
    """
    if not HYPERBOLIC_API_KEY:
        raise RuntimeError("Missing HYPERBOLIC_API_KEY (or HYPERBOLIC_API_TOKEN) in Fly secrets/env")

    env_extra = {
        "HYPERBOLIC_API_KEY": HYPERBOLIC_API_KEY,
        "HYPERBOLIC_API_TOKEN": HYPERBOLIC_API_KEY,
    }

    # Check status first (non-fatal)
    p = run(["hyperbolic", "auth", "status"], check=False, env_extra=env_extra)
    status_text = (p.stdout + p.stderr).lower()
    if p.returncode == 0 and any(x in status_text for x in ("authenticated", "logged", "valid", "success")):
        return

    errors: List[str] = []

    # stdin style
    p1 = run(["hyperbolic", "auth", "set-key"], input_text=HYPERBOLIC_API_KEY + "\n", check=False, env_extra=env_extra)
    if p1.returncode != 0:
        errors.append(f"set-key(stdin) rc={p1.returncode} stderr={p1.stderr.strip()[:300]}")

    # arg style
    if p1.returncode != 0:
        p2 = run(["hyperbolic", "auth", "set-key", HYPERBOLIC_API_KEY], check=False, env_extra=env_extra)
        if p2.returncode != 0:
            errors.append(f"set-key <key> rc={p2.returncode} stderr={p2.stderr.strip()[:300]}")
        else:
            p1 = p2

    # flag styles
    if p1.returncode != 0:
        for flag in ("--api-key", "--key"):
            p3 = run(["hyperbolic", "auth", "set-key", flag, HYPERBOLIC_API_KEY], check=False, env_extra=env_extra)
            if p3.returncode == 0:
                p1 = p3
                break
            errors.append(f"set-key {flag} rc={p3.returncode} stderr={p3.stderr.strip()[:300]}")

    # Re-check status
    p = run(["hyperbolic", "auth", "status"], check=False, env_extra=env_extra)
    status_text = (p.stdout + p.stderr).lower()
    if p.returncode == 0 and any(x in status_text for x in ("authenticated", "logged", "valid", "success")):
        return

    raise RuntimeError(
        "Hyperbolic CLI auth failed (non-interactive). Details:\n"
        + "\n".join(errors[-8:])
        + (f"\n\nFINAL STATUS STDOUT:\n{p.stdout}" if p.stdout else "")
        + (f"\n\nFINAL STATUS STDERR:\n{p.stderr}" if p.stderr else "")
    )


# ----------------------------
# JSON parsing helpers (FIX)
# ----------------------------

def _extract_json_payload(text: str) -> str:
    """
    Extract the first JSON object/array from a text blob that may include warnings/banners.
    Returns "" if no JSON-looking payload exists.
    """
    if not text:
        return ""
    s = text.strip()
    if not s:
        return ""

    # If it already starts with JSON, use it
    if s[0] in "[{":
        return s

    # Otherwise find first JSON start
    idx_obj = s.find("{")
    idx_arr = s.find("[")
    idxs = [i for i in (idx_obj, idx_arr) if i != -1]
    if not idxs:
        return ""
    start = min(idxs)
    return s[start:]


def _parse_instances_json(stdout: str, stderr: str) -> List[dict]:
    """
    Tolerant parsing:
      - empty => []
      - prefer stdout else stderr
      - strip junk before JSON
      - accept {"instances":[...]} or [...]
    """
    primary = stdout.strip() if stdout and stdout.strip() else ""
    secondary = stderr.strip() if stderr and stderr.strip() else ""

    if not primary and not secondary:
        # Some CLIs output nothing for "no instances"
        return []

    payload = _extract_json_payload(primary) or _extract_json_payload(secondary)
    if not payload:
        # still nothing JSON-like
        head = (primary or secondary)[:400]
        tail = (primary or secondary)[-400:]
        raise RuntimeError(
            "hyperbolic instances --json returned non-JSON output.\n"
            f"HEAD:\n{head}\n\nTAIL:\n{tail}\n"
        )

    try:
        data = json.loads(payload)
    except Exception as e:
        head = payload[:400]
        tail = payload[-400:]
        raise RuntimeError(
            f"Failed to parse hyperbolic JSON: {e}\nHEAD:\n{head}\n\nTAIL:\n{tail}\n"
        ) from None

    if isinstance(data, dict) and isinstance(data.get("instances"), list):
        return data["instances"]
    if isinstance(data, list):
        return data
    return []


# ----------------------------
# Hyperbolic CLI helpers
# ----------------------------

def hyperbolic_instances_json() -> List[dict]:
    ensure_hyperbolic_auth()

    # Don't let run(check=True) hide useful output: we want stdout/stderr always
    p = run(["hyperbolic", "instances", "--json"], check=False, timeout=30)

    if p.returncode != 0:
        raise RuntimeError(
            "hyperbolic instances --json failed\n"
            f"exit_code={p.returncode}\n"
            + (f"STDOUT:\n{p.stdout}\n" if p.stdout else "")
            + (f"STDERR:\n{p.stderr}\n" if p.stderr else "")
        )

    return _parse_instances_json(p.stdout or "", p.stderr or "")


def hyperbolic_rent_vm(gpu_count: int) -> str:
    ensure_hyperbolic_auth()
    cmd = ["hyperbolic", "rent", "ondemand", "--instance-type", "virtual-machine", "--gpu-count", str(gpu_count)]
    p = run(cmd, check=True, timeout=120)

    out = (p.stdout or "").strip()
    if out.startswith("{") or out.startswith("["):
        try:
            j = json.loads(out)
            for k in ("instance_id", "id", "rental_id"):
                if isinstance(j, dict) and j.get(k):
                    return str(j[k])
        except Exception:
            pass

    m = re.search(r"([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})", out, re.I)
    if m:
        return m.group(1)

    inst = hyperbolic_instances_json()
    if inst:
        inst_sorted = sorted(inst, key=lambda x: x.get("created_at", ""), reverse=True)
        iid = inst_sorted[0].get("id") or inst_sorted[0].get("instance_id")
        if iid:
            return str(iid)

    raise RuntimeError(f"Could not determine instance id from rent output.\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")


def hyperbolic_terminate(instance_id: str) -> None:
    ensure_hyperbolic_auth()
    run(["hyperbolic", "terminate", instance_id], check=True, timeout=60)


# ----------------------------
# SSH + bootstrap
# ----------------------------

@dataclass
class SSHInfo:
    user: str
    host: str
    port: int


def _write_ssh_keyfile() -> str:
    if not SSH_PRIVATE_KEY:
        raise RuntimeError("Missing SSH_PRIVATE_KEY in Fly secrets/env")
    fd, path = tempfile.mkstemp(prefix="hyperbolic_ssh_", dir="/tmp")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        f.write(SSH_PRIVATE_KEY.strip() + "\n")
    os.chmod(path, 0o600)
    return path


def _extract_ssh_info(instance: dict) -> Optional[SSHInfo]:
    ssh_cmd = instance.get("ssh_command") or instance.get("sshCommand") or ""
    if isinstance(ssh_cmd, str) and "ssh" in ssh_cmd and "@" in ssh_cmd:
        port = 22
        mport = re.search(r"-p\s+(\d+)", ssh_cmd)
        if mport:
            port = int(mport.group(1))
        muserhost = re.search(r"([a-zA-Z0-9._-]+)@([a-zA-Z0-9.\-]+)", ssh_cmd)
        if muserhost:
            return SSHInfo(user=muserhost.group(1), host=muserhost.group(2), port=port)

    ssh_obj = instance.get("ssh")
    if isinstance(ssh_obj, dict):
        host = ssh_obj.get("host") or ssh_obj.get("ip")
        port = int(ssh_obj.get("port") or 22)
        user = ssh_obj.get("user") or ssh_obj.get("username") or "root"
        if host:
            return SSHInfo(user=str(user), host=str(host), port=port)

    host = instance.get("host") or instance.get("ip") or instance.get("public_ip") or instance.get("publicIp")
    port = int(instance.get("port") or instance.get("ssh_port") or instance.get("sshPort") or 22)
    user = instance.get("user") or instance.get("username") or "root"
    if host:
        return SSHInfo(user=str(user), host=str(host), port=port)

    return None


def ssh_run(ssh: SSHInfo, script: str, *, timeout_s: int = 600) -> None:
    keyfile = _write_ssh_keyfile()
    try:
        cmd = [
            "ssh",
            "-i", keyfile,
            "-p", str(ssh.port),
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            f"{ssh.user}@{ssh.host}",
            "bash", "-lc", script,
        ]
        p = run(cmd, check=False, timeout=timeout_s)
        if p.returncode != 0:
            raise RuntimeError(
                f"SSH bootstrap failed rc={p.returncode}\nHOST={ssh.user}@{ssh.host}:{ssh.port}\n"
                + (f"STDOUT:\n{p.stdout}\n" if p.stdout else "")
                + (f"STDERR:\n{p.stderr}\n" if p.stderr else "")
            )
    finally:
        try:
            os.remove(keyfile)
        except Exception:
            pass


def bootstrap_executor_vm(ssh: SSHInfo) -> None:
    if not EXECUTOR_TOKEN:
        raise RuntimeError("Missing EXECUTOR_TOKEN in Fly secrets/env (needed to write hyper_executor.env on VMs)")
    for k, v in {
        "B2_BUCKET": B2_BUCKET,
        "B2_S3_ENDPOINT": B2_S3_ENDPOINT,
        "AWS_ACCESS_KEY_ID": AWS_ACCESS_KEY_ID,
        "AWS_SECRET_ACCESS_KEY": AWS_SECRET_ACCESS_KEY,
    }.items():
        if not v:
            raise RuntimeError(f"Missing {k} in Fly secrets/env (needed to write /data/secrets/b2.env on VMs)")

    remote = f"""
set -euo pipefail

sudo mkdir -p /data/secrets /data/bin

sudo bash -c 'cat > /data/secrets/b2.env <<EOF
B2_SYNC={B2_SYNC}
B2_BUCKET={B2_BUCKET}
B2_S3_ENDPOINT={B2_S3_ENDPOINT}
AWS_ACCESS_KEY_ID={AWS_ACCESS_KEY_ID}
AWS_SECRET_ACCESS_KEY={AWS_SECRET_ACCESS_KEY}
AWS_DEFAULT_REGION={AWS_DEFAULT_REGION}
EOF'
sudo chmod 600 /data/secrets/b2.env

sudo bash -c 'cat > /data/secrets/hyper_executor.env <<EOF
EXECUTOR_TOKEN="{EXECUTOR_TOKEN}"
BRAIN_URL="{BRAIN_URL}"
ASSIGNED_WORKER_ID="{ASSIGNED_WORKER_ID}"
POLL_SECONDS={POLL_SECONDS}
IDLE_SECONDS=3600
EXECUTOR_ID="exec-$(hostname)"
EOF'
sudo chmod 600 /data/secrets/hyper_executor.env

sudo curl -fsSL "{EXECUTOR_LOOP_URL}" -o /data/bin/hyper_executor_loop.sh
sudo chmod +x /data/bin/hyper_executor_loop.sh

sudo tee /etc/systemd/system/hyper-executor.service >/dev/null <<'EOF'
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
"""
    ssh_run(ssh, remote, timeout_s=900)


# ----------------------------
# DB helpers
# ----------------------------

def promote_queued_jobs(conn, *, limit: int = 500) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH cte AS (
              SELECT job_id
              FROM public.jobs
              WHERE status = 'QUEUED'
              ORDER BY created_at
              LIMIT %s
              FOR UPDATE SKIP LOCKED
            )
            UPDATE public.jobs j
            SET status = 'RUNNING',
                assigned_worker_id = %s,
                started_at = now()
            FROM cte
            WHERE j.job_id = cte.job_id
            RETURNING j.job_id
            """,
            (limit, ASSIGNED_WORKER_ID),
        )
        rows = cur.fetchall()
    conn.commit()
    return len(rows)


def count_unclaimed_running(conn) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*) AS n
            FROM public.jobs
            WHERE status = 'RUNNING'
              AND assigned_worker_id = %s
              AND executor_id IS NULL
            """,
            (ASSIGNED_WORKER_ID,),
        )
        row = cur.fetchone()
    return int(row["n"])


# ----------------------------
# Autoscaler core
# ----------------------------

def desired_instance_count(backlog: int) -> int:
    if backlog <= 0:
        return 0
    return int(math.ceil(backlog / float(BACKLOG_PER_INSTANCE)))


def autoscale_once() -> None:
    with get_conn() as conn:
        promoted = promote_queued_jobs(conn, limit=500)
        backlog = count_unclaimed_running(conn)

    desired = desired_instance_count(backlog)

    instances = hyperbolic_instances_json()
    current = len(instances)

    log(f"promoted={promoted} backlog_unclaimed_running={backlog} desired_instances={desired} current_instances={current}")

    if desired == current:
        return

    if desired < current:
        inst_sorted = sorted(instances, key=lambda x: x.get("created_at", ""))
        to_kill = inst_sorted[: (current - desired)]
        for inst in to_kill:
            iid = inst.get("id") or inst.get("instance_id")
            if not iid:
                continue
            log(f"terminating instance {iid} (scale down)")
            try:
                hyperbolic_terminate(str(iid))
            except Exception as e:
                log(f"terminate failed for {iid}: {e}")
        return

    need = min(desired - current, MAX_NEW_INSTANCES_PER_LOOP)
    for _ in range(need):
        rented_id = None
        last_err = None
        for g in GPU_COUNTS_TRY:
            try:
                log(f"renting VM gpu_count={g}")
                rented_id = hyperbolic_rent_vm(g)
                log(f"rented instance_id={rented_id}")
                break
            except Exception as e:
                last_err = e
                log(f"rent failed gpu_count={g}: {e}")
        if not rented_id:
            raise RuntimeError(f"Failed to rent any VM size. last_err={last_err}")

        time.sleep(3)
        inst_list = hyperbolic_instances_json()
        inst = None
        for x in inst_list:
            if str(x.get("id") or x.get("instance_id")) == str(rented_id):
                inst = x
                break

        if not inst:
            log(f"warning: could not find instance details for {rented_id} in instances list yet")
            continue

        ssh = _extract_ssh_info(inst)
        if not ssh:
            raise RuntimeError(f"Could not extract SSH info from instance JSON for {rented_id}: {inst}")

        log(f"bootstrapping executor on {ssh.user}@{ssh.host}:{ssh.port}")
        bootstrap_executor_vm(ssh)
        log(f"bootstrap complete for instance_id={rented_id}")


def main() -> None:
    log(f"worker start: ASSIGNED_WORKER_ID={ASSIGNED_WORKER_ID} POLL_SECONDS={POLL_SECONDS} BACKLOG_PER_INSTANCE={BACKLOG_PER_INSTANCE}")
    while True:
        try:
            autoscale_once()
        except Exception as e:
            log(f"autoscaler error: {e}")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
