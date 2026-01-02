# app/worker.py
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import textwrap
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .db import get_conn


LOG_PREFIX = "[brain-worker]"


def log(msg: str) -> None:
    print(f"{LOG_PREFIX} {msg}", flush=True)


# ---------- Hyperbolic CLI helpers ----------

HYPERBOLIC_API_KEY = os.getenv("HYPERBOLIC_API_KEY", "").strip()

# Where we force hyperbolic-cli to store its config on Fly
HYPERBOLIC_CLI_HOME = os.getenv("HYPERBOLIC_CLI_HOME", "/tmp/hyperbolic-cli-home").strip()

HYPERBOLIC_SSH_USER = os.getenv("HYPERBOLIC_SSH_USER", "root").strip()
SSH_PRIVATE_KEY = os.getenv("SSH_PRIVATE_KEY", "").strip()

# Executor/brain config passed down to VMs (bootstrap later; safe to keep here)
EXECUTOR_TOKEN = os.getenv("EXECUTOR_TOKEN", "").strip()
BRAIN_URL = os.getenv("BRAIN_URL", "https://hyper-brain.fly.dev").strip()
ASSIGNED_WORKER_ID = os.getenv("ASSIGNED_WORKER_ID", "hyperbolic-pool").strip()

# B2 env to place onto the VM
B2_BUCKET = os.getenv("B2_BUCKET", "").strip()
B2_S3_ENDPOINT = os.getenv("B2_S3_ENDPOINT", "").strip()
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "").strip()
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "").strip()
B2_SYNC = os.getenv("B2_SYNC", "1").strip()

# Your repo script location on the VM
EXECUTOR_LOOP_URL = os.getenv(
    "EXECUTOR_LOOP_URL",
    "https://raw.githubusercontent.com/Showstopper902/hyperbolic_project/main/scripts/hyper_executor_loop.sh",
).strip()

# Scaling rule params
BACKLOG_PER_VM = int(os.getenv("BACKLOG_PER_VM", "250"))
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "3"))
SCALE_POLL_SECONDS = int(os.getenv("SCALE_POLL_SECONDS", "10"))  # autoscaler loop interval

# Hyperbolic rent settings (your constraints)
GPU_COUNTS_PREFERENCE = [1, 2, 4]  # try 1 then 2 then 4; but we rent 1 VM at a time here


def _hyper_env() -> Dict[str, str]:
    """
    Force hyperbolic-cli to use a deterministic HOME so config file exists
    for the worker process (Fly machines/users vary).
    """
    env = dict(os.environ)
    env["HOME"] = HYPERBOLIC_CLI_HOME
    env["XDG_CONFIG_HOME"] = HYPERBOLIC_CLI_HOME
    env["XDG_CACHE_HOME"] = HYPERBOLIC_CLI_HOME
    os.makedirs(HYPERBOLIC_CLI_HOME, exist_ok=True)
    return env


def run_cmd(
    args: List[str],
    *,
    env: Optional[Dict[str, str]] = None,
    input_text: Optional[str] = None,
    timeout: int = 120,
) -> Tuple[int, str, str]:
    p = subprocess.run(
        args,
        input=input_text,
        text=True,
        capture_output=True,
        env=env,
        timeout=timeout,
    )
    return p.returncode, p.stdout or "", p.stderr or ""


def ensure_hyperbolic_auth() -> None:
    """
    Always ensure the hyperbolic-cli config exists and has the API key.
    This avoids 'config file not found' across Fly machines/users.
    """
    if not HYPERBOLIC_API_KEY:
        raise RuntimeError("Missing HYPERBOLIC_API_KEY in environment (Fly secret).")

    env = _hyper_env()

    # Set-key MUST be passed as an argument (not stdin)
    rc, out, err = run_cmd(["hyperbolic", "auth", "set-key", HYPERBOLIC_API_KEY], env=env, timeout=60)

    if rc != 0:
        raise RuntimeError(
            "hyperbolic auth set-key failed. "
            f"rc={rc} stdout_head={out[:200]!r} stderr_head={err[:200]!r}"
        )

    # Optional: verify
    rc2, out2, err2 = run_cmd(["hyperbolic", "auth", "status"], env=env, timeout=30)
    if rc2 != 0:
        raise RuntimeError(
            "hyperbolic auth status failed after set-key. "
            f"rc={rc2} stdout_head={out2[:200]!r} stderr_head={err2[:200]!r}"
        )


def _extract_json_from_output(s: str) -> Optional[Any]:
    """
    Try to parse JSON even if the CLI prints extra text.
    """
    s = (s or "").strip()
    if not s:
        return None
    # quick path
    if s.startswith("{") or s.startswith("["):
        try:
            return json.loads(s)
        except Exception:
            pass

    # try to locate first json object/array substring
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = s.find(start_char)
        end = s.rfind(end_char)
        if 0 <= start < end:
            try:
                return json.loads(s[start : end + 1])
            except Exception:
                continue
    return None


# ---------- DB helpers / backlog ----------

def ensure_tables() -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS hyper_instances (
                    instance_id TEXT PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    terminated_at TIMESTAMPTZ NULL,
                    last_note TEXT NULL
                );
                """
            )
        conn.commit()


def get_backlog_count() -> int:
    """
    Your rule: backlog = RUNNING jobs where executor_id IS NULL.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM jobs
                WHERE status = 'RUNNING'
                  AND executor_id IS NULL;
                """
            )
            (n,) = cur.fetchone()
            return int(n)


def get_tracked_live_instances() -> List[str]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT instance_id
                FROM hyper_instances
                WHERE terminated_at IS NULL
                ORDER BY created_at ASC;
                """
            )
            return [r[0] for r in cur.fetchall()]


def mark_instance_created(instance_id: str, note: str = "") -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO hyper_instances (instance_id, last_note)
                VALUES (%s, %s)
                ON CONFLICT (instance_id) DO UPDATE
                SET last_note = EXCLUDED.last_note;
                """,
                (instance_id, note[:500]),
            )
        conn.commit()


def mark_instance_terminated(instance_id: str, note: str = "") -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE hyper_instances
                SET terminated_at = now(),
                    last_note = %s
                WHERE instance_id = %s;
                """,
                (note[:500], instance_id),
            )
        conn.commit()


# ---------- Hyperbolic instance operations ----------

_INSTANCE_ID_HEX_RE = re.compile(r"\b[0-9a-f]{16}\b", re.IGNORECASE)
_INSTANCE_ID_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE
)


def parse_instance_id_from_rent_output(stdout: str, stderr: str) -> Optional[str]:
    blob = "\n".join([stdout or "", stderr or ""])
    # Prefer UUID if present
    m = _INSTANCE_ID_UUID_RE.search(blob)
    if m:
        return m.group(0)
    m2 = _INSTANCE_ID_HEX_RE.search(blob)
    if m2:
        return m2.group(0)
    return None


def rent_vm(gpu_count: int = 1) -> str:
    ensure_hyperbolic_auth()
    env = _hyper_env()

    args = [
        "hyperbolic",
        "rent",
        "ondemand",
        "--instance-type",
        "virtual-machine",
        "--gpu-count",
        str(gpu_count),
    ]

    rc, out, err = run_cmd(args, env=env, timeout=300)
    if rc != 0:
        raise RuntimeError(
            "hyperbolic rent failed. "
            f"rc={rc} stdout_head={out[:300]!r} stderr_head={err[:300]!r}"
        )

    instance_id = parse_instance_id_from_rent_output(out, err)
    if not instance_id:
        # If the rent output is JSON-ish, try that too
        j = _extract_json_from_output(out)
        if isinstance(j, dict):
            for k in ["instance_id", "id"]:
                if j.get(k):
                    instance_id = str(j[k])
                    break

    if not instance_id:
        raise RuntimeError(
            "Could not determine instance_id after rent. "
            f"rent_stdout_head={out[:300]!r} rent_stderr_head={err[:300]!r}"
        )

    return instance_id


def terminate_vm(instance_id: str) -> None:
    ensure_hyperbolic_auth()
    env = _hyper_env()
    rc, out, err = run_cmd(["hyperbolic", "terminate", instance_id], env=env, timeout=180)
    if rc != 0:
        raise RuntimeError(
            "hyperbolic terminate failed. "
            f"id={instance_id} rc={rc} stdout_head={out[:300]!r} stderr_head={err[:300]!r}"
        )


# ---------- Optional: Bootstrap VM over SSH ----------
# (This is ready when you want to turn it on; you can keep it enabled now too.)

def _write_private_key_to_file() -> str:
    if not SSH_PRIVATE_KEY:
        raise RuntimeError("Missing SSH_PRIVATE_KEY in environment (Fly secret).")

    fd, path = tempfile.mkstemp(prefix="hyperbolic_ssh_", text=True)
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        f.write(SSH_PRIVATE_KEY.strip() + "\n")
    os.chmod(path, 0o600)
    return path


def bootstrap_vm_over_ssh(host: str, *, port: int = 22, user: str = "root") -> None:
    """
    If/when you have the VM's reachable host/ip, this will install the executor loop + systemd unit.
    NOTE: hyperbolic-cli rent output format varies; wire host discovery when ready.
    """
    key_path = _write_private_key_to_file()

    # Build remote bootstrap script
    remote = textwrap.dedent(
        f"""
        set -euo pipefail

        mkdir -p /data/secrets /data/bin

        cat > /data/secrets/b2.env <<'EOF'
        B2_SYNC={B2_SYNC}
        B2_BUCKET={B2_BUCKET}
        B2_S3_ENDPOINT={B2_S3_ENDPOINT}
        AWS_ACCESS_KEY_ID={AWS_ACCESS_KEY_ID}
        AWS_SECRET_ACCESS_KEY={AWS_SECRET_ACCESS_KEY}
        AWS_DEFAULT_REGION={AWS_DEFAULT_REGION}
        EOF
        chmod 600 /data/secrets/b2.env

        cat > /data/secrets/hyper_executor.env <<'EOF'
        EXECUTOR_TOKEN="{EXECUTOR_TOKEN}"
        BRAIN_URL="{BRAIN_URL}"
        ASSIGNED_WORKER_ID="{ASSIGNED_WORKER_ID}"
        POLL_SECONDS=3
        IDLE_SECONDS=3600
        EXECUTOR_ID="exec-$(hostname)"
        EOF
        chmod 600 /data/secrets/hyper_executor.env

        curl -fsSL "{EXECUTOR_LOOP_URL}" -o /data/bin/hyper_executor_loop.sh
        chmod +x /data/bin/hyper_executor_loop.sh

        cat > /etc/systemd/system/hyper-executor.service <<'EOF'
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

        systemctl daemon-reload
        systemctl enable --now hyper-executor.service
        systemctl restart hyper-executor.service
        """
    ).strip()

    ssh_cmd = [
        "ssh",
        "-i",
        key_path,
        "-p",
        str(port),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        f"{user}@{host}",
        "bash",
        "-lc",
        remote,
    ]

    rc, out, err = run_cmd(ssh_cmd, timeout=600)
    try:
        os.remove(key_path)
    except Exception:
        pass

    if rc != 0:
        raise RuntimeError(
            "VM bootstrap over SSH failed. "
            f"rc={rc} stdout_head={out[:300]!r} stderr_head={err[:300]!r}"
        )


# ---------- Main autoscaler loop ----------

def desired_vm_count(backlog: int) -> int:
    if backlog <= 0:
        return 0
    # 1..BACKLOG_PER_VM => 1, BACKLOG_PER_VM+1..2* => 2, etc
    return ((backlog - 1) // BACKLOG_PER_VM) + 1


def autoscaler_loop() -> None:
    ensure_tables()
    log("worker loop started")

    while True:
        try:
            backlog = get_backlog_count()
            want = desired_vm_count(backlog)

            tracked = get_tracked_live_instances()
            have = len(tracked)

            log(f"backlog={backlog} want_vms={want} have_vms={have}")

            # scale up
            while have < want:
                # For now: always rent 1-GPU VM (your choice: 1 job per VM)
                log("renting 1x GPU VM on Hyperbolic...")
                instance_id = rent_vm(gpu_count=1)
                mark_instance_created(instance_id, note="rented by autoscaler")
                have += 1
                log(f"rented instance_id={instance_id}")

                # Bootstrapping host discovery is the last missing piece:
                # once we can reliably extract host/ip from hyperbolic-cli output,
                # call bootstrap_vm_over_ssh(host=..., user=HYPERBOLIC_SSH_USER).
                log("NOTE: VM bootstrap over SSH not executed (host discovery not wired yet).")

            # scale down (terminate oldest tracked instances)
            while have > want:
                instance_id = tracked[0]
                log(f"terminating instance_id={instance_id}...")
                terminate_vm(instance_id)
                mark_instance_terminated(instance_id, note="terminated by autoscaler")
                tracked = get_tracked_live_instances()
                have = len(tracked)
                log(f"terminated instance_id={instance_id}")

        except Exception as e:
            log(f"autoscaler error: {e}")

        time.sleep(SCALE_POLL_SECONDS)


def main() -> None:
    autoscaler_loop()


if __name__ == "__main__":
    main()
