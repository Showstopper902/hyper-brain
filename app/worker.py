# app/worker.py
import json
import logging
import os
import re
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import psycopg

LOG = logging.getLogger("brain-worker")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s][brain-worker] %(message)s",
)

# ----------------------------
# Config
# ----------------------------

DATABASE_URL = os.environ.get("DATABASE_URL", "")
HYPERBOLIC_API_KEY = os.environ.get("HYPERBOLIC_API_KEY", "")
ASSIGNED_WORKER_ID = os.environ.get("ASSIGNED_WORKER_ID", "hyperbolic-pool")

# Scaling policy
MAX_UNCLAIMED_PER_VM = int(os.getenv("MAX_UNCLAIMED_PER_VM", "250"))
GPU_CHOICES = [1, 2, 4]  # prefer fewer GPUs (1 > 2 > 4)
LOOP_SECONDS = float(os.getenv("AUTOSCALER_LOOP_SECONDS", "4"))

# Hyperbolic CLI deterministic home (critical on Fly)
HCLI_HOME = os.getenv("HYPERBOLIC_CLI_HOME", "/tmp/hyperbolic-cli-home")

# SSH/bootstrap
SSH_PRIVATE_KEY = os.environ.get("SSH_PRIVATE_KEY", "")
HYPERBOLIC_SSH_USER = os.getenv("HYPERBOLIC_SSH_USER", "root")

BRAIN_URL = os.getenv("BRAIN_URL", "https://hyper-brain.fly.dev")
EXECUTOR_TOKEN = os.getenv("EXECUTOR_TOKEN", "")

# Executor behavior
EXECUTOR_POLL_SECONDS = int(os.getenv("EXECUTOR_POLL_SECONDS", "3"))
EXECUTOR_IDLE_SECONDS = int(os.getenv("EXECUTOR_IDLE_SECONDS", "3600"))

# Where to fetch the executor loop script from
EXECUTOR_LOOP_URL = os.getenv(
    "EXECUTOR_LOOP_URL",
    "https://raw.githubusercontent.com/Showstopper902/hyperbolic_project/main/scripts/hyper_executor_loop.sh",
)

# Secrets for B2 env file we place onto the VM
B2_SYNC = os.getenv("B2_SYNC", "1")
B2_BUCKET = os.getenv("B2_BUCKET", "")
B2_S3_ENDPOINT = os.getenv("B2_S3_ENDPOINT", "")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-west-004")


@dataclass
class Instance:
    instance_id: str
    raw: Dict[str, Any]


# ----------------------------
# DB helpers
# ----------------------------

def db_conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL missing")
    return psycopg.connect(DATABASE_URL)


def get_counts() -> Tuple[int, int]:
    """
    Returns:
      (running_total, unclaimed_running)
    We scale UP based on unclaimed_running, but we keep at least 1 VM if running_total>0
    to avoid terminating mid-job.
    """
    with db_conn() as conn:
        with conn.cursor() as cur:
            # tolerate different schemas by being defensive
            cur.execute(
                """
                SELECT
                  COALESCE(SUM(CASE WHEN status='RUNNING' THEN 1 ELSE 0 END),0) AS running_total,
                  COALESCE(SUM(CASE WHEN status='RUNNING' AND executor_id IS NULL THEN 1 ELSE 0 END),0) AS unclaimed_running
                FROM jobs
                """
            )
            row = cur.fetchone()
            running_total = int(row[0] or 0)
            unclaimed = int(row[1] or 0)
            return running_total, unclaimed


def want_vms(running_total: int, unclaimed: int) -> int:
    if running_total <= 0:
        return 0
    # at least 1 if anything is RUNNING
    base = 1
    # scale beyond 1 only if unclaimed backlog is large
    extra = (unclaimed + MAX_UNCLAIMED_PER_VM - 1) // MAX_UNCLAIMED_PER_VM
    return max(base, extra)


# ----------------------------
# Hyperbolic CLI wrapper
# ----------------------------

def _hcli_env() -> Dict[str, str]:
    env = os.environ.copy()
    env["HOME"] = HCLI_HOME
    env["XDG_CONFIG_HOME"] = HCLI_HOME
    env["XDG_CACHE_HOME"] = HCLI_HOME
    return env


def _run(cmd: List[str], timeout: int = 30, check: bool = True) -> subprocess.CompletedProcess:
    env = _hcli_env()
    os.makedirs(HCLI_HOME, exist_ok=True)
    p = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        env=env,
        timeout=timeout,
    )
    if check and p.returncode != 0:
        raise RuntimeError(
            f"command failed rc={p.returncode} cmd={cmd} "
            f"stdout_head={p.stdout[:400]!r} stderr_head={p.stderr[:400]!r}"
        )
    return p


def ensure_hcli_auth() -> None:
    if not HYPERBOLIC_API_KEY:
        raise RuntimeError("HYPERBOLIC_API_KEY missing in env")
    # Always set-key before doing anything else (Fly machines/users differ)
    _run(["hyperbolic", "auth", "set-key", HYPERBOLIC_API_KEY], timeout=20, check=True)


def parse_json_or_raise(output: str, context: str) -> Any:
    s = output.strip()
    if not s.startswith("{") and not s.startswith("["):
        raise RuntimeError(f"{context} returned non-JSON output: {s[:500]!r}")
    try:
        return json.loads(s)
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON from {context}: {e}. head={s[:500]!r}") from e


def list_instances() -> List[Instance]:
    ensure_hcli_auth()
    p = _run(["hyperbolic", "instances", "--json"], timeout=30, check=True)
    j = parse_json_or_raise(p.stdout, "hyperbolic instances --json")

    # normalize to list
    if isinstance(j, dict) and "instances" in j and isinstance(j["instances"], list):
        items = j["instances"]
    elif isinstance(j, list):
        items = j
    else:
        # last resort: if it's a dict that looks like one instance
        items = [j] if isinstance(j, dict) else []

    out: List[Instance] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        iid = str(it.get("instance_id") or it.get("id") or it.get("instanceId") or "")
        if iid:
            out.append(Instance(instance_id=iid, raw=it))
    return out


def instance_detail(instance_id: str) -> Dict[str, Any]:
    ensure_hcli_auth()
    p = _run(["hyperbolic", "instances", instance_id, "--json"], timeout=30, check=True)
    return parse_json_or_raise(p.stdout, f"hyperbolic instances {instance_id} --json")


def terminate_instance(instance_id: str) -> None:
    ensure_hcli_auth()
    _run(["hyperbolic", "terminate", instance_id], timeout=30, check=True)


def _gpu_count_of(raw: Dict[str, Any]) -> Optional[int]:
    for k in ("gpu_count", "gpuCount", "gpus", "gpu"):
        v = raw.get(k)
        if isinstance(v, int):
            return v
        if isinstance(v, str) and v.isdigit():
            return int(v)
    # sometimes nested
    spec = raw.get("spec") or raw.get("configuration") or {}
    if isinstance(spec, dict):
        for k in ("gpu_count", "gpuCount", "gpus"):
            v = spec.get(k)
            if isinstance(v, int):
                return v
            if isinstance(v, str) and v.isdigit():
                return int(v)
    return None


def rent_vm(gpu_count: int) -> str:
    """
    Never parse rent stdout (unstable). Instead:
      - snapshot instances
      - run rent
      - poll instances and find the new one
    """
    ensure_hcli_auth()

    before = {i.instance_id for i in list_instances()}

    p = _run(
        ["hyperbolic", "rent", "ondemand", "--instance-type", "virtual-machine", "--gpu-count", str(gpu_count)],
        timeout=60,
        check=False,  # we want the real error payload if it fails
    )
    if p.returncode != 0:
        # This is where your 404 is coming from; surface it clearly.
        raise RuntimeError(
            f"rent failed gpu_count={gpu_count} rc={p.returncode} "
            f"stdout_head={p.stdout[:600]!r} stderr_head={p.stderr[:600]!r}"
        )

    # Poll for a new instance showing up
    for _ in range(30):  # ~60s
        time.sleep(2)
        after_list = list_instances()
        after = {i.instance_id for i in after_list}
        new_ids = list(after - before)
        if not new_ids:
            continue

        # Prefer one matching gpu_count
        for iid in new_ids:
            inst = next((x for x in after_list if x.instance_id == iid), None)
            if inst and _gpu_count_of(inst.raw) == gpu_count:
                return iid

        # Otherwise return newest-looking id
        return new_ids[0]

    raise RuntimeError("rent succeeded but could not discover new instance_id via instances --json")


# ----------------------------
# SSH/bootstrap
# ----------------------------

def _write_temp_key() -> str:
    if not SSH_PRIVATE_KEY.strip():
        raise RuntimeError("SSH_PRIVATE_KEY missing/empty")
    fd, path = tempfile.mkstemp(prefix="hyperbolic_ssh_", text=True)
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        f.write(SSH_PRIVATE_KEY.strip() + "\n")
    os.chmod(path, 0o600)
    return path


def _extract_ssh_target(detail: Dict[str, Any]) -> Tuple[str, int, str]:
    """
    Returns (host, port, user). We try multiple shapes.
    """
    # Common: ssh_command like "ssh user@host" or includes -p
    ssh_cmd = detail.get("ssh_command") or detail.get("sshCommand") or detail.get("ssh")
    if isinstance(ssh_cmd, str) and "ssh" in ssh_cmd:
        parts = shlex.split(ssh_cmd)
        # find -p and user@host
        port = 22
        target = None
        i = 0
        while i < len(parts):
            if parts[i] == "-p" and i + 1 < len(parts):
                try:
                    port = int(parts[i + 1])
                except Exception:
                    pass
                i += 2
                continue
            if parts[i].endswith("@") or parts[i].startswith("-"):
                i += 1
                continue
            if "@" in parts[i] and not parts[i].startswith("-") and parts[i] != "ssh":
                target = parts[i]
            i += 1

        if target:
            user, host = target.split("@", 1)
            return host, port, user

    # Alternate: structured fields
    host = detail.get("host") or detail.get("ip") or detail.get("ip_address") or detail.get("public_ip")
    port = detail.get("port") or detail.get("ssh_port") or 22
    user = detail.get("user") or detail.get("ssh_user") or HYPERBOLIC_SSH_USER

    if isinstance(host, str) and host:
        try:
            port = int(port)
        except Exception:
            port = 22
        return host, port, str(user)

    raise RuntimeError(f"Could not extract ssh target from instance detail keys={list(detail.keys())[:30]}")


def ssh_run(host: str, port: int, user: str, script: str) -> None:
    key_path = _write_temp_key()
    try:
        cmd = [
            "ssh",
            "-i", key_path,
            "-p", str(port),
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=10",
            f"{user}@{host}",
            "bash -lc 'cat > /tmp/bootstrap.sh && sudo bash /tmp/bootstrap.sh'",
        ]
        p = subprocess.run(
            cmd,
            input=script,
            text=True,
            capture_output=True,
            timeout=180,
        )
        if p.returncode != 0:
            raise RuntimeError(
                f"ssh bootstrap failed rc={p.returncode} host={host} "
                f"stdout_head={p.stdout[:600]!r} stderr_head={p.stderr[:600]!r}"
            )
    finally:
        try:
            os.remove(key_path)
        except Exception:
            pass


def build_bootstrap_script() -> str:
    """
    Generates the exact VM-side bootstrap steps (your "way 2"):
      - write env files
      - install executor loop script
      - install+enable systemd service
    """
    # DO NOT log this script (contains secrets)
    if not EXECUTOR_TOKEN:
        raise RuntimeError("EXECUTOR_TOKEN missing (needed to run executor loop on VM)")
    if not B2_BUCKET or not B2_S3_ENDPOINT or not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        raise RuntimeError("B2/AWS env vars missing (bucket/endpoint/access key/secret)")

    return f"""#!/usr/bin/env bash
set -euo pipefail

sudo mkdir -p /data/secrets /data/bin

# ---- B2 env ----
sudo bash -c 'cat > /data/secrets/b2.env <<EOF
B2_SYNC={B2_SYNC}
B2_BUCKET={B2_BUCKET}
B2_S3_ENDPOINT={B2_S3_ENDPOINT}
AWS_ACCESS_KEY_ID={AWS_ACCESS_KEY_ID}
AWS_SECRET_ACCESS_KEY={AWS_SECRET_ACCESS_KEY}
AWS_DEFAULT_REGION={AWS_DEFAULT_REGION}
EOF'
sudo chmod 600 /data/secrets/b2.env

# ---- executor env ----
sudo bash -c 'cat > /data/secrets/hyper_executor.env <<EOF
EXECUTOR_TOKEN="{EXECUTOR_TOKEN}"
BRAIN_URL="{BRAIN_URL}"
ASSIGNED_WORKER_ID="{ASSIGNED_WORKER_ID}"
POLL_SECONDS={EXECUTOR_POLL_SECONDS}
IDLE_SECONDS={EXECUTOR_IDLE_SECONDS}
EXECUTOR_ID="exec-$(hostname)"
EOF'
sudo chmod 600 /data/secrets/hyper_executor.env

# ---- executor loop script ----
sudo curl -fsSL "{EXECUTOR_LOOP_URL}" -o /data/bin/hyper_executor_loop.sh
sudo chmod +x /data/bin/hyper_executor_loop.sh

# ---- systemd unit ----
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
sudo systemctl restart hyper-executor.service || true
"""


# ----------------------------
# Main loop
# ----------------------------

def autoscaler_tick() -> None:
    running_total, unclaimed = get_counts()
    want = want_vms(running_total, unclaimed)

    inst = list_instances()
    have = len(inst)

    LOG.info("running_total=%s unclaimed=%s want_vms=%s have_vms=%s", running_total, unclaimed, want, have)

    # Scale down only when truly no RUNNING jobs exist.
    if want == 0:
        if have > 0:
            LOG.info("no backlog -> terminating %s instance(s)", have)
        for i in inst:
            try:
                terminate_instance(i.instance_id)
                LOG.info("terminated instance_id=%s", i.instance_id)
            except Exception as e:
                LOG.info("terminate failed instance_id=%s err=%s", i.instance_id, e)
        return

    # If we already have at least one VM, do nothing (your “1 VM handles many jobs” rule)
    # Only scale up beyond 1 when unclaimed backlog exceeds thresholds.
    if have >= want:
        return

    # Need +1 VM
    for gpu in GPU_CHOICES:
        try:
            LOG.info("renting %sx GPU VM on Hyperbolic...", gpu)
            iid = rent_vm(gpu)
            LOG.info("rented instance_id=%s; bootstrapping...", iid)

            detail = instance_detail(iid)
            host, port, user = _extract_ssh_target(detail)

            script = build_bootstrap_script()
            # Wait for SSH to come up
            for attempt in range(30):
                try:
                    ssh_run(host, port, user, script)
                    LOG.info("bootstrap ok instance_id=%s host=%s", iid, host)
                    return
                except Exception as e:
                    if attempt == 29:
                        raise
                    time.sleep(4)

        except Exception as e:
            LOG.info("rent/bootstrap attempt failed for gpu_count=%s: %s", gpu, e)

    raise RuntimeError("could not rent+bootstrap any VM size (1/2/4 failed)")


def main() -> None:
    LOG.info("autoscaler starting (HCLI_HOME=%s)", HCLI_HOME)
    while True:
        try:
            autoscaler_tick()
        except Exception:
            # full traceback (this is how we stop getting useless "error: 0")
            LOG.exception("autoscaler error")
        time.sleep(LOOP_SECONDS)


if __name__ == "__main__":
    main()
