import os
import re
import json
import time
import base64
import logging
import subprocess
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import psycopg


# ---------------------------
# Logging
# ---------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s][brain-worker] %(message)s",
)
log = logging.getLogger("brain-worker")


# ---------------------------
# Config
# ---------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "")

POLL_SECONDS = int(os.getenv("POLL_SECONDS", "3"))

# Scaling rule: 0 backlog => 0 VMs; else 1 VM per 250 unclaimed RUNNING jobs
BACKLOG_PER_VM = int(os.getenv("BACKLOG_PER_VM", "250"))

# Hyperbolic preferences (you said: VM, 1>2>4 GPUs, prefer us-central-1, ethernet, payg)
GPU_CHOICES = [1, 2, 4]

# Where to fetch the executor loop script on the Hyperbolic VM
EXECUTOR_LOOP_URL = os.getenv(
    "EXECUTOR_LOOP_URL",
    "https://raw.githubusercontent.com/Showstopper902/hyperbolic_project/main/scripts/hyper_executor_loop.sh",
)

# Brain URL (used by executor env on the rented VM)
BRAIN_URL = os.getenv("BRAIN_URL", "https://hyper-brain.fly.dev")
ASSIGNED_WORKER_ID = os.getenv("ASSIGNED_WORKER_ID", "hyperbolic-pool")
EXECUTOR_TOKEN = os.getenv("EXECUTOR_TOKEN", "")

# SSH
HYPERBOLIC_SSH_USER = os.getenv("HYPERBOLIC_SSH_USER", "ubuntu")  # you set this as a secret
SSH_PRIVATE_KEY = os.getenv("SSH_PRIVATE_KEY", "")

# Hyperbolic API Key for CLI
HYPERBOLIC_API_KEY = (
    os.getenv("HYPERBOLIC_API_KEY")
    or os.getenv("HYPERBOLIC_KEY")
    or os.getenv("HYPERBOLIC_TOKEN")
    or ""
)

# IMPORTANT: try to force the API base if Hyperbolic moved domains (common cause of your 404s)
# We set multiple possible env vars so whichever the CLI respects will stick.
HYPERBOLIC_API_BASE = os.getenv("HYPERBOLIC_API_BASE", "https://api.hyperbolic.xyz")
HYPERBOLIC_APP_BASE = os.getenv("HYPERBOLIC_APP_BASE", "https://app.hyperbolic.xyz")

# B2 env (passed down to executor VM)
B2_SYNC = os.getenv("B2_SYNC", "1")
B2_BUCKET = os.getenv("B2_BUCKET", "")
B2_S3_ENDPOINT = os.getenv("B2_S3_ENDPOINT", "")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "")


# ---------------------------
# DB helpers
# ---------------------------
def db_conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is missing")
    return psycopg.connect(DATABASE_URL)


def ensure_tables():
    # Keep it minimal: jobs table is assumed to exist already.
    # We add a small table to track rented Hyperbolic VMs (instance_id + ssh info)
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS hyper_vms (
              instance_id TEXT PRIMARY KEY,
              gpu_count INT NOT NULL,
              ssh_user TEXT,
              ssh_host TEXT,
              ssh_port INT,
              raw_ssh_cmd TEXT,
              created_at TIMESTAMPTZ DEFAULT now(),
              terminated_at TIMESTAMPTZ
            );
            """
        )
        conn.commit()


def backlog_counts() -> Tuple[int, int]:
    """
    Returns:
      unclaimed = count of RUNNING jobs where executor_id IS NULL
      inflight  = count of RUNNING jobs where executor_id IS NOT NULL
    """
    with db_conn() as conn, conn.cursor() as cur:
        # Your requirement: use RUNNING + executor_id IS NULL for backlog.
        cur.execute(
            """
            SELECT
              COALESCE(SUM(CASE WHEN status='RUNNING' AND executor_id IS NULL THEN 1 ELSE 0 END),0) AS unclaimed,
              COALESCE(SUM(CASE WHEN status='RUNNING' AND executor_id IS NOT NULL THEN 1 ELSE 0 END),0) AS inflight
            FROM jobs;
            """
        )
        row = cur.fetchone()
        if not row:
            return 0, 0
        return int(row[0]), int(row[1])


def list_tracked_vms() -> List[Dict[str, Any]]:
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT instance_id, gpu_count, ssh_user, ssh_host, ssh_port, raw_ssh_cmd, terminated_at
            FROM hyper_vms
            WHERE terminated_at IS NULL
            ORDER BY created_at ASC;
            """
        )
        rows = cur.fetchall() or []
        out = []
        for r in rows:
            out.append(
                dict(
                    instance_id=r[0],
                    gpu_count=int(r[1]),
                    ssh_user=r[2],
                    ssh_host=r[3],
                    ssh_port=int(r[4]) if r[4] is not None else None,
                    raw_ssh_cmd=r[5],
                    terminated_at=r[6],
                )
            )
        return out


def track_vm(instance_id: str, gpu_count: int, ssh_user: str, ssh_host: str, ssh_port: int, raw_ssh_cmd: str):
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO hyper_vms (instance_id, gpu_count, ssh_user, ssh_host, ssh_port, raw_ssh_cmd)
            VALUES (%s,%s,%s,%s,%s,%s)
            ON CONFLICT (instance_id) DO UPDATE
              SET gpu_count=EXCLUDED.gpu_count,
                  ssh_user=EXCLUDED.ssh_user,
                  ssh_host=EXCLUDED.ssh_host,
                  ssh_port=EXCLUDED.ssh_port,
                  raw_ssh_cmd=EXCLUDED.raw_ssh_cmd,
                  terminated_at=NULL;
            """
            ,
            (instance_id, gpu_count, ssh_user, ssh_host, ssh_port, raw_ssh_cmd),
        )
        conn.commit()


def mark_terminated(instance_id: str):
    with db_conn() as conn, conn.cursor() as cur:
        cur.execute("UPDATE hyper_vms SET terminated_at=now() WHERE instance_id=%s;", (instance_id,))
        conn.commit()


# ---------------------------
# Subprocess helpers
# ---------------------------
def run_cmd(cmd: List[str], env: Optional[Dict[str, str]] = None, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        timeout=timeout,
    )


def hyperbolic_env() -> Dict[str, str]:
    """
    Force a clean config HOME so the CLI always finds its auth config,
    even if Fly runs as different users.
    Also try forcing API base URLs to avoid the 404 you're seeing.
    """
    e = dict(os.environ)

    # Force a stable CLI home/config location
    cli_home = "/tmp/hyperbolic-cli-home"
    e["HOME"] = cli_home
    e["XDG_CONFIG_HOME"] = cli_home
    e["XDG_CACHE_HOME"] = cli_home

    # Domain/base overrides (the CLI will ignore unknown vars, but if it supports any of these, it fixes 404)
    e["HYPERBOLIC_API_BASE_URL"] = HYPERBOLIC_API_BASE
    e["HYPERBOLIC_API_URL"] = HYPERBOLIC_API_BASE
    e["HYPERBOLIC_BASE_URL"] = HYPERBOLIC_API_BASE
    e["HYPERBOLIC_ENDPOINT"] = HYPERBOLIC_API_BASE
    e["HYPERBOLIC_APP_URL"] = HYPERBOLIC_APP_BASE

    return e


def ensure_hyperbolic_auth():
    if not HYPERBOLIC_API_KEY:
        raise RuntimeError("HYPERBOLIC_API_KEY is empty in the worker environment (Fly secret not visible).")

    e = hyperbolic_env()
    # Create the HOME dir
    run_cmd(["bash", "-lc", "mkdir -p /tmp/hyperbolic-cli-home"], env=e, timeout=10)

    p = run_cmd(["hyperbolic", "auth", "set-key", HYPERBOLIC_API_KEY], env=e, timeout=20)
    if p.returncode != 0:
        raise RuntimeError(f"hyperbolic auth set-key failed rc={p.returncode} stdout={p.stdout[:400]!r} stderr={p.stderr[:400]!r}")


# ---------------------------
# Parse helpers
# ---------------------------
UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)

def parse_instance_id(text: str) -> Optional[str]:
    # Prefer UUID if present
    m = UUID_RE.search(text)
    if m:
        return m.group(0)

    # Fallback: "instance-id: XXX" / "instance_id=XXX"
    m = re.search(r"(?:instance[_ -]?id)\s*[:=]\s*([A-Za-z0-9\-]+)", text, re.I)
    if m:
        return m.group(1)

    return None


def parse_ssh_cmd(text: str) -> Optional[str]:
    # Find the first "ssh ..." line
    m = re.search(r"(ssh\b[^\n\r]+)", text)
    if not m:
        return None
    return m.group(1).strip()


def parse_ssh_target(ssh_cmd: str) -> Optional[Tuple[str, str, int]]:
    """
    Extract (user, host, port) from an ssh command.
    Supports:
      ssh user@host
      ssh -p 2222 user@host
      ssh user@host -p 2222
    """
    port = 22
    # port
    m = re.search(r"\s-p\s+(\d+)", ssh_cmd)
    if m:
        port = int(m.group(1))

    # user@host
    m = re.search(r"\b([A-Za-z0-9._-]+)@([A-Za-z0-9.\-]+)\b", ssh_cmd)
    if not m:
        return None
    return m.group(1), m.group(2), port


# ---------------------------
# SSH bootstrap
# ---------------------------
def write_ssh_key() -> str:
    if not SSH_PRIVATE_KEY.strip():
        raise RuntimeError("SSH_PRIVATE_KEY secret is empty.")
    key_path = "/tmp/hyperbolic_ssh_key"
    with open(key_path, "w", encoding="utf-8") as f:
        f.write(SSH_PRIVATE_KEY.strip() + "\n")
    os.chmod(key_path, 0o600)
    return key_path


def ssh_run(host: str, port: int, user: str, key_path: str, remote_bash: str, timeout: int = 900) -> subprocess.CompletedProcess:
    cmd = [
        "ssh",
        "-i", key_path,
        "-p", str(port),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        f"{user}@{host}",
        "bash", "-lc", remote_bash,
    ]
    return run_cmd(cmd, timeout=timeout)


def bootstrap_executor_vm(instance_id: str, host: str, port: int, ssh_user: str, key_path: str):
    """
    Writes /data/secrets/b2.env + /data/secrets/hyper_executor.env on the Hyperbolic VM,
    installs /data/bin/hyper_executor_loop.sh (curl), installs/enables systemd unit.
    """
    if not EXECUTOR_TOKEN:
        raise RuntimeError("EXECUTOR_TOKEN missing in Fly secrets/env.")

    # Build env file contents locally (then base64 to avoid quoting issues)
    b2_env = "\n".join([
        f"B2_SYNC={B2_SYNC}",
        f"B2_BUCKET={B2_BUCKET}",
        f"B2_S3_ENDPOINT={B2_S3_ENDPOINT}",
        f"AWS_ACCESS_KEY_ID={AWS_ACCESS_KEY_ID}",
        f"AWS_SECRET_ACCESS_KEY={AWS_SECRET_ACCESS_KEY}",
        f"AWS_DEFAULT_REGION={AWS_DEFAULT_REGION}",
        "",
    ])
    hyper_exec_env = "\n".join([
        f'EXECUTOR_TOKEN="{EXECUTOR_TOKEN}"',
        f'BRAIN_URL="{BRAIN_URL}"',
        f'ASSIGNED_WORKER_ID="{ASSIGNED_WORKER_ID}"',
        "POLL_SECONDS=3",
        "IDLE_SECONDS=3600",
        f'EXECUTOR_ID="exec-{instance_id}"',
        "",
    ])

    b2_b64 = base64.b64encode(b2_env.encode("utf-8")).decode("ascii")
    ex_b64 = base64.b64encode(hyper_exec_env.encode("utf-8")).decode("ascii")

    # Use sudo where needed
    remote = f"""
set -euo pipefail

sudo mkdir -p /data/secrets /data/bin

echo "{b2_b64}" | base64 -d | sudo tee /data/secrets/b2.env >/dev/null
echo "{ex_b64}" | base64 -d | sudo tee /data/secrets/hyper_executor.env >/dev/null
sudo chmod 600 /data/secrets/b2.env /data/secrets/hyper_executor.env

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
sudo systemctl restart hyper-executor.service

echo "BOOTSTRAP_OK"
""".strip()

    p = ssh_run(host, port, ssh_user, key_path, remote, timeout=900)
    if p.returncode != 0 or "BOOTSTRAP_OK" not in (p.stdout or ""):
        raise RuntimeError(
            f"bootstrap failed rc={p.returncode} stdout_head={p.stdout[:600]!r} stderr_head={p.stderr[:600]!r}"
        )


# ---------------------------
# Hyperbolic actions (CLI)
# ---------------------------
def hyperbolic_instances_json() -> List[Dict[str, Any]]:
    """
    Best-effort. If Hyperbolic returns non-JSON (your current situation), we return [] but log full context.
    """
    ensure_hyperbolic_auth()
    e = hyperbolic_env()
    p = run_cmd(["hyperbolic", "instances", "--json"], env=e, timeout=30)
    out = (p.stdout or "").strip()
    err = (p.stderr or "").strip()

    if p.returncode != 0:
        log.warning("hyperbolic instances rc=%s stdout_head=%r stderr_head=%r", p.returncode, out[:400], err[:400])
        return []

    try:
        j = json.loads(out)
        if isinstance(j, list):
            return j
        # some CLIs return {"instances":[...]}
        if isinstance(j, dict) and "instances" in j and isinstance(j["instances"], list):
            return j["instances"]
    except Exception:
        log.warning("hyperbolic instances --json returned non-JSON output. stdout_head=%r stderr_head=%r", out[:500], err[:500])
        return []

    return []


def hyperbolic_rent_vm(gpu_count: int) -> Tuple[str, str, str, int]:
    """
    Rent a VM with gpu_count and return:
      (instance_id, raw_ssh_cmd, host, port)
    """
    ensure_hyperbolic_auth()
    e = hyperbolic_env()

    p = run_cmd(
        ["hyperbolic", "rent", "ondemand", "--instance-type", "virtual-machine", "--gpu-count", str(gpu_count)],
        env=e,
        timeout=120,
    )
    out = (p.stdout or "")
    err = (p.stderr or "")

    if p.returncode != 0:
        raise RuntimeError(f"rent failed rc={p.returncode} stdout_head={out[:600]!r} stderr_head={err[:600]!r}")

    instance_id = parse_instance_id(out)
    ssh_cmd = parse_ssh_cmd(out)

    if not instance_id:
        raise RuntimeError(f"Could not determine instance_id after rent. rent_stdout_head={out[:700]!r}")

    if not ssh_cmd:
        raise RuntimeError(f"Could not parse ssh command from rent output. rent_stdout_head={out[:700]!r}")

    parsed = parse_ssh_target(ssh_cmd)
    if not parsed:
        raise RuntimeError(f"Could not parse ssh user/host/port. ssh_cmd={ssh_cmd!r}")

    ssh_user, host, port = parsed
    return instance_id, ssh_cmd, host, port


def hyperbolic_terminate(instance_id: str):
    ensure_hyperbolic_auth()
    e = hyperbolic_env()
    p = run_cmd(["hyperbolic", "terminate", instance_id], env=e, timeout=60)
    if p.returncode != 0:
        raise RuntimeError(f"terminate failed rc={p.returncode} stdout_head={p.stdout[:400]!r} stderr_head={p.stderr[:400]!r}")


# ---------------------------
# Autoscaler tick
# ---------------------------
def desired_vm_count(unclaimed: int, inflight: int) -> int:
    if unclaimed <= 0 and inflight <= 0:
        return 0
    want = 0
    if unclaimed > 0:
        want = (unclaimed - 1) // BACKLOG_PER_VM + 1
    # never scale down below inflight jobs (so we don't kill active processing)
    want = max(want, inflight)
    return want


def autoscaler_tick():
    unclaimed, inflight = backlog_counts()
    want_vms = desired_vm_count(unclaimed, inflight)
    tracked = list_tracked_vms()
    have_vms = len(tracked)

    log.info("backlog=%s inflight=%s want_vms=%s have_vms=%s", unclaimed, inflight, want_vms, have_vms)

    # Scale up
    if want_vms > have_vms:
        missing = want_vms - have_vms
        log.info("need %s more VM(s)", missing)

        key_path = write_ssh_key()

        for _ in range(missing):
            last_err = None
            for g in GPU_CHOICES:
                try:
                    log.info("renting %sx GPU VM on Hyperbolic...", g)
                    instance_id, ssh_cmd, host, port = hyperbolic_rent_vm(g)

                    # You told me: 1 job per VM for now; executor_id must be deterministic
                    # We keep ssh_user from the rent output (more accurate than secret default)
                    parsed = parse_ssh_target(ssh_cmd)
                    assert parsed is not None
                    ssh_user, _, _ = parsed

                    log.info("rented instance_id=%s host=%s port=%s ssh_user=%s", instance_id, host, port, ssh_user)

                    # Bootstrap executor on the rented VM
                    bootstrap_executor_vm(instance_id, host, port, ssh_user, key_path)

                    # Track in DB
                    track_vm(instance_id, g, ssh_user, host, port, ssh_cmd)

                    # Success -> break gpu choice loop
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    log.info("rent attempt failed for gpu_count=%s: %s", g, str(e))
                    continue

            if last_err is not None:
                raise RuntimeError(f"could not rent+bootstrap any VM size: {last_err}")

    # Scale down (only when truly nothing is running and no backlog)
    # If you later want aggressive scale-down, we can do it safely by mapping executor_id -> instance_id.
    if want_vms < have_vms and unclaimed == 0 and inflight == 0:
        extra = have_vms - want_vms
        log.info("scaling down: terminating %s VM(s)", extra)

        # terminate oldest first
        for vm in tracked[:extra]:
            iid = vm["instance_id"]
            try:
                log.info("terminating instance_id=%s ...", iid)
                hyperbolic_terminate(iid)
                mark_terminated(iid)
                log.info("terminated instance_id=%s", iid)
            except Exception as e:
                log.info("terminate failed for %s: %s", iid, str(e))


# ---------------------------
# Main loop
# ---------------------------
def main():
    log.info("worker boot: poll=%ss backlog_per_vm=%s hyper_api_base=%s", POLL_SECONDS, BACKLOG_PER_VM, HYPERBOLIC_API_BASE)
    ensure_tables()

    while True:
        try:
            autoscaler_tick()
        except Exception as e:
            # Never crash the whole worker; log the real error
            log.error("autoscaler error", exc_info=True)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
