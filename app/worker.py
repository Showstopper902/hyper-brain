#!/usr/bin/env python3
"""
Hyper-brain autoscaler worker.

Goals:
- Scale Hyperbolic GPU VMs up/down based on backlog (unclaimed RUNNING/QUEUED jobs).
- "No backlog = no instances."
- Preference: VM, 1 GPU if possible (fallback 2 then 4), 1 executor per VM.
- Bootstrap VMs over SSH (Way 2) by installing hyper_executor_loop.sh + systemd unit.

This file is intentionally stdlib-heavy to avoid dependency fragility.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse


# ----------------------------
# Logging
# ----------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s][brain-worker] %(message)s",
)
log = logging.getLogger("brain-worker")


# ----------------------------
# Config
# ----------------------------

BRAIN_URL = os.getenv("BRAIN_URL", "https://hyper-brain.fly.dev").rstrip("/")
ASSIGNED_WORKER_ID = os.getenv("ASSIGNED_WORKER_ID", "hyperbolic-pool")

# Backlog policy: 0 backlog => 0 VMs.
BACKLOG_PER_VM = int(os.getenv("BACKLOG_PER_VM", "250"))  # your rule: +1 VM per 250 unclaimed
TICK_SECONDS = float(os.getenv("AUTOSCALER_TICK_SECONDS", "4"))

# GPU sizing preference
GPU_FALLBACKS = [int(x) for x in os.getenv("GPU_FALLBACKS", "1,2,4").split(",") if x.strip()]
MAX_GPUS = int(os.getenv("MAX_GPUS", "4"))

# Hyperbolic CLI
HYPERBOLIC_BIN = os.getenv("HYPERBOLIC_BIN", "hyperbolic")
HYPERBOLIC_API_KEY = os.getenv("HYPERBOLIC_API_KEY", "").strip()

# If the CLI supports an override env var for base URL, you can set it in Fly secrets:
# flyctl secrets set -a hyper-brain HYPERBOLIC_API_BASE_URL=https://api.hyperbolic.ai
HYPERBOLIC_API_BASE_URL = os.getenv("HYPERBOLIC_API_BASE_URL", "").strip()

# SSH / bootstrap
SSH_PRIVATE_KEY = os.getenv("SSH_PRIVATE_KEY", "")
HYPERBOLIC_SSH_USER = os.getenv("HYPERBOLIC_SSH_USER", "root")
# The script the VM should download (your hyperbolic_project repo)
EXECUTOR_LOOP_URL = os.getenv(
    "EXECUTOR_LOOP_URL",
    "https://raw.githubusercontent.com/Showstopper902/hyperbolic_project/main/scripts/hyper_executor_loop.sh",
)

# VM local secrets to write
EXECUTOR_TOKEN = os.getenv("EXECUTOR_TOKEN", "").strip()

# B2 env (passed to VM)
B2_SYNC = os.getenv("B2_SYNC", "1")
B2_BUCKET = os.getenv("B2_BUCKET", "")
B2_S3_ENDPOINT = os.getenv("B2_S3_ENDPOINT", "")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "")

# Safety: only terminate instances that contain this marker in their name/metadata (if present).
# If your CLI JSON includes tags/names, you can enforce this. Default: allow all.
INSTANCE_NAME_PREFIX = os.getenv("INSTANCE_NAME_PREFIX", "").strip()


# ----------------------------
# Utilities
# ----------------------------

def _redact(s: str, keep: int = 200) -> str:
    s = s or ""
    s = s.replace(HYPERBOLIC_API_KEY, "REDACTED_API_KEY") if HYPERBOLIC_API_KEY else s
    s = s.replace(EXECUTOR_TOKEN, "REDACTED_EXECUTOR_TOKEN") if EXECUTOR_TOKEN else s
    return s[:keep]


def _which(cmd: str) -> str:
    from shutil import which
    p = which(cmd)
    return p or cmd


def _json_extract_first(payload: str) -> Optional[str]:
    """
    Extract the first JSON object/array from a string, if any.
    Handles cases where CLI prints extra lines before JSON.
    """
    if not payload:
        return None
    start = None
    for i, ch in enumerate(payload):
        if ch in "{[":
            start = i
            break
    if start is None:
        return None

    stack = []
    in_str = False
    esc = False
    for i in range(start, len(payload)):
        ch = payload[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if not stack:
                    return None
                op = stack.pop()
                if (op == "{" and ch != "}") or (op == "[" and ch != "]"):
                    return None
                if not stack:
                    return payload[start : i + 1]
    return None


def _parse_json_or_raise(stdout: str, context: str) -> Any:
    raw = (stdout or "").strip()
    js = _json_extract_first(raw)
    if not js:
        raise RuntimeError(f"{context} returned non-JSON output. stdout_head={raw[:500]!r}")
    try:
        return json.loads(js)
    except Exception as e:
        raise RuntimeError(f"{context} returned JSON but parsing failed: {e}. json_head={js[:500]!r}")


# ----------------------------
# DB backlog (stdlib-first)
# ----------------------------

def _db_connect():
    """
    Supports:
    - sqlite:///path
    - postgres:// / postgresql:// via psycopg2 or psycopg (if installed)
    """
    db_url = os.getenv("DATABASE_URL", "").strip()
    if not db_url:
        raise RuntimeError("DATABASE_URL missing")

    if db_url.startswith("sqlite:"):
        import sqlite3
        # sqlite:////data/db.sqlite -> /data/db.sqlite
        path = db_url.split("sqlite:", 1)[1]
        path = path.lstrip("/")
        path = "/" + path
        conn = sqlite3.connect(path)
        return ("sqlite", conn)

    # Postgres
    scheme = urlparse(db_url).scheme.lower()
    if scheme in ("postgres", "postgresql"):
        try:
            import psycopg2  # type: ignore
            conn = psycopg2.connect(db_url)
            return ("psycopg2", conn)
        except Exception:
            pass
        try:
            import psycopg  # type: ignore
            conn = psycopg.connect(db_url)
            return ("psycopg", conn)
        except Exception as e:
            raise RuntimeError(f"Postgres driver missing (install psycopg2-binary or psycopg): {e}")

    raise RuntimeError(f"Unsupported DATABASE_URL scheme: {db_url}")


def _db_scalar(conn_kind: str, conn, sql: str, params: Tuple[Any, ...] = ()) -> int:
    if conn_kind == "sqlite":
        cur = conn.cursor()
        cur.execute(sql, params)
        row = cur.fetchone()
        return int(row[0] if row and row[0] is not None else 0)

    # psycopg2/psycopg
    cur = conn.cursor()
    cur.execute(sql, params)
    row = cur.fetchone()
    return int(row[0] if row and row[0] is not None else 0)


def backlog_counts() -> Tuple[int, int]:
    """
    Returns (unclaimed_backlog, inflight_claimed).
    We only scale on unclaimed_backlog per your rule.
    """
    kind, conn = _db_connect()
    try:
        # Be permissive: some jobs might be QUEUED, some marked RUNNING immediately.
        unclaimed = _db_scalar(
            kind, conn,
            "SELECT COUNT(*) FROM jobs WHERE executor_id IS NULL AND status IN ('QUEUED','RUNNING')"
        )
        inflight = _db_scalar(
            kind, conn,
            "SELECT COUNT(*) FROM jobs WHERE executor_id IS NOT NULL AND status IN ('QUEUED','RUNNING')"
        )
        return unclaimed, inflight
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ----------------------------
# Hyperbolic CLI wrapper
# ----------------------------

@dataclass
class CmdResult:
    rc: int
    stdout: str
    stderr: str


def _hyperbolic_env() -> Dict[str, str]:
    """
    Ensure CLI has a stable config home so it can find auth config.
    We do NOT rely on whatever $HOME is in Fly; we force it.
    """
    env = dict(os.environ)

    # Stable writable config home
    env["HOME"] = env.get("HOME", "/root")
    env["XDG_CONFIG_HOME"] = env.get("XDG_CONFIG_HOME", "/tmp/hyperbolic-cli-config")
    env["XDG_CACHE_HOME"] = env.get("XDG_CACHE_HOME", "/tmp/hyperbolic-cli-cache")

    # Optional base URL override (only helps if CLI supports it)
    if HYPERBOLIC_API_BASE_URL:
        env["HYPERBOLIC_API_BASE_URL"] = HYPERBOLIC_API_BASE_URL

    return env


def run_hcli(args: List[str], timeout: int = 60) -> CmdResult:
    cmd = [_which(HYPERBOLIC_BIN)] + args
    p = subprocess.run(
        cmd,
        env=_hyperbolic_env(),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return CmdResult(p.returncode, p.stdout or "", p.stderr or "")


def ensure_hcli_auth() -> None:
    """
    Idempotently write config using the API key.
    """
    if not HYPERBOLIC_API_KEY:
        raise RuntimeError("HYPERBOLIC_API_KEY missing in env")

    # set-key is safe to run repeatedly
    r = run_hcli(["auth", "set-key", HYPERBOLIC_API_KEY], timeout=30)
    if r.rc != 0:
        raise RuntimeError(f"hyperbolic auth set-key failed rc={r.rc} stdout={_redact(r.stdout)!r} stderr={_redact(r.stderr)!r}")


def list_instances() -> List[Dict[str, Any]]:
    ensure_hcli_auth()
    r = run_hcli(["instances", "--json"], timeout=60)
    if r.rc != 0:
        raise RuntimeError(f"hyperbolic instances --json failed rc={r.rc} stdout={_redact(r.stdout,500)!r} stderr={_redact(r.stderr,500)!r}")
    j = _parse_json_or_raise(r.stdout, "hyperbolic instances --json")

    # The CLI may return either a list or an object containing list.
    if isinstance(j, list):
        return j
    if isinstance(j, dict):
        # common patterns
        for k in ("instances", "data", "result"):
            if k in j and isinstance(j[k], list):
                return j[k]
    # fallback: treat as empty
    return []


def terminate_instance(instance_id: str) -> None:
    ensure_hcli_auth()
    r = run_hcli(["terminate", instance_id], timeout=60)
    if r.rc != 0:
        raise RuntimeError(f"terminate failed id={instance_id} rc={r.rc} stdout={_redact(r.stdout,500)!r} stderr={_redact(r.stderr,500)!r}")


def rent_vm(gpu_count: int) -> Tuple[str, Optional[str]]:
    """
    Returns (instance_id, ssh_command_if_present).
    """
    ensure_hcli_auth()
    gpu_count = int(gpu_count)
    args = ["rent", "ondemand", "--instance-type", "virtual-machine", "--gpu-count", str(gpu_count)]
    r = run_hcli(args, timeout=120)

    out = (r.stdout or "") + "\n" + (r.stderr or "")
    if r.rc != 0:
        raise RuntimeError(f"rent failed rc={r.rc} out_head={_redact(out,500)!r}")

    # Try to extract instance_id
    instance_id = ""
    patterns = [
        r'instance[_\s-]*id["\s:=]+([a-zA-Z0-9\-]+)',
        r'Instance\s*ID\s*[:=]\s*([a-zA-Z0-9\-]+)',
        r'"instance_id"\s*:\s*"([^"]+)"',
    ]
    for pat in patterns:
        m = re.search(pat, out, flags=re.IGNORECASE)
        if m:
            instance_id = m.group(1).strip()
            break
    if not instance_id:
        # Sometimes CLI prints a full JSON blob; try parse JSON fragment
        try:
            j = _parse_json_or_raise(out, "hyperbolic rent")
            if isinstance(j, dict):
                instance_id = str(j.get("instance_id") or j.get("id") or "")
        except Exception:
            pass

    # Try extract ssh command line
    ssh_cmd = None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("ssh "):
            ssh_cmd = line
            break

    if not instance_id:
        raise RuntimeError(f"Could not determine instance_id after rent. rent_stdout_head={_redact(out,500)!r}")

    return instance_id, ssh_cmd


# ----------------------------
# SSH bootstrap
# ----------------------------

def _write_temp_ssh_key() -> str:
    if not SSH_PRIVATE_KEY.strip():
        raise RuntimeError("SSH_PRIVATE_KEY missing in env")
    fd, path = tempfile.mkstemp(prefix="hyperbolic_ssh_", text=True)
    os.write(fd, SSH_PRIVATE_KEY.encode("utf-8"))
    os.close(fd)
    os.chmod(path, 0o600)
    return path


def _ssh_run(ssh_cmd: str, remote_script: str, timeout: int = 600) -> None:
    """
    Run a remote bash script using an ssh command we got from Hyperbolic CLI output.
    We append our own options to avoid prompts.
    """
    key_path = _write_temp_ssh_key()
    try:
        base = shlex.split(ssh_cmd)

        # If the ssh command already includes -i, keep it; else add ours.
        if "-i" not in base:
            base += ["-i", key_path]

        # Avoid host key prompts in Fly
        base += [
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=10",
        ]

        # Execute the remote script
        remote = "bash -lc " + shlex.quote(remote_script)
        cmd = base + [remote]

        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if p.returncode != 0:
            raise RuntimeError(
                f"ssh bootstrap failed rc={p.returncode} stdout={_redact(p.stdout,800)!r} stderr={_redact(p.stderr,800)!r}"
            )
    finally:
        try:
            os.remove(key_path)
        except Exception:
            pass


def bootstrap_vm_over_ssh(ssh_cmd: str) -> None:
    """
    Installs executor loop script + systemd unit + secrets on the VM.
    """
    if not EXECUTOR_TOKEN:
        raise RuntimeError("EXECUTOR_TOKEN missing in env (needed on VM)")

    # Remote script: careful quoting, NO python f-string injection inside the worker source.
    script = textwrap.dedent(f"""
        set -euo pipefail

        sudo mkdir -p /data/secrets /data/bin

        # B2 env
        sudo bash -c 'cat > /data/secrets/b2.env << "EOF"
B2_SYNC={B2_SYNC}
B2_BUCKET={B2_BUCKET}
B2_S3_ENDPOINT={B2_S3_ENDPOINT}
AWS_ACCESS_KEY_ID={AWS_ACCESS_KEY_ID}
AWS_SECRET_ACCESS_KEY={AWS_SECRET_ACCESS_KEY}
AWS_DEFAULT_REGION={AWS_DEFAULT_REGION}
EOF'
        sudo chmod 600 /data/secrets/b2.env

        # Executor env
        sudo bash -c 'cat > /data/secrets/hyper_executor.env << "EOF"
EXECUTOR_TOKEN="{EXECUTOR_TOKEN}"
BRAIN_URL="{BRAIN_URL}"
ASSIGNED_WORKER_ID="{ASSIGNED_WORKER_ID}"
POLL_SECONDS=3
IDLE_SECONDS=3600
EXECUTOR_ID="exec-$(hostname)"
EOF'
        sudo chmod 600 /data/secrets/hyper_executor.env

        # Download loop script
        sudo curl -fsSL {shlex.quote(EXECUTOR_LOOP_URL)} -o /data/bin/hyper_executor_loop.sh
        sudo chmod +x /data/bin/hyper_executor_loop.sh

        # systemd unit
        sudo tee /etc/systemd/system/hyper-executor.service >/dev/null << "EOF"
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

        echo "BOOTSTRAP_OK"
    """).strip()

    _ssh_run(ssh_cmd, script, timeout=900)


# ----------------------------
# Autoscaler logic
# ----------------------------

def compute_want_vms(unclaimed_backlog: int) -> int:
    if unclaimed_backlog <= 0:
        return 0
    return int(math.ceil(unclaimed_backlog / float(BACKLOG_PER_VM)))


def filter_active_instances(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep only "active" instances. The exact schema depends on CLI.
    We attempt common keys: status/state and instance id keys.
    """
    out = []
    for inst in raw:
        if not isinstance(inst, dict):
            continue
        status = str(inst.get("status") or inst.get("state") or "").upper()
        # If status missing, keep it (better than terminating incorrectly)
        if status and status not in ("RUNNING", "ACTIVE", "STARTED"):
            continue
        if INSTANCE_NAME_PREFIX:
            name = str(inst.get("name") or inst.get("instance_name") or "")
            if not name.startswith(INSTANCE_NAME_PREFIX):
                continue
        out.append(inst)
    return out


def get_instance_id(inst: Dict[str, Any]) -> str:
    return str(inst.get("instance_id") or inst.get("id") or inst.get("instanceId") or "").strip()


def get_instance_ssh(inst: Dict[str, Any]) -> Optional[str]:
    """
    If CLI JSON includes ssh command or connection string, use it.
    Otherwise return None.
    """
    for k in ("ssh", "ssh_command", "sshCommand", "connect", "connection"):
        v = inst.get(k)
        if isinstance(v, str) and v.strip().startswith("ssh "):
            return v.strip()
    return None


def autoscaler_tick() -> None:
    unclaimed, inflight = backlog_counts()
    want = compute_want_vms(unclaimed)

    # If Hyperbolic list API is broken (404), we still want clean logs and no crash-loop.
    have = 0
    instances: List[Dict[str, Any]] = []
    try:
        instances = filter_active_instances(list_instances())
        have = len(instances)
    except Exception as e:
        # IMPORTANT: don’t crash the worker; log and proceed to rent attempts (or termination if want=0)
        log.error(f"autoscaler error listing instances: {e}")
        instances = []
        have = 0

    log.info(f"backlog={unclaimed} inflight={inflight} want_vms={want} have_vms={have}")

    # No backlog => terminate everything we can see (only if we can list).
    if want == 0:
        if have == 0:
            return
        # If jobs are still inflight, do not kill instances.
        if inflight > 0:
            log.info("inflight>0; not terminating instances yet")
            return

        for inst in instances:
            iid = get_instance_id(inst)
            if not iid:
                continue
            try:
                log.info(f"terminating instance {iid} (no backlog)")
                terminate_instance(iid)
            except Exception as e:
                log.error(f"terminate failed for {iid}: {e}")
        return

    # Need more instances
    if have >= want:
        return

    need = want - have
    log.info(f"need {need} more VM(s)")

    for _ in range(need):
        last_err = None
        instance_id = None
        ssh_cmd = None

        for g in GPU_FALLBACKS:
            if g > MAX_GPUS:
                continue
            try:
                log.info(f"renting {g}x GPU VM on Hyperbolic...")
                instance_id, ssh_cmd = rent_vm(g)
                log.info(f"rented instance_id={instance_id}")

                # Bootstrap if we have ssh command
                if ssh_cmd:
                    log.info(f"bootstrapping instance_id={instance_id} over SSH...")
                    bootstrap_vm_over_ssh(ssh_cmd)
                    log.info(f"bootstrap complete for instance_id={instance_id}")
                else:
                    log.warning(f"no ssh command found in rent output for instance_id={instance_id}; bootstrap skipped")
                break
            except Exception as e:
                last_err = e
                log.info(f"rent attempt failed for gpu_count={g}: {e}")

        if not instance_id:
            raise RuntimeError(f"autoscaler error: could not rent any VM size: {last_err}")


def main() -> None:
    # Validate required env early (so failures are obvious)
    missing = []
    if not os.getenv("DATABASE_URL"):
        missing.append("DATABASE_URL")
    if not HYPERBOLIC_API_KEY:
        missing.append("HYPERBOLIC_API_KEY")
    if not EXECUTOR_TOKEN:
        missing.append("EXECUTOR_TOKEN")
    if not SSH_PRIVATE_KEY.strip():
        missing.append("SSH_PRIVATE_KEY")
    if missing:
        raise SystemExit(f"Missing required env: {', '.join(missing)}")

    # Basic sanity: print CLI location (helps debugging container builds)
    log.info(f"using hyperbolic bin: {_which(HYPERBOLIC_BIN)}")
    if HYPERBOLIC_API_BASE_URL:
        log.info(f"HYPERBOLIC_API_BASE_URL override set")

    # Loop
    stop = False

    def _handle(_sig, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)

    while not stop:
        try:
            autoscaler_tick()
        except Exception as e:
            log.error(f"autoscaler error: {e}")
        time.sleep(TICK_SECONDS)


if __name__ == "__main__":
    main()
