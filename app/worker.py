# app/worker.py
from __future__ import annotations

import json
import logging
import math
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psycopg


LOG = logging.getLogger("brain-worker")

# -----------------------------
# Config
# -----------------------------
DB_URL = os.environ.get("DATABASE_URL", "")
ASSIGNED_WORKER_ID = os.environ.get("ASSIGNED_WORKER_ID")  # optional
BACKLOG_PER_VM = int(os.environ.get("BACKLOG_PER_VM", "250"))
MAX_VMS = int(os.environ.get("MAX_VMS", "4"))  # safety cap

# Hyperbolic
HYPERBOLIC_KEY = (
    os.environ.get("HYPERBOLIC_API_KEY")
    or os.environ.get("HYPERBOLIC_KEY")
    or os.environ.get("HYPERBOLIC_TOKEN")
    or ""
).strip()

# SSH
SSH_PRIVATE_KEY = os.environ.get("SSH_PRIVATE_KEY", "")
HYPERBOLIC_SSH_USER = os.environ.get("HYPERBOLIC_SSH_USER", "root")

# Executor bootstrap
BRAIN_URL = os.environ.get("BRAIN_URL", "https://hyper-brain.fly.dev").rstrip("/")
EXECUTOR_TOKEN = os.environ.get("EXECUTOR_TOKEN", "")
EXECUTOR_LOOP_PATH = "/data/bin/hyper_executor_loop.sh"
SERVICE_NAME = "hyper-executor"

STATE_DIR = Path(os.environ.get("STATE_DIR", "/tmp/brain-state"))
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "hyperbolic_instances.json"

HCLI_HOME = Path(os.environ.get("HCLI_HOME", "/tmp/hyperbolic-cli-home"))
HCLI_HOME.mkdir(parents=True, exist_ok=True)

TICK_SECONDS = float(os.environ.get("TICK_SECONDS", "4"))

# Candidate GPU counts to try when renting (smallest first).
GPU_COUNT_CANDIDATES = [1, 2, 4]

# Embedded executor loop script (written to the rented VM at /data/bin/hyper_executor_loop.sh)
# NOTE: this is pulled from your uploaded hyper_executor_loop.sh
HYPER_EXECUTOR_LOOP_SH = '#!/usr/bin/env bash\nset -euo pipefail\n\n# hyper_executor_loop.sh\n# Long-running executor loop:\n# - polls Brain (/executors/claim) for a job\n# - downloads inputs/models from B2 (S3-compatible)\n# - runs the right docker image for the job\n# - uploads outputs back to B2\n# - reports completion to Brain\n#\n# Requires env in /data/secrets/hyper_executor.env and /data/secrets/b2.env\n\nlog() { echo \"[$(date -u +\"%Y-%m-%dT%H:%M:%SZ\")] $*\"; }\n\n# Load env files if present\nif [[ -f \"/data/secrets/hyper_executor.env\" ]]; then\n  # shellcheck disable=SC1091\n  source \"/data/secrets/hyper_executor.env\"\nfi\n\nif [[ -f \"/data/secrets/b2.env\" ]]; then\n  # shellcheck disable=SC1091\n  source \"/data/secrets/b2.env\"\nfi\n\n: \"${BRAIN_URL:?BRAIN_URL is required}\"\n: \"${EXECUTOR_TOKEN:?EXECUTOR_TOKEN is required}\"\n\nASSIGNED_WORKER_ID=\"${ASSIGNED_WORKER_ID:-}\"\n\n# Optional tuning\nPOLL_SECONDS=\"${POLL_SECONDS:-2}\"\nIDLE_SHUTDOWN_SECONDS=\"${IDLE_SHUTDOWN_SECONDS:-900}\"\n\nlast_job_ts=$(date +%s)\n\nclaim_job() {\n  local url=\"$BRAIN_URL/executors/claim\"\n\n  # If you’re using ASSIGNED_WORKER_ID routing, send it.\n  if [[ -n \"$ASSIGNED_WORKER_ID\" ]]; then\n    curl -fsS -X POST \"$url\" \\\n      -H \"Authorization: Bearer $EXECUTOR_TOKEN\" \\\n      -H \"Content-Type: application/json\" \\\n      -d \"{\\\"assigned_worker_id\\\":\\\"$ASSIGNED_WORKER_ID\\\"}\" || return 1\n  else\n    curl -fsS -X POST \"$url\" \\\n      -H \"Authorization: Bearer $EXECUTOR_TOKEN\" \\\n      -H \"Content-Type: application/json\" \\\n      -d \"{}\" || return 1\n  fi\n}\n\nreport_done() {\n  local job_id=\"$1\"\n  local status=\"$2\"\n  local message=\"$3\"\n\n  curl -fsS -X POST \"$BRAIN_URL/executors/complete\" \\\n    -H \"Authorization: Bearer $EXECUTOR_TOKEN\" \\\n    -H \"Content-Type: application/json\" \\\n    -d \"{\\\"job_id\\\":\\\"$job_id\\\",\\\"status\\\":\\\"$status\\\",\\\"message\\\":\\\"$message\\\"}\" >/dev/null\n}\n\nwhile true; do\n  now=$(date +%s)\n  idle=$((now - last_job_ts))\n\n  if (( idle > IDLE_SHUTDOWN_SECONDS )); then\n    log \"Idle for ${idle}s; powering off\"\n    sudo poweroff || true\n    sleep 10\n  fi\n\n  resp=\"\"\n  if ! resp=$(claim_job 2>/dev/null); then\n    sleep \"$POLL_SECONDS\"\n    continue\n  fi\n\n  # Expect JSON, either {\"job\":null} or {\"job\":{...}}\n  job=$(echo \"$resp\" | python3 -c 'import sys,json; j=json.load(sys.stdin); print(json.dumps(j.get(\"job\")))')\n\n  if [[ \"$job\" == \"null\" ]]; then\n    sleep \"$POLL_SECONDS\"\n    continue\n  fi\n\n  last_job_ts=$(date +%s)\n  job_id=$(echo \"$job\" | python3 -c 'import sys,json; j=json.load(sys.stdin); print(j.get(\"job_id\") or j.get(\"id\") or \"\")')\n  job_type=$(echo \"$job\" | python3 -c 'import sys,json; j=json.load(sys.stdin); print(j.get(\"job_type\") or \"\")')\n\n  log \"Claimed job_id=$job_id type=$job_type\"\n\n  # TODO: your actual run logic here (download -> docker run -> upload)\n  # For now, mark done so autoscaling + claiming is exercised.\n  report_done \"$job_id\" \"SUCCEEDED\" \"stub executor: completed\"\n\n  sleep 0.2\ndone\n'

SYSTEMD_UNIT = f"""[Unit]
Description=Hyper Executor Loop
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
Restart=always
RestartSec=2
EnvironmentFile=-/data/secrets/hyper_executor.env
EnvironmentFile=-/data/secrets/b2.env
ExecStart={EXECUTOR_LOOP_PATH}
WorkingDirectory=/data
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""


@dataclass
class CmdResult:
    args: List[str]
    rc: int
    stdout: str
    stderr: str


def _sh(cmd: List[str], *, env: Optional[Dict[str, str]] = None, timeout: int = 120) -> CmdResult:
    p = subprocess.run(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    return CmdResult(cmd, p.returncode, p.stdout or "", p.stderr or "")


def _head(s: str, n: int = 400) -> str:
    s = (s or "").replace("\r", "")
    return s[:n]


def _load_state() -> Dict[str, Any]:
    if not STATE_FILE.exists():
        return {"instances": []}
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {"instances": []}


def _save_state(state: Dict[str, Any]) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True))


def _hcli_env() -> Dict[str, str]:
    env = dict(os.environ)
    # Force Hyperbolic CLI to read/write config to a writable location.
    env["HOME"] = str(HCLI_HOME)
    env["XDG_CONFIG_HOME"] = str(HCLI_HOME)
    env["XDG_CACHE_HOME"] = str(HCLI_HOME)
    return env


def ensure_hyperbolic_auth() -> None:
    """Save API key to Hyperbolic CLI config (needed because some subcommands ignore env vars)."""
    if not HYPERBOLIC_KEY:
        raise RuntimeError("HYPERBOLIC_API_KEY is not set in env.")
    env = _hcli_env()
    r = _sh(["hyperbolic", "auth", "set-key", HYPERBOLIC_KEY], env=env, timeout=60)
    if r.rc != 0 and "saved" not in r.stdout.lower():
        raise RuntimeError(
            f"hyperbolic auth set-key failed rc={r.rc} stdout={_head(r.stdout)!r} stderr={_head(r.stderr)!r}"
        )


def parse_json_or_raise(output: str, context: str) -> Any:
    s = (output or "").strip()
    try:
        return json.loads(s)
    except Exception:
        raise RuntimeError(f"{context} returned non-JSON output: {s[:700]!r}")


def list_instances() -> List[Dict[str, Any]]:
    env = _hcli_env()
    r = _sh(["hyperbolic", "instances", "--json"], env=env, timeout=60)
    if r.rc != 0:
        raise RuntimeError(
            "hyperbolic instances --json failed. "
            f"rc={r.rc} stdout={_head(r.stdout,800)!r} stderr={_head(r.stderr,800)!r}"
        )
    j = parse_json_or_raise(r.stdout, "hyperbolic instances --json")
    if isinstance(j, dict) and "instances" in j and isinstance(j["instances"], list):
        return j["instances"]
    if isinstance(j, list):
        return j
    return []


def _instance_id_from_obj(obj: Dict[str, Any]) -> Optional[str]:
    for k in ("id", "instance_id", "instanceId", "instanceID"):
        if k in obj:
            return str(obj[k])
    return None


def _instance_ssh_from_obj(obj: Dict[str, Any]) -> Optional[str]:
    for k in ("ssh_command", "sshCommand", "ssh", "ssh_cmd"):
        v = obj.get(k)
        if isinstance(v, str) and "ssh " in v:
            return v.strip()
    return None


def _extract_ssh_command(text: str) -> Optional[str]:
    for line in (text or "").splitlines():
        if line.strip().startswith("ssh "):
            return line.strip()
    m = re.search(r"(ssh\s+[^\n]+)", text or "")
    return m.group(1).strip() if m else None


def rent_vm(gpu_count: int) -> Tuple[str, Optional[str]]:
    """Rent a VM and return (instance_id, ssh_command)."""
    env = _hcli_env()

    before: List[Dict[str, Any]] = []
    try:
        before = list_instances()
    except Exception as e:
        LOG.warning("list_instances before rent failed (will fallback to stdout parsing): %s", e)

    before_ids = {_instance_id_from_obj(x) for x in before if _instance_id_from_obj(x)}

    r = _sh(
        ["hyperbolic", "rent", "ondemand", "--instance-type", "virtual-machine", "--gpu-count", str(gpu_count)],
        env=env,
        timeout=180,
    )

    if r.rc != 0:
        raise RuntimeError(f"rent rc={r.rc} stdout={_head(r.stdout,800)!r} stderr={_head(r.stderr,800)!r}")

    # Prefer diffing instances (most reliable across CLI output formats)
    try:
        after = list_instances()
        after_ids = {_instance_id_from_obj(x) for x in after if _instance_id_from_obj(x)}
        new_ids = [i for i in after_ids if i and i not in before_ids]
        if new_ids:
            def _as_int(s: str) -> int:
                try:
                    return int(s)
                except Exception:
                    return -1

            new_id = sorted(new_ids, key=_as_int, reverse=True)[0]
            ssh_cmd = None
            for inst in after:
                if _instance_id_from_obj(inst) == new_id:
                    ssh_cmd = _instance_ssh_from_obj(inst)
                    break
            return new_id, ssh_cmd
    except Exception as e:
        LOG.warning("list_instances after rent failed (will fallback to stdout parsing): %s", e)

    # Fallback: parse stdout for ID and SSH command
    m = re.search(r"\b(?:instance\s*id|id)\s*[:=]\s*([0-9]+)\b", r.stdout, re.IGNORECASE)
    if m:
        return m.group(1), _extract_ssh_command(r.stdout)

    for line in (r.stdout or "").splitlines():
        if re.fullmatch(r"\s*[0-9]{4,}\s*", line):
            return line.strip(), _extract_ssh_command(r.stdout)

    raise RuntimeError(f"Could not determine instance_id after rent. rent_stdout_head={_head(r.stdout,900)!r}")


def terminate_vm(instance_id: str) -> None:
    env = _hcli_env()
    r = _sh(["hyperbolic", "terminate", str(instance_id)], env=env, timeout=60)
    if r.rc != 0:
        raise RuntimeError(
            f"terminate rc={r.rc} stdout={_head(r.stdout,800)!r} stderr={_head(r.stderr,800)!r}"
        )


def _write_private_key() -> Path:
    if not SSH_PRIVATE_KEY.strip():
        raise RuntimeError("SSH_PRIVATE_KEY is missing; cannot bootstrap executor VM.")
    key_path = STATE_DIR / "id_ed25519"
    key_path.write_text(SSH_PRIVATE_KEY.strip() + "\n")
    os.chmod(key_path, 0o600)
    return key_path


def _build_env_file_lines(prefixes: Tuple[str, ...]) -> str:
    lines: List[str] = []
    for k, v in os.environ.items():
        if any(k.startswith(p) for p in prefixes) and v != "":
            vv = v.replace("\n", "\\n")
            lines.append(f"{k}={vv}")
    return "\n".join(sorted(lines)) + "\n"


def _ssh_tokens_from_cmd(ssh_cmd: str, key_path: Path) -> List[str]:
    tokens = shlex.split(ssh_cmd)
    if not tokens or tokens[0] != "ssh":
        raise RuntimeError(f"Unexpected ssh_command format: {ssh_cmd!r}")

    # Ensure -i <key>
    if "-i" in tokens:
        i = tokens.index("-i")
        if i + 1 < len(tokens):
            tokens[i + 1] = str(key_path)
        else:
            tokens.extend([str(key_path)])
    else:
        tokens.extend(["-i", str(key_path)])

    # Ensure non-interactive host key behavior
    tokens.extend(["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"])
    return tokens


def bootstrap_executor_over_ssh(ssh_cmd: Optional[str], *, instance_id: str) -> None:
    """Bootstraps the executor systemd service on the rented VM."""
    key_path = _write_private_key()

    if ssh_cmd:
        ssh_tokens = _ssh_tokens_from_cmd(ssh_cmd, key_path)
    else:
        host = os.environ.get("HYPERBOLIC_SSH_HOST", "").strip()
        if not host:
            raise RuntimeError("No ssh_command from Hyperbolic CLI and HYPERBOLIC_SSH_HOST not set.")
        ssh_tokens = [
            "ssh",
            "-i",
            str(key_path),
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            f"{HYPERBOLIC_SSH_USER}@{host}",
        ]

    if not EXECUTOR_TOKEN:
        raise RuntimeError("EXECUTOR_TOKEN missing; executor cannot authenticate to brain.")

    hyper_env = "\n".join(
        [
            f"BRAIN_URL={BRAIN_URL}",
            f"EXECUTOR_TOKEN={EXECUTOR_TOKEN}",
            f"ASSIGNED_WORKER_ID={ASSIGNED_WORKER_ID or ''}",
            f"INSTANCE_ID={instance_id}",
            "",
        ]
    )
    b2_env = _build_env_file_lines(("B2_", "AWS_"))

    remote = f"""set -euo pipefail
sudo mkdir -p /data/bin /data/secrets
sudo tee {EXECUTOR_LOOP_PATH} >/dev/null <<'EOF'
{HYPER_EXECUTOR_LOOP_SH}
EOF
sudo chmod +x {EXECUTOR_LOOP_PATH}

sudo tee /data/secrets/hyper_executor.env >/dev/null <<'EOF'
{hyper_env}
EOF

sudo tee /data/secrets/b2.env >/dev/null <<'EOF'
{b2_env}
EOF

sudo tee /etc/systemd/system/{SERVICE_NAME}.service >/dev/null <<'EOF'
{SYSTEMD_UNIT}
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now {SERVICE_NAME}
sudo systemctl is-active --quiet {SERVICE_NAME}
echo "{SERVICE_NAME} started"
"""

    # Run: ssh ... bash -lc "<remote script>"
    cmd = ssh_tokens + ["bash", "-lc", remote]
    r = _sh(cmd, timeout=300)
    if r.rc != 0:
        raise RuntimeError(
            f"bootstrap failed rc={r.rc} stdout={_head(r.stdout,800)!r} stderr={_head(r.stderr,800)!r}"
        )


def get_backlog_count() -> int:
    if not DB_URL:
        raise RuntimeError("DATABASE_URL is not set.")
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            if ASSIGNED_WORKER_ID:
                try:
                    cur.execute(
                        "SELECT COUNT(*) FROM jobs WHERE status='RUNNING' AND executor_id IS NULL AND assigned_worker_id=%s",
                        (ASSIGNED_WORKER_ID,),
                    )
                    row = cur.fetchone()
                    return int(row[0] if row else 0)
                except Exception:
                    conn.rollback()
            cur.execute("SELECT COUNT(*) FROM jobs WHERE status='RUNNING' AND executor_id IS NULL")
            row = cur.fetchone()
            return int(row[0] if row else 0)


def desired_vm_count(backlog: int) -> int:
    if backlog <= 0:
        return 0
    return min(MAX_VMS, max(1, math.ceil(backlog / BACKLOG_PER_VM)))


def autoscaler_tick() -> None:
    ensure_hyperbolic_auth()

    backlog = get_backlog_count()
    want = desired_vm_count(backlog)

    state = _load_state()
    known_ids = [str(x) for x in state.get("instances", []) if x is not None]

    # If Hyperbolic listing is currently broken (your 404), we still allow scale-up using our state,
    # but we refuse to scale down for safety.
    inst_ids: Optional[set] = None
    try:
        inst = list_instances()
        inst_ids = {_instance_id_from_obj(x) for x in inst if _instance_id_from_obj(x)}
    except Exception as e:
        LOG.error("list_instances failed (will not scale down): %s", e)

    live_ids = known_ids if inst_ids is None else [i for i in known_ids if i in inst_ids]
    have = len(live_ids)

    LOG.info("backlog=%s want_vms=%s have_vms=%s", backlog, want, have)

    # Scale up
    if have < want:
        need = want - have
        for _ in range(need):
            last_err: Optional[Exception] = None
            for gpu in GPU_COUNT_CANDIDATES:
                LOG.info("renting %sx GPU VM on Hyperbolic...", gpu)
                try:
                    instance_id, ssh_cmd = rent_vm(gpu)
                    LOG.info("rented instance_id=%s", instance_id)

                    live_ids.append(instance_id)
                    state["instances"] = live_ids
                    _save_state(state)

                    # bootstrap best-effort
                    try:
                        bootstrap_executor_over_ssh(ssh_cmd, instance_id=instance_id)
                    except Exception as be:
                        LOG.error("bootstrap failed for instance_id=%s: %s", instance_id, be)

                    break
                except Exception as e:
                    last_err = e
                    LOG.info("rent attempt failed for gpu_count=%s: %s", gpu, e)
            else:
                raise RuntimeError(f"could not rent any VM size: {last_err}")

    # Scale down only if we can list instances (safety)
    elif have > want and inst_ids is not None:
        extra = have - want
        to_kill = list(reversed(live_ids))[:extra]
        for instance_id in to_kill:
            LOG.info("terminating instance_id=%s", instance_id)
            terminate_vm(instance_id)
            live_ids.remove(instance_id)
        state["instances"] = live_ids
        _save_state(state)


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s][brain-worker] %(message)s",
    )

    if not DB_URL:
        LOG.error("DATABASE_URL missing; worker cannot run.")
        return
    if not HYPERBOLIC_KEY:
        LOG.error("HYPERBOLIC_API_KEY missing; autoscaler cannot rent/terminate.")
        return

    LOG.info("worker started. assigned_worker_id=%s backlog_per_vm=%s", ASSIGNED_WORKER_ID, BACKLOG_PER_VM)

    while True:
        try:
            autoscaler_tick()
        except Exception:
            LOG.exception("autoscaler error")
        time.sleep(TICK_SECONDS)


if __name__ == "__main__":
    main()
