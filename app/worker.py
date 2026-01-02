# worker.py
import json
import logging
import os
import re
import shlex
import subprocess
import time
from datetime import datetime, timezone
from math import ceil
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("brain-worker")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s][brain-worker] %(message)s")

# =========================
# Autoscaling policy (your rule)
# =========================
BACKLOG_PER_VM = int(os.getenv("BACKLOG_PER_VM", "250"))  # 250, 500, 750...
POLL_SECONDS = int(os.getenv("AUTOSCALER_POLL_SECONDS", "4"))

# Prefer 1 > 2 > 4 GPUs for a VM (you said 1 job per VM for now)
GPU_CHOICES = [1, 2, 4]

# Hyperbolic CLI commands
HYPERBOLIC_BIN = os.getenv("HYPERBOLIC_BIN", "hyperbolic")

# This is the single biggest fix:
# Force Hyperbolic CLI to always read/write config in one predictable place.
HCLI_HOME = os.getenv("HYPERBOLIC_CLI_HOME", "/tmp/hyperbolic-cli-home")

# Optional: run bootstrap over SSH after renting
ENABLE_BOOTSTRAP = os.getenv("ENABLE_VM_BOOTSTRAP", "1") == "1"

# Your Brain URL & tokens (passed to VM executor in bootstrap)
BRAIN_URL = os.getenv("BRAIN_URL", "https://hyper-brain.fly.dev")
EXECUTOR_TOKEN = os.getenv("EXECUTOR_TOKEN", "")
ASSIGNED_WORKER_ID = os.getenv("ASSIGNED_WORKER_ID", "hyperbolic-pool")

# B2 env that will be written onto the VM (you said /data/secrets/hyperbolic.env is fine;
# on the VM we write /data/secrets/b2.env and /data/secrets/hyper_executor.env)
B2_SYNC = os.getenv("B2_SYNC", "")
B2_BUCKET = os.getenv("B2_BUCKET", "")
B2_S3_ENDPOINT = os.getenv("B2_S3_ENDPOINT", "")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "")

# SSH private key for bootstrapping VMs
SSH_PRIVATE_KEY = os.getenv("SSH_PRIVATE_KEY", "")
HYPERBOLIC_SSH_USER = os.getenv("HYPERBOLIC_SSH_USER", "user")

# Where executor loop script is pulled from on the VM
HYPER_EXECUTOR_LOOP_URL = os.getenv(
    "HYPER_EXECUTOR_LOOP_URL",
    "https://raw.githubusercontent.com/Showstopper902/hyperbolic_project/main/scripts/hyper_executor_loop.sh",
)

# =========================
# DB backlog query
# =========================
# This worker queries your DB directly using DATABASE_URL with SQLAlchemy if present.
# If your project already has db helpers, you can swap this section later.
DATABASE_URL = os.getenv("DATABASE_URL", "")


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env_for_hyperbolic() -> Dict[str, str]:
    env = dict(os.environ)

    # Force config/cache dirs so set-key + later commands use the SAME config.
    env["HOME"] = HCLI_HOME
    env["XDG_CONFIG_HOME"] = HCLI_HOME
    env["XDG_CACHE_HOME"] = HCLI_HOME

    # Ensure directory exists (some environments don't create HOME automatically)
    try:
        os.makedirs(HCLI_HOME, exist_ok=True)
    except Exception:
        pass

    return env


def _run(cmd: List[str], timeout: int = 60, check: bool = False) -> subprocess.CompletedProcess:
    logger.debug("exec: %s", " ".join(shlex.quote(c) for c in cmd))
    cp = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_env_for_hyperbolic(),
        timeout=timeout,
    )
    if check and cp.returncode != 0:
        raise RuntimeError(
            f"Command failed rc={cp.returncode}: {' '.join(cmd)}\nstdout={cp.stdout}\nstderr={cp.stderr}"
        )
    return cp


def _extract_json_maybe(text: str) -> Optional[Any]:
    """
    Hyperbolic sometimes prints non-JSON errors.
    This tries to find the first JSON object/array inside output.
    """
    if not text:
        return None
    # quick path
    s = text.strip()
    if s.startswith("{") or s.startswith("["):
        try:
            return json.loads(s)
        except Exception:
            pass

    # find first { or [
    m = re.search(r"(\{|\[)", text)
    if not m:
        return None
    start = m.start()
    candidate = text[start:].strip()
    try:
        return json.loads(candidate)
    except Exception:
        return None


def _hyperbolic_api_key() -> str:
    # accept a few names just in case
    return (
        os.getenv("HYPERBOLIC_API_KEY")
        or os.getenv("HYPERBOLIC_KEY")
        or os.getenv("HYPERBOLIC_TOKEN")
        or ""
    )


def ensure_hyperbolic_auth() -> None:
    key = _hyperbolic_api_key()
    if not key:
        raise RuntimeError("Missing HYPERBOLIC_API_KEY in environment.")

    cp = _run([HYPERBOLIC_BIN, "auth", "set-key", key], timeout=30, check=False)
    if cp.returncode != 0:
        # Show heads only (avoid dumping secrets)
        raise RuntimeError(
            f"hyperbolic auth set-key failed rc={cp.returncode}. "
            f"stdout_head={cp.stdout[:300]!r} stderr_head={cp.stderr[:300]!r}"
        )


def hyperbolic_instances_json() -> List[Dict[str, Any]]:
    """
    Returns list of instances if JSON works; otherwise empty list, with a useful error.
    """
    cp = _run([HYPERBOLIC_BIN, "instances", "--json"], timeout=60, check=False)
    j = _extract_json_maybe(cp.stdout)

    if cp.returncode != 0 or j is None:
        # This is what you were seeing: non-JSON output or auth problems
        raise RuntimeError(
            "hyperbolic instances --json returned non-JSON output. "
            f"stdout={cp.stdout[:500]!r} stderr={cp.stderr[:500]!r} parse_err="
            f"{'no JSON object/array found in output' if j is None else 'n/a'}"
        )

    # Some CLIs return {"instances":[...]} vs [...]
    if isinstance(j, dict) and "instances" in j and isinstance(j["instances"], list):
        return j["instances"]
    if isinstance(j, list):
        return j
    if isinstance(j, dict):
        # best effort
        for k, v in j.items():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
    return []


def rent_vm(gpu_count: int) -> Tuple[str, str]:
    """
    Returns (instance_id, ssh_hint_text).
    ssh_hint_text may be empty if not printed by CLI.
    """
    cmd = [HYPERBOLIC_BIN, "rent", "ondemand", "--instance-type", "virtual-machine", "--gpu-count", str(gpu_count)]
    cp = _run(cmd, timeout=180, check=False)
    out = (cp.stdout or "") + "\n" + (cp.stderr or "")

    if cp.returncode != 0:
        raise RuntimeError(
            f"rent failed rc={cp.returncode}. "
            f"rent_stdout_head={cp.stdout[:400]!r} rent_stderr_head={cp.stderr[:400]!r}"
        )

    # Parse instance_id from output (best-effort)
    # Support uuid or opaque id
    m = re.search(r"(?:instance[_\s-]*id|Instance ID)\s*[:=]\s*([A-Za-z0-9._:-]+)", out, re.IGNORECASE)
    if not m:
        # maybe first UUID in output
        m2 = re.search(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", out, re.IGNORECASE)
        if m2:
            instance_id = m2.group(0)
        else:
            raise RuntimeError(
                "Could not determine instance_id after rent. "
                f"rent_stdout_head={cp.stdout[:400]!r} rent_stderr_head={cp.stderr[:400]!r}"
            )
    else:
        instance_id = m.group(1).strip()

    # Extract an ssh command line if present
    ssh_line = ""
    for line in out.splitlines():
        if line.strip().startswith("ssh "):
            ssh_line = line.strip()
            break

    return instance_id, ssh_line


def terminate_vm(instance_id: str) -> None:
    cp = _run([HYPERBOLIC_BIN, "terminate", instance_id], timeout=60, check=False)
    if cp.returncode != 0:
        raise RuntimeError(
            f"terminate failed rc={cp.returncode} stdout={cp.stdout[:300]!r} stderr={cp.stderr[:300]!r}"
        )


def _write_temp_ssh_key() -> str:
    if not SSH_PRIVATE_KEY.strip():
        raise RuntimeError("SSH_PRIVATE_KEY secret is missing; cannot bootstrap VM.")

    path = "/tmp/hyperbolic_ssh_key"
    with open(path, "w", encoding="utf-8") as f:
        f.write(SSH_PRIVATE_KEY.strip() + "\n")
    os.chmod(path, 0o600)
    return path


def _parse_ssh_target(ssh_line: str) -> Tuple[str, int]:
    """
    Very best-effort: returns (host, port).
    ssh_line example: ssh -p 2222 user@1.2.3.4
    """
    port = 22
    host = ""

    if not ssh_line:
        return host, port

    parts = shlex.split(ssh_line)
    for i, p in enumerate(parts):
        if p == "-p" and i + 1 < len(parts):
            try:
                port = int(parts[i + 1])
            except Exception:
                pass

    # last token usually user@host
    for tok in reversed(parts):
        if "@" in tok:
            host = tok.split("@", 1)[1]
            break
        # could be just host
        if re.match(r"^[A-Za-z0-9.\-]+$", tok):
            host = tok
            break

    return host, port


def bootstrap_vm_over_ssh(ssh_line: str) -> None:
    """
    Writes /data/secrets/b2.env and /data/secrets/hyper_executor.env,
    downloads hyper_executor_loop.sh, installs/enables systemd unit.
    """
    if not ENABLE_BOOTSTRAP:
        logger.info("bootstrap disabled (ENABLE_VM_BOOTSTRAP=0)")
        return

    if not ssh_line:
        raise RuntimeError("No ssh command/hint returned by rent; cannot bootstrap VM automatically yet.")

    ssh_key_path = _write_temp_ssh_key()
    host, port = _parse_ssh_target(ssh_line)
    if not host:
        raise RuntimeError(f"Could not parse host from ssh_line={ssh_line!r}")

    # Build remote bootstrap script
    # (No secrets are printed to logs; they are injected via SSH stdin.)
    remote = f"""#!/usr/bin/env bash
set -euo pipefail

sudo mkdir -p /data/secrets /data/bin

# b2.env
sudo bash -c 'cat > /data/secrets/b2.env <<EOF
B2_SYNC={shlex.quote(B2_SYNC)}
B2_BUCKET={shlex.quote(B2_BUCKET)}
B2_S3_ENDPOINT={shlex.quote(B2_S3_ENDPOINT)}
AWS_ACCESS_KEY_ID={shlex.quote(AWS_ACCESS_KEY_ID)}
AWS_SECRET_ACCESS_KEY={shlex.quote(AWS_SECRET_ACCESS_KEY)}
AWS_DEFAULT_REGION={shlex.quote(AWS_DEFAULT_REGION)}
EOF'
sudo chmod 600 /data/secrets/b2.env

# hyper_executor.env
sudo bash -c 'cat > /data/secrets/hyper_executor.env <<EOF
EXECUTOR_TOKEN={shlex.quote(EXECUTOR_TOKEN)}
BRAIN_URL={shlex.quote(BRAIN_URL)}
ASSIGNED_WORKER_ID={shlex.quote(ASSIGNED_WORKER_ID)}
POLL_SECONDS=3
IDLE_SECONDS=3600
EXECUTOR_ID=exec-$(hostname)
EOF'
sudo chmod 600 /data/secrets/hyper_executor.env

# executor loop
sudo curl -fsSL {shlex.quote(HYPER_EXECUTOR_LOOP_URL)} -o /data/bin/hyper_executor_loop.sh
sudo chmod +x /data/bin/hyper_executor_loop.sh

# systemd unit
sudo bash -c 'cat > /etc/systemd/system/hyper-executor.service <<EOF
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
EOF'

sudo systemctl daemon-reload
sudo systemctl enable --now hyper-executor.service
sudo systemctl is-active --quiet hyper-executor.service && echo "OK: hyper-executor active"
"""

    ssh_cmd = [
        "ssh",
        "-i",
        ssh_key_path,
        "-p",
        str(port),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "ConnectTimeout=20",
        f"{HYPERBOLIC_SSH_USER}@{host}",
        "bash -s",
    ]

    cp = subprocess.run(ssh_cmd, input=remote, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
    if cp.returncode != 0:
        raise RuntimeError(f"VM bootstrap failed rc={cp.returncode} stdout={cp.stdout[:400]!r} stderr={cp.stderr[:400]!r}")


# =========================
# Backlog calculation
# =========================
def get_backlog_count() -> int:
    """
    Count RUNNING jobs where executor_id IS NULL
    (Your exact rule.)
    """
    if not DATABASE_URL:
        # If DB isn't configured, don't scale.
        return 0

    try:
        from sqlalchemy import create_engine, text as sql_text  # type: ignore
    except Exception as e:
        raise RuntimeError(f"SQLAlchemy missing; cannot read backlog. {e}")

    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

    q = sql_text("SELECT COUNT(*) AS c FROM jobs WHERE status='RUNNING' AND executor_id IS NULL")
    with engine.connect() as conn:
        row = conn.execute(q).fetchone()
        if not row:
            return 0
        # row[0] is numeric; never try to parse column name strings
        return int(row[0])


def desired_vm_count(backlog: int) -> int:
    if backlog <= 0:
        return 0
    return int(ceil(backlog / float(BACKLOG_PER_VM)))


# =========================
# Main loop
# =========================
def run_forever() -> None:
    logger.info("brain-worker starting. BACKLOG_PER_VM=%s poll=%ss", BACKLOG_PER_VM, POLL_SECONDS)

    while True:
        try:
            ensure_hyperbolic_auth()

            backlog = get_backlog_count()
            want_vms = desired_vm_count(backlog)

            have_vms = 0
            try:
                instances = hyperbolic_instances_json()
                # Best-effort count of *all* active instances (we’ll refine tagging later)
                have_vms = len(instances)
            except Exception as e:
                # keep going: treat as zero, but log the real reason
                logger.info("autoscaler error: %s", e)

            logger.info("backlog=%s want_vms=%s have_vms=%s", backlog, want_vms, have_vms)

            # scale up only for now (safe); we’ll add tracked termination once rent works reliably
            if want_vms > have_vms:
                to_add = want_vms - have_vms
                for _ in range(to_add):
                    last_err = None
                    for gpu_count in GPU_CHOICES:
                        try:
                            logger.info("renting %sx GPU VM on Hyperbolic...", gpu_count)
                            instance_id, ssh_line = rent_vm(gpu_count)
                            logger.info("rented instance_id=%s", instance_id)

                            # bootstrap (optional)
                            if ENABLE_BOOTSTRAP:
                                bootstrap_vm_over_ssh(ssh_line)
                                logger.info("bootstrapped instance_id=%s", instance_id)
                            break
                        except Exception as e:
                            last_err = e
                            logger.info("rent attempt failed for gpu_count=%s: %s", gpu_count, e)
                            continue

                    if last_err:
                        raise RuntimeError(f"autoscaler error: could not rent any VM size: {last_err}")

            # If want_vms == 0, do nothing here (we’ll terminate once we track instance ids cleanly)

        except Exception as e:
            logger.info("autoscaler error: %s", e)

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    run_forever()
