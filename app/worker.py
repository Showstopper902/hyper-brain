# app/worker.py
import json
import logging
import math
import os
import re
import shlex
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

from .db import get_conn

log = logging.getLogger("brain-worker")
logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")


# -----------------------------
# DB helpers
# -----------------------------
def get_backlog_unclaimed_running() -> int:
    """
    User requirement:
      count of RUNNING jobs where executor_id IS NULL
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*) AS count
            FROM jobs
            WHERE status = 'RUNNING'
              AND executor_id IS NULL
            """
        )
        row = cur.fetchone() or {}
        # IMPORTANT: db.py uses dict_row, so row is a dict like {"count": 12}
        return int(row.get("count", 0))


def dispatch_one_job() -> bool:
    """
    Minimal dispatcher:
    - Find one QUEUED job
    - Mark it RUNNING
    - Let executors claim it via /executors/claim (your existing flow)
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT job_id
            FROM jobs
            WHERE status = 'QUEUED'
            ORDER BY created_at ASC
            LIMIT 1
            FOR UPDATE SKIP LOCKED
            """
        )
        job = cur.fetchone()
        if not job:
            return False

        job_id = job["job_id"]
        cur.execute(
            """
            UPDATE jobs
            SET status = 'RUNNING', started_at = NOW()
            WHERE job_id = %s
            """,
            (job_id,),
        )
        conn.commit()
        return True


# -----------------------------
# Hyperbolic CLI wrapper
# -----------------------------
def _hcli_env() -> Dict[str, str]:
    """
    Make Hyperbolic CLI non-interactive and consistent on Fly:
    - Force HOME/XDG paths into /tmp (writable, per-machine)
    """
    env = os.environ.copy()
    h = "/tmp/hcli"
    env["HOME"] = h
    env["XDG_CONFIG_HOME"] = h
    env["XDG_CACHE_HOME"] = h
    return env


def _run(cmd: List[str], timeout: int = 60) -> Tuple[int, str, str]:
    p = subprocess.run(
        cmd,
        env=_hcli_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    return p.returncode, p.stdout or "", p.stderr or ""


def _ensure_hyperbolic_auth() -> None:
    key = os.environ.get("HYPERBOLIC_API_KEY", "")
    if not key:
        raise RuntimeError("Missing HYPERBOLIC_API_KEY in environment")

    # Ensure config dir exists
    subprocess.run(["bash", "-lc", "mkdir -p /tmp/hcli"], check=False)

    # Always set-key (cheap, avoids config-not-found issues on fresh machines)
    rc, out, err = _run(["hyperbolic", "auth", "set-key", key], timeout=30)
    if rc != 0:
        raise RuntimeError(
            f"hyperbolic auth set-key failed rc={rc} stdout={out[:200]!r} stderr={err[:200]!r}"
        )


def _extract_json(stdout: str) -> Any:
    """
    Hyperbolic sometimes prints non-JSON text.
    Try to locate the first { or [ and parse from there.
    """
    s = stdout.strip()
    m = re.search(r"(\{|\[)", s)
    if not m:
        raise ValueError("no JSON object/array found in output")
    s2 = s[m.start() :]
    return json.loads(s2)


def hyperbolic_instances_json() -> List[Dict[str, Any]]:
    """
    Returns list of instances (best-effort).
    If non-json / auth issues happen, raise so caller can log & continue.
    """
    _ensure_hyperbolic_auth()
    rc, out, err = _run(["hyperbolic", "instances", "--json"], timeout=60)
    if rc != 0:
        raise RuntimeError(
            f"hyperbolic instances failed rc={rc} stdout={out[:300]!r} stderr={err[:300]!r}"
        )
    try:
        data = _extract_json(out)
    except Exception as e:
        raise RuntimeError(
            f"hyperbolic instances --json returned non-JSON output. stdout={out[:300]!r} stderr={err[:300]!r} parse_err={e}"
        )
    # Some CLIs return {"instances":[...]} while others return [...]
    if isinstance(data, dict) and "instances" in data and isinstance(data["instances"], list):
        return data["instances"]
    if isinstance(data, list):
        return data
    # unknown shape
    return []


_INSTANCE_ID_PATTERNS = [
    re.compile(r"\b([0-9a-f]{16,64})\b", re.IGNORECASE),  # hex-ish ids
    re.compile(r"\b([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b", re.IGNORECASE),  # uuid
    re.compile(r"\b(i-[0-9a-f]{8,})\b", re.IGNORECASE),  # aws-ish
    re.compile(r"\b(vm-[\w-]+)\b", re.IGNORECASE),
    re.compile(r"\b(inst-[\w-]+)\b", re.IGNORECASE),
]


def _parse_instance_id(stdout: str) -> Optional[str]:
    for pat in _INSTANCE_ID_PATTERNS:
        m = pat.search(stdout)
        if m:
            return m.group(1)
    return None


def _parse_ssh_command(stdout: str) -> Optional[str]:
    """
    Try to find an ssh command line in rent output.
    """
    for line in stdout.splitlines():
        if "ssh " in line.lower():
            # return the first line that looks like an ssh command
            return line.strip()
    return None


def hyperbolic_rent_vm(gpu_count: int) -> Tuple[str, Optional[str], str]:
    """
    Returns (instance_id, ssh_cmd, raw_stdout)
    """
    _ensure_hyperbolic_auth()

    cmd = ["hyperbolic", "rent", "ondemand", "--instance-type", "virtual-machine", "--gpu-count", str(gpu_count)]
    rc, out, err = _run(cmd, timeout=120)

    if rc != 0:
        raise RuntimeError(
            f"rent failed rc={rc} stdout={out[:400]!r} stderr={err[:400]!r}"
        )

    instance_id = _parse_instance_id(out)
    ssh_cmd = _parse_ssh_command(out)
    return instance_id or "", ssh_cmd, out


def hyperbolic_terminate(instance_id: str) -> None:
    _ensure_hyperbolic_auth()
    rc, out, err = _run(["hyperbolic", "terminate", instance_id], timeout=60)
    if rc != 0:
        raise RuntimeError(
            f"terminate failed rc={rc} stdout={out[:300]!r} stderr={err[:300]!r}"
        )


# -----------------------------
# Autoscaler logic
# -----------------------------
BACKLOG_PER_VM = int(os.environ.get("BACKLOG_PER_VM", "250"))
PREFER_GPU_COUNTS = [1, 2, 4]  # your preference (try in order)


def autoscaler_tick() -> None:
    backlog = get_backlog_unclaimed_running()
    want_vms = 0 if backlog <= 0 else int(math.ceil(backlog / BACKLOG_PER_VM))

    # Best-effort: count current Hyperbolic instances.
    # If listing fails, assume 0 so you at least *try* to recover.
    try:
        instances = hyperbolic_instances_json()
        # Count “active-ish” instances. Field names may vary; be permissive.
        have_vms = 0
        for inst in instances:
            status = str(inst.get("status", inst.get("state", ""))).lower()
            if status in ("running", "started", "active", "provisioning"):
                have_vms += 1
    except Exception as e:
        log.info(f"autoscaler error: {e}")
        instances = []
        have_vms = 0

    log.info(f"backlog={backlog} want_vms={want_vms} have_vms={have_vms}")

    # No backlog -> no instances (you chose this).
    if want_vms <= 0:
        return

    # Scale up only (for now). Scale-down/terminate later once scale-up is stable.
    if have_vms >= want_vms:
        return

    need = want_vms - have_vms
    for _ in range(need):
        rented = False
        last_err = None
        for g in PREFER_GPU_COUNTS:
            try:
                log.info(f"renting {g}x GPU VM on Hyperbolic...")
                instance_id, ssh_cmd, raw = hyperbolic_rent_vm(g)

                # If instance_id couldn't be parsed, show the head for debugging.
                if not instance_id:
                    raise RuntimeError(
                        f"Could not determine instance_id after rent. rent_stdout_head={raw[:400]!r}"
                    )

                log.info(f"rented instance_id={instance_id} gpu_count={g}")
                if ssh_cmd:
                    log.info(f"rent output included ssh cmd: {ssh_cmd}")
                else:
                    log.info("rent output did not include an ssh cmd (ok for now).")

                # Bootstrapping via SSH is the next step (we can add once rent is stable).
                # For now, just rent successfully.
                rented = True
                break
            except Exception as e:
                last_err = e
                # If 1/2 GPU VM is not offered, Hyperbolic may 404; try next size.
                log.info(f"rent attempt failed for gpu_count={g}: {e}")

        if not rented and last_err:
            raise RuntimeError(f"autoscaler error: could not rent any VM size: {last_err}")


def main() -> None:
    # Worker does two jobs:
    # 1) Dispatch queued jobs -> RUNNING
    # 2) Autoscale based on unclaimed RUNNING jobs (executor_id IS NULL)
    while True:
        try:
            # Dispatch as fast as possible (small batches)
            dispatched = False
            for _ in range(10):
                if dispatch_one_job():
                    dispatched = True
                else:
                    break

            # Autoscaler tick every loop (cheap)
            autoscaler_tick()

            # If nothing happened, sleep a bit
            if not dispatched:
                time.sleep(3)
        except Exception as e:
            log.info(f"autoscaler error: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()
