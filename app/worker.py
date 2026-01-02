# app/worker.py
"""
Worker loop for Hyper-Brain.

Responsibilities
1) Move QUEUED -> RUNNING (so executors can claim)
2) Autoscale Hyperbolic on-demand VMs based on backlog:
   backlog = count(RUNNING jobs where executor_id IS NULL)

User policy (as of this iteration):
- If backlog == 0: zero instances (terminate managed VMs)
- Otherwise: want_vms = ceil(backlog / BACKLOG_PER_VM)
- VM preferences: virtual-machine, smallest GPU count (1 > 2 > 4), slight region preference us-central-1 (best-effort)
- One job per VM (for now)
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
import os
import re
import subprocess
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .db import get_conn

log = logging.getLogger("brain-worker")
logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")


# -----------------------------
# Config
# -----------------------------
BACKLOG_PER_VM = int(os.environ.get("BACKLOG_PER_VM", "250"))
POLL_SECONDS = float(os.environ.get("POLL_SECONDS", "3"))
# Try smallest first; may be overridden via env "PREFER_GPU_COUNTS" like "1,2,4"
_PREFER_GPU_COUNTS_ENV = os.environ.get("PREFER_GPU_COUNTS", "")
if _PREFER_GPU_COUNTS_ENV.strip():
    try:
        PREFER_GPU_COUNTS = [int(x.strip()) for x in _PREFER_GPU_COUNTS_ENV.split(",") if x.strip()]
    except Exception:
        PREFER_GPU_COUNTS = [1, 2, 4]
else:
    PREFER_GPU_COUNTS = [1, 2, 4]

# Slight preference only (we cannot always force region via CLI); used only when interpreting ondemand listing.
PREFERRED_REGION = os.environ.get("PREFERRED_REGION", "us-central-1")

# Where we store Hyperbolic CLI config on Fly machines (writable).
HCLI_HOME = os.environ.get("HCLI_HOME", "/tmp/hcli")

# Required secrets
HYPERBOLIC_API_KEY = os.environ.get("HYPERBOLIC_API_KEY", "")
# Optional: identify instances created by this app (stored in DB; also useful as worker_id passed to executors)
ASSIGNED_WORKER_ID = os.environ.get("ASSIGNED_WORKER_ID", "hyperbolic-pool")


# -----------------------------
# DB helpers
# -----------------------------
def _ensure_db_schema() -> None:
    """Create the minimal table(s) we need to safely terminate only VMs we created."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS hyper_instances (
              instance_id TEXT PRIMARY KEY,
              gpu_count   INT,
              ssh_command TEXT,
              status      TEXT,
              created_at  TIMESTAMPTZ DEFAULT NOW(),
              last_seen_at TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )
        conn.commit()


def _db_try_advisory_lock(lock_key: int = 93278123) -> bool:
    """Prevent multiple Fly worker machines from autoscaling at the same time."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT pg_try_advisory_lock(%s) AS ok", (lock_key,))
        row = cur.fetchone() or {}
        return bool(row.get("ok", False))


def get_backlog_unclaimed_running() -> int:
    """
    User requirement:
      count of RUNNING jobs where executor_id IS NULL
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*)::int AS count
            FROM jobs
            WHERE status='RUNNING' AND executor_id IS NULL
            """
        )
        row = cur.fetchone() or {}
        return int(row.get("count", 0))


def dispatch_one_job() -> bool:
    """
    Minimal dispatcher:
      - Find one QUEUED job
      - Mark it RUNNING (executors will later claim it)
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT job_id
            FROM jobs
            WHERE status='QUEUED'
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
            SET status='RUNNING', started_at=NOW()
            WHERE job_id=%s
            """,
            (job_id,),
        )
        conn.commit()
        return True


def list_managed_instances() -> List[Dict[str, Any]]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT instance_id, gpu_count, ssh_command, status, created_at, last_seen_at
            FROM hyper_instances
            ORDER BY created_at ASC
            """
        )
        return list(cur.fetchall() or [])


def upsert_managed_instance(instance_id: str, gpu_count: Optional[int], ssh_command: Optional[str], status: str) -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO hyper_instances(instance_id, gpu_count, ssh_command, status, created_at, last_seen_at)
            VALUES (%s, %s, %s, %s, NOW(), NOW())
            ON CONFLICT (instance_id) DO UPDATE
            SET gpu_count=COALESCE(EXCLUDED.gpu_count, hyper_instances.gpu_count),
                ssh_command=COALESCE(EXCLUDED.ssh_command, hyper_instances.ssh_command),
                status=EXCLUDED.status,
                last_seen_at=NOW()
            """,
            (instance_id, gpu_count, ssh_command, status),
        )
        conn.commit()


def mark_instance_terminated(instance_id: str) -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE hyper_instances SET status='terminated', last_seen_at=NOW() WHERE instance_id=%s",
            (instance_id,),
        )
        conn.commit()


# -----------------------------
# Hyperbolic CLI wrapper
# -----------------------------
_JSON_START_RE = re.compile(r"(\{|\[)")

# NOTE: Hyperbolic CLI output formats may change; keep parsing defensive.
_INSTANCE_ID_PATTERNS = [
    re.compile(r"\b([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b", re.I),
    re.compile(r"\b([0-9a-f]{16,64})\b", re.I),
    re.compile(r"\b(i-[0-9a-f]{8,})\b", re.I),
    re.compile(r"\b(vm-[\w-]+)\b", re.I),
    re.compile(r"\b(inst-[\w-]+)\b", re.I),
]


def _hcli_env() -> Dict[str, str]:
    env = os.environ.copy()
    env["HOME"] = HCLI_HOME
    env["XDG_CONFIG_HOME"] = HCLI_HOME
    env["XDG_CACHE_HOME"] = HCLI_HOME
    # Keep it as quiet / non-interactive as possible
    env.setdefault("TERM", "dumb")
    # Some CLIs will read the API key directly; doesn't hurt to pass.
    if HYPERBOLIC_API_KEY:
        env["HYPERBOLIC_API_KEY"] = HYPERBOLIC_API_KEY
    return env


def _run(cmd: Sequence[str], timeout: int = 90, input_text: Optional[str] = None) -> Tuple[int, str, str]:
    # Ensure config dir exists for every call
    try:
        os.makedirs(HCLI_HOME, exist_ok=True)
    except Exception:
        pass

    p = subprocess.run(
        list(cmd),
        env=_hcli_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        input=input_text,
        timeout=timeout,
    )
    return p.returncode, (p.stdout or ""), (p.stderr or "")


_AUTH_OK = False


def _ensure_hyperbolic_auth() -> None:
    """Make sure Hyperbolic CLI has a saved API key (config stored under HCLI_HOME)."""
    global _AUTH_OK
    if _AUTH_OK:
        return

    if not HYPERBOLIC_API_KEY:
        raise RuntimeError("Missing HYPERBOLIC_API_KEY in environment (Fly secrets).")

    # Always set-key on first run (cheap, and avoids 'config file not found' on fresh machines).
    rc, out, err = _run(["hyperbolic", "auth", "set-key", HYPERBOLIC_API_KEY], timeout=30)
    if rc != 0:
        raise RuntimeError(
            f"hyperbolic auth set-key failed rc={rc} stdout_head={out[:200]!r} stderr_head={err[:200]!r}"
        )
    _AUTH_OK = True


def _extract_json(stdout: str) -> Any:
    s = (stdout or "").strip()
    m = _JSON_START_RE.search(s)
    if not m:
        raise ValueError("no JSON object/array found in output")
    return json.loads(s[m.start():])


def _looks_like_404(out: str, err: str) -> bool:
    blob = (out or "") + "\n" + (err or "")
    return "status 404" in blob.lower() or '"httpstatus":404' in blob.lower() or "httpstatus\":404" in blob.lower()


def hyperbolic_instances_json(instance_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Return list of instances (best-effort).
    Some CLI versions:
      - return a list
      - return {"instances":[...]}
      - may return 404 when there are no instances (treat as empty)
    """
    _ensure_hyperbolic_auth()

    cmd: List[str] = ["hyperbolic", "instances"]
    if instance_id:
        cmd.append(instance_id)
    cmd.append("--json")

    rc, out, err = _run(cmd, timeout=60)
    if rc != 0:
        # If API returns 404 for "none", treat as empty list.
        if _looks_like_404(out, err):
            return []
        raise RuntimeError(f"hyperbolic instances failed rc={rc} stdout_head={out[:300]!r} stderr_head={err[:300]!r}")

    try:
        data = _extract_json(out)
    except Exception as e:
        # Some versions print errors to stdout with rc=0; detect and treat as empty on 404.
        if _looks_like_404(out, err):
            return []
        raise RuntimeError(
            f"hyperbolic instances --json returned non-JSON output. stdout_head={out[:300]!r} stderr_head={err[:300]!r} parse_err={e}"
        )

    if isinstance(data, dict) and isinstance(data.get("instances"), list):
        return data["instances"]
    if isinstance(data, list):
        return data
    # Some `instances <id> --json` might return a single dict.
    if isinstance(data, dict) and data:
        return [data]
    return []


def hyperbolic_ondemand_options() -> List[Dict[str, Any]]:
    """
    Return list of available on-demand configs (best-effort).
    This is useful to avoid calling `rent` with a GPU count that isn't offered.
    """
    _ensure_hyperbolic_auth()
    rc, out, err = _run(["hyperbolic", "ondemand", "--json"], timeout=60)
    if rc != 0:
        if _looks_like_404(out, err):
            return []
        raise RuntimeError(f"hyperbolic ondemand failed rc={rc} stdout_head={out[:300]!r} stderr_head={err[:300]!r}")
    try:
        data = _extract_json(out)
    except Exception:
        if _looks_like_404(out, err):
            return []
        raise RuntimeError(f"hyperbolic ondemand --json returned non-JSON output. stdout_head={out[:300]!r} stderr_head={err[:300]!r}")
    if isinstance(data, dict) and isinstance(data.get("instances"), list):
        return data["instances"]
    if isinstance(data, list):
        return data
    return []


def _get_int(d: Dict[str, Any], *keys: str) -> Optional[int]:
    for k in keys:
        v = d.get(k)
        if v is None:
            continue
        try:
            return int(v)
        except Exception:
            continue
    return None


def _get_str(d: Dict[str, Any], *keys: str) -> str:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def pick_gpu_count_from_ondemand(prefer: Sequence[int]) -> Optional[int]:
    """
    Choose smallest preferred GPU count that is present in ondemand listing for virtual-machine.
    If listing is unavailable/empty, return None to fall back to trying the prefer list directly.
    """
    options = hyperbolic_ondemand_options()
    if not options:
        return None

    available: set[int] = set()

    for o in options:
        if not isinstance(o, dict):
            continue
        instance_type = _get_str(o, "instance_type", "instanceType", "type")
        if instance_type and instance_type.lower() not in ("virtual-machine", "virtual_machine", "vm", "virtual"):
            continue

        # best-effort: network type constraint (for bare metal). For VMs, often omitted.
        network = _get_str(o, "network_type", "networkType", "network")
        if network and network.lower() == "infiniband":
            continue

        gpu_count = _get_int(o, "gpu_count", "gpuCount", "gpus", "gpu")
        if gpu_count is None:
            continue
        available.add(gpu_count)

    for g in prefer:
        if g in available:
            return g
    return None


def _parse_instance_id(stdout: str) -> Optional[str]:
    s = stdout or ""
    for pat in _INSTANCE_ID_PATTERNS:
        m = pat.search(s)
        if m:
            return m.group(1)
    return None


def _parse_ssh_command(stdout: str) -> Optional[str]:
    for line in (stdout or "").splitlines():
        if "ssh " in line.lower():
            return line.strip()
    return None


def _parse_iso_dt(s: str) -> Optional[dt.datetime]:
    if not s:
        return None
    try:
        # Handle Z and offset forms
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def _discover_new_instance_id(known_ids: Iterable[str]) -> Optional[str]:
    """
    If rent output doesn't include an ID, best-effort: list instances and pick the most recent
    one that is not already known.
    """
    known = set(known_ids)
    instances = hyperbolic_instances_json()
    candidates: List[Tuple[dt.datetime, str]] = []

    now = dt.datetime.now(dt.timezone.utc)

    for inst in instances:
        if not isinstance(inst, dict):
            continue
        iid = _get_str(inst, "id", "instance_id", "instanceId")
        if not iid or iid in known:
            continue
        created = _get_str(inst, "created_at", "createdAt", "created")
        created_dt = _parse_iso_dt(created) or now
        candidates.append((created_dt, iid))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def hyperbolic_rent_vm(gpu_count: int, known_instance_ids: Iterable[str]) -> Tuple[str, Optional[str], str]:
    """
    Rent an on-demand VM with the given gpu_count.
    Returns (instance_id, ssh_command, raw_stdout).
    """
    _ensure_hyperbolic_auth()

    cmd = [
        "hyperbolic",
        "rent",
        "ondemand",
        "--instance-type",
        "virtual-machine",
        "--gpu-count",
        str(gpu_count),
    ]
    rc, out, err = _run(cmd, timeout=180)

    if rc != 0:
        raise RuntimeError(f"rent failed rc={rc} stdout_head={out[:400]!r} stderr_head={err[:200]!r}")

    # Defensive: some versions print API errors to stdout even with rc==0
    if "error response from api" in out.lower() or "api request failed" in out.lower():
        raise RuntimeError(f"rent returned error output. rent_stdout_head={out[:400]!r}")

    instance_id = _parse_instance_id(out) or _discover_new_instance_id(known_instance_ids)
    ssh_cmd = _parse_ssh_command(out)

    return (instance_id or ""), ssh_cmd, out


def hyperbolic_terminate(instance_id: str) -> None:
    _ensure_hyperbolic_auth()
    rc, out, err = _run(["hyperbolic", "terminate", instance_id], timeout=90)
    if rc != 0:
        # If already terminated/not found, treat as ok.
        if _looks_like_404(out, err):
            return
        raise RuntimeError(f"terminate failed rc={rc} stdout_head={out[:300]!r} stderr_head={err[:300]!r}")


# -----------------------------
# Autoscaler logic
# -----------------------------
def _count_active_managed_instances() -> int:
    rows = list_managed_instances()
    active = 0
    for r in rows:
        st = str(r.get("status") or "").lower()
        if st in ("running", "active", "provisioning", "started", "bootstrapping", ""):
            active += 1
    return active


def _known_instance_ids() -> List[str]:
    return [r["instance_id"] for r in list_managed_instances() if r.get("instance_id")]


def terminate_all_managed_instances() -> None:
    rows = list_managed_instances()
    for r in rows:
        iid = r.get("instance_id")
        st = str(r.get("status") or "").lower()
        if not iid:
            continue
        if st == "terminated":
            continue
        try:
            log.info(f"terminating instance_id={iid} (no backlog)...")
            hyperbolic_terminate(iid)
        except Exception as e:
            log.info(f"autoscaler error: terminate failed for {iid}: {e}")
        finally:
            mark_instance_terminated(iid)


def autoscaler_tick() -> None:
    backlog = get_backlog_unclaimed_running()
    want_vms = 0 if backlog <= 0 else int(math.ceil(backlog / BACKLOG_PER_VM))
    have_vms = _count_active_managed_instances()

    log.info(f"backlog={backlog} want_vms={want_vms} have_vms={have_vms}")

    # User policy: backlog=0 -> no instances.
    if want_vms <= 0:
        if have_vms > 0:
            terminate_all_managed_instances()
        return

    if have_vms >= want_vms:
        return

    need = want_vms - have_vms
    known_ids = _known_instance_ids()

    # Avoid trying GPU sizes that are not offered (prevents 404 rent on unsupported sizes)
    chosen_from_list = None
    try:
        chosen_from_list = pick_gpu_count_from_ondemand(PREFER_GPU_COUNTS)
    except Exception as e:
        log.info(f"autoscaler error: could not read ondemand listing (will try sizes directly): {e}")

    for _ in range(need):
        last_err: Optional[Exception] = None

        gpu_try_order = ([chosen_from_list] if chosen_from_list else []) + [g for g in PREFER_GPU_COUNTS if g != chosen_from_list]
        gpu_try_order = [g for g in gpu_try_order if g is not None]

        for g in gpu_try_order:
            try:
                log.info(f"renting {g}x GPU VM on Hyperbolic...")
                instance_id, ssh_cmd, raw = hyperbolic_rent_vm(int(g), known_ids)

                if not instance_id:
                    raise RuntimeError(f"Could not determine instance_id after rent. rent_stdout_head={raw[:400]!r}")

                known_ids.append(instance_id)
                upsert_managed_instance(instance_id, int(g), ssh_cmd, "provisioning")
                log.info(f"rented instance_id={instance_id} gpu_count={g}")
                if ssh_cmd:
                    log.info(f"rent output included ssh cmd: {ssh_cmd}")
                else:
                    log.info("rent output did not include an ssh cmd (ok).")

                # Bootstrapping step comes next (SSH + install executor loop). Add once rent is stable.
                upsert_managed_instance(instance_id, int(g), ssh_cmd, "running")
                last_err = None
                break
            except Exception as e:
                last_err = e
                log.info(f"rent attempt failed for gpu_count={g}: {e}")

        if last_err is not None:
            raise RuntimeError(f"could not rent any VM size (tried {gpu_try_order}): {last_err}")


# -----------------------------
# Main loop
# -----------------------------
def main() -> None:
    _ensure_db_schema()

    # Single-leader behavior
    if not _db_try_advisory_lock():
        log.info("another worker holds the advisory lock; sleeping.")
        while True:
            time.sleep(10)

    while True:
        try:
            # Dispatch a small batch to keep throughput up without a tight loop
            dispatched_any = False
            for _ in range(10):
                if dispatch_one_job():
                    dispatched_any = True
                else:
                    break

            autoscaler_tick()

            # If nothing happened, sleep a bit
            if not dispatched_any:
                time.sleep(POLL_SECONDS)
        except Exception as e:
            log.info(f"autoscaler error: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()
