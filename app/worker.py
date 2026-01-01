# app/worker.py
import json
import math
import os
import re
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

from app.db import get_conn

LOG_PREFIX = "[brain-worker]"

ASSIGNED_WORKER_ID = os.getenv("ASSIGNED_WORKER_ID", "hyperbolic-pool")

# Autoscale rule: 1 VM can chew through up to 250 unclaimed RUNNING jobs sequentially.
BACKLOG_PER_VM = int(os.getenv("BACKLOG_PER_VM", "250"))
POLL_SECONDS = float(os.getenv("WORKER_POLL_SECONDS", "4"))

# Prefer region (we only pass it if CLI supports it)
PREFERRED_REGION = os.getenv("HYPERBOLIC_REGION", "us-central-1")

# Always VM, always smallest GPU count (1 job per VM for now)
RENT_INSTANCE_TYPE = "virtual-machine"
RENT_GPU_COUNT = 1


def log(msg: str) -> None:
    print(f"{LOG_PREFIX} {msg}", flush=True)


def _hyper_env() -> Dict[str, str]:
    """
    Make Hyperbolic CLI auth/config deterministic on Fly:
    force config into writable /tmp.
    """
    env = os.environ.copy()
    env.setdefault("HOME", "/tmp/hyperbolic_home")
    env.setdefault("XDG_CONFIG_HOME", "/tmp/hyperbolic_home/.config")
    env.setdefault("XDG_DATA_HOME", "/tmp/hyperbolic_home/.local/share")
    # Create dirs is handled by worker loop before first CLI call.
    return env


def _run(cmd: List[str], timeout: int = 30) -> Tuple[int, str, str]:
    """
    Run a command and capture stdout/stderr.
    """
    env = _hyper_env()
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
        )
        return p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout after {timeout}s"
    except Exception as e:
        return 125, "", f"exception: {e!r}"


def _extract_json(text: str) -> Any:
    """
    Hyperbolic CLI sometimes prints non-JSON lines.
    Extract the first JSON object/array substring and parse it.
    """
    s = text.strip()
    if not s:
        raise ValueError("empty output")

    # Fast path
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        return json.loads(s)

    # Try to find first JSON object/array
    m = re.search(r"(\{.*\}|\[.*\])", s, flags=re.DOTALL)
    if not m:
        raise ValueError("no JSON object/array found in output")
    blob = m.group(1)
    return json.loads(blob)


def hyperbolic_ensure_auth() -> bool:
    """
    Ensure CLI is authenticated. We always run set-key because Fly machines are ephemeral
    and the CLI otherwise errors with "config file not found".
    """
    api_key = os.getenv("HYPERBOLIC_API_KEY", "").strip()
    if not api_key:
        log("autoscaler error: HYPERBOLIC_API_KEY is missing/empty in env")
        return False

    # Ensure writable dirs exist
    env = _hyper_env()
    os.makedirs(env["HOME"], exist_ok=True)
    os.makedirs(env["XDG_CONFIG_HOME"], exist_ok=True)
    os.makedirs(env["XDG_DATA_HOME"], exist_ok=True)

    rc, out, err = _run(["hyperbolic", "auth", "set-key", api_key], timeout=15)
    if rc != 0:
        log(f"autoscaler error: hyperbolic auth set-key failed rc={rc} stdout={out!r} stderr={err!r}")
        return False
    return True


def hyperbolic_instances() -> List[Dict[str, Any]]:
    """
    Return list of instances from CLI JSON.
    """
    rc, out, err = _run(["hyperbolic", "instances", "--json"], timeout=20)
    if rc != 0:
        raise RuntimeError(f"hyperbolic instances rc={rc} stdout={out} stderr={err}")

    try:
        data = _extract_json(out)
    except Exception as e:
        raise RuntimeError(
            f"hyperbolic instances --json returned non-JSON output. stdout={out!r} stderr={err!r} parse_err={e}"
        )

    # Normalize to list
    if isinstance(data, dict):
        # Some CLIs wrap as {"instances":[...]}
        if "instances" in data and isinstance(data["instances"], list):
            return data["instances"]
        return [data]
    if isinstance(data, list):
        return data
    raise RuntimeError(f"unexpected instances JSON type: {type(data)}")


def _instance_is_active(inst: Dict[str, Any]) -> bool:
    """
    Best-effort filter: count instances that are not terminated.
    We don't assume exact schema; we check common fields.
    """
    status = str(inst.get("status") or inst.get("state") or "").lower()
    if not status:
        return True  # unknown => count it rather than undercount
    if "terminat" in status or status in {"dead", "stopped"}:
        return False
    return True


def _instance_id(inst: Dict[str, Any]) -> Optional[str]:
    for k in ("id", "instance_id", "instanceId", "instanceID"):
        v = inst.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _rent_supports_flag(flag: str) -> bool:
    """
    Detect optional flags by checking help text (safe to call; fast).
    """
    rc, out, err = _run(["hyperbolic", "rent", "ondemand", "--help"], timeout=10)
    help_text = (out or "") + "\n" + (err or "")
    return flag in help_text


def hyperbolic_rent_vm() -> None:
    """
    Rent a new on-demand VM with minimal GPUs.
    Prefer us-central-1 IF the CLI supports a region flag.
    """
    cmd = [
        "hyperbolic",
        "rent",
        "ondemand",
        "--instance-type",
        RENT_INSTANCE_TYPE,
        "--gpu-count",
        str(RENT_GPU_COUNT),
    ]

    # Only add region if the CLI actually supports it
    # (avoids hard failing on unknown flags).
    for candidate in ("--region", "--location", "--datacenter", "--data-center"):
        if _rent_supports_flag(candidate):
            cmd += [candidate, PREFERRED_REGION]
            break

    rc, out, err = _run(cmd, timeout=120)
    if rc != 0:
        raise RuntimeError(f"rent failed rc={rc} stdout={out!r} stderr={err!r}")

    log(f"rented VM ok. stdout_head={out[:300]!r} stderr_head={err[:300]!r}")


def hyperbolic_terminate(instance_id: str) -> None:
    rc, out, err = _run(["hyperbolic", "terminate", instance_id], timeout=60)
    if rc != 0:
        raise RuntimeError(f"terminate {instance_id} failed rc={rc} stdout={out!r} stderr={err!r}")
    log(f"terminated instance_id={instance_id}")


def db_backlog_and_running() -> Tuple[int, int]:
    """
    backlog = RUNNING jobs where executor_id IS NULL (and assigned to our worker pool)
    running_total = all RUNNING jobs
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                select count(*)::int as c
                from public.jobs
                where status='RUNNING'
                  and executor_id is null
                  and assigned_worker_id = %s
                """,
                (ASSIGNED_WORKER_ID,),
            )
            backlog = cur.fetchone()["c"]

            cur.execute(
                """
                select count(*)::int as c
                from public.jobs
                where status='RUNNING'
                """
            )
            running_total = cur.fetchone()["c"]

    return backlog, running_total


def promote_queued_to_running(limit: int = 25) -> int:
    """
    Optional: move QUEUED -> RUNNING so executors can claim.
    Safe if you already do it elsewhere; it just becomes a no-op.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                with picked as (
                    select job_id
                    from public.jobs
                    where status='QUEUED'
                    order by created_at asc
                    limit %s
                    for update skip locked
                )
                update public.jobs j
                set status='RUNNING',
                    started_at=coalesce(started_at, now()),
                    assigned_worker_id=coalesce(assigned_worker_id, %s)
                from picked
                where j.job_id = picked.job_id
                returning j.job_id
                """,
                (limit, ASSIGNED_WORKER_ID),
            )
            rows = cur.fetchall()
            return len(rows)


def main() -> None:
    log(f"worker start: ASSIGNED_WORKER_ID={ASSIGNED_WORKER_ID} BACKLOG_PER_VM={BACKLOG_PER_VM}")

    while True:
        try:
            # 1) Promote queued -> running (if applicable)
            promoted = promote_queued_to_running()
            if promoted:
                log(f"promoted queued->running: {promoted}")

            # 2) Compute backlog & desired VM count
            backlog, running_total = db_backlog_and_running()
            want_vms = 0 if backlog == 0 else int(math.ceil(backlog / BACKLOG_PER_VM))

            # 3) Hyperbolic autoscale (only if authenticated)
            if not hyperbolic_ensure_auth():
                time.sleep(POLL_SECONDS)
                continue

            instances = hyperbolic_instances()
            active = [i for i in instances if _instance_is_active(i)]
            have_vms = len(active)

            log(f"backlog={backlog} want_vms={want_vms} have_vms={have_vms}")

            if want_vms > have_vms:
                to_add = want_vms - have_vms
                for _ in range(to_add):
                    hyperbolic_rent_vm()

            elif want_vms < have_vms:
                # Safety: only scale down if there are *no* RUNNING jobs at all.
                if running_total == 0:
                    to_kill = have_vms - want_vms
                    killed = 0
                    for inst in active:
                        if killed >= to_kill:
                            break
                        iid = _instance_id(inst)
                        if iid:
                            hyperbolic_terminate(iid)
                            killed += 1
                else:
                    log(f"scale-down skipped (running_total={running_total} > 0)")

        except Exception as e:
            log(f"autoscaler error: {e}")

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
