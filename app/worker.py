import json
import math
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .db import get_conn

LOG_PREFIX = "[brain-worker]"

ASSIGNED_WORKER_ID = os.getenv("ASSIGNED_WORKER_ID", "hyperbolic-pool")

# Backlog rule (your requirement):
# - backlog == 0 => 0 instances
# - backlog 1..250 => 1 instance
# - 251..500 => 2
# - 501..750 => 3
# - etc
BACKLOG_PER_VM = int(os.getenv("BACKLOG_PER_VM", "250"))

# Autoscaler loop frequency
POLL_SECONDS = int(os.getenv("AUTOSCALER_POLL_SECONDS", "10"))

# We currently rent 1 GPU per VM (you chose 1 job per VM for now)
GPU_COUNT_PER_VM = int(os.getenv("GPU_COUNT_PER_VM", "1"))

# Hyperbolic API key must be present as Fly secret env var
HYPERBOLIC_API_KEY = os.getenv("HYPERBOLIC_API_KEY", "")

# Where we force the CLI to store config so it works reliably in Fly machines
# (Do NOT rely on whatever HOME happens to be for root/non-root)
HYPERBOLIC_HOME = os.getenv("HYPERBOLIC_HOME", "/data/hyperbolic-cli-home")


def log(msg: str) -> None:
    print(f"{LOG_PREFIX} {msg}", flush=True)


def _ensure_dirs() -> None:
    Path(HYPERBOLIC_HOME).mkdir(parents=True, exist_ok=True)
    # Some CLIs use XDG_CONFIG_HOME; keep it stable
    Path(HYPERBOLIC_HOME, ".config").mkdir(parents=True, exist_ok=True)


def _hyperbolic_env() -> Dict[str, str]:
    """
    Return env for subprocess so Hyperbolic CLI can always find/write config.
    """
    _ensure_dirs()
    env = dict(os.environ)
    env["HOME"] = HYPERBOLIC_HOME
    env["XDG_CONFIG_HOME"] = str(Path(HYPERBOLIC_HOME, ".config"))
    return env


def _run(cmd: List[str], timeout: int = 60) -> Tuple[int, str, str]:
    env = _hyperbolic_env()
    p = subprocess.run(
        cmd,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    return p.returncode, p.stdout or "", p.stderr or ""


def _redact(s: str, keep_tail: int = 6) -> str:
    if not s:
        return ""
    if len(s) <= keep_tail:
        return "***"
    return "***" + s[-keep_tail:]


def ensure_hyperbolic_auth() -> None:
    """
    Always try to set-key so the config exists for `hyperbolic instances` / `rent` / `terminate`.
    This prevents the 'config file not found - please run hyperbolic auth' error.
    """
    if not HYPERBOLIC_API_KEY:
        raise RuntimeError("HYPERBOLIC_API_KEY env var is missing (Fly secret not injected?).")

    # This is idempotent for the CLI; it overwrites/saves the key.
    rc, out, err = _run(["hyperbolic", "auth", "set-key", HYPERBOLIC_API_KEY], timeout=30)
    if rc != 0:
        raise RuntimeError(
            f"hyperbolic auth set-key failed rc={rc} out={out.strip()[:300]} err={err.strip()[:300]}"
        )

    # Optional: sanity check (don’t fail autoscaler on this)
    rc2, out2, err2 = _run(["hyperbolic", "auth", "status"], timeout=30)
    if rc2 == 0:
        log(f"hyperbolic auth ok (key={_redact(HYPERBOLIC_API_KEY)})")
    else:
        log(f"hyperbolic auth status nonzero rc={rc2} (continuing). err={err2.strip()[:200]}")


def _extract_json_from_mixed_output(text: str) -> Optional[Any]:
    """
    Hyperbolic CLI sometimes prints human text even when `--json` is requested.
    This extracts the first JSON object/array found in the output.

    Returns parsed JSON or None.
    """
    if not text:
        return None
    # Find first { or [
    m = re.search(r"[\{\[]", text)
    if not m:
        return None
    start = m.start()
    # Find last } or ]
    end = max(text.rfind("}"), text.rfind("]"))
    if end <= start:
        return None
    blob = text[start : end + 1]
    try:
        return json.loads(blob)
    except Exception:
        return None


def hyperbolic_instances_json() -> List[Dict[str, Any]]:
    """
    Try to list instances as JSON. If CLI returns NOT_FOUND or non-json, return [] (don’t crash loop).
    """
    rc, out, err = _run(["hyperbolic", "instances", "--json"], timeout=60)
    combined = (out + "\n" + err).strip()

    if rc != 0:
        # Treat "not found" as "no instances"
        if "NOT_FOUND" in combined or "Not found" in combined:
            return []
        # Auth/config errors show up here too
        log(f"autoscaler warn: hyperbolic instances --json rc={rc} (treating as empty). msg={combined[:240]}")
        return []

    parsed = _extract_json_from_mixed_output(out) or _extract_json_from_mixed_output(combined)
    if parsed is None:
        # Sometimes stdout is empty but stderr has the message
        log(f"autoscaler warn: hyperbolic instances --json returned non-JSON output (treating as empty). head={combined[:240]}")
        return []

    # Normalize to list[dict]
    if isinstance(parsed, list):
        return [x for x in parsed if isinstance(x, dict)]
    if isinstance(parsed, dict):
        for key in ("instances", "data", "items", "results"):
            v = parsed.get(key)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
        # Could be a single object
        return [parsed]

    return []


def _instance_id_from_obj(obj: Dict[str, Any]) -> Optional[str]:
    for k in ("instance_id", "id", "instanceId", "instanceID"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _ssh_cmd_from_obj(obj: Dict[str, Any]) -> Optional[str]:
    for k in ("ssh_command", "sshCommand", "ssh", "ssh_cmd", "connection", "connect"):
        v = obj.get(k)
        if isinstance(v, str) and "ssh " in v:
            return v.strip()
    return None


def _parse_instance_id_from_rent_output(text: str) -> Optional[str]:
    """
    Last-resort parsing for instance IDs when instances list is broken.
    """
    if not text:
        return None

    # Common patterns: "Instance ID: xxx", "instance_id=xxx", "id: xxx"
    patterns = [
        r"instance[\s_-]*id\s*[:=]\s*([A-Za-z0-9\-_]+)",
        r"\bid\s*[:=]\s*([A-Za-z0-9\-_]+)",
        # UUID
        r"\b([0-9a-fA-F]{8}\-[0-9a-fA-F]{4}\-[0-9a-fA-F]{4}\-[0-9a-fA-F]{4}\-[0-9a-fA-F]{12})\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def _parse_ssh_from_text(text: str) -> Optional[str]:
    """
    Extract an ssh command line if present.
    """
    if not text:
        return None
    for line in text.splitlines():
        if "ssh " in line:
            # return from 'ssh' onward
            idx = line.find("ssh ")
            return line[idx:].strip()
    return None


def rent_vm() -> Tuple[str, Optional[str], str, str]:
    """
    Rent a VM and return (instance_id, ssh_cmd, stdout, stderr).
    We DO NOT rely solely on parsing rent output; we prefer instances --json diff.
    """
    before = { _instance_id_from_obj(x) for x in hyperbolic_instances_json() }
    before = {x for x in before if x}

    cmd = [
        "hyperbolic", "rent", "ondemand",
        "--instance-type", "virtual-machine",
        "--gpu-count", str(GPU_COUNT_PER_VM),
    ]
    rc, out, err = _run(cmd, timeout=300)
    if rc != 0:
        raise RuntimeError(f"hyperbolic rent failed rc={rc} out={out.strip()[:400]} err={err.strip()[:400]}")

    # Prefer: find new instance via instances diff
    after_objs = hyperbolic_instances_json()
    after_ids = []
    for obj in after_objs:
        iid = _instance_id_from_obj(obj)
        if iid:
            after_ids.append(iid)

    new_ids = [iid for iid in after_ids if iid not in before]
    if new_ids:
        instance_id = new_ids[0]
        # try to get ssh from the instance object (if present)
        ssh_cmd = None
        for obj in after_objs:
            if _instance_id_from_obj(obj) == instance_id:
                ssh_cmd = _ssh_cmd_from_obj(obj)
                break
        return instance_id, ssh_cmd, out, err

    # Fallback: parse rent output
    combined = (out + "\n" + err).strip()
    instance_id = _parse_instance_id_from_rent_output(combined)
    ssh_cmd = _parse_ssh_from_text(combined)
    if instance_id:
        return instance_id, ssh_cmd, out, err

    # If we still can’t figure it out, bubble up a *useful* error with stdout/stderr attached
    raise RuntimeError(
        "Could not determine instance_id after rent. "
        f"rent_stdout_head={out.strip()[:400]} rent_stderr_head={err.strip()[:400]}"
    )


def terminate_instance(instance_id: str) -> None:
    rc, out, err = _run(["hyperbolic", "terminate", instance_id], timeout=120)
    if rc != 0:
        raise RuntimeError(f"hyperbolic terminate failed id={instance_id} rc={rc} out={out.strip()[:300]} err={err.strip()[:300]}")


def ensure_tables() -> None:
    """
    Track instances we rent so we can scale down reliably even if CLI listing is flaky.
    """
    sql = """
    CREATE TABLE IF NOT EXISTS autoscaler_instances (
      instance_id TEXT PRIMARY KEY,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      terminated_at TIMESTAMPTZ NULL,
      ssh_command TEXT NULL
    );
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)


def db_backlog_count() -> int:
    """
    backlog = count of RUNNING jobs with executor_id IS NULL (unclaimed) for this worker pool.
    """
    sql = """
    SELECT COUNT(*) AS n
    FROM jobs
    WHERE status = 'RUNNING'
      AND executor_id IS NULL
      AND assigned_worker_id = %s;
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (ASSIGNED_WORKER_ID,))
            row = cur.fetchone()
            return int(row["n"] if isinstance(row, dict) else row[0])


def db_active_instances() -> List[Dict[str, Any]]:
    sql = """
    SELECT instance_id, ssh_command, created_at
    FROM autoscaler_instances
    WHERE terminated_at IS NULL
    ORDER BY created_at ASC;
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            return list(cur.fetchall())


def db_record_instance(instance_id: str, ssh_cmd: Optional[str]) -> None:
    sql = """
    INSERT INTO autoscaler_instances (instance_id, ssh_command)
    VALUES (%s, %s)
    ON CONFLICT (instance_id) DO UPDATE SET ssh_command = COALESCE(EXCLUDED.ssh_command, autoscaler_instances.ssh_command);
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (instance_id, ssh_cmd))


def db_mark_terminated(instance_id: str) -> None:
    sql = """
    UPDATE autoscaler_instances
    SET terminated_at = NOW()
    WHERE instance_id = %s;
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (instance_id,))


def desired_vm_count(backlog: int) -> int:
    if backlog <= 0:
        return 0
    return int(math.ceil(backlog / float(BACKLOG_PER_VM)))


def main_loop() -> None:
    ensure_tables()
    ensure_hyperbolic_auth()

    log(f"autoscaler started assigned_worker_id={ASSIGNED_WORKER_ID} backlog_per_vm={BACKLOG_PER_VM} poll={POLL_SECONDS}s")

    while True:
        try:
            backlog = db_backlog_count()
            want = desired_vm_count(backlog)
            active = db_active_instances()
            have = len(active)

            log(f"backlog={backlog} want_vms={want} have_vms={have}")

            # Scale up
            if have < want:
                to_add = want - have
                for _ in range(to_add):
                    log(f"renting {GPU_COUNT_PER_VM}x GPU VM on Hyperbolic...")
                    instance_id, ssh_cmd, out, err = rent_vm()
                    db_record_instance(instance_id, ssh_cmd)

                    # Important: show *something* helpful if ssh isn’t detected yet
                    if ssh_cmd:
                        log(f"rented instance_id={instance_id} ssh='{ssh_cmd[:160]}'")
                    else:
                        log(f"rented instance_id={instance_id} (ssh cmd not detected yet)")

            # Scale down
            elif have > want:
                to_kill = have - want
                # terminate newest first (reverse)
                kill_list = list(reversed(active))[:to_kill]
                for row in kill_list:
                    iid = row["instance_id"] if isinstance(row, dict) else row[0]
                    log(f"terminating instance_id={iid} ...")
                    terminate_instance(iid)
                    db_mark_terminated(iid)
                    log(f"terminated instance_id={iid}")

        except Exception as e:
            log(f"autoscaler error: {e}")

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main_loop()
