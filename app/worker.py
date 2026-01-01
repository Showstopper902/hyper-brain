import os
import time
import json
import math
import shlex
import tempfile
import subprocess
import threading
from datetime import datetime, timezone

import psycopg


# ----------------------------
# Logging
# ----------------------------
def log(msg: str):
    # Fly logs will show this under your worker process
    print(f"[brain-worker] {msg}", flush=True)


# ----------------------------
# DB helpers
# ----------------------------
DB_DSN = os.getenv("DATABASE_URL") or os.getenv("DB_DSN")
if not DB_DSN:
    raise RuntimeError("Missing DATABASE_URL (or DB_DSN)")

ASSIGNED_WORKER_ID = os.getenv("ASSIGNED_WORKER_ID", "hyperbolic-pool")

DISPATCH_POLL_SECONDS = float(os.getenv("DISPATCH_POLL_SECONDS", "2"))
AUTOSCALE_POLL_SECONDS = float(os.getenv("AUTOSCALE_POLL_SECONDS", "12"))

# your scaling rule
BACKLOG_PER_VM = int(os.getenv("BACKLOG_PER_VM", "250"))

# Hyperbolic constraints (for now: VM + 1 GPU)
HYP_INSTANCE_TYPE = os.getenv("HYP_INSTANCE_TYPE", "virtual-machine")
HYP_GPU_COUNT = int(os.getenv("HYP_GPU_COUNT", "1"))

# optional safety: if you ever have other instances in the account
# set this to 1 to allow terminating instances; otherwise it will only scale up.
ALLOW_TERMINATE = os.getenv("ALLOW_TERMINATE", "1") == "1"

# Hyperbolic API key secret on Fly
HYPERBOLIC_API_KEY = (
    os.getenv("HYPERBOLIC_API_KEY")
    or os.getenv("HYPERBOLIC_KEY")
    or os.getenv("HYPERBOLIC_TOKEN")
)
if not HYPERBOLIC_API_KEY:
    log("WARNING: Missing HYPERBOLIC_API_KEY (or HYPERBOLIC_KEY/HYPERBOLIC_TOKEN). Autoscaler will fail auth.")


def db_connect():
    return psycopg.connect(DB_DSN)


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def count_backlog(conn) -> int:
    """
    backlog = RUNNING jobs assigned to our worker pool but not yet claimed by an executor
            = status='RUNNING' AND assigned_worker_id=... AND executor_id IS NULL
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*)
            FROM jobs
            WHERE status='RUNNING'
              AND assigned_worker_id=%s
              AND executor_id IS NULL
            """,
            (ASSIGNED_WORKER_ID,),
        )
        return int(cur.fetchone()[0])


def claim_next_queued_job(conn):
    """
    Move one QUEUED job to RUNNING and assign to the worker pool.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH next_job AS (
              SELECT job_id
              FROM jobs
              WHERE status='QUEUED'
              ORDER BY created_at ASC
              LIMIT 1
              FOR UPDATE SKIP LOCKED
            )
            UPDATE jobs
            SET status='RUNNING',
                assigned_worker_id=%s,
                started_at=NOW()
            WHERE job_id IN (SELECT job_id FROM next_job)
            RETURNING job_id, job_type, username, model_name, input_key
            """,
            (ASSIGNED_WORKER_ID,),
        )
        row = cur.fetchone()
        if row:
            conn.commit()
            return {
                "job_id": row[0],
                "job_type": row[1],
                "username": row[2],
                "model_name": row[3],
                "input_key": row[4],
            }
        conn.rollback()
        return None


# ----------------------------
# Hyperbolic CLI helpers
# ----------------------------
def run_cmd(cmd, *, env=None, input_text=None, timeout=60):
    """
    Run a command and return (rc, stdout, stderr).
    """
    try:
        p = subprocess.run(
            cmd,
            input=input_text,
            text=True,
            capture_output=True,
            env=env,
            timeout=timeout,
        )
        return p.returncode, p.stdout or "", p.stderr or ""
    except Exception as e:
        return 999, "", f"{type(e).__name__}: {e}"


def ensure_hyperbolic_auth() -> dict:
    """
    Hyperbolic CLI needs a config file. On Fly, assume it's not there.
    We create an isolated HOME and run auth set-key in multiple compatible ways.
    Returns env override dict you should pass to all hyperbolic subprocess calls.
    """
    if not HYPERBOLIC_API_KEY:
        raise RuntimeError("No HYPERBOLIC_API_KEY env var found.")

    # Put Hyperbolic CLI config in a stable path in the container.
    # /tmp is fine because we re-auth on each worker start anyway.
    home_dir = "/tmp/hyperbolic_home"
    os.makedirs(home_dir, exist_ok=True)

    env = os.environ.copy()
    env["HOME"] = home_dir

    # Try multiple auth methods to be compatible across hyperbolic-cli versions.
    attempts = [
        # common cobra style: positional arg
        (["hyperbolic", "auth", "set-key", HYPERBOLIC_API_KEY], None),
        # flag variants
        (["hyperbolic", "auth", "set-key", "--api-key", HYPERBOLIC_API_KEY], None),
        (["hyperbolic", "auth", "set-key", "--key", HYPERBOLIC_API_KEY], None),
        # stdin (some CLIs prompt)
        (["hyperbolic", "auth", "set-key"], HYPERBOLIC_API_KEY + "\n"),
        # older/alternate syntax (your error text mentioned it)
        (["hyperbolic", "auth", HYPERBOLIC_API_KEY], None),
    ]

    last = None
    for cmd, stdin in attempts:
        rc, out, err = run_cmd(cmd, env=env, input_text=stdin, timeout=45)
        last = (cmd, rc, out, err)
        if rc == 0:
            # verify with auth status (best-effort)
            rc2, out2, err2 = run_cmd(["hyperbolic", "auth", "status"], env=env, timeout=20)
            log(f"hyperbolic auth OK (rc={rc}); status_rc={rc2}")
            return env

    # If we got here, auth failed
    cmd, rc, out, err = last
    raise RuntimeError(
        "Hyperbolic auth failed.\n"
        f"cmd={shlex.join(cmd)} rc={rc}\n"
        f"stdout={out[:400]!r}\n"
        f"stderr={err[:400]!r}"
    )


def hyperbolic_instances_json(hyp_env: dict):
    rc, out, err = run_cmd(["hyperbolic", "instances", "--json"], env=hyp_env, timeout=45)

    # Some failures still return rc=0 but output is error text
    out_stripped = (out or "").strip()
    if not out_stripped:
        raise RuntimeError(f"hyperbolic instances --json returned empty output. stderr={err[:300]!r}")

    try:
        data = json.loads(out_stripped)
        return data
    except Exception as e:
        raise RuntimeError(
            "hyperbolic instances --json returned non-JSON output.\n"
            f"stdout={out_stripped[:400]!r}\n"
            f"stderr={err[:400]!r}\n"
            f"parse_err={e}"
        )


def extract_active_instances(instances_payload):
    """
    Be defensive: payload might be a list or a dict with 'instances'.
    """
    if isinstance(instances_payload, list):
        inst_list = instances_payload
    elif isinstance(instances_payload, dict):
        inst_list = instances_payload.get("instances") or instances_payload.get("data") or []
        if not isinstance(inst_list, list):
            inst_list = []
    else:
        inst_list = []

    active = []
    for inst in inst_list:
        if not isinstance(inst, dict):
            continue
        status = str(inst.get("status") or "").lower()
        # treat these as "active enough"
        if status in ("running", "active", "ready", "provisioning"):
            active.append(inst)
        elif status == "" and inst.get("id"):
            # if status missing, assume active (better than scaling wrong)
            active.append(inst)

    return active


def instance_id(inst: dict) -> str:
    return str(inst.get("id") or inst.get("instance_id") or "")


def rent_one_vm(hyp_env: dict):
    cmd = [
        "hyperbolic",
        "rent",
        "ondemand",
        "--instance-type",
        HYP_INSTANCE_TYPE,
        "--gpu-count",
        str(HYP_GPU_COUNT),
    ]
    rc, out, err = run_cmd(cmd, env=hyp_env, timeout=180)
    if rc != 0:
        raise RuntimeError(f"rent failed rc={rc} stdout={out[:400]!r} stderr={err[:400]!r}")
    log(f"rented VM (stdout_head={out.strip()[:200]!r})")


def terminate_vm(hyp_env: dict, inst_id: str):
    if not inst_id:
        return
    rc, out, err = run_cmd(["hyperbolic", "terminate", inst_id], env=hyp_env, timeout=60)
    if rc != 0:
        raise RuntimeError(f"terminate {inst_id} failed rc={rc} stdout={out[:300]!r} stderr={err[:300]!r}")
    log(f"terminated VM {inst_id} (stdout_head={out.strip()[:200]!r})")


# ----------------------------
# Loops
# ----------------------------
def dispatch_loop():
    log("dispatcher loop starting")
    conn = db_connect()
    try:
        while True:
            job = claim_next_queued_job(conn)
            if job:
                log(f"dispatched job_id={job['job_id']} type={job['job_type']} user={job['username']} model={job['model_name']}")
            time.sleep(DISPATCH_POLL_SECONDS)
    except Exception as e:
        log(f"dispatcher error: {type(e).__name__}: {e}")
        raise
    finally:
        try:
            conn.close()
        except Exception:
            pass


def autoscale_loop():
    log("autoscaler loop starting")
    hyp_env = None
    # authenticate once per worker start (safe even if already authed)
    try:
        hyp_env = ensure_hyperbolic_auth()
    except Exception as e:
        log(f"autoscaler error: {e}")
        # keep looping; maybe secrets not set yet
        hyp_env = None

    conn = db_connect()
    try:
        while True:
            try:
                backlog = count_backlog(conn)
                want_vms = 0 if backlog == 0 else int(math.ceil(backlog / BACKLOG_PER_VM))

                if hyp_env is None:
                    # retry auth periodically
                    hyp_env = ensure_hyperbolic_auth()

                inst_payload = hyperbolic_instances_json(hyp_env)
                active = extract_active_instances(inst_payload)
                have_vms = len(active)

                log(f"backlog={backlog} want_vms={want_vms} have_vms={have_vms}")

                # scale up
                if want_vms > have_vms:
                    for _ in range(want_vms - have_vms):
                        rent_one_vm(hyp_env)

                # scale down (optional but matches your "no backlog=no instances")
                if ALLOW_TERMINATE and want_vms < have_vms:
                    # terminate extras (oldest-first if we can sort)
                    # best-effort: if created_at exists, use it; else just slice.
                    def created_key(i):
                        v = i.get("created_at") or i.get("createdAt") or ""
                        return str(v)

                    active_sorted = sorted(active, key=created_key)
                    to_kill = active_sorted[: (have_vms - want_vms)]
                    for inst in to_kill:
                        terminate_vm(hyp_env, instance_id(inst))

            except Exception as e:
                log(f"autoscaler error: {e}")

            time.sleep(AUTOSCALE_POLL_SECONDS)

    finally:
        try:
            conn.close()
        except Exception:
            pass


def main():
    # run both loops
    t1 = threading.Thread(target=dispatch_loop, daemon=True)
    t2 = threading.Thread(target=autoscale_loop, daemon=True)
    t1.start()
    t2.start()

    # keep process alive
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
