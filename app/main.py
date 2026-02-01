import os
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel

from .db import get_conn




# --- executor command mapping (brain -> RunPod executor) ---
RVC_SCRIPT = os.getenv("RVC_SCRIPT", "/opt/rvc_inferencing/auto_pitch_entry.py")
TRAIN_SCRIPT = os.getenv("TRAIN_SCRIPT", "/opt/hyperbolic_project/scripts/train_all_linux_hyper.sh")

def _cap_python(capabilities: dict, key: str, default: str) -> str:
    try:
        v = (capabilities or {}).get(key)
        return str(v) if v else default
    except Exception:
        return default

def _build_cmd(job: dict, capabilities: dict) -> dict:
    """Map a DB job row to the executor contract fields (cmd/workdir/env)."""
    jtype = str(job.get("job_type") or "").upper()
    username = str(job.get("username") or "")
    model_name = str(job.get("model_name") or "")
    input_key = job.get("input_key")

    rvc_py = _cap_python(capabilities, "rvc_python", os.getenv("RVC_PYTHON", "/opt/venv_rvc/bin/python"))
    uvr_py = _cap_python(capabilities, "uvr_python", os.getenv("UVR_PYTHON", "/opt/venv_uvr/bin/python"))

    env = {
        "RVC_PYTHON": rvc_py,
        "UVR_PYTHON": uvr_py,
        "USERNAME": username,
        "MODEL_NAME": model_name,
    }

    if jtype == "INFER":
        if not input_key:
            raise HTTPException(status_code=422, detail="INFER job missing input_key")
        # auto_pitch_entry.py expects --input (not --input_key)
        cmd = [rvc_py, RVC_SCRIPT, "--user", username, "--model_name", model_name, "--input", str(input_key)]
        return {"cmd": cmd, "workdir": None, "env": env}

    if jtype == "TRAIN":
        cmd = ["bash", "-lc", f"bash {TRAIN_SCRIPT} {username} {model_name}"]
        return {"cmd": cmd, "workdir": None, "env": env}

    raise HTTPException(status_code=422, detail=f"Unsupported job_type: {jtype}")

def _env(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name)
    return v if v is not None and v != "" else (default or "")


def _require_bearer_token(req: Request) -> str:
    """Simple Bearer-token auth for executor <-> brain calls."""
    auth = req.headers.get("authorization") or req.headers.get("Authorization")
    if not auth or not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    return auth.split(" ", 1)[1].strip()


def require_executor_auth(req: Request) -> None:
    expected = _env("EXECUTOR_TOKEN")
    if not expected:
        # Misconfiguration; fail closed.
        raise HTTPException(status_code=500, detail="Server misconfigured: EXECUTOR_TOKEN not set")
    got = _require_bearer_token(req)
    if got != expected:
        raise HTTPException(status_code=403, detail="Invalid token")


app = FastAPI()


class ClaimRequest(BaseModel):
    executor_id: str
    assigned_worker_id: str
    hostname: Optional[str] = None
    capabilities: Optional[Dict[str, Any]] = None


class HeartbeatRequest(BaseModel):
    job_id: str
    executor_id: str


class CompleteRequest(BaseModel):
    job_id: str
    executor_id: str
    ok: bool
    error_text: Optional[str] = None
    exit_code: Optional[int] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None


def _row_to_job(row: Dict[str, Any]) -> Dict[str, Any]:
    """Return a JSON-friendly dict, and include an `id` alias."""
    out = dict(row)
    # For backwards compat: executor_loop historically expected `id`.
    if "job_id" in out and "id" not in out:
        out["id"] = out["job_id"]
    return out


@app.post("/executors/claim")

def executors_claim(payload: ClaimRequest, _: None = Depends(require_executor_auth)):
    """Claim the oldest QUEUED job for this worker pool."""
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
WITH candidate AS (
  SELECT job_id
  FROM jobs
  WHERE status = 'QUEUED'
    AND (assigned_worker_id IS NULL OR assigned_worker_id = %s)
  ORDER BY created_at ASC
  FOR UPDATE SKIP LOCKED
  LIMIT 1
)
UPDATE jobs j
SET status = 'RUNNING',
    executor_id = %s,
    assigned_worker_id = %s,
    claimed_at = NOW(),
    heartbeat_at = NOW(),
    started_at = COALESCE(started_at, NOW()),
    attempts = COALESCE(attempts, 0) + 1
FROM candidate c
WHERE j.job_id = c.job_id
RETURNING j.*
                    """,
                    (payload.assigned_worker_id, payload.executor_id, payload.assigned_worker_id),
                )
                row = cur.fetchone()
                if not row:
                    return {"job": None}

                cols = [d.name for d in cur.description]
                job = {cols[i]: row[i] for i in range(len(cols))}
                job.update(_build_cmd(job, payload.capabilities or {}))
                return {"job": _row_to_job(job)}
    finally:
        conn.close()


@app.post("/executors/heartbeat")

def executors_heartbeat(payload: HeartbeatRequest, _: None = Depends(require_executor_auth)):
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
UPDATE jobs
SET heartbeat_at = NOW()
WHERE job_id = %s AND executor_id = %s AND status = 'RUNNING'
                    """,
                    (payload.job_id, payload.executor_id),
                )
                if cur.rowcount != 1:
                    raise HTTPException(status_code=404, detail="Job not found for executor")
        return {"ok": True}
    finally:
        conn.close()


@app.post("/executors/complete")
def executors_complete(payload: CompleteRequest, _: None = Depends(require_executor_auth)):
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                status = "SUCCEEDED" if payload.ok else "FAILED"
                err = payload.error_text
                if not payload.ok and not err:
                    # Prefer stderr, otherwise a generic message.
                    err = (payload.stderr or "").strip() or "Executor reported failure"

                cur.execute(
                    """
UPDATE jobs
SET status = %s,
    finished_at = NOW(),
    heartbeat_at = NOW(),
    error_text = %s
WHERE job_id = %s AND executor_id = %s
                    """,
                    (status, err, payload.job_id, payload.executor_id),
                )
                if cur.rowcount != 1:
                    raise HTTPException(status_code=404, detail="Job not found for executor")
        return {"ok": True}
    finally:
        conn.close()
