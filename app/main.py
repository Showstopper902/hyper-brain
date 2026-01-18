import os
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel

from .db import get_conn


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
    """Claim the oldest unclaimed RUNNING job for this worker."""
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                # Atomically claim one job.
                cur.execute(
                    """
WITH candidate AS (
  SELECT job_id
  FROM jobs
  WHERE status = 'RUNNING'
    AND assigned_worker_id = %s
    AND executor_id IS NULL
  ORDER BY created_at ASC
  FOR UPDATE SKIP LOCKED
  LIMIT 1
)
UPDATE jobs j
SET executor_id = %s,
    claimed_at = NOW(),
    heartbeat_at = NOW(),
    started_at = NOW(),
    attempts = COALESCE(attempts, 0) + 1
FROM candidate c
WHERE j.job_id = c.job_id
RETURNING j.*
                    """,
                    (payload.assigned_worker_id, payload.executor_id),
                )
                row = cur.fetchone()
                if not row:
                    return {"job": None}
                # psycopg returns tuples unless configured; map columns.
                cols = [d.name for d in cur.description]
                job = {cols[i]: row[i] for i in range(len(cols))}
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
