# app/main.py
import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from psycopg import sql

from app.db import get_conn

app = FastAPI()


# ----------------------------
# Auth
# ----------------------------
def _executor_token_expected() -> str:
    tok = os.environ.get("EXECUTOR_TOKEN")
    if not tok:
        raise RuntimeError("Missing EXECUTOR_TOKEN in environment (Fly secret).")
    return tok


def require_executor_auth(authorization: Optional[str] = None):
    # FastAPI injects headers via parameter name "authorization"
    expected = _executor_token_expected()
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    got = authorization.split(" ", 1)[1].strip()
    if got != expected:
        raise HTTPException(status_code=403, detail="Invalid token")


# ----------------------------
# Models
# ----------------------------
class TrainJobRequest(BaseModel):
    username: str
    model_name: str


class InferJobRequest(BaseModel):
    username: str
    model_name: str
    # IMPORTANT: filename only (e.g. "test_song.wav"), NOT "input/test_song.wav"
    input_key: str


class ClaimRequest(BaseModel):
    assigned_worker_id: str
    executor_id: str


class CompleteRequest(BaseModel):
    job_id: str
    executor_id: str
    ok: bool
    error_text: Optional[str] = None


# ----------------------------
# Job creation
# ----------------------------
@app.post("/jobs/train")
def create_train_job(req: TrainJobRequest):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into public.jobs (job_type, username, model_name)
                values ('TRAIN', %s, %s)
                returning job_id
                """,
                (req.username, req.model_name),
            )
            row = cur.fetchone()
            conn.commit()
            return {"job_id": row["job_id"]}


@app.post("/jobs/infer")
def create_infer_job(req: InferJobRequest):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into public.jobs (job_type, username, model_name, input_key)
                values ('INFER', %s, %s, %s)
                returning job_id
                """,
                (req.username, req.model_name, req.input_key),
            )
            row = cur.fetchone()
            conn.commit()
            return {"job_id": row["job_id"]}


# ----------------------------
# Executor endpoints
# ----------------------------
@app.post("/executors/claim", dependencies=[Depends(require_executor_auth)])
def executor_claim(req: ClaimRequest):
    """
    Claim the oldest RUNNING job for assigned_worker_id that doesn't have executor_id yet.
    This matches your SQL approach: FOR UPDATE SKIP LOCKED, oldest first.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                update public.jobs
                   set executor_id = %s,
                       claimed_at  = now()
                 where job_id = (
                    select job_id
                      from public.jobs
                     where status = 'RUNNING'
                       and assigned_worker_id = %s
                       and executor_id is null
                     order by created_at asc
                     for update skip locked
                     limit 1
                 )
                 returning
                   job_id, job_type, status,
                   username, model_name,
                   input_key,
                   created_at, started_at, finished_at,
                   assigned_worker_id, executor_id, claimed_at
                """,
                (req.executor_id, req.assigned_worker_id),
            )
            row = cur.fetchone()
            conn.commit()

    if not row:
        return {"job": None}
    return {"job": row}


@app.post("/executors/complete", dependencies=[Depends(require_executor_auth)])
def executor_complete(req: CompleteRequest):
    """
    Mark SUCCEEDED / FAILED and write finished_at + error_text.
    """
    new_status = "SUCCEEDED" if req.ok else "FAILED"

    with get_conn() as conn:
        with conn.cursor() as cur:
            # Guardrail: only the claiming executor can complete it
            cur.execute(
                """
                update public.jobs
                   set status      = %s,
                       finished_at = now(),
                       error_text  = %s
                 where job_id = %s
                   and executor_id = %s
                 returning job_id
                """,
                (new_status, req.error_text, req.job_id, req.executor_id),
            )
            row = cur.fetchone()
            conn.commit()

    if not row:
        raise HTTPException(status_code=404, detail="Job not found for this executor_id")

    return {"ok": True, "job_id": req.job_id, "status": new_status}
