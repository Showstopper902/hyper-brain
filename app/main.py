from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from .db import get_conn
import os

app = FastAPI()

# =========================
# Auth (simple shared secret)
# =========================
EXECUTOR_TOKEN = os.getenv("EXECUTOR_TOKEN", "")

def require_executor_auth(authorization: str | None):
    # Expect: Authorization: Bearer <token>
    if not EXECUTOR_TOKEN:
        # If you forgot to set it, fail closed so you notice.
        raise HTTPException(status_code=500, detail="EXECUTOR_TOKEN not configured")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if token != EXECUTOR_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

class TrainRequest(BaseModel):
    username: str
    model_name: str

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/jobs/train")
def create_train_job(req: TrainRequest):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into public.jobs (job_type, username, model_name)
                values ('TRAIN', %s, %s)
                returning job_id, status, created_at
                """,
                (req.username, req.model_name),
            )
            row = cur.fetchone()
    return row

# =========================
# Executor API
# =========================

class ClaimRequest(BaseModel):
    assigned_worker_id: str          # e.g. "hyperbolic-pool"
    executor_id: str                 # unique per GPU machine, e.g. hostname/instance id

@app.post("/executors/claim")
def claim_next_job(req: ClaimRequest, authorization: str | None = Header(default=None)):
    require_executor_auth(authorization)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                update public.jobs
                   set executor_id = %s,
                       claimed_at = now()
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
                returning job_id, job_type, username, model_name, created_at;
                """,
                (req.executor_id, req.assigned_worker_id),
            )
            job = cur.fetchone()

    # Return null when no work (clean for curl loops)
    return {"job": job}

class CompleteRequest(BaseModel):
    job_id: str
    executor_id: str
    ok: bool
    error_text: str | None = None

@app.post("/executors/complete")
def complete_job(req: CompleteRequest, authorization: str | None = Header(default=None)):
    require_executor_auth(authorization)

    status = "SUCCEEDED" if req.ok else "FAILED"

    with get_conn() as conn:
        with conn.cursor() as cur:
            # executor_id guard prevents one machine from completing another's job
            cur.execute(
                """
                update public.jobs
                   set status = %s,
                       finished_at = now(),
                       error_text = %s
                 where job_id = %s
                   and executor_id = %s
                returning job_id, status, finished_at;
                """,
                (status, req.error_text, req.job_id, req.executor_id),
            )
            row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=409, detail="Job not found or executor mismatch")

    return row
