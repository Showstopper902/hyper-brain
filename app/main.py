from fastapi import FastAPI
from pydantic import BaseModel
from .db import get_conn

app = FastAPI()

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
