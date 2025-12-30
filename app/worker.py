import os, time
from .db import get_conn

POLL_SECONDS = int(os.getenv("POLL_SECONDS", "3"))

def claim_one_job():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                update public.jobs
                   set status='RUNNING',
                       started_at=now()
                 where job_id = (
                    select job_id
                      from public.jobs
                     where status='QUEUED'
                     order by created_at asc
                     for update skip locked
                     limit 1
                 )
                returning job_id, username, model_name;
                """
            )
            job = cur.fetchone()
    return job

def mark_done(job_id: str, ok: bool, err: str | None = None):
    status = "SUCCEEDED" if ok else "FAILED"
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                update public.jobs
                   set status=%s,
                       finished_at=now(),
                       error_text=%s
                 where job_id=%s
                """,
                (status, err, job_id),
            )

def run():
    print("[worker] started")
    while True:
        job = claim_one_job()
        if not job:
            time.sleep(POLL_SECONDS)
            continue

        job_id = str(job["job_id"])
        user = job["username"]
        model = job["model_name"]
        print(f"[worker] claimed {job_id} {user}/{model}")

        try:
            # Placeholder for now: marks success quickly.
            # Next: we’ll replace this with Hyperbolic provisioning + docker run.
            time.sleep(2)
            mark_done(job_id, ok=True)
        except Exception as e:
            mark_done(job_id, ok=False, err=str(e))

if __name__ == "__main__":
    run()
