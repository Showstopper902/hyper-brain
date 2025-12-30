import os, time
from .db import get_conn

POLL_SECONDS = int(os.getenv("POLL_SECONDS", "3"))

# TEMP until we implement real registered GPU workers
DEFAULT_WORKER_ID = os.getenv("DEFAULT_WORKER_ID", "hyperbolic-pool")

def dispatch_one_job():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                update public.jobs
                   set status='RUNNING',
                       assigned_worker_id=%s,
                       started_at=now()
                 where job_id = (
                    select job_id
                      from public.jobs
                     where status='QUEUED'
                     order by created_at asc
                     for update skip locked
                     limit 1
                 )
                returning job_id, username, model_name, job_type;
                """,
                (DEFAULT_WORKER_ID,),
            )
            job = cur.fetchone()
    return job

def run():
    print("[worker] started")
    while True:
        job = dispatch_one_job()
        if not job:
            time.sleep(POLL_SECONDS)
            continue

        job_id = str(job["job_id"])
        user = job["username"]
        model = job["model_name"]
        jtype = job.get("job_type", "TRAIN")
        print(f"[worker] claimed {job_id} {jtype} {user}/{model} -> {DEFAULT_WORKER_ID}")

        # IMPORTANT: submit+return means we do NOT mark SUCCEEDED here.
        # Next phase: actually provision Hyperbolic worker(s) and notify them.

if __name__ == "__main__":
    run()
