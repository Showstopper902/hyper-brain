# app/worker.py
import os
import time

from app.db import get_conn


POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "3"))
ASSIGNED_WORKER_ID = os.environ.get("ASSIGNED_WORKER_ID", "hyperbolic-pool")


def main():
    while True:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    update public.jobs
                       set status = 'RUNNING',
                           assigned_worker_id = %s,
                           started_at = coalesce(started_at, now())
                     where job_id = (
                        select job_id
                          from public.jobs
                         where status = 'QUEUED'
                         order by created_at asc
                         for update skip locked
                         limit 1
                     )
                    returning job_id, job_type, username, model_name
                    """,
                    (ASSIGNED_WORKER_ID,),
                )
                row = cur.fetchone()
                conn.commit()

        # If no work, just sleep
        if not row:
            time.sleep(POLL_SECONDS)
            continue

        # This worker's only job is dispatching QUEUED->RUNNING.
        # The Hyperbolic executors will claim RUNNING jobs and do the work.
        time.sleep(0.1)


if __name__ == "__main__":
    main()
