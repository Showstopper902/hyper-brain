"""Fly worker process: autoscale RunPod pods for this brain.

This worker does *not* execute jobs. It only:
  - reads backlog from Postgres
  - lists existing RunPod pods with a name prefix
  - creates/terminates pods to reach the desired count

Scaling rule (default):
  - backlog = count(jobs where status='RUNNING' and executor_id is NULL
                   and assigned_worker_id matches this brain)
  - want_pods = 0 if backlog == 0 else ceil(backlog / BACKLOG_PER_POD)

RunPod pods should run an "executor loop" container which calls:
  POST /executors/claim and POST /executors/complete on this brain.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import string
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import psycopg

from .runpod_client import RunPodClient, RunPodError


LOG = logging.getLogger("brain-worker")


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except ValueError:
        raise RuntimeError(f"{name} must be an int, got {v!r}")


def _env_str(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return default if v is None else v


def _rand_suffix(n: int = 6) -> str:
    return "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(n))


@dataclass(frozen=True)
class Config:
    database_url: str
    assigned_worker_id: str

    # RunPod
    runpod_api_key: str
    runpod_api_base: str
    runpod_cloud_type: str
    runpod_executor_image: str
    runpod_name_prefix: str

    # Scaling
    backlog_per_pod: int
    max_pods: int
    poll_seconds: int
    scale_down_grace_seconds: int

    # Pod resources
    container_disk_gb: int
    volume_gb: int
    min_vcpu: int
    min_mem_gb: int

    # Selection
    gpu_ladder: List[str]
    datacenter_ladder: List[str]

    # Env pass-through
    pod_env_passthrough: List[str]


def load_config() -> Config:
    database_url = _env_str("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required")

    assigned_worker_id = _env_str("ASSIGNED_WORKER_ID")
    if not assigned_worker_id:
        raise RuntimeError("ASSIGNED_WORKER_ID is required")

    runpod_api_key = _env_str("RUNPOD_API_KEY")
    if not runpod_api_key:
        raise RuntimeError("RUNPOD_API_KEY is required")

    runpod_executor_image = _env_str("RUNPOD_EXECUTOR_IMAGE")
    if not runpod_executor_image:
        raise RuntimeError("RUNPOD_EXECUTOR_IMAGE is required")

    runpod_api_base = _env_str("RUNPOD_API_BASE", "https://api.runpod.io")
    # Using ALL gives the allocator the widest set of hosts; you can tighten
    # this to SECURE later if you only want verified/secure providers.
    runpod_cloud_type = _env_str("RUNPOD_CLOUD_TYPE", "ALL")
    runpod_name_prefix = _env_str("RUNPOD_POD_NAME_PREFIX", "hyper-exec")

    backlog_per_pod = _env_int("RUNPOD_BACKLOG_PER_POD", 250)
    max_pods = _env_int("RUNPOD_MAX_PODS", 25)
    poll_seconds = _env_int("RUNPOD_POLL_SECONDS", 5)
    scale_down_grace_seconds = _env_int("RUNPOD_SCALE_DOWN_GRACE_SECONDS", 0)

    container_disk_gb = _env_int("RUNPOD_CONTAINER_DISK_GB", 50)
    volume_gb = _env_int("RUNPOD_VOLUME_GB", 80)
    min_vcpu = _env_int("RUNPOD_MIN_VCPU", 4)
    min_mem_gb = _env_int("RUNPOD_MIN_MEM_GB", 16)

    gpu_ladder = [s.strip() for s in _env_str(
        "RUNPOD_GPU_LADDER",
        "A40,RTX A6000,L40S,L40,RTX 6000 Ada,A100,H100,H200",
    ).split(",") if s.strip()]

    datacenter_ladder = [s.strip() for s in _env_str("RUNPOD_DATACENTER_LADDER", "").split(",") if s.strip()]

    # Comma-separated list of env var names to pass through to the executor pod.
    # Keep this list tight—only secrets/values the executor needs.
    pod_env_passthrough = [s.strip() for s in _env_str(
        "RUNPOD_POD_ENV_PASSTHROUGH",
        "EXECUTOR_TOKEN,BRAIN_URL,ASSIGNED_WORKER_ID,B2_S3_ENDPOINT,B2_BUCKET,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,AWS_DEFAULT_REGION",
    ).split(",") if s.strip()]

    return Config(
        database_url=database_url,
        assigned_worker_id=assigned_worker_id,
        runpod_api_key=runpod_api_key,
        runpod_api_base=runpod_api_base,
        runpod_cloud_type=runpod_cloud_type,
        runpod_executor_image=runpod_executor_image,
        runpod_name_prefix=runpod_name_prefix,
        backlog_per_pod=backlog_per_pod,
        max_pods=max_pods,
        poll_seconds=poll_seconds,
        scale_down_grace_seconds=scale_down_grace_seconds,
        container_disk_gb=container_disk_gb,
        volume_gb=volume_gb,
        min_vcpu=min_vcpu,
        min_mem_gb=min_mem_gb,
        gpu_ladder=gpu_ladder,
        datacenter_ladder=datacenter_ladder,
        pod_env_passthrough=pod_env_passthrough,
    )


def get_backlog_and_inflight(conn: psycopg.Connection, assigned_worker_id: str) -> Tuple[int, int]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
              SUM(CASE WHEN status='RUNNING' AND executor_id IS NULL THEN 1 ELSE 0 END) AS backlog,
              SUM(CASE WHEN status='RUNNING' AND executor_id IS NOT NULL THEN 1 ELSE 0 END) AS inflight
            FROM jobs
            WHERE assigned_worker_id = %s
            """,
            (assigned_worker_id,),
        )
        row = cur.fetchone()
        backlog = int(row[0] or 0)
        inflight = int(row[1] or 0)
        return backlog, inflight


def want_pods_for_backlog(backlog: int, backlog_per_pod: int) -> int:
    if backlog <= 0:
        return 0
    return int(math.ceil(backlog / float(backlog_per_pod)))


def filter_managed_pods(pods: List[Dict[str, Any]], name_prefix: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in pods:
        name = (p.get("name") or "")
        if not name.startswith(name_prefix):
            continue
        status = (p.get("desiredStatus") or "").upper()
        if status in {"TERMINATED", "DELETED"}:
            continue
        out.append(p)
    return out


def pod_uptime_seconds(p: Dict[str, Any]) -> int:
    rt = p.get("runtime") or {}
    try:
        return int(rt.get("uptimeInSeconds") or 0)
    except Exception:
        return 0


def build_executor_env(cfg: Config) -> List[Dict[str, str]]:
    env_items: List[Dict[str, str]] = []
    for key in cfg.pod_env_passthrough:
        val = os.getenv(key)
        if val is None or val == "":
            continue
        env_items.append({"key": key, "value": val})
    # Convenience: expose the worker prefix so executors can include it in logs.
    env_items.append({"key": "RUNPOD_POD_NAME_PREFIX", "value": cfg.runpod_name_prefix})
    return env_items


def try_create_one_pod(client: RunPodClient, cfg: Config) -> Dict[str, Any]:
    gpu_types = client.list_gpu_types()

    # Build a resolved list of actual gpuTypeId strings to try.
    resolved: List[str] = []
    for token in cfg.gpu_ladder:
        match = client.resolve_gpu_type_id(gpu_types, token, min_vram_gb=48, disallow_blackwell=True)
        if match and match not in resolved:
            resolved.append(match)

    if not resolved:
        raise RuntimeError("Could not resolve any GPU types from ladder; check RUNPOD_GPU_LADDER")

    # We try: for gpu in resolved: for dc in [None] + datacenter_ladder
    dcs = [None] + (cfg.datacenter_ladder or [])
    last_err: Optional[BaseException] = None
    for gpu_type_id in resolved:
        for dc in dcs:
            name = f"{cfg.runpod_name_prefix}-{_rand_suffix()}"
            try:
                LOG.info("creating RunPod pod name=%s gpu=%s dc=%s", name, gpu_type_id, dc or "(auto)")
                pod = client.create_on_demand_pod(
                    name=name,
                    image_name=cfg.runpod_executor_image,
                    gpu_count=1,
                    gpu_type_id=gpu_type_id,
                    cloud_type=cfg.runpod_cloud_type,
                    container_disk_gb=cfg.container_disk_gb,
                    volume_gb=cfg.volume_gb,
                    min_vcpu=cfg.min_vcpu,
                    min_mem_gb=cfg.min_mem_gb,
                    env=build_executor_env(cfg),
                    datacenter_id=dc,
                )
                return pod
            except Exception as e:
                last_err = e
                LOG.warning("pod create failed gpu=%s dc=%s err=%s", gpu_type_id, dc or "(auto)", e)
                continue

    raise RuntimeError(f"All allocation attempts failed; last error: {last_err}")


def autoscaler_loop() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s][%(name)s] %(message)s",
    )
    cfg = load_config()
    client = RunPodClient(api_key=cfg.runpod_api_key, api_base=cfg.runpod_api_base)
    conn = psycopg.connect(cfg.database_url, autocommit=True)

    backlog_zero_since: Optional[float] = None

    while True:
        try:
            backlog, inflight = get_backlog_and_inflight(conn, cfg.assigned_worker_id)
            want = want_pods_for_backlog(backlog, cfg.backlog_per_pod)
            want = min(want, cfg.max_pods)

            # Optional scale-down grace to avoid churn (disabled by default).
            now = time.time()
            if backlog == 0 and cfg.scale_down_grace_seconds > 0:
                if backlog_zero_since is None:
                    backlog_zero_since = now
                if (now - backlog_zero_since) < cfg.scale_down_grace_seconds:
                    want = max(want, 1)  # keep one warm pod
            else:
                backlog_zero_since = None

            pods = filter_managed_pods(client.list_pods(), cfg.runpod_name_prefix)
            have = len(pods)

            LOG.info("backlog=%s inflight=%s want_pods=%s have_pods=%s", backlog, inflight, want, have)

            if have < want:
                need = want - have
                LOG.info("need %s more pod(s)", need)
                for _ in range(need):
                    try_create_one_pod(client, cfg)
                    time.sleep(0.25)

            elif have > want:
                extra = have - want
                LOG.info("terminating %s pod(s)", extra)
                # Terminate newest first (smallest uptime), so older pods stay warm.
                pods_sorted = sorted(pods, key=pod_uptime_seconds)
                for p in pods_sorted[:extra]:
                    pid = p.get("id")
                    if not pid:
                        continue
                    try:
                        LOG.info("terminating pod id=%s name=%s", pid, p.get("name"))
                        client.terminate_pod(str(pid))
                    except Exception as e:
                        LOG.warning("terminate failed id=%s err=%s", pid, e)

        except (RunPodError, psycopg.Error) as e:
            LOG.error("autoscaler error: %s", e, exc_info=True)
        except Exception as e:
            LOG.error("autoscaler error", exc_info=True)

        time.sleep(cfg.poll_seconds)


def main() -> None:
    autoscaler_loop()


if __name__ == "__main__":
    main()
