# app/worker.py
import json
import logging
import math
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import psycopg  # in requirements already

from app.db import get_conn

LOG = logging.getLogger("brain-worker")

# ---- Hyperbolic Marketplace API ----
HYP_API_KEY = os.getenv("HYPERBOLIC_API_KEY", "")
# Marketplace API base that is actually documented/live (not the CLI ondemand endpoints)
HYP_MARKET_BASE = os.getenv("HYPERBOLIC_MARKETPLACE_BASE", "https://api.hyperbolic.xyz/v1/marketplace")

# ---- Scaling policy ----
POLL_SECONDS = float(os.getenv("POLL_SECONDS", "3"))
AUTOSCALE_SECONDS = float(os.getenv("AUTOSCALE_SECONDS", "6"))
BACKLOG_PER_VM = int(os.getenv("BACKLOG_PER_VM", "250"))

# ---- Executor bootstrap ----
BRAIN_URL = os.getenv("BRAIN_URL", "https://hyper-brain.fly.dev")
EXECUTOR_TOKEN = os.getenv("EXECUTOR_TOKEN", "")
ASSIGNED_WORKER_ID = os.getenv("ASSIGNED_WORKER_ID", "hyperbolic-pool")
POLL_SECONDS_VM = int(os.getenv("EXECUTOR_POLL_SECONDS", "3"))
IDLE_SECONDS_VM = int(os.getenv("EXECUTOR_IDLE_SECONDS", "3600"))

SSH_PRIVATE_KEY = os.getenv("SSH_PRIVATE_KEY", "")
HYP_SSH_USER_DEFAULT = os.getenv("HYPERBOLIC_SSH_USER", "ubuntu")

# Where to fetch the executor loop script from (your repo)
EXECUTOR_SCRIPT_URL = os.getenv(
    "EXECUTOR_SCRIPT_URL",
    "https://raw.githubusercontent.com/Showstopper902/hyperbolic_project/main/scripts/hyper_executor_loop.sh",
)

# B2 env values passed to the VM
B2_SYNC = os.getenv("B2_SYNC", "1")
B2_BUCKET = os.getenv("B2_BUCKET", "")
B2_S3_ENDPOINT = os.getenv("B2_S3_ENDPOINT", "")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "")

# User preference hints (best-effort; only applied if the marketplace objects expose these fields)
PREF_LOCATION = os.getenv("HYP_PREFERRED_LOCATION", "us-central-1")
FORCE_ETHERNET = os.getenv("HYP_NETWORK_PREFERENCE", "ethernet").lower() == "ethernet"


@dataclass
class Offer:
    cluster_name: str
    node_name: str
    location: str
    gpus_total: int
    gpus_available: int
    gpu_hourly_cost: float
    raw: Dict[str, Any]


def _http_json(method: str, url: str, payload: Optional[dict] = None) -> Any:
    if not HYP_API_KEY:
        raise RuntimeError("Missing HYPERBOLIC_API_KEY in env")

    data = None
    headers = {
        "Authorization": f"Bearer {HYP_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return json.loads(body) if body else None
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Hyperbolic API HTTP {e.code} for {url}. body_head={body[:300]!r}") from None
    except Exception as e:
        raise RuntimeError(f"Hyperbolic API request failed for {url}: {e}") from None


def list_offers() -> List[Offer]:
    j = _http_json("GET", HYP_MARKET_BASE)
    offers: List[Offer] = []

    # Docs show clusters each with nodes; we handle flexible shapes.
    clusters = j if isinstance(j, list) else j.get("clusters") or j.get("data") or []
    for c in clusters:
        cluster_name = c.get("cluster_name") or c.get("clusterName") or c.get("name") or ""
        nodes = c.get("nodes") or c.get("data") or []
        for n in nodes:
            node_name = n.get("node_name") or n.get("nodeName") or n.get("name") or ""
            location = n.get("location") or n.get("region") or ""
            gpus_total = int(n.get("gpus_total") or n.get("gpusTotal") or n.get("gpus") or 0)
            gpus_available = int(n.get("gpus_available") or n.get("gpusAvailable") or n.get("available_gpus") or gpus_total)
            cost = n.get("gpu_hourly_cost") or n.get("gpuHourlyCost") or n.get("hourly") or 0
            try:
                gpu_hourly_cost = float(cost)
            except Exception:
                gpu_hourly_cost = 0.0

            if cluster_name and node_name and gpus_total > 0:
                offers.append(
                    Offer(
                        cluster_name=cluster_name,
                        node_name=node_name,
                        location=location,
                        gpus_total=gpus_total,
                        gpus_available=gpus_available,
                        gpu_hourly_cost=gpu_hourly_cost,
                        raw=n,
                    )
                )
    return offers


def pick_offer(required_gpus: int) -> Optional[Offer]:
    offers = [o for o in list_offers() if o.gpus_available >= required_gpus]

    # Best-effort filters if fields exist
    def is_vm(o: Offer) -> bool:
        t = (o.raw.get("instance_type") or o.raw.get("instanceType") or "").lower()
        return ("virtual" in t) if t else True  # if unknown, don't block

    def is_ethernet(o: Offer) -> bool:
        net = (o.raw.get("network_type") or o.raw.get("networkType") or "").lower()
        if not net:
            return True  # unknown
        return ("ethernet" in net)

    offers = [o for o in offers if is_vm(o)]
    if FORCE_ETHERNET:
        offers = [o for o in offers if is_ethernet(o)]

    if not offers:
        return None

    def score(o: Offer) -> Tuple[int, float, int]:
        # lower is better
        loc_penalty = 0 if (PREF_LOCATION and PREF_LOCATION in (o.location or "")) else 1
        cost = o.gpu_hourly_cost if o.gpu_hourly_cost > 0 else 1e9
        # prefer nodes with closer-fit capacity
        cap_penalty = max(0, o.gpus_total - required_gpus)
        return (loc_penalty, cost, cap_penalty)

    offers.sort(key=score)
    return offers[0]


def ensure_tables() -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS autoscaler_instances (
                    instance_id TEXT PRIMARY KEY,
                    gpu_count   INT  NOT NULL,
                    cluster_name TEXT,
                    node_name    TEXT,
                    created_at  TIMESTAMPTZ DEFAULT NOW(),
                    bootstrapped_at TIMESTAMPTZ,
                    terminated_at TIMESTAMPTZ
                );
                """
            )


def backlog_count() -> int:
    # backlog definition per your spec:
    # RUNNING jobs where executor_id IS NULL
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM jobs
                WHERE status = 'RUNNING'
                  AND executor_id IS NULL
                """
            )
            row = cur.fetchone()
            return int(row[0]) if row else 0


def desired_vm_count(backlog: int) -> int:
    if backlog <= 0:
        return 0
    return int(math.ceil(backlog / float(BACKLOG_PER_VM)))


def tracked_instances() -> List[str]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT instance_id
                FROM autoscaler_instances
                WHERE terminated_at IS NULL
                ORDER BY created_at ASC
                """
            )
            return [r[0] for r in cur.fetchall()]


def record_instance(instance_id: str, gpu_count: int, cluster_name: str, node_name: str) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO autoscaler_instances (instance_id, gpu_count, cluster_name, node_name)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (instance_id) DO NOTHING
                """,
                (instance_id, gpu_count, cluster_name, node_name),
            )


def mark_bootstrapped(instance_id: str) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE autoscaler_instances
                SET bootstrapped_at = NOW()
                WHERE instance_id = %s
                """,
                (instance_id,),
            )


def mark_terminated(instance_id: str) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE autoscaler_instances
                SET terminated_at = NOW()
                WHERE instance_id = %s
                """,
                (instance_id,),
            )


def list_user_instances() -> List[Dict[str, Any]]:
    j = _http_json("GET", f"{HYP_MARKET_BASE}/instances")
    if isinstance(j, list):
        return j
    return j.get("instances") or j.get("data") or []


def find_instance(instance_id: str) -> Optional[Dict[str, Any]]:
    for inst in list_user_instances():
        iid = inst.get("instance_id") or inst.get("id") or inst.get("instanceId")
        if iid == instance_id:
            return inst
    return None


def create_instance(required_gpus: int) -> str:
    offer = pick_offer(required_gpus)
    if not offer:
        raise RuntimeError(f"No marketplace offer available with >= {required_gpus} GPUs")

    payload = {
        "cluster_name": offer.cluster_name,
        "node_name": offer.node_name,
        "gpu_count": required_gpus,
        # We are NOT using the optional `image` mode here.
        # We want a normal VM we can SSH into and bootstrap.
    }
    resp = _http_json("POST", f"{HYP_MARKET_BASE}/instances/create", payload)
    # Accept flexible response shapes
    instance_id = None
    if isinstance(resp, dict):
        instance_id = resp.get("instance_id") or resp.get("id") or resp.get("instanceId")
        if not instance_id and isinstance(resp.get("data"), dict):
            d = resp["data"]
            instance_id = d.get("instance_id") or d.get("id") or d.get("instanceId")

    if not instance_id:
        raise RuntimeError(f"Create instance did not return instance_id. resp_head={str(resp)[:300]!r}")

    record_instance(instance_id, required_gpus, offer.cluster_name, offer.node_name)
    return instance_id


def terminate_instance(instance_id: str) -> None:
    _http_json("POST", f"{HYP_MARKET_BASE}/instances/terminate", {"instance_id": instance_id})
    mark_terminated(instance_id)


def _write_ssh_key() -> str:
    if not SSH_PRIVATE_KEY.strip():
        raise RuntimeError("Missing SSH_PRIVATE_KEY in Fly secrets")

    path = "/tmp/hyperbolic_ssh_key"
    # Normalize line endings
    key = SSH_PRIVATE_KEY.replace("\r\n", "\n").strip() + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(key)
    os.chmod(path, 0o600)
    return path


def _extract_ssh_target(inst: Dict[str, Any]) -> Tuple[str, str, Optional[int]]:
    """
    Returns (user, host, port)
    Tries to parse common shapes like:
      - ssh_command: "ssh ubuntu@1.2.3.4"
      - ip_address + ssh_port
    """
    ssh_cmd = inst.get("ssh_command") or inst.get("sshCommand") or ""
    if ssh_cmd:
        # try parse "ssh user@host" and optional "-p 22"
        port = None
        m_port = re.search(r"\s-p\s+(\d+)", ssh_cmd)
        if m_port:
            port = int(m_port.group(1))
        m = re.search(r"ssh\s+(?:[^@]*\s+)?([A-Za-z0-9._-]+)@([A-Za-z0-9.\-]+)", ssh_cmd)
        if m:
            return m.group(1), m.group(2), port

    host = inst.get("ip_address") or inst.get("ipAddress") or inst.get("host") or inst.get("public_ip") or ""
    port = inst.get("ssh_port") or inst.get("sshPort")
    port = int(port) if port else None
    user = inst.get("ssh_user") or inst.get("sshUser") or HYP_SSH_USER_DEFAULT
    if host:
        return user, host, port
    raise RuntimeError(f"Could not determine SSH target for instance. keys={list(inst.keys())}")


def _ssh_run(user: str, host: str, key_path: str, cmd: str, port: Optional[int] = None) -> None:
    base = [
        "ssh",
        "-i",
        key_path,
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "ConnectTimeout=15",
    ]
    if port:
        base += ["-p", str(port)]
    base += [f"{user}@{host}", "bash", "-lc", cmd]
    subprocess.run(base, check=True)


def bootstrap_executor(instance_id: str) -> None:
    """
    Waits for SSH info then bootstraps the VM.
    """
    key_path = _write_ssh_key()

    # Poll instance until SSH info shows up
    inst = None
    for _ in range(90):  # ~90 * 2s = 3 min
        inst = find_instance(instance_id)
        if inst:
            try:
                _extract_ssh_target(inst)
                break
            except Exception:
                pass
        time.sleep(2)

    if not inst:
        raise RuntimeError("Instance never appeared in /instances list")

    user, host, port = _extract_ssh_target(inst)

    # Build the remote bootstrap script as a single bash -lc string
    # NOTE: We keep quoting conservative by using EOF blocks on the remote.
    remote = f"""
set -euo pipefail

sudo mkdir -p /data/secrets /data/bin

# --- B2 env ---
sudo bash -c 'cat > /data/secrets/b2.env <<\\EOF
B2_SYNC={shlex.quote(B2_SYNC)}
B2_BUCKET={shlex.quote(B2_BUCKET)}
B2_S3_ENDPOINT={shlex.quote(B2_S3_ENDPOINT)}
AWS_ACCESS_KEY_ID={shlex.quote(AWS_ACCESS_KEY_ID)}
AWS_SECRET_ACCESS_KEY={shlex.quote(AWS_SECRET_ACCESS_KEY)}
AWS_DEFAULT_REGION={shlex.quote(AWS_DEFAULT_REGION)}
EOF'
sudo chmod 600 /data/secrets/b2.env

# --- executor env ---
sudo bash -c 'cat > /data/secrets/hyper_executor.env <<\\EOF
EXECUTOR_TOKEN={shlex.quote(EXECUTOR_TOKEN)}
BRAIN_URL={shlex.quote(BRAIN_URL)}
ASSIGNED_WORKER_ID={shlex.quote(ASSIGNED_WORKER_ID)}
POLL_SECONDS={POLL_SECONDS_VM}
IDLE_SECONDS={IDLE_SECONDS_VM}
EXECUTOR_ID="exec-$(hostname)"
EOF'
sudo chmod 600 /data/secrets/hyper_executor.env

# --- executor loop ---
sudo curl -fsSL {shlex.quote(EXECUTOR_SCRIPT_URL)} -o /data/bin/hyper_executor_loop.sh
sudo chmod +x /data/bin/hyper_executor_loop.sh

# --- systemd unit ---
sudo tee /etc/systemd/system/hyper-executor.service >/dev/null <<'EOF'
[Unit]
Description=Hyperbolic GPU Executor Loop
After=network-online.target docker.service
Wants=network-online.target

[Service]
Type=simple
EnvironmentFile=-/data/secrets/hyper_executor.env
ExecStart=/data/bin/hyper_executor_loop.sh
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now hyper-executor.service
"""

    _ssh_run(user, host, key_path, remote, port=port)
    mark_bootstrapped(instance_id)


def autoscale_once() -> None:
    ensure_tables()

    b = backlog_count()
    want = desired_vm_count(b)
    tracked = tracked_instances()

    # Clean up tracked ids that no longer exist
    alive_ids = set()
    for iid in tracked:
        inst = find_instance(iid)
        if inst:
            alive_ids.add(iid)
        else:
            # if it vanished, mark terminated so we don't keep counting it
            mark_terminated(iid)

    have = len(alive_ids)
    LOG.info("backlog=%s want_vms=%s have_vms=%s", b, want, have)

    # Scale up
    while have < want:
        # 1 job per VM for now, and you prefer as few GPUs as possible
        for gpu_count in (1, 2, 4):
            try:
                LOG.info("renting %sx GPU VM on Hyperbolic Marketplace...", gpu_count)
                iid = create_instance(gpu_count)
                LOG.info("rented instance_id=%s (gpu_count=%s) - bootstrapping...", iid, gpu_count)
                bootstrap_executor(iid)
                have += 1
                break
            except Exception as e:
                LOG.info("rent attempt failed for gpu_count=%s: %s", gpu_count, e)
        else:
            raise RuntimeError("could not rent any VM size (1/2/4)")

    # Scale down (only if you explicitly want 0)
    if want == 0 and have > 0:
        # terminate everything we created (saves money vs waiting for VM idle shutdown)
        for iid in list(alive_ids):
            try:
                LOG.info("terminating instance_id=%s (no backlog)", iid)
                terminate_instance(iid)
            except Exception as e:
                LOG.info("terminate failed for %s: %s", iid, e)


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s][brain-worker] %(message)s",
        stream=sys.stdout,
    )

    if not EXECUTOR_TOKEN:
        LOG.info("WARNING: EXECUTOR_TOKEN missing; autoscaler can rent VMs but executor bootstrap will not work.")
    if not SSH_PRIVATE_KEY.strip():
        LOG.info("WARNING: SSH_PRIVATE_KEY missing; autoscaler can rent VMs but cannot bootstrap them.")

    last = 0.0
    while True:
        now = time.time()
        if now - last >= AUTOSCALE_SECONDS:
            last = now
            try:
                autoscale_once()
            except Exception as e:
                LOG.info("autoscaler error: %s", e)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
