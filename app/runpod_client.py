"""
RunPod GraphQL client used by the brain autoscaler (Fly worker).

- Endpoint: https://api.runpod.io/graphql
- Auth header: Authorization: Bearer <RUNPOD_API_KEY>

This client intentionally keeps to a small surface area:
- list_gpu_types()
- list_pods()
- create_on_demand_pod()
- terminate_pod()
- resolve_gpu_type_id() helper used by worker
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import random
import time

import requests


class RunPodError(RuntimeError):
    pass


@dataclass
class RunPodClient:
    api_key: str
    api_base: str = "https://api.runpod.io"
    timeout_s: int = 60

    # Retry knobs (kept conservative; Fly networking can be spiky)
    max_attempts: int = 5
    backoff_s: float = 0.8  # base backoff; grows exponentially

    def __post_init__(self) -> None:
        self.endpoint = self.api_base.rstrip("/") + "/graphql"
        self.session = requests.Session()

    def _gql(self, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {"query": query}
        if variables is not None:
            payload["variables"] = variables

        last_err: Optional[BaseException] = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                r = self.session.post(
                    self.endpoint,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout_s,
                )

                # Retry on common transient HTTP statuses
                if r.status_code in (429, 500, 502, 503, 504):
                    if attempt < self.max_attempts:
                        sleep_s = (self.backoff_s * (2 ** (attempt - 1))) + random.uniform(0, 0.25)
                        time.sleep(sleep_s)
                        continue
                    raise RunPodError(f"RunPod HTTP {r.status_code}: {r.text[:500]}")

                if not r.ok:
                    raise RunPodError(f"RunPod HTTP {r.status_code}: {r.text[:500]}")

                data = r.json()
                if "errors" in data and data["errors"]:
                    msg = data["errors"][0].get("message") or str(data["errors"][0])
                    raise RunPodError(f"RunPod GraphQL error: {msg}")

                return data.get("data") or {}

            except requests.exceptions.RequestException as e:
                # Covers TLS handshake resets, ConnectionResetError, etc.
                last_err = e
                if attempt < self.max_attempts:
                    sleep_s = (self.backoff_s * (2 ** (attempt - 1))) + random.uniform(0, 0.25)
                    time.sleep(sleep_s)
                    continue
                raise RunPodError(f"RunPod request failed after {self.max_attempts} attempts: {e}") from e

        # Should never reach here
        raise RunPodError(f"RunPod request failed: {last_err}")

    # ---------- Queries ----------

    def list_gpu_types(self) -> List[Dict[str, Any]]:
        q = """
        query gpuTypes($input: GpuTypeFilter) {
          gpuTypes(input: $input) {
            id
            displayName
            memoryInGb
            secureCloud
            communityCloud
          }
        }
        """
        out = self._gql(q, {"input": None})
        return out.get("gpuTypes") or []

    def list_pods(self) -> List[Dict[str, Any]]:
        # IMPORTANT:
        # RunPod schema does NOT expose gpuTypeId directly on Pod.
        # It's under pod.machine.gpuTypeId (if present).
        q = """
        query myself {
          myself {
            pods {
              id
              name
              desiredStatus
              createdAt
              gpuCount
              runtime { uptimeInSeconds }
              machine {
                gpuTypeId
                gpuDisplayName
              }
            }
          }
        }
        """
        out = self._gql(q)
        myself = out.get("myself") or {}
        return myself.get("pods") or []

    # ---------- Mutations ----------

    def create_on_demand_pod(
        self,
        *,
        name: str,
        image_name: str,
        gpu_count: int,
        gpu_type_id: str,
        cloud_type: str = "SECURE",
        container_disk_gb: int = 60,
        volume_gb: int = 0,
        min_vcpu: Optional[int] = None,
        min_mem_gb: Optional[int] = None,
        env: Optional[List[Dict[str, str]]] = None,
        datacenter_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        q = """
        mutation podFindAndDeployOnDemand($input: PodFindAndDeployOnDemandInput!) {
          podFindAndDeployOnDemand(input: $input) {
            id
            name
            desiredStatus
            createdAt
            gpuCount
            machine {
              gpuTypeId
              gpuDisplayName
            }
          }
        }
        """
        inp: Dict[str, Any] = {
            "name": name,
            "imageName": image_name,
            "cloudType": cloud_type,
            "gpuCount": gpu_count,
            "gpuTypeId": gpu_type_id,
            "containerDiskInGb": container_disk_gb,
            "startJupyter": False,
            "startSsh": False,
            "supportPublicIp": False,
        }
        if env:
            inp["env"] = env
        if volume_gb and volume_gb > 0:
            inp["volumeInGb"] = volume_gb
            inp["volumeMountPath"] = "/workspace"
        if min_vcpu is not None:
            inp["minVcpuCount"] = int(min_vcpu)
        if min_mem_gb is not None:
            inp["minMemoryInGb"] = int(min_mem_gb)
        if datacenter_id:
            inp["dataCenterId"] = datacenter_id

        out = self._gql(q, {"input": inp})
        return out.get("podFindAndDeployOnDemand") or {}

    def terminate_pod(self, pod_id: str) -> None:
        q = """
        mutation podTerminate($input: PodTerminateInput!) {
          podTerminate(input: $input)
        }
        """
        self._gql(q, {"input": {"podId": pod_id}})

    # ---------- Helpers ----------

    @staticmethod
    def resolve_gpu_type_id(
        gpu_types: List[Dict[str, Any]],
        token: str,
        *,
        min_vram_gb: int = 48,
        disallow_blackwell: bool = True,
    ) -> Optional[str]:
        """
        Resolve `token` (either a display name like 'A40' or an id) to a gpuTypeId.

        The worker passes tokens from RUNPOD_GPU_LADDER in order of preference.
        """
        if not token:
            return None

        def is_disallowed(name: str) -> bool:
            if not disallow_blackwell:
                return False
            n = (name or "").lower()
            return any(x in n for x in ["blackwell", "b200", "b100", "rtx pro 6000"])

        # First pass: exact match by id or displayName
        for g in gpu_types:
            gid = g.get("id")
            dname = g.get("displayName") or ""
            mem = int(g.get("memoryInGb") or 0)
            if mem < min_vram_gb:
                continue
            if is_disallowed(dname):
                continue
            if token == gid or token.lower() == dname.lower():
                return gid

        # Second pass: substring match (helps if ladder uses "A100 80GB", etc.)
        t = token.lower()
        for g in gpu_types:
            gid = g.get("id")
            dname = (g.get("displayName") or "").lower()
            mem = int(g.get("memoryInGb") or 0)
            if mem < min_vram_gb:
                continue
            if is_disallowed(dname):
                continue
            if t in dname:
                return gid

        return None
