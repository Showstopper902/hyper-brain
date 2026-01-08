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

import requests


class RunPodError(RuntimeError):
    pass


@dataclass
class RunPodClient:
    api_key: str
    api_base: str = "https://api.runpod.io"
    timeout_s: int = 60

    def __post_init__(self) -> None:
        self.endpoint = self.api_base.rstrip("/") + "/graphql"

    def _gql(self, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {"query": query}
        if variables is not None:
            payload["variables"] = variables

        r = requests.post(self.endpoint, json=payload, headers=headers, timeout=self.timeout_s)
        if not r.ok:
            raise RunPodError(f"RunPod HTTP {r.status_code}: {r.text[:500]}")
        data = r.json()
        if "errors" in data and data["errors"]:
            # GraphQL errors often include useful messages
            msg = data["errors"][0].get("message") or str(data["errors"][0])
            raise RunPodError(f"RunPod GraphQL error: {msg}")
        return data.get("data") or {}

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
        # Using `myself { pods { ... } }` is the simplest way to list your pods.
        q = """
        query myself {
          myself {
            pods {
              id
              name
              desiredStatus
              createdAt
              gpuTypeId
              runtime {
                uptimeInSeconds
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
            gpuTypeId
            createdAt
          }
        }
        """
        inp: Dict[str, Any] = {
            "name": name,
            "imageName": image_name,
            "cloudType": cloud_type,
            "gpuCount": gpu_count,
            # prefer list form so RunPod can pick from a set if you choose to pass multiple
            "gpuTypeId": gpu_type_id,
            "containerDiskInGb": container_disk_gb,
            "startJupyter": False,
            "startSsh": False,
            "supportPublicIp": False,
        }
        if env:
            # GraphQL spec defines env as [EnvironmentVariableInput]
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
            # Conservative block list. Extend as needed.
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

        # Second pass: allow substring match (helps when ladder uses 'A100 80GB' etc.)
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
