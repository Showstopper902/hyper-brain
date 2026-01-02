# app/worker.py
from __future__ import annotations

import base64
import json
import logging
import math
import os
import sys
import time
import traceback
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

LOG = logging.getLogger("brain-worker")


# ----------------------------
# Small utilities
# ----------------------------

def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v not in (None, "") else default


def _now() -> float:
    return time.time()


def _sleep_s(s: float) -> None:
    time.sleep(max(0.0, s))


def _head(s: str, n: int = 500) -> str:
    s = s or ""
    return s[:n]


# ----------------------------
# DB backlog counting
#   We try psycopg (v3), psycopg2, then SQLAlchemy if installed.
#   This keeps the worker from dying depending on which lib is in requirements.
# ----------------------------

BACKLOG_SQL = "SELECT COUNT(*) FROM jobs WHERE status='RUNNING' AND executor_id IS NULL"
INFLIGHT_SQL = "SELECT COUNT(*) FROM jobs WHERE status='RUNNING' AND executor_id IS NOT NULL"

def get_job_counts() -> Tuple[int, int]:
    db_url = _env("DATABASE_URL")
    if not db_url:
        return (0, 0)

    # Try psycopg (v3)
    try:
        import psycopg  # type: ignore
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(BACKLOG_SQL)
                backlog = int(cur.fetchone()[0])
                cur.execute(INFLIGHT_SQL)
                inflight = int(cur.fetchone()[0])
        return backlog, inflight
    except Exception:
        pass

    # Try psycopg2
    try:
        import psycopg2  # type: ignore
        conn = psycopg2.connect(db_url)
        try:
            cur = conn.cursor()
            cur.execute(BACKLOG_SQL)
            backlog = int(cur.fetchone()[0])
            cur.execute(INFLIGHT_SQL)
            inflight = int(cur.fetchone()[0])
            cur.close()
            return backlog, inflight
        finally:
            conn.close()
    except Exception:
        pass

    # Try SQLAlchemy (if present)
    try:
        from sqlalchemy import create_engine, text  # type: ignore
        eng = create_engine(db_url, pool_pre_ping=True)
        with eng.connect() as c:
            backlog = int(c.execute(text(BACKLOG_SQL)).scalar_one())
            inflight = int(c.execute(text(INFLIGHT_SQL)).scalar_one())
        return backlog, inflight
    except ModuleNotFoundError as e:
        raise RuntimeError(
            f"Cannot read backlog: no postgres client library installed. "
            f"Install one of: psycopg, psycopg2-binary, sqlalchemy. ({e})"
        )
    except Exception as e:
        raise RuntimeError(f"Cannot read backlog from DB: {e}")


# ----------------------------
# Hyperbolic on-demand (tRPC) client
#   Matches the browser calls you captured:
#   /v2/ondemand.getRentalOptions
#   /v2/ondemand.getActiveVirtualMachineRentals
#   /v2/ondemand.createVirtualMachineRental
#   /v2/ondemand.terminateVirtualMachineRental
# ----------------------------

@dataclass
class HyperbolicTRPC:
    base_url: str
    token: str
    timeout_s: int = 20

    def _headers(self) -> Dict[str, str]:
        # Some stacks accept api keys in different headers. This is harmless if ignored.
        return {
            "accept": "*/*",
            "content-type": "application/json",
            "trpc-accept": "application/jsonl",
            "x-trpc-source": "brain-worker",
            "authorization": f"Bearer {self.token}",
            "x-api-key": self.token,
            "X-API-Key": self.token,
        }

    def _url(self, procedure: str, query: str) -> str:
        return f"{self.base_url}/v2/{procedure}{query}"

    def _request(self, method: str, url: str, body: Optional[bytes]) -> Tuple[int, str]:
        req = urllib.request.Request(url, method=method, data=body, headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read()
                # response is usually gzip in browsers; urllib handles transparently sometimes,
                # but if not, we still treat as bytes->utf-8 best-effort:
                txt = raw.decode("utf-8", errors="replace")
                return resp.status, txt
        except Exception as e:
            # Normalize urllib errors that still contain response bodies
            if hasattr(e, "code"):
                code = getattr(e, "code", 0) or 0
                try:
                    raw = e.read()  # type: ignore
                    txt = raw.decode("utf-8", errors="replace")
                except Exception:
                    txt = str(e)
                return int(code), txt
            raise

    def _parse_trpc(self, txt: str) -> Any:
        """
        tRPC batch responses commonly look like:
          [{"result":{"data":{"json": ...}}}]
        sometimes with JSONL lines. We support both.
        """
        s = (txt or "").strip()
        if not s:
            raise ValueError("empty response")

        # JSON array
        if s.startswith("["):
            arr = json.loads(s)
            return arr

        # JSONL: one json object per line
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        if not lines:
            raise ValueError("no JSON lines")
        objs = [json.loads(ln) for ln in lines]
        return objs

    def _unwrap_json(self, parsed: Any) -> Any:
        """
        Unwrap the first batch item into its `result.data.json` payload if present.
        """
        if isinstance(parsed, list) and parsed:
            item = parsed[0]
            if isinstance(item, dict):
                # tRPC batch
                j = item
                for k in ("result", "data", "json"):
                    if isinstance(j, dict) and k in j:
                        j = j[k]
                    else:
                        break
                return j
        return parsed

    def get_rental_options(self) -> Any:
        input_param = urllib.parse.quote(json.dumps({"0": {"json": {}}}))
        url = self._url("ondemand.getRentalOptions", f"?batch=1&input={input_param}")
        code, txt = self._request("GET", url, None)
        if code != 200:
            raise RuntimeError(f"getRentalOptions HTTP {code}: {_head(txt)}")
        return self._unwrap_json(self._parse_trpc(txt))

    def get_active_rentals(self) -> List[Dict[str, Any]]:
        input_param = urllib.parse.quote(json.dumps({"0": {"json": {}}}))
        url = self._url("ondemand.getActiveVirtualMachineRentals", f"?batch=1&input={input_param}")
        code, txt = self._request("GET", url, None)
        if code != 200:
            raise RuntimeError(f"getActiveVirtualMachineRentals HTTP {code}: {_head(txt)}")
        data = self._unwrap_json(self._parse_trpc(txt))
        # Expecting a list-ish structure; return best-effort
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        if isinstance(data, dict) and "rentals" in data and isinstance(data["rentals"], list):
            return [x for x in data["rentals"] if isinstance(x, dict)]
        # Unknown schema, return empty instead of crashing
        return []

    def create_rental(
        self,
        gpu_count: int,
        region: str,
        gpu_type: str,
        label: Optional[str],
        term_type: str = "on-demand",
        monitoring_enabled: Optional[bool] = None,
    ) -> Any:
        payload: Dict[str, Any] = {
            "gpuCount": gpu_count,
            "region": region,
            "gpuType": gpu_type,
            "label": label,
            "termType": term_type,
            "monitoringEnabled": monitoring_enabled,
        }
        # Match browser "meta.values" pattern so undefineds serialize consistently
        meta = {"values": {"label": ["undefined"] if label is None else ["string"],
                           "monitoringEnabled": ["undefined"] if monitoring_enabled is None else ["boolean"]}}
        body = json.dumps({"0": {"json": payload, "meta": meta}}).encode("utf-8")
        url = self._url("ondemand.createVirtualMachineRental", "?batch=1")
        code, txt = self._request("POST", url, body)
        if code != 200:
            raise RuntimeError(f"createVirtualMachineRental HTTP {code}: {_head(txt)}")
        return self._unwrap_json(self._parse_trpc(txt))

    def terminate_rental(self, rental_id: int) -> Any:
        body = json.dumps({"0": {"json": {"rentalId": int(rental_id)}}}).encode("utf-8")
        url = self._url("ondemand.terminateVirtualMachineRental", "?batch=1")
        code, txt = self._request("POST", url, body)
        if code != 200:
            raise RuntimeError(f"terminateVirtualMachineRental HTTP {code}: {_head(txt)}")
        return self._unwrap_json(self._parse_trpc(txt))


# ----------------------------
# Autoscaler
# ----------------------------

class Autoscaler:
    def __init__(self) -> None:
        self.api_base = _env("HYPERBOLIC_API_BASE", "https://api.hyperbolic.xyz")
        # IMPORTANT: prefer a real auth token if you have it
        self.token = _env("HYPERBOLIC_AUTH_TOKEN") or _env("HYPERBOLIC_API_KEY") or ""
        self.label_prefix = _env("HYPERBOLIC_LABEL_PREFIX", "hyper-brain")
        self.region = _env("HYPERBOLIC_REGION", "us-central-1")
        self.gpu_type = _env("HYPERBOLIC_GPU_TYPE", "h100")
        self.gpu_per_vm = int(_env("HYPERBOLIC_GPU_PER_VM", "1") or "1")
        self.max_backlog_per_vm = int(_env("MAX_BACKLOG_PER_VM", "250") or "250")
        self.tick_s = float(_env("AUTOSCALER_TICK_SECONDS", "4") or "4")
        self.cooldown_s = float(_env("AUTOSCALER_COOLDOWN_SECONDS", "15") or "15")
        self._last_scale_at = 0.0

        if not self.token:
            raise RuntimeError("Missing HYPERBOLIC_AUTH_TOKEN or HYPERBOLIC_API_KEY")

        self.h = HyperbolicTRPC(base_url=self.api_base, token=self.token)

    def _want_vms(self, backlog: int) -> int:
        if backlog <= 0:
            return 0
        return max(1, int(math.ceil(backlog / float(self.max_backlog_per_vm))))

    def _is_ours(self, r: Dict[str, Any]) -> bool:
        # We set label; if API doesn’t return label, we can’t filter — so we keep it conservative.
        lab = r.get("label")
        if isinstance(lab, str) and lab.startswith(self.label_prefix):
            return True
        return False

    def _extract_rental_id(self, r: Dict[str, Any]) -> Optional[int]:
        # browser terminate uses rentalId integer
        rid = r.get("rentalId") or r.get("id")
        try:
            return int(rid)
        except Exception:
            return None

    def _summarize_rentals(self, rentals: List[Dict[str, Any]]) -> str:
        ids = []
        for r in rentals:
            rid = self._extract_rental_id(r)
            if rid is not None:
                ids.append(str(rid))
        return ",".join(ids[:20]) + ("..." if len(ids) > 20 else "")

    def tick(self) -> None:
        backlog, inflight = get_job_counts()
        want = self._want_vms(backlog)

        rentals = self.h.get_active_rentals()
        ours = [r for r in rentals if self._is_ours(r)]
        have = len(ours)

        LOG.info("backlog=%s inflight=%s want_vms=%s have_vms=%s", backlog, inflight, want, have)

        # Cooldown to avoid thrashing on noisy counts
        now = _now()
        if now - self._last_scale_at < self.cooldown_s:
            return

        if have < want:
            need = want - have
            LOG.info("need %s more VM(s)", need)
            for i in range(need):
                label = f"{self.label_prefix}-{int(now)}-{i}"
                LOG.info("renting %sx GPU VM on Hyperbolic (gpuType=%s region=%s label=%s)...",
                         self.gpu_per_vm, self.gpu_type, self.region, label)
                try:
                    resp = self.h.create_rental(
                        gpu_count=self.gpu_per_vm,
                        region=self.region,
                        gpu_type=self.gpu_type,
                        label=label,
                        term_type="on-demand",
                        monitoring_enabled=None,
                    )
                    LOG.info("rent success: %s", _head(json.dumps(resp, default=str), 800))
                except Exception as e:
                    LOG.error("rent failed: %s", e)
                    LOG.error("rent failed traceback:\n%s", traceback.format_exc())

            self._last_scale_at = now

        elif have > want:
            excess = have - want
            LOG.info("have %s excess VM(s); terminating...", excess)
            # terminate arbitrary ones (best-effort). If API returns timestamps, you can sort here.
            terminated = 0
            for r in ours:
                if terminated >= excess:
                    break
                rid = self._extract_rental_id(r)
                if rid is None:
                    continue
                LOG.info("terminating rentalId=%s", rid)
                try:
                    resp = self.h.terminate_rental(rid)
                    LOG.info("terminate success: %s", _head(json.dumps(resp, default=str), 500))
                    terminated += 1
                except Exception as e:
                    LOG.error("terminate failed rentalId=%s: %s", rid, e)
                    LOG.error("terminate traceback:\n%s", traceback.format_exc())

            self._last_scale_at = now


def configure_logging() -> None:
    lvl = (_env("LOG_LEVEL", "INFO") or "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, lvl, logging.INFO),
        format="%(asctime)s [%(levelname)s][brain-worker] %(message)s",
        stream=sys.stdout,
    )


def main() -> None:
    configure_logging()
    LOG.info("worker booting")

    # Validate DB (optional)
    try:
        b, i = get_job_counts()
        LOG.info("db ok (backlog=%s inflight=%s)", b, i)
    except Exception as e:
        LOG.error("db check failed (autoscaling will break): %s", e)

    # Validate Hyperbolic auth + endpoint reachability
    try:
        a = Autoscaler()
        # quick probe to produce a clean error if auth is wrong
        _ = a.h.get_rental_options()
        LOG.info("hyperbolic probe ok (getRentalOptions)")
    except Exception as e:
        LOG.error("hyperbolic probe failed: %s", e)
        LOG.error("This usually means your token/key is not accepted by /v2/ondemand.* endpoints.")
        LOG.error("See notes below in chat on what token to use.")
        # Don't crash the machine; keep looping so deploy succeeds and logs remain visible.
        a = None  # type: ignore

    while True:
        try:
            if a is not None:
                a.tick()
        except Exception:
            LOG.error("autoscaler error:\n%s", traceback.format_exc())
        _sleep_s(float(_env("AUTOSCALER_TICK_SECONDS", "4") or "4"))


if __name__ == "__main__":
    main()
