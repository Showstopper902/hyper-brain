import os
import re
from typing import Any, Dict, Optional
from collections.abc import Mapping

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import boto3
from botocore.config import Config as BotoConfig

from .db import get_conn


# --- executor command mapping (brain -> RunPod executor) ---
RVC_SCRIPT = os.getenv("RVC_SCRIPT", "/opt/rvc_inferencing/auto_pitch_entry.py")
TRAIN_SCRIPT = os.getenv("TRAIN_SCRIPT", "/opt/hyperbolic_project/scripts/train_all_linux_hyper.sh")


def _cap_python(capabilities: dict, key: str, default: str) -> str:
    try:
        v = (capabilities or {}).get(key)
        return str(v) if v else default
    except Exception:
        return default


def _build_cmd(job: dict, capabilities: dict) -> dict:
    """Map a DB job row to the executor contract fields (cmd/workdir/env)."""
    jtype = str(job.get("job_type") or "").upper()
    username = str(job.get("username") or "")
    model_name = str(job.get("model_name") or "")
    input_key = job.get("input_key")

    rvc_py = _cap_python(capabilities, "rvc_python", os.getenv("RVC_PYTHON", "/opt/venv_rvc/bin/python"))
    uvr_py = _cap_python(capabilities, "uvr_python", os.getenv("UVR_PYTHON", "/opt/venv_uvr/bin/python"))

    # Baseline environment variables required by downstream scripts.
    env = {
        "RVC_PYTHON": rvc_py,
        "UVR_PYTHON": uvr_py,
        "USERNAME": username,
        "MODEL_NAME": model_name,
    }
    if input_key is not None:
        env["INPUT_KEY"] = input_key

    # Legacy inference jobs (convert an uploaded song).
    if jtype == "INFER":
        if not input_key:
            raise HTTPException(status_code=422, detail="INFER job missing input_key")
        cmd = [rvc_py, RVC_SCRIPT, "--user", username, "--model_name", model_name, "--input", str(input_key)]
        return {"cmd": cmd, "workdir": None, "env": env}

    # Training jobs by invoking the training shell script.
    if jtype == "TRAIN":
        cmd = ["bash", "-lc", f"bash {TRAIN_SCRIPT} {username} {model_name}"]
        return {"cmd": cmd, "workdir": None, "env": env}

    # INFER_SONG jobs
    if jtype == "INFER_SONG":
        song_name = str(job.get("song_name") or "")
        if not song_name:
            ik = str(input_key or "")
            parts = ik.split("/")
            if len(parts) >= 3:
                song_name = parts[-1]
        if not song_name:
            raise HTTPException(status_code=422, detail="INFER_SONG job missing song_name")

        env["SONG_NAME"] = song_name

        # Temporary: pass MiniMax prompt/lyrics/settings via env.
        env["MINIMAX_LYRICS"] = str(job.get("lyrics") or "")
        env["MINIMAX_PROMPT"] = str(job.get("prompt_full") or job.get("prompt_user") or "")
        env["MINIMAX_MODEL"] = str(job.get("minimax_model") or "music-2.5")
        env["MINIMAX_OUTPUT_FORMAT"] = "url"
        env["MINIMAX_AUDIO_FORMAT"] = str(job.get("minimax_audio_format") or "mp3")
        env["MINIMAX_SAMPLE_RATE"] = str(job.get("minimax_sample_rate") or 44100)
        env["MINIMAX_BITRATE"] = str(job.get("minimax_bitrate") or 256000)

        script_path = "/opt/hyperbolic_project/runpod_executor/infer_song_pipeline.py"
        cmd = [uvr_py, script_path]
        return {"cmd": cmd, "workdir": None, "env": env}

    raise HTTPException(status_code=422, detail=f"Unsupported job_type: {jtype}")


def _env(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name)
    return v if v is not None and v != "" else (default or "")


# =========================
# Auth: executor (existing)
# =========================
def _require_bearer_token(req: Request) -> str:
    """Simple Bearer-token auth for executor <-> brain calls."""
    auth = req.headers.get("authorization") or req.headers.get("Authorization")
    if not auth or not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    return auth.split(" ", 1)[1].strip()


def require_executor_auth(req: Request) -> None:
    expected = _env("EXECUTOR_TOKEN")
    if not expected:
        raise HTTPException(status_code=500, detail="Server misconfigured: EXECUTOR_TOKEN not set")
    got = _require_bearer_token(req)
    if got != expected:
        raise HTTPException(status_code=403, detail="Invalid token")


# =========================
# Auth: Base44 (new)
# =========================
def require_base44_api_key(req: Request) -> None:
    expected = _env("BASE44_API_KEY")
    if not expected:
        raise HTTPException(status_code=500, detail="Server misconfigured: BASE44_API_KEY not set")

    got = req.headers.get("x-api-key") or req.headers.get("X-API-KEY") or req.headers.get("X-API-Key")
    if not got:
        raise HTTPException(status_code=401, detail="Missing X-API-KEY")
    if got != expected:
        raise HTTPException(status_code=403, detail="Invalid API key")


app = FastAPI()

# =========================
# CORS (new)
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://voiceforge.base44.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# B2 Presign (new)
# =========================
_ALLOWED_EXT = {".wav", ".mp3", ".m4a", ".flac"}
_SAFE_NAME_RE = re.compile(r"^[a-zA-Z0-9._ -]+$")


def _b2_s3_client():
    endpoint = _env("B2_S3_ENDPOINT")
    region = _env("AWS_DEFAULT_REGION", "us-west-004") or "us-west-004"
    if not endpoint:
        raise HTTPException(status_code=500, detail="B2_S3_ENDPOINT not set")

    # Credentials come from env vars you already use in RunPod
    aws_key = _env("AWS_ACCESS_KEY_ID")
    aws_secret = _env("AWS_SECRET_ACCESS_KEY")
    if not aws_key or not aws_secret:
        raise HTTPException(status_code=500, detail="AWS credentials not set for presign")

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret,
        config=BotoConfig(signature_version="s3v4"),
    )


def _validate_object_key(key: str) -> str:
    key = (key or "").strip()
    if not key or key.startswith("/") or ".." in key:
        raise HTTPException(status_code=400, detail="Invalid key")
    return key


def _validate_upload_filename(name: str) -> str:
    name = (name or "").strip()
    if not name or not _SAFE_NAME_RE.match(name):
        raise HTTPException(status_code=400, detail=f"Invalid filename: {name!r}")
    lower = name.lower()
    ext = "." + lower.split(".")[-1] if "." in lower else ""
    if ext not in _ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"Disallowed file extension: {ext}")
    return name


class PresignUploadFile(BaseModel):
    name: str
    content_type: str


class PresignUploadRequest(BaseModel):
    username: str
    model_name: str
    files: list[PresignUploadFile]


class PresignUploadItem(BaseModel):
    key: str
    url: str
    headers: Dict[str, str]


class PresignUploadResponse(BaseModel):
    bucket: str
    uploads: list[PresignUploadItem]


class PresignReadRequest(BaseModel):
    key: str
    download: Optional[bool] = False
    filename: Optional[str] = None


@app.post("/b2/presign-upload", response_model=PresignUploadResponse)
def b2_presign_upload(payload: PresignUploadRequest, _: None = Depends(require_base44_api_key)):
    bucket = _env("B2_BUCKET")
    if not bucket:
        raise HTTPException(status_code=500, detail="B2_BUCKET not set")

    s3 = _b2_s3_client()
    uploads: list[PresignUploadItem] = []

    username = (payload.username or "").strip()
    model_name = (payload.model_name or "").strip()
    if not username or not model_name:
        raise HTTPException(status_code=400, detail="username and model_name are required")

    expires = int(os.getenv("B2_PRESIGN_UPLOAD_EXPIRES", "900"))  # 15 min

    for f in payload.files:
        fname = _validate_upload_filename(f.name)
        ctype = (f.content_type or "application/octet-stream").strip()

        key = f"data/users/{username}/{model_name}/input/{fname}"
        key = _validate_object_key(key)

        url = s3.generate_presigned_url(
            ClientMethod="put_object",
            Params={"Bucket": bucket, "Key": key, "ContentType": ctype},
            ExpiresIn=expires,
        )

        uploads.append(PresignUploadItem(key=key, url=url, headers={"Content-Type": ctype}))

    return PresignUploadResponse(bucket=bucket, uploads=uploads)


@app.post("/b2/presign-read")
def b2_presign_read(payload: PresignReadRequest, _: None = Depends(require_base44_api_key)):
    bucket = _env("B2_BUCKET")
    if not bucket:
        raise HTTPException(status_code=500, detail="B2_BUCKET not set")

    s3 = _b2_s3_client()
    key = _validate_object_key(payload.key)

    params: Dict[str, Any] = {"Bucket": bucket, "Key": key}

    if payload.download:
        fname = (payload.filename or os.path.basename(key) or "audio.wav").strip()
        params["ResponseContentDisposition"] = f'attachment; filename="{fname}"'

    expires = int(os.getenv("B2_PRESIGN_READ_EXPIRES", "3600"))  # 60 min

    url = s3.generate_presigned_url(
        ClientMethod="get_object",
        Params=params,
        ExpiresIn=expires,
    )
    return {"url": url}


# =========================
# Existing executor routes
# =========================
class ClaimRequest(BaseModel):
    executor_id: str
    assigned_worker_id: str
    hostname: Optional[str] = None
    capabilities: Optional[Dict[str, Any]] = None


class HeartbeatRequest(BaseModel):
    job_id: str
    executor_id: str


class CompleteRequest(BaseModel):
    job_id: str
    executor_id: str
    ok: bool
    error_text: Optional[str] = None
    exit_code: Optional[int] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None


def _row_to_job(row: Dict[str, Any]) -> Dict[str, Any]:
    """Return a JSON-friendly dict, and include an `id` alias."""
    out = dict(row)
    if "job_id" in out and "id" not in out:
        out["id"] = out["job_id"]
    return out


@app.post("/executors/claim")
def executors_claim(payload: ClaimRequest, _: None = Depends(require_executor_auth)):
    """Claim the oldest QUEUED job for this worker pool (or reclaim a stale RUNNING job)."""
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
WITH candidate AS (
  SELECT job_id
  FROM jobs
  WHERE
    (
      status = 'QUEUED'
      AND (assigned_worker_id IS NULL OR assigned_worker_id = %s)
    )
    OR
    (
      status = 'RUNNING'
      AND heartbeat_at IS NOT NULL
      AND heartbeat_at < NOW() - INTERVAL '3 minutes'
      AND (assigned_worker_id IS NULL OR assigned_worker_id = %s)
    )
  ORDER BY created_at ASC
  FOR UPDATE SKIP LOCKED
  LIMIT 1
)
UPDATE jobs j
SET status = 'RUNNING',
    executor_id = %s,
    assigned_worker_id = %s,
    claimed_at = NOW(),
    heartbeat_at = NOW(),
    started_at = COALESCE(started_at, NOW()),
    attempts = COALESCE(attempts, 0) + 1
FROM candidate c
WHERE j.job_id = c.job_id
RETURNING j.*
                    """,
                    (
                        payload.assigned_worker_id,
                        payload.assigned_worker_id,
                        payload.executor_id,
                        payload.assigned_worker_id,
                    ),
                )
                row = cur.fetchone()
                if not row:
                    return {"job": None}
                if isinstance(row, Mapping):
                    job = dict(row)
                else:
                    cols = [d.name for d in cur.description]
                    job = {cols[i]: row[i] for i in range(len(cols))}
                job.update(_build_cmd(job, payload.capabilities or {}))
                return {"job": _row_to_job(job)}
    finally:
        conn.close()


@app.post("/executors/heartbeat")
def executors_heartbeat(payload: HeartbeatRequest, _: None = Depends(require_executor_auth)):
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
UPDATE jobs
SET heartbeat_at = NOW()
WHERE job_id = %s AND executor_id = %s AND status = 'RUNNING'
                    """,
                    (payload.job_id, payload.executor_id),
                )
                if cur.rowcount != 1:
                    raise HTTPException(status_code=404, detail="Job not found for executor")
        return {"ok": True}
    finally:
        conn.close()


@app.post("/executors/complete")
def executors_complete(payload: CompleteRequest, _: None = Depends(require_executor_auth)):
    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                status = "SUCCEEDED" if payload.ok else "FAILED"
                err = payload.error_text
                if not payload.ok and not err:
                    err = (payload.stderr or "").strip() or "Executor reported failure"
                cur.execute(
                    """
UPDATE jobs
SET status = %s,
    finished_at = NOW(),
    heartbeat_at = NOW(),
    error_text = %s
WHERE job_id = %s AND executor_id = %s
                    """,
                    (status, err, payload.job_id, payload.executor_id),
                )
                if cur.rowcount != 1:
                    raise HTTPException(status_code=404, detail="Job not found for executor")
        return {"ok": True}
    finally:
        conn.close()
