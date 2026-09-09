#!/usr/bin/env python3
"""OpenAI-compatible `/v1/audio/speech` in front of the RunPod endpoint.

Why: many tools can point their OpenAI TTS base URL at a custom server but
cannot send a reference clip — they send a voice *name*. This shim accepts
OpenAI's request shape, maps `voice` onto a name in the volume-backed voice
registry, and returns audio bytes.

What: a Starlette app with one route. It authenticates the caller against
`SHIM_API_KEY`, calls the RunPod endpoint (`/runsync`, falling back to `/run`
plus polling for long inputs), and transcodes the 24 kHz WAV the worker returns.

Test: ``tests/test_openai_shim.py``
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import requests
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

BASE_URL = "https://api.runpod.ai/v2"
DEFAULT_ENV_FILE = Path(__file__).resolve().parents[1] / ".env.local"

# Above this many characters a cold worker can outlast the /runsync window, so
# the request is queued and polled instead.
RUNSYNC_TEXT_LIMIT = 600
MAX_INPUT_CHARS = 5000

CONTENT_TYPES = {"wav": "audio/wav", "mp3": "audio/mpeg"}
# OpenAI defaults to mp3. `opus`, `aac`, `flac` and `pcm` are declined rather
# than silently served as something else.
SUPPORTED_FORMATS = tuple(CONTENT_TYPES)
DEFAULT_FORMAT = "mp3"


class ShimError(Exception):
    """A request the shim rejects, carrying the status code to return."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def read_env(name: str, env_file: Path = DEFAULT_ENV_FILE) -> str | None:
    """Read a setting from the environment, falling back to a dotenv file.

    Test: `test_read_env_prefers_environment`, `test_read_env_reads_env_file`
    """
    value = os.environ.get(name)
    if value:
        return value.strip()
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            key, separator, raw = line.strip().partition("=")
            if separator and key.strip() == name:
                return raw.strip().strip("'\"")
    return None


def check_authorization(header: str | None) -> None:
    """Reject a caller whose bearer token does not match `SHIM_API_KEY`.

    The shim is meant to sit on a network, so a missing `SHIM_API_KEY` is a
    configuration error rather than an invitation to serve everyone.

    Test: `test_missing_bearer_token_is_rejected`, `test_wrong_bearer_token_is_rejected`
    """
    expected = read_env("SHIM_API_KEY")
    if not expected:
        raise ShimError(500, "SHIM_API_KEY is not configured on this server.")
    if not header or not header.startswith("Bearer "):
        raise ShimError(401, "Missing bearer token.")
    if header[len("Bearer ") :].strip() != expected:
        raise ShimError(401, "Invalid bearer token.")


def validate_speech_request(body: object) -> dict[str, object]:
    """Map OpenAI's speech body onto a clone job.

    `model` is accepted and ignored — this server has one model. `speed` is
    rejected for anything but 1.0: nothing in the generation path takes a rate
    or duration parameter (`FastStreamingConfig`, `models/fast_streaming.py`),
    so honouring it would mean silently ignoring it.

    Test: `test_validate_speech_request_defaults_to_mp3`,
    `test_validate_speech_request_rejects_speed`
    """
    if not isinstance(body, dict):
        raise ShimError(400, "Request body must be a JSON object.")

    text = body.get("input")
    if not isinstance(text, str) or not text.strip():
        raise ShimError(400, "'input' is required and must be a non-empty string.")
    if len(text) > MAX_INPUT_CHARS:
        raise ShimError(400, f"'input' exceeds {MAX_INPUT_CHARS} characters.")

    voice = body.get("voice")
    if not isinstance(voice, str) or not voice.strip():
        raise ShimError(
            400, "'voice' is required and must name a voice registered on the endpoint."
        )

    response_format = body.get("response_format", DEFAULT_FORMAT)
    if response_format not in SUPPORTED_FORMATS:
        raise ShimError(
            400,
            f"'response_format' must be one of {list(SUPPORTED_FORMATS)}; "
            f"got {response_format!r}.",
        )

    speed = body.get("speed", 1.0)
    if not isinstance(speed, (int, float)) or isinstance(speed, bool):
        raise ShimError(400, "'speed' must be a number.")
    if float(speed) != 1.0:
        raise ShimError(
            400,
            "'speed' is not supported: the generation path exposes no rate or "
            "duration control, so any value but 1.0 would be ignored.",
        )

    return {"text": text, "voice": voice.strip(), "response_format": response_format}


def to_mp3(wav_bytes: bytes) -> bytes:
    """Transcode WAV to MP3 with ffmpeg.

    ffmpeg is used rather than a pure-python encoder because the serverless
    image already installs it (`docker/Dockerfile`) and no maintained pure-python
    MP3 *encoder* exists — the pure-python libraries decode only.

    Test: `test_to_mp3_reports_missing_ffmpeg`
    """
    binary = shutil.which("ffmpeg")
    if binary is None:
        raise ShimError(
            500,
            "response_format=mp3 needs ffmpeg on PATH. Install it, or request "
            "response_format=wav.",
        )
    result = subprocess.run(
        [
            binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "wav",
            "-i",
            "pipe:0",
            "-codec:a",
            "libmp3lame",
            "-q:a",
            "2",
            "-f",
            "mp3",
            "pipe:1",
        ],
        input=wav_bytes,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ShimError(500, f"ffmpeg failed: {result.stderr.decode()[:400]}")
    return result.stdout


def call_endpoint(text: str, voice: str, *, timeout: int = 900) -> dict[str, object]:
    """Run one clone job on the RunPod endpoint.

    Test: `test_call_endpoint_uses_runsync_for_short_text`,
    `test_call_endpoint_polls_for_long_text`
    """
    api_key = read_env("RUNPOD_API_KEY")
    endpoint_id = read_env("RUNPOD_ENDPOINT_ID")
    if not api_key or not endpoint_id:
        raise ShimError(
            500, "RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID must be set for the shim."
        )

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"input": {"op": "clone", "text": text, "voice": voice}}

    if len(text) <= RUNSYNC_TEXT_LIMIT:
        response = requests.post(
            f"{BASE_URL}/{endpoint_id}/runsync",
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()

    submitted = requests.post(
        f"{BASE_URL}/{endpoint_id}/run", headers=headers, json=payload, timeout=60
    )
    submitted.raise_for_status()
    job_id = submitted.json()["id"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = requests.get(
            f"{BASE_URL}/{endpoint_id}/status/{job_id}", headers=headers, timeout=60
        )
        status.raise_for_status()
        body = status.json()
        if body.get("status") in {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}:
            return body
        time.sleep(2.0)
    raise ShimError(504, f"Job {job_id} did not finish within {timeout}s.")


async def speech(request: Request) -> Response:
    """`POST /v1/audio/speech` — OpenAI's text-to-speech route."""
    try:
        check_authorization(request.headers.get("authorization"))
        try:
            body = json.loads(await request.body())
        except json.JSONDecodeError as exc:
            raise ShimError(400, f"Body is not valid JSON: {exc}") from exc
        parsed = validate_speech_request(body)

        result = call_endpoint(str(parsed["text"]), str(parsed["voice"]))
        output = result.get("output")
        if result.get("status") != "COMPLETED" or not isinstance(output, dict):
            raise ShimError(502, f"Endpoint job status {result.get('status')!r}.")
        if "error" in output:
            raise ShimError(400, str(output["error"]))

        audio = base64.b64decode(str(output["audio_b64"]))
        if parsed["response_format"] == "mp3":
            audio = to_mp3(audio)
        return Response(
            audio,
            media_type=CONTENT_TYPES[str(parsed["response_format"])],
            headers={"X-Truncated": str(bool(output.get("truncated"))).lower()},
        )
    except ShimError as exc:
        return JSONResponse(
            {"error": {"message": exc.message, "type": "invalid_request_error"}},
            status_code=exc.status,
        )


async def health(_request: Request) -> Response:
    return JSONResponse({"status": "ok"})


app = Starlette(
    routes=[
        Route("/v1/audio/speech", speech, methods=["POST"]),
        Route("/health", health, methods=["GET"]),
    ]
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("SHIM_HOST", "127.0.0.1"),
        port=int(os.environ.get("SHIM_PORT", "8080")),
        log_level="info",
    )
