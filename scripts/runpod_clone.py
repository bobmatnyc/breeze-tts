#!/usr/bin/env python3
"""Invoke the RunPod Serverless voice-clone endpoint and save the returned WAV.

Why: the endpoint replaces `pod-synth.sh`'s SSH-into-a-pod clone mode. This
script is the only client needed — it reads the API key, ships the reference
clip as base64, and writes the result to disk.

What: POSTs to `/runsync` by default and falls back to `/run` plus `/status`
polling when `--async` is passed or a long text would outrun the sync timeout.

Usage:
    python scripts/runpod_clone.py \\
        --endpoint-id <id> \\
        --text "This is a test of voice cloning" \\
        --ref-audio outputs/Recording_1_ref20.wav \\
        --ref-text "..." \\
        --output outputs/clone.wav

Test: `test_load_api_key_prefers_environment`, `test_build_payload_encodes_audio`
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import requests

BASE_URL = "https://api.runpod.ai/v2"
DEFAULT_ENV_FILE = Path(__file__).resolve().parents[1] / ".env.local"
API_KEY_NAME = "RUNPOD_API_KEY"


def load_api_key(env_file: Path = DEFAULT_ENV_FILE) -> str:
    """Read the RunPod key from the environment, else from a dotenv file.

    Test: `test_load_api_key_prefers_environment`, `test_load_api_key_reads_env_file`
    """
    key = os.environ.get(API_KEY_NAME)
    if key:
        return key.strip()
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            name, separator, value = line.strip().partition("=")
            if separator and name.strip() == API_KEY_NAME:
                return value.strip().strip("'\"")
    raise SystemExit(f"{API_KEY_NAME} is not set and was not found in {env_file}.")


def build_payload(
    text: str,
    ref_audio: Path,
    ref_text: str,
    *,
    seed: int,
    cfg_scale: float,
) -> dict[str, object]:
    """Assemble the job body the worker's `validate_job_input` expects.

    Test: `test_build_payload_encodes_audio`
    """
    audio = ref_audio.read_bytes()
    if not audio:
        raise SystemExit(f"Reference audio is empty: {ref_audio}")
    return {
        "input": {
            "text": text,
            "ref_audio_b64": base64.b64encode(audio).decode("ascii"),
            "ref_text": ref_text,
            "seed": seed,
            "cfg_scale": cfg_scale,
        }
    }


def _headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def run_sync(
    endpoint_id: str, api_key: str, payload: dict[str, object], timeout: int
) -> dict[str, object]:
    response = requests.post(
        f"{BASE_URL}/{endpoint_id}/runsync",
        headers=_headers(api_key),
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def run_async(
    endpoint_id: str,
    api_key: str,
    payload: dict[str, object],
    timeout: int,
    poll_seconds: float = 2.0,
) -> dict[str, object]:
    """Submit to `/run`, then poll `/status` until the job leaves the queue."""
    submitted = requests.post(
        f"{BASE_URL}/{endpoint_id}/run",
        headers=_headers(api_key),
        json=payload,
        timeout=60,
    )
    submitted.raise_for_status()
    job_id = submitted.json()["id"]
    print(f"queued job {job_id}", file=sys.stderr)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = requests.get(
            f"{BASE_URL}/{endpoint_id}/status/{job_id}",
            headers=_headers(api_key),
            timeout=60,
        )
        status.raise_for_status()
        body = status.json()
        if body.get("status") in {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}:
            return body
        time.sleep(poll_seconds)
    raise SystemExit(f"Job {job_id} did not finish within {timeout}s.")


def _summarize(body: dict[str, object]) -> None:
    """Print everything except the base64 audio body."""
    redacted = json.loads(json.dumps(body))
    output = redacted.get("output")
    if isinstance(output, dict) and "audio_b64" in output:
        output["audio_b64"] = f"<{len(body['output']['audio_b64'])} base64 chars>"
    print(json.dumps(redacted, indent=2), file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--endpoint-id", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--ref-audio", type=Path, required=True)
    parser.add_argument("--ref-text", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument(
        "--async",
        dest="use_async",
        action="store_true",
        help="Submit to /run and poll /status instead of blocking on /runsync.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Seconds to wait for the job to finish.",
    )
    args = parser.parse_args()

    if not args.ref_audio.is_file():
        raise SystemExit(f"Reference audio not found: {args.ref_audio}")

    api_key = load_api_key()
    payload = build_payload(
        args.text,
        args.ref_audio,
        args.ref_text,
        seed=args.seed,
        cfg_scale=args.cfg_scale,
    )

    started = time.perf_counter()
    if args.use_async:
        body = run_async(args.endpoint_id, api_key, payload, args.timeout)
    else:
        body = run_sync(args.endpoint_id, api_key, payload, args.timeout)
    wall_seconds = time.perf_counter() - started

    _summarize(body)

    output = body.get("output")
    if body.get("status") != "COMPLETED" or not isinstance(output, dict):
        print(f"job did not complete: status={body.get('status')}", file=sys.stderr)
        return 1
    if "error" in output:
        print(f"worker error: {output['error']}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(base64.b64decode(output["audio_b64"]))

    delay_ms = body.get("delayTime")
    execution_ms = body.get("executionTime")
    print(f"saved {args.output}")
    print(f"  sample rate      : {output.get('sample_rate')} Hz")
    print(f"  audio duration   : {output.get('audio_seconds')} s")
    print(f"  client wall time : {wall_seconds:.2f} s")
    print(f"  endpoint delay   : {delay_ms} ms (queue + cold start)")
    print(f"  endpoint execute : {execution_ms} ms")
    print(f"  worker model load: {output.get('model_load_seconds')} s")
    print(f"  worker generate  : {output.get('generation_seconds')} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
