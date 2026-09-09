#!/usr/bin/env python3
"""Client for the RunPod Serverless voice-clone endpoint.

Why: the endpoint replaces `pod-synth.sh`'s SSH-into-a-pod clone mode. This
script is the only client needed — it reads the API key, ships payloads, and
writes returned audio to disk.

What: four subcommands — `clone`, `register`, `list`, `delete`. A clone takes
either `--voice <name>` for a registered voice or `--ref-audio` with
`--ref-text`. Requests go to `/runsync` by default and to `/run` plus `/status`
polling with `--async`; a `/runsync` call that comes back still queued falls
through to the same polling rather than failing.

Usage:
    python scripts/runpod_clone.py register --endpoint-id <id> \\
        --name bob --ref-audio outputs/Recording_1_ref40.wav --ref-text "..."
    python scripts/runpod_clone.py clone --endpoint-id <id> \\
        --voice bob --text "Hello" --output outputs/hello.wav

Test: `test_load_api_key_prefers_environment`, `test_build_clone_payload_uses_voice`
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
TERMINAL_STATUSES = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}


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


def encode_audio(path: Path) -> str:
    """Base64-encode a reference clip, refusing an empty file.

    Test: `test_encode_audio_rejects_empty_file`
    """
    audio = path.read_bytes()
    if not audio:
        raise SystemExit(f"Reference audio is empty: {path}")
    return base64.b64encode(audio).decode("ascii")


def build_clone_payload(args: argparse.Namespace) -> dict[str, object]:
    """Assemble a clone job, using exactly one voice source.

    Test: `test_build_clone_payload_uses_voice`,
    `test_build_clone_payload_uses_inline_reference`,
    `test_build_clone_payload_rejects_both_sources`
    """
    job: dict[str, object] = {
        "op": "clone",
        "text": args.text,
        "seed": args.seed,
        "cfg_scale": args.cfg_scale,
    }
    if args.voice and args.ref_audio:
        raise SystemExit("Pass either --voice or --ref-audio, not both.")
    if args.voice:
        job["voice"] = args.voice
    elif args.ref_audio:
        if not args.ref_text:
            raise SystemExit("--ref-audio requires --ref-text.")
        job["ref_audio_b64"] = encode_audio(args.ref_audio)
        job["ref_text"] = args.ref_text
    else:
        raise SystemExit("A clone needs --voice or --ref-audio with --ref-text.")
    return {"input": job}


def _headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def poll_status(
    endpoint_id: str,
    api_key: str,
    job_id: str,
    *,
    timeout: int,
    poll_seconds: float = 2.0,
) -> dict[str, object]:
    """Poll `/status/<job_id>` until the job reaches a terminal status.

    Test: `test_poll_status_returns_terminal_body`
    """
    deadline = time.monotonic() + timeout
    while True:
        status = requests.get(
            f"{BASE_URL}/{endpoint_id}/status/{job_id}",
            headers=_headers(api_key),
            timeout=60,
        )
        status.raise_for_status()
        body = status.json()
        if body.get("status") in TERMINAL_STATUSES:
            return body
        if time.monotonic() >= deadline:
            raise SystemExit(f"Job {job_id} did not finish within {timeout}s.")
        time.sleep(poll_seconds)


def submit(
    endpoint_id: str,
    api_key: str,
    payload: dict[str, object],
    *,
    timeout: int,
    use_async: bool,
    poll_seconds: float = 2.0,
) -> dict[str, object]:
    """Run a job, blocking on `/runsync` or polling `/status` after `/run`.

    A throttled pool answers `/runsync` with `IN_QUEUE` or `IN_PROGRESS` and a
    job id rather than a finished job, so the synchronous path falls through to
    the same `/status` polling the async path uses instead of treating a
    non-terminal body as the result.

    Test: `test_submit_polls_when_runsync_returns_in_queue`,
    `test_submit_returns_terminal_runsync_body`
    """
    if not use_async:
        response = requests.post(
            f"{BASE_URL}/{endpoint_id}/runsync",
            headers=_headers(api_key),
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        body = response.json()
        if body.get("status") in TERMINAL_STATUSES:
            return body
        job_id = body.get("id")
        if not job_id:
            return body
        print(
            f"/runsync returned {body.get('status')}; polling job {job_id}",
            file=sys.stderr,
        )
        return poll_status(
            endpoint_id,
            api_key,
            str(job_id),
            timeout=timeout,
            poll_seconds=poll_seconds,
        )

    submitted = requests.post(
        f"{BASE_URL}/{endpoint_id}/run",
        headers=_headers(api_key),
        json=payload,
        timeout=60,
    )
    submitted.raise_for_status()
    job_id = submitted.json()["id"]
    print(f"queued job {job_id}", file=sys.stderr)
    return poll_status(
        endpoint_id, api_key, job_id, timeout=timeout, poll_seconds=poll_seconds
    )


def summarize(body: dict[str, object]) -> None:
    """Print the response with the base64 audio body elided.

    Test: `test_summarize_elides_audio_body`
    """
    redacted = json.loads(json.dumps(body))
    output = redacted.get("output")
    if isinstance(output, dict) and "audio_b64" in output:
        output["audio_b64"] = f"<{len(output['audio_b64'])} base64 chars>"
    print(json.dumps(redacted, indent=2), file=sys.stderr)


def job_output(body: dict[str, object]) -> dict[str, object]:
    """Return the worker output, or exit describing what went wrong."""
    output = body.get("output")
    if body.get("status") != "COMPLETED" or not isinstance(output, dict):
        summarize(body)
        raise SystemExit(f"Job did not complete: status={body.get('status')}")
    if "error" in output:
        summarize(body)
        raise SystemExit(f"Worker error: {output['error']}")
    return output


def run_clone(args: argparse.Namespace, api_key: str) -> int:
    started = time.perf_counter()
    body = submit(
        args.endpoint_id,
        api_key,
        build_clone_payload(args),
        timeout=args.timeout,
        use_async=args.use_async,
    )
    wall_seconds = time.perf_counter() - started
    summarize(body)
    output = job_output(body)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(base64.b64decode(output["audio_b64"]))

    print(f"saved {args.output}")
    print(f"  sample rate      : {output.get('sample_rate')} Hz")
    print(f"  audio duration   : {output.get('audio_seconds')} s")
    print(f"  client wall time : {wall_seconds:.2f} s")
    print(f"  endpoint delay   : {body.get('delayTime')} ms (queue + cold start)")
    print(f"  endpoint execute : {body.get('executionTime')} ms")
    print(f"  worker model load: {output.get('model_load_seconds')} s")
    print(f"  worker generate  : {output.get('generation_seconds')} s")
    if output.get("truncated"):
        print(
            "WARNING: generation stopped at the token ceiling after "
            f"{output.get('decode_steps')} steps, so the audio is cut off "
            "mid-utterance. Send shorter text.",
            file=sys.stderr,
        )
    return 0


def run_register(args: argparse.Namespace, api_key: str) -> int:
    payload = {
        "input": {
            "op": "register_voice",
            "name": args.name,
            "ref_audio_b64": encode_audio(args.ref_audio),
            "ref_text": args.ref_text,
            "overwrite": args.overwrite,
            "source_filename": args.ref_audio.name,
        }
    }
    body = submit(
        args.endpoint_id,
        api_key,
        payload,
        timeout=args.timeout,
        use_async=args.use_async,
    )
    summarize(body)
    voice = job_output(body)["voice"]
    print(f"registered voice {voice['name']}")
    print(f"  duration : {voice.get('duration_seconds')} s")
    print(f"  sha256   : {voice.get('sha256')}")
    return 0


def run_list(args: argparse.Namespace, api_key: str) -> int:
    body = submit(
        args.endpoint_id,
        api_key,
        {"input": {"op": "list_voices"}},
        timeout=args.timeout,
        use_async=args.use_async,
    )
    voices = job_output(body)["voices"]
    if not voices:
        print("no voices registered")
        return 0
    for voice in voices:
        print(
            f"{voice['name']:<20} {voice.get('duration_seconds')}s  "
            f"{voice.get('created_at')}  {voice.get('source_filename')}"
        )
    return 0


def run_delete(args: argparse.Namespace, api_key: str) -> int:
    body = submit(
        args.endpoint_id,
        api_key,
        {"input": {"op": "delete_voice", "name": args.name}},
        timeout=args.timeout,
        use_async=args.use_async,
    )
    job_output(body)
    print(f"deleted voice {args.name}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--endpoint-id", required=True)
        sub.add_argument(
            "--timeout", type=int, default=900, help="Seconds to wait for the job."
        )
        sub.add_argument(
            "--async",
            dest="use_async",
            action="store_true",
            help="Submit to /run and poll /status instead of blocking on /runsync.",
        )

    clone = subparsers.add_parser("clone", help="Generate speech.")
    common(clone)
    clone.add_argument("--text", required=True)
    clone.add_argument("--output", type=Path, required=True)
    clone.add_argument("--voice", help="Name of a registered voice.")
    clone.add_argument(
        "--ref-audio", type=Path, help="Reference WAV, instead of --voice."
    )
    clone.add_argument("--ref-text", help="Exact transcript of --ref-audio.")
    clone.add_argument("--seed", type=int, default=42)
    clone.add_argument("--cfg-scale", type=float, default=1.0)
    clone.set_defaults(func=run_clone)

    register = subparsers.add_parser("register", help="Store a named voice.")
    common(register)
    register.add_argument("--name", required=True)
    register.add_argument("--ref-audio", type=Path, required=True)
    register.add_argument("--ref-text", required=True)
    register.add_argument("--overwrite", action="store_true")
    register.set_defaults(func=run_register)

    listing = subparsers.add_parser("list", help="List registered voices.")
    common(listing)
    listing.set_defaults(func=run_list)

    delete = subparsers.add_parser("delete", help="Remove a registered voice.")
    common(delete)
    delete.add_argument("--name", required=True)
    delete.set_defaults(func=run_delete)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    ref_audio = getattr(args, "ref_audio", None)
    if ref_audio is not None and not ref_audio.is_file():
        raise SystemExit(f"Reference audio not found: {ref_audio}")
    return args.func(args, load_api_key())


if __name__ == "__main__":
    raise SystemExit(main())
