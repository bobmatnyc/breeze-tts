"""RunPod Serverless worker for Breeze TTS 2 voice cloning.

Why: the FastAPI server in ``breeze_infer/api.py`` assumes a long-lived pod. A
RunPod Serverless worker is already the request boundary, so this module runs
the same in-process sequence without uvicorn: load the model once at import,
then per job prepare inputs, stream chunks, and return a base64 WAV.

What: resolves the checkpoint on the network volume (downloading it from Hugging
Face on first boot), loads it via ``load_runtime``, and registers ``handler``
with ``runpod.serverless.start``. A job is
``{text, ref_audio_b64, ref_text, seed?, cfg_scale?}`` and the reply is
``{audio_b64, sample_rate, ...timings}``.

Test: ``tests/test_rp_handler.py``
"""

from __future__ import annotations

import base64
import binascii
import io
import math
import os
import tempfile
import time
import uuid
import wave
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent

# The checkpoint bundles the audio tokenizer, so one directory is the whole
# model (README.md:62). Kept on a RunPod network volume so cold starts do not
# re-download 7.2 GB.
DEFAULT_VOLUME_ROOT = "/runpod-volume"
DEFAULT_HF_REPO = "BreezeBlue/breeze-tts-2"
CHECKPOINT_DIR_NAME = "breeze-tts-2"

# Mirrors infer.py:25-27 and breeze_infer/api.py:37-39.
MAX_NEW_TOKENS = 1500
MAX_SEQ_LEN = 2048
REPETITION_PENALTY = 1.1
DEFAULT_SEED = 42
DEFAULT_CFG_SCALE = 1.0

# `/runsync` accepts 20 MB and `/run` 10 MB of JSON, and base64 inflates by 4/3.
MAX_REF_AUDIO_BYTES = 6 * 1024 * 1024
MAX_TEXT_CHARS = 5000


class InputError(ValueError):
    """A job input the worker rejects before touching the GPU."""


# --------------------------------------------------------------------------
# WAV helpers
# --------------------------------------------------------------------------


def encode_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    """Serialize mono float32 PCM in [-1, 1] as a 16-bit WAV container.

    Test: `test_encode_wav_roundtrips_through_decode`
    """
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2", copy=False)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def decode_wav(payload: bytes) -> tuple[np.ndarray, int]:
    """Read a WAV byte string as mono float32 PCM plus its sample rate.

    Why: ``encode_prompt_audio`` (``breeze_infer/audio.py:13-22``) reads a path
    with ``soundfile`` and downmixes to mono, so the reference must arrive as a
    file soundfile can open. Parsing here first turns a malformed upload into a
    clear job error instead of a stack trace deep in the tokenizer.

    What: uses the stdlib ``wave`` reader for 8/16/32-bit PCM, and falls back to
    ``soundfile`` for anything else it cannot parse (float WAV, 24-bit, FLAC).

    Test: `test_decode_wav_reads_stereo_as_mono`, `test_decode_wav_rejects_garbage`
    """
    try:
        with wave.open(io.BytesIO(payload), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            frames = handle.readframes(handle.getnframes())
    except (wave.Error, EOFError) as exc:
        return _decode_wav_via_soundfile(payload, exc)

    dtypes = {1: np.uint8, 2: "<i2", 4: "<i4"}
    if width not in dtypes:
        return _decode_wav_via_soundfile(
            payload, ValueError(f"unsupported sample width {width}")
        )

    raw = np.frombuffer(frames, dtype=dtypes[width])
    if width == 1:
        samples = (raw.astype(np.float32) - 128.0) / 128.0
    else:
        samples = raw.astype(np.float32) / float(2 ** (8 * width - 1))

    if channels > 1:
        usable = (samples.size // channels) * channels
        samples = samples[:usable].reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(samples, dtype=np.float32), int(sample_rate)


def _decode_wav_via_soundfile(
    payload: bytes, cause: Exception
) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf
    except ImportError:
        raise InputError(f"ref_audio_b64 is not a readable WAV: {cause}") from cause

    try:
        data, sample_rate = sf.read(
            io.BytesIO(payload), always_2d=True, dtype="float32"
        )
    except Exception as exc:
        raise InputError(f"ref_audio_b64 is not readable audio: {exc}") from exc
    return np.ascontiguousarray(np.mean(data, axis=1), dtype=np.float32), int(
        sample_rate
    )


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------


def validate_job_input(job_input: Any) -> dict[str, Any]:
    """Normalize a job payload into the fields the clone path needs.

    Preconditions the caller must meet: ``text`` and ``ref_text`` are non-blank
    strings, ``ref_audio_b64`` is base64-encoded WAV under
    ``MAX_REF_AUDIO_BYTES``, ``seed`` is an int, and ``cfg_scale`` is exactly
    1.0 — the ``ref_clone_tata`` template defines no negative prompt
    (``breeze_infer/templates.py:120-124``), so ``prepare_inputs`` rejects any
    other guidance scale (``breeze_infer/templates.py:342-346``).

    Test: `test_validate_job_input_accepts_minimal_payload`,
    `test_validate_job_input_rejects_bad_fields`

    Returns the decoded reference audio alongside the scalar fields.
    """
    if not isinstance(job_input, dict):
        raise InputError("Job input must be a JSON object.")

    text = job_input.get("text")
    if not isinstance(text, str) or not text.strip():
        raise InputError("'text' is required and must be a non-empty string.")
    if len(text) > MAX_TEXT_CHARS:
        raise InputError(f"'text' exceeds {MAX_TEXT_CHARS} characters.")

    ref_text = job_input.get("ref_text")
    if not isinstance(ref_text, str) or not ref_text.strip():
        raise InputError(
            "'ref_text' is required and must be the transcript of 'ref_audio_b64'."
        )

    ref_audio_b64 = job_input.get("ref_audio_b64")
    if not isinstance(ref_audio_b64, str) or not ref_audio_b64.strip():
        raise InputError("'ref_audio_b64' is required and must be a base64 WAV string.")
    try:
        ref_audio = base64.b64decode(ref_audio_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InputError(f"'ref_audio_b64' is not valid base64: {exc}") from exc
    if not ref_audio:
        raise InputError("'ref_audio_b64' decoded to zero bytes.")
    if len(ref_audio) > MAX_REF_AUDIO_BYTES:
        raise InputError(
            f"'ref_audio_b64' decodes to {len(ref_audio)} bytes, over the "
            f"{MAX_REF_AUDIO_BYTES} byte limit. Send a shorter reference clip."
        )

    seed = job_input.get("seed", DEFAULT_SEED)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise InputError("'seed' must be an integer.")

    cfg_scale = job_input.get("cfg_scale", DEFAULT_CFG_SCALE)
    if isinstance(cfg_scale, bool) or not isinstance(cfg_scale, (int, float)):
        raise InputError("'cfg_scale' must be a number.")
    cfg_scale = float(cfg_scale)
    if not math.isfinite(cfg_scale) or cfg_scale <= 0:
        raise InputError("'cfg_scale' must be greater than 0.")
    if cfg_scale != 1.0:
        raise InputError(
            "'cfg_scale' must be 1.0 for voice cloning: the ref_clone_tata "
            "template defines no negative prompt, so guidance is unavailable."
        )

    samples, sample_rate = decode_wav(ref_audio)
    if samples.size == 0:
        raise InputError("'ref_audio_b64' contains no audio frames.")

    return {
        "text": text,
        "ref_text": ref_text.strip(),
        "ref_samples": samples,
        "ref_sample_rate": sample_rate,
        "seed": seed,
        "cfg_scale": cfg_scale,
    }


# --------------------------------------------------------------------------
# Checkpoint + model load
# --------------------------------------------------------------------------


def resolve_checkpoint(
    volume_root: str | Path = DEFAULT_VOLUME_ROOT,
    repo_id: str = DEFAULT_HF_REPO,
) -> Path:
    """Return a checkpoint directory, downloading it to the volume if absent.

    Why: a fresh network volume is empty, and a separate seeding pod is one more
    thing to provision and forget. The worker seeds itself on first boot and
    every later cold start finds the weights already there.

    What: treats ``<volume_root>/breeze-tts-2/audio_tokenizer`` as the presence
    marker, since ``load_runtime`` hard-fails without it
    (``breeze_infer/runtime.py:99-105``).

    Test: `test_resolve_checkpoint_uses_existing_directory`
    """
    override = os.environ.get("BREEZE_CHECKPOINT_DIR")
    if override:
        return Path(override)

    target = Path(volume_root) / CHECKPOINT_DIR_NAME
    if (target / "audio_tokenizer").is_dir():
        print(f"checkpoint present at {target}", flush=True)
        return target

    from huggingface_hub import snapshot_download

    print(f"checkpoint missing at {target}; downloading {repo_id}", flush=True)
    started = time.perf_counter()
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        max_workers=8,
        token=os.environ.get("HF_TOKEN") or None,
    )
    print(
        f"checkpoint download: {time.perf_counter() - started:.2f} s -> {target}",
        flush=True,
    )
    if not (target / "audio_tokenizer").is_dir():
        raise RuntimeError(
            f"{repo_id} downloaded to {target} but has no audio_tokenizer/ directory."
        )
    return target


def load_state() -> dict[str, Any]:
    """Load tokenizer, model and streaming runtime once, mirroring api.py:127-158.

    Eager attention matches the hardcoded choice in ``infer.py:73`` and
    ``breeze_infer/api.py:131``; the CUDA-graph fast path stays off because its
    warmup cost would be paid on every serverless cold start.
    """
    from breeze_infer.runtime import (
        load_runtime,
        resolve_device,
        update_generation_config_for_breeze,
    )
    from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig

    checkpoint = resolve_checkpoint()
    device = resolve_device()
    started = time.perf_counter()
    tokenizer, model, audio_tokenizer = load_runtime(
        checkpoint, device=device, attn_implementation="eager"
    )
    update_generation_config_for_breeze(model)

    runtime = FastBreezeStreamingRuntime(
        model,
        audio_tokenizer,
        FastStreamingConfig(
            max_new_tokens=MAX_NEW_TOKENS,
            max_seq_len=MAX_SEQ_LEN,
            fast_all=False,
            repetition_penalty=REPETITION_PENALTY,
        ),
        tokenizer=tokenizer,
    )
    load_seconds = time.perf_counter() - started
    print(
        f"model load: {load_seconds:.2f} s device={device} "
        f"sample_rate={runtime.sample_rate}",
        flush=True,
    )
    return {
        "tokenizer": tokenizer,
        "model": model,
        "audio_tokenizer": audio_tokenizer,
        "runtime": runtime,
        "load_seconds": load_seconds,
        "checkpoint": str(checkpoint),
    }


# `BREEZE_SKIP_MODEL_LOAD=1` lets CPU-only tests import this module. Every other
# process loads at import so the first job never pays for the load twice.
STATE: dict[str, Any] | None = (
    None if os.environ.get("BREEZE_SKIP_MODEL_LOAD") == "1" else load_state()
)


# --------------------------------------------------------------------------
# Handler
# --------------------------------------------------------------------------


def handler(job: dict[str, Any]) -> dict[str, Any]:
    """Generate one cloned utterance and return it as a base64 WAV.

    Test: `test_handler_returns_wav_with_stubbed_runtime`,
    `test_handler_reports_validation_error`
    """
    started = time.perf_counter()
    try:
        request = validate_job_input(job.get("input"))
    except InputError as exc:
        return {"error": str(exc)}

    if STATE is None:
        return {"error": "Model is not loaded in this worker."}

    from breeze_infer.runtime import set_all_seeds
    from breeze_infer.templates import (
        get_template,
        prepare_inputs,
        select_template_name,
    )

    runtime = STATE["runtime"]
    request_id = job.get("id") or f"rp-{uuid.uuid4().hex}"

    with tempfile.TemporaryDirectory(prefix="breeze_ref_") as workdir:
        reference_path = Path(workdir) / "reference.wav"
        reference_path.write_bytes(
            encode_wav(request["ref_samples"], request["ref_sample_rate"])
        )

        job_request = {
            "id": request_id,
            "text": request["text"],
            "speaker": "S0",
            "ref_audio_path": str(reference_path),
            "ref_text": request["ref_text"],
        }
        set_all_seeds(request["seed"])
        inputs = prepare_inputs(
            STATE["tokenizer"],
            STATE["audio_tokenizer"],
            STATE["model"],
            [job_request],
            get_template(select_template_name(job_request)),
            guidance_scale=request["cfg_scale"],
            guidance_scale_ref=None,
            guidance_scale_ins=None,
        )
        prepare_seconds = time.perf_counter() - started

        generation_started = time.perf_counter()
        set_all_seeds(request["seed"])
        pieces = [
            np.asarray(chunk.audio, dtype=np.float32).reshape(-1)
            for chunk in runtime.iter_audio_chunks(
                inputs, request_id=request_id, seed=request["seed"]
            )
        ]
        generation_seconds = time.perf_counter() - generation_started

    audio = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
    if audio.size == 0:
        return {"error": "Model produced no audio for this request."}

    sample_rate = int(runtime.sample_rate)
    wav = encode_wav(audio, sample_rate)
    audio_seconds = audio.size / sample_rate
    print(
        f"request {request_id}: prepare={prepare_seconds:.2f}s "
        f"generate={generation_seconds:.2f}s audio={audio_seconds:.2f}s "
        f"rtf={generation_seconds / audio_seconds:.3f}",
        flush=True,
    )
    return {
        "audio_b64": base64.b64encode(wav).decode("ascii"),
        "sample_rate": sample_rate,
        "audio_seconds": round(audio_seconds, 3),
        "prepare_seconds": round(prepare_seconds, 3),
        "generation_seconds": round(generation_seconds, 3),
        "model_load_seconds": round(STATE["load_seconds"], 3),
    }


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
