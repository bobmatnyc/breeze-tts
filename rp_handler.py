"""RunPod Serverless worker for Breeze TTS 2 voice cloning.

Why: the FastAPI server in ``breeze_infer/api.py`` assumes a long-lived pod. A
RunPod Serverless worker is already the request boundary, so this module runs
the same in-process sequence without uvicorn: load the model once at import,
then per job prepare inputs, stream chunks, and return a base64 WAV.

What: resolves the checkpoint on the network volume (downloading it from Hugging
Face on first boot), loads it via ``load_runtime``, and registers ``handler``
with ``runpod.serverless.start``. A job carries an ``op`` — ``clone`` (the
default), ``register_voice``, ``list_voices`` or ``delete_voice`` — and a clone
takes either a registered ``voice`` name or an inline ``ref_audio_b64`` plus
``ref_text``.

Test: ``tests/test_rp_handler.py``
"""

from __future__ import annotations

import base64
import binascii
import math
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from serverless.locking import is_complete, publish_directory, volume_lock
from serverless.registry import RegistryError, VoiceRegistry, validate_name
from serverless.wav import AudioError, decode_wav, encode_wav

REPO_ROOT = Path(__file__).resolve().parent

# The checkpoint bundles the audio tokenizer, so one directory is the whole
# model (README.md:62). Kept on a RunPod network volume so cold starts do not
# re-download 7.2 GB.
DEFAULT_VOLUME_ROOT = "/runpod-volume"
DEFAULT_HF_REPO = "BreezeBlue/breeze-tts-2"
CHECKPOINT_DIR_NAME = "breeze-tts-2"
CHECKPOINT_LOCK_NAME = ".breeze-checkpoint.lock"

# Mirrors infer.py:25-27 and breeze_infer/api.py:37-39.
MAX_NEW_TOKENS = 1500
MAX_SEQ_LEN = 2048
REPETITION_PENALTY = 1.1
DEFAULT_SEED = 42
DEFAULT_CFG_SCALE = 1.0

# `/runsync` accepts 20 MB and `/run` 10 MB of JSON, and base64 inflates by 4/3.
MAX_REF_AUDIO_BYTES = 6 * 1024 * 1024
MAX_TEXT_CHARS = 5000

OPERATIONS = ("clone", "register_voice", "list_voices", "delete_voice")


class InputError(ValueError):
    """A job input the worker rejects before touching the GPU."""


# Re-exported so callers and tests have one import site for the audio helpers.
__all__ = [
    "AudioError",
    "InputError",
    "decode_wav",
    "encode_wav",
    "handler",
    "resolve_checkpoint",
]


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------


def _require_text(job_input: dict[str, Any], field: str, *, limit: int) -> str:
    value = job_input.get(field)
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"'{field}' is required and must be a non-empty string.")
    if len(value) > limit:
        raise InputError(f"'{field}' exceeds {limit} characters.")
    return value


def _decode_reference_audio(job_input: dict[str, Any]) -> tuple[np.ndarray, int]:
    encoded = job_input.get("ref_audio_b64")
    if not isinstance(encoded, str) or not encoded.strip():
        raise InputError("'ref_audio_b64' is required and must be a base64 WAV string.")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InputError(f"'ref_audio_b64' is not valid base64: {exc}") from exc
    if not payload:
        raise InputError("'ref_audio_b64' decoded to zero bytes.")
    if len(payload) > MAX_REF_AUDIO_BYTES:
        raise InputError(
            f"'ref_audio_b64' decodes to {len(payload)} bytes, over the "
            f"{MAX_REF_AUDIO_BYTES} byte limit. Send a shorter reference clip."
        )
    try:
        samples, sample_rate = decode_wav(payload)
    except AudioError as exc:
        raise InputError(str(exc)) from exc
    if samples.size == 0:
        raise InputError("'ref_audio_b64' contains no audio frames.")
    return samples, sample_rate


def _validate_scalars(job_input: dict[str, Any]) -> tuple[int, float]:
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
    return seed, cfg_scale


def validate_job_input(job_input: Any) -> dict[str, Any]:
    """Normalize a job payload and dispatch it to the right operation's shape.

    Preconditions: ``op`` is one of ``OPERATIONS`` (default ``clone``). A clone
    supplies exactly one voice source — a registered ``voice`` name, or an
    inline ``ref_audio_b64`` plus ``ref_text`` pair. ``cfg_scale`` must be 1.0;
    the ``ref_clone_tata`` template defines no negative prompt
    (``breeze_infer/templates.py:120-124``), so ``prepare_inputs`` rejects any
    other guidance scale (``breeze_infer/templates.py:342-346``).

    Test: `test_validate_job_input_accepts_minimal_payload`,
    `test_validate_job_input_rejects_bad_fields`,
    `test_clone_requires_exactly_one_voice_source`
    """
    if not isinstance(job_input, dict):
        raise InputError("Job input must be a JSON object.")

    op = job_input.get("op", "clone")
    if op not in OPERATIONS:
        raise InputError(f"'op' must be one of {list(OPERATIONS)}; got {op!r}.")

    if op == "list_voices":
        return {"op": op}

    if op == "delete_voice":
        try:
            return {"op": op, "name": validate_name(job_input.get("name"))}
        except RegistryError as exc:
            raise InputError(str(exc)) from exc

    if op == "register_voice":
        try:
            name = validate_name(job_input.get("name"))
        except RegistryError as exc:
            raise InputError(str(exc)) from exc
        ref_text = _require_text(job_input, "ref_text", limit=MAX_TEXT_CHARS)
        samples, sample_rate = _decode_reference_audio(job_input)
        overwrite = job_input.get("overwrite", False)
        if not isinstance(overwrite, bool):
            raise InputError("'overwrite' must be a boolean.")
        return {
            "op": op,
            "name": name,
            "ref_text": ref_text.strip(),
            "ref_samples": samples,
            "ref_sample_rate": sample_rate,
            "overwrite": overwrite,
            "source_filename": job_input.get("source_filename"),
        }

    text = _require_text(job_input, "text", limit=MAX_TEXT_CHARS)
    seed, cfg_scale = _validate_scalars(job_input)

    has_voice = "voice" in job_input and job_input["voice"] is not None
    has_inline = any(
        job_input.get(field) is not None for field in ("ref_audio_b64", "ref_text")
    )
    if has_voice and has_inline:
        raise InputError(
            "Provide either 'voice' or the 'ref_audio_b64'/'ref_text' pair, not both."
        )
    if not has_voice and not has_inline:
        raise InputError(
            "A clone needs a voice source: either 'voice' naming a registered "
            "voice, or 'ref_audio_b64' with its 'ref_text'."
        )

    request: dict[str, Any] = {
        "op": op,
        "text": text,
        "seed": seed,
        "cfg_scale": cfg_scale,
    }
    if has_voice:
        try:
            request["voice"] = validate_name(job_input["voice"])
        except RegistryError as exc:
            raise InputError(str(exc)) from exc
        return request

    request["ref_text"] = _require_text(
        job_input, "ref_text", limit=MAX_TEXT_CHARS
    ).strip()
    request["ref_samples"], request["ref_sample_rate"] = _decode_reference_audio(
        job_input
    )
    return request


# --------------------------------------------------------------------------
# Checkpoint + model load
# --------------------------------------------------------------------------


def _download_checkpoint(repo_id: str, destination: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(destination),
        max_workers=8,
        token=os.environ.get("HF_TOKEN") or None,
    )
    if not (destination / "audio_tokenizer").is_dir():
        raise RuntimeError(
            f"{repo_id} downloaded to {destination} but has no audio_tokenizer/ "
            "directory, which load_runtime requires."
        )


def resolve_checkpoint(
    volume_root: str | Path = DEFAULT_VOLUME_ROOT,
    repo_id: str = DEFAULT_HF_REPO,
    download: Any = None,
) -> Path:
    """Return a complete checkpoint directory, downloading it once if absent.

    Why: a fresh network volume is empty, and a separate seeding pod is one more
    thing to provision and forget. The worker seeds itself on first boot.

    What: readiness is the completion marker written after a successful
    download, not the presence of ``audio_tokenizer/``. ``snapshot_download``
    populates the tree incrementally, so a worker killed mid-download would
    otherwise leave a directory that looks finished forever. The download runs
    into a staging directory under an ``flock``, so a second worker cold-starting
    on the same volume waits for the first rather than racing it, and an
    interrupted attempt is discarded and retried on the next start.

    Test: `test_resolve_checkpoint_uses_existing_directory`,
    `test_resolve_checkpoint_rejects_partial_download`,
    `test_resolve_checkpoint_retries_after_interrupted_download`
    """
    override = os.environ.get("BREEZE_CHECKPOINT_DIR")
    if override:
        return Path(override)

    fetch = download or (lambda destination: _download_checkpoint(repo_id, destination))
    root = Path(volume_root)
    target = root / CHECKPOINT_DIR_NAME
    if is_complete(target):
        print(f"checkpoint present at {target}", flush=True)
        return target

    with volume_lock(root / CHECKPOINT_LOCK_NAME):
        # Another worker may have finished while this one waited for the lock.
        if is_complete(target):
            print(f"checkpoint completed by another worker at {target}", flush=True)
            return target
        print(f"checkpoint missing at {target}; downloading {repo_id}", flush=True)
        started = time.perf_counter()
        publish_directory(target, fetch)
        print(
            f"checkpoint download: {time.perf_counter() - started:.2f} s -> {target}",
            flush=True,
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
        "registry": VoiceRegistry(
            os.environ.get("BREEZE_VOLUME_ROOT", DEFAULT_VOLUME_ROOT)
        ),
    }


# `BREEZE_SKIP_MODEL_LOAD=1` lets CPU-only tests import this module. Every other
# process loads at import so the first job never pays for the load twice.
STATE: dict[str, Any] | None = (
    None if os.environ.get("BREEZE_SKIP_MODEL_LOAD") == "1" else load_state()
)


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


def _reference_for(request: dict[str, Any], workdir: Path) -> tuple[Path, str]:
    """Materialize the reference clip the clone path reads from disk."""
    if "voice" in request:
        voice = STATE["registry"].get(request["voice"])
        return voice.reference_path, voice.ref_text

    path = workdir / "reference.wav"
    path.write_bytes(encode_wav(request["ref_samples"], request["ref_sample_rate"]))
    return path, request["ref_text"]


def _generate(request: dict[str, Any], request_id: str) -> dict[str, Any]:
    """Run one clone and return the response body.

    Truncation: ``iter_audio_chunks`` ends on EOS, on ``max_new_tokens``, or on
    ``max_seq_len`` (``models/fast_streaming.py:850-854,878``), and every path
    sets ``is_final`` on the last chunk, so ``is_final`` cannot tell them apart.
    The token observer fires once per decode step, so the step count can: a run
    that reached either ceiling was cut off mid-utterance.

    Test: `test_handler_reports_truncation_at_the_token_ceiling`
    """
    from breeze_infer.runtime import set_all_seeds
    from breeze_infer.templates import (
        get_template,
        prepare_inputs,
        select_template_name,
    )

    started = time.perf_counter()
    runtime = STATE["runtime"]

    with tempfile.TemporaryDirectory(prefix="breeze_ref_") as workdir:
        reference_path, ref_text = _reference_for(request, Path(workdir))
        job_request = {
            "id": request_id,
            "text": request["text"],
            "speaker": "S0",
            "ref_audio_path": str(reference_path),
            "ref_text": ref_text,
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

        steps = 0

        def count_step(_frame: Any) -> None:
            nonlocal steps
            steps += 1

        generation_started = time.perf_counter()
        set_all_seeds(request["seed"])
        pieces = [
            np.asarray(chunk.audio, dtype=np.float32).reshape(-1)
            for chunk in runtime.iter_audio_chunks(
                inputs,
                request_id=request_id,
                seed=request["seed"],
                token_observer=count_step,
            )
        ]
        generation_seconds = time.perf_counter() - generation_started

    audio = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
    if audio.size == 0:
        return {"error": "Model produced no audio for this request."}

    prompt_tokens = _prompt_length(inputs)
    truncated = steps >= MAX_NEW_TOKENS or (
        prompt_tokens is not None and prompt_tokens + steps >= MAX_SEQ_LEN - 1
    )

    sample_rate = int(runtime.sample_rate)
    wav = encode_wav(audio, sample_rate)
    audio_seconds = audio.size / sample_rate
    print(
        f"request {request_id}: prepare={prepare_seconds:.2f}s "
        f"generate={generation_seconds:.2f}s audio={audio_seconds:.2f}s "
        f"steps={steps} truncated={truncated}",
        flush=True,
    )
    return {
        "audio_b64": base64.b64encode(wav).decode("ascii"),
        "sample_rate": sample_rate,
        "audio_seconds": round(audio_seconds, 3),
        "truncated": truncated,
        "decode_steps": steps,
        "prepare_seconds": round(prepare_seconds, 3),
        "generation_seconds": round(generation_seconds, 3),
        "model_load_seconds": round(STATE["load_seconds"], 3),
    }


def _prompt_length(inputs: Any) -> int | None:
    """Prompt token count, used for the max_seq_len truncation check."""
    try:
        return int(inputs["input_ids"].shape[1])
    except (AttributeError, KeyError, IndexError, TypeError):
        return None


# --------------------------------------------------------------------------
# Handler
# --------------------------------------------------------------------------


def handler(job: dict[str, Any]) -> dict[str, Any]:
    """Dispatch one job to its operation and return a JSON-serializable reply.

    Test: `test_handler_returns_wav_with_stubbed_runtime`,
    `test_handler_reports_validation_error`, `test_handler_registers_a_voice`
    """
    try:
        request = validate_job_input(job.get("input"))
    except InputError as exc:
        return {"error": str(exc)}

    if STATE is None:
        return {"error": "Model is not loaded in this worker."}

    registry: VoiceRegistry = STATE["registry"]
    request_id = job.get("id") or f"rp-{uuid.uuid4().hex}"

    try:
        if request["op"] == "list_voices":
            return {"voices": registry.list_voices()}
        if request["op"] == "delete_voice":
            return registry.delete(request["name"])
        if request["op"] == "register_voice":
            return {
                "voice": registry.register(
                    request["name"],
                    audio=request["ref_samples"],
                    sample_rate=request["ref_sample_rate"],
                    ref_text=request["ref_text"],
                    overwrite=request["overwrite"],
                    source_filename=request["source_filename"],
                )
            }
        return _generate(request, request_id)
    except (RegistryError, AudioError) as exc:
        return {"error": str(exc)}


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
