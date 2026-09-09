"""CPU-only tests for the RunPod Serverless worker.

The worker loads a 7.2 GB checkpoint onto a GPU at import time, so every test
here sets ``BREEZE_SKIP_MODEL_LOAD`` first and stubs the two ``breeze_infer``
modules ``handler`` imports lazily. That keeps the module importable — and the
request contract checkable — on a machine with neither torch nor a GPU.
"""

from __future__ import annotations

import base64
import io
import os
import sys
import types
import wave
from pathlib import Path

import numpy as np
import pytest

os.environ["BREEZE_SKIP_MODEL_LOAD"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rp_handler


def make_wav(samples: np.ndarray, sample_rate: int = 16000, channels: int = 1) -> bytes:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


@pytest.fixture
def tone() -> np.ndarray:
    t = np.linspace(0.0, 0.25, 4000, endpoint=False, dtype=np.float32)
    return (0.5 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


@pytest.fixture
def job_input(tone: np.ndarray) -> dict[str, object]:
    return {
        "text": "This is a test of voice cloning.",
        "ref_text": "Reference transcript.",
        "ref_audio_b64": base64.b64encode(make_wav(tone)).decode("ascii"),
    }


def test_module_imports_without_model_loaded() -> None:
    assert rp_handler.STATE is None
    assert callable(rp_handler.handler)


# --- WAV helpers ----------------------------------------------------------


def test_encode_wav_roundtrips_through_decode(tone: np.ndarray) -> None:
    payload = rp_handler.encode_wav(tone, 24000)

    with wave.open(io.BytesIO(payload), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 24000

    decoded, sample_rate = rp_handler.decode_wav(payload)
    assert sample_rate == 24000
    assert decoded.dtype == np.float32
    assert decoded.shape == tone.shape
    # 16-bit quantization is the only loss on this round trip.
    assert np.max(np.abs(decoded - tone)) < 1e-4


def test_encode_wav_clips_out_of_range_samples() -> None:
    payload = rp_handler.encode_wav(np.array([-4.0, 0.0, 4.0], dtype=np.float32), 8000)
    decoded, _ = rp_handler.decode_wav(payload)
    assert decoded[0] == pytest.approx(-1.0, abs=1e-4)
    assert decoded[2] == pytest.approx(1.0, abs=1e-4)


def test_decode_wav_reads_stereo_as_mono() -> None:
    left = np.full(64, 0.5, dtype=np.float32)
    right = np.full(64, -0.1, dtype=np.float32)
    interleaved = np.empty(128, dtype=np.float32)
    interleaved[0::2] = left
    interleaved[1::2] = right

    decoded, sample_rate = rp_handler.decode_wav(
        make_wav(interleaved, sample_rate=22050, channels=2)
    )
    assert sample_rate == 22050
    assert decoded.shape == (64,)
    assert np.allclose(decoded, 0.2, atol=1e-4)


def test_decode_wav_rejects_garbage() -> None:
    with pytest.raises(rp_handler.InputError, match="not a readable WAV|not readable"):
        rp_handler.decode_wav(b"definitely not a wav file")


# --- Input validation -----------------------------------------------------


def test_validate_job_input_accepts_minimal_payload(job_input) -> None:
    validated = rp_handler.validate_job_input(job_input)

    assert validated["text"] == job_input["text"]
    assert validated["ref_text"] == "Reference transcript."
    assert validated["ref_sample_rate"] == 16000
    assert validated["seed"] == rp_handler.DEFAULT_SEED
    assert validated["cfg_scale"] == 1.0
    assert validated["ref_samples"].dtype == np.float32
    assert validated["ref_samples"].size == 4000


def test_validate_job_input_strips_ref_text_whitespace(job_input) -> None:
    job_input["ref_text"] = "  padded transcript  "
    assert rp_handler.validate_job_input(job_input)["ref_text"] == "padded transcript"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"text": None}, "'text' is required"),
        ({"text": "   "}, "'text' is required"),
        ({"text": "x" * (rp_handler.MAX_TEXT_CHARS + 1)}, "exceeds"),
        ({"ref_text": ""}, "'ref_text' is required"),
        ({"ref_text": 7}, "'ref_text' is required"),
        ({"ref_audio_b64": None}, "'ref_audio_b64' is required"),
        ({"ref_audio_b64": "not base64!!"}, "not valid base64"),
        ({"ref_audio_b64": ""}, "'ref_audio_b64' is required"),
        ({"seed": "42"}, "'seed' must be an integer"),
        ({"seed": True}, "'seed' must be an integer"),
        ({"cfg_scale": "1.0"}, "'cfg_scale' must be a number"),
        ({"cfg_scale": 0}, "greater than 0"),
        ({"cfg_scale": -1.0}, "greater than 0"),
        ({"cfg_scale": float("nan")}, "greater than 0"),
        ({"cfg_scale": 2.0}, "must be 1.0 for voice cloning"),
    ],
)
def test_validate_job_input_rejects_bad_fields(job_input, mutation, message) -> None:
    job_input.update(mutation)
    with pytest.raises(rp_handler.InputError, match=message):
        rp_handler.validate_job_input(job_input)


def test_validate_job_input_rejects_non_object() -> None:
    with pytest.raises(rp_handler.InputError, match="must be a JSON object"):
        rp_handler.validate_job_input(["text"])


def test_validate_job_input_rejects_oversized_reference(job_input, monkeypatch) -> None:
    monkeypatch.setattr(rp_handler, "MAX_REF_AUDIO_BYTES", 128)
    with pytest.raises(rp_handler.InputError, match="over the 128 byte limit"):
        rp_handler.validate_job_input(job_input)


def test_validate_job_input_rejects_empty_audio() -> None:
    empty = make_wav(np.zeros(0, dtype=np.float32))
    with pytest.raises(rp_handler.InputError, match="no audio frames"):
        rp_handler.validate_job_input(
            {
                "text": "hello",
                "ref_text": "hello",
                "ref_audio_b64": base64.b64encode(empty).decode("ascii"),
            }
        )


# --- Checkpoint resolution ------------------------------------------------


def test_resolve_checkpoint_uses_existing_directory(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("BREEZE_CHECKPOINT_DIR", raising=False)
    checkpoint = tmp_path / rp_handler.CHECKPOINT_DIR_NAME
    (checkpoint / "audio_tokenizer").mkdir(parents=True)

    assert rp_handler.resolve_checkpoint(volume_root=tmp_path) == checkpoint


def test_resolve_checkpoint_honours_override(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BREEZE_CHECKPOINT_DIR", str(tmp_path / "elsewhere"))
    assert rp_handler.resolve_checkpoint() == tmp_path / "elsewhere"


# --- Handler --------------------------------------------------------------


class _Chunk:
    def __init__(self, audio: np.ndarray) -> None:
        self.audio = audio


class _Runtime:
    sample_rate = 24000

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def iter_audio_chunks(self, inputs, *, request_id, seed):
        self.calls.append({"inputs": inputs, "request_id": request_id, "seed": seed})
        yield _Chunk(np.full(1200, 0.25, dtype=np.float32))
        yield _Chunk(np.full(1200, -0.25, dtype=np.float32))


@pytest.fixture
def stubbed_runtime(monkeypatch) -> _Runtime:
    """Replace the lazily imported breeze_infer modules and the loaded STATE."""
    seen: dict[str, object] = {}

    def prepare_inputs(tokenizer, audio_tokenizer, model, requests, template, **kwargs):
        seen["request"] = requests[0]
        seen["template"] = template
        seen["kwargs"] = kwargs
        return {"input_ids": "stub"}

    runtime_module = types.ModuleType("breeze_infer.runtime")
    runtime_module.set_all_seeds = lambda seed: seen.setdefault("seeds", []).append(
        seed
    )

    templates_module = types.ModuleType("breeze_infer.templates")
    templates_module.prepare_inputs = prepare_inputs
    templates_module.get_template = lambda name: f"template:{name}"
    templates_module.select_template_name = lambda request: (
        "ref_clone_tata" if request.get("ref_audio_path") else "tts_plain"
    )

    monkeypatch.setitem(sys.modules, "breeze_infer.runtime", runtime_module)
    monkeypatch.setitem(sys.modules, "breeze_infer.templates", templates_module)

    runtime = _Runtime()
    runtime.seen = seen
    monkeypatch.setattr(
        rp_handler,
        "STATE",
        {
            "tokenizer": object(),
            "model": object(),
            "audio_tokenizer": object(),
            "runtime": runtime,
            "load_seconds": 12.5,
            "checkpoint": "/runpod-volume/breeze-tts-2",
        },
    )
    return runtime


def test_handler_returns_wav_with_stubbed_runtime(job_input, stubbed_runtime) -> None:
    result = rp_handler.handler({"id": "job-1", "input": job_input})

    assert "error" not in result
    assert result["sample_rate"] == 24000
    assert result["model_load_seconds"] == 12.5
    assert result["audio_seconds"] == pytest.approx(2400 / 24000, abs=1e-3)

    audio, sample_rate = rp_handler.decode_wav(base64.b64decode(result["audio_b64"]))
    assert sample_rate == 24000
    assert audio.size == 2400
    assert audio[0] == pytest.approx(0.25, abs=1e-4)
    assert audio[-1] == pytest.approx(-0.25, abs=1e-4)


def test_handler_routes_to_the_clone_template(job_input, stubbed_runtime) -> None:
    rp_handler.handler({"id": "job-2", "input": job_input})

    seen = stubbed_runtime.seen
    assert seen["template"] == "template:ref_clone_tata"
    assert seen["kwargs"] == {
        "guidance_scale": 1.0,
        "guidance_scale_ref": None,
        "guidance_scale_ins": None,
    }
    assert seen["request"]["ref_text"] == "Reference transcript."
    assert seen["request"]["speaker"] == "S0"
    assert stubbed_runtime.calls[0]["request_id"] == "job-2"
    assert stubbed_runtime.calls[0]["seed"] == 42


def test_handler_writes_a_readable_reference_file(job_input, stubbed_runtime) -> None:
    written: dict[str, bytes] = {}
    original = rp_handler.encode_wav

    def capture(audio, sample_rate):
        payload = original(audio, sample_rate)
        written.setdefault("reference", payload)
        return payload

    rp_handler.encode_wav = capture
    try:
        rp_handler.handler({"id": "job-3", "input": job_input})
    finally:
        rp_handler.encode_wav = original

    # The clone path reads this back from disk via soundfile, so it must parse.
    decoded, sample_rate = rp_handler.decode_wav(written["reference"])
    assert sample_rate == 16000
    assert decoded.size == 4000


def test_handler_reports_validation_error(stubbed_runtime) -> None:
    result = rp_handler.handler({"id": "job-4", "input": {"text": "no reference"}})
    assert result == {
        "error": "'ref_text' is required and must be the transcript of 'ref_audio_b64'."
    }


def test_handler_reports_unloaded_model(job_input, monkeypatch) -> None:
    monkeypatch.setattr(rp_handler, "STATE", None)
    result = rp_handler.handler({"id": "job-5", "input": job_input})
    assert result == {"error": "Model is not loaded in this worker."}
