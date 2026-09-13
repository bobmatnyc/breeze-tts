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
from serverless.locking import COMPLETE_MARKER
from serverless.registry import VoiceRegistry


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


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_encode_wav_rejects_non_finite_samples(bad) -> None:
    """NaN and Inf survive np.clip and then hit an undefined int16 cast."""
    audio = np.array([0.1, bad, -0.1], dtype=np.float32)
    with pytest.raises(rp_handler.AudioError, match="non-finite"):
        rp_handler.encode_wav(audio, 24000)


def test_encode_wav_accepts_empty_audio() -> None:
    assert rp_handler.encode_wav(np.zeros(0, dtype=np.float32), 24000)


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
    with pytest.raises(rp_handler.AudioError, match="not a readable WAV|not readable"):
        rp_handler.decode_wav(b"definitely not a wav file")


# --- Input validation -----------------------------------------------------


def test_validate_job_input_accepts_minimal_payload(job_input) -> None:
    validated = rp_handler.validate_job_input(job_input)

    assert validated["op"] == "clone"
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
        ({"ref_audio_b64": "not base64!!"}, "not valid base64"),
        ({"seed": "42"}, "'seed' must be an integer"),
        ({"seed": True}, "'seed' must be an integer"),
        ({"cfg_scale": "1.0"}, "'cfg_scale' must be a number"),
        ({"cfg_scale": 0}, "greater than 0"),
        ({"cfg_scale": -1.0}, "greater than 0"),
        ({"cfg_scale": float("nan")}, "greater than 0"),
        ({"cfg_scale": 2.0}, "must be 1.0 without an 'instruction'"),
        ({"instruction": ""}, "'instruction' must be a non-empty string"),
        ({"instruction": 7}, "'instruction' must be a non-empty string"),
        ({"instruction": "x" * 1001}, "'instruction' exceeds"),
        ({"temperature": "hot"}, "'temperature' must be a number"),
        ({"temperature": 0.0}, "'temperature' must be between"),
        ({"temperature": 5.0}, "'temperature' must be between"),
        ({"top_p": 1.5}, "'top_p' must be between"),
        ({"top_p": True}, "'top_p' must be a number"),
        ({"top_k": 0}, "'top_k' must be between"),
        ({"top_k": "8"}, "'top_k' must be an integer"),
        ({"op": "sing"}, "'op' must be one of"),
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


# --- exactly one voice source --------------------------------------------


def test_clone_accepts_a_registered_voice_name() -> None:
    validated = rp_handler.validate_job_input({"text": "hello", "voice": "bob"})
    assert validated["voice"] == "bob"
    assert "ref_samples" not in validated


def test_clone_requires_exactly_one_voice_source(job_input) -> None:
    both = dict(job_input, voice="bob")
    with pytest.raises(rp_handler.InputError, match="not both"):
        rp_handler.validate_job_input(both)

    with pytest.raises(rp_handler.InputError, match="needs a voice source"):
        rp_handler.validate_job_input({"text": "hello"})


def test_clone_rejects_an_illegal_voice_name() -> None:
    with pytest.raises(rp_handler.InputError, match="Voice name must be"):
        rp_handler.validate_job_input({"text": "hello", "voice": "../escape"})


def test_clone_with_ref_audio_still_requires_ref_text(tone) -> None:
    with pytest.raises(rp_handler.InputError, match="'ref_text' is required"):
        rp_handler.validate_job_input(
            {
                "text": "hello",
                "ref_audio_b64": base64.b64encode(make_wav(tone)).decode("ascii"),
            }
        )


# --- registry operations --------------------------------------------------


def test_validate_register_voice(job_input) -> None:
    payload = dict(job_input, op="register_voice", name="bob", overwrite=True)
    validated = rp_handler.validate_job_input(payload)

    assert validated["op"] == "register_voice"
    assert validated["name"] == "bob"
    assert validated["overwrite"] is True
    assert validated["ref_samples"].size == 4000


def test_validate_register_voice_rejects_non_boolean_overwrite(job_input) -> None:
    payload = dict(job_input, op="register_voice", name="bob", overwrite="yes")
    with pytest.raises(rp_handler.InputError, match="'overwrite' must be a boolean"):
        rp_handler.validate_job_input(payload)


def test_validate_list_and_delete_voice() -> None:
    assert rp_handler.validate_job_input({"op": "list_voices"}) == {"op": "list_voices"}
    assert rp_handler.validate_job_input({"op": "delete_voice", "name": "bob"}) == {
        "op": "delete_voice",
        "name": "bob",
    }


# --- Checkpoint resolution ------------------------------------------------


def test_resolve_checkpoint_uses_existing_directory(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("BREEZE_CHECKPOINT_DIR", raising=False)
    checkpoint = tmp_path / rp_handler.CHECKPOINT_DIR_NAME
    (checkpoint / "audio_tokenizer").mkdir(parents=True)
    (checkpoint / COMPLETE_MARKER).write_text("")

    def fail(_destination):
        raise AssertionError("should not download a complete checkpoint")

    assert rp_handler.resolve_checkpoint(tmp_path, download=fail) == checkpoint


def test_resolve_checkpoint_rejects_partial_download(tmp_path, monkeypatch) -> None:
    """A killed download leaves real files but no marker, so it is redone."""
    monkeypatch.delenv("BREEZE_CHECKPOINT_DIR", raising=False)
    partial = tmp_path / rp_handler.CHECKPOINT_DIR_NAME
    (partial / "audio_tokenizer").mkdir(parents=True)
    (partial / "audio_tokenizer" / "config.json").write_text("{}")

    calls: list[Path] = []

    def download(destination: Path) -> None:
        calls.append(destination)
        (destination / "audio_tokenizer").mkdir(parents=True)
        (destination / "model.safetensors").write_bytes(b"complete")

    resolved = rp_handler.resolve_checkpoint(tmp_path, download=download)

    assert len(calls) == 1
    assert (resolved / "model.safetensors").read_bytes() == b"complete"
    assert not (resolved / "audio_tokenizer" / "config.json").exists()


def test_resolve_checkpoint_retries_after_interrupted_download(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("BREEZE_CHECKPOINT_DIR", raising=False)
    attempts: list[int] = []

    def flaky(destination: Path) -> None:
        attempts.append(1)
        (destination / "partial.bin").write_bytes(b"half")
        if len(attempts) == 1:
            raise RuntimeError("connection reset")
        (destination / "audio_tokenizer").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="connection reset"):
        rp_handler.resolve_checkpoint(tmp_path, download=flaky)
    assert not (tmp_path / rp_handler.CHECKPOINT_DIR_NAME).exists()

    resolved = rp_handler.resolve_checkpoint(tmp_path, download=flaky)
    assert attempts == [1, 1]
    assert (resolved / "audio_tokenizer").is_dir()


def test_resolve_checkpoint_honours_override(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BREEZE_CHECKPOINT_DIR", str(tmp_path / "elsewhere"))
    assert rp_handler.resolve_checkpoint() == tmp_path / "elsewhere"


# --- Handler --------------------------------------------------------------


class _Chunk:
    def __init__(self, audio: np.ndarray) -> None:
        self.audio = audio


class _Runtime:
    sample_rate = 24000

    def __init__(self, steps: int = 4) -> None:
        self.calls: list[dict[str, object]] = []
        self.steps = steps

    def iter_audio_chunks(self, inputs, *, request_id, seed, token_observer=None):
        self.calls.append({"inputs": inputs, "request_id": request_id, "seed": seed})
        for _ in range(self.steps):
            if token_observer is not None:
                token_observer(object())
        yield _Chunk(np.full(1200, 0.25, dtype=np.float32))
        yield _Chunk(np.full(1200, -0.25, dtype=np.float32))


class _Inputs(dict):
    """Stands in for prepare_inputs' return, exposing a prompt length."""

    def __init__(self, prompt_tokens: int = 64) -> None:
        super().__init__(input_ids=types.SimpleNamespace(shape=(1, prompt_tokens)))


@pytest.fixture
def stubbed_runtime(monkeypatch, tmp_path) -> _Runtime:
    """Replace the lazily imported breeze_infer modules and the loaded STATE."""
    seen: dict[str, object] = {}

    def prepare_inputs(tokenizer, audio_tokenizer, model, requests, template, **kwargs):
        seen["request"] = requests[0]
        seen["template"] = template
        seen["kwargs"] = kwargs
        seen["reference_bytes"] = Path(requests[0]["ref_audio_path"]).read_bytes()
        return _Inputs()

    runtime_module = types.ModuleType("breeze_infer.runtime")
    runtime_module.set_all_seeds = lambda seed: seen.setdefault("seeds", []).append(
        seed
    )

    templates_module = types.ModuleType("breeze_infer.templates")
    templates_module.prepare_inputs = prepare_inputs
    templates_module.get_template = lambda name: f"template:{name}"
    templates_module.select_template_name = lambda request: (
        ("ref_edit_tata" if request.get("instruction") else "ref_clone_tata")
        if request.get("ref_audio_path")
        else "tts_plain"
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
            "model": types.SimpleNamespace(
                generation_config=types.SimpleNamespace(
                    temperature=0.9, top_p=1.0, top_k=50
                )
            ),
            "audio_tokenizer": object(),
            "runtime": runtime,
            "load_seconds": 12.5,
            "checkpoint": str(tmp_path / "breeze-tts-2"),
            "registry": VoiceRegistry(tmp_path),
        },
    )
    return runtime


def test_handler_returns_wav_with_stubbed_runtime(job_input, stubbed_runtime) -> None:
    result = rp_handler.handler({"id": "job-1", "input": job_input})

    assert "error" not in result
    assert result["sample_rate"] == 24000
    assert result["model_load_seconds"] == 12.5
    assert result["truncated"] is False
    assert result["decode_steps"] == 4
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
    rp_handler.handler({"id": "job-3", "input": job_input})

    # The clone path reads this back from disk via soundfile, so it must parse.
    decoded, sample_rate = rp_handler.decode_wav(
        stubbed_runtime.seen["reference_bytes"]
    )
    assert sample_rate == 16000
    assert decoded.size == 4000


def test_handler_reports_truncation_at_the_token_ceiling(
    job_input, stubbed_runtime, monkeypatch
) -> None:
    """Reaching max_new_tokens cuts the utterance off; EOS does not."""
    monkeypatch.setattr(stubbed_runtime, "steps", rp_handler.MAX_NEW_TOKENS)

    result = rp_handler.handler({"id": "job-t", "input": job_input})

    assert result["truncated"] is True
    assert result["decode_steps"] == rp_handler.MAX_NEW_TOKENS


def test_handler_reports_truncation_at_the_sequence_ceiling(
    job_input, stubbed_runtime, monkeypatch
) -> None:
    monkeypatch.setattr(stubbed_runtime, "steps", 100)
    templates = sys.modules["breeze_infer.templates"]
    monkeypatch.setattr(
        templates,
        "prepare_inputs",
        lambda *a, **k: _Inputs(prompt_tokens=rp_handler.MAX_SEQ_LEN - 50),
    )

    result = rp_handler.handler({"id": "job-s", "input": job_input})
    assert result["truncated"] is True


def test_handler_reports_validation_error(stubbed_runtime) -> None:
    result = rp_handler.handler({"id": "job-4", "input": {"text": "no reference"}})
    assert "voice source" in result["error"]


def test_handler_reports_unloaded_model(job_input, monkeypatch) -> None:
    monkeypatch.setattr(rp_handler, "STATE", None)
    result = rp_handler.handler({"id": "job-5", "input": job_input})
    assert result == {"error": "Model is not loaded in this worker."}


# --- registry through the handler ----------------------------------------


def test_handler_registers_lists_clones_and_deletes(job_input, stubbed_runtime) -> None:
    registered = rp_handler.handler(
        {
            "id": "reg",
            "input": dict(job_input, op="register_voice", name="bob"),
        }
    )
    assert registered["voice"]["name"] == "bob"

    listed = rp_handler.handler({"id": "ls", "input": {"op": "list_voices"}})
    assert [v["name"] for v in listed["voices"]] == ["bob"]

    cloned = rp_handler.handler(
        {"id": "cl", "input": {"text": "speak please", "voice": "bob"}}
    )
    assert "audio_b64" in cloned
    assert stubbed_runtime.seen["request"]["ref_text"] == "Reference transcript."

    deleted = rp_handler.handler(
        {"id": "del", "input": {"op": "delete_voice", "name": "bob"}}
    )
    assert deleted == {"name": "bob", "deleted": True}


def test_handler_reports_unknown_voice(stubbed_runtime) -> None:
    result = rp_handler.handler({"id": "x", "input": {"text": "hi", "voice": "nobody"}})
    assert "Unknown voice 'nobody'" in result["error"]


def test_handler_reports_duplicate_registration(job_input, stubbed_runtime) -> None:
    payload = dict(job_input, op="register_voice", name="bob")
    rp_handler.handler({"id": "a", "input": payload})
    again = rp_handler.handler({"id": "b", "input": payload})
    assert "already exists" in again["error"]

    forced = rp_handler.handler({"id": "c", "input": dict(payload, overwrite=True)})
    assert forced["voice"]["name"] == "bob"


# --- Voice Direction and per-request sampling -----------------------------


def test_validate_job_input_accepts_guidance_with_an_instruction(job_input) -> None:
    job_input.update({"instruction": "  Speak conversationally.  ", "cfg_scale": 4.0})

    validated = rp_handler.validate_job_input(job_input)

    assert validated["instruction"] == "Speak conversationally."
    assert validated["cfg_scale"] == 4.0


def test_validate_job_input_defaults_the_optional_generation_fields(
    job_input,
) -> None:
    validated = rp_handler.validate_job_input(job_input)

    assert validated["instruction"] is None
    assert validated["temperature"] is None
    assert validated["top_p"] is None
    assert validated["top_k"] is None


def test_validate_job_input_accepts_sampling_in_range(job_input) -> None:
    job_input.update({"temperature": 1.1, "top_p": 0.9, "top_k": 80})

    validated = rp_handler.validate_job_input(job_input)

    assert (validated["temperature"], validated["top_p"], validated["top_k"]) == (
        1.1,
        0.9,
        80,
    )


def test_handler_routes_an_instruction_to_the_direction_template(
    job_input, stubbed_runtime
) -> None:
    job_input.update({"instruction": "Speak conversationally.", "cfg_scale": 4.0})

    rp_handler.handler({"id": "job-i", "input": job_input})

    seen = stubbed_runtime.seen
    assert seen["template"] == "template:ref_edit_tata"
    assert seen["request"]["instruction"] == "Speak conversationally."
    assert seen["kwargs"]["guidance_scale"] == 4.0


def test_handler_sends_no_instruction_field_without_one(
    job_input, stubbed_runtime
) -> None:
    rp_handler.handler({"id": "job-n", "input": job_input})

    assert "instruction" not in stubbed_runtime.seen["request"]


def _watch_sampling(stubbed_runtime, monkeypatch) -> list[tuple]:
    """Record the model's live sampling settings as each generation starts."""
    config = rp_handler.STATE["model"].generation_config
    seen: list[tuple] = []
    original = stubbed_runtime.iter_audio_chunks

    def spy(inputs, **kwargs):
        seen.append((config.temperature, config.top_p, config.top_k))
        return original(inputs, **kwargs)

    monkeypatch.setattr(stubbed_runtime, "iter_audio_chunks", spy)
    return seen


def test_handler_applies_sampling_overrides_for_one_request(
    job_input, stubbed_runtime, monkeypatch
) -> None:
    seen = _watch_sampling(stubbed_runtime, monkeypatch)
    job_input.update({"temperature": 1.2, "top_p": 0.8, "top_k": 80})

    result = rp_handler.handler({"id": "job-s", "input": job_input})

    assert "error" not in result
    assert seen == [(1.2, 0.8, 80)]


def test_handler_restores_sampling_after_a_request(
    job_input, stubbed_runtime, monkeypatch
) -> None:
    """An override on one job must not become this worker's new default."""
    seen = _watch_sampling(stubbed_runtime, monkeypatch)

    rp_handler.handler({"id": "job-1", "input": dict(job_input, temperature=1.5)})
    rp_handler.handler({"id": "job-2", "input": job_input})

    assert seen == [(1.5, 1.0, 50), (0.9, 1.0, 50)]
    config = rp_handler.STATE["model"].generation_config
    assert (config.temperature, config.top_p, config.top_k) == (0.9, 1.0, 50)


def test_handler_leaves_the_model_untouched_without_overrides(
    job_input, stubbed_runtime
) -> None:
    config = rp_handler.STATE["model"].generation_config
    del config.temperature

    result = rp_handler.handler({"id": "job-u", "input": job_input})

    assert "error" not in result
    assert not hasattr(config, "temperature")
