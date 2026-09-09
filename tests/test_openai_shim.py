"""Tests for the OpenAI-compatible speech shim, with the RunPod call stubbed."""

from __future__ import annotations

import base64
import importlib.util
import io
import wave
from pathlib import Path

import numpy as np
import pytest
from starlette.testclient import TestClient

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "openai_shim.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("openai_shim", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shim = _load_module()


def make_wav_b64(seconds: float = 0.25, sample_rate: int = 24000) -> str:
    samples = np.zeros(int(seconds * sample_rate), dtype="<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@pytest.fixture
def client(monkeypatch, tmp_path):
    """A shim whose settings come from the environment and whose endpoint is stubbed."""
    monkeypatch.setattr(shim, "DEFAULT_ENV_FILE", tmp_path / "absent.env")
    monkeypatch.setenv("SHIM_API_KEY", "secret-token")
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-key")
    monkeypatch.setenv("RUNPOD_ENDPOINT_ID", "endpoint-1")

    calls: list[tuple[str, str]] = []

    def fake_call(text: str, voice: str, *, timeout: int = 900):
        calls.append((text, voice))
        return {
            "status": "COMPLETED",
            "output": {
                "audio_b64": make_wav_b64(),
                "sample_rate": 24000,
                "truncated": False,
            },
        }

    monkeypatch.setattr(shim, "call_endpoint", fake_call)
    test_client = TestClient(shim.app)
    test_client.calls = calls
    return test_client


AUTH = {"Authorization": "Bearer secret-token"}


# --- settings lookup ------------------------------------------------------


def test_read_env_prefers_environment(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env.local"
    env_file.write_text("SHIM_API_KEY=from-file\n")
    monkeypatch.setenv("SHIM_API_KEY", "from-environment")
    assert shim.read_env("SHIM_API_KEY", env_file) == "from-environment"


def test_read_env_reads_env_file(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("RUNPOD_ENDPOINT_ID", raising=False)
    env_file = tmp_path / ".env.local"
    env_file.write_text("RUNPOD_ENDPOINT_ID='abc123'\n")
    assert shim.read_env("RUNPOD_ENDPOINT_ID", env_file) == "abc123"


def test_read_env_returns_none_when_absent(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("NOT_SET_ANYWHERE", raising=False)
    assert shim.read_env("NOT_SET_ANYWHERE", tmp_path / "missing.env") is None


# --- authorization --------------------------------------------------------


def test_missing_bearer_token_is_rejected(client) -> None:
    response = client.post(
        "/v1/audio/speech", json={"model": "tts-1", "input": "hi", "voice": "bob"}
    )
    assert response.status_code == 401
    assert "Missing bearer token" in response.json()["error"]["message"]


def test_wrong_bearer_token_is_rejected(client) -> None:
    response = client.post(
        "/v1/audio/speech",
        headers={"Authorization": "Bearer wrong"},
        json={"input": "hi", "voice": "bob"},
    )
    assert response.status_code == 401


def test_unconfigured_shim_key_is_a_server_error(client, monkeypatch) -> None:
    monkeypatch.delenv("SHIM_API_KEY", raising=False)
    response = client.post(
        "/v1/audio/speech", headers=AUTH, json={"input": "hi", "voice": "bob"}
    )
    assert response.status_code == 500
    assert "SHIM_API_KEY is not configured" in response.json()["error"]["message"]


# --- request validation ---------------------------------------------------


def test_validate_speech_request_defaults_to_mp3() -> None:
    parsed = shim.validate_speech_request(
        {"model": "tts-1-hd", "input": "hello", "voice": "bob"}
    )
    assert parsed == {"text": "hello", "voice": "bob", "response_format": "mp3"}


def test_validate_speech_request_rejects_speed() -> None:
    with pytest.raises(shim.ShimError, match="'speed' is not supported"):
        shim.validate_speech_request({"input": "x", "voice": "bob", "speed": 1.5})


def test_validate_speech_request_allows_speed_one() -> None:
    parsed = shim.validate_speech_request({"input": "x", "voice": "bob", "speed": 1.0})
    assert parsed["voice"] == "bob"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"voice": "bob"}, "'input' is required"),
        ({"input": "  ", "voice": "bob"}, "'input' is required"),
        ({"input": "x"}, "'voice' is required"),
        ({"input": "x", "voice": "bob", "response_format": "opus"}, "response_format"),
        ({"input": "x", "voice": "bob", "speed": "fast"}, "'speed' must be a number"),
        ({"input": "x" * (shim.MAX_INPUT_CHARS + 1), "voice": "b"}, "exceeds"),
    ],
)
def test_validate_speech_request_rejects_bad_bodies(body, message) -> None:
    with pytest.raises(shim.ShimError, match=message):
        shim.validate_speech_request(body)


def test_non_json_body_is_rejected(client) -> None:
    response = client.post(
        "/v1/audio/speech",
        headers={**AUTH, "Content-Type": "application/json"},
        content=b"not json",
    )
    assert response.status_code == 400
    assert "not valid JSON" in response.json()["error"]["message"]


# --- successful responses -------------------------------------------------


def test_wav_response_carries_audio_and_content_type(client) -> None:
    response = client.post(
        "/v1/audio/speech",
        headers=AUTH,
        json={
            "model": "tts-1",
            "input": "hello",
            "voice": "bob",
            "response_format": "wav",
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["x-truncated"] == "false"
    assert response.content[:4] == b"RIFF"
    assert client.calls == [("hello", "bob")]


def test_worker_error_becomes_a_client_error(client, monkeypatch) -> None:
    monkeypatch.setattr(
        shim,
        "call_endpoint",
        lambda *a, **k: {
            "status": "COMPLETED",
            "output": {"error": "Unknown voice 'x'"},
        },
    )
    response = client.post(
        "/v1/audio/speech",
        headers=AUTH,
        json={"input": "hi", "voice": "x", "response_format": "wav"},
    )
    assert response.status_code == 400
    assert "Unknown voice" in response.json()["error"]["message"]


def test_failed_job_becomes_a_gateway_error(client, monkeypatch) -> None:
    monkeypatch.setattr(shim, "call_endpoint", lambda *a, **k: {"status": "FAILED"})
    response = client.post(
        "/v1/audio/speech",
        headers=AUTH,
        json={"input": "hi", "voice": "bob", "response_format": "wav"},
    )
    assert response.status_code == 502


def test_missing_runpod_settings_is_a_server_error(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(shim, "DEFAULT_ENV_FILE", tmp_path / "absent.env")
    monkeypatch.delenv("RUNPOD_ENDPOINT_ID", raising=False)
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    with pytest.raises(shim.ShimError, match="RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID"):
        shim.call_endpoint("hi", "bob")


def test_health_route(client) -> None:
    assert client.get("/health").json() == {"status": "ok"}


# --- transcoding ----------------------------------------------------------


def test_to_mp3_reports_missing_ffmpeg(monkeypatch) -> None:
    monkeypatch.setattr(shim.shutil, "which", lambda _name: None)
    with pytest.raises(shim.ShimError, match="needs ffmpeg on PATH"):
        shim.to_mp3(b"RIFF")


def test_mp3_response_when_ffmpeg_is_available(client, monkeypatch) -> None:
    monkeypatch.setattr(shim, "to_mp3", lambda wav: b"ID3-fake-mp3")
    response = client.post(
        "/v1/audio/speech", headers=AUTH, json={"input": "hello", "voice": "bob"}
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.content == b"ID3-fake-mp3"
