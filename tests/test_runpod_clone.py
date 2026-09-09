"""Tests for the RunPod endpoint client's offline helpers.

Only the key lookup and payload assembly are covered; the HTTP calls are the
endpoint's contract, exercised by a live invocation rather than a mock.
"""

from __future__ import annotations

import base64
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "runpod_clone.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("runpod_clone", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runpod_clone = _load_module()


def test_load_api_key_prefers_environment(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env.local"
    env_file.write_text("RUNPOD_API_KEY=from-file\n")
    monkeypatch.setenv("RUNPOD_API_KEY", "from-environment")

    assert runpod_clone.load_api_key(env_file) == "from-environment"


def test_load_api_key_reads_env_file(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    env_file = tmp_path / ".env.local"
    env_file.write_text("OTHER=1\nRUNPOD_API_KEY='quoted-value'\n")

    assert runpod_clone.load_api_key(env_file) == "quoted-value"


def test_load_api_key_exits_when_absent(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="is not set"):
        runpod_clone.load_api_key(tmp_path / "missing.env")


def test_build_payload_encodes_audio(tmp_path) -> None:
    audio = tmp_path / "ref.wav"
    audio.write_bytes(b"RIFF-fake-wav-bytes")

    payload = runpod_clone.build_payload(
        "hello world", audio, "reference text", seed=7, cfg_scale=1.0
    )

    job = payload["input"]
    assert job["text"] == "hello world"
    assert job["ref_text"] == "reference text"
    assert job["seed"] == 7
    assert job["cfg_scale"] == 1.0
    assert base64.b64decode(job["ref_audio_b64"]) == b"RIFF-fake-wav-bytes"


def test_build_payload_rejects_empty_audio(tmp_path) -> None:
    audio = tmp_path / "empty.wav"
    audio.write_bytes(b"")
    with pytest.raises(SystemExit, match="Reference audio is empty"):
        runpod_clone.build_payload("hi", audio, "ref", seed=1, cfg_scale=1.0)
