"""Tests for the RunPod endpoint client's offline helpers.

Only key lookup, argument parsing and payload assembly are covered; the HTTP
calls are the endpoint's contract, exercised by a live invocation.
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


def parse(*argv):
    return runpod_clone.build_parser().parse_args(argv)


# --- key lookup -----------------------------------------------------------


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


# --- payload assembly -----------------------------------------------------


def test_encode_audio_rejects_empty_file(tmp_path) -> None:
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    with pytest.raises(SystemExit, match="Reference audio is empty"):
        runpod_clone.encode_audio(empty)


def test_build_clone_payload_uses_voice(tmp_path) -> None:
    args = parse(
        "clone",
        "--endpoint-id",
        "e1",
        "--text",
        "hello",
        "--output",
        str(tmp_path / "o.wav"),
        "--voice",
        "bob",
    )
    job = runpod_clone.build_clone_payload(args)["input"]

    assert job == {
        "op": "clone",
        "text": "hello",
        "seed": 42,
        "cfg_scale": 1.0,
        "voice": "bob",
    }


def test_build_clone_payload_uses_inline_reference(tmp_path) -> None:
    audio = tmp_path / "ref.wav"
    audio.write_bytes(b"RIFF-fake-wav-bytes")
    args = parse(
        "clone",
        "--endpoint-id",
        "e1",
        "--text",
        "hello",
        "--output",
        str(tmp_path / "o.wav"),
        "--ref-audio",
        str(audio),
        "--ref-text",
        "reference text",
        "--seed",
        "7",
    )
    job = runpod_clone.build_clone_payload(args)["input"]

    assert job["seed"] == 7
    assert job["ref_text"] == "reference text"
    assert base64.b64decode(job["ref_audio_b64"]) == b"RIFF-fake-wav-bytes"
    assert "voice" not in job


def test_build_clone_payload_rejects_both_sources(tmp_path) -> None:
    audio = tmp_path / "ref.wav"
    audio.write_bytes(b"RIFF")
    args = parse(
        "clone",
        "--endpoint-id",
        "e1",
        "--text",
        "hi",
        "--output",
        str(tmp_path / "o.wav"),
        "--voice",
        "bob",
        "--ref-audio",
        str(audio),
        "--ref-text",
        "t",
    )
    with pytest.raises(SystemExit, match="not both"):
        runpod_clone.build_clone_payload(args)


def test_build_clone_payload_rejects_neither_source(tmp_path) -> None:
    args = parse(
        "clone",
        "--endpoint-id",
        "e1",
        "--text",
        "hi",
        "--output",
        str(tmp_path / "o.wav"),
    )
    with pytest.raises(SystemExit, match="needs --voice or --ref-audio"):
        runpod_clone.build_clone_payload(args)


def test_build_clone_payload_rejects_ref_audio_without_transcript(tmp_path) -> None:
    audio = tmp_path / "ref.wav"
    audio.write_bytes(b"RIFF")
    args = parse(
        "clone",
        "--endpoint-id",
        "e1",
        "--text",
        "hi",
        "--output",
        str(tmp_path / "o.wav"),
        "--ref-audio",
        str(audio),
    )
    with pytest.raises(SystemExit, match="requires --ref-text"):
        runpod_clone.build_clone_payload(args)


# --- subcommands ----------------------------------------------------------


def test_subcommands_are_registered(tmp_path) -> None:
    assert parse("list", "--endpoint-id", "e1").func is runpod_clone.run_list
    assert parse("delete", "--endpoint-id", "e1", "--name", "bob").name == "bob"

    audio = tmp_path / "r.wav"
    audio.write_bytes(b"RIFF")
    register = parse(
        "register",
        "--endpoint-id",
        "e1",
        "--name",
        "bob",
        "--ref-audio",
        str(audio),
        "--ref-text",
        "t",
        "--overwrite",
    )
    assert register.overwrite is True
    assert register.func is runpod_clone.run_register


def test_summarize_elides_audio_body(capsys) -> None:
    runpod_clone.summarize(
        {
            "status": "COMPLETED",
            "output": {"audio_b64": "A" * 120, "sample_rate": 24000},
        }
    )
    printed = capsys.readouterr().err
    assert "120 base64 chars" in printed
    assert "A" * 120 not in printed
    assert '"sample_rate": 24000' in printed
