"""Tests for the volume-backed named voice registry.

A temporary directory stands in for ``/runpod-volume`` throughout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serverless.locking import COMPLETE_MARKER
from serverless.registry import (
    RegistryError,
    VoiceRegistry,
    validate_name,
)


@pytest.fixture
def tone() -> np.ndarray:
    t = np.linspace(0.0, 0.5, 8000, endpoint=False, dtype=np.float32)
    return (0.4 * np.sin(2 * np.pi * 180.0 * t)).astype(np.float32)


@pytest.fixture
def registry(tmp_path) -> VoiceRegistry:
    return VoiceRegistry(tmp_path)


# --- name validation ------------------------------------------------------


@pytest.mark.parametrize("name", ["bob", "a", "voice_1", "my-voice", "a" * 32, "007"])
def test_validate_name_accepts_legal_names(name) -> None:
    assert validate_name(name) == name


@pytest.mark.parametrize(
    "name",
    [
        "",
        "a" * 33,
        "Bob",
        "has space",
        "dots.here",
        "slash/es",
        "../escape",
        None,
        7,
        "unicodé",
    ],
)
def test_validate_name_rejects_illegal_names(name) -> None:
    with pytest.raises(RegistryError, match="Voice name must be"):
        validate_name(name)


# --- round trips ----------------------------------------------------------


def test_register_then_get_round_trips(registry, tone, tmp_path) -> None:
    meta = registry.register(
        "bob",
        audio=tone,
        sample_rate=16000,
        ref_text="  the reference transcript  ",
        source_filename="Recording_1_ref40.wav",
    )

    assert meta["name"] == "bob"
    assert meta["sample_rate"] == 16000
    assert meta["duration_seconds"] == pytest.approx(0.5, abs=1e-3)
    assert meta["source_filename"] == "Recording_1_ref40.wav"
    assert len(str(meta["sha256"])) == 64
    assert meta["created_at"].endswith("Z")

    voice = registry.get("bob")
    assert voice.name == "bob"
    assert voice.ref_text == "the reference transcript"
    assert voice.reference_path.is_file()

    stored = tmp_path / "voices" / "bob"
    assert (stored / COMPLETE_MARKER).is_file()
    assert json.loads((stored / "meta.json").read_text())["sha256"] == meta["sha256"]


def test_register_requires_a_transcript(registry, tone) -> None:
    with pytest.raises(RegistryError, match="non-empty 'ref_text'"):
        registry.register("bob", audio=tone, sample_rate=16000, ref_text="   ")


def test_register_refuses_existing_without_overwrite(registry, tone) -> None:
    registry.register("bob", audio=tone, sample_rate=16000, ref_text="first")
    with pytest.raises(RegistryError, match="already exists"):
        registry.register("bob", audio=tone, sample_rate=16000, ref_text="second")


def test_register_overwrites_when_asked(registry, tone) -> None:
    registry.register("bob", audio=tone, sample_rate=16000, ref_text="first")
    registry.register(
        "bob", audio=tone, sample_rate=16000, ref_text="second", overwrite=True
    )
    assert registry.get("bob").ref_text == "second"


def test_register_rejects_an_illegal_name(registry, tone) -> None:
    with pytest.raises(RegistryError, match="Voice name must be"):
        registry.register("../escape", audio=tone, sample_rate=16000, ref_text="x")


# --- lookup and listing ---------------------------------------------------


def test_get_reports_unknown_voice(registry, tone) -> None:
    registry.register("bob", audio=tone, sample_rate=16000, ref_text="hello")
    with pytest.raises(RegistryError, match="Unknown voice 'alice'.*bob"):
        registry.get("alice")


def test_get_on_empty_registry_says_none_registered(registry) -> None:
    with pytest.raises(RegistryError, match="none registered"):
        registry.get("alice")


def test_list_voices_is_empty_before_any_registration(registry) -> None:
    assert registry.list_voices() == []


def test_list_voices_skips_incomplete_entries(registry, tone, tmp_path) -> None:
    registry.register("bob", audio=tone, sample_rate=16000, ref_text="hello")

    # A registration killed part-way leaves a directory with no marker.
    half = tmp_path / "voices" / "halfwritten"
    half.mkdir(parents=True)
    (half / "meta.json").write_text('{"name": "halfwritten"}')

    listed = [v["name"] for v in registry.list_voices()]
    assert listed == ["bob"]
    with pytest.raises(RegistryError, match="Unknown voice"):
        registry.get("halfwritten")


# --- deletion -------------------------------------------------------------


def test_delete_removes_a_voice(registry, tone) -> None:
    registry.register("bob", audio=tone, sample_rate=16000, ref_text="hello")
    assert registry.delete("bob") == {"name": "bob", "deleted": True}
    assert registry.list_voices() == []


def test_delete_reports_unknown_voice(registry) -> None:
    with pytest.raises(RegistryError, match="nothing to delete"):
        registry.delete("alice")
