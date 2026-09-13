"""Tests for the join half of a reading: gap planning and WAV assembly.

Everything here is pure or filesystem-local — no endpoint, no model.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "reading_audio.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("reading_audio", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["reading_audio"] = module
    spec.loader.exec_module(module)
    return module


reading_audio = _load_module()


def records(count: int, *, paragraph_every: int = 3) -> list[dict]:
    """`count` single-part chunk records, every `paragraph_every`th ending one."""
    return [
        {
            "index": index,
            "ends_paragraph": index % paragraph_every == paragraph_every - 1,
            "parts": [{"wav": f"chunk_{index:04d}_p00.wav"}],
        }
        for index in range(count)
    ]


# --- Gap plan -------------------------------------------------------------


def test_gap_plan_round_trips_through_its_dict() -> None:
    plan = reading_audio.GapPlan(
        sentence_ms=600, paragraph_ms=1200, jitter=0.25, seed=7
    )

    assert plan.as_dict() == {
        "sentence_ms": 600,
        "paragraph_ms": 1200,
        "jitter": 0.25,
        "seed": 7,
    }
    assert reading_audio.GapPlan(**plan.as_dict()).as_dict() == plan.as_dict()


def test_validate_jitter_rejects_a_fraction_that_would_zero_a_gap() -> None:
    assert reading_audio.validate_jitter(0.25) == 0.25
    for bad in (-0.1, 1.0, 2.0):
        try:
            reading_audio.validate_jitter(bad)
        except SystemExit as exit_code:
            assert "--gap-jitter" in str(exit_code)
        else:  # pragma: no cover - the assertion below reports the miss
            raise AssertionError(f"{bad} was accepted")


# --- Drawing one gap ------------------------------------------------------


def test_draw_gap_returns_the_mean_without_jitter() -> None:
    assert reading_audio.draw_gap(600, 0.0, seed=42, index=3, part=0) == 600


def test_draw_gap_is_deterministic() -> None:
    first = reading_audio.draw_gap(600, 0.25, seed=42, index=3, part=0)
    second = reading_audio.draw_gap(600, 0.25, seed=42, index=3, part=0)

    assert first == second


def test_draw_gap_stays_in_range() -> None:
    drawn = [
        reading_audio.draw_gap(600, 0.25, seed=42, index=index, part=0)
        for index in range(200)
    ]

    assert min(drawn) >= 450
    assert max(drawn) <= 750
    assert len(set(drawn)) > 20, "the draw collapsed onto a handful of values"


def test_draw_gap_does_not_depend_on_draw_order() -> None:
    """A resumed run plans gap 40 without having planned gaps 0-39 first."""
    alone = reading_audio.draw_gap(600, 0.25, seed=42, index=40, part=0)
    after = [
        reading_audio.draw_gap(600, 0.25, seed=42, index=index, part=0)
        for index in range(41)
    ][40]

    assert alone == after


# --- Planning a whole reading ---------------------------------------------


def test_plan_gaps_widens_at_paragraph_ends(tmp_path: Path) -> None:
    planned = reading_audio.plan_gaps(
        [
            {"ends_paragraph": False, "parts": [{"wav": "a.wav"}]},
            {"ends_paragraph": True, "parts": [{"wav": "b.wav"}, {"wav": "c.wav"}]},
        ],
        tmp_path,
        sentence_gap_ms=350,
        paragraph_gap_ms=700,
    )

    assert [gap for _, gap in planned] == [350, 350, 700]
    assert [path.name for path, _ in planned] == ["a.wav", "b.wav", "c.wav"]


def test_plan_gaps_varies_the_gaps_under_jitter(tmp_path: Path) -> None:
    planned = reading_audio.plan_gaps(
        records(10),
        tmp_path,
        sentence_gap_ms=600,
        paragraph_gap_ms=1200,
        jitter=0.25,
        seed=42,
    )

    gaps = [gap for _, gap in planned]
    assert len(set(gaps)) > 1, "every join got the same silence"
    assert 600 not in gaps or gaps.count(600) < len(gaps)
    assert all(450 <= gap <= 750 or 900 <= gap <= 1500 for gap in gaps)


def test_plan_gaps_gives_every_join_the_mean_at_zero_jitter(tmp_path: Path) -> None:
    planned = reading_audio.plan_gaps(
        records(10),
        tmp_path,
        sentence_gap_ms=600,
        paragraph_gap_ms=1200,
        jitter=0.0,
        seed=42,
    )

    assert set(gap for _, gap in planned) == {600, 1200}


def test_plan_gaps_records_each_gap_on_its_part(tmp_path: Path) -> None:
    planned_records = records(4)

    planned = reading_audio.plan_gaps(
        planned_records,
        tmp_path,
        sentence_gap_ms=600,
        paragraph_gap_ms=1200,
        jitter=0.25,
        seed=42,
    )

    assert [record["parts"][0]["gap_ms"] for record in planned_records] == [
        gap for _, gap in planned
    ]


def test_plan_gaps_replays_recorded_gaps(tmp_path: Path) -> None:
    """A resumed run joins with the silences the first run recorded."""
    planned_records = records(3)
    for record in planned_records:
        record["parts"][0]["gap_ms"] = 1234

    planned = reading_audio.plan_gaps(
        planned_records,
        tmp_path,
        sentence_gap_ms=600,
        paragraph_gap_ms=1200,
        jitter=0.25,
        seed=42,
        reuse_recorded=True,
    )

    assert [gap for _, gap in planned] == [1234, 1234, 1234]


def test_plan_gaps_redraws_when_not_replaying(tmp_path: Path) -> None:
    planned_records = records(3)
    for record in planned_records:
        record["parts"][0]["gap_ms"] = 1234

    planned = reading_audio.plan_gaps(
        planned_records,
        tmp_path,
        sentence_gap_ms=600,
        paragraph_gap_ms=1200,
        jitter=0.25,
        seed=42,
    )

    assert 1234 not in [gap for _, gap in planned]


# --- Transcode ------------------------------------------------------------


def test_write_mp3_reports_missing_ffmpeg(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)

    assert reading_audio.write_mp3(tmp_path / "a.wav", tmp_path / "a.mp3") is False
    assert "ffmpeg not found" in capsys.readouterr().err
