"""Tests for the storyboard builder.

Everything here is filesystem-local: a fake work directory of silent WAVs plus a
manifest stands in for a real reading, so scene ordering, image cycling and
chunk merging are checked without calling either the TTS endpoint or HeyGen.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import wave
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_storyboard.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_storyboard", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_storyboard"] = module
    spec.loader.exec_module(module)
    return module


build_storyboard = _load_module()
runpod_read = build_storyboard.runpod_read


def write_wav(path: Path, seconds: float) -> Path:
    """Write a silent 24 kHz mono 16-bit WAV of the requested length."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(runpod_read.SAMPLE_RATE)
        handle.writeframes(b"\x00" * int(runpod_read.SAMPLE_RATE * 2 * seconds))
    return path


def make_work_dir(
    tmp_path: Path, count: int, *, parts_for: dict[int, int] | None = None
):
    """Build a work directory of `count` chunks, some with several part WAVs."""
    parts_for = parts_for or {}
    work_dir = tmp_path / "body"
    records = []
    for index in range(count):
        part_count = parts_for.get(index, 1)
        parts = []
        for position in range(part_count):
            name = f"chunk_{index:04d}_p{position:02d}.wav"
            write_wav(work_dir / name, 1.0)
            parts.append(
                {"wav": name, "text": f"text {index}.{position}", "duration": 1.0}
            )
        records.append(
            {
                "index": index,
                "text": f"text {index}",
                "duration": float(part_count),
                "ends_paragraph": True,
                "parts": parts,
            }
        )
    (work_dir / "manifest.json").write_text(json.dumps({"chunks": records}, indent=2))
    return work_dir, records


# --- Manifest and grouping -------------------------------------------------


def test_load_records_orders_by_index(tmp_path):
    work_dir, records = make_work_dir(tmp_path, 3)
    shuffled = {"chunks": [records[2], records[0], records[1]]}
    (work_dir / "manifest.json").write_text(json.dumps(shuffled))
    assert [r["index"] for r in build_storyboard.load_records(work_dir)] == [0, 1, 2]


def test_load_records_without_a_manifest_exits(tmp_path):
    with pytest.raises(SystemExit):
        build_storyboard.load_records(tmp_path)


def test_group_records_keeps_one_chunk_per_group_when_it_fits():
    records = [{"index": index} for index in range(5)]
    groups = build_storyboard.group_records(records, 10)
    assert [[r["index"] for r in group] for group in groups] == [
        [0],
        [1],
        [2],
        [3],
        [4],
    ]


def test_group_records_merges_adjacent_chunks():
    records = [{"index": index} for index in range(9)]
    groups = build_storyboard.group_records(records, 4)
    assert [[r["index"] for r in group] for group in groups] == [
        [0, 1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
    ]
    flattened = [record["index"] for group in groups for record in group]
    assert flattened == list(range(9))


def test_group_records_rejects_a_zero_budget():
    with pytest.raises(SystemExit):
        build_storyboard.group_records([{"index": 0}], 0)


def test_cycle_images_repeats_in_order():
    images = [Path("hero.png"), Path("b1.png"), Path("b2.png")]
    picks = build_storyboard.cycle_images(images, 7)
    assert [path.name for path in picks] == [
        "hero.png",
        "b1.png",
        "b2.png",
        "hero.png",
        "b1.png",
        "b2.png",
        "hero.png",
    ]


def test_cycle_images_without_images_exits():
    with pytest.raises(SystemExit):
        build_storyboard.cycle_images([], 3)


# --- Scene audio -----------------------------------------------------------


def test_group_audio_reuses_a_single_part_chunk(tmp_path):
    work_dir, records = make_work_dir(tmp_path, 2)
    path, seconds = build_storyboard.group_audio(
        [records[1]],
        work_dir,
        tmp_path / "merged",
        sentence_gap_ms=350,
        paragraph_gap_ms=700,
    )
    assert path == work_dir / "chunk_0001_p00.wav"
    assert seconds == 1.0
    assert not (tmp_path / "merged").exists()


def test_group_audio_concatenates_a_merged_group(tmp_path):
    work_dir, records = make_work_dir(tmp_path, 3)
    path, seconds = build_storyboard.group_audio(
        records[:2],
        work_dir,
        tmp_path / "merged",
        sentence_gap_ms=350,
        paragraph_gap_ms=700,
    )
    assert path == tmp_path / "merged" / "scene_0000_0001.wav"
    # Two one-second chunks plus the paragraph gap between them.
    assert seconds == pytest.approx(2.7, abs=0.01)


def test_group_parts_uses_the_readers_gaps(tmp_path):
    work_dir, records = make_work_dir(tmp_path, 2, parts_for={0: 2})
    planned = build_storyboard.group_parts(
        records[:2], work_dir, sentence_gap_ms=350, paragraph_gap_ms=700
    )
    assert [gap for _, gap in planned] == [350, 700, 700]


def test_group_audio_joins_a_resplit_chunks_parts(tmp_path):
    work_dir, records = make_work_dir(tmp_path, 1, parts_for={0: 2})
    path, seconds = build_storyboard.group_audio(
        records,
        work_dir,
        tmp_path / "merged",
        sentence_gap_ms=350,
        paragraph_gap_ms=700,
    )
    assert path == tmp_path / "merged" / "scene_0000_0000.wav"
    assert seconds == pytest.approx(2.35, abs=0.01)


# --- Storyboard assembly ---------------------------------------------------


def build(tmp_path, groups, audio, images, captions=False, engine="avatar_iii"):
    return build_storyboard.build_storyboard(
        intro_audio=tmp_path / "audio" / "intro.wav",
        closing_audio=tmp_path / "audio" / "closing.wav",
        groups=groups,
        audio=audio,
        images=images,
        look_id="look_1",
        engine=engine,
        title="Test",
        aspect_ratio="16:9",
        resolution="1080p",
        base=tmp_path,
        captions=captions,
    )


def test_build_storyboard_brackets_the_body(tmp_path):
    work_dir, records = make_work_dir(tmp_path, 2)
    groups = [[records[0]], [records[1]]]
    audio = [
        (work_dir / "chunk_0000_p00.wav", 1.0),
        (work_dir / "chunk_0001_p00.wav", 1.0),
    ]
    storyboard = build(tmp_path, groups, audio, [tmp_path / "hero.png"])
    kinds = [scene["type"] for scene in storyboard["scenes"]]
    assert kinds == ["avatar", "image", "image", "avatar"]
    roles = [scene["role"] for scene in storyboard["scenes"]]
    assert roles == ["intro", "body", "body", "closing"]
    assert storyboard["scenes"][0]["audio"] == "audio/intro.wav"
    assert storyboard["scenes"][0]["avatar_id"] == "look_1"
    assert storyboard["scenes"][0]["engine"] == "avatar_iii"
    assert storyboard["scenes"][-1]["audio"] == "audio/closing.wav"
    assert storyboard["aspect_ratio"] == "16:9"
    assert storyboard["resolution"] == "1080p"
    assert storyboard["title"] == "Test"


def test_build_storyboard_omits_an_unset_engine(tmp_path):
    work_dir, records = make_work_dir(tmp_path, 1)
    audio = [(work_dir / "chunk_0000_p00.wav", 1.0)]
    storyboard = build(
        tmp_path, [[records[0]]], audio, [tmp_path / "hero.png"], engine=None
    )
    avatars = [scene for scene in storyboard["scenes"] if scene["type"] == "avatar"]
    assert len(avatars) == 2
    assert all("engine" not in scene for scene in avatars)


def test_build_storyboard_cycles_images_across_body_scenes(tmp_path):
    work_dir, records = make_work_dir(tmp_path, 4)
    groups = [[record] for record in records]
    audio = [(work_dir / f"chunk_{i:04d}_p00.wav", 1.0) for i in range(4)]
    images = [tmp_path / "hero.png", tmp_path / "b1.png", tmp_path / "b2.png"]
    storyboard = build(tmp_path, groups, audio, images)
    body = [scene for scene in storyboard["scenes"] if scene["role"] == "body"]
    assert [scene["image"] for scene in body] == [
        "hero.png",
        "b1.png",
        "b2.png",
        "hero.png",
    ]


def test_build_storyboard_records_merged_chunk_indices(tmp_path):
    work_dir, records = make_work_dir(tmp_path, 4)
    groups = [records[:2], records[2:]]
    audio = [
        (work_dir / "chunk_0000_p00.wav", 2.7),
        (work_dir / "chunk_0002_p00.wav", 2.7),
    ]
    storyboard = build(tmp_path, groups, audio, [tmp_path / "hero.png"], captions=True)
    body = [scene for scene in storyboard["scenes"] if scene["role"] == "body"]
    assert [scene["chunks"] for scene in body] == [[0, 1], [2, 3]]
    assert body[0]["caption"] == "text 0 text 1"
    assert body[0]["duration"] == 2.7


def test_run_merges_when_the_scene_budget_is_tight(tmp_path, capsys):
    work_dir, _ = make_work_dir(tmp_path, 9)
    write_wav(tmp_path / "audio" / "intro.wav", 1.0)
    write_wav(tmp_path / "audio" / "closing.wav", 1.0)
    output = tmp_path / "storyboard.json"
    options = build_storyboard.build_parser().parse_args(
        [
            "--intro-audio",
            str(tmp_path / "audio" / "intro.wav"),
            "--body-dir",
            str(work_dir),
            "--closing-audio",
            str(tmp_path / "audio" / "closing.wav"),
            "--image",
            str(tmp_path / "hero.png"),
            "--image",
            str(tmp_path / "b1.png"),
            "--look-id",
            "look_1",
            "--output",
            str(output),
            "--max-scenes",
            "6",
        ]
    )
    assert build_storyboard.run(options) == 0
    storyboard = json.loads(output.read_text())
    assert len(storyboard["scenes"]) == 6
    body = [scene for scene in storyboard["scenes"] if scene["role"] == "body"]
    assert [scene["chunks"] for scene in body] == [[0, 1, 2], [3, 4], [5, 6], [7, 8]]
    assert "merged adjacent chunks into 4 scenes" in capsys.readouterr().err


def test_run_keeps_one_scene_per_chunk_when_they_fit(tmp_path):
    work_dir, _ = make_work_dir(tmp_path, 9)
    write_wav(tmp_path / "audio" / "intro.wav", 1.0)
    write_wav(tmp_path / "audio" / "closing.wav", 1.0)
    output = tmp_path / "storyboard.json"
    options = build_storyboard.build_parser().parse_args(
        [
            "--intro-audio",
            str(tmp_path / "audio" / "intro.wav"),
            "--body-dir",
            str(work_dir),
            "--closing-audio",
            str(tmp_path / "audio" / "closing.wav"),
            "--image",
            str(tmp_path / "hero.png"),
            "--look-id",
            "look_1",
            "--output",
            str(output),
        ]
    )
    assert build_storyboard.run(options) == 0
    storyboard = json.loads(output.read_text())
    assert len(storyboard["scenes"]) == 11
    assert not (tmp_path / "merged").exists()
