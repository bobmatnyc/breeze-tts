#!/usr/bin/env python3
"""Turn a reading's chunk WAVs into a HeyGen storyboard: avatar, images, avatar.

Why: `scripts/runpod_read.py` leaves a work directory of per-chunk WAVs and a
manifest, and `scripts/heygen_video.py` wants an ordered scene list. The join
between them is mechanical but fiddly — every body chunk needs its own single
audio file, the article images have to cycle in order underneath them, and
HeyGen refuses a request with more than 50 scenes.

What: reads the body work directory's `manifest.json`, gives each chunk one
scene, cycles the supplied images across those scenes (hero first), and brackets
them with an avatar scene for the intro audio and another for the closing.
A chunk the reader had to re-split arrives as several part WAVs, so its parts are
concatenated with the reader's own sentence gap. When the chunk count would push
the video past `--max-scenes`, adjacent chunks are merged the same way — with the
wider paragraph gap between them — and the merge is reported on stderr.

Usage:
    python scripts/build_storyboard.py --intro-audio audio/intro.wav \\
        --body-dir audio/body --closing-audio audio/closing.wav \\
        --image hero.png --image body-1.png --look-id <avatar look id> \\
        --output storyboard.json --title "..."

Test: `test_group_records_merges_adjacent_chunks`,
`test_cycle_images_repeats_in_order`, `test_build_storyboard_brackets_the_body`
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

_READER = Path(__file__).resolve().parent / "runpod_read.py"


def _load_reader():
    """Import the reader as a module so its WAV assembly is reused, not copied."""
    spec = importlib.util.spec_from_file_location("runpod_read", _READER)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("runpod_read", module)
    spec.loader.exec_module(module)
    return module


runpod_read = _load_reader()

DEFAULT_MAX_SCENES = 50
DEFAULT_ENGINE = "avatar_iii"


def load_records(work_dir: Path) -> list[dict]:
    """Return the work directory's chunk records in index order.

    Test: `test_load_records_orders_by_index`
    """
    manifest = work_dir / "manifest.json"
    if not manifest.is_file():
        raise SystemExit(f"No manifest.json in {work_dir}.")
    chunks = json.loads(manifest.read_text())["chunks"]
    return sorted(chunks, key=lambda record: record["index"])


def group_records(records: list[dict], max_groups: int) -> list[list[dict]]:
    """Split chunks into at most `max_groups` contiguous, near-equal groups.

    Order is preserved and no chunk is dropped, so a merge only ever joins
    neighbours. Under the ceiling every chunk keeps its own group.

    Test: `test_group_records_keeps_one_chunk_per_group_when_it_fits`,
    `test_group_records_merges_adjacent_chunks`
    """
    if max_groups < 1:
        raise SystemExit("--max-scenes leaves no room for body scenes.")
    total = len(records)
    if total <= max_groups:
        return [[record] for record in records]
    size, remainder = divmod(total, max_groups)
    groups: list[list[dict]] = []
    start = 0
    for position in range(max_groups):
        stop = start + size + (1 if position < remainder else 0)
        groups.append(records[start:stop])
        start = stop
    return groups


def cycle_images(images: list[Path], count: int) -> list[Path]:
    """Assign `count` scenes an image each, cycling the list in order.

    Test: `test_cycle_images_repeats_in_order`
    """
    if not images:
        raise SystemExit("At least one --image is required.")
    return [images[position % len(images)] for position in range(count)]


def group_parts(
    group: list[dict],
    work_dir: Path,
    *,
    sentence_gap_ms: int,
    paragraph_gap_ms: int,
) -> list[tuple[Path, int]]:
    """Pair every part WAV in a group with the silence that follows it.

    Delegates the gap rule to the reader: parts inside one chunk take the
    sentence gap, and only a chunk that ended a paragraph takes the wider one.

    Test: `test_group_parts_uses_the_readers_gaps`
    """
    return runpod_read.plan_gaps(
        group,
        work_dir,
        sentence_gap_ms=sentence_gap_ms,
        paragraph_gap_ms=paragraph_gap_ms,
    )


def group_audio(
    group: list[dict],
    work_dir: Path,
    merged_dir: Path,
    *,
    sentence_gap_ms: int,
    paragraph_gap_ms: int,
) -> tuple[Path, float]:
    """Return one WAV for a group of chunks, concatenating only when needed.

    A single-part chunk already is its own scene audio, so it is used in place.

    Test: `test_group_audio_reuses_a_single_part_chunk`,
    `test_group_audio_concatenates_a_merged_group`
    """
    parts = group_parts(
        group,
        work_dir,
        sentence_gap_ms=sentence_gap_ms,
        paragraph_gap_ms=paragraph_gap_ms,
    )
    if len(parts) == 1:
        path = parts[0][0]
        return path, float(group[0]["duration"])
    first, last = group[0]["index"], group[-1]["index"]
    output = merged_dir / f"scene_{first:04d}_{last:04d}.wav"
    seconds = runpod_read.concatenate(parts, output)
    return output, round(seconds, 3)


def relative_to(path: Path, base: Path) -> str:
    """Storyboard paths stay relative to the storyboard when they can."""
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def build_storyboard(
    *,
    intro_audio: Path,
    closing_audio: Path,
    groups: list[list[dict]],
    audio: list[tuple[Path, float]],
    images: list[Path],
    look_id: str,
    engine: str,
    title: str,
    aspect_ratio: str,
    resolution: str,
    base: Path,
    captions: bool,
) -> dict:
    """Assemble the storyboard: intro avatar scene, image scenes, closing avatar.

    Test: `test_build_storyboard_brackets_the_body`,
    `test_build_storyboard_cycles_images_across_body_scenes`
    """
    picks = cycle_images(images, len(groups))
    scenes: list[dict] = [
        {
            "type": "avatar",
            "avatar_id": look_id,
            "engine": engine,
            "audio": relative_to(intro_audio, base),
            "role": "intro",
        }
    ]
    for group, (path, seconds), image in zip(groups, audio, picks, strict=True):
        scene = {
            "type": "image",
            "image": relative_to(image, base),
            "audio": relative_to(path, base),
            "role": "body",
            "chunks": [record["index"] for record in group],
            "duration": seconds,
        }
        if captions:
            scene["caption"] = " ".join(record["text"] for record in group)
        scenes.append(scene)
    scenes.append(
        {
            "type": "avatar",
            "avatar_id": look_id,
            "engine": engine,
            "audio": relative_to(closing_audio, base),
            "role": "closing",
        }
    )
    return {
        "title": title,
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
        "scenes": scenes,
    }


def run(options: argparse.Namespace) -> int:
    """Read the manifest, write any merged WAVs, and save the storyboard."""
    records = load_records(options.body_dir)
    budget = options.max_scenes - 2
    groups = group_records(records, budget)
    if len(groups) < len(records):
        print(
            f"{len(records)} body chunks exceed the {budget}-scene budget "
            f"({options.max_scenes} scenes less the two avatar scenes); merged "
            f"adjacent chunks into {len(groups)} scenes.",
            file=sys.stderr,
        )
    merged_dir = options.merged_dir or (options.output.parent / "merged")
    audio = [
        group_audio(
            group,
            options.body_dir,
            merged_dir,
            sentence_gap_ms=options.sentence_gap_ms,
            paragraph_gap_ms=options.paragraph_gap_ms,
        )
        for group in groups
    ]
    storyboard = build_storyboard(
        intro_audio=options.intro_audio,
        closing_audio=options.closing_audio,
        groups=groups,
        audio=audio,
        images=options.image,
        look_id=options.look_id,
        engine=options.engine,
        title=options.title,
        aspect_ratio=options.aspect_ratio,
        resolution=options.resolution,
        base=options.output.resolve().parent,
        captions=options.captions,
    )
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(storyboard, indent=2) + "\n")
    body_seconds = sum(seconds for _, seconds in audio)
    print(
        f"{options.output}: {len(storyboard['scenes'])} scenes, "
        f"{len(groups)} body scenes totalling {body_seconds:.2f}s"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--intro-audio", type=Path, required=True)
    parser.add_argument("--body-dir", type=Path, required=True)
    parser.add_argument("--closing-audio", type=Path, required=True)
    parser.add_argument(
        "--image",
        type=Path,
        action="append",
        required=True,
        help="Repeat in display order; the first is the hero.",
    )
    parser.add_argument("--look-id", required=True, help="HeyGen avatar look id.")
    parser.add_argument("--engine", default=DEFAULT_ENGINE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--merged-dir", type=Path)
    parser.add_argument("--title", default="Breeze TTS video")
    parser.add_argument("--aspect-ratio", default="16:9")
    parser.add_argument("--resolution", default="1080p")
    parser.add_argument("--max-scenes", type=int, default=DEFAULT_MAX_SCENES)
    parser.add_argument(
        "--sentence-gap-ms", type=int, default=runpod_read.DEFAULT_SENTENCE_GAP_MS
    )
    parser.add_argument(
        "--paragraph-gap-ms", type=int, default=runpod_read.DEFAULT_PARAGRAPH_GAP_MS
    )
    parser.add_argument(
        "--captions",
        action="store_true",
        help="Carry each body scene's spoken text as its caption.",
    )
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
