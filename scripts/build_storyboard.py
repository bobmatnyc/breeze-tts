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
concatenated with the reader's own sentence gap. Adjacent chunks are packed into
one scene up to `--scene-seconds`, joined the same way — with the wider paragraph
gap between chunks that ended a paragraph — and the packer counts that silence,
so a scene's rendered length is what the target bounds. `--max-scenes` merges
further if the packed count still exceeds HeyGen's ceiling. Both are reported on
stderr.

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
# Long enough that a 15-minute reading fits in ~30 scenes, short enough that a
# still image never sits on screen past the point a viewer stops looking at it.
DEFAULT_SCENE_SECONDS = 40.0


def load_records(work_dir: Path) -> list[dict]:
    """Return the chunk records this reading covers, in index order.

    A manifest written by a reader run carries `reading`, the indices that run
    actually read. Dropping a section leaves its already-synthesised WAVs in
    `chunks` — they cost money and a later run may want them back — so honouring
    `reading` is what keeps them out of the video. A manifest without the key
    predates it, and every chunk in it counts.

    Test: `test_load_records_orders_by_index`,
    `test_load_records_honours_the_reading_subset`
    """
    manifest = work_dir / "manifest.json"
    if not manifest.is_file():
        raise SystemExit(f"No manifest.json in {work_dir}.")
    body = json.loads(manifest.read_text())
    chunks = sorted(body["chunks"], key=lambda record: record["index"])
    reading = body.get("reading")
    if reading is None:
        return chunks
    wanted = set(reading)
    return [record for record in chunks if record["index"] in wanted]


def group_seconds(
    group: list[dict], *, sentence_gap_ms: int, paragraph_gap_ms: int
) -> float:
    """Seconds a group of chunks will render to, silence included.

    Why: `group_audio` joins a group's part WAVs with the reader's gaps, so a
    group's rendered length is longer than the sum of its chunk durations by one
    gap per interior join. Packing against the durations alone let scenes run
    past `--scene-seconds` — the full-article run's longest came out 41.1 s
    against a 40 s target.

    The gap rule is `plan_gaps`': every part takes the sentence gap unless it is
    the last part of a chunk that ended a paragraph. `concatenate` drops the
    final gap, so a group of n parts carries n-1 of them.

    Test: `test_group_seconds_matches_what_group_audio_renders`
    """
    seconds = 0.0
    gaps_ms: list[int] = []
    for record in group:
        parts = record.get("parts") or [record]
        for position, part in enumerate(parts):
            seconds += float(part.get("duration") or 0.0)
            last = position == len(parts) - 1
            gaps_ms.append(
                paragraph_gap_ms
                if last and record.get("ends_paragraph")
                else sentence_gap_ms
            )
    return seconds + sum(gaps_ms[:-1]) / 1000.0


def pack_records(
    records: list[dict],
    target_seconds: float,
    *,
    sentence_gap_ms: int,
    paragraph_gap_ms: int,
) -> list[list[dict]]:
    """Pack adjacent chunks into scenes of at most `target_seconds` each.

    Why: one scene per chunk gives a 15-minute reading 70 scenes, past HeyGen's
    50-scene ceiling, and splitting by chunk count instead produces scenes from
    5 s to 60 s because chunk length follows the source paragraphs. Packing by
    duration is what puts every scene in the same range.

    The rule: walk the chunks in order and start a new scene whenever adding the
    next chunk would carry the current one — silence included, via
    `group_seconds` — past the target. A chunk is never split, so a single chunk
    longer than the target becomes its own scene and is the only way a scene
    exceeds it.

    Test: `test_pack_records_fills_to_the_target`,
    `test_pack_records_never_splits_a_chunk`,
    `test_pack_records_counts_the_gaps_it_will_render`
    """
    if target_seconds <= 0:
        raise SystemExit("--scene-seconds must be positive.")
    gaps = {"sentence_gap_ms": sentence_gap_ms, "paragraph_gap_ms": paragraph_gap_ms}
    groups: list[list[dict]] = []
    current: list[dict] = []
    for record in records:
        if current and group_seconds([*current, record], **gaps) > target_seconds:
            groups.append(current)
            current = []
        current.append(record)
    if current:
        groups.append(current)
    return groups


def group_records(records: list, max_groups: int) -> list[list]:
    """Split a sequence into at most `max_groups` contiguous, near-equal groups.

    Order is preserved and nothing is dropped, so a merge only ever joins
    neighbours. Under the ceiling every item keeps its own group. Applied to
    chunks it merges chunks; applied to packed scenes it merges scenes, which is
    how `--max-scenes` stays a hard ceiling over `--scene-seconds`.

    Test: `test_group_records_keeps_one_chunk_per_group_when_it_fits`,
    `test_group_records_merges_adjacent_chunks`
    """
    if max_groups < 1:
        raise SystemExit("--max-scenes leaves no room for body scenes.")
    total = len(records)
    if total <= max_groups:
        return [[record] for record in records]
    size, remainder = divmod(total, max_groups)
    groups: list[list] = []
    start = 0
    for position in range(max_groups):
        stop = start + size + (1 if position < remainder else 0)
        groups.append(records[start:stop])
        start = stop
    return groups


def plan_scenes(
    records: list[dict],
    *,
    target_seconds: float,
    max_groups: int,
    sentence_gap_ms: int = runpod_read.DEFAULT_SENTENCE_GAP_MS,
    paragraph_gap_ms: int = runpod_read.DEFAULT_PARAGRAPH_GAP_MS,
) -> list[list[dict]]:
    """Pack chunks to the duration target, then merge further if still too many.

    The `--max-scenes` fallback overrides the target, so a scene it produces can
    run past `--scene-seconds`. That is the ceiling doing its job: HeyGen refuses
    a 51-scene request outright, and a long scene only looks wrong.

    Test: `test_plan_scenes_falls_back_to_the_ceiling`
    """
    packed = pack_records(
        records,
        target_seconds,
        sentence_gap_ms=sentence_gap_ms,
        paragraph_gap_ms=paragraph_gap_ms,
    )
    if len(packed) <= max_groups:
        return packed
    return [
        [record for scene in merged for record in scene]
        for merged in group_records(packed, max_groups)
    ]


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
    engine: str | None,
    title: str,
    aspect_ratio: str,
    resolution: str,
    base: Path,
    captions: bool,
) -> dict:
    """Assemble the storyboard: intro avatar scene, image scenes, closing avatar.

    An `engine` of None leaves the key out, so HeyGen applies its own default —
    Avatar IV — which is what a photo-avatar look wants unless told otherwise.

    Test: `test_build_storyboard_brackets_the_body`,
    `test_build_storyboard_cycles_images_across_body_scenes`,
    `test_build_storyboard_omits_an_unset_engine`
    """

    def avatar_scene(audio: Path, role: str) -> dict:
        scene = {
            "type": "avatar",
            "avatar_id": look_id,
            "audio": relative_to(audio, base),
            "role": role,
        }
        if engine:
            scene["engine"] = engine
        return scene

    picks = cycle_images(images, len(groups))
    scenes: list[dict] = [avatar_scene(intro_audio, "intro")]
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
    scenes.append(avatar_scene(closing_audio, "closing"))
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
    groups = plan_scenes(
        records,
        target_seconds=options.scene_seconds,
        max_groups=budget,
        sentence_gap_ms=options.sentence_gap_ms,
        paragraph_gap_ms=options.paragraph_gap_ms,
    )
    if len(groups) < len(records):
        print(
            f"{len(records)} body chunks, {budget}-scene budget "
            f"({options.max_scenes} scenes less the two avatar scenes); packed "
            f"adjacent chunks to ~{options.scene_seconds:.0f}s into "
            f"{len(groups)} scenes.",
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
    parser.add_argument(
        "--engine",
        help="Force an avatar engine (avatar_iii, avatar_iv, avatar_v). "
        "Omit to let HeyGen apply its default, Avatar IV.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--merged-dir", type=Path)
    parser.add_argument("--title", default="Breeze TTS video")
    parser.add_argument("--aspect-ratio", default="16:9")
    parser.add_argument("--resolution", default="1080p")
    parser.add_argument("--max-scenes", type=int, default=DEFAULT_MAX_SCENES)
    parser.add_argument(
        "--scene-seconds",
        type=float,
        default=DEFAULT_SCENE_SECONDS,
        help="Pack adjacent chunks up to this many rendered seconds per image "
        "scene, counting the silence between them.",
    )
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
