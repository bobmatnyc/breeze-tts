#!/usr/bin/env python3
"""Join a reading's chunk WAVs, and decide how long each join's silence is.

Why: `scripts/runpod_read.py` had grown past the 500-line cap again, and the
half of it that turns finished chunk WAVs into one file — format checks, PCM
concatenation, the MP3 transcode — is one concern with no dependency on RunPod,
chunking or the manifest. Gap planning moves with it because a gap *is* a join:
the silence between two chunks is the only prosody the client itself controls.

What: `read_pcm` refuses a WAV that is not whole, `concatenate` writes the parts
with their gaps, and `plan_gaps` decides each gap. Fixed gaps were the
pipeline's most audible mechanical cue — every sentence in a reading followed by
exactly the same dead air — so a gap is now a seeded draw from a bounded
distribution around a configurable mean. The draw is keyed by chunk and part
rather than taken from one sequence, so a resumed run rebuilds the same joins.

Test: `tests/test_reading_audio.py`
"""

from __future__ import annotations

import random
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

SAMPLE_RATE = 24_000
SAMPLE_WIDTH = 2
CHANNELS = 1

# Within-sentence pauses rate most natural at about 0.6 s and between-sentence
# ones at 0.6-1.2 s (the pause-tuning study collected in
# articles/hyperdev/research/tts-naturalness-techniques.md, section 5). No
# source gives a paragraph-boundary number, so the paragraph gap keeps the 2x
# ratio the pipeline already used, landing at the top of that band.
DEFAULT_SENTENCE_GAP_MS = 600
DEFAULT_PARAGRAPH_GAP_MS = 1200
# Every naturalness number in that research is a distribution and never a single
# value. Plus or minus a quarter of the mean keeps a sentence gap inside the
# rated 0.45-0.75 s band while breaking the metronome.
DEFAULT_GAP_JITTER = 0.25


# --- WAV assembly ----------------------------------------------------------


def read_pcm(path: Path) -> bytes:
    """Read a chunk WAV, refusing anything but a complete 24 kHz mono 16-bit file.

    Why: `wave.readframes` hands back whatever bytes are present when a file was
    truncated mid-write, with no error. A part WAV cut short by a crashed or
    interrupted run then reads back as valid-but-short, and the finished reading
    silently loses that speech while exiting 0.

    Test: `test_read_pcm_rejects_a_mismatched_format`,
    `test_read_pcm_rejects_a_truncated_payload`
    """
    with wave.open(str(path), "rb") as handle:
        actual = (handle.getnchannels(), handle.getsampwidth(), handle.getframerate())
        if actual != (CHANNELS, SAMPLE_WIDTH, SAMPLE_RATE):
            raise SystemExit(
                f"{path} is {actual}, expected {(CHANNELS, SAMPLE_WIDTH, SAMPLE_RATE)}."
            )
        frames = handle.getnframes()
        data = handle.readframes(frames)
    expected = frames * SAMPLE_WIDTH * CHANNELS
    if len(data) != expected:
        raise SystemExit(
            f"{path} holds {len(data)} bytes of audio; its header claims {expected}."
        )
    return data


def silence(milliseconds: int) -> bytes:
    """Return `milliseconds` of digital silence in the output PCM format."""
    return b"\x00" * (SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS * milliseconds // 1000)


def concatenate(parts: list[tuple[Path, int]], output: Path) -> float:
    """Join chunk WAVs, inserting each part's trailing gap, and return seconds.

    Each entry pairs a WAV with the silence in milliseconds that follows it; the
    final entry's gap is dropped so the reading does not end on silence.

    Test: `test_concatenate_inserts_the_requested_gaps`
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as out:
        out.setnchannels(CHANNELS)
        out.setsampwidth(SAMPLE_WIDTH)
        out.setframerate(SAMPLE_RATE)
        for position, (path, gap_ms) in enumerate(parts):
            out.writeframes(read_pcm(path))
            if position < len(parts) - 1:
                out.writeframes(silence(gap_ms))
        frames = out.getnframes()
    return frames / SAMPLE_RATE


def write_mp3(source: Path, destination: Path) -> bool:
    """Transcode to MP3 with ffmpeg, reporting when ffmpeg is unavailable.

    Test: `test_write_mp3_reports_missing_ffmpeg`
    """
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found on PATH; skipping the MP3.", file=sys.stderr)
        return False
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-codec:a",
            "libmp3lame",
            "-b:a",
            "128k",
            str(destination),
        ],
        check=True,
    )
    return True


# --- Gap planning ----------------------------------------------------------


@dataclass(frozen=True)
class GapPlan:
    """Everything that decides a reading's joins, as recorded in the manifest.

    Test: `test_gap_plan_round_trips_through_its_dict`
    """

    sentence_ms: int = DEFAULT_SENTENCE_GAP_MS
    paragraph_ms: int = DEFAULT_PARAGRAPH_GAP_MS
    jitter: float = DEFAULT_GAP_JITTER
    seed: int = 42

    def as_dict(self) -> dict[str, float]:
        """The manifest form, compared verbatim to decide whether to redraw."""
        return {
            "sentence_ms": int(self.sentence_ms),
            "paragraph_ms": int(self.paragraph_ms),
            "jitter": float(self.jitter),
            "seed": int(self.seed),
        }


def validate_jitter(jitter: float) -> float:
    """Reject a jitter fraction that would put a gap at or below zero."""
    if not 0.0 <= jitter < 1.0:
        raise SystemExit("--gap-jitter must be at least 0 and less than 1.")
    return float(jitter)


def draw_gap(mean_ms: int, jitter: float, *, seed: int, index: int, part: int) -> int:
    """Draw one join's silence from a bounded distribution around `mean_ms`.

    A triangular draw over `mean_ms` plus or minus `jitter` of it keeps most
    gaps near the rated-natural mean while never reaching a length a listener
    would hear as a stall. Each gap gets its own stream, keyed by seed, chunk,
    part and mean, rather than being pulled from one shared sequence: planning
    gap 40 then gives the same answer whether or not gaps 0 to 39 were drawn
    first, which is what lets a resumed or partly re-synthesised reading join
    exactly as the first run did.

    Test: `test_draw_gap_is_deterministic`, `test_draw_gap_stays_in_range`,
    `test_draw_gap_returns_the_mean_without_jitter`
    """
    if jitter <= 0:
        return int(mean_ms)
    low = max(1, round(mean_ms * (1.0 - jitter)))
    high = max(low, round(mean_ms * (1.0 + jitter)))
    stream = random.Random(f"{seed}:{index}:{part}:{mean_ms}")
    return int(round(stream.triangular(low, high, float(mean_ms))))


def plan_gaps(
    records: list[dict],
    work_dir: Path,
    *,
    sentence_gap_ms: int = DEFAULT_SENTENCE_GAP_MS,
    paragraph_gap_ms: int = DEFAULT_PARAGRAPH_GAP_MS,
    jitter: float = 0.0,
    seed: int = 42,
    reuse_recorded: bool = False,
) -> list[tuple[Path, int]]:
    """Pair every part WAV with the silence that follows it, recording each gap.

    Parts inside one chunk are halves of a re-split sentence run, so they take
    the sentence gap; only a chunk that ends a paragraph takes the wider one.
    Every gap is written back onto its part record, so the manifest carries the
    joins a listener actually heard. `reuse_recorded` replays those recorded
    gaps instead of drawing, which is what the caller asks for when the plan
    parameters have not changed since the run that recorded them.

    Test: `test_plan_gaps_widens_at_paragraph_ends`,
    `test_plan_gaps_varies_the_gaps_under_jitter`,
    `test_plan_gaps_replays_recorded_gaps`
    """
    planned: list[tuple[Path, int]] = []
    for record in records:
        parts = record["parts"]
        index = int(record.get("index", 0))
        for position, part in enumerate(parts):
            last = position == len(parts) - 1
            mean = (
                paragraph_gap_ms
                if last and record.get("ends_paragraph")
                else sentence_gap_ms
            )
            recorded = part.get("gap_ms") if reuse_recorded else None
            gap = (
                int(recorded)
                if isinstance(recorded, int)
                else draw_gap(mean, jitter, seed=seed, index=index, part=position)
            )
            part["gap_ms"] = gap
            planned.append((work_dir / part["wav"], gap))
    return planned
