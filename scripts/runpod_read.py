#!/usr/bin/env python3
"""Read a long text or Markdown file aloud through the RunPod clone endpoint.

Why: `scripts/runpod_clone.py` sends one request and gets one clip back, but the
worker stops at `MAX_NEW_TOKENS` (1500 decode steps, about 12.6 per second of
audio) and returns `truncated: true`. Anything longer than roughly 115 s of
speech has to be split, synthesised piecewise and stitched back together.

What: `scripts/speech_text.py` reduces the document to the words to be spoken;
this script splits those at sentence boundaries under a word budget, clones each
chunk under its own derived seed, and hands the finished parts to
`scripts/reading_audio.py`, which joins them with a jittered silence gap that
widens at paragraph breaks. Per-chunk WAVs and a manifest live in a work
directory, so a re-run resynthesises only what is missing — and the manifest
records each chunk's seed, its settings and its gap, so a resumed run rebuilds
the same reading.

Usage:
    python scripts/runpod_read.py --endpoint-id <id> --voice bob \\
        --input article.md --output outputs/article.wav \\
        --work-dir outputs/reads/article --mp3

Test: `test_chunk_text_respects_word_budget`,
`test_synthesise_resumes_from_manifest`,
`test_synthesise_redoes_only_the_chunks_a_lexicon_change_touched`
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import reading_audio  # noqa: E402  (path is set up immediately above)
import runpod_clone  # noqa: E402  (same)
import speech_text  # noqa: E402  (same)
from chunking import (  # noqa: E402  (same)
    DEFAULT_MAX_WORDS,
    DEFAULT_WORD_BUDGET,
    Chunk,
    chunk_text,
    split_in_half,
    text_sha,
)
from reading_audio import (  # noqa: E402  (same)
    CHANNELS,
    DEFAULT_GAP_JITTER,
    DEFAULT_PARAGRAPH_GAP_MS,
    DEFAULT_SENTENCE_GAP_MS,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
    GapPlan,
    concatenate,
    plan_gaps,
    read_pcm,
    silence,
    validate_jitter,
    write_mp3,
)
from speech_text import (  # noqa: E402  (same)
    AUTHOR_FOOTER_PREFIX,
    DEFAULT_PRONUNCIATIONS,
    apply_pronunciations,
    drop_section,
    inject_disfluencies,
    lexicon_sha,
    load_pronunciations,
    markdown_to_speech,
    select_section,
    split_sentences,
    strip_frontmatter,
)

__all__ = [
    "AUTHOR_FOOTER_PREFIX",
    "CHANNELS",
    "Chunk",
    "chunk_text",
    "split_in_half",
    "text_sha",
    "SAMPLE_RATE",
    "SAMPLE_WIDTH",
    "GapPlan",
    "apply_pronunciations",
    "concatenate",
    "drop_section",
    "inject_disfluencies",
    "lexicon_sha",
    "load_pronunciations",
    "markdown_to_speech",
    "plan_gaps",
    "read_pcm",
    "reading_audio",
    "select_section",
    "silence",
    "speech_text",
    "split_sentences",
    "strip_frontmatter",
    "write_mp3",
]

# RTX 4090 at $1.10/hr, the most expensive tier in the endpoint's GPU pool.
DEFAULT_RATE_PER_SECOND = 0.000306
MAX_RESPLIT_DEPTH = 3
# The worker's reported seconds and the WAV it sent agree exactly on every
# reading on record, so this only absorbs float rounding in the manifest.
PART_DURATION_TOLERANCE_S = 0.05

SEED_MODES = ("fixed", "vary")
DEFAULT_SEED = 42
DEFAULT_SEED_MODE = "vary"
# Oviatt's measured rate for spontaneous monologue, and the variant Bob picked
# out of the A/B listening pass (see docker/README.serverless.md, Naturalness).
DEFAULT_DISFLUENCY_RATE = 3.6


# --- Request parameters ----------------------------------------------------


def part_seed(base_seed: int, index: int, part: int, mode: str) -> int:
    """The seed one request uses, varied per chunk unless the mode says fixed.

    Why: one seed for a whole reading means every chunk samples the same
    trajectory through the same sampler, and a long article comes out sounding
    like one sentence read seventy times. `--seed` stays the reading's
    reproducibility handle; it now seeds a per-chunk stream rather than every
    request directly.

    What: a `vary` seed is the first 31 bits of sha256 over the base seed, the
    chunk index and the part index. It is a pure function of those three, so a
    re-run derives the same seed for the same chunk, an edit to chunk 3 leaves
    chunk 4's seed alone, and the manifest resume path stays exact.

    Test: `test_part_seed_varies_by_chunk`, `test_part_seed_is_reproducible`,
    `test_part_seed_is_the_base_seed_when_fixed`
    """
    if mode not in SEED_MODES:
        raise SystemExit(f"--seed-mode must be one of {list(SEED_MODES)}.")
    if mode == "fixed":
        return int(base_seed)
    digest = hashlib.sha256(f"{base_seed}:{index}:{part}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def params_sha(params: dict[str, object]) -> str:
    """Short hash of a request's generation settings, stable under key order.

    Why: a chunk's cached WAV is only reusable if the settings that produced it
    still apply. Text has always been checked; seed, instruction and sampling
    overrides now are too, so switching `--seed-mode` or adding an
    `--instruction` re-synthesises exactly as a text edit does.

    Test: `test_params_sha_ignores_key_order`,
    `test_synthesise_redoes_a_chunk_when_the_seed_mode_changes`
    """
    canonical = json.dumps(params, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# --- Synthesis -------------------------------------------------------------


def load_manifest(path: Path) -> dict[str, dict]:
    """Read the work directory's manifest, treating a missing file as empty."""
    if not path.is_file():
        return {}
    records = json.loads(path.read_text())["chunks"]
    return {str(record["index"]): record for record in records}


def manifest_lexicon(path: Path) -> str | None:
    """The lexicon sha a previous run recorded, or None when it recorded none.

    Test: `test_synthesise_records_the_lexicon_sha`
    """
    if not path.is_file():
        return None
    return json.loads(path.read_text()).get("lexicon")


def manifest_gaps(path: Path) -> dict | None:
    """The gap plan a previous run recorded, or None when it recorded none.

    Test: `test_synthesise_records_nothing_about_gaps_until_they_are_planned`,
    `test_run_replays_recorded_gaps_on_a_resumed_run`
    """
    if not path.is_file():
        return None
    return json.loads(path.read_text()).get("gaps")


def save_manifest(
    path: Path,
    records: dict[str, dict],
    *,
    lexicon: str | None = None,
    reading: list[int] | None = None,
    gaps: dict | None = None,
) -> None:
    """Write the manifest back in chunk order.

    `chunks` is the work directory's whole cache, which outlives any one run.
    `reading` names the subset this run actually read, so dropping a section
    leaves its already-paid-for WAVs on disk without them re-entering the video;
    `lexicon` records which respellings produced those texts, and `gaps` the
    plan that drew the silences between them.

    Test: `test_save_manifest_records_the_reading_and_lexicon`,
    `test_save_manifest_records_the_gap_plan`
    """
    ordered = [records[key] for key in sorted(records, key=int)]
    body: dict[str, object] = {}
    if lexicon is not None:
        body["lexicon"] = lexicon
    if reading is not None:
        body["reading"] = reading
    if gaps is not None:
        body["gaps"] = gaps
    body["chunks"] = ordered
    path.write_text(json.dumps(body, indent=2) + "\n")


def clone_part(
    text: str,
    part_path: Path,
    options: argparse.Namespace,
    api_key: str,
    *,
    seed: int | None = None,
) -> dict:
    """Synthesise one request's text, returning the worker's own report.

    `seed` overrides the reading's base seed for this one request, which is how
    `--seed-mode vary` gives each chunk its own sampling trajectory.

    Test: `test_clone_part_writes_audio_and_reports_truncation`,
    `test_clone_part_sends_the_voice_direction_fields`
    """
    params = runpod_clone.generation_fields(options, seed=seed)
    payload = {"input": {"op": "clone", "text": text, "voice": options.voice, **params}}
    body = runpod_clone.submit(
        options.endpoint_id,
        api_key,
        payload,
        timeout=options.timeout,
        use_async=not options.sync,
    )
    output = runpod_clone.job_output(body)
    part_path.parent.mkdir(parents=True, exist_ok=True)
    part_path.write_bytes(base64.b64decode(output["audio_b64"]))
    return {
        "wav": part_path.name,
        "text": text,
        "sha": text_sha(text),
        "seed": params["seed"],
        "duration": float(output.get("audio_seconds") or 0.0),
        "executionTime": float(body.get("executionTime") or 0) / 1000.0,
        "truncated": bool(output.get("truncated")),
    }


def synthesise_chunk(
    chunk: Chunk, work_dir: Path, options: argparse.Namespace, api_key: str
) -> dict:
    """Clone a chunk, halving and retrying any part the worker truncated.

    Every part carries its own seed under `--seed-mode vary`, so a re-split
    chunk's halves do not repeat one trajectory either.

    Test: `test_synthesise_chunk_resplits_on_truncation`,
    `test_synthesise_records_a_different_seed_per_chunk`
    """
    mode = getattr(options, "seed_mode", "fixed")
    pending = [(chunk.text, 0)]
    parts: list[dict] = []
    resplits = 0
    while pending:
        text, depth = pending.pop(0)
        part_path = work_dir / f"chunk_{chunk.index:04d}_p{len(parts):02d}.wav"
        seed = part_seed(options.seed, chunk.index, len(parts), mode)
        report = clone_part(text, part_path, options, api_key, seed=seed)
        if report["truncated"] and depth < MAX_RESPLIT_DEPTH:
            halves = split_in_half(text)
            if len(halves) > 1:
                print(
                    f"chunk {chunk.index} truncated; re-splitting into "
                    f"{len(halves)} parts",
                    file=sys.stderr,
                )
                part_path.unlink(missing_ok=True)
                resplits += 1
                pending = [(half, depth + 1) for half in halves] + pending
                continue
        if report["truncated"]:
            raise SystemExit(
                f"chunk {chunk.index} still truncated after "
                f"{MAX_RESPLIT_DEPTH} re-splits; lower --word-budget."
            )
        parts.append(report)
    return {
        "index": chunk.index,
        "text": chunk.text,
        "sha": text_sha(chunk.text),
        "seed": parts[0]["seed"],
        "params": params_sha(
            runpod_clone.generation_fields(options, seed=parts[0]["seed"])
        ),
        "duration": round(sum(part["duration"] for part in parts), 3),
        "executionTime": round(sum(part["executionTime"] for part in parts), 3),
        "truncated": False,
        "ends_paragraph": chunk.ends_paragraph,
        "resplits": resplits,
        "parts": parts,
    }


def _part_is_intact(path: Path, expected_seconds: float | None) -> bool:
    """True when a part WAV decodes whole and is as long as the manifest says.

    Why: existence is not enough. A zero-frame or truncated WAV satisfies
    `is_file()`, so resume treats a chunk that never finished as done and the
    reading loses it. Rejecting it here re-synthesises that chunk instead.

    Test: `test_part_is_intact_rejects_a_header_only_wav`,
    `test_part_is_intact_rejects_a_duration_mismatch`
    """
    if not path.is_file():
        return False
    try:
        data = read_pcm(path)
    except (SystemExit, wave.Error, EOFError, OSError):
        return False
    seconds = len(data) / (SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS)
    if seconds <= 0:
        return False
    if expected_seconds is None:
        return True
    return abs(seconds - expected_seconds) <= PART_DURATION_TOLERANCE_S


def _is_reusable(
    record: dict, chunk: Chunk, work_dir: Path, options: argparse.Namespace
) -> bool:
    """True when a manifest record still matches the request and its WAVs are whole.

    A record written before this script recorded generation settings carries no
    `params`, and every such record was made with the base seed and nothing else
    set — so it is read as exactly that rather than discarded, and a work
    directory from an older run still resumes under `--seed-mode fixed`.

    Test: `test_synthesise_resumes_from_manifest`,
    `test_synthesise_redoes_a_chunk_whose_wav_is_truncated`,
    `test_synthesise_redoes_a_chunk_when_the_seed_mode_changes`,
    `test_synthesise_resumes_a_manifest_written_before_params`
    """
    if record.get("sha") != text_sha(chunk.text) or record.get("truncated"):
        return False
    mode = getattr(options, "seed_mode", "fixed")
    wanted = params_sha(
        runpod_clone.generation_fields(
            options, seed=part_seed(options.seed, chunk.index, 0, mode)
        )
    )
    legacy = params_sha({"seed": options.seed, "cfg_scale": options.cfg_scale})
    if record.get("params", legacy) != wanted:
        return False
    parts = record.get("parts") or []
    return bool(parts) and all(
        _part_is_intact(work_dir / part["wav"], part.get("duration")) for part in parts
    )


def synthesise(
    chunks: list[Chunk],
    work_dir: Path,
    options: argparse.Namespace,
    api_key: str,
    *,
    lexicon: str | None = None,
) -> list[dict]:
    """Synthesise every chunk, skipping those the work directory already holds.

    A lexicon change is detected by comparing `lexicon` with the sha the previous
    run recorded, but it never invalidates a chunk on its own: respelling happens
    before chunking, so a changed lexicon shows up as a changed chunk text, and
    the per-chunk sha check already re-synthesises exactly those chunks and no
    others. The comparison only says so out loud.

    Test: `test_synthesise_resumes_from_manifest`,
    `test_synthesise_redoes_only_the_chunks_a_lexicon_change_touched`
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = work_dir / "manifest.json"
    previous = manifest_lexicon(manifest_path)
    if lexicon is not None and previous is not None and previous != lexicon:
        print(
            f"pronunciation lexicon changed ({previous} -> {lexicon}); "
            "re-synthesising every chunk whose text it altered",
            file=sys.stderr,
        )
    records = load_manifest(manifest_path)
    reading = [chunk.index for chunk in chunks]
    for chunk in chunks:
        existing = records.get(str(chunk.index))
        if existing and _is_reusable(existing, chunk, work_dir, options):
            print(f"chunk {chunk.index}: cached", file=sys.stderr)
            continue
        print(
            f"chunk {chunk.index}/{len(chunks) - 1}: {len(chunk.text.split())} words",
            file=sys.stderr,
        )
        records[str(chunk.index)] = synthesise_chunk(chunk, work_dir, options, api_key)
        save_manifest(manifest_path, records, lexicon=lexicon, reading=reading)
    save_manifest(manifest_path, records, lexicon=lexicon, reading=reading)
    return [records[str(chunk.index)] for chunk in chunks]


def report(records: list[dict], seconds: float, options: argparse.Namespace) -> None:
    """Print the reading's chunk count, duration, endpoint time and cost."""
    execution = sum(record["executionTime"] for record in records)
    resplit = [record["index"] for record in records if record.get("resplits")]
    print(f"chunks              : {len(records)}")
    print(f"total audio seconds : {seconds:.2f}")
    print(f"total execute secs  : {execution:.2f}")
    print(f"estimated cost      : ${execution * options.rate_per_second:.4f}")
    print(f"re-split chunks     : {resplit if resplit else 'none'}")


def resolve_lexicon(options: argparse.Namespace) -> dict[str, str]:
    """The pronunciation map this run applies, empty when it is switched off.

    Test: `test_resolve_lexicon_is_empty_when_disabled`
    """
    if options.no_pronunciations:
        return {}
    return load_pronunciations(options.pronunciations)


def speech_for(options: argparse.Namespace, lexicon: dict[str, str]) -> str:
    """Reduce the input to spoken words: Markdown, sections, respelling, fillers.

    Disfluencies go in last, after respelling, so the respelled forms are
    already in the text and can be protected from being split. The injected
    words are part of the chunk text, and therefore part of its sha and of the
    manifest, so what the model was asked to say is inspectable.

    Test: `test_speech_for_applies_the_lexicon_after_markdown`,
    `test_speech_for_drops_a_named_section`,
    `test_speech_for_injects_disfluencies_at_the_requested_rate`
    """
    speech = markdown_to_speech(
        options.input.read_text(encoding="utf-8"),
        section=options.section,
        drop_sections=options.drop_section,
    )
    speech = apply_pronunciations(speech, lexicon)
    return inject_disfluencies(
        speech,
        rate=getattr(options, "disfluency_rate", DEFAULT_DISFLUENCY_RATE),
        seed=options.seed,
        protected=tuple(lexicon.values()),
    )


def gap_plan_for(options: argparse.Namespace) -> GapPlan:
    """The join plan this command line asks for.

    Test: `test_gap_plan_for_reads_the_command_line`
    """
    return GapPlan(
        sentence_ms=options.sentence_gap_ms,
        paragraph_ms=options.paragraph_gap_ms,
        jitter=validate_jitter(options.gap_jitter),
        seed=options.seed,
    )


def run(options: argparse.Namespace) -> int:
    started = time.perf_counter()
    api_key = runpod_clone.load_api_key(options.env_file)
    lexicon = resolve_lexicon(options)
    speech = speech_for(options, lexicon)
    if not speech.strip():
        raise SystemExit("The input produced no speakable text.")
    chunks = chunk_text(
        speech, word_budget=options.word_budget, max_words=options.max_words
    )
    work_dir = options.work_dir or options.output.with_suffix("")
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "speech.txt").write_text(speech + "\n", encoding="utf-8")

    manifest_path = work_dir / "manifest.json"
    plan = gap_plan_for(options)
    replay = manifest_gaps(manifest_path) == plan.as_dict()
    records = synthesise(
        chunks, work_dir, options, api_key, lexicon=lexicon_sha(lexicon)
    )
    planned = plan_gaps(
        records,
        work_dir,
        sentence_gap_ms=plan.sentence_ms,
        paragraph_gap_ms=plan.paragraph_ms,
        jitter=plan.jitter,
        seed=plan.seed,
        reuse_recorded=replay,
    )
    cache = load_manifest(manifest_path)
    cache.update({str(record["index"]): record for record in records})
    save_manifest(
        manifest_path,
        cache,
        lexicon=lexicon_sha(lexicon),
        reading=[chunk.index for chunk in chunks],
        gaps=plan.as_dict(),
    )
    seconds = concatenate(planned, options.output)
    print(f"saved {options.output}")
    if options.mp3:
        destination = options.output.with_suffix(".mp3")
        if write_mp3(options.output, destination):
            print(f"saved {destination}")
    report(records, seconds, options)
    print(f"client wall time    : {time.perf_counter() - started:.1f} s")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--endpoint-id", required=True)
    parser.add_argument("--voice", required=True, help="Name of a registered voice.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Per-chunk WAVs and manifest. Defaults to --output without its suffix.",
    )
    parser.add_argument(
        "--section", help='Read one Markdown section, e.g. "## Script".'
    )
    parser.add_argument(
        "--drop-section",
        action="append",
        default=[],
        metavar="HEADING",
        help='Remove a section and everything under it, e.g. "Related reading". '
        "Repeat for more than one. Matches an ATX heading or a bold line.",
    )
    parser.add_argument(
        "--pronunciations",
        type=Path,
        default=DEFAULT_PRONUNCIATIONS,
        help="JSON map of written form to spoken respelling.",
    )
    parser.add_argument(
        "--no-pronunciations",
        action="store_true",
        help="Read every word as written, applying no respellings.",
    )
    parser.add_argument("--word-budget", type=int, default=DEFAULT_WORD_BUDGET)
    parser.add_argument("--max-words", type=int, default=DEFAULT_MAX_WORDS)
    parser.add_argument(
        "--sentence-gap-ms",
        "--gap-sentence-ms",
        type=int,
        default=DEFAULT_SENTENCE_GAP_MS,
        help="Mean silence after a chunk that does not end a paragraph.",
    )
    parser.add_argument(
        "--paragraph-gap-ms",
        "--gap-paragraph-ms",
        type=int,
        default=DEFAULT_PARAGRAPH_GAP_MS,
        help="Mean silence after a chunk that ends a paragraph.",
    )
    parser.add_argument(
        "--gap-jitter",
        type=float,
        default=DEFAULT_GAP_JITTER,
        metavar="FRACTION",
        help="Spread of each gap around its mean, as a fraction of it. "
        "0 gives every join exactly the mean, as before.",
    )
    parser.add_argument(
        "--disfluency-rate",
        type=float,
        default=DEFAULT_DISFLUENCY_RATE,
        metavar="PER_100_WORDS",
        help="Filled pauses to insert per 100 words, from um / uh / so / well / "
        "you know. 0 inserts none; the default is the measured monologue rate.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--seed-mode",
        choices=SEED_MODES,
        default=DEFAULT_SEED_MODE,
        help="'vary' derives a per-chunk seed from --seed, so a long reading "
        "does not sample one trajectory throughout; 'fixed' sends --seed with "
        "every request, as before. Changing this re-synthesises the reading.",
    )
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument(
        "--instruction",
        help="Voice Direction: a natural-language steer on tone, emotion, pace "
        'and delivery, e.g. "Speak conversationally, with natural pacing." '
        "Needs --cfg-scale above 1 (4 is the documented starting point).",
    )
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument(
        "--rate-per-second", type=float, default=DEFAULT_RATE_PER_SECOND
    )
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Block on /runsync instead of /run plus /status polling.",
    )
    parser.add_argument("--mp3", action="store_true", help="Also write an MP3.")
    parser.add_argument("--env-file", type=Path, default=runpod_clone.DEFAULT_ENV_FILE)
    return parser


def main() -> int:
    options = build_parser().parse_args()
    if not options.input.is_file():
        raise SystemExit(f"Input not found: {options.input}")
    return run(options)


if __name__ == "__main__":
    raise SystemExit(main())
