#!/usr/bin/env python3
"""Read a long text or Markdown file aloud through the RunPod clone endpoint.

Why: `scripts/runpod_clone.py` sends one request and gets one clip back, but the
worker stops at `MAX_NEW_TOKENS` (1500 decode steps, about 12.6 per second of
audio) and returns `truncated: true`. Anything longer than roughly 115 s of
speech has to be split, synthesised piecewise and stitched back together.

What: strips Markdown down to speakable prose, splits it at sentence boundaries
under a word budget, clones each chunk with a fixed seed, and concatenates the
chunk WAVs with a silence gap that widens at paragraph breaks. Per-chunk WAVs
and a manifest live in a work directory, so a re-run resynthesises only what is
missing.

Usage:
    python scripts/runpod_read.py --endpoint-id <id> --voice bob \\
        --input article.md --output outputs/article.wav \\
        --work-dir outputs/reads/article --mp3

Test: `test_markdown_to_speech_drops_frontmatter_and_footer`,
`test_chunk_text_respects_word_budget`, `test_synthesise_resumes_from_manifest`
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import runpod_clone  # noqa: E402  (path is set up immediately above)

SAMPLE_RATE = 24_000
SAMPLE_WIDTH = 2
CHANNELS = 1

DEFAULT_WORD_BUDGET = 70
# 12.6 decode steps per second of audio against a 1500-step ceiling is ~115 s.
# The voice reads at a measured ~165 words per minute, so 110 words is ~40 s.
DEFAULT_MAX_WORDS = 110
DEFAULT_SENTENCE_GAP_MS = 350
DEFAULT_PARAGRAPH_GAP_MS = 700
# RTX 4090 at $1.10/hr, the most expensive tier in the endpoint's GPU pool.
DEFAULT_RATE_PER_SECOND = 0.000306
MAX_RESPLIT_DEPTH = 3
# The worker's reported seconds and the WAV it sent agree exactly on every
# reading on record, so this only absorbs float rounding in the manifest.
PART_DURATION_TOLERANCE_S = 0.05

AUTHOR_FOOTER_PREFIX = "Bob Matsuoka is CTO"

_ABBREVIATIONS = frozenset(
    """mr. mrs. ms. dr. prof. sr. jr. st. inc. ltd. co. corp. vs. etc. e.g. i.e.
    cf. approx. fig. no. dept. est. al. a.m. p.m. u.s. u.k.""".split()
)


# --- Markdown to speakable text -------------------------------------------


def strip_frontmatter(text: str) -> str:
    """Drop a leading YAML frontmatter block delimited by `---` lines.

    Test: `test_markdown_to_speech_drops_frontmatter_and_footer`
    """
    if not text.startswith("---"):
        return text
    match = re.match(r"^---\r?\n.*?\r?\n---[ \t]*\r?\n?", text, flags=re.DOTALL)
    return text[match.end() :] if match else text


def select_section(text: str, heading: str) -> str:
    """Return one Markdown section's body, named by its heading.

    The heading may be given with or without its `#` markers. The section runs
    to the next heading at the same or a shallower level.

    Test: `test_select_section_returns_only_that_section`,
    `test_select_section_rejects_unknown_heading`
    """
    wanted = heading.strip()
    hashes, _, title = wanted.partition(" ")
    title = title.strip() if set(hashes) == {"#"} else wanted
    for match in re.finditer(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", text, flags=re.MULTILINE):
        if match.group(2).strip().casefold() != title.casefold():
            continue
        level = len(match.group(1))
        body = text[match.end() :]
        following = re.search(rf"^#{{1,{level}}}[ \t]+", body, flags=re.MULTILINE)
        return body[: following.start()] if following else body
    raise SystemExit(f"No section titled {title!r} in the input.")


def _strip_blocks(text: str) -> str:
    """Remove fenced code, HTML, footnote definitions and link definitions.

    An HTML element that starts its own line is a block — a figure, an embed,
    a caption — so it goes entirely, contents included. A tag sitting inside a
    sentence loses only the tag, because the words around it are still prose.

    A reference link definition — `[label]: https://… "Title"` on its own line —
    is machinery for the `[text][label]` form, never speech, so the whole line
    goes. `_strip_inline` cannot do it: with no parentheses and no second
    bracket, the line matches none of its link rules and the URL is read aloud.

    Test: `test_markdown_to_speech_drops_reference_link_definitions`
    """
    text = re.sub(
        r"^[ \t]*(```|~~~).*?^[ \t]*\1[ \t]*$",
        "",
        text,
        flags=re.DOTALL | re.MULTILINE,
    )
    text = re.sub(r"^\[\^[^\]]+\]:.*(?:\n[ \t]+\S.*)*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(
        r"^[ \t]{0,3}\[[^\^\]][^\]]*\]:[ \t]*<?\S+>?"
        r"(?:[ \t]+(?:\"[^\"]*\"|'[^']*'|\([^)]*\)))?[ \t]*$\n?",
        "",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(
        r"^[ \t]*<([A-Za-z][\w-]*)\b[^>]*>.*?</\1[ \t]*>[ \t]*$",
        "",
        text,
        flags=re.DOTALL | re.MULTILINE,
    )
    text = re.sub(r"^[ \t]*<[A-Za-z/!][^>]*>[ \t]*$", "", text, flags=re.MULTILINE)
    return re.sub(r"<[^<>\n]{1,200}>", "", text)


def _strip_inline(line: str) -> str:
    """Reduce inline Markdown on one line to the words a reader would say."""
    line = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", line)
    line = re.sub(r"\[\^[^\]]+\]", "", line)
    line = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", line)
    line = re.sub(r"\[([^\]]*)\]\[[^\]]*\]", r"\1", line)
    line = re.sub(r"`([^`]*)`", r"\1", line)
    line = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
    line = re.sub(r"\*([^*]+)\*", r"\1", line)
    line = re.sub(r"(?<![\w])__([^_]+)__(?![\w])", r"\1", line)
    line = re.sub(r"(?<![\w])_([^_]+)_(?![\w])", r"\1", line)
    line = re.sub(r"~~([^~]+)~~", r"\1", line)
    return line


def _speakable_line(line: str) -> str | None:
    """Convert one Markdown line to prose, or None when it carries no speech."""
    stripped = line.strip()
    if not stripped:
        return ""
    if re.fullmatch(r"(\*\s*){3,}|(-\s*){3,}|(_\s*){3,}", stripped):
        return None
    if stripped.startswith("|"):
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", cell) for cell in cells if cell):
            return None
        return _strip_inline(", ".join(cell for cell in cells if cell))
    stripped = re.sub(r"^#{1,6}[ \t]+", "", stripped)
    stripped = re.sub(r"^>[ \t]?", "", stripped)
    stripped = re.sub(r"^([-*+]|\d+\.)[ \t]+", "", stripped)
    spoken = _strip_inline(stripped).strip()
    return spoken if spoken else None


def markdown_to_speech(text: str, *, section: str | None = None) -> str:
    """Turn a Markdown document into paragraphs of speakable plain text.

    Frontmatter, images, code blocks, HTML, link URLs, footnote markers and
    bodies, and the closing italic author footer all come out. Heading text,
    link text and table cells stay. Nothing else is expanded or rewritten.

    Test: `test_markdown_to_speech_drops_frontmatter_and_footer`,
    `test_markdown_to_speech_keeps_link_text_and_drops_code`
    """
    body = strip_frontmatter(text)
    if section:
        body = select_section(body, section)
    body = _strip_blocks(body)

    paragraphs: list[str] = []
    current: list[str] = []
    for raw in body.splitlines():
        spoken = _speakable_line(raw)
        if spoken is None:
            continue
        if spoken == "":
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        current.append(spoken)
    if current:
        paragraphs.append(" ".join(current))

    kept = [
        re.sub(r"\s+", " ", paragraph).strip()
        for paragraph in paragraphs
        if not paragraph.startswith(AUTHOR_FOOTER_PREFIX)
    ]
    return "\n\n".join(paragraph for paragraph in kept if paragraph)


# --- Chunking --------------------------------------------------------------


def _is_abbreviation(before: str) -> bool:
    token = re.search(r"[\w.]+$", before)
    if not token:
        return False
    word = token.group(0).casefold()
    return word in _ABBREVIATIONS or bool(re.fullmatch(r"[a-z]\.", word))


def split_sentences(paragraph: str) -> list[str]:
    """Split one paragraph into sentences, holding common abbreviations back.

    Test: `test_split_sentences_finds_boundaries`,
    `test_split_sentences_keeps_abbreviations_together`
    """
    sentences: list[str] = []
    start = 0
    for match in re.finditer(r"""[.!?]+["')\]]*\s+""", paragraph):
        following = paragraph[match.end() : match.end() + 1]
        if following and not re.match(r"""[A-Z0-9"'(\[]""", following):
            continue
        if _is_abbreviation(paragraph[start : match.start() + 1]):
            continue
        sentences.append(paragraph[start : match.end()].strip())
        start = match.end()
    tail = paragraph[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def _word_count(text: str) -> int:
    return len(text.split())


def _split_long_sentence(sentence: str, max_words: int) -> list[str]:
    """Break an over-long sentence at clause boundaries, then at word count."""
    if _word_count(sentence) <= max_words:
        return [sentence]
    pieces = [
        piece.strip()
        for piece in re.split(r"(?<=[,;:])\s+|\s+(?=--\s|—)", sentence)
        if piece.strip()
    ]
    grouped: list[str] = []
    for piece in pieces:
        if grouped and _word_count(f"{grouped[-1]} {piece}") <= max_words:
            grouped[-1] = f"{grouped[-1]} {piece}"
        else:
            grouped.append(piece)
    final: list[str] = []
    for piece in grouped:
        words = piece.split()
        while len(words) > max_words:
            final.append(" ".join(words[:max_words]))
            words = words[max_words:]
        if words:
            final.append(" ".join(words))
    return final


@dataclass(frozen=True)
class Chunk:
    """One request's worth of text, and whether a paragraph ends with it."""

    index: int
    text: str
    ends_paragraph: bool


def chunk_text(
    text: str,
    *,
    word_budget: int = DEFAULT_WORD_BUDGET,
    max_words: int = DEFAULT_MAX_WORDS,
) -> list[Chunk]:
    """Group sentences into chunks under `word_budget`, never mid-sentence.

    A chunk grows until the next sentence would carry it past the budget. Only a
    single sentence longer than `max_words` on its own is cut, at clause
    boundaries, because no sentence boundary is available to cut at.

    Test: `test_chunk_text_respects_word_budget`,
    `test_chunk_text_marks_paragraph_ends`,
    `test_chunk_text_splits_an_oversized_sentence`
    """
    if max_words < word_budget:
        raise SystemExit("--max-words must be at least --word-budget.")
    chunks: list[Chunk] = []
    for paragraph in (block for block in text.split("\n\n") if block.strip()):
        pending: list[str] = []
        started = len(chunks)
        for sentence in split_sentences(paragraph.strip()):
            for piece in _split_long_sentence(sentence, max_words):
                joined = " ".join([*pending, piece])
                if pending and _word_count(joined) > word_budget:
                    chunks.append(Chunk(len(chunks), " ".join(pending), False))
                    pending = [piece]
                else:
                    pending.append(piece)
        if pending:
            chunks.append(Chunk(len(chunks), " ".join(pending), False))
        if len(chunks) > started:
            last = chunks[-1]
            chunks[-1] = Chunk(last.index, last.text, True)
    return chunks


def split_in_half(text: str) -> list[str]:
    """Cut a chunk near its middle, preferring a sentence boundary.

    Test: `test_split_in_half_prefers_a_sentence_boundary`
    """
    sentences = split_sentences(text)
    if len(sentences) < 2:
        words = text.split()
        if len(words) < 2:
            return [text]
        middle = len(words) // 2
        return [" ".join(words[:middle]), " ".join(words[middle:])]
    total = _word_count(text)
    running = 0
    for cut in range(1, len(sentences)):
        running += _word_count(sentences[cut - 1])
        if running >= total / 2:
            return [
                " ".join(sentences[:cut]).strip(),
                " ".join(sentences[cut:]).strip(),
            ]
    return [sentences[0], " ".join(sentences[1:]).strip()]


def text_sha(text: str) -> str:
    """Short content hash used to decide whether a cached chunk still applies."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


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


def plan_gaps(
    records: list[dict],
    work_dir: Path,
    *,
    sentence_gap_ms: int,
    paragraph_gap_ms: int,
) -> list[tuple[Path, int]]:
    """Pair every part WAV with the silence that follows it.

    Parts inside one chunk are halves of a re-split sentence run, so they take
    the sentence gap; only a chunk that ends a paragraph takes the wider one.

    Test: `test_plan_gaps_widens_at_paragraph_ends`
    """
    planned: list[tuple[Path, int]] = []
    for record in records:
        parts = record["parts"]
        for position, part in enumerate(parts):
            last = position == len(parts) - 1
            gap = (
                paragraph_gap_ms
                if last and record.get("ends_paragraph")
                else sentence_gap_ms
            )
            planned.append((work_dir / part["wav"], gap))
    return planned


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


# --- Synthesis -------------------------------------------------------------


def load_manifest(path: Path) -> dict[str, dict]:
    """Read the work directory's manifest, treating a missing file as empty."""
    if not path.is_file():
        return {}
    records = json.loads(path.read_text())["chunks"]
    return {str(record["index"]): record for record in records}


def save_manifest(path: Path, records: dict[str, dict]) -> None:
    """Write the manifest back in chunk order."""
    ordered = [records[key] for key in sorted(records, key=int)]
    path.write_text(json.dumps({"chunks": ordered}, indent=2) + "\n")


def clone_part(
    text: str, part_path: Path, options: argparse.Namespace, api_key: str
) -> dict:
    """Synthesise one request's text, returning the worker's own report.

    Test: `test_clone_part_writes_audio_and_reports_truncation`
    """
    payload = {
        "input": {
            "op": "clone",
            "text": text,
            "seed": options.seed,
            "cfg_scale": options.cfg_scale,
            "voice": options.voice,
        }
    }
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
        "duration": float(output.get("audio_seconds") or 0.0),
        "executionTime": float(body.get("executionTime") or 0) / 1000.0,
        "truncated": bool(output.get("truncated")),
    }


def synthesise_chunk(
    chunk: Chunk, work_dir: Path, options: argparse.Namespace, api_key: str
) -> dict:
    """Clone a chunk, halving and retrying any part the worker truncated.

    Test: `test_synthesise_chunk_resplits_on_truncation`
    """
    pending = [(chunk.text, 0)]
    parts: list[dict] = []
    resplits = 0
    while pending:
        text, depth = pending.pop(0)
        part_path = work_dir / f"chunk_{chunk.index:04d}_p{len(parts):02d}.wav"
        report = clone_part(text, part_path, options, api_key)
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


def _is_reusable(record: dict, chunk: Chunk, work_dir: Path) -> bool:
    """True when a manifest record still matches the chunk and its WAVs are whole.

    Test: `test_synthesise_resumes_from_manifest`,
    `test_synthesise_redoes_a_chunk_whose_wav_is_truncated`
    """
    if record.get("sha") != text_sha(chunk.text) or record.get("truncated"):
        return False
    parts = record.get("parts") or []
    return bool(parts) and all(
        _part_is_intact(work_dir / part["wav"], part.get("duration")) for part in parts
    )


def synthesise(
    chunks: list[Chunk], work_dir: Path, options: argparse.Namespace, api_key: str
) -> list[dict]:
    """Synthesise every chunk, skipping those the work directory already holds.

    Test: `test_synthesise_resumes_from_manifest`
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = work_dir / "manifest.json"
    records = load_manifest(manifest_path)
    for chunk in chunks:
        existing = records.get(str(chunk.index))
        if existing and _is_reusable(existing, chunk, work_dir):
            print(f"chunk {chunk.index}: cached", file=sys.stderr)
            continue
        print(
            f"chunk {chunk.index}/{len(chunks) - 1}: {_word_count(chunk.text)} words",
            file=sys.stderr,
        )
        records[str(chunk.index)] = synthesise_chunk(chunk, work_dir, options, api_key)
        save_manifest(manifest_path, records)
    save_manifest(manifest_path, records)
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


def run(options: argparse.Namespace) -> int:
    started = time.perf_counter()
    api_key = runpod_clone.load_api_key(options.env_file)
    speech = markdown_to_speech(
        options.input.read_text(encoding="utf-8"), section=options.section
    )
    if not speech.strip():
        raise SystemExit("The input produced no speakable text.")
    chunks = chunk_text(
        speech, word_budget=options.word_budget, max_words=options.max_words
    )
    work_dir = options.work_dir or options.output.with_suffix("")
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "speech.txt").write_text(speech + "\n", encoding="utf-8")

    records = synthesise(chunks, work_dir, options, api_key)
    seconds = concatenate(
        plan_gaps(
            records,
            work_dir,
            sentence_gap_ms=options.sentence_gap_ms,
            paragraph_gap_ms=options.paragraph_gap_ms,
        ),
        options.output,
    )
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
    parser.add_argument("--word-budget", type=int, default=DEFAULT_WORD_BUDGET)
    parser.add_argument("--max-words", type=int, default=DEFAULT_MAX_WORDS)
    parser.add_argument("--sentence-gap-ms", type=int, default=DEFAULT_SENTENCE_GAP_MS)
    parser.add_argument(
        "--paragraph-gap-ms", type=int, default=DEFAULT_PARAGRAPH_GAP_MS
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
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
