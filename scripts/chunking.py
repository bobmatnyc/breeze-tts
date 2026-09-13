#!/usr/bin/env python3
"""Group a document's speakable text into one request's worth of words at a time.

Why: the worker stops at 1500 decode steps, about 115 s of speech, so a reading
is only possible if the text is cut into pieces that each finish inside that
ceiling. Cutting anywhere but a sentence boundary is audible, so the budget is
spent in whole sentences.

What: `chunk_text` fills chunks to `word_budget` without crossing a sentence
boundary, cutting a single over-long sentence at its clauses only when no
sentence boundary is available. `split_in_half` is the retry path for a chunk
the worker still truncated, and `text_sha` is the cache key that decides whether
a chunk's audio still applies.

Test: `tests/test_runpod_read.py`
"""

from __future__ import annotations

import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from speech_text import split_sentences  # noqa: E402  (path set up above)

DEFAULT_WORD_BUDGET = 70
# 12.6 decode steps per second of audio against a 1500-step ceiling is ~115 s.
# The voice reads at a measured ~165 words per minute, so 110 words is ~40 s.
DEFAULT_MAX_WORDS = 110


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
