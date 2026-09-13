#!/usr/bin/env python3
"""Turn a document into the exact words a reader should say.

Why: `scripts/runpod_read.py` had grown past the 500-line cap, and everything
that decides *what words come out* — Markdown stripping, section selection,
section dropping, pronunciation respelling — is one concern with no dependency
on RunPod, WAV assembly or the manifest. Splitting it out keeps the reader to
synthesis and lets the text rules be tested on their own.

What: `markdown_to_speech` reduces Markdown to speakable paragraphs, keeping or
dropping named sections; `apply_pronunciations` respells written forms the model
mis-reads. The reader calls them in that order, before chunking, so a chunk's
text — and therefore its manifest sha — already carries every text decision.

Test: `test_markdown_to_speech_drops_frontmatter_and_footer`,
`test_apply_pronunciations_matches_whole_words_only`,
`test_drop_section_removes_a_bold_pseudo_heading`
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

AUTHOR_FOOTER_PREFIX = "Bob Matsuoka is CTO"

DEFAULT_PRONUNCIATIONS = Path(__file__).resolve().parent / "pronunciations.json"

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


def _normalise_title(heading: str) -> str:
    """Reduce a heading argument to its bare title, without `#` or a colon."""
    wanted = heading.strip()
    hashes, _, title = wanted.partition(" ")
    title = title.strip() if set(hashes) == {"#"} else wanted
    return title.rstrip(":").strip()


def _heading_spans(text: str) -> list[tuple[int, int, int, str]]:
    """Every heading in the document as `(start, end, level, title)`.

    An ATX heading takes its own `#` count as its level. A line that is nothing
    but bold or italic text — `**Related reading:**` — is a pseudo-heading: real
    documents use it where a heading belongs, so it gets level 7, deeper than any
    ATX heading, and therefore ends at the next heading of any level.

    Test: `test_drop_section_removes_a_bold_pseudo_heading`
    """
    spans: list[tuple[int, int, int, str]] = []
    for match in re.finditer(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", text, flags=re.MULTILINE):
        spans.append((match.start(), match.end(), len(match.group(1)), match.group(2)))
    for match in re.finditer(
        r"^[ \t]{0,3}(\*\*|__)(.+?)\1[ \t]*$", text, flags=re.MULTILINE
    ):
        spans.append((match.start(), match.end(), 7, match.group(2)))
    return sorted(spans)


def _section_bounds(text: str, heading: str) -> tuple[int, int, int]:
    """Locate one section as `(heading_start, body_start, section_end)`.

    The section runs to the next heading at the same or a shallower level, or to
    the end of the document. A pseudo-heading sits at level 7, so every heading
    is same-or-shallower and it ends at the next heading of any kind — including
    the next pseudo-heading. Ending it only at a real ATX heading would let a
    drop of `**Related reading:**` swallow a following `**About the author:**`
    with nothing in the output to show it happened.

    Test: `test_select_section_returns_only_that_section`,
    `test_drop_section_stops_at_the_next_heading`,
    `test_drop_section_stops_at_the_next_pseudo_heading`
    """
    title = _normalise_title(heading)
    spans = _heading_spans(text)
    for position, (start, end, level, found) in enumerate(spans):
        if _normalise_title(found).casefold() != title.casefold():
            continue
        stop = len(text)
        for next_start, _, next_level, _ in spans[position + 1 :]:
            if next_level <= level:
                stop = next_start
                break
        return start, end, stop
    raise SystemExit(f"No section titled {title!r} in the input.")


def select_section(text: str, heading: str) -> str:
    """Return one Markdown section's body, named by its heading.

    Test: `test_select_section_returns_only_that_section`,
    `test_select_section_rejects_unknown_heading`
    """
    _, body_start, stop = _section_bounds(text, heading)
    return text[body_start:stop]


def drop_section(text: str, heading: str) -> str:
    """Return the document without one section — its heading and its body.

    Why: an article's trailing "Related reading" list is link furniture. Read
    aloud it becomes a run of titles and taglines with no sentences in it, and
    editing it out of the source by hand before every run is the step that gets
    forgotten.

    Test: `test_drop_section_removes_a_bold_pseudo_heading`,
    `test_drop_section_stops_at_the_next_heading`,
    `test_drop_section_rejects_unknown_heading`
    """
    start, _, stop = _section_bounds(text, heading)
    return text[:start] + text[stop:]


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


def markdown_to_speech(
    text: str,
    *,
    section: str | None = None,
    drop_sections: list[str] | None = None,
) -> str:
    """Turn a Markdown document into paragraphs of speakable plain text.

    Frontmatter, images, code blocks, HTML, link URLs, footnote markers and
    bodies, and the closing italic author footer all come out. Heading text,
    link text and table cells stay. Nothing else is expanded or rewritten.

    `section` keeps one named section; every name in `drop_sections` removes
    one, and both are resolved against the Markdown, before any stripping, so a
    heading is still a heading when it is matched.

    Test: `test_markdown_to_speech_drops_frontmatter_and_footer`,
    `test_markdown_to_speech_keeps_link_text_and_drops_code`,
    `test_markdown_to_speech_drops_a_named_section`
    """
    body = strip_frontmatter(text)
    if section:
        body = select_section(body, section)
    for heading in drop_sections or []:
        body = drop_section(body, heading)
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


# --- Pronunciation lexicon -------------------------------------------------


def load_pronunciations(path: Path) -> dict[str, str]:
    """Read a written-form to spoken-form JSON map, refusing an ambiguous one.

    Why: a respelling is a guess about how the model reads a word, so it belongs
    in data a human can edit and diff, not in the reader's source. Two keys that
    differ only in case are rejected because matching is case-insensitive, so
    which of them applied would depend on dictionary order.

    Test: `test_load_pronunciations_rejects_a_non_string_value`,
    `test_load_pronunciations_rejects_keys_differing_only_in_case`
    """
    if not path.is_file():
        raise SystemExit(f"Pronunciation file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{path} must hold a JSON object of written: spoken.")
    seen: dict[str, str] = {}
    for written, spoken in data.items():
        if not written.strip() or not isinstance(spoken, str) or not spoken.strip():
            raise SystemExit(f"{path}: {written!r} needs a non-empty spoken form.")
        folded = written.casefold()
        if folded in seen:
            raise SystemExit(
                f"{path}: {written!r} and {seen[folded]!r} differ only in case."
            )
        seen[folded] = written
    return data


def lexicon_sha(lexicon: dict[str, str]) -> str:
    """Short content hash of a lexicon, stable under key order.

    Test: `test_lexicon_sha_ignores_key_order`
    """
    canonical = json.dumps(lexicon, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _match_case(written: str, matched: str, spoken: str) -> str:
    """Carry the matched text's capitalisation onto the spoken form.

    An exact hit on the key keeps the spoken form verbatim, so a lexicon can
    spell a respelling however it likes. Otherwise only the three unambiguous
    shapes — all lower, all upper, leading capital — are transferred.

    Test: `test_apply_pronunciations_preserves_case`
    """
    if matched == written:
        return spoken
    if matched.islower():
        return spoken.lower()
    if matched.isupper() and len(matched) > 1:
        return spoken.upper()
    if matched[:1].isupper() and matched[1:].islower():
        return spoken[:1].upper() + spoken[1:]
    return spoken


def apply_pronunciations(text: str, lexicon: dict[str, str]) -> str:
    """Respell every whole-word occurrence of a lexicon key.

    Matching is case-insensitive and the replacement carries the matched text's
    case. Keys are tried longest first, so a multi-word entry wins over a key
    that is only its first word. `(?<!\\w)`/`(?!\\w)` rather than `\\b` because a
    respelling may end in a hyphen or a dot, where `\\b` flips meaning.

    Test: `test_apply_pronunciations_matches_whole_words_only`,
    `test_apply_pronunciations_prefers_the_longest_key`,
    `test_apply_pronunciations_preserves_case`
    """
    if not lexicon:
        return text
    by_fold = {
        written.casefold(): (written, spoken) for written, spoken in lexicon.items()
    }
    ordered = sorted(lexicon, key=len, reverse=True)
    pattern = re.compile(
        r"(?<!\w)(" + "|".join(re.escape(key) for key in ordered) + r")(?!\w)",
        flags=re.IGNORECASE,
    )

    def replace(match: re.Match[str]) -> str:
        matched = match.group(1)
        written, spoken = by_fold[matched.casefold()]
        return _match_case(written, matched, spoken)

    return pattern.sub(replace, text)


# --- Sentence splitting ----------------------------------------------------


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


# --- Disfluency injection --------------------------------------------------

# Oviatt (1995), reported in Bortfeld et al. 2001, measured about 3.6
# disfluencies per 100 words in monologue against 5.5-8.8 in dialogue, and
# Boomer/Shriberg put them at sentence and turn beginnings where planning load
# is highest. Clark and Fox Tree separate the two fillers: "um" marks a strong
# boundary and a longer delay, "uh" a weaker clause-internal one. Sources in
# articles/hyperdev/research/tts-naturalness-techniques.md, section 3.
SENTENCE_FILLERS = (("Um,", 0.70), ("So,", 0.15), ("You know,", 0.15))
CLAUSE_FILLER = "uh,"
SENTENCE_START_SHARE = 0.75

# A filler needs a sentence long enough to have a planning load worth marking.
MIN_FILLER_SENTENCE_WORDS = 6
# A clause filler needs real words on both sides of the boundary it follows.
MIN_CLAUSE_SIDE_WORDS = 3

# Any sentence carrying one of these is left alone outright rather than
# reasoned about span by span: a quotation is someone else's words, a
# parenthetical may be a vocal-event tag the model renders, a backtick is a
# code span that escaped the Markdown pass, and a URL is not read as prose.
_UNSAFE_CHARACTERS = "()[]{}\"'`“”‘’"
_URL_MARKERS = ("://", "www.", "@")
# Lowercased when a filler takes over the capital slot. Only closed-class
# openers, so a sentence starting on a proper noun keeps its capital.
_LOWERABLE_OPENERS = frozenset(
    """a all an and any as at because before both but by each either every for
    from he her here his how if in it its many most much neither no none not
    nothing on once one other our over she some something such that the their
    them then there these they this those to two under we what when where while
    why with you your""".split()
)


def _sentence_is_eligible(sentence: str, protected: tuple[str, ...]) -> bool:
    """True when a filler can be inserted into this sentence without damage.

    A sentence with no terminal punctuation is a heading or a fragment — the
    Markdown pass flattens headings into bare paragraphs, so terminal
    punctuation is what distinguishes prose from a title at this stage.

    Test: `test_inject_disfluencies_skips_headings_and_quotes`
    """
    if not re.search(r"[.!?]['\")\]]*$", sentence):
        return False
    if len(sentence.split()) < MIN_FILLER_SENTENCE_WORDS:
        return False
    if any(character in sentence for character in _UNSAFE_CHARACTERS):
        return False
    if any(marker in sentence for marker in _URL_MARKERS):
        return False
    return not any(word and word in sentence for word in protected)


def _clause_offsets(sentence: str) -> list[int]:
    """Offsets of clause boundaries inside a sentence, with words either side."""
    offsets: list[int] = []
    for match in re.finditer(r"[,;:]\s", sentence):
        cut = match.start()
        if (
            len(sentence[:cut].split()) >= MIN_CLAUSE_SIDE_WORDS
            and len(sentence[cut + 1 :].split()) >= MIN_CLAUSE_SIDE_WORDS
        ):
            offsets.append(cut)
    return offsets


def _weighted_filler(draw: float) -> str:
    """Pick a sentence-initial filler from the weighted pool."""
    running = 0.0
    for filler, weight in SENTENCE_FILLERS:
        running += weight
        if draw < running:
            return filler
    return SENTENCE_FILLERS[-1][0]


def _open_sentence_with(sentence: str, filler: str) -> str:
    """Put `filler` in front, lowercasing the displaced opener when it is safe."""
    head, separator, tail = sentence.partition(" ")
    if head[:1].isupper() and head.casefold().strip(",.;:") in _LOWERABLE_OPENERS:
        head = head[:1].lower() + head[1:]
    return f"{filler} {head}{separator}{tail}"


def inject_disfluencies(
    text: str,
    *,
    rate: float,
    seed: int,
    protected: tuple[str, ...] = (),
) -> str:
    """Insert filled pauses into speakable text at the measured monologue rate.

    Why: the reader's cadence is even because written prose is even. Spontaneous
    monologue carries about 3.6 disfluencies per 100 words, clustered where
    planning load is highest, and none of that reaches the model unless the text
    carries it.

    What: `rate` is fillers per 100 words. Placement is seeded and deterministic,
    so the same text, rate and seed produce byte-identical output and therefore
    the same chunk shas and the same cached audio. One filler per sentence at
    most. "Um" and its "so," / "you know," variants open a sentence; "uh" sits
    at a clause boundary inside one. Sentences that quote, parenthesise, carry a
    URL or a code span, or hold one of the `protected` respellings are skipped
    whole, as are headings, which reach this stage as paragraphs with no
    terminal punctuation. A rate of 0 returns the text unchanged.

    Test: `test_inject_disfluencies_hits_the_requested_rate`,
    `test_inject_disfluencies_is_byte_identical_for_a_seed`,
    `test_inject_disfluencies_is_a_no_op_at_rate_zero`,
    `test_inject_disfluencies_never_doubles_in_one_sentence`
    """
    if rate <= 0:
        return text

    paragraphs = text.split("\n\n")
    split: list[list[str] | str] = []
    candidates: list[tuple[int, int]] = []
    for position, paragraph in enumerate(paragraphs):
        sentences = split_sentences(paragraph)
        # Rebuilding must be lossless, or injection would reflow prose it was
        # only asked to add words to.
        if not sentences or " ".join(sentences) != paragraph.strip():
            split.append(paragraph)
            continue
        split.append(sentences)
        for order, sentence in enumerate(sentences):
            if _sentence_is_eligible(sentence, protected):
                candidates.append((position, order))

    target = min(int(round(rate * len(text.split()) / 100.0)), len(candidates))
    if target <= 0:
        return text

    stream = random.Random(f"disfluency:{seed}")
    chosen = sorted(stream.sample(range(len(candidates)), target))
    for slot in chosen:
        position, order = candidates[slot]
        sentences = split[position]
        sentence = sentences[order]
        offsets = _clause_offsets(sentence)
        if offsets and stream.random() >= SENTENCE_START_SHARE:
            cut = offsets[stream.randrange(len(offsets))] + 1
            sentences[order] = f"{sentence[:cut]} {CLAUSE_FILLER}{sentence[cut:]}"
        else:
            sentences[order] = _open_sentence_with(
                sentence, _weighted_filler(stream.random())
            )

    return "\n\n".join(
        piece if isinstance(piece, str) else " ".join(piece) for piece in split
    )
