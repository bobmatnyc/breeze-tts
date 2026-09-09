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
import re
from pathlib import Path

AUTHOR_FOOTER_PREFIX = "Bob Matsuoka is CTO"

DEFAULT_PRONUNCIATIONS = Path(__file__).resolve().parent / "pronunciations.json"


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
    the end of the document. A pseudo-heading's level 7 is clamped to 6 when
    looking for that next heading, so it ends at the next real heading.

    Test: `test_select_section_returns_only_that_section`,
    `test_drop_section_stops_at_the_next_heading`
    """
    title = _normalise_title(heading)
    spans = _heading_spans(text)
    for position, (start, end, level, found) in enumerate(spans):
        if _normalise_title(found).casefold() != title.casefold():
            continue
        stop = len(text)
        for next_start, _, next_level, _ in spans[position + 1 :]:
            if next_level <= min(level, 6):
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
