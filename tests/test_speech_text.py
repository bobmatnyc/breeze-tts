"""Tests for the document-to-spoken-words rules.

Section dropping and the pronunciation lexicon both change which words reach the
endpoint, so they are the two places where a silent mistake costs a re-synthesis
of the whole reading. Everything here is pure text.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "speech_text.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("speech_text", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["speech_text"] = module
    spec.loader.exec_module(module)
    return module


speech_text = _load_module()


# --- Section dropping -----------------------------------------------------


ARTICLE = """## The argument

The body of the piece.

**Related reading:**
- [What Is Harness Engineering?](https://example.com/harness). Why it matters
- [AI Power Ranking](https://example.com/rank). Benchmarks
"""


def test_drop_section_removes_a_bold_pseudo_heading() -> None:
    """`**Related reading:**` sits where a heading belongs, so it drops like one."""
    speech = speech_text.markdown_to_speech(ARTICLE, drop_sections=["Related reading"])

    assert speech == "The argument\n\nThe body of the piece."
    assert "Harness Engineering" not in speech
    assert "AI Power Ranking" not in speech


TWO_PSEUDO_HEADINGS = """## The argument

The body of the piece.

**Related reading:**
- [What Is Harness Engineering?](https://example.com/harness). Why it matters

**About the author:**
Bob writes about AI-augmented engineering practice.
"""


def test_drop_section_stops_at_the_next_pseudo_heading() -> None:
    """Two bold pseudo-headings in a row: dropping the first must keep the second.

    The regression: a pseudo-heading's section was terminated only by a real ATX
    heading, so with no `#` between them the drop swallowed the bio as well —
    silently, because a dropped section leaves no trace in the output.
    """
    speech = speech_text.markdown_to_speech(
        TWO_PSEUDO_HEADINGS, drop_sections=["Related reading"]
    )

    assert "Harness Engineering" not in speech
    assert "About the author:" in speech
    assert "Bob writes about AI-augmented engineering practice." in speech


def test_select_section_stops_at_the_next_pseudo_heading() -> None:
    """Selecting the first of two adjacent pseudo-headings excludes the second."""
    body = speech_text.select_section(TWO_PSEUDO_HEADINGS, "Related reading")

    assert "Harness Engineering" in body
    assert "About the author" not in body
    assert "Bob writes about" not in body


def test_drop_section_stops_at_the_next_heading() -> None:
    document = "# One\n\nKept.\n\n## Two\n\nDropped.\n\n## Three\n\nAlso kept.\n"

    speech = speech_text.markdown_to_speech(document, drop_sections=["## Two"])

    assert "Kept." in speech
    assert "Also kept." in speech
    assert "Dropped." not in speech
    assert "Two" not in speech


def test_drop_section_removes_a_deeper_nested_subsection_too() -> None:
    document = (
        "## Keep\n\nA.\n\n## Go\n\nB.\n\n### Under go\n\nC.\n\n## Keep two\n\nD.\n"
    )

    speech = speech_text.markdown_to_speech(document, drop_sections=["Go"])

    assert "C." not in speech
    assert "Under go" not in speech
    assert "D." in speech


def test_drop_section_accepts_several_headings() -> None:
    document = "## A\n\nOne.\n\n## B\n\nTwo.\n\n## C\n\nThree.\n"

    speech = speech_text.markdown_to_speech(document, drop_sections=["A", "C"])

    assert speech == "B\n\nTwo."


def test_drop_section_rejects_unknown_heading() -> None:
    with pytest.raises(SystemExit, match="No section titled"):
        speech_text.drop_section(ARTICLE, "Nowhere")


def test_markdown_to_speech_drops_a_named_section() -> None:
    """The drop happens on Markdown, so the heading is still a heading."""
    speech = speech_text.markdown_to_speech(
        "---\ntitle: t\n---\n## Body\n\nSpoken.\n\n## Notes\n\nUnspoken.\n",
        drop_sections=["Notes"],
    )

    assert speech == "Body\n\nSpoken."


# --- Loading the lexicon ---------------------------------------------------


def write_lexicon(path: Path, mapping: dict) -> Path:
    path.write_text(json.dumps(mapping), encoding="utf-8")
    return path


def test_load_pronunciations_reads_the_map(tmp_path: Path) -> None:
    path = write_lexicon(tmp_path / "p.json", {"Matsuoka": "Mah-tsu-oh-ka"})

    assert speech_text.load_pronunciations(path) == {"Matsuoka": "Mah-tsu-oh-ka"}


def test_load_pronunciations_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="not found"):
        speech_text.load_pronunciations(tmp_path / "absent.json")


def test_load_pronunciations_rejects_a_non_string_value(tmp_path: Path) -> None:
    path = write_lexicon(tmp_path / "p.json", {"Matsuoka": 7})

    with pytest.raises(SystemExit, match="non-empty spoken form"):
        speech_text.load_pronunciations(path)


def test_load_pronunciations_rejects_a_non_object(tmp_path: Path) -> None:
    (tmp_path / "p.json").write_text('["Matsuoka"]', encoding="utf-8")

    with pytest.raises(SystemExit, match="JSON object"):
        speech_text.load_pronunciations(tmp_path / "p.json")


def test_load_pronunciations_rejects_keys_differing_only_in_case(
    tmp_path: Path,
) -> None:
    """Matching is case-insensitive, so two such keys have no defined winner."""
    path = write_lexicon(tmp_path / "p.json", {"Duetto": "Doo-etto", "duetto": "x"})

    with pytest.raises(SystemExit, match="differ only in case"):
        speech_text.load_pronunciations(path)


def test_the_shipped_lexicon_loads_and_respells_the_surname() -> None:
    shipped = speech_text.load_pronunciations(speech_text.DEFAULT_PRONUNCIATIONS)

    assert shipped["Matsuoka"] == "Mah-tsu-oh-ka"
    assert speech_text.apply_pronunciations("I'm Bob Matsuoka.", shipped) == (
        "I'm Bob Mah-tsu-oh-ka."
    )


def test_lexicon_sha_ignores_key_order() -> None:
    first = speech_text.lexicon_sha({"a": "1", "b": "2"})
    second = speech_text.lexicon_sha({"b": "2", "a": "1"})

    assert first == second
    assert first != speech_text.lexicon_sha({"a": "1", "b": "3"})


# --- Applying the lexicon --------------------------------------------------


LEXICON = {"Matsuoka": "Mah-tsu-oh-ka"}


def test_apply_pronunciations_matches_whole_words_only() -> None:
    text = "Matsuoka wrote it. Matsuokas did not. Not xMatsuoka either."

    spoken = speech_text.apply_pronunciations(text, LEXICON)

    assert spoken == (
        "Mah-tsu-oh-ka wrote it. Matsuokas did not. Not xMatsuoka either."
    )


def test_apply_pronunciations_matches_across_punctuation() -> None:
    text = "hyperdev dot matsuoka dot com, and (Matsuoka) too."

    spoken = speech_text.apply_pronunciations(text, LEXICON)

    assert spoken == ("hyperdev dot mah-tsu-oh-ka dot com, and (Mah-tsu-oh-ka) too.")


def test_apply_pronunciations_preserves_case() -> None:
    text = "Matsuoka, matsuoka, MATSUOKA."

    spoken = speech_text.apply_pronunciations(text, LEXICON)

    assert spoken == "Mah-tsu-oh-ka, mah-tsu-oh-ka, MAH-TSU-OH-KA."


def test_apply_pronunciations_prefers_the_longest_key() -> None:
    """A two-word entry must win over a key that is only its first word."""
    lexicon = {"Duetto": "Doo-etto", "Duetto Research": "Doo-etto Ree-search"}

    spoken = speech_text.apply_pronunciations("Duetto Research and Duetto.", lexicon)

    assert spoken == "Doo-etto Ree-search and Doo-etto."


def test_apply_pronunciations_is_a_single_pass() -> None:
    """A replacement is never itself rewritten, however the lexicon chains."""
    lexicon = {"one": "two", "two": "three"}

    assert speech_text.apply_pronunciations("one two", lexicon) == "two three"


def test_apply_pronunciations_leaves_text_alone_when_empty() -> None:
    assert speech_text.apply_pronunciations("Matsuoka.", {}) == "Matsuoka."
