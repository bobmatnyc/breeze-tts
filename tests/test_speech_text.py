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


# --- Disfluency injection -------------------------------------------------

# Expository prose of the shape an article reduces to once the Markdown pass has
# run: four paragraphs, no quotation, parenthetical, URL or code span.
PASSAGE = """The reader splits an article into chunks before it sends anything to the endpoint. Each chunk carries about seventy words, which the worker turns into roughly twenty five seconds of speech. That budget exists because the decode loop stops after a fixed number of steps. A longer chunk would come back cut off in the middle of a word. The client then joins the finished chunks with a short silence between them, and a longer one wherever a paragraph ended.

Every one of those decisions used to be a constant, and the result was a read that never varied at all. A listener notices the sameness long before they can say what it is. The pauses land on a grid, the pace never shifts, and the whole article arrives in a single register. None of that comes from the model, which samples a fresh trajectory whenever it is handed a fresh seed. It comes from the client, which handed it the same seed every single time.

Human speech does none of this. A speaker plans the next clause while still finishing the current one, and that planning shows up as variation in tempo, and as the occasional filled pause. The measured rate in monologue is under four such pauses per hundred words, far lower than conversation but far higher than zero. They cluster at the start of a sentence, where the planning load is heaviest, and they thin out in the middle of a clause. A reader that inserts them evenly has traded one mechanical pattern for another.

So the levers here are deliberately modest. A seed that moves per chunk, a gap drawn from a range rather than fixed, and a filler rate that starts at zero until somebody listens to both versions and decides. None of them retrains."""  # noqa: E501


def filler_count(text: str) -> int:
    """Count the inserted fillers by the exact forms the module inserts."""
    fillers = [f"{filler} " for filler, _ in speech_text.SENTENCE_FILLERS]
    return sum(text.count(filler) for filler in fillers) + text.count(
        f" {speech_text.CLAUSE_FILLER} "
    )


def test_the_passage_is_three_hundred_words() -> None:
    """The rate assertions below are only meaningful against a known length."""
    assert len(PASSAGE.split()) == 300


def test_inject_disfluencies_hits_the_requested_rate() -> None:
    spoken = speech_text.inject_disfluencies(PASSAGE, rate=3.6, seed=42)

    assert 10 <= filler_count(spoken) <= 12


def test_inject_disfluencies_is_byte_identical_for_a_seed() -> None:
    first = speech_text.inject_disfluencies(PASSAGE, rate=3.6, seed=42)
    second = speech_text.inject_disfluencies(PASSAGE, rate=3.6, seed=42)

    assert first == second
    assert first != PASSAGE


def test_inject_disfluencies_moves_with_the_seed() -> None:
    first = speech_text.inject_disfluencies(PASSAGE, rate=3.6, seed=42)
    second = speech_text.inject_disfluencies(PASSAGE, rate=3.6, seed=43)

    assert first != second


def test_inject_disfluencies_is_a_no_op_at_rate_zero() -> None:
    assert speech_text.inject_disfluencies(PASSAGE, rate=0.0, seed=42) == PASSAGE


def test_inject_disfluencies_never_doubles_in_one_sentence() -> None:
    spoken = speech_text.inject_disfluencies(PASSAGE, rate=8.0, seed=7)

    for paragraph in spoken.split("\n\n"):
        for sentence in speech_text.split_sentences(paragraph):
            assert filler_count(sentence + " ") <= 1, sentence


def test_inject_disfluencies_keeps_every_original_word() -> None:
    """Injection adds words. It never drops, reorders or reflows them."""
    spoken = speech_text.inject_disfluencies(PASSAGE, rate=3.6, seed=42)

    added = {filler for filler, _ in speech_text.SENTENCE_FILLERS}
    added.add(speech_text.CLAUSE_FILLER)
    kept = [
        word
        for word in spoken.split()
        if word not in added and word not in {"you", "know,"}
    ]
    original = PASSAGE.split()
    assert len(kept) >= len(original) - 12
    assert kept[-1] == original[-1]


def test_inject_disfluencies_skips_headings_and_quotes() -> None:
    """A heading has no terminal punctuation; a quotation is someone else's."""
    document = (
        "What the Reader Does\n\n"
        'He said "the pauses land on a grid and the pace never shifts at all."\n\n'
        "The client joins the finished chunks with a short silence between them."
    )

    spoken = speech_text.inject_disfluencies(document, rate=40.0, seed=3)

    paragraphs = spoken.split("\n\n")
    assert paragraphs[0] == "What the Reader Does"
    assert paragraphs[1] == document.split("\n\n")[1]
    assert filler_count(paragraphs[2] + " ") == 1


def test_inject_disfluencies_leaves_a_respelled_form_alone() -> None:
    document = (
        "The reader respells a surname so the model says it correctly aloud. "
        "I am Bob Mah-tsu-oh-ka and this is the reading you asked me for."
    )

    spoken = speech_text.inject_disfluencies(
        document, rate=40.0, seed=5, protected=("Mah-tsu-oh-ka",)
    )

    assert "Bob Mah-tsu-oh-ka and this is the reading" in spoken
    assert filler_count(spoken + " ") == 1


def test_inject_disfluencies_lowercases_only_a_safe_opener() -> None:
    sentence = "The client joins the finished chunks with a short silence."
    proper = "Anthropic ships a model that samples a fresh trajectory each time."

    lowered = speech_text.inject_disfluencies(sentence, rate=100.0, seed=1)
    kept = speech_text.inject_disfluencies(proper, rate=100.0, seed=1)

    assert " the client joins" in lowered
    assert " Anthropic ships" in kept
