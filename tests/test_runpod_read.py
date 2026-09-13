"""Tests for the long-text reader's offline halves.

Markdown-to-speech conversion, chunking, WAV assembly and the resume path are
all pure or filesystem-local, so they run without touching the endpoint. The
few tests that need a job response stub `runpod_clone.submit`.
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import sys
import wave
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "runpod_read.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("runpod_read", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["runpod_read"] = module
    spec.loader.exec_module(module)
    return module


runpod_read = _load_module()


def write_wav(path: Path, seconds: float) -> Path:
    """Write a silent 24 kHz mono 16-bit WAV of the requested length."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(runpod_read.SAMPLE_RATE)
        handle.writeframes(b"\x01\x00" * int(runpod_read.SAMPLE_RATE * seconds))
    return path


def wav_bytes(seconds: float, tmp_path: Path) -> bytes:
    return write_wav(tmp_path / "stub.wav", seconds).read_bytes()


def options(**overrides) -> argparse.Namespace:
    defaults = {
        "endpoint_id": "e1",
        "voice": "bob",
        "seed": 42,
        "seed_mode": "fixed",
        "cfg_scale": 1.0,
        "instruction": None,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "timeout": 60,
        "sync": False,
        "rate_per_second": 0.000306,
    }
    return argparse.Namespace(**{**defaults, **overrides})


ARTICLE = """---
title: "A Title"
tags:
  - one
---
# A Heading

Text with a [link](https://example.com/x) and an ![image](images/a.png) in it.
It cites a claim.[^1]

```python
print("this is code")
```

<div class="note">markup</div>

| Decision | Question |
|---|---|
| Outcome | What state should exist? |

[^1]: The footnote body, which nobody reads aloud.

---

*Bob Matsuoka is CTO of [Duetto](https://www.duettocloud.com/), a platform.*
"""


# --- Markdown to speech ---------------------------------------------------


def test_markdown_to_speech_drops_frontmatter_and_footer() -> None:
    speech = runpod_read.markdown_to_speech(ARTICLE)

    assert "title:" not in speech
    assert speech.startswith("A Heading")
    assert "Bob Matsuoka is CTO" not in speech
    assert "Duetto" not in speech


def test_markdown_to_speech_keeps_link_text_and_drops_code() -> None:
    speech = runpod_read.markdown_to_speech(ARTICLE)

    assert "Text with a link and an in it." in speech
    assert "https://example.com/x" not in speech
    assert "images/a.png" not in speech
    assert "this is code" not in speech
    assert "markup" not in speech
    assert 'class="note"' not in speech


def test_markdown_to_speech_drops_footnote_marker_and_body() -> None:
    speech = runpod_read.markdown_to_speech(ARTICLE)

    assert "It cites a claim." in speech
    assert "[^1]" not in speech
    assert "nobody reads aloud" not in speech


def test_markdown_to_speech_drops_reference_link_definitions() -> None:
    """`[label]: url "Title"` is machinery for `[text][label]`, never speech."""
    document = (
        "Delegation is a skill, as the [2001 study][study] showed.\n\n"
        '[study]: https://example.com/paper.pdf "The 2001 Study"\n'
        "[plain]: http://example.org/other\n"
        "[angle]: <https://example.net/spaced> 'Single quoted'\n"
    )

    speech = runpod_read.markdown_to_speech(document)

    assert speech == "Delegation is a skill, as the 2001 study showed."
    assert "example.com" not in speech
    assert "example.org" not in speech
    assert "example.net" not in speech


def test_markdown_to_speech_keeps_a_colon_inside_a_sentence() -> None:
    """The definition rule must not eat prose that merely contains brackets."""
    document = "The tag [draft]: a working title, stays in the sentence.\n"

    speech = runpod_read.markdown_to_speech(document)

    assert "a working title, stays in the sentence" in speech


def test_markdown_to_speech_reads_table_rows_without_pipes() -> None:
    speech = runpod_read.markdown_to_speech(ARTICLE)

    assert "Decision, Question" in speech
    assert "Outcome, What state should exist?" in speech
    assert "|" not in speech
    assert "---" not in speech


def test_markdown_to_speech_rejects_empty_input() -> None:
    assert runpod_read.markdown_to_speech("---\ntitle: x\n---\n") == ""


# --- Section selection ----------------------------------------------------


SECTIONED = """## Script

Spoken words here.

## Tone and delivery

Not spoken.

### Nested under tone

Also not spoken.
"""


def test_select_section_returns_only_that_section() -> None:
    speech = runpod_read.markdown_to_speech(SECTIONED, section="## Script")

    assert speech == "Spoken words here."


def test_select_section_accepts_a_bare_title() -> None:
    assert runpod_read.select_section(SECTIONED, "Script").strip() == (
        "Spoken words here."
    )


def test_select_section_rejects_unknown_heading() -> None:
    with pytest.raises(SystemExit, match="No section titled"):
        runpod_read.select_section(SECTIONED, "## Missing")


# --- Sentence splitting and chunking --------------------------------------


def test_split_sentences_finds_boundaries() -> None:
    sentences = runpod_read.split_sentences(
        'One thing happened. "Then another!" And a third? Yes.'
    )

    assert sentences == [
        "One thing happened.",
        '"Then another!"',
        "And a third?",
        "Yes.",
    ]


def test_split_sentences_keeps_abbreviations_together() -> None:
    sentences = runpod_read.split_sentences(
        "Dr. Smith spoke at 9 a.m. Everyone listened."
    )

    assert sentences == ["Dr. Smith spoke at 9 a.m. Everyone listened."]


def test_chunk_text_respects_word_budget() -> None:
    paragraph = " ".join(f"Sentence number {n} runs on for a while." for n in range(12))
    chunks = runpod_read.chunk_text(paragraph, word_budget=20, max_words=40)

    assert len(chunks) > 1
    assert all(len(chunk.text.split()) <= 20 for chunk in chunks[:-1])
    assert " ".join(chunk.text for chunk in chunks) == paragraph


def test_chunk_text_never_splits_mid_sentence() -> None:
    paragraph = " ".join(f"This is sentence {n} of the run." for n in range(8))
    chunks = runpod_read.chunk_text(paragraph, word_budget=15, max_words=40)

    for chunk in chunks:
        assert chunk.text.endswith(".")
        assert chunk.text.startswith("This is sentence")


def test_chunk_text_marks_paragraph_ends() -> None:
    text = "First one. Second one.\n\nA new paragraph starts."
    chunks = runpod_read.chunk_text(text, word_budget=3, max_words=20)

    ends = [chunk.ends_paragraph for chunk in chunks]
    assert ends[-1] is True
    assert ends.count(True) == 2
    assert chunks[ends.index(True)].text == "Second one."


def test_chunk_text_splits_an_oversized_sentence() -> None:
    sentence = "Alpha " * 30 + ", beta " * 30 + "end."
    chunks = runpod_read.chunk_text(sentence, word_budget=20, max_words=25)

    assert all(len(chunk.text.split()) <= 25 for chunk in chunks)


def test_chunk_text_rejects_a_cap_below_the_budget() -> None:
    with pytest.raises(SystemExit, match="at least"):
        runpod_read.chunk_text("Words.", word_budget=70, max_words=10)


def test_split_in_half_prefers_a_sentence_boundary() -> None:
    halves = runpod_read.split_in_half("One two three. Four five six. Seven eight.")

    assert halves == ["One two three. Four five six.", "Seven eight."]


def test_split_in_half_falls_back_to_words() -> None:
    halves = runpod_read.split_in_half("one two three four")

    assert halves == ["one two", "three four"]


# --- WAV assembly ---------------------------------------------------------


def test_concatenate_inserts_the_requested_gaps(tmp_path: Path) -> None:
    first = write_wav(tmp_path / "a.wav", 1.0)
    second = write_wav(tmp_path / "b.wav", 1.0)
    output = tmp_path / "joined.wav"

    seconds = runpod_read.concatenate([(first, 500), (second, 700)], output)

    assert seconds == pytest.approx(2.5, abs=0.001)
    with wave.open(str(output), "rb") as handle:
        assert (handle.getnchannels(), handle.getsampwidth()) == (1, 2)
        assert handle.getframerate() == runpod_read.SAMPLE_RATE


def test_concatenate_drops_the_trailing_gap(tmp_path: Path) -> None:
    only = write_wav(tmp_path / "a.wav", 0.5)

    seconds = runpod_read.concatenate([(only, 700)], tmp_path / "one.wav")

    assert seconds == pytest.approx(0.5, abs=0.001)


def test_read_pcm_rejects_a_mismatched_format(tmp_path: Path) -> None:
    path = tmp_path / "wrong.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(48_000)
        handle.writeframes(b"\x00\x00\x00\x00")

    with pytest.raises(SystemExit, match="expected"):
        runpod_read.read_pcm(path)


def test_plan_gaps_widens_at_paragraph_ends(tmp_path: Path) -> None:
    records = [
        {"ends_paragraph": False, "parts": [{"wav": "a.wav"}]},
        {"ends_paragraph": True, "parts": [{"wav": "b.wav"}, {"wav": "c.wav"}]},
    ]

    planned = runpod_read.plan_gaps(
        records, tmp_path, sentence_gap_ms=350, paragraph_gap_ms=700
    )

    assert [gap for _, gap in planned] == [350, 350, 700]
    assert [path.name for path, _ in planned] == ["a.wav", "b.wav", "c.wav"]


# --- Synthesis and resume -------------------------------------------------


class StubEndpoint:
    """Records every submitted text and answers with a fixed-length clip."""

    def __init__(self, audio: bytes, truncate: set[str] | None = None) -> None:
        self.audio = audio
        self.truncate = truncate or set()
        self.texts: list[str] = []

    def submit(self, endpoint_id, api_key, payload, **kwargs):
        text = payload["input"]["text"]
        self.texts.append(text)
        return {
            "status": "COMPLETED",
            "executionTime": 2000,
            "output": {
                "audio_b64": base64.b64encode(self.audio).decode("ascii"),
                "sample_rate": runpod_read.SAMPLE_RATE,
                "audio_seconds": 1.0,
                "truncated": text in self.truncate,
            },
        }


def test_clone_part_writes_audio_and_reports_truncation(tmp_path, monkeypatch) -> None:
    stub = StubEndpoint(wav_bytes(1.0, tmp_path), truncate={"Too long."})
    monkeypatch.setattr(runpod_read.runpod_clone, "submit", stub.submit)
    part = tmp_path / "work" / "chunk_0000_p00.wav"

    report = runpod_read.clone_part("Too long.", part, options(), "key")

    assert part.is_file()
    assert report["truncated"] is True
    assert report["executionTime"] == pytest.approx(2.0)
    assert report["sha"] == runpod_read.text_sha("Too long.")


def test_synthesise_chunk_resplits_on_truncation(tmp_path, monkeypatch) -> None:
    whole = "One two three. Four five six."
    stub = StubEndpoint(wav_bytes(1.0, tmp_path), truncate={whole})
    monkeypatch.setattr(runpod_read.runpod_clone, "submit", stub.submit)
    chunk = runpod_read.Chunk(0, whole, True)

    record = runpod_read.synthesise_chunk(chunk, tmp_path / "work", options(), "key")

    assert stub.texts == [whole, "One two three.", "Four five six."]
    assert record["resplits"] == 1
    assert record["truncated"] is False
    assert [part["text"] for part in record["parts"]] == [
        "One two three.",
        "Four five six.",
    ]


def test_synthesise_chunk_gives_up_after_the_retry_limit(tmp_path, monkeypatch) -> None:
    stub = StubEndpoint(wav_bytes(1.0, tmp_path))
    stub.truncate = {"word"}
    monkeypatch.setattr(runpod_read.runpod_clone, "submit", stub.submit)

    with pytest.raises(SystemExit, match="still truncated"):
        runpod_read.synthesise_chunk(
            runpod_read.Chunk(0, "word", True), tmp_path / "w", options(), "key"
        )


def test_synthesise_resumes_from_manifest(tmp_path, monkeypatch) -> None:
    stub = StubEndpoint(wav_bytes(1.0, tmp_path))
    monkeypatch.setattr(runpod_read.runpod_clone, "submit", stub.submit)
    work = tmp_path / "work"
    chunks = [
        runpod_read.Chunk(0, "First chunk.", True),
        runpod_read.Chunk(1, "Second chunk.", True),
    ]

    runpod_read.synthesise(chunks, work, options(), "key")
    assert stub.texts == ["First chunk.", "Second chunk."]

    stub.texts.clear()
    records = runpod_read.synthesise(chunks, work, options(), "key")

    assert stub.texts == []
    assert [record["index"] for record in records] == [0, 1]


def test_synthesise_redoes_a_chunk_whose_text_changed(tmp_path, monkeypatch) -> None:
    stub = StubEndpoint(wav_bytes(1.0, tmp_path))
    monkeypatch.setattr(runpod_read.runpod_clone, "submit", stub.submit)
    work = tmp_path / "work"
    runpod_read.synthesise(
        [runpod_read.Chunk(0, "Original.", True)], work, options(), "k"
    )

    stub.texts.clear()
    runpod_read.synthesise(
        [runpod_read.Chunk(0, "Rewritten.", True)], work, options(), "k"
    )

    assert stub.texts == ["Rewritten."]


def test_synthesise_redoes_a_chunk_whose_wav_vanished(tmp_path, monkeypatch) -> None:
    stub = StubEndpoint(wav_bytes(1.0, tmp_path))
    monkeypatch.setattr(runpod_read.runpod_clone, "submit", stub.submit)
    work = tmp_path / "work"
    chunks = [runpod_read.Chunk(0, "Only chunk.", True)]
    runpod_read.synthesise(chunks, work, options(), "k")
    (work / "chunk_0000_p00.wav").unlink()

    stub.texts.clear()
    runpod_read.synthesise(chunks, work, options(), "k")

    assert stub.texts == ["Only chunk."]


def test_synthesise_redoes_a_chunk_whose_wav_is_truncated(
    tmp_path, monkeypatch
) -> None:
    """A header-only WAV marked done in the manifest must not pass resume.

    The regression: `_is_reusable` checked only that the file existed, so an
    interrupted synthesis left a chunk that resume skipped and `concatenate`
    contributed nothing to — a reading silently short by one chunk, exit 0.
    """
    stub = StubEndpoint(wav_bytes(1.0, tmp_path))
    monkeypatch.setattr(runpod_read.runpod_clone, "submit", stub.submit)
    work = tmp_path / "work"
    chunks = [runpod_read.Chunk(0, "Only chunk.", True)]
    runpod_read.synthesise(chunks, work, options(), "k")

    part = work / "chunk_0000_p00.wav"
    write_wav(part, 0.0)  # header intact, payload gone
    manifest = json.loads((work / "manifest.json").read_text())
    assert manifest["chunks"][0]["parts"][0]["duration"] == 1.0

    stub.texts.clear()
    records = runpod_read.synthesise(chunks, work, options(), "k")

    assert stub.texts == ["Only chunk."], "the truncated chunk was not re-synthesised"
    planned = runpod_read.plan_gaps(
        records, work, sentence_gap_ms=350, paragraph_gap_ms=700
    )
    seconds = runpod_read.concatenate(planned, tmp_path / "out.wav")
    assert seconds > 0.0, "the reading concatenated to silence"


def test_part_is_intact_rejects_a_header_only_wav(tmp_path) -> None:
    path = write_wav(tmp_path / "empty.wav", 0.0)
    assert runpod_read._part_is_intact(path, 1.0) is False


def test_part_is_intact_rejects_a_duration_mismatch(tmp_path) -> None:
    path = write_wav(tmp_path / "short.wav", 0.5)
    assert runpod_read._part_is_intact(path, 1.0) is False
    assert runpod_read._part_is_intact(path, 0.5) is True


def test_part_is_intact_accepts_a_record_without_a_duration(tmp_path) -> None:
    path = write_wav(tmp_path / "part.wav", 0.5)
    assert runpod_read._part_is_intact(path, None) is True


def test_part_is_intact_rejects_a_missing_file(tmp_path) -> None:
    assert runpod_read._part_is_intact(tmp_path / "absent.wav", 1.0) is False


def test_read_pcm_rejects_a_truncated_payload(tmp_path) -> None:
    """A WAV whose data chunk is shorter than its header claims must not read."""
    path = write_wav(tmp_path / "cut.wav", 1.0)
    whole = path.read_bytes()
    path.write_bytes(whole[: len(whole) - 8000])

    with pytest.raises(SystemExit):
        runpod_read.read_pcm(path)


def test_manifest_records_the_required_fields(tmp_path, monkeypatch) -> None:
    stub = StubEndpoint(wav_bytes(1.0, tmp_path))
    monkeypatch.setattr(runpod_read.runpod_clone, "submit", stub.submit)
    work = tmp_path / "work"

    runpod_read.synthesise(
        [runpod_read.Chunk(0, "A chunk.", True)], work, options(), "k"
    )

    record = json.loads((work / "manifest.json").read_text())["chunks"][0]
    assert set(record) >= {
        "index",
        "text",
        "sha",
        "duration",
        "executionTime",
        "truncated",
    }
    assert record["text"] == "A chunk."
    assert record["duration"] == pytest.approx(1.0)


# --- Pronunciation lexicon and section dropping ---------------------------


def read_options(tmp_path: Path, document: str, **overrides) -> argparse.Namespace:
    """A parsed reader command line over a one-file input."""
    source = tmp_path / "input.md"
    source.write_text(document, encoding="utf-8")
    argv = [
        "--endpoint-id",
        "e1",
        "--voice",
        "bob",
        "--input",
        str(source),
        "--output",
        str(tmp_path / "out.wav"),
    ]
    for flag, value in overrides.items():
        name = "--" + flag.replace("_", "-")
        argv.append(name)
        if value is not True:
            argv.append(str(value))
    return runpod_read.build_parser().parse_args(argv)


def test_resolve_lexicon_reads_the_shipped_file_by_default(tmp_path: Path) -> None:
    options = read_options(tmp_path, "Text.\n")

    assert runpod_read.resolve_lexicon(options)["Matsuoka"] == "Mah-tsu-oh-ka"


def test_resolve_lexicon_is_empty_when_disabled(tmp_path: Path) -> None:
    options = read_options(tmp_path, "Text.\n", no_pronunciations=True)

    assert runpod_read.resolve_lexicon(options) == {}


def test_speech_for_applies_the_lexicon_after_markdown(tmp_path: Path) -> None:
    """The link text survives the Markdown pass and is then respelled."""
    options = read_options(tmp_path, "See [Matsuoka](https://example.com).\n")

    speech = runpod_read.speech_for(options, runpod_read.resolve_lexicon(options))

    assert speech == "See Mah-tsu-oh-ka."


def test_speech_for_leaves_the_word_alone_with_no_pronunciations(
    tmp_path: Path,
) -> None:
    options = read_options(tmp_path, "I'm Bob Matsuoka.\n", no_pronunciations=True)

    speech = runpod_read.speech_for(options, runpod_read.resolve_lexicon(options))

    assert speech == "I'm Bob Matsuoka."


def test_speech_for_drops_a_named_section(tmp_path: Path) -> None:
    document = "## Body\n\nSpoken.\n\n**Related reading:**\n- [A](https://a.example)\n"
    options = read_options(tmp_path, document, drop_section="Related reading")

    speech = runpod_read.speech_for(options, {})

    assert speech == "Body\n\nSpoken."


def test_save_manifest_records_the_reading_and_lexicon(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    records = {"0": {"index": 0}, "1": {"index": 1}}

    runpod_read.save_manifest(path, records, lexicon="abc123", reading=[0])

    body = json.loads(path.read_text())
    assert body["lexicon"] == "abc123"
    assert body["reading"] == [0]
    assert [record["index"] for record in body["chunks"]] == [0, 1]


def test_synthesise_records_the_lexicon_sha(tmp_path, monkeypatch) -> None:
    stub = StubEndpoint(wav_bytes(1.0, tmp_path))
    monkeypatch.setattr(runpod_read.runpod_clone, "submit", stub.submit)
    work = tmp_path / "work"

    runpod_read.synthesise(
        [runpod_read.Chunk(0, "A chunk.", True)], work, options(), "k", lexicon="sha1"
    )

    assert runpod_read.manifest_lexicon(work / "manifest.json") == "sha1"


def test_manifest_lexicon_is_none_for_a_manifest_without_one(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"chunks": []}))

    assert runpod_read.manifest_lexicon(path) is None


def test_synthesise_redoes_only_the_chunks_a_lexicon_change_touched(
    tmp_path, monkeypatch, capsys
) -> None:
    """A new respelling must not re-buy audio for chunks that never said the word.

    Respelling happens before chunking, so a lexicon change reaches the manifest
    as a changed chunk sha. Only the chunk whose text moved is re-synthesised;
    the rest stay cached, which is the difference between a few seconds of
    endpoint time and a whole reading of it.
    """
    stub = StubEndpoint(wav_bytes(1.0, tmp_path))
    monkeypatch.setattr(runpod_read.runpod_clone, "submit", stub.submit)
    work = tmp_path / "work"
    before = [
        runpod_read.Chunk(0, "No surname here.", True),
        runpod_read.Chunk(1, "I'm Bob Matsuoka.", True),
    ]
    runpod_read.synthesise(before, work, options(), "k", lexicon="lex-one")
    assert stub.texts == ["No surname here.", "I'm Bob Matsuoka."]

    stub.texts.clear()
    after = [
        runpod_read.Chunk(0, "No surname here.", True),
        runpod_read.Chunk(1, "I'm Bob Mah-tsu-oh-ka.", True),
    ]
    runpod_read.synthesise(after, work, options(), "k", lexicon="lex-two")

    assert stub.texts == ["I'm Bob Mah-tsu-oh-ka."]
    assert runpod_read.manifest_lexicon(work / "manifest.json") == "lex-two"
    assert "lexicon changed (lex-one -> lex-two)" in capsys.readouterr().err


def test_synthesise_reuses_everything_when_the_lexicon_is_unchanged(
    tmp_path, monkeypatch
) -> None:
    stub = StubEndpoint(wav_bytes(1.0, tmp_path))
    monkeypatch.setattr(runpod_read.runpod_clone, "submit", stub.submit)
    work = tmp_path / "work"
    chunks = [runpod_read.Chunk(0, "I'm Bob Mah-tsu-oh-ka.", True)]
    runpod_read.synthesise(chunks, work, options(), "k", lexicon="lex-one")

    stub.texts.clear()
    runpod_read.synthesise(chunks, work, options(), "k", lexicon="lex-one")

    assert stub.texts == []


# --- Summary --------------------------------------------------------------


def test_manifest_gaps_is_none_for_a_manifest_without_one(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"chunks": []}))

    assert runpod_read.manifest_gaps(path) is None


def test_save_manifest_records_the_gap_plan(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    plan = runpod_read.GapPlan(sentence_ms=600, paragraph_ms=1200, jitter=0.25, seed=42)

    runpod_read.save_manifest(path, {"0": {"index": 0}}, gaps=plan.as_dict())

    assert runpod_read.manifest_gaps(path) == plan.as_dict()


# --- Seeds and request parameters -----------------------------------------


def test_part_seed_is_the_base_seed_when_fixed() -> None:
    assert runpod_read.part_seed(42, 0, 0, "fixed") == 42
    assert runpod_read.part_seed(42, 7, 1, "fixed") == 42


def test_part_seed_varies_by_chunk() -> None:
    seeds = [runpod_read.part_seed(42, index, 0, "vary") for index in range(10)]

    assert len(set(seeds)) == 10
    assert all(0 <= seed <= 0x7FFFFFFF for seed in seeds)


def test_part_seed_is_reproducible() -> None:
    assert runpod_read.part_seed(42, 3, 0, "vary") == runpod_read.part_seed(
        42, 3, 0, "vary"
    )
    assert runpod_read.part_seed(43, 3, 0, "vary") != runpod_read.part_seed(
        42, 3, 0, "vary"
    )
    assert runpod_read.part_seed(42, 3, 1, "vary") != runpod_read.part_seed(
        42, 3, 0, "vary"
    )


def test_part_seed_rejects_an_unknown_mode() -> None:
    with pytest.raises(SystemExit, match="--seed-mode"):
        runpod_read.part_seed(42, 0, 0, "random")


def test_params_sha_ignores_key_order() -> None:
    first = runpod_read.params_sha({"seed": 1, "cfg_scale": 1.0})
    second = runpod_read.params_sha({"cfg_scale": 1.0, "seed": 1})

    assert first == second
    assert first != runpod_read.params_sha({"seed": 2, "cfg_scale": 1.0})


def test_clone_part_sends_the_voice_direction_fields(tmp_path, monkeypatch) -> None:
    sent: list[dict] = []

    def submit(endpoint_id, api_key, payload, **kwargs):
        sent.append(payload["input"])
        return {
            "status": "COMPLETED",
            "executionTime": 1000,
            "output": {
                "audio_b64": base64.b64encode(wav_bytes(1.0, tmp_path)).decode("ascii"),
                "audio_seconds": 1.0,
                "truncated": False,
            },
        }

    monkeypatch.setattr(runpod_read.runpod_clone, "submit", submit)
    chosen = options(
        instruction="Speak conversationally.", cfg_scale=4.0, temperature=1.0
    )

    report = runpod_read.clone_part("Hello.", tmp_path / "p.wav", chosen, "k", seed=99)

    assert sent[0]["instruction"] == "Speak conversationally."
    assert sent[0]["cfg_scale"] == 4.0
    assert sent[0]["temperature"] == 1.0
    assert sent[0]["seed"] == 99
    assert "top_p" not in sent[0], "an unset override must not be sent"
    assert report["seed"] == 99


def test_synthesise_records_a_different_seed_per_chunk(tmp_path, monkeypatch) -> None:
    """Two consecutive chunks must not share a seed under --seed-mode vary."""
    stub = StubEndpoint(wav_bytes(1.0, tmp_path))
    monkeypatch.setattr(runpod_read.runpod_clone, "submit", stub.submit)
    work = tmp_path / "work"
    chunks = [
        runpod_read.Chunk(0, "First chunk.", True),
        runpod_read.Chunk(1, "Second chunk.", True),
    ]

    records = runpod_read.synthesise(chunks, work, options(seed_mode="vary"), "k")

    seeds = [record["seed"] for record in records]
    assert seeds[0] != seeds[1]
    assert seeds == [
        runpod_read.part_seed(42, 0, 0, "vary"),
        runpod_read.part_seed(42, 1, 0, "vary"),
    ]
    written = json.loads((work / "manifest.json").read_text())["chunks"]
    assert [record["seed"] for record in written] == seeds


def test_synthesise_records_the_same_seeds_on_a_re_run(tmp_path, monkeypatch) -> None:
    stub = StubEndpoint(wav_bytes(1.0, tmp_path))
    monkeypatch.setattr(runpod_read.runpod_clone, "submit", stub.submit)
    chunks = [
        runpod_read.Chunk(0, "First chunk.", True),
        runpod_read.Chunk(1, "Second chunk.", True),
    ]

    first = runpod_read.synthesise(
        chunks, tmp_path / "a", options(seed_mode="vary"), "k"
    )
    second = runpod_read.synthesise(
        chunks, tmp_path / "b", options(seed_mode="vary"), "k"
    )

    assert [record["seed"] for record in first] == [record["seed"] for record in second]


def test_synthesise_redoes_a_chunk_when_the_seed_mode_changes(
    tmp_path, monkeypatch
) -> None:
    stub = StubEndpoint(wav_bytes(1.0, tmp_path))
    monkeypatch.setattr(runpod_read.runpod_clone, "submit", stub.submit)
    work = tmp_path / "work"
    chunks = [runpod_read.Chunk(0, "Only chunk.", True)]
    runpod_read.synthesise(chunks, work, options(seed_mode="fixed"), "k")

    stub.texts.clear()
    runpod_read.synthesise(chunks, work, options(seed_mode="vary"), "k")

    assert stub.texts == ["Only chunk."], "a seed-mode change reused stale audio"


def test_synthesise_redoes_a_chunk_when_an_instruction_arrives(
    tmp_path, monkeypatch
) -> None:
    stub = StubEndpoint(wav_bytes(1.0, tmp_path))
    monkeypatch.setattr(runpod_read.runpod_clone, "submit", stub.submit)
    work = tmp_path / "work"
    chunks = [runpod_read.Chunk(0, "Only chunk.", True)]
    runpod_read.synthesise(chunks, work, options(), "k")

    stub.texts.clear()
    runpod_read.synthesise(
        chunks, work, options(instruction="Slow down.", cfg_scale=4.0), "k"
    )

    assert stub.texts == ["Only chunk."]


def test_synthesise_resumes_a_manifest_written_before_params(
    tmp_path, monkeypatch
) -> None:
    """A work directory from an older run still resumes under --seed-mode fixed."""
    stub = StubEndpoint(wav_bytes(1.0, tmp_path))
    monkeypatch.setattr(runpod_read.runpod_clone, "submit", stub.submit)
    work = tmp_path / "work"
    chunks = [runpod_read.Chunk(0, "Only chunk.", True)]
    runpod_read.synthesise(chunks, work, options(), "k")

    manifest = work / "manifest.json"
    body = json.loads(manifest.read_text())
    for record in body["chunks"]:
        record.pop("params")
        record.pop("seed")
        for part in record["parts"]:
            part.pop("seed")
    manifest.write_text(json.dumps(body))

    stub.texts.clear()
    runpod_read.synthesise(chunks, work, options(), "k")

    assert stub.texts == [], "an older manifest was thrown away"


# --- Gaps through a whole run ---------------------------------------------

TEN_CHUNKS = "\n\n".join(
    f"Alpha bravo charlie delta {word}. Echo foxtrot golf hotel {word}."
    for word in ("one", "two", "three", "four", "five")
)


def run_options(tmp_path: Path, **overrides) -> argparse.Namespace:
    chosen = read_options(tmp_path, TEN_CHUNKS + "\n", word_budget=5, **overrides)
    chosen.work_dir = tmp_path / "work"
    return chosen


def recorded_gaps(work: Path) -> list[int]:
    body = json.loads((work / "manifest.json").read_text())
    return [part["gap_ms"] for record in body["chunks"] for part in record["parts"]]


def test_run_jitters_the_recorded_gaps(tmp_path, monkeypatch) -> None:
    stub = StubEndpoint(wav_bytes(1.0, tmp_path))
    monkeypatch.setattr(runpod_read.runpod_clone, "submit", stub.submit)
    monkeypatch.setattr(runpod_read.runpod_clone, "load_api_key", lambda path: "k")
    chosen = run_options(tmp_path, gap_jitter=0.25)

    assert runpod_read.run(chosen) == 0

    gaps = recorded_gaps(chosen.work_dir)
    assert len(gaps) == 10
    assert len(set(gaps)) > 1, "every join in the reading got the same silence"


def test_run_uses_exactly_the_means_at_zero_jitter(tmp_path, monkeypatch) -> None:
    stub = StubEndpoint(wav_bytes(1.0, tmp_path))
    monkeypatch.setattr(runpod_read.runpod_clone, "submit", stub.submit)
    monkeypatch.setattr(runpod_read.runpod_clone, "load_api_key", lambda path: "k")
    chosen = run_options(tmp_path, gap_jitter=0)

    assert runpod_read.run(chosen) == 0

    gaps = recorded_gaps(chosen.work_dir)
    assert len(gaps) == 10
    assert set(gaps) == {chosen.sentence_gap_ms, chosen.paragraph_gap_ms}


def test_run_replays_recorded_gaps_on_a_resumed_run(tmp_path, monkeypatch) -> None:
    """A resumed run joins exactly as the first one did, without redrawing."""
    stub = StubEndpoint(wav_bytes(1.0, tmp_path))
    monkeypatch.setattr(runpod_read.runpod_clone, "submit", stub.submit)
    monkeypatch.setattr(runpod_read.runpod_clone, "load_api_key", lambda path: "k")
    chosen = run_options(tmp_path, gap_jitter=0.25)
    runpod_read.run(chosen)
    first = recorded_gaps(chosen.work_dir)

    # Pin the recorded gaps to values no draw would produce, then resume.
    body = json.loads((chosen.work_dir / "manifest.json").read_text())
    for record in body["chunks"]:
        for part in record["parts"]:
            part["gap_ms"] = 1234
    (chosen.work_dir / "manifest.json").write_text(json.dumps(body))

    stub.texts.clear()
    runpod_read.run(chosen)

    assert stub.texts == [], "a resumed run re-synthesised cached chunks"
    assert recorded_gaps(chosen.work_dir) == [1234] * len(first)


def test_run_redraws_gaps_when_the_plan_changes(tmp_path, monkeypatch) -> None:
    stub = StubEndpoint(wav_bytes(1.0, tmp_path))
    monkeypatch.setattr(runpod_read.runpod_clone, "submit", stub.submit)
    monkeypatch.setattr(runpod_read.runpod_clone, "load_api_key", lambda path: "k")
    chosen = run_options(tmp_path, gap_jitter=0.25)
    runpod_read.run(chosen)

    chosen.gap_jitter = 0.0
    runpod_read.run(chosen)

    assert set(recorded_gaps(chosen.work_dir)) == {
        chosen.sentence_gap_ms,
        chosen.paragraph_gap_ms,
    }


def test_gap_plan_for_reads_the_command_line(tmp_path: Path) -> None:
    chosen = read_options(tmp_path, "Text.\n", gap_jitter=0.1, seed=7)

    plan = runpod_read.gap_plan_for(chosen)

    assert plan.jitter == 0.1
    assert plan.seed == 7
    assert plan.sentence_ms == runpod_read.DEFAULT_SENTENCE_GAP_MS
    assert plan.paragraph_ms == runpod_read.DEFAULT_PARAGRAPH_GAP_MS


def test_gap_plan_for_rejects_an_impossible_jitter(tmp_path: Path) -> None:
    chosen = read_options(tmp_path, "Text.\n", gap_jitter=1.5)

    with pytest.raises(SystemExit, match="--gap-jitter"):
        runpod_read.gap_plan_for(chosen)


def test_the_gap_flags_have_a_second_spelling(tmp_path: Path) -> None:
    chosen = read_options(tmp_path, "Text.\n", gap_sentence_ms=500)

    assert chosen.sentence_gap_ms == 500


# --- Disfluencies through the reader --------------------------------------


def test_speech_for_injects_no_disfluencies_by_default(tmp_path: Path) -> None:
    chosen = read_options(tmp_path, "The client joins the chunks with silence.\n")

    assert chosen.disfluency_rate == 0.0
    assert (
        runpod_read.speech_for(chosen, {})
        == "The client joins the chunks with silence."
    )


def test_speech_for_injects_disfluencies_at_the_requested_rate(
    tmp_path: Path,
) -> None:
    document = " ".join(
        f"The client joins the finished chunks with a short silence number {word}."
        for word in ("one", "two", "three", "four", "five")
    )
    chosen = read_options(tmp_path, document + "\n", disfluency_rate=20)

    spoken = runpod_read.speech_for(chosen, {})

    assert spoken != document
    assert "Um," in spoken or "So," in spoken or "You know," in spoken


def test_report_totals_execution_and_cost(capsys) -> None:
    records = [
        {"index": 0, "executionTime": 10.0, "resplits": 0},
        {"index": 1, "executionTime": 5.0, "resplits": 1},
    ]

    runpod_read.report(records, 42.5, options(rate_per_second=0.001))

    printed = capsys.readouterr().out
    assert "chunks              : 2" in printed
    assert "total audio seconds : 42.50" in printed
    assert "total execute secs  : 15.00" in printed
    assert "estimated cost      : $0.0150" in printed
    assert "re-split chunks     : [1]" in printed
