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
        "cfg_scale": 1.0,
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


def test_write_mp3_reports_missing_ffmpeg(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(runpod_read.shutil, "which", lambda name: None)

    assert runpod_read.write_mp3(tmp_path / "a.wav", tmp_path / "a.mp3") is False
    assert "ffmpeg not found" in capsys.readouterr().err


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


# --- Summary --------------------------------------------------------------


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
