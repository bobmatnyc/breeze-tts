"""Named voice registry stored on the RunPod network volume.

Why: sending a multi-megabyte reference clip with every request is wasteful when
the same speaker is used repeatedly, and an OpenAI-compatible client has nowhere
to put one — it sends a voice *name*. Registering a voice once turns every later
request into a short JSON body.

What: each voice is a directory under ``<volume>/voices/<name>/`` holding
``reference.wav``, ``transcript.txt`` and ``meta.json``. Writes go through the
same lock and staging discipline the checkpoint download uses, so a killed
worker never leaves a half-registered voice behind.

Test: ``tests/test_voice_registry.py``
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from serverless.locking import is_complete, publish_directory, volume_lock
from serverless.wav import decode_wav, encode_wav

VOICES_DIR_NAME = "voices"
REFERENCE_NAME = "reference.wav"
TRANSCRIPT_NAME = "transcript.txt"
META_NAME = "meta.json"

# Names become path segments and appear in OpenAI-style request bodies, so the
# character set is deliberately narrow rather than merely path-safe.
NAME_PATTERN = re.compile(r"^[a-z0-9_-]{1,32}$")


class RegistryError(ValueError):
    """A voice-registry operation the worker rejects."""


@dataclass(frozen=True)
class Voice:
    """A registered reference clip and the transcript that goes with it."""

    name: str
    reference_path: Path
    ref_text: str
    meta: dict[str, object]


def validate_name(name: object) -> str:
    """Return ``name`` if it is a legal voice name.

    Test: `test_validate_name_accepts_legal_names`,
    `test_validate_name_rejects_illegal_names`
    """
    if not isinstance(name, str) or not NAME_PATTERN.match(name):
        raise RegistryError(
            "Voice name must be 1-32 characters of lowercase letters, digits, "
            f"underscore or hyphen; got {name!r}."
        )
    return name


class VoiceRegistry:
    """Voice storage rooted at a volume path.

    The root is injectable so tests can stand a temporary directory in for
    ``/runpod-volume``.
    """

    def __init__(self, volume_root: str | Path) -> None:
        self.root = Path(volume_root) / VOICES_DIR_NAME
        self._lock_path = Path(volume_root) / ".breeze-voices.lock"

    def _dir(self, name: str) -> Path:
        return self.root / name

    def list_voices(self) -> list[dict[str, object]]:
        """Every fully registered voice, oldest first.

        Partially written directories are skipped rather than reported, so a
        crashed registration never shows up as a usable voice.

        Test: `test_list_voices_skips_incomplete_entries`
        """
        if not self.root.is_dir():
            return []
        voices = []
        for entry in sorted(self.root.iterdir()):
            if not entry.is_dir() or not is_complete(entry):
                continue
            try:
                meta = json.loads((entry / META_NAME).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            voices.append(meta)
        voices.sort(key=lambda item: str(item.get("created_at", "")))
        return voices

    def get(self, name: object) -> Voice:
        """Load a registered voice, failing closed when it is unknown.

        Test: `test_get_reports_unknown_voice`, `test_register_then_get_round_trips`
        """
        validated = validate_name(name)
        directory = self._dir(validated)
        if not directory.is_dir() or not is_complete(directory):
            known = [str(v.get("name")) for v in self.list_voices()]
            available = ", ".join(known) if known else "none registered"
            raise RegistryError(
                f"Unknown voice {validated!r}. Registered voices: {available}."
            )
        return Voice(
            name=validated,
            reference_path=directory / REFERENCE_NAME,
            ref_text=(directory / TRANSCRIPT_NAME).read_text(encoding="utf-8"),
            meta=json.loads((directory / META_NAME).read_text(encoding="utf-8")),
        )

    def register(
        self,
        name: object,
        *,
        audio: np.ndarray,
        sample_rate: int,
        ref_text: str,
        overwrite: bool = False,
        source_filename: str | None = None,
    ) -> dict[str, object]:
        """Store a voice, replacing an existing one only when asked.

        Test: `test_register_then_get_round_trips`,
        `test_register_refuses_existing_without_overwrite`,
        `test_register_overwrites_when_asked`
        """
        validated = validate_name(name)
        transcript = ref_text.strip()
        if not transcript:
            raise RegistryError("A registered voice needs a non-empty 'ref_text'.")

        wav_bytes = encode_wav(audio, sample_rate)
        meta: dict[str, object] = {
            "name": validated,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_filename": source_filename,
            "sample_rate": int(sample_rate),
            "duration_seconds": round(float(np.asarray(audio).size) / sample_rate, 3),
            "sha256": hashlib.sha256(wav_bytes).hexdigest(),
            "bytes": len(wav_bytes),
        }

        with volume_lock(self._lock_path):
            directory = self._dir(validated)
            if directory.is_dir() and is_complete(directory) and not overwrite:
                raise RegistryError(
                    f"Voice {validated!r} already exists. Pass overwrite=true to replace it."
                )

            def build(staging: Path) -> None:
                (staging / REFERENCE_NAME).write_bytes(wav_bytes)
                (staging / TRANSCRIPT_NAME).write_text(transcript, encoding="utf-8")
                (staging / META_NAME).write_text(
                    json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )

            publish_directory(directory, build)
        return meta

    def delete(self, name: object) -> dict[str, object]:
        """Remove a registered voice.

        Test: `test_delete_removes_a_voice`, `test_delete_reports_unknown_voice`
        """
        import shutil

        validated = validate_name(name)
        with volume_lock(self._lock_path):
            directory = self._dir(validated)
            if not directory.is_dir():
                raise RegistryError(f"Unknown voice {validated!r}; nothing to delete.")
            shutil.rmtree(directory)
        return {"name": validated, "deleted": True}


def decode_reference(payload: bytes) -> tuple[np.ndarray, int]:
    """Decode an uploaded reference clip to mono float32 plus its sample rate."""
    return decode_wav(payload)
