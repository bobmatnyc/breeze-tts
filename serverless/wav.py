"""WAV encoding and decoding for the serverless worker.

Why: reference audio arrives base64 in a job body and generated audio leaves the
same way, while ``encode_prompt_audio`` (``breeze_infer/audio.py:13-22``) reads a
*path* with soundfile. Parsing here turns a malformed upload into a clear job
error instead of a stack trace inside the tokenizer, and keeps both directions
on the stdlib so they stay testable without libsndfile.
"""

from __future__ import annotations

import io
import wave

import numpy as np


class AudioError(ValueError):
    """Audio that cannot be represented as a 16-bit PCM WAV."""


def encode_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    """Serialize mono float32 PCM in [-1, 1] as a 16-bit WAV container.

    Preconditions: every sample is finite. NaN and Inf survive ``np.clip``
    unchanged and then reach an undefined int16 cast, which would ship silence
    or full-scale noise as if it were audio, so they are rejected instead.

    Test: `test_encode_wav_roundtrips_through_decode`,
    `test_encode_wav_rejects_non_finite_samples`
    """
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size and not np.isfinite(samples).all():
        raise AudioError(
            "Audio contains non-finite samples (NaN or Inf); refusing to encode."
        )
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2", copy=False)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def decode_wav(payload: bytes) -> tuple[np.ndarray, int]:
    """Read a WAV byte string as mono float32 PCM plus its sample rate.

    What: uses the stdlib ``wave`` reader for 8/16/32-bit PCM, and falls back to
    ``soundfile`` for anything else it cannot parse (float WAV, 24-bit, FLAC).

    Test: `test_decode_wav_reads_stereo_as_mono`, `test_decode_wav_rejects_garbage`
    """
    try:
        with wave.open(io.BytesIO(payload), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            frames = handle.readframes(handle.getnframes())
    except (wave.Error, EOFError) as exc:
        return _decode_via_soundfile(payload, exc)

    dtypes = {1: np.uint8, 2: "<i2", 4: "<i4"}
    if width not in dtypes:
        return _decode_via_soundfile(
            payload, ValueError(f"unsupported sample width {width}")
        )

    raw = np.frombuffer(frames, dtype=dtypes[width])
    if width == 1:
        samples = (raw.astype(np.float32) - 128.0) / 128.0
    else:
        samples = raw.astype(np.float32) / float(2 ** (8 * width - 1))

    if channels > 1:
        usable = (samples.size // channels) * channels
        samples = samples[:usable].reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(samples, dtype=np.float32), int(sample_rate)


def _decode_via_soundfile(payload: bytes, cause: Exception) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf
    except ImportError:
        raise AudioError(f"Reference audio is not a readable WAV: {cause}") from cause

    try:
        data, sample_rate = sf.read(
            io.BytesIO(payload), always_2d=True, dtype="float32"
        )
    except Exception as exc:
        raise AudioError(f"Reference audio is not readable audio: {exc}") from exc
    return (
        np.ascontiguousarray(np.mean(data, axis=1), dtype=np.float32),
        int(sample_rate),
    )
