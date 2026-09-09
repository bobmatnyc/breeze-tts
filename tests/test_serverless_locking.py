"""Tests for the network-volume lock and atomic directory publication."""

from __future__ import annotations

import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serverless.locking import (
    COMPLETE_MARKER,
    is_complete,
    publish_directory,
    volume_lock,
    write_file_atomically,
)


def test_is_complete_rejects_partial_directory(tmp_path) -> None:
    partial = tmp_path / "breeze-tts-2"
    (partial / "audio_tokenizer").mkdir(parents=True)
    (partial / "audio_tokenizer" / "config.json").write_text("{}")

    # Real files are present, but the download never finished.
    assert not is_complete(partial)

    (partial / COMPLETE_MARKER).write_text("")
    assert is_complete(partial)


def test_publish_directory_is_atomic(tmp_path) -> None:
    target = tmp_path / "payload"
    seen: list[Path] = []

    def build(staging: Path) -> None:
        seen.append(staging)
        # The target must not exist while the build is still running.
        assert not target.exists()
        (staging / "weights.bin").write_bytes(b"x" * 32)

    published = publish_directory(target, build)

    assert published == target
    assert is_complete(target)
    assert (target / "weights.bin").read_bytes() == b"x" * 32
    assert not seen[0].exists()


def test_publish_directory_discards_failed_build(tmp_path) -> None:
    target = tmp_path / "payload"

    def build(staging: Path) -> None:
        (staging / "half.bin").write_bytes(b"partial")
        raise RuntimeError("connection reset")

    with pytest.raises(RuntimeError, match="connection reset"):
        publish_directory(target, build)

    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_publish_directory_replaces_an_existing_target(tmp_path) -> None:
    target = tmp_path / "payload"
    publish_directory(target, lambda s: (s / "v.txt").write_text("first"))
    publish_directory(target, lambda s: (s / "v.txt").write_text("second"))

    assert (target / "v.txt").read_text() == "second"
    assert is_complete(target)


def test_write_file_atomically_replaces_content(tmp_path) -> None:
    path = tmp_path / "meta.json"
    write_file_atomically(path, b"one")
    write_file_atomically(path, b"two")

    assert path.read_bytes() == b"two"
    assert [p.name for p in tmp_path.iterdir()] == ["meta.json"]


def test_volume_lock_times_out_when_held_elsewhere(tmp_path) -> None:
    holder = _spawn_lock_holder(tmp_path / "busy.lock", hold_seconds=5.0)
    try:
        _wait_for_holder(tmp_path / "busy.ready")
        with (
            pytest.raises(TimeoutError, match="Timed out"),
            volume_lock(tmp_path / "busy.lock", timeout=0.5),
        ):
            pass
    finally:
        holder.kill()
        holder.wait(timeout=10)


def test_volume_lock_serializes_two_processes(tmp_path) -> None:
    """A lock held by another process blocks, then succeeds once released."""
    holder = _spawn_lock_holder(tmp_path / "busy.lock", hold_seconds=1.5)
    try:
        _wait_for_holder(tmp_path / "busy.ready")
        started = time.monotonic()
        with volume_lock(tmp_path / "busy.lock", timeout=30):
            waited = time.monotonic() - started
    finally:
        holder.wait(timeout=15)

    # It waited for the other process rather than proceeding concurrently.
    assert waited > 0.4
    assert holder.returncode == 0


def _spawn_lock_holder(lock_path: Path, *, hold_seconds: float) -> subprocess.Popen:
    script = textwrap.dedent(
        f"""
        import fcntl, os, time
        handle = os.open({str(lock_path)!r}, os.O_CREAT | os.O_RDWR, 0o644)
        fcntl.flock(handle, fcntl.LOCK_EX)
        open({str(lock_path.with_suffix(".ready"))!r}, "w").close()
        time.sleep({hold_seconds})
        fcntl.flock(handle, fcntl.LOCK_UN)
        os.close(handle)
        """
    )
    return subprocess.Popen([sys.executable, "-c", script])


def _wait_for_holder(ready_path: Path, *, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if ready_path.exists():
            return
        time.sleep(0.02)
    raise AssertionError(f"lock holder never signalled readiness at {ready_path}")
