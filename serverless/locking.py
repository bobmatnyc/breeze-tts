"""Interprocess locking and atomic directory publication on the network volume.

Why: a RunPod network volume is shared by every worker on the endpoint, and a
worker can be killed at any point. Both the checkpoint download and the voice
registry therefore need the same two guarantees — only one writer at a time, and
a directory that is either absent or complete, never half-written.

What: ``volume_lock`` is an ``flock`` on a lock file beside the payload;
``publish_directory`` builds into a staging directory and renames it into place,
which is atomic within one filesystem.

Test: ``tests/test_serverless_locking.py``
"""

from __future__ import annotations

import fcntl
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

# Written into a directory only after its contents are complete. Presence of the
# payload alone is not a readiness signal: an interrupted download leaves real
# files behind that look finished.
COMPLETE_MARKER = ".breeze-complete"


@contextmanager
def volume_lock(lock_path: Path, *, timeout: float = 1800.0) -> Iterator[None]:
    """Hold an exclusive interprocess lock, waiting for any current holder.

    Test: `test_volume_lock_serializes_two_processes`
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    handle = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out after {timeout}s waiting for {lock_path}."
                    ) from None
                time.sleep(0.25)
        yield
    finally:
        try:
            fcntl.flock(handle, fcntl.LOCK_UN)
        finally:
            os.close(handle)


def is_complete(target: Path) -> bool:
    """Whether ``target`` was published in full.

    Test: `test_is_complete_rejects_partial_directory`
    """
    return (target / COMPLETE_MARKER).is_file()


def publish_directory(target: Path, build: Callable[[Path], None]) -> Path:
    """Run ``build`` against a staging directory, then rename it onto ``target``.

    The staging directory sits beside ``target`` so the rename stays within one
    filesystem and is therefore atomic. A failed build leaves the staging
    directory removed and ``target`` untouched, so the next attempt retries from
    scratch rather than inheriting a partial result.

    Space: staging beside the target means both exist at once, which needs room
    for two copies. An *incomplete* target is unusable by definition, so it is
    deleted before staging starts rather than kept — the 7.2 GB checkpoint lives
    on a 10 GB volume, where holding two copies would fill the disk and make
    every retry fail the same way. A complete target is kept until the swap, so
    replacing one still needs the room.

    Test: `test_publish_directory_is_atomic`,
    `test_publish_directory_discards_failed_build`,
    `test_publish_directory_reclaims_an_incomplete_target`
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not is_complete(target):
        shutil.rmtree(target, ignore_errors=True)
    _clear_stale_staging(target)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=str(target.parent))
    )
    try:
        build(staging)
        (staging / COMPLETE_MARKER).write_text("", encoding="utf-8")
        if target.exists():
            doomed = staging.with_name(f".{target.name}.old-{os.getpid()}")
            target.rename(doomed)
            shutil.rmtree(doomed, ignore_errors=True)
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def _clear_stale_staging(target: Path) -> None:
    """Delete staging directories a previous killed attempt left behind.

    They are only ever reachable by this helper — nothing renames them into
    place after the process that made them dies — so on a small volume they are
    pure leaked space that would starve the next download.

    Test: `test_publish_directory_clears_stale_staging`
    """
    for leftover in target.parent.glob(f".{target.name}.staging-*"):
        shutil.rmtree(leftover, ignore_errors=True)
    for leftover in target.parent.glob(f".{target.name}.old-*"):
        shutil.rmtree(leftover, ignore_errors=True)


def write_file_atomically(path: Path, payload: bytes) -> None:
    """Replace ``path`` with ``payload`` in one step.

    Test: `test_write_file_atomically_replaces_content`
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
