"""Cross-process advisory file locks for shared, user-owned state."""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def try_lock(path: Path) -> Iterator[bool]:
    """Take the lock without waiting; yield True only when it is ours.

    A second extractor has nothing useful to do — the first is already covering
    the same work. Waiting 30 seconds and then raising killed the run and left
    nothing but a traceback in a log file nobody opens, which is how a backlog
    of hundreds of turns quietly stopped ever being processed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fh = open(path, "a+b")
    except OSError:
        yield False
        return
    acquired = False
    with fh:
        try:
            if os.name == "nt":
                import msvcrt

                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError:
            acquired = False
        try:
            yield acquired
        finally:
            if acquired:
                if os.name == "nt":
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


@contextmanager
def file_lock(path: Path, timeout: float = 30.0) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    # Windows can transiently deny open() while another process is creating the
    # same lock file; retry the open the same way we retry the lock itself.
    while True:
        try:
            fh = open(path, "a+b")
            break
        except OSError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"could not open lock file: {path.name}")
            time.sleep(0.05)
    with fh:
        if os.name == "nt":
            import msvcrt
        else:
            import fcntl
        while True:
            try:
                if os.name == "nt":
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"could not acquire lock: {path.name}")
                time.sleep(0.05)
        try:
            yield
        finally:
            if os.name == "nt":
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
