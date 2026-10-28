"""Shared concurrency helpers.

Two kinds of work:
- I/O and native code that releases the GIL (cv2 decode/encode, file reads): ``thread_map``.
- Pure-Python CPU work over independent items (sequences, tracks): ``process_map``.

``ANIMCV_WORKERS`` overrides the worker count (``1`` forces serial execution everywhere).
Both helpers preserve input order and fall back to a plain loop for tiny inputs or one worker.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def worker_count(kind: str = "cpu") -> int:
    env = os.environ.get("ANIMCV_WORKERS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    cpus = os.cpu_count() or 1
    return min(32, cpus * 2) if kind == "io" else max(1, cpus - 1)


def thread_map(fn: Callable[[T], R], items: Iterable[T], workers: int | None = None) -> list[R]:
    items = list(items)
    n = workers if workers is not None else worker_count("io")
    if n <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=min(n, len(items))) as pool:
        return list(pool.map(fn, items))


def process_map(fn: Callable[[T], R], items: Iterable[T], workers: int | None = None, min_items: int = 2,
                chunksize: int = 1) -> list[R]:
    """``fn`` must be a picklable top-level function; results are returned in input order."""
    items = list(items)
    n = workers if workers is not None else worker_count("cpu")
    if n <= 1 or len(items) < min_items:
        return [fn(x) for x in items]
    with ProcessPoolExecutor(max_workers=min(n, len(items))) as pool:
        return list(pool.map(fn, items, chunksize=chunksize))
