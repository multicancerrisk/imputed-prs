"""In-process measurement utilities used *inside* the isolated child.

* :func:`peak_rss_bytes` — current process peak RSS (monotonic high-water mark).
* :func:`numpy_domain_is_live` — probe whether numpy allocations are captured by
  tracemalloc's numpy domain (they are on numpy 2.4.3; the analytical byte model is the
  fallback if a future wheel stops populating it).
* :func:`phase` / :class:`PhaseRegistry` — sub-phase wall timing (+ optional tracemalloc
  deltas) at the public-call boundary (load → fit → predict → export).
* :class:`TraceMallocAttribution` — whole-run peak + per-source-line attribution, walking
  each allocation's traceback out to the first library frame. Non-invasive: the library is
  never modified.
* :class:`RSSWatchdog` — soft-memory-ceiling daemon thread. Needed because Darwin rejects
  ``RLIMIT_AS`` (verified), so there is no hard rlimit backstop.
"""
from __future__ import annotations

import resource
import threading
import time
import tracemalloc
from contextlib import contextmanager
from typing import Callable, List, Optional, Sequence

from benchmarks.harness import SiteAttribution, SubphaseTiming, normalize_maxrss

#: Frames inside these are "internal" and skipped when picking an attribution frame.
_INTERNAL_HINTS = (
    "/numpy/",
    "site-packages/numpy",
    "tracemalloc.py",
    "<frozen",
    "/importlib/",
    "/benchmarks/meters.py",
)


def peak_rss_bytes() -> int:
    """Current process peak RSS in bytes (``ru_maxrss``, unit-normalized)."""
    return normalize_maxrss(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def numpy_domain_is_live() -> bool:
    """True iff numpy array data buffers are traced under numpy's tracemalloc domain."""
    try:
        import numpy as np
    except Exception:
        return False
    domain = getattr(np.lib, "tracemalloc_domain", None)
    if domain is None:
        return False
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start(4)
    try:
        filt = [tracemalloc.DomainFilter(True, domain)]
        before = sum(s.size for s in tracemalloc.take_snapshot().filter_traces(filt).statistics("filename"))
        arr = np.ones(2_000_000, dtype=np.float64)  # ~16 MiB
        after = sum(s.size for s in tracemalloc.take_snapshot().filter_traces(filt).statistics("filename"))
        del arr
        return (after - before) > 8_000_000
    finally:
        if not was_tracing:
            tracemalloc.stop()


class PhaseRegistry:
    """Thread-safe collector of :class:`SubphaseTiming` records."""

    def __init__(self) -> None:
        self._phases: List[SubphaseTiming] = []
        self._lock = threading.Lock()

    def record(self, timing: SubphaseTiming) -> None:
        with self._lock:
            self._phases.append(timing)

    def to_list(self) -> List[SubphaseTiming]:
        with self._lock:
            return list(self._phases)


@contextmanager
def phase(name: str, registry: PhaseRegistry, trace: bool = False):
    """Time a sub-phase (``time.perf_counter``); optionally record tracemalloc deltas.

    Deliberately does **not** call ``reset_peak`` — that would clobber the whole-run peak
    tracked by :class:`TraceMallocAttribution`. Per-phase memory is reported as the net
    live-bytes delta instead.
    """
    tracing = trace and tracemalloc.is_tracing()
    start_cur = tracemalloc.get_traced_memory()[0] if tracing else None
    t0 = time.perf_counter()
    try:
        yield
    finally:
        wall = time.perf_counter() - t0
        cur = delta = None
        if tracing:
            cur = tracemalloc.get_traced_memory()[0]
            delta = cur - (start_cur or 0)
        registry.record(
            SubphaseTiming(name=name, wall_seconds=wall, tm_current_bytes=cur, tm_delta_bytes=delta)
        )


class TraceMallocAttribution:
    """Context manager: whole-run tracemalloc peak + per-file:line attribution.

    On exit, ``peak_bytes`` is the true peak of traced (Python + numpy-domain) memory and
    ``sites`` maps the largest *surviving* allocations to their source line, preferring the
    first frame matching ``markers`` (e.g. ``imputed_prs``), else the deepest non-internal
    frame. Transient allocations freed before exit contribute to ``peak_bytes`` but not to
    ``sites`` (documented limitation).
    """

    def __init__(
        self,
        depth: int = 30,
        top_n: int = 15,
        markers: Sequence[str] = ("imputed_prs",),
    ) -> None:
        self.depth = depth
        self.top_n = top_n
        self.markers = tuple(markers)
        self._started = False
        self.peak_bytes: Optional[int] = None
        self.sites: List[SiteAttribution] = []

    def __enter__(self) -> "TraceMallocAttribution":
        if not tracemalloc.is_tracing():
            tracemalloc.start(self.depth)
            self._started = True
        tracemalloc.reset_peak()
        return self

    def __exit__(self, *exc) -> bool:
        self.peak_bytes = tracemalloc.get_traced_memory()[1]
        try:
            self.sites = self._attribute()
        except Exception:
            self.sites = []
        if self._started:
            tracemalloc.stop()
        return False

    def _numpy_domain(self):
        try:
            import numpy as np

            return getattr(np.lib, "tracemalloc_domain", None)
        except Exception:
            return None

    def _pick_frame(self, traceback):
        frames = list(traceback)  # oldest -> most recent
        if not frames:
            return None
        for fr in reversed(frames):  # most-recent (deepest) first
            if any(m in fr.filename for m in self.markers):
                return fr
        for fr in reversed(frames):
            if not any(h in fr.filename for h in _INTERNAL_HINTS):
                return fr
        return frames[-1]

    def _attribute(self) -> List[SiteAttribution]:
        snap = tracemalloc.take_snapshot()
        domain = self._numpy_domain()
        if domain is not None:
            snap = snap.filter_traces([tracemalloc.DomainFilter(True, domain)])
        agg: dict = {}
        for st in snap.statistics("traceback"):
            frame = self._pick_frame(st.traceback)
            if frame is None:
                continue
            key = (frame.filename, frame.lineno)
            entry = agg.setdefault(key, [0, 0])
            entry[0] += st.size
            entry[1] += st.count
        items = sorted(agg.items(), key=lambda kv: kv[1][0], reverse=True)[: self.top_n]
        return [
            SiteAttribution(filename=fn, lineno=ln, size_bytes=size, count=count)
            for (fn, ln), (size, count) in items
        ]


class RSSWatchdog(threading.Thread):
    """Daemon thread that polls peak RSS and fires ``on_breach`` once past ``ceiling``."""

    def __init__(
        self,
        ceiling_bytes: int,
        on_breach: Callable[[int], None],
        interval: float = 0.05,
    ) -> None:
        super().__init__(daemon=True, name="rss-watchdog")
        self.ceiling_bytes = int(ceiling_bytes)
        self.on_breach = on_breach
        self.interval = interval
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.is_set():
            current = peak_rss_bytes()
            if current >= self.ceiling_bytes:
                self.on_breach(current)  # expected to terminate the process
                return
            self._stop.wait(self.interval)
