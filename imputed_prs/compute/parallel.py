"""Process-level fan-out for the streaming fit / reference-CV (Phase 7).

Chromosome shards are **zero-halo** — windows and region merges are strictly
within-chromosome and a fresh band buffer is built per chromosome — so a
chromosome's models are bit-identical regardless of sharding. This module maps a
fitter's per-chromosome method (``_run_one_chromosome``) across a process pool and
returns the per-chromosome partials; the caller reduces them in a canonical order
(``StreamingFitResult.reduce`` / ``reduce_cv_collectors``), so the merged artifact
is reproducible regardless of worker scheduling.

Process fan-out is **CPU-only** (the GPU backend already parallelizes the intra-pass
GEMMs; one process per GPU is enough), so a non-CPU device resolves to a single
in-process worker. The default ``n_workers=1`` degrades to a serial, BLAS-unpinned
in-process map — bit-identical to (and as fast as) the pre-Phase-7 single loop.

The pool is joblib's **loky** backend (vendored inside ``joblib`` — no new top-level
dependency): it auto-memmaps large read-only numpy args, so an in-RAM
``InMemoryGenotypeSource`` matrix (reference CV / sensitivity) is *shared* across
workers rather than pickle-copied per task, and ``inner_max_num_threads=1`` pins
BLAS/OMP to one thread in every worker — the oversubscription guard (P workers ×
multithreaded BLAS would thrash the cores).
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence

_PERF_CORES: Optional[int] = None


def _performance_cores() -> int:
    """Physical performance-core count (Apple Silicon), else the logical CPU count.

    On Darwin the P-cores live at ``hw.perflevel0.physicalcpu`` (E-cores are
    ``perflevel1``); sizing a GEMM-heavy pool to the P-cores avoids the slower
    efficiency cores. Cached after the first query.
    """
    global _PERF_CORES
    if _PERF_CORES is not None:
        return _PERF_CORES
    cores = os.cpu_count() or 1
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "hw.perflevel0.physicalcpu"],
                capture_output=True,
                text=True,
                timeout=2.0,
                check=True,
            )
            val = int(out.stdout.strip())
            if val > 0:
                cores = val
        except (subprocess.SubprocessError, ValueError, OSError):
            pass
    _PERF_CORES = cores
    return cores


def resolve_n_workers(n_workers: Optional[int], *, device: str = "cpu") -> int:
    """Resolve a requested worker count to a concrete positive int.

    ``-1`` → performance cores; ``None``/``1`` → 1; otherwise clamp to the logical CPU
    count. Process fan-out is CPU-only, so any non-CPU ``device`` forces 1 (the GPU
    already parallelizes the intra-pass GEMMs).
    """
    if device not in (None, "cpu"):
        return 1
    if n_workers is None or n_workers == 1:
        return 1
    ncpu = os.cpu_count() or 1
    if n_workers < 0:
        return max(1, min(_performance_cores(), ncpu))
    return max(1, min(int(n_workers), ncpu))


@dataclass
class _ChromTask:
    """Picklable per-chromosome unit of work: run one chromosome, return its partial.

    ``fitter`` and ``source`` are picklable — the plan carries global platform indices,
    the already-built ``fitter.folds`` permutation is pickled (so every worker shares
    the identical fold assignment), path-based sources hold no live handle, and an
    in-RAM source's dosage matrix is memmapped read-only by loky. BLAS is *not* pinned
    here; the parallel path pins it via ``parallel_config(inner_max_num_threads=1)`` so
    the serial (``n_workers=1``) path stays unpinned and fast.
    """

    fitter: object
    source: object

    def __call__(self, chrom):
        return self.fitter._run_one_chromosome(self.source, chrom)


def fan_out_chromosomes(
    fitter,
    source,
    chroms: Sequence[str],
    *,
    n_workers: int = 1,
    device: str = "cpu",
) -> List[object]:
    """Map ``fitter._run_one_chromosome`` over ``chroms`` → list of per-chrom partials.

    ``n_workers <= 1`` (or a non-CPU ``device``) runs a serial in-process map, unpinned
    — preserving the fast, multi-threaded-BLAS default and bit-identical to the
    pre-Phase-7 single loop. With ``n_workers > 1`` each chromosome is one task on a
    loky process pool, BLAS pinned to one thread per worker. Return order is not relied
    upon — the caller reduces in canonical (``_chrom_sort_key``) order.
    """
    chroms = list(chroms)
    workers = min(resolve_n_workers(n_workers, device=device), max(1, len(chroms)))
    task = _ChromTask(fitter, source)
    if workers <= 1:
        return [task(c) for c in chroms]
    from joblib import Parallel, delayed, parallel_config

    with parallel_config(backend="loky", inner_max_num_threads=1):
        return Parallel(n_jobs=workers, return_as="list")(
            delayed(task)(c) for c in chroms
        )
