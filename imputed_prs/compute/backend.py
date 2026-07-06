"""Compute-backend protocol for CPU/GPU dispatch of the sample-free fit kernel.

Phase 2 restructured training so that every variant/region model is solved from a
local Gram block over a shared, banded chip buffer (``compute/sufficient_stats.py``).
The two hot kernels are the *banded cross-product / OOF GEMMs* and the *batched local
Gram solve* — both are pure linear algebra that can run on CPU (numpy/scipy) or GPU
(torch on MPS/CUDA). :class:`ComputeBackend` is the seam: the streaming fitters dispatch
one chunk of co-windowed units through ``run_fit_batch`` and the backend decides where
the math runs.

The seam is at the *chunk* level (not per-op) so a GPU backend can transfer the shared
band to the device once and keep the GEMMs, the solve, and the out-of-fold reduction on
the device before returning the small per-unit coefficients to the host.

The CPU backend (:mod:`imputed_prs.compute.cpu_backend`) is the always-available default
and reproduces the Phase-2 result bit-for-bit; the GPU backend
(:mod:`imputed_prs.compute.gpu_backend`, lazy ``torch``) matches it within the sanctioned
statistical-parity band. See :func:`imputed_prs.compute.device.get_backend`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ComputeBackend(Protocol):
    """Protocol for a compute backend that runs the batched local-Gram fit kernel.

    Implementations must be interchangeable: for a given input the CPU backend defines
    the reference result and any GPU backend must agree within statistical-parity
    tolerance (float32 on MPS). All array inputs/outputs are host ``numpy`` objects —
    the backend owns any host↔device transfer internally.
    """

    @property
    def device_name(self) -> str:
        """Device identifier: ``"cpu"``, ``"mps"``, or ``"cuda"``."""
        ...

    def make_buffer(self, n_samples: int, folds, lazy_fold_gram: bool = False):
        """Construct the sliding band-Gram buffer this backend accumulates into.

        The CPU backend returns the host ``numpy`` ``_ChipGramBuffer``; a GPU backend
        returns a device-resident buffer that keeps the band ``Z`` and the incremental
        Gram on the device so the O(n) accumulation matmuls run there (Phase 3D). The
        streaming fitters call this once per chromosome and drive it with
        ``add``/``evict_below``/``clear``; ``run_fit_batch`` consumes the same object.

        ``lazy_fold_gram=True`` (projection) skips the incremental full + per-fold Gram
        and recomputes them on-demand at ``gather`` over each unit's ≤max_predictors fit
        predictors, so a chromosome-spanning merged region never allocates a
        (K, cap, cap) tensor (Finding-#1 band-limited per-fold Gram, Phase 3E).
        """
        ...

    def run_fit_batch(
        self,
        jobs,
        buf,
        folds,
        alpha: float,
        l1_ratio: float,
        cv_folds: int,
        s_true,
        s_cv,
        batch_cap: int,
    ) -> int:
        """Fit one group of co-located units from the shared banded buffer.

        Mirrors ``compute.sufficient_stats._run_fit_batch``: consumes the chunk's
        ``_FitJob`` list plus the shared ``_ChipGramBuffer`` ``buf`` and ``GlobalFolds``
        ``folds``, mutates the calibration accumulators ``s_true``/``s_cv`` in place, and
        returns the number of *calibrating* intercept-only models produced.
        """
        ...
