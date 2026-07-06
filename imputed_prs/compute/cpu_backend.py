"""CPU compute backend — the validated Phase-2 numpy/scipy kernel.

This backend is a thin dispatcher over ``compute.sufficient_stats._run_fit_batch`` (and,
through it, ``compute.gram_solve.fit_from_local_gram``). ``device="cpu"`` therefore
reproduces the Phase-2 result **bit-for-bit** and is the correctness oracle every GPU
backend is validated against. numpy/scipy only — no ``torch``.
"""

from __future__ import annotations


class CpuBackend:
    """Runs the batched local-Gram fit kernel on the CPU (numpy/scipy)."""

    @property
    def device_name(self) -> str:
        return "cpu"

    def make_buffer(self, n_samples, folds, lazy_fold_gram: bool = False):
        # The validated Phase-2 host buffer — accumulation stays in numpy float64.
        # ``lazy_fold_gram`` (projection) recomputes the per-fold Gram on-demand so a
        # chromosome-spanning region never allocates a (K, cap, cap) tensor.
        from imputed_prs.compute.sufficient_stats import _ChipGramBuffer

        return _ChipGramBuffer(n_samples, folds, lazy_fold_gram=lazy_fold_gram)

    def run_fit_batch(
        self, jobs, buf, folds, alpha, l1_ratio, cv_folds, s_true, s_cv, batch_cap
    ) -> int:
        # Lazy import breaks the sufficient_stats -> device -> cpu_backend cycle: by the
        # time a fit runs, sufficient_stats is fully imported.
        from imputed_prs.compute.sufficient_stats import _run_fit_batch

        return _run_fit_batch(
            jobs, buf, folds, alpha, l1_ratio, cv_folds, s_true, s_cv, batch_cap
        )
