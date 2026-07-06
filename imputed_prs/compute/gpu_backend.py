"""GPU compute backend (torch on Apple MPS / NVIDIA CUDA).

Reimplements the sample-free chunk kernel (``sufficient_stats._run_fit_chunk`` +
``gram_solve.fit_from_local_gram``) with a **device-resident band buffer** (Phase 3D): the
sliding chip band ``Z`` and its incremental full + per-fold Gram live on the GPU, so the two
O(n) accumulation kernels run there — the per-column band-Gram update (``_GpuChipGramBuffer.add``,
Seam B) and the per-chunk cross-product / OOF GEMMs (``_run_fit_chunk``, Seam A). Only a single
new column is uploaded per ``add``; the band itself never leaves the device.

The per-unit local Gram solve is **batched** on-device: every co-windowed unit's gathered Gram
block is padded to a uniform ``P`` predictors, standardized per fold, and solved together —
batched Cholesky for ridge (``l1_ratio=0``, exact) or batched masked FISTA for ElasticNet
(``l1_ratio>0``, matches the CPU coordinate-descent optimum within statistical-parity tolerance).
Fold MSE, cv_r2, and the back-transform to raw scale are reproduced as batched tensor algebra.

``s_true`` (the observed calibration target, Σ β·y over real dosages) is reduced on the host in
float64 so it stays bit-identical to the CPU oracle for free; the out-of-fold ``s_cv`` GEMM runs
against the resident band. ``torch`` is imported lazily so importing this module never requires
it. float32 on MPS (no float64), float64 on CUDA / torch-CPU — the whole band + Gram inherit that
dtype (float32 is exact for 0/1/2 dosages and within the sanctioned band after mean-imputation).

The CPU backend (``cpu_backend.CpuBackend``) is the correctness oracle; this backend is validated
against it (``tests/test_compute_backend.py``, ``tests/test_gpu_buffer.py``).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from imputed_prs.compute.gram_solve import (
    FoldModel,
    GramFitResult,
    _intercept_only_result,
)

# Standardization near-zero-variance guard — matches gram_solve._STD_EPS / metrics.py.
_STD_EPS = 1e-8
_INTERCEPT_ONLY_ATOL = 1e-10
# Max columns folded into one band-Gram GEMM update (throughput sweet spot on MPS; a wider
# group wastes work on the full symmetric new×new block — see _GpuChipGramBuffer.add_batch).
_ADD_SUBBATCH = 128
# Device-memory budget for one padded (·,K,P,P) fold-Gram solve sub-batch. Bounds *both* how
# many units' Grams are gathered simultaneously and the padded solve tensor, so a wide chunk
# (small n ⇒ large batch width) can never balloon device memory with per-unit fold-Grams.
_SOLVE_GRAM_BUDGET = 512 * 1024 * 1024


class _GpuChipGramBuffer:
    """Device-resident analogue of ``sufficient_stats._ChipGramBuffer``.

    Keeps the sliding band ``Z`` (n × cap) and its incremental full/per-fold Gram + column
    moments as **device tensors**, so the O(n) accumulation matmuls in ``add`` run on the GPU.
    Only the small bookkeeping (positions, allele frequencies, platform-index → slot map) stays
    on the host, where the window/eviction logic needs it. The public contract mirrors the numpy
    buffer's (``add``/``evict_below``/``clear`` for the streaming driver; ``gather`` for the
    kernel), except ``gather`` returns device tensors + the host slot indices.
    """

    def __init__(self, n_samples, folds, torch, device, dtype, capacity=256,
                 lazy_fold_gram=False):
        self._torch = torch
        self._device = device
        self._dtype = dtype
        self.n = n_samples
        self.folds = folds
        self.K = folds.n_folds
        self.cap = capacity
        self.m = 0
        # Projection (few, wide units): recompute the full + per-fold Gram on-demand at
        # gather rather than keep a (K, cap, cap) device tensor for a chromosome-spanning
        # merged region (Finding-#1 band-limited per-fold Gram, Phase 3E).
        self.lazy_fold_gram = lazy_fold_gram
        dev, dt = device, dtype
        self.Z = torch.zeros((n_samples, capacity), device=dev, dtype=dt)
        self.Gfull = None if lazy_fold_gram else torch.zeros((capacity, capacity), device=dev, dtype=dt)
        self.Ghold = (
            None if lazy_fold_gram
            else torch.zeros((self.K, capacity, capacity), device=dev, dtype=dt)
        )
        self.zsum = torch.zeros(capacity, device=dev, dtype=dt)
        self.zsqsum = torch.zeros(capacity, device=dev, dtype=dt)
        self.zsum_h = torch.zeros((self.K, capacity), device=dev, dtype=dt)
        self.zsqsum_h = torch.zeros((self.K, capacity), device=dev, dtype=dt)
        # Host bookkeeping (window logic + eviction searchsorted; small, O(cap)).
        self.pos = np.zeros(capacity, dtype=np.int64)
        self.af = np.zeros(capacity, dtype=np.float64)
        self.pidx = np.full(capacity, -1, dtype=np.int64)
        self.slot_of = {}
        # Contiguous device fold row-ranges (fold-block permuted order).
        self._fold_slices = [
            (int(folds.bounds[k]), int(folds.bounds[k + 1])) for k in range(self.K)
        ]
        self._fold_n_t = torch.as_tensor(
            [b - a for a, b in self._fold_slices], device=dev, dtype=dt
        )

    def _grow(self) -> None:
        torch = self._torch
        dev, dt = self._device, self._dtype
        new_cap = self.cap * 2
        Znew = torch.zeros((self.n, new_cap), device=dev, dtype=dt)
        Znew[:, : self.cap] = self.Z
        self.Z = Znew
        if not self.lazy_fold_gram:  # else the Grams are recomputed on-demand at gather
            G = torch.zeros((new_cap, new_cap), device=dev, dtype=dt)
            G[: self.cap, : self.cap] = self.Gfull
            self.Gfull = G
            Gh = torch.zeros((self.K, new_cap, new_cap), device=dev, dtype=dt)
            Gh[:, : self.cap, : self.cap] = self.Ghold
            self.Ghold = Gh
        for name in ("zsum", "zsqsum"):
            arr = getattr(self, name)
            new = torch.zeros(new_cap, device=dev, dtype=dt)
            new[: self.cap] = arr
            setattr(self, name, new)
        for name in ("zsum_h", "zsqsum_h"):
            arr = getattr(self, name)
            new = torch.zeros((self.K, new_cap), device=dev, dtype=dt)
            new[:, : self.cap] = arr
            setattr(self, name, new)
        for name in ("pos", "af", "pidx"):
            arr = getattr(self, name)
            new = np.zeros(new_cap, dtype=arr.dtype)
            new[: self.cap] = arr
            setattr(self, name, new)
        self.cap = new_cap

    def add(self, col_perm, platform_idx, position, af) -> None:
        """Append one chip column and update the band Gram on-device.

        Only ``col_perm`` (the new column) is uploaded; the ``Zᵀcol`` full + per-fold matvecs
        run against the resident band. All device writes stay 0-dim/1-dim tensors (no host sync)
        so the accumulation loop never round-trips to the CPU.
        """
        torch = self._torch
        if self.m >= self.cap:
            self._grow()
        s = self.m
        m = self.m
        col = torch.as_tensor(col_perm, device=self._device, dtype=self._dtype)
        if not self.lazy_fold_gram:  # incremental band Gram; else recomputed at gather
            dots = self.Z[:, :m].t() @ col  # (m,)
            self.Gfull[:m, s] = dots
            self.Gfull[s, :m] = dots
            self.Gfull[s, s] = col @ col
        for k in range(self.K):
            a, b = self._fold_slices[k]
            ck = col[a:b]
            cc = ck @ ck
            if not self.lazy_fold_gram:
                dk = self.Z[a:b, :m].t() @ ck  # (m,)
                self.Ghold[k, :m, s] = dk
                self.Ghold[k, s, :m] = dk
                self.Ghold[k, s, s] = cc
            self.zsum_h[k, s] = ck.sum()
            self.zsqsum_h[k, s] = cc
        self.Z[:, s] = col
        self.zsum[s] = col.sum()
        self.zsqsum[s] = col @ col
        self.pos[s] = position
        self.af[s] = af
        self.pidx[s] = platform_idx
        self.slot_of[platform_idx] = s
        self.m += 1

    def add_batch(self, cols, platform_indices, positions, afs) -> None:
        """Append a group of chip columns with the band-Gram update as GEMMs.

        Uploads the whole group ``Cnew = (n, B)`` once and folds its band-Gram contribution
        into matrix products — old×new ``Zᵀ·Cnew`` and new×new ``Cnewᵀ·Cnew`` (+ the per-fold
        analogues) — instead of ``B`` memory-bound per-column GEMVs. This is the on-device
        accumulation win: a compute-bound GEMM saturates the GPU where a stream of GEMVs does
        not. The resulting Gram equals the per-column path to float32 (symmetric sample sums).
        """
        torch = self._torch
        B = len(platform_indices)
        if B == 0:
            return
        # Sub-batch wide blocks: a single Cnewᵀ·Cnew over a huge group computes the full
        # symmetric new×new (2× the needed work); splitting keeps each new×new small and
        # turns the cross-group pairs into the (already-needed) old×new GEMM. ~128 is the
        # measured throughput sweet spot on MPS.
        if B > _ADD_SUBBATCH:
            for s in range(0, B, _ADD_SUBBATCH):
                e = min(s + _ADD_SUBBATCH, B)
                self.add_batch(
                    cols[s:e], platform_indices[s:e], positions[s:e], afs[s:e]
                )
            return
        while self.m + B > self.cap:
            self._grow()
        s = self.m
        Cnew = torch.as_tensor(
            np.stack(cols, axis=1), device=self._device, dtype=self._dtype
        )  # (n, B)
        if not self.lazy_fold_gram:  # incremental band Gram; else recomputed at gather
            if s > 0:
                cross = self.Z[:, :s].t() @ Cnew  # (s, B) old×new
                self.Gfull[:s, s : s + B] = cross
                self.Gfull[s : s + B, :s] = cross.t()
            self.Gfull[s : s + B, s : s + B] = Cnew.t() @ Cnew  # (B, B) new×new
        for k in range(self.K):
            a, b = self._fold_slices[k]
            Ck = Cnew[a:b]  # (n_k, B)
            if not self.lazy_fold_gram:
                if s > 0:
                    cross_k = self.Z[a:b, :s].t() @ Ck  # (s, B)
                    self.Ghold[k, :s, s : s + B] = cross_k
                    self.Ghold[k, s : s + B, :s] = cross_k.t()
                self.Ghold[k, s : s + B, s : s + B] = Ck.t() @ Ck
            self.zsum_h[k, s : s + B] = Ck.sum(0)
            self.zsqsum_h[k, s : s + B] = (Ck * Ck).sum(0)
        self.Z[:, s : s + B] = Cnew
        self.zsum[s : s + B] = Cnew.sum(0)
        self.zsqsum[s : s + B] = (Cnew * Cnew).sum(0)
        for i in range(B):
            self.pos[s + i] = positions[i]
            self.af[s + i] = afs[i]
            self.pidx[s + i] = platform_indices[i]
            self.slot_of[platform_indices[i]] = s + i
        self.m += B

    def evict_below(self, min_pos) -> None:
        """Drop leading (lowest-position) columns with ``pos < min_pos`` (device slice-shift)."""
        m = self.m
        if m == 0:
            return
        c = int(np.searchsorted(self.pos[:m], min_pos, side="left"))
        if c == 0:
            return
        keep = m - c
        # ``.clone()`` the RHS so the overlapping in-place shift cannot alias.
        self.Z[:, :keep] = self.Z[:, c:m].clone()
        if not self.lazy_fold_gram:
            self.Gfull[:keep, :keep] = self.Gfull[c:m, c:m].clone()
            self.Ghold[:, :keep, :keep] = self.Ghold[:, c:m, c:m].clone()
        self.zsum[:keep] = self.zsum[c:m].clone()
        self.zsqsum[:keep] = self.zsqsum[c:m].clone()
        self.zsum_h[:, :keep] = self.zsum_h[:, c:m].clone()
        self.zsqsum_h[:, :keep] = self.zsqsum_h[:, c:m].clone()
        for name in ("pos", "af", "pidx"):
            arr = getattr(self, name)
            arr[:keep] = arr[c:m].copy()
        self.m = keep
        self.slot_of = {int(self.pidx[i]): i for i in range(keep)}

    def clear(self) -> None:
        self.m = 0
        self.slot_of = {}

    def resolve(self, platform_indices):
        """Cheap predictor resolution: ``(idx_host, idx_dev, af)`` with **no** Gram slices.

        Separated from :meth:`gather_slices` so the chunk kernel can partition every unit
        (fallback checks + af) without materializing the O(K·p²) per-fold Gram for all units
        at once — the device Grams are gathered later, one memory-bounded sub-batch at a time.
        """
        idx_host = np.fromiter(
            (self.slot_of[int(p)] for p in platform_indices),
            dtype=np.int64, count=len(platform_indices),
        )
        idx = self._torch.as_tensor(idx_host, device=self._device, dtype=self._torch.long)
        return idx_host, idx, self.af[idx_host]

    def gather_slices(self, idx):
        """Device slices of the resident Gram/moments for the ``idx`` predictors (no af)."""
        m = self.m
        if self.lazy_fold_gram:
            # Recompute the full + per-fold Gram on-demand over this unit's ≤max_predictors
            # fit predictors from the resident band ``Z`` — never a (K, cap, cap) device
            # tensor. ``G`` and ``fold_G`` come from the same ``Zsub`` slice so
            # ``G == Σ_k fold_G[k]`` (Finding-#1 band-limited per-fold Gram, Phase 3E).
            torch = self._torch
            Zsub = self.Z.index_select(1, idx)  # (n, p) on-demand sub-band
            fold_G = torch.stack(
                [Zsub[a:b].t() @ Zsub[a:b] for (a, b) in self._fold_slices]
            )
            return {
                "G": Zsub.t() @ Zsub,
                "fold_G": fold_G,
                "zsum": self.zsum[:m].index_select(0, idx),
                "zsqsum": self.zsqsum[:m].index_select(0, idx),
                "fold_zsum": self.zsum_h[:, :m].index_select(1, idx),
                "fold_zsqsum": self.zsqsum_h[:, :m].index_select(1, idx),
            }
        return {
            "G": self.Gfull[:m, :m].index_select(0, idx).index_select(1, idx),
            "fold_G": self.Ghold[:, :m, :m].index_select(1, idx).index_select(2, idx),
            "zsum": self.zsum[:m].index_select(0, idx),
            "zsqsum": self.zsqsum[:m].index_select(0, idx),
            "fold_zsum": self.zsum_h[:, :m].index_select(1, idx),
            "fold_zsqsum": self.zsqsum_h[:, :m].index_select(1, idx),
        }

    def gather(self, platform_indices):
        """Return ``(idx_host, idx_dev, blocks)`` for a unit's predictors, in the given order.

        ``blocks`` holds device slices of the resident Gram/moments (no host round-trip);
        ``idx_host`` is the numpy slot vector the caller uses to scatter OOF fold coefficients,
        and ``idx_dev`` slices the chunk cross-products ``C``/``Ck`` on-device. Cross-products
        themselves are not gathered here — they are batched GEMMs in ``_run_fit_chunk``.
        (Kept as the one-shot form for :meth:`GpuBackend._solve_blocks` and parity tests.)
        """
        idx_host, idx, af = self.resolve(platform_indices)
        blocks = self.gather_slices(idx)
        blocks["af"] = af
        return idx_host, idx, blocks


class GpuBackend:
    """torch-backed compute backend: device-resident band + batched local-Gram solve."""

    def __init__(self, device: str):
        import torch  # lazy: importing this module must not require torch

        self._device_name = device
        self._torch = torch
        self._device = torch.device(device)
        # MPS has no float64; float32 is fine for 0/1/2 dosages (statistical parity).
        self._dtype = torch.float32 if device == "mps" else torch.float64
        # FISTA controls (ElasticNet only). Iterate to a tight coef-change tolerance so
        # the solution sits on the same optimum sklearn's CD approximates.
        self._fista_max_iter = 2000
        self._fista_tol = 1e-7
        self._power_iters = 12

    @property
    def device_name(self) -> str:
        return self._device_name

    def make_buffer(self, n_samples, folds, lazy_fold_gram: bool = False):
        return _GpuChipGramBuffer(
            n_samples, folds, self._torch, self._device, self._dtype,
            lazy_fold_gram=lazy_fold_gram,
        )

    # ------------------------------------------------------------------ public
    def run_fit_batch(
        self, jobs, buf, folds, alpha, l1_ratio, cv_folds, s_true, s_cv, batch_cap
    ) -> int:
        n_io = 0
        for s in range(0, len(jobs), batch_cap):
            n_io += self._run_fit_chunk(
                jobs[s : s + batch_cap], buf, folds, alpha, l1_ratio, cv_folds,
                s_true, s_cv,
            )
        return n_io

    # ------------------------------------------------------------------ kernel
    def _run_fit_chunk(
        self, jobs, buf, folds, alpha, l1_ratio, cv_folds, s_true, s_cv
    ) -> int:
        """Device-native mirror of ``sufficient_stats._run_fit_chunk``.

        Consumes the device-resident ``buf``: the Seam-A cross-products (``C``, per-fold
        ``Ck``) and y-moments are batched GEMMs over the resident band; each unit's Gram
        sub-block is gathered on-device and fed straight into the padded solve; the OOF
        reduction runs against the band. Only the small per-unit results and the ``s_cv``
        delta come back to the host; ``s_true`` is reduced on the host float64 target
        (exact, matches the CPU oracle). ``s_true``/``s_cv`` are mutated in place; returns
        the number of *calibrating* intercept-only models.
        """
        torch = self._torch
        dev, dt = self._device, self._dtype
        n, K, T = folds.n, folds.n_folds, len(jobs)
        m = buf.m
        Zband = buf.Z[:, :m]  # (n, m) device band view — never copied

        # --- Upload targets + device Seam-A GEMMs (C, per-fold Ck, y-moments). ---
        Yh = np.empty((n, T), dtype=np.float64)
        for j, job in enumerate(jobs):
            Yh[:, j] = job.col
        Y = torch.as_tensor(Yh, device=dev, dtype=dt)
        C = Zband.t() @ Y  # (m, T)
        Ck: List = [None] * K
        fysum_KT = torch.empty((K, T), device=dev, dtype=dt)
        fysqsum_KT = torch.empty((K, T), device=dev, dtype=dt)
        for k in range(K):
            a, b = buf._fold_slices[k]
            Yk = Y[a:b]
            Ck[k] = Zband[a:b].t() @ Yk  # (m, T)
            fysum_KT[k] = Yk.sum(0)
            fysqsum_KT[k] = (Yk * Yk).sum(0)
        # Host y-moments for the fallback/ss_tot scalars (T,).
        ysum_all = Yh.sum(0)
        ysqsum_all = np.einsum("ij,ij->j", Yh, Yh)
        fn_t = buf._fold_n_t

        # --- Fallback partition (cheap: resolve idx + af, no Gram slices yet). ---
        # The per-unit device Gram is O(K·p²); gathering it for *every* unit up front OOMs the
        # GPU when the chunk is wide (small n ⇒ large _batch_cap, which is sized for the (n×T)
        # Seam-A arrays, not the Gram). So only resolve indices here; gather the Grams later,
        # one memory-bounded sub-batch at a time.
        results: List[Optional[GramFitResult]] = [None] * T
        idx_host_of: List[Optional[np.ndarray]] = [None] * T
        pred_afs: List[Optional[np.ndarray]] = [None] * T
        real: List = []  # (j, idx_host, gidx, p, ysum, ysqsum)
        for j, job in enumerate(jobs):
            try:
                pred_idx = job.pred_idx
                p = len(pred_idx)
                ysum_j = float(ysum_all[j])
                ysqsum_j = float(ysqsum_all[j])
                if p == 0:
                    results[j] = _intercept_only_result(ysum_j, ysqsum_j, n, 0)
                    pred_afs[j] = np.empty(0)
                    continue
                idx_host, gidx, af = buf.resolve(pred_idx)
                idx_host_of[j] = idx_host
                pred_afs[j] = af
                ybar = ysum_j / n
                if n < cv_folds or np.sqrt(max(ysqsum_j / n - ybar * ybar, 0.0)) < 1e-10:
                    results[j] = _intercept_only_result(ysum_j, ysqsum_j, n, p)
                    continue
                real.append((j, idx_host, gidx, p, ysum_j, ysqsum_j))
            except Exception as exc:  # noqa: BLE001 - record, don't crash (mirror legacy)
                job.fail(exc)

        # --- Memory-bounded gather + batched solve. Sort by predictor count so each sub-batch
        # pads to a similar P (minimal waste); grow it until the padded (·,K,P,P) fold-Gram would
        # exceed the budget. Only one sub-batch's device Grams are ever resident. ---
        real.sort(key=lambda r: r[3])
        itemsize = 4 if self._dtype == torch.float32 else 8
        i = 0
        while i < len(real):
            cnt = 1
            while i + cnt < len(real):
                pnext = real[i + cnt][3]  # sorted ascending ⇒ the sub-batch's new max P
                if (cnt + 1) * K * pnext * pnext * itemsize > _SOLVE_GRAM_BUDGET:
                    break
                cnt += 1
            sub = real[i : i + cnt]
            blks = []
            for (_j, _idx_host, gidx, p, ysum_j, ysqsum_j) in sub:
                gg = buf.gather_slices(gidx)
                fc = torch.stack([Ck[k].index_select(0, gidx)[:, _j] for k in range(K)])
                blks.append({
                    "p": p, "n": n, "ysum": ysum_j, "ysqsum": ysqsum_j,
                    "G": gg["G"], "c": C.index_select(0, gidx)[:, _j],
                    "zsum": gg["zsum"], "zsqsum": gg["zsqsum"],
                    "fG": gg["fold_G"], "fc": fc,
                    "fzsum": gg["fold_zsum"], "fzsqsum": gg["fold_zsqsum"],
                    "fysum": fysum_KT[:, _j], "fysqsum": fysqsum_KT[:, _j], "fn": fn_t,
                })
            for (jj, *_rest), res in zip(sub, self._solve_padded_core(blks, alpha, l1_ratio)):
                results[jj] = res
            del blks
            i += cnt

        # --- Store + OOF + calibration reduction. ---
        Wk_h = np.zeros((K, m, T), dtype=np.float64)
        bk = np.zeros((K, T), dtype=np.float64)
        coef_vec = np.zeros(T, dtype=np.float64)
        n_io = 0
        for j, job in enumerate(jobs):
            result = results[j]
            if result is None:  # gather failed above → already job.fail'd
                continue
            try:
                job.store(result, job.pred_idx, pred_afs[j])
                if job.is_calibrating:
                    if result.fold_models:
                        idx_host = idx_host_of[j]
                        for k, fm in enumerate(result.fold_models):
                            Wk_h[k, idx_host, j] = fm.coef
                            bk[k, j] = fm.intercept
                    else:  # intercept-only ⇒ constant OOF = intercept on every sample
                        bk[:, j] = result.intercept
                    coef_vec[j] = job.calib_coef
                    if result.is_intercept_only:
                        n_io += 1
            except Exception as exc:  # noqa: BLE001
                job.fail(exc)

        if coef_vec.any():
            # s_true: host float64 reduction of the stacked target (exact, matches CPU).
            s_true += Yh @ coef_vec
            # s_cv: batched OOF GEMM against the resident band (one per fold), reduced.
            Wk = torch.as_tensor(Wk_h, device=dev, dtype=dt)
            bk_t = torch.as_tensor(bk, device=dev, dtype=dt)
            coef_t = torch.as_tensor(coef_vec, device=dev, dtype=dt)
            oof = torch.empty((n, T), device=dev, dtype=dt)
            for k in range(K):
                a, b = buf._fold_slices[k]
                oof[a:b] = Zband[a:b] @ Wk[k] + bk_t[k]
            s_cv += (oof @ coef_t).detach().cpu().numpy().astype(np.float64)
        return n_io

    # ------------------------------------------------------------ batched solve
    def _solve_blocks(
        self, blocks, alpha, l1_ratio, cv_folds
    ) -> List[Optional[GramFitResult]]:
        """Batched analogue of ``[fit_from_local_gram(b) for b in blocks]`` for **numpy**
        ``LocalGramBlock`` inputs (the block-level parity oracle in tests).

        Partitions intercept-only fallbacks on the host, uploads the rest into the shared
        device solve core. The streaming kernel gathers device blocks directly instead
        (``_run_fit_chunk``); both funnel through ``_solve_real_padded``.
        """
        torch = self._torch
        dev, dt = self._device, self._dtype
        tt = lambda a: torch.as_tensor(a, device=dev, dtype=dt)  # noqa: E731
        results: List[Optional[GramFitResult]] = [None] * len(blocks)
        real_blks: List = []
        real_pos: List[int] = []
        for i, b in enumerate(blocks):
            if b is None:
                continue
            p, n = b.n_predictors, b.n
            if p == 0 or n < cv_folds:
                results[i] = _intercept_only_result(b.ysum, b.ysqsum, n, p)
                continue
            ybar = b.ysum / n
            if np.sqrt(max(b.ysqsum / n - ybar * ybar, 0.0)) < 1e-10:
                results[i] = _intercept_only_result(b.ysum, b.ysqsum, n, p)
                continue
            real_blks.append({
                "p": p, "n": n, "ysum": b.ysum, "ysqsum": b.ysqsum,
                "G": tt(b.G), "c": tt(b.c), "zsum": tt(b.zsum), "zsqsum": tt(b.zsqsum),
                "fG": tt(np.stack(b.fold_G)), "fc": tt(np.stack(b.fold_c)),
                "fzsum": tt(np.stack(b.fold_zsum)), "fzsqsum": tt(np.stack(b.fold_zsqsum)),
                "fysum": tt(np.asarray(b.fold_ysum)), "fysqsum": tt(np.asarray(b.fold_ysqsum)),
                "fn": tt(np.asarray(b.fold_n, dtype=np.float64)),
            })
            real_pos.append(i)
        for pos, res in zip(
            real_pos, self._solve_real_padded(real_blks, alpha, l1_ratio, cv_folds)
        ):
            results[pos] = res
        return results

    def _solve_real_padded(self, blks, alpha, l1_ratio, cv_folds) -> List[GramFitResult]:
        """Solve a list of on-device Gram blocks, sub-batched so the padded ``(B,K,P,P)``
        fold-Gram tensor stays ≲ 512 MB."""
        if not blks:
            return []
        torch = self._torch
        P = max(b["p"] for b in blks)
        K = int(blks[0]["fn"].shape[0])
        itemsize = 4 if self._dtype == torch.float32 else 8
        cap = max(1, _SOLVE_GRAM_BUDGET // max(K * P * P * itemsize, 1))
        out: List[GramFitResult] = []
        for s in range(0, len(blks), cap):
            out.extend(self._solve_padded_core(blks[s : s + cap], alpha, l1_ratio))
        return out

    def _solve_padded_core(self, blks, alpha, l1_ratio) -> List[GramFitResult]:
        """Pad on-device Gram blocks to a uniform ``P`` and solve final + K fold models,
        reconstructing held-out MSE / cv_r2 as batched tensor algebra (matches
        ``fit_from_local_gram``)."""
        torch = self._torch
        dev, dt = self._device, self._dtype
        B = len(blks)
        P = max(b["p"] for b in blks)
        K = int(blks[0]["fn"].shape[0])
        z = lambda *shape: torch.zeros(*shape, device=dev, dtype=dt)  # noqa: E731

        Gp = z(B, P, P); cp = z(B, P); zsp = z(B, P); zsqp = z(B, P); mask = z(B, P)
        fGp = z(B, K, P, P); fcp = z(B, K, P); fzsp = z(B, K, P); fzsqp = z(B, K, P)
        fysum_m = z(B, K); fysqsum_m = z(B, K)
        ysum_v = z(B); ysqsum_v = z(B); n_v = z(B)
        for i, b in enumerate(blks):
            p = b["p"]
            mask[i, :p] = 1.0
            Gp[i, :p, :p] = b["G"]; cp[i, :p] = b["c"]
            zsp[i, :p] = b["zsum"]; zsqp[i, :p] = b["zsqsum"]
            fGp[i, :, :p, :p] = b["fG"]; fcp[i, :, :p] = b["fc"]
            fzsp[i, :, :p] = b["fzsum"]; fzsqp[i, :, :p] = b["fzsqsum"]
            fysum_m[i] = b["fysum"]; fysqsum_m[i] = b["fysqsum"]
            ysum_v[i] = b["ysum"]; ysqsum_v[i] = b["ysqsum"]; n_v[i] = float(b["n"])
        fn_v = blks[0]["fn"]  # (K,) identical fold sizes across the batch

        # Final model on the full valid subset.
        _w, coef_raw, intercept_raw, is_io = self._standardize_solve_backtransform(
            Gp, cp, zsp, zsqp, ysum_v, ysqsum_v, n_v, mask, alpha, l1_ratio
        )
        # K fold models (train = full − held-out-k).
        fcoef = z(B, K, P); fint = z(B, K)
        for k in range(K):
            _wk, coef_k, int_k, _iok = self._standardize_solve_backtransform(
                Gp - fGp[:, k], cp - fcp[:, k], zsp - fzsp[:, k], zsqp - fzsqp[:, k],
                ysum_v - fysum_m[:, k], ysqsum_v - fysqsum_m[:, k], n_v - fn_v[k],
                mask, alpha, l1_ratio,
            )
            fcoef[:, k] = coef_k; fint[:, k] = int_k

        # Held-out MSE + cv metrics, batched: sse = Σy² + aᵀGa + n·b² − 2aᵀc − 2b·Σy + 2b·aᵀΣz.
        a = fcoef
        Ga = torch.bmm(fGp.reshape(B * K, P, P), a.reshape(B * K, P, 1)).reshape(B, K, P)
        aGa = (a * Ga).sum(-1); ac = (a * fcp).sum(-1); az = (a * fzsp).sum(-1)
        bt = fint
        nval = fn_v.view(1, K)
        sse = fysqsum_m + aGa + nval * bt * bt - 2.0 * ac - 2.0 * bt * fysum_m + 2.0 * bt * az
        cv_mse = (sse / nval).mean(1)
        ss_res = sse.sum(1)
        ss_tot = ysqsum_v - ysum_v * ysum_v / n_v
        cv_r2 = torch.where(ss_tot >= 1e-10, 1.0 - ss_res / ss_tot, torch.zeros_like(ss_tot))

        to_host = lambda x: x.detach().cpu().numpy().astype(np.float64)  # noqa: E731
        coef_np = to_host(coef_raw); int_np = to_host(intercept_raw)
        io_np = is_io.detach().cpu().numpy()
        fcoef_np = to_host(fcoef); fint_np = to_host(fint)
        cvm = to_host(cv_mse); cvr = to_host(cv_r2)

        out: List[GramFitResult] = []
        for i, b in enumerate(blks):
            p = b["p"]
            fold_models = [
                FoldModel(coef=fcoef_np[i, k, :p].copy(), intercept=float(fint_np[i, k]))
                for k in range(K)
            ]
            out.append(GramFitResult(
                coefficients=coef_np[i, :p].copy(),
                intercept=float(int_np[i]),
                cv_mse=float(cvm[i]),
                cv_r2=float(cvr[i]),
                is_intercept_only=bool(io_np[i]),
                n_predictors=p,
                n_samples=int(b["n"]),
                fold_models=fold_models,
            ))
        return out

    # --------------------------------------------------------------- primitives
    def _standardize_solve_backtransform(
        self, G, c, zsum, zsqsum, ysum, ysqsum, n, mask, alpha, l1_ratio
    ):
        torch = self._torch
        B, P, _ = G.shape
        ninv = 1.0 / n
        mean = zsum * ninv[:, None]
        var = zsqsum * ninv[:, None] - mean * mean
        std = torch.sqrt(torch.clamp(var, min=0.0))
        scale = torch.where(std < _STD_EPS, torch.ones_like(std), std)
        ybar = ysum * ninv
        inv_scale = 1.0 / scale
        ms = mean * inv_scale
        G_std = G * inv_scale[:, :, None] * inv_scale[:, None, :]
        G_std = G_std - n[:, None, None] * ms[:, :, None] * ms[:, None, :]
        outer = mask[:, :, None] * mask[:, None, :]
        G_std = G_std * outer + torch.diag_embed(1.0 - mask)  # padded rows → identity
        q_std = (c - n[:, None] * mean * ybar[:, None]) * inv_scale
        q_std = q_std * mask

        if l1_ratio == 0.0:
            w_std = self._batched_ridge(G_std, q_std, n, alpha)
        else:
            w_std = self._batched_fista(G_std, q_std, n, alpha, l1_ratio, mask)
        w_std = w_std * mask

        coef_raw = (w_std * inv_scale) * mask
        intercept_raw = ybar - (w_std * (mean * inv_scale)).sum(dim=1)
        is_io = (w_std.abs().amax(dim=1) <= _INTERCEPT_ONLY_ATOL)
        return w_std, coef_raw, intercept_raw, is_io

    def _batched_ridge(self, G_std, q_std, n, alpha):
        torch = self._torch
        B, P, _ = G_std.shape
        l2 = (alpha * n)[:, None].expand(B, P)
        A = G_std + torch.diag_embed(l2)
        L = torch.linalg.cholesky(A)
        return torch.cholesky_solve(q_std[:, :, None], L)[:, :, 0]

    def _batched_fista(self, G_std, q_std, n, alpha, l1_ratio, mask):
        torch = self._torch
        B, P, _ = G_std.shape
        l1_reg = alpha * l1_ratio * n           # (B,)
        l2_reg = alpha * (1.0 - l1_ratio) * n    # (B,)
        H = G_std + torch.diag_embed(l2_reg[:, None].expand(B, P))
        Lip = self._power_lipschitz(H, mask)     # (B,)
        step = 1.0 / Lip
        thr = (l1_reg * step)[:, None]
        w = torch.zeros(B, P, device=G_std.device, dtype=G_std.dtype)
        w_prev = w.clone()
        tk = 1.0
        # Sync the convergence scalar to the host only every ``check_every`` iterations —
        # a per-iteration ``float(delta)`` serializes the GPU (device→host sync each step).
        check_every = 25
        for it in range(self._fista_max_iter):
            tk_new = 0.5 * (1.0 + (1.0 + 4.0 * tk * tk) ** 0.5)
            mom = (tk - 1.0) / tk_new
            v = (w + mom * (w - w_prev)) * mask
            grad = torch.bmm(H, v[:, :, None])[:, :, 0] - q_std
            zt = v - step[:, None] * grad
            w_new = torch.sign(zt) * torch.clamp(zt.abs() - thr, min=0.0)
            w_new = w_new * mask
            converged = (
                it % check_every == check_every - 1
                and float((w_new - w).abs().amax()) < self._fista_tol
            )
            w_prev, w, tk = w, w_new, tk_new
            if converged:
                break
        return w

    def _power_lipschitz(self, H, mask):
        torch = self._torch
        v = mask.clone()
        v = v / (v.norm(dim=1, keepdim=True) + 1e-12)
        for _ in range(self._power_iters):
            hv = torch.bmm(H, v[:, :, None])[:, :, 0] * mask
            v = hv / (hv.norm(dim=1, keepdim=True) + 1e-12)
        hv = torch.bmm(H, v[:, :, None])[:, :, 0] * mask
        lam = (v * hv).sum(dim=1)
        return torch.clamp(lam, min=1e-12)
