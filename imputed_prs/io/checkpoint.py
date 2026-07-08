"""Opt-in disk checkpoint for resumable streaming fits (Phase 10).

A streaming fit (and, more expensively, a k-fold reference CV) contracts the sample
dimension into one **per-chromosome partial** per chromosome — the fitted models plus the
two calibration accumulators — produced by ``fitter._run_one_chromosome`` and merged by an
**order-independent reduce** (sorted by ``_chrom_sort_key``). Chromosome shards are
zero-halo, so a chromosome's partial is bit-identical regardless of sharding.

That makes the per-chromosome partial a clean checkpoint unit: persist each partial as it
completes, and a killed run resumes by loading the finished chromosomes and computing only
the missing ones. Mixing disk-loaded and freshly-computed partials is bit-identical because
the reduce sorts and (for the calibration vectors) sums additively in canonical order.

**It is inert until a caller passes ``checkpoint_dir``.** Nothing in the default path
touches disk; ``checkpoint_dir=None`` is byte-for-byte the current fit.

Storage model
-------------
On disk each key is a directory ``<checkpoint_dir>/<digest>/`` holding one ``chr{c}.ckpt``
shard per completed chromosome (a :func:`joblib.dump` of the partial — the same serialization
loky already uses to move partials between processes) plus a ``manifest.json`` header written
first, atomically. A shard is committed atomically (``joblib.dump`` to ``.chr{c}.ckpt.tmp.{pid}``
then :func:`os.replace`), so **shard-file presence is the completion record** and an
interrupted write leaves only an ignored ``.tmp`` — that chromosome simply recomputes.

The key
-------
Unlike the Phase-9 stats cache (whose beta-free raw Gram legitimately omits effect sizes), a
checkpoint stores **models and calibration derived from the PRS betas**, so the key **must**
bind the effect sizes and orientation: a re-weighted or re-oriented PRS on the same variants
must miss. The reduce's disjointness assert only catches *same-chromosome* id collisions, so
key completeness here is a correctness requirement, not a convenience. The digest folds in the
reference panel identity, the predictor (chip) set, a PRS-content digest over
``(role, id, effect_flip, beta)`` for every target/fallback/observed term, the window params,
``alpha``/``l1_ratio``, ``device`` + solve-mode, and the ``mode`` ("fit"/"cv") + fold-partition
key. Because the digest names the directory, a different config lands in a different directory
and never reuses stale shards.

Invalidation mirrors the stats cache: a manifest whose ``schema_version`` or key components
disagree is reset (``rmtree`` + fresh). A ``git_sha`` drift is a **warning, not a reset** — an
unrelated commit must not destroy a multi-day checkpoint — but it flags that resuming may not
be bit-identical if the fit/solver code changed. Bump :data:`_CHECKPOINT_SCHEMA_VERSION` when
any pickled partial class changes shape. Conventions (``clear_*``/``*_info`` helpers, silent
corrupt-tolerant reads) mirror :mod:`imputed_prs.io.stats_cache`.
"""

from __future__ import annotations

import json
import os
import shutil
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np

from imputed_prs.io.stats_cache import _chip_set_hash, _git_sha, _ref_digest, _sha1

# Bump when the pickled partial classes (``_ImputeChromPartial``, ``_ProjectChromPartial``,
# ``ImputedVariantModel``, ``ProjectionRegionModel``, or the ``cv_collector`` shape) change
# in a way that makes an old shard unreadable/wrong. A manifest with a different
# schema_version is treated as incompatible → reset (recompute).
_CHECKPOINT_SCHEMA_VERSION = 1
_MANIFEST_NAME = "manifest.json"
_SHARD_SUFFIX = ".ckpt"

# A PRS content token binds (role, variant_id, effect_flip, beta); ``PrsTerm`` is that tuple.
PrsTerm = Tuple[str, str, bool, float]


# ---------------------------------------------------------------------------
# Key computation
# ---------------------------------------------------------------------------
def _prs_content_hash(prs_terms: Sequence[PrsTerm]) -> str:
    """Order-independent digest binding each PRS term's ``(role, id, flip, beta)``.

    ``repr(float(beta))`` is the shortest round-trippable representation, so distinct effect
    sizes never collide. ``role`` (``t``/``b``/``o`` for target/fallback/observed) keeps the
    roles distinct even in the (impossible-by-construction) event two share an id.
    """
    tokens = sorted(
        f"{role}:{vid}:{int(bool(flip))}:{beta!r}"
        for role, vid, flip, beta in (
            (r, v, f, float(b)) for r, v, f, b in prs_terms
        )
    )
    return _sha1(*tokens)


@dataclass(frozen=True)
class CheckpointKey:
    """The identity of one checkpointable fit, split into its provenance components.

    ``digest`` (truncated combined hash) names the on-disk directory; the full components +
    ``config`` are re-checked against the manifest on resume, so a truncation collision
    degrades to a reset, never a wrong resume.
    """

    digest: str
    ref_digest: str
    predictor_hash: str
    prs_hash: str
    config: Dict[str, Any]


def make_checkpoint_key(
    *,
    sample_ids: Sequence[str],
    predictor_ids: Sequence[str],
    prs_terms: Sequence[PrsTerm],
    window_size: int,
    max_predictors: Optional[int],
    cv_folds: int,
    random_state: Optional[int],
    alpha: float,
    l1_ratio: float,
    device: str,
    solve_mode: str,
    mode: str,
    fold_key: Optional[str] = None,
    source_file: Optional[Union[str, Path]] = None,
    n_variants: Optional[int] = None,
) -> CheckpointKey:
    """Build the :class:`CheckpointKey` for a streaming fit / reference-CV run.

    ``mode`` is ``"fit"`` or ``"cv"``; a CV key also passes ``fold_key`` = a digest of the
    outer fold partition. Unlike the stats-cache key, ``alpha``/``l1_ratio``/``device``/
    solve-mode ARE part of the key: they change the per-chromosome bits a checkpoint stores.
    """
    ref = _ref_digest(sample_ids, source_file=source_file, n_variants=n_variants)
    pred = _chip_set_hash(predictor_ids)
    prs = _prs_content_hash(prs_terms)
    config = {
        "window_size": window_size,
        "max_predictors": max_predictors,
        "cv_folds": cv_folds,
        "random_state": random_state,
        "alpha": float(alpha),
        "l1_ratio": float(l1_ratio),
        "device": device,
        "solve_mode": solve_mode,
        "mode": mode,
        "fold_key": fold_key,
    }
    config_hash = _sha1(*[f"{k}={config[k]!r}" for k in sorted(config)])
    digest = _sha1(_CHECKPOINT_SCHEMA_VERSION, ref, pred, prs, config_hash)[:16]
    return CheckpointKey(
        digest=digest,
        ref_digest=ref,
        predictor_hash=pred,
        prs_hash=prs,
        config=config,
    )


# ---------------------------------------------------------------------------
# Plan → key material (shared by the plain-fit and reference-CV key builders, so a fit
# and a CV over the same PRS share the predictor/PRS digests and differ only in mode/folds)
# ---------------------------------------------------------------------------
def imputation_plan_terms(plan) -> Tuple[List[str], List[PrsTerm]]:
    """``(predictor_ids, prs_terms)`` for an imputation ``StreamPlan`` — the content to key."""
    predictor_ids = list(plan.chip_ids)
    prs_terms: List[PrsTerm] = [
        ("t", vid, tv.effect_flip, tv.beta) for vid, tv in plan.targets.items()
    ]
    prs_terms += [
        ("b", vid, tv.effect_flip, tv.beta) for vid, tv in plan.fallback_targets.items()
    ]
    prs_terms += [
        ("o", vid, ov.effect_flip, ov.beta) for vid, ov in plan.observed.items()
    ]
    return predictor_ids, prs_terms


def projection_plan_terms(plan) -> Tuple[List[str], List[PrsTerm]]:
    """``(predictor_ids, prs_terms)`` for a projection ``ProjectionStreamPlan``.

    Binds each region-member's ``(ref_id:region_index, effect_flip, beta)`` — the streamed
    ``S_R`` accumulation terms — plus the observed / fallback terms.
    """
    predictor_ids = list(plan.chip_ids)
    prs_terms: List[PrsTerm] = [
        ("r", f"{ref_id}:{region_index}", effect_flip, beta)
        for ref_id, members in plan.region_members.items()
        for region_index, beta, effect_flip in members
    ]
    prs_terms += [
        ("b", vid, tv.effect_flip, tv.beta) for vid, tv in plan.fallback_targets.items()
    ]
    prs_terms += [
        ("o", vid, ov.effect_flip, ov.beta) for vid, ov in plan.observed.items()
    ]
    return predictor_ids, prs_terms


def fold_partition_key(fold_indices: Sequence[Sequence[int]]) -> str:
    """Digest of a reference-CV outer partition (order-independent within a fold).

    Fold ``k``'s membership defines the leave-one-out split, so the digest is
    order-independent within a fold but order-sensitive across folds.
    """
    parts = [_sha1(*sorted(int(i) for i in idx)) for idx in fold_indices]
    return _sha1(*parts)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def _dep_versions() -> Dict[str, str]:
    """numpy/scipy/sklearn versions for the manifest (bit-identity is dependency-versioned)."""
    versions = {"numpy": np.__version__}
    for name, mod in (("scipy", "scipy"), ("sklearn", "sklearn")):
        try:
            versions[name] = __import__(mod).__version__
        except Exception:  # noqa: BLE001 - provenance is informational, never fatal
            versions[name] = "unknown"
    return versions


def _safe_chrom(chrom: str) -> str:
    """A filesystem-safe token for a chromosome (chroms are ``1``..``22``/``X``/``Y``/``MT``)."""
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(chrom))


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------
class CheckpointStore:
    """A per-chromosome checkpoint under ``<checkpoint_dir>/<key.digest>/``.

    On construction it validates any existing entry against ``key`` (reset on a schema /
    key-component mismatch, warn on a git drift) and writes a fresh ``manifest.json`` first,
    atomically. Thereafter :meth:`save` commits one chromosome's partial atomically and
    :meth:`load` returns a completed chromosome's partial (or ``None``). Both are best-effort:
    a disk error disables further checkpointing but never raises into the fit.
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        key: CheckpointKey,
        *,
        verbose: int = 0,
    ) -> None:
        self.key = key
        self.verbose = verbose
        self.root = Path(checkpoint_dir)
        self.dir = self.root / key.digest
        self._ok = True
        self._prepare()

    # -- lifecycle ----------------------------------------------------------
    def _prepare(self) -> None:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            self._ok = False  # unwritable → checkpointing silently disabled
            return
        manifest_path = self.dir / _MANIFEST_NAME
        existing: Optional[Dict[str, Any]] = None
        if manifest_path.exists():
            try:
                existing = json.loads(manifest_path.read_text())
            except Exception:  # noqa: BLE001 - a corrupt manifest resets
                existing = None
        if existing is not None and self._compatible(existing):
            self._maybe_warn_git(existing)
            self._sweep_tmp()
            return
        # Absent, corrupt, or incompatible (schema / component / truncation collision):
        # wipe any stale shards and write a fresh header.
        try:
            if self.dir.exists():
                shutil.rmtree(self.dir)
            self.dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            self._ok = False
            return
        self._write_manifest()

    def _compatible(self, manifest: Dict[str, Any]) -> bool:
        if manifest.get("schema_version") != _CHECKPOINT_SCHEMA_VERSION:
            return False
        k = manifest.get("key", {})
        return (
            k.get("ref_digest") == self.key.ref_digest
            and k.get("predictor_hash") == self.key.predictor_hash
            and k.get("prs_hash") == self.key.prs_hash
            and k.get("config") == self.key.config
        )

    def _maybe_warn_git(self, manifest: Dict[str, Any]) -> None:
        prev = manifest.get("provenance", {}).get("git_sha")
        cur = _git_sha()
        if prev and cur and "unknown" not in (prev, cur) and prev != cur:
            warnings.warn(
                f"Checkpoint {self.key.digest} was written at git {prev} but the code is "
                f"now at {cur}; resuming will not be bit-identical if the fit/solver code "
                f"changed. Clear the checkpoint dir to force a clean recompute.",
                UserWarning,
                stacklevel=3,
            )

    def _write_manifest(self) -> None:
        manifest = {
            "schema_version": _CHECKPOINT_SCHEMA_VERSION,
            "key": {
                "digest": self.key.digest,
                "ref_digest": self.key.ref_digest,
                "predictor_hash": self.key.predictor_hash,
                "prs_hash": self.key.prs_hash,
                "config": self.key.config,
            },
            "provenance": {
                "git_sha": _git_sha(),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                **_dep_versions(),
            },
        }
        self._atomic_write_json(self.dir / _MANIFEST_NAME, manifest)

    def _atomic_write_json(self, path: Path, obj: Any) -> None:
        tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        with open(tmp, "w") as fh:
            json.dump(obj, fh)
        os.replace(tmp, path)

    def _sweep_tmp(self) -> None:
        """Remove killed-run tmp debris so it can't accumulate across resumes."""
        for pattern in (f".chr*{_SHARD_SUFFIX}.tmp.*", f".{_MANIFEST_NAME}.tmp.*"):
            for p in self.dir.glob(pattern):
                try:
                    p.unlink()
                except OSError:
                    pass

    # -- shard IO -----------------------------------------------------------
    def _shard_path(self, chrom: str) -> Path:
        return self.dir / f"chr{_safe_chrom(chrom)}{_SHARD_SUFFIX}"

    def save(self, chrom: str, partial: Any) -> bool:
        """Atomically persist one chromosome's partial. Best-effort → returns success."""
        if not self._ok:
            return False
        final = self._shard_path(chrom)
        tmp = final.with_name(f".{final.name}.tmp.{os.getpid()}")
        try:
            joblib.dump(partial, tmp)
            os.replace(tmp, final)
            return True
        except Exception:  # noqa: BLE001 - a checkpoint write must never break a fit
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            return False

    def load(self, chrom: str) -> Any:
        """Return a completed chromosome's partial, or ``None`` on absent/corrupt/error."""
        if not self._ok:
            return None
        final = self._shard_path(chrom)
        try:
            if not final.exists():
                return None
            return joblib.load(final)
        except Exception:  # noqa: BLE001 - a corrupt/partial shard → recompute
            return None

    def done_chromosomes(self) -> List[str]:
        """The ``_safe_chrom`` tokens of chromosomes with a committed shard (for reporting)."""
        if not self._ok:
            return []
        n = len(_SHARD_SUFFIX)
        return sorted(p.name[3:-n] for p in self.dir.glob(f"chr*{_SHARD_SUFFIX}"))


# ---------------------------------------------------------------------------
# Maintenance helpers (mirror io/stats_cache.py)
# ---------------------------------------------------------------------------
def clear_checkpoint(checkpoint_dir: Union[str, Path]) -> int:
    """Remove all checkpoint entries under ``checkpoint_dir``. Returns entries removed."""
    root = Path(checkpoint_dir)
    if not root.exists():
        return 0
    count = 0
    for child in root.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
            count += 1
        elif child.is_file():
            child.unlink()
    return count


def get_checkpoint_info(checkpoint_dir: Union[str, Path]) -> Dict[str, Any]:
    """Summarize the checkpoint dir: path, entry count, on-disk size, per-entry shard counts."""
    root = Path(checkpoint_dir)
    if not root.exists():
        return {"path": str(root), "n_entries": 0, "size_bytes": 0, "size_mb": 0.0, "entries": []}

    entries: List[Dict[str, Any]] = []
    total = 0
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        size = sum(f.stat().st_size for f in child.rglob("*") if f.is_file())
        total += size
        n_shards = len(list(child.glob(f"chr*{_SHARD_SUFFIX}")))
        entries.append({"digest": child.name, "n_shards": n_shards, "size_bytes": size})

    return {
        "path": str(root),
        "n_entries": len(entries),
        "size_bytes": total,
        "size_mb": round(total / (1024 * 1024), 2),
        "entries": entries,
    }
