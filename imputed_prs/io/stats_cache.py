"""Opt-in persisted store for streaming sufficient-statistics (Phase 9 stats cache).

The streaming imputation fit accumulates, in one O(n) pass over the reference panel,
a local raw Gram block per target (:class:`~imputed_prs.compute.gram_solve.LocalGramBlock`
— ``G = ZᵀZ``, ``c = Zᵀy`` and the per-fold held-out moments). Those blocks depend only
on the **reference genotypes**, the **chip (predictor) set**, and the **window params**
— *not* on ``(alpha, l1_ratio)``. So re-tuning, cross-validation, or a re-run over the
same panel re-does the expensive accumulation to reach the same blocks.

This module lets a caller persist the collected blocks (the list
:meth:`StreamingImputationFitter.run_collect` returns) keyed on exactly those inputs, so
a later invocation can skip the stream and re-solve the cached blocks at any grid point.

**It is inert until a caller passes ``cache_dir``** (Phase-9 wiring, Commit 6). Nothing in
the default path touches disk.

What a warm hit reproduces
--------------------------
Solving a cached block reproduces the fit's **coefficients and CV R²/MSE bit-identically**
(the block round-trips through float64 ``.npz`` exactly). It does **not** carry per-sample
calibration: that is ``s_cv += (Zband @ Wk) @ coef`` against the *resident raw-dosage band*,
which is evicted per window and never stored (storing it would be the full genotype matrix).
A warm-hit consumer that needs calibration re-streams a light calibration-only pass; a
consumer that scores raw models (the sensitivity grid) needs nothing further.

Storage model
-------------
On disk each key is a directory ``<cache_dir>/<digest>/`` holding one ``.npz`` shard per
chromosome plus a ``manifest.json``. Size is linear in ``Σ_target predictors²`` — practical
for a manageable score (hundreds–thousands of targets), **not** a full-2M fit. This is a
documented reuse convenience, not a scale-out store.

The key (``ref_digest``, ``chip_set_hash``, ``window_params``) deliberately **excludes**
``alpha``/``l1_ratio`` — that is what lets a hyperparameter sweep reuse one accumulation.
Conventions (default dir under ``~/.cache/imputed_prs``, ``cache_dir`` override, silent
corrupt-tolerant reads, ``clear_*``/``*_info`` helpers) mirror :mod:`imputed_prs.io.pgs_catalog`.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

# Bump when the on-disk layout changes in a way that makes old shards unreadable /
# wrong; a manifest with a different schema_version is treated as a miss (recompute).
_CACHE_SCHEMA_VERSION = 1
_MANIFEST_NAME = "manifest.json"

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "imputed_prs" / "stats_cache"


# ---------------------------------------------------------------------------
# Key computation
# ---------------------------------------------------------------------------
def _sha1(*parts: Any) -> str:
    """SHA-1 of ``parts`` joined by ``|`` (full 40-char hexdigest)."""
    h = hashlib.sha1()
    h.update("|".join(str(p) for p in parts).encode("utf-8"))
    return h.hexdigest()


def _ref_digest(
    sample_ids: Sequence[str],
    *,
    source_file: Optional[Union[str, Path]] = None,
    n_variants: Optional[int] = None,
) -> str:
    """Identity of the reference panel underlying an accumulation.

    With a ``source_file`` on disk: ``sha1(realpath | st_size | st_mtime_ns |
    n_samples | sha1(sample_ids))`` — the file's stat plus the sample set. Without one
    (an in-RAM fold with no backing file) it degrades to ``sha1(sha1(sample_ids) |
    n_variants)``: the sample set plus, if known, the variant count. ``sample_ids`` are
    hashed in the given (deterministic) order — a reorder is a conservative miss, never a
    wrong hit (``G`` is order-invariant, so recompute is the only cost).
    """
    ids_hash = _sha1(*sample_ids)
    if source_file is not None:
        p = Path(source_file)
        try:
            st = p.stat()
            return _sha1(p.resolve(), st.st_size, st.st_mtime_ns, len(sample_ids), ids_hash)
        except OSError:
            # File named but unstattable: fall through to the sample-only identity.
            pass
    return _sha1(ids_hash, n_variants if n_variants is not None else len(sample_ids))


def _chip_set_hash(chip_ids: Sequence[str]) -> str:
    """Order-independent identity of the predictor (chip) variant set."""
    return _sha1(*sorted(chip_ids))


def _window_params_hash(
    window_size: int,
    max_predictors: Optional[int],
    cv_folds: int,
    random_state: Optional[int],
    fold_key: Optional[str],
) -> str:
    """Identity of the accumulation window params (+ any explicit fold partition)."""
    return _sha1(window_size, max_predictors, cv_folds, random_state, fold_key)


@dataclass(frozen=True)
class StatsKey:
    """The cache identity for one accumulation, split into its provenance components.

    ``digest`` (truncated combined hash) is the on-disk directory name; the full
    components are re-checked against the manifest on load, so a truncation collision
    can never surface a wrong hit — it degrades to a miss.
    """

    digest: str
    ref_digest: str
    chip_set_hash: str
    window_params: str
    window: Dict[str, Any]


def make_stats_key(
    *,
    sample_ids: Sequence[str],
    chip_ids: Sequence[str],
    window_size: int,
    max_predictors: Optional[int],
    cv_folds: int,
    random_state: Optional[int],
    source_file: Optional[Union[str, Path]] = None,
    n_variants: Optional[int] = None,
    fold_key: Optional[str] = None,
) -> StatsKey:
    """Build the :class:`StatsKey` for an accumulation over ``sample_ids`` / ``chip_ids``.

    ``alpha``/``l1_ratio`` are intentionally absent — the whole grid shares one key.
    ``fold_key`` distinguishes reference-CV fold partitions that share window params.
    """
    ref = _ref_digest(sample_ids, source_file=source_file, n_variants=n_variants)
    chip = _chip_set_hash(chip_ids)
    win = _window_params_hash(window_size, max_predictors, cv_folds, random_state, fold_key)
    digest = _sha1(_CACHE_SCHEMA_VERSION, ref, chip, win)[:16]
    return StatsKey(
        digest=digest,
        ref_digest=ref,
        chip_set_hash=chip,
        window_params=win,
        window={
            "window_size": window_size,
            "max_predictors": max_predictors,
            "cv_folds": cv_folds,
            "random_state": random_state,
            "fold_key": fold_key,
        },
    )


# ---------------------------------------------------------------------------
# Cache dir + provenance
# ---------------------------------------------------------------------------
def _get_cache_dir(cache_dir: Optional[Union[str, Path]] = None) -> Path:
    """Resolve (and create) the cache root."""
    root = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    root.mkdir(parents=True, exist_ok=True)
    return root


def _git_sha() -> str:
    """Best-effort short git SHA for the provenance stamp (mirrors benchmarks/harness)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:  # noqa: BLE001 - provenance is informational, never fatal
        return "unknown"


# ---------------------------------------------------------------------------
# Serialization of a collected-block list
# ---------------------------------------------------------------------------
def _target_record(collected) -> Dict[str, Any]:
    """The JSON-serializable spec/scalar record for one collected target."""
    spec = collected.spec
    return {
        "prs_variant_id": spec.prs_variant_id,
        "chromosome": str(spec.chromosome),
        "position": int(spec.position),
        "effect_allele": spec.effect_allele,
        "other_allele": spec.other_allele,
        "beta": float(spec.beta),
        "effect_flip": bool(spec.effect_flip),
        "af": float(collected.af),
        "is_fallback": bool(collected.is_fallback),
        "n_folds": len(collected.block.fold_n),
    }


def _target_arrays(idx: int, collected) -> Dict[str, np.ndarray]:
    """The ``.npz`` arrays (heavy Gram data) for one collected target, keyed ``t{idx}_*``."""
    b = collected.block
    p = f"t{idx}_"
    n_folds = len(b.fold_n)
    arrays: Dict[str, np.ndarray] = {
        p + "G": np.asarray(b.G, dtype=np.float64),
        p + "c": np.asarray(b.c, dtype=np.float64),
        p + "zsum": np.asarray(b.zsum, dtype=np.float64),
        p + "zsqsum": np.asarray(b.zsqsum, dtype=np.float64),
        p + "pred_idx": np.asarray(collected.pred_idx, dtype=np.int64),
        p + "pred_af": np.asarray(collected.pred_af, dtype=np.float64),
        p + "scalars": np.array([b.n, b.ysum, b.ysqsum], dtype=np.float64),
    }
    if n_folds:
        arrays[p + "fold_G"] = np.stack([np.asarray(g, dtype=np.float64) for g in b.fold_G])
        arrays[p + "fold_c"] = np.stack([np.asarray(c, dtype=np.float64) for c in b.fold_c])
        arrays[p + "fold_zsum"] = np.stack(
            [np.asarray(z, dtype=np.float64) for z in b.fold_zsum]
        )
        arrays[p + "fold_zsqsum"] = np.stack(
            [np.asarray(z, dtype=np.float64) for z in b.fold_zsqsum]
        )
        arrays[p + "fold_scalars"] = np.array(
            [[b.fold_ysum[k], b.fold_ysqsum[k], b.fold_n[k]] for k in range(n_folds)],
            dtype=np.float64,
        )
    return arrays


def store_collected(
    key: StatsKey,
    collected: Sequence,
    *,
    cache_dir: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    """Persist a ``run_collect`` block list under ``key``. Returns the entry dir (or None).

    Shards by chromosome (one ``.npz`` per chromosome); writes ``manifest.json`` **last**
    so an interrupted write leaves no manifest → a later load misses and recomputes. A
    pre-existing entry for the same key is replaced. Never raises: a failed write returns
    ``None`` (the caller keeps the freshly-computed blocks it already has).
    """
    try:
        root = _get_cache_dir(cache_dir)
        entry = root / key.digest
        if entry.exists():
            shutil.rmtree(entry)
        entry.mkdir(parents=True, exist_ok=True)

        # Group targets by chromosome, preserving run_collect's append order globally.
        by_chrom: Dict[str, List[Any]] = {}
        chrom_order: List[str] = []
        for c in collected:
            chrom = str(c.spec.chromosome)
            if chrom not in by_chrom:
                by_chrom[chrom] = []
                chrom_order.append(chrom)
            by_chrom[chrom].append(c)

        targets_manifest: Dict[str, List[Dict[str, Any]]] = {}
        shards: Dict[str, str] = {}
        for chrom in chrom_order:
            items = by_chrom[chrom]
            arrays: Dict[str, np.ndarray] = {}
            records: List[Dict[str, Any]] = []
            for i, c in enumerate(items):
                arrays.update(_target_arrays(i, c))
                records.append(_target_record(c))
            shard_name = f"chr{chrom}.npz"
            with open(entry / shard_name, "wb") as fh:
                np.savez(fh, **arrays)
            shards[chrom] = shard_name
            targets_manifest[chrom] = records

        manifest = {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "key": {
                "digest": key.digest,
                "ref_digest": key.ref_digest,
                "chip_set_hash": key.chip_set_hash,
                "window_params": key.window_params,
                "window": key.window,
            },
            "provenance": {
                "git_sha": _git_sha(),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "numpy": np.__version__,
            },
            "n_targets": int(len(collected)),
            "chromosomes": chrom_order,
            "shards": shards,
            "targets": targets_manifest,
        }
        with open(entry / _MANIFEST_NAME, "w") as fh:
            json.dump(manifest, fh)
        return entry
    except Exception:  # noqa: BLE001 - a cache write must never break a fit
        return None


def _rebuild_target(record: Dict[str, Any], npz, idx: int):
    """Reconstruct one ``_CollectedTarget`` from its manifest record + shard arrays."""
    from imputed_prs.compute.gram_solve import LocalGramBlock
    from imputed_prs.compute.sufficient_stats import TargetVar, _CollectedTarget

    p = f"t{idx}_"
    scalars = npz[p + "scalars"]
    n = int(round(float(scalars[0])))
    ysum = float(scalars[1])
    ysqsum = float(scalars[2])
    n_folds = int(record["n_folds"])

    fold_G: List[np.ndarray] = []
    fold_c: List[np.ndarray] = []
    fold_zsum: List[np.ndarray] = []
    fold_zsqsum: List[np.ndarray] = []
    fold_ysum: List[float] = []
    fold_ysqsum: List[float] = []
    fold_n: List[int] = []
    if n_folds:
        fg, fc = npz[p + "fold_G"], npz[p + "fold_c"]
        fzs, fzq = npz[p + "fold_zsum"], npz[p + "fold_zsqsum"]
        fsc = npz[p + "fold_scalars"]
        for k in range(n_folds):
            fold_G.append(np.array(fg[k]))
            fold_c.append(np.array(fc[k]))
            fold_zsum.append(np.array(fzs[k]))
            fold_zsqsum.append(np.array(fzq[k]))
            fold_ysum.append(float(fsc[k, 0]))
            fold_ysqsum.append(float(fsc[k, 1]))
            fold_n.append(int(round(float(fsc[k, 2]))))

    block = LocalGramBlock(
        n=n,
        G=np.array(npz[p + "G"]),
        c=np.array(npz[p + "c"]),
        zsum=np.array(npz[p + "zsum"]),
        zsqsum=np.array(npz[p + "zsqsum"]),
        ysum=ysum,
        ysqsum=ysqsum,
        fold_G=fold_G,
        fold_c=fold_c,
        fold_zsum=fold_zsum,
        fold_zsqsum=fold_zsqsum,
        fold_ysum=fold_ysum,
        fold_ysqsum=fold_ysqsum,
        fold_n=fold_n,
    )
    spec = TargetVar(
        prs_variant_id=record["prs_variant_id"],
        chromosome=record["chromosome"],
        position=int(record["position"]),
        effect_allele=record["effect_allele"],
        other_allele=record["other_allele"],
        beta=float(record["beta"]),
        effect_flip=bool(record["effect_flip"]),
    )
    return _CollectedTarget(
        spec=spec,
        af=float(record["af"]),
        block=block,
        pred_idx=np.array(npz[p + "pred_idx"]),
        pred_af=np.array(npz[p + "pred_af"]),
        is_fallback=bool(record["is_fallback"]),
    )


def load_collected(
    key: StatsKey,
    *,
    cache_dir: Optional[Union[str, Path]] = None,
) -> Optional[List]:
    """Load the collected block list for ``key``, or ``None`` on any miss.

    A miss is: no entry dir, missing/corrupt manifest, schema-version mismatch, any key
    component disagreeing with the manifest (invalidation), or a missing/corrupt shard.
    Every failure is swallowed → ``None`` so the caller silently recomputes.
    """
    try:
        root = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
        entry = root / key.digest
        manifest_path = entry / _MANIFEST_NAME
        if not manifest_path.exists():
            return None
        with open(manifest_path) as fh:
            manifest = json.load(fh)

        if manifest.get("schema_version") != _CACHE_SCHEMA_VERSION:
            return None
        mkey = manifest.get("key", {})
        if (
            mkey.get("ref_digest") != key.ref_digest
            or mkey.get("chip_set_hash") != key.chip_set_hash
            or mkey.get("window_params") != key.window_params
        ):
            return None  # key-change invalidation (a truncation collision lands here too)

        collected: List[Any] = []
        for chrom in manifest["chromosomes"]:
            shard = entry / manifest["shards"][chrom]
            records = manifest["targets"][chrom]
            with np.load(shard) as npz:
                for i, record in enumerate(records):
                    collected.append(_rebuild_target(record, npz, i))
        return collected
    except Exception:  # noqa: BLE001 - corrupt/partial cache → silent recompute
        return None


# ---------------------------------------------------------------------------
# Maintenance helpers (mirror io/pgs_catalog.py)
# ---------------------------------------------------------------------------
def clear_stats_cache(cache_dir: Optional[Union[str, Path]] = None) -> int:
    """Remove all cached accumulations. Returns the number of entries removed."""
    root = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
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


def get_stats_cache_info(cache_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Summarize the stats cache: path, entry count, on-disk size, and per-entry digests."""
    root = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    if not root.exists():
        return {
            "path": str(root),
            "n_entries": 0,
            "size_bytes": 0,
            "size_mb": 0.0,
            "entries": [],
        }

    entries: List[Dict[str, Any]] = []
    total = 0
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        size = sum(f.stat().st_size for f in child.rglob("*") if f.is_file())
        total += size
        n_targets = None
        try:
            with open(child / _MANIFEST_NAME) as fh:
                n_targets = json.load(fh).get("n_targets")
        except Exception:  # noqa: BLE001 - a corrupt entry still counts toward size
            pass
        entries.append({"digest": child.name, "n_targets": n_targets, "size_bytes": size})

    return {
        "path": str(root),
        "n_entries": len(entries),
        "size_bytes": total,
        "size_mb": round(total / (1024 * 1024), 2),
        "entries": entries,
    }
