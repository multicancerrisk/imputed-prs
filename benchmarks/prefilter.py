"""Self-contained bcftools prefilter for the benchmark panel.

Lifted and generalized from ``analysis/prs313_evaluation/evaluate_prs313.py`` (which lives
under the git-ignored ``analysis/`` tree, so it is copied here rather than imported).
Differences from the original:

* parameterized file pattern / directory (targets the GRCh38 NYGC high-coverage panel);
* uses ``bcftools view -R`` (index seek — the ``.tbi`` files are published) instead of the
  whole-file ``-T`` scan;
* auto-detects the VCF contig prefix (GRCh38 panels use ``chr1``; GRCh37 used ``1``) when
  writing the regions file, since bcftools does not normalize contig names;
* caches per-chromosome filtered output so growing the chromosome set is incremental.

Phase 1 is expected to replace this external step with a streaming/tabix loader inside the
library; until then this keeps the benchmark reproducible without the ``analysis/`` tree.
"""
from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: GRCh38 NYGC high-coverage 3,202-sample phased panel, per chromosome.
GRCH38_HIGHCOV_PATTERN = (
    "1kGP_high_coverage_Illumina.chr{chrom}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz"
)
AUTOSOMES = [str(c) for c in range(1, 23)]


class BcftoolsError(RuntimeError):
    pass


def require_bcftools() -> str:
    path = shutil.which("bcftools")
    if path is None:
        raise BcftoolsError("bcftools not found on PATH (required for the benchmark prefilter).")
    return path


def _run(args: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run([str(a) for a in args], capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise BcftoolsError(f"command failed ({proc.returncode}): {' '.join(map(str, args))}\n{proc.stderr}")
    return proc


def _norm_chrom(chrom: str) -> str:
    c = str(chrom).strip()
    if c.lower().startswith("chr"):
        c = c[3:]
    # Repair float artifacts like "22.0" that leak from pandas' low_memory chunked dtype
    # inference on the large PGS000027 harmonized file (its hm_chr column is mixed-type).
    if "." in c and c.replace(".", "", 1).isdigit():
        c = str(int(float(c)))
    return "M" if c in ("MT", "M") else c


def needed_positions(
    prs_df: pd.DataFrame,
    platform_variants: Set[str],
    *,
    window_size: int = 1_000_000,
    restrict_chroms: Optional[Set[str]] = None,
) -> Dict[str, Set[int]]:
    """Per-chromosome positions to extract: all PRS positions ∪ platform variants within
    ±``window_size`` of any PRS variant. Chromosome keys are normalized (no ``chr``)."""
    restrict = {_norm_chrom(c) for c in restrict_chroms} if restrict_chroms else None
    prs_windows: Dict[str, List[tuple]] = {}
    prs_positions: Dict[str, Set[int]] = {}
    for _, row in prs_df.iterrows():
        chrom = _norm_chrom(row["chromosome"])
        if restrict is not None and chrom not in restrict:
            continue
        pos = int(row["position"])
        prs_positions.setdefault(chrom, set()).add(pos)
        prs_windows.setdefault(chrom, []).append((max(1, pos - window_size), pos + window_size))

    platform_by_chrom: Dict[str, Set[int]] = {}
    for var_id in platform_variants:
        parts = str(var_id).split(":")
        if len(parts) >= 2:
            chrom = _norm_chrom(parts[0])
            if restrict is not None and chrom not in restrict:
                continue
            try:
                platform_by_chrom.setdefault(chrom, set()).add(int(parts[1]))
            except ValueError:
                continue

    needed: Dict[str, Set[int]] = {}
    for chrom, windows in prs_windows.items():
        positions = set(prs_positions.get(chrom, set()))
        plat = platform_by_chrom.get(chrom, set())
        if plat:
            arr = np.array(sorted(plat))
            for lo, hi in windows:
                positions.update(arr[(arr >= lo) & (arr <= hi)].tolist())
        if positions:
            needed[chrom] = positions
    return needed


def find_kg_vcf(kg_dir: Path, chrom: str, pattern: str = GRCH38_HIGHCOV_PATTERN) -> Optional[Path]:
    chrom = _norm_chrom(chrom)
    for name in (pattern.format(chrom=chrom), pattern.format(chrom=f"chr{chrom}")):
        p = kg_dir / name
        if p.exists() and p.stat().st_size > 1000:
            return p
    # glob fallback on the invariant tail
    for match in sorted(kg_dir.glob(f"*chr{chrom}.*.vcf.gz")):
        if match.stat().st_size > 1000:
            return match
    return None


def detect_chrom_prefix(vcf: Path) -> str:
    """Return ``"chr"`` if the VCF's contigs are ``chr``-prefixed, else ``""``."""
    proc = _run(["bcftools", "view", "-h", str(vcf)], check=False)
    for line in proc.stdout.splitlines():
        if line.startswith("##contig="):
            # ##contig=<ID=chr1,...>
            seg = line.split("ID=", 1)[1]
            contig = seg.split(",", 1)[0].rstrip(">")
            return "chr" if contig.lower().startswith("chr") else ""
    return ""


def list_samples(vcf: Path) -> List[str]:
    proc = _run(["bcftools", "query", "-l", str(vcf)])
    return [s for s in proc.stdout.splitlines() if s.strip()]


def subsample_samples(all_ids: Sequence[str], n: int, *, seed: int = 42) -> List[str]:
    """Deterministic sample subset (seeded shuffle of sorted IDs)."""
    ids = sorted(all_ids)
    if n >= len(ids):
        return ids
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ids))[:n]
    return [ids[i] for i in sorted(idx)]


def subset_samples_vcf(in_vcf: Path, sample_ids: Sequence[str], out_vcf: Path) -> Path:
    """Write ``out_vcf`` containing only ``sample_ids`` (via ``bcftools view -S``).

    The public ``fit`` loads every sample in its reference VCF, so varying n_samples in the
    blocker sweep means baking the subset into the VCF here.
    """
    require_bcftools()
    out_vcf = Path(out_vcf)
    out_vcf.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        sfile = Path(fh.name)
        fh.write("\n".join(sample_ids) + "\n")
    try:
        _run(
            ["bcftools", "view", "-S", str(sfile), "--force-samples",
             "-O", "z", "-o", str(out_vcf), str(in_vcf)]
        )
    finally:
        sfile.unlink(missing_ok=True)
    _run(["bcftools", "index", "-t", str(out_vcf)], check=False)
    return out_vcf


def _filter_one_chrom(chrom: str, positions: Set[int], vcf: Path, out_vcf: Path) -> bool:
    prefix = detect_chrom_prefix(vcf)
    with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False) as fh:
        regions = Path(fh.name)
        for pos in sorted(positions):
            fh.write(f"{prefix}{chrom}\t{pos}\n")
    try:
        # -R uses the .tbi index (seek) rather than scanning the whole file.
        proc = _run(
            ["bcftools", "view", "-R", str(regions), "-O", "z", "-o", str(out_vcf), str(vcf)],
            check=False,
        )
        if proc.returncode != 0:  # fall back to the streaming -T targets scan
            proc = _run(
                ["bcftools", "view", "-T", str(regions), "-O", "z", "-o", str(out_vcf), str(vcf)],
                check=False,
            )
        if proc.returncode != 0:
            log.warning("bcftools failed for chr%s: %s", chrom, proc.stderr)
            return False
    finally:
        regions.unlink(missing_ok=True)
    _run(["bcftools", "index", "-t", str(out_vcf)], check=False)
    return True


def prefilter_vcf(
    needed: Dict[str, Set[int]],
    kg_dir: Path,
    out_vcf: Path,
    *,
    file_pattern: str = GRCH38_HIGHCOV_PATTERN,
    per_chrom_cache: Optional[Path] = None,
) -> Path:
    """Extract ``needed`` positions from per-chromosome VCFs into a single ``out_vcf``.

    Per-chromosome filtered files are cached (positions for a chromosome are stable across
    grid cells), so adding a chromosome to a sweep only filters the new one.
    """
    require_bcftools()
    kg_dir = Path(kg_dir)
    out_vcf = Path(out_vcf)
    out_vcf.parent.mkdir(parents=True, exist_ok=True)
    cache = Path(per_chrom_cache) if per_chrom_cache else out_vcf.parent / "per_chrom_cache"
    cache.mkdir(parents=True, exist_ok=True)

    chrom_vcfs: List[str] = []
    for chrom in sorted(needed, key=lambda c: int(c) if c.isdigit() else 99):
        positions = needed[chrom]
        pos_hash = hashlib.sha1(",".join(map(str, sorted(positions))).encode()).hexdigest()[:12]
        cached = cache / f"chr{chrom}.{pos_hash}.vcf.gz"
        if not (cached.exists() and cached.stat().st_size > 0):
            vcf = find_kg_vcf(kg_dir, chrom, file_pattern)
            if vcf is None:
                log.warning("no VCF for chr%s in %s; skipping", chrom, kg_dir)
                continue
            log.info("chr%s: extracting %d positions from %s", chrom, len(positions), vcf.name)
            if not _filter_one_chrom(chrom, positions, vcf, cached):
                continue
        chrom_vcfs.append(str(cached))

    if not chrom_vcfs:
        raise BcftoolsError("no variants extracted from any chromosome")

    proc = _run(
        ["bcftools", "concat", "-O", "z", "-o", str(out_vcf), "--naive", *chrom_vcfs],
        check=False,
    )
    if proc.returncode != 0:  # --naive requires identical headers; fall back
        _run(["bcftools", "concat", "-O", "z", "-o", str(out_vcf), *chrom_vcfs])
    _run(["bcftools", "index", "-t", str(out_vcf)], check=False)
    return out_vcf
