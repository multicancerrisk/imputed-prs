"""Fast, hermetic end-to-end shakedown of the baseline driver.

Synthesizes a small, block-correlated VCF + a consistent PRS definition + chip variant
list (no network, no bcftools, no 20 GB panel), then exercises the full pipeline:
fit both methods -> oracle -> export -> determinism check -> a few load-only points ->
scaling projection. Proves the plumbing in seconds. Statistical quality is not the point;
correctness of the mechanics is.
"""
from __future__ import annotations

import logging
import math
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from benchmarks.harness import WorkSpec, collect_metadata, dumps, load_results, measure
from benchmarks.oracle import oracle_matches
from benchmarks.run_baseline import _fit_spec, _prs_to_csv

log = logging.getLogger("benchmarks.smoke")

_ALLELES = ["A", "C", "G", "T"]
_GT = ("0/0", "0/1", "1/1")


def make_synthetic_vcf(
    path: Path,
    n_samples: int,
    n_variants: int,
    *,
    seed: int = 0,
    chrom: str = "22",
    start: int = 16_050_000,
    step: int = 2000,
    block_size: int = 8,
) -> pd.DataFrame:
    """Write a small VCF with block-correlated genotypes; return variant info."""
    rng = np.random.default_rng(seed)
    samples = [f"S{i:04d}" for i in range(n_samples)]
    n_blocks = max(1, math.ceil(n_variants / block_size))
    tags = rng.integers(0, 3, size=(n_samples, n_blocks))  # per-block "tag" genotype
    info: List[dict] = []
    lines = [
        "##fileformat=VCFv4.2",
        f"##contig=<ID={chrom}>",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples),
    ]
    for j in range(n_variants):
        b = j // block_size
        pos = start + j * step
        pair = rng.choice(_ALLELES, size=2, replace=False)
        ref, alt = str(pair[0]), str(pair[1])
        vid = f"rs{100000 + j}"
        copy = rng.random(n_samples) < 0.7  # 70% track the block tag => LD
        indep = rng.binomial(2, 0.3, n_samples)
        dos = np.where(copy, tags[:, b], indep).astype(int)
        gt = "\t".join(_GT[d] for d in dos)
        lines.append(f"{chrom}\t{pos}\t{vid}\t{ref}\t{alt}\t.\tPASS\t.\tGT\t{gt}")
        info.append({"variant_id": vid, "chromosome": chrom, "position": pos,
                     "ref": ref, "alt": alt, "block": b})
    Path(path).write_text("\n".join(lines) + "\n")
    return pd.DataFrame(info)


def build_synth_prs_and_chip(info: pd.DataFrame, *, seed: int = 1) -> Tuple[pd.DataFrame, List[str]]:
    """A chip (≈60% of variants, ≥1 per block) and a PRS (≈50% of variants, mixed)."""
    rng = np.random.default_rng(seed)
    chip_idx = set()
    for _, grp in info.groupby("block"):
        idxs = list(grp.index)
        k = max(1, int(round(0.6 * len(idxs))))
        for i in rng.choice(idxs, size=k, replace=False):
            chip_idx.add(int(i))
    chip = [f"{info.loc[i, 'chromosome']}:{int(info.loc[i, 'position'])}" for i in sorted(chip_idx)]

    n_prs = max(4, int(round(0.5 * len(info))))
    prs_idx = sorted(int(i) for i in rng.choice(info.index, size=n_prs, replace=False))
    rows = [
        {
            "variant_id": info.loc[i, "variant_id"],
            "chromosome": info.loc[i, "chromosome"],
            "position": int(info.loc[i, "position"]),
            "effect_allele": info.loc[i, "alt"],
            "other_allele": info.loc[i, "ref"],
            "beta": float(rng.standard_normal()),
        }
        for i in prs_idx
    ]
    return pd.DataFrame(rows), chip


def run_smoke(args) -> int:
    workdir = Path(args.workdir) / "smoke"
    workdir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir) / "smoke"
    shutil.rmtree(results_dir, ignore_errors=True)  # fresh each run (no cross-run accumulation)
    results_dir.mkdir(parents=True, exist_ok=True)
    meta = collect_metadata()
    config = {"window_size": 1_000_000, "cv_folds": 3, "tuning_scope": "none", "l1_ratio": 0.5,
              "alpha": 0.01, "n_jobs": 1, "random_state": 42, "verbose": 0}

    info = make_synthetic_vcf(workdir / "synth.vcf", 40, 64, seed=0)
    prs_df, chip = build_synth_prs_and_chip(info)
    prs_csv = _prs_to_csv(prs_df, workdir / "prs.csv")
    chip_file = workdir / "chip.txt"
    chip_file.write_text("\n".join(chip) + "\n")
    print(f"[smoke] synth panel: 40 samples x 64 variants; PRS={len(prs_df)} chip={len(chip)}")

    oracles = {}
    for method in args.methods:
        spec = _fit_spec(method, workdir / "synth.vcf", prs_csv, chip_file, "GRCh38",
                         label=f"smoke_{method}", n_samples=40, n_variants=len(prs_df),
                         config=config, soft_ceiling=None, tracemalloc=False,
                         export_dir=workdir / "export" / method)
        res = measure(spec, results_dir, metadata=meta)
        ok = res.outcome == "completed" and res.result
        s = (res.result or {}).get("summary", {})
        observed = s.get("n_observed", s.get("n_observed_variants"))
        imputed = s.get("n_imputed", s.get("n_missing_variants"))
        print(f"[smoke] fit {method}: {res.outcome} peak={_mb(res.peak_rss_bytes)}MiB "
              f"wall={_f(res.wall_seconds)}s observed={observed} imputed/missing={imputed} "
              f"export_ok={(res.result or {}).get('export_ok')}")
        if not ok:
            print(f"[smoke] FAILED — stderr tail:\n{res.stderr_tail}")
            return 1
        oracles[method] = res.result

    if "imputation" in args.methods:
        spec = _fit_spec("imputation", workdir / "synth.vcf", prs_csv, chip_file, "GRCh38",
                         label="smoke_imputation_repeat", n_samples=40, n_variants=len(prs_df),
                         config=config, soft_ceiling=None, tracemalloc=False)
        res2 = measure(spec, results_dir, metadata=meta)
        # Compare the parity-relevant core only (export paths/flags are per-run volatile).
        same = oracle_matches(_core_oracle(oracles["imputation"]), _core_oracle(res2.result or {}))
        print(f"[smoke] determinism (imputation oracle reproducible): {same}")
        if not same:
            print("[smoke] FAILED — oracle not reproducible across identical fits")
            return 1

    for ns in (10, 20, 40):
        for nv in (32, 64):
            vpath = workdir / f"s{ns}_v{nv}.vcf"
            make_synthetic_vcf(vpath, ns, nv, seed=0)
            spec = WorkSpec(operation="load_genotypes", label=f"load_s{ns}_v{nv}",
                            params={"path": str(vpath)}, n_samples=ns, n_variants=nv)
            measure(spec, results_dir, metadata=meta)

    from benchmarks import scaling_projection

    report = scaling_projection.make_report(load_results(results_dir))
    (results_dir / "projection.json").write_text(dumps(report))
    print(scaling_projection.format_report(report))
    print("\n[smoke] OK")
    return 0


def _core_oracle(o: dict) -> dict:
    """Parity-relevant subset of an oracle (drops per-run-volatile export fields)."""
    return {k: o.get(k) for k in ("method", "summary", "calibration", "r2_summary", "n_units")}


def _mb(n) -> str:
    return "n/a" if n is None else f"{n / (1024 * 1024):.0f}"


def _f(v) -> str:
    return "n/a" if v is None else f"{v:.2f}"
