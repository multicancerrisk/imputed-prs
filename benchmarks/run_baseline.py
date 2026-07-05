"""Phase-0 baseline + blocker driver.

Modes (``--mode``):

* ``confirm``  — fetch PGS000027 (scale, ~2.1M) and PGS000004 (PRS-313) on GRCh38, report
  advertised vs loaded variant counts. Offline-graceful.
* ``baseline`` — fit both methods on PRS-313 + the GRCh38 panel; capture the
  statistical-parity **oracle** (counts, R² summary, calibration, predict probe) plus
  per-phase wall + peak RSS. Written to ``results/baseline/{method}.json``.
* ``blocker``  — grow PGS000027 by chromosome and by sample count until RAM/time explodes,
  each cell in an isolated subprocess with a soft memory ceiling; attribute the failure.
* ``all``      — confirm + baseline + blocker, then scaling projection.

``--smoke`` runs the whole pipeline fast on a tiny slice (no 20 GB panel).

Everything is driven through the library's public API; nothing in ``imputed_prs/`` is
modified. Real-data modes require the GRCh38 panel (see ``benchmarks/data_prep/``) and
``bcftools``.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from benchmarks import grid, prefilter
from benchmarks.harness import (
    DEFAULT_RESULTS_DIR,
    MeasurementResult,
    RunMetadata,
    WorkSpec,
    collect_metadata,
    dumps,
    measure,
)
from benchmarks.oracle import extract_oracle, oracle_matches

log = logging.getLogger("benchmarks.run_baseline")

PGS_SCALE = "PGS000027"  # BMI, ~2.1M variants (GRCh38 harmonized)
PGS_BASELINE = "PGS000004"  # PRS-313, 313 variants
GENOME_BUILD = "GRCh38"
_BENCH_DATA = Path(__file__).resolve().parent / "data"
DEFAULT_KG_DIR = _BENCH_DATA / "1kg_grch38"
DEFAULT_CHIP_FILE = _BENCH_DATA / "23andme_v5_GRCh38_variants.txt"


# --------------------------------------------------------------------------------------
# PRS / chip helpers
# --------------------------------------------------------------------------------------
def _fetch_prs(pgs_id: str, build: str):
    from imputed_prs import fetch_pgs_catalog_score

    df, meta = fetch_pgs_catalog_score(pgs_id, genome_build=build)
    # Repair mixed-type chromosome values ("22.0" vs "22") from the library's low_memory
    # CSV read of the large harmonized file, so both the prefilter and the CSV handed back
    # to the library carry clean, matchable chromosomes. (Library-side fix: read the PGS
    # scoring file with low_memory=False / dtype={"hm_chr": str} in io/pgs_catalog.py.)
    df = df.copy()
    df["chromosome"] = df["chromosome"].map(prefilter._norm_chrom)
    return df, meta


def _fetch_metadata(pgs_id: str):
    from imputed_prs.io.pgs_catalog import fetch_pgs_catalog_metadata

    return fetch_pgs_catalog_metadata(pgs_id)


def _prs_to_csv(df, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _load_chip_set(chip_file: Path) -> Set[str]:
    if not Path(chip_file).exists():
        raise FileNotFoundError(
            f"GRCh38 chip file not found: {chip_file}. "
            "Generate it with benchmarks/data_prep/liftover_chip.py."
        )
    with open(chip_file) as fh:
        return {ln.strip() for ln in fh if ln.strip()}


def _prs_chroms(prs_df) -> List[str]:
    chroms = {prefilter._norm_chrom(c) for c in prs_df["chromosome"]}
    return sorted(chroms, key=lambda c: int(c) if c.isdigit() else 99)


# --------------------------------------------------------------------------------------
# Reference preparation (positions prefilter + sample subset)
# --------------------------------------------------------------------------------------
def _prepare_reference(
    prs_df,
    chip_set: Set[str],
    chroms: Sequence[str],
    n_samples: Optional[int],
    kg_dir: Path,
    workdir: Path,
    file_pattern: str,
    window_size: int,
) -> Tuple[Path, int, int]:
    """Return ``(reference_vcf, n_samples_used, n_positions)`` for a cell."""
    needed = prefilter.needed_positions(
        prs_df, chip_set, window_size=window_size, restrict_chroms=set(chroms)
    )
    n_positions = sum(len(v) for v in needed.values())
    key = "+".join(chroms)
    pos_vcf = workdir / "pos" / f"{key}.vcf.gz"
    prefilter.prefilter_vcf(
        needed, kg_dir, pos_vcf, file_pattern=file_pattern, per_chrom_cache=workdir / "per_chrom_cache"
    )
    all_samples = prefilter.list_samples(pos_vcf)
    if n_samples and n_samples < len(all_samples):
        subset = prefilter.subsample_samples(all_samples, n_samples)
        cell_vcf = workdir / "cells" / f"{key}_s{n_samples}.vcf.gz"
        prefilter.subset_samples_vcf(pos_vcf, subset, cell_vcf)
        return cell_vcf, len(subset), n_positions
    return pos_vcf, len(all_samples), n_positions


def _restricted_prs(prs_df, chroms, workdir: Path, pgs_id: str, build: str, cache: Dict) -> Tuple:
    """PRS restricted to ``chroms`` (+ cached CSV). The blocker grows the problem by
    chromosome, so each cell must see only the PRS variants on its active chromosomes —
    otherwise the fit processes the whole genome-wide PRS against a partial reference."""
    key = "+".join(chroms)
    if key in cache:
        return cache[key]
    keep = {prefilter._norm_chrom(c) for c in chroms}
    sub = prs_df[prs_df["chromosome"].map(prefilter._norm_chrom).isin(keep)].copy()
    csv = _prs_to_csv(sub, workdir / "prs" / f"{pgs_id}_{build}_{key}.csv")
    cache[key] = (sub, csv)
    return sub, csv


def _fit_config(args) -> Dict:
    return {
        "window_size": args.window_size,
        "cv_folds": args.cv_folds,
        "tuning_scope": "none",
        "l1_ratio": 0.5,
        "alpha": 0.01,
        "n_jobs": args.n_jobs,
        "random_state": args.random_state,
        "verbose": 1,
    }


def _fit_spec(method, ref_vcf, prs_csv, chip_file, build, *, label, n_samples, n_variants,
              config, soft_ceiling, tracemalloc, probe=None, export_dir=None,
              reference_panel_id="1000G_highcov_GRCh38", training_ancestry="ALL") -> WorkSpec:
    params = {
        "method": method,
        "reference_genotypes": str(ref_vcf),
        "prs_definition": str(prs_csv),
        "platform_variants_file": str(chip_file),
        "genome_build": build,
        # Provenance is required for a deployable JSON export (json_export.py).
        "reference_panel_id": reference_panel_id,
        "training_ancestry": training_ancestry,
    }
    if probe is not None:
        params["predict_probe"] = probe
    if export_dir is not None:
        params["export_dir"] = str(export_dir)
    return WorkSpec(
        operation="fit", label=label, params=params, config=config,
        n_samples=n_samples, n_variants=n_variants, seed=config.get("random_state"),
        soft_ceiling_bytes=soft_ceiling, tracemalloc=tracemalloc,
    )


# --------------------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------------------
def cmd_confirm(args) -> Dict:
    out: Dict[str, Dict] = {}
    for pgs in (args.pgs_scale, args.pgs_baseline):
        entry: Dict = {"pgs_id": pgs, "genome_build": args.genome_build}
        try:
            meta = _fetch_metadata(pgs)
            entry["advertised_variants"] = getattr(meta, "variants_number", None)
        except Exception as exc:
            entry["status"] = "metadata_unavailable"
            entry["error"] = str(exc)
            out[pgs] = entry
            log.warning("metadata fetch failed for %s: %s", pgs, exc)
            continue
        try:
            df, _ = _fetch_prs(pgs, args.genome_build)
            entry["loaded_rows"] = int(len(df))
            entry["status"] = "ok"
        except Exception as exc:
            entry["status"] = "score_unavailable"
            entry["error"] = str(exc)
        out[pgs] = entry

    (args.results_dir).mkdir(parents=True, exist_ok=True)
    (args.results_dir / "confirm_pgs.json").write_text(dumps(out))
    print("PGS confirmation:")
    for pgs, e in out.items():
        print(f"  {pgs}: advertised={e.get('advertised_variants')} "
              f"loaded={e.get('loaded_rows')} status={e.get('status')}")
    scale = out.get(args.pgs_scale, {})
    if scale.get("advertised_variants") and scale["advertised_variants"] < 1_500_000:
        log.warning("%s advertises only %s variants (<1.5M)", args.pgs_scale,
                    scale["advertised_variants"])
    return out


def cmd_baseline(args, meta: RunMetadata) -> Dict:
    prs_df, _ = _fetch_prs(args.pgs_baseline, args.genome_build)
    prs_csv = _prs_to_csv(prs_df, args.workdir / "prs" / f"{args.pgs_baseline}_{args.genome_build}.csv")
    chip_set = _load_chip_set(args.chip_file)
    chroms = _prs_chroms(prs_df)
    ref_vcf, n_used, n_pos = _prepare_reference(
        prs_df, chip_set, chroms, args.baseline_samples, args.kg_dir, args.workdir,
        args.file_pattern, args.window_size,
    )
    log.info("baseline reference: %s (%d samples, %d positions)", ref_vcf.name, n_used, n_pos)

    baseline_dir = args.results_dir / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Dict] = {}
    for method in args.methods:
        spec = _fit_spec(
            method, ref_vcf, prs_csv, args.chip_file, args.genome_build,
            label=f"baseline_{method}", n_samples=n_used, n_variants=len(prs_df),
            config=_fit_config(args), soft_ceiling=None, tracemalloc=False,
            export_dir=args.workdir / "export" / method,
        )
        res = measure(spec, args.results_dir, timeout_s=args.cell_timeout_s, metadata=meta)
        record = {
            "oracle": res.result,
            "measurement": {
                "outcome": res.outcome,
                "wall_seconds": res.wall_seconds,
                "peak_rss_bytes": res.peak_rss_bytes,
                "phases": [p.__dict__ for p in res.phases],
            },
            "config": spec.config,
            "reference": {"vcf": ref_vcf.name, "n_samples": n_used, "n_positions": n_pos,
                          "pgs_id": args.pgs_baseline, "genome_build": args.genome_build},
        }
        (baseline_dir / f"{method}.json").write_text(dumps(record))
        summary[method] = record
        _print_baseline(method, res)
    return summary


def _print_baseline(method: str, res: MeasurementResult) -> None:
    if res.outcome != "completed" or not res.result:
        print(f"  {method}: {res.outcome} (no oracle)")
        return
    o = res.result
    s = o.get("summary", {})
    cal = o.get("calibration") or {}
    r2 = o.get("r2_summary", {})
    # imputation and projection use different summary key names
    observed = s.get("n_observed", s.get("n_observed_variants"))
    imputed = s.get("n_imputed", s.get("n_missing_variants"))
    icept = s.get("n_intercept_only", s.get("n_intercept_only_regions"))
    print(f"  {method}: observed={observed} imputed/missing={imputed} intercept_only={icept} "
          f"| mean_r2={_fmt(r2.get('mean'))} | calib_slope={_fmt(cal.get('scaling_factor'))} "
          f"| export_ok={o.get('export_ok')} | peak={_mb(res.peak_rss_bytes)} MiB "
          f"wall={_fmt(res.wall_seconds)}s")


def cmd_blocker(args, meta: RunMetadata) -> List[MeasurementResult]:
    prs_df, _ = _fetch_prs(args.pgs_scale, args.genome_build)
    chip_set = _load_chip_set(args.chip_file)
    prs_cache: Dict[str, Tuple] = {}

    chrom_order = tuple(args.chroms) if args.chroms else grid.DEFAULT_CHROM_ORDER
    cells = grid.order_for_growth(
        list(grid.iter_grid(chrom_order=chrom_order, sample_sizes=args.samples, methods=args.methods,
                            include_load_only=not args.no_load_only))
    )
    ceiling = int(args.mem_limit_gb * 1e9) if args.mem_limit_gb else _default_ceiling(meta)
    log.info("blocker: %d cells, soft ceiling %.1f GB", len(cells), ceiling / 1e9)

    failed_sweeps: Set[Tuple[str, str, int]] = set()
    results: List[MeasurementResult] = []
    for cell in cells:
        skey = grid.sweep_key(cell)
        if skey in failed_sweeps:
            log.info("skip %s (earlier cell in sweep already failed)", cell.label)
            continue
        cell_prs, cell_prs_csv = _restricted_prs(
            prs_df, cell.chroms, args.workdir, args.pgs_scale, args.genome_build, prs_cache
        )
        try:
            ref_vcf, n_used, n_pos = _prepare_reference(
                cell_prs, chip_set, cell.chroms, cell.n_samples, args.kg_dir, args.workdir,
                args.file_pattern, args.window_size,
            )
        except Exception as exc:
            log.warning("prefilter failed for %s: %s", cell.label, exc)
            continue
        if cell.kind == "load_only":
            spec = WorkSpec(
                operation="load_genotypes", label=cell.label, params={"path": str(ref_vcf)},
                n_samples=n_used, n_variants=n_pos, soft_ceiling_bytes=ceiling,
                tracemalloc=args.tracemalloc,
            )
        else:
            spec = _fit_spec(
                cell.method, ref_vcf, cell_prs_csv, args.chip_file, args.genome_build,
                label=cell.label, n_samples=n_used, n_variants=n_pos,
                config=_fit_config(args), soft_ceiling=ceiling, tracemalloc=args.tracemalloc,
            )
        res = measure(spec, args.results_dir, timeout_s=args.cell_timeout_s, metadata=meta)
        results.append(res)
        _print_cell(cell, res, n_pos)
        if res.outcome != "completed":
            failed_sweeps.add(skey)
            _print_attribution(res)
    return results


def _print_cell(cell: grid.Cell, res: MeasurementResult, n_pos: int) -> None:
    loaded = (res.result or {}).get("n_variants") if res.result else None
    print(f"  {cell.label:32s} {res.outcome:22s} peak={_mb(res.peak_rss_bytes):>8} MiB "
          f"wall={_fmt(res.wall_seconds):>7}s pos={n_pos} loaded={loaded}")


def _print_attribution(res: MeasurementResult) -> None:
    if res.per_site:
        print("      top memory sites:")
        for s in res.per_site[:5]:
            print(f"        {Path(s.filename).name}:{s.lineno}  {_mb(s.size_bytes)} MiB")


def _profile_one_fit(args, meta: RunMetadata) -> Dict:
    """cProfile a small fit to attribute wall-time to the O(n_variants) pandas hotspots."""
    import cProfile
    import pstats

    from benchmarks.scenarios import _fit_kwargs, _make_model

    prs_df, _ = _fetch_prs(args.pgs_scale, args.genome_build)
    chip_set = _load_chip_set(args.chip_file)
    cell_prs, cell_prs_csv = _restricted_prs(prs_df, ("22",), args.workdir, args.pgs_scale,
                                             args.genome_build, {})
    ref_vcf, n_used, n_pos = _prepare_reference(
        cell_prs, chip_set, ("22",), 500, args.kg_dir, args.workdir, args.file_pattern, args.window_size
    )
    model = _make_model("imputation", _fit_config(args))
    params = {"method": "imputation", "reference_genotypes": str(ref_vcf),
              "prs_definition": str(cell_prs_csv), "platform_variants_file": str(args.chip_file),
              "genome_build": args.genome_build}
    pr = cProfile.Profile()
    pr.enable()
    model.fit(**_fit_kwargs(params))
    pr.disable()

    st = pstats.Stats(pr)
    hotspots = ("filter_to_local_window", "match_oriented_dosage", "merge_variant_windows",
                "_normalize_chromosome", "build_reference_allele_index")
    found = {}
    total = sum(v[3] for v in st.stats.values())  # cumulative time root approximation
    for (fname, lineno, func), (cc, nc, tt, ct, callers) in st.stats.items():
        if func in hotspots:
            found[func] = {"file": fname, "line": lineno, "ncalls": nc,
                           "tottime_s": tt, "cumtime_s": ct}
    report = {"reference": {"chroms": ["22"], "n_samples": n_used, "n_positions": n_pos},
              "hotspots": found}
    (args.results_dir).mkdir(parents=True, exist_ok=True)
    (args.results_dir / "profile_imputation_chr22_s500.json").write_text(dumps(report))
    print("cProfile hotspots (imputation fit, chr22, 500 samples):")
    for func, d in sorted(found.items(), key=lambda kv: kv[1]["cumtime_s"], reverse=True):
        print(f"  {func:28s} cumtime={d['cumtime_s']:.3f}s ncalls={d['ncalls']}")
    return report


# --------------------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------------------
def _mb(n) -> str:
    return "n/a" if n is None else f"{n / (1024 * 1024):.0f}"


def _fmt(v) -> str:
    return "n/a" if v is None else f"{v:.4g}"


def _default_ceiling(meta: RunMetadata) -> int:
    if meta.total_ram_bytes:
        return int(0.75 * meta.total_ram_bytes)
    return 24 * 1024 ** 3


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def _parse_int_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _parse_str_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["confirm", "baseline", "blocker", "profile", "all"], default="all")
    ap.add_argument("--smoke", action="store_true", help="fast path; no large panel required")
    ap.add_argument("--methods", type=_parse_str_list, default=list(grid.DEFAULT_METHODS))
    ap.add_argument("--samples", type=_parse_int_list, default=list(grid.DEFAULT_SAMPLE_SIZES))
    ap.add_argument("--chroms", type=_parse_str_list, default=None, help="growth order, e.g. 22,21,20")
    ap.add_argument("--baseline-samples", type=int, default=None, help="samples for the oracle (default: all)")
    ap.add_argument("--pgs-scale", default=PGS_SCALE)
    ap.add_argument("--pgs-baseline", default=PGS_BASELINE)
    ap.add_argument("--genome-build", default=GENOME_BUILD)
    ap.add_argument("--kg-dir", dest="kg_dir", type=Path, default=DEFAULT_KG_DIR)
    ap.add_argument("--chip-file", type=Path, default=DEFAULT_CHIP_FILE)
    ap.add_argument("--file-pattern", default=prefilter.GRCH38_HIGHCOV_PATTERN)
    ap.add_argument("--window-size", type=int, default=1_000_000)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--mem-limit-gb", type=float, default=None)
    # Per-cell TIME budget (X). A cell exceeding it is killed and recorded as outcome
    # "timeout" -- the *time wall*, analogous to the memory wall -- and larger cells in that
    # sweep are skipped. Default 30 min: the scaling plan targets the optimized full-scale
    # fit at order-of-minutes, so a cell that cannot finish a fraction of the genome at
    # <=3202 samples within 30 min is decisively impractical at 500K x 2M. Later phases must
    # bring runs below this.
    ap.add_argument("--cell-timeout-s", type=float, default=1800.0)
    ap.add_argument("--tracemalloc", action="store_true", help="enable per-site attribution (inflates RSS)")
    ap.add_argument("--no-load-only", action="store_true")
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--workdir", type=Path, default=_BENCH_DATA / "work")
    ap.add_argument("--no-projection", action="store_true", help="skip scaling projection at the end")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    if args.smoke:
        return _run_smoke(args)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.workdir.mkdir(parents=True, exist_ok=True)
    meta = collect_metadata()

    if args.mode in ("confirm", "all"):
        cmd_confirm(args)
    if args.mode in ("baseline", "all"):
        cmd_baseline(args, meta)
    if args.mode in ("blocker", "all"):
        cmd_blocker(args, meta)
    if args.mode == "profile":
        _profile_one_fit(args, meta)
    if args.mode in ("blocker", "all") and not args.no_projection:
        _run_projection(args)
    return 0


def _run_projection(args) -> None:
    from benchmarks import scaling_projection

    scaling_projection.main(["--results-dir", str(args.results_dir)])


def _run_smoke(args) -> int:
    """Fast end-to-end shakedown; see benchmarks/smoke.py for the implementation."""
    from benchmarks.smoke import run_smoke

    return run_smoke(args)


if __name__ == "__main__":
    raise SystemExit(main())
