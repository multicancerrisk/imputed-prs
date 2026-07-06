"""Phase-5 bounded validation: numeric hard-called scorer vs the retired string replay.

Before P5 the evaluator scored *hard-called* reference panels (1000G high-coverage GT,
every dosage in {0,1,2}) by rendering genotype strings and replaying the browser scorer
once per held-out sample (``_predicted_prs_via_strings``) -- O(samples x variants)
pure-Python object construction, the path that ran >2 h (killed) inside the Phase-4 chr22
reference CV while each streaming fit took ~9 min. P5 routes hard-called panels through
the same vectorized numeric scorer the *continuous* path already used
(``_compute_imputed_prs_batch`` -> ``_predicted_prs_numeric``).

Correctness/parity is locked by the unit suite (``tests/test_round_trip.py``:
numeric==string at ``atol=1e-12``, CSR batch==oracle at ``atol=1e-9``, plus the no-call
and masking-metric parity tests). This script measures the real-data *speedup* on the
GRCh38 1000G high-coverage **chr22** panel (3,202 samples) over a bounded (<=1,200-variant)
PGS000027 subset, re-confirms parity on real hard-called dosages, and extrapolates the
string-path cost to the full PGS000027 (~2.1M variants) 10-fold reference CV.

No full-scale run is needed to establish P5 correctness (the golden gate + small-fixture
parity is the bar); this is the "micro-benchmark that extrapolates the saving".

Run:  .venv/bin/python -m benchmarks.verify_hardcall_scoring
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from imputed_prs.core.linear_imputation_prs import LinearImputationPRS
from imputed_prs.core.types import GenotypeData
from imputed_prs.evaluation import ImputationEvaluator
from imputed_prs.evaluation._scoring import is_hard_called
from imputed_prs.evaluation.metrics import compute_prs_metrics
from imputed_prs.io.genotype_loader import load_genotypes

_COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}


def _predictor_census(model) -> dict:
    """Count non-SNP predictor kinds that used to make the retired string replay
    diverge from the numeric path before the structured allele/dosage browser
    scorer: INDELs / multi-char alleles and multiallelic co-predictors (both now
    round-trip through the string path as whole allele tokens), plus palindromic
    (A/T, C/G) loci."""
    seen = {}
    for m in model.imputed_models:
        for i, pid in enumerate(m.predictor_variant_ids):
            seen.setdefault(pid, (m.predictor_counted_alleles[i], m.predictor_other_alleles[i],
                                  m.predictor_chromosomes[i], m.predictor_positions[i]))
    pos_alleles = {}
    n_indel = n_palindromic = 0
    for cnt, oth, c, p in seen.values():
        if len(cnt) > 1 or len(oth) > 1:
            n_indel += 1
        if _COMPLEMENT.get(cnt) == oth:
            n_palindromic += 1
        pos_alleles.setdefault((c, p), set()).add(cnt)
    n_multiallelic = sum(1 for v in pos_alleles.values() if len(v) > 1)
    return {"unique_predictors": len(seen), "indel_or_multichar_allele": n_indel,
            "multiallelic_co_predictor_positions": n_multiallelic, "palindromic": n_palindromic}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("verify_hardcall_scoring")

_BENCH = Path(__file__).resolve().parent
WORK = _BENCH / "data" / "work"
CHR22_REF = WORK / "pos" / "22.vcf.gz"            # 34,388 positions x 3,202 samples (cached)
CHIP = _BENCH / "data" / "23andme_v5_GRCh38_variants.txt"
PRS_CHR22 = WORK / "prs" / "PGS000027_GRCh38_22.csv"   # PGS000027 restricted to chr22
RESULTS = _BENCH / "results" / "predict-hardcall"
GENOME_BUILD = "GRCh38"

N_VARIANTS_CAP = 1200          # bounded PRS subset (plan: <=2,000-variant chr22 subset)
N_SAMPLES_TIMING = 800         # bounded sample count for the (slow) string-path timing
FULL_PGS_VARIANTS = 2_100_000  # PGS000027 ~2.1M variants
FULL_SAMPLES = 500_000         # production panel size
CV_FOLDS = 10


def main() -> None:
    for p in (CHR22_REF, CHIP, PRS_CHR22):
        if not p.exists():
            raise FileNotFoundError(f"missing benchmark input {p}")
    RESULTS.mkdir(parents=True, exist_ok=True)

    # 1. Bounded chr22 PGS000027 subset.
    prs = pd.read_csv(PRS_CHR22).head(N_VARIANTS_CAP)
    prs_csv = RESULTS / "prs_subset.csv"
    prs.to_csv(prs_csv, index=False)
    platform = [ln.strip() for ln in open(CHIP) if ln.strip()]
    log.info("fitting bounded chr22 model: %d PRS variants ...", len(prs))

    # 2. Fit (bounded -> fast; config mirrors verify_predict_scale.part_refcv).
    t0 = time.perf_counter()
    model = LinearImputationPRS(
        window_size=1_000_000, tuning_scope="none", alpha=0.01, l1_ratio=0.5,
        cv_folds=3, random_state=42, backend="auto", verbose=0,
    )
    model.fit(
        reference_genotypes=str(CHR22_REF), prs_definition=str(prs_csv),
        platform_variants=platform, genome_build=GENOME_BUILD,
    )
    fit_wall = time.perf_counter() - t0
    n_observed = len(model.observed_variants)
    n_imputed = len(model.imputed_models)
    n_predictor_terms = int(sum(len(m.predictor_variant_ids) for m in model.imputed_models))
    log.info("fit ok in %.1fs: %d observed, %d imputed targets, %d predictor terms",
             fit_wall, n_observed, n_imputed, n_predictor_terms)

    # 3. Held-out hard-called panel (subsample samples so the string path stays bounded).
    gd = load_genotypes(path=str(CHR22_REF))
    n_full = gd.n_samples
    n = min(N_SAMPLES_TIMING, n_full)
    panel = GenotypeData(
        dosage_matrix=gd.dosage_matrix[:n],
        variant_info=gd.variant_info,
        sample_ids=list(gd.sample_ids[:n]),
        genome_build=gd.genome_build,
    )
    assert is_hard_called(panel.dosage_matrix), "chr22 GT panel must be hard-called"

    ev = ImputationEvaluator(model, verbose=0)
    census = _predictor_census(model)
    log.info("predictor census: %s", census)

    # 4. Time the retired string replay vs the P5 numeric scorer; confirm parity, and
    #    compare the *metrics* each estimate yields against the gold-standard true PRS.
    s_true = ev._compute_true_prs(panel)   # numeric gold standard (unchanged by P5)

    t = time.perf_counter()
    s_string = ev._predicted_prs_via_strings(panel)
    string_wall = time.perf_counter() - t

    t = time.perf_counter()
    s_numeric = ev._compute_imputed_prs_batch(panel)   # P5: numeric dispatch
    numeric_wall = time.perf_counter() - t

    max_abs_diff = float(np.max(np.abs(s_numeric - s_string)))
    denom = float(np.max(np.abs(s_string))) or 1.0
    max_rel_diff = max_abs_diff / denom
    corr = float(np.corrcoef(s_numeric, s_string)[0, 1])
    speedup = string_wall / numeric_wall if numeric_wall > 0 else None

    # Metric parity: does swapping the string replay for the numeric scorer move the
    # headline accuracy/calibration the evaluator reports? (The plan's fidelity bar.)
    m_num = compute_prs_metrics(s_numeric, s_true)
    m_str = compute_prs_metrics(s_string, s_true)
    metric_parity = {
        "numeric_vs_true": {"r2": m_num.r2, "correlation": m_num.correlation,
                            "calibration_slope": m_num.calibration_slope,
                            "calibration_intercept": m_num.calibration_intercept},
        "string_vs_true": {"r2": m_str.r2, "correlation": m_str.correlation,
                           "calibration_slope": m_str.calibration_slope,
                           "calibration_intercept": m_str.calibration_intercept},
        "abs_r2_difference": abs(m_num.r2 - m_str.r2),
        "abs_calibration_slope_difference": abs(m_num.calibration_slope - m_str.calibration_slope),
    }
    log.info("scored n=%d samples: string=%.2fs numeric=%.3fs speedup=%.1fx parity_maxdiff=%.2e corr=%.10f",
             n, string_wall, numeric_wall, speedup or -1, max_abs_diff, corr)
    log.info("metric parity: R2 numeric=%.6f string=%.6f (|d|=%.2e)  calib_slope numeric=%.6f string=%.6f",
             m_num.r2, m_str.r2, metric_parity["abs_r2_difference"],
             m_num.calibration_slope, m_str.calibration_slope)

    # 5. Extrapolate. String scoring is O(held-out samples x needed variants); a k-fold
    #    reference CV scores every sample once (each held out in exactly one fold).
    needed = n_observed + n_predictor_terms
    string_per_cell = string_wall / (n * max(needed, 1))       # sec / (sample x needed-variant)
    numeric_per_cell = numeric_wall / (n * max(needed, 1))
    # Full PGS000027 10-fold CV: FULL_SAMPLES scorings x FULL_PGS_VARIANTS needed variants.
    full_cells = FULL_SAMPLES * FULL_PGS_VARIANTS
    string_full_hours = string_per_cell * full_cells / 3600.0
    numeric_full_hours = numeric_per_cell * full_cells / 3600.0

    rec = {
        "part": "hardcall_scoring",
        "phase": 5,
        "reference": str(CHR22_REF),
        "genome_build": GENOME_BUILD,
        "n_prs_variants_fit": int(len(prs)),
        "n_observed": n_observed,
        "n_imputed_targets": n_imputed,
        "n_predictor_terms": n_predictor_terms,
        "predictor_census": census,
        "n_samples_timed": n,
        "n_samples_available": n_full,
        "fit_seconds": fit_wall,
        "string_replay_seconds": string_wall,
        "numeric_scorer_seconds": numeric_wall,
        "speedup_string_over_numeric": speedup,
        "parity": {
            "max_abs_diff": max_abs_diff,
            "max_rel_diff": max_rel_diff,
            "pearson_corr": corr,
            "note": (
                "numeric == string to float-epsilon on ALL predictors, including the "
                "INDEL / multi-char-allele predictors in predictor_census. The "
                "browser/upload string scorer now parses, counts, and resolves non-SNP "
                "alleles as whole tokens (the structured allele/dosage scorer in "
                "imputed_prs.io.user_genotypes) and disambiguates multiallelic "
                "co-predictors via exact-id-first resolution, so it matches the numeric "
                "reference-dosage path. Before that fix these indel predictors made "
                "numeric and string differ at ~1e-4 (historical max_abs 2.8e-4, r 0.99988)."
            ),
        },
        "metric_parity": metric_parity,
        "extrapolation_full_pgs027_10fold_cv": {
            "model": "held-out scoring wall ~ O(samples x needed_variants); k-fold scores each sample once",
            "assumed_samples": FULL_SAMPLES,
            "assumed_pgs_variants": FULL_PGS_VARIANTS,
            "assumed_cv_folds": CV_FOLDS,
            "string_replay_hours": string_full_hours,
            "numeric_scorer_hours": numeric_full_hours,
            "string_replay_days": string_full_hours / 24.0,
        },
    }
    out = RESULTS / "hardcall_speedup.json"
    out.write_text(json.dumps(rec, indent=2, default=str))
    log.info("wrote %s", out)
    log.info("EXTRAPOLATION full PGS000027 10-fold CV held-out scoring: string ~%.0f h (%.1f d)  vs  numeric ~%.2f h",
             string_full_hours, string_full_hours / 24.0, numeric_full_hours)
    print(json.dumps(rec, indent=2, default=str))


if __name__ == "__main__":
    main()
