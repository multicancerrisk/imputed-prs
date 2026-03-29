"""Integration tests: equivalence, divergence, and calibration of projection vs imputation."""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest

from imputed_prs import LinearImputationPRS, LinearProjectionPRS
from imputed_prs.evaluation import ImputationEvaluator, ProjectionEvaluator


# =============================================================================
# Helpers
# =============================================================================


def _gt_from_dosage(d: int) -> str:
    """Convert integer dosage (0, 1, 2) to VCF GT string."""
    if d == 0:
        return "0/0"
    elif d == 1:
        return "0/1"
    else:
        return "1/1"


def _generate_synthetic_vcf(
    tmp_path: Path,
    rng: np.random.Generator,
    n_samples: int,
    variant_specs: List[Tuple[str, int, str, str, str, np.ndarray]],
    filename: str = "synthetic.vcf",
) -> Path:
    """Generate a synthetic VCF from variant specifications.

    Args:
        tmp_path: Temporary directory for the VCF file.
        rng: NumPy random generator.
        n_samples: Number of samples.
        variant_specs: List of (chrom, pos, rsid, ref, alt, dosage_array).
            dosage_array has shape (n_samples,) with values in {0, 1, 2}.
        filename: Output filename.

    Returns:
        Path to the generated VCF file.
    """
    # Collect unique chromosomes for contig headers
    chroms = sorted(set(spec[0] for spec in variant_specs))

    lines = []
    lines.append("##fileformat=VCFv4.2")
    for chrom in chroms:
        lines.append(f"##contig=<ID={chrom},length=249250621>")
    lines.append('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')

    sample_names = [f"S{i+1}" for i in range(n_samples)]
    header = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(sample_names)
    lines.append(header)

    for chrom, pos, rsid, ref, alt, dosages in variant_specs:
        gts = "\t".join(_gt_from_dosage(int(d)) for d in dosages)
        lines.append(f"{chrom}\t{pos}\t{rsid}\t{ref}\t{alt}\t.\t.\t.\tGT\t{gts}")

    vcf_path = tmp_path / filename
    vcf_path.write_text("\n".join(lines) + "\n")
    return vcf_path


def _build_synthetic_data(
    tmp_path: Path,
    rng: np.random.Generator,
    n_samples: int = 200,
    n_platform: int = 30,
    n_missing: int = 5,
    chromosomes: List[str] = None,
    missing_positions_start: int = 115_000,
    missing_spacing: int = 1_000,
    platform_positions_start: int = 100_000,
    platform_spacing: int = 1_000,
    betas: np.ndarray = None,
):
    """Build synthetic VCF + PRS data for integration tests.

    Platform variants and missing variants are on specified chromosomes.
    Missing variants have LD with nearby platform variants.

    Returns:
        (vcf_path, prs_df, platform_variant_ids)
    """
    if chromosomes is None:
        chromosomes = ["1"]
    if betas is None:
        betas = rng.uniform(0.1, 0.5, n_missing * len(chromosomes))

    variant_specs = []
    prs_rows = []
    platform_ids = []

    variant_counter = 1000
    missing_counter = 0

    for chrom in chromosomes:
        # Platform variants on this chromosome (use rs-prefixed IDs)
        platform_dosages = {}
        for i in range(n_platform):
            rsid = f"rs{variant_counter}"
            variant_counter += 1
            pos = platform_positions_start + i * platform_spacing
            af = rng.uniform(0.15, 0.45)
            dosages = rng.binomial(2, af, n_samples)
            variant_specs.append((chrom, pos, rsid, "A", "G", dosages))
            platform_dosages[i] = dosages.astype(float)
            platform_ids.append(rsid)

        # Missing PRS variants with LD to platform variants (use rs-prefixed IDs)
        for j in range(n_missing):
            rsid = f"rs{variant_counter}"
            variant_counter += 1
            pos = missing_positions_start + j * missing_spacing
            # Create LD: linear combination of nearby platform variants + noise
            n_pred = min(5, n_platform)
            pred_indices = rng.choice(n_platform, n_pred, replace=False)
            weights = rng.uniform(0.2, 0.5, n_pred)
            signal = sum(
                platform_dosages[k] * w for k, w in zip(pred_indices, weights)
            )
            signal = (signal - signal.mean()) / (signal.std() + 1e-10)
            noise = rng.normal(0, 0.5, n_samples)
            raw = signal + noise + 1.0
            dosages = np.clip(np.round(raw), 0, 2).astype(int)
            variant_specs.append((chrom, pos, rsid, "C", "T", dosages))

            beta = float(betas[missing_counter])
            prs_rows.append({
                "variant_id": rsid,
                "chromosome": chrom,
                "position": pos,
                "effect_allele": "T",
                "other_allele": "C",
                "beta": beta,
            })
            missing_counter += 1

    # Also add some platform variants as "observed" PRS variants
    # (pick the first few per chromosome to have nonzero observed component)
    plat_idx = 0
    for chrom in chromosomes:
        for i in range(min(3, n_platform)):
            rsid = platform_ids[plat_idx + i]
            pos = platform_positions_start + i * platform_spacing
            prs_rows.append({
                "variant_id": rsid,
                "chromosome": chrom,
                "position": pos,
                "effect_allele": "G",
                "other_allele": "A",
                "beta": float(rng.uniform(0.05, 0.2)),
            })
        plat_idx += n_platform

    prs_df = pd.DataFrame(prs_rows)
    vcf_path = _generate_synthetic_vcf(tmp_path, rng, n_samples, variant_specs)

    return vcf_path, prs_df, platform_ids


# =============================================================================
# Equivalence tests
# =============================================================================


class TestEquivalenceWithoutRegularization:
    """When regularization is near-zero, projection and imputation give the same PRS."""

    def test_equivalence_single_region(self, tmp_path):
        """Single region: PRS values from both methods match within tolerance.

        Note: Exact equivalence requires no regularization AND no dosage clipping.
        The imputation approach clips predicted dosages to [0, 2] while the
        projection approach doesn't clip. We use very small regularization
        (near-OLS) and accept a tolerance that accounts for clipping effects.
        """
        cyvcf2 = pytest.importorskip("cyvcf2")

        rng = np.random.default_rng(42)
        vcf_path, prs_df, platform_ids = _build_synthetic_data(
            tmp_path, rng, n_samples=200, n_platform=30, n_missing=5,
            chromosomes=["1"],
        )

        common_kwargs = dict(
            window_size=500_000,
            alpha=1e-6,
            l1_ratio=0.01,
            cv_folds=3,
            random_state=42,
            verbose=0,
        )

        imp_model = LinearImputationPRS(tuning_scope="none", **common_kwargs)
        imp_model.fit(
            reference_genotypes=vcf_path,
            prs_definition=prs_df,
            platform_variants=platform_ids,
        )

        proj_model = LinearProjectionPRS(**common_kwargs)
        proj_model.fit(
            reference_genotypes=vcf_path,
            prs_definition=prs_df,
            platform_variants=platform_ids,
        )

        # Use actual genotype data from VCF for realistic predictions
        from imputed_prs.io.genotype_loader import load_genotypes

        needed = set(prs_df["variant_id"]) | set(platform_ids)
        geno_data = load_genotypes(path=vcf_path, variant_ids=needed)

        var_to_idx = {}
        for idx, row in geno_data.variant_info.iterrows():
            var_to_idx[row["variant_id"]] = idx

        prs_imp_values = []
        prs_proj_values = []
        for i in range(geno_data.n_samples):
            user = {}
            for var_id in set(platform_ids) | set(prs_df["variant_id"]):
                geno_idx = var_to_idx.get(var_id)
                if geno_idx is not None:
                    d = geno_data.dosage_matrix[i, geno_idx]
                    if not np.isnan(d):
                        user[var_id] = float(d)

            prs_imp_values.append(imp_model.predict(user, apply_calibration=False).prs)
            prs_proj_values.append(proj_model.predict(user, apply_calibration=False).prs)

        # With near-zero regularization, the two methods should be highly
        # correlated. Absolute differences arise from dosage clipping in
        # the imputation approach (which the projection approach skips).
        correlation = np.corrcoef(prs_imp_values, prs_proj_values)[0, 1]
        assert correlation > 0.95, (
            f"Correlation {correlation:.4f} too low -- "
            "methods should be near-equivalent with minimal regularization"
        )

    def test_equivalence_multiple_regions(self, tmp_path):
        """Multiple regions across chromosomes: equivalence holds."""
        cyvcf2 = pytest.importorskip("cyvcf2")

        rng = np.random.default_rng(123)
        vcf_path, prs_df, platform_ids = _build_synthetic_data(
            tmp_path, rng, n_samples=200, n_platform=20, n_missing=3,
            chromosomes=["1", "2"],
        )

        common_kwargs = dict(
            window_size=500_000,
            alpha=1e-6,
            l1_ratio=0.01,
            cv_folds=3,
            random_state=123,
            verbose=0,
        )

        imp_model = LinearImputationPRS(tuning_scope="none", **common_kwargs)
        imp_model.fit(
            reference_genotypes=vcf_path,
            prs_definition=prs_df,
            platform_variants=platform_ids,
        )

        proj_model = LinearProjectionPRS(**common_kwargs)
        proj_model.fit(
            reference_genotypes=vcf_path,
            prs_definition=prs_df,
            platform_variants=platform_ids,
        )

        from imputed_prs.io.genotype_loader import load_genotypes

        needed = set(prs_df["variant_id"]) | set(platform_ids)
        geno_data = load_genotypes(path=vcf_path, variant_ids=needed)

        var_to_idx = {}
        for idx, row in geno_data.variant_info.iterrows():
            var_to_idx[row["variant_id"]] = idx

        prs_imp_values = []
        prs_proj_values = []
        for i in range(geno_data.n_samples):
            user = {}
            for var_id in set(platform_ids) | set(prs_df["variant_id"]):
                geno_idx = var_to_idx.get(var_id)
                if geno_idx is not None:
                    d = geno_data.dosage_matrix[i, geno_idx]
                    if not np.isnan(d):
                        user[var_id] = float(d)

            prs_imp_values.append(imp_model.predict(user, apply_calibration=False).prs)
            prs_proj_values.append(proj_model.predict(user, apply_calibration=False).prs)

        correlation = np.corrcoef(prs_imp_values, prs_proj_values)[0, 1]
        assert correlation > 0.95, (
            f"Correlation {correlation:.4f} too low -- "
            "methods should be near-equivalent with minimal regularization"
        )


class TestDivergenceWithRegularization:
    """With meaningful regularization, the two approaches diverge."""

    def test_divergence_moderate_alpha(self, tmp_path):
        """alpha=0.1: PRS values differ between methods."""
        cyvcf2 = pytest.importorskip("cyvcf2")

        rng = np.random.default_rng(99)
        vcf_path, prs_df, platform_ids = _build_synthetic_data(
            tmp_path, rng, n_samples=200, n_platform=30, n_missing=5,
            chromosomes=["1"],
        )

        common_kwargs = dict(
            window_size=500_000,
            alpha=0.1,
            l1_ratio=0.5,
            cv_folds=3,
            random_state=99,
            verbose=0,
        )

        imp_model = LinearImputationPRS(tuning_scope="none", **common_kwargs)
        imp_model.fit(
            reference_genotypes=vcf_path,
            prs_definition=prs_df,
            platform_variants=platform_ids,
        )

        proj_model = LinearProjectionPRS(**common_kwargs)
        proj_model.fit(
            reference_genotypes=vcf_path,
            prs_definition=prs_df,
            platform_variants=platform_ids,
        )

        diffs = []
        for i in range(50):
            user = {}
            for var_id in platform_ids:
                user[var_id] = float(rng.choice([0, 1, 2]))
            for _, row in prs_df.iterrows():
                if row["variant_id"] not in user:
                    user[row["variant_id"]] = float(rng.choice([0, 1, 2]))

            prs_imp = imp_model.predict(user, apply_calibration=False).prs
            prs_proj = proj_model.predict(user, apply_calibration=False).prs
            diffs.append(abs(prs_imp - prs_proj))

        max_diff = max(diffs)
        assert max_diff > 1e-3, (
            f"Max PRS difference {max_diff:.6f} is too small -- "
            "approaches should diverge with moderate regularization"
        )


class TestProjectionAdvantage:
    """Cases where projection should be at least competitive with imputation."""

    def test_heterogeneous_betas(self, tmp_path):
        """Widely varying betas: projection achieves comparable or higher R^2."""
        cyvcf2 = pytest.importorskip("cyvcf2")

        rng = np.random.default_rng(77)
        # Create betas with wide range: some tiny, some large
        betas = np.array([0.01, 0.01, 0.02, 1.0, 0.8])

        vcf_path, prs_df, platform_ids = _build_synthetic_data(
            tmp_path, rng, n_samples=200, n_platform=30, n_missing=5,
            chromosomes=["1"], betas=betas,
        )

        common_kwargs = dict(
            window_size=500_000,
            alpha=0.05,
            l1_ratio=0.5,
            cv_folds=3,
            random_state=77,
            verbose=0,
        )

        imp_model = LinearImputationPRS(tuning_scope="none", **common_kwargs)
        imp_model.fit(
            reference_genotypes=vcf_path,
            prs_definition=prs_df,
            platform_variants=platform_ids,
        )

        proj_model = LinearProjectionPRS(**common_kwargs)
        proj_model.fit(
            reference_genotypes=vcf_path,
            prs_definition=prs_df,
            platform_variants=platform_ids,
        )

        # Evaluate both on same data (in-sample)
        imp_eval = ImputationEvaluator(imp_model, verbose=0)
        proj_eval = ProjectionEvaluator(proj_model, verbose=0)

        imp_metrics = imp_eval.evaluate(vcf_path)
        proj_metrics = proj_eval.evaluate(vcf_path)

        # Projection should be at least competitive
        assert proj_metrics.r2 >= imp_metrics.r2 - 0.05, (
            f"Projection R²={proj_metrics.r2:.4f} much worse than "
            f"imputation R²={imp_metrics.r2:.4f}"
        )


class TestSECalibration:
    """Verify that the projection SE provides reasonable coverage."""

    def test_coverage_95ci(self, tmp_path):
        """A meaningful fraction of true PRS values fall within projected 95% CIs."""
        cyvcf2 = pytest.importorskip("cyvcf2")

        rng = np.random.default_rng(55)
        n_total = 500
        vcf_path, prs_df, platform_ids = _build_synthetic_data(
            tmp_path, rng, n_samples=n_total, n_platform=30, n_missing=5,
            chromosomes=["1"],
        )

        # Fit on all data
        model = LinearProjectionPRS(
            window_size=500_000,
            alpha=0.01,
            l1_ratio=0.5,
            cv_folds=5,
            random_state=55,
            verbose=0,
        )
        model.fit(
            reference_genotypes=vcf_path,
            prs_definition=prs_df,
            platform_variants=platform_ids,
        )

        # Compute true PRS for held-out "test" samples
        # Use evaluator to get true PRS
        from imputed_prs.core.types import GenotypeData
        from imputed_prs.io.genotype_loader import load_genotypes

        needed = set(prs_df["variant_id"]) | set(platform_ids)
        geno_data = load_genotypes(path=vcf_path, variant_ids=needed)

        evaluator = ProjectionEvaluator(model, verbose=0)
        true_prs = evaluator._compute_true_prs(geno_data)

        # Predict with CIs for a subset of samples
        n_test = min(100, n_total)
        var_to_idx = {}
        for idx, row in geno_data.variant_info.iterrows():
            var_to_idx[row["variant_id"]] = idx

        n_covered = 0
        n_valid = 0
        for i in range(n_test):
            user = {}
            for var_id in platform_ids:
                geno_idx = var_to_idx.get(var_id)
                if geno_idx is not None:
                    d = geno_data.dosage_matrix[i, geno_idx]
                    user[var_id] = float(d) if not np.isnan(d) else None

            result = model.predict(user, apply_calibration=False)
            if result.se > 0:
                if result.ci_lower <= true_prs[i] <= result.ci_upper:
                    n_covered += 1
                n_valid += 1

        if n_valid > 0:
            coverage = n_covered / n_valid
            # Generous bounds: CI coverage should be meaningful
            assert coverage > 0.70, f"Coverage {coverage:.2f} is too low"
            assert coverage <= 1.0
