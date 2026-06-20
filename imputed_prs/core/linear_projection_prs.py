"""Main LinearProjectionPRS class for training and prediction."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd

from imputed_prs.core.exceptions import ModelNotFittedError, ValidationError
from imputed_prs.core.harmonizer import (
    _is_ambiguous_snp,
    _normalize_chromosome,
    build_reference_allele_index,
    check_predict_compatibility,
    match_oriented_dosage,
    partition_variants,
    validate_genome_build,
)
from imputed_prs.core.types import (
    CalibrationParams,
    PredictionResult,
    ProjectionRegionModel,
    ProjectionTrainingResult,
    VariantInfo,
)
from imputed_prs.evaluation.calibration import estimate_cv_calibration
from imputed_prs.io.genotype_loader import load_genotypes
from imputed_prs.io.pgs_catalog import download_pgs_catalog_score
from imputed_prs.io.platform_loader import (
    load_platform_from_manifest,
    load_platform_from_name,
    load_platform_variants_from_list,
)
from imputed_prs.io.prs_loader import load_prs_from_dataframe, load_prs_from_file
from imputed_prs.io.user_genotypes import (
    load_raw_user_genotypes,
    load_user_genotypes,
)
from imputed_prs.models.projection_predictor import ProjectionPredictor
from imputed_prs.models.projection_trainer import ProjectionRegionTrainer


class LinearProjectionPRS:
    """High-level API for training and using projection-based PRS models.

    Mirrors LinearImputationPRS but uses the linear projection approach:
    instead of imputing individual missing variant dosages, it directly
    learns platform-variant weights to approximate each region's PRS
    contribution.

    Example:
        >>> model = LinearProjectionPRS(window_size=1_000_000, cv_folds=5)
        >>> model.fit(
        ...     reference_genotypes="1000g_eur.vcf.gz",
        ...     prs_definition="PGS000004",
        ...     platform_name="23andme_v5",
        ... )
        >>> result = model.predict("user_genotypes.txt")
        >>> print(f"PRS: {result.prs:.3f} "
        ...       f"(95% CI: {result.ci_lower:.3f}-{result.ci_upper:.3f})")
    """

    def __init__(
        self,
        window_size: int = 1_000_000,
        l1_ratio: float = 0.5,
        alpha: float = 0.01,
        cv_folds: int = 5,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
        max_predictors: Optional[int] = None,
        exclude_ambiguous: bool = False,
        ambiguous_maf_threshold: float = 0.4,
        verbose: int = 1,
    ):
        """Initialize the projection PRS model.

        Args:
            window_size: Size of genomic window (bp) for defining regions
                and selecting predictor variants. Default: 1,000,000 (1 Mb).
            l1_ratio: ElasticNet L1/L2 mixing parameter (0=Ridge, 1=Lasso).
                Default: 0.5.
            alpha: ElasticNet regularization strength. Default: 0.01.
            cv_folds: Number of cross-validation folds. Default: 5.
            n_jobs: Number of parallel jobs for training. Default: 1.
            random_state: Random seed for reproducibility. Default: None.
            max_predictors: Maximum predictor variants per region.
                Default: None (no limit).
            verbose: Verbosity level (0=silent, 1=progress, 2=debug).
                Default: 1.
        """
        self.window_size = window_size
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.cv_folds = cv_folds
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.max_predictors = max_predictors
        self.exclude_ambiguous = exclude_ambiguous
        self.ambiguous_maf_threshold = ambiguous_maf_threshold
        self.verbose = verbose

        # Fitted state
        self._is_fitted: bool = False
        self._observed_variants: Optional[List[VariantInfo]] = None
        self._region_models: Optional[List[ProjectionRegionModel]] = None
        self._calibration_params: Optional[CalibrationParams] = None
        self._training_result: Optional[ProjectionTrainingResult] = None
        self._platform_variant_index: Optional[Dict[str, int]] = None
        # Per-variant disposition records (see LinearImputationPRS for the schema).
        self._variant_dispositions: Optional[List[Dict[str, Any]]] = None

        # Metadata
        self._prs_id: Optional[str] = None
        self._platform_name: Optional[str] = None
        self._genome_build: Optional[str] = None
        self._model_name: Optional[str] = None

    def fit(
        self,
        reference_genotypes: Union[str, Path],
        prs_definition: Union[str, Path, pd.DataFrame],
        platform_name: Optional[str] = None,
        platform_manifest: Optional[Union[str, Path]] = None,
        platform_variants: Optional[List[str]] = None,
        genome_build: Optional[str] = None,
        prs_id: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> "LinearProjectionPRS":
        """Train projection models on reference genotype data.

        Args:
            reference_genotypes: Path to reference genotype file (VCF/PLINK).
            prs_definition: PRS definition as DataFrame, file path, or PGS
                Catalog ID (e.g., "PGS000004").
            platform_name: Pre-built platform name (e.g., "23andme_v5").
            platform_manifest: Path to platform manifest file.
            platform_variants: List of platform variant IDs.
            genome_build: Reference genome build (e.g., "GRCh37").
            prs_id: PRS identifier for metadata.
            model_name: Model name for metadata.

        Returns:
            self (for method chaining).

        Raises:
            ValidationError: If inputs are invalid or incompatible.
            DataLoadError: If files cannot be loaded.
        """
        # Step 1: Input validation - exactly one platform source
        platform_sources = [
            platform_name is not None,
            platform_manifest is not None,
            platform_variants is not None,
        ]
        if sum(platform_sources) != 1:
            raise ValidationError(
                "Exactly one platform source must be provided: "
                "platform_name, platform_manifest, or platform_variants"
            )

        # Track effective metadata values
        effective_prs_id = prs_id
        effective_platform_name = platform_name
        effective_genome_build = genome_build
        pgs_metadata = None

        # Step 2: Load PRS definition
        if isinstance(prs_definition, pd.DataFrame):
            prs_df = load_prs_from_dataframe(prs_definition)
        elif isinstance(prs_definition, str) and prs_definition.upper().startswith("PGS"):
            prs_df, pgs_metadata = download_pgs_catalog_score(
                prs_definition,
                genome_build=genome_build or "GRCh37",
            )
            if effective_prs_id is None:
                effective_prs_id = prs_definition.upper()
            if effective_genome_build is None and pgs_metadata:
                effective_genome_build = pgs_metadata.genome_build
        else:
            prs_df = load_prs_from_file(Path(prs_definition))

        if self.verbose >= 2:
            print(f"Loaded PRS definition with {len(prs_df)} variants")

        # Step 3: Load platform variants
        platform_info = None
        if platform_name is not None:
            platform_variant_set, platform_info = load_platform_from_name(platform_name)
            effective_platform_name = platform_name
            if effective_genome_build is None and platform_info:
                effective_genome_build = platform_info.genome_build
        elif platform_manifest is not None:
            platform_variant_set, _ = load_platform_from_manifest(str(platform_manifest))
            effective_platform_name = Path(platform_manifest).stem
        else:
            platform_variant_set = load_platform_variants_from_list(platform_variants)
            effective_platform_name = "custom"

        if self.verbose >= 2:
            print(f"Loaded {len(platform_variant_set)} platform variants")

        # Step 4: Partition variants into observed and missing
        partition_result = partition_variants(prs_df, platform_variant_set)
        observed_variant_ids = partition_result.observed
        missing_variant_ids = partition_result.missing

        if self.verbose >= 1:
            print(
                f"Partitioned variants: {len(observed_variant_ids)} observed, "
                f"{len(missing_variant_ids)} missing"
            )

        # Step 5: Load reference genotypes
        prs_chrpos = set()
        for _, row in prs_df.iterrows():
            chrom = str(row["chromosome"]).upper()
            if chrom.startswith("CHR"):
                chrom = chrom[3:]
            prs_chrpos.add(f"{chrom}:{int(row['position'])}")
        all_needed_variants = set(prs_df["variant_id"]) | platform_variant_set | prs_chrpos
        genotype_data = load_genotypes(
            path=reference_genotypes, variant_ids=all_needed_variants
        )

        if self.verbose >= 2:
            print(
                f"Loaded genotypes: {genotype_data.n_samples} samples, "
                f"{genotype_data.n_variants} variants"
            )

        # Step 6: Validate genome build
        prs_build = effective_genome_build
        build_result = validate_genome_build(
            prs_build, genotype_data.genome_build, strict=False
        )
        if build_result.warning and self.verbose >= 1:
            print(f"Warning: {build_result.warning}")
        if build_result.genotype_build:
            effective_genome_build = build_result.genotype_build
        elif build_result.prs_build:
            effective_genome_build = build_result.prs_build

        # Step 7: Build allele-aware reference indices (see LinearImputationPRS).
        reference_index = build_reference_allele_index(genotype_data.variant_info)
        reference_contigs = {
            _normalize_chromosome(str(c))
            for c in genotype_data.variant_info["chromosome"].unique()
        }

        # Step 8: Build training matrices.
        # Mapping for platform predictor lookups (first occurrence wins).
        geno_var_to_idx: Dict[str, int] = {}
        for idx, row in genotype_data.variant_info.iterrows():
            geno_var_to_idx.setdefault(row["variant_id"], idx)
            chrom = _normalize_chromosome(str(row["chromosome"]))
            pos = str(int(row["position"]))
            geno_var_to_idx.setdefault(f"{chrom}:{pos}", idx)

        # Optional QC — exclude strand-ambiguous SNPs with high reference MAF.
        ambiguous_excluded_ids: Set[str] = set()
        if self.exclude_ambiguous:
            for _, row in prs_df.iterrows():
                effect = str(row["effect_allele"]).upper()
                other = row.get("other_allele")
                if pd.isna(other):
                    continue
                if not _is_ambiguous_snp(effect, str(other).upper()):
                    continue
                match = match_oriented_dosage(
                    row["chromosome"], int(row["position"]), effect, other,
                    genotype_data.variant_info, genotype_data.dosage_matrix,
                    reference_index,
                )
                if match is None:
                    continue
                dosage = match[1]
                valid = ~np.isnan(dosage)
                if not np.any(valid):
                    continue
                af = float(np.mean(dosage[valid]) / 2.0)
                if min(af, 1.0 - af) > self.ambiguous_maf_threshold:
                    ambiguous_excluded_ids.add(row["variant_id"])

        # Allele-aware observed inclusion (see LinearImputationPRS): require an
        # observed variant's (effect, other) alleles to be compatible with the
        # reference at its locus. Locus-in-reference but allele-incompatible ->
        # reclassify to missing (recovered by projection where possible, else
        # dropped-with-reason "allele_mismatch"). Locus absent from reference ->
        # keep observed (still scoreable directly from the user's genotype).
        observed_variant_ids = set(observed_variant_ids)
        missing_variant_ids = set(missing_variant_ids)
        observed_allele_reclassified: Set[str] = set()
        for _, row in prs_df[
            prs_df["variant_id"].isin(observed_variant_ids)
        ].iterrows():
            var_id = row["variant_id"]
            if var_id in ambiguous_excluded_ids:
                continue
            locus = (
                f"{_normalize_chromosome(str(row['chromosome']))}:"
                f"{int(row['position'])}"
            )
            if locus not in reference_index:
                continue  # platform-measured but absent from reference: keep observed
            match = match_oriented_dosage(
                row["chromosome"], int(row["position"]),
                row["effect_allele"], row.get("other_allele"),
                genotype_data.variant_info, genotype_data.dosage_matrix,
                reference_index,
            )
            if match is None:
                observed_allele_reclassified.add(var_id)
        if observed_allele_reclassified:
            observed_variant_ids -= observed_allele_reclassified
            missing_variant_ids |= observed_allele_reclassified
            if self.verbose >= 1:
                print(
                    f"Reclassified {len(observed_allele_reclassified)} observed "
                    f"variants to missing (allele-incompatible with reference)"
                )

        # Build platform variant info DataFrame and Z matrix
        platform_variant_indices = []
        platform_variant_rows = []
        for var_id in platform_variant_set:
            if var_id in geno_var_to_idx:
                idx = geno_var_to_idx[var_id]
                platform_variant_indices.append(idx)
                row = genotype_data.variant_info.iloc[idx]
                platform_variant_rows.append({
                    "variant_id": row["variant_id"],
                    "chromosome": row["chromosome"],
                    "position": row["position"],
                    "ref_allele": row["ref_allele"],
                    "alt_allele": row["alt_allele"],
                })
            elif var_id.lower() in geno_var_to_idx:
                idx = geno_var_to_idx[var_id.lower()]
                platform_variant_indices.append(idx)
                row = genotype_data.variant_info.iloc[idx]
                platform_variant_rows.append({
                    "variant_id": row["variant_id"],
                    "chromosome": row["chromosome"],
                    "position": row["position"],
                    "ref_allele": row["ref_allele"],
                    "alt_allele": row["alt_allele"],
                })

        if platform_variant_rows:
            platform_variant_info = pd.DataFrame(platform_variant_rows)
            Z = genotype_data.dosage_matrix[:, platform_variant_indices]
        else:
            platform_variant_info = pd.DataFrame(
                columns=["variant_id", "chromosome", "position", "ref_allele", "alt_allele"]
            )
            Z = np.empty((genotype_data.n_samples, 0))

        # Build the missing-variant target matrix X using effect-allele-oriented
        # dosages, recording a reason for any variant that cannot be placed.
        missing_drop_reason: Dict[str, str] = {}
        missing_prs_rows: List[pd.Series] = []
        missing_columns: List[np.ndarray] = []
        for _, row in prs_df[prs_df["variant_id"].isin(missing_variant_ids)].iterrows():
            var_id = row["variant_id"]
            if var_id in ambiguous_excluded_ids:
                missing_drop_reason[var_id] = "ambiguous_excluded"
                continue
            match = match_oriented_dosage(
                row["chromosome"], int(row["position"]),
                row["effect_allele"], row.get("other_allele"),
                genotype_data.variant_info, genotype_data.dosage_matrix,
                reference_index,
            )
            if match is None:
                chrom_n = _normalize_chromosome(str(row["chromosome"]))
                if chrom_n not in reference_contigs:
                    missing_drop_reason[var_id] = "reference_contig_missing"
                elif f"{chrom_n}:{int(row['position'])}" in reference_index:
                    missing_drop_reason[var_id] = "allele_mismatch"
                else:
                    missing_drop_reason[var_id] = "not_in_reference"
                continue
            missing_columns.append(match[1])
            missing_prs_rows.append(row)

        if missing_prs_rows:
            missing_prs_df = pd.DataFrame(missing_prs_rows).reset_index(drop=True)
            X = np.column_stack(missing_columns).astype(np.float32)
        else:
            missing_prs_df = prs_df.iloc[0:0].copy()
            X = np.empty((genotype_data.n_samples, 0), dtype=np.float32)

        if self.verbose >= 2:
            print(
                f"Training matrices: Z={Z.shape}, X={X.shape}, "
                f"missing_prs_df={len(missing_prs_df)} variants"
            )

        # Step 9: Train projection models
        if X.shape[1] > 0:
            if self.verbose >= 1:
                print(f"Training projection models for {len(missing_prs_df)} missing variants...")

            trainer = ProjectionRegionTrainer(
                window_size=self.window_size,
                l1_ratio=self.l1_ratio,
                alpha=self.alpha,
                cv_folds=self.cv_folds,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                max_predictors=self.max_predictors,
                verbose=self.verbose,
            )
            training_result = trainer.fit_all_regions(
                Z=Z,
                X=X,
                prs_variants=missing_prs_df,
                platform_variant_info=platform_variant_info,
            )

            if self.verbose >= 1:
                print(
                    f"Trained {training_result.n_regions_trained} region models, "
                    f"{training_result.n_intercept_only} intercept-only"
                )
        else:
            training_result = ProjectionTrainingResult(
                region_models={},
                cv_predictions={},
                n_regions_trained=0,
                n_regions_failed=0,
                n_intercept_only=0,
                training_summary={
                    "mean_r2": 0.0,
                    "median_r2": 0.0,
                    "std_r2": 0.0,
                    "min_r2": 0.0,
                    "max_r2": 0.0,
                    "n_high_quality": 0,
                    "n_medium_quality": 0,
                    "n_low_quality": 0,
                    "mean_n_predictors": 0.0,
                    "mean_n_prs_variants_per_region": 0.0,
                },
            )

        # Step 10: Compute calibration parameters
        calibration_params = None
        if training_result.cv_predictions and len(training_result.cv_predictions) > 0:
            try:
                # Build the placed-variant matrix with effect-oriented dosages.
                covered_ids = set()
                for region in training_result.region_models.values():
                    covered_ids.update(region.prs_variant_ids)

                placed_columns: List[np.ndarray] = []
                placed_var_ids: List[str] = []
                placed_betas: List[float] = []
                for _, prs_row in prs_df.iterrows():
                    var_id = prs_row["variant_id"]
                    is_observed = (
                        var_id in observed_variant_ids
                        and var_id not in ambiguous_excluded_ids
                    )
                    if not (is_observed or var_id in covered_ids):
                        continue
                    match = match_oriented_dosage(
                        prs_row["chromosome"], int(prs_row["position"]),
                        prs_row["effect_allele"], prs_row.get("other_allele"),
                        genotype_data.variant_info, genotype_data.dosage_matrix,
                        reference_index,
                    )
                    if match is None:
                        continue
                    placed_columns.append(match[1])
                    placed_var_ids.append(var_id)
                    placed_betas.append(float(prs_row["beta"]))

                if placed_columns:
                    X_full = np.nan_to_num(
                        np.column_stack(placed_columns).astype(np.float32)
                    )
                    all_betas = np.array(placed_betas)
                    id_to_col = {vid: i for i, vid in enumerate(placed_var_ids)}

                    # Observed component for S_cv (oriented true genotypes x betas)
                    observed_indices = np.array(
                        [id_to_col[vid] for vid in placed_var_ids
                         if vid in observed_variant_ids],
                        dtype=int,
                    )
                    observed_betas = np.array(
                        [all_betas[i] for i in observed_indices]
                    )

                    n_samples = X_full.shape[0]
                    s_cv = np.zeros(n_samples)
                    if len(observed_indices) > 0:
                        s_cv += X_full[:, observed_indices] @ observed_betas

                    # Projected component: sum region CV predictions of S_R
                    # (already S_R = X_R @ beta_R on oriented dosages).
                    for region_id, cv_pred in training_result.cv_predictions.items():
                        s_cv += cv_pred

                    # True PRS from the same effect-oriented matrix
                    s_true = X_full @ all_betas

                    # Estimate calibration parameters
                    calibration_params = estimate_cv_calibration(s_cv, s_true)

                    if self.verbose >= 2:
                        print(
                            f"Calibration: R²={calibration_params.calibration_r2:.3f}, "
                            f"scaling={calibration_params.scaling_factor:.3f}"
                        )

            except (ValueError, IndexError) as e:
                if self.verbose >= 1:
                    print(f"Warning: Could not compute calibration parameters: {e}")
                calibration_params = None

        # Step 11: Build observed VariantInfo objects (excluding QC-dropped SNPs)
        observed_variants_list: List[VariantInfo] = []
        for _, row in prs_df[prs_df["variant_id"].isin(observed_variant_ids)].iterrows():
            if row["variant_id"] in ambiguous_excluded_ids:
                continue
            other_allele = row.get("other_allele")
            if pd.isna(other_allele):
                other_allele = None

            observed_variants_list.append(
                VariantInfo(
                    variant_id=row["variant_id"],
                    chromosome=str(row["chromosome"]),
                    position=int(row["position"]),
                    effect_allele=row["effect_allele"],
                    other_allele=other_allele,
                    beta=float(row["beta"]),
                )
            )

        # Step 11b: Record a disposition for every input PRS variant so coverage
        # is reported honestly (no silent loss).
        observed_kept_ids = {v.variant_id for v in observed_variants_list}
        covered_ids: Set[str] = set()
        for region in training_result.region_models.values():
            covered_ids.update(region.prs_variant_ids)
        self._variant_dispositions = self._build_variant_dispositions(
            prs_df=prs_df,
            observed_kept_ids=observed_kept_ids,
            covered_ids=covered_ids,
            ambiguous_excluded_ids=ambiguous_excluded_ids,
            missing_drop_reason=missing_drop_reason,
        )

        # Step 12: Populate instance state
        self._is_fitted = True
        self._observed_variants = observed_variants_list
        self._region_models = list(training_result.region_models.values())
        self._calibration_params = calibration_params
        self._training_result = training_result

        # Build platform variant index mapping
        self._platform_variant_index = {}
        for i, row in enumerate(platform_variant_rows):
            self._platform_variant_index[row["variant_id"]] = i

        self._prs_id = effective_prs_id
        self._platform_name = effective_platform_name
        self._genome_build = effective_genome_build
        self._model_name = model_name

        if self.verbose >= 1:
            print(
                f"Model fitted: {len(self._observed_variants)} observed variants, "
                f"{training_result.n_regions_trained} projection regions"
            )

        return self

    def _build_variant_dispositions(
        self,
        prs_df: pd.DataFrame,
        observed_kept_ids: Set[str],
        covered_ids: Set[str],
        ambiguous_excluded_ids: Set[str],
        missing_drop_reason: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """Build one disposition record per input PRS variant.

        Every row of ``prs_df`` yields exactly one record. ``status`` is one of
        {observed, projected, dropped}; ``reason`` is None for placed variants,
        or one of {ambiguous_excluded, reference_contig_missing, allele_mismatch,
        not_in_reference, training_failed} for dropped ones.
        """
        dispositions: List[Dict[str, Any]] = []
        for _, row in prs_df.iterrows():
            var_id = row["variant_id"]
            other_allele = row.get("other_allele")
            if pd.isna(other_allele):
                other_allele = None

            reason: Optional[str] = None
            if var_id in ambiguous_excluded_ids:
                status, reason = "dropped", "ambiguous_excluded"
            elif var_id in observed_kept_ids:
                status = "observed"
            elif var_id in covered_ids:
                status = "projected"
            else:
                status = "dropped"
                reason = missing_drop_reason.get(var_id, "training_failed")

            dispositions.append({
                "variant_id": var_id,
                "chromosome": str(row["chromosome"]),
                "position": int(row["position"]),
                "effect_allele": row["effect_allele"],
                "other_allele": other_allele,
                "beta": float(row["beta"]),
                "status": status,
                "reason": reason,
            })
        return dispositions

    def predict(
        self,
        user_genotypes: Union[str, Path, pd.DataFrame, Dict[str, float]],
        apply_calibration: bool = True,
        *,
        genome_build: Optional[str] = None,
        platform_id: Optional[str] = None,
        strict: bool = True,
    ) -> PredictionResult:
        """Compute PRS for user genotypes using projection models.

        Args:
            user_genotypes: User genotype data as file path, DataFrame, or dict.
            apply_calibration: Whether to apply calibration scaling. Default: True.
            genome_build: Genome build of the user genotypes (e.g. "GRCh37").
                Overrides auto-detection. For file inputs the build is
                auto-detected when omitted; DataFrame/dict inputs are not.
            platform_id: Genotyping platform the user genotypes came from. When
                provided, it is checked against the platform the model was
                trained for.
            strict: If True (default), an incompatible genome build or a declared
                platform mismatch raises. If False, the mismatch is downgraded to
                a blocking UserWarning and scoring proceeds.

        Returns:
            PredictionResult with PRS value and uncertainty estimates.

        Raises:
            ModelNotFittedError: If model has not been fitted.
            IncompatibleBuildError: If strict and the user build is known and
                mismatches the model's build.
            IncompatiblePlatformError: If strict and platform_id mismatches the
                model's platform.

        Warns:
            UserWarning: If the user build cannot be determined while the model
                declares one, or if strict=False downgrades a build/platform
                mismatch.
        """
        if not self._is_fitted:
            raise ModelNotFittedError(
                "Model has not been fitted. Call fit() before predict()."
            )

        # Refuse (or hard-block) before scoring when the upload is incompatible
        # with the model's training build / platform.
        check_predict_compatibility(
            model_build=self._genome_build,
            model_platform=self._platform_name,
            user_input=user_genotypes,
            declared_build=genome_build,
            declared_platform=platform_id,
            strict=strict,
        )

        expected_variants = self._get_expected_variants()

        # See LinearImputationPRS.predict: real uploads also load a multi-key raw
        # collection for allele-aware observed scoring; the projected component
        # still reads the legacy dosage dict (unchanged until P1.4), hence the
        # deliberate double parse.
        raw_genotypes = None
        if isinstance(user_genotypes, dict):
            user_dosages = user_genotypes
        else:
            user_dosages = load_user_genotypes(user_genotypes, expected_variants)
            raw_genotypes = load_raw_user_genotypes(user_genotypes)

        predictor = ProjectionPredictor(
            observed_variants=self._observed_variants,
            region_models=self._region_models,
            calibration_params=self._calibration_params,
        )
        return predictor.predict(
            user_dosages,
            apply_calibration=apply_calibration,
            raw_genotypes=raw_genotypes,
        )

    def _get_expected_variants(self) -> Set[str]:
        """Get set of all variant IDs needed for prediction."""
        expected = set()
        for var in self._observed_variants or []:
            expected.add(var.variant_id)
        for model in self._region_models or []:
            expected.update(model.predictor_variant_ids)
        return expected

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted."""
        return self._is_fitted

    @property
    def observed_variants(self) -> List[VariantInfo]:
        """List of observed (directly measured) variants.

        Raises:
            ModelNotFittedError: If fit() has not been called.
        """
        if not self._is_fitted:
            raise ModelNotFittedError(
                "Model has not been fitted. Call fit() first."
            )
        return self._observed_variants or []

    @property
    def region_models(self) -> List[ProjectionRegionModel]:
        """List of trained projection region models.

        Raises:
            ModelNotFittedError: If fit() has not been called.
        """
        if not self._is_fitted:
            raise ModelNotFittedError(
                "Model has not been fitted. Call fit() first."
            )
        return self._region_models or []

    @property
    def calibration_params(self) -> Optional[CalibrationParams]:
        """Calibration parameters from CV training.

        Raises:
            ModelNotFittedError: If fit() has not been called.
        """
        if not self._is_fitted:
            raise ModelNotFittedError(
                "Model has not been fitted. Call fit() first."
            )
        return self._calibration_params

    @property
    def summary(self) -> Dict[str, Any]:
        """Model summary with region counts and quality statistics.

        Raises:
            ModelNotFittedError: If fit() has not been called.
        """
        if not self._is_fitted:
            raise ModelNotFittedError(
                "Model has not been fitted. Call fit() first."
            )

        n_observed = len(self._observed_variants or [])
        n_missing = sum(
            len(m.prs_variant_ids) for m in (self._region_models or [])
        )
        n_regions = len(self._region_models or [])
        n_intercept_only = sum(
            1 for m in (self._region_models or []) if m.is_intercept_only
        )

        calibration_dict = None
        if self._calibration_params is not None:
            from dataclasses import asdict
            calibration_dict = asdict(self._calibration_params)

        # Honest coverage from the per-variant disposition record.
        if self._variant_dispositions is not None:
            n_definition = len(self._variant_dispositions)
            dropped_by_reason: Dict[str, int] = {}
            for d in self._variant_dispositions:
                if d["status"] == "dropped":
                    key = d["reason"] or "unknown"
                    dropped_by_reason[key] = dropped_by_reason.get(key, 0) + 1
            n_dropped = sum(dropped_by_reason.values())
        else:
            n_definition = n_observed + n_missing
            dropped_by_reason = {}
            n_dropped = 0
        coverage = (n_observed + n_missing) / n_definition if n_definition else 0.0

        return {
            "n_observed_variants": n_observed,
            "n_missing_variants": n_missing,
            "n_definition_variants": n_definition,
            "n_dropped": n_dropped,
            "dropped_by_reason": dropped_by_reason,
            "coverage": coverage,
            "n_regions": n_regions,
            "n_intercept_only_regions": n_intercept_only,
            "training_summary": (
                self._training_result.training_summary
                if self._training_result else {}
            ),
            "calibration": calibration_dict,
            "prs_id": self._prs_id,
            "platform_name": self._platform_name,
            "genome_build": self._genome_build,
            "model_name": self._model_name,
            "window_size": self.window_size,
            "cv_folds": self.cv_folds,
        }

    @property
    def variant_dispositions(self) -> pd.DataFrame:
        """Per-variant disposition table (status/reason for every PRS variant).

        Raises:
            ModelNotFittedError: If fit() has not been called.
        """
        if not self._is_fitted:
            raise ModelNotFittedError(
                "Model has not been fitted. Call fit() first."
            )
        return pd.DataFrame(self._variant_dispositions or [])

    @property
    def variant_table(self) -> pd.DataFrame:
        """Per-region summary table.

        Returns:
            DataFrame with columns: region_id, chromosome, start, end,
            n_prs_variants, n_predictors, cv_r2, cv_mse, is_intercept_only,
            prs_variant_ids.

        Raises:
            ModelNotFittedError: If fit() has not been called.
        """
        if not self._is_fitted:
            raise ModelNotFittedError(
                "Model has not been fitted. Call fit() first."
            )

        rows = []
        for model in self._region_models or []:
            rows.append({
                "region_id": model.region_id,
                "chromosome": model.chromosome,
                "start": model.start,
                "end": model.end,
                "n_prs_variants": len(model.prs_variant_ids),
                "n_predictors": len(model.predictor_variant_ids),
                "cv_r2": model.cv_r2,
                "cv_mse": model.cv_mse,
                "is_intercept_only": model.is_intercept_only,
                "prs_variant_ids": model.prs_variant_ids,
            })

        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        """String representation of the model."""
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"LinearProjectionPRS(window_size={self.window_size}, "
            f"cv_folds={self.cv_folds}, status={status})"
        )
