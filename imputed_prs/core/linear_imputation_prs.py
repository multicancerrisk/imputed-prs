"""Main LinearImputationPRS class for training and prediction."""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Union

import numpy as np
import pandas as pd

from imputed_prs.core.exceptions import DataLoadError, ModelNotFittedError, ValidationError
from imputed_prs.core.harmonizer import (
    align_effect_alleles,
    partition_variants,
    validate_genome_build,
)
from imputed_prs.core.types import (
    CalibrationParams,
    EvaluationMetrics,
    GenotypeData,
    ImputedVariantModel,
    PredictionResult,
    TrainingResult,
    VariantInfo,
)
from imputed_prs.evaluation.calibration import (
    compute_cv_predicted_prs,
    estimate_cv_calibration,
)
from imputed_prs.io.genotype_loader import load_genotypes
from imputed_prs.io.pgs_catalog import download_pgs_catalog_score
from imputed_prs.io.platform_loader import (
    load_platform_from_manifest,
    load_platform_from_name,
    load_platform_variants_from_list,
)
from imputed_prs.io.exporters import (
    export_to_arrow,
    export_to_hdf5,
    export_to_json,
    export_to_parquet,
    export_variant_table,
)
from imputed_prs.io.loaders import load_model_hdf5
from imputed_prs.io.prs_loader import load_prs_from_dataframe, load_prs_from_file
from imputed_prs.io.user_genotypes import load_user_genotypes
from imputed_prs.models.predictor import PRSPredictor
from imputed_prs.models.trainer import ImputationModelTrainer
from imputed_prs.models.tuning import global_hyperparameter_search


class LinearImputationPRS:
    """High-level API for training and using imputation-based PRS models.

    This class provides a unified interface for:
    - Loading PRS definitions and platform information
    - Training imputation models on reference genotype data
    - Computing PRS predictions with uncertainty estimates
    - Exporting trained models to portable formats

    Example:
        >>> model = LinearImputationPRS(window_size=1_000_000, cv_folds=5)
        >>> model.fit(
        ...     reference_genotypes="1000g_eur.vcf.gz",
        ...     prs_definition="PGS000004",
        ...     platform_name="23andme_v5",
        ... )
        >>> result = model.predict("user_genotypes.txt")
        >>> print(f"PRS: {result.prs:.3f} (95% CI: {result.ci_lower:.3f}-{result.ci_upper:.3f})")

    Attributes:
        window_size: Size of genomic window (bp) for selecting predictor variants.
        tuning_scope: Hyperparameter tuning strategy ("global", "per_variant", or "none").
        l1_ratio: ElasticNet L1/L2 mixing parameter (0=Ridge, 1=Lasso).
        alpha: ElasticNet regularization strength.
        cv_folds: Number of cross-validation folds.
        n_jobs: Number of parallel jobs for training.
        random_state: Random seed for reproducibility.
        verbose: Verbosity level (0=silent, 1=progress, 2=debug).
    """

    def __init__(
        self,
        window_size: int = 1_000_000,
        tuning_scope: Literal["global", "per_variant", "none"] = "global",
        l1_ratio: float = 0.5,
        alpha: float = 0.01,
        cv_folds: int = 5,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
        max_predictors: Optional[int] = None,
        verbose: int = 1,
    ):
        """Initialize LinearImputationPRS model.

        Args:
            window_size: Size of genomic window (bp) for selecting predictor variants.
                Larger windows include more potential predictors but increase computation.
                Default: 1,000,000 (1 Mb).
            tuning_scope: Hyperparameter tuning strategy:
                - "global": Tune once on subset of variants (recommended)
                - "per_variant": Tune separately for each variant (slow)
                - "none": Use provided l1_ratio and alpha directly
                Default: "global".
            l1_ratio: ElasticNet L1/L2 mixing parameter. 0=pure Ridge, 1=pure Lasso.
                Only used when tuning_scope="none". Default: 0.5.
            alpha: ElasticNet regularization strength. Larger values = more regularization.
                Only used when tuning_scope="none". Default: 0.01.
            cv_folds: Number of cross-validation folds for training and calibration.
                Default: 5.
            n_jobs: Number of parallel jobs for training (-1 for all CPUs).
                Default: 1 (sequential).
            random_state: Random seed for reproducibility. Default: None.
            max_predictors: Maximum number of predictor variants per model.
                If None, uses all variants in window. Default: None.
            verbose: Verbosity level. 0=silent, 1=progress bar, 2=debug output.
                Default: 1.
        """
        # Configuration parameters
        self.window_size = window_size
        self.tuning_scope = tuning_scope
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.cv_folds = cv_folds
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.max_predictors = max_predictors
        self.verbose = verbose

        # Fitted state (populated by fit())
        self._is_fitted: bool = False
        self._observed_variants: Optional[List[VariantInfo]] = None
        self._imputed_models: Optional[List[ImputedVariantModel]] = None
        self._calibration_params: Optional[CalibrationParams] = None
        self._evaluation_metrics: Optional[EvaluationMetrics] = None
        self._training_result: Optional[TrainingResult] = None
        self._platform_variant_index: Optional[Dict[str, int]] = None

        # Metadata (populated by fit())
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
        evaluation_genotypes: Optional[Union[str, Path]] = None,
    ) -> "LinearImputationPRS":
        """Train imputation models on reference genotype data.

        Exactly one platform source must be provided (platform_name, platform_manifest,
        or platform_variants).

        Args:
            reference_genotypes: Path to reference genotype file (VCF or PLINK format).
            prs_definition: PRS definition as PGS Catalog ID (e.g., "PGS000004"),
                file path, or DataFrame with variant weights.
            platform_name: Name of pre-built platform (e.g., "23andme_v5").
            platform_manifest: Path to platform manifest file.
            platform_variants: List of platform variant IDs.
            genome_build: Genome build ("GRCh37" or "GRCh38"). Auto-detected if None.
            prs_id: PRS identifier for metadata.
            model_name: Human-readable model name for metadata.
            evaluation_genotypes: Optional holdout genotypes for external evaluation.

        Returns:
            self (for method chaining).

        Raises:
            ValidationError: If inputs are invalid or incompatible.
            DataLoadError: If files cannot be loaded.
        """
        # Step 1: Input validation - exactly one platform source must be provided
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
            # PGS Catalog ID
            prs_df, pgs_metadata = download_pgs_catalog_score(
                prs_definition,
                genome_build=genome_build or "GRCh37",
            )
            if effective_prs_id is None:
                effective_prs_id = prs_definition.upper()
            if effective_genome_build is None and pgs_metadata:
                effective_genome_build = pgs_metadata.genome_build
        else:
            # File path
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
            # platform_variants list
            platform_variant_set = load_platform_variants_from_list(platform_variants)
            effective_platform_name = "custom"

        if self.verbose >= 2:
            print(f"Loaded {len(platform_variant_set)} platform variants")

        # Step 4: Partition variants into observed and missing
        partition_result = partition_variants(prs_df, platform_variant_set)
        observed_variant_ids = partition_result.observed  # FrozenSet[str]
        missing_variant_ids = partition_result.missing  # FrozenSet[str]

        if self.verbose >= 1:
            print(
                f"Partitioned variants: {len(observed_variant_ids)} observed, "
                f"{len(missing_variant_ids)} missing"
            )

        # Step 5: Load reference genotypes
        # Include PRS variant chr:pos so variants are loaded even if rsIDs
        # in the reference VCF differ from those in the PRS definition
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

        # Step 7: Align effect alleles for observed variants
        alignment_result = align_effect_alleles(
            prs_df, genotype_data, observed_variant_ids
        )

        if self.verbose >= 2:
            print(
                f"Allele alignment: {alignment_result.n_matched} matched, "
                f"{alignment_result.n_flipped} flipped"
            )

        # Step 8: Build training matrices
        # Build a mapping from variant_id to index in genotype_data
        geno_var_to_idx: Dict[str, int] = {}
        for idx, row in genotype_data.variant_info.iterrows():
            geno_var_to_idx[row["variant_id"]] = idx
            # Also add chr:pos lookup
            chrom = str(row["chromosome"]).upper()
            if chrom.startswith("CHR"):
                chrom = chrom[3:]
            pos = str(int(row["position"]))
            geno_var_to_idx[f"{chrom}:{pos}"] = idx

        # Build platform variant info DataFrame and Z matrix
        platform_variant_indices = []
        platform_variant_rows = []
        for var_id in platform_variant_set:
            # Try to find in genotype data
            if var_id in geno_var_to_idx:
                idx = geno_var_to_idx[var_id]
                platform_variant_indices.append(idx)
                row = genotype_data.variant_info.iloc[idx]
                platform_variant_rows.append({
                    "variant_id": row["variant_id"],
                    "chromosome": row["chromosome"],
                    "position": row["position"],
                })
            elif var_id.lower() in geno_var_to_idx:
                idx = geno_var_to_idx[var_id.lower()]
                platform_variant_indices.append(idx)
                row = genotype_data.variant_info.iloc[idx]
                platform_variant_rows.append({
                    "variant_id": row["variant_id"],
                    "chromosome": row["chromosome"],
                    "position": row["position"],
                })

        if platform_variant_rows:
            platform_variant_info = pd.DataFrame(platform_variant_rows)
            Z = genotype_data.dosage_matrix[:, platform_variant_indices]
        else:
            platform_variant_info = pd.DataFrame(
                columns=["variant_id", "chromosome", "position"]
            )
            Z = np.empty((genotype_data.n_samples, 0))

        # Build missing PRS DataFrame and X matrix
        missing_prs_df = prs_df[prs_df["variant_id"].isin(missing_variant_ids)].copy()
        missing_prs_df = missing_prs_df.reset_index(drop=True)

        missing_variant_indices = []
        for _, row in missing_prs_df.iterrows():
            var_id = row["variant_id"]
            chrom = str(row["chromosome"]).upper()
            if chrom.startswith("CHR"):
                chrom = chrom[3:]
            chrpos = f"{chrom}:{int(row['position'])}"
            if var_id in geno_var_to_idx:
                missing_variant_indices.append(geno_var_to_idx[var_id])
            elif var_id.lower() in geno_var_to_idx:
                missing_variant_indices.append(geno_var_to_idx[var_id.lower()])
            elif chrpos in geno_var_to_idx:
                missing_variant_indices.append(geno_var_to_idx[chrpos])

        if missing_variant_indices:
            X = genotype_data.dosage_matrix[:, missing_variant_indices]
        else:
            X = np.empty((genotype_data.n_samples, 0))

        # Update missing_prs_df to only include variants found in genotype data
        valid_missing_mask = []
        for _, row in missing_prs_df.iterrows():
            var_id = row["variant_id"]
            chrom = str(row["chromosome"]).upper()
            if chrom.startswith("CHR"):
                chrom = chrom[3:]
            chrpos = f"{chrom}:{int(row['position'])}"
            found = (var_id in geno_var_to_idx
                     or var_id.lower() in geno_var_to_idx
                     or chrpos in geno_var_to_idx)
            valid_missing_mask.append(found)
        missing_prs_df = missing_prs_df[valid_missing_mask].reset_index(drop=True)

        if self.verbose >= 2:
            print(
                f"Training matrices: Z={Z.shape}, X={X.shape}, "
                f"missing_prs_df={len(missing_prs_df)} variants"
            )

        # Step 9: Hyperparameter tuning (if tuning_scope="global")
        if self.tuning_scope == "global" and X.shape[1] > 0 and Z.shape[1] > 0:
            if self.verbose >= 1:
                print("Running global hyperparameter search...")
            try:
                grid_result = global_hyperparameter_search(
                    Z=Z,
                    X_missing=X,
                    sample_indices=None,  # Use all variants
                    cv_folds=self.cv_folds,
                    random_state=self.random_state,
                )
                effective_l1_ratio = grid_result.best_l1_ratio
                effective_alpha = grid_result.best_alpha
                if self.verbose >= 1:
                    print(
                        f"Best hyperparameters: l1_ratio={effective_l1_ratio}, "
                        f"alpha={effective_alpha}"
                    )
            except ValidationError:
                # Fall back to defaults if tuning fails
                effective_l1_ratio = self.l1_ratio
                effective_alpha = self.alpha
                if self.verbose >= 1:
                    print("Hyperparameter search failed, using defaults")
        elif self.tuning_scope == "none" or X.shape[1] == 0 or Z.shape[1] == 0:
            effective_l1_ratio = self.l1_ratio
            effective_alpha = self.alpha
        else:
            # per_variant tuning - let trainer handle it
            effective_l1_ratio = self.l1_ratio
            effective_alpha = self.alpha

        # Step 10: Train imputation models
        if X.shape[1] > 0:
            if self.verbose >= 1:
                print(f"Training imputation models for {X.shape[1]} variants...")

            trainer = ImputationModelTrainer(
                window_size=self.window_size,
                l1_ratio=effective_l1_ratio,
                alpha=effective_alpha,
                cv_folds=self.cv_folds,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                max_predictors=self.max_predictors,
                verbose=self.verbose,
            )
            training_result = trainer.fit_all_variants(
                Z=Z,
                X=X,
                prs_variants=missing_prs_df,
                platform_variant_info=platform_variant_info,
            )

            if self.verbose >= 1:
                print(
                    f"Trained {training_result.n_variants_trained} models, "
                    f"{training_result.n_intercept_only} intercept-only"
                )
        else:
            # No missing variants to impute
            training_result = TrainingResult(
                models={},
                cv_predictions={},
                n_variants_trained=0,
                n_variants_failed=0,
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
                },
            )

        # Step 11: Compute calibration parameters
        calibration_params = None
        if training_result.cv_predictions and len(training_result.cv_predictions) > 0:
            try:
                # Build full dosage matrix X_full for all PRS variants
                all_prs_indices = []
                for _, prs_row in prs_df.iterrows():
                    var_id = prs_row["variant_id"]
                    chrom = str(prs_row["chromosome"]).upper()
                    if chrom.startswith("CHR"):
                        chrom = chrom[3:]
                    chrpos = f"{chrom}:{int(prs_row['position'])}"
                    if var_id in geno_var_to_idx:
                        all_prs_indices.append(geno_var_to_idx[var_id])
                    elif var_id.lower() in geno_var_to_idx:
                        all_prs_indices.append(geno_var_to_idx[var_id.lower()])
                    elif chrpos in geno_var_to_idx:
                        all_prs_indices.append(geno_var_to_idx[chrpos])

                if all_prs_indices:
                    X_full = genotype_data.dosage_matrix[:, all_prs_indices]

                    # Get observed variant indices and betas
                    observed_prs_df = prs_df[
                        prs_df["variant_id"].isin(observed_variant_ids)
                    ]
                    observed_indices_in_full = []
                    observed_betas = []
                    prs_var_list = list(prs_df["variant_id"])
                    for var_id in observed_prs_df["variant_id"]:
                        if var_id in prs_var_list:
                            idx = prs_var_list.index(var_id)
                            observed_indices_in_full.append(idx)
                            observed_betas.append(
                                observed_prs_df[
                                    observed_prs_df["variant_id"] == var_id
                                ]["beta"].values[0]
                            )

                    observed_indices = np.array(observed_indices_in_full, dtype=int)
                    observed_betas = np.array(observed_betas)

                    # Build CV predictions dict with indices into full matrix
                    cv_preds_by_idx: Dict[int, np.ndarray] = {}
                    missing_betas = []
                    for var_id, cv_pred in training_result.cv_predictions.items():
                        if var_id in prs_var_list:
                            idx = prs_var_list.index(var_id)
                            cv_preds_by_idx[idx] = cv_pred
                            beta_val = prs_df[
                                prs_df["variant_id"] == var_id
                            ]["beta"].values[0]
                            missing_betas.append(beta_val)

                    missing_betas = np.array(missing_betas)

                    # Compute CV-predicted PRS
                    s_cv = compute_cv_predicted_prs(
                        X=X_full,
                        observed_variant_indices=observed_indices,
                        observed_betas=observed_betas,
                        cv_predictions=cv_preds_by_idx,
                        missing_betas=missing_betas,
                    )

                    # Compute true PRS
                    all_betas = prs_df["beta"].values
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

        # Step 12: Build observed VariantInfo objects
        observed_variants_list: List[VariantInfo] = []
        for _, row in prs_df[prs_df["variant_id"].isin(observed_variant_ids)].iterrows():
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

        # Step 13: Populate instance state
        self._is_fitted = True
        self._observed_variants = observed_variants_list
        self._imputed_models = list(training_result.models.values())
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
                f"Model fitted: {len(self._observed_variants)} observed, "
                f"{len(self._imputed_models)} imputed variants"
            )

        # Step 14: Return self for method chaining
        return self

    def predict(
        self,
        user_genotypes: Union[str, Path, pd.DataFrame, Dict[str, float]],
        apply_calibration: bool = True,
    ) -> PredictionResult:
        """Compute PRS for user genotypes.

        Args:
            user_genotypes: User genotype data as:
                - File path (DTC format auto-detected)
                - DataFrame with variant_id and genotype columns
                - Dict mapping variant_id to dosage values
            apply_calibration: Whether to apply calibration scaling.
                Default: True.

        Returns:
            PredictionResult with PRS value, uncertainty estimates, and diagnostics.

        Raises:
            ModelNotFittedError: If fit() has not been called.
            DataLoadError: If user genotype file cannot be loaded.
        """
        if not self._is_fitted:
            raise ModelNotFittedError(
                "Model has not been fitted. Call fit() before predict()."
            )

        # Get expected variants for filtering
        expected_variants = self._get_expected_variants()

        # Handle dict input directly, otherwise use loader
        if isinstance(user_genotypes, dict):
            user_dosages = user_genotypes
        else:
            user_dosages = load_user_genotypes(user_genotypes, expected_variants)

        # Create PRSPredictor and compute prediction
        predictor = PRSPredictor(
            observed_variants=self._observed_variants,
            imputed_models=self._imputed_models,
            calibration_params=self._calibration_params,
        )
        return predictor.predict(user_dosages, apply_calibration=apply_calibration)

    def _get_expected_variants(self) -> Set[str]:
        """Get set of all variant IDs needed for prediction.

        Returns:
            Set of variant IDs including observed variants and predictor variants
            from imputation models.
        """
        expected = set()
        for var in self._observed_variants or []:
            expected.add(var.variant_id)
        for model in self._imputed_models or []:
            expected.update(model.predictor_variant_ids)
        return expected

    def export(
        self,
        output_dir: Union[str, Path],
        model_name: Optional[str] = None,
        formats: Optional[List[str]] = None,
        include_variance_scaling: bool = True,
    ) -> Dict[str, Path]:
        """Export trained model to portable formats.

        Args:
            output_dir: Directory for output files.
            model_name: Base name for output files. Uses self._model_name if None.
            formats: List of formats to export. Options: "json", "arrow", "parquet",
                "hdf5", "csv". Default: ["json", "hdf5"].
            include_variance_scaling: Whether to include variance/SE components.
                Default: True.

        Returns:
            Dict mapping format name to output file path.

        Raises:
            ModelNotFittedError: If fit() has not been called.
            ValueError: If an unsupported format is requested.
        """
        if not self._is_fitted:
            raise ModelNotFittedError(
                "Model has not been fitted. Call fit() before export()."
            )

        # Default formats
        if formats is None:
            formats = ["json", "hdf5"]

        # Resolve model name
        effective_model_name = model_name or self._model_name or "imputed_prs_model"

        # Convert output_dir to Path
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get training summary if available
        training_summary = None
        if self._training_result is not None:
            training_summary = self._training_result.training_summary

        # Common kwargs for all export functions
        common_kwargs = {
            "observed_variants": self._observed_variants or [],
            "imputed_models": self._imputed_models or [],
            "calibration_params": self._calibration_params,
            "evaluation_metrics": self._evaluation_metrics,
            "platform_name": self._platform_name,
            "prs_id": self._prs_id,
            "genome_build": self._genome_build,
            "model_name": effective_model_name,
            "include_variance_scaling": include_variance_scaling,
            "training_summary": training_summary,
        }

        # Valid formats
        valid_formats = {"json", "arrow", "parquet", "hdf5", "csv"}
        invalid_formats = set(formats) - valid_formats
        if invalid_formats:
            raise ValueError(
                f"Unsupported export formats: {invalid_formats}. "
                f"Valid formats: {valid_formats}"
            )

        # Export to each requested format
        output_paths: Dict[str, Path] = {}

        for fmt in formats:
            if fmt == "json":
                output_path = output_dir / f"{effective_model_name}.json"
                export_to_json(output_path=output_path, **common_kwargs)
                output_paths["json"] = output_path

            elif fmt == "arrow":
                output_path = output_dir / f"{effective_model_name}.arrow"
                export_to_arrow(output_path=output_path, **common_kwargs)
                output_paths["arrow"] = output_path

            elif fmt == "parquet":
                parquet_dir = output_dir / f"{effective_model_name}_parquet"
                parquet_paths = export_to_parquet(output_path=parquet_dir, **common_kwargs)
                # Store the directory path as the main parquet output
                output_paths["parquet"] = parquet_dir

            elif fmt == "hdf5":
                output_path = output_dir / f"{effective_model_name}.h5"
                export_to_hdf5(output_path=output_path, **common_kwargs)
                output_paths["hdf5"] = output_path

            elif fmt == "csv":
                output_path = output_dir / f"{effective_model_name}_variants.csv"
                export_variant_table(
                    output_path=output_path,
                    observed_variants=self._observed_variants or [],
                    imputed_models=self._imputed_models or [],
                    include_variance_scaling=include_variance_scaling,
                )
                output_paths["csv"] = output_path

        return output_paths

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LinearImputationPRS":
        """Load a trained model from file.

        Args:
            path: Path to saved model file (HDF5 or JSON format).

        Returns:
            Loaded LinearImputationPRS instance ready for prediction.

        Raises:
            DataLoadError: If file cannot be loaded or format is unsupported.
        """
        path = Path(path)

        if not path.exists():
            raise DataLoadError(f"Model file not found: {path}")

        # Detect format by extension
        suffix = path.suffix.lower()

        if suffix in (".h5", ".hdf5"):
            return cls._load_from_hdf5(path)
        elif suffix == ".json":
            return cls._load_from_json(path)
        else:
            raise DataLoadError(
                f"Unsupported model file format: {suffix}. "
                "Supported formats: .h5, .hdf5, .json"
            )

    @classmethod
    def _load_from_hdf5(cls, path: Path) -> "LinearImputationPRS":
        """Load model from HDF5 file."""
        try:
            observed, imputed, calib, metrics, metadata = load_model_hdf5(path)
        except Exception as e:
            raise DataLoadError(f"Failed to load HDF5 model: {e}") from e

        # Create instance with default parameters
        instance = cls()

        # Populate fitted state
        instance._is_fitted = True
        instance._observed_variants = observed
        instance._imputed_models = imputed
        instance._calibration_params = calib
        instance._evaluation_metrics = metrics

        # Populate metadata from loaded file
        instance._prs_id = metadata.get("prs_id") or None
        instance._platform_name = metadata.get("platform_name") or None
        instance._genome_build = metadata.get("genome_build") or None
        instance._model_name = metadata.get("model_name") or None

        # Handle empty string values from HDF5
        if instance._prs_id == "":
            instance._prs_id = None
        if instance._platform_name == "":
            instance._platform_name = None
        if instance._genome_build == "":
            instance._genome_build = None
        if instance._model_name == "":
            instance._model_name = None

        return instance

    @classmethod
    def _load_from_json(cls, path: Path) -> "LinearImputationPRS":
        """Load model from JSON file."""
        from imputed_prs.io.loaders import load_model_json

        try:
            data = load_model_json(path)
        except Exception as e:
            raise DataLoadError(f"Failed to load JSON model: {e}") from e

        # Create instance with default parameters
        instance = cls()

        # Parse observed variants
        observed_variants = []
        for v in data.get("observed_variants", []):
            observed_variants.append(
                VariantInfo(
                    variant_id=v["variant_id"],
                    chromosome=v["chromosome"],
                    position=v["position"],
                    effect_allele=v["effect_allele"],
                    other_allele=v.get("other_allele"),
                    beta=v["beta"],
                )
            )

        # Parse imputed models
        imputed_models = []
        for m in data.get("imputed_variants", []):
            imputed_models.append(
                ImputedVariantModel(
                    variant_id=m["variant_id"],
                    chromosome=m["chromosome"],
                    position=m["position"],
                    effect_allele=m["effect_allele"],
                    other_allele=m.get("other_allele"),
                    beta=m["beta"],
                    allele_frequency=m["allele_frequency"],
                    imputation_r2=m["imputation_r2"],
                    residual_variance=m.get("residual_variance", 0.0),
                    intercept=m["intercept"],
                    predictor_variant_ids=m.get("predictor_variant_ids", []),
                    coefficients=np.array(m.get("coefficients", [])),
                    is_intercept_only=m.get("is_intercept_only", False),
                )
            )

        # Parse calibration params if present
        calib_params = None
        if "calibration_params" in data and data["calibration_params"]:
            calib_params = CalibrationParams(**data["calibration_params"])

        # Parse evaluation metrics if present
        eval_metrics = None
        if "evaluation_metrics" in data and data["evaluation_metrics"]:
            eval_metrics = EvaluationMetrics(**data["evaluation_metrics"])

        # Populate fitted state
        instance._is_fitted = True
        instance._observed_variants = observed_variants
        instance._imputed_models = imputed_models
        instance._calibration_params = calib_params
        instance._evaluation_metrics = eval_metrics

        # Populate metadata
        metadata = data.get("metadata", {})
        instance._prs_id = metadata.get("prs_id")
        instance._platform_name = metadata.get("platform_name")
        instance._genome_build = metadata.get("genome_build")
        instance._model_name = metadata.get("model_name")

        return instance

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted."""
        return self._is_fitted

    @property
    def variant_table(self) -> pd.DataFrame:
        """Per-variant summary table with status and quality metrics.

        Returns:
            DataFrame with columns: variant_id, chromosome, position, effect_allele,
            other_allele, beta, status, imputation_r2, allele_frequency, n_predictors.

        Raises:
            ModelNotFittedError: If fit() has not been called.
        """
        if not self._is_fitted:
            raise ModelNotFittedError(
                "Model has not been fitted. Call fit() first."
            )
        # Build variant table from observed and imputed variants
        rows = []

        for var in self._observed_variants or []:
            rows.append({
                "variant_id": var.variant_id,
                "chromosome": var.chromosome,
                "position": var.position,
                "effect_allele": var.effect_allele,
                "other_allele": var.other_allele,
                "beta": var.beta,
                "status": "observed",
                "imputation_r2": None,
                "allele_frequency": None,
                "n_predictors": 0,
            })

        for model in self._imputed_models or []:
            rows.append({
                "variant_id": model.variant_id,
                "chromosome": model.chromosome,
                "position": model.position,
                "effect_allele": model.effect_allele,
                "other_allele": model.other_allele,
                "beta": model.beta,
                "status": "intercept_only" if model.is_intercept_only else "imputed",
                "imputation_r2": model.imputation_r2,
                "allele_frequency": model.allele_frequency,
                "n_predictors": len(model.predictor_variant_ids),
            })

        return pd.DataFrame(rows)

    @property
    def summary(self) -> Dict[str, Any]:
        """Model summary with counts and quality statistics.

        Returns:
            Dict with keys: n_total_variants, n_observed, n_imputed, n_intercept_only,
            mean_imputation_r2, prs_id, platform_name, genome_build, etc.

        Raises:
            ModelNotFittedError: If fit() has not been called.
        """
        if not self._is_fitted:
            raise ModelNotFittedError(
                "Model has not been fitted. Call fit() first."
            )

        n_observed = len(self._observed_variants or [])
        n_imputed = len(self._imputed_models or [])
        n_intercept_only = sum(
            1 for m in (self._imputed_models or []) if m.is_intercept_only
        )

        # Compute mean R² for imputed variants
        r2_values = [m.imputation_r2 for m in (self._imputed_models or [])
                     if not m.is_intercept_only]
        mean_r2 = float(np.mean(r2_values)) if r2_values else None

        return {
            "n_total_variants": n_observed + n_imputed,
            "n_observed": n_observed,
            "n_imputed": n_imputed,
            "n_intercept_only": n_intercept_only,
            "mean_imputation_r2": mean_r2,
            "prs_id": self._prs_id,
            "platform_name": self._platform_name,
            "genome_build": self._genome_build,
            "model_name": self._model_name,
            "window_size": self.window_size,
            "cv_folds": self.cv_folds,
        }

    @property
    def evaluation_metrics(self) -> Optional[EvaluationMetrics]:
        """Evaluation metrics from training (if available).

        Returns:
            EvaluationMetrics or None if no evaluation was performed.

        Raises:
            ModelNotFittedError: If fit() has not been called.
        """
        if not self._is_fitted:
            raise ModelNotFittedError(
                "Model has not been fitted. Call fit() first."
            )
        return self._evaluation_metrics

    @property
    def calibration_params(self) -> Optional[CalibrationParams]:
        """Calibration parameters from CV training.

        Returns:
            CalibrationParams or None if calibration was not performed.

        Raises:
            ModelNotFittedError: If fit() has not been called.
        """
        if not self._is_fitted:
            raise ModelNotFittedError(
                "Model has not been fitted. Call fit() first."
            )
        return self._calibration_params

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
    def imputed_models(self) -> List[ImputedVariantModel]:
        """List of imputed variant models.

        Raises:
            ModelNotFittedError: If fit() has not been called.
        """
        if not self._is_fitted:
            raise ModelNotFittedError(
                "Model has not been fitted. Call fit() first."
            )
        return self._imputed_models or []

    def __repr__(self) -> str:
        """String representation of the model."""
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"LinearImputationPRS(window_size={self.window_size}, "
            f"cv_folds={self.cv_folds}, status={status})"
        )
