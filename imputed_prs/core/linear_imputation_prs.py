"""Main LinearImputationPRS class for training and prediction."""

import dataclasses
import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Union

import numpy as np
import pandas as pd

from imputed_prs.core.exceptions import DataLoadError, ModelNotFittedError, ValidationError
from imputed_prs.core.harmonizer import (
    _is_ambiguous_snp,
    _normalize_chromosome,
    ReferenceAlleleResolver,
    check_predict_compatibility,
    hoist_columns,
    normalize_chromosome_array,
    partition_variants,
    validate_genome_build,
)
from imputed_prs.core.types import (
    CalibrationParams,
    EvaluationMetrics,
    GenotypeData,
    ImputedVariantModel,
    PredictionResult,
    TrainingFailure,
    TrainingResult,
    VariantInfo,
)
from imputed_prs.evaluation.calibration import (
    compute_cv_predicted_prs,
    estimate_cv_calibration,
    mean_impute_columns,
)
from imputed_prs.io.genotype_loader import load_genotypes
from imputed_prs.io.genotype_source import make_genotype_source
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
from imputed_prs.io.loaders import (
    load_model_arrow,
    load_model_csv,
    load_model_hdf5,
    load_model_parquet,
)
from imputed_prs.io.prs_loader import load_prs_from_dataframe, load_prs_from_file
from imputed_prs.io.user_genotypes import (
    load_raw_user_genotypes,
    load_user_genotypes,
)
from imputed_prs.models.predictor import PRSPredictor
from imputed_prs.models.trainer import ImputationModelTrainer
from imputed_prs.models.tuning import global_hyperparameter_search

# backend="auto" streams when the estimated dense dosage matrix
# (n_samples x |needed variants| x 4 bytes) exceeds this. Chosen so test-sized
# inputs (<< 1 GB) stay on the dense oracle — keeping the golden gate exact — while
# real reference panels (1000G/UK Biobank scale) select the streaming path.
_AUTO_STREAMING_BYTES_THRESHOLD = 8 * 1024**3  # 8 GiB


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
        max_tuning_variants: Optional[int] = 50,
        exclude_ambiguous: bool = False,
        ambiguous_maf_threshold: float = 0.4,
        backend: Literal["auto", "dense", "streaming"] = "auto",
        verbose: int = 1,
    ):
        """Initialize LinearImputationPRS model.

        Args:
            window_size: Size of genomic window (bp) for selecting predictor variants.
                Larger windows include more potential predictors but increase computation.
                Default: 1,000,000 (1 Mb).
            tuning_scope: Hyperparameter tuning strategy. All modes tune on the
                same local-window matrices used in training:
                - "global": Tune once on a bounded, stratified sample of missing
                  variants (by chromosome / MAF / |beta|) and apply the winning
                  l1_ratio/alpha to every variant (recommended).
                - "per_variant": Grid-search each variant's own local window and
                  give each variant its own l1_ratio/alpha (slow).
                - "none": Use provided l1_ratio and alpha directly.
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
            max_tuning_variants: Cap on the number of missing variants sampled for
                tuning_scope="global". None tunes on all missing variants. Must be
                positive when set. Default: 50.
            exclude_ambiguous: If True, drop strand-ambiguous (palindromic A/T and
                C/G) SNPs whose reference minor-allele frequency exceeds
                ``ambiguous_maf_threshold``, since their strand cannot be resolved
                reliably. Default: False.
            ambiguous_maf_threshold: MAF above which ambiguous SNPs are excluded
                when ``exclude_ambiguous`` is True. Default: 0.4.
            backend: Training backend for reference genotypes.
                - "dense": load the whole dosage matrix in RAM and train per-variant
                  (the original path; the correctness oracle). Feasible only when the
                  matrix fits in memory (≈ n_samples × n_variants × 4 bytes).
                - "streaming": stream the panel once via a chunked ``GenotypeSource``
                  and train from banded sufficient statistics, never materializing the
                  full matrix (Phase 2). Scales to 2M variants × 500K samples in GB of
                  RAM. Requires ``tuning_scope`` in {"none", "global"} and
                  ``exclude_ambiguous=False`` (AF-based streaming QC is a follow-up).
                - "auto": pick "streaming" when the estimated dense matrix is large
                  (so test-sized inputs stay on the dense oracle, keeping the golden
                  gate exact), else "dense". Default: "auto".
            verbose: Verbosity level. 0=silent, 1=progress bar, 2=debug output.
                Default: 1.
        """
        # Configuration parameters
        if tuning_scope not in ("global", "per_variant", "none"):
            raise ValidationError(
                f"tuning_scope must be 'global', 'per_variant', or 'none', "
                f"got {tuning_scope!r}"
            )
        if backend not in ("auto", "dense", "streaming"):
            raise ValidationError(
                f"backend must be 'auto', 'dense', or 'streaming', got {backend!r}"
            )
        if max_tuning_variants is not None and max_tuning_variants <= 0:
            raise ValidationError(
                f"max_tuning_variants must be positive or None, "
                f"got {max_tuning_variants}"
            )
        self.window_size = window_size
        self.tuning_scope = tuning_scope
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.cv_folds = cv_folds
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.max_predictors = max_predictors
        self.max_tuning_variants = max_tuning_variants
        self.exclude_ambiguous = exclude_ambiguous
        self.ambiguous_maf_threshold = ambiguous_maf_threshold
        self.backend = backend
        self.verbose = verbose

        # Fitted state (populated by fit())
        self._is_fitted: bool = False
        self._observed_variants: Optional[List[VariantInfo]] = None
        self._imputed_models: Optional[List[ImputedVariantModel]] = None
        self._calibration_params: Optional[CalibrationParams] = None
        self._evaluation_metrics: Optional[EvaluationMetrics] = None
        self._training_result: Optional[TrainingResult] = None
        self._platform_variant_index: Optional[Dict[str, int]] = None
        # Per-variant disposition records: one dict per input PRS variant with
        # keys variant_id, chromosome, position, effect_allele, other_allele,
        # beta, status, reason. None until fit() is called (and for loaded models).
        self._variant_dispositions: Optional[List[Dict[str, Any]]] = None

        # Metadata (populated by fit())
        self._prs_id: Optional[str] = None
        self._platform_name: Optional[str] = None
        self._genome_build: Optional[str] = None
        self._model_name: Optional[str] = None

        # Provenance for the deployable v2 export (consumed by the build/platform
        # compatibility check). Reference panel / ancestry are set at fit(); the
        # ambiguity policy default follows the deploy decision (training-side
        # exclude_ambiguous stays False, so the model still carries palindromes).
        self._reference_panel_id: Optional[str] = None
        self._training_ancestry: Optional[str] = None
        self._ambiguous_policy: str = "exclude_unless_platform_strand_known"

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
        reference_panel_id: Optional[str] = None,
        training_ancestry: Optional[str] = None,
        evaluation_genotypes: Optional[Union[str, Path]] = None,
        allow_alt_as_effect: bool = False,
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
            reference_panel_id: Provenance — reference panel used for training
                (e.g., "1000G_phase3_EUR"). Recorded in the deployable export.
            training_ancestry: Provenance — ancestry of the training cohort
                (e.g., "EUR"). Recorded in the deployable export.
            evaluation_genotypes: Optional holdout genotypes for external evaluation.
            allow_alt_as_effect: If True, permit a PRS definition that supplies an
                ``alt`` column (but no explicit ``effect_allele``) to be loaded by
                treating ALT as the effect allele. Defaults to False, which raises.

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
            prs_df = load_prs_from_dataframe(
                prs_definition, allow_alt_as_effect=allow_alt_as_effect
            )
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
            prs_df = load_prs_from_file(
                Path(prs_definition), allow_alt_as_effect=allow_alt_as_effect
            )

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
        _chroms, _pos = hoist_columns(prs_df, "chromosome", "position")
        for _c, _p in zip(_chroms, _pos):
            # Inline normalization preserved verbatim (upper + strip "chr" only,
            # intentionally NOT the full _normalize_chromosome): this only widens
            # the load filter, and changing it would change which variants load.
            _c = str(_c).upper()
            if _c.startswith("CHR"):
                _c = _c[3:]
            prs_chrpos.add(f"{_c}:{int(_p)}")
        all_needed_variants = set(prs_df["variant_id"]) | platform_variant_set | prs_chrpos

        # Backend selection (Phase 2 streaming seam). The dense in-RAM path below is
        # the untouched correctness oracle; the streaming path trains from banded
        # sufficient statistics without ever materializing the dosage matrix. "auto"
        # streams only when the estimated dense matrix is large, so test-sized inputs
        # stay on the oracle (golden gate exact).
        if self.backend != "dense":
            # "auto" must not regress formats the dense oracle supports: if the
            # streaming source cannot read this path (e.g. PLINK1 .bed), fall back
            # to dense. Explicit backend="streaming" surfaces the error instead.
            try:
                source = make_genotype_source(
                    reference_genotypes, variant_ids=all_needed_variants
                )
            except DataLoadError:
                if self.backend == "streaming":
                    raise
                source = None
            if source is not None and (
                self.backend == "streaming"
                or self._auto_should_stream(source, all_needed_variants)
            ):
                return self._fit_streaming(
                    source=source,
                    prs_df=prs_df,
                    platform_variant_set=platform_variant_set,
                    effective_prs_id=effective_prs_id,
                    effective_platform_name=effective_platform_name,
                    effective_genome_build=effective_genome_build,
                    model_name=model_name,
                    reference_panel_id=reference_panel_id,
                    training_ancestry=training_ancestry,
                )

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

        # Step 7: Build allele-aware reference indices.
        # reference_index maps chr:pos -> candidate reference rows (multi-allelic
        # records are split into one row per ALT), enabling exact chr:pos:ref:alt
        # matching and effect-allele-oriented dosage extraction.
        resolver = ReferenceAlleleResolver(genotype_data.variant_info)
        reference_index = resolver.locus_to_rows
        reference_contigs = {
            _normalize_chromosome(str(c))
            for c in genotype_data.variant_info["chromosome"].unique()
        }

        # Mapping from variant_id / chr:pos to a reference row index, used for
        # platform predictor lookups (first occurrence wins, for determinism).
        geno_var_to_idx: Dict[str, int] = {}
        _gv_ids, _gv_pos = hoist_columns(
            genotype_data.variant_info, "variant_id", "position"
        )
        _gv_chroms = normalize_chromosome_array(
            genotype_data.variant_info["chromosome"]
        ).tolist()
        for idx in range(len(_gv_ids)):
            # idx is positional (variant_info has a RangeIndex); setdefault keeps
            # first-occurrence-wins for determinism, exactly as the iterrows loop.
            geno_var_to_idx.setdefault(_gv_ids[idx], idx)
            geno_var_to_idx.setdefault(
                f"{_gv_chroms[idx]}:{int(_gv_pos[idx])}", idx
            )

        # Step 7b: Optional QC — exclude strand-ambiguous (palindromic) SNPs whose
        # reference MAF is too high for the strand to be resolved reliably.
        ambiguous_excluded_ids: Set[str] = set()
        if self.exclude_ambiguous:
            _vids, _chroms, _pos, _effs, _oths = hoist_columns(
                prs_df, "variant_id", "chromosome", "position",
                "effect_allele", "other_allele",
            )
            for i in range(len(prs_df)):
                effect = str(_effs[i]).upper()
                other = _oths[i]
                if pd.isna(other):
                    continue
                if not _is_ambiguous_snp(effect, str(other).upper()):
                    continue
                match = resolver.resolve(
                    _chroms[i], int(_pos[i]), effect, other,
                    genotype_data.dosage_matrix,
                )
                if match is None:
                    continue
                dosage = match[1]
                valid = ~np.isnan(dosage)
                if not np.any(valid):
                    continue
                af = float(np.mean(dosage[valid]) / 2.0)
                if min(af, 1.0 - af) > self.ambiguous_maf_threshold:
                    ambiguous_excluded_ids.add(_vids[i])
            if self.verbose >= 1 and ambiguous_excluded_ids:
                print(
                    f"Excluded {len(ambiguous_excluded_ids)} ambiguous SNPs "
                    f"(MAF > {self.ambiguous_maf_threshold})"
                )

        # Step 7c: Allele-aware observed inclusion. partition_variants matched the
        # platform by id/locus only; additionally require that an observed variant's
        # (effect, other) alleles are compatible with the reference at its locus. A
        # variant whose locus is in the reference but whose alleles match no
        # reference row (even on the complementary strand) is reclassified to
        # missing, so it is recovered by imputation where possible and otherwise
        # dropped-with-reason ("allele_mismatch") instead of being mis-scored as
        # observed. A variant whose locus is absent from the reference (e.g. a
        # non-autosomal platform SNP) is kept observed: it remains directly
        # scoreable from the user's genotype.
        observed_variant_ids = set(observed_variant_ids)
        missing_variant_ids = set(missing_variant_ids)
        observed_allele_reclassified: Set[str] = set()
        _obs_df = prs_df[prs_df["variant_id"].isin(observed_variant_ids)]
        _vids, _chroms, _pos, _effs, _oths = hoist_columns(
            _obs_df, "variant_id", "chromosome", "position",
            "effect_allele", "other_allele",
        )
        for i in range(len(_obs_df)):
            var_id = _vids[i]
            if var_id in ambiguous_excluded_ids:
                continue
            locus = (
                f"{_normalize_chromosome(str(_chroms[i]))}:"
                f"{int(_pos[i])}"
            )
            if locus not in reference_index:
                continue  # platform-measured but absent from reference: keep observed
            match = resolver.resolve(
                _chroms[i], int(_pos[i]),
                _effs[i], _oths[i],
                genotype_data.dosage_matrix,
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

        # Step 8: Build training matrices

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
        # dosages. Variants absent from the reference (or excluded by QC) are
        # recorded with a reason rather than silently dropped.
        missing_drop_reason: Dict[str, str] = {}
        missing_prs_rows: List[pd.Series] = []
        missing_columns: List[np.ndarray] = []
        for _, row in prs_df[prs_df["variant_id"].isin(missing_variant_ids)].iterrows():
            var_id = row["variant_id"]
            if var_id in ambiguous_excluded_ids:
                missing_drop_reason[var_id] = "ambiguous_excluded"
                continue
            match = resolver.resolve(
                row["chromosome"], int(row["position"]),
                row["effect_allele"], row.get("other_allele"),
                genotype_data.dosage_matrix,
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

        # Step 9: Hyperparameter tuning. "global" searches a stratified sample of
        # missing variants on their local windows (the same matrices training uses)
        # and applies one winning (l1_ratio, alpha) to every variant. "per_variant"
        # defers to the trainer, which tunes each variant on its own window.
        # "none" (and the no-data cases) use the configured l1_ratio/alpha.
        effective_l1_ratio = self.l1_ratio
        effective_alpha = self.alpha
        if self.tuning_scope == "global" and X.shape[1] > 0 and Z.shape[1] > 0:
            if self.verbose >= 1:
                print("Running global hyperparameter search...")
            try:
                grid_result = global_hyperparameter_search(
                    Z=Z,
                    X_missing=X,
                    missing_variant_info=missing_prs_df,
                    platform_variant_info=platform_variant_info,
                    window_size=self.window_size,
                    max_predictors=self.max_predictors,
                    max_tuning_variants=self.max_tuning_variants,
                    cv_folds=self.cv_folds,
                    random_state=self.random_state,
                )
                effective_l1_ratio = grid_result.best_l1_ratio
                effective_alpha = grid_result.best_alpha
                if self.verbose >= 1:
                    print(
                        f"Best hyperparameters: l1_ratio={effective_l1_ratio}, "
                        f"alpha={effective_alpha} "
                        f"(tuned on {grid_result.n_variants_sampled} variants)"
                    )
            except ValidationError:
                # Fall back to defaults if tuning fails
                if self.verbose >= 1:
                    print("Hyperparameter search failed, using defaults")

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
                tuning_scope=self.tuning_scope,
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
                # Build the placed-variant matrix with effect-oriented dosages.
                # Column order defines indexing for the observed/imputed components.
                trained_ids = set(training_result.models.keys())
                placed_columns: List[np.ndarray] = []
                placed_var_ids: List[str] = []
                placed_betas: List[float] = []
                _vids, _chroms, _pos, _effs, _oths, _betas = hoist_columns(
                    prs_df, "variant_id", "chromosome", "position",
                    "effect_allele", "other_allele", "beta",
                )
                for i in range(len(prs_df)):
                    var_id = _vids[i]
                    is_observed = (
                        var_id in observed_variant_ids
                        and var_id not in ambiguous_excluded_ids
                    )
                    if not (is_observed or var_id in trained_ids):
                        continue
                    match = resolver.resolve(
                        _chroms[i], int(_pos[i]),
                        _effs[i], _oths[i],
                        genotype_data.dosage_matrix,
                    )
                    if match is None:
                        continue
                    placed_columns.append(match[1])
                    placed_var_ids.append(var_id)
                    placed_betas.append(float(_betas[i]))

                if placed_columns:
                    # Per-column (per-variant) mean imputation of missing reference
                    # dosages: a NaN sample is filled with the column mean (≈ 2*AF, the
                    # population-expected dosage under HWE), NOT 0 (= homozygous
                    # non-effect), which would bias s_true and the observed part of
                    # s_cv toward zero. See evaluation.calibration.mean_impute_columns.
                    X_full = mean_impute_columns(
                        np.column_stack(placed_columns).astype(np.float32)
                    )
                    all_betas = np.array(placed_betas)
                    id_to_col = {vid: i for i, vid in enumerate(placed_var_ids)}

                    # Observed component: columns of X_full that are observed.
                    observed_indices = np.array(
                        [id_to_col[vid] for vid in placed_var_ids
                         if vid in observed_variant_ids],
                        dtype=int,
                    )
                    observed_betas = np.array(
                        [all_betas[i] for i in observed_indices]
                    )

                    # Imputed component: CV predictions paired with their betas
                    # (built in one pass so order stays aligned).
                    cv_preds_by_idx: Dict[int, np.ndarray] = {}
                    missing_betas = []
                    for var_id, cv_pred in training_result.cv_predictions.items():
                        if var_id in id_to_col:
                            cv_preds_by_idx[id_to_col[var_id]] = cv_pred
                            missing_betas.append(all_betas[id_to_col[var_id]])
                    missing_betas = np.array(missing_betas)

                    # Compute CV-predicted PRS
                    s_cv = compute_cv_predicted_prs(
                        X=X_full,
                        observed_variant_indices=observed_indices,
                        observed_betas=observed_betas,
                        cv_predictions=cv_preds_by_idx,
                        missing_betas=missing_betas,
                    )

                    # Compute true PRS from the same effect-oriented matrix
                    s_true = X_full @ all_betas

                    # Estimate calibration parameters
                    calibration_params = estimate_cv_calibration(s_cv, s_true)

                    # Inject the full-data (no-missingness) diagonal SE — the
                    # optimistic lower bound the empirical residual SD replaces
                    # (P4.1). Only imputed terms carry variance; exact observed
                    # terms contribute zero.
                    diag_var = 0.0
                    for var_id in training_result.cv_predictions:
                        model = training_result.models.get(var_id)
                        if model is not None and var_id in id_to_col:
                            beta = float(all_betas[id_to_col[var_id]])
                            diag_var += beta**2 * model.residual_variance
                    calibration_params = dataclasses.replace(
                        calibration_params,
                        diagonal_model_se_lower_bound=float(np.sqrt(diag_var)),
                    )

                    if self.verbose >= 2:
                        print(
                            f"Calibration: R²={calibration_params.calibration_r2:.3f}, "
                            f"scaling={calibration_params.scaling_factor:.3f}"
                        )

            except (ValueError, IndexError) as e:
                if self.verbose >= 1:
                    print(f"Warning: Could not compute calibration parameters: {e}")
                calibration_params = None

        # Step 12: Build observed VariantInfo objects (excluding QC-dropped SNPs),
        # each carrying an optional per-variant fallback imputation model (P1.8) so
        # an observed variant the user's upload cannot resolve/call is recovered
        # rather than silently dropped.
        #
        # Pass 1: collect the kept observed rows and, for each whose effect-allele
        # dosage is extractable from the reference, an effect-oriented target column
        # to train its fallback from (local-window platform predictors, the target
        # locus auto-excluded by filter_to_local_window).
        kept_observed_rows: List[pd.Series] = []
        fb_target_pos: List[int] = []  # indices into kept_observed_rows with a target
        fb_columns: List[np.ndarray] = []
        fallback_no_target_ids: Set[str] = set()
        for _, row in prs_df[prs_df["variant_id"].isin(observed_variant_ids)].iterrows():
            if row["variant_id"] in ambiguous_excluded_ids:
                continue
            pos_in_kept = len(kept_observed_rows)
            kept_observed_rows.append(row)
            if Z.shape[1] == 0:
                continue
            match = resolver.resolve(
                row["chromosome"], int(row["position"]),
                row["effect_allele"], row.get("other_allele"),
                genotype_data.dosage_matrix,
            )
            if match is None:
                # Platform-measured but no reference target (locus absent): still
                # scored directly when the upload resolves it, but there is no panel
                # to train a fallback from.
                fallback_no_target_ids.add(row["variant_id"])
                continue
            fb_target_pos.append(pos_in_kept)
            fb_columns.append(match[1])

        # Train the fallbacks with the same trainer/config as the imputed models.
        # The trainer keys results by variant_id, which is not unique at
        # duplicate-rsID multiallelic loci, so train against a unique synthetic key
        # and reset each model's identity to its real variant afterwards.
        fallback_by_pos: Dict[int, ImputedVariantModel] = {}
        if fb_columns:
            if self.verbose >= 1:
                print(
                    f"Training {len(fb_columns)} observed-variant fallback models..."
                )
            fb_prs_rows = []
            for k, pos_in_kept in enumerate(fb_target_pos):
                fb_row = kept_observed_rows[pos_in_kept].copy()
                fb_row["variant_id"] = str(k)
                fb_prs_rows.append(fb_row)
            fb_prs_df = pd.DataFrame(fb_prs_rows).reset_index(drop=True)
            X_observed = np.column_stack(fb_columns).astype(np.float32)
            fb_trainer = ImputationModelTrainer(
                window_size=self.window_size,
                l1_ratio=effective_l1_ratio,
                alpha=effective_alpha,
                cv_folds=self.cv_folds,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                max_predictors=self.max_predictors,
                tuning_scope=self.tuning_scope,
                verbose=0,
            )
            fb_result = fb_trainer.fit_all_variants(
                Z=Z,
                X=X_observed,
                prs_variants=fb_prs_df,
                platform_variant_info=platform_variant_info,
            )
            for k, pos_in_kept in enumerate(fb_target_pos):
                model = fb_result.models.get(str(k))
                if model is None:
                    continue
                model.variant_id = kept_observed_rows[pos_in_kept]["variant_id"]
                fallback_by_pos[pos_in_kept] = model

        # Pass 2: build the VariantInfos, attaching each fallback by position.
        observed_variants_list: List[VariantInfo] = []
        for pos_in_kept, row in enumerate(kept_observed_rows):
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
                    fallback=fallback_by_pos.get(pos_in_kept),
                )
            )

        # Step 12b: Record a disposition for every input PRS variant so coverage
        # is reported honestly (no silent loss).
        observed_kept_ids = {v.variant_id for v in observed_variants_list}
        observed_fallback_ids = {m.variant_id for m in fallback_by_pos.values()}
        intercept_only_ids = {
            vid for vid, m in training_result.models.items() if m.is_intercept_only
        }
        self._variant_dispositions = self._build_variant_dispositions(
            prs_df=prs_df,
            observed_kept_ids=observed_kept_ids,
            trained_models=training_result.models,
            intercept_only_ids=intercept_only_ids,
            ambiguous_excluded_ids=ambiguous_excluded_ids,
            missing_drop_reason=missing_drop_reason,
            observed_fallback_ids=observed_fallback_ids,
            fallback_no_target_ids=fallback_no_target_ids,
            training_failures=training_result.failures,
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
        self._reference_panel_id = reference_panel_id
        self._training_ancestry = training_ancestry

        if self.verbose >= 1:
            print(
                f"Model fitted: {len(self._observed_variants)} observed, "
                f"{len(self._imputed_models)} imputed variants"
            )

        # Step 14: Return self for method chaining
        return self

    # ------------------------------------------------------------------
    # Streaming backend (Phase 2): train from banded sufficient statistics
    # without ever materializing the reference dosage matrix.
    # ------------------------------------------------------------------
    def _auto_should_stream(self, source, all_needed_variants: Set[str]) -> bool:
        """backend='auto': stream when the estimated dense matrix is large.

        Estimated dense bytes = ``n_samples × |needed variants| × 4``. Test-sized
        inputs fall well below the threshold and stay on the dense oracle (keeping
        the golden gate exact); real reference panels select streaming.
        """
        try:
            n_samples = len(source.sample_ids)
        except Exception:  # noqa: BLE001 - can't size it → safe default (dense oracle)
            return False
        est_bytes = n_samples * max(len(all_needed_variants), 1) * 4
        stream = est_bytes > _AUTO_STREAMING_BYTES_THRESHOLD
        if stream and self.verbose >= 1:
            print(
                f"backend='auto': estimated dense matrix ~{est_bytes / 1e9:.1f} GB "
                f"(> {_AUTO_STREAMING_BYTES_THRESHOLD / 1e9:.0f} GB) → streaming."
            )
        return stream

    def _fit_streaming(
        self,
        source,
        prs_df: pd.DataFrame,
        platform_variant_set: Set[str],
        effective_prs_id: Optional[str],
        effective_platform_name: Optional[str],
        effective_genome_build: Optional[str],
        model_name: Optional[str],
        reference_panel_id: Optional[str],
        training_ancestry: Optional[str],
    ) -> "LinearImputationPRS":
        """Train via a single streaming pass over the panel (Phase 2 backend).

        Produces the same fitted state the dense tail (Steps 11–13) would — imputed
        models, calibration params, observed ``VariantInfo`` list (with per-variant
        fallbacks), dispositions, platform index — but from banded sufficient
        statistics, never materializing the dosage matrix and never building the
        per-variant ``cv_predictions`` dict (``s_true``/``s_cv`` are reduced in-stream).

        Sanctioned deviations from the dense oracle (all documented; parity is exact
        on a dense, no-missing panel like 1000G):

        - Mean-imputation, not listwise deletion: NaN dosages are mean-imputed at
          accumulation (a shared Gram cannot drop per-variant-varying rows), so under
          panel missingness ``n_valid`` and the intercept-only triggers can differ.
        - Calibration via two O(n) accumulators: ``s_true``/``s_cv`` are reduced
          in-stream, so ``TrainingResult.cv_predictions`` is ``None`` (no per-variant
          dict — the calibration blocker never materializes).
        - float64 accumulation matmuls (more accurate than a float32 path; the
          float32/GPU tradeoff is deferred to Phase 3).
        - Genome build is not auto-detected from the panel (the streaming source
          carries none — the PRS/platform build is used).
        - Not yet supported on streaming: ``exclude_ambiguous`` (raises
          ``NotImplementedError``) and hyperparameter tuning (``tuning_scope != "none"``
          warns and uses the configured ``l1_ratio``/``alpha``). Use ``backend="dense"``.
        """
        from imputed_prs.compute.sufficient_stats import (
            StreamingImputationFitter,
            _chrom_sort_key,
            build_stream_plan,
            collect_reference_variant_info,
        )
        from imputed_prs.evaluation.streaming_calibration import (
            finalize_imputation_calibration,
        )
        from imputed_prs.models.trainer import _compute_training_summary

        if self.exclude_ambiguous:
            raise NotImplementedError(
                "backend='streaming' does not yet support exclude_ambiguous=True "
                "(AF-based streaming QC is a follow-up). Use backend='dense'."
            )

        effective_l1_ratio = self.l1_ratio
        effective_alpha = self.alpha
        if self.tuning_scope != "none":
            # Bounded streaming hyperparameter tuning (windowed dense pre-pass) is a
            # follow-up. Warn unconditionally so a default streaming fit never
            # *silently* drops tuning, then use the configured (l1_ratio, alpha).
            # tuning_scope='none' silences this; backend='dense' tunes.
            warnings.warn(
                f"backend='streaming': tuning_scope={self.tuning_scope!r} is not yet "
                f"supported on the streaming path; using the configured "
                f"l1_ratio={effective_l1_ratio}, alpha={effective_alpha} "
                f"(no hyperparameter tuning performed).",
                UserWarning,
                stacklevel=2,
            )

        if self.verbose >= 1:
            print(f"Streaming backend: {len(source.sample_ids)} reference samples")

        # Metadata scan → harmonized stream plan (targets, observed, chip, fallbacks).
        chroms = sorted(
            {_normalize_chromosome(str(c)) for c in prs_df["chromosome"].unique()},
            key=_chrom_sort_key,
        )
        ref_info = collect_reference_variant_info(source, chroms)
        plan, missing_drop_reason = build_stream_plan(
            ref_info,
            prs_df,
            platform_variant_set,
            sample_ids=source.sample_ids,
            window_size=self.window_size,
            max_predictors=self.max_predictors,
            alpha=effective_alpha,
            l1_ratio=effective_l1_ratio,
            cv_folds=self.cv_folds,
            random_state=self.random_state,
        )

        if self.verbose >= 1:
            print(
                f"Stream plan: {len(plan.targets)} missing targets, "
                f"{len(plan.observed)} observed calibration terms, "
                f"{len(plan.chip_ids)} chip predictors"
            )

        # Single streaming pass: imputed models + observed fallbacks + calibration.
        fitter = StreamingImputationFitter(plan)
        result = fitter.run(source)

        calibration_params = finalize_imputation_calibration(
            result.s_true, result.s_cv, result.models
        )

        # TrainingResult with cv_predictions=None — the calibration blocker never
        # materializes; s_true/s_cv were accumulated during the streaming pass.
        training_failures: Dict[str, TrainingFailure] = {}
        for vid, msg in result.failures.items():
            etype, _, emsg = msg.partition(": ")
            training_failures[vid] = TrainingFailure(
                unit_id=vid, error_type=etype or "Error", error_message=emsg or msg
            )
        training_result = TrainingResult(
            models=result.models,
            cv_predictions=None,
            n_variants_trained=result.n_trained,
            n_variants_failed=result.n_failed,
            n_intercept_only=result.n_intercept_only,
            training_summary=_compute_training_summary(result.models),
            failures=training_failures,
        )

        # Observed VariantInfo list, each carrying its per-variant fallback (P1.8),
        # in PRS order (mirrors dense Step 12).
        observed_variants_list: List[VariantInfo] = []
        for _, row in prs_df[
            prs_df["variant_id"].isin(plan.observed_prs_ids)
        ].iterrows():
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
                    fallback=result.fallback_models.get(row["variant_id"]),
                )
            )

        # Per-variant dispositions (one record per input PRS variant; no silent loss).
        intercept_only_ids = {
            vid for vid, m in result.models.items() if m.is_intercept_only
        }
        self._variant_dispositions = self._build_variant_dispositions(
            prs_df=prs_df,
            observed_kept_ids=set(plan.observed_prs_ids),
            trained_models=result.models,
            intercept_only_ids=intercept_only_ids,
            ambiguous_excluded_ids=set(),
            missing_drop_reason=missing_drop_reason,
            observed_fallback_ids=set(result.fallback_models.keys()),
            fallback_no_target_ids=set(plan.fallback_no_target_ids),
            training_failures=training_failures,
        )

        # Populate instance state (mirrors dense Step 13).
        self._is_fitted = True
        self._observed_variants = observed_variants_list
        self._imputed_models = list(result.models.values())
        self._calibration_params = calibration_params
        self._training_result = training_result
        self._platform_variant_index = {
            vid: i
            for i, vid in enumerate(plan.platform_variant_info["variant_id"].tolist())
        }
        self._prs_id = effective_prs_id
        self._platform_name = effective_platform_name
        self._genome_build = effective_genome_build
        self._model_name = model_name
        self._reference_panel_id = reference_panel_id
        self._training_ancestry = training_ancestry

        if self.verbose >= 1:
            print(
                f"Model fitted (streaming): {len(self._observed_variants)} observed, "
                f"{len(self._imputed_models)} imputed variants"
            )
        return self

    def _build_variant_dispositions(
        self,
        prs_df: pd.DataFrame,
        observed_kept_ids: Set[str],
        trained_models: Dict[str, ImputedVariantModel],
        intercept_only_ids: Set[str],
        ambiguous_excluded_ids: Set[str],
        missing_drop_reason: Dict[str, str],
        observed_fallback_ids: Optional[Set[str]] = None,
        fallback_no_target_ids: Optional[Set[str]] = None,
        training_failures: Optional[Dict[str, TrainingFailure]] = None,
    ) -> List[Dict[str, Any]]:
        """Build one disposition record per input PRS variant.

        Every row of ``prs_df`` yields exactly one record so no variant is
        silently lost. ``status`` is one of {observed, imputed, intercept_only,
        dropped}; ``reason`` is None for placed variants, or one of
        {ambiguous_excluded, reference_contig_missing, allele_mismatch,
        not_in_reference, training_failed} for dropped ones.

        For observed variants, ``has_fallback`` records whether a per-variant
        fallback model was trained (P1.8) and ``fallback_reason`` explains its
        absence ({no_reference_target, no_fallback_model}); both are False/None
        for non-observed rows.

        ``training_failures`` (variant_id -> TrainingFailure) adds the
        ``failure_error_type``/``failure_error_message``/``failure_n_valid_samples``/
        ``failure_target_variance`` columns explaining *why* a training fit raised
        (P5.1); all None for variants that did not fail training. ``reason`` is
        left unchanged ("training_failed") so existing aggregation is stable.
        """
        observed_fallback_ids = observed_fallback_ids or set()
        fallback_no_target_ids = fallback_no_target_ids or set()
        training_failures = training_failures or {}
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
            elif var_id in intercept_only_ids:
                status = "intercept_only"
            elif var_id in trained_models:
                status = "imputed"
            else:
                status = "dropped"
                reason = missing_drop_reason.get(var_id, "training_failed")

            has_fallback = False
            fallback_reason: Optional[str] = None
            if status == "observed":
                has_fallback = var_id in observed_fallback_ids
                if not has_fallback:
                    fallback_reason = (
                        "no_reference_target"
                        if var_id in fallback_no_target_ids
                        else "no_fallback_model"
                    )

            failure = training_failures.get(var_id)
            dispositions.append({
                "variant_id": var_id,
                "chromosome": str(row["chromosome"]),
                "position": int(row["position"]),
                "effect_allele": row["effect_allele"],
                "other_allele": other_allele,
                "beta": float(row["beta"]),
                "status": status,
                "reason": reason,
                "has_fallback": has_fallback,
                "fallback_reason": fallback_reason,
                "failure_error_type": failure.error_type if failure is not None else None,
                "failure_error_message": (
                    failure.error_message if failure is not None else None
                ),
                "failure_n_valid_samples": (
                    failure.n_valid_samples if failure is not None else None
                ),
                "failure_target_variance": (
                    failure.target_variance if failure is not None else None
                ),
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
        """Compute PRS for user genotypes.

        Args:
            user_genotypes: User genotype data as:
                - File path (DTC format auto-detected) — scored allele-aware
                  (genotypes are oriented against each variant's effect/other
                  alleles). Recommended for correct scoring.
                - DataFrame with variant_id and genotype columns — also scored
                  allele-aware. Recommended for correct scoring.
                - Dict mapping variant_id to numeric dosage — a legacy,
                  allele-blind fallback that bypasses allele orientation (it
                  trusts the dosages as-is). Prefer a file/DataFrame input.
            apply_calibration: Whether to apply calibration scaling.
                Default: True.
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
            PredictionResult with PRS value, uncertainty estimates, and diagnostics.

        Raises:
            ModelNotFittedError: If fit() has not been called.
            DataLoadError: If user genotype file cannot be loaded.
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

        # Get expected variants for filtering
        expected_variants = self._get_expected_variants()

        # Handle dict input directly, otherwise use the loaders. For real uploads
        # (file / DataFrame) we load a multi-key raw collection so that BOTH the
        # observed and imputed components are scored allele-aware (the predictor
        # passes raw_genotypes to its oriented scorers). The legacy dosage dict
        # (user_dosages) is still loaded alongside it to drive the allele-blind
        # back-compat path and the missing-variant accounting; that is why the
        # input is parsed twice here. A numeric dict input takes only the legacy
        # allele-blind path (raw_genotypes stays None).
        raw_genotypes = None
        if isinstance(user_genotypes, dict):
            user_dosages = user_genotypes
        else:
            user_dosages = load_user_genotypes(user_genotypes, expected_variants)
            raw_genotypes = load_raw_user_genotypes(user_genotypes)

        # Create PRSPredictor and compute prediction
        predictor = PRSPredictor(
            observed_variants=self._observed_variants,
            imputed_models=self._imputed_models,
            calibration_params=self._calibration_params,
        )
        return predictor.predict(
            user_dosages,
            apply_calibration=apply_calibration,
            raw_genotypes=raw_genotypes,
        )

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
            # Provenance is consumed by the JSON/HDF5/Arrow/Parquet exporters (the
            # flat CSV per-variant table is written separately and carries none).
            "reference_panel_id": self._reference_panel_id,
            "training_ancestry": self._training_ancestry,
            "ambiguous_policy": self._ambiguous_policy,
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
            path: Path to a saved model — an ``.h5``/``.hdf5``, ``.json``, ``.arrow``
                or ``*_variants.csv`` file, or a Parquet export directory.

        Returns:
            Loaded LinearImputationPRS instance ready for prediction.

        Raises:
            DataLoadError: If file cannot be loaded or format is unsupported.
        """
        path = Path(path)

        if not path.exists():
            raise DataLoadError(f"Model file not found: {path}")

        # A Parquet export is a directory of per-table .parquet files.
        if path.is_dir():
            return cls._load_from_parquet(path)

        # Detect format by extension
        suffix = path.suffix.lower()

        if suffix in (".h5", ".hdf5"):
            return cls._load_from_hdf5(path)
        elif suffix == ".json":
            return cls._load_from_json(path)
        elif suffix == ".arrow":
            return cls._load_from_arrow(path)
        elif suffix == ".csv":
            return cls._load_from_csv(path)
        else:
            raise DataLoadError(
                f"Unsupported model file format: {suffix}. Supported formats: "
                ".h5, .hdf5, .json, .arrow, .csv, or a Parquet export directory."
            )

    @classmethod
    def _from_components(
        cls,
        observed: List[VariantInfo],
        imputed: List[ImputedVariantModel],
        calib: Optional[CalibrationParams],
        metrics: Optional[EvaluationMetrics],
        metadata: Dict[str, Any],
    ) -> "LinearImputationPRS":
        """Build a fitted instance from loaded components + identity/provenance.

        Shared by the HDF5/Arrow/Parquet/CSV load paths. Normalizes empty-string
        metadata (HDF5 stores ``""`` for absent optional strings) to ``None`` and
        restores the provenance fields when present.
        """

        def _clean(value: Any) -> Optional[str]:
            if value is None or value == "":
                return None
            return value

        instance = cls()
        instance._is_fitted = True
        instance._observed_variants = observed
        instance._imputed_models = imputed
        instance._calibration_params = calib
        instance._evaluation_metrics = metrics

        instance._prs_id = _clean(metadata.get("prs_id"))
        instance._platform_name = _clean(metadata.get("platform_name"))
        instance._genome_build = _clean(metadata.get("genome_build"))
        instance._model_name = _clean(metadata.get("model_name"))
        instance._reference_panel_id = _clean(metadata.get("reference_panel_id"))
        instance._training_ancestry = _clean(metadata.get("training_ancestry"))
        ambiguous_policy = _clean(metadata.get("ambiguous_policy"))
        if ambiguous_policy:
            instance._ambiguous_policy = ambiguous_policy

        return instance

    @classmethod
    def _load_from_hdf5(cls, path: Path) -> "LinearImputationPRS":
        """Load model from HDF5 file."""
        try:
            components = load_model_hdf5(path)
        except Exception as e:
            raise DataLoadError(f"Failed to load HDF5 model: {e}") from e
        return cls._from_components(*components)

    @classmethod
    def _load_from_arrow(cls, path: Path) -> "LinearImputationPRS":
        """Load model from Arrow IPC file."""
        try:
            components = load_model_arrow(path)
        except Exception as e:
            raise DataLoadError(f"Failed to load Arrow model: {e}") from e
        return cls._from_components(*components)

    @classmethod
    def _load_from_parquet(cls, path: Path) -> "LinearImputationPRS":
        """Load model from a Parquet export directory."""
        try:
            components = load_model_parquet(path)
        except Exception as e:
            raise DataLoadError(f"Failed to load Parquet model: {e}") from e
        return cls._from_components(*components)

    @classmethod
    def _load_from_csv(cls, path: Path) -> "LinearImputationPRS":
        """Load model from the CSV per-variant table + companion coefficients."""
        try:
            components = load_model_csv(path)
        except Exception as e:
            raise DataLoadError(f"Failed to load CSV model: {e}") from e
        return cls._from_components(*components)

    @classmethod
    def _load_from_json(cls, path: Path) -> "LinearImputationPRS":
        """Load model from JSON file."""
        from imputed_prs.io.loaders import load_model_json, parse_imputed_model_json

        try:
            data = load_model_json(path)
        except Exception as e:
            raise DataLoadError(f"Failed to load JSON model: {e}") from e

        # Create instance with default parameters
        instance = cls()

        # `parse_imputed_model_json` restores the self-describing v2 `predictors`
        # list (or the v1.0 parallel arrays), including the P1.3 predictor allele
        # metadata the oriented scorer indexes. Shared with the projection loader
        # (P2.2) so the two products reconstruct identically.

        # Parse observed variants, each with an optional fallback model (P1.8).
        observed_variants = []
        for v in data.get("observed_variants", []):
            fb = v.get("fallback")
            observed_variants.append(
                VariantInfo(
                    variant_id=v["variant_id"],
                    chromosome=v["chromosome"],
                    position=v["position"],
                    effect_allele=v["effect_allele"],
                    other_allele=v.get("other_allele"),
                    beta=v["beta"],
                    fallback=parse_imputed_model_json(fb) if fb else None,
                )
            )

        # Parse imputed models.
        imputed_models = [
            parse_imputed_model_json(m) for m in data.get("imputed_variants", [])
        ]

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

        # Populate metadata + provenance (v2). Fall back to `metadata` for the
        # identity fields that predate the provenance block so v1.0 files still
        # populate genome build / platform.
        metadata = data.get("metadata", {})
        provenance = data.get("provenance", {})
        instance._prs_id = metadata.get("prs_id")
        instance._platform_name = (
            metadata.get("platform_name") or provenance.get("platform_id")
        )
        instance._genome_build = (
            metadata.get("genome_build") or provenance.get("genome_build")
        )
        instance._model_name = metadata.get("model_name")
        instance._reference_panel_id = provenance.get("reference_panel_id")
        instance._training_ancestry = provenance.get("training_ancestry")
        if provenance.get("ambiguous_policy"):
            instance._ambiguous_policy = provenance["ambiguous_policy"]

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

        # Loaded models carry no disposition record; fall back to the
        # observed + imputed view (dropped variants are not recoverable).
        if self._variant_dispositions is None:
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
                    "reason": None,
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
                    "reason": None,
                    "imputation_r2": model.imputation_r2,
                    "allele_frequency": model.allele_frequency,
                    "n_predictors": len(model.predictor_variant_ids),
                })

            return pd.DataFrame(rows)

        # Disposition-driven table: every input PRS variant appears exactly once,
        # including dropped variants (annotated with a reason).
        model_by_id = {m.variant_id: m for m in (self._imputed_models or [])}
        rows = []
        for d in self._variant_dispositions:
            model = model_by_id.get(d["variant_id"])
            rows.append({
                "variant_id": d["variant_id"],
                "chromosome": d["chromosome"],
                "position": d["position"],
                "effect_allele": d["effect_allele"],
                "other_allele": d["other_allele"],
                "beta": d["beta"],
                "status": d["status"],
                "reason": d["reason"],
                "imputation_r2": model.imputation_r2 if model is not None else None,
                "allele_frequency": model.allele_frequency if model is not None else None,
                "n_predictors": len(model.predictor_variant_ids) if model is not None else 0,
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
        # Observed variants recoverable via a per-variant fallback model (P1.8).
        # Computed from state so it is correct for both fitted and loaded models.
        n_observed_with_fallback = sum(
            1 for v in (self._observed_variants or []) if v.fallback is not None
        )

        # Compute mean R² for imputed variants
        r2_values = [m.imputation_r2 for m in (self._imputed_models or [])
                     if not m.is_intercept_only]
        mean_r2 = float(np.mean(r2_values)) if r2_values else None

        # Honest coverage from the per-variant disposition record. Loaded models
        # have no disposition record -> fall back to the placed count.
        if self._variant_dispositions is not None:
            n_definition = len(self._variant_dispositions)
            dropped_by_reason: Dict[str, int] = {}
            for d in self._variant_dispositions:
                if d["status"] == "dropped":
                    key = d["reason"] or "unknown"
                    dropped_by_reason[key] = dropped_by_reason.get(key, 0) + 1
            n_dropped = sum(dropped_by_reason.values())
        else:
            n_definition = n_observed + n_imputed
            dropped_by_reason = {}
            n_dropped = 0

        coverage = (n_observed + n_imputed) / n_definition if n_definition else 0.0

        # Training-failure breakdown by exception class (P5.1). Loaded models have
        # no training result -> empty/zero.
        if self._training_result is not None:
            training_failures_by_type: Dict[str, int] = {}
            for f in self._training_result.failures.values():
                training_failures_by_type[f.error_type] = (
                    training_failures_by_type.get(f.error_type, 0) + 1
                )
            n_training_failed = len(self._training_result.failures)
        else:
            training_failures_by_type = {}
            n_training_failed = 0

        return {
            "n_total_variants": n_definition,
            "n_definition_variants": n_definition,
            "n_observed": n_observed,
            "n_observed_with_fallback": n_observed_with_fallback,
            "n_imputed": n_imputed,
            "n_intercept_only": n_intercept_only,
            "n_dropped": n_dropped,
            "dropped_by_reason": dropped_by_reason,
            "n_training_failed": n_training_failed,
            "training_failures_by_type": training_failures_by_type,
            "coverage": coverage,
            "mean_imputation_r2": mean_r2,
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

        Every input PRS variant appears exactly once. Empty for models loaded
        from disk (dispositions are not serialized).

        Raises:
            ModelNotFittedError: If fit() has not been called.
        """
        if not self._is_fitted:
            raise ModelNotFittedError(
                "Model has not been fitted. Call fit() first."
            )
        return pd.DataFrame(self._variant_dispositions or [])

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
