"""Main LinearProjectionPRS class for training and prediction."""

import dataclasses
import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Union

import numpy as np
import pandas as pd

from imputed_prs.core.exceptions import (
    DataLoadError,
    ModelNotFittedError,
    ValidationError,
)
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
    GenotypeData,
    ImputedVariantModel,
    PredictionResult,
    ProjectionRegionModel,
    ProjectionTrainingResult,
    TrainingFailure,
    VariantInfo,
)
from imputed_prs.evaluation.calibration import (
    estimate_cv_calibration,
    mean_impute_columns,
)
from imputed_prs.io.genotype_loader import load_genotypes
from imputed_prs.io.genotype_source import (
    GenotypeSource,
    InMemoryGenotypeSource,
    PgenGenotypeSource,
    is_pgen_path,
    make_genotype_source,
)
from imputed_prs.io.pgs_catalog import download_pgs_catalog_score
from imputed_prs.io.platform_loader import resolve_platform_variant_set
from imputed_prs.io.prs_loader import load_prs_from_dataframe, load_prs_from_file
from imputed_prs.io.user_genotypes import (
    load_raw_user_genotypes,
    load_user_genotypes,
)
from imputed_prs.io.exporters.projection_json_export import export_projection_to_json
from imputed_prs.models.projection_predictor import ProjectionPredictor
from imputed_prs.models.projection_trainer import ProjectionRegionTrainer
from imputed_prs.models.trainer import ImputationModelTrainer
from imputed_prs.models.tuning import projection_hyperparameter_search

# backend="auto" streams when the estimated dense dosage matrix (n_samples ×
# |needed variants| × 4 bytes) exceeds this; test-sized inputs stay on the dense
# oracle so the golden gate is exact. Mirrors the imputation orchestrator.
_AUTO_STREAMING_BYTES_THRESHOLD = 8 * 1024**3  # 8 GiB


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
        tuning_scope: Literal["global", "none"] = "global",
        l1_ratio: float = 0.5,
        alpha: float = 0.01,
        cv_folds: int = 5,
        n_jobs: int = 1,
        n_workers: int = 1,
        random_state: Optional[int] = None,
        max_predictors: Optional[int] = None,
        max_tuning_regions: Optional[int] = 50,
        exclude_ambiguous: bool = False,
        ambiguous_maf_threshold: float = 0.4,
        backend: Literal["auto", "dense", "streaming"] = "auto",
        device: Literal["auto", "cpu", "mps", "cuda"] = "auto",
        verbose: int = 1,
    ):
        """Initialize the projection PRS model.

        Args:
            window_size: Size of genomic window (bp) for defining regions
                and selecting predictor variants. Default: 1,000,000 (1 Mb).
            tuning_scope: Hyperparameter tuning strategy:
                - "global": Tune once on a bounded, stratified sample of regions
                  using the same region matrices as training (recommended).
                - "none": Use the provided l1_ratio/alpha directly.
                Default: "global".
            l1_ratio: ElasticNet L1/L2 mixing parameter (0=Ridge, 1=Lasso).
                Only used when tuning_scope="none". Default: 0.5.
            alpha: ElasticNet regularization strength. Only used when
                tuning_scope="none". Default: 0.01.
            cv_folds: Number of cross-validation folds. Default: 5.
            n_jobs: Number of parallel jobs for training. Default: 1.
            n_workers: Process-level fan-out for the streaming backend — shards the
                per-chromosome accumulation + local solves across a process pool
                (Phase 7). ``-1`` = performance cores; ``1`` (default) = off. CPU-only
                (GPU stays single-process) and reproducible. Orthogonal to ``n_jobs``.
            random_state: Random seed for reproducibility. Default: None.
            max_predictors: Maximum predictor variants per region.
                Default: None (no limit).
            max_tuning_regions: Cap on the number of regions sampled for
                tuning_scope="global". None tunes on all regions. Must be positive
                when set. Default: 50.
            device: Compute device for the streaming backend's Gram kernels
                ("auto"/"cpu"/"mps"/"cuda"). "auto" uses the GPU via ``torch`` (the
                ``gpu`` extra) when available, else CPU; with no ``torch`` it resolves
                to "cpu". Only the streaming path is device-aware. Default: "auto".
            verbose: Verbosity level (0=silent, 1=progress, 2=debug).
                Default: 1.
        """
        if tuning_scope not in ("global", "none"):
            raise ValidationError(
                f"tuning_scope must be 'global' or 'none', got {tuning_scope!r}"
            )
        if max_tuning_regions is not None and max_tuning_regions <= 0:
            raise ValidationError(
                f"max_tuning_regions must be positive or None, "
                f"got {max_tuning_regions}"
            )
        if backend not in ("auto", "dense", "streaming"):
            raise ValidationError(
                f"backend must be 'auto', 'dense', or 'streaming', got {backend!r}"
            )
        if device not in ("auto", "cpu", "mps", "cuda"):
            raise ValidationError(
                f"device must be 'auto', 'cpu', 'mps', or 'cuda', got {device!r}"
            )
        self.backend = backend
        self.device = device
        self.window_size = window_size
        self.tuning_scope = tuning_scope
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.cv_folds = cv_folds
        self.n_jobs = n_jobs
        self.n_workers = n_workers
        self.random_state = random_state
        self.max_predictors = max_predictors
        self.max_tuning_regions = max_tuning_regions
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

        # Provenance for the deployable v2 export (consumed by the build/platform
        # compatibility check). Reference panel / ancestry are set at fit(); the
        # ambiguity policy default follows the deploy decision (training-side
        # exclude_ambiguous stays False, so the model still carries palindromes).
        self._reference_panel_id: Optional[str] = None
        self._training_ancestry: Optional[str] = None
        self._ambiguous_policy: str = "exclude_unless_platform_strand_known"

    def fit(
        self,
        reference_genotypes: Union[str, Path, GenotypeData, GenotypeSource],
        prs_definition: Union[str, Path, pd.DataFrame],
        platform_name: Optional[str] = None,
        platform_manifest: Optional[Union[str, Path]] = None,
        platform_variants: Optional[List[str]] = None,
        genome_build: Optional[str] = None,
        prs_id: Optional[str] = None,
        model_name: Optional[str] = None,
        reference_panel_id: Optional[str] = None,
        training_ancestry: Optional[str] = None,
        allow_alt_as_effect: bool = False,
        _platform_variant_set: Optional[Set[str]] = None,
        _platform_info: Optional[Any] = None,
    ) -> "LinearProjectionPRS":
        """Train projection models on reference genotype data.

        Args:
            reference_genotypes: Path to a reference genotype file, a ``GenotypeData``,
                or a streaming ``GenotypeSource``. Paths may be VCF
                (``.vcf/.vcf.gz/.bcf``), PLINK1 (``.bed``, dense backend only), or
                PLINK2 **PGEN** (``.pgen``). PGEN is the preferred production reader
                (bgzipped-VCF reads are far slower) and always streams; convert a VCF
                reference once with ``plink2 --vcf ref.vcf.gz --make-pgen --out ref``.
            prs_definition: PRS definition as DataFrame, file path, or PGS
                Catalog ID (e.g., "PGS000004").
            platform_name: Pre-built platform name (e.g., "23andme_v5").
            platform_manifest: Path to platform manifest file.
            platform_variants: List of platform variant IDs.
            genome_build: Reference genome build (e.g., "GRCh37").
            prs_id: PRS identifier for metadata.
            model_name: Model name for metadata.
            reference_panel_id: Provenance — reference panel used for training
                (e.g., "1000G_phase3_EUR"). Recorded in the deployable export.
            training_ancestry: Provenance — ancestry of the training cohort
                (e.g., "EUR"). Recorded in the deployable export.
            allow_alt_as_effect: If True, permit a PRS definition that supplies an
                ``alt`` column (but no explicit ``effect_allele``) to be loaded by
                treating ALT as the effect allele. Defaults to False, which raises.

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
            prs_df = load_prs_from_dataframe(
                prs_definition, allow_alt_as_effect=allow_alt_as_effect
            )
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
            prs_df = load_prs_from_file(
                Path(prs_definition), allow_alt_as_effect=allow_alt_as_effect
            )

        if self.verbose >= 2:
            print(f"Loaded PRS definition with {len(prs_df)} variants")

        # Step 3: Resolve platform variants. A caller (cross_validate / a fold fit)
        # may pass a pre-resolved ``_platform_variant_set`` so the reference is read
        # once and threaded through — skip the load, but still derive the metadata
        # label/build from the original source args.
        if platform_name is not None:
            effective_platform_name = platform_name
        elif platform_manifest is not None:
            effective_platform_name = Path(platform_manifest).stem
        else:
            effective_platform_name = "custom"

        if _platform_variant_set is not None:
            platform_variant_set = _platform_variant_set
            platform_info = _platform_info
        else:
            platform_variant_set, platform_info, _ = resolve_platform_variant_set(
                platform_name, platform_manifest, platform_variants
            )
        if effective_genome_build is None and platform_info:
            effective_genome_build = platform_info.genome_build

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

        # Phase 4: accept in-memory genotypes so callers (cross-validation,
        # sensitivity analysis) need not serialize a fold to a temp VCF. A bare
        # GenotypeSource is streaming-only; a GenotypeData feeds either backend.
        if isinstance(reference_genotypes, GenotypeSource):
            if self.backend == "dense":
                raise ValidationError(
                    "backend='dense' cannot ingest a GenotypeSource; pass a path "
                    "or GenotypeData."
                )
            return self._fit_streaming(
                source=reference_genotypes,
                prs_df=prs_df,
                platform_variant_set=platform_variant_set,
                effective_prs_id=effective_prs_id,
                effective_platform_name=effective_platform_name,
                effective_genome_build=effective_genome_build,
                model_name=model_name,
                reference_panel_id=reference_panel_id,
                training_ancestry=training_ancestry,
            )
        in_memory_gd = (
            reference_genotypes
            if isinstance(reference_genotypes, GenotypeData)
            else None
        )

        # Backend selection (Phase 2 streaming seam, M3). The dense in-RAM path below
        # is the untouched correctness oracle; the streaming path trains region models
        # from banded sufficient statistics without materializing the dosage matrix.
        # "auto" streams only for large panels, so test-sized inputs stay on the oracle.
        if self.backend != "dense":
            if in_memory_gd is not None:
                source = InMemoryGenotypeSource(
                    in_memory_gd, variant_ids=all_needed_variants
                )
            else:
                # "auto" must not regress formats the dense oracle supports: if the
                # streaming source cannot read this path (e.g. PLINK1 .bed), fall
                # back to dense. Explicit backend="streaming" surfaces the error.
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
                # PGEN is streaming-native (no dense reader): always stream it, even
                # for a small panel the size gate would otherwise route to dense.
                or isinstance(source, PgenGenotypeSource)
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

        if in_memory_gd is not None:
            genotype_data = in_memory_gd
        else:
            if is_pgen_path(reference_genotypes):
                raise ValidationError(
                    "PGEN input requires the streaming backend (there is no dense "
                    "PGEN reader). Use backend='auto' (the default) or "
                    "backend='streaming', or convert the reference to VCF."
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

        # Step 7: Build allele-aware reference indices (see LinearImputationPRS).
        resolver = ReferenceAlleleResolver(genotype_data.variant_info)
        reference_index = resolver.locus_to_rows
        reference_contigs = {
            _normalize_chromosome(str(c))
            for c in genotype_data.variant_info["chromosome"].unique()
        }

        # Step 8: Build training matrices.
        # Mapping for platform predictor lookups (first occurrence wins).
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

        # Optional QC — exclude strand-ambiguous SNPs with high reference MAF.
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

        # Allele-aware observed inclusion (see LinearImputationPRS): require an
        # observed variant's (effect, other) alleles to be compatible with the
        # reference at its locus. Locus-in-reference but allele-incompatible ->
        # reclassify to missing (recovered by projection where possible, else
        # dropped-with-reason "allele_mismatch"). Locus absent from reference ->
        # keep observed (still scoreable directly from the user's genotype).
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

        # Step 8.5: Hyperparameter tuning. "global" searches a stratified sample of
        # regions on the same region matrices training uses (predictors selected by
        # the identical region/window logic; target = sum of PRS contributions) and
        # applies one winning (l1_ratio, alpha) to every region. "none" (and the
        # no-data cases) use the configured l1_ratio/alpha.
        effective_l1_ratio = self.l1_ratio
        effective_alpha = self.alpha
        if self.tuning_scope == "global" and X.shape[1] > 0 and Z.shape[1] > 0:
            if self.verbose >= 1:
                print("Running projection hyperparameter search...")
            try:
                grid_result = projection_hyperparameter_search(
                    Z=Z,
                    X=X,
                    prs_variants=missing_prs_df,
                    platform_variant_info=platform_variant_info,
                    window_size=self.window_size,
                    max_predictors=self.max_predictors,
                    max_tuning_regions=self.max_tuning_regions,
                    cv_folds=self.cv_folds,
                    random_state=self.random_state,
                )
                effective_l1_ratio = grid_result.best_l1_ratio
                effective_alpha = grid_result.best_alpha
                if self.verbose >= 1:
                    print(
                        f"Best hyperparameters: l1_ratio={effective_l1_ratio}, "
                        f"alpha={effective_alpha} "
                        f"(tuned on {grid_result.n_variants_sampled} regions)"
                    )
            except ValidationError:
                # Fall back to defaults if tuning fails
                if self.verbose >= 1:
                    print("Hyperparameter search failed, using defaults")

        # Step 9: Train projection models
        if X.shape[1] > 0:
            if self.verbose >= 1:
                print(f"Training projection models for {len(missing_prs_df)} missing variants...")

            trainer = ProjectionRegionTrainer(
                window_size=self.window_size,
                l1_ratio=effective_l1_ratio,
                alpha=effective_alpha,
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
                    if not (is_observed or var_id in covered_ids):
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

                    # Inject the full-data (no-missingness) diagonal SE — the
                    # optimistic lower bound the empirical residual SD replaces
                    # (P4.1). Projection sums each region's out-of-fold CV MSE.
                    diag_var = 0.0
                    for region_id in training_result.cv_predictions:
                        region = training_result.region_models.get(region_id)
                        if region is not None:
                            diag_var += region.cv_mse
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

        # Step 11: Build observed VariantInfo objects (excluding QC-dropped SNPs),
        # each carrying an optional per-variant fallback imputation model (P2.4) so
        # an observed variant the user's upload cannot resolve/call is recovered
        # rather than silently dropped — parity with the imputation product (P1.8).
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

        # Train the observed-variant fallbacks. Region tuning (Step 8.5) does not
        # apply here: these are imputation-style single-variant recoveries, so they
        # use the configured l1_ratio / alpha rather than the region-tuned values.
        # The trainer keys results by variant_id, which is not unique at duplicate-rsID
        # multiallelic loci, so train against a unique synthetic key and reset each
        # model's identity to its real variant afterwards.
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
                l1_ratio=self.l1_ratio,
                alpha=self.alpha,
                cv_folds=self.cv_folds,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                max_predictors=self.max_predictors,
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

        # Step 11b: Record a disposition for every input PRS variant so coverage
        # is reported honestly (no silent loss).
        observed_kept_ids = {v.variant_id for v in observed_variants_list}
        observed_fallback_ids = {m.variant_id for m in fallback_by_pos.values()}
        covered_ids: Set[str] = set()
        for region in training_result.region_models.values():
            covered_ids.update(region.prs_variant_ids)
        self._variant_dispositions = self._build_variant_dispositions(
            prs_df=prs_df,
            observed_kept_ids=observed_kept_ids,
            covered_ids=covered_ids,
            ambiguous_excluded_ids=ambiguous_excluded_ids,
            missing_drop_reason=missing_drop_reason,
            observed_fallback_ids=observed_fallback_ids,
            fallback_no_target_ids=fallback_no_target_ids,
            training_failures=training_result.failures,
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
        self._reference_panel_id = reference_panel_id
        self._training_ancestry = training_ancestry

        if self.verbose >= 1:
            print(
                f"Model fitted: {len(self._observed_variants)} observed variants, "
                f"{training_result.n_regions_trained} projection regions"
            )

        return self

    # ------------------------------------------------------------------
    # Streaming backend (Phase 2, M3): region models from banded sufficient stats.
    # ------------------------------------------------------------------
    def _auto_should_stream(self, source, all_needed_variants: Set[str]) -> bool:
        """backend='auto': stream when the estimated dense matrix is large.

        Estimated dense bytes = ``n_samples × |needed variants| × 4``. Test-sized
        inputs fall well below the threshold and stay on the dense oracle (keeping the
        golden gate exact); real reference panels select streaming.
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

    # ------------------------------------------------------------------
    # Reference cross-validation (Phase 6): per-outer-fold region models by
    # one streaming pass + additive subtraction S_full − S_fold(k).
    # ------------------------------------------------------------------
    @classmethod
    def _from_components(
        cls,
        observed: List[VariantInfo],
        region_models: List[ProjectionRegionModel],
        calib: Optional[CalibrationParams],
        metrics,
        metadata: Dict[str, Any],
    ) -> "LinearProjectionPRS":
        """Build a fitted instance from components (hermetic; no fit/IO).

        Mirrors ``LinearImputationPRS._from_components`` and the load paths' fitted-state
        population; used by the Phase-6 reference-CV fast-path to score per-fold models.
        ``metrics`` is accepted for signature symmetry (a projection model carries no
        evaluation-metrics attribute) and ignored.
        """

        def _clean(value: Any) -> Optional[str]:
            if value is None or value == "":
                return None
            return value

        instance = cls()
        instance._is_fitted = True
        instance._observed_variants = observed
        instance._region_models = region_models
        instance._calibration_params = calib
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

    def _resolve_reference_cv_inputs(
        self,
        prs_definition,
        platform_name,
        platform_manifest,
        platform_variants,
        genome_build,
        allow_alt_as_effect,
        _platform_variant_set=None,
    ):
        """Resolve ``(prs_df, platform_variant_set, all_needed_variants)`` for the
        reference-CV fast-path (mirrors ``fit`` Steps 2-3/5). See the imputation twin."""
        if isinstance(prs_definition, pd.DataFrame):
            prs_df = load_prs_from_dataframe(
                prs_definition, allow_alt_as_effect=allow_alt_as_effect
            )
        elif isinstance(prs_definition, str) and prs_definition.upper().startswith("PGS"):
            prs_df, _ = download_pgs_catalog_score(
                prs_definition, genome_build=genome_build or "GRCh37"
            )
        else:
            prs_df = load_prs_from_file(
                Path(prs_definition), allow_alt_as_effect=allow_alt_as_effect
            )

        if _platform_variant_set is not None:
            platform_variant_set = _platform_variant_set
        else:
            platform_variant_set, _, _ = resolve_platform_variant_set(
                platform_name, platform_manifest, platform_variants
            )

        prs_chrpos = set()
        _chroms, _pos = hoist_columns(prs_df, "chromosome", "position")
        for _c, _p in zip(_chroms, _pos):
            _c = str(_c).upper()
            if _c.startswith("CHR"):
                _c = _c[3:]
            prs_chrpos.add(f"{_c}:{int(_p)}")
        all_needed_variants = (
            set(prs_df["variant_id"]) | platform_variant_set | prs_chrpos
        )
        return prs_df, platform_variant_set, all_needed_variants

    def _reference_cv_fold_models(
        self,
        genotype_data: "GenotypeData",
        prs_definition,
        *,
        platform_name=None,
        platform_manifest=None,
        platform_variants=None,
        fold_indices,
        genome_build=None,
        allow_alt_as_effect: bool = False,
        _platform_variant_set=None,
    ):
        """One streaming pass → per-outer-fold projection region models by subtraction.

        Returns a ``ReferenceCVModels`` when this model's resolved backend streams, or
        ``None`` for the dense backend / ``exclude_ambiguous`` (caller uses the refit
        oracle). See :meth:`LinearImputationPRS._reference_cv_fold_models`.
        """
        from imputed_prs.compute.cv_stats import streaming_reference_cv_project

        if self.exclude_ambiguous:
            return None
        prs_df, platform_variant_set, all_needed = self._resolve_reference_cv_inputs(
            prs_definition,
            platform_name,
            platform_manifest,
            platform_variants,
            genome_build,
            allow_alt_as_effect,
            _platform_variant_set=_platform_variant_set,
        )
        source = InMemoryGenotypeSource(genotype_data, variant_ids=all_needed)
        use_stream = self.backend == "streaming" or (
            self.backend == "auto" and self._auto_should_stream(source, all_needed)
        )
        if not use_stream:
            return None
        return streaming_reference_cv_project(
            source,
            prs_df,
            platform_variant_set,
            fold_indices=fold_indices,
            window_size=self.window_size,
            max_predictors=self.max_predictors,
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            cv_folds=self.cv_folds,
            random_state=self.random_state,
            device="cpu",
            n_workers=self.n_workers,
        )

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
    ) -> "LinearProjectionPRS":
        """Train region projection models via one streaming pass over the panel.

        Produces the same fitted state the dense tail (Steps 10–12) would — region
        models, calibration params, observed ``VariantInfo`` list (with per-variant
        fallbacks), dispositions, platform index — but from banded sufficient
        statistics, never materializing the dosage matrix and never building the
        per-region ``cv_predictions`` dict (``s_true``/``s_cv`` reduced in-stream).

        Sanctioned deviations from the dense oracle (all documented; parity is exact
        on a dense, no-missing panel like 1000G):

        - Region targets are assembled as the length-``n`` vector
          ``S_R = Σ β_j x_eff_j`` accumulated in-stream — no ``X_region`` matrix and no
          per-region PRS-variant Gram ``X_RᵀX_R``; the projection SE uses ``var(S_R)``
          and ``Σ_R cv_mse_R``.
        - Mean-imputation, not listwise deletion (NaN dosages mean-imputed at
          accumulation).
        - Calibration via two O(n) accumulators: ``s_true``/``s_cv`` reduced in-stream,
          so ``ProjectionTrainingResult.cv_predictions`` is ``None``.
        - float64 accumulation matmuls (float32/GPU tradeoff deferred to Phase 3).
        - Genome build is not auto-detected from the panel (the PRS/platform build is
          used).
        - Not yet supported on streaming: ``exclude_ambiguous`` (raises
          ``NotImplementedError``) and hyperparameter tuning (``tuning_scope != "none"``
          warns and uses the configured ``l1_ratio``/``alpha``). Use ``backend="dense"``.

        Scaling note (dense scores): the region-scoped chip band buffer holds a per-fold
        Gram ``Ghold=(K, cap, cap)`` where ``cap`` is the number of chip columns buffered
        for the widest open region. On a *uniformly dense* score like PGS000027, ±W windows
        merge into ~one region per chromosome (thousands of predictors), so ``cap`` grows to
        many thousands and this Gram dominates RAM (measured ~12 GB on chr22, independent of
        n_samples) — and the mega-region fit is itself low-R². Such scores are better served
        by ``LinearImputationPRS`` (per-variant windows stay small: measured 3.7 GB, mean
        R²≈0.77 on the same chr22 data); projection targets scores whose PRS variants cluster
        into a modest number of separated regions. Bounding ``max_predictors`` caps the *fit*
        predictors but not the buffered ``cap``; a band-limited per-fold Gram is a Phase-3
        optimization (see benchmarks/results/streaming/).
        """
        from imputed_prs.compute.projection_stream import (
            StreamingProjectionFitter,
            build_projection_stream_plan,
        )
        from imputed_prs.compute.sufficient_stats import (
            _chrom_sort_key,
            collect_reference_variant_info,
        )
        from imputed_prs.evaluation.streaming_calibration import (
            finalize_projection_calibration,
        )
        from imputed_prs.models.projection_trainer import (
            _compute_projection_training_summary,
        )

        if self.exclude_ambiguous:
            raise NotImplementedError(
                "backend='streaming' does not yet support exclude_ambiguous=True "
                "(AF-based streaming QC is a follow-up). Use backend='dense'."
            )
        effective_l1_ratio = self.l1_ratio
        effective_alpha = self.alpha
        if self.tuning_scope != "none":
            # Streaming projection tuning is a follow-up. Warn unconditionally so a
            # default streaming fit never *silently* drops tuning, then use the
            # configured (l1_ratio, alpha). tuning_scope='none' silences this.
            warnings.warn(
                f"backend='streaming': tuning_scope={self.tuning_scope!r} is not yet "
                f"supported on the streaming projection path; using the configured "
                f"l1_ratio={effective_l1_ratio}, alpha={effective_alpha} "
                f"(no hyperparameter tuning performed).",
                UserWarning,
                stacklevel=2,
            )
        if self.verbose >= 1:
            print(f"Streaming backend: {len(source.sample_ids)} reference samples")

        # Metadata scan → harmonized stream plan (regions, observed, chip, fallbacks).
        chroms = sorted(
            {_normalize_chromosome(str(c)) for c in prs_df["chromosome"].unique()},
            key=_chrom_sort_key,
        )
        ref_info = collect_reference_variant_info(source, chroms)
        plan, missing_drop_reason = build_projection_stream_plan(
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
                f"Stream plan: {len(plan.regions)} regions, "
                f"{len(plan.observed)} observed calibration terms, "
                f"{len(plan.chip_ids)} chip predictors"
            )

        # Single streaming pass: region models + observed fallbacks + calibration.
        result = StreamingProjectionFitter(plan, device=self.device).run(
            source, n_workers=self.n_workers
        )

        calibration_params = None
        if result.has_calibration_terms:
            calibration_params = finalize_projection_calibration(
                result.s_true, result.s_cv, result.diag_var
            )

        # Region failures → TrainingFailure carrying member_ids (per-variant attribution).
        members_by_region = {r.region_id: list(r.prs_variant_ids) for r in plan.regions}
        training_failures: Dict[str, TrainingFailure] = {}
        for rid, msg in result.failures.items():
            etype, _, emsg = msg.partition(": ")
            training_failures[rid] = TrainingFailure(
                unit_id=rid, error_type=etype or "Error", error_message=emsg or msg,
                member_ids=tuple(members_by_region.get(rid, ())),
            )
        training_result = ProjectionTrainingResult(
            region_models=result.region_models,
            cv_predictions=None,  # the calibration blocker never materializes
            n_regions_trained=result.n_regions_trained,
            n_regions_failed=result.n_regions_failed,
            n_intercept_only=result.n_intercept_only,
            training_summary=_compute_projection_training_summary(result.region_models),
            failures=training_failures,
        )

        # Observed VariantInfo list, each carrying its per-variant fallback (P2.4).
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

        covered_ids: Set[str] = set()
        for region in result.region_models.values():
            covered_ids.update(region.prs_variant_ids)
        observed_kept_ids = {v.variant_id for v in observed_variants_list}
        self._variant_dispositions = self._build_variant_dispositions(
            prs_df=prs_df,
            observed_kept_ids=observed_kept_ids,
            covered_ids=covered_ids,
            ambiguous_excluded_ids=set(),
            missing_drop_reason=missing_drop_reason,
            observed_fallback_ids=set(result.fallback_models.keys()),
            fallback_no_target_ids=set(plan.fallback_no_target_ids),
            training_failures=training_failures,
        )

        # Populate instance state (mirrors dense Step 12).
        self._is_fitted = True
        self._observed_variants = observed_variants_list
        self._region_models = list(result.region_models.values())
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
        observed_fallback_ids: Optional[Set[str]] = None,
        fallback_no_target_ids: Optional[Set[str]] = None,
        training_failures: Optional[Dict[str, TrainingFailure]] = None,
    ) -> List[Dict[str, Any]]:
        """Build one disposition record per input PRS variant.

        Every row of ``prs_df`` yields exactly one record. ``status`` is one of
        {observed, projected, dropped}; ``reason`` is None for placed variants,
        or one of {ambiguous_excluded, reference_contig_missing, allele_mismatch,
        not_in_reference, training_failed} for dropped ones.

        For observed variants, ``has_fallback`` records whether a per-variant
        fallback model was trained (P2.4) and ``fallback_reason`` explains its
        absence ({no_reference_target, no_fallback_model}); both are False/None
        for non-observed rows.

        ``training_failures`` (region_id -> TrainingFailure) adds the
        ``failure_error_type``/``failure_error_message``/``failure_n_valid_samples``/
        ``failure_target_variance`` columns explaining *why* a region fit raised
        (P5.1). A region failure is attributed to each of its member PRS variants;
        all None for variants whose region did not fail. ``reason`` is left
        unchanged ("training_failed") so existing aggregation is stable.
        """
        observed_fallback_ids = observed_fallback_ids or set()
        fallback_no_target_ids = fallback_no_target_ids or set()
        # A region failure affects every PRS variant in that region; index the
        # failure by member variant id so each gets the explanation (P5.1).
        failure_by_member: Dict[str, TrainingFailure] = {}
        for failure in (training_failures or {}).values():
            for member_id in failure.member_ids:
                failure_by_member[member_id] = failure
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

            failure = failure_by_member.get(var_id)
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
        """Compute PRS for user genotypes using projection models.

        Args:
            user_genotypes: User genotype data as a file path (DTC format
                auto-detected) or DataFrame — both scored allele-aware
                (genotypes are oriented against each variant's effect/other
                alleles); recommended for correct scoring. A dict mapping
                variant_id to numeric dosage is a legacy, allele-blind fallback
                that bypasses allele orientation (it trusts the dosages as-is);
                prefer a file/DataFrame input.
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

        # See LinearImputationPRS.predict: for real uploads we load a multi-key
        # raw collection so BOTH the observed and projected components are scored
        # allele-aware (the predictor passes raw_genotypes to its oriented
        # scorers). The legacy dosage dict (user_dosages) is loaded alongside it
        # for the allele-blind back-compat path and missing-variant accounting,
        # hence the double parse. A numeric dict input takes only the legacy
        # allele-blind path (raw_genotypes stays None).
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
        # Observed variants recoverable via a per-variant fallback model (P2.4).
        # Computed from state so it is correct for both fitted and loaded models.
        n_observed_with_fallback = sum(
            1 for v in (self._observed_variants or []) if v.fallback is not None
        )
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

        # Training-failure breakdown by exception class (P5.1). Keyed by region
        # (the projection training unit); loaded models have no training result.
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
            "n_observed_variants": n_observed,
            "n_observed_with_fallback": n_observed_with_fallback,
            "n_missing_variants": n_missing,
            "n_definition_variants": n_definition,
            "n_dropped": n_dropped,
            "dropped_by_reason": dropped_by_reason,
            "n_training_failed": n_training_failed,
            "training_failures_by_type": training_failures_by_type,
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

    def export(
        self,
        output_dir: Union[str, Path],
        model_name: Optional[str] = None,
        formats: Optional[List[str]] = None,
        include_variance_scaling: bool = True,
    ) -> Dict[str, Path]:
        """Export trained projection model to portable formats.

        Args:
            output_dir: Directory for output files.
            model_name: Base name for output files. Uses self._model_name if None.
            formats: List of formats to export. Only "json" is supported (the
                browser-deployable artifact). Default: ["json"].
            include_variance_scaling: Accepted for parity with the imputation
                exporter; projection has no per-region residual-variance field.

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

        if formats is None:
            formats = ["json"]

        effective_model_name = (
            model_name or self._model_name or "projection_prs_model"
        )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        training_summary = None
        if self._training_result is not None:
            training_summary = self._training_result.training_summary

        common_kwargs = {
            "observed_variants": self._observed_variants or [],
            "region_models": self._region_models or [],
            "calibration_params": self._calibration_params,
            "platform_name": self._platform_name,
            "prs_id": self._prs_id,
            "genome_build": self._genome_build,
            "model_name": effective_model_name,
            "include_variance_scaling": include_variance_scaling,
            "training_summary": training_summary,
            "reference_panel_id": self._reference_panel_id,
            "training_ancestry": self._training_ancestry,
            "ambiguous_policy": self._ambiguous_policy,
        }

        # JSON is the only projection format today (HDF5/Arrow/CSV are imputation-
        # only; a projection loader arrives in P2.2).
        valid_formats = {"json"}
        invalid_formats = set(formats) - valid_formats
        if invalid_formats:
            raise ValueError(
                f"Unsupported export formats: {invalid_formats}. "
                f"Valid formats: {valid_formats}"
            )

        output_paths: Dict[str, Path] = {}
        for fmt in formats:
            if fmt == "json":
                output_path = output_dir / f"{effective_model_name}.json"
                export_projection_to_json(output_path=output_path, **common_kwargs)
                output_paths["json"] = output_path

        return output_paths

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LinearProjectionPRS":
        """Load a trained projection model from a JSON artifact.

        Projection models export to JSON only (the browser-deployable artifact; the
        HDF5/Arrow/CSV formats remain imputation-only), so ``load`` accepts a
        ``.json`` file and raises on any other format.

        Args:
            path: Path to a saved projection model (a ``.json`` file).

        Returns:
            Loaded LinearProjectionPRS instance ready for prediction.

        Raises:
            DataLoadError: If the file is missing or its format is unsupported.
        """
        path = Path(path)

        if not path.exists():
            raise DataLoadError(f"Model file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".json":
            return cls._load_from_json(path)
        raise DataLoadError(
            f"Unsupported projection model file format: {suffix or path.name}. "
            "Projection models are exported to JSON only (.json)."
        )

    @classmethod
    def _load_from_json(cls, path: Path) -> "LinearProjectionPRS":
        """Reconstruct a fitted projection model from a JSON file (schema v2.0).

        Mirrors :meth:`LinearImputationPRS._load_from_json`. Restores exactly the
        state ``predict()`` consumes: the region models (with P1.3 predictor allele
        metadata), the observed terms (and their optional P2.4 fallbacks),
        calibration, and the identity/provenance the build/platform guard checks.
        Training-time diagnostics (``_training_result``/``_variant_dispositions``)
        are not serialized and stay ``None``; ``summary``/``variant_dispositions``
        tolerate that.
        """
        from imputed_prs.io.loaders import (
            load_projection_model_json,
            parse_imputed_model_json,
            parse_projection_region_model,
        )

        try:
            data = load_projection_model_json(path)
        except Exception as e:
            raise DataLoadError(
                f"Failed to load projection JSON model: {e}"
            ) from e

        instance = cls()

        # Observed variants, each with an optional per-variant fallback (P2.4). The
        # fallback block is the same self-describing shape the imputation product
        # emits, so it reconstructs through the shared reader.
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

        # Region models — each predictor / projected PRS variant is self-describing,
        # so the index-aligned dataclass arrays rebuild via the shared reader.
        region_models = [
            parse_projection_region_model(r)
            for r in data.get("region_models", [])
        ]

        # Calibration: the top-level block mirrors the imputation export; fall back
        # to the provenance centering/scaling copy when only that is present.
        metadata = data.get("metadata", {})
        provenance = data.get("provenance", {})
        calib_source = data.get("calibration_params") or provenance.get(
            "centering_scaling"
        )
        calib_params = CalibrationParams(**calib_source) if calib_source else None

        # Populate fitted state.
        instance._is_fitted = True
        instance._observed_variants = observed_variants
        instance._region_models = region_models
        instance._calibration_params = calib_params
        instance._platform_variant_index = data.get("platform_variant_index")

        # Identity + provenance, with the same precedence as the imputation loader:
        # the provenance block is authoritative for the deploy fields, falling back
        # to `metadata` for the identity fields that predate it.
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

    def __repr__(self) -> str:
        """String representation of the model."""
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"LinearProjectionPRS(window_size={self.window_size}, "
            f"cv_folds={self.cv_folds}, status={status})"
        )
