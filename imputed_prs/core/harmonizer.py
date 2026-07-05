"""Variant harmonization between PRS definitions, platforms, and reference data.

This module provides functions to harmonize variants across different data sources:
- Partition PRS variants into observed (on platform) and missing (need imputation) sets
- Align effect alleles between PRS and genotype data
- Validate genome build compatibility
- Filter variants to local genomic windows
"""

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from imputed_prs.core.exceptions import (
    DataLoadError,
    IncompatibleBuildError,
    IncompatiblePlatformError,
)
from imputed_prs.core.types import GenotypeData


# Build aliases for normalization (case-insensitive)
BUILD_ALIASES = {
    "hg19": "GRCh37",
    "hg38": "GRCh38",
    "grch37": "GRCh37",
    "grch38": "GRCh38",
    "b37": "GRCh37",
    "b38": "GRCh38",
}

# Complement mapping for strand flipping
_COMPLEMENT_MAP = {"A": "T", "T": "A", "C": "G", "G": "C"}


def _complement(allele: str) -> str:
    """Get the complement of an allele.

    Maps A<->T, C<->G for strand flipping.

    Parameters
    ----------
    allele : str
        Single nucleotide allele (A, T, C, or G).

    Returns
    -------
    str
        Complement allele, or original if not a standard nucleotide.
    """
    return _COMPLEMENT_MAP.get(allele.upper(), allele.upper())


def _is_ambiguous_snp(a1: str, a2: str) -> bool:
    """Check if a SNP is ambiguous (A/T or C/G pair).

    Ambiguous SNPs cannot be reliably strand-flipped because
    the complement is the same as the original.

    Parameters
    ----------
    a1 : str
        First allele.
    a2 : str
        Second allele.

    Returns
    -------
    bool
        True if the allele pair is ambiguous.
    """
    a1_upper = a1.upper()
    a2_upper = a2.upper()

    # A/T pairs
    if (a1_upper == "A" and a2_upper == "T") or (a1_upper == "T" and a2_upper == "A"):
        return True

    # C/G pairs
    if (a1_upper == "C" and a2_upper == "G") or (a1_upper == "G" and a2_upper == "C"):
        return True

    return False


def _normalize_build(build: Optional[str]) -> Optional[str]:
    """Normalize a genome build string.

    Parameters
    ----------
    build : str or None
        Genome build string (e.g., "hg19", "GRCh37", "b38").

    Returns
    -------
    str or None
        Normalized build ("GRCh37" or "GRCh38"), or None if input is None.
    """
    if build is None:
        return None

    build_lower = build.strip().lower()

    if build_lower in BUILD_ALIASES:
        return BUILD_ALIASES[build_lower]

    # Already normalized
    if build in ("GRCh37", "GRCh38"):
        return build

    # Return as-is for unknown builds
    return build


# Float-stringified numeric chromosome, e.g. "22.0" produced when a mixed-dtype
# source column is inferred as float. Guarded to purely-numeric names so that
# versioned scaffold accessions (e.g. "GL000220.0") are never touched.
_CHROM_FLOAT_RE = re.compile(r"^(\d+)\.0$")


def _normalize_chromosome(chrom: str) -> str:
    """Normalize chromosome name.

    Strips 'chr' prefix, repairs float-stringified numeric chromosomes, and
    normalizes sex/mitochondrial chromosomes.

    Parameters
    ----------
    chrom : str
        Chromosome string (e.g., "chr1", "CHR22", "chrX", "22.0").

    Returns
    -------
    str
        Normalized chromosome (e.g., "1", "22", "X").
    """
    chrom = str(chrom).upper()
    # Strip chr prefix
    if chrom.startswith("CHR"):
        chrom = chrom[3:]
    # Repair float artifacts like "22.0" -> "22" (see _CHROM_FLOAT_RE). This
    # arises when a harmonized PGS Catalog file's chromosome column is inferred
    # as float; without repair, "22.0" fails to match a "chr22" reference and
    # those variants are silently dropped.
    match = _CHROM_FLOAT_RE.match(chrom)
    if match:
        chrom = match.group(1)
    # Normalize common aliases
    if chrom == "MT":
        chrom = "M"
    return chrom


def normalize_chromosome_array(chromosomes) -> np.ndarray:
    """Vectorized chromosome normalization, element-wise identical to
    :func:`_normalize_chromosome`.

    Normalizes only the *unique* input values through the scalar function (a
    handful of distinct chromosomes even for millions of rows) and maps them
    back. Equivalent to ``pd.Series(chromosomes).apply(lambda x:
    _normalize_chromosome(str(x)))`` but at a fraction of the cost, which is why
    it can replace the per-call ``.apply``/``.iterrows`` normalization in the
    windowing and reference-indexing hotspots.

    Parameters
    ----------
    chromosomes : pandas.Series or array-like
        Raw chromosome values.

    Returns
    -------
    numpy.ndarray
        Object array of normalized chromosome strings, aligned to the input.
    """
    # ``.map(str)`` applies Python ``str`` element-wise, reproducing the scalar
    # path's ``str(x)`` exactly (NaN -> "nan", None -> "None") and turning every
    # value into a real string so the subsequent dict-map has no NaN-key
    # mismatch. Only the handful of *distinct* strings pay the regex cost.
    as_str = pd.Series(chromosomes).map(str)
    mapping = {value: _normalize_chromosome(value) for value in as_str.unique()}
    return as_str.map(mapping).to_numpy()


def hoist_columns(df: pd.DataFrame, *names: str) -> List[list]:
    """Return per-column Python lists for fast index-based row iteration.

    A drop-in replacement for ``df.iterrows()`` in per-variant harmonization
    loops: ``iterrows`` builds a Series per row (~60x slower at PRS scale and it
    can upcast dtypes), whereas indexing pre-extracted lists does neither. An
    absent optional column yields ``[None] * len(df)`` to mirror ``row.get``.

    Parameters
    ----------
    df : pd.DataFrame
        Frame to hoist.
    *names : str
        Column names, in the order the returned lists should be unpacked.

    Returns
    -------
    List[list]
        One Python list per requested name, each aligned to ``df`` row order.
    """
    n = len(df)
    return [
        df[name].tolist() if name in df.columns else [None] * n
        for name in names
    ]


def _build_variant_lookup(
    variant_ids: Set[str],
) -> Tuple[Set[str], Set[str]]:
    """Build lookup sets for variant filtering.

    Parameters
    ----------
    variant_ids : Set[str]
        Set of variant IDs in rsID or chr:pos format.

    Returns
    -------
    Tuple[Set[str], Set[str]]
        Tuple of (rsid_set, chrpos_set) for matching.
        rsID set is lowercase for case-insensitive matching.
        chrpos set has normalized chromosome names.
    """
    rsid_set = set()
    chrpos_set = set()

    for vid in variant_ids:
        if vid.lower().startswith("rs"):
            rsid_set.add(vid.lower())
        elif ":" in vid:
            # Normalize chr:pos format
            parts = vid.split(":")
            if len(parts) >= 2:
                chrom = _normalize_chromosome(parts[0])
                pos = parts[1]
                chrpos_set.add(f"{chrom}:{pos}")
        else:
            # Assume it's an rsID-like identifier
            rsid_set.add(vid.lower())

    return rsid_set, chrpos_set


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class BuildValidationResult:
    """Result from genome build validation.

    Attributes
    ----------
    is_compatible : bool
        True if builds are compatible.
    prs_build : str or None
        Normalized PRS build.
    genotype_build : str or None
        Normalized genotype build.
    warning : str or None
        Warning message if builds could not be determined.
    """

    is_compatible: bool
    prs_build: Optional[str]
    genotype_build: Optional[str]
    warning: Optional[str]


@dataclass
class PartitionResult:
    """Result from partitioning PRS variants.

    Attributes
    ----------
    observed : FrozenSet[str]
        Variant IDs found on the platform.
    missing : FrozenSet[str]
        Variant IDs not found on the platform (need imputation).
    observed_by_rsid : int
        Count of variants matched by rsID.
    observed_by_chrpos : int
        Count of variants matched by chr:pos.
    prs_to_platform_id : Dict[str, str]
        Mapping from PRS variant ID to matched platform variant ID.
    """

    observed: FrozenSet[str]
    missing: FrozenSet[str]
    observed_by_rsid: int
    observed_by_chrpos: int
    prs_to_platform_id: Dict[str, str]


@dataclass
class AlleleAlignmentResult:
    """Result from aligning effect alleles.

    Attributes
    ----------
    aligned_dosage_matrix : np.ndarray
        Dosage matrix with flipped values where needed (n_samples x n_variants).
    variant_mask : np.ndarray
        Boolean mask indicating which PRS variants were matched in genotype data.
    flip_mask : np.ndarray
        Boolean mask indicating which variants were flipped.
    n_matched : int
        Number of variants successfully matched.
    n_flipped : int
        Number of variants that required flipping.
    n_ambiguous : int
        Number of ambiguous A/T or C/G SNPs.
    n_unmatched_alleles : int
        Number of variants where alleles could not be matched.
    """

    aligned_dosage_matrix: np.ndarray
    variant_mask: np.ndarray
    flip_mask: np.ndarray
    n_matched: int
    n_flipped: int
    n_ambiguous: int
    n_unmatched_alleles: int


@dataclass
class WindowFilterResult:
    """Result from filtering to a local genomic window.

    Attributes
    ----------
    variant_ids : List[str]
        Variant IDs within the window.
    variant_indices : np.ndarray
        Indices into the original variant_info DataFrame.
    distances : np.ndarray
        Distance from target position (in base pairs).
    n_variants : int
        Number of variants in the window.
    """

    variant_ids: List[str]
    variant_indices: np.ndarray
    distances: np.ndarray
    n_variants: int


# =============================================================================
# Main Functions
# =============================================================================


def validate_genome_build(
    prs_build: Optional[str],
    genotype_build: Optional[str],
    strict: bool = True,
) -> BuildValidationResult:
    """Check genome build compatibility between PRS and genotype data.

    Parameters
    ----------
    prs_build : str or None
        Genome build of the PRS (e.g., "GRCh37", "hg38").
    genotype_build : str or None
        Genome build of the genotype data.
    strict : bool
        If True, raise IncompatibleBuildError on mismatch.
        If False, return is_compatible=False.

    Returns
    -------
    BuildValidationResult
        Result containing compatibility status and normalized builds.

    Raises
    ------
    IncompatibleBuildError
        If strict=True and builds are incompatible.
    """
    # Normalize builds
    prs_normalized = _normalize_build(prs_build)
    genotype_normalized = _normalize_build(genotype_build)

    # Handle None values
    warning = None
    if prs_normalized is None and genotype_normalized is None:
        warning = "Both PRS and genotype builds are unknown; cannot verify compatibility"
        return BuildValidationResult(
            is_compatible=True,  # Assume compatible if both unknown
            prs_build=None,
            genotype_build=None,
            warning=warning,
        )

    if prs_normalized is None:
        warning = f"PRS build is unknown; assuming compatible with {genotype_normalized}"
        return BuildValidationResult(
            is_compatible=True,
            prs_build=None,
            genotype_build=genotype_normalized,
            warning=warning,
        )

    if genotype_normalized is None:
        warning = f"Genotype build is unknown; assuming compatible with {prs_normalized}"
        return BuildValidationResult(
            is_compatible=True,
            prs_build=prs_normalized,
            genotype_build=None,
            warning=warning,
        )

    # Both builds known - check compatibility
    is_compatible = prs_normalized == genotype_normalized

    if not is_compatible and strict:
        raise IncompatibleBuildError(
            f"Genome build mismatch: PRS is {prs_normalized}, "
            f"genotype data is {genotype_normalized}. "
            "Use strict=False to skip this check, but results may be unreliable."
        )

    return BuildValidationResult(
        is_compatible=is_compatible,
        prs_build=prs_normalized,
        genotype_build=genotype_normalized,
        warning=None,
    )


# Map the integer build numbers returned by user_genotypes.detect_genome_build
# onto the canonical build names understood by validate_genome_build.
_BUILD_NUMBER_TO_NAME = {37: "GRCh37", 38: "GRCh38"}


def check_predict_compatibility(
    model_build: Optional[str],
    model_platform: Optional[str],
    user_input,
    declared_build: Optional[str] = None,
    declared_platform: Optional[str] = None,
    strict: bool = True,
) -> None:
    """Guard ``predict`` against an incompatible genome build / platform.

    Enforces, at the scoring boundary, that a deployable model is not silently
    applied to an upload on a different genome build or genotyping array.

    Genome build
        The user build is taken from ``declared_build`` if given; otherwise, when
        the model declares a build *and* ``user_input`` is a file path, it is
        auto-detected with :func:`imputed_prs.io.user_genotypes.detect_genome_build`.
        DataFrame/dict inputs carry no build and are never auto-detected.

        - A *known mismatch* against ``model_build`` raises
          :class:`IncompatibleBuildError` when ``strict`` (the default), or emits a
          blocking ``UserWarning`` when ``strict=False``.
        - An *unknown* user build, when the model build is known, emits a blocking
          ``UserWarning`` and scoring proceeds.
        - When the model build itself is unknown, no check is possible and nothing
          is emitted (research models without a declared build stay quiet).

    Platform
        Only checked when the caller declares ``declared_platform`` and the model
        records one. A mismatch raises :class:`IncompatiblePlatformError` when
        ``strict``, else emits a blocking ``UserWarning``.

    Parameters
    ----------
    model_build : str or None
        Genome build recorded on the model (e.g. ``"GRCh37"``).
    model_platform : str or None
        Genotyping platform recorded on the model (e.g. ``"23andme_v5"``).
    user_input : str, pathlib.Path, pandas.DataFrame, or dict
        The raw value passed to ``predict``; used to auto-detect the build for
        file-path inputs.
    declared_build : str or None
        Build explicitly declared by the caller; overrides auto-detection.
    declared_platform : str or None
        Platform explicitly declared by the caller.
    strict : bool
        If True (default), a known build/platform mismatch raises. If False, the
        mismatch is downgraded to a blocking ``UserWarning`` and scoring proceeds.

    Raises
    ------
    IncompatibleBuildError
        If ``strict`` and the user build is known and mismatches ``model_build``.
    IncompatiblePlatformError
        If ``strict`` and ``declared_platform`` mismatches ``model_platform``.
    """
    # Resolve the user's genome build. Auto-detection is bounded to the case
    # where it can actually help: the model declares a build and the input is a
    # file path (DataFrames and dicts carry no build to detect). This also keeps
    # the extra full-file parse off the in-memory research paths.
    resolved_build = declared_build
    if (
        resolved_build is None
        and model_build is not None
        and isinstance(user_input, (str, Path))
    ):
        # Function-local import: harmonizer lives in ``core`` and
        # detect_genome_build in ``io`` (which imports ``core``), so a
        # module-level import would create a cycle.
        from imputed_prs.io.user_genotypes import detect_genome_build

        try:
            build_number = detect_genome_build(user_input)
        except DataLoadError:
            # The build probe failed (e.g. the optional ``snps`` package is not
            # installed, or the file is unreadable). Degrade to unknown; the
            # genotype loaders re-parse the file next and raise their own clear
            # error if it is genuinely bad.
            build_number = None
        resolved_build = _BUILD_NUMBER_TO_NAME.get(build_number)

    # Genome build compatibility. validate_genome_build raises on a known
    # mismatch when strict=True; otherwise it returns is_compatible=False with no
    # warning text, and we surface the blocking warning ourselves.
    result = validate_genome_build(model_build, resolved_build, strict=strict)
    if not result.is_compatible:
        # Only reachable with strict=False (strict=True has already raised).
        warnings.warn(
            f"Genome build mismatch: model is {result.prs_build}, user data is "
            f"{result.genotype_build}. Proceeding because strict=False; results "
            "may be unreliable.",
            UserWarning,
            stacklevel=2,
        )
    elif resolved_build is None and model_build is not None:
        # The user build could not be determined but the model declares one.
        # Block loudly, then score. (Silent when the model build is also unknown.)
        warnings.warn(
            "Could not determine the genome build of the user genotypes; the "
            f"model was trained on {model_build}. Pass genome_build= to verify "
            "compatibility. Proceeding, but results may be unreliable if the "
            "builds differ.",
            UserWarning,
            stacklevel=2,
        )

    # Platform compatibility (only when both sides are known). Platform names are
    # controlled free-form values, so an exact string comparison is used.
    if (
        declared_platform is not None
        and model_platform is not None
        and declared_platform != model_platform
    ):
        message = (
            f"Genotyping platform mismatch: model was trained for "
            f"'{model_platform}', but '{declared_platform}' was declared."
        )
        if strict:
            raise IncompatiblePlatformError(
                message + " Use strict=False to skip this check, but results may "
                "be unreliable."
            )
        warnings.warn(
            message + " Proceeding because strict=False; results may be "
            "unreliable.",
            UserWarning,
            stacklevel=2,
        )


def partition_variants(
    prs_df: pd.DataFrame,
    platform_variants: Set[str],
) -> PartitionResult:
    """Partition PRS variants into observed (on platform) and missing sets.

    Tries to match variants by rsID first, then falls back to chr:pos matching.

    Parameters
    ----------
    prs_df : pd.DataFrame
        PRS DataFrame with columns: variant_id, chromosome, position.
    platform_variants : Set[str]
        Set of variant IDs on the genotyping platform.
        Can contain rsIDs (e.g., "rs123") or chr:pos (e.g., "1:12345").

    Returns
    -------
    PartitionResult
        Result containing observed/missing sets and match statistics.
    """
    # Build lookup sets from platform variants
    platform_rsid_set, platform_chrpos_set = _build_variant_lookup(platform_variants)

    observed = set()
    missing = set()
    observed_by_rsid = 0
    observed_by_chrpos = 0
    prs_to_platform_id: Dict[str, str] = {}

    for _, row in prs_df.iterrows():
        variant_id = row["variant_id"]
        chrom = _normalize_chromosome(str(row["chromosome"]))
        pos = str(int(row["position"]))
        chrpos = f"{chrom}:{pos}"

        # Try rsID match first (case-insensitive)
        variant_id_lower = variant_id.lower()
        if variant_id_lower in platform_rsid_set:
            observed.add(variant_id)
            observed_by_rsid += 1
            prs_to_platform_id[variant_id] = variant_id
            continue

        # Try chr:pos match as fallback
        if chrpos in platform_chrpos_set:
            observed.add(variant_id)
            observed_by_chrpos += 1
            prs_to_platform_id[variant_id] = chrpos
            continue

        # No match found
        missing.add(variant_id)

    return PartitionResult(
        observed=frozenset(observed),
        missing=frozenset(missing),
        observed_by_rsid=observed_by_rsid,
        observed_by_chrpos=observed_by_chrpos,
        prs_to_platform_id=prs_to_platform_id,
    )


def filter_to_local_window(
    target_chrom: str,
    target_pos: int,
    variant_info: pd.DataFrame,
    window_size: int = 1_000_000,
    exclude_target: bool = True,
    max_variants: Optional[int] = None,
) -> WindowFilterResult:
    """Filter variants to a local genomic window around a target position.

    Parameters
    ----------
    target_chrom : str
        Target chromosome (e.g., "1", "chr1", "X").
    target_pos : int
        Target genomic position.
    variant_info : pd.DataFrame
        DataFrame with variant information. Must have columns:
        variant_id, chromosome, position.
    window_size : int
        Window size in base pairs on each side of target (default 1Mb).
    exclude_target : bool
        If True, exclude variants at the exact target position.
    max_variants : int or None
        If specified, limit to the closest N variants.

    Returns
    -------
    WindowFilterResult
        Result containing filtered variant IDs, indices, and distances.
    """
    # Normalize target chromosome
    target_chrom_norm = _normalize_chromosome(target_chrom)

    # Normalize variant chromosomes for comparison
    chroms = variant_info["chromosome"].apply(lambda x: _normalize_chromosome(str(x)))

    # Filter to same chromosome
    chrom_mask = chroms == target_chrom_norm

    if not chrom_mask.any():
        return WindowFilterResult(
            variant_ids=[],
            variant_indices=np.array([], dtype=int),
            distances=np.array([], dtype=int),
            n_variants=0,
        )

    # Compute distances
    positions = variant_info["position"].values
    distances = np.abs(positions - target_pos)

    # Apply window filter
    window_mask = (distances <= window_size) & chrom_mask.values

    # Optionally exclude target position
    if exclude_target:
        not_target = positions != target_pos
        window_mask = window_mask & not_target

    # Get indices of variants in window
    indices = np.where(window_mask)[0]

    if len(indices) == 0:
        return WindowFilterResult(
            variant_ids=[],
            variant_indices=np.array([], dtype=int),
            distances=np.array([], dtype=int),
            n_variants=0,
        )

    # Get distances for these variants
    filtered_distances = distances[indices]

    # Sort by distance if max_variants specified
    if max_variants is not None and len(indices) > max_variants:
        sort_order = np.argsort(filtered_distances)[:max_variants]
        indices = indices[sort_order]
        filtered_distances = filtered_distances[sort_order]

    # Get variant IDs
    variant_ids = variant_info.iloc[indices]["variant_id"].tolist()

    return WindowFilterResult(
        variant_ids=variant_ids,
        variant_indices=indices,
        distances=filtered_distances,
        n_variants=len(indices),
    )


def align_effect_alleles(
    prs_df: pd.DataFrame,
    genotype_data: GenotypeData,
    observed_variants: FrozenSet[str],
) -> AlleleAlignmentResult:
    """Align effect alleles between PRS and genotype data.

    Flips dosages when the PRS effect_allele matches the genotype ref_allele,
    since dosage typically counts the alt allele.

    Parameters
    ----------
    prs_df : pd.DataFrame
        PRS DataFrame with columns: variant_id, effect_allele, other_allele.
    genotype_data : GenotypeData
        Loaded genotype data with dosage_matrix and variant_info.
    observed_variants : FrozenSet[str]
        Set of PRS variant IDs that are observed in the genotype data.

    Returns
    -------
    AlleleAlignmentResult
        Result containing aligned dosage matrix and alignment statistics.
    """
    n_samples = genotype_data.n_samples
    n_prs_variants = len(prs_df)

    # Initialize output matrix (all NaN initially)
    aligned_dosage = np.full((n_samples, n_prs_variants), np.nan, dtype=np.float32)

    # Build lookup from genotype variant_id to index
    geno_var_info = genotype_data.variant_info
    geno_var_to_idx: Dict[str, int] = {}
    geno_chrpos_to_idx: Dict[str, int] = {}

    for idx, row in geno_var_info.iterrows():
        var_id = row["variant_id"]
        chrom = _normalize_chromosome(str(row["chromosome"]))
        pos = str(int(row["position"]))
        chrpos = f"{chrom}:{pos}"

        geno_var_to_idx[var_id.lower()] = idx
        geno_chrpos_to_idx[chrpos] = idx

    # Track statistics
    variant_mask = np.zeros(n_prs_variants, dtype=bool)
    flip_mask = np.zeros(n_prs_variants, dtype=bool)
    n_matched = 0
    n_flipped = 0
    n_ambiguous = 0
    n_unmatched_alleles = 0

    for prs_idx, row in prs_df.iterrows():
        variant_id = row["variant_id"]

        # Skip if not in observed set
        if variant_id not in observed_variants:
            continue

        # Find corresponding genotype variant
        geno_idx = None

        # Try rsID match
        var_id_lower = variant_id.lower()
        if var_id_lower in geno_var_to_idx:
            geno_idx = geno_var_to_idx[var_id_lower]
        else:
            # Try chr:pos match
            chrom = _normalize_chromosome(str(row["chromosome"]))
            pos = str(int(row["position"]))
            chrpos = f"{chrom}:{pos}"
            if chrpos in geno_chrpos_to_idx:
                geno_idx = geno_chrpos_to_idx[chrpos]

        if geno_idx is None:
            continue

        # Get alleles
        prs_effect = row["effect_allele"].upper()
        prs_other = row.get("other_allele", "")
        if pd.notna(prs_other):
            prs_other = str(prs_other).upper()
        else:
            prs_other = ""

        geno_row = geno_var_info.iloc[geno_idx]
        geno_ref = str(geno_row["ref_allele"]).upper()
        geno_alt = str(geno_row["alt_allele"]).upper()

        # Get the row index in the DataFrame (for accessing aligned_dosage)
        prs_row_idx = prs_df.index.get_loc(prs_idx)

        # Get dosage values for this variant
        dosage = genotype_data.dosage_matrix[:, geno_idx].copy()

        # Check for ambiguous SNP
        is_ambiguous = _is_ambiguous_snp(geno_ref, geno_alt)
        if is_ambiguous:
            n_ambiguous += 1

        # Determine if flip is needed
        # Dosage counts alt allele, so:
        # - effect_allele == alt_allele -> no flip
        # - effect_allele == ref_allele -> flip (dosage = 2 - dosage)
        need_flip = False
        matched = False

        # Direct match: effect == alt
        if prs_effect == geno_alt:
            matched = True
            need_flip = False
        # Direct match: effect == ref
        elif prs_effect == geno_ref:
            matched = True
            need_flip = True
        # Try complement for strand issues
        elif _complement(prs_effect) == geno_alt:
            matched = True
            need_flip = False
        elif _complement(prs_effect) == geno_ref:
            matched = True
            need_flip = True
        # Also check if other_allele matches (for validation)
        elif prs_other and prs_other == geno_alt:
            # effect_allele should be ref
            matched = True
            need_flip = True
        elif prs_other and prs_other == geno_ref:
            # effect_allele should be alt
            matched = True
            need_flip = False

        if not matched:
            n_unmatched_alleles += 1
            continue

        # Apply flip if needed
        if need_flip:
            dosage = 2.0 - dosage
            flip_mask[prs_row_idx] = True
            n_flipped += 1

        aligned_dosage[:, prs_row_idx] = dosage
        variant_mask[prs_row_idx] = True
        n_matched += 1

    return AlleleAlignmentResult(
        aligned_dosage_matrix=aligned_dosage,
        variant_mask=variant_mask,
        flip_mask=flip_mask,
        n_matched=n_matched,
        n_flipped=n_flipped,
        n_ambiguous=n_ambiguous,
        n_unmatched_alleles=n_unmatched_alleles,
    )


def build_reference_allele_index(variant_info: pd.DataFrame) -> Dict[str, List[int]]:
    """Map each normalized ``chr:pos`` to the reference rows at that locus.

    After multi-allelic records are split (one row per ALT allele), a single
    genomic position can map to several rows. This index groups the candidate
    rows so allele-aware matching can select the correct one.

    Parameters
    ----------
    variant_info : pd.DataFrame
        Reference variant metadata with columns chromosome, position,
        ref_allele, alt_allele.

    Returns
    -------
    Dict[str, List[int]]
        Mapping from "chrom:pos" to a list of *positional* row indices.
    """
    n = len(variant_info)
    if n == 0:
        return {}
    chroms = normalize_chromosome_array(variant_info["chromosome"]).tolist()
    positions = variant_info["position"].to_numpy().tolist()
    # int(p) matches the scalar path and avoids float-tainted keys ("22:1.0").
    keys = [f"{chrom}:{int(pos)}" for chrom, pos in zip(chroms, positions)]
    # groupby(sort=False).indices yields, per key in first-appearance order, the
    # ascending positional row indices at that locus -- identical to the
    # enumerate/setdefault loop this replaces. .tolist() is required: consumers
    # do ``if not candidates`` / iterate the list, which breaks on an ndarray.
    grouped = pd.Series(np.arange(n)).groupby(keys, sort=False).indices
    return {key: rows.tolist() for key, rows in grouped.items()}


def match_oriented_dosage(
    chromosome: str,
    position: int,
    effect_allele: str,
    other_allele: Optional[str],
    variant_info: pd.DataFrame,
    dosage_matrix: np.ndarray,
    reference_index: Dict[str, List[int]],
) -> Optional[Tuple[int, np.ndarray, bool]]:
    """Resolve a PRS variant against the reference and return effect-oriented dosage.

    Matching keys on ``chr:pos`` and then resolves the exact reference row by
    comparing the PRS ``(effect, other)`` alleles against each candidate's
    ``(ref, alt)`` — directly and via the complementary strand. The returned
    dosage counts copies of the *effect* allele: it is flipped to ``2 - dosage``
    when the effect allele is the reference allele, since the per-row dosage
    counts the ALT allele.

    Parameters
    ----------
    chromosome, position : str, int
        PRS variant locus.
    effect_allele, other_allele : str, Optional[str]
        PRS effect and other alleles. ``other_allele`` may be None/NaN.
    variant_info : pd.DataFrame
        Reference variant metadata (must include ref_allele, alt_allele).
    dosage_matrix : np.ndarray
        Reference dosage matrix (n_samples x n_variants), ALT-allele counts.
    reference_index : Dict[str, List[int]]
        Output of :func:`build_reference_allele_index` for ``variant_info``.

    Returns
    -------
    Optional[Tuple[int, np.ndarray, bool]]
        ``(reference_row_index, oriented_dosage, was_flipped)``, or None when no
        allele-compatible reference variant exists at the locus.
    """
    chrom = _normalize_chromosome(str(chromosome))
    candidates = reference_index.get(f"{chrom}:{int(position)}")
    if not candidates:
        return None

    effect = str(effect_allele).upper()
    if other_allele is None or (isinstance(other_allele, float) and pd.isna(other_allele)):
        other = ""
    else:
        other = str(other_allele).upper()

    # Pass 1 matches alleles directly; pass 2 retries on the complementary strand.
    for use_complement in (False, True):
        eff = _complement(effect) if use_complement else effect
        oth = _complement(other) if (use_complement and other) else other
        for idx in candidates:
            row = variant_info.iloc[idx]
            ref = str(row["ref_allele"]).upper()
            alt = str(row["alt_allele"]).upper()
            dosage = dosage_matrix[:, idx]
            # Effect allele is the ALT -> dosage already counts the effect allele.
            if eff == alt and (oth == ref or oth == ""):
                return idx, dosage.copy(), False
            # Effect allele is the REF -> flip so dosage counts the effect allele.
            if eff == ref and (oth == alt or oth == ""):
                return idx, 2.0 - dosage, True

    return None


class ReferenceAlleleResolver:
    """Effect-allele-oriented dosage resolver over a fixed reference panel.

    Built once from a reference ``variant_info``; :meth:`resolve` reproduces
    :func:`match_oriented_dosage` exactly but resolves candidate rows through
    pre-extracted numpy arrays instead of a per-candidate
    ``variant_info.iloc[idx]``. The ``.iloc`` Series construction is the dominant
    per-variant cost when a whole PRS is harmonized (``match_oriented_dosage`` is
    called once per PRS variant across several passes), so this array access is
    the hotspot fix. ``match_oriented_dosage`` is retained unchanged as the
    differential oracle.

    Parameters
    ----------
    variant_info : pd.DataFrame
        Reference variant metadata with columns chromosome, position,
        ref_allele, alt_allele.
    """

    def __init__(self, variant_info: pd.DataFrame):
        self.locus_to_rows: Dict[str, List[int]] = build_reference_allele_index(
            variant_info
        )
        # Pre-uppercase with element-wise ``str(x).upper()`` (NOT
        # ``Series.str.upper``, which maps None/NaN to NaN rather than
        # "NONE"/"NAN") so the comparisons in resolve() are byte-identical to the
        # scalar oracle, including for ALT-less records whose alt_allele is None.
        self._ref = np.array(
            [str(a).upper() for a in variant_info["ref_allele"].to_numpy()],
            dtype=object,
        )
        self._alt = np.array(
            [str(a).upper() for a in variant_info["alt_allele"].to_numpy()],
            dtype=object,
        )

    def resolve(
        self,
        chromosome: str,
        position: int,
        effect_allele: str,
        other_allele: Optional[str],
        dosage_matrix: np.ndarray,
    ) -> Optional[Tuple[int, np.ndarray, bool]]:
        """Resolve a PRS variant to effect-oriented dosage.

        Equivalent to ``match_oriented_dosage(chromosome, position,
        effect_allele, other_allele, variant_info, dosage_matrix,
        self.locus_to_rows)`` for the ``variant_info`` this resolver was built
        from. Returns ``(reference_row_index, oriented_dosage, was_flipped)`` or
        None when no allele-compatible reference row exists at the locus.
        """
        chrom = _normalize_chromosome(str(chromosome))
        candidates = self.locus_to_rows.get(f"{chrom}:{int(position)}")
        if not candidates:
            return None

        effect = str(effect_allele).upper()
        if other_allele is None or (
            isinstance(other_allele, float) and pd.isna(other_allele)
        ):
            other = ""
        else:
            other = str(other_allele).upper()

        # Pass 1 matches alleles directly; pass 2 retries the complementary strand.
        for use_complement in (False, True):
            eff = _complement(effect) if use_complement else effect
            oth = _complement(other) if (use_complement and other) else other
            for idx in candidates:
                ref = self._ref[idx]
                alt = self._alt[idx]
                dosage = dosage_matrix[:, idx]
                # Effect allele is the ALT -> dosage already counts it.
                if eff == alt and (oth == ref or oth == ""):
                    return idx, dosage.copy(), False
                # Effect allele is the REF -> flip so dosage counts the effect.
                if eff == ref and (oth == alt or oth == ""):
                    return idx, 2.0 - dosage, True

        return None

    def would_resolve(
        self,
        chromosome: str,
        position: int,
        effect_allele: str,
        other_allele: Optional[str],
    ) -> Optional[Tuple[int, bool]]:
        """Metadata-only :meth:`resolve`: the reference row + flip, no dosage needed.

        Returns ``(reference_row_index, was_flipped)`` for the same reference row
        :meth:`resolve` would select (``was_flipped`` True ⇒ effect allele is the
        reference REF, so the effect-oriented dosage is ``2 − ALT``), or None when no
        allele-compatible row exists. This lets the streaming path classify PRS
        variants and precompute per-variant flip flags without materializing the
        dosage matrix.
        """
        chrom = _normalize_chromosome(str(chromosome))
        candidates = self.locus_to_rows.get(f"{chrom}:{int(position)}")
        if not candidates:
            return None

        effect = str(effect_allele).upper()
        if other_allele is None or (
            isinstance(other_allele, float) and pd.isna(other_allele)
        ):
            other = ""
        else:
            other = str(other_allele).upper()

        for use_complement in (False, True):
            eff = _complement(effect) if use_complement else effect
            oth = _complement(other) if (use_complement and other) else other
            for idx in candidates:
                ref = self._ref[idx]
                alt = self._alt[idx]
                if eff == alt and (oth == ref or oth == ""):
                    return idx, False
                if eff == ref and (oth == alt or oth == ""):
                    return idx, True

        return None
