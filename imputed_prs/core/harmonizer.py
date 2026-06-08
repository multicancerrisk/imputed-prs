"""Variant harmonization between PRS definitions, platforms, and reference data.

This module provides functions to harmonize variants across different data sources:
- Partition PRS variants into observed (on platform) and missing (need imputation) sets
- Align effect alleles between PRS and genotype data
- Validate genome build compatibility
- Filter variants to local genomic windows
"""

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from imputed_prs.core.exceptions import IncompatibleBuildError
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


def _normalize_chromosome(chrom: str) -> str:
    """Normalize chromosome name.

    Strips 'chr' prefix and normalizes sex/mitochondrial chromosomes.

    Parameters
    ----------
    chrom : str
        Chromosome string (e.g., "chr1", "CHR22", "chrX").

    Returns
    -------
    str
        Normalized chromosome (e.g., "1", "22", "X").
    """
    chrom = str(chrom).upper()
    # Strip chr prefix
    if chrom.startswith("CHR"):
        chrom = chrom[3:]
    # Normalize common aliases
    if chrom == "MT":
        chrom = "M"
    return chrom


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
    index: Dict[str, List[int]] = {}
    for pos_idx, (_, row) in enumerate(variant_info.iterrows()):
        chrom = _normalize_chromosome(str(row["chromosome"]))
        pos = int(row["position"])
        index.setdefault(f"{chrom}:{pos}", []).append(pos_idx)
    return index


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
