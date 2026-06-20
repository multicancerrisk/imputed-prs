"""User genotype loading for DTC genetic testing files.

This module provides functions for loading user genotype data from direct-to-consumer
(DTC) genetic testing files (23andMe, AncestryDNA, etc.) and converting them to
dosage values for PRS calculation.

Supported input formats:
- File paths: DTC format auto-detected via snps package
- SNPs objects: Direct usage of pre-loaded snps.SNPs instances
- DataFrames: Must have 'rsid' index or column and 'genotype' column
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd

from imputed_prs.core.exceptions import DataLoadError, ValidationError
from imputed_prs.core.harmonizer import (
    _complement,
    _is_ambiguous_snp,
    _normalize_chromosome,
)
from imputed_prs.core.types import VariantIdentity

if TYPE_CHECKING:
    from snps import SNPs

logger = logging.getLogger(__name__)

# Tokens that indicate a missing / no-call genotype.
MISSING_TOKENS = ("--", "", "NA", "N/A", "NULL", "..", "NN", "00")

# Valid single-nucleotide bases. Membership is tested against this set rather
# than the literal "ACGT" string: ``"" in "ACGT"`` and ``"CG" in "ACGT"`` are
# both True (substring semantics) and would wrongly accept empty/multi-base
# allele arguments.
_VALID_BASES = frozenset({"A", "C", "G", "T"})


def load_user_genotypes(
    input_data: Union[str, Path, "SNPs", pd.DataFrame],
    expected_variants: Optional[Set[str]] = None,
) -> Dict[str, Optional[float]]:
    """Load user genotypes and convert to dosage values.

    Supports multiple input types:
    - File path: DTC format auto-detected via snps package (23andMe, AncestryDNA, etc.)
    - SNPs object: Direct usage of pre-loaded snps.SNPs instance
    - DataFrame: Must have 'rsid' index or column and 'genotype' column

    Args:
        input_data: User genotype data (file path, SNPs object, or DataFrame).
        expected_variants: Optional set of variant IDs to extract. If None,
            returns all variants found in the input.

    Returns:
        Dictionary mapping variant_id to dosage value (0.0, 1.0, 2.0) or None
        for missing/invalid genotypes.

    Raises:
        DataLoadError: If file cannot be loaded or parsed.
        ValidationError: If input data format is invalid.
    """
    # Import SNPs here to check instance type without requiring it at module load
    try:
        from snps import SNPs
    except ImportError:
        SNPs = None

    if isinstance(input_data, (str, Path)):
        return _load_from_file(input_data, expected_variants)
    elif SNPs is not None and isinstance(input_data, SNPs):
        return _load_from_snps(input_data, expected_variants)
    elif isinstance(input_data, pd.DataFrame):
        return _load_from_dataframe(input_data, expected_variants)
    else:
        raise ValidationError(f"Unsupported input type: {type(input_data)}")


def genotype_to_dosage(genotype: str) -> Optional[float]:
    """Convert genotype string to dosage value.

    Counts alleles to determine dosage:
    - Homozygous (AA, GG, etc.) -> 2.0
    - Heterozygous (AG, CT, etc.) -> 1.0
    - Missing/invalid -> None

    .. deprecated::
        This function is **allele-blind**: it counts homozygosity without knowing
        which allele is the effect allele, so multiplying its output by a PRS beta
        is only correct when the user genotype happens to be oriented to the effect
        allele. Use :func:`count_allele` for oriented scoring. Retained for
        backward compatibility.

    Note: For PRS calculation, the effect allele direction is determined by
    the PRS weights, not the raw genotypes. This function simply counts
    alleles without assuming which is the effect allele.

    Args:
        genotype: Genotype string (e.g., "AA", "AG", "GG", "--", "I", "D")

    Returns:
        Dosage value (0.0, 1.0, 2.0) or None if missing/invalid.
    """
    if genotype is None or pd.isna(genotype):
        return None

    genotype = str(genotype).strip().upper()

    # Missing data indicators
    if genotype in ("--", "", "NA", "N/A", "NULL", "..", "NN", "00"):
        return None

    # Single character (haploid or indel)
    if len(genotype) == 1:
        if genotype in ("I", "D", "N", "-", "0"):
            return None
        # Single allele - treat as missing (conservative approach)
        return None

    # Standard diploid genotype (2 characters)
    if len(genotype) == 2:
        allele1, allele2 = genotype[0], genotype[1]

        # Check for missing alleles
        if allele1 in ("-", "N", "0") or allele2 in ("-", "N", "0"):
            return None

        # Indel genotypes
        if allele1 in ("I", "D") or allele2 in ("I", "D"):
            return None

        # Count matching alleles (homozygous = 2, heterozygous = 1)
        if allele1 == allele2:
            return 2.0  # Homozygous
        else:
            return 1.0  # Heterozygous

    return None  # Unrecognized format


def _parse_snp_genotype(genotype: object) -> Optional[List[str]]:
    """Parse a diploid SNP genotype string into its two alleles.

    Normalizes case/whitespace, strips ``/`` and ``|`` separators (so ``"A/G"``
    and ``"A|G"`` parse like ``"AG"``), and rejects missing, haploid, indel, and
    non-SNP tokens. Returns the two alleles as a *list* (order preserved, so the
    caller can count multiplicity), or ``None`` when the genotype is not a clean
    A/C/G/T diploid call.

    Args:
        genotype: Raw genotype cell (str, NaN, numeric, etc.).

    Returns:
        ``[allele1, allele2]`` with both in {A, C, G, T}, or ``None``.
    """
    if genotype is None:
        return None
    # Guard NaN without calling pd.isna on non-scalars (it returns an array and
    # would raise in a boolean context); mirrors harmonizer.match_oriented_dosage.
    if isinstance(genotype, float) and pd.isna(genotype):
        return None

    s = str(genotype).strip().upper().replace("/", "").replace("|", "")

    if s in MISSING_TOKENS:
        return None
    if len(s) != 2 or any(base not in _VALID_BASES for base in s):
        return None

    return [s[0], s[1]]


def count_allele(
    genotype: object,
    counted_allele: str,
    other_allele: str,
    *,
    allow_ambiguous: bool,
    allow_strand_flip: bool,
) -> Optional[float]:
    """Count copies of a named allele in a genotype string.

    This is the oriented-scoring primitive: it mirrors the training-side
    :func:`imputed_prs.core.harmonizer.match_oriented_dosage` semantics on raw
    genotype strings instead of a reference dosage matrix. The genotype is only
    counted when its alleles are a subset of the declared ``{counted, other}``
    pair; a partial overlap (e.g. ``"AC"`` against ``{A, G}``) is *unresolved*
    (``None``), never silently scored as a single copy.

    Args:
        genotype: Raw user genotype string (e.g. ``"AG"``, ``"A/G"``, ``"GG"``).
        counted_allele: Allele whose copies are counted (role-specific: the
            effect allele for observed terms, the ALT allele for predictors).
        other_allele: The complementary allele of the biallelic pair. Always
            required (browser-safe scoring depends on it).
        allow_ambiguous: If False, palindromic loci (A/T, C/G) return ``None``
            because their strand cannot be resolved from alleles alone.
        allow_strand_flip: If True and the genotype does not match the pair
            directly, retry once on the complementary strand.

    Returns:
        ``0.0`` / ``1.0`` / ``2.0`` copies of ``counted_allele``, or ``None`` if
        the genotype is missing/invalid or cannot be resolved against the pair.
    """
    alleles = _parse_snp_genotype(genotype)
    if alleles is None:
        return None

    counted = str(counted_allele).strip().upper()
    other = str(other_allele).strip().upper()
    if counted not in _VALID_BASES or other not in _VALID_BASES:
        return None
    if counted == other:
        # Degenerate pair: not a valid biallelic locus. (_is_ambiguous_snp does
        # not catch this — e.g. A/A is not "ambiguous".)
        return None
    if _is_ambiguous_snp(counted, other) and not allow_ambiguous:
        return None

    pair = {counted, other}
    if set(alleles) <= pair:
        return float(alleles.count(counted))

    if allow_strand_flip:
        complemented = [_complement(allele) for allele in alleles]
        if set(complemented) <= pair:
            return float(complemented.count(counted))

    return None


def render_genotype_string(
    ref_allele: object,
    alt_allele: object,
    dosage: object,
    *,
    tol: float = 1e-9,
) -> Optional[str]:
    """Render a diploid genotype string from a biallelic locus and an ALT dosage.

    The inverse of :func:`count_allele` for hard-called (integer) dosages: given a
    reference row's ``ref``/``alt`` alleles and an ALT-allele count ``dosage`` in
    ``{0, 1, 2}``, return the two-allele genotype string (``ref+ref`` / ``ref+alt``
    / ``alt+alt``). It lets the evaluators replay the browser string scorer over a
    hard-called reference panel (P1.6), so the evaluation path is literally the
    upload path.

    The rendered string is *raw* (ALT-counted), never effect-oriented: callers
    count a role-specific allele from it via :func:`count_allele`, which re-orients
    per role. Rendering an already-oriented dosage would double-orient.

    Args:
        ref_allele: Reference allele (single A/C/G/T base).
        alt_allele: Alternate allele (single A/C/G/T base).
        dosage: ALT-allele count; only an integer 0/1/2 (within ``tol``) renders.
        tol: Absolute tolerance for treating ``dosage`` as an integer.

    Returns:
        The genotype string, or ``None`` when the dosage is missing / non-integer /
        out of ``[0, 2]`` or the alleles are not a clean, distinct A/C/G/T pair.
    """
    if dosage is None:
        return None
    try:
        d = float(dosage)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(d):
        return None
    n_alt = round(d)
    if abs(d - n_alt) > tol or n_alt < 0 or n_alt > 2:
        return None

    ref = str(ref_allele).strip().upper()
    alt = str(alt_allele).strip().upper()
    if ref not in _VALID_BASES or alt not in _VALID_BASES or ref == alt:
        return None

    n_alt = int(n_alt)
    return ref * (2 - n_alt) + alt * n_alt


def detect_genome_build(input_data: Union[str, Path, "SNPs"]) -> Optional[int]:
    """Detect genome build from user genotype data.

    Uses the snps package's build detection algorithm which examines
    known SNP positions to determine if the data is GRCh37 (build 37)
    or GRCh38 (build 38).

    Args:
        input_data: File path or SNPs object.

    Returns:
        Build number (37 or 38) or None if not detected.

    Raises:
        DataLoadError: If file cannot be loaded.
    """
    try:
        from snps import SNPs
    except ImportError:
        raise DataLoadError(
            "snps package is required for genome build detection. "
            "Install with: pip install snps"
        )

    if isinstance(input_data, (str, Path)):
        path = Path(input_data)
        if not path.exists():
            raise DataLoadError(f"File not found: {path}")
        try:
            snps_obj = SNPs(str(path))
        except Exception as e:
            raise DataLoadError(f"Failed to parse genotype file: {e}") from e
    else:
        snps_obj = input_data

    return snps_obj.build


def _load_from_file(
    path: Union[str, Path],
    expected_variants: Optional[Set[str]],
) -> Dict[str, Optional[float]]:
    """Load genotypes from a DTC file using the snps package.

    Args:
        path: Path to the genotype file.
        expected_variants: Optional set of variant IDs to extract.

    Returns:
        Dictionary mapping variant_id to dosage value.

    Raises:
        DataLoadError: If file cannot be loaded.
    """
    try:
        from snps import SNPs
    except ImportError:
        raise DataLoadError(
            "snps package is required for loading DTC genotype files. "
            "Install with: pip install snps"
        )

    path = Path(path)
    if not path.exists():
        raise DataLoadError(f"File not found: {path}")

    try:
        snps_obj = SNPs(str(path))
    except Exception as e:
        raise DataLoadError(f"Failed to parse genotype file: {e}") from e

    if snps_obj.snps is None or snps_obj.snps.empty:
        raise DataLoadError(f"No genotype data found in {path}")

    logger.info(
        f"Loaded {len(snps_obj.snps)} variants from {path} "
        f"(source: {snps_obj.source}, build: {snps_obj.build})"
    )

    return _load_from_snps(snps_obj, expected_variants)


def _load_from_snps(
    snps_obj: "SNPs",
    expected_variants: Optional[Set[str]],
) -> Dict[str, Optional[float]]:
    """Load genotypes from a snps.SNPs object.

    Args:
        snps_obj: Pre-loaded SNPs object.
        expected_variants: Optional set of variant IDs to extract.

    Returns:
        Dictionary mapping variant_id to dosage value.
    """
    df = snps_obj.snps  # DataFrame with index=rsid, columns=[chrom, pos, genotype]

    dosages: Dict[str, Optional[float]] = {}

    if expected_variants is not None:
        # Only process expected variants
        for variant_id in expected_variants:
            if variant_id in df.index:
                genotype = df.loc[variant_id, "genotype"]
                dosages[variant_id] = genotype_to_dosage(genotype)
            else:
                dosages[variant_id] = None  # Mark as missing
    else:
        # Process all variants
        for variant_id in df.index:
            genotype = df.loc[variant_id, "genotype"]
            dosages[variant_id] = genotype_to_dosage(genotype)

    return dosages


def _load_from_dataframe(
    df: pd.DataFrame,
    expected_variants: Optional[Set[str]],
) -> Dict[str, Optional[float]]:
    """Load genotypes from a DataFrame.

    Args:
        df: DataFrame with variant IDs as index or in 'rsid'/'variant_id' column,
            and genotypes in 'genotype' column.
        expected_variants: Optional set of variant IDs to extract.

    Returns:
        Dictionary mapping variant_id to dosage value.

    Raises:
        ValidationError: If DataFrame format is invalid.
    """
    # Make a copy to avoid modifying the input
    df = df.copy()

    # Ensure we have a genotype column
    if "genotype" not in df.columns:
        raise ValidationError("DataFrame must have a 'genotype' column")

    # Set index from rsid or variant_id column if not already indexed
    if df.index.name != "rsid" and "rsid" not in str(df.index.dtype):
        if "rsid" in df.columns:
            df = df.set_index("rsid")
        elif "variant_id" in df.columns:
            df = df.set_index("variant_id")
        # Otherwise assume index is already variant IDs

    dosages: Dict[str, Optional[float]] = {}

    if expected_variants is not None:
        for variant_id in expected_variants:
            if variant_id in df.index:
                genotype = df.loc[variant_id, "genotype"]
                dosages[variant_id] = genotype_to_dosage(genotype)
            else:
                dosages[variant_id] = None
    else:
        for variant_id in df.index:
            genotype = df.loc[variant_id, "genotype"]
            dosages[variant_id] = genotype_to_dosage(genotype)

    return dosages


# =============================================================================
# Oriented, multi-key user-genotype representation (P1.1b)
# =============================================================================


@dataclass(frozen=True)
class RawUserGenotype:
    """A single raw genotype call from a user file, before allele orientation.

    Attributes:
        raw_id: The identifier as it appears in the user's file (rsID, etc.).
        chromosome: Chromosome string if available, else None.
        position: Genomic position if available, else None.
        genotype: Raw genotype string (e.g. "AG"); kept verbatim, not collapsed
            to a dosage, so the counted allele can be resolved later.
    """

    raw_id: str
    chromosome: Optional[str]
    position: Optional[int]
    genotype: str


@dataclass(frozen=True)
class GenotypeResolution:
    """Outcome of resolving one variant against a user-genotype collection.

    Attributes:
        genotype: The resolved raw genotype string, or None when unresolved.
        status: One of ``"resolved"``, ``"not_found"``, ``"duplicate_conflict"``.
        matched_id: The user-file id that produced the match, for diagnostics.
    """

    genotype: Optional[str]
    status: str
    matched_id: Optional[str]


class RawUserGenotypeCollection:
    """User genotypes indexed for multi-key (rsID and chr:pos) resolution.

    Built from a user file / DataFrame / SNPs object without collapsing
    genotypes to dosages. Resolution tries every identifier a
    :class:`~imputed_prs.core.types.VariantIdentity` declares plus its chr:pos,
    and refuses to guess when the matched user entries conflict (different
    genotype, or a different locus for the same id).
    """

    def __init__(self, records: List[RawUserGenotype]) -> None:
        self.records: List[RawUserGenotype] = list(records)
        # Multi-maps so duplicate ids/loci are first-class, not silently merged.
        self._by_rsid: Dict[str, List[RawUserGenotype]] = {}
        self._by_chrpos: Dict[str, List[RawUserGenotype]] = {}
        for record in self.records:
            self._by_rsid.setdefault(record.raw_id.lower(), []).append(record)
            if record.chromosome is not None and record.position is not None:
                key = f"{_normalize_chromosome(record.chromosome)}:{record.position}"
                self._by_chrpos.setdefault(key, []).append(record)

    def to_genotype_strings(self) -> Dict[str, str]:
        """Project to a simple ``{raw_id: genotype}`` dict (last write wins).

        Allele-blind and conflict-blind by construction; use :meth:`resolve`
        for oriented, conflict-aware lookup.
        """
        return {record.raw_id: record.genotype for record in self.records}

    def resolve(self, identity: VariantIdentity) -> GenotypeResolution:
        """Resolve one variant's user genotype across all of its identifiers.

        Args:
            identity: The variant to look up; its ``accepted_ids`` and chr:pos
                are all tried.

        Returns:
            A :class:`GenotypeResolution`. ``status`` is ``"duplicate_conflict"``
            when matched user entries disagree on genotype or locus, so a caller
            never scores an arbitrary first match.
        """
        # Dedup by object identity: a record matched via both its rsID and its
        # chr:pos must count once, not look like a conflict.
        matched: Dict[int, RawUserGenotype] = {}

        for accepted_id in identity.accepted_ids:
            aid = str(accepted_id)
            # Classify exactly like harmonizer._build_variant_lookup: an
            # rs-prefixed id is always an rsID, even if it contains a colon.
            if ":" in aid and not aid.lower().startswith("rs"):
                parts = aid.split(":")
                if len(parts) >= 2:
                    key = f"{_normalize_chromosome(parts[0])}:{parts[1]}"
                    for record in self._by_chrpos.get(key, []):
                        matched[id(record)] = record
            else:
                for record in self._by_rsid.get(aid.lower(), []):
                    matched[id(record)] = record

        # Always also try the identity's own chr:pos.
        try:
            chrpos_key: Optional[str] = (
                f"{_normalize_chromosome(str(identity.chromosome))}:"
                f"{int(identity.position)}"
            )
        except (TypeError, ValueError):
            chrpos_key = None
        if chrpos_key is not None:
            for record in self._by_chrpos.get(chrpos_key, []):
                matched[id(record)] = record

        records = list(matched.values())
        if not records:
            return GenotypeResolution(None, "not_found", None)

        # Conflict 1: matched entries disagree on the genotype itself. Compare on
        # the parsed allele multiset so "AG"/"GA"/"A/G" are equal; fall back to
        # the raw string so two distinct unparseable calls still conflict.
        genotype_groups = set()
        for record in records:
            parsed = _parse_snp_genotype(record.genotype)
            if parsed is not None:
                genotype_groups.add(tuple(sorted(parsed)))
            else:
                genotype_groups.add(("__raw__", str(record.genotype).strip().upper()))
        if len(genotype_groups) > 1:
            return GenotypeResolution(None, "duplicate_conflict", None)

        # Conflict 2: the same id resolved to different loci. A missing chr:pos is
        # compatible with any present one, so only present-and-differing conflict.
        chrpos_values = {
            f"{_normalize_chromosome(record.chromosome)}:{record.position}"
            for record in records
            if record.chromosome is not None and record.position is not None
        }
        if len(chrpos_values) > 1:
            return GenotypeResolution(None, "duplicate_conflict", None)

        chosen = records[0]
        return GenotypeResolution(chosen.genotype, "resolved", chosen.raw_id)


def resolve_counted_dosage(
    raw_genotypes: RawUserGenotypeCollection,
    *,
    variant_id: str,
    chromosome: str,
    position: int,
    counted_allele: str,
    other_allele: str,
    allow_ambiguous: bool,
    allow_strand_flip: bool,
) -> Optional[float]:
    """Resolve one locus against the user collection and count the named allele.

    Combines the two primitives every oriented scorer needs: build a multi-key
    :class:`~imputed_prs.core.types.VariantIdentity` (matched by ``variant_id``
    plus ``chr:pos``), resolve it via :meth:`RawUserGenotypeCollection.resolve`,
    then count copies of ``counted_allele`` with :func:`count_allele`. This is the
    resolve→count dance inlined in ``compute_observed_prs_oriented``; predictor
    scorers (imputed and projected) reuse it so a missing/unresolvable predictor
    can be detected (``None``) and mean-substituted by the caller.

    Args:
        raw_genotypes: User genotypes as a multi-key resolvable collection.
        variant_id: Predictor identifier (rsID or source id).
        chromosome: Predictor chromosome; normalized with ``_normalize_chromosome``.
        position: Predictor genomic position.
        counted_allele: Allele whose copies are counted (for predictors, the ALT
            allele the reference ``Z`` column was built from).
        other_allele: The complementary allele of the biallelic pair (REF).
        allow_ambiguous: Whether palindromic (A/T, C/G) loci may be counted.
        allow_strand_flip: Whether to retry on the complementary strand.

    Returns:
        ``0.0`` / ``1.0`` / ``2.0`` copies of ``counted_allele``, or ``None`` when
        the locus is unresolved (not found / duplicate-conflict) or uncountable
        (missing/invalid/partial-overlap/palindromic under the active policy).
    """
    chrom = _normalize_chromosome(str(chromosome))
    identity = VariantIdentity(
        feature_id=f"{chrom}:{position}:{other_allele}:{counted_allele}",
        variant_id=variant_id,
        accepted_ids=(variant_id, f"{chrom}:{position}"),
        chromosome=chrom,
        position=position,
        counted_allele=counted_allele,
        other_allele=other_allele,
    )
    resolution = raw_genotypes.resolve(identity)
    if resolution.status != "resolved":
        # not_found or duplicate_conflict: never count an arbitrary match.
        return None

    return count_allele(
        resolution.genotype,
        counted_allele,
        other_allele,
        allow_ambiguous=allow_ambiguous,
        allow_strand_flip=allow_strand_flip,
    )


def _safe_int(value: object) -> Optional[int]:
    """Best-effort int conversion for a position cell; None when not parseable."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _make_raw_record(
    raw_id: object,
    chromosome: object,
    position: object,
    genotype: object,
) -> RawUserGenotype:
    """Build a RawUserGenotype, normalizing missing chrom/pos/genotype cells."""
    chrom_value: Optional[str] = None
    if chromosome is not None and not (
        isinstance(chromosome, float) and pd.isna(chromosome)
    ):
        chrom_value = str(chromosome)
    if genotype is None or (isinstance(genotype, float) and pd.isna(genotype)):
        genotype_value = ""
    else:
        genotype_value = str(genotype)
    return RawUserGenotype(str(raw_id), chrom_value, _safe_int(position), genotype_value)


def load_user_genotype_strings(
    input_data: Union[str, Path, "SNPs", pd.DataFrame],
    expected_variants: Optional[Set[str]] = None,
) -> Dict[str, str]:
    """Load user genotypes as raw strings keyed by id (no dosage collapse).

    The allele-aware counterpart to :func:`load_user_genotypes`: it preserves the
    genotype strings so the counted allele can be resolved at scoring time via
    :func:`count_allele`. Duplicate ids collapse last-write-wins in the returned
    dict; use :func:`load_raw_user_genotypes` +
    :meth:`RawUserGenotypeCollection.resolve` when duplicate/locus conflicts must
    be detected.

    Args:
        input_data: User genotype data (file path, SNPs object, or DataFrame).
        expected_variants: Optional set of ids to keep. Ids absent from the input
            are omitted (there is no string to return for them).

    Returns:
        Dictionary mapping id to its raw genotype string.
    """
    strings = load_raw_user_genotypes(input_data).to_genotype_strings()
    if expected_variants is not None:
        return {
            variant_id: strings[variant_id]
            for variant_id in expected_variants
            if variant_id in strings
        }
    return strings


def load_raw_user_genotypes(
    input_data: Union[str, Path, "SNPs", pd.DataFrame],
) -> RawUserGenotypeCollection:
    """Load user genotypes into a multi-key resolvable collection.

    Mirrors :func:`load_user_genotypes`' entrypoints (file / SNPs / DataFrame)
    but keeps raw genotype strings plus chr/pos so lookups can match by rsID or
    chr:pos and detect duplicates.

    Args:
        input_data: User genotype data (file path, SNPs object, or DataFrame).

    Returns:
        A :class:`RawUserGenotypeCollection`.

    Raises:
        DataLoadError: If a file cannot be loaded or parsed.
        ValidationError: If the input type/format is invalid.
    """
    try:
        from snps import SNPs
    except ImportError:
        SNPs = None

    if isinstance(input_data, (str, Path)):
        return _raw_from_file(input_data)
    elif SNPs is not None and isinstance(input_data, SNPs):
        return _raw_from_snps(input_data)
    elif isinstance(input_data, pd.DataFrame):
        return _raw_from_dataframe(input_data)
    else:
        raise ValidationError(f"Unsupported input type: {type(input_data)}")


def _raw_from_file(path: Union[str, Path]) -> RawUserGenotypeCollection:
    """Load a raw genotype collection from a DTC file via the snps package."""
    try:
        from snps import SNPs
    except ImportError:
        raise DataLoadError(
            "snps package is required for loading DTC genotype files. "
            "Install with: pip install snps"
        )

    path = Path(path)
    if not path.exists():
        raise DataLoadError(f"File not found: {path}")

    try:
        snps_obj = SNPs(str(path))
    except Exception as e:
        raise DataLoadError(f"Failed to parse genotype file: {e}") from e

    if snps_obj.snps is None or snps_obj.snps.empty:
        raise DataLoadError(f"No genotype data found in {path}")

    return _raw_from_snps(snps_obj)


def _raw_from_snps(snps_obj: "SNPs") -> RawUserGenotypeCollection:
    """Build a raw genotype collection from a snps.SNPs object."""
    df = snps_obj.snps
    has_chrom = "chrom" in df.columns
    has_pos = "pos" in df.columns
    has_genotype = "genotype" in df.columns
    records: List[RawUserGenotype] = []
    for raw_id, row in df.iterrows():
        records.append(
            _make_raw_record(
                raw_id,
                row["chrom"] if has_chrom else None,
                row["pos"] if has_pos else None,
                row["genotype"] if has_genotype else None,
            )
        )
    return RawUserGenotypeCollection(records)


def _raw_from_dataframe(df: pd.DataFrame) -> RawUserGenotypeCollection:
    """Build a raw genotype collection from a DataFrame.

    Iterates rows (never ``df.loc[id]``) so a duplicated index yields multiple
    records instead of a Series, keeping duplicate-conflict detection reachable.
    """
    if "genotype" not in df.columns:
        raise ValidationError("DataFrame must have a 'genotype' column")

    id_col = (
        "rsid"
        if "rsid" in df.columns
        else ("variant_id" if "variant_id" in df.columns else None)
    )
    chrom_col = next((c for c in ("chrom", "chromosome") if c in df.columns), None)
    pos_col = next((c for c in ("pos", "position") if c in df.columns), None)

    records: List[RawUserGenotype] = []
    for index_value, row in df.iterrows():
        raw_id = row[id_col] if id_col is not None else index_value
        records.append(
            _make_raw_record(
                raw_id,
                row[chrom_col] if chrom_col is not None else None,
                row[pos_col] if pos_col is not None else None,
                row["genotype"],
            )
        )
    return RawUserGenotypeCollection(records)
