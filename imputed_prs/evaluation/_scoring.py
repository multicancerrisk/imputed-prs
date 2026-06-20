"""Shared allele-oriented scoring helpers for the evaluators (P1.6).

Both evaluators route their *predicted*-PRS computation through one of two
allele-oriented paths, selected by the reference dosage mode, so the evaluation
path stays in lock-step with the browser/upload path and they cannot diverge:

- **hard-called** integer dosages (parsed from VCF ``GT``) → render genotype
  strings and replay the browser scorer (``PRSPredictor`` / ``ProjectionPredictor``
  with ``raw_genotypes``). The evaluation path is then *literally* the upload path.
- **continuous** DS/GP dosages → a role-aware numeric scorer that orients each
  predictor via :func:`match_oriented_dosage` using the stored counted/other
  alleles (no string can be rendered from a fractional dosage).

The two paths agree on integer biallelic data — locked by the golden test in
``tests/test_round_trip.py``. ``match_oriented_dosage`` and ``count_allele`` apply
the same effective orientation under the orchestrator policy
(``allow_ambiguous=True, allow_strand_flip=True``), which is also the
``PRSPredictor`` / ``ProjectionPredictor`` constructor default.
"""

from typing import Iterator, List, Sequence, Tuple

import numpy as np

from imputed_prs.core.harmonizer import (
    build_reference_allele_index,
    match_oriented_dosage,
)
from imputed_prs.core.types import GenotypeData, VariantInfo
from imputed_prs.io.user_genotypes import (
    RawUserGenotype,
    RawUserGenotypeCollection,
    render_genotype_string,
)

# A variant the scorer must resolve: (variant_id, chromosome, position, allele_a,
# allele_b). The allele pair only selects the reference row; orientation/counting
# happens later (count_allele for the string path, the dosage flip in the numeric
# path), so either role's (counted/effect, other) pair works here.
NeededVariant = Tuple[str, str, int, str, str]

# Absolute tolerance for treating a dosage as an integer (hard call).
_INT_DOSAGE_TOL = 1e-9


def is_hard_called(dosage_matrix: np.ndarray, *, tol: float = _INT_DOSAGE_TOL) -> bool:
    """Return True iff every non-NaN dosage is an integer in ``{0, 1, 2}``.

    Hard-called reference data is scored by rendering genotype strings and
    replaying the browser scorer; continuous DS/GP data is scored numerically. An
    empty or all-NaN matrix is treated as hard-called — there is no continuous
    evidence, and the render path simply produces no-calls.
    """
    finite = dosage_matrix[~np.isnan(dosage_matrix)]
    if finite.size == 0:
        return True
    rounded = np.round(finite)
    return bool(
        np.all(np.abs(finite - rounded) <= tol)
        and np.all(rounded >= 0)
        and np.all(rounded <= 2)
    )


def observed_component_numeric(
    genotype_data: GenotypeData,
    reference_index: dict,
    observed_variants: Sequence[VariantInfo],
) -> np.ndarray:
    """Effect-allele-oriented observed PRS contribution over all samples.

    Identical to the loop both evaluators already used for the observed
    component; kept here so the numeric path shares one implementation.
    """
    n_samples = genotype_data.n_samples
    observed_prs = np.zeros(n_samples)
    for var in observed_variants:
        match = match_oriented_dosage(
            var.chromosome,
            var.position,
            var.effect_allele,
            var.other_allele,
            genotype_data.variant_info,
            genotype_data.dosage_matrix,
            reference_index,
        )
        if match is None:
            continue
        # Compute in float64: the reference matrix is float32, and a float32
        # dosage * beta loses ~1e-7 precision (NEP-50 weak-scalar promotion),
        # which would split the numeric path from the float64 string path.
        dosages = np.asarray(match[1], dtype=np.float64)
        valid_mask = ~np.isnan(dosages)
        observed_prs[valid_mask] += dosages[valid_mask] * var.beta
    return observed_prs


def oriented_predictor_matrix(
    genotype_data: GenotypeData,
    reference_index: dict,
    chromosomes: Sequence[str],
    positions: Sequence[int],
    counted_alleles: Sequence[str],
    other_alleles: Sequence[str],
    allele_frequencies: np.ndarray,
) -> np.ndarray:
    """Build an ``(n_samples, n_predictors)`` oriented predictor dosage matrix.

    Column ``i`` counts copies of ``counted_alleles[i]`` (the stored ALT allele
    the reference ``Z`` column was built from) via :func:`match_oriented_dosage`.
    Samples with a NaN dosage, and predictors with no allele-compatible reference
    row, fall back to the population mean dosage ``2 * allele_frequencies[i]`` —
    mirroring the per-predictor mean-substitution in the oriented per-user scorers.
    """
    n_samples = genotype_data.n_samples
    n_pred = len(counted_alleles)
    matrix = np.empty((n_samples, n_pred), dtype=np.float64)
    for i in range(n_pred):
        mean_dosage = 2.0 * float(allele_frequencies[i])
        match = match_oriented_dosage(
            chromosomes[i],
            positions[i],
            counted_alleles[i],
            other_alleles[i],
            genotype_data.variant_info,
            genotype_data.dosage_matrix,
            reference_index,
        )
        if match is None:
            matrix[:, i] = mean_dosage
            continue
        col = np.asarray(match[1], dtype=np.float64).copy()
        col[np.isnan(col)] = mean_dosage
        matrix[:, i] = col
    return matrix


def iter_sample_collections(
    genotype_data: GenotypeData,
    needed: Sequence[NeededVariant],
) -> Iterator[RawUserGenotypeCollection]:
    """Yield one :class:`RawUserGenotypeCollection` per sample (string-replay path).

    For each *needed* variant the exact reference row is selected once via
    :func:`match_oriented_dosage` (the allele pair only picks the row); the row's
    **raw** (ALT-counted) genotype string is rendered per sample from
    ``(ref_allele, alt_allele, dosage)``. Exactly one record per needed variant is
    emitted (keyed by ``variant_id``, at the row's locus): rendering *every*
    reference row would put several genotypes at one ``chr:pos`` and trip the
    collection's duplicate-conflict guard at multi-allelic loci. ``count_allele``
    re-orients per role at scoring time, so the raw string is role-neutral.

    Variants with no allele-compatible reference row are skipped (omitted from the
    collection), so the scorer resolves them as missing — observed → unresolved,
    predictor → mean-substituted — matching the numeric path.
    """
    variant_info = genotype_data.variant_info
    dosage_matrix = genotype_data.dosage_matrix
    n_samples = genotype_data.n_samples
    reference_index = build_reference_allele_index(variant_info)

    # Resolve each needed variant to its reference row once and pre-render its
    # per-sample genotype strings. Dedup by variant_id so a locus that is both an
    # observed term and a predictor yields a single (role-neutral) record.
    rendered: List[Tuple[str, str, int, List]] = []
    seen = set()
    for variant_id, chrom, pos, allele_a, allele_b in needed:
        if variant_id in seen:
            continue
        seen.add(variant_id)
        match = match_oriented_dosage(
            chrom,
            pos,
            allele_a,
            allele_b,
            variant_info,
            dosage_matrix,
            reference_index,
        )
        if match is None:
            continue
        row_idx = match[0]
        row = variant_info.iloc[row_idx]
        ref = str(row["ref_allele"])
        alt = str(row["alt_allele"])
        dosage_col = dosage_matrix[:, row_idx]
        strings = [render_genotype_string(ref, alt, d) for d in dosage_col]
        rendered.append((variant_id, chrom, int(pos), strings))

    for sample_idx in range(n_samples):
        records: List[RawUserGenotype] = []
        for variant_id, chrom, pos, strings in rendered:
            geno = strings[sample_idx]
            if geno is None:
                continue
            records.append(RawUserGenotype(variant_id, chrom, pos, geno))
        yield RawUserGenotypeCollection(records)
