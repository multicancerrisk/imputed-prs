"""Shared allele-oriented scoring helpers for the evaluators (P1.6, P5).

Both evaluators compute their *predicted*-PRS through the role-aware **numeric**
scorer (:func:`observed_component_numeric` + the vectorized/oracle imputation or
projection solve), for **both** hard-called and continuous dosage modes (P5). It
orients each predictor via :func:`match_oriented_dosage` (through
:class:`ReferenceAlleleResolver`) using the stored counted/other alleles, so the
evaluation path applies the same effective orientation as the browser/upload path
under the orchestrator policy (``allow_ambiguous=True, allow_strand_flip=True``,
the ``PRSPredictor`` / ``ProjectionPredictor`` constructor default).

The **string-replay** path (render genotype strings and replay the browser scorer
with ``raw_genotypes``, i.e. *literally* the upload path) is retained on the
evaluators as ``_predicted_prs_via_strings`` but is no longer on the metric path.
On hard-called integer data the numeric path is byte-identical to it — including
**indel / multi-character alleles** and **multiallelic co-predictor loci**, now
that the browser/upload primitives parse, count, and resolve non-SNP alleles as
whole tokens (the structured allele/dosage scorer in
:mod:`imputed_prs.io.user_genotypes`; parity locked by ``TestIndelBrowserParity``
and ``TestMultiallelicCoPredictorParity`` in ``tests/test_round_trip.py``). So the
string replay now serves only as the browser-faithful oracle for those tests.
Before P5 the hard-called metric path replayed strings per sample
(O(samples × variants) pure-Python), which dominated reference-CV wall-clock;
routing it through the numeric scorer removes that cost while keeping metrics
byte-identical.

One documented, benign deviation from the retired string path:

- **P1.8 observed fallback.** :func:`observed_component_numeric` does not recover
  an unresolvable/no-call observed variant from its trained fallback model (a
  loud :class:`UserWarning` fires if it ever could) — see its docstring. Not
  exercised on fully-called panels. The numeric path resolves the reference dosage
  the model was **trained** on (by ``chr:pos`` + alleles, via
  :class:`ReferenceAlleleResolver`), so it is the training-faithful path.
"""

import warnings
from typing import Iterator, List, Sequence, Tuple

import numpy as np

from imputed_prs.core.harmonizer import (
    ReferenceAlleleResolver,
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

# Number of target units (imputed variants / placed variants / regions) at or
# above which the evaluator switches from the exact per-unit oracle loop to the
# vectorized CSR batch path (Phase 4). Below it the oracle runs, keeping the
# golden ``atol=1e-12`` tests on the byte-exact path; the batch path is validated
# at ``atol~1e-9``. Every golden fixture has far fewer than this many units.
_BATCH_MIN_TARGETS = 256


def should_use_batch(n_units: int) -> bool:
    """True when the vectorized batch path should score ``n_units`` targets."""
    return n_units >= _BATCH_MIN_TARGETS


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


def _warn_dropped_fallback(variant_id: str, reason: str) -> None:
    """Loud P1.8 guard: the numeric observed scorer dropped a fallback-eligible
    variant that the string-replay path would have recovered. Unreachable in
    normal evaluator usage (see :func:`observed_component_numeric`); surfaced so
    the deviation is never silent."""
    warnings.warn(
        f"Observed variant {variant_id!r} has a trained fallback model but "
        f"cannot be scored directly ({reason}); the numeric evaluator path drops "
        f"it, whereas the browser/string path would recover it from the fallback "
        f"(P1.8), so evaluator metrics may differ for this variant. Use the "
        f"_predicted_prs_via_strings oracle for exact browser parity.",
        UserWarning,
        stacklevel=3,
    )


def observed_component_numeric(
    genotype_data: GenotypeData,
    resolver: ReferenceAlleleResolver,
    observed_variants: Sequence[VariantInfo],
) -> np.ndarray:
    """Effect-allele-oriented observed PRS contribution over all samples.

    Identical to the loop both evaluators already used for the observed
    component; kept here so the numeric path shares one implementation.
    ``resolver.resolve`` is byte-identical to :func:`match_oriented_dosage` but
    avoids the per-candidate ``variant_info.iloc`` (Phase 4 hotspot fix).

    **P1.8 deviation (documented, not silent).** Unlike the string-replay path
    (:func:`~imputed_prs.models.predictor.compute_observed_prs_oriented`), this
    does not recover an unresolvable or no-call observed variant from its trained
    fallback model: an unresolved variant (``match is None``) is dropped and NaN
    (no-call) samples contribute zero. In evaluator usage this is unreachable —
    observed variants are on the genotyping platform, always retained by
    ``mask_reference_to_platform``, and fully called on hard-called GT — and it is
    already the behavior of the pre-P5 continuous path. Should it ever occur (an
    observed variant carrying a fallback that is unresolved or has any NaN
    sample), a loud :class:`UserWarning` naming the variant is emitted; the
    retained ``_predicted_prs_via_strings`` oracle gives exact browser parity.
    """
    n_samples = genotype_data.n_samples
    observed_prs = np.zeros(n_samples)
    for var in observed_variants:
        match = resolver.resolve(
            var.chromosome,
            var.position,
            var.effect_allele,
            var.other_allele,
            genotype_data.dosage_matrix,
        )
        if match is None:
            if var.fallback is not None:
                _warn_dropped_fallback(var.variant_id, "no allele-compatible reference row")
            continue
        # Compute in float64: the reference matrix is float32, and a float32
        # dosage * beta loses ~1e-7 precision (NEP-50 weak-scalar promotion),
        # which would split the numeric path from the float64 string path.
        dosages = np.asarray(match[1], dtype=np.float64)
        valid_mask = ~np.isnan(dosages)
        if var.fallback is not None and not valid_mask.all():
            _warn_dropped_fallback(var.variant_id, "no-call (NaN) samples")
        observed_prs[valid_mask] += dosages[valid_mask] * var.beta
    return observed_prs


def oriented_predictor_matrix(
    genotype_data: GenotypeData,
    resolver: ReferenceAlleleResolver,
    chromosomes: Sequence[str],
    positions: Sequence[int],
    counted_alleles: Sequence[str],
    other_alleles: Sequence[str],
    allele_frequencies: np.ndarray,
) -> np.ndarray:
    """Build an ``(n_samples, n_predictors)`` oriented predictor dosage matrix.

    Column ``i`` counts copies of ``counted_alleles[i]`` (the stored ALT allele
    the reference ``Z`` column was built from) via ``resolver.resolve`` (byte-
    identical to :func:`match_oriented_dosage`). Samples with a NaN dosage, and
    predictors with no allele-compatible reference row, fall back to the population
    mean dosage ``2 * allele_frequencies[i]`` — mirroring the per-predictor
    mean-substitution in the oriented per-user scorers.
    """
    n_samples = genotype_data.n_samples
    n_pred = len(counted_alleles)
    matrix = np.empty((n_samples, n_pred), dtype=np.float64)
    for i in range(n_pred):
        mean_dosage = 2.0 * float(allele_frequencies[i])
        match = resolver.resolve(
            chromosomes[i],
            positions[i],
            counted_alleles[i],
            other_alleles[i],
            genotype_data.dosage_matrix,
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
