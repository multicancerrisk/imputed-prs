"""Streaming genotype sources: read a panel in blocks without materializing the
full dosage matrix.

The eager :func:`imputed_prs.io.genotype_loader.load_genotypes` builds the entire
(n_samples x n_variants) dense float32 matrix in RAM, and its VCF path scans the
whole file with no region pushdown. A :class:`GenotypeSource` instead yields
blocks of at most ``block_size`` variants (all samples) via
:meth:`iter_variant_blocks`, optionally tabix-seeked to a ``region``, so peak RAM
is O(n_samples x block_size) regardless of panel size.

VCF dosages are produced by the same ``variant_to_records`` splitter the eager
loader uses, so a streamed block is bit-identical to the eager result on the same
variants. Delivered + verified standalone in Phase 1; the Phase-2
sufficient-statistics pass consumes it (streaming ZᵀZ/ZᵀX accumulation).
``load_genotypes`` remains the eager small-data path.
"""

import abc
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from imputed_prs.core.exceptions import DataLoadError, ValidationError
from imputed_prs.core.harmonizer import normalize_chromosome_array
from imputed_prs.io.genotype_loader import (
    _build_variant_lookup,
    _normalize_chromosome,
    _variant_matches,
    variant_to_records,
)


@dataclass
class VariantBlock:
    """A contiguous chunk of streamed variants for all samples.

    Attributes
    ----------
    variant_info : pd.DataFrame
        Columns ``variant_id, chromosome, position, ref_allele, alt_allele`` --
        the same schema the eager loader produces.
    dosages : np.ndarray
        ``(n_samples x n_block_variants)`` float32 ALT-count dosages (NaN for
        missing), column-aligned to ``variant_info``.
    """

    variant_info: pd.DataFrame
    dosages: np.ndarray

    @property
    def n_variants(self) -> int:
        return self.variant_info.shape[0]


class GenotypeSource(abc.ABC):
    """Abstract chunked reader over a reference panel.

    Implementations never materialize the whole dosage matrix. ``PgenGenotypeSource``
    is the production (500K-sample) backend; ``VcfGenotypeSource`` is the
    verification backend used on the 1000G panels.
    """

    @property
    @abc.abstractmethod
    def sample_ids(self) -> List[str]:
        """Ordered sample IDs (rows of every block's dosage matrix)."""

    @abc.abstractmethod
    def iter_variant_blocks(
        self, region: Optional[str] = None, block_size: int = 4096
    ) -> Iterator[VariantBlock]:
        """Yield ``VariantBlock`` chunks, optionally restricted to ``region``."""

    def iter_sample_chunks(self, *args, **kwargs):
        """Stream the sample dimension (Phase-2 / 500K-sample path).

        Implemented where it is natural (PGEN). For VCF, all samples are read
        per variant, so ``iter_variant_blocks`` is the streaming primitive.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement sample-chunk streaming; "
            "use iter_variant_blocks (the Phase-1 streaming primitive)."
        )


class VcfGenotypeSource(GenotypeSource):
    """Chunked VCF reader using cyvcf2, with tabix region pushdown.

    Parameters
    ----------
    path : str or Path
        Path to a (bgzipped, tabix-indexed for region queries) VCF/BCF.
    samples : list of str, optional
        Subset of samples to read (delegated to cyvcf2).
    dosage_field : str
        ``"auto"`` (DS -> GT -> GP), or an explicit ``"DS"``/``"GT"``/``"GP"``.
    variant_ids : set of str, optional
        Restrict to these rsIDs / ``chr:pos`` keys (Python-side, same matching as
        the eager loader).
    """

    def __init__(
        self,
        path,
        samples: Optional[List[str]] = None,
        dosage_field: str = "auto",
        variant_ids: Optional[Set[str]] = None,
    ):
        self.path = str(path)
        if not Path(self.path).exists():
            raise DataLoadError(f"VCF file not found: {self.path}")
        self._samples = samples
        self._dosage_field = dosage_field
        self._rsid_set, self._chrpos_set = _build_variant_lookup(variant_ids)
        self._filter = bool(self._rsid_set or self._chrpos_set)

        from cyvcf2 import VCF

        vcf = VCF(self.path, samples=samples)
        self._sample_ids = list(vcf.samples)
        # Raw contig names (region queries must use these, e.g. "chr22" not "22").
        self._seqnames = set(vcf.seqnames)
        vcf.close()
        if not self._sample_ids:
            raise ValidationError("VCF file contains no samples")
        # Region pushdown needs a tabix/CSI index; without one we fall back to a
        # whole-file scan filtered to the region's contig (correct, just slower —
        # fine for per-chromosome or small panels; index large multi-contig VCFs).
        self._has_index = (
            Path(self.path + ".tbi").exists() or Path(self.path + ".csi").exists()
        )
        self._warned_no_index = False

    @property
    def sample_ids(self) -> List[str]:
        return self._sample_ids

    @property
    def contigs(self) -> List[str]:
        """Raw contig names in the file (region queries must use these spellings)."""
        return sorted(self._seqnames)

    def _check_region(self, region: str) -> None:
        # cyvcf2 returns an *empty* iterator (no error) if the contig name does
        # not match the file's, so validate up front against seqnames.
        contig = region.split(":", 1)[0]
        if contig not in self._seqnames:
            raise DataLoadError(
                f"Region contig {contig!r} is not in the VCF's contigs. Use the "
                f"file's raw naming (e.g. 'chr22', not '22'). Available example: "
                f"{sorted(self._seqnames)[:3]}"
            )

    def iter_variant_blocks(
        self, region: Optional[str] = None, block_size: int = 4096
    ) -> Iterator[VariantBlock]:
        if block_size <= 0:
            raise ValidationError("block_size must be positive")

        from cyvcf2 import VCF

        # Region pushdown when indexed; otherwise scan the whole file and keep only
        # records on the region's contig (+ span) Python-side.
        scan_region = region
        contig_filter = span_filter = None
        if region is not None:
            self._check_region(region)
            if not self._has_index:
                scan_region = None
                contig_filter, span_filter = _parse_region(region)
                if not self._warned_no_index:
                    warnings.warn(
                        f"VCF {self.path!r} has no tabix/CSI index; streaming falls "
                        f"back to a full-file scan per region (slower). Index it "
                        f"(bgzip + tabix) for large multi-contig panels.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self._warned_no_index = True

        vcf = VCF(self.path, samples=self._samples)
        n = len(self._sample_ids)
        records: List[dict] = []
        columns: List[np.ndarray] = []
        try:
            iterator = vcf(scan_region) if scan_region is not None else vcf
            for variant in iterator:
                if contig_filter is not None:
                    if variant.CHROM != contig_filter:
                        continue
                    if span_filter is not None and not (
                        span_filter[0] <= variant.POS <= span_filter[1]
                    ):
                        continue
                if self._filter:
                    chrom = _normalize_chromosome(variant.CHROM)
                    var_id = variant.ID if variant.ID else f"{chrom}:{variant.POS}"
                    if not _variant_matches(
                        var_id, chrom, variant.POS, self._rsid_set, self._chrpos_set
                    ):
                        continue
                for record, dosage in variant_to_records(
                    variant, self._dosage_field, n
                ):
                    records.append(record)
                    columns.append(dosage)
                    if len(records) >= block_size:
                        yield _make_block(records, columns)
                        records, columns = [], []
            if records:
                yield _make_block(records, columns)
        finally:
            vcf.close()


def _parse_region(region: str) -> Tuple[str, Optional[Tuple[int, int]]]:
    """Split ``contig[:start-end]`` into ``(contig, (lo, hi) | None)``.

    Used for the no-index fallback scan (the raw contig name is matched against
    ``variant.CHROM``). A bare contig yields ``None`` span (whole contig).
    """
    contig, _, span = region.partition(":")
    if not span:
        return contig, None
    start, _, end = span.partition("-")
    lo = int(start) if start else 0
    hi = int(end) if end else (1 << 62)
    return contig, (lo, hi)


def _make_block(records: List[dict], columns: List[np.ndarray]) -> VariantBlock:
    # Same stack+transpose+float32 cast the eager loader applies, so a block is
    # bit-identical to the eager matrix on the same variants.
    variant_info = pd.DataFrame(records)
    dosages = np.vstack(columns).T.astype(np.float32)
    return VariantBlock(variant_info=variant_info, dosages=dosages)


# =============================================================================
# PLINK 2 PGEN backend (production 500K-panel target)
# =============================================================================

# read_dosages / read fill missing entries with -9 (see pgenlib); the rest of
# the codebase uses NaN, so PGEN dosages are converted at read time.
_PGEN_MISSING = -9


def _read_pvar(path: Path) -> pd.DataFrame:
    """Parse a PLINK 2 ``.pvar`` into the standard variant_info schema.

    Handles ``##`` header lines and the ``#CHROM`` column header. Chromosomes
    are normalized (matching the VCF source and the harmonizer), positions are
    int64.
    """
    header: Optional[List[str]] = None
    rows: List[List[str]] = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM") or line.startswith("#chrom"):
                header = line[1:].rstrip("\n").split("\t")
                continue
            if not line.strip():
                continue
            rows.append(line.rstrip("\n").split("\t"))
    if header is None:
        # Headerless .pvar: PLINK 2 guarantees the first five columns.
        header = ["CHROM", "POS", "ID", "REF", "ALT"]
    idx = {name: header.index(name) for name in ("CHROM", "POS", "ID", "REF", "ALT")}
    chrom = [r[idx["CHROM"]] for r in rows]
    return pd.DataFrame({
        "variant_id": [r[idx["ID"]] for r in rows],
        "chromosome": normalize_chromosome_array(chrom),
        "position": np.array([int(r[idx["POS"]]) for r in rows], dtype=np.int64),
        "ref_allele": [r[idx["REF"]] for r in rows],
        "alt_allele": [r[idx["ALT"]] for r in rows],
    })


def _read_psam(path: Path) -> List[str]:
    """Parse sample IDs (IID column) from a PLINK 2 ``.psam``."""
    with open(path) as fh:
        lines = [ln for ln in fh if not ln.startswith("##") and ln.strip()]
    if not lines:
        raise ValidationError(f"{path} contains no samples")
    header = lines[0].lstrip("#").rstrip("\n").split("\t")
    iid = header.index("IID") if "IID" in header else (
        header.index("iid") if "iid" in header else 0
    )
    return [ln.rstrip("\n").split("\t")[iid] for ln in lines[1:]]


class PgenGenotypeSource(GenotypeSource):
    """Chunked PLINK 2 ``.pgen`` reader via ``pgenlib`` (lazy import).

    Variant metadata comes from the companion ``.pvar`` and sample IDs from the
    ``.psam``. Dosages are ALT-allele counts (0-2, NaN missing), matching the VCF
    source. ``region`` is applied on the ``.pvar`` (no genomic index in a
    ``.pgen``): the contig is matched on the *normalized* chromosome.

    Requires the ``scale`` extra (``pip install imputed-prs[scale]``). Currently
    biallelic ``.pgen`` (one ALT per variant); multiallelic ALT selection is a
    Phase-2 extension.
    """

    def __init__(
        self,
        pgen_path,
        pvar_path=None,
        psam_path=None,
        samples: Optional[List[str]] = None,
        variant_ids: Optional[Set[str]] = None,
    ):
        self._pgen_path = str(pgen_path)
        if not Path(self._pgen_path).exists():
            raise DataLoadError(f"PGEN file not found: {self._pgen_path}")
        stem = self._pgen_path[:-5] if self._pgen_path.endswith(".pgen") else self._pgen_path
        pvar = Path(pvar_path) if pvar_path else Path(stem + ".pvar")
        psam = Path(psam_path) if psam_path else Path(stem + ".psam")
        for p in (pvar, psam):
            if not p.exists():
                raise DataLoadError(f"PGEN companion file not found: {p}")

        self._variant_info = _read_pvar(pvar)
        self._all_sample_ids = _read_psam(psam)

        if samples is not None:
            pos = {s: i for i, s in enumerate(self._all_sample_ids)}
            missing = [s for s in samples if s not in pos]
            if missing:
                raise ValidationError(f"Samples not in .psam: {missing[:5]}")
            self._sample_subset = np.array(sorted(pos[s] for s in samples), dtype=np.uint32)
            self._sample_ids = [self._all_sample_ids[i] for i in self._sample_subset]
        else:
            self._sample_subset = None
            self._sample_ids = list(self._all_sample_ids)

        self._rsid_set, self._chrpos_set = _build_variant_lookup(variant_ids)
        self._filter = bool(self._rsid_set or self._chrpos_set)

    @property
    def sample_ids(self) -> List[str]:
        return self._sample_ids

    def _selected_variant_indices(self, region: Optional[str]) -> np.ndarray:
        """Row indices into ``self._variant_info`` passing the region + id filter."""
        info = self._variant_info
        mask = np.ones(len(info), dtype=bool)
        if region is not None:
            contig, _, span = region.partition(":")
            chrom_norm = _normalize_chromosome(contig)
            mask &= (info["chromosome"].to_numpy() == chrom_norm)
            if span:
                start, _, end = span.partition("-")
                pos = info["position"].to_numpy()
                if start:
                    mask &= pos >= int(start)
                if end:
                    mask &= pos <= int(end)
        if self._filter:
            chroms = info["chromosome"].to_numpy()
            positions = info["position"].to_numpy()
            vids = info["variant_id"].to_numpy()
            id_mask = np.array([
                _variant_matches(vids[i], chroms[i], int(positions[i]),
                                 self._rsid_set, self._chrpos_set)
                for i in range(len(info))
            ])
            mask &= id_mask
        return np.nonzero(mask)[0]

    def iter_variant_blocks(
        self, region: Optional[str] = None, block_size: int = 4096
    ) -> Iterator[VariantBlock]:
        if block_size <= 0:
            raise ValidationError("block_size must be positive")
        try:
            import pgenlib
        except ImportError:
            raise DataLoadError(
                "pgenlib is required for PGEN reading. Install the scale extra: "
                "pip install 'imputed-prs[scale]'"
            )

        selected = self._selected_variant_indices(region)
        if selected.size == 0:
            return

        reader = pgenlib.PgenReader(
            self._pgen_path.encode(), sample_subset=self._sample_subset
        )
        try:
            n = len(self._sample_ids)
            col = np.empty(n, dtype=np.float32)
            for start in range(0, len(selected), block_size):
                block_idx = selected[start : start + block_size]
                dosages = np.empty((n, len(block_idx)), dtype=np.float32)
                for j, vidx in enumerate(block_idx):
                    reader.read_dosages(int(vidx), col)
                    dosages[:, j] = col
                # pgenlib marks missing as -9; convert to NaN like the VCF path.
                dosages[dosages < 0] = np.nan
                info = self._variant_info.iloc[block_idx].reset_index(drop=True)
                yield VariantBlock(variant_info=info, dosages=dosages)
        finally:
            reader.close()


class InMemoryGenotypeSource(GenotypeSource):
    """``GenotypeSource`` view over an in-RAM ``GenotypeData`` (optional subsets).

    Yields ``VariantBlock``\\ s by column-slicing the dense dosage matrix — it never
    copies the full (subset) matrix, so peak extra RAM is O(n_sub x block_size). This
    lets cross-validation / sensitivity analysis refit on an in-RAM fold without
    serializing it to a temporary VCF, and lets the streaming fitter run over a fold
    subset (the Phase-6 full-streaming-CV seam).

    On the same variants a streamed block is bit-identical to what the dense path
    reads from the same ``GenotypeData`` (a plain column slice), so a streaming fit
    from here matches a dense fit within the sanctioned streaming deviations
    (mean-imputation, float64 accumulation, ``cv_predictions=None``).

    Correctness details:

    - **No ``contigs`` property** — so ``compute.sufficient_stats._region_for`` falls
      back to the normalized chromosome, and region filtering here is on the
      normalized chromosome (mirrors :class:`PgenGenotypeSource`).
    - **Position-sorted within each region** — the streaming fitter advances a
      position frontier and evicts below it, so an out-of-order region would close
      targets prematurely.

    Parameters
    ----------
    genotype_data : GenotypeData
        In-RAM panel (dense dosage matrix + variant_info + sample_ids).
    sample_indices : np.ndarray, optional
        Row indices selecting a sample subset (e.g. a CV train fold). None → all rows.
    variant_ids : set of str, optional
        Restrict to these variants (rsID or ``"chr:pos"``), same semantics as
        :func:`make_genotype_source`.
    """

    def __init__(self, genotype_data, *, sample_indices=None, variant_ids=None):
        self._gd = genotype_data
        variant_info = genotype_data.variant_info
        if sample_indices is None:
            self._sample_rows = np.arange(genotype_data.n_samples, dtype=np.int64)
        else:
            self._sample_rows = np.asarray(sample_indices, dtype=np.int64)
        self._sample_ids = [genotype_data.sample_ids[i] for i in self._sample_rows]

        self._chrom_norm = normalize_chromosome_array(
            variant_info["chromosome"].to_numpy()
        )
        # Grouping codes (order irrelevant) so lexsort keeps each chromosome's
        # variants contiguous and position-ascending.
        self._chrom_codes = pd.factorize(self._chrom_norm)[0]
        self._positions = variant_info["position"].to_numpy()
        self._vids = variant_info["variant_id"].to_numpy()
        self._rsid_set, self._chrpos_set = _build_variant_lookup(variant_ids)
        self._filter = bool(self._rsid_set or self._chrpos_set)

    @property
    def sample_ids(self) -> List[str]:
        return self._sample_ids

    def _selected_variant_indices(self, region: Optional[str]) -> np.ndarray:
        """Row indices passing the region + id filter, position-sorted per chromosome."""
        n = len(self._positions)
        mask = np.ones(n, dtype=bool)
        if region is not None:
            contig, _, span = region.partition(":")
            chrom_norm = _normalize_chromosome(contig)
            mask &= self._chrom_norm == chrom_norm
            if span:
                start, _, end = span.partition("-")
                if start:
                    mask &= self._positions >= int(start)
                if end:
                    mask &= self._positions <= int(end)
        if self._filter:
            id_mask = np.array(
                [
                    _variant_matches(
                        self._vids[i],
                        self._chrom_norm[i],
                        int(self._positions[i]),
                        self._rsid_set,
                        self._chrpos_set,
                    )
                    for i in range(n)
                ]
            )
            mask &= id_mask
        selected = np.nonzero(mask)[0]
        if selected.size == 0:
            return selected
        order = np.lexsort(
            (self._positions[selected], self._chrom_codes[selected])
        )
        return selected[order]

    def iter_variant_blocks(
        self, region: Optional[str] = None, block_size: int = 4096
    ) -> Iterator[VariantBlock]:
        if block_size <= 0:
            raise ValidationError("block_size must be positive")
        selected = self._selected_variant_indices(region)
        if selected.size == 0:
            return
        dosage_matrix = self._gd.dosage_matrix
        variant_info = self._gd.variant_info
        rows = self._sample_rows
        for start in range(0, len(selected), block_size):
            cols = selected[start : start + block_size]
            dosages = np.asarray(dosage_matrix[np.ix_(rows, cols)], dtype=np.float32)
            info = variant_info.iloc[cols].reset_index(drop=True)
            yield VariantBlock(variant_info=info, dosages=dosages)


def make_genotype_source(
    path, samples: Optional[List[str]] = None, variant_ids: Optional[Set[str]] = None
) -> GenotypeSource:
    """Construct the right streaming source for ``path`` from its extension.

    ``.pgen`` (or a path with a companion ``.pgen``) → :class:`PgenGenotypeSource`
    (the production 500K-sample backend); ``.vcf/.vcf.gz/.bcf`` →
    :class:`VcfGenotypeSource` (the 1000G verification backend). Mirrors the
    format detection of :func:`io.genotype_loader.load_genotypes` so the streaming
    ``backend`` accepts the same reference paths as the dense one.
    """
    p = str(path)
    low = p.lower()
    if low.endswith(".pgen") or Path(p + ".pgen").exists():
        pgen_path = p if low.endswith(".pgen") else p + ".pgen"
        return PgenGenotypeSource(pgen_path, samples=samples, variant_ids=variant_ids)
    if low.endswith((".vcf", ".vcf.gz", ".vcf.bgz", ".bcf")):
        return VcfGenotypeSource(p, samples=samples, variant_ids=variant_ids)
    raise DataLoadError(
        f"Cannot build a streaming GenotypeSource for {p!r}: expected a "
        f".vcf/.vcf.gz/.bcf or .pgen path. Use backend='dense' for other formats "
        f"(e.g. PLINK1 .bed)."
    )
