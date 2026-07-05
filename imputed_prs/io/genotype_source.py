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
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Set

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

    @property
    def sample_ids(self) -> List[str]:
        return self._sample_ids

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

        if region is not None:
            self._check_region(region)

        vcf = VCF(self.path, samples=self._samples)
        n = len(self._sample_ids)
        records: List[dict] = []
        columns: List[np.ndarray] = []
        try:
            iterator = vcf(region) if region is not None else vcf
            for variant in iterator:
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
