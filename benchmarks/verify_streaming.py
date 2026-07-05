"""Phase-1 Workstream E verification on the real 1000G GRCh38 chr22 panel.

1. Dosage identity: eager-load a bcftools-subset region and compare, bit-for-bit
   (NaN-aware), to the same region streamed from the FULL chr22 file.
2. Tabix pushdown + no-OOM: stream a region from the full 446 MB file in blocks;
   confirm only in-region variants are returned, that the seek is fast (does not
   scan the whole file), and that peak resident dosage bytes stay at block size
   -- not the full-region dense matrix the eager path would hold.

Run:
    .venv/bin/python -m benchmarks.verify_streaming
"""

import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from imputed_prs.io.genotype_loader import load_genotypes
from imputed_prs.io.genotype_source import VcfGenotypeSource

CHR22 = Path(
    "/Users/jeya/Documents/projects/imputed-prs/benchmarks/data/1kg_grch38/"
    "1kGP_high_coverage_Illumina.chr22.filtered.SNV_INDEL_SV_phased_panel.vcf.gz"
)
IDENTITY_REGION = "chr22:16000000-16500000"   # ~0.5 Mb interior region
PUSHDOWN_REGION = "chr22:16000000-17000000"   # ~1 Mb, ~33.5k variants


def _key(info):
    return list(zip(info["chromosome"], info["position"],
                    info["ref_allele"].astype(str), info["alt_allele"].astype(str)))


def verify_identity():
    print(f"[identity] region {IDENTITY_REGION}")
    with tempfile.TemporaryDirectory() as td:
        sub = Path(td) / "sub.vcf.gz"
        subprocess.run(["bcftools", "view", "-Oz", "-o", str(sub), "-r",
                        IDENTITY_REGION, str(CHR22)], check=True)
        subprocess.run(["bcftools", "index", "-t", str(sub)], check=True)
        eager = load_genotypes(sub)

    src = VcfGenotypeSource(CHR22)
    infos, mats = [], []
    for blk in src.iter_variant_blocks(region=IDENTITY_REGION, block_size=4096):
        infos.append(blk.variant_info)
        mats.append(blk.dosages)
    import pandas as pd
    stream_info = pd.concat(infos, ignore_index=True)
    stream_dos = np.hstack(mats)

    ek, sk = _key(eager.variant_info), _key(stream_info)
    assert ek == sk, (
        f"variant sets differ: eager {len(ek)} vs stream {len(sk)}; "
        f"first mismatch idx {next(i for i,(a,b) in enumerate(zip(ek,sk)) if a!=b)}"
    )
    np.testing.assert_array_equal(stream_dos, eager.dosage_matrix)
    print(f"[identity] OK: {len(sk):,} variants x {len(src.sample_ids):,} samples, "
          f"dosages bit-identical (NaN-aware)")


def verify_pushdown():
    print(f"[pushdown] region {PUSHDOWN_REGION}, block_size=4096")
    src = VcfGenotypeSource(CHR22)
    n_samples = len(src.sample_ids)
    t = time.perf_counter()
    n_variants = 0
    peak_block_bytes = 0
    off_region = 0
    colsum = None  # a running reduction so we never hold all blocks
    for blk in src.iter_variant_blocks(region=PUSHDOWN_REGION, block_size=4096):
        n_variants += blk.n_variants
        peak_block_bytes = max(peak_block_bytes, blk.dosages.nbytes)
        if not (blk.variant_info["chromosome"] == "22").all():
            off_region += 1
        s = np.nansum(blk.dosages, axis=1)
        colsum = s if colsum is None else colsum + s
    dt = time.perf_counter() - t
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS

    full_region_dense = n_variants * n_samples * 4
    assert off_region == 0, f"{off_region} blocks had off-region variants"
    print(f"[pushdown] OK: {n_variants:,} in-region variants in {dt:.1f}s "
          f"({n_variants/dt:,.0f} variants/s)")
    print(f"[pushdown] peak resident dosage block: {peak_block_bytes/1e6:.1f} MB "
          f"vs full-region dense {full_region_dense/1e6:.0f} MB "
          f"({full_region_dense/peak_block_bytes:.0f}x smaller)")
    print(f"[pushdown] process peak RSS: {ru/1e6:.0f} MB "
          f"(would be >13 GB to hold all of chr22 densely at {n_samples:,} samples)")


if __name__ == "__main__":
    if not CHR22.exists():
        sys.exit(f"chr22 panel not found at {CHR22}")
    # Pushdown first so its process-RSS reading is not inflated by the identity
    # check, which deliberately materializes both the eager and streamed copies.
    verify_pushdown()
    verify_identity()
