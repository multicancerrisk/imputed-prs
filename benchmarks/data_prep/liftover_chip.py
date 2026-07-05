"""Lift the packaged 23andMe v5 chip positions GRCh37 -> GRCh38.

The packaged manifest (``imputed_prs/data/platforms/23andme_v5_variants.txt.gz``) is
635,991 ``chr:pos`` entries on **GRCh37**, matched positionally by the library with no
build validation. Fitting against a GRCh38 reference therefore needs GRCh38 chip
positions, or PRS variants silently fall into "missing". This script lifts them with
``pyliftover`` (UCSC hg19->hg38 chain, auto-downloaded on first use) into a
benchmark-local artifact; the packaged GRCh37 manifest is left untouched.

Usage:  python -m benchmarks.data_prep.liftover_chip [--chip PATH] [--out PATH] [--chain PATH]

pyliftover uses 0-based coordinates (like UCSC liftOver / BED); chip and VCF positions are
1-based, so we convert 1->0 on input and 0->1 on output.
"""
from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path
from typing import Iterator, Optional, Sequence, Tuple

_OUT_DEFAULT = Path(__file__).resolve().parent.parent / "data" / "23andme_v5_GRCh38_variants.txt"


def default_chip_gz() -> Path:
    import imputed_prs

    return (
        Path(imputed_prs.__file__).resolve().parent
        / "data" / "platforms" / "23andme_v5_variants.txt.gz"
    )


def read_chip_positions(path: Path) -> Iterator[Tuple[str, int]]:
    open_fn = gzip.open if str(path).endswith(".gz") else open
    with open_fn(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line or ":" not in line:
                continue
            chrom, _, pos = line.partition(":")
            try:
                yield chrom, int(pos)
            except ValueError:
                continue


def _norm(chrom: str) -> str:
    return chrom[3:] if chrom.lower().startswith("chr") else chrom


def liftover(chip_path: Path, out_path: Path, chain: Optional[Path] = None) -> dict:
    try:
        from pyliftover import LiftOver
    except ImportError:
        raise SystemExit(
            "pyliftover is not installed. Install the bench extra:\n"
            "  .venv/bin/pip install -e '.[bench]'   (or: pip install pyliftover)"
        )
    lo = LiftOver(str(chain)) if chain else LiftOver("hg19", "hg38")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_in = n_out = n_unmapped = n_chrom_switch = 0
    with open(out_path, "w") as w:
        for chrom, pos in read_chip_positions(chip_path):
            n_in += 1
            res = lo.convert_coordinate(f"chr{chrom}", pos - 1)  # pyliftover is 0-based
            if not res:
                n_unmapped += 1
                continue
            new_chrom = _norm(res[0][0])
            if new_chrom != chrom:
                n_chrom_switch += 1  # lifted onto a different contig; drop for a clean panel
                n_unmapped += 1
                continue
            w.write(f"{new_chrom}:{res[0][1] + 1}\n")
            n_out += 1
    return {
        "n_input": n_in,
        "n_lifted": n_out,
        "n_unmapped": n_unmapped,
        "n_chrom_switch": n_chrom_switch,
        "fraction_lifted": (n_out / n_in if n_in else 0.0),
        "out": str(out_path),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chip", type=Path, default=None, help="GRCh37 chip file (default: packaged 23andMe v5)")
    ap.add_argument("--out", type=Path, default=_OUT_DEFAULT)
    ap.add_argument("--chain", type=Path, default=None, help="local hg19->hg38 chain (default: auto-download)")
    args = ap.parse_args(argv)

    chip = args.chip or default_chip_gz()
    if not Path(chip).exists():
        raise SystemExit(f"chip file not found: {chip}")
    print(f"lifting {chip} -> {args.out}")
    stats = liftover(Path(chip), Path(args.out), args.chain)
    print(
        f"lifted {stats['n_lifted']:,}/{stats['n_input']:,} "
        f"({stats['fraction_lifted']:.1%}); unmapped {stats['n_unmapped']:,} "
        f"(chrom-switch {stats['n_chrom_switch']:,})"
    )
    if stats["fraction_lifted"] < 0.95:
        print("WARNING: <95% lifted — check the chain file and input build", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
