"""Growth-grid definition for the blocker demonstration.

Two axes:

* **variants** — cumulative chromosome prefixes, smallest-first (22, 22+21, 22+21+20, …),
  so each step grows the problem by one chromosome's worth of PGS + windowed-chip variants.
* **samples** — subsample the panel to {500, 1000, 2000, 3202}.

Plus **load-only** cells (a single full chromosome through ``load_genotypes`` with no fit)
that isolate the dense ``dosage_matrix`` term from the fit-only terms (``Z``/``X``/
``X_full`` and, for imputation, the ``cv_predictions`` dict).

Each cell becomes one isolated ``measure()`` subprocess, so an OOM on a large cell is a
recorded data point rather than a driver crash.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Sequence, Tuple

# Smallest autosomes first so the sweep starts cheap and grows.
DEFAULT_CHROM_ORDER: Tuple[str, ...] = tuple(str(c) for c in (22, 21, 20, 19, 18, 17, 16, 15, 14,
                                                              13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1))
DEFAULT_SAMPLE_SIZES: Tuple[int, ...] = (500, 1000, 2000, 3202)
DEFAULT_METHODS: Tuple[str, ...] = ("imputation", "projection")


@dataclass(frozen=True)
class Cell:
    kind: str  # "fit" | "load_only"
    method: str  # "imputation" | "projection" | "" for load_only
    chroms: Tuple[str, ...]  # cumulative chromosome set for this cell
    n_samples: int
    label: str = field(default="")

    @property
    def chrom_key(self) -> str:
        return "+".join(self.chroms)


def _cumulative(chrom_order: Sequence[str]) -> List[Tuple[str, ...]]:
    return [tuple(chrom_order[: i + 1]) for i in range(len(chrom_order))]


def iter_grid(
    chrom_order: Sequence[str] = DEFAULT_CHROM_ORDER,
    sample_sizes: Sequence[int] = DEFAULT_SAMPLE_SIZES,
    methods: Sequence[str] = DEFAULT_METHODS,
    *,
    include_load_only: bool = True,
) -> Iterator[Cell]:
    prefixes = _cumulative(chrom_order)

    # load-only isolation: each single chromosome at each sample size
    if include_load_only:
        for chrom in chrom_order:
            for n in sample_sizes:
                yield Cell(
                    kind="load_only",
                    method="",
                    chroms=(chrom,),
                    n_samples=n,
                    label=f"load_chr{chrom}_s{n}",
                )

    # fit cells: cumulative chromosome prefixes x sample sizes x methods
    for method in methods:
        for prefix in prefixes:
            for n in sample_sizes:
                yield Cell(
                    kind="fit",
                    method=method,
                    chroms=prefix,
                    n_samples=n,
                    label=f"fit_{method}_{len(prefix)}chr_s{n}",
                )


def order_for_growth(cells: Sequence[Cell]) -> List[Cell]:
    """Order cells so a sweep grows gently: by kind, then sample size, then #chromosomes.

    This makes the OOM knee appear as the sweep advances, and lets the driver skip larger
    cells on the same (kind, method, n_samples) sweep once one fails.
    """
    return sorted(
        cells,
        key=lambda c: (c.kind != "load_only", c.method, c.n_samples, len(c.chroms)),
    )


def sweep_key(cell: Cell) -> Tuple[str, str, int]:
    """Identifies the monotone sweep a cell belongs to (grows only in #chromosomes)."""
    return (cell.kind, cell.method, cell.n_samples)
