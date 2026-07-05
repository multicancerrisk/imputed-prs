"""Benchmark and profiling harness for ``imputed-prs`` (Phase 0 of the scaling work).

This package is **dev tooling only** — it is not part of the shipped ``imputed_prs``
wheel (``[tool.setuptools.packages.find]`` includes only ``imputed_prs*``) and it never
modifies or is imported by the library. All measurement is external: subprocess
isolation for peak RSS (``resource.getrusage``), ``tracemalloc`` for per-allocation-site
attribution, ``time.perf_counter`` for wall-clock, and ``cProfile`` for hot-function
attribution.

Public entry points:

- :func:`benchmarks.harness.measure` / :func:`benchmarks.harness.sweep` — run a unit of
  work in an isolated subprocess and record wall-clock + peak RSS + attribution.
- ``python -m benchmarks.run_baseline`` — fetch the scale PGS, capture the baseline
  oracle, and demonstrate the scaling blocker.
- ``python -m benchmarks.scaling_projection`` — fit + extrapolate time/memory to 500K
  samples.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
