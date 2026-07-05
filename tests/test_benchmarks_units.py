"""Hermetic unit tests for the pure benchmark logic (no bcftools, no cyvcf2, no data):
window/position computation, the growth grid, and the oracle comparator."""
from __future__ import annotations

import pandas as pd

from benchmarks.grid import Cell, iter_grid, order_for_growth, sweep_key
from benchmarks.oracle import _stats, compare_oracle, oracle_matches
from benchmarks.prefilter import needed_positions, subsample_samples


def test_needed_positions_window_and_prs():
    prs = pd.DataFrame(
        [{"chromosome": "22", "position": 1_000_000}, {"chromosome": "22", "position": 5_000_000}]
    )
    chip = {"22:1000500", "22:900000", "22:3000000", "22:9000000"}
    nd = needed_positions(prs, chip, window_size=1_000_000)
    pos = nd["22"]
    assert {1_000_000, 5_000_000} <= pos  # PRS positions always included
    assert 1000500 in pos and 900000 in pos  # within +/-1Mb of a PRS variant
    assert 3000000 not in pos and 9000000 not in pos  # outside every window


def test_needed_positions_restrict_chroms():
    prs = pd.DataFrame(
        [{"chromosome": "22", "position": 100}, {"chromosome": "chr21", "position": 200}]
    )
    nd = needed_positions(prs, set(), window_size=1000, restrict_chroms={"22"})
    assert set(nd) == {"22"}  # chr21 excluded; "chr21" normalized to "21"


def test_subsample_samples_deterministic():
    ids = [f"S{i}" for i in range(100)]
    a = subsample_samples(ids, 10, seed=42)
    b = subsample_samples(ids, 10, seed=42)
    assert a == b and len(a) == 10 and set(a) <= set(ids)
    assert subsample_samples(ids, 500) == sorted(ids)  # n >= len -> all


def test_grid_cumulative_and_load_only():
    cells = list(iter_grid(chrom_order=("22", "21", "20"), sample_sizes=(500, 1000),
                           methods=("imputation",)))
    fit = [c for c in cells if c.kind == "fit"]
    load = [c for c in cells if c.kind == "load_only"]
    assert len(fit) == 3 * 2  # 3 cumulative prefixes x 2 sample sizes
    assert len(load) == 3 * 2  # 3 single chroms x 2 sample sizes
    assert any(c.chroms == ("22",) for c in fit)
    assert any(c.chroms == ("22", "21", "20") for c in fit)
    assert all(len(c.chroms) == 1 for c in load)


def test_order_for_growth_puts_load_first_and_grows():
    cells = list(iter_grid(chrom_order=("22", "21"), sample_sizes=(500,), methods=("imputation",)))
    ordered = order_for_growth(cells)
    assert ordered[0].kind == "load_only"
    fit = [c for c in ordered if c.kind == "fit"]
    assert [len(c.chroms) for c in fit] == sorted(len(c.chroms) for c in fit)  # grows


def test_sweep_key_groups_by_kind_method_samples():
    a = Cell("fit", "imputation", ("22",), 500)
    b = Cell("fit", "imputation", ("22", "21"), 500)
    c = Cell("fit", "imputation", ("22",), 1000)
    assert sweep_key(a) == sweep_key(b)  # same sweep (grows in #chroms)
    assert sweep_key(a) != sweep_key(c)  # different sample count


def test_oracle_compare_and_matches():
    a = {"summary": {"n": 10}, "calibration": {"scaling_factor": 1.0}}
    close = {"summary": {"n": 10}, "calibration": {"scaling_factor": 1.0 + 1e-11}}
    far = {"summary": {"n": 10}, "calibration": {"scaling_factor": 1.1}}
    assert oracle_matches(a, close)
    assert not oracle_matches(a, far)
    diff = compare_oracle(a, far)
    assert diff["calibration.scaling_factor"][2] is False


def test_oracle_stats():
    s = _stats([1.0, 2.0, 3.0])
    assert s["n"] == 3 and s["mean"] == 2.0 and s["min"] == 1.0 and s["max"] == 3.0
    assert _stats([])["n"] == 0
