"""Tests for region decomposition and projection data types."""

import numpy as np
import pandas as pd
import pytest

from imputed_prs.core.regions import (
    GenomicRegion,
    RegionDecompositionResult,
    merge_variant_windows,
)
from imputed_prs.core.types import (
    ProjectionRegionModel,
    ProjectionTrainingResult,
)


def _make_prs_df(variant_ids, chromosomes, positions):
    """Helper to create a PRS variants DataFrame."""
    return pd.DataFrame({
        "variant_id": variant_ids,
        "chromosome": chromosomes,
        "position": positions,
    })


class TestMergeVariantWindows:
    """Tests for the merge_variant_windows function."""

    def test_single_variant_single_region(self):
        """One variant produces one region spanning [pos-W, pos+W]."""
        df = _make_prs_df(["rs1"], ["1"], [5_000_000])
        result = merge_variant_windows(df, window_size=1_000_000)

        assert result.n_regions == 1
        region = result.regions[0]
        assert region.chromosome == "1"
        assert region.start == 4_000_000
        assert region.end == 6_000_000
        assert region.prs_variant_ids == ["rs1"]

    def test_two_variants_same_chrom_overlapping(self):
        """Two variants with overlapping windows merge into 1 region."""
        # Positions 5M and 6M with W=1M: [4M,6M] and [5M,7M] overlap
        df = _make_prs_df(["rs1", "rs2"], ["1", "1"], [5_000_000, 6_000_000])
        result = merge_variant_windows(df, window_size=1_000_000)

        assert result.n_regions == 1
        region = result.regions[0]
        assert region.start == 4_000_000
        assert region.end == 7_000_000
        assert set(region.prs_variant_ids) == {"rs1", "rs2"}

    def test_two_variants_same_chrom_non_overlapping(self):
        """Two variants far apart produce 2 separate regions."""
        # Positions 5M and 50M with W=1M: [4M,6M] and [49M,51M] don't overlap
        df = _make_prs_df(["rs1", "rs2"], ["1", "1"], [5_000_000, 50_000_000])
        result = merge_variant_windows(df, window_size=1_000_000)

        assert result.n_regions == 2
        assert result.regions[0].prs_variant_ids == ["rs1"]
        assert result.regions[1].prs_variant_ids == ["rs2"]

    def test_multiple_chromosomes(self):
        """Variants on different chromosomes never merge."""
        df = _make_prs_df(
            ["rs1", "rs2", "rs3"],
            ["1", "2", "1"],
            [5_000_000, 5_000_000, 5_500_000],
        )
        result = merge_variant_windows(df, window_size=1_000_000)

        # rs1 and rs3 on chr1 overlap -> 1 region; rs2 on chr2 -> 1 region
        assert result.n_regions == 2
        chr1_regions = [r for r in result.regions if r.chromosome == "1"]
        chr2_regions = [r for r in result.regions if r.chromosome == "2"]
        assert len(chr1_regions) == 1
        assert len(chr2_regions) == 1
        assert set(chr1_regions[0].prs_variant_ids) == {"rs1", "rs3"}

    def test_chain_merge(self):
        """Chain of variants where each overlaps the next: all merge into 1."""
        # Positions 0, 1.5M, 3M with W=1M:
        # [0, 1M] overlaps [0.5M, 2.5M] overlaps [2M, 4M] -> one region [0, 4M]
        df = _make_prs_df(
            ["rs1", "rs2", "rs3"],
            ["1", "1", "1"],
            [0, 1_500_000, 3_000_000],
        )
        result = merge_variant_windows(df, window_size=1_000_000)

        assert result.n_regions == 1
        region = result.regions[0]
        assert region.start == 0
        assert region.end == 4_000_000
        assert len(region.prs_variant_ids) == 3

    def test_variant_ids_tracked_correctly(self):
        """Each region contains exactly the variant IDs whose windows contributed."""
        df = _make_prs_df(
            ["rs1", "rs2", "rs3", "rs4"],
            ["1", "1", "1", "1"],
            [1_000_000, 1_500_000, 10_000_000, 10_200_000],
        )
        result = merge_variant_windows(df, window_size=1_000_000)

        assert result.n_regions == 2
        assert set(result.regions[0].prs_variant_ids) == {"rs1", "rs2"}
        assert set(result.regions[1].prs_variant_ids) == {"rs3", "rs4"}

    def test_variant_indices_tracked_correctly(self):
        """prs_variant_indices match DataFrame row indices."""
        df = _make_prs_df(
            ["rs1", "rs2", "rs3"],
            ["1", "2", "1"],
            [5_000_000, 5_000_000, 5_500_000],
        )
        result = merge_variant_windows(df, window_size=1_000_000)

        # chr1 region has rs1 (index 0) and rs3 (index 2)
        chr1_region = [r for r in result.regions if r.chromosome == "1"][0]
        assert set(chr1_region.prs_variant_indices) == {0, 2}

        # chr2 region has rs2 (index 1)
        chr2_region = [r for r in result.regions if r.chromosome == "2"][0]
        assert chr2_region.prs_variant_indices == [1]

    def test_empty_dataframe(self):
        """Empty input produces 0 regions."""
        df = pd.DataFrame(columns=["variant_id", "chromosome", "position"])
        result = merge_variant_windows(df, window_size=1_000_000)

        assert result.n_regions == 0
        assert result.regions == []
        assert result.n_variants_in_regions == 0
        assert result.variants_per_region == []
        assert result.max_region_span_bp == 0

    def test_position_at_zero(self):
        """Variant at position 0: region start is clamped to 0 (not negative)."""
        df = _make_prs_df(["rs1"], ["1"], [0])
        result = merge_variant_windows(df, window_size=1_000_000)

        assert result.n_regions == 1
        assert result.regions[0].start == 0
        assert result.regions[0].end == 1_000_000

    def test_regions_sorted(self):
        """Regions are returned sorted by (chromosome, start)."""
        df = _make_prs_df(
            ["rs1", "rs2", "rs3", "rs4"],
            ["2", "1", "1", "2"],
            [50_000_000, 90_000_000, 5_000_000, 5_000_000],
        )
        result = merge_variant_windows(df, window_size=1_000_000)

        chroms = [r.chromosome for r in result.regions]
        starts = [r.start for r in result.regions]
        # Should be sorted by chromosome then start
        for i in range(len(result.regions) - 1):
            assert (chroms[i], starts[i]) <= (chroms[i + 1], starts[i + 1])

    def test_window_size_zero(self):
        """Window size 0: each variant is its own region (point intervals don't overlap)."""
        df = _make_prs_df(
            ["rs1", "rs2", "rs3"],
            ["1", "1", "1"],
            [1_000_000, 2_000_000, 3_000_000],
        )
        result = merge_variant_windows(df, window_size=0)

        assert result.n_regions == 3
        for region in result.regions:
            assert len(region.prs_variant_ids) == 1
            assert region.start == region.end

    def test_result_statistics(self):
        """n_regions, n_variants_in_regions, variants_per_region, max_region_span_bp are correct."""
        df = _make_prs_df(
            ["rs1", "rs2", "rs3", "rs4"],
            ["1", "1", "1", "2"],
            [1_000_000, 1_500_000, 50_000_000, 5_000_000],
        )
        result = merge_variant_windows(df, window_size=1_000_000)

        # chr1: rs1+rs2 merge (span=2.5M), rs3 alone (span=2M); chr2: rs4 alone (span=2M)
        assert result.n_regions == 3
        assert result.n_variants_in_regions == 4
        assert sorted(result.variants_per_region) == [1, 1, 2]
        # rs1+rs2 region: [0, 2.5M] -> span = 2.5M
        assert result.max_region_span_bp == 2_500_000

    def test_chromosome_normalization(self):
        """'chr1' and '1' are treated as the same chromosome."""
        df = _make_prs_df(
            ["rs1", "rs2"],
            ["chr1", "1"],
            [5_000_000, 5_500_000],
        )
        result = merge_variant_windows(df, window_size=1_000_000)

        # Both should be on normalized chromosome "1" and merge
        assert result.n_regions == 1
        assert result.regions[0].chromosome == "1"
        assert set(result.regions[0].prs_variant_ids) == {"rs1", "rs2"}


class TestGenomicRegion:
    """Tests for the GenomicRegion dataclass."""

    def test_construction(self):
        """GenomicRegion can be constructed with all required fields."""
        region = GenomicRegion(
            chromosome="1",
            start=1_000_000,
            end=3_000_000,
            prs_variant_ids=["rs1", "rs2"],
            prs_variant_indices=[0, 1],
        )
        assert region.chromosome == "1"
        assert region.start == 1_000_000
        assert region.end == 3_000_000
        assert region.prs_variant_ids == ["rs1", "rs2"]
        assert region.prs_variant_indices == [0, 1]


class TestProjectionRegionModel:
    """Tests for the ProjectionRegionModel dataclass."""

    def test_to_dict(self):
        """to_dict() converts numpy arrays to lists."""
        model = ProjectionRegionModel(
            region_id="chr1:1000000-3000000",
            chromosome="1",
            start=1_000_000,
            end=3_000_000,
            prs_variant_ids=["rs1", "rs2"],
            betas=np.array([0.3, 0.5]),
            predictor_variant_ids=["rs10", "rs11", "rs12"],
            coefficients=np.array([0.1, 0.2, 0.3]),
            intercept=0.05,
            cv_mse=0.01,
            cv_r2=0.85,
            is_intercept_only=False,
            mean_prs_contribution=0.42,
            predictor_allele_frequencies=np.array([0.3, 0.4, 0.25]),
            predictor_chromosomes=["1", "1", "1"],
            predictor_positions=[1_500_000, 2_000_000, 2_500_000],
            predictor_counted_alleles=["A", "C", "G"],
            predictor_other_alleles=["G", "T", "A"],
        )
        d = model.to_dict()

        assert isinstance(d["betas"], list)
        assert isinstance(d["coefficients"], list)
        assert isinstance(d["predictor_allele_frequencies"], list)
        assert d["betas"] == pytest.approx([0.3, 0.5])
        assert d["coefficients"] == pytest.approx([0.1, 0.2, 0.3])
        assert d["predictor_allele_frequencies"] == pytest.approx([0.3, 0.4, 0.25])
        assert d["region_id"] == "chr1:1000000-3000000"
        assert d["is_intercept_only"] is False
        # Predictor allele metadata (index-aligned with coefficients) round-trips.
        assert d["predictor_chromosomes"] == ["1", "1", "1"]
        assert d["predictor_positions"] == [1_500_000, 2_000_000, 2_500_000]
        assert d["predictor_counted_alleles"] == ["A", "C", "G"]
        assert d["predictor_other_alleles"] == ["G", "T", "A"]

    def test_region_id_format(self):
        """region_id follows 'chr{chrom}:{start}-{end}' format."""
        model = ProjectionRegionModel(
            region_id="chr1:1000000-3000000",
            chromosome="1",
            start=1_000_000,
            end=3_000_000,
            prs_variant_ids=["rs1"],
            betas=np.array([0.3]),
            predictor_variant_ids=[],
            coefficients=np.array([]),
            intercept=0.05,
            cv_mse=0.02,
            cv_r2=0.0,
            is_intercept_only=True,
            mean_prs_contribution=0.15,
            predictor_allele_frequencies=np.array([]),
        )
        assert model.region_id == f"chr{model.chromosome}:{model.start}-{model.end}"
        # New predictor allele metadata defaults to empty lists.
        assert model.predictor_chromosomes == []
        assert model.predictor_positions == []
        assert model.predictor_counted_alleles == []
        assert model.predictor_other_alleles == []


class TestProjectionTrainingResult:
    """Tests for the ProjectionTrainingResult dataclass."""

    def test_construction(self):
        """ProjectionTrainingResult can be constructed with all required fields."""
        result = ProjectionTrainingResult(
            region_models={},
            cv_predictions={},
            n_regions_trained=5,
            n_regions_failed=1,
            n_intercept_only=2,
            training_summary={
                "mean_r2": 0.75,
                "median_r2": 0.80,
                "std_r2": 0.15,
            },
        )
        assert result.n_regions_trained == 5
        assert result.n_regions_failed == 1
        assert result.n_intercept_only == 2
        assert result.training_summary["mean_r2"] == 0.75
