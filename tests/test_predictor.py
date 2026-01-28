"""Tests for the predictor module."""

from imputed_prs.core.types import VariantInfo
from imputed_prs.models.predictor import compute_observed_prs


class TestComputeObservedPrs:
    """Tests for compute_observed_prs function."""

    def test_basic_calculation(self):
        """Basic calculation with known values."""
        observed = [
            VariantInfo("rs1", "1", 100, "A", "G", 0.1),
            VariantInfo("rs2", "1", 200, "C", "T", 0.2),
        ]
        dosages = {"rs1": 2.0, "rs2": 1.0}

        prs, n_used = compute_observed_prs(dosages, observed)

        # PRS = 2.0 * 0.1 + 1.0 * 0.2 = 0.2 + 0.2 = 0.4
        assert abs(prs - 0.4) < 1e-10
        assert n_used == 2

    def test_missing_variants_none_dosages(self):
        """Handle missing variants (None dosages)."""
        observed = [
            VariantInfo("rs1", "1", 100, "A", "G", 0.1),
            VariantInfo("rs2", "1", 200, "C", "T", 0.2),
        ]
        dosages = {"rs1": 2.0, "rs2": None}

        prs, n_used = compute_observed_prs(dosages, observed)

        # PRS = 2.0 * 0.1 = 0.2 (rs2 is skipped)
        assert abs(prs - 0.2) < 1e-10
        assert n_used == 1

    def test_empty_observed_variants(self):
        """Empty observed variants list returns zero."""
        dosages = {"rs1": 2.0, "rs2": 1.0}

        prs, n_used = compute_observed_prs(dosages, [])

        assert prs == 0.0
        assert n_used == 0

    def test_mixed_valid_missing_dosages(self):
        """Mixed valid and missing dosages."""
        observed = [
            VariantInfo("rs1", "1", 100, "A", "G", 0.1),
            VariantInfo("rs2", "1", 200, "C", "T", 0.2),
            VariantInfo("rs3", "1", 300, "G", "A", 0.3),
            VariantInfo("rs4", "1", 400, "T", "C", 0.4),
        ]
        dosages = {
            "rs1": 1.0,
            "rs2": None,
            "rs3": 2.0,
            # rs4 not in dosages dict
        }

        prs, n_used = compute_observed_prs(dosages, observed)

        # PRS = 1.0 * 0.1 + 2.0 * 0.3 = 0.1 + 0.6 = 0.7
        assert abs(prs - 0.7) < 1e-10
        assert n_used == 2

    def test_negative_betas(self):
        """Handle negative beta values."""
        observed = [
            VariantInfo("rs1", "1", 100, "A", "G", 0.1),
            VariantInfo("rs2", "1", 200, "C", "T", -0.05),
        ]
        dosages = {"rs1": 2.0, "rs2": 1.0}

        prs, n_used = compute_observed_prs(dosages, observed)

        # PRS = 2.0 * 0.1 + 1.0 * (-0.05) = 0.2 - 0.05 = 0.15
        assert abs(prs - 0.15) < 1e-10
        assert n_used == 2

    def test_zero_dosage(self):
        """Handle zero dosage values."""
        observed = [
            VariantInfo("rs1", "1", 100, "A", "G", 0.5),
            VariantInfo("rs2", "1", 200, "C", "T", 0.3),
        ]
        dosages = {"rs1": 0.0, "rs2": 2.0}

        prs, n_used = compute_observed_prs(dosages, observed)

        # PRS = 0.0 * 0.5 + 2.0 * 0.3 = 0.0 + 0.6 = 0.6
        assert abs(prs - 0.6) < 1e-10
        assert n_used == 2

    def test_all_missing_dosages(self):
        """All variants have missing dosages."""
        observed = [
            VariantInfo("rs1", "1", 100, "A", "G", 0.1),
            VariantInfo("rs2", "1", 200, "C", "T", 0.2),
        ]
        dosages = {"rs1": None, "rs2": None}

        prs, n_used = compute_observed_prs(dosages, observed)

        assert prs == 0.0
        assert n_used == 0

    def test_empty_dosages_dict(self):
        """Empty dosages dictionary."""
        observed = [
            VariantInfo("rs1", "1", 100, "A", "G", 0.1),
        ]
        dosages = {}

        prs, n_used = compute_observed_prs(dosages, observed)

        assert prs == 0.0
        assert n_used == 0

    def test_fractional_dosages(self):
        """Handle fractional dosage values (imputed-like)."""
        observed = [
            VariantInfo("rs1", "1", 100, "A", "G", 0.1),
            VariantInfo("rs2", "1", 200, "C", "T", 0.2),
        ]
        dosages = {"rs1": 1.5, "rs2": 0.7}

        prs, n_used = compute_observed_prs(dosages, observed)

        # PRS = 1.5 * 0.1 + 0.7 * 0.2 = 0.15 + 0.14 = 0.29
        assert abs(prs - 0.29) < 1e-10
        assert n_used == 2
