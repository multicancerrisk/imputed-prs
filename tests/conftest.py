"""Shared pytest fixtures for the test suite.

Hosts the only expensive, genuinely shared fixtures — the cyvcf2-fitted reference
models used by the round-trip (``test_round_trip.py``) and consolidated golden
(``test_golden.py``) suites. A single tiny 1000G-style panel is fit once per test
that needs it.

This file is purely **additive**: modules that still define a local fixture of the
same name (``test_round_trip.py``, ``test_api.py``) shadow these conftest versions,
so adding it cannot change their behavior. ``test_golden.py`` defines no locals and
picks these up.
"""

import pandas as pd
import pytest

from imputed_prs import LinearImputationPRS, LinearProjectionPRS

# 20-sample biallelic reference. effect_allele == ALT for every PRS variant (the
# common case); platform is a partial overlap so a fitted model carries both
# observed (rs1-rs3) and imputed/projected (rs4, rs5) terms with real predictors.
_VCF = """##fileformat=VCFv4.2
##contig=<ID=1,length=249250621>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\tS3\tS4\tS5\tS6\tS7\tS8\tS9\tS10\tS11\tS12\tS13\tS14\tS15\tS16\tS17\tS18\tS19\tS20
1\t100000\trs1\tA\tG\t.\t.\t.\tGT\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1
1\t100500\trs2\tC\tT\t.\t.\t.\tGT\t0/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0
1\t101000\trs3\tG\tA\t.\t.\t.\tGT\t1/1\t0/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1\t0/0\t0/1\t1/1
1\t101500\trs4\tT\tC\t.\t.\t.\tGT\t0/0\t1/1\t0/1\t0/0\t1/1\t0/1\t0/0\t1/1\t0/1\t0/0\t1/1\t0/1\t0/0\t1/1\t0/1\t0/0\t1/1\t0/1\t0/0\t1/1
1\t102000\trs5\tG\tC\t.\t.\t.\tGT\t0/1\t0/1\t0/1\t0/0\t0/0\t1/1\t1/1\t0/1\t0/0\t0/1\t0/1\t0/1\t0/0\t0/0\t1/1\t1/1\t0/1\t0/0\t0/1\t0/1
"""

_PRS_DF = pd.DataFrame(
    {
        "variant_id": ["rs1", "rs2", "rs3", "rs4", "rs5"],
        "chromosome": ["1", "1", "1", "1", "1"],
        "position": [100000, 100500, 101000, 101500, 102000],
        "effect_allele": ["G", "T", "A", "C", "C"],
        "other_allele": ["A", "C", "G", "T", "G"],
        "beta": [0.1, -0.05, 0.2, 0.15, -0.1],
    }
)

_PLATFORM = ["rs1", "rs2", "rs3"]


@pytest.fixture
def vcf_file(tmp_path):
    path = tmp_path / "ref.vcf"
    path.write_text(_VCF)
    return path


@pytest.fixture
def fitted_imputation_model(vcf_file):
    pytest.importorskip("cyvcf2")
    model = LinearImputationPRS(
        window_size=500_000, cv_folds=3, tuning_scope="none", verbose=0, random_state=42
    )
    model.fit(
        reference_genotypes=vcf_file,
        prs_definition=_PRS_DF,
        platform_variants=_PLATFORM,
        # Provenance so a JSON/HDF5 round trip exercises a deployable artifact (P1.7).
        genome_build="GRCh37",
        reference_panel_id="1000G_phase3_EUR",
        training_ancestry="EUR",
    )
    return model


@pytest.fixture
def fitted_projection_model(vcf_file):
    pytest.importorskip("cyvcf2")
    model = LinearProjectionPRS(
        window_size=500_000, cv_folds=3, verbose=0, random_state=42
    )
    model.fit(
        reference_genotypes=vcf_file,
        prs_definition=_PRS_DF,
        platform_variants=_PLATFORM,
        # Symmetric with fitted_imputation_model.
        genome_build="GRCh37",
        reference_panel_id="1000G_phase3_EUR",
        training_ancestry="EUR",
    )
    return model
