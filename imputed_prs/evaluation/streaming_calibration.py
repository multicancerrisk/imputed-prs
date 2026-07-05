"""Finalize CV calibration from streaming accumulators (Phase 2).

The streaming fitter reduces every unit's out-of-fold prediction directly into two
length-``n`` accumulators (``s_true``, ``s_cv``) instead of building the per-variant
``cv_predictions`` dict (the ~8 EB calibration blocker). This module turns those two
vectors into a ``CalibrationParams`` using the *existing* ``estimate_cv_calibration``
(no new statistics), then injects the diagonal-SE lower bound exactly as the dense
orchestrator does.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Dict, Optional

import numpy as np

from imputed_prs.core.types import CalibrationParams, ImputedVariantModel
from imputed_prs.evaluation.calibration import estimate_cv_calibration


def finalize_imputation_calibration(
    s_true: np.ndarray,
    s_cv: np.ndarray,
    models: Dict[str, ImputedVariantModel],
) -> Optional[CalibrationParams]:
    """Build CalibrationParams from streaming accumulators + trained models.

    Mirrors ``core/linear_imputation_prs.py`` Step 11: regress ``s_true`` on ``s_cv``
    (``estimate_cv_calibration``), then set ``diagonal_model_se_lower_bound`` to
    ``sqrt(Σ_j β_j²·residual_variance_j)`` over the trained missing variants. Returns
    None if calibration cannot be estimated (fewer than 3 usable samples), matching
    the dense path's ``except (ValueError, IndexError) -> None``.
    """
    try:
        params = estimate_cv_calibration(s_cv, s_true)
    except (ValueError, IndexError):
        return None

    diag_var = 0.0
    for model in models.values():
        diag_var += float(model.beta) ** 2 * float(model.residual_variance)
    return dataclasses.replace(
        params, diagonal_model_se_lower_bound=math.sqrt(diag_var)
    )


def finalize_projection_calibration(
    s_true: np.ndarray,
    s_cv: np.ndarray,
    diag_var: float,
) -> Optional[CalibrationParams]:
    """Build CalibrationParams from streaming projection accumulators.

    Mirrors ``core/linear_projection_prs.py`` Step 10: regress ``s_true`` on ``s_cv``
    (``estimate_cv_calibration``), then set ``diagonal_model_se_lower_bound`` to
    ``sqrt(Σ_R cv_mse_R)`` over the trained regions (``diag_var`` is that sum,
    accumulated during the streaming pass). Returns None if calibration cannot be
    estimated, matching the dense path's ``except (ValueError, IndexError) -> None``.
    """
    try:
        params = estimate_cv_calibration(s_cv, s_true)
    except (ValueError, IndexError):
        return None
    return dataclasses.replace(
        params, diagonal_model_se_lower_bound=math.sqrt(max(diag_var, 0.0))
    )
