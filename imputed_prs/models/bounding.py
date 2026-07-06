"""Dosage bounding with variance adjustment for truncated normal distributions.

When imputed dosage predictions are clipped to valid bounds [0, 2], the variance
of the truncated distribution is reduced. This module provides functions to
compute the proper variance adjustment.
"""

from typing import Tuple

import numpy as np
from scipy.stats import norm

_MIN_SIGMA = 1e-10
_MIN_Z = 1e-15


def truncated_normal_variance(
    mu: float,
    sigma: float,
    lower: float = 0.0,
    upper: float = 2.0,
) -> float:
    """Compute variance of a truncated normal distribution.

    For X ~ N(mu, sigma²) truncated to [lower, upper], computes Var(X | lower <= X <= upper).

    The formula uses standardized bounds z_lower = (lower - mu) / sigma and
    z_upper = (upper - mu) / sigma:

        Var = sigma² * [1 + (z_lower·φ(z_lower) - z_upper·φ(z_upper))/Z
              - ((φ(z_lower) - φ(z_upper))/Z)²]

    where φ is the standard normal PDF, Φ is the standard normal CDF,
    and Z = Φ(z_upper) - Φ(z_lower).

    Args:
        mu: Mean of the untruncated normal distribution.
        sigma: Standard deviation of the untruncated normal distribution.
        lower: Lower truncation bound (default 0.0 for dosage).
        upper: Upper truncation bound (default 2.0 for dosage).

    Returns:
        Variance of the truncated distribution. Always non-negative and
        less than or equal to sigma².
    """
    if sigma < _MIN_SIGMA:
        return 0.0

    z_lower = (lower - mu) / sigma
    z_upper = (upper - mu) / sigma

    phi_lo, phi_hi = norm.pdf(z_lower), norm.pdf(z_upper)
    Phi_lo, Phi_hi = norm.cdf(z_lower), norm.cdf(z_upper)

    Z = Phi_hi - Phi_lo
    if Z < _MIN_Z:
        # Nearly all mass is outside bounds - return minimum variance floor
        return _MIN_SIGMA**2

    term1 = (z_lower * phi_lo - z_upper * phi_hi) / Z
    term2 = ((phi_lo - phi_hi) / Z) ** 2

    return max(0.0, sigma**2 * (1 + term1 - term2))


def truncated_normal_variance_array(
    mu: np.ndarray,
    sigma: np.ndarray,
    lower: float = 0.0,
    upper: float = 2.0,
) -> np.ndarray:
    """Vectorized :func:`truncated_normal_variance` over element-aligned arrays.

    Elementwise-identical to the scalar oracle (validated at ``atol=1e-12``). The
    scalar early-returns (``sigma < _MIN_SIGMA`` → 0, ``Z < _MIN_Z`` →
    ``_MIN_SIGMA**2``) become a *compute-then-select* over **safe denominators**:
    ``np.where`` evaluates both branches, so the guarded regions must never divide
    by zero and poison unrelated elements. Guard precedence matches the scalar —
    the ``sigma`` guard is applied last, overriding the ``Z`` guard, mirroring the
    scalar's early ``return 0.0``.

    Args:
        mu: Means of the untruncated normals (any broadcastable shape).
        sigma: Standard deviations, broadcastable against ``mu``.
        lower: Lower truncation bound (default 0.0 for dosage).
        upper: Upper truncation bound (default 2.0 for dosage).

    Returns:
        Array of truncated variances with the broadcast shape of ``mu``/``sigma``.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    mu, sigma = np.broadcast_arrays(mu, sigma)

    # Safe sigma: in the sigma-guarded region the result is overwritten with 0.0
    # below, so substituting 1.0 here only prevents a 0/0 that would NaN-poison
    # nothing (masked) yet still avoids a divide warning. Elsewhere sig_safe==sigma.
    sig_safe = np.where(sigma < _MIN_SIGMA, 1.0, sigma)
    z_lower = (lower - mu) / sig_safe
    z_upper = (upper - mu) / sig_safe

    phi_lo, phi_hi = norm.pdf(z_lower), norm.pdf(z_upper)
    Phi_lo, Phi_hi = norm.cdf(z_lower), norm.cdf(z_upper)

    Z = Phi_hi - Phi_lo
    Z_safe = np.where(Z < _MIN_Z, 1.0, Z)

    term1 = (z_lower * phi_lo - z_upper * phi_hi) / Z_safe
    term2 = ((phi_lo - phi_hi) / Z_safe) ** 2
    var = np.maximum(0.0, sig_safe**2 * (1.0 + term1 - term2))

    # Guard precedence identical to the scalar: Z floor first, then the sigma
    # early-return wins (applied last).
    var = np.where(Z < _MIN_Z, _MIN_SIGMA**2, var)
    var = np.where(sigma < _MIN_SIGMA, 0.0, var)
    return var


def truncated_normal_mean(
    mu: float,
    sigma: float,
    lower: float = 0.0,
    upper: float = 2.0,
) -> float:
    """Compute mean of a truncated normal distribution.

    For X ~ N(mu, sigma²) truncated to [lower, upper], computes E[X | lower <= X <= upper].

    The formula uses standardized bounds:

        E[X] = mu + sigma * (φ(z_lower) - φ(z_upper)) / Z

    where z_lower = (lower - mu) / sigma, z_upper = (upper - mu) / sigma,
    φ is the standard normal PDF, and Z = Φ(z_upper) - Φ(z_lower).

    Args:
        mu: Mean of the untruncated normal distribution.
        sigma: Standard deviation of the untruncated normal distribution.
        lower: Lower truncation bound (default 0.0 for dosage).
        upper: Upper truncation bound (default 2.0 for dosage).

    Returns:
        Mean of the truncated distribution. Always in [lower, upper].
    """
    if sigma < _MIN_SIGMA:
        # Zero variance - just clip mu to bounds
        return max(lower, min(upper, mu))

    z_lower = (lower - mu) / sigma
    z_upper = (upper - mu) / sigma

    phi_lo, phi_hi = norm.pdf(z_lower), norm.pdf(z_upper)
    Phi_lo, Phi_hi = norm.cdf(z_lower), norm.cdf(z_upper)

    Z = Phi_hi - Phi_lo
    if Z < _MIN_Z:
        # Nearly all mass is outside bounds
        # Return the bound closest to mu
        if mu < lower:
            return lower
        elif mu > upper:
            return upper
        else:
            return mu

    truncated_mean = mu + sigma * (phi_lo - phi_hi) / Z
    # Ensure result is within bounds (numerical safety)
    return max(lower, min(upper, truncated_mean))


def clip_and_adjust_variance(
    raw_prediction: float,
    residual_variance: float,
    lower: float = 0.0,
    upper: float = 2.0,
) -> Tuple[float, float]:
    """Clip prediction to bounds and adjust variance for truncation effect.

    When a prediction from a normal distribution is clipped to valid dosage
    bounds [0, 2], the effective variance is reduced. This function returns
    both the clipped prediction and the properly adjusted variance.

    Args:
        raw_prediction: The unclipped prediction (mu of the normal distribution).
        residual_variance: Variance of the residuals (sigma² of the normal distribution).
        lower: Lower bound for clipping (default 0.0 for dosage).
        upper: Upper bound for clipping (default 2.0 for dosage).

    Returns:
        Tuple of (clipped_prediction, adjusted_variance).

    Raises:
        ValueError: If residual_variance is negative.

    Example:
        >>> # Well within bounds - variance nearly unchanged
        >>> pred, var = clip_and_adjust_variance(1.0, 0.25)
        >>> pred
        1.0
        >>> np.isclose(var, 0.25, rtol=0.01)
        True

        >>> # Near boundary - variance is reduced
        >>> pred, var = clip_and_adjust_variance(-0.2, 0.25)
        >>> pred
        0.0
        >>> var < 0.25
        True
    """
    if residual_variance < 0:
        raise ValueError(f"residual_variance must be non-negative, got {residual_variance}")

    clipped = max(lower, min(upper, raw_prediction))

    if residual_variance < _MIN_SIGMA**2:
        return clipped, 0.0

    sigma = np.sqrt(residual_variance)
    adjusted_variance = truncated_normal_variance(raw_prediction, sigma, lower, upper)

    return clipped, adjusted_variance


def clip_and_adjust_variance_array(
    raw_prediction: np.ndarray,
    residual_variance: np.ndarray,
    lower: float = 0.0,
    upper: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized :func:`clip_and_adjust_variance` over element-aligned arrays.

    Elementwise-identical to the scalar oracle. ``residual_variance`` below the
    variance floor ``_MIN_SIGMA**2`` yields ``adjusted = 0.0`` (as the scalar's
    early return); the ``sqrt`` uses a safe substitute there to avoid feeding a
    degenerate sigma into the array variance kernel.

    Args:
        raw_prediction: Unclipped predictions (mu), any broadcastable shape.
        residual_variance: Residual variances (sigma²), broadcastable against
            ``raw_prediction``. Must be non-negative.
        lower: Lower clip/truncation bound (default 0.0).
        upper: Upper clip/truncation bound (default 2.0).

    Returns:
        Tuple ``(clipped, adjusted_variance)`` of broadcast-shaped arrays.

    Raises:
        ValueError: If any ``residual_variance`` element is negative.
    """
    raw = np.asarray(raw_prediction, dtype=np.float64)
    rv = np.asarray(residual_variance, dtype=np.float64)
    raw, rv = np.broadcast_arrays(raw, rv)

    if np.any(rv < 0):
        raise ValueError("residual_variance must be non-negative")

    clipped = np.clip(raw, lower, upper)

    below_floor = rv < _MIN_SIGMA**2
    rv_safe = np.where(below_floor, 1.0, rv)
    sigma = np.sqrt(rv_safe)
    adjusted = truncated_normal_variance_array(raw, sigma, lower, upper)
    adjusted = np.where(below_floor, 0.0, adjusted)

    return clipped, adjusted


def compute_truncation_adjustment_factor(
    mu: float,
    sigma: float,
    lower: float = 0.0,
    upper: float = 2.0,
) -> float:
    """Compute variance reduction factor due to truncation.

    Returns the ratio of truncated variance to original variance, which
    indicates how much the variance is reduced by truncation.

    Args:
        mu: Mean of the untruncated normal distribution.
        sigma: Standard deviation of the untruncated normal distribution.
        lower: Lower truncation bound (default 0.0 for dosage).
        upper: Upper truncation bound (default 2.0 for dosage).

    Returns:
        Factor in [0, 1] where 1 means no reduction (mu far from bounds)
        and values closer to 0 mean significant reduction (mu near or outside bounds).
    """
    if sigma < _MIN_SIGMA:
        return 0.0

    original_variance = sigma**2
    truncated_var = truncated_normal_variance(mu, sigma, lower, upper)

    return truncated_var / original_variance
