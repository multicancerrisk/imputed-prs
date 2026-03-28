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
