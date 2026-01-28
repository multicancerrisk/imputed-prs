"""Utility functions for the imputed-prs library."""

import numpy as np
from numpy.typing import ArrayLike


def clip_dosage(value: float, lower: float = 0.0, upper: float = 2.0) -> float:
    """Clip dosage value to valid range [0, 2].

    Args:
        value: The dosage value to clip.
        lower: Lower bound (default 0.0).
        upper: Upper bound (default 2.0).

    Returns:
        The clipped dosage value.
    """
    return max(lower, min(upper, value))


def hardy_weinberg_variance(allele_freq: float) -> float:
    """Calculate Hardy-Weinberg expected variance: 2 * q * (1-q).

    Under Hardy-Weinberg equilibrium, the variance of genotype dosage
    (0, 1, or 2 copies of the effect allele) is 2 * q * (1-q), where
    q is the allele frequency.

    Args:
        allele_freq: Population allele frequency (between 0 and 1).

    Returns:
        The Hardy-Weinberg expected variance.
    """
    return 2.0 * allele_freq * (1.0 - allele_freq)


def compute_residual_variance(allele_freq: float, r2: float) -> float:
    """Calculate residual variance: 2 * q * (1-q) * (1 - r2).

    The residual variance is the unexplained variance after imputation,
    computed as the Hardy-Weinberg variance multiplied by (1 - R²).

    Args:
        allele_freq: Population allele frequency (between 0 and 1).
        r2: Imputation R² (coefficient of determination, between 0 and 1).

    Returns:
        The residual variance after imputation.
    """
    return hardy_weinberg_variance(allele_freq) * (1.0 - r2)


def compute_standard_error(
    betas: ArrayLike,
    residual_variances: ArrayLike,
) -> float:
    """Calculate SE of PRS from variant contributions: sqrt(sum(beta^2 * var)).

    The standard error of a PRS is computed by propagating the uncertainty
    from each variant's imputation through the linear combination.

    Args:
        betas: Array of effect sizes (beta coefficients) for each variant.
        residual_variances: Array of residual variances for each variant.

    Returns:
        The standard error of the PRS.
    """
    betas = np.asarray(betas)
    residual_variances = np.asarray(residual_variances)

    if len(betas) == 0 or len(residual_variances) == 0:
        return 0.0

    variance_sum = np.sum(betas**2 * residual_variances)
    return float(np.sqrt(variance_sum))
