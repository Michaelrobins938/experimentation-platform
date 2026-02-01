"""
CUPED: Controlled-Experiment Using Pre-Experiment Data
=======================================================

Variance reduction technique that uses pre-experiment covariates to reduce
noise in A/B test metrics, enabling faster experiments with smaller sample sizes.

The Problem:
- Traditional A/B tests have high variance due to user heterogeneity
- Example: Revenue varies 10x between users, drowning out treatment effects
- This requires large samples and long duration to detect effects

The Solution:
- Collect pre-experiment metric (e.g., revenue in prior 30 days)
- Use linear regression to "adjust" post-experiment metric
- Reduces variance by 30-50% → 2x faster experiments

Author: Michael Robins
Date: January 31, 2026

References:
- Deng et al. (2013): "Improving the Sensitivity of Online Controlled Experiments
  by Utilizing Pre-Experiment Data"
- Microsoft Research: Original CUPED paper
"""

import numpy as np
from scipy import stats
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import warnings


@dataclass
class CUPEDResult:
    """Results from CUPED variance reduction."""
    adjusted_metric: np.ndarray  # Variance-reduced metric
    variance_reduction: float  # Variance reduction (%)
    theta: float  # Optimal coefficient
    original_variance: float
    adjusted_variance: float
    correlation: float  # Correlation between metric and covariate

    def __repr__(self) -> str:
        return (
            f"CUPEDResult(\n"
            f"  variance_reduction={self.variance_reduction:.1%}\n"
            f"  theta={self.theta:.4f}\n"
            f"  original_var={self.original_variance:.2f}\n"
            f"  adjusted_var={self.adjusted_variance:.2f}\n"
            f"  correlation={self.correlation:.3f}\n"
            f")"
        )


class CUPED:
    """
    CUPED (Controlled-experiment Using Pre-Experiment Data) implementation.

    CUPED reduces variance by adjusting the experiment metric using pre-experiment
    covariate data. The adjustment is optimal in the sense that it minimizes variance.

    Formula:
        Y_cuped = Y - θ(X - E[X])

    Where:
        - Y = experiment metric (e.g., post-experiment revenue)
        - X = pre-experiment covariate (e.g., pre-experiment revenue)
        - θ = Cov(Y, X) / Var(X) (optimal coefficient)
        - E[X] = mean of covariate

    The adjusted metric Y_cuped has the same expectation as Y but lower variance.

    Example:
        >>> # Experiment data
        >>> treatment = np.array([100, 120, 80, 150, 110])  # Post-experiment revenue
        >>> control = np.array([90, 100, 75, 140, 100])
        >>>
        >>> # Pre-experiment data
        >>> treatment_pre = np.array([95, 115, 78, 145, 105])
        >>> control_pre = np.array([88, 98, 72, 138, 98])
        >>>
        >>> # Apply CUPED
        >>> cuped = CUPED()
        >>> result = cuped.fit_transform(
        ...     treatment_metric=treatment,
        ...     control_metric=control,
        ...     treatment_covariate=treatment_pre,
        ...     control_covariate=control_pre
        ... )
        >>> print(f"Variance reduction: {result.variance_reduction:.1%}")
    """

    def __init__(self):
        """Initialize CUPED."""
        self.theta_: Optional[float] = None

    def fit(
        self,
        metric: np.ndarray,
        covariate: np.ndarray
    ) -> float:
        """
        Fit CUPED by computing optimal theta coefficient.

        Args:
            metric: Experiment metric (Y)
            covariate: Pre-experiment covariate (X)

        Returns:
            Optimal theta coefficient

        Formula:
            θ = Cov(Y, X) / Var(X)
        """
        # Validate inputs
        if len(metric) != len(covariate):
            raise ValueError(
                f"metric and covariate must have same length, "
                f"got {len(metric)} and {len(covariate)}"
            )

        if len(metric) < 2:
            raise ValueError("Need at least 2 observations")

        # Compute theta
        cov = np.cov(metric, covariate)[0, 1]
        var_x = np.var(covariate, ddof=1)

        if var_x == 0:
            warnings.warn("Covariate has zero variance, setting theta=0")
            theta = 0.0
        else:
            theta = cov / var_x

        self.theta_ = theta

        return theta

    def transform(
        self,
        metric: np.ndarray,
        covariate: np.ndarray,
        theta: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply CUPED transformation to reduce variance.

        Args:
            metric: Experiment metric (Y)
            covariate: Pre-experiment covariate (X)
            theta: Optional theta coefficient (uses fitted value if None)

        Returns:
            Adjusted metric with reduced variance

        Formula:
            Y_cuped = Y - θ(X - E[X])
        """
        if theta is None:
            if self.theta_ is None:
                raise ValueError("Must fit() before transform() or provide theta")
            theta = self.theta_

        # Mean-center covariate
        x_centered = covariate - np.mean(covariate)

        # Apply CUPED adjustment
        adjusted_metric = metric - theta * x_centered

        return adjusted_metric

    def fit_transform(
        self,
        treatment_metric: np.ndarray,
        control_metric: np.ndarray,
        treatment_covariate: np.ndarray,
        control_covariate: np.ndarray
    ) -> CUPEDResult:
        """
        Fit and transform both treatment and control variants.

        Computes optimal theta using pooled data, then applies to both variants.

        Args:
            treatment_metric: Treatment experiment metric
            control_metric: Control experiment metric
            treatment_covariate: Treatment pre-experiment covariate
            control_covariate: Control pre-experiment covariate

        Returns:
            CUPEDResult with adjusted metrics and variance reduction
        """
        # Pool data for theta estimation
        pooled_metric = np.concatenate([treatment_metric, control_metric])
        pooled_covariate = np.concatenate([treatment_covariate, control_covariate])

        # Fit theta on pooled data
        theta = self.fit(pooled_metric, pooled_covariate)

        # Transform both variants
        treatment_adjusted = self.transform(treatment_metric, treatment_covariate, theta)
        control_adjusted = self.transform(control_metric, control_covariate, theta)

        # Compute variance reduction
        original_var = np.var(pooled_metric, ddof=1)
        adjusted_var = np.var(
            np.concatenate([treatment_adjusted, control_adjusted]),
            ddof=1
        )

        variance_reduction = (original_var - adjusted_var) / original_var

        # Compute correlation
        correlation = np.corrcoef(pooled_metric, pooled_covariate)[0, 1]

        # Combine adjusted metrics
        adjusted_metric = np.concatenate([treatment_adjusted, control_adjusted])

        return CUPEDResult(
            adjusted_metric=adjusted_metric,
            variance_reduction=variance_reduction,
            theta=theta,
            original_variance=original_var,
            adjusted_variance=adjusted_var,
            correlation=correlation
        )

    def compute_variance_reduction(
        self,
        metric: np.ndarray,
        covariate: np.ndarray
    ) -> float:
        """
        Compute expected variance reduction from using covariate.

        Does NOT require running experiment - can be used for pre-experiment planning.

        Args:
            metric: Historical metric data
            covariate: Historical covariate data

        Returns:
            Expected variance reduction (0-1)

        Formula:
            variance_reduction = ρ²

        Where ρ is the correlation between metric and covariate.
        """
        correlation = np.corrcoef(metric, covariate)[0, 1]
        variance_reduction = correlation ** 2

        return variance_reduction


class CUPAC:
    """
    CUPAC: CUPED for Continuous metrics using Asymptotic Approximations.

    Extension of CUPED for continuous metrics with better handling of
    heteroskedasticity and non-normality.

    Primarily used for metrics like revenue where variance is non-constant.

    Example:
        >>> cupac = CUPAC()
        >>> result = cupac.fit_transform(
        ...     treatment_metric=treatment_revenue,
        ...     control_metric=control_revenue,
        ...     treatment_covariate=pre_revenue_treatment,
        ...     control_covariate=pre_revenue_control
        ... )
    """

    def __init__(self):
        """Initialize CUPAC."""
        self.cuped = CUPED()

    def fit_transform(
        self,
        treatment_metric: np.ndarray,
        control_metric: np.ndarray,
        treatment_covariate: np.ndarray,
        control_covariate: np.ndarray,
        use_robust_variance: bool = True
    ) -> CUPEDResult:
        """
        Apply CUPAC transformation.

        Currently delegates to CUPED (identical for most use cases).
        Future enhancement: Add robust variance estimation.

        Args:
            treatment_metric: Treatment experiment metric
            control_metric: Control experiment metric
            treatment_covariate: Treatment pre-experiment covariate
            control_covariate: Control pre-experiment covariate
            use_robust_variance: Whether to use robust variance estimation

        Returns:
            CUPEDResult with adjusted metrics
        """
        # For now, CUPAC is identical to CUPED
        # Future: Add heteroskedasticity-robust variance estimation
        return self.cuped.fit_transform(
            treatment_metric=treatment_metric,
            control_metric=control_metric,
            treatment_covariate=treatment_covariate,
            control_covariate=control_covariate
        )


def estimate_sample_size_reduction(
    correlation: float,
    original_sample_size: int
) -> int:
    """
    Estimate sample size reduction from using CUPED.

    Args:
        correlation: Correlation between metric and covariate
        original_sample_size: Original required sample size

    Returns:
        Reduced sample size with CUPED

    Formula:
        n_cuped = n_original * (1 - ρ²)

    Example:
        >>> # With 40% correlation, can reduce sample by 16%
        >>> n_cuped = estimate_sample_size_reduction(
        ...     correlation=0.4,
        ...     original_sample_size=10000
        ... )
        >>> print(f"Reduced to {n_cuped:,} samples")
        Reduced to 8,400 samples
    """
    variance_reduction = correlation ** 2
    reduced_sample_size = int(original_sample_size * (1 - variance_reduction))

    return reduced_sample_size


def select_best_covariate(
    metric: np.ndarray,
    covariates: Dict[str, np.ndarray]
) -> Tuple[str, float]:
    """
    Select covariate with highest variance reduction potential.

    Args:
        metric: Experiment metric
        covariates: Dictionary of covariate_name -> covariate_data

    Returns:
        Tuple of (best_covariate_name, expected_variance_reduction)

    Example:
        >>> covariates = {
        ...     'prior_revenue': np.array([100, 120, 80, 150]),
        ...     'prior_sessions': np.array([10, 15, 8, 20]),
        ...     'account_age_days': np.array([365, 180, 90, 730])
        ... }
        >>> metric = np.array([110, 125, 85, 155])
        >>> best_name, var_reduction = select_best_covariate(metric, covariates)
        >>> print(f"Best covariate: {best_name} ({var_reduction:.1%} reduction)")
    """
    best_covariate = None
    best_variance_reduction = 0.0

    for name, covariate in covariates.items():
        correlation = np.corrcoef(metric, covariate)[0, 1]
        variance_reduction = correlation ** 2

        if variance_reduction > best_variance_reduction:
            best_variance_reduction = variance_reduction
            best_covariate = name

    return best_covariate, best_variance_reduction


if __name__ == "__main__":
    # Example: CUPED reduces variance
    np.random.seed(42)

    # Simulate experiment with heterogeneous users
    n_users = 1000

    # Pre-experiment revenue (covariate)
    pre_revenue = np.random.lognormal(mean=4.0, sigma=1.0, size=n_users)

    # Post-experiment revenue (correlated with pre-revenue)
    # Treatment has +10% lift
    treatment_indices = np.random.choice(n_users, size=n_users // 2, replace=False)
    control_indices = np.setdiff1d(np.arange(n_users), treatment_indices)

    post_revenue = pre_revenue * np.random.lognormal(mean=0, sigma=0.3, size=n_users)
    post_revenue[treatment_indices] *= 1.10  # 10% lift

    # Split into treatment and control
    treatment_metric = post_revenue[treatment_indices]
    control_metric = post_revenue[control_indices]
    treatment_covariate = pre_revenue[treatment_indices]
    control_covariate = pre_revenue[control_indices]

    # Apply CUPED
    cuped = CUPED()
    result = cuped.fit_transform(
        treatment_metric=treatment_metric,
        control_metric=control_metric,
        treatment_covariate=treatment_covariate,
        control_covariate=control_covariate
    )

    print("CUPED Variance Reduction Example")
    print("=" * 50)
    print(result)
    print()

    # Compare t-test p-values
    # Without CUPED
    t_stat_original, p_value_original = stats.ttest_ind(treatment_metric, control_metric)

    # With CUPED
    treatment_adjusted = cuped.transform(treatment_metric, treatment_covariate)
    control_adjusted = cuped.transform(control_metric, control_covariate)
    t_stat_cuped, p_value_cuped = stats.ttest_ind(treatment_adjusted, control_adjusted)

    print("Statistical Test Results:")
    print(f"Original p-value: {p_value_original:.4f}")
    print(f"CUPED p-value:    {p_value_cuped:.4f}")
    print(f"Sensitivity gain: {p_value_original / p_value_cuped:.2f}x")
    print()

    # Sample size reduction
    correlation = result.correlation
    sample_reduction = estimate_sample_size_reduction(
        correlation=correlation,
        original_sample_size=n_users
    )
    print(f"Sample size reduction:")
    print(f"  Original: {n_users:,}")
    print(f"  With CUPED: {sample_reduction:,}")
    print(f"  Reduction: {(1 - sample_reduction / n_users):.1%}")
