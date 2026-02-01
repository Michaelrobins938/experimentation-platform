"""
Power Analysis Calculator for A/B Testing
==========================================

Provides sample size, minimum detectable effect (MDE), and statistical power
calculations for two-sample proportion and mean tests.

Key Functions:
- calculate_sample_size(): Determine required sample size for target power
- calculate_mde(): Compute minimum detectable effect given sample size
- calculate_power(): Compute statistical power given effect size and sample size
- plot_power_curve(): Visualize power vs sample size relationship

Author: Michael Robins
Date: January 31, 2026
"""

import numpy as np
from scipy import stats
from typing import Dict, Optional, Tuple, Literal
from dataclasses import dataclass
import warnings


@dataclass
class PowerAnalysisResult:
    """Container for power analysis results."""
    sample_size_per_variant: int
    total_sample_size: int
    statistical_power: float
    minimum_detectable_effect: float
    alpha: float
    baseline_rate: float
    test_type: str
    duration_days: Optional[float] = None

    def __repr__(self) -> str:
        return (
            f"PowerAnalysisResult(\n"
            f"  sample_size_per_variant={self.sample_size_per_variant:,}\n"
            f"  total_sample_size={self.total_sample_size:,}\n"
            f"  statistical_power={self.statistical_power:.1%}\n"
            f"  minimum_detectable_effect={self.minimum_detectable_effect:.2%}\n"
            f"  alpha={self.alpha}\n"
            f"  baseline_rate={self.baseline_rate:.2%}\n"
            f"  duration_days={self.duration_days}\n"
            f")"
        )


class PowerAnalyzer:
    """
    Statistical power analysis for A/B testing.

    Supports both proportion tests (e.g., conversion rate) and mean tests (e.g., revenue).
    Uses normal approximation for large samples.

    Example:
        >>> analyzer = PowerAnalyzer()
        >>> result = analyzer.calculate_sample_size(
        ...     baseline_rate=0.10,
        ...     mde=0.02,  # Detect 2pp lift (10% → 12%)
        ...     alpha=0.05,
        ...     power=0.80
        ... )
        >>> print(f"Need {result.sample_size_per_variant:,} per variant")
    """

    def __init__(self):
        """Initialize power analyzer."""
        pass

    def calculate_sample_size(
        self,
        baseline_rate: float,
        mde: float,
        alpha: float = 0.05,
        power: float = 0.80,
        test_type: Literal["two-sided", "one-sided"] = "two-sided",
        metric_type: Literal["proportion", "continuous"] = "proportion",
        baseline_std: Optional[float] = None,
        daily_traffic: Optional[int] = None
    ) -> PowerAnalysisResult:
        """
        Calculate required sample size per variant.

        Args:
            baseline_rate: Baseline conversion rate or mean (e.g., 0.10 for 10%)
            mde: Minimum detectable effect (absolute, e.g., 0.02 for 2pp lift)
            alpha: Significance level (Type I error rate, default: 0.05)
            power: Statistical power (1 - Type II error rate, default: 0.80)
            test_type: "two-sided" or "one-sided" test
            metric_type: "proportion" (conversion) or "continuous" (revenue)
            baseline_std: Standard deviation for continuous metrics
            daily_traffic: Optional traffic per day to estimate duration

        Returns:
            PowerAnalysisResult with sample size and metadata

        Raises:
            ValueError: If inputs are invalid
        """
        # Validate inputs
        self._validate_inputs(baseline_rate, mde, alpha, power, metric_type, baseline_std)

        # Get Z-scores for alpha and beta
        if test_type == "two-sided":
            z_alpha = stats.norm.ppf(1 - alpha / 2)
        else:
            z_alpha = stats.norm.ppf(1 - alpha)

        z_beta = stats.norm.ppf(power)

        # Calculate sample size based on metric type
        if metric_type == "proportion":
            n = self._sample_size_proportion(
                p1=baseline_rate,
                p2=baseline_rate + mde,
                z_alpha=z_alpha,
                z_beta=z_beta
            )
        else:  # continuous
            if baseline_std is None:
                raise ValueError("baseline_std required for continuous metrics")
            n = self._sample_size_continuous(
                mu1=baseline_rate,
                mu2=baseline_rate + mde,
                sigma=baseline_std,
                z_alpha=z_alpha,
                z_beta=z_beta
            )

        # Round up to next integer
        n = int(np.ceil(n))

        # Calculate duration if traffic provided
        duration = None
        if daily_traffic is not None:
            # Total sample size / (daily traffic / 2 variants)
            duration = (n * 2) / daily_traffic

        return PowerAnalysisResult(
            sample_size_per_variant=n,
            total_sample_size=n * 2,
            statistical_power=power,
            minimum_detectable_effect=mde,
            alpha=alpha,
            baseline_rate=baseline_rate,
            test_type=test_type,
            duration_days=duration
        )

    def calculate_mde(
        self,
        baseline_rate: float,
        sample_size_per_variant: int,
        alpha: float = 0.05,
        power: float = 0.80,
        test_type: Literal["two-sided", "one-sided"] = "two-sided",
        metric_type: Literal["proportion", "continuous"] = "proportion",
        baseline_std: Optional[float] = None
    ) -> float:
        """
        Calculate minimum detectable effect given sample size.

        Args:
            baseline_rate: Baseline conversion rate or mean
            sample_size_per_variant: Sample size per variant
            alpha: Significance level
            power: Statistical power
            test_type: "two-sided" or "one-sided"
            metric_type: "proportion" or "continuous"
            baseline_std: Standard deviation for continuous metrics

        Returns:
            Minimum detectable effect (absolute)
        """
        # Get Z-scores
        if test_type == "two-sided":
            z_alpha = stats.norm.ppf(1 - alpha / 2)
        else:
            z_alpha = stats.norm.ppf(1 - alpha)

        z_beta = stats.norm.ppf(power)

        n = sample_size_per_variant

        if metric_type == "proportion":
            # For proportions: MDE = (z_α + z_β) * sqrt(2 * p * (1-p) / n)
            # Using pooled variance approximation
            pooled_variance = baseline_rate * (1 - baseline_rate)
            mde = (z_alpha + z_beta) * np.sqrt(2 * pooled_variance / n)
        else:  # continuous
            if baseline_std is None:
                raise ValueError("baseline_std required for continuous metrics")
            # For means: MDE = (z_α + z_β) * sigma * sqrt(2/n)
            mde = (z_alpha + z_beta) * baseline_std * np.sqrt(2 / n)

        return mde

    def calculate_power(
        self,
        baseline_rate: float,
        treatment_rate: float,
        sample_size_per_variant: int,
        alpha: float = 0.05,
        test_type: Literal["two-sided", "one-sided"] = "two-sided",
        metric_type: Literal["proportion", "continuous"] = "proportion",
        baseline_std: Optional[float] = None
    ) -> float:
        """
        Calculate statistical power given effect size and sample size.

        Args:
            baseline_rate: Baseline conversion rate or mean
            treatment_rate: Treatment conversion rate or mean
            sample_size_per_variant: Sample size per variant
            alpha: Significance level
            test_type: "two-sided" or "one-sided"
            metric_type: "proportion" or "continuous"
            baseline_std: Standard deviation for continuous metrics

        Returns:
            Statistical power (0-1)
        """
        # Get critical value
        if test_type == "two-sided":
            z_alpha = stats.norm.ppf(1 - alpha / 2)
        else:
            z_alpha = stats.norm.ppf(1 - alpha)

        n = sample_size_per_variant
        effect_size = treatment_rate - baseline_rate

        if metric_type == "proportion":
            # Pooled variance under H0
            pooled_var_h0 = baseline_rate * (1 - baseline_rate)
            # Standard error under H0
            se_h0 = np.sqrt(2 * pooled_var_h0 / n)

            # Standard error under H1
            var_treatment = treatment_rate * (1 - treatment_rate)
            var_baseline = baseline_rate * (1 - baseline_rate)
            se_h1 = np.sqrt((var_baseline + var_treatment) / n)

            # Non-centrality parameter
            ncp = effect_size / se_h1

            # Power = P(reject H0 | H1 true)
            # = P(|Z| > z_α | Z ~ N(ncp, 1))
            if test_type == "two-sided":
                power = 1 - stats.norm.cdf(z_alpha - ncp) + stats.norm.cdf(-z_alpha - ncp)
            else:
                power = 1 - stats.norm.cdf(z_alpha - ncp)

        else:  # continuous
            if baseline_std is None:
                raise ValueError("baseline_std required for continuous metrics")

            se = baseline_std * np.sqrt(2 / n)
            ncp = effect_size / se

            if test_type == "two-sided":
                power = 1 - stats.norm.cdf(z_alpha - ncp) + stats.norm.cdf(-z_alpha - ncp)
            else:
                power = 1 - stats.norm.cdf(z_alpha - ncp)

        return power

    def generate_power_curve(
        self,
        baseline_rate: float,
        mde: float,
        alpha: float = 0.05,
        test_type: Literal["two-sided", "one-sided"] = "two-sided",
        metric_type: Literal["proportion", "continuous"] = "proportion",
        baseline_std: Optional[float] = None,
        sample_size_range: Optional[Tuple[int, int]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate power curve data (power vs sample size).

        Args:
            baseline_rate: Baseline conversion rate or mean
            mde: Minimum detectable effect
            alpha: Significance level
            test_type: "two-sided" or "one-sided"
            metric_type: "proportion" or "continuous"
            baseline_std: Standard deviation for continuous metrics
            sample_size_range: Optional (min, max) sample sizes

        Returns:
            Dictionary with 'sample_sizes' and 'power' arrays
        """
        treatment_rate = baseline_rate + mde

        # Determine sample size range
        if sample_size_range is None:
            # Auto-determine range based on power=0.8
            n_80 = self.calculate_sample_size(
                baseline_rate=baseline_rate,
                mde=mde,
                alpha=alpha,
                power=0.80,
                test_type=test_type,
                metric_type=metric_type,
                baseline_std=baseline_std
            ).sample_size_per_variant

            min_n = max(100, int(n_80 * 0.3))
            max_n = int(n_80 * 2.0)
        else:
            min_n, max_n = sample_size_range

        # Generate sample size grid
        sample_sizes = np.linspace(min_n, max_n, num=50, dtype=int)

        # Calculate power for each sample size
        powers = np.array([
            self.calculate_power(
                baseline_rate=baseline_rate,
                treatment_rate=treatment_rate,
                sample_size_per_variant=n,
                alpha=alpha,
                test_type=test_type,
                metric_type=metric_type,
                baseline_std=baseline_std
            )
            for n in sample_sizes
        ])

        return {
            'sample_sizes': sample_sizes,
            'power': powers
        }

    def _sample_size_proportion(
        self,
        p1: float,
        p2: float,
        z_alpha: float,
        z_beta: float
    ) -> float:
        """
        Calculate sample size for proportion test.

        Uses pooled variance under H0 and unpooled under H1.

        Formula:
        n = [(z_α * sqrt(2p̄(1-p̄)) + z_β * sqrt(p1(1-p1) + p2(1-p2)))]² / (p2 - p1)²

        where p̄ = (p1 + p2) / 2
        """
        # Pooled proportion under H0
        p_pooled = (p1 + p2) / 2

        # Variance under H0 (pooled)
        var_h0 = 2 * p_pooled * (1 - p_pooled)

        # Variance under H1 (unpooled)
        var_h1 = p1 * (1 - p1) + p2 * (1 - p2)

        # Effect size
        delta = p2 - p1

        # Sample size per variant
        n = ((z_alpha * np.sqrt(var_h0) + z_beta * np.sqrt(var_h1)) / delta) ** 2

        return n

    def _sample_size_continuous(
        self,
        mu1: float,
        mu2: float,
        sigma: float,
        z_alpha: float,
        z_beta: float
    ) -> float:
        """
        Calculate sample size for continuous metric test.

        Formula:
        n = 2 * σ² * (z_α + z_β)² / (μ2 - μ1)²
        """
        delta = mu2 - mu1
        variance = sigma ** 2

        n = 2 * variance * (z_alpha + z_beta) ** 2 / (delta ** 2)

        return n

    def _validate_inputs(
        self,
        baseline_rate: float,
        mde: float,
        alpha: float,
        power: float,
        metric_type: str,
        baseline_std: Optional[float]
    ) -> None:
        """Validate input parameters."""
        if metric_type == "proportion":
            if not 0 < baseline_rate < 1:
                raise ValueError(f"baseline_rate must be in (0, 1), got {baseline_rate}")
            if not 0 < baseline_rate + mde < 1:
                raise ValueError(
                    f"baseline_rate + mde must be in (0, 1), "
                    f"got {baseline_rate + mde}"
                )

        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        if not 0 < power < 1:
            raise ValueError(f"power must be in (0, 1), got {power}")

        if mde <= 0:
            raise ValueError(f"mde must be positive, got {mde}")

        if metric_type == "continuous" and baseline_std is not None:
            if baseline_std <= 0:
                raise ValueError(f"baseline_std must be positive, got {baseline_std}")


# Convenience functions
def calculate_sample_size(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
    **kwargs
) -> PowerAnalysisResult:
    """
    Convenience function for sample size calculation.

    See PowerAnalyzer.calculate_sample_size() for full documentation.
    """
    analyzer = PowerAnalyzer()
    return analyzer.calculate_sample_size(
        baseline_rate=baseline_rate,
        mde=mde,
        alpha=alpha,
        power=power,
        **kwargs
    )


def calculate_mde(
    baseline_rate: float,
    sample_size_per_variant: int,
    alpha: float = 0.05,
    power: float = 0.80,
    **kwargs
) -> float:
    """
    Convenience function for MDE calculation.

    See PowerAnalyzer.calculate_mde() for full documentation.
    """
    analyzer = PowerAnalyzer()
    return analyzer.calculate_mde(
        baseline_rate=baseline_rate,
        sample_size_per_variant=sample_size_per_variant,
        alpha=alpha,
        power=power,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage
    analyzer = PowerAnalyzer()

    # Calculate sample size
    result = analyzer.calculate_sample_size(
        baseline_rate=0.10,  # 10% baseline conversion
        mde=0.02,  # Detect 2pp lift (10% → 12%)
        alpha=0.05,
        power=0.80,
        daily_traffic=10000
    )

    print(result)
    print(f"\nTo detect a 2pp lift from 10% → 12% with 80% power:")
    print(f"  Need {result.sample_size_per_variant:,} users per variant")
    print(f"  Total sample size: {result.total_sample_size:,}")
    print(f"  Estimated duration: {result.duration_days:.1f} days")
