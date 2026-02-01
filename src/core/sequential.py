"""
Sequential Testing for A/B Experiments
=======================================

Enables valid early stopping of experiments without inflating false positive rates.
Implements Group Sequential Testing with O'Brien-Fleming spending function.

The Problem:
- "Peeking" at experiment results inflates Type I error from 5% → 25%
- Example: Checking significance every day and stopping when p<0.05
- This leads to false positives and bad product launches

The Solution:
- Pre-specify analysis times (e.g., 25%, 50%, 75%, 100% of data)
- Use adjusted significance thresholds at each look
- O'Brien-Fleming: Conservative early, lenient late
- Maintains overall α = 0.05 across all looks

Author: Michael Robins
Date: January 31, 2026

References:
- O'Brien & Fleming (1979): "A multiple testing procedure for clinical trials"
- Jennison & Turnbull (1999): "Group Sequential Methods with Applications to Clinical Trials"
"""

import numpy as np
from scipy import stats
from typing import List, Optional, Tuple, Literal
from dataclasses import dataclass
import warnings


@dataclass
class SequentialBoundary:
    """Sequential testing boundary at a single analysis."""
    analysis_number: int  # Which analysis (1, 2, 3, ...)
    information_fraction: float  # Fraction of total sample size (0-1)
    sample_size: int  # Sample size at this analysis
    z_threshold: float  # Z-score threshold for significance
    p_threshold: float  # P-value threshold for significance
    efficacy_boundary: float  # Upper boundary (reject H0 if Z > this)
    futility_boundary: Optional[float] = None  # Lower boundary (stop for futility)


@dataclass
class SequentialTestResult:
    """Result from sequential test analysis."""
    reject_null: bool  # Whether to reject null hypothesis
    stop_early: bool  # Whether to stop experiment early
    reason: str  # "efficacy", "futility", "continue", or "final"
    z_statistic: float  # Observed Z-statistic
    p_value: float  # Nominal p-value (not adjusted)
    adjusted_p_value: float  # Adjusted p-value accounting for multiple looks
    boundary: SequentialBoundary  # Boundary at this analysis

    def __repr__(self) -> str:
        return (
            f"SequentialTestResult(\n"
            f"  reject_null={self.reject_null}\n"
            f"  stop_early={self.stop_early}\n"
            f"  reason={self.reason}\n"
            f"  z_statistic={self.z_statistic:.3f}\n"
            f"  p_value={self.p_value:.4f}\n"
            f"  adjusted_p_value={self.adjusted_p_value:.4f}\n"
            f"  boundary={self.boundary.information_fraction:.0%}\n"
            f")"
        )


class GroupSequentialTest:
    """
    Group Sequential Testing with O'Brien-Fleming boundaries.

    Allows interim analyses at pre-specified times while controlling
    family-wise Type I error rate.

    The O'Brien-Fleming approach:
    - Very conservative boundaries early in the experiment
    - Boundaries relax as more data accumulates
    - Final analysis uses approximately α = 0.05

    Example:
        >>> # Setup: 4 looks at 25%, 50%, 75%, 100% of data
        >>> gst = GroupSequentialTest(
        ...     n_looks=4,
        ...     alpha=0.05,
        ...     information_fractions=[0.25, 0.50, 0.75, 1.0]
        ... )
        >>>
        >>> # First interim analysis at 25% of data
        >>> result = gst.analyze(
        ...     treatment_mean=0.12,
        ...     control_mean=0.10,
        ...     treatment_n=1000,
        ...     control_n=1000,
        ...     analysis_number=1
        ... )
        >>> print(f"Stop early: {result.stop_early}")
    """

    def __init__(
        self,
        n_looks: int = 4,
        alpha: float = 0.05,
        information_fractions: Optional[List[float]] = None,
        spending_function: Literal["obrien-fleming", "pocock"] = "obrien-fleming",
        two_sided: bool = True
    ):
        """
        Initialize group sequential test.

        Args:
            n_looks: Number of interim analyses + final analysis
            alpha: Overall Type I error rate (e.g., 0.05)
            information_fractions: Fraction of total sample size at each look
                                   (e.g., [0.25, 0.5, 0.75, 1.0])
            spending_function: "obrien-fleming" or "pocock"
            two_sided: Whether test is two-sided
        """
        self.n_looks = n_looks
        self.alpha = alpha
        self.two_sided = two_sided
        self.spending_function = spending_function

        # Set information fractions
        if information_fractions is None:
            # Default: equally spaced looks
            self.information_fractions = np.linspace(
                1 / n_looks, 1.0, n_looks
            ).tolist()
        else:
            if len(information_fractions) != n_looks:
                raise ValueError(
                    f"information_fractions must have length {n_looks}, "
                    f"got {len(information_fractions)}"
                )
            if not all(0 < f <= 1 for f in information_fractions):
                raise ValueError("information_fractions must be in (0, 1]")
            if information_fractions[-1] != 1.0:
                warnings.warn("Last information fraction should be 1.0")
            self.information_fractions = information_fractions

        # Compute boundaries
        self.boundaries = self._compute_boundaries()

    def _compute_boundaries(self) -> List[SequentialBoundary]:
        """
        Compute sequential boundaries using spending function.

        Returns:
            List of SequentialBoundary objects
        """
        boundaries = []

        for k in range(1, self.n_looks + 1):
            t_k = self.information_fractions[k - 1]

            # Compute Z-threshold using spending function
            if self.spending_function == "obrien-fleming":
                z_threshold = self._obrien_fleming_boundary(k, t_k)
            elif self.spending_function == "pocock":
                z_threshold = self._pocock_boundary(k)
            else:
                raise ValueError(f"Unknown spending function: {self.spending_function}")

            # Convert to p-value threshold
            if self.two_sided:
                p_threshold = 2 * (1 - stats.norm.cdf(z_threshold))
            else:
                p_threshold = 1 - stats.norm.cdf(z_threshold)

            boundary = SequentialBoundary(
                analysis_number=k,
                information_fraction=t_k,
                sample_size=0,  # Will be set during analysis
                z_threshold=z_threshold,
                p_threshold=p_threshold,
                efficacy_boundary=z_threshold,
                futility_boundary=None  # Optional: add futility bounds
            )

            boundaries.append(boundary)

        return boundaries

    def _obrien_fleming_boundary(self, k: int, t_k: float) -> float:
        """
        Compute O'Brien-Fleming boundary using alpha-spending function.

        The O'Brien-Fleming spending function is:
            alpha(t) = 2 * [1 - Phi(z_alpha/2 / sqrt(t))]

        For exact boundaries, we compute the incremental alpha spent at each look
        and find the z-score that corresponds to that spending.

        Args:
            k: Analysis number (1-indexed)
            t_k: Information fraction at look k

        Returns:
            Z-score threshold for significance
        """
        if self.two_sided:
            z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        else:
            z_alpha = stats.norm.ppf(1 - self.alpha)

        # Use exact tabulated O'Brien-Fleming boundaries for common cases
        # For alpha=0.05, two-sided, 4 equally-spaced looks
        if (self.two_sided and abs(self.alpha - 0.05) < 1e-6 and
            self.n_looks == 4 and
            np.allclose(self.information_fractions, [0.25, 0.5, 0.75, 1.0])):
            # Exact O'Brien-Fleming boundaries from literature
            exact_boundaries = [4.049, 2.863, 2.337, 2.004]
            return exact_boundaries[k - 1]

        # For other cases, use the O'Brien-Fleming approximation with adjustment
        # Compute cumulative alpha spent using O'Brien-Fleming spending function
        def obf_spending(t):
            """O'Brien-Fleming alpha spending function"""
            if t <= 0:
                return 0
            if self.two_sided:
                return 2 * (1 - stats.norm.cdf(z_alpha / np.sqrt(t)))
            else:
                return 1 - stats.norm.cdf(z_alpha / np.sqrt(t))

        # Cumulative alpha spent up to this look
        alpha_spent_current = obf_spending(t_k)

        # Cumulative alpha spent up to previous look
        if k == 1:
            alpha_spent_previous = 0
        else:
            t_k_minus_1 = self.information_fractions[k - 2]
            alpha_spent_previous = obf_spending(t_k_minus_1)

        # Incremental alpha for this look
        incremental_alpha = alpha_spent_current - alpha_spent_previous

        # Find z-score corresponding to this incremental alpha
        if self.two_sided:
            z_k = stats.norm.ppf(1 - incremental_alpha / 2)
        else:
            z_k = stats.norm.ppf(1 - incremental_alpha)

        return z_k

    def _pocock_boundary(self, k: int) -> float:
        """
        Compute Pocock boundary (constant across looks).

        Pocock boundaries are approximately constant across all looks.
        For 4 looks and α=0.05 (two-sided), threshold ≈ 2.36.

        Args:
            k: Analysis number (1-indexed)

        Returns:
            Z-score threshold for significance
        """
        # Simplified Pocock approximation
        # For exact values, use numerical integration
        if self.n_looks == 4 and self.alpha == 0.05 and self.two_sided:
            return 2.361  # Exact value for 4 looks
        elif self.n_looks == 3 and self.alpha == 0.05 and self.two_sided:
            return 2.289
        elif self.n_looks == 5 and self.alpha == 0.05 and self.two_sided:
            return 2.413
        else:
            # Approximation
            c = self.n_looks ** 0.25
            if self.two_sided:
                z_alpha = stats.norm.ppf(1 - self.alpha / 2)
            else:
                z_alpha = stats.norm.ppf(1 - self.alpha)
            return z_alpha * c / np.sqrt(self.n_looks)

    def analyze(
        self,
        treatment_mean: float,
        control_mean: float,
        treatment_n: int,
        control_n: int,
        analysis_number: int,
        treatment_std: Optional[float] = None,
        control_std: Optional[float] = None,
        pooled_std: Optional[float] = None
    ) -> SequentialTestResult:
        """
        Perform sequential analysis at interim or final look.

        Args:
            treatment_mean: Treatment group mean
            control_mean: Control group mean
            treatment_n: Treatment group sample size
            control_n: Control group sample size
            analysis_number: Which analysis (1 = first interim, n_looks = final)
            treatment_std: Treatment standard deviation (optional)
            control_std: Control standard deviation (optional)
            pooled_std: Pooled standard deviation (optional)

        Returns:
            SequentialTestResult with decision
        """
        if analysis_number < 1 or analysis_number > self.n_looks:
            raise ValueError(
                f"analysis_number must be 1-{self.n_looks}, got {analysis_number}"
            )

        # Get boundary for this analysis
        boundary = self.boundaries[analysis_number - 1]
        boundary.sample_size = treatment_n + control_n

        # Compute Z-statistic
        effect_size = treatment_mean - control_mean

        # Estimate standard error
        if pooled_std is not None:
            se = pooled_std * np.sqrt(1 / treatment_n + 1 / control_n)
        elif treatment_std is not None and control_std is not None:
            # Unpooled variance
            se = np.sqrt(
                (treatment_std ** 2) / treatment_n +
                (control_std ** 2) / control_n
            )
        else:
            # For proportions, use pooled proportion
            # Assume binary outcome with mean = proportion
            p_pooled = (treatment_mean * treatment_n + control_mean * control_n) / (
                treatment_n + control_n
            )
            se = np.sqrt(p_pooled * (1 - p_pooled) * (1 / treatment_n + 1 / control_n))

        z_statistic = effect_size / se if se > 0 else 0.0

        # Nominal p-value (not adjusted for multiple looks)
        if self.two_sided:
            p_value = 2 * (1 - stats.norm.cdf(abs(z_statistic)))
        else:
            p_value = 1 - stats.norm.cdf(z_statistic)

        # Compare to boundary
        reject_null = abs(z_statistic) > boundary.z_threshold

        # Determine stopping decision
        is_final = (analysis_number == self.n_looks)

        if is_final:
            # Final analysis: always stop
            stop_early = False
            reason = "final"
        elif reject_null:
            # Crossed efficacy boundary: stop for efficacy
            stop_early = True
            reason = "efficacy"
        else:
            # Did not cross boundary: continue
            stop_early = False
            reason = "continue"

        # Adjusted p-value (conservative estimate)
        # For O'Brien-Fleming, approximately:
        # adjusted_p = p_value * sqrt(1 / t_k)
        adjusted_p_value = min(
            p_value / np.sqrt(boundary.information_fraction),
            1.0
        )

        return SequentialTestResult(
            reject_null=reject_null,
            stop_early=stop_early,
            reason=reason,
            z_statistic=z_statistic,
            p_value=p_value,
            adjusted_p_value=adjusted_p_value,
            boundary=boundary
        )

    def get_boundary_summary(self) -> str:
        """
        Get summary of sequential boundaries.

        Returns:
            Formatted string with boundary information
        """
        lines = [
            f"Group Sequential Design: {self.spending_function.upper()}",
            f"Number of looks: {self.n_looks}",
            f"Overall alpha: {self.alpha}",
            f"Two-sided: {self.two_sided}",
            "",
            "Boundaries:",
            "-" * 70
        ]

        for boundary in self.boundaries:
            lines.append(
                f"  Look {boundary.analysis_number}: "
                f"{boundary.information_fraction:.0%} of data | "
                f"Z > {boundary.z_threshold:.3f} | "
                f"p < {boundary.p_threshold:.4f}"
            )

        return "\n".join(lines)


def calculate_expected_sample_size_reduction(
    n_looks: int,
    true_effect_size: float,
    null_effect_size: float = 0.0,
    spending_function: str = "obrien-fleming"
) -> float:
    """
    Estimate expected sample size reduction from sequential testing.

    Sequential testing typically saves 20-30% of sample size when there's
    a real effect, because you can stop early after crossing boundary.

    Args:
        n_looks: Number of interim analyses
        true_effect_size: True underlying effect (Cohen's d)
        null_effect_size: Effect under null hypothesis (usually 0)
        spending_function: "obrien-fleming" or "pocock"

    Returns:
        Expected sample size reduction (0-1)

    Note:
        This is a rough approximation. Actual savings depend on effect size,
        variance, and boundary crossing probabilities.
    """
    # Rough heuristic: O'Brien-Fleming saves ~23% on average
    # when there's a moderate effect (d ≈ 0.2-0.5)
    if spending_function == "obrien-fleming":
        if abs(true_effect_size) > 0.3:
            return 0.23  # 23% savings
        elif abs(true_effect_size) > 0.2:
            return 0.15
        else:
            return 0.05  # Minimal savings for small effects
    else:  # Pocock
        # Pocock more aggressive, higher early stopping probability
        if abs(true_effect_size) > 0.3:
            return 0.30
        elif abs(true_effect_size) > 0.2:
            return 0.20
        else:
            return 0.08

    return 0.0


if __name__ == "__main__":
    # Example: Sequential testing with 4 looks
    np.random.seed(42)

    gst = GroupSequentialTest(
        n_looks=4,
        alpha=0.05,
        spending_function="obrien-fleming"
    )

    print(gst.get_boundary_summary())
    print()

    # Simulate experiment with true 2pp lift
    baseline_rate = 0.10
    treatment_rate = 0.12  # 2pp lift

    total_n = 10000  # Total planned sample size per variant

    print("Simulating Experiment:")
    print("=" * 70)

    for k in range(1, 5):
        # Sample size at this look
        info_frac = gst.information_fractions[k - 1]
        n_at_look = int(total_n * info_frac)

        # Simulate data
        control_conversions = np.random.binomial(n_at_look, baseline_rate)
        treatment_conversions = np.random.binomial(n_at_look, treatment_rate)

        control_mean = control_conversions / n_at_look
        treatment_mean = treatment_conversions / n_at_look

        # Analyze
        result = gst.analyze(
            treatment_mean=treatment_mean,
            control_mean=control_mean,
            treatment_n=n_at_look,
            control_n=n_at_look,
            analysis_number=k
        )

        print(f"Look {k} ({info_frac:.0%} of data, n={n_at_look:,} per variant):")
        print(f"  Control: {control_mean:.2%}")
        print(f"  Treatment: {treatment_mean:.2%}")
        print(f"  Z-statistic: {result.z_statistic:.3f}")
        print(f"  Threshold: {result.boundary.z_threshold:.3f}")
        print(f"  Reject null: {result.reject_null}")
        print(f"  Decision: {result.reason.upper()}")
        print()

        if result.stop_early:
            print(f"STOP: Crossed efficacy boundary at look {k}!")
            print(f"Sample size savings: {(1 - info_frac):.0%}")
            break
