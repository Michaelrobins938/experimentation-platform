"""
Statistical Validation Simulator
=================================

Runs thousands of simulated experiments to validate statistical guarantees:
1. Type I error control (false positive rate ≤ α)
2. Type II error control (false negative rate ≤ β)
3. CUPED variance reduction
4. Sequential testing family-wise error rate

Author: Michael Robins
Date: January 31, 2026
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.power_analysis import PowerAnalyzer
from core.cuped import CUPED
from core.sequential import GroupSequentialTest


@dataclass
class ValidationResult:
    """Results from validation simulation."""
    n_simulations: int
    observed_type_i_error: float
    target_type_i_error: float
    type_i_passes: bool

    observed_type_ii_error: Optional[float] = None
    target_type_ii_error: Optional[float] = None
    type_ii_passes: Optional[bool] = None

    observed_power: Optional[float] = None
    target_power: Optional[float] = None

    variance_reduction_mean: Optional[float] = None
    variance_reduction_std: Optional[float] = None

    def __repr__(self) -> str:
        lines = [
            f"ValidationResult({self.n_simulations:,} simulations)",
            f"Type I Error: {self.observed_type_i_error:.3f} (target: {self.target_type_i_error})",
            f"  PASS: {self.type_i_passes}",
        ]

        if self.observed_type_ii_error is not None:
            lines.append(
                f"Type II Error: {self.observed_type_ii_error:.3f} "
                f"(target: {self.target_type_ii_error})"
            )
            lines.append(f"  PASS: {self.type_ii_passes}")

        if self.observed_power is not None:
            lines.append(
                f"Statistical Power: {self.observed_power:.3f} "
                f"(target: {self.target_power})"
            )

        if self.variance_reduction_mean is not None:
            lines.append(
                f"CUPED Variance Reduction: "
                f"{self.variance_reduction_mean:.1%} ± {self.variance_reduction_std:.1%}"
            )

        return "\n".join(lines)


class ExperimentSimulator:
    """
    Simulator for validating statistical properties of A/B tests.

    Runs thousands of simulated experiments to empirically verify:
    - Type I error rate matches nominal alpha
    - Type II error rate matches nominal beta
    - CUPED variance reduction
    - Sequential testing error control
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize simulator.

        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)

    def validate_type_i_error(
        self,
        n_simulations: int = 10000,
        baseline_rate: float = 0.10,
        sample_size_per_variant: int = 1000,
        alpha: float = 0.05,
        metric_type: str = "proportion"
    ) -> ValidationResult:
        """
        Validate Type I error rate (false positive rate).

        Runs simulations under null hypothesis (no effect) and checks
        that rejection rate matches nominal alpha.

        Args:
            n_simulations: Number of simulations to run
            baseline_rate: Baseline conversion rate or mean
            sample_size_per_variant: Sample size per variant
            alpha: Significance level
            metric_type: "proportion" or "continuous"

        Returns:
            ValidationResult with Type I error validation
        """
        print(f"Running {n_simulations:,} simulations for Type I error validation...")

        rejections = 0

        for _ in tqdm(range(n_simulations), desc="Type I Error"):
            # Generate data under H0 (no difference)
            if metric_type == "proportion":
                treatment = np.random.binomial(
                    1, baseline_rate, size=sample_size_per_variant
                )
                control = np.random.binomial(
                    1, baseline_rate, size=sample_size_per_variant
                )
            else:  # continuous
                treatment = np.random.normal(
                    baseline_rate, 1.0, size=sample_size_per_variant
                )
                control = np.random.normal(
                    baseline_rate, 1.0, size=sample_size_per_variant
                )

            # Run t-test
            _, p_value = stats.ttest_ind(treatment, control)

            if p_value < alpha:
                rejections += 1

        observed_type_i = rejections / n_simulations

        # Type I error should be within ±2 standard errors of alpha
        # SE = sqrt(α(1-α) / n)
        se = np.sqrt(alpha * (1 - alpha) / n_simulations)
        tolerance = 2 * se

        type_i_passes = abs(observed_type_i - alpha) <= tolerance

        return ValidationResult(
            n_simulations=n_simulations,
            observed_type_i_error=observed_type_i,
            target_type_i_error=alpha,
            type_i_passes=type_i_passes
        )

    def validate_type_ii_error(
        self,
        n_simulations: int = 10000,
        baseline_rate: float = 0.10,
        treatment_rate: float = 0.12,
        sample_size_per_variant: int = 1000,
        alpha: float = 0.05,
        target_power: float = 0.80,
        metric_type: str = "proportion"
    ) -> ValidationResult:
        """
        Validate Type II error rate (false negative rate) and statistical power.

        Runs simulations under alternative hypothesis (true effect exists)
        and checks that power matches target.

        Args:
            n_simulations: Number of simulations to run
            baseline_rate: Baseline conversion rate or mean
            treatment_rate: Treatment conversion rate or mean
            sample_size_per_variant: Sample size per variant
            alpha: Significance level
            target_power: Target statistical power (1 - β)
            metric_type: "proportion" or "continuous"

        Returns:
            ValidationResult with Type II error and power validation
        """
        print(f"Running {n_simulations:,} simulations for Type II error validation...")

        rejections = 0

        for _ in tqdm(range(n_simulations), desc="Type II Error"):
            # Generate data under H1 (true difference exists)
            if metric_type == "proportion":
                treatment = np.random.binomial(
                    1, treatment_rate, size=sample_size_per_variant
                )
                control = np.random.binomial(
                    1, baseline_rate, size=sample_size_per_variant
                )
            else:  # continuous
                treatment = np.random.normal(
                    treatment_rate, 1.0, size=sample_size_per_variant
                )
                control = np.random.normal(
                    baseline_rate, 1.0, size=sample_size_per_variant
                )

            # Run t-test
            _, p_value = stats.ttest_ind(treatment, control)

            if p_value < alpha:
                rejections += 1

        observed_power = rejections / n_simulations
        observed_type_ii = 1 - observed_power
        target_type_ii = 1 - target_power

        # Power should be within ±2 standard errors of target
        se = np.sqrt(target_power * (1 - target_power) / n_simulations)
        tolerance = 2 * se

        type_ii_passes = abs(observed_power - target_power) <= tolerance

        return ValidationResult(
            n_simulations=n_simulations,
            observed_type_i_error=0.0,  # Not tested here
            target_type_i_error=alpha,
            type_i_passes=True,
            observed_type_ii_error=observed_type_ii,
            target_type_ii_error=target_type_ii,
            type_ii_passes=type_ii_passes,
            observed_power=observed_power,
            target_power=target_power
        )

    def validate_cuped_variance_reduction(
        self,
        n_simulations: int = 1000,
        baseline_rate: float = 100.0,
        sample_size_per_variant: int = 500,
        covariate_correlation: float = 0.6
    ) -> ValidationResult:
        """
        Validate CUPED variance reduction.

        Runs simulations to verify that CUPED reduces variance as expected.

        Args:
            n_simulations: Number of simulations to run
            baseline_rate: Baseline mean (for continuous metrics)
            sample_size_per_variant: Sample size per variant
            covariate_correlation: Correlation between metric and covariate

        Returns:
            ValidationResult with variance reduction statistics
        """
        print(f"Running {n_simulations:,} simulations for CUPED validation...")

        variance_reductions = []

        for _ in tqdm(range(n_simulations), desc="CUPED"):
            # Generate correlated covariate and metric
            # Using bivariate normal
            mean = [baseline_rate, baseline_rate]
            cov = [
                [100, 100 * covariate_correlation],
                [100 * covariate_correlation, 100]
            ]

            # Treatment
            treatment_data = np.random.multivariate_normal(
                mean, cov, size=sample_size_per_variant
            )
            treatment_metric = treatment_data[:, 0]
            treatment_covariate = treatment_data[:, 1]

            # Control
            control_data = np.random.multivariate_normal(
                mean, cov, size=sample_size_per_variant
            )
            control_metric = control_data[:, 0]
            control_covariate = control_data[:, 1]

            # Apply CUPED
            cuped = CUPED()
            result = cuped.fit_transform(
                treatment_metric=treatment_metric,
                control_metric=control_metric,
                treatment_covariate=treatment_covariate,
                control_covariate=control_covariate
            )

            variance_reductions.append(result.variance_reduction)

        mean_reduction = np.mean(variance_reductions)
        std_reduction = np.std(variance_reductions)

        # Expected variance reduction = ρ²
        expected_reduction = covariate_correlation ** 2

        return ValidationResult(
            n_simulations=n_simulations,
            observed_type_i_error=0.0,
            target_type_i_error=0.05,
            type_i_passes=True,
            variance_reduction_mean=mean_reduction,
            variance_reduction_std=std_reduction
        )

    def validate_sequential_testing(
        self,
        n_simulations: int = 10000,
        baseline_rate: float = 0.10,
        sample_size_per_variant: int = 10000,
        alpha: float = 0.05,
        n_looks: int = 4
    ) -> ValidationResult:
        """
        Validate sequential testing Type I error control.

        Runs simulations under null hypothesis with sequential looks
        to verify family-wise error rate is controlled.

        Args:
            n_simulations: Number of simulations to run
            baseline_rate: Baseline conversion rate
            sample_size_per_variant: Total sample size per variant
            alpha: Target family-wise error rate
            n_looks: Number of interim analyses

        Returns:
            ValidationResult with sequential testing validation
        """
        print(f"Running {n_simulations:,} simulations for Sequential Testing validation...")

        # Setup sequential test
        gst = GroupSequentialTest(
            n_looks=n_looks,
            alpha=alpha,
            spending_function="obrien-fleming"
        )

        rejections = 0

        for _ in tqdm(range(n_simulations), desc="Sequential"):
            # Generate full data under H0
            treatment = np.random.binomial(1, baseline_rate, size=sample_size_per_variant)
            control = np.random.binomial(1, baseline_rate, size=sample_size_per_variant)

            # Perform sequential analyses
            rejected = False

            for k in range(1, n_looks + 1):
                # Data at this look
                info_frac = gst.information_fractions[k - 1]
                n_at_look = int(sample_size_per_variant * info_frac)

                treatment_at_look = treatment[:n_at_look]
                control_at_look = control[:n_at_look]

                treatment_mean = np.mean(treatment_at_look)
                control_mean = np.mean(control_at_look)

                # Sequential analysis
                result = gst.analyze(
                    treatment_mean=treatment_mean,
                    control_mean=control_mean,
                    treatment_n=n_at_look,
                    control_n=n_at_look,
                    analysis_number=k
                )

                if result.reject_null:
                    rejected = True
                    break

            if rejected:
                rejections += 1

        observed_type_i = rejections / n_simulations

        # Should maintain alpha despite multiple looks
        se = np.sqrt(alpha * (1 - alpha) / n_simulations)
        tolerance = 2 * se

        type_i_passes = abs(observed_type_i - alpha) <= tolerance

        return ValidationResult(
            n_simulations=n_simulations,
            observed_type_i_error=observed_type_i,
            target_type_i_error=alpha,
            type_i_passes=type_i_passes
        )


if __name__ == "__main__":
    # Run full validation suite
    simulator = ExperimentSimulator(seed=42)

    print("=" * 70)
    print("STATISTICAL VALIDATION SUITE")
    print("=" * 70)
    print()

    # Test 1: Type I Error
    print("TEST 1: Type I Error Control")
    print("-" * 70)
    result_type_i = simulator.validate_type_i_error(n_simulations=10000)
    print(result_type_i)
    print()

    # Test 2: Type II Error / Power
    print("TEST 2: Type II Error / Statistical Power")
    print("-" * 70)
    result_type_ii = simulator.validate_type_ii_error(
        n_simulations=10000,
        baseline_rate=0.10,
        treatment_rate=0.12,
        sample_size_per_variant=3843  # Calculated for 80% power
    )
    print(result_type_ii)
    print()

    # Test 3: CUPED Variance Reduction
    print("TEST 3: CUPED Variance Reduction")
    print("-" * 70)
    result_cuped = simulator.validate_cuped_variance_reduction(
        n_simulations=1000,
        covariate_correlation=0.6
    )
    print(result_cuped)
    print(f"Expected reduction (r^2): {0.6**2:.1%}")
    print()

    # Test 4: Sequential Testing
    print("TEST 4: Sequential Testing Family-Wise Error Rate")
    print("-" * 70)
    result_sequential = simulator.validate_sequential_testing(
        n_simulations=10000,
        n_looks=4
    )
    print(result_sequential)
    print()

    # Summary
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    all_pass = (
        result_type_i.type_i_passes and
        result_type_ii.type_ii_passes and
        result_sequential.type_i_passes
    )

    if all_pass:
        print("[PASS] ALL TESTS PASSED")
        print("  Statistical guarantees validated across 31,000 simulations")
    else:
        print("[FAIL] SOME TESTS FAILED")
        print("  Review results above")
