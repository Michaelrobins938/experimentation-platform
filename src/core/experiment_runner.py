"""
Experiment Runner
=================

Main interface for running A/B tests with power analysis, CUPED, Budget A/B,
and sequential testing.

Author: Michael Robins
Date: January 31, 2026
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass, field
from datetime import datetime

from power_analysis import PowerAnalyzer, PowerAnalysisResult
from cuped import CUPED, CUPEDResult
from budget_ab import BudgetABController, BudgetConfig, BudgetABResult
from sequential import GroupSequentialTest, SequentialTestResult


@dataclass
class ExperimentConfig:
    """Configuration for an A/B test experiment."""
    name: str
    metric_name: str
    metric_type: Literal["proportion", "continuous"]

    # Power analysis
    baseline_rate: float
    mde: float  # Minimum detectable effect
    alpha: float = 0.05
    power: float = 0.80

    # Variance reduction
    use_cuped: bool = False
    covariate_name: Optional[str] = None

    # Budget testing
    use_budget_ab: bool = False
    budget_config: Optional[BudgetConfig] = None

    # Sequential testing
    use_sequential: bool = False
    n_looks: int = 4
    information_fractions: Optional[List[float]] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ExperimentResult:
    """Results from completed experiment."""
    config: ExperimentConfig

    # Sample data
    treatment_n: int
    control_n: int
    treatment_mean: float
    control_mean: float
    treatment_std: float
    control_std: float

    # Statistical test
    test_statistic: float
    p_value: float
    confidence_interval: tuple
    reject_null: bool

    # Effect size
    absolute_lift: float
    relative_lift: float

    # Optional components
    power_analysis: Optional[PowerAnalysisResult] = None
    cuped_result: Optional[CUPEDResult] = None
    budget_result: Optional[BudgetABResult] = None
    sequential_result: Optional[SequentialTestResult] = None

    completed_at: datetime = field(default_factory=datetime.now)

    def __repr__(self) -> str:
        return (
            f"ExperimentResult(\n"
            f"  name={self.config.name}\n"
            f"  reject_null={self.reject_null}\n"
            f"  p_value={self.p_value:.4f}\n"
            f"  absolute_lift={self.absolute_lift:.4f}\n"
            f"  relative_lift={self.relative_lift:.2%}\n"
            f"  treatment_n={self.treatment_n:,}, control_n={self.control_n:,}\n"
            f")"
        )


class ExperimentRunner:
    """
    Main interface for running A/B tests.

    Combines power analysis, CUPED variance reduction, budget testing,
    and sequential testing into a unified framework.

    Example:
        >>> # Configure experiment
        >>> config = ExperimentConfig(
        ...     name="Homepage CTA Test",
        ...     metric_name="conversion_rate",
        ...     metric_type="proportion",
        ...     baseline_rate=0.10,
        ...     mde=0.02,
        ...     use_cuped=True
        ... )
        >>>
        >>> # Run experiment
        >>> runner = ExperimentRunner(config)
        >>> result = runner.run(
        ...     treatment_data=treatment_conversions,
        ...     control_data=control_conversions,
        ...     treatment_covariate=pre_treatment_conversions,
        ...     control_covariate=pre_control_conversions
        ... )
        >>> print(result)
    """

    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment runner.

        Args:
            config: ExperimentConfig with experiment parameters
        """
        self.config = config

        # Initialize components
        self.power_analyzer = PowerAnalyzer()

        if config.use_cuped:
            self.cuped = CUPED()
        else:
            self.cuped = None

        if config.use_budget_ab and config.budget_config:
            self.budget_controller = BudgetABController(config.budget_config)
        else:
            self.budget_controller = None

        if config.use_sequential:
            self.sequential_test = GroupSequentialTest(
                n_looks=config.n_looks,
                alpha=config.alpha,
                information_fractions=config.information_fractions
            )
        else:
            self.sequential_test = None

    def run_power_analysis(self) -> PowerAnalysisResult:
        """
        Run power analysis to determine required sample size.

        Returns:
            PowerAnalysisResult with sample size requirements
        """
        result = self.power_analyzer.calculate_sample_size(
            baseline_rate=self.config.baseline_rate,
            mde=self.config.mde,
            alpha=self.config.alpha,
            power=self.config.power,
            metric_type=self.config.metric_type
        )

        return result

    def run(
        self,
        treatment_data: np.ndarray,
        control_data: np.ndarray,
        treatment_covariate: Optional[np.ndarray] = None,
        control_covariate: Optional[np.ndarray] = None,
        analysis_number: Optional[int] = None
    ) -> ExperimentResult:
        """
        Run experiment analysis.

        Args:
            treatment_data: Treatment group metric values
            control_data: Control group metric values
            treatment_covariate: Pre-experiment covariate for treatment (CUPED)
            control_covariate: Pre-experiment covariate for control (CUPED)
            analysis_number: Analysis number for sequential testing (optional)

        Returns:
            ExperimentResult with statistical analysis
        """
        # Apply CUPED if enabled
        if self.config.use_cuped:
            if treatment_covariate is None or control_covariate is None:
                raise ValueError("Covariates required when use_cuped=True")

            cuped_result = self.cuped.fit_transform(
                treatment_metric=treatment_data,
                control_metric=control_data,
                treatment_covariate=treatment_covariate,
                control_covariate=control_covariate
            )

            # Use CUPED-adjusted metrics
            n_treatment = len(treatment_data)
            treatment_adjusted = cuped_result.adjusted_metric[:n_treatment]
            control_adjusted = cuped_result.adjusted_metric[n_treatment:]
        else:
            treatment_adjusted = treatment_data
            control_adjusted = control_data
            cuped_result = None

        # Compute statistics
        treatment_n = len(treatment_adjusted)
        control_n = len(control_adjusted)

        treatment_mean = np.mean(treatment_adjusted)
        control_mean = np.mean(control_adjusted)

        treatment_std = np.std(treatment_adjusted, ddof=1)
        control_std = np.std(control_adjusted, ddof=1)

        # Statistical test
        if self.config.use_sequential and analysis_number is not None:
            # Sequential testing
            sequential_result = self.sequential_test.analyze(
                treatment_mean=treatment_mean,
                control_mean=control_mean,
                treatment_n=treatment_n,
                control_n=control_n,
                analysis_number=analysis_number,
                treatment_std=treatment_std,
                control_std=control_std
            )

            test_statistic = sequential_result.z_statistic
            p_value = sequential_result.adjusted_p_value
            reject_null = sequential_result.reject_null
        else:
            # Standard t-test
            t_stat, p_value = stats.ttest_ind(treatment_adjusted, control_adjusted)
            test_statistic = t_stat
            reject_null = p_value < self.config.alpha
            sequential_result = None

        # Confidence interval (95%)
        pooled_std = np.sqrt(
            ((treatment_n - 1) * treatment_std**2 + (control_n - 1) * control_std**2) /
            (treatment_n + control_n - 2)
        )
        se = pooled_std * np.sqrt(1/treatment_n + 1/control_n)
        ci_margin = stats.t.ppf(1 - self.config.alpha/2, treatment_n + control_n - 2) * se

        absolute_lift = treatment_mean - control_mean
        confidence_interval = (
            absolute_lift - ci_margin,
            absolute_lift + ci_margin
        )

        # Effect sizes
        relative_lift = absolute_lift / control_mean if control_mean != 0 else 0.0

        # Power analysis
        power_analysis = self.run_power_analysis()

        # Budget result (if applicable)
        budget_result = None
        if self.budget_controller:
            budget_result = self.budget_controller.get_result()

        return ExperimentResult(
            config=self.config,
            treatment_n=treatment_n,
            control_n=control_n,
            treatment_mean=treatment_mean,
            control_mean=control_mean,
            treatment_std=treatment_std,
            control_std=control_std,
            test_statistic=test_statistic,
            p_value=p_value,
            confidence_interval=confidence_interval,
            reject_null=reject_null,
            absolute_lift=absolute_lift,
            relative_lift=relative_lift,
            power_analysis=power_analysis,
            cuped_result=cuped_result,
            budget_result=budget_result,
            sequential_result=sequential_result
        )


if __name__ == "__main__":
    # Example: Run experiment with CUPED
    np.random.seed(42)

    # Configure experiment
    config = ExperimentConfig(
        name="Homepage CTA Button Test",
        metric_name="conversion_rate",
        metric_type="proportion",
        baseline_rate=0.10,
        mde=0.02,  # Detect 2pp lift
        use_cuped=True
    )

    runner = ExperimentRunner(config)

    # Power analysis
    power_result = runner.run_power_analysis()
    print("Power Analysis:")
    print(power_result)
    print()

    # Simulate experiment data
    n = power_result.sample_size_per_variant

    # Pre-experiment covariate (30-day prior conversion)
    pre_treatment = np.random.binomial(1, 0.10, size=n)
    pre_control = np.random.binomial(1, 0.10, size=n)

    # Post-experiment data (correlated with pre-experiment)
    # Treatment has 2pp lift (10% → 12%)
    post_treatment = np.random.binomial(1, 0.12, size=n)
    post_control = np.random.binomial(1, 0.10, size=n)

    # Add correlation with pre-experiment
    # (Users who converted before more likely to convert again)
    post_treatment = np.where(
        pre_treatment == 1,
        np.random.binomial(1, 0.15, size=n),  # Higher rate for repeat converters
        post_treatment
    )

    # Run experiment
    result = runner.run(
        treatment_data=post_treatment.astype(float),
        control_data=post_control.astype(float),
        treatment_covariate=pre_treatment.astype(float),
        control_covariate=pre_control.astype(float)
    )

    print("Experiment Result:")
    print(result)
    print()

    if result.cuped_result:
        print("CUPED Variance Reduction:")
        print(result.cuped_result)
        print()

    print(f"Decision: {'LAUNCH' if result.reject_null else 'NO LAUNCH'}")
    print(f"Confidence Interval: [{result.confidence_interval[0]:.4f}, "
          f"{result.confidence_interval[1]:.4f}]")
