"""
Complete Experimentation Platform Example
==========================================

Demonstrates all features:
1. Power Analysis
2. Budget A/B Testing
3. CUPED Variance Reduction
4. Sequential Testing
5. Statistical Validation

Author: Michael Robins
Date: January 31, 2026
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.power_analysis import PowerAnalyzer
from core.budget_ab import BudgetABController, BudgetConfig
from core.cuped import CUPED
from core.sequential import GroupSequentialTest
from core.experiment_runner import ExperimentConfig, ExperimentRunner


def example_1_power_analysis():
    """Example 1: Power Analysis for sample size determination."""
    print("=" * 70)
    print("EXAMPLE 1: Power Analysis")
    print("=" * 70)
    print()

    analyzer = PowerAnalyzer()

    # Scenario: Homepage CTA button test
    # Want to detect 2pp lift from 10% baseline with 80% power
    result = analyzer.calculate_sample_size(
        baseline_rate=0.10,
        mde=0.02,  # 2 percentage points
        alpha=0.05,
        power=0.80,
        daily_traffic=5000
    )

    print("Scenario: Homepage CTA Button Color Test")
    print(f"Baseline conversion rate: {result.baseline_rate:.1%}")
    print(f"Minimum detectable effect: {result.minimum_detectable_effect:.1%}")
    print(f"Target power: {result.statistical_power:.0%}")
    print()
    print("Required Sample Size:")
    print(f"  Per variant: {result.sample_size_per_variant:,}")
    print(f"  Total: {result.total_sample_size:,}")
    print(f"  Duration: {result.duration_days:.1f} days")
    print()

    # What if we want 90% power instead?
    result_90 = analyzer.calculate_sample_size(
        baseline_rate=0.10,
        mde=0.02,
        power=0.90  # Higher power
    )

    print("With 90% power:")
    print(f"  Per variant: {result_90.sample_size_per_variant:,}")
    print(f"  Increase: {(result_90.sample_size_per_variant / result.sample_size_per_variant - 1):.0%}")
    print()


def example_2_budget_ab_testing():
    """Example 2: Budget A/B Testing for paid acquisition."""
    print("=" * 70)
    print("EXAMPLE 2: Budget A/B Testing")
    print("=" * 70)
    print()

    # Setup: Testing new Google Ads bidding strategy
    # Control: Current strategy (CPC = $2.00)
    # Treatment: New strategy (CPC = $3.00, better targeting)

    config = BudgetConfig(
        total_budget=10000,
        budget_per_variant=5000,
        max_imbalance=0.10,
        throttle_threshold=0.08
    )

    controller = BudgetABController(config)

    print("Scenario: Google Ads Bidding Strategy Test")
    print(f"Total budget: ${config.total_budget:,.2f}")
    print(f"Budget per variant: ${config.budget_per_variant:,.2f}")
    print(f"Max imbalance allowed: {config.max_imbalance:.0%}")
    print()

    # Simulate 2000 clicks
    np.random.seed(42)
    control_cpc = 2.00
    treatment_cpc = 3.00  # 50% higher (better targeting, higher quality)

    print("Simulating 2000 clicks...")
    print(f"Control CPC: ${control_cpc:.2f}")
    print(f"Treatment CPC: ${treatment_cpc:.2f}")
    print()

    for i in range(2000):
        # Random assignment
        variant = "control" if np.random.random() < 0.5 else "treatment"

        # Check throttle
        if not controller.should_serve(variant):
            continue

        # Event
        if variant == "control":
            cost = control_cpc
            conversion = np.random.random() < 0.10  # 10% CVR
        else:
            cost = treatment_cpc
            conversion = np.random.random() < 0.13  # 13% CVR (better targeting)

        controller.record_event(
            variant=variant,
            cost=cost,
            click=True,
            conversion=conversion
        )

    # Results
    result = controller.get_result()
    print("Results:")
    print(f"  Control spend: ${result.control_metrics.spend:,.2f}")
    print(f"  Treatment spend: ${result.treatment_metrics.spend:,.2f}")
    print(f"  Budget imbalance: {result.budget_imbalance:.1%}")
    print(f"  Is fair: {result.is_fair}")
    print()

    stats = controller.get_summary_stats()
    print("Performance Metrics:")
    print(f"  Control CPA: ${stats['control']['cpa']:.2f}")
    print(f"  Treatment CPA: ${stats['treatment']['cpa']:.2f}")
    print(f"  CPA improvement: {(1 - stats['treatment']['cpa'] / stats['control']['cpa']):.1%}")
    print()


def example_3_cuped_variance_reduction():
    """Example 3: CUPED for faster experiments."""
    print("=" * 70)
    print("EXAMPLE 3: CUPED Variance Reduction")
    print("=" * 70)
    print()

    np.random.seed(42)

    # Scenario: Testing email campaign
    # High variance in revenue, but strong correlation with past revenue

    n = 1000

    # Pre-experiment revenue (30 days prior)
    pre_revenue_treatment = np.random.lognormal(mean=4.0, sigma=1.0, size=n)
    pre_revenue_control = np.random.lognormal(mean=4.0, sigma=1.0, size=n)

    # Post-experiment revenue (correlated with pre-revenue)
    # Treatment has 15% lift
    post_revenue_treatment = pre_revenue_treatment * np.random.lognormal(
        mean=np.log(1.15), sigma=0.3, size=n
    )
    post_revenue_control = pre_revenue_control * np.random.lognormal(
        mean=0, sigma=0.3, size=n
    )

    print("Scenario: Email Campaign Revenue Test")
    print(f"Sample size: {n:,} per variant")
    print()

    # Without CUPED
    from scipy import stats as sp_stats
    t_stat, p_value_original = sp_stats.ttest_ind(
        post_revenue_treatment,
        post_revenue_control
    )

    print("Without CUPED:")
    print(f"  p-value: {p_value_original:.4f}")
    print(f"  Significant: {p_value_original < 0.05}")
    print()

    # With CUPED
    cuped = CUPED()
    cuped_result = cuped.fit_transform(
        treatment_metric=post_revenue_treatment,
        control_metric=post_revenue_control,
        treatment_covariate=pre_revenue_treatment,
        control_covariate=pre_revenue_control
    )

    treatment_adjusted = cuped_result.adjusted_metric[:n]
    control_adjusted = cuped_result.adjusted_metric[n:]

    t_stat_cuped, p_value_cuped = sp_stats.ttest_ind(
        treatment_adjusted,
        control_adjusted
    )

    print("With CUPED:")
    print(f"  p-value: {p_value_cuped:.4f}")
    print(f"  Significant: {p_value_cuped < 0.05}")
    print()

    print("CUPED Results:")
    print(f"  Variance reduction: {cuped_result.variance_reduction:.1%}")
    print(f"  Sensitivity improvement: {p_value_original / p_value_cuped:.1f}x")
    print(f"  Equivalent sample size gain: {1 / (1 - cuped_result.variance_reduction):.1f}x")
    print()


def example_4_sequential_testing():
    """Example 4: Sequential Testing for early stopping."""
    print("=" * 70)
    print("EXAMPLE 4: Sequential Testing")
    print("=" * 70)
    print()

    np.random.seed(42)

    # Setup sequential test with 4 looks
    gst = GroupSequentialTest(
        n_looks=4,
        alpha=0.05,
        spending_function="obrien-fleming"
    )

    print("Scenario: Product Page Redesign")
    print(gst.get_boundary_summary())
    print()

    # Simulate strong effect (will likely stop early)
    baseline_rate = 0.10
    treatment_rate = 0.14  # 4pp lift (strong effect)
    total_n = 10000

    print(f"True effect: {baseline_rate:.1%} → {treatment_rate:.1%}")
    print()

    for k in range(1, 5):
        info_frac = gst.information_fractions[k - 1]
        n_at_look = int(total_n * info_frac)

        # Simulate data
        control = np.random.binomial(1, baseline_rate, size=n_at_look)
        treatment = np.random.binomial(1, treatment_rate, size=n_at_look)

        control_mean = np.mean(control)
        treatment_mean = np.mean(treatment)

        # Analyze
        result = gst.analyze(
            treatment_mean=treatment_mean,
            control_mean=control_mean,
            treatment_n=n_at_look,
            control_n=n_at_look,
            analysis_number=k
        )

        print(f"Look {k} ({info_frac:.0%} of data, n={n_at_look:,}):")
        print(f"  Control: {control_mean:.2%}")
        print(f"  Treatment: {treatment_mean:.2%}")
        print(f"  Z-statistic: {result.z_statistic:.3f} (threshold: {result.boundary.z_threshold:.3f})")
        print(f"  Decision: {result.reason.upper()}")

        if result.stop_early:
            print()
            print(f"EARLY STOP at {info_frac:.0%}!")
            print(f"Sample size savings: {(1 - info_frac) * 100:.0f}%")
            print(f"Time savings: ~{(1 - info_frac) * 100:.0f}% of planned duration")
            break

        print()


def example_5_full_experiment():
    """Example 5: Full experiment using ExperimentRunner."""
    print("=" * 70)
    print("EXAMPLE 5: Complete Experiment with All Features")
    print("=" * 70)
    print()

    np.random.seed(42)

    # Configure experiment
    config = ExperimentConfig(
        name="Checkout Flow Optimization",
        metric_name="conversion_rate",
        metric_type="proportion",
        baseline_rate=0.08,  # 8% baseline
        mde=0.01,  # Detect 1pp lift
        alpha=0.05,
        power=0.80,
        use_cuped=True,
        covariate_name="prior_30d_purchases"
    )

    runner = ExperimentRunner(config)

    # Power analysis
    print("1. POWER ANALYSIS")
    print("-" * 70)
    power_result = runner.run_power_analysis()
    print(power_result)
    print()

    # Simulate data
    n = power_result.sample_size_per_variant

    # Pre-experiment covariate (prior 30-day purchases)
    pre_treatment = np.random.binomial(1, 0.08, size=n)
    pre_control = np.random.binomial(1, 0.08, size=n)

    # Post-experiment conversion (treatment has 1.5pp lift)
    post_treatment = np.random.binomial(1, 0.095, size=n)
    post_control = np.random.binomial(1, 0.080, size=n)

    # Add correlation with pre-experiment
    post_treatment = np.where(
        pre_treatment == 1,
        np.random.binomial(1, 0.12, size=n),
        post_treatment
    )

    # Run experiment
    print("2. EXPERIMENT RESULTS")
    print("-" * 70)
    result = runner.run(
        treatment_data=post_treatment.astype(float),
        control_data=post_control.astype(float),
        treatment_covariate=pre_treatment.astype(float),
        control_covariate=pre_control.astype(float)
    )

    print(f"Control conversion: {result.control_mean:.2%}")
    print(f"Treatment conversion: {result.treatment_mean:.2%}")
    print(f"Absolute lift: {result.absolute_lift:.2%}")
    print(f"Relative lift: {result.relative_lift:.1%}")
    print(f"p-value: {result.p_value:.4f}")
    print(f"95% CI: [{result.confidence_interval[0]:.3%}, {result.confidence_interval[1]:.3%}]")
    print()

    if result.cuped_result:
        print("3. CUPED VARIANCE REDUCTION")
        print("-" * 70)
        print(f"Variance reduction: {result.cuped_result.variance_reduction:.1%}")
        print(f"Correlation: {result.cuped_result.correlation:.3f}")
        print()

    print("4. DECISION")
    print("-" * 70)
    if result.reject_null:
        print("✓ LAUNCH - Statistically significant improvement detected")
    else:
        print("✗ NO LAUNCH - No significant difference detected")
    print()


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "EXPERIMENTATION PLATFORM DEMO" + " " * 24 + "║")
    print("║" + " " * 12 + "Production-Grade A/B Testing Framework" + " " * 17 + "║")
    print("╚" + "=" * 68 + "╝")
    print("\n")

    # Run all examples
    example_1_power_analysis()
    input("Press Enter to continue to Example 2...")
    print("\n")

    example_2_budget_ab_testing()
    input("Press Enter to continue to Example 3...")
    print("\n")

    example_3_cuped_variance_reduction()
    input("Press Enter to continue to Example 4...")
    print("\n")

    example_4_sequential_testing()
    input("Press Enter to continue to Example 5...")
    print("\n")

    example_5_full_experiment()

    print("\n")
    print("=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print()
    print("All features demonstrated:")
    print("  ✓ Power Analysis")
    print("  ✓ Budget A/B Testing")
    print("  ✓ CUPED Variance Reduction")
    print("  ✓ Sequential Testing")
    print("  ✓ Complete Experiment Runner")
    print()
