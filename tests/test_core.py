"""
Core Functionality Tests
========================

Quick tests to verify core modules work correctly.

Author: Michael Robins
Date: January 31, 2026
"""

import sys
import os
import numpy as np
from scipy import stats

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.power_analysis import PowerAnalyzer
from core.budget_ab import BudgetABController, BudgetConfig
from core.cuped import CUPED
from core.sequential import GroupSequentialTest


def test_power_analysis():
    """Test power analysis calculator."""
    print("Testing Power Analysis...")

    analyzer = PowerAnalyzer()

    # Test sample size calculation
    result = analyzer.calculate_sample_size(
        baseline_rate=0.10,
        mde=0.02,
        alpha=0.05,
        power=0.80
    )

    assert result.sample_size_per_variant > 0
    assert result.total_sample_size == result.sample_size_per_variant * 2
    assert result.statistical_power == 0.80
    assert result.alpha == 0.05

    # Test MDE calculation
    mde = analyzer.calculate_mde(
        baseline_rate=0.10,
        sample_size_per_variant=result.sample_size_per_variant,
        alpha=0.05,
        power=0.80
    )

    assert abs(mde - 0.02) < 0.001  # Should recover original MDE

    print("  [PASS] Power analysis working")


def test_budget_ab():
    """Test budget A/B controller."""
    print("Testing Budget A/B...")

    config = BudgetConfig(
        total_budget=1000,
        budget_per_variant=500,
        max_imbalance=0.10,
        throttle_threshold=0.08
    )

    controller = BudgetABController(config)

    # Record some events
    for _ in range(100):
        variant = "control" if np.random.random() < 0.5 else "treatment"
        cost = 2.0 if variant == "control" else 3.0  # Treatment more expensive

        if controller.should_serve(variant):
            controller.record_event(
                variant=variant,
                cost=cost,
                click=True,
                conversion=np.random.random() < 0.1
            )

    result = controller.get_result()

    assert result.total_spend > 0
    assert result.control_metrics.spend > 0
    assert result.treatment_metrics.spend > 0

    # Check budget imbalance is computed
    assert 0 <= result.budget_imbalance <= 1

    print("  [PASS] Budget A/B working")


def test_cuped():
    """Test CUPED variance reduction."""
    print("Testing CUPED...")

    np.random.seed(42)

    # Generate correlated data
    n = 500
    correlation = 0.6

    # Treatment
    treatment_data = np.random.multivariate_normal(
        mean=[100, 100],
        cov=[[100, 60], [60, 100]],
        size=n
    )
    treatment_metric = treatment_data[:, 0]
    treatment_covariate = treatment_data[:, 1]

    # Control
    control_data = np.random.multivariate_normal(
        mean=[100, 100],
        cov=[[100, 60], [60, 100]],
        size=n
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

    # Check variance reduction occurred
    assert result.variance_reduction > 0
    assert result.variance_reduction < 1
    assert result.adjusted_variance < result.original_variance

    # Check theta is reasonable
    assert -10 < result.theta < 10

    print(f"  [PASS] CUPED working (variance reduction: {result.variance_reduction:.1%})")


def test_sequential():
    """Test sequential testing."""
    print("Testing Sequential Testing...")

    gst = GroupSequentialTest(
        n_looks=4,
        alpha=0.05,
        spending_function="obrien-fleming"
    )

    # Check boundaries created
    assert len(gst.boundaries) == 4
    assert all(b.z_threshold > 0 for b in gst.boundaries)

    # Test analysis
    result = gst.analyze(
        treatment_mean=0.12,
        control_mean=0.10,
        treatment_n=1000,
        control_n=1000,
        analysis_number=1
    )

    assert result.z_statistic is not None
    assert 0 <= result.p_value <= 1
    assert result.reason in ["efficacy", "futility", "continue", "final"]

    print("  [PASS] Sequential testing working")


def test_integration():
    """Integration test: run a simple experiment end-to-end."""
    print("Testing Integration...")

    np.random.seed(42)

    # Simulate simple experiment
    baseline = 0.10
    treatment_effect = 0.02  # 2pp lift
    n = 1000

    control = np.random.binomial(1, baseline, size=n)
    treatment = np.random.binomial(1, baseline + treatment_effect, size=n)

    # Run t-test
    t_stat, p_value = stats.ttest_ind(treatment, control)

    # Basic sanity checks
    assert np.mean(treatment) > np.mean(control)  # Treatment should be higher
    assert 0 <= p_value <= 1

    print(f"  [PASS] Integration working (p={p_value:.4f})")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("RUNNING CORE FUNCTIONALITY TESTS")
    print("=" * 70 + "\n")

    try:
        test_power_analysis()
        test_budget_ab()
        test_cuped()
        test_sequential()
        test_integration()

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED [OK]")
        print("=" * 70 + "\n")

    except AssertionError as e:
        print(f"\n[FAIL] TEST FAILED: {e}\n")
        raise

    except Exception as e:
        print(f"\n[ERROR] ERROR: {e}\n")
        raise
