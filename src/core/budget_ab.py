"""
Budget A/B Testing Framework
==============================

Fair budget allocation for paid acquisition A/B tests.
Implements DoorDash's Budget A/B methodology.

The Problem:
- Equal traffic splits (50/50) ≠ equal budget allocation in paid acquisition
- Example: If treatment bids higher CPCs, 50/50 traffic → 70/30 budget split
- This creates unfair comparisons and biased results

The Solution:
- Track spend in real-time per variant
- Throttle higher-spending variant to maintain budget parity
- Ensure statistical validity while controlling costs

Author: Michael Robins
Date: January 31, 2026
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import warnings


@dataclass
class BudgetConfig:
    """Configuration for budget A/B test."""
    total_budget: float  # Total experiment budget ($)
    budget_per_variant: float  # Budget allocated per variant ($)
    max_imbalance: float = 0.10  # Max allowed budget imbalance (10% = 0.10)
    throttle_threshold: float = 0.08  # Throttle when imbalance > 8%
    check_frequency_minutes: int = 60  # Check budget every N minutes

    def __post_init__(self):
        """Validate configuration."""
        if self.budget_per_variant * 2 > self.total_budget:
            warnings.warn(
                f"Budget per variant ({self.budget_per_variant * 2}) exceeds "
                f"total budget ({self.total_budget})"
            )


@dataclass
class BudgetMetrics:
    """Real-time budget tracking metrics."""
    variant_name: str
    spend: float = 0.0
    impressions: int = 0
    clicks: int = 0
    conversions: int = 0
    throttle_rate: float = 1.0  # 1.0 = no throttling, 0.5 = 50% throttled
    is_throttled: bool = False
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def cpc(self) -> float:
        """Cost per click."""
        return self.spend / self.clicks if self.clicks > 0 else 0.0

    @property
    def cpa(self) -> float:
        """Cost per acquisition."""
        return self.spend / self.conversions if self.conversions > 0 else 0.0

    @property
    def conversion_rate(self) -> float:
        """Conversion rate."""
        return self.conversions / self.clicks if self.clicks > 0 else 0.0


@dataclass
class BudgetABResult:
    """Results from budget A/B test."""
    control_metrics: BudgetMetrics
    treatment_metrics: BudgetMetrics
    budget_config: BudgetConfig
    total_spend: float
    budget_utilization: float  # % of budget used
    budget_imbalance: float  # Absolute difference in spend (%)
    is_fair: bool  # Budget imbalance within acceptable range
    throttle_events: List[Dict] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"BudgetABResult(\n"
            f"  Control: ${self.control_metrics.spend:,.2f} "
            f"({self.control_metrics.conversions} conv, "
            f"throttle={self.control_metrics.throttle_rate:.1%})\n"
            f"  Treatment: ${self.treatment_metrics.spend:,.2f} "
            f"({self.treatment_metrics.conversions} conv, "
            f"throttle={self.treatment_metrics.throttle_rate:.1%})\n"
            f"  Budget Imbalance: {self.budget_imbalance:.1%}\n"
            f"  Fair: {self.is_fair}\n"
            f")"
        )


class BudgetABController:
    """
    Controller for budget-constrained A/B testing.

    Maintains fair budget allocation across variants by:
    1. Tracking spend in real-time
    2. Computing budget imbalance
    3. Throttling higher-spending variant when needed
    4. Recording throttle events for analysis

    Example:
        >>> config = BudgetConfig(
        ...     total_budget=10000,
        ...     budget_per_variant=5000,
        ...     max_imbalance=0.10
        ... )
        >>> controller = BudgetABController(config)
        >>>
        >>> # Record events
        >>> controller.record_event("control", cost=2.50, conversion=True)
        >>> controller.record_event("treatment", cost=3.75, conversion=True)
        >>>
        >>> # Check throttling
        >>> throttle_rates = controller.get_throttle_rates()
        >>> print(f"Treatment throttle: {throttle_rates['treatment']:.1%}")
    """

    def __init__(self, config: BudgetConfig):
        """
        Initialize budget A/B controller.

        Args:
            config: BudgetConfig with budget constraints
        """
        self.config = config

        # Initialize metrics for both variants
        self.control_metrics = BudgetMetrics(variant_name="control")
        self.treatment_metrics = BudgetMetrics(variant_name="treatment")

        # Throttle event log
        self.throttle_events: List[Dict] = []

    def record_event(
        self,
        variant: str,
        cost: float,
        impression: bool = False,
        click: bool = False,
        conversion: bool = False
    ) -> None:
        """
        Record a paid acquisition event.

        Args:
            variant: "control" or "treatment"
            cost: Cost of this event ($)
            impression: Whether this was an impression
            click: Whether this was a click
            conversion: Whether this was a conversion
        """
        metrics = self._get_metrics(variant)

        # Update spend
        metrics.spend += cost

        # Update counts
        if impression:
            metrics.impressions += 1
        if click:
            metrics.clicks += 1
        if conversion:
            metrics.conversions += 1

        metrics.timestamp = datetime.now()

        # Check if we need to update throttle rates
        self._update_throttle_rates()

    def _get_metrics(self, variant: str) -> BudgetMetrics:
        """Get metrics for specified variant."""
        if variant == "control":
            return self.control_metrics
        elif variant == "treatment":
            return self.treatment_metrics
        else:
            raise ValueError(f"Unknown variant: {variant}")

    def get_budget_imbalance(self) -> float:
        """
        Calculate budget imbalance between variants.

        Returns:
            Budget imbalance as fraction (e.g., 0.15 = 15% imbalance)

        Formula:
            imbalance = |spend_control - spend_treatment| / avg_spend
        """
        control_spend = self.control_metrics.spend
        treatment_spend = self.treatment_metrics.spend

        if control_spend == 0 and treatment_spend == 0:
            return 0.0

        avg_spend = (control_spend + treatment_spend) / 2

        if avg_spend == 0:
            return 0.0

        imbalance = abs(control_spend - treatment_spend) / avg_spend

        return imbalance

    def _update_throttle_rates(self) -> None:
        """
        Update throttle rates based on budget imbalance.

        Throttling Logic:
        1. If imbalance < threshold: No throttling (both at 100%)
        2. If imbalance >= threshold: Throttle higher-spending variant
        3. Throttle rate proportional to imbalance severity
        """
        imbalance = self.get_budget_imbalance()

        # No throttling needed
        if imbalance < self.config.throttle_threshold:
            self.control_metrics.is_throttled = False
            self.treatment_metrics.is_throttled = False
            self.control_metrics.throttle_rate = 1.0
            self.treatment_metrics.throttle_rate = 1.0
            return

        # Determine which variant is spending more
        control_spend = self.control_metrics.spend
        treatment_spend = self.treatment_metrics.spend

        if treatment_spend > control_spend:
            # Throttle treatment
            higher_spender = self.treatment_metrics
            lower_spender = self.control_metrics
            higher_variant = "treatment"
        else:
            # Throttle control
            higher_spender = self.control_metrics
            lower_spender = self.treatment_metrics
            higher_variant = "control"

        # Calculate throttle rate
        # Linear throttle: more imbalance → more throttling
        # throttle_rate = 1 - min(imbalance / max_imbalance, 1.0)
        throttle_intensity = min(
            (imbalance - self.config.throttle_threshold) /
            (self.config.max_imbalance - self.config.throttle_threshold),
            1.0
        )

        # Throttle between 50% and 100% (never fully stop)
        throttle_rate = 1.0 - (0.5 * throttle_intensity)

        # Apply throttle
        higher_spender.throttle_rate = throttle_rate
        higher_spender.is_throttled = True
        lower_spender.throttle_rate = 1.0
        lower_spender.is_throttled = False

        # Log throttle event
        self.throttle_events.append({
            'timestamp': datetime.now(),
            'throttled_variant': higher_variant,
            'throttle_rate': throttle_rate,
            'budget_imbalance': imbalance,
            'control_spend': control_spend,
            'treatment_spend': treatment_spend
        })

    def get_throttle_rates(self) -> Dict[str, float]:
        """
        Get current throttle rates for both variants.

        Returns:
            Dictionary with throttle rates (1.0 = no throttling)
        """
        return {
            'control': self.control_metrics.throttle_rate,
            'treatment': self.treatment_metrics.throttle_rate
        }

    def should_serve(self, variant: str) -> bool:
        """
        Determine if request should be served to variant (throttle-aware).

        Uses probabilistic throttling: throttle_rate = probability of serving.

        Args:
            variant: "control" or "treatment"

        Returns:
            True if request should be served, False if throttled
        """
        metrics = self._get_metrics(variant)

        # If not throttled, always serve
        if not metrics.is_throttled:
            return True

        # Probabilistic throttle: random draw against throttle_rate
        return np.random.random() < metrics.throttle_rate

    def is_budget_fair(self) -> bool:
        """
        Check if budget allocation is fair.

        Returns:
            True if imbalance within acceptable range
        """
        imbalance = self.get_budget_imbalance()
        return imbalance <= self.config.max_imbalance

    def get_result(self) -> BudgetABResult:
        """
        Get final experiment result with budget metrics.

        Returns:
            BudgetABResult with complete metrics
        """
        total_spend = self.control_metrics.spend + self.treatment_metrics.spend
        budget_utilization = total_spend / self.config.total_budget
        budget_imbalance = self.get_budget_imbalance()
        is_fair = self.is_budget_fair()

        return BudgetABResult(
            control_metrics=self.control_metrics,
            treatment_metrics=self.treatment_metrics,
            budget_config=self.config,
            total_spend=total_spend,
            budget_utilization=budget_utilization,
            budget_imbalance=budget_imbalance,
            is_fair=is_fair,
            throttle_events=self.throttle_events
        )

    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics for both variants.

        Returns:
            Dictionary with key metrics
        """
        return {
            'control': {
                'spend': self.control_metrics.spend,
                'conversions': self.control_metrics.conversions,
                'clicks': self.control_metrics.clicks,
                'cpc': self.control_metrics.cpc,
                'cpa': self.control_metrics.cpa,
                'conversion_rate': self.control_metrics.conversion_rate,
                'throttle_rate': self.control_metrics.throttle_rate
            },
            'treatment': {
                'spend': self.treatment_metrics.spend,
                'conversions': self.treatment_metrics.conversions,
                'clicks': self.treatment_metrics.clicks,
                'cpc': self.treatment_metrics.cpc,
                'cpa': self.treatment_metrics.cpa,
                'conversion_rate': self.treatment_metrics.conversion_rate,
                'throttle_rate': self.treatment_metrics.throttle_rate
            },
            'budget_imbalance': self.get_budget_imbalance(),
            'is_fair': self.is_budget_fair(),
            'throttle_event_count': len(self.throttle_events)
        }


def calculate_fair_budget_split(
    control_cpc: float,
    treatment_cpc: float,
    total_budget: float
) -> Tuple[float, float]:
    """
    Calculate fair budget split given different CPCs.

    If treatment has higher CPC, needs lower traffic to maintain budget parity.

    Args:
        control_cpc: Control variant cost per click
        treatment_cpc: Treatment variant cost per click
        total_budget: Total budget to split

    Returns:
        Tuple of (control_budget, treatment_budget)

    Example:
        >>> # Treatment CPC is 50% higher
        >>> control_budget, treatment_budget = calculate_fair_budget_split(
        ...     control_cpc=2.00,
        ...     treatment_cpc=3.00,
        ...     total_budget=10000
        ... )
        >>> print(f"Control: ${control_budget:.2f}, Treatment: ${treatment_budget:.2f}")
        Control: $5000.00, Treatment: $5000.00
    """
    # For budget parity, split budget equally
    # (Not traffic, which would be proportional to inverse CPC)
    control_budget = total_budget / 2
    treatment_budget = total_budget / 2

    return control_budget, treatment_budget


if __name__ == "__main__":
    # Example: Budget A/B test simulation
    np.random.seed(42)

    config = BudgetConfig(
        total_budget=10000,
        budget_per_variant=5000,
        max_imbalance=0.10,
        throttle_threshold=0.08
    )

    controller = BudgetABController(config)

    # Simulate 1000 events
    # Treatment has 50% higher CPC (more expensive)
    control_cpc = 2.00
    treatment_cpc = 3.00

    print("Simulating Budget A/B Test...")
    print(f"Control CPC: ${control_cpc:.2f}")
    print(f"Treatment CPC: ${treatment_cpc:.2f}")
    print()

    for i in range(1000):
        # Random assignment (50/50 before throttling)
        variant = "control" if np.random.random() < 0.5 else "treatment"

        # Check throttle
        if not controller.should_serve(variant):
            continue  # Skip this request (throttled)

        # Determine cost
        if variant == "control":
            cost = control_cpc
            conversion = np.random.random() < 0.10  # 10% CVR
        else:
            cost = treatment_cpc
            conversion = np.random.random() < 0.12  # 12% CVR (better)

        # Record event
        controller.record_event(
            variant=variant,
            cost=cost,
            click=True,
            conversion=conversion
        )

    # Get results
    result = controller.get_result()
    print(result)
    print()

    stats = controller.get_summary_stats()
    print("Summary Statistics:")
    print(f"Control CPA: ${stats['control']['cpa']:.2f}")
    print(f"Treatment CPA: ${stats['treatment']['cpa']:.2f}")
    print(f"Budget Imbalance: {stats['budget_imbalance']:.1%}")
    print(f"Fair: {stats['is_fair']}")
    print(f"Throttle Events: {stats['throttle_event_count']}")
