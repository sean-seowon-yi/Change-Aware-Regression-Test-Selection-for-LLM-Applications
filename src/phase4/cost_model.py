"""CI/CD cost projections for CARTS.

Parameterized model that projects monthly LLM evaluation costs under
full-rerun vs. CARTS strategies for various team sizes and model tiers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CICDScenario:
    """A realistic CI/CD team configuration."""

    name: str
    suite_size: int
    prs_per_month: int
    cost_per_call: float


STANDARD_SCENARIOS: list[CICDScenario] = [
    CICDScenario("small_team_cheap", 100, 50, 0.001),
    CICDScenario("medium_team_cheap", 300, 100, 0.001),
    CICDScenario("large_team_standard", 300, 200, 0.01),
    CICDScenario("enterprise_premium", 500, 200, 0.05),
]


def project_savings(
    scenario: CICDScenario,
    call_reduction: float,
    escalation_rate: float,
) -> dict:
    """Compute monthly costs and savings for a CI/CD scenario.

    Args:
        scenario: Team configuration.
        call_reduction: Observed fraction of tests skipped (0–1).
        escalation_rate: Fraction of PRs where sentinel triggers full rerun (0–1).

    Returns:
        Dict with full_rerun_cost, carts_cost, savings_absolute,
        savings_pct, and calls_saved_per_month.
    """
    N = scenario.suite_size
    P = scenario.prs_per_month
    c = scenario.cost_per_call
    r = call_reduction
    e = escalation_rate

    full_cost = P * N * c
    carts_cost = P * N * c * (1 - r * (1 - e))
    savings = full_cost - carts_cost
    savings_pct = savings / full_cost if full_cost > 0 else 0.0
    calls_saved = P * N * r * (1 - e)

    return {
        "scenario": scenario.name,
        "suite_size": N,
        "prs_per_month": P,
        "cost_per_call": c,
        "call_reduction": round(r, 4),
        "escalation_rate": round(e, 4),
        "full_rerun_cost": round(full_cost, 2),
        "carts_cost": round(carts_cost, 2),
        "savings_absolute": round(savings, 2),
        "savings_pct": round(savings_pct, 4),
        "calls_saved_per_month": round(calls_saved, 0),
        "annual_savings": round(savings * 12, 2),
    }


def project_all_scenarios(
    call_reduction: float,
    escalation_rate: float,
    *,
    scenarios: list[CICDScenario] | None = None,
) -> list[dict]:
    """Run cost projections for all standard scenarios."""
    if scenarios is None:
        scenarios = STANDARD_SCENARIOS
    return [project_savings(s, call_reduction, escalation_rate) for s in scenarios]


def break_even_suite_size(
    cost_per_call: float,
    call_reduction: float,
    escalation_rate: float,
    *,
    overhead_seconds: float = 3.0,
    avg_call_latency_seconds: float = 2.0,
) -> float:
    """Minimum suite size where CARTS saves more time than it costs.

    The overhead is the per-PR time for change classification, feature
    extraction, and model inference.  The benefit is the time saved by
    not making ``N * r * (1 - e)`` LLM calls.

    Returns the break-even N (fractional; ceil for practical purposes).
    """
    time_saved_per_test = call_reduction * (1 - escalation_rate) * avg_call_latency_seconds
    if time_saved_per_test <= 0:
        return float("inf")
    return overhead_seconds / time_saved_per_test


def cost_vs_suite_size(
    call_reduction: float,
    escalation_rate: float,
    cost_per_call: float,
    prs_per_month: int,
    *,
    suite_sizes: list[int] | None = None,
) -> list[dict]:
    """Compute savings at varying suite sizes for a single cost tier.

    Useful for generating the F7 cost savings curve.
    """
    if suite_sizes is None:
        suite_sizes = [50, 100, 150, 200, 300, 400, 500, 750, 1000]
    results = []
    for N in suite_sizes:
        s = CICDScenario(f"N={N}", N, prs_per_month, cost_per_call)
        results.append(project_savings(s, call_reduction, escalation_rate))
    return results
