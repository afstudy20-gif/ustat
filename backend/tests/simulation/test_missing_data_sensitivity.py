"""
Phase 3 - Missing Data Sensitivity Simulation Tests
"""

import pytest
from services.missing_data_sensitivity import (
    simulate_missingness,
    delta_adjustment_sensitivity,
)
from services.simulation_generators import generate_logistic_data, generate_linear_data


@pytest.mark.simulation
def test_delta_adjustment_changes_estimate():
    """
    Basic sanity check: applying different delta values should move the estimate
    in a predictable direction for MNAR.
    """
    df, _ = generate_logistic_data(n=500, seed=123)

    # Create MAR missingness first
    df_miss = simulate_missingness(df, ["event", "X1", "X2"], mechanism="MAR", missing_rate=0.25)

    sens = delta_adjustment_sensitivity(
        df_miss,
        outcome="event",
        predictors=["X1", "X2"],
        model_type="logistic",
        delta_range=(-1.0, 1.0),
        n_steps=5,
    )

    results = sens["results"]
    assert len(results) == 5

    # Check that the log-odds for X1 changes across deltas
    log_odds_values = [r["log_odds"] for r in results if "log_odds" in r]
    assert len(log_odds_values) >= 3
    assert max(log_odds_values) - min(log_odds_values) > 0.1, \
        "Delta adjustment did not meaningfully change the coefficient"
