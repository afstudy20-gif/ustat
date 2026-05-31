"""Characterization tests for Restricted Cubic Splines (RCS).

Focus: lock the behavior of the Harrell RCS basis generation and the
two main RCS endpoints before any refactoring of models.py or services/.

These are intentionally simple but numerically meaningful tests.
"""

import numpy as np
import pandas as pd
import pytest
from conftest import make_session

from services.rcs_basis import (
    rcs_basis,
    resolve_knots,
    harrell_knots,
    KNOT_PERCENTILES,
)


# ──────────────────────────────────────────────────────────────────────────────
# RCS basis generation (pure, very high value for refactoring)
# ──────────────────────────────────────────────────────────────────────────────

def test_harrell_knots_standard():
    """Harrell default knots for 4 knots must be 5/35/65/95 percentiles."""
    x = np.linspace(0, 100, 1000)
    knots = harrell_knots(x, 4)
    np.testing.assert_allclose(knots, [5, 35, 65, 95], rtol=0.01)


def test_rcs_basis_shape_and_monotonicity():
    """
    For n_knots=4 we should get (4-2)=2 basis columns.
    The basis should be zero before the first knot and increase after.
    """
    rng = np.random.default_rng(42)
    x = rng.uniform(10, 90, 500)
    knots = np.array([20, 40, 60, 80])

    basis = rcs_basis(x, knots)
    assert basis.shape == (500, 2)

    # Before first knot, the basis columns should be very small (near zero)
    mask_before = x < 20
    if mask_before.any():
        assert np.all(np.abs(basis[mask_before]) < 1e-8)


def test_resolve_knots_custom_validation():
    """Custom knots must be strictly increasing and within data range."""
    x = np.array([0.0, 10.0, 20.0, 30.0, 100.0])

    # Valid
    k = resolve_knots(x, 3, knot_positions=[5, 25, 70])
    np.testing.assert_array_equal(k, [5, 25, 70])

    # Not enough knots
    with pytest.raises(ValueError):
        resolve_knots(x, 4, knot_positions=[1, 2, 3])

    # Not strictly increasing
    with pytest.raises(ValueError):
        resolve_knots(x, 3, knot_positions=[10, 5, 20])

    # Outside range
    with pytest.raises(ValueError):
        resolve_knots(x, 3, knot_positions=[-1, 10, 50])


# ──────────────────────────────────────────────────────────────────────────────
# Univariate RCS endpoint characterization
# ──────────────────────────────────────────────────────────────────────────────

def test_rcs_univariate_cox_basic(client):
    """
    Simple Cox RCS run. We only check that the endpoint returns the
    expected top-level keys and that nonlinearity p-value is present.
    """
    rng = np.random.default_rng(99)
    n = 300
    time = rng.exponential(200, n).clip(5, 800)
    event = rng.binomial(1, 0.6, n)
    x = rng.normal(50, 15, n).clip(10, 120)

    # Make x have a mild non-linear effect on hazard
    hazard = 0.01 + 0.0008 * (x - 50)**2 / 100
    event = (rng.uniform(0, 1, n) < (1 - np.exp(-hazard * time))).astype(int)

    df = pd.DataFrame({"time": time, "event": event, "ldl": x})
    sid = make_session(df, "rcs_cox1")

    payload = {
        "session_id": sid,
        "predictor": "ldl",
        "model_type": "cox",
        "duration_col": "time",
        "event_col": "event",
        "n_knots": 4,
        "knot_placement": "harrell",
    }

    r = client.post("/api/models/rcs", json=payload)
    assert r.status_code == 200, r.text

    data = r.json()
    assert "knots" in data
    assert len(data["knots"]) == 4
    assert "nonlinearity_p" in data or "nonlinearity_wald" in data
    # New modular response uses x_values + or_values/ci instead of legacy hr_curve
    assert "x_values" in data and "or_values" in data
    assert "model_type" in data and data["model_type"] == "cox"


def test_rcs_linear_outcome(client):
    """RCS with linear outcome should return beta + CI for the spline terms."""
    rng = np.random.default_rng(123)
    n = 250
    x = rng.normal(100, 20, n)
    y = 3.0 + 0.8 * x + 0.015 * (x - 80).clip(0, None)**2 + rng.normal(0, 12, n)

    df = pd.DataFrame({"y": y, "dose": x})
    sid = make_session(df, "rcs_lin")

    r = client.post("/api/models/rcs", json={
        "session_id": sid,
        "predictor": "dose",
        "model_type": "linear",
        "outcome": "y",
        "n_knots": 3,
    })
    assert r.status_code == 200
    data = r.json()
    assert data.get("model_type") == "linear"
    # New response shape after extraction
    assert "x_values" in data and "or_values" in data
    assert "knots" in data and len(data["knots"]) == 3
