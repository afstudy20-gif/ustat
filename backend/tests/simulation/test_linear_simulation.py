"""
Example simulation-based tests for Linear Regression.

These tests generate data with known parameters and check that the
implemented methods recover them within reasonable tolerance.
"""

import numpy as np
import pytest

import sys
import os
import pytest

# Ensure project root is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tests.conftest import make_session


@pytest.mark.simulation
def test_linear_recovers_coefficients(client):
    """
    Basic simulation check: does the linear regression endpoint
    recover the true coefficients reasonably well?
    """
    # Lazy import to avoid collection-time import issues
    from services.simulation_generators import generate_linear_data

    df, truth = generate_linear_data(n=800, seed=42)
    sid = make_session(df, "sim_linear_1")

    payload = {
        "session_id": sid,
        "outcome": "y",
        "predictors": ["X1", "X2", "X3", "X4", "X5"],
    }

    r = client.post("/api/models/linear", json=payload)
    assert r.status_code == 200, r.text

    data = r.json()
    coefs = {c["variable"]: c["estimate"] for c in data["coefficients"] if c["variable"] != "const"}

    true_beta = np.array(truth["beta"])

    # Check recovery within reasonable tolerance (allows for noise)
    recovered = np.array([coefs.get(f"X{i+1}", 0.0) for i in range(5)])

    # Mean absolute error should be small
    mae = np.mean(np.abs(recovered - true_beta))
    assert mae < 0.25, f"MAE too high: {mae:.3f}"

    # Sign and rough magnitude should match for non-zero coefficients
    for i, beta in enumerate(true_beta):
        if abs(beta) > 0.3:
            est = coefs.get(f"X{i+1}", 0.0)
            assert np.sign(est) == np.sign(beta), f"Wrong sign for X{i+1}"
            assert abs(est) > 0.3 * abs(beta), f"Underestimated effect for X{i+1}"


@pytest.mark.simulation
def test_linear_r2_reasonable(client):
    """R² should be reasonably high when signal is strong."""
    df, _ = generate_linear_data(n=600, noise_sd=0.8, seed=123)
    sid = make_session(df, "sim_linear_r2")

    r = client.post("/api/models/linear", json={
        "session_id": sid,
        "outcome": "y",
        "predictors": ["X1", "X2", "X3", "X4", "X5"],
    })
    assert r.status_code == 200
    r2 = r.json()["r_squared"]
    assert r2 > 0.65, f"R² too low for strong signal: {r2:.3f}"
