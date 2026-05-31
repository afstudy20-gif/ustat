"""
Simulation checks for Logistic Regression.
"""

import numpy as np
import pytest

import sys
import os
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tests.conftest import make_session


@pytest.mark.simulation
def test_logistic_recovers_log_odds(client):
    """Basic check that logistic regression recovers direction and rough magnitude."""
    from services.simulation_generators import generate_logistic_data

    df, truth = generate_logistic_data(n=1200, seed=99)
    sid = make_session(df, "sim_logistic_1")

    r = client.post("/api/models/logistic", json={
        "session_id": sid,
        "outcome": "event",
        "predictors": ["X1", "X2", "X3", "X4"],
    })
    assert r.status_code == 200, r.text

    data = r.json()
    coefs = {c["variable"]: c["log_odds"] for c in data["coefficients"] if c["variable"] != "const"}

    true_beta = np.array(truth["beta"])
    recovered = np.array([coefs.get(f"X{i+1}", 0.0) for i in range(4)])

    # Direction should mostly match
    sign_agreement = np.sum(np.sign(recovered) == np.sign(true_beta))
    assert sign_agreement >= 3, "Too many sign errors in logistic coefficients"

    # Magnitude check for stronger effects
    for i, beta in enumerate(true_beta):
        if abs(beta) > 0.6:
            est = coefs.get(f"X{i+1}", 0.0)
            assert abs(est) > 0.4 * abs(beta), f"Effect for X{i+1} severely underestimated"
