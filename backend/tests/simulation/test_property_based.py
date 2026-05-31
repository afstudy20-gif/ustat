"""
Property-based tests using Hypothesis (Phase 2 - Strengthened).

These tests generate many varied datasets and check important statistical
invariants and recovery properties across Linear, Logistic, Survival, and
PSM/IPTW methods.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

try:
    from hypothesis import given, settings
    from hypothesis import strategies as st
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    given = lambda *a, **kw: (lambda f: f)
    settings = lambda **kw: (lambda f: f)

from tests.conftest import make_session
from services.simulation_generators import (
    generate_linear_data,
    generate_logistic_data,
    generate_survival_data,
    generate_psm_iptw_data,
)

pytestmark = pytest.mark.skipif(
    not HYPOTHESIS_AVAILABLE,
    reason="hypothesis not installed. Run: pip install hypothesis"
)


# =============================================================================
# Linear Regression Properties
# =============================================================================

@settings(max_examples=40, deadline=4000)
@given(
    n=st.integers(150, 1200),
    noise_sd=st.floats(0.3, 2.5),
    seed=st.integers(0, 10000),
)
def test_linear_recovers_sign_and_magnitude(client, n, noise_sd, seed):
    """
    Property: When there is a reasonably strong linear signal, the model should
    recover the correct sign and at least 40% of the true magnitude for
    non-negligible coefficients.
    """
    df, truth = generate_linear_data(n=n, noise_sd=noise_sd, seed=seed)
    sid = make_session(df, f"prop_linear_{n}_{seed}")

    r = client.post("/api/models/linear", json={
        "session_id": sid,
        "outcome": "y",
        "predictors": ["X1", "X2", "X3", "X4", "X5"],
    })
    assert r.status_code == 200, r.text

    data = r.json()
    coefs = {c["variable"]: c["estimate"] for c in data["coefficients"] if c["variable"] != "const"}

    true_beta = np.array(truth["beta"])
    recovered = np.array([coefs.get(f"X{i+1}", 0.0) for i in range(5)])

    # Sign agreement for coefficients with |beta| > 0.4
    for i, beta in enumerate(true_beta):
        if abs(beta) > 0.4:
            est = recovered[i]
            assert np.sign(est) == np.sign(beta), f"Wrong sign recovered for X{i+1}"

            # At least 40% magnitude recovery
            assert abs(est) >= 0.4 * abs(beta), f"Severe underestimation for X{i+1}"


# =============================================================================
# Logistic Regression Properties
# =============================================================================

@settings(max_examples=35, deadline=5000)
@given(
    n=st.integers(300, 1500),
    seed=st.integers(0, 10000),
)
def test_logistic_has_discrimination_power(client, n, seed):
    """
    Property: On data with real signal, logistic regression should achieve
    AUC meaningfully above 0.5.
    """
    df, _ = generate_logistic_data(n=n, seed=seed)
    sid = make_session(df, f"prop_logistic_{n}_{seed}")

    r = client.post("/api/models/logistic", json={
        "session_id": sid,
        "outcome": "event",
        "predictors": ["X1", "X2", "X3", "X4"],
    })
    assert r.status_code == 200, r.text

    auc = r.json().get("auc")
    assert auc is not None
    assert auc > 0.58, f"AUC too close to random guessing: {auc:.3f}"


# =============================================================================
# Survival (Cox) Properties
# =============================================================================

@settings(max_examples=30, deadline=6000)
@given(
    n=st.integers(250, 900),
    seed=st.integers(0, 10000),
)
def test_cox_recovers_direction(client, n, seed):
    """
    Property: On data with real prognostic signal, Cox model should recover
    the correct direction of effect for stronger predictors.
    """
    df, truth = generate_survival_data(n=n, seed=seed)
    sid = make_session(df, f"prop_cox_{n}_{seed}")

    r = client.post("/api/models/survival/cox", json={
        "session_id": sid,
        "duration_col": "duration",
        "event_col": "event",
        "predictors": ["X1", "X2", "X3"],
    })
    assert r.status_code == 200, r.text

    coefs = {c["variable"]: c["estimate"] for c in r.json()["coefficients"]}
    true_beta = np.array(truth["beta"])

    for i, beta in enumerate(true_beta):
        if abs(beta) > 0.5:
            est = coefs.get(f"X{i+1}", 0.0)
            assert np.sign(est) == np.sign(beta), f"Wrong direction recovered for X{i+1} in Cox model"


# =============================================================================
# PSM / IPTW Properties
# =============================================================================

@settings(max_examples=25, deadline=5000)
@given(
    n=st.integers(400, 1500),
    treatment_effect=st.floats(0.4, 1.8),
    seed=st.integers(0, 10000),
)
def test_psm_reduces_confounding(client, n, treatment_effect, seed):
    """
    Property: After PSM, average SMD across confounders should be substantially
    lower than before matching (basic balance improvement check).
    """
    df, _ = generate_psm_iptw_data(n=n, treatment_effect=treatment_effect, seed=seed)
    sid = make_session(df, f"prop_psm_{n}_{seed}")

    payload = {
        "session_id": sid,
        "treatment_col": "treat",
        "covariates": ["x1", "x2", "x3"],
        "caliper": 0.25,
        "matching_method": "greedy",
        "score_method": "logistic",
        "ratio": 1,
    }

    r = client.post("/api/models/psm", json=payload)
    assert r.status_code == 200, r.text

    data = r.json()
    # Basic sanity: we should see some improvement in balance
    assert data["avg_smd_after"] < data["avg_smd_before"] * 0.85, \
        f"PSM did not meaningfully improve balance (before={data['avg_smd_before']:.3f}, after={data['avg_smd_after']:.3f})"


# =============================================================================
# Invariant Tests (Model should not crash on varied data)
# =============================================================================

@settings(max_examples=25, deadline=3000)
@given(
    n=st.integers(100, 800),
    n_pred=st.integers(2, 7),
)
def test_linear_does_not_crash_on_varied_data(client, n, n_pred):
    """Linear regression should handle a wide range of reasonable inputs without crashing."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(n, n_pred))
    y = X @ np.random.randn(n_pred) + rng.normal(0, 1.5, n)

    df = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(n_pred)])
    df["y"] = y

    sid = make_session(df, f"invariant_linear_{n}_{n_pred}")

    r = client.post("/api/models/linear", json={
        "session_id": sid,
        "outcome": "y",
        "predictors": [f"X{i+1}" for i in range(n_pred)],
    })
    assert r.status_code == 200, f"Linear regression crashed on n={n}, p={n_pred}"
    assert "r_squared" in r.json()


# =============================================================================
# Cross-Phase: Simulation under controlled assumption stress (links Phase 1)
# =============================================================================

@settings(max_examples=20, deadline=4000)
@given(
    noise_sd=st.floats(0.5, 4.0),   # High noise can violate normality/homoscedasticity
    seed=st.integers(0, 10000),
)
def test_linear_produces_assumption_warnings_under_stress(client, noise_sd, seed):
    """
    Property: When we deliberately stress assumptions (very high noise),
    the new assumption checking system (Phase 1) should detect issues
    and return warnings in the response.
    """
    df, _ = generate_linear_data(n=350, noise_sd=noise_sd, seed=seed)
    sid = make_session(df, f"stress_assumption_{seed}")

    r = client.post("/api/models/linear", json={
        "session_id": sid,
        "outcome": "y",
        "predictors": ["X1", "X2", "X3", "X4", "X5"],
    })
    assert r.status_code == 200

    data = r.json()
    # Either we have explicit assumption warnings or the overall severity is not "ok"
    has_warnings = bool(data.get("warnings"))
    assumptions = data.get("assumptions", {})
    severity = assumptions.get("overall_severity", "ok")

    # In high noise regimes we expect the system to raise some flag
    if noise_sd > 2.8:
        assert has_warnings or severity in ("warning", "critical"), \
            f"High noise ({noise_sd:.2f}) did not trigger any assumption warning"
