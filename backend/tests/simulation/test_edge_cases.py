import pytest
import numpy as np
import pandas as pd
from tests.conftest import make_session

@pytest.mark.simulation
def test_edge_case_zero_events_survival(client):
    """
    Edge Case: No events in survival data (all right-censored).
    The system must handle this gracefully instead of throwing a 500 error.
    """
    df = pd.DataFrame({
        "time": [10.0, 20.0, 30.0, 40.0],
        "event": [0, 0, 0, 0],  # Zero events
        "x": [1.2, 0.8, -0.4, 2.1]
    })
    
    sid = make_session(df, "edge_zero_events")
    
    r = client.post("/api/models/survival/cox", json={
        "session_id": sid,
        "duration_col": "time",
        "event_col": "event",
        "predictors": ["x"]
    })
    
    # Either a 400 Bad Request or a warning in a 200 response is expected
    assert r.status_code in (400, 200)
    if r.status_code == 200:
        data = r.json()
        assert "warning" in data or "warnings" in data or "error" in data


@pytest.mark.simulation
def test_edge_case_constant_covariate(client):
    """
    Edge Case: A predictor covariate is constant (no variance).
    The regression service should handle this gracefully (warnings or errors).
    """
    df = pd.DataFrame({
        "y": [1.0, 2.5, 3.8, 1.9],
        "x_const": [5.0, 5.0, 5.0, 5.0],  # Constant predictor
        "x_normal": [1.2, -0.8, 2.5, 0.4]
    })
    
    sid = make_session(df, "edge_constant_cov")
    
    r = client.post("/api/models/linear", json={
        "session_id": sid,
        "outcome": "y",
        "predictors": ["x_const", "x_normal"]
    })
    
    # Handled gracefully
    assert r.status_code in (400, 200)
    if r.status_code == 200:
        data = r.json()
        # Should flag collinearity/warnings
        assert "warnings" in data or "warning" in data or data.get("r_squared") is not None


@pytest.mark.simulation
def test_edge_case_perfect_separation_logistic(client):
    """
    Edge Case: Perfect separation in logistic regression.
    The logistic endpoint should detect or handle infinite coefficients.
    """
    df = pd.DataFrame({
        "event": [1, 1, 0, 0],
        "x_sep": [10.0, 5.0, -2.0, -10.0]  # x_sep > 0 perfectly predicts event = 1
    })
    
    sid = make_session(df, "edge_perfect_sep")
    
    r = client.post("/api/models/logistic", json={
        "session_id": sid,
        "outcome": "event",
        "predictors": ["x_sep"]
    })
    
    assert r.status_code in (400, 200)
    if r.status_code == 200:
        data = r.json()
        # Warning/alert should be raised in the assumptions or warnings block
        assert "warnings" in data or data.get("auc") == 1.0 or "perfect separation" in str(data).lower()


@pytest.mark.simulation
def test_edge_case_single_cluster_frailty():
    """
    Edge Case: Single cluster in Shared Frailty model fitting.
    Shared frailty expects multiple clusters to estimate unobserved cluster-specific variance.
    If only 1 cluster is present, the service should raise an error or warning.
    """
    from services.frailty import fit_shared_gamma_frailty
    
    df = pd.DataFrame({
        "duration": [12.0, 8.5, 15.0, 20.0] * 5,
        "event": [1, 0, 1, 1] * 5,
        "cluster": ["A"] * 20,  # Single cluster
        "x": [0.4, -0.6, 1.2, 0.1] * 5
    })
    
    with pytest.raises(ValueError) as excinfo:
        fit_shared_gamma_frailty(
            df,
            duration_col="duration",
            event_col="event",
            cluster_col="cluster",
            predictors=["x"],
            frailty_distribution="gamma"
        )
    
    assert "clusters" in str(excinfo.value).lower()
