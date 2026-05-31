import pytest
import numpy as np
import pandas as pd
from tests.conftest import make_session

from services.simulation_generators import (
    generate_shared_frailty_survival_data,
    generate_multistate_data
)
from services.frailty import fit_shared_gamma_frailty

@pytest.mark.simulation
def test_shared_frailty_parameter_recovery():
    """
    Test that the shared frailty model recovers the simulated frailty variance (theta)
    and prognostic predictors direction correctly.
    """
    # 1. Clustered survival data with Gamma frailty
    df, gt = generate_shared_frailty_survival_data(
        n_subjects=400,
        n_clusters=35,
        cluster_effect_sd=0.8, # theta = cluster_effect_sd**2 = 0.64
        seed=123
    )
    
    # 2. Fit the shared frailty model
    res = fit_shared_gamma_frailty(
        df,
        duration_col="duration",
        event_col="event",
        cluster_col="cluster",
        predictors=["X1", "X2", "X3"],
        frailty_distribution="gamma"
    )
    
    # 3. Assert statistical correctness
    assert res is not None
    assert "theta" in res
    assert "coefficients" in res
    
    recovered_theta = res["theta"]
    true_theta = gt["theta"]
    
    # Check that recovered theta is positive and within a reasonable bounds of the true value (0.64)
    assert recovered_theta > 0.05
    assert abs(recovered_theta - true_theta) / true_theta < 0.60
    
    # Check that predictor directions are recovered correctly
    coef_dict = {c["variable"]: c["estimate"] for c in res["coefficients"]}
    true_beta = gt["beta"]
    
    assert np.sign(coef_dict["X1"]) == np.sign(true_beta[0])
    assert np.sign(coef_dict["X2"]) == np.sign(true_beta[1])
    assert np.sign(coef_dict["X3"]) == np.sign(true_beta[2])


@pytest.mark.simulation
def test_recurrent_events_lwyy_pipeline(client):
    """
    Test recurrent event modeling (LWYY model, Nelson-Aalen MCF, Diagnostics)
    using the /recurrent_lwyy endpoint with simulated recurrent event data.
    """
    # Generate multi-state data, which naturally contains recurrent events (multiple rows/events per subject)
    df, gt = generate_multistate_data(n=250, seed=99)
    
    # Reconstruct columns for recurrent format
    df = df.rename(columns={
        "id": "subject_id",
        "entry": "start_time",
        "exit": "stop_time",
        "event": "event_indicator"
    })
    
    # Generate a dummy terminal event indicator for diagnostics
    rng = np.random.default_rng(99)
    terminal_df = pd.DataFrame({
        "subject_id": df["subject_id"].unique(),
        "terminal_event": rng.binomial(1, 0.4, size=df["subject_id"].nunique()),
        "terminal_time": rng.uniform(8.0, 10.0, size=df["subject_id"].nunique())
    })
    
    df = df.merge(terminal_df, on="subject_id", how="left")
    
    sid = make_session(df, "recurrent_events_session")
    
    payload = {
        "session_id": sid,
        "id_col": "subject_id",
        "start_col": "start_time",
        "stop_col": "stop_time",
        "event_col": "event_indicator",
        "predictors": ["X1", "X2", "X3"],
        "model_type": "lwyy",          # Lin-Wei-Yang-Ying robust SE model
        "time_scale": "total",          # Total time scale (Andersen-Gill)
        "terminal_event_col": "terminal_event",
        "terminal_time_col": "terminal_time"
    }
    
    r = client.post("/api/survival_advanced/recurrent_lwyy", json=payload)
    assert r.status_code == 200, r.text
    
    res = r.json()
    assert "coefficients" in res
    assert "mcf" in res
    assert "recurrent_diagnostics" in res
    assert "events_per_subject" in res
    
    # Check that LWYY robust standard errors are calculated and coefficients are parsed correctly
    coefs = res["coefficients"]
    assert len(coefs) == 3
    for c in coefs:
        assert "robust_se" in c
        assert "estimate" in c
        assert "rate_ratio" in c
        
    # Check MCF curve properties
    mcf = res["mcf"]
    assert "overall" in mcf
    overall_pts = mcf["overall"]
    assert len(overall_pts) > 0
    assert "t" in overall_pts[0]
    assert "mcf" in overall_pts[0]
    assert overall_pts[-1]["mcf"] >= overall_pts[0]["mcf"] # should be monotonic increasing
