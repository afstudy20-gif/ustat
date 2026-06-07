"""Chained Predictive Mean Matching engine (services.missing_data.mice_multiple)."""
import numpy as np
import pandas as pd

from services.missing_data import mice_multiple, _column_method


def _data():
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(0, 1, n)
    y = 2 * x + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"x": x, "y": y, "sex": rng.choice(["k", "e"], n), "dm": rng.integers(0, 2, n)})
    observed_y = set(np.round(df["y"], 6))
    df.loc[rng.choice(n, 40, replace=False), "y"] = np.nan
    df.loc[rng.choice(n, 20, replace=False), "sex"] = np.nan
    return df, observed_y


def test_method_selection_by_type():
    df, _ = _data()
    assert _column_method(df["y"]) == "pmm"        # continuous → PMM
    assert _column_method(df["dm"]) == "logreg"    # binary 0/1 → logistic
    assert _column_method(df["sex"]) == "polyreg"  # categorical → hot-deck


def test_pmm_fills_with_observed_donors_and_varies():
    df, observed_y = _data()
    miss = df["y"].isna().to_numpy()
    res = mice_multiple(df, ["x", "y", "sex", "dm"], n_imputations=5, max_iter=10, random_state=1)
    assert res.method == "pmm"
    assert len(res.imputed_datasets) == 5
    d0, d1 = res.imputed_datasets[0], res.imputed_datasets[1]
    # All cells filled.
    assert d0["y"].isna().sum() == 0 and d0["sex"].isna().sum() == 0
    # PMM only ever uses real observed values (no synthetic extrapolation).
    imputed = set(np.round(d0["y"].to_numpy()[miss], 6))
    assert imputed.issubset(observed_y)
    # Proper multiple imputation → datasets differ.
    assert not np.allclose(d0["y"].to_numpy()[miss], d1["y"].to_numpy()[miss])
    # Relationship preserved (true slope = 2).
    slope = float(np.polyfit(d0["x"], d0["y"], 1)[0])
    assert 1.5 < slope < 2.5
