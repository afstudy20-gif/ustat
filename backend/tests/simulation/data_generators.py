"""
Data generators for simulation studies in uSTAT.

These functions generate synthetic data with known parameters so we can
validate that the statistical methods recover the ground truth reasonably well.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_linear_data(
    n: int = 500,
    n_predictors: int = 5,
    true_beta: np.ndarray | None = None,
    noise_sd: float = 1.5,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """
    Generate data for linear regression with known coefficients.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, n_predictors))
    if true_beta is None:
        true_beta = np.array([2.0, -1.5, 0.8, 0.0, -0.3])

    y = X @ true_beta + rng.normal(0, noise_sd, n)

    df = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(n_predictors)])
    df["y"] = y

    ground_truth = {
        "beta": true_beta.tolist(),
        "noise_sd": noise_sd,
        "model": "linear",
    }
    return df, ground_truth


def generate_logistic_data(
    n: int = 800,
    n_predictors: int = 4,
    true_beta: np.ndarray | None = None,
    seed: int = 123,
) -> tuple[pd.DataFrame, dict]:
    """
    Generate binary outcome data for logistic regression.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, n_predictors))
    if true_beta is None:
        true_beta = np.array([0.5, -0.8, 1.2, 0.0])

    logit = X @ true_beta
    p = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, p)

    df = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(n_predictors)])
    df["event"] = y

    ground_truth = {
        "beta": true_beta.tolist(),
        "model": "logistic",
    }
    return df, ground_truth


def generate_survival_data(
    n: int = 600,
    n_predictors: int = 3,
    true_beta: np.ndarray | None = None,
    censoring_rate: float = 0.3,
    seed: int = 99,
) -> tuple[pd.DataFrame, dict]:
    """
    Generate right-censored survival data (Weibull baseline + Cox model).
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, n_predictors))
    if true_beta is None:
        true_beta = np.array([0.7, -0.4, 0.9])

    # Weibull baseline hazard
    shape = 1.5
    scale = 100.0
    u = rng.uniform(0, 1, n)
    # Inverse transform for Weibull
    baseline_time = scale * (-np.log(u)) ** (1 / shape)

    # Cox multiplier
    lp = X @ true_beta
    time = baseline_time * np.exp(-lp)

    # Censoring
    censor_time = rng.exponential(150, n)
    observed_time = np.minimum(time, censor_time)
    event = (time <= censor_time).astype(int)

    df = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(n_predictors)])
    df["duration"] = observed_time
    df["event"] = event

    ground_truth = {
        "beta": true_beta.tolist(),
        "model": "cox",
        "censoring_rate": censoring_rate,
    }
    return df, ground_truth


def generate_psm_data(
    n: int = 1000,
    treatment_effect: float = 0.8,
    seed: int = 77,
) -> tuple[pd.DataFrame, dict]:
    """
    Generate confounded data suitable for PSM / IPTW testing.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)  # noise

    # Treatment assignment depends on x1, x2 (confounding)
    logit_t = -0.5 + 1.2 * x1 + 0.9 * x2
    p_t = 1 / (1 + np.exp(-logit_t))
    treat = rng.binomial(1, p_t)

    # Outcome depends on treatment + confounders
    y = 3.0 + treatment_effect * treat + 1.5 * x1 + 1.2 * x2 + rng.normal(0, 2, n)

    df = pd.DataFrame({
        "treat": treat,
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "outcome": y,
    })

    ground_truth = {
        "treatment_effect": treatment_effect,
        "confounders": ["x1", "x2"],
        "model": "psm_iptw",
    }
    return df, ground_truth
