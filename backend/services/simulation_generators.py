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
    baseline_shape: float = 1.5,
    seed: int = 99,
) -> tuple[pd.DataFrame, dict]:
    """
    Generate right-censored survival data (Weibull baseline + Cox PH model).
    Improved version with controllable parameters.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, n_predictors))
    if true_beta is None:
        true_beta = np.array([0.7, -0.4, 0.9])

    # Weibull baseline
    scale = 100.0
    u = rng.uniform(0, 1, n)
    baseline_time = scale * (-np.log(u)) ** (1 / baseline_shape)

    lp = X @ true_beta
    time = baseline_time * np.exp(-lp)

    # Censoring (exponential)
    censor_time = rng.exponential(150 / (1 - censoring_rate + 0.01), n)
    observed_time = np.minimum(time, censor_time)
    event = (time <= censor_time).astype(int)

    df = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(n_predictors)])
    df["duration"] = observed_time
    df["event"] = event

    ground_truth = {
        "beta": true_beta.tolist(),
        "model": "cox",
        "censoring_rate": float(event.mean()),
        "baseline_shape": baseline_shape,
    }
    return df, ground_truth


def generate_psm_iptw_data(
    n: int = 1000,
    treatment_effect: float = 0.8,
    confounding_strength: float = 1.2,
    seed: int = 77,
    add_noise_covariate: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Generate confounded data suitable for PSM and IPTW testing.
    Stronger and more flexible version.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n) if add_noise_covariate else np.zeros(n)

    # Treatment assignment (confounding)
    logit_t = -0.5 + confounding_strength * x1 + confounding_strength * 0.75 * x2
    p_t = 1 / (1 + np.exp(-logit_t))
    treat = rng.binomial(1, p_t)

    # Outcome
    y = (3.0 +
         treatment_effect * treat +
         1.5 * x1 +
         1.2 * x2 +
         rng.normal(0, 2.0, n))

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
        "noise_covariates": ["x3"] if add_noise_covariate else [],
        "model": "psm_iptw",
    }
    return df, ground_truth


def add_correlated_predictors(df: pd.DataFrame, correlation: float = 0.7, seed: int = 42) -> pd.DataFrame:
    """Helper to add correlated versions of existing predictors (useful for multicollinearity tests)."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    for col in df.columns:
        if col.startswith("X") or col in ["x1", "x2"]:
            noise = rng.normal(0, np.sqrt(1 - correlation**2), len(df))
            df[f"{col}_corr"] = correlation * df[col] + noise
    return df


def generate_shared_frailty_survival_data(
    n_subjects: int = 400,
    n_clusters: int = 40,
    cluster_effect_sd: float = 0.8,   # log-frailty variance (theta)
    true_beta: np.ndarray | None = None,
    censoring_rate: float = 0.35,
    seed: int = 2026,
) -> tuple[pd.DataFrame, dict]:
    """
    Generate clustered survival data with shared gamma frailty (for testing frailty models).

    Each cluster has a multiplicative frailty term ~ Gamma(1/theta, theta) on the hazard scale.
    This induces within-cluster correlation while marginal effects remain identifiable.
    """
    rng = np.random.default_rng(seed)

    if true_beta is None:
        true_beta = np.array([0.6, -0.4, 0.9])

    n_per_cluster = max(1, n_subjects // n_clusters)
    n_subjects = n_per_cluster * n_clusters

    cluster_ids = np.repeat(np.arange(n_clusters), n_per_cluster)

    # Gamma frailty (mean 1, variance = theta = cluster_effect_sd**2)
    theta = max(0.01, cluster_effect_sd ** 2)
    shape = 1.0 / theta
    frailties = rng.gamma(shape=shape, scale=theta, size=n_clusters)
    frailty = frailties[cluster_ids]

    X = rng.normal(0, 1, size=(n_subjects, len(true_beta)))

    # Conditional hazard multiplier = exp(X @ beta) * frailty
    lp = X @ true_beta
    conditional_multiplier = np.exp(lp) * frailty

    # Weibull baseline (shape 1.4, scale 80)
    shape_b = 1.4
    scale_b = 80.0
    u = rng.uniform(1e-8, 1.0, n_subjects)
    time = scale_b * (-np.log(u) / conditional_multiplier) ** (1.0 / shape_b)

    # Independent censoring
    censor_time = rng.exponential(120.0, n_subjects)
    observed_time = np.minimum(time, censor_time)
    event = (time <= censor_time).astype(int)

    df = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(len(true_beta))])
    df["cluster"] = cluster_ids
    df["duration"] = observed_time
    df["event"] = event
    df["true_frailty"] = frailty   # for validation only

    ground_truth = {
        "beta": true_beta.tolist(),
        "theta": float(theta),
        "n_clusters": n_clusters,
        "model": "shared_frailty_cox",
        "censoring_rate": float(event.mean()),
    }
    return df, ground_truth


def generate_multistate_data(
    n: int = 600,
    states: tuple[int, ...] = (0, 1, 2),           # 0=healthy, 1=disease, 2=death (illness-death)
    true_transitions: dict[tuple[int, int], float] | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """
    Generate data for multi-state / illness-death models.

    Produces individual-level data with possible intermediate states.
    Uses exponential waiting times per allowed transition (simple but
    sufficient for testing transition intensity recovery).

    Returns a long-format DataFrame suitable for stacked transition-specific
    modeling + ground truth transition rates.
    """
    rng = np.random.default_rng(seed)

    if true_transitions is None:
        # Default illness-death structure
        # 0 -> 1 (healthy -> disease)
        # 0 -> 2 (healthy -> death, competing)
        # 1 -> 2 (disease -> death)
        true_transitions = {
            (0, 1): 0.15,
            (0, 2): 0.08,
            (1, 2): 0.35,
        }

    records = []
    for i in range(n):
        current_state = 0
        time = 0.0
        X = rng.normal(0, 1, 3)  # 3 covariates for demo

        history = []  # list of (from_state, to_state, entry_time, exit_time)

        max_time = 10.0
        while current_state != max(states) and time < max_time:
            possible = [t for t in true_transitions if t[0] == current_state]
            if not possible:
                break

            # Simple exponential waiting times (no covariates for generator simplicity;
            # real models will add effects)
            rates = [true_transitions[t] for t in possible]
            total_rate = sum(rates)
            if total_rate <= 0:
                break

            wait = rng.exponential(1 / total_rate)
            next_time = time + wait

            # Choose which transition
            probs = [r / total_rate for r in rates]
            chosen_idx = rng.choice(len(possible), p=probs)
            next_state = possible[chosen_idx][1]

            history.append({
                "id": i,
                "from_state": current_state,
                "to_state": next_state,
                "entry": time,
                "exit": next_time,
                "X1": X[0],
                "X2": X[1],
                "X3": X[2],
            })

            current_state = next_state
            time = next_time

        # Censor if still in non-absorbing state at max_time
        if history and history[-1]["to_state"] != max(states):
            history[-1]["exit"] = min(history[-1]["exit"], max_time)
            history[-1]["to_state"] = -1  # censored indicator for that transition

        records.extend(history)

    if not records:
        # Fallback
        records = [{"id": 0, "from_state": 0, "to_state": 2, "entry": 0, "exit": 5, "X1": 0, "X2": 0, "X3": 0}]

    df = pd.DataFrame(records)
    df["event"] = (df["to_state"] != -1).astype(int)

    ground_truth = {
        "true_transitions": true_transitions,
        "n_subjects": n,
        "model": "multistate_illness_death",
    }
    return df, ground_truth


def generate_joint_longitudinal_survival_data(
    n_subjects: int = 400,
    n_measurements: int = 5,
    measurement_times: np.ndarray | None = None,
    true_beta_long: np.ndarray | None = None,
    true_beta_surv: float = 0.4,
    random_effect_sd: float = 0.8,
    censoring_rate: float = 0.3,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """
    Generate data for joint longitudinal-survival models (shared random effects).

    Longitudinal part:
    - Repeated measurements of a biomarker Y(t) with subject-specific random intercept + slope.
    - Fixed effects + error.

    Survival part:
    - Hazard depends on current biomarker value + shared random effects (association parameter).
    - Weibull or exponential baseline.

    Returns:
    - long_df: longitudinal measurements (id, time, Y, covariates)
    - surv_df: survival data (id, duration, event, covariates)
    - ground_truth with true parameters.
    """
    rng = np.random.default_rng(seed)

    if measurement_times is None:
        measurement_times = np.array([0, 1, 2, 3, 4, 5])[:n_measurements]

    if true_beta_long is None:
        true_beta_long = np.array([2.0, 0.3])  # intercept, slope

    # Subject-specific random effects (shared with survival)
    u0 = rng.normal(0, random_effect_sd, n_subjects)  # random intercept
    u1 = rng.normal(0, random_effect_sd * 0.5, n_subjects)  # random slope

    long_records = []
    surv_records = []

    for i in range(n_subjects):
        # Covariates (baseline)
        X = rng.normal(0, 1, 2)

        # Longitudinal trajectory
        for t in measurement_times:
            y = (
                true_beta_long[0]
                + true_beta_long[1] * t
                + u0[i]
                + u1[i] * t
                + rng.normal(0, 0.8)
            )
            long_records.append({
                "id": i,
                "time": t,
                "Y": y,
                "X1": X[0],
                "X2": X[1],
            })

        # Survival time (depends on current biomarker level + random effects)
        # Simplified: hazard proportional to exp( true_beta_surv * current_Y + gamma * u )
        # Use inverse transform for Weibull-like
        shape = 1.3
        scale = 8.0

        # Approximate current Y at t=0 for hazard multiplier (or integrate, but simplify)
        y0 = true_beta_long[0] + u0[i] + rng.normal(0, 0.5)
        hazard_mult = np.exp(true_beta_surv * y0 + 0.6 * u0[i])

        u = rng.uniform(1e-6, 1.0)
        surv_time = scale * (-np.log(u) / hazard_mult) ** (1.0 / shape)

        # Censoring
        cens = rng.exponential(12.0)
        observed_time = min(surv_time, cens)
        event = 1 if surv_time <= cens else 0

        surv_records.append({
            "id": i,
            "duration": observed_time,
            "event": event,
            "X1": X[0],
            "X2": X[1],
            "true_u0": u0[i],
        })

    long_df = pd.DataFrame(long_records)
    surv_df = pd.DataFrame(surv_records)

    # Ensure no duplicate columns (defensive for joint modeling)
    long_df = long_df.loc[:, ~long_df.columns.duplicated()]
    surv_df = surv_df.loc[:, ~surv_df.columns.duplicated()]

    ground_truth = {
        "beta_long": true_beta_long.tolist(),
        "beta_surv_association": true_beta_surv,
        "random_effect_sd": random_effect_sd,
        "model": "joint_shared_random_effects",
        "censoring_rate": float(1 - surv_df["event"].mean()),
    }

    return long_df, surv_df, ground_truth


def generate_external_validation_cohorts(
    n_dev: int = 600,
    n_val: int = 400,
    shift_covariates: float = 0.4,   # mean shift in validation cohort
    hazard_multiplier: float = 1.3,  # different baseline hazard in validation
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Generate a development cohort and an external validation cohort with
    realistic transportability differences (different covariate distribution
    and different baseline risk). Useful for testing prediction model
    evaluation, calibration, and external validation workflows.
    """
    rng = np.random.default_rng(seed)

    # Development cohort (reference population)
    dev_long, dev_surv, _ = generate_joint_longitudinal_survival_data(
        n_subjects=n_dev,
        true_beta_surv=0.45,
        seed=seed,
    )
    dev_surv["cohort"] = "development"

    # Validation cohort (shifted population)
    val_long, val_surv, _ = generate_joint_longitudinal_survival_data(
        n_subjects=n_val,
        true_beta_surv=0.45,
        seed=seed + 1,
    )

    # Apply realistic transportability shifts
    for col in ["X1", "X2"]:
        val_long[col] = val_long[col] + rng.normal(shift_covariates, 0.3, len(val_long))
        val_surv[col] = val_surv[col] + rng.normal(shift_covariates, 0.3, len(val_surv))

    # Different baseline hazard (multiply duration by factor)
    val_surv["duration"] = val_surv["duration"] / hazard_multiplier

    val_surv["cohort"] = "validation"

    long_df = pd.concat([dev_long, val_long], ignore_index=True)
    surv_df = pd.concat([dev_surv, val_surv], ignore_index=True)

    ground_truth = {
        "dev_n": n_dev,
        "val_n": n_val,
        "covariate_shift": shift_covariates,
        "hazard_multiplier": hazard_multiplier,
        "model": "external_validation_cohorts",
    }

    return long_df, surv_df, ground_truth


def generate_survival_ml_benchmark_data(
    n: int = 800,
    n_predictors: int = 8,
    non_linear: bool = True,
    interaction: bool = True,
    censoring_rate: float = 0.35,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """
    Generate survival data designed to benchmark classical Cox vs ML models.
    When non_linear=True or interaction=True, tree-based methods (RSF, GB) typically
    outperform linear Cox on discrimination and calibration in the validation set.
    """
    rng = np.random.default_rng(seed)

    X = rng.normal(0, 1, size=(n, n_predictors))
    cols = [f"X{i+1}" for i in range(n_predictors)]

    # True log-hazard (can be non-linear + interactions)
    lp = np.zeros(n)
    lp += 0.7 * X[:, 0]
    lp += -0.5 * X[:, 1]

    if non_linear:
        lp += 0.8 * np.sin(X[:, 2])                    # strong non-linearity
        lp += 0.6 * (X[:, 3] ** 2 - 1)                 # quadratic

    if interaction:
        lp += 0.9 * X[:, 4] * X[:, 5]                  # interaction

    # Weibull baseline
    shape = 1.4
    scale = 6.0
    u = rng.uniform(1e-8, 1.0, n)
    time = scale * (-np.log(u)) ** (1 / shape) * np.exp(-lp)

    # Censoring
    cens = rng.exponential(9.0, n)
    observed_time = np.minimum(time, cens)
    event = (time <= cens).astype(int)

    df = pd.DataFrame(X, columns=cols)
    df["duration"] = observed_time
    df["event"] = event

    ground_truth = {
        "non_linear": non_linear,
        "interaction": interaction,
        "model": "survival_ml_benchmark",
        "censoring_rate": float(1 - event.mean()),
        "n_predictors": n_predictors,
    }
    return df, ground_truth


def generate_arima_series(
    n: int = 200,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
    sigma: float = 1.0,
    seed: int = 42,
) -> tuple[pd.Series, dict]:
    """
    Generate a time series from a known (S)ARIMA process for testing recovery.
    Returns a pandas Series with DatetimeIndex and ground truth parameters.
    """
    rng = np.random.default_rng(seed)
    from statsmodels.tsa.arima_process import ArmaProcess

    p, d, q = order
    P, D, Q, s = seasonal_order

    # Non-seasonal ARMA
    ar_params = [1.0] + [-0.6] * min(p, 1)  # simple AR(1) example if p>=1
    ma_params = [1.0] + [0.4] * min(q, 1)

    arma_process = ArmaProcess(ar_params[:p+1] if p > 0 else [1.0], ma_params[:q+1] if q > 0 else [1.0])
    y = arma_process.generate_sample(nsample=n + 50, scale=sigma, distrvs=rng.normal)

    # Simple differencing simulation for d and D (approximate)
    if d > 0:
        y = np.cumsum(y)
    if D > 0 and s > 1:
        for _ in range(D):
            y[s:] = y[s:] + y[:-s]

    y = y[-n:]
    idx = pd.date_range("2020-01-01", periods=n, freq="MS")
    series = pd.Series(y, index=idx, name="y")

    ground_truth = {
        "order": order,
        "seasonal_order": seasonal_order,
        "sigma": sigma,
        "model": "arima_simulation",
    }
    return series, ground_truth


def generate_dca_binary_data(
    n: int = 600,
    prevalence: float = 0.25,
    auc: float = 0.78,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """
    Generate synthetic binary data with known discrimination (AUC) for DCA testing.

    The generator produces a predictor whose logistic relationship with the outcome
    yields approximately the requested AUC. Ground truth includes the true prevalence
    and the fact that a perfect model would have a specific net benefit curve.
    """
    rng = np.random.default_rng(seed)

    # Simulate a latent risk that gives the desired AUC via a simple probit-like mechanism
    # Higher latent → higher event probability
    latent = rng.normal(0, 1, n)
    # Shift to achieve target prevalence
    shift = np.quantile(latent, 1 - prevalence)
    event_prob = 1 / (1 + np.exp(-(latent - shift) * 1.8))  # scaling gives reasonable AUC range

    # Add noise so that the observable predictor has the target AUC
    noise_sd = np.sqrt((1 / (auc ** 2) - 1))  # rough relationship
    predictor = latent + rng.normal(0, noise_sd, n)

    y = rng.binomial(1, event_prob)

    df = pd.DataFrame({
        "outcome": y,
        "predictor": predictor,
    })

    ground_truth = {
        "prevalence": float(np.mean(y)),
        "target_auc": auc,
        "model": "dca_binary_simulation",
        "n": n,
    }
    return df, ground_truth
