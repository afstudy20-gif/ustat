"""
Causal Sensitivity Analysis Module (Phase 5)

Provides tools for assessing robustness of causal effect estimates
to unmeasured confounding.

Core methods:
- E-value (VanderWeele & Ding 2017) for RR, OR, HR
- Improved OR -> RR conversion (supports baseline risk / prevalence)
- Simple bias factor / quantitative bias analysis (Greenland / Schlesselman style)
- Integration helpers for PSM/IPTW results

All functions are pure and return new immutable dictionaries.
"""

from __future__ import annotations

from typing import Dict, Any, Literal, Optional
import math


def _safe_sqrt(x: float) -> float:
    return math.sqrt(max(x, 0.0))


def _or_to_rr(or_val: float, baseline_risk: float = 0.1) -> float:
    """
    Convert odds ratio to approximate risk ratio using a baseline risk.
    Formula: RR ≈ OR / ((1 - p0) + p0 * OR)   (Zhang & Yu style approximation)
    """
    if or_val <= 0:
        return 1.0
    p0 = max(0.001, min(0.99, baseline_risk))
    return or_val / ((1.0 - p0) + p0 * or_val)


def e_value(
    estimate: float,
    ci_low: Optional[float] = None,
    ci_high: Optional[float] = None,
    measure: Literal["rr", "or", "hr"] = "rr",
    rare_outcome: bool = False,
    baseline_risk: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Calculate the E-value (VanderWeele & Ding) for a point estimate and optional CI.

    For OR with common outcomes, supply baseline_risk (0 < p0 < 1) for a better
    approximation than the rare-disease assumption.

    Returns a new immutable dict with e_value_point_estimate, e_value_ci, interpretation.
    """
    if measure not in ("rr", "or", "hr"):
        raise ValueError("measure must be 'rr', 'or', or 'hr'")
    if estimate <= 0:
        raise ValueError("estimate must be positive")

    # Convert everything to RR scale (immutably)
    if measure == "rr" or measure == "hr":
        rr = float(estimate)
        rr_ci = (
            [float(ci_low), float(ci_high)]
            if ci_low is not None and ci_high is not None
            else None
        )
    else:  # OR
        if rare_outcome or baseline_risk is None:
            # Rare outcome or user did not supply prevalence → use conservative approx
            p0 = 0.1 if baseline_risk is None else baseline_risk
        else:
            p0 = baseline_risk
        rr = _or_to_rr(float(estimate), p0)
        if ci_low is not None and ci_high is not None:
            rr_ci = [_or_to_rr(float(ci_low), p0), _or_to_rr(float(ci_high), p0)]
        else:
            rr_ci = None

    # Invert if protective (RR < 1) — work on copies
    if rr <= 1.0:
        rr = 1.0 / rr if rr > 0 else 1.0
        if rr_ci:
            rr_ci = [1.0 / x if x > 0 else float("inf") for x in sorted(rr_ci, reverse=True)]

    # E-value point estimate (classic formula)
    ev_point = rr + _safe_sqrt(rr * (rr - 1.0)) if rr > 1.0 else 1.0

    # E-value for CI (bound closest to null)
    ev_ci = None
    if rr_ci:
        lo, hi = sorted(rr_ci)
        bound = lo if lo < 1.0 else hi
        if bound > 1.0:
            ev_ci = bound + _safe_sqrt(bound * (bound - 1.0))
        else:
            ev_ci = 1.0

    interpretation = (
        f"An unmeasured confounder would need to be associated with both the exposure and the outcome "
        f"by a risk ratio of at least {ev_point:.2f} (point estimate)"
    )
    if ev_ci and ev_ci > 1.0:
        interpretation += f" and at least {ev_ci:.2f} (CI) to explain away the observed association."

    return {
        "measure": measure,
        "e_value_point_estimate": round(ev_point, 3) if ev_point > 1.0 else 1.0,
        "e_value_ci": round(ev_ci, 3) if ev_ci and ev_ci > 1.0 else 1.0,
        "interpretation": interpretation,
        "baseline_risk_used": baseline_risk if measure == "or" else None,
    }


def bias_factor(
    observed_rr: float,
    confounding_rr_exposure_outcome: float,
    confounding_prevalence_exposed: float = 0.5,
    confounding_prevalence_unexposed: float = 0.5,
) -> float:
    """
    Approximate bias factor by which the observed RR may be inflated
    due to an unmeasured binary confounder (Greenland/Schlesselman formulation).
    Returns a new float (immutable result).
    """
    if observed_rr <= 0 or confounding_rr_exposure_outcome <= 0:
        return float("nan")

    pe = max(0.0, min(1.0, confounding_prevalence_exposed))
    pu = max(0.0, min(1.0, confounding_prevalence_unexposed))

    numerator = confounding_rr_exposure_outcome * pe + (1.0 - pe)
    denominator = confounding_rr_exposure_outcome * pu + (1.0 - pu)

    if denominator < 1e-12:
        return float("inf")

    return round(numerator / denominator, 3)


def quantitative_bias_analysis(
    observed_estimate: float,
    measure: Literal["rr", "or", "hr"] = "rr",
    confounding_strength: float = 2.0,
    prevalence_exposed: float = 0.5,
    prevalence_unexposed: float = 0.5,
) -> Dict[str, Any]:
    """
    Simple quantitative bias analysis (QBA) for unmeasured confounding.
    Returns a fresh dict; never mutates inputs.
    """
    if observed_estimate <= 0:
        observed_estimate = 1.0

    bias = bias_factor(
        observed_rr=observed_estimate if measure != "or" else max(observed_estimate, 0.1),
        confounding_rr_exposure_outcome=confounding_strength,
        confounding_prevalence_exposed=prevalence_exposed,
        confounding_prevalence_unexposed=prevalence_unexposed,
    )

    if bias <= 0 or math.isnan(bias):
        bias = 1.0

    if measure == "rr":
        corrected = observed_estimate / bias
    else:
        corrected = observed_estimate / bias   # rough on the ratio scale

    direction = "away from the null" if bias > 1.01 else "toward the null (or none)"

    return {
        "observed_estimate": round(observed_estimate, 3),
        "assumed_confounder_risk_ratio": confounding_strength,
        "prevalence_exposed": prevalence_exposed,
        "prevalence_unexposed": prevalence_unexposed,
        "bias_factor": bias,
        "bias_corrected_estimate": round(corrected, 3),
        "bias_direction": direction,
        "interpretation": (
            f"Under an unmeasured confounder with RR={confounding_strength} and prevalences "
            f"{prevalence_exposed:.0%} vs {prevalence_unexposed:.0%}, bias factor ≈ {bias}. "
            f"Corrected estimate ≈ {round(corrected, 3)} ({direction})."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Integration helper for PSM / IPTW results (Phase 5 requirement)
# ═══════════════════════════════════════════════════════════════════════════════

def e_value_from_psm_or_iptw(
    treatment_effect_on_ratio_scale: float,
    ci_low: Optional[float] = None,
    ci_high: Optional[float] = None,
    measure: Literal["rr", "or", "hr"] = "rr",
) -> Dict[str, Any]:
    """
    Convenience wrapper: compute E-value directly from a marginal treatment effect
    typically obtained after PSM or IPTW (e.g. marginal RR or HR).

    This is the main integration point with the existing PSM/IPTW modules.
    """
    return e_value(
        estimate=treatment_effect_on_ratio_scale,
        ci_low=ci_low,
        ci_high=ci_high,
        measure=measure,
        rare_outcome=False,
    )
