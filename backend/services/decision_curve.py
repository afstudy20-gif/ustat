"""
Decision Curve Analysis (DCA) — Phase 13

Production-grade implementation following the project's established patterns
(external_validation, survival_ml, frailty, etc.):

- Pure functions, immutable returns (no mutation of inputs).
- Support for both binary logistic predictions and pre-computed survival risk scores.
- Full net benefit, standardized net benefit, and clinical utility summaries.
- Rich metadata: assumptions, warnings, result_text, export-ready rows.
- Designed for direct integration with Phase 9 external validation and
  Phase 12 survival ML benchmark outputs (risk scores + survival probabilities).

No new dependencies beyond the existing stack (numpy, pandas, scipy).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _safe(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


def _compute_net_benefit(
    y: np.ndarray,
    p: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """
    Vectorized net benefit calculation.

    NB(pt) = (TP / n) - (FP / n) * (pt / (1 - pt))

    Parameters
    ----------
    y : binary outcome (0/1)
    p : predicted probability or risk score (higher = more likely event / worse prognosis)
    thresholds : array of decision thresholds in (0, 1)

    Returns
    -------
    net_benefit : array same length as thresholds
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    n = len(y)
    if n == 0:
        return np.zeros_like(thresholds, dtype=float)

    nb = np.zeros_like(thresholds, dtype=float)
    prevalence = float(y.mean())

    for i, pt in enumerate(thresholds):
        if pt <= 0 or pt >= 1:
            # At 0 or 1 the formula is degenerate; treat as boundary cases
            if pt <= 0:
                nb[i] = prevalence  # treat all
            else:
                nb[i] = 0.0  # treat none
            continue

        pred_pos = p >= pt
        tp = float((pred_pos & (y == 1)).sum()) / n
        fp = float((pred_pos & (y == 0)).sum()) / n
        nb[i] = tp - (fp * pt / (1.0 - pt))

    return nb


def _standardized_net_benefit(nb: np.ndarray, prevalence: float) -> np.ndarray:
    """
    Standardized net benefit (sNB) = NB / prevalence

    This puts NB on a scale comparable across different outcome prevalences.
    """
    if prevalence <= 0:
        return np.zeros_like(nb)
    return nb / prevalence


def decision_curve_analysis_binary(
    y: np.ndarray,
    p: np.ndarray,
    *,
    thresholds: Optional[np.ndarray] = None,
    n_thresholds: int = 101,
    threshold_range: Tuple[float, float] = (0.01, 0.99),
) -> Dict[str, Any]:
    """
    Core DCA for a binary outcome with predicted probabilities.

    Returns a rich, immutable result dictionary suitable for API responses.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)

    if len(y) != len(p):
        raise ValueError("y and p must have the same length")
    if len(y) < 20:
        return {"error": "Need at least 20 observations for reliable DCA"}

    prevalence = float(np.mean(y))
    if prevalence <= 0 or prevalence >= 1:
        return {"error": "Outcome must have both events and non-events"}

    if thresholds is None:
        lo, hi = threshold_range
        thresholds = np.linspace(max(0.001, lo), min(0.999, hi), n_thresholds)

    thresholds = np.asarray(thresholds, dtype=float)

    model_nb = _compute_net_benefit(y, p, thresholds)
    all_nb = _compute_net_benefit(y, np.ones_like(p), thresholds)  # treat all
    none_nb = np.zeros_like(thresholds)

    snb_model = _standardized_net_benefit(model_nb, prevalence)
    snb_all = _standardized_net_benefit(all_nb, prevalence)

    # Clinical summaries
    positive_nb_mask = model_nb > 0
    if positive_nb_mask.any():
        useful_thresholds = thresholds[positive_nb_mask]
        useful_min = float(useful_thresholds.min())
        useful_max = float(useful_thresholds.max())
        range_text = f"Positive net benefit for threshold probabilities {useful_min:.3f}–{useful_max:.3f}."
    else:
        useful_min = useful_max = None
        range_text = "No threshold shows positive net benefit over treat-none."

    # Max net benefit and the threshold at which it occurs
    max_idx = int(np.argmax(model_nb))
    max_nb = float(model_nb[max_idx])
    max_nb_threshold = float(thresholds[max_idx])

    # "Harm" threshold: where model NB drops below treat-none (0)
    harm_mask = model_nb < 0
    harm_threshold = float(thresholds[harm_mask][0]) if harm_mask.any() else None

    # Interventions avoided (rough clinical translation at a reference threshold, e.g. max NB pt)
    # For simplicity we report at the max-NB threshold
    ref_pt = max_nb_threshold
    if 0 < ref_pt < 1:
        interventions_avoided = (max_nb / (ref_pt / (1 - ref_pt))) * 100 if ref_pt > 0 else 0.0
    else:
        interventions_avoided = 0.0

    assumptions = [
        "Net benefit assumes the clinical consequences of a false positive vs false negative are captured by the chosen threshold probability.",
        "Standardized net benefit (sNB) allows comparison across populations with different outcome prevalence.",
        "Predictions p are assumed to be well-calibrated probabilities or risk scores monotonically related to event risk.",
    ]
    warnings: List[str] = []
    if len(y) < 100:
        warnings.append("Small sample; net benefit estimates have high variance — interpret ranges cautiously.")
    if prevalence < 0.05 or prevalence > 0.95:
        warnings.append("Extreme prevalence; standardized net benefit interpretation requires care.")

    result_text = (
        f"Decision curve analysis (n={len(y)}, prevalence={prevalence:.3f}). "
        f"{range_text} "
        f"Maximum net benefit {max_nb:.4f} at threshold {max_nb_threshold:.3f}. "
        f"Approximately {interventions_avoided:.1f} interventions avoided per 100 patients at that threshold (relative to treat-all)."
    )

    curves = {
        "thresholds": [round(float(t), 4) for t in thresholds],
        "model_net_benefit": [round(float(v), 6) for v in model_nb],
        "treat_all_net_benefit": [round(float(v), 6) for v in all_nb],
        "treat_none_net_benefit": [round(float(v), 6) for v in none_nb],
        "model_snb": [round(float(v), 6) for v in snb_model],
        "treat_all_snb": [round(float(v), 6) for v in snb_all],
    }

    summary = {
        "prevalence": round(prevalence, 4),
        "n": int(len(y)),
        "max_net_benefit": round(max_nb, 6),
        "max_net_benefit_threshold": round(max_nb_threshold, 4),
        "positive_nb_range": [round(useful_min, 4), round(useful_max, 4)] if useful_min is not None else None,
        "harm_threshold": round(harm_threshold, 4) if harm_threshold is not None else None,
        "interventions_avoided_per_100_at_max": round(interventions_avoided, 2),
    }

    export_rows = [
        ["threshold", "model_nb", "treat_all_nb", "treat_none_nb", "model_snb"],
        *[
            [
                round(float(thresholds[i]), 4),
                round(float(model_nb[i]), 6),
                round(float(all_nb[i]), 6),
                0.0,
                round(float(snb_model[i]), 6),
            ]
            for i in range(len(thresholds))
        ],
    ]

    return {
        "test": "Decision Curve Analysis",
        "mode": "binary",
        "curves": curves,
        "summary": summary,
        "assumptions": assumptions,
        "warnings": warnings,
        "result_text": result_text,
        "export_rows": export_rows,
        "n": int(len(y)),
        "prevalence": round(prevalence, 4),
    }


def decision_curve_analysis_survival(
    duration: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
    *,
    time_horizon: Optional[float] = None,
    thresholds: Optional[np.ndarray] = None,
    n_thresholds: int = 101,
    threshold_range: Tuple[float, float] = (0.01, 0.50),  # narrower for survival thresholds
) -> Dict[str, Any]:
    """
    DCA for survival / time-to-event using a risk score (higher = worse prognosis).

    We convert risk to an approximate "probability of event by time_horizon"
    using the empirical distribution at the chosen horizon (simple but robust).
    This allows the same net-benefit logic to be applied.

    For production use with full survival curves, feed 1 - S(t_horizon) as the
    "probability" input to the binary function instead.
    """
    duration = np.asarray(duration, dtype=float)
    event = np.asarray(event, dtype=float).astype(int)
    risk = np.asarray(risk, dtype=float)

    if not (len(duration) == len(event) == len(risk)):
        raise ValueError("duration, event, and risk must have identical length")

    if time_horizon is None:
        time_horizon = float(np.percentile(duration[event == 1], 75))

    # Approximate event probability by the chosen horizon (Kaplan-Meier style indicator)
    event_by_h = ((duration <= time_horizon) & (event == 1)).astype(float)

    # Use risk directly as the "score" (higher risk → higher chance of being classified positive)
    # Many survival DCA papers use the linear predictor or -S(t) directly.
    p_approx = 1 / (1 + np.exp(-0.8 * (risk - np.median(risk))))  # gentle monotonic map to (0,1)

    res = decision_curve_analysis_binary(
        y=event_by_h,
        p=p_approx,
        thresholds=thresholds,
        n_thresholds=n_thresholds,
        threshold_range=threshold_range,
    )

    res["mode"] = "survival"
    res["time_horizon"] = round(float(time_horizon), 2)
    res["note"] = (
        "Survival DCA uses an approximate event probability by the chosen time horizon "
        "derived from risk scores. For highest fidelity, pass 1 - S(t) from a full survival model."
    )
    if "assumptions" in res:
        res["assumptions"].append(
            f"Event probability approximated at t = {time_horizon:.1f} using monotonic transform of risk score."
        )
    return res
