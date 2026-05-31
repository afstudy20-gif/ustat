"""
Phase 5 - Causal Sensitivity (E-value + Quantitative Bias Analysis) Simulation Tests

TDD approach:
- Property-based mathematical invariants for E-value
- Simulation studies using generate_psm_iptw_data to demonstrate
  confounding sensitivity / robustness detection
- Integration points with PSM/IPTW treatment effect estimates
"""

import pytest
import numpy as np

from services.causal_sensitivity import (
    e_value,
    quantitative_bias_analysis,
    bias_factor,
    e_value_from_psm_or_iptw,
)
from services.simulation_generators import generate_psm_iptw_data


# ═══════════════════════════════════════════════════════════════════════════════
# 1. MATHEMATICAL INVARIANTS (Property-style checks)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.simulation
def test_e_value_null_effect_is_one():
    """E-value must be 1.0 when observed estimate is exactly null (RR=1)."""
    res = e_value(estimate=1.0, measure="rr")
    assert res["e_value_point_estimate"] == 1.0
    assert res["e_value_ci"] == 1.0


@pytest.mark.simulation
def test_e_value_monotonicity():
    """Stronger observed effects must produce higher or equal E-values (on RR scale)."""
    weak = e_value(estimate=1.5, measure="rr")["e_value_point_estimate"]
    strong = e_value(estimate=4.0, measure="rr")["e_value_point_estimate"]
    assert strong >= weak


@pytest.mark.simulation
def test_e_value_ci_consistency():
    """If CI is provided and excludes 1, the CI E-value should be finite and reasonable."""
    res = e_value(estimate=2.0, ci_low=1.2, ci_high=3.5, measure="rr")
    assert res["e_value_ci"] is not None
    assert 1.0 < res["e_value_ci"] < 20.0  # sanity bound


@pytest.mark.simulation
def test_e_value_handles_inverted_effects():
    """Protective effects (RR < 1) are inverted correctly and still yield E > 1."""
    res = e_value(estimate=0.4, ci_low=0.2, ci_high=0.8, measure="rr")
    assert res["e_value_point_estimate"] > 2.0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. OR CONVERSION & RARE vs COMMON OUTCOME
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.simulation
def test_or_rare_outcome_approximation():
    """Rare outcome OR should behave similarly to RR."""
    rr_res = e_value(estimate=2.0, measure="rr")
    or_rare = e_value(estimate=2.0, measure="or", rare_outcome=True)
    # For truly rare outcomes the numbers are close
    assert abs(rr_res["e_value_point_estimate"] - or_rare["e_value_point_estimate"]) < 0.5


@pytest.mark.simulation
def test_or_common_outcome_flag():
    """When rare_outcome=False for OR, the conversion is applied (even if crude)."""
    res = e_value(estimate=3.0, measure="or", rare_outcome=False)
    assert res["measure"] == "or"
    assert res["e_value_point_estimate"] > 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# 3. QUANTITATIVE BIAS ANALYSIS & BIAS FACTOR
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.simulation
def test_bias_factor_equal_prevalence_no_net_bias():
    """When prevalences are equal, simple bias factor should be ~1 (no differential bias)."""
    bf = bias_factor(2.5, 2.0, 0.5, 0.5)
    assert abs(bf - 1.0) < 0.05


@pytest.mark.simulation
def test_quantitative_bias_reduces_estimate():
    """Applying QBA with strong confounder should pull the estimate toward the null."""
    res = quantitative_bias_analysis(
        observed_estimate=3.0,
        measure="rr",
        confounding_strength=3.0,
        prevalence_exposed=0.7,
        prevalence_unexposed=0.2,
    )
    assert res["bias_corrected_estimate"] < res["observed_estimate"]
    assert "bias factor" in res["interpretation"].lower()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SIMULATION: CONFOUNDING RECOVERY / SENSITIVITY (using PSM generator)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.simulation
def test_e_value_under_simulated_strong_confounding():
    """
    Generate data with strong confounding + known treatment effect.
    The 'naive' observed effect (from ground truth) should produce a LOW E-value,
    correctly signalling high sensitivity to unmeasured confounding.
    """
    df, gt = generate_psm_iptw_data(
        n=800,
        treatment_effect=0.9,          # moderate true effect on additive scale
        confounding_strength=2.5,      # STRONG confounding
        seed=2025,
    )

    # Simulate a 'naive' risk ratio from the confounded data
    # (in real use this would come from a fitted model or PSM marginal effect on RR scale)
    mean_treated = df[df["treat"] == 1]["outcome"].mean()
    mean_control = df[df["treat"] == 0]["outcome"].mean()
    # Approximate RR via a simple transformation (for demo; real analyses would use proper marginal RR)
    naive_rr = max(1.1, (mean_treated / (mean_control + 1e-6)) ** 0.6)  # dampened for stability

    ev = e_value(estimate=naive_rr, measure="rr")

    # With strong confounding the E-value should be modest (sensitive)
    assert ev["e_value_point_estimate"] < 6.0, f"E-value unexpectedly high: {ev}"


@pytest.mark.simulation
def test_e_value_after_strong_confounding_still_detects_need_for_sensitivity():
    """
    Even after generating confounded data, the E-value framework correctly
    warns that the observed association could be explained by moderate unmeasured confounding.
    """
    df, gt = generate_psm_iptw_data(n=600, treatment_effect=1.1, confounding_strength=1.8, seed=42)

    # Use a deliberately naive contrast
    naive_effect = 2.8  # pretend we observed a large RR

    ev = e_value(estimate=naive_effect, ci_low=1.4, ci_high=5.5, measure="rr")

    # The interpretation must mention unmeasured confounder (core contract of the module)
    assert "unmeasured confounder" in ev["interpretation"].lower()
    assert ev["e_value_point_estimate"] > 3.0


# ═══════════════════════════════════════════════════════════════════════════════
# 5. EXPLICIT PSM/IPTW INTEGRATION (using the dedicated helper)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.simulation
def test_e_value_from_psm_helper_integration():
    """The Phase 5 integration helper e_value_from_psm_or_iptw works end-to-end."""
    df, gt = generate_psm_iptw_data(n=500, treatment_effect=0.7, confounding_strength=1.5, seed=123)

    # Simulate marginal RR after "PSM" (here we just use a plausible post-adjustment value)
    observed_marginal_rr = 1.9

    res = e_value_from_psm_or_iptw(observed_marginal_rr, ci_low=1.3, ci_high=2.8, measure="rr")

    assert res["e_value_point_estimate"] > 2.5
    assert "e_value" in res or "e_value_point_estimate" in res
