"""Canonical number formatting policy (services.number_format)."""
from services.number_format import (
    format_p, format_pct, format_num, format_or_hr, format_effect_ci, format_beta,
)


def test_format_p_policy():
    assert format_p(0.0005) == "<0.001"
    assert format_p(0.0009999) == "<0.001"
    assert format_p(0.001) == "0.001"
    assert format_p(0.0234) == "0.023"
    assert format_p(0.035) == "0.035"   # never rounded to 0.04
    assert format_p(0.043) == "0.043"
    assert format_p(0.42) == "0.420"
    assert format_p(1.0) == "1.000"
    assert format_p(None) == "—"
    assert format_p(float("nan")) == "—"


def test_format_p_prefix():
    assert format_p(0.0008, prefix=True) == "p<0.001"
    assert format_p(0.035, prefix=True) == "p=0.035"


def test_summary_stat_formatters():
    assert format_pct(45.3) == "45.3"
    assert format_pct(12.0) == "12.0"
    assert format_or_hr(1.456) == "1.46"
    assert format_effect_ci(1.45, 1.12, 1.89) == "1.45 (1.12–1.89)"
    assert format_effect_ci(1.45, None, None) == "1.45"
    assert format_beta(0.0234) == "0.023"
    assert format_num(None) == "—"
