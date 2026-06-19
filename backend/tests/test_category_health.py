"""Rare-category warnings for regression endpoints.

Dirty/typed categorical levels (a sex column with "M", "F", "x", "Female")
used to silently dummy-encode into extra predictors with n<5, destabilising
the fit. The endpoint now surfaces a warning so the user knows.
"""

import pandas as pd
from services.category_health import rare_level_warnings


def test_rare_level_warnings_basic():
    df = pd.DataFrame({"sex": ["M"] * 50 + ["F"] * 40 + ["x"] * 1 + ["Female"] * 1})
    out = rare_level_warnings(df, ["sex"])
    assert len(out) == 1
    w = out[0]
    assert w["variable"] == "sex"
    assert {r["level"] for r in w["rare_levels"]} == {"x", "Female"}
    assert {r["level"] for r in w["kept_levels"]} == {"M", "F"}


def test_skips_clean_binary():
    # M / F only — no rare levels, no warning
    df = pd.DataFrame({"sex": ["M"] * 50 + ["F"] * 50})
    assert rare_level_warnings(df, ["sex"]) == []


def test_skips_numeric():
    df = pd.DataFrame({"age": [30, 40, 50, 60, 70] * 20})
    assert rare_level_warnings(df, ["age"]) == []


def test_no_warning_when_below_three_levels():
    # 2 levels even if one is rare — get_dummies produces only 1 extra column,
    # no dummy bloat to warn about.
    df = pd.DataFrame({"sex": ["M"] * 99 + ["x"] * 1})
    assert rare_level_warnings(df, ["sex"]) == []
