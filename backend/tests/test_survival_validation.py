"""Range checks shared by the survival endpoints."""

import pandas as pd
import pytest
from fastapi import HTTPException
from services.survival_validation import validate_survival_inputs, warn_dev_eq_val


def test_accepts_clean_inputs():
    df = pd.DataFrame({"t": [10, 20, 30, 40, 50], "e": [0, 1, 0, 1, 1]})
    validate_survival_inputs(df, "t", "e")


def test_rejects_non_positive_duration():
    df = pd.DataFrame({"t": [10, -5, 20, 30], "e": [1, 0, 1, 0]})
    with pytest.raises(HTTPException) as ex:
        validate_survival_inputs(df, "t", "e")
    assert ex.value.status_code == 400
    assert "non-positive" in ex.value.detail


def test_rejects_zero_duration():
    df = pd.DataFrame({"t": [10, 0, 20], "e": [1, 0, 1]})
    with pytest.raises(HTTPException):
        validate_survival_inputs(df, "t", "e")


def test_rejects_non_binary_event():
    df = pd.DataFrame({"t": [10, 20, 30], "e": [1, 2, 0]})
    with pytest.raises(HTTPException) as ex:
        validate_survival_inputs(df, "t", "e")
    assert ex.value.status_code == 400
    assert "binary 0/1" in ex.value.detail


def test_rejects_missing_column():
    df = pd.DataFrame({"t": [10, 20], "e": [0, 1]})
    with pytest.raises(HTTPException):
        validate_survival_inputs(df, "fu_days", "e")


def test_dev_eq_val_warning():
    assert warn_dev_eq_val("sid_A", "sid_A") is not None
    assert warn_dev_eq_val("sid_A", "sid_B") is None
    assert warn_dev_eq_val(None, "sid_A") is None
