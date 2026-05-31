"""
models.py

This file is intentionally kept very small after the major refactoring.
Most statistical logic has been moved to dedicated routers under routers/
( regression.py, survival.py, psm.py, rcs.py, etc. ).

It now only re-exports a couple of very commonly used helpers so that
existing code and tests continue to work with minimal changes.
"""

from fastapi import APIRouter

from services.store import get_df as _get_df
from services.regression import compute_vif as _compute_vif

router = APIRouter()

# Re-export for backward compatibility with older imports
get_df = _get_df
compute_vif = _compute_vif


