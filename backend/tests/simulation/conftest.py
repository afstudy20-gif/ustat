"""
Pytest + Hypothesis configuration for simulation tests.

This file provides:
- Custom Hypothesis strategies for realistic statistical data generation
- Common settings for simulation-based property testing
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from hypothesis import strategies as st
    from hypothesis.extra.numpy import arrays
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    st = None
    arrays = None
    HYPOTHESIS_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Hypothesis Strategies
# ──────────────────────────────────────────────────────────────────────────────

def numeric_predictors(
    n_rows: int = 300,
    n_cols: int = 5,
    min_value: float = -3.0,
    max_value: float = 3.0,
) -> st.SearchStrategy[pd.DataFrame]:
    """Generate a DataFrame of numeric predictors."""
    return arrays(
        dtype=np.float64,
        shape=(n_rows, n_cols),
        elements=st.floats(min_value=min_value, max_value=max_value, allow_nan=False, allow_infinity=False),
    ).map(lambda arr: pd.DataFrame(arr, columns=[f"X{i+1}" for i in range(n_cols)]))


def binary_outcome(n: int) -> st.SearchStrategy[np.ndarray]:
    return st.lists(st.integers(0, 1), min_size=n, max_size=n).map(np.array)


def realistic_coefficients(n: int) -> st.SearchStrategy[np.ndarray]:
    """Generate plausible coefficient vectors for simulation."""
    return arrays(
        dtype=np.float64,
        shape=(n,),
        elements=st.floats(-2.0, 2.0, allow_nan=False, allow_infinity=False),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Hypothesis Settings for Simulation Tests
# ──────────────────────────────────────────────────────────────────────────────

# We use a relatively high number of examples for simulation studies
# because statistical properties need more samples than typical unit tests.
SIMULATION_SETTINGS = {
    "max_examples": 50,          # Reasonable for CI while still giving good coverage
    "deadline": 5000,            # Some simulations can be slow
    "suppress_health_check": [], # Allow slow tests in this category
}
