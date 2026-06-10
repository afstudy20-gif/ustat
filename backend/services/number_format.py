"""Canonical numeric formatting for uSTAT — one policy, used everywhere.

Mirrors the frontend `lib/format.ts` so a value renders identically whether it
was formatted server-side (Table 1, Results paragraphs, DOCX export, figure
captions) or client-side. Import these instead of re-rolling `f"{p:.3f}"`.

P-value policy (journal standard, precision-preserving):
    p < 0.001  → "<0.001"   (never "0.000")
    otherwise  → 3 decimals (exact p; 0.035 stays 0.035, not 0.04)
Summary-stat policy:
    percentage → 1 decimal · OR/HR/RR + CI → 2 decimals · beta/r → 3 decimals.
Mean±SD / median[IQR] use the caller's column-aware decimals so every number in
one row (median, Q1, Q3) shares the same precision.
"""
from __future__ import annotations

import math

DASH = "—"  # em dash for missing


def _finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def format_p(p, *, prefix: bool = False) -> str:
    """Canonical p-value string: '<0.001' or exact 3-decimal (0.035 → '0.035').
    `prefix=True` → 'p<0.001' / 'p=0.035'."""
    if not _finite(p):
        return DASH
    n = float(p)
    if n < 0.001:
        return "p<0.001" if prefix else "<0.001"
    body = f"{n:.3f}"
    return f"p={body}" if prefix else body


def format_pct(x, decimals: int = 1) -> str:
    """Percentage value (the number only, no % sign) — 1 decimal by default."""
    if not _finite(x):
        return DASH
    return f"{float(x):.{decimals}f}"


def format_num(x, decimals: int = 2) -> str:
    """Generic fixed-decimal number with em-dash for missing/non-finite."""
    if not _finite(x):
        return DASH
    return f"{float(x):.{decimals}f}"


def format_or_hr(x, decimals: int = 2) -> str:
    """Odds/Hazard/Risk ratio — 2 decimals."""
    return format_num(x, decimals)


def format_effect_ci(estimate, lo, hi, decimals: int = 2) -> str:
    """'1.45 (1.12–1.89)' — effect estimate with 95% CI, 2 decimals."""
    if not _finite(estimate):
        return DASH
    if _finite(lo) and _finite(hi):
        return f"{float(estimate):.{decimals}f} ({float(lo):.{decimals}f}–{float(hi):.{decimals}f})"
    return f"{float(estimate):.{decimals}f}"


def format_beta(x, decimals: int = 3) -> str:
    """Regression coefficient / correlation r — 3 decimals."""
    return format_num(x, decimals)
