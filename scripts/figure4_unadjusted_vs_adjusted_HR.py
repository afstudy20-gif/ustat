#!/usr/bin/env python3
"""
Figure 4 — Unadjusted vs Adjusted hazard ratios (paired Cox forest plot).

Each row shows two estimates for one variable:
  - Unadjusted (univariable Cox)  -> grey circle
  - Adjusted   (multivariable Cox) -> blue square

Right-hand columns print the numeric HR (95% CI). Adjusted entries that were
not retained in the multivariable model print "—".

Edit the ROWS list with your own values, then run:

    python3 scripts/figure4_unadjusted_vs_adjusted_HR.py

Output: Figure_4_unadjusted_vs_adjusted_HR.png (300 DPI, white background).

Only dependency: matplotlib  ->  pip install matplotlib
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── Style ─────────────────────────────────────────────────────────────────────
GREY = "#9aa0a6"   # unadjusted
BLUE = "#1f4e79"   # adjusted
OFFSET = 0.18       # vertical separation of the two markers within a row
PLT_FONT = "DejaVu Sans"

plt.rcParams.update({
    "font.family": PLT_FONT,
    "font.size": 13,
    "axes.linewidth": 1.0,
    "svg.fonttype": "none",
})


@dataclass
class Row:
    label: str
    # Unadjusted: point estimate + 95% CI (always shown)
    u_hr: float
    u_lo: float
    u_hi: float
    # Adjusted: point estimate + optional 95% CI (None CI -> marker only).
    # a_hr None -> variable dropped from the multivariable model ("—").
    a_hr: Optional[float] = None
    a_ci: Optional[Tuple[float, float]] = None
    # Override the printed adjusted text (e.g. show CI for some rows, point
    # estimate only for others, "—" when dropped). Auto-derived if None.
    a_text: Optional[str] = None


def _ci(hr: float, lo: float, hi: float) -> str:
    return f"{hr:.2f} ({lo:.2f}–{hi:.2f})"


# ── DATA — replace with your real model output ─────────────────────────────────
# Order top-to-bottom as you want them to appear.
ROWS = [
    Row("LDL-C <100 mg/dL (vs >130)",   2.03, 1.02, 4.03, a_hr=1.43, a_ci=(0.69, 2.96)),
    Row("LDL-C 100–130 mg/dL (vs >130)", 0.75, 0.35, 1.63, a_hr=None),
    Row("Age, per year",                1.08, 1.06, 1.11, a_hr=1.07, a_ci=(1.04, 1.10), a_text="1.07"),
    Row("Male sex",                     0.50, 0.27, 0.95, a_hr=1.08, a_ci=(0.55, 2.12), a_text="1.08"),
    Row("Diabetes mellitus",            2.29, 1.29, 4.07, a_hr=1.92, a_ci=(1.05, 3.51), a_text="1.92"),
    Row("Ejection fraction, per %",     0.93, 0.90, 0.95, a_hr=0.95, a_ci=(0.92, 0.98), a_text="0.95"),
    Row("Blood urea, per mg/dL",        1.03, 1.00, 1.06, a_hr=None),
]

XLABEL = "Hazard ratio for all-cause mortality (log scale)"
XTICKS = [0.5, 1, 2, 4]
OUTFILE = "Figure_4_unadjusted_vs_adjusted_HR.png"
# ───────────────────────────────────────────────────────────────────────────────


def main() -> None:
    n = len(ROWS)
    # Row 0 at top: give descending y so first row is highest.
    ys = list(range(n, 0, -1))

    fig, ax = plt.subplots(figsize=(15, 0.95 * n + 1.6))

    for y, r in zip(ys, ROWS):
        # Unadjusted (grey circle, full CI) — slightly above row centre.
        yu = y + OFFSET
        ax.plot([r.u_lo, r.u_hi], [yu, yu], color=GREY, lw=2.0, zorder=2,
                solid_capstyle="butt")
        # CI end caps
        for xc in (r.u_lo, r.u_hi):
            ax.plot([xc, xc], [yu - 0.07, yu + 0.07], color=GREY, lw=2.0, zorder=2)
        ax.plot(r.u_hr, yu, "o", color=GREY, ms=10, zorder=3)

        # Adjusted (blue square) — slightly below row centre.
        if r.a_hr is not None:
            ya = y - OFFSET
            if r.a_ci is not None:
                ax.plot([r.a_ci[0], r.a_ci[1]], [ya, ya], color=BLUE, lw=2.0, zorder=2,
                        solid_capstyle="butt")
                for xc in r.a_ci:
                    ax.plot([xc, xc], [ya - 0.07, ya + 0.07], color=BLUE, lw=2.0, zorder=2)
            ax.plot(r.a_hr, ya, "s", color=BLUE, ms=9, zorder=3)

    # Reference line at HR = 1
    ax.axvline(1.0, color="#9aa0a6", ls="--", lw=1.2, zorder=1)

    # Axes
    ax.set_xscale("log")
    ax.set_xticks(XTICKS)
    ax.set_xticklabels([str(t) for t in XTICKS])
    ax.set_xlim(min(XTICKS) * 0.7, max(XTICKS) * 1.4)
    ax.set_yticks(ys)
    ax.set_yticklabels([r.label for r in ROWS])
    ax.set_ylim(0.3, n + 1.1)
    ax.set_xlabel(XLABEL)
    ax.tick_params(axis="y", length=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    # Right-hand numeric columns (figure coords via blended transform).
    x_unadj = 1.05   # axes-fraction x for the Unadjusted column
    x_adj = 1.34     # axes-fraction x for the Adjusted column
    trans = ax.get_yaxis_transform()  # x in axes frac, y in data coords

    ax.text(x_unadj, n + 0.9, "Unadjusted", transform=trans, color=GREY,
            fontsize=13, fontweight="bold", ha="left", va="center", clip_on=False)
    ax.text(x_adj, n + 0.9, "Adjusted", transform=trans, color=BLUE,
            fontsize=13, fontweight="bold", ha="left", va="center", clip_on=False)

    for y, r in zip(ys, ROWS):
        ax.text(x_unadj, y, _ci(r.u_hr, r.u_lo, r.u_hi), transform=trans,
                color=GREY, ha="left", va="center", clip_on=False)
        if r.a_text is not None:
            a_txt = r.a_text
        elif r.a_hr is None:
            a_txt = "—"
        elif r.a_ci is not None:
            a_txt = _ci(r.a_hr, r.a_ci[0], r.a_ci[1])
        else:
            a_txt = f"{r.a_hr:.2f}"
        ax.text(x_adj, y, a_txt, transform=trans, color=BLUE,
                ha="left", va="center", clip_on=False)

    # Legend
    handles = [
        Line2D([0], [0], marker="o", color=GREY, lw=2, ms=10, label="Unadjusted"),
        Line2D([0], [0], marker="s", color=BLUE, lw=2, ms=9, label="Adjusted"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=12,
              bbox_to_anchor=(0.62, 0.02))

    fig.subplots_adjust(left=0.26, right=0.70, top=0.93, bottom=0.13)
    fig.savefig(OUTFILE, dpi=300, facecolor="white")
    print(f"wrote {OUTFILE}")


if __name__ == "__main__":
    main()
