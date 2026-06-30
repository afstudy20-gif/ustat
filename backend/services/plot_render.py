"""Server-side static rendering of Plotly figures to image bytes.

The frontend renders charts interactively with Plotly.js and exports PNG/SVG
client-side (see ``PlotExporter``). This service provides the *headless*
counterpart: turn a Plotly figure spec (the same ``{data, layout}`` the client
already builds) into a static image on the server, for API clients, scheduled
reports, and PDF pipelines that have no browser.

Rendering uses Plotly's ``to_image`` (kaleido). Both libraries bundle a
headless Chromium and are heavy, so the import is **lazy**: a deployment that
omits them still boots, and the ``/api/charts/render`` endpoint returns a clear
503 instead of crashing at startup.

Styling lives only in the caller's figure spec — this module never builds or
mutates traces, so there is no risk of the server drifting from what the client
draws.
"""
from __future__ import annotations

from typing import Any, Optional

# Formats kaleido can emit. Kept explicit so an unexpected value fails fast with
# a 400 rather than surfacing as an opaque kaleido error.
ALLOWED_FORMATS = frozenset({"png", "svg", "jpeg", "pdf", "webp"})

MIME_TYPES = {
    "png": "image/png",
    "svg": "image/svg+xml",
    "jpeg": "image/jpeg",
    "pdf": "application/pdf",
    "webp": "image/webp",
}

# Guard rails against resource abuse: a single render must not ask kaleido for a
# poster-sized canvas at 4x scale. These bound pixels, not correctness.
MAX_DIMENSION = 4000
MAX_SCALE = 4.0


class RenderUnavailable(RuntimeError):
    """Raised when plotly/kaleido are not installed on the server."""


def _cap_dimension(value: Optional[int]) -> Optional[int]:
    """Clamp a width/height to (0, MAX_DIMENSION]; ``None`` keeps the figure's
    own layout size."""
    if value is None:
        return None
    v = int(value)
    if v <= 0:
        raise ValueError("width/height must be positive")
    return min(v, MAX_DIMENSION)


def render_figure(
    figure: Any,
    fmt: str = "png",
    width: Optional[int] = None,
    height: Optional[int] = None,
    scale: float = 2.0,
) -> bytes:
    """Render a Plotly figure spec to image bytes.

    Args:
        figure: A Plotly figure as a dict — must contain a ``data`` key
            (``layout`` is optional). This is the exact structure the frontend
            passes to ``react-plotly.js``.
        fmt: One of :data:`ALLOWED_FORMATS`.
        width / height: Output size in pixels; ``None`` uses the figure's own
            layout dimensions. Capped at :data:`MAX_DIMENSION`.
        scale: Device-pixel multiplier (e.g. 2 for retina / print). Capped at
            :data:`MAX_SCALE`.

    Returns:
        The encoded image as ``bytes``.

    Raises:
        ValueError: Bad format, malformed figure, or non-positive dimension.
        RenderUnavailable: plotly/kaleido not installed.
    """
    fmt = (fmt or "png").lower()
    if fmt not in ALLOWED_FORMATS:
        raise ValueError(
            f"Unsupported format '{fmt}'. Allowed: {sorted(ALLOWED_FORMATS)}"
        )
    if not isinstance(figure, dict) or "data" not in figure:
        raise ValueError("figure must be a dict with a 'data' key")

    width = _cap_dimension(width)
    height = _cap_dimension(height)
    try:
        scale_val = min(float(scale or 1.0), MAX_SCALE)
    except (TypeError, ValueError):
        raise ValueError("scale must be a number")
    if scale_val <= 0:
        raise ValueError("scale must be positive")

    try:
        import plotly.io as pio
    except ImportError as exc:  # pragma: no cover - depends on deploy extras
        raise RenderUnavailable(
            "Server-side chart rendering is unavailable: install 'plotly' and "
            "'kaleido' to enable POST /api/charts/render."
        ) from exc

    return pio.to_image(
        figure, format=fmt, width=width, height=height, scale=scale_val
    )
