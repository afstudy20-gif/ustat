"""Server-side Plotly render endpoint: POST /api/charts/render.

Verifies the figure-spec → image-bytes contract, format/MIME handling, input
validation, and graceful degradation when kaleido is absent.
"""

import importlib.util

import pytest
from fastapi.testclient import TestClient

from main import app
from services import plot_render

client = TestClient(app)

_HAS_KALEIDO = importlib.util.find_spec("kaleido") is not None

FIGURE = {
    "data": [{"type": "scatter", "mode": "lines", "x": [0, 1, 2], "y": [0, 1, 4]}],
    "layout": {"title": {"text": "t"}},
}


@pytest.mark.skipif(not _HAS_KALEIDO, reason="kaleido not installed")
def test_render_png_returns_image_bytes():
    r = client.post("/api/charts/render",
                    json={"figure": FIGURE, "format": "png", "width": 400, "height": 300})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"   # PNG magic number
    assert len(r.content) > 1000


@pytest.mark.skipif(not _HAS_KALEIDO, reason="kaleido not installed")
def test_render_svg_mime():
    r = client.post("/api/charts/render", json={"figure": FIGURE, "format": "svg"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/svg+xml"
    assert b"<svg" in r.content[:200]


def test_render_rejects_unknown_format():
    r = client.post("/api/charts/render", json={"figure": FIGURE, "format": "bmp"})
    assert r.status_code == 400


def test_render_rejects_malformed_figure():
    r = client.post("/api/charts/render", json={"figure": {"layout": {}}, "format": "png"})
    assert r.status_code == 400


def test_render_caps_dimensions():
    # Service clamps oversized dimensions rather than forwarding them to kaleido.
    assert plot_render._cap_dimension(99999) == plot_render.MAX_DIMENSION
    assert plot_render._cap_dimension(None) is None
    with pytest.raises(ValueError):
        plot_render._cap_dimension(0)


def test_service_unavailable_maps_to_503(monkeypatch):
    def _boom(*a, **k):
        raise plot_render.RenderUnavailable("nope")
    monkeypatch.setattr(plot_render, "render_figure", _boom)
    r = client.post("/api/charts/render", json={"figure": FIGURE})
    assert r.status_code == 503
