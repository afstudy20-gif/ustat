"""Tests for routers/article_parser.py.

POST /api/article_parser/parse takes a multipart file upload (PDF/DOCX/TXT) and
extracts statistical results via pure-Python regex parsing — no external/paid
API is involved (pdfplumber, PyPDF2, pdfminer, python-docx are all local
libraries). This means the endpoint is fully hermetically testable:

  * .txt uploads exercise the full extract-then-parse pipeline end-to-end
    without any third-party file-format library.
  * .docx uploads exercise python-docx (available in the venv) end-to-end.
  * .pdf extraction relies on pdfplumber/PyPDF2/pdfminer; since building a
    real PDF fixture requires an extra dependency (reportlab/PyPDF2, neither
    installed), the PDF branch is covered by monkeypatching pdfplumber.open
    so the router's control flow (calling _extract_text_from_pdf and feeding
    the result into _extract_stats) is still verified without a real PDF file.
  * The pure regex/parsing helper `_extract_stats` is additionally unit
    tested directly for a wide range of statistical-reporting formats and
    edge cases (empty string, unicode dashes, multiple findings, etc).
  * Input validation paths (no filename, unsupported extension, empty file)
    are tested via the API.
"""
import io
import math

import pytest

from routers.article_parser import _extract_stats, _parse_p, _clean_num, _cohens_d_from_t


# ── Input validation (no external dependency needed) ─────────────────────────

def test_parse_no_file_uploaded(client):
    # No file part in the multipart request at all -> FastAPI itself will
    # reject with 422 since `file` is a required UploadFile field.
    r = client.post("/api/article_parser/parse")
    assert r.status_code == 422


def test_parse_unsupported_extension(client):
    r = client.post(
        "/api/article_parser/parse",
        files={"file": ("results.xlsx", io.BytesIO(b"not really excel"), "application/octet-stream")},
    )
    assert r.status_code == 400
    assert "Unsupported file type" in r.json()["detail"]


def test_parse_empty_file(client):
    r = client.post(
        "/api/article_parser/parse",
        files={"file": ("results.txt", io.BytesIO(b""), "text/plain")},
    )
    assert r.status_code == 400
    assert "Empty file" in r.json()["detail"]


def test_parse_no_filename(client):
    # UploadFile with an empty filename string triggers the explicit
    # `if not file.filename` check in the router.
    r = client.post(
        "/api/article_parser/parse",
        files={"file": ("", io.BytesIO(b"hello"), "text/plain")},
    )
    assert r.status_code in (400, 422)


# ── TXT end-to-end (real text extraction path, no external lib) ──────────────

def test_parse_txt_extracts_findings(client):
    text = (
        "In this study (N = 120 participants), an independent-samples "
        "t-test showed t(118) = 2.34, p = 0.023. "
        "A correlation was found, r = 0.45, p < 0.001. "
        "The odds ratio was OR = 2.5 (95% CI 1.2-5.1)."
    )
    r = client.post(
        "/api/article_parser/parse",
        files={"file": ("article.txt", io.BytesIO(text.encode("utf-8")), "text/plain")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["filename"] == "article.txt"
    assert body["n_chars"] == len(text)
    assert body["n_findings"] >= 3
    types = {f["type"] for f in body["findings"]}
    assert "t_test" in types
    assert "correlation" in types
    assert "odds_ratio" in types
    assert isinstance(body["text_preview"], str) and body["text_preview"]


def test_parse_txt_no_extractable_stats_returns_empty_findings(client):
    text = "This document contains no statistical results whatsoever."
    r = client.post(
        "/api/article_parser/parse",
        files={"file": ("plain.txt", io.BytesIO(text.encode("utf-8")), "text/plain")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["n_findings"] == 0
    assert body["findings"] == []


def test_parse_txt_whitespace_only_is_422(client):
    r = client.post(
        "/api/article_parser/parse",
        files={"file": ("blank.txt", io.BytesIO(b"   \n\n   "), "text/plain")},
    )
    assert r.status_code == 422
    assert "No text could be extracted" in r.json()["detail"]


# ── DOCX end-to-end (python-docx is a real local dependency) ─────────────────

def test_parse_docx_extracts_findings(client):
    docx = pytest.importorskip("docx")
    doc = docx.Document()
    doc.add_paragraph("Sample size was 200 participants.")
    doc.add_paragraph("An ANOVA revealed F(2, 87) = 4.56, p = 0.013.")
    doc.add_paragraph("Cronbach's alpha = 0.82 indicated good reliability.")
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)

    r = client.post(
        "/api/article_parser/parse",
        files={"file": ("study.docx", buf, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["filename"] == "study.docx"
    types = {f["type"] for f in body["findings"]}
    assert "anova" in types
    assert "reliability" in types


def test_parse_docx_malformed_content_is_400(client):
    # Not a real docx zip/xml structure -> python-docx will raise, router
    # catches generic Exception and returns 400.
    r = client.post(
        "/api/article_parser/parse",
        files={"file": ("broken.docx", io.BytesIO(b"this is not a real docx file"), "application/octet-stream")},
    )
    assert r.status_code == 400
    assert "Failed to extract text" in r.json()["detail"]


# ── PDF path: monkeypatched at the pdfplumber boundary ────────────────────────
# A real PDF fixture would require reportlab or PyPDF2 (neither installed in
# this venv) to author the multipart body. Since pdfplumber itself is a pure
# local library (no network/paid API), we exercise the router's PDF branch by
# monkeypatching `pdfplumber.open` so the "text extraction -> regex parsing"
# control flow is verified without needing a real binary PDF fixture.

def test_parse_pdf_extracts_findings(client, monkeypatch):
    import pdfplumber

    class _FakePage:
        def extract_text(self):
            return "Chi-square test: chi2(2) = 8.45, p = 0.015."

    class _FakePDF:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        @property
        def pages(self):
            return [_FakePage()]

    monkeypatch.setattr(pdfplumber, "open", lambda *_a, **_k: _FakePDF())

    r = client.post(
        "/api/article_parser/parse",
        files={"file": ("paper.pdf", io.BytesIO(b"%PDF-1.4 fake bytes"), "application/pdf")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    types = {f["type"] for f in body["findings"]}
    assert "chi_square" in types


def test_parse_pdf_no_library_available_is_500(client, monkeypatch):
    import routers.article_parser as ap

    # Force every internal fallback to fail by patching the whole helper.
    monkeypatch.setattr(
        ap, "_extract_text_from_pdf",
        lambda _b: (_ for _ in ()).throw(
            __import__("fastapi").HTTPException(status_code=500, detail="No PDF library available")
        ),
    )
    r = client.post(
        "/api/article_parser/parse",
        files={"file": ("paper.pdf", io.BytesIO(b"%PDF-1.4 fake bytes"), "application/pdf")},
    )
    assert r.status_code == 500
    assert "No PDF library available" in r.json()["detail"]


# ── Unit tests for the pure parsing helpers (no HTTP layer at all) ───────────

def test_parse_p_handles_lt_and_spaces():
    assert _parse_p("<0.001") == 0.001
    assert _parse_p("0.023") == 0.023
    assert _parse_p("0. 001") == 0.001
    assert _parse_p("garbage") == 0.001  # unparseable default


def test_clean_num_strips_pdf_artifacts():
    assert _clean_num("45. 2") == 45.2
    assert _clean_num("45.2") == 45.2


def test_cohens_d_from_t():
    d = _cohens_d_from_t(2.0, 100)
    assert math.isclose(d, 2 * 2.0 / math.sqrt(100))


def test_extract_stats_empty_string_returns_empty_list():
    assert _extract_stats("") == []


def test_extract_stats_t_test_and_effect_size():
    findings = _extract_stats("t(48) = 2.34, p = 0.023")
    assert len(findings) == 1
    f = findings[0]
    assert f["type"] == "t_test"
    assert f["df"] == 48
    assert f["statistic"] == 2.34
    assert f["p"] == 0.023
    assert f["effect_size"] > 0


def test_extract_stats_hazard_ratio_with_ci():
    findings = _extract_stats("HR = 1.8 (95% CI: 1.2-2.7)")
    hr = [f for f in findings if f["type"] == "hazard_ratio"]
    assert len(hr) == 1
    assert hr[0]["statistic"] == 1.8
    assert hr[0]["ci_low"] == 1.2
    assert hr[0]["ci_high"] == 2.7


def test_extract_stats_proportions_cohens_h():
    findings = _extract_stats("45% vs 32% of patients responded.")
    prop = [f for f in findings if f["type"] == "proportions"]
    assert len(prop) == 1
    assert prop[0]["p1"] == 0.45
    assert prop[0]["p2"] == 0.32
    assert prop[0]["effect_size"] > 0


def test_extract_stats_unicode_dash_and_negative_correlation():
    findings = _extract_stats("R = −0.45, p < 0.001")
    corr = [f for f in findings if f["type"] == "correlation"]
    assert len(corr) == 1
    assert corr[0]["statistic"] == -0.45
    assert corr[0]["effect_size"] == 0.45


def test_extract_stats_multiple_findings_in_one_document():
    text = (
        "N = 250 participants were enrolled. "
        "t(248) = 3.1, p = 0.002. "
        "F(2, 240) = 5.6, p = 0.004. "
        "OR = 1.9 (95% CI 1.1-3.2). "
        "Cronbach's alpha = 0.88."
    )
    findings = _extract_stats(text)
    types = {f["type"] for f in findings}
    assert {"t_test", "anova", "odds_ratio", "reliability"}.issubset(types)


def test_extract_stats_invalid_eta_squared_out_of_range_ignored():
    # eta² >= 1 should not produce a finding via the standalone branch.
    findings = _extract_stats("eta² = 1.5")
    assert all(f["type"] != "effect_size" for f in findings)
