"""
Article Parser router — extract statistical parameters from PDF/DOCX for power analysis.

Endpoint
--------
POST /parse  — Upload PDF or DOCX, extract statistical results via regex patterns.
"""

from __future__ import annotations

import io
import math
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

router = APIRouter()


# ── Regex patterns for common statistical reporting ──────────────────────────
# Patterns are intentionally flexible to handle PDF text extraction artifacts
# (extra spaces, unicode chars, line breaks within values, etc.)

_S = r'[\s\u00a0]*'  # flexible space (incl. non-breaking)
_NUM = r'[\d]+[.\s]?[\d]*'  # number that may have space in decimal (PDF artifact)
_P_VAL = r'[pP]' + _S + r'[=<≤>]' + _S + r'([\d.<\s]+)'  # p = 0.023 or p < 0.001
_DASH = r'[\-–—−\u2012\u2013\u2014\u2212,\s]+'  # any dash variant + spaces
_CI_BLOCK = r'(?:\(?' + _S + r'95' + _S + r'%?' + _S + r'CI' + _S + r'[:\s=]*' + _S + r'([\d.]+)' + _DASH + r'([\d.]+)' + _S + r'\)?)?'

# t-test: t(48) = 2.34, p = 0.023 / t = 2.34, df = 48, p < 0.001
_T_TEST = re.compile(
    r't' + _S + r'\(' + _S + r'(\d+)' + _S + r'\)' + _S + r'=' + _S + r'([\d.]+)' + _S + r',?' + _S + _P_VAL
    + r'|t' + _S + r'=' + _S + r'([\d.]+)' + _S + r',?' + _S + r'(?:df' + _S + r'=' + _S + r'(\d+)' + _S + r',?' + _S + r')?' + _P_VAL,
    re.IGNORECASE
)

# Cohen's d: d = 0.67 / Cohen's d = 0.67
_COHENS_D = re.compile(
    r"(?:Cohen['\u2019]?s?\s+)?d\s*=\s*([\d.]+)",
    re.IGNORECASE
)

# ANOVA: F(2, 87) = 4.56, p = 0.013
_ANOVA_F = re.compile(
    r'F' + _S + r'\(' + _S + r'(\d+)' + _S + r',' + _S + r'(\d+)' + _S + r'\)' + _S + r'=' + _S + r'([\d.]+)' + _S + r',?' + _S + _P_VAL,
    re.IGNORECASE
)

# Eta squared: η² = 0.095 / eta² = 0.095 / partial η² = 0.12
_ETA_SQ = re.compile(
    r'(?:partial\s+)?(?:\u03B7|eta)\s*[²2]\s*=\s*([\d.]+)',
    re.IGNORECASE
)

# Correlation: r = 0.45, p < 0.001 / ρ = 0.32 / R = −0.45
_CORR = re.compile(
    r'(?:\b[rR]|ρ|\u03C1)\s*=\s*([−\-\u2212]?[\d.]+)\s*,?\s*' + _P_VAL,
    re.IGNORECASE
)

# Odds Ratio — very flexible: OR = 2.5 / OR: 2.5 / OR 2.5 (95% CI 1.2-5.1) / OR 2.5 [1.2–5.1]
_OR = re.compile(
    r'(?:OR|[Oo]dds\s*[Rr]atio)' + _S + r'[=:\s]' + _S + r'([\d.]+)' + _S
    + r'(?:'
    + r'\(?' + _S + r'(?:95' + _S + r'%?' + _S + r'CI' + _S + r'[:\s=]*)?' + _S + r'([\d.]+)' + _DASH + r'([\d.]+)' + _S + r'[\)\]]?'
    + r')?',
    re.IGNORECASE
)

# Hazard Ratio — very flexible: HR = 1.8 / HR: 1.8 / HR 1.8 (95% CI: 1.2-2.7)
_HR = re.compile(
    r'(?:HR|[Hh]azard\s*[Rr]atio)' + _S + r'[=:\s]' + _S + r'([\d.]+)' + _S
    + r'(?:'
    + r'\(?' + _S + r'(?:95' + _S + r'%?' + _S + r'CI' + _S + r'[:\s=]*)?' + _S + r'([\d.]+)' + _DASH + r'([\d.]+)' + _S + r'[\)\]]?'
    + r')?',
    re.IGNORECASE
)

# Risk Ratio: RR = 1.5 (95% CI: 1.1-2.0)
_RR = re.compile(
    r'(?:RR|[Rr]elative\s*[Rr]isk|[Rr]isk\s*[Rr]atio)' + _S + r'[=:\s]' + _S + r'([\d.]+)' + _S
    + r'(?:'
    + r'\(?' + _S + r'(?:95' + _S + r'%?' + _S + r'CI' + _S + r'[:\s=]*)?' + _S + r'([\d.]+)' + _DASH + r'([\d.]+)' + _S + r'[\)\]]?'
    + r')?',
    re.IGNORECASE
)

# Chi-square: χ²(2) = 8.45, p = 0.015 / χ2 = 8.45 / chi-square = 8.45
_CHI2 = re.compile(
    r'(?:\u03C7|\u03A7|chi)[\s\-]*(?:square)?[\s²2]*' + _S + r'\(?(\d+)?\)?' + _S + r'=' + _S + r'([\d.]+)' + _S + r',?' + _S + _P_VAL,
    re.IGNORECASE
)

# p-value standalone: p = 0.023 / P < 0.001 / p-value = 0.045
_P_STANDALONE = re.compile(
    r'(?:p[\s-]*value|[pP])' + _S + r'[=<≤]' + _S + r'([\d.]+(?:\s*[\d]+)?)',
)

# Mean ± SD: 45.2 ± 12.3 / 45.2±12.3 / 45.2 +/- 12.3
_MEAN_SD = re.compile(
    r'([\d.]+)\s*[±\+/\u00b1]\s*-?\s*([\d.]+)',
)

# Sample size: n = 120 / N = 450 / sample size of 120 / (n = 85)
_SAMPLE_N = re.compile(
    r'(?:(?:sample\s+size|total|enrolled|included|recruited|participants?)\s+(?:of\s+|was\s+|were\s+)?|[nN]\s*=\s*)\(?(\d{2,})\)?',
    re.IGNORECASE
)

# AUC: AUC = 0.82 / AUC 0.82 / c-statistic = 0.78 / AUC of 0.82 / AUROC 0.82
_AUC = re.compile(
    r'(?:AUC|[cC][\s-]*statistic|AUROC)' + _S + r'(?:of|was|=|:)?' + _S + r'(0\.\d+|1\.0+)',
    re.IGNORECASE
)

# Proportions: 45% vs 32% / 45.2% vs. 32.1% / 45% versus 32%
_PROPORTIONS = re.compile(
    r'(\d+(?:\.\d+)?)\s*%\s*(?:vs\.?|versus|compared\s+(?:to|with))\s*(\d+(?:\.\d+)?)\s*%',
    re.IGNORECASE
)

# Beta coefficient: β = 0.45 / beta = 0.45
_BETA = re.compile(
    r'(?:\u03B2|[Bb]eta)\s*[=:]\s*([−\-\u2212]?[\d.]+)\s*,?\s*' + _P_VAL,
    re.IGNORECASE
)

# Confidence interval standalone: 95% CI 1.2–5.1 / 95%CI: 1.2-5.1
_CI_STANDALONE = re.compile(
    r'95\s*%?\s*CI\s*[:\s=]*\s*([\d.]+)' + _DASH + r'([\d.]+)',
    re.IGNORECASE
)


def _parse_p(p_str: str) -> float:
    """Parse p-value string like '<0.001' or '0.023' or '0. 001'."""
    p_str = p_str.strip().replace("<", "").replace("≤", "").replace(">", "").replace(" ", "")
    try:
        return float(p_str)
    except ValueError:
        return 0.001  # default for unparseable


def _clean_num(s: str) -> float:
    """Parse a number that may have PDF artifacts like spaces in decimal."""
    return float(s.strip().replace(" ", "").replace("\u00a0", ""))


def _cohens_d_from_t(t: float, n: int) -> float:
    """Approximate Cohen's d from t-statistic and total df."""
    return 2 * t / math.sqrt(n)


def _cohens_f_from_eta2(eta2: float) -> float:
    """Convert eta-squared to Cohen's f."""
    if eta2 >= 1:
        return 1.0
    return math.sqrt(eta2 / (1 - eta2))


def _cohens_h(p1: float, p2: float) -> float:
    """Cohen's h for two proportions."""
    return 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))


def _extract_stats(text: str) -> List[Dict[str, Any]]:
    """Extract all statistical results from text using regex patterns."""
    findings: List[Dict[str, Any]] = []

    # Sample sizes
    sample_ns = [int(m.group(1)) for m in _SAMPLE_N.finditer(text)]
    default_n = max(sample_ns) if sample_ns else None

    # t-tests
    for m in _T_TEST.finditer(text):
        if m.group(1):  # t(df) = stat format
            df = int(m.group(1))
            t_val = float(m.group(2))
            p_val = _parse_p(m.group(3))
        else:  # t = stat format
            t_val = float(m.group(4))
            df = int(m.group(5)) if m.group(5) else (default_n - 2 if default_n else 48)
            p_val = _parse_p(m.group(6))
        d = _cohens_d_from_t(t_val, df)
        findings.append({
            "type": "t_test",
            "test_label": "t-test",
            "power_test": "t_two",
            "statistic": round(t_val, 4),
            "df": df,
            "p": round(p_val, 6),
            "effect_size": round(abs(d), 4),
            "effect_label": "Cohen's d",
            "n_approx": df + 2,
            "source": m.group(0).strip(),
        })

    # Cohen's d (standalone, not already captured)
    for m in _COHENS_D.finditer(text):
        d_val = float(m.group(1))
        # Check it's not already associated with a t-test
        already = any(f["type"] == "t_test" and abs(f["effect_size"] - d_val) < 0.01 for f in findings)
        if not already:
            findings.append({
                "type": "effect_size",
                "test_label": "Cohen's d",
                "power_test": "t_two",
                "effect_size": round(d_val, 4),
                "effect_label": "Cohen's d",
                "n_approx": default_n,
                "source": m.group(0).strip(),
            })

    # ANOVA F-tests
    for m in _ANOVA_F.finditer(text):
        df1 = int(m.group(1))
        df2 = int(m.group(2))
        f_val = float(m.group(3))
        p_val = _parse_p(m.group(4))
        k = df1 + 1
        # Approximate eta² from F
        eta2 = (f_val * df1) / (f_val * df1 + df2)
        f_effect = _cohens_f_from_eta2(eta2)
        findings.append({
            "type": "anova",
            "test_label": "ANOVA",
            "power_test": "anova",
            "statistic": round(f_val, 4),
            "df1": df1, "df2": df2,
            "p": round(p_val, 6),
            "effect_size": round(f_effect, 4),
            "effect_label": "Cohen's f",
            "eta_squared": round(eta2, 4),
            "k_groups": k,
            "n_approx": df1 + df2 + 1,
            "source": m.group(0).strip(),
        })

    # Eta-squared (standalone)
    for m in _ETA_SQ.finditer(text):
        eta2 = float(m.group(1))
        already = any(f.get("eta_squared") and abs(f["eta_squared"] - eta2) < 0.001 for f in findings)
        if not already and 0 < eta2 < 1:
            findings.append({
                "type": "effect_size",
                "test_label": "η²",
                "power_test": "anova",
                "effect_size": round(_cohens_f_from_eta2(eta2), 4),
                "effect_label": "Cohen's f",
                "eta_squared": round(eta2, 4),
                "k_groups": 3,
                "source": m.group(0).strip(),
            })

    # Correlations
    for m in _CORR.finditer(text):
        r_val = float(m.group(1).replace("−", "-"))
        p_val = _parse_p(m.group(2))
        findings.append({
            "type": "correlation",
            "test_label": "Correlation",
            "power_test": "correlation",
            "statistic": round(r_val, 4),
            "p": round(p_val, 6),
            "effect_size": round(abs(r_val), 4),
            "effect_label": "r",
            "n_approx": default_n,
            "source": m.group(0).strip(),
        })

    # Odds Ratios
    for m in _OR.finditer(text):
        or_val = float(m.group(1))
        ci_lo = float(m.group(2)) if m.group(2) else None
        ci_hi = float(m.group(3)) if m.group(3) else None
        # Convert OR to Cohen's d approximation: d ≈ ln(OR) * √3 / π
        if or_val > 0:
            d_approx = abs(math.log(or_val)) * math.sqrt(3) / math.pi
        else:
            d_approx = 0
        findings.append({
            "type": "odds_ratio",
            "test_label": "Odds Ratio",
            "power_test": "t_two",
            "statistic": round(or_val, 4),
            "ci_low": round(ci_lo, 4) if ci_lo else None,
            "ci_high": round(ci_hi, 4) if ci_hi else None,
            "effect_size": round(d_approx, 4),
            "effect_label": "Cohen's d (from OR)",
            "source": m.group(0).strip(),
        })

    # Hazard Ratios
    for m in _HR.finditer(text):
        hr_val = float(m.group(1))
        ci_lo = float(m.group(2)) if m.group(2) else None
        ci_hi = float(m.group(3)) if m.group(3) else None
        findings.append({
            "type": "hazard_ratio",
            "test_label": "Hazard Ratio",
            "power_test": "t_two",
            "statistic": round(hr_val, 4),
            "ci_low": round(ci_lo, 4) if ci_lo else None,
            "ci_high": round(ci_hi, 4) if ci_hi else None,
            "effect_size": round(abs(math.log(hr_val)) * math.sqrt(3) / math.pi, 4) if hr_val > 0 else 0,
            "effect_label": "Cohen's d (from HR)",
            "source": m.group(0).strip(),
        })

    # Chi-square
    for m in _CHI2.finditer(text):
        df = int(m.group(1)) if m.group(1) else 1
        chi2_val = float(m.group(2))
        p_val = _parse_p(m.group(3))
        n = default_n or 100
        # Cohen's w = sqrt(chi2/n)
        w = math.sqrt(chi2_val / n) if n > 0 else 0
        findings.append({
            "type": "chi_square",
            "test_label": "Chi-square",
            "power_test": "chi2",
            "statistic": round(chi2_val, 4),
            "df": df,
            "p": round(p_val, 6),
            "effect_size": round(w, 4),
            "effect_label": "Cohen's w",
            "n_approx": n,
            "k_groups": df + 1,
            "source": m.group(0).strip(),
        })

    # Proportions (e.g. 45% vs 32%)
    for m in _PROPORTIONS.finditer(text):
        p1 = float(m.group(1)) / 100
        p2 = float(m.group(2)) / 100
        h = abs(_cohens_h(p1, p2))
        findings.append({
            "type": "proportions",
            "test_label": "Two proportions",
            "power_test": "proportion",
            "p1": round(p1, 4),
            "p2": round(p2, 4),
            "effect_size": round(h, 4),
            "effect_label": "Cohen's h",
            "source": m.group(0).strip(),
        })

    # AUC values
    for m in _AUC.finditer(text):
        auc_val = float(m.group(1))
        if 0.5 <= auc_val <= 1.0:
            findings.append({
                "type": "auc",
                "test_label": "AUC/C-statistic",
                "statistic": round(auc_val, 4),
                "source": m.group(0).strip(),
            })

    # Beta coefficients: β = 0.45, p = 0.012
    for m in _BETA.finditer(text):
        try:
            beta_val = _clean_num(m.group(1))
            p_val = _parse_p(m.group(2))
            findings.append({
                "type": "beta",
                "test_label": "β coefficient",
                "power_test": "t_two",
                "statistic": round(beta_val, 4),
                "p": round(p_val, 6),
                "effect_size": round(abs(beta_val), 4),
                "effect_label": "Standardized β",
                "source": m.group(0).strip(),
            })
        except (ValueError, TypeError):
            pass

    # P-values standalone (collect for context, no direct power mapping)
    p_values_found = []
    for m in _P_STANDALONE.finditer(text):
        try:
            pv = _parse_p(m.group(1))
            if 0 < pv < 1:
                p_values_found.append(round(pv, 6))
        except (ValueError, TypeError):
            pass

    # Mean ± SD pairs (useful context)
    means = []
    for m in _MEAN_SD.finditer(text):
        try:
            mean_val = _clean_num(m.group(1))
            sd_val = _clean_num(m.group(2))
            if 0 < sd_val < mean_val * 10:  # sanity check
                means.append({"mean": round(mean_val, 3), "sd": round(sd_val, 3), "source": m.group(0).strip()})
        except (ValueError, TypeError):
            pass

    # Compute Cohen's d from consecutive mean±SD pairs
    if len(means) >= 2:
        for i in range(0, len(means) - 1, 2):
            m1, s1 = means[i]["mean"], means[i]["sd"]
            m2, s2 = means[i+1]["mean"], means[i+1]["sd"]
            pooled_sd = math.sqrt((s1**2 + s2**2) / 2) if s1 > 0 and s2 > 0 else 1
            d = abs(m1 - m2) / pooled_sd
            if 0 < d < 10:
                findings.append({
                    "type": "mean_diff",
                    "test_label": "Mean difference",
                    "power_test": "t_two",
                    "mean1": m1, "sd1": s1,
                    "mean2": m2, "sd2": s2,
                    "effect_size": round(d, 4),
                    "effect_label": "Cohen's d",
                    "source": f"{means[i]['source']} vs {means[i+1]['source']}",
                })

    return findings


def _extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF using pdfplumber (preferred) or PyPDF2 (fallback)."""
    import sys
    errors = []

    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
        return "\n".join(text_parts)
    except ImportError as e:
        errors.append(f"pdfplumber: {e}")
    except Exception as e:
        errors.append(f"pdfplumber error: {e}")

    # Fallback: PyPDF2
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except ImportError as e:
        errors.append(f"PyPDF2: {e}")
    except Exception as e:
        errors.append(f"PyPDF2 error: {e}")

    # Fallback: pdfminer (already installed as pdfplumber dependency)
    try:
        from pdfminer.high_level import extract_text as _pdf_extract
        return _pdf_extract(io.BytesIO(file_bytes))
    except ImportError as e:
        errors.append(f"pdfminer: {e}")
    except Exception as e:
        errors.append(f"pdfminer error: {e}")

    raise HTTPException(
        status_code=500,
        detail=f"No PDF library available. Python: {sys.executable} | Errors: {'; '.join(errors)} | Fix: pip install pdfplumber"
    )


def _extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from DOCX using python-docx."""
    try:
        import docx
    except ImportError:
        raise HTTPException(status_code=500, detail="DOCX parsing requires python-docx. Install: pip install python-docx")
    doc = docx.Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


@router.post("/parse")
async def parse_article(file: UploadFile = File(...)):
    """Parse a PDF or DOCX file and extract statistical parameters for power analysis."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    ext = file.filename.lower().rsplit(".", 1)[-1] if "." in file.filename else ""
    if ext not in ("pdf", "docx", "doc", "txt"):
        raise HTTPException(status_code=400, detail=f"Unsupported file type: .{ext}. Use PDF, DOCX, or TXT.")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        if ext == "pdf":
            text = _extract_text_from_pdf(content)
        elif ext in ("docx", "doc"):
            text = _extract_text_from_docx(content)
        else:  # txt
            text = content.decode("utf-8", errors="replace")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to extract text: {exc}")

    if not text.strip():
        raise HTTPException(status_code=422, detail="No text could be extracted from the file")

    findings = _extract_stats(text)

    # Text preview for debugging (first 500 chars of extracted text)
    preview = text[:500].replace("\n", " ").strip()

    return {
        "filename": file.filename,
        "n_chars": len(text),
        "n_findings": len(findings),
        "findings": findings,
        "text_preview": preview,
    }
