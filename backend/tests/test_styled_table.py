"""Generic styled-table DOCX export (/api/pub_export/styled_table)."""
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def test_styled_table_returns_docx():
    r = client.post("/api/pub_export/styled_table", json={
        "title": "Table 3. Cox HR",
        "caption": "HR = hazard ratio.",
        "columns": ["Variable", "Univariable HR (95% CI), p"],
        "rows": [["Age (per 1 year)", "1.08 (1.06-1.11), p<0.001"], ["Sex", "0.50 (0.27-0.95), p=0.033"]],
        "filename": "cox_hr_table",
    })
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == DOCX_MIME
    assert "cox_hr_table.docx" in r.headers.get("content-disposition", "")
    # .docx is a zip → starts with PK
    assert r.content[:2] == b"PK"
    assert len(r.content) > 200


def test_styled_table_rejects_empty_columns():
    r = client.post("/api/pub_export/styled_table", json={"columns": [], "rows": []})
    assert r.status_code == 422


def test_styled_table_handles_ragged_rows():
    r = client.post("/api/pub_export/styled_table", json={
        "columns": ["A", "B", "C"],
        "rows": [["1"], ["1", "2", "3"]],  # short row should pad, not crash
    })
    assert r.status_code == 200, r.text
    assert r.content[:2] == b"PK"
