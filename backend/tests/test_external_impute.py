import io

import numpy as np
import pandas as pd

from services import store


def _seed(sid: str) -> str:
    df = pd.DataFrame({
        "age": [50, 60, 70, 80],
        "bmi": [24.0, 28.0, 31.0, 35.0],
        "ldl": [110.0, np.nan, np.nan, 170.0],
    })
    store.save(sid, df)
    return sid


def _reference_file() -> tuple[str, io.BytesIO, str]:
    ref = pd.DataFrame({
        "age": [58, 62, 68, 75],
        "bmi": [27.0, 29.0, 32.0, 34.0],
        "ldl": [126.0, 134.0, 150.0, 164.0],
    })
    return "reference.csv", io.BytesIO(ref.to_csv(index=False).encode("utf-8")), "text/csv"


def _mapped_reference_file() -> tuple[str, io.BytesIO, str]:
    ref = pd.DataFrame({
        "AGE_YEARS": [58, 62, 68, 75],
        "BMI": [27.0, 29.0, 32.0, 34.0],
        "LDL_VALUE": [126.0, 134.0, 150.0, 164.0],
    })
    return "reference.csv", io.BytesIO(ref.to_csv(index=False).encode("utf-8")), "text/csv"


def _form(sid: str) -> dict:
    return {
        "session_id": sid,
        "target": "ldl",
        "predictors": '["age","bmi"]',
        "method": "pmm",
        "mechanism": "MAR",
        "max_iter": "5",
        "random_state": "11",
    }


def _mapped_form(sid: str) -> dict:
    return {
        "session_id": sid,
        "target": "ldl",
        "reference_target": "LDL_VALUE",
        "predictors": '["AGE_YEARS","BMI"]',
        "predictor_mappings": '{"AGE_YEARS":"age","BMI":"bmi"}',
        "method": "pmm",
        "mechanism": "MAR",
        "max_iter": "5",
        "random_state": "11",
    }


def test_external_impute_preview_uses_reference_dataset(client):
    sid = _seed("external_preview")
    response = client.post(
        "/api/missing_data/external_impute_preview",
        data=_form(sid),
        files={"file": _reference_file()},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["target"] == "ldl"
    assert body["predictors"] == ["age", "bmi"]
    assert body["n_missing_target"] == 2
    assert body["n_imputed"] == 2
    assert body["reference_rows"] == 4
    assert {row["row_index"] for row in body["preview_rows"]} == {1, 2}
    assert store.get(sid)["ldl"].isna().sum() == 2


def test_external_impute_preview_accepts_explicit_column_mapping(client):
    sid = _seed("external_preview_mapping")
    response = client.post(
        "/api/missing_data/external_impute_preview",
        data=_mapped_form(sid),
        files={"file": _mapped_reference_file()},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["target"] == "ldl"
    assert body["reference_target"] == "LDL_VALUE"
    assert body["predictors"] == ["age", "bmi"]
    assert body["reference_predictors"] == ["AGE_YEARS", "BMI"]
    assert body["predictor_mappings"] == {"AGE_YEARS": "age", "BMI": "bmi"}
    assert body["n_imputed"] == 2


def test_external_impute_preview_matches_columns_case_insensitively(client):
    sid = _seed("external_preview_case")
    response = client.post(
        "/api/missing_data/external_impute_preview",
        data={
            **_form(sid),
            "target": "LDL",
            "predictors": '["AGE","BMI"]',
        },
        files={"file": _reference_file()},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["target"] == "ldl"
    assert body["reference_target"] == "ldl"
    assert body["predictors"] == ["age", "bmi"]
    assert body["reference_predictors"] == ["age", "bmi"]


def test_external_reference_columns_reads_uploaded_dataset(client):
    response = client.post(
        "/api/missing_data/external_impute_reference_columns",
        files={"file": _reference_file()},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["n_rows"] == 4
    assert [col["name"] for col in body["columns"]] == ["age", "bmi", "ldl"]
    assert body["columns"][0]["kind"] == "numeric"


def test_external_impute_apply_writes_back_to_current_session(client):
    sid = _seed("external_apply")
    response = client.post(
        "/api/missing_data/external_impute_apply",
        data=_form(sid),
        files={"file": _reference_file()},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["applied"] is True
    assert body["n_imputed"] == 2

    df = store.get(sid)
    assert df["ldl"].isna().sum() == 0
    assert float(df.loc[0, "ldl"]) == 110.0
    assert float(df.loc[3, "ldl"]) == 170.0

    audit = store.get_audit(sid)
    assert audit[-1]["action"] == "external_reference_impute"
    assert audit[-1]["params"]["target"] == "ldl"


def test_external_impute_apply_respects_active_case_filter(client):
    sid = _seed("external_apply_filter")
    filter_response = client.post(f"/api/sessions/{sid}/select_cases", json={
        "conditions": [{"column": "age", "operator": "gt", "value": 65, "join": "AND"}],
    })
    assert filter_response.status_code == 200, filter_response.text

    response = client.post(
        "/api/missing_data/external_impute_apply",
        data=_form(sid),
        files={"file": _reference_file()},
    )
    assert response.status_code == 200, response.text
    assert response.json()["n_imputed"] == 1

    df = store.get(sid)
    assert pd.isna(df.loc[1, "ldl"])
    assert pd.notna(df.loc[2, "ldl"])
