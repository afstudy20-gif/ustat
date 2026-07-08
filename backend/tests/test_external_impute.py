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


def test_external_impute_transfer_writes_previewed_values(client):
    sid = _seed("external_transfer")
    response = client.post("/api/missing_data/external_impute_transfer", json={
        "session_id": sid,
        "target": "LDL",
        "preview_rows": [
            {"row_index": 1, "imputed_value": 133.0},
            {"row_index": 2, "imputed_value": 151.0},
        ],
    })
    assert response.status_code == 200, response.text
    assert response.json()["n_imputed"] == 2

    df = store.get(sid)
    assert float(df.loc[1, "ldl"]) == 133.0
    assert float(df.loc[2, "ldl"]) == 151.0
    assert float(df.loc[0, "ldl"]) == 110.0

    audit = store.get_audit(sid)
    assert audit[-1]["action"] == "external_reference_impute_transfer"


def test_external_impute_transfer_does_not_overwrite_observed_values(client):
    sid = _seed("external_transfer_observed")
    response = client.post("/api/missing_data/external_impute_transfer", json={
        "session_id": sid,
        "target": "ldl",
        "preview_rows": [{"row_index": 0, "imputed_value": 999.0}],
    })
    assert response.status_code == 400
    assert "No currently missing" in response.text
    assert float(store.get(sid).loc[0, "ldl"]) == 110.0


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


def _stratified_seed(sid: str) -> str:
    df = pd.DataFrame({
        "age": [50, 60, 70, 80],
        "dm": [0, 0, 1, 1],
        "glucose": [90.0, np.nan, 300.0, np.nan],
    })
    store.save(sid, df)
    return sid


def _stratified_reference_file() -> tuple[str, io.BytesIO, str]:
    ref = pd.DataFrame({
        "age": [52, 58, 72, 78],
        "dm": [0, 0, 1, 1],
        "glucose": [92.0, 88.0, 295.0, 305.0],
    })
    return "reference.csv", io.BytesIO(ref.to_csv(index=False).encode("utf-8")), "text/csv"


def test_external_impute_stratify_keeps_donors_within_stratum(client):
    sid = _stratified_seed("external_stratify")
    response = client.post(
        "/api/missing_data/external_impute_preview",
        data={
            "session_id": sid,
            "target": "glucose",
            "predictors": '["age"]',
            "stratify_by": "dm",
            "method": "pmm",
            "mechanism": "MAR",
            "max_iter": "5",
            "random_state": "11",
        },
        files={"file": _stratified_reference_file()},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["n_imputed"] == 2
    assert "stratified by 'dm'" in body["result_text"]

    by_row = {row["row_index"]: row for row in body["preview_rows"]}
    # Row 1 is dm=0, so its imputed glucose should stay near the non-diabetic reference pool (~90).
    assert by_row[1]["imputed_value"] < 150
    # Row 3 is dm=1, so its imputed glucose should stay near the diabetic reference pool (~300).
    assert by_row[3]["imputed_value"] > 200


def test_external_impute_stratify_skips_stratum_missing_in_reference(client):
    sid = _stratified_seed("external_stratify_missing_ref")
    ref = pd.DataFrame({
        "age": [52, 58],
        "dm": [0, 0],
        "glucose": [92.0, 88.0],
    })
    ref_file = ("reference.csv", io.BytesIO(ref.to_csv(index=False).encode("utf-8")), "text/csv")
    response = client.post(
        "/api/missing_data/external_impute_preview",
        data={
            "session_id": sid,
            "target": "glucose",
            "predictors": '["age"]',
            "stratify_by": "dm",
            "method": "pmm",
            "mechanism": "MAR",
            "max_iter": "5",
            "random_state": "11",
        },
        files={"file": ref_file},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    # Only the dm=0 stratum (row 1) can be imputed; dm=1 stratum (row 3) is skipped.
    assert body["n_imputed"] == 1
    by_row = {row["row_index"]: row for row in body["preview_rows"]}
    assert 1 in by_row
    assert 3 not in by_row
    assert any("Skipped stratum" in w for w in body["warnings"])


def test_external_impute_stratify_fails_when_all_strata_missing_in_reference(client):
    sid = _stratified_seed("external_stratify_all_missing_ref")
    ref = pd.DataFrame({
        "age": [52, 58],
        "dm": [2, 2],
        "glucose": [92.0, 88.0],
    })
    ref_file = ("reference.csv", io.BytesIO(ref.to_csv(index=False).encode("utf-8")), "text/csv")
    response = client.post(
        "/api/missing_data/external_impute_preview",
        data={
            "session_id": sid,
            "target": "glucose",
            "predictors": '["age"]',
            "stratify_by": "dm",
            "method": "pmm",
            "mechanism": "MAR",
            "max_iter": "5",
            "random_state": "11",
        },
        files={"file": ref_file},
    )
    assert response.status_code == 422
    assert "No strata" in response.text


def test_external_impute_stratify_allows_stratify_column_as_predictor(client):
    sid = _stratified_seed("external_stratify_predictor")
    response = client.post(
        "/api/missing_data/external_impute_preview",
        data={
            "session_id": sid,
            "target": "glucose",
            "predictors": '["age","dm"]',
            "stratify_by": "dm",
            "method": "pmm",
            "mechanism": "MAR",
            "max_iter": "5",
            "random_state": "11",
        },
        files={"file": _stratified_reference_file()},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["n_imputed"] == 2
    # dm is dropped from the per-stratum predictor list because it is constant within each stratum.
    assert "dm" not in body["predictors"]

    by_row = {row["row_index"]: row for row in body["preview_rows"]}
    assert by_row[1]["imputed_value"] < 150
    assert by_row[3]["imputed_value"] > 200
