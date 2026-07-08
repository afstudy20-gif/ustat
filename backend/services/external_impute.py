from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from fastapi import HTTPException

from services.missing_data import mice_multiple


SUPPORTED_METHODS = {"pmm", "mice"}
SUPPORTED_MECHANISMS = {"unknown", "MCAR", "MAR", "MNAR"}


@dataclass
class ExternalImputeResult:
    target: str
    predictors: list[str]
    method: str
    mechanism: str
    missing_rows: list[int]
    filled_values: dict[int, Any]
    result: dict[str, Any]


def _missing_mask(series: pd.Series) -> pd.Series:
    return series.isna() | (series.astype(str).str.strip() == "")


def _clean_scalar(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        return int(value) if np.isfinite(value) and value.is_integer() else value
    return value


def _parse_predictors(predictors: Iterable[str]) -> list[str]:
    out = []
    for item in predictors:
        name = str(item).strip()
        if name and name not in out:
            out.append(name)
    return out


def _norm_name(name: str) -> str:
    return str(name).strip().casefold()


def _resolve_column(columns: Iterable[str], requested: str, *, dataset_name: str) -> str:
    columns = [str(col) for col in columns]
    if requested in columns:
        return requested

    norm_requested = _norm_name(requested)
    matches = [col for col in columns if _norm_name(col) == norm_requested]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise HTTPException(
            status_code=400,
            detail=f"Ambiguous column '{requested}' in {dataset_name}: {matches}",
        )
    raise HTTPException(status_code=400, detail=f"Column missing in {dataset_name}: {requested}")


def _stratum_value(value: Any) -> str:
    """Stable string key for grouping categorical strata."""
    if value is None:
        return "__missing__"
    try:
        if pd.isna(value):
            return "__missing__"
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _external_reference_impute_single(
    current_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    *,
    target: str,
    predictors: list[str],
    reference_target: str,
    predictor_mappings: dict[str, str],
    method: str,
    mechanism: str,
    max_iter: int,
    random_state: int,
    stratum_label: Optional[str] = None,
) -> ExternalImputeResult:
    """Core external-reference imputation for a single (possibly stratified) subset."""
    current_target = _resolve_column(current_df.columns, target, dataset_name="current data")
    reference_target_col = _resolve_column(reference_df.columns, reference_target, dataset_name="reference data")
    current_predictors = [
        _resolve_column(current_df.columns, predictor_mappings.get(predictor, predictor), dataset_name="current data")
        for predictor in predictors
    ]
    if current_target in current_predictors:
        raise HTTPException(status_code=400, detail="Target column cannot also be a mapped predictor.")
    reference_predictors = [
        _resolve_column(reference_df.columns, predictor, dataset_name="reference data")
        for predictor in predictors
    ]

    needed = [current_target] + current_predictors
    reference_needed = [reference_target_col] + reference_predictors

    target_missing = _missing_mask(current_df[current_target])
    missing_rows = [int(i) for i in current_df.index[target_missing].tolist()]
    if not missing_rows:
        raise HTTPException(status_code=400, detail=f"Column '{current_target}' has no missing values to impute.")

    current_part = current_df[needed].copy()
    reference_part = reference_df[reference_needed].copy()
    reference_part.columns = needed
    current_part["__ustat_source"] = "current"
    reference_part["__ustat_source"] = "reference"
    current_part["__ustat_row_index"] = list(current_df.index)
    reference_part["__ustat_row_index"] = -1
    combined = pd.concat([current_part, reference_part], ignore_index=True)

    observed_target = combined.loc[~_missing_mask(combined[current_target]), current_target]
    if observed_target.empty:
        raise HTTPException(status_code=422, detail=f"No observed '{current_target}' values found in current or reference data.")

    imputation_cols = needed
    imputed = mice_multiple(
        combined[imputation_cols],
        imputation_cols,
        n_imputations=1,
        max_iter=max(1, int(max_iter)),
        random_state=int(random_state),
    ).imputed_datasets[0]

    imputed[current_target] = imputed[current_target].where(
        ~_missing_mask(imputed[current_target]), combined[current_target]
    )
    current_positions = combined.index[(combined["__ustat_source"] == "current") & combined["__ustat_row_index"].isin(missing_rows)]
    filled_values = {
        int(combined.loc[pos, "__ustat_row_index"]): _clean_scalar(imputed.loc[pos, current_target])
        for pos in current_positions
    }
    filled_values = {k: v for k, v in filled_values.items() if v is not None}
    if not filled_values:
        raise HTTPException(status_code=422, detail=f"Could not impute any missing '{current_target}' values.")

    preview_rows = []
    for row_index, value in filled_values.items():
        pred_missing = int(_missing_mask(current_df.loc[row_index, current_predictors]).sum())
        row = {
            "row_index": row_index,
            "imputed_value": value,
            "predictors_missing": pred_missing,
        }
        if stratum_label is not None:
            row["stratum"] = stratum_label
        preview_rows.append(row)

    ref_complete = int(reference_part[needed].dropna().shape[0])
    current_observed = int((~_missing_mask(current_df[current_target])).sum())
    warnings: list[str] = []
    if mechanism == "MNAR":
        warnings.append("MNAR selected: PMM/MICE remains a MAR reference imputation; use sensitivity analysis for MNAR assumptions.")
    if ref_complete < max(5, len(predictors) + 2):
        warnings.append("Reference dataset has few complete donor rows; inspect imputed values carefully.")

    result = {
        "target": current_target,
        "reference_target": reference_target_col,
        "predictors": current_predictors,
        "reference_predictors": reference_predictors,
        "predictor_mappings": dict(zip(reference_predictors, current_predictors)),
        "method": "PMM" if method == "pmm" else "MICE/PMM",
        "mechanism": mechanism,
        "n_missing_target": len(missing_rows),
        "n_imputed": len(filled_values),
        "current_observed_target": current_observed,
        "reference_rows": int(len(reference_df)),
        "reference_complete_rows": ref_complete,
        "preview_rows": preview_rows[:200],
        "warnings": warnings,
        "result_text": (
            f"{len(filled_values)} missing value(s) in '{current_target}' were imputed using "
            f"{len(current_predictors)} predictor(s) and a reference dataset with {len(reference_df)} row(s)."
        ),
        "methods_text": (
            f"External reference-assisted imputation used current data plus an uploaded reference dataset. "
            f"The target variable was {current_target}; predictors were {', '.join(current_predictors)}. "
            f"Missing target values were filled by chained equations with predictive mean matching "
            f"({max(1, int(max_iter))} iterations; random seed {int(random_state)}), under a {mechanism} mechanism label."
        ),
        "export_rows": [
            ["Row", "Imputed value", "Predictors missing"],
            *[[r["row_index"], r["imputed_value"], r["predictors_missing"]] for r in preview_rows],
        ],
    }
    if stratum_label is not None:
        result["stratum"] = stratum_label

    return ExternalImputeResult(
        target=current_target,
        predictors=current_predictors,
        method=method,
        mechanism=mechanism,
        missing_rows=missing_rows,
        filled_values=filled_values,
        result=result,
    )


def external_reference_impute(
    current_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    *,
    target: str,
    predictors: Iterable[str],
    reference_target: str | None = None,
    predictor_mappings: dict[str, str] | None = None,
    method: str = "pmm",
    mechanism: str = "unknown",
    max_iter: int = 20,
    random_state: int = 42,
    stratify_by: Optional[str] = None,
) -> ExternalImputeResult:
    target = str(target).strip()
    predictors = _parse_predictors(predictors)
    method = (method or "pmm").strip().lower()
    mechanism = (mechanism or "unknown").strip()
    stratify_by = str(stratify_by).strip() if stratify_by else None

    if method not in SUPPORTED_METHODS:
        raise HTTPException(status_code=400, detail=f"Unknown external imputation method '{method}'.")
    if mechanism not in SUPPORTED_MECHANISMS:
        raise HTTPException(status_code=400, detail=f"Unknown missingness mechanism '{mechanism}'.")
    if not target:
        raise HTTPException(status_code=400, detail="Select a target column.")
    if not predictors:
        raise HTTPException(status_code=400, detail="Select at least one predictor column.")
    if _norm_name(target) in {_norm_name(p) for p in predictors}:
        raise HTTPException(status_code=400, detail="Target column cannot also be a predictor.")

    predictor_mappings = predictor_mappings or {}
    reference_target_name = str(reference_target or target).strip()

    if not stratify_by:
        return _external_reference_impute_single(
            current_df,
            reference_df,
            target=target,
            predictors=predictors,
            reference_target=reference_target_name,
            predictor_mappings=predictor_mappings,
            method=method,
            mechanism=mechanism,
            max_iter=max_iter,
            random_state=random_state,
        )

    # Stratified imputation: run separately per stratum and combine results.
    current_stratify_col = _resolve_column(current_df.columns, stratify_by, dataset_name="current data")
    reference_stratify_col = _resolve_column(reference_df.columns, stratify_by, dataset_name="reference data")

    if current_stratify_col == _resolve_column(current_df.columns, target, dataset_name="current data"):
        raise HTTPException(status_code=400, detail="Stratify column cannot be the target column.")
    if current_stratify_col in [
        _resolve_column(current_df.columns, predictor_mappings.get(p, p), dataset_name="current data")
        for p in predictors
    ]:
        raise HTTPException(status_code=400, detail="Stratify column cannot also be a predictor.")

    target_missing = _missing_mask(current_df[target])
    missing_rows = current_df.index[target_missing]
    strata = sorted({
        _stratum_value(v)
        for v in current_df.loc[missing_rows, current_stratify_col]
    })
    if not strata:
        raise HTTPException(status_code=400, detail=f"No strata found in stratify column '{current_stratify_col}' for missing rows.")

    combined_result: Optional[ExternalImputeResult] = None
    for stratum in strata:
        current_mask = current_df[current_stratify_col].apply(_stratum_value) == stratum
        reference_mask = reference_df[reference_stratify_col].apply(_stratum_value) == stratum
        current_subset = current_df.loc[current_mask].copy()
        reference_subset = reference_df.loc[reference_mask].copy()

        if reference_subset.empty:
            raise HTTPException(
                status_code=422,
                detail=f"Stratum '{stratum}' has no matching reference rows; cannot impute."
            )

        stratum_result = _external_reference_impute_single(
            current_subset,
            reference_subset,
            target=target,
            predictors=predictors,
            reference_target=reference_target_name,
            predictor_mappings=predictor_mappings,
            method=method,
            mechanism=mechanism,
            max_iter=max_iter,
            random_state=random_state,
            stratum_label=stratum,
        )

        if combined_result is None:
            combined_result = stratum_result
        else:
            combined_result.missing_rows.extend(stratum_result.missing_rows)
            combined_result.filled_values.update(stratum_result.filled_values)
            combined_result.result["preview_rows"].extend(stratum_result.result["preview_rows"])

    assert combined_result is not None
    combined_result.result["preview_rows"] = combined_result.result["preview_rows"][:200]
    combined_result.result["n_missing_target"] = len(combined_result.missing_rows)
    combined_result.result["n_imputed"] = len(combined_result.filled_values)
    combined_result.result["reference_rows"] = int(len(reference_df))
    combined_result.result["reference_complete_rows"] = None  # not meaningful across strata
    combined_result.result["result_text"] = (
        f"{len(combined_result.filled_values)} missing value(s) in '{combined_result.target}' were imputed "
        f"using {len(combined_result.predictors)} predictor(s), stratified by '{current_stratify_col}'."
    )
    combined_result.result["methods_text"] = (
        f"External reference-assisted imputation used current data plus an uploaded reference dataset, "
        f"stratified by '{current_stratify_col}'. The target variable was {combined_result.target}; "
        f"predictors were {', '.join(combined_result.predictors)}. Missing target values were filled "
        f"by chained equations with predictive mean matching within each stratum "
        f"({max(1, int(max_iter))} iterations; random seed {int(random_state)}), under a {mechanism} mechanism label."
    )
    combined_result.result["export_rows"] = [
        ["Row", "Imputed value", "Predictors missing", "Stratum"],
        *[[
            r["row_index"], r["imputed_value"], r.get("predictors_missing"), r.get("stratum", "")
        ] for r in combined_result.result["preview_rows"]],
    ]
    combined_result.result["warnings"].insert(
        0,
        f"Imputation was stratified by '{current_stratify_col}'. Donors were drawn only from the same stratum."
    )
    return combined_result
