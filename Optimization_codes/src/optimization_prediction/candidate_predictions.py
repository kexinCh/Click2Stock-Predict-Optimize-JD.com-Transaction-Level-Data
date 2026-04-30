from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class CandidatePredictionArtifacts:
    aligned_predictions: pd.DataFrame
    diagnostics: dict


def _read_prediction_file(path: Path) -> tuple[pd.DataFrame, dict]:
    read_attempts: list[dict] = []
    strategies = [
        ("csv", {"encoding": "utf-8-sig"}),
        ("csv", {"encoding": "utf-8"}),
        ("excel", {}),
        ("csv", {"encoding": "latin1"}),
    ]
    last_error: Exception | None = None
    for reader, kwargs in strategies:
        try:
            if reader == "excel":
                frame = pd.read_excel(path)
            else:
                frame = pd.read_csv(path, **kwargs)
            return frame, {"reader": reader, **kwargs}
        except Exception as exc:  # pragma: no cover - diagnostics path
            last_error = exc
            read_attempts.append({"reader": reader, "kwargs": kwargs, "error": type(exc).__name__})
    raise RuntimeError(f"Unable to read prediction file {path.name}: {last_error}; attempts={read_attempts}")


def _normalize_date_column(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all() and numeric.min() > 40000:
        return pd.to_datetime("1899-12-30") + pd.to_timedelta(numeric, unit="D")
    return pd.to_datetime(series, errors="coerce")


def load_candidate_prediction_file(
    path: Path,
    actual_test_panel: pd.DataFrame,
    benchmark_role: str | None = None,
) -> CandidatePredictionArtifacts:
    raw, read_metadata = _read_prediction_file(path)
    required = {"date", "cluster_id", "warehouse_id", "actual_demand", "predicted_demand"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")

    frame = raw.rename(columns={"cluster_id": "sku_cluster_ID", "warehouse_id": "warehouse"}).copy()
    frame["date"] = _normalize_date_column(frame["date"])
    frame["sku_cluster_ID"] = pd.to_numeric(frame["sku_cluster_ID"], errors="coerce").astype("Int64")
    frame["warehouse"] = pd.to_numeric(frame["warehouse"], errors="coerce").astype("Int64")
    frame["actual_demand"] = pd.to_numeric(frame["actual_demand"], errors="coerce")
    frame["predicted_demand"] = pd.to_numeric(frame["predicted_demand"], errors="coerce").fillna(0.0).clip(lower=0.0)

    frame = frame.dropna(subset=["date", "sku_cluster_ID", "warehouse", "actual_demand"]).copy()
    frame["sku_cluster_ID"] = frame["sku_cluster_ID"].astype(int)
    frame["warehouse"] = frame["warehouse"].astype(int)
    frame["actual_demand"] = frame["actual_demand"].astype(float)

    if "model_name" not in frame.columns:
        frame["model_name"] = path.stem
    if "feature_set" not in frame.columns:
        frame["feature_set"] = "unknown"

    source_scope = frame[
        [
            "date",
            "warehouse",
            "sku_cluster_ID",
            "actual_demand",
            "predicted_demand",
            "model_name",
            "feature_set",
        ]
    ].drop_duplicates(["date", "warehouse", "sku_cluster_ID"])

    aligned = actual_test_panel.merge(
        source_scope,
        on=["date", "warehouse", "sku_cluster_ID"],
        how="left",
    )
    actual_mismatch = aligned["demand"].sub(aligned["actual_demand"]).abs().fillna(0.0).sum()
    if actual_mismatch > 1e-6:
        raise ValueError(
            f"Actual demand mismatch between test panel and {path.name}. Absolute mismatch sum={actual_mismatch}"
        )

    model_name = str(source_scope["model_name"].dropna().mode().iloc[0]) if source_scope["model_name"].notna().any() else path.stem
    feature_set = (
        str(source_scope["feature_set"].dropna().mode().iloc[0]) if source_scope["feature_set"].notna().any() else "unknown"
    )
    role = benchmark_role or path.stem
    model_token = model_name.lower().replace(" ", "_")
    feature_token = feature_set.lower().replace(" ", "_")
    role_token = role.lower().replace(" ", "_").replace("-", "_")

    aligned["predicted_demand"] = aligned["predicted_demand"].fillna(0.0)
    aligned["model_id"] = f"{role_token}__{model_token}__feature_{feature_token}"
    aligned["approach"] = "candidate_file"
    aligned["prediction_level"] = "direct warehouse-cluster"
    aligned["model_name"] = model_name
    aligned["family"] = "candidate_file"
    aligned["feature_set"] = feature_set
    aligned["source_file"] = path.name

    diagnostics = {
        "source_file": path.name,
        "read_method": read_metadata,
        "source_rows": int(len(raw)),
        "aligned_rows": int(len(aligned)),
        "rows_missing_from_file_filled_with_zero": int(aligned["actual_demand"].isna().sum()),
        "actual_demand_mismatch_sum": float(actual_mismatch),
        "model_name": model_name,
        "feature_set": feature_set,
        "min_date": str(aligned["date"].min().date()) if not aligned.empty else None,
        "max_date": str(aligned["date"].max().date()) if not aligned.empty else None,
    }
    return CandidatePredictionArtifacts(
        aligned_predictions=aligned[
            [
                "date",
                "warehouse",
                "sku_cluster_ID",
                "demand",
                "predicted_demand",
                "model_id",
                "approach",
                "prediction_level",
                "model_name",
                "family",
                "feature_set",
                "source_file",
            ]
        ].rename(columns={"demand": "actual_demand"}),
        diagnostics=diagnostics,
    )
