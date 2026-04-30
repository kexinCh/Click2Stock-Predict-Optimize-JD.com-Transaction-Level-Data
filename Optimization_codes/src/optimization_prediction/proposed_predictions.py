from __future__ import annotations

from pathlib import Path

import pandas as pd


def harmonize_proposed_prediction_csv(
    csv_path: Path,
    actual_test_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    proposed = pd.read_csv(csv_path)
    required = {
        "date",
        "cluster_id",
        "warehouse_id",
        "actual_demand",
        "predicted_demand",
    }
    missing = required.difference(proposed.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path.name}: {sorted(missing)}")

    proposed = proposed.rename(
        columns={
            "cluster_id": "sku_cluster_ID",
            "warehouse_id": "warehouse",
        }
    ).copy()
    proposed["date"] = pd.to_datetime(proposed["date"])
    proposed["sku_cluster_ID"] = proposed["sku_cluster_ID"].astype(int)
    proposed["warehouse"] = proposed["warehouse"].astype(int)
    proposed["actual_demand"] = proposed["actual_demand"].astype(float)
    proposed["predicted_demand"] = proposed["predicted_demand"].astype(float).clip(lower=0.0)

    if "model_name" not in proposed.columns:
        proposed["model_name"] = "proposed_model"
    if "feature_set" not in proposed.columns:
        proposed["feature_set"] = "unknown"

    csv_scope = proposed[
        ["date", "warehouse", "sku_cluster_ID", "actual_demand", "predicted_demand", "model_name", "feature_set"]
    ].drop_duplicates(["date", "warehouse", "sku_cluster_ID"])

    aligned = actual_test_panel.merge(
        csv_scope,
        on=["date", "warehouse", "sku_cluster_ID"],
        how="left",
    )
    actual_mismatch = aligned["demand"].sub(aligned["actual_demand"]).abs().fillna(0.0).sum()
    if actual_mismatch > 1e-6:
        raise ValueError(
            f"Actual demand mismatch between test panel and {csv_path.name}. Absolute mismatch sum={actual_mismatch}"
        )

    model_name = str(csv_scope["model_name"].dropna().mode().iloc[0]) if csv_scope["model_name"].notna().any() else "proposed_model"
    feature_set = (
        str(csv_scope["feature_set"].dropna().mode().iloc[0]) if csv_scope["feature_set"].notna().any() else "unknown"
    )
    model_token = model_name.lower().replace(" ", "_")
    feature_token = feature_set.lower().replace(" ", "_")

    aligned["predicted_demand"] = aligned["predicted_demand"].fillna(0.0)
    aligned["model_id"] = f"proposed_model__{model_token}__feature_{feature_token}"
    aligned["approach"] = "proposed_model"
    aligned["prediction_level"] = "direct warehouse-cluster"
    aligned["model_name"] = model_name
    aligned["family"] = "proposed_model"
    aligned["feature_set"] = feature_set

    diagnostics = {
        "source_file": str(csv_path),
        "source_rows": int(len(proposed)),
        "aligned_rows": int(len(aligned)),
        "rows_missing_from_csv_filled_with_zero": int(aligned["actual_demand"].isna().sum()),
        "actual_demand_mismatch_sum": float(actual_mismatch),
        "model_name": model_name,
        "feature_set": feature_set,
    }
    return (
        aligned[
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
            ]
        ].rename(columns={"demand": "actual_demand"}),
        diagnostics,
    )
