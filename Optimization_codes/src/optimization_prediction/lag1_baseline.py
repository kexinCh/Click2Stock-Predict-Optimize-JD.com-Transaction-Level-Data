from __future__ import annotations

import pandas as pd


def build_lag1_baseline(
    train_panel: pd.DataFrame,
    test_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    cluster_universe = sorted(test_panel["sku_cluster_ID"].astype(int).unique().tolist())
    warehouse_universe = sorted(test_panel["warehouse"].astype(int).unique().tolist())
    all_dates = pd.date_range(train_panel["date"].min(), test_panel["date"].max(), freq="D")
    full_index = pd.MultiIndex.from_product(
        [all_dates, warehouse_universe, cluster_universe],
        names=["date", "warehouse", "sku_cluster_ID"],
    )

    actual_full = (
        pd.concat(
            [
                train_panel[["date", "warehouse", "sku_cluster_ID", "demand"]],
                test_panel[["date", "warehouse", "sku_cluster_ID", "demand"]],
            ],
            ignore_index=True,
        )
        .groupby(["date", "warehouse", "sku_cluster_ID"], as_index=False)["demand"]
        .sum()
        .set_index(["date", "warehouse", "sku_cluster_ID"])
        .reindex(full_index, fill_value=0.0)
        .reset_index()
        .sort_values(["warehouse", "sku_cluster_ID", "date"])
    )

    actual_full["predicted_demand"] = (
        actual_full.groupby(["warehouse", "sku_cluster_ID"], sort=False)["demand"].shift(1).fillna(0.0)
    )
    baseline = actual_full[actual_full["date"].between(test_panel["date"].min(), test_panel["date"].max())].copy()
    baseline["model_id"] = "baseline__lag_1_actual_demand"
    baseline["approach"] = "lag_1_baseline"
    baseline["prediction_level"] = "direct warehouse-cluster"
    baseline["model_name"] = "lag_1_actual_demand"
    baseline["family"] = "naive_baseline"

    first_test_date = pd.to_datetime(test_panel["date"].min())
    first_day_rows = baseline[baseline["date"] == first_test_date].copy()
    prior_day = first_test_date - pd.Timedelta(days=1)
    prior_day_source = actual_full[actual_full["date"] == prior_day].copy()
    prior_day_source = prior_day_source.rename(columns={"demand": "prior_day_actual"})[
        ["warehouse", "sku_cluster_ID", "prior_day_actual"]
    ]
    first_day_rows = first_day_rows.merge(prior_day_source, on=["warehouse", "sku_cluster_ID"], how="left")

    diagnostics = {
        "baseline_definition": "Lag-1 naive forecast using prior day's actual demand for each warehouse-cluster pair",
        "first_test_day": str(first_test_date.date()),
        "first_test_day_lag_source": str(prior_day.date()),
        "aligned_rows": int(len(baseline)),
        "rows_with_zero_lag": int((baseline["predicted_demand"] == 0).sum()),
        "first_day_rows_with_missing_prior_actual": int(first_day_rows["prior_day_actual"].isna().sum()),
    }
    return (
        baseline[
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
            ]
        ].copy(),
        diagnostics,
    )
