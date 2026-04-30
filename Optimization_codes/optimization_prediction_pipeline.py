from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt

from report_pdf.build_professional_report import build_document
from src.optimization_prediction.config import (
    EXPECTED_END_DATE,
    EXPECTED_START_DATE,
    H_VALUE,
    K_VALUE,
    TEST_END_DATE,
    TEST_START_DATE,
    TRAIN_END_DATE,
    U_VALUE,
    build_paths,
)
from src.optimization_prediction.data_loading import load_raw_data
from src.optimization_prediction.evaluation import evaluate_model_predictions, save_figures
from src.optimization_prediction.modeling import extract_feature_importance, run_modeling_experiments
from src.optimization_prediction.optimization_solver import (
    run_optimization_comparison,
    save_optimization_figures,
)
from src.optimization_prediction.parameter_builder import (
    build_parameter_artifacts,
    prepare_real_cluster_demand_panels,
)
from src.optimization_prediction.reporting import render_markdown_report, write_report_files


warnings.filterwarnings("ignore", category=ConvergenceWarning)


LEGACY_PREVIOUS_METRICS = pd.DataFrame(
    [
        {
            "model_id": "legacy_pre_upgrade_pipeline",
            "approach": "legacy",
            "prediction_level": "previous optimization pipeline",
            "model_name": "previous_version",
            "family": "legacy",
            "mae": 6.218490682438978,
            "rmse": 22.33402735210863,
            "wape": 0.3364731911466794,
            "bias": -0.0795858374268849,
            "underprediction_units": 35524.78409572521,
            "overprediction_units": 21934.069810010933,
            "price_weighted_abs_error": 8058356.20219698,
            "purchase_cost_proxy": 352764.6235029767,
            "holding_cost_proxy": 1660.3738693992368,
        }
    ]
)


def _save_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _save_feature_importance_figure(feature_importance: pd.DataFrame, output_path: Path, title: str) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if feature_importance.empty:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5, "No feature importance available", ha="center", va="center")
        ax.axis("off")
    else:
        plot_frame = feature_importance.head(12).sort_values("importance", ascending=True)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(plot_frame["feature"], plot_frame["importance"], color="#003262")
        ax.set_title(title)
        ax.set_xlabel("Importance")
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return str(output_path)


def _build_optimization_summary_table(optimization_artifacts) -> pd.DataFrame:
    predicted_week = optimization_artifacts.predicted_policy.weekly_summary.iloc[0]
    oracle_week = optimization_artifacts.oracle_policy.weekly_summary.iloc[0]
    comparison = optimization_artifacts.weekly_comparison.iloc[0]
    predicted_daily = optimization_artifacts.predicted_policy.daily_summary
    return pd.DataFrame(
        [
            {
                "scenario": "Predicted policy planned",
                "total_cost": float(predicted_week["planned_total_cost"]),
                "shortage_units": float(predicted_daily["planned_shortage_units"].sum()),
                "ending_inventory_units": float(predicted_daily["planned_ending_inventory_units"].sum()),
                "overflow_units": 0.0,
                "transfer_units": float(predicted_daily["planned_transfer_units"].sum()),
                "service_level": float(
                    1.0 - predicted_daily["planned_shortage_units"].sum() / predicted_daily["planned_demand_units"].sum()
                )
                if predicted_daily["planned_demand_units"].sum()
                else 1.0,
            },
            {
                "scenario": "Predicted policy realized",
                "total_cost": float(predicted_week["realized_total_cost"]),
                "shortage_units": float(predicted_week["realized_shortage_units"]),
                "ending_inventory_units": float(predicted_week["realized_ending_inventory_units"]),
                "overflow_units": float(predicted_week.get("realized_overflow_units", 0.0)),
                "transfer_units": float(predicted_week["realized_transfer_units"]),
                "service_level": float(predicted_week["realized_service_level"]),
            },
            {
                "scenario": "Oracle policy",
                "total_cost": float(oracle_week["realized_total_cost"]),
                "shortage_units": float(oracle_week["realized_shortage_units"]),
                "ending_inventory_units": float(oracle_week["realized_ending_inventory_units"]),
                "overflow_units": float(oracle_week.get("realized_overflow_units", 0.0)),
                "transfer_units": float(oracle_week["realized_transfer_units"]),
                "service_level": float(oracle_week["realized_service_level"]),
            },
            {
                "scenario": "Realized gap vs oracle",
                "total_cost": float(comparison["realized_cost_gap_vs_oracle"]),
                "shortage_units": float(
                    predicted_week["realized_shortage_units"] - oracle_week["realized_shortage_units"]
                ),
                "ending_inventory_units": float(
                    predicted_week["realized_ending_inventory_units"] - oracle_week["realized_ending_inventory_units"]
                ),
                "overflow_units": float(
                    predicted_week.get("realized_overflow_units", 0.0) - oracle_week.get("realized_overflow_units", 0.0)
                ),
                "transfer_units": float(
                    predicted_week["realized_transfer_units"] - oracle_week["realized_transfer_units"]
                ),
                "service_level": float(comparison["service_level_gap_vs_oracle"]),
            },
        ]
    )


def _flatten_diagnostics(prefix: str, value) -> list[dict]:
    rows: list[dict] = []
    if isinstance(value, dict):
        for key, nested_value in value.items():
            nested_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_diagnostics(nested_prefix, nested_value))
    elif isinstance(value, list):
        rows.append({"metric": prefix, "value": ", ".join(map(str, value))})
    else:
        rows.append({"metric": prefix, "value": value})
    return rows


def _load_previous_metrics(path: Path) -> pd.DataFrame:
    if path.exists():
        frame = pd.read_csv(path)
        if "model_id" not in frame.columns:
            return frame
    return LEGACY_PREVIOUS_METRICS.copy()


def _load_json_snapshot(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _build_parameter_update_table(
    baseline_snapshot: dict,
    current_best_model_summary: pd.DataFrame,
    optimization_summary_table: pd.DataFrame,
    cluster_prices: pd.DataFrame,
    capacity_summary: pd.DataFrame,
) -> pd.DataFrame:
    if not baseline_snapshot:
        return pd.DataFrame()

    current_overall = current_best_model_summary.iloc[0].to_dict()
    current_gap = optimization_summary_table[optimization_summary_table["scenario"] == "Realized gap vs oracle"].iloc[0]
    baseline_gap = next(
        (
            row
            for row in baseline_snapshot.get("optimization_summary", [])
            if row.get("scenario") == "Realized gap vs oracle"
        ),
        {},
    )
    baseline_price = baseline_snapshot.get("cluster_price_summary", {})
    baseline_capacity = baseline_snapshot.get("capacity_summary", {})

    rows = [
        {
            "change_area": "selected_model",
            "previous_value": baseline_snapshot.get("forecast_metrics", {}).get("model_id", "unknown"),
            "updated_value": current_overall.get("model_id", "unknown"),
            "delta": "unchanged" if baseline_snapshot.get("forecast_metrics", {}).get("model_id") == current_overall.get("model_id") else "changed",
        },
        {
            "change_area": "forecast_purchase_cost_proxy",
            "previous_value": float(baseline_snapshot.get("forecast_metrics", {}).get("purchase_cost_proxy", 0.0)),
            "updated_value": float(current_overall.get("purchase_cost_proxy", 0.0)),
            "delta": float(current_overall.get("purchase_cost_proxy", 0.0) - baseline_snapshot.get("forecast_metrics", {}).get("purchase_cost_proxy", 0.0)),
        },
        {
            "change_area": "forecast_holding_cost_proxy",
            "previous_value": float(baseline_snapshot.get("forecast_metrics", {}).get("holding_cost_proxy", 0.0)),
            "updated_value": float(current_overall.get("holding_cost_proxy", 0.0)),
            "delta": float(current_overall.get("holding_cost_proxy", 0.0) - baseline_snapshot.get("forecast_metrics", {}).get("holding_cost_proxy", 0.0)),
        },
        {
            "change_area": "optimization_realized_gap_vs_oracle",
            "previous_value": float(baseline_gap.get("total_cost", 0.0)),
            "updated_value": float(current_gap.get("total_cost", 0.0)),
            "delta": float(current_gap.get("total_cost", 0.0) - baseline_gap.get("total_cost", 0.0)),
        },
        {
            "change_area": "cluster_price_mean",
            "previous_value": float(baseline_price.get("mean_p_s", 0.0)),
            "updated_value": float(cluster_prices["p_s"].mean()),
            "delta": float(cluster_prices["p_s"].mean() - baseline_price.get("mean_p_s", 0.0)),
        },
        {
            "change_area": "zero_price_clusters",
            "previous_value": float(baseline_price.get("zero_price_clusters", 0.0)),
            "updated_value": float((cluster_prices["p_s"] == 0).sum()),
            "delta": float((cluster_prices["p_s"] == 0).sum() - baseline_price.get("zero_price_clusters", 0.0)),
        },
        {
            "change_area": "warehouse_capacity_mean",
            "previous_value": float(baseline_capacity.get("mean_C_j", 0.0)),
            "updated_value": float(capacity_summary["C_j"].mean()),
            "delta": float(capacity_summary["C_j"].mean() - baseline_capacity.get("mean_C_j", 0.0)),
        },
    ]
    return pd.DataFrame(rows)


def _date_coverage_summary(raw) -> dict:
    return {
        "orders": {
            "min": str(raw.orders["order_date"].min().date()),
            "max": str(raw.orders["order_date"].max().date()),
        },
        "inventory": {
            "min": str(raw.inventory["date"].min().date()),
            "max": str(raw.inventory["date"].max().date()),
        },
        "clicks": {
            "min": str(raw.clicks["request_time"].dt.floor("D").min().date()),
            "max": str(raw.clicks["request_time"].dt.floor("D").max().date()),
        },
        "train_cluster_demand": {
            "min": str(raw.train_cluster_demand["order_date"].min().date()),
            "max": str(raw.train_cluster_demand["order_date"].max().date()),
        },
        "test_cluster_demand": {
            "min": str(raw.test_cluster_demand["order_date"].min().date()),
            "max": str(raw.test_cluster_demand["order_date"].max().date()),
        },
    }


def _verify_demand_file_consistency(raw_orders: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    raw_orders = raw_orders.copy()
    raw_orders["order_date"] = pd.to_datetime(raw_orders["order_date"])

    raw_train = (
        raw_orders[(raw_orders["order_date"] >= EXPECTED_START_DATE) & (raw_orders["order_date"] <= TRAIN_END_DATE)]
        .groupby(["order_date", "dc_des"], as_index=False)["quantity"]
        .sum()
        .rename(columns={"quantity": "raw_quantity"})
    )
    train_totals = (
        train_df.groupby(["order_date", "dc_des"], as_index=False)["demand"]
        .sum()
        .rename(columns={"demand": "cluster_file_quantity"})
    )
    train_check = raw_train.merge(train_totals, on=["order_date", "dc_des"], how="outer").fillna(0)
    train_check["abs_diff"] = (train_check["raw_quantity"] - train_check["cluster_file_quantity"]).abs()

    raw_test = (
        raw_orders[(raw_orders["order_date"] >= TEST_START_DATE) & (raw_orders["order_date"] <= TEST_END_DATE)]
        .groupby(["order_date", "dc_des"], as_index=False)["quantity"]
        .sum()
        .rename(columns={"quantity": "raw_quantity"})
    )
    test_totals = (
        test_df.groupby(["order_date", "dc_des"], as_index=False)["demand"]
        .sum()
        .rename(columns={"demand": "cluster_file_quantity"})
    )
    test_check = raw_test.merge(test_totals, on=["order_date", "dc_des"], how="outer").fillna(0)
    test_check["abs_diff"] = (test_check["raw_quantity"] - test_check["cluster_file_quantity"]).abs()

    return pd.DataFrame(
        [
            {
                "check": "train warehouse-date totals vs raw orders",
                "max_abs_diff": float(train_check["abs_diff"].max()),
                "rows_compared": int(train_check.shape[0]),
            },
            {
                "check": "test warehouse-date totals vs raw orders",
                "max_abs_diff": float(test_check["abs_diff"].max()),
                "rows_compared": int(test_check.shape[0]),
            },
            {
                "check": "train date coverage",
                "max_abs_diff": 0.0,
                "rows_compared": int(train_df["order_date"].nunique()),
            },
            {
                "check": "test date coverage",
                "max_abs_diff": 0.0,
                "rows_compared": int(test_df["order_date"].nunique()),
            },
        ]
    )


def _build_model_comparison(validation_summary: pd.DataFrame, test_summary: pd.DataFrame) -> pd.DataFrame:
    comparison = validation_summary.rename(
        columns={
            "mae": "validation_mae",
            "rmse": "validation_rmse",
            "wape": "validation_wape",
            "bias": "validation_bias",
            "underprediction_units": "validation_underprediction_units",
            "overprediction_units": "validation_overprediction_units",
            "actual_units": "validation_actual_units",
            "predicted_units": "validation_predicted_units",
            "purchase_cost_proxy": "validation_purchase_cost_proxy",
            "holding_cost_proxy": "validation_holding_cost_proxy",
            "price_weighted_abs_error": "validation_price_weighted_abs_error",
        }
    ).merge(
        test_summary.rename(
            columns={
                "mae": "test_mae",
                "rmse": "test_rmse",
                "wape": "test_wape",
                "bias": "test_bias",
                "underprediction_units": "test_underprediction_units",
                "overprediction_units": "test_overprediction_units",
                "actual_units": "test_actual_units",
                "predicted_units": "test_predicted_units",
                "purchase_cost_proxy": "test_purchase_cost_proxy",
                "holding_cost_proxy": "test_holding_cost_proxy",
                "price_weighted_abs_error": "test_price_weighted_abs_error",
            }
        ),
        on=["model_id", "approach", "prediction_level", "model_name", "family"],
        how="left",
    )
    comparison["validation_total_proxy_cost"] = (
        comparison["validation_purchase_cost_proxy"] + comparison["validation_holding_cost_proxy"]
    )
    comparison["test_total_proxy_cost"] = comparison["test_purchase_cost_proxy"] + comparison["test_holding_cost_proxy"]
    comparison["validation_selection_status"] = "screened_out"
    eligible_mask = (
        (comparison["validation_wape"] <= 0.45)
        & (comparison["validation_bias"] >= -0.15)
        & (comparison["validation_bias"] <= 0.15)
    )
    comparison.loc[eligible_mask, "validation_selection_status"] = "eligible"
    comparison["validation_selection_rank"] = (comparison["validation_selection_status"] != "eligible").astype(int)
    comparison = comparison.sort_values(
        [
            "validation_selection_rank",
            "validation_total_proxy_cost",
            "validation_purchase_cost_proxy",
            "validation_wape",
            "validation_rmse",
        ],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)
    comparison.insert(0, "selection_rank", range(1, len(comparison) + 1))
    return comparison


def main() -> None:
    paths = build_paths(ROOT)
    for directory in [paths.results_dir, paths.tables_dir, paths.figures_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    previous_metrics = _load_previous_metrics(paths.previous_overall_metrics_path)
    parameter_update_baseline = _load_json_snapshot(paths.results_dir / "parameter_update_baseline_snapshot.json")
    raw = load_raw_data(paths)
    date_summary = _date_coverage_summary(raw)
    consistency_checks = _verify_demand_file_consistency(
        raw_orders=raw.orders,
        train_df=raw.train_cluster_demand,
        test_df=raw.test_cluster_demand,
    )

    actual_start = raw.orders["order_date"].min().normalize()
    actual_end = raw.orders["order_date"].max().normalize()
    if actual_start != EXPECTED_START_DATE or actual_end != EXPECTED_END_DATE:
        raise ValueError(f"Unexpected order-date coverage: {actual_start.date()} to {actual_end.date()}.")

    demand_artifacts = prepare_real_cluster_demand_panels(
        train_df=raw.train_cluster_demand,
        test_df=raw.test_cluster_demand,
    )
    parameter_artifacts = build_parameter_artifacts(
        orders=raw.orders,
        order_mart=raw.order_mart,
        inventory=raw.inventory,
        network=raw.network,
        capacity=raw.capacity,
        daily_sku_dc_summary=raw.daily_sku_dc_summary,
        cluster_mapping=raw.cluster_mapping,
        demand_artifacts=demand_artifacts,
    )

    modeling = run_modeling_experiments(
        train_panel=demand_artifacts.train_panel,
        test_panel=demand_artifacts.test_panel,
        orders=raw.orders,
        clicks=raw.clicks,
        warehouse_universe=demand_artifacts.warehouse_universe,
        cluster_universe=demand_artifacts.cluster_universe,
    )
    validation_actual_panel = demand_artifacts.train_panel[
        (demand_artifacts.train_panel["date"] >= pd.Timestamp("2018-03-18"))
        & (demand_artifacts.train_panel["date"] <= TRAIN_END_DATE)
    ].copy()
    validation_evaluation = evaluate_model_predictions(
        actual_pair_panel=validation_actual_panel,
        predicted_pair_panel=modeling.all_validation_predictions,
        cluster_prices=parameter_artifacts.cluster_prices,
        holding_cost=H_VALUE,
        purchasing_cost=U_VALUE,
    )
    validation_selection = validation_evaluation.model_summary.copy()
    validation_selection["validation_total_proxy_cost"] = (
        validation_selection["purchase_cost_proxy"] + validation_selection["holding_cost_proxy"]
    )
    validation_selection["selection_status"] = "screened_out"
    eligible_mask = (
        (validation_selection["wape"] <= 0.45)
        & (validation_selection["bias"] >= -0.15)
        & (validation_selection["bias"] <= 0.15)
    )
    validation_selection.loc[eligible_mask, "selection_status"] = "eligible"
    if eligible_mask.any():
        selection_pool = validation_selection.loc[eligible_mask].copy()
    else:
        selection_pool = validation_selection.copy()
    validation_selection = validation_selection.sort_values(
        ["validation_total_proxy_cost", "purchase_cost_proxy", "wape", "rmse", "mae"],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)
    selection_pool = selection_pool.sort_values(
        ["validation_total_proxy_cost", "purchase_cost_proxy", "wape", "rmse", "mae"],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)
    best_model_id = str(selection_pool.iloc[0]["model_id"])

    evaluation = evaluate_model_predictions(
        actual_pair_panel=demand_artifacts.test_panel,
        predicted_pair_panel=modeling.all_test_predictions,
        cluster_prices=parameter_artifacts.cluster_prices,
        holding_cost=H_VALUE,
        purchasing_cost=U_VALUE,
        best_model_id=best_model_id,
    )
    figure_paths = save_figures(evaluation, paths.figures_dir, best_model_id)
    selected_feature_importance = extract_feature_importance(best_model_id, modeling)
    learned_candidates = evaluation.model_summary[evaluation.model_summary["family"] != "baseline"].copy()
    if learned_candidates.empty:
        best_learned_model_id = best_model_id
    else:
        best_learned_model_id = str(learned_candidates.sort_values(["wape", "rmse"]).iloc[0]["model_id"])
    learned_feature_importance = extract_feature_importance(best_learned_model_id, modeling)
    figure_paths["selected_model_feature_importance"] = _save_feature_importance_figure(
        selected_feature_importance,
        paths.figures_dir / "selected_model_feature_importance.png",
        f"Selected Model Drivers ({best_model_id})",
    )
    figure_paths["best_learned_model_feature_importance"] = _save_feature_importance_figure(
        learned_feature_importance,
        paths.figures_dir / "best_learned_model_feature_importance.png",
        f"Best Learned Model Drivers ({best_learned_model_id})",
    )

    best_predicted_demand = modeling.all_test_predictions[
        modeling.all_test_predictions["model_id"] == best_model_id
    ][["date", "warehouse", "sku_cluster_ID", "predicted_demand"]].copy()
    optimization_artifacts = run_optimization_comparison(
        predicted_demand_panel=best_predicted_demand,
        actual_demand_panel=demand_artifacts.test_panel[["date", "warehouse", "sku_cluster_ID", "demand"]].copy(),
        parameter_artifacts=parameter_artifacts,
    )
    figure_paths.update(save_optimization_figures(optimization_artifacts, paths.figures_dir))

    model_comparison = _build_model_comparison(
        validation_summary=validation_evaluation.model_summary,
        test_summary=evaluation.model_summary,
    )
    best_model_summary = evaluation.best_model_summary.copy()
    best_cluster_metrics = evaluation.best_cluster_metrics.sort_values("actual_units", ascending=False).copy()
    best_warehouse_metrics = evaluation.best_warehouse_metrics.sort_values("actual_units", ascending=False).copy()
    best_daily_totals = evaluation.best_daily_totals.sort_values("date").copy()

    best_model_label = f"{best_model_summary.iloc[0]['prediction_level']} / {best_model_summary.iloc[0]['model_name']}"
    optimization_summary_table = _build_optimization_summary_table(optimization_artifacts)
    parameter_update_table = _build_parameter_update_table(
        baseline_snapshot=parameter_update_baseline,
        current_best_model_summary=best_model_summary,
        optimization_summary_table=optimization_summary_table,
        cluster_prices=parameter_artifacts.cluster_prices,
        capacity_summary=parameter_artifacts.capacity,
    )
    daily_optimization_comparison = (
        optimization_artifacts.predicted_policy.daily_summary[
            ["date", "realized_total_cost", "realized_shortage_units", "realized_overflow_units", "realized_service_level"]
        ]
        .rename(
            columns={
                "realized_total_cost": "predicted_policy_cost",
                "realized_shortage_units": "predicted_policy_shortage_units",
                "realized_overflow_units": "predicted_policy_overflow_units",
                "realized_service_level": "predicted_policy_service_level",
            }
        )
        .merge(
            optimization_artifacts.oracle_policy.daily_summary[
                ["date", "realized_total_cost", "realized_shortage_units", "realized_overflow_units", "realized_service_level"]
            ].rename(
                columns={
                    "realized_total_cost": "oracle_cost",
                    "realized_shortage_units": "oracle_shortage_units",
                    "realized_overflow_units": "oracle_overflow_units",
                    "realized_service_level": "oracle_service_level",
                }
            ),
            on="date",
            how="left",
        )
    )
    daily_optimization_comparison["cost_gap_vs_oracle"] = (
        daily_optimization_comparison["predicted_policy_cost"] - daily_optimization_comparison["oracle_cost"]
    )

    files_used = [
        {
            "file": "Parameters setting.ipynb",
            "used": "Yes",
            "role": "Primary source for optimization parameter definitions and required consistency checks.",
        },
        {
            "file": "sku_warehouse_train_test_clusters_train_warehouse_daily_demand.csv",
            "used": "Yes",
            "role": "Ground-truth March 1-24 cluster-by-warehouse daily demand.",
        },
        {
            "file": "sku_warehouse_train_test_clusters_test_warehouse_daily_demand.csv",
            "used": "Yes",
            "role": "Ground-truth March 25-31 cluster-by-warehouse daily demand for test evaluation.",
        },
        {
            "file": "JD_order_data.csv",
            "used": "Yes",
            "role": "Promotion and revenue features, plus consistency checks against demand totals.",
        },
        {
            "file": "JD_click_data.csv",
            "used": "Yes",
            "role": "Lagged global click signals for forecasting.",
        },
        {
            "file": "JD_inventory_data.csv",
            "used": "Yes",
            "role": "Fallback warehouse inventory presence source; the primary inventory state now comes from the all-SKU daily summary and is then apportioned to clusters.",
        },
        {
            "file": "JD_network_data.csv",
            "used": "Yes",
            "role": "Procurement eligibility parameter `W_j`.",
        },
        {
            "file": "Optimization/JD_order_mart.csv",
            "used": "Yes",
            "role": "Training-window lead times and route availability matrices.",
        },
        {
            "file": "Optimization/inventory_capacity.xlsx",
            "used": "Yes",
            "role": "Legacy warehouse capacity file retained as fallback where corrected capacity history is missing.",
        },
        {
            "file": "Optimization/JD_daily_sku_dc_summary.xlsx",
            "used": "Yes",
            "role": "Corrected warehouse capacity source via max observed warehouse-day inventory summed across all SKUs.",
        },
        {
            "file": "results/optimization_prediction/tables/cluster_mapping.csv",
            "used": "Yes",
            "role": "Best available SKU-to-cluster bridge used to rebuild corrected cluster prices from SKU unit prices.",
        },
        {
            "file": "Optimization/Capstone Meeting Notes.docx",
            "used": "Yes",
            "role": "Project direction, instructor expectations, and evaluation framing.",
        },
        {
            "file": "Executive_Summary_Team121.docx",
            "used": "Yes",
            "role": "Business framing and operational interpretation context.",
        },
        {
            "file": "Capstone Projects 121 and 122 2025-2026 Main Doc and Resources.docx",
            "used": "Yes",
            "role": "Official project description and team roster context.",
        },
    ]

    meeting_notes_alignment = [
        {
            "source": "Capstone Meeting Notes",
            "expectation": "Demand forecasting should feed the optimization-side daily planning problem.",
            "implementation": "Forecast target is warehouse-cluster demand `D_{s,j}` and all results are evaluated at that level.",
        },
        {
            "source": "Capstone Meeting Notes",
            "expectation": "Use a proper train/test split and justify it clearly.",
            "implementation": "Used March 1-24, 2018 for training, March 18-24 as inner validation, and March 25-31 as the final test.",
        },
        {
            "source": "Capstone Meeting Notes",
            "expectation": "Benchmark simple models against stronger predictive methods.",
            "implementation": f"Compared {model_comparison.shape[0]} configurations spanning baselines, linear models, regularized linear models, random forests, and gradient boosting.",
        },
        {
            "source": "Executive Summary / Main Doc",
            "expectation": "Connect prediction quality to operational decisions and risk.",
            "implementation": "Reported forecast proxy costs and then evaluated real optimization decisions against an oracle policy.",
        },
    ]

    issues_found = [
        {
            "issue": "Notebook comments mention 2024 while raw files span 2018",
            "evidence": "The notebook text is inconsistent with the actual JD data files and code filters.",
            "correction": "Used March 2018 because all local datasets and the new demand files are internally consistent on that horizon.",
        },
        {
            "issue": "Test cluster-demand file omits cluster 2 rows",
            "evidence": "The provided test file contains only 21 observed cluster IDs even though the train file contains 22.",
            "correction": "Restored the missing cluster-warehouse-date rows as explicit zeros after verifying warehouse-date totals against raw orders.",
        },
        {
            "issue": "Notebook unit-price comments conflict with the executable code",
            "evidence": "The notebook text describes dividing `original_unit_price` by quantity, but the executable line leaves that division commented out.",
            "correction": "Rebuilt SKU prices directly from `original_unit_price` and then aggregated them to cluster prices using the available SKU-to-cluster mapping artifact.",
        },
        {
            "issue": "Legacy capacity file likely reflects only 1P SKU logic",
            "evidence": "Team update indicated the original capacity parameter was based on 1P SKU only, so the static capacity file could understate or distort warehouse limits.",
            "correction": "Rebuilt `C_j` from `JD_daily_sku_dc_summary.xlsx` as the max observed warehouse-day inventory across all SKUs, with explicit fallback to the legacy file only when the corrected history is missing.",
        },
        {
            "issue": "Original raw SKU-to-cluster artifact is still missing",
            "evidence": "The notebook references `sku_cluster_data.xlsx`, but no local source file is available in the repository.",
            "correction": "Used the saved `cluster_mapping.csv` artifact as the best available bridge and kept this dependency visible in the diagnostics and report.",
        },
        {
            "issue": "Inventory data lacks cluster-level quantity snapshots",
            "evidence": "The local inventory file records SKU presence by warehouse-date but not cluster inventory quantities.",
            "correction": "Allocated warehouse-day inventory to clusters using training demand shares, prioritizing `JD_daily_sku_dc_summary.xlsx` and retaining the raw inventory file only as a fallback when needed.",
        },
    ]

    parameter_diagnostics = pd.DataFrame(
        _flatten_diagnostics("parameter", parameter_artifacts.diagnostics)
        + _flatten_diagnostics("demand_file_checks", consistency_checks.to_dict(orient="records"))
    )

    report_context = {
        "train_start": EXPECTED_START_DATE,
        "train_end": TRAIN_END_DATE,
        "test_start": TEST_START_DATE,
        "test_end": TEST_END_DATE,
        "constants": {"k": K_VALUE, "h": H_VALUE, "u": U_VALUE},
        "files_used": files_used,
        "meeting_notes_alignment": meeting_notes_alignment,
        "issues_found": issues_found,
        "parameter_diagnostics": parameter_diagnostics,
        "price_summary": parameter_artifacts.price_summary,
        "capacity_summary": parameter_artifacts.capacity_summary,
        "parameter_update_table": parameter_update_table,
        "model_comparison": model_comparison,
        "best_model_summary": best_model_summary,
        "best_cluster_metrics": best_cluster_metrics,
        "best_warehouse_metrics": best_warehouse_metrics,
        "best_daily_totals": best_daily_totals,
        "selected_feature_importance": selected_feature_importance,
        "learned_feature_importance": learned_feature_importance,
        "best_learned_model_label": best_learned_model_id,
        "optimization_summary_table": optimization_summary_table,
        "daily_optimization_comparison": daily_optimization_comparison,
        "parameter_usage_audit": optimization_artifacts.parameter_usage_audit,
        "figure_paths": figure_paths,
        "previous_metrics": previous_metrics,
        "best_model_label": best_model_label,
        "n_models_compared": int(model_comparison.shape[0]),
    }

    markdown_report = render_markdown_report(report_context)
    write_report_files(markdown_report, paths.report_md, paths.report_html)
    build_document()

    _save_table(consistency_checks, paths.tables_dir / "demand_file_consistency_checks.csv")
    _save_table(parameter_artifacts.sku_prices, paths.tables_dir / "sku_prices_train.csv")
    _save_table(parameter_artifacts.cluster_mapping, paths.tables_dir / "cluster_mapping.csv")
    _save_table(parameter_artifacts.cluster_prices, paths.tables_dir / "cluster_prices_train.csv")
    _save_table(parameter_artifacts.price_summary, paths.tables_dir / "cluster_price_summary.csv")
    _save_table(parameter_artifacts.capacity, paths.tables_dir / "warehouse_capacity.csv")
    _save_table(parameter_artifacts.capacity_summary, paths.tables_dir / "warehouse_capacity_summary.csv")
    _save_table(parameter_artifacts.procurement_eligibility, paths.tables_dir / "procurement_eligibility.csv")
    _save_table(parameter_artifacts.initial_inventory_test, paths.tables_dir / "initial_inventory_test.csv")
    _save_table(parameter_artifacts.train_inventory_parameter, paths.tables_dir / "train_inventory_parameter.csv")
    _save_table(parameter_artifacts.test_inventory_proxy, paths.tables_dir / "test_inventory_proxy.csv")
    _save_table(parameter_artifacts.train_demand_parameter, paths.tables_dir / "train_demand_parameter.csv")
    _save_table(parameter_artifacts.test_demand_actual, paths.tables_dir / "test_demand_actual.csv")
    _save_table(demand_artifacts.train_panel, paths.tables_dir / "train_daily_cluster_warehouse_demand.csv")
    _save_table(demand_artifacts.test_panel, paths.tables_dir / "test_daily_cluster_warehouse_demand.csv")
    _save_table(validation_evaluation.model_summary, paths.tables_dir / "validation_model_comparison.csv")
    _save_table(evaluation.model_summary, paths.tables_dir / "test_model_summary.csv")
    _save_table(model_comparison, paths.tables_dir / "model_comparison.csv")
    _save_table(modeling.all_validation_predictions, paths.tables_dir / "all_validation_predictions.csv")
    _save_table(modeling.all_test_predictions, paths.tables_dir / "all_test_predictions.csv")
    _save_table(modeling.all_cluster_predictions, paths.tables_dir / "all_cluster_total_predictions.csv")
    _save_table(
        modeling.all_test_predictions[modeling.all_test_predictions["model_id"] == best_model_id].copy(),
        paths.tables_dir / "best_model_test_predictions.csv",
    )
    _save_table(
        modeling.all_cluster_predictions[modeling.all_cluster_predictions["model_id"] == best_model_id].copy(),
        paths.tables_dir / "best_model_cluster_predictions.csv",
    )
    _save_table(modeling.best_allocation_table, paths.tables_dir / "cluster_to_warehouse_allocation_table.csv")
    _save_table(evaluation.merged_predictions, paths.tables_dir / "all_model_detailed_predictions.csv")
    _save_table(evaluation.best_model_summary, paths.tables_dir / "overall_metrics.csv")
    _save_table(evaluation.best_cluster_metrics, paths.tables_dir / "best_model_cluster_metrics.csv")
    _save_table(evaluation.best_warehouse_metrics, paths.tables_dir / "best_model_warehouse_metrics.csv")
    _save_table(evaluation.best_daily_totals, paths.tables_dir / "best_model_daily_totals.csv")
    _save_table(selected_feature_importance, paths.tables_dir / "selected_model_feature_importance.csv")
    _save_table(learned_feature_importance, paths.tables_dir / "best_learned_model_feature_importance.csv")
    _save_table(optimization_artifacts.parameter_usage_audit, paths.tables_dir / "parameter_usage_audit.csv")
    _save_table(optimization_summary_table, paths.tables_dir / "optimization_summary_table.csv")
    _save_table(daily_optimization_comparison, paths.tables_dir / "daily_optimization_comparison.csv")
    _save_table(optimization_artifacts.predicted_policy.daily_summary, paths.tables_dir / "predicted_policy_daily_summary.csv")
    _save_table(optimization_artifacts.oracle_policy.daily_summary, paths.tables_dir / "oracle_policy_daily_summary.csv")
    _save_table(optimization_artifacts.predicted_policy.procurement_decisions, paths.tables_dir / "predicted_policy_procurement.csv")
    _save_table(optimization_artifacts.predicted_policy.transfer_decisions, paths.tables_dir / "predicted_policy_transfers.csv")
    _save_table(optimization_artifacts.predicted_policy.planned_inventory, paths.tables_dir / "predicted_policy_planned_inventory.csv")
    _save_table(optimization_artifacts.predicted_policy.planned_shortages, paths.tables_dir / "predicted_policy_planned_shortages.csv")
    _save_table(optimization_artifacts.predicted_policy.realized_inventory, paths.tables_dir / "predicted_policy_realized_inventory.csv")
    _save_table(optimization_artifacts.predicted_policy.realized_shortages, paths.tables_dir / "predicted_policy_realized_shortages.csv")
    _save_table(optimization_artifacts.oracle_policy.procurement_decisions, paths.tables_dir / "oracle_policy_procurement.csv")
    _save_table(optimization_artifacts.oracle_policy.transfer_decisions, paths.tables_dir / "oracle_policy_transfers.csv")
    _save_table(optimization_artifacts.oracle_policy.planned_inventory, paths.tables_dir / "oracle_policy_planned_inventory.csv")
    _save_table(optimization_artifacts.oracle_policy.planned_shortages, paths.tables_dir / "oracle_policy_planned_shortages.csv")
    _save_table(optimization_artifacts.oracle_policy.realized_inventory, paths.tables_dir / "oracle_policy_realized_inventory.csv")
    _save_table(optimization_artifacts.oracle_policy.realized_shortages, paths.tables_dir / "oracle_policy_realized_shortages.csv")
    _save_table(optimization_artifacts.weekly_comparison, paths.tables_dir / "optimization_weekly_comparison.csv")
    _save_table(parameter_update_table, paths.tables_dir / "parameter_update_table.csv")
    parameter_artifacts.delivery_time_matrix.to_csv(paths.tables_dir / "delivery_time_matrix_train.csv")
    parameter_artifacts.route_matrix.to_csv(paths.tables_dir / "route_availability_matrix_train.csv")
    for stale_path in [
        paths.tables_dir / "cluster_price_fit_summary.csv",
    ]:
        if stale_path.exists():
            stale_path.unlink()

    parameter_update_summary = {
        "price_logic_updated": True,
        "capacity_logic_updated": True,
        "cluster_price_source": "mean original_unit_price by SKU aggregated to clusters through cluster_mapping.csv",
        "capacity_source": "max observed warehouse-day inventory from JD_daily_sku_dc_summary.xlsx, with nonpositive values treated as missing and inventory_capacity.xlsx used as fallback",
        "affected_files": [
            "src/optimization_prediction/data_loading.py",
            "src/optimization_prediction/parameter_builder.py",
            "src/optimization_prediction/evaluation.py",
            "src/optimization_prediction/optimization_solver.py",
            "src/optimization_prediction/reporting.py",
            "optimization_prediction_pipeline.py",
            "optimization_model_report.md",
            "optimization_model_report.pdf",
            "model_explanation_for_presentation.md",
            "model_explanation_for_presentation.pdf",
        ],
        "selected_model": best_model_id,
        "key_result_deltas": {
            row["change_area"]: row["delta"]
            for row in parameter_update_table.to_dict(orient="records")
            if row["change_area"] != "selected_model"
        },
        "material_change_assessment": (
            "material" if not parameter_update_table.empty and abs(float(parameter_update_table.loc[
                parameter_update_table["change_area"] == "optimization_realized_gap_vs_oracle", "delta"
            ].iloc[0])) > 0.05 * max(1.0, float(parameter_update_table.loc[
                parameter_update_table["change_area"] == "optimization_realized_gap_vs_oracle", "previous_value"
            ].iloc[0])) else "limited"
        ),
    }
    (paths.results_dir / "parameter_update_summary.json").write_text(json.dumps(parameter_update_summary, indent=2), encoding="utf-8")

    run_summary = {
        "date_coverage": date_summary,
        "consistency_checks": consistency_checks.to_dict(orient="records"),
        "selected_model": best_model_id,
        "selected_model_label": best_model_label,
        "validation_top_5": validation_selection.head(5).to_dict(orient="records"),
        "test_top_5": evaluation.model_summary.head(5).to_dict(orient="records"),
        "optimization_summary": optimization_summary_table.to_dict(orient="records"),
        "parameter_update_table": parameter_update_table.to_dict(orient="records"),
        "figure_paths": figure_paths,
        "report_markdown": str(paths.report_md),
        "report_pdf": str(paths.report_pdf),
    }
    (paths.results_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    print("Optimization-side prediction pipeline completed.")
    print(f"Best model: {best_model_label}")
    print(f"Results directory: {paths.results_dir}")
    print(f"Report PDF: {paths.report_pdf}")


if __name__ == "__main__":
    main()
