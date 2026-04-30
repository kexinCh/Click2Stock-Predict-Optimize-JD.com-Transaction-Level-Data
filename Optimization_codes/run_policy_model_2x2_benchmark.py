from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.optimization_prediction.baseline_policy import run_rule_based_policy
from src.optimization_prediction.candidate_predictions import load_candidate_prediction_file
from src.optimization_prediction.config import build_paths
from src.optimization_prediction.data_loading import load_raw_data
from src.optimization_prediction.lag1_baseline import build_lag1_baseline
from src.optimization_prediction.optimization_solver import run_receding_horizon_policy
from src.optimization_prediction.parameter_builder import build_parameter_artifacts, prepare_real_cluster_demand_panels


BERKELEY_BLUE = "#003262"
GOLD = "#FDB515"
SLATE = "#4A5568"
TEAL = "#2C7A7B"


def _save_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _build_combination_records(
    weekly: pd.DataFrame,
    daily: pd.DataFrame,
    prediction_input: str,
    policy_model: str,
    policy_family: str,
) -> tuple[dict, pd.DataFrame]:
    row = weekly.iloc[0].to_dict()
    row["prediction_input"] = prediction_input
    row["policy_model"] = policy_model
    row["policy_family"] = policy_family
    row["combination_id"] = f"{prediction_input.lower().replace(' ', '_')}__{policy_family.lower().replace(' ', '_')}"
    row["combination_label"] = f"{prediction_input} + {policy_model}"
    daily_frame = daily.copy()
    daily_frame["prediction_input"] = prediction_input
    daily_frame["policy_model"] = policy_model
    daily_frame["policy_family"] = policy_family
    daily_frame["combination_id"] = row["combination_id"]
    daily_frame["combination_label"] = row["combination_label"]
    return row, daily_frame


def _make_matrix_view(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    prediction_order = ["Best Prediction", "Lag-1 Baseline"]
    policy_order = ["Optimization Model", "Baseline Policy"]
    for prediction_input in prediction_order:
        row = {"prediction_input": prediction_input}
        subset = comparison[comparison["prediction_input"] == prediction_input]
        for policy_model in policy_order:
            match = subset[subset["policy_model"] == policy_model].iloc[0]
            prefix = "optimization_model" if policy_model == "Optimization Model" else "baseline_policy"
            row[f"{prefix}_cost"] = float(match["realized_total_cost"])
            row[f"{prefix}_service_level"] = float(match["realized_service_level"])
            row[f"{prefix}_shortage_units"] = float(match["realized_shortage_units"])
            row[f"{prefix}_overflow_units"] = float(match["realized_overflow_units"])
            row[f"{prefix}_transfer_units"] = float(match["realized_transfer_units"])
            row[f"{prefix}_procurement_units"] = float(match["realized_procurement_units"])
            row[f"{prefix}_ending_inventory_units"] = float(match["realized_ending_inventory_units"])
        rows.append(row)
    return pd.DataFrame(rows)


def _build_summary(comparison: pd.DataFrame) -> dict:
    ordered = comparison.sort_values(
        ["realized_total_cost", "realized_shortage_units", "realized_service_level"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    best = ordered.iloc[0]

    def _lookup(prediction_input: str, policy_model: str) -> pd.Series:
        match = comparison[
            (comparison["prediction_input"] == prediction_input) & (comparison["policy_model"] == policy_model)
        ]
        return match.iloc[0]

    best_pred_opt = _lookup("Best Prediction", "Optimization Model")
    best_pred_base = _lookup("Best Prediction", "Baseline Policy")
    lag_pred_opt = _lookup("Lag-1 Baseline", "Optimization Model")
    lag_pred_base = _lookup("Lag-1 Baseline", "Baseline Policy")

    return {
        "min_cost_combination": {
            "prediction_input": best["prediction_input"],
            "policy_model": best["policy_model"],
            "combination_label": best["combination_label"],
            "realized_total_cost": float(best["realized_total_cost"]),
            "realized_service_level": float(best["realized_service_level"]),
            "realized_shortage_units": float(best["realized_shortage_units"]),
        },
        "optimization_gain_best_prediction": {
            "cost_delta_vs_baseline_policy": float(best_pred_base["realized_total_cost"] - best_pred_opt["realized_total_cost"]),
            "service_level_delta_vs_baseline_policy": float(best_pred_opt["realized_service_level"] - best_pred_base["realized_service_level"]),
            "shortage_delta_vs_baseline_policy": float(best_pred_opt["realized_shortage_units"] - best_pred_base["realized_shortage_units"]),
        },
        "optimization_gain_lag1": {
            "cost_delta_vs_baseline_policy": float(lag_pred_base["realized_total_cost"] - lag_pred_opt["realized_total_cost"]),
            "service_level_delta_vs_baseline_policy": float(lag_pred_opt["realized_service_level"] - lag_pred_base["realized_service_level"]),
            "shortage_delta_vs_baseline_policy": float(lag_pred_opt["realized_shortage_units"] - lag_pred_base["realized_shortage_units"]),
        },
        "prediction_gain_with_optimization_model": {
            "cost_delta_best_vs_lag1": float(lag_pred_opt["realized_total_cost"] - best_pred_opt["realized_total_cost"]),
            "service_level_delta_best_vs_lag1": float(best_pred_opt["realized_service_level"] - lag_pred_opt["realized_service_level"]),
            "shortage_delta_best_vs_lag1": float(best_pred_opt["realized_shortage_units"] - lag_pred_opt["realized_shortage_units"]),
        },
        "prediction_gain_with_baseline_policy": {
            "cost_delta_best_vs_lag1": float(lag_pred_base["realized_total_cost"] - best_pred_base["realized_total_cost"]),
            "service_level_delta_best_vs_lag1": float(best_pred_base["realized_service_level"] - lag_pred_base["realized_service_level"]),
            "shortage_delta_best_vs_lag1": float(best_pred_base["realized_shortage_units"] - lag_pred_base["realized_shortage_units"]),
        },
    }


def _create_cost_figure(comparison: pd.DataFrame, figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plot = comparison.copy()
    plot["display_label"] = plot["prediction_input"].map(
        {
            "Best Prediction": "Best Pred.",
            "Lag-1 Baseline": "Lag-1",
        }
    ) + "\n" + plot["policy_model"].map(
        {
            "Optimization Model": "Optimization",
            "Baseline Policy": "Baseline Policy",
        }
    )
    color_map = {
        "Best Prediction + Optimization Model": BERKELEY_BLUE,
        "Best Prediction + Baseline Policy": TEAL,
        "Lag-1 Baseline + Optimization Model": GOLD,
        "Lag-1 Baseline + Baseline Policy": SLATE,
    }
    colors = [color_map[label] for label in plot["combination_label"]]

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    bars = ax.bar(plot["display_label"], plot["realized_total_cost"], color=colors)
    ax.set_title("Realized Weekly Cost Across Prediction-Policy Combinations", fontsize=13, weight="bold")
    ax.set_ylabel("Realized total cost")
    ax.grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, plot["realized_total_cost"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 160,
            f"{value:,.0f}",
            ha="center",
            va="bottom",
            fontsize=8.8,
        )
    fig.tight_layout()
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    paths = build_paths(ROOT)
    raw = load_raw_data(paths)
    demand = prepare_real_cluster_demand_panels(raw.train_cluster_demand, raw.test_cluster_demand)
    parameters = build_parameter_artifacts(
        orders=raw.orders,
        order_mart=raw.order_mart,
        inventory=raw.inventory,
        network=raw.network,
        capacity=raw.capacity,
        daily_sku_dc_summary=raw.daily_sku_dc_summary,
        cluster_mapping=raw.cluster_mapping,
        demand_artifacts=demand,
    )
    actual_test = demand.test_panel.copy()

    best_prediction_artifact = load_candidate_prediction_file(
        ROOT / "test_predictions.csv",
        actual_test_panel=actual_test,
        benchmark_role="proposed_xgboost_b",
    )
    best_prediction_panel = best_prediction_artifact.aligned_predictions.rename(
        columns={"predicted_demand": "demand"}
    )[["date", "warehouse", "sku_cluster_ID", "demand"]]

    lag1_panel, lag1_diag = build_lag1_baseline(demand.train_panel, demand.test_panel)
    lag1_prediction_panel = lag1_panel[
        ["date", "warehouse", "sku_cluster_ID", "predicted_demand"]
    ].rename(columns={"predicted_demand": "demand"})
    actual_panel = actual_test[["date", "warehouse", "sku_cluster_ID", "demand"]].copy()

    policy_runs = []
    daily_frames = []

    best_opt = run_receding_horizon_policy(
        policy_name="best_prediction_optimization_model",
        planning_basis="Best Prediction",
        planning_demand_panel=best_prediction_panel,
        actual_demand_panel=actual_panel,
        initial_inventory=parameters.initial_inventory_test,
        cluster_prices_frame=parameters.cluster_prices,
        procurement_eligibility_frame=parameters.procurement_eligibility,
        capacity_frame=parameters.capacity,
        route_matrix=parameters.route_matrix,
        delivery_time_matrix=parameters.delivery_time_matrix,
        constants=parameters.constants,
    )
    row, daily = _build_combination_records(
        best_opt.weekly_summary,
        best_opt.daily_summary,
        prediction_input="Best Prediction",
        policy_model="Optimization Model",
        policy_family="optimization_model",
    )
    policy_runs.append(row)
    daily_frames.append(daily)

    best_base = run_rule_based_policy(
        policy_name="best_prediction_baseline_policy",
        planning_basis="Best Prediction",
        planning_demand_panel=best_prediction_panel,
        actual_demand_panel=actual_panel,
        initial_inventory=parameters.initial_inventory_test,
        cluster_prices_frame=parameters.cluster_prices,
        procurement_eligibility_frame=parameters.procurement_eligibility,
        capacity_frame=parameters.capacity,
        route_matrix=parameters.route_matrix,
        delivery_time_matrix=parameters.delivery_time_matrix,
        constants=parameters.constants,
    )
    row, daily = _build_combination_records(
        best_base.weekly_summary,
        best_base.daily_summary,
        prediction_input="Best Prediction",
        policy_model="Baseline Policy",
        policy_family="baseline_policy",
    )
    policy_runs.append(row)
    daily_frames.append(daily)

    lag_opt = run_receding_horizon_policy(
        policy_name="lag1_baseline_optimization_model",
        planning_basis="Lag-1 Baseline",
        planning_demand_panel=lag1_prediction_panel,
        actual_demand_panel=actual_panel,
        initial_inventory=parameters.initial_inventory_test,
        cluster_prices_frame=parameters.cluster_prices,
        procurement_eligibility_frame=parameters.procurement_eligibility,
        capacity_frame=parameters.capacity,
        route_matrix=parameters.route_matrix,
        delivery_time_matrix=parameters.delivery_time_matrix,
        constants=parameters.constants,
    )
    row, daily = _build_combination_records(
        lag_opt.weekly_summary,
        lag_opt.daily_summary,
        prediction_input="Lag-1 Baseline",
        policy_model="Optimization Model",
        policy_family="optimization_model",
    )
    policy_runs.append(row)
    daily_frames.append(daily)

    lag_base = run_rule_based_policy(
        policy_name="lag1_baseline_policy",
        planning_basis="Lag-1 Baseline",
        planning_demand_panel=lag1_prediction_panel,
        actual_demand_panel=actual_panel,
        initial_inventory=parameters.initial_inventory_test,
        cluster_prices_frame=parameters.cluster_prices,
        procurement_eligibility_frame=parameters.procurement_eligibility,
        capacity_frame=parameters.capacity,
        route_matrix=parameters.route_matrix,
        delivery_time_matrix=parameters.delivery_time_matrix,
        constants=parameters.constants,
    )
    row, daily = _build_combination_records(
        lag_base.weekly_summary,
        lag_base.daily_summary,
        prediction_input="Lag-1 Baseline",
        policy_model="Baseline Policy",
        policy_family="baseline_policy",
    )
    policy_runs.append(row)
    daily_frames.append(daily)

    weekly_comparison = pd.DataFrame(policy_runs).sort_values(
        ["realized_total_cost", "realized_shortage_units", "realized_service_level"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    weekly_comparison["cost_rank"] = range(1, len(weekly_comparison) + 1)
    weekly_comparison["service_rank"] = weekly_comparison["realized_service_level"].rank(
        ascending=False, method="dense"
    ).astype(int)
    weekly_comparison["shortage_rank"] = weekly_comparison["realized_shortage_units"].rank(
        ascending=True, method="dense"
    ).astype(int)

    daily_comparison = pd.concat(daily_frames, ignore_index=True).sort_values(
        ["date", "prediction_input", "policy_model"]
    ).reset_index(drop=True)
    matrix_view = _make_matrix_view(weekly_comparison)
    summary = _build_summary(weekly_comparison)
    summary["proposed_model_file"] = "test_predictions.csv"
    summary["baseline_definition"] = "lag-1 naive forecast using previous day's actual demand"
    summary["lag1_diagnostics"] = lag1_diag
    summary["best_prediction_diagnostics"] = best_prediction_artifact.diagnostics

    tables_dir = ROOT / "results" / "optimization_prediction" / "tables"
    _save_table(weekly_comparison, tables_dir / "policy_model_2x2_weekly_comparison.csv")
    _save_table(daily_comparison, tables_dir / "policy_model_2x2_daily_comparison.csv")
    _save_table(matrix_view, tables_dir / "policy_model_2x2_matrix.csv")

    summary_path = tables_dir / "policy_model_2x2_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    figure_path = ROOT / "report_figures" / "figure5_policy_model_2x2_cost.png"
    _create_cost_figure(weekly_comparison, figure_path)

    print(
        json.dumps(
            {
                "weekly_comparison_path": str(tables_dir / "policy_model_2x2_weekly_comparison.csv"),
                "matrix_path": str(tables_dir / "policy_model_2x2_matrix.csv"),
                "summary_path": str(summary_path),
                "figure_path": str(figure_path),
                "min_cost_combination": summary["min_cost_combination"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
