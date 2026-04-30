from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.optimization_prediction.candidate_predictions import load_candidate_prediction_file
from src.optimization_prediction.config import build_paths
from src.optimization_prediction.data_loading import load_raw_data
from src.optimization_prediction.evaluation import evaluate_model_predictions
from src.optimization_prediction.lag1_baseline import build_lag1_baseline
from src.optimization_prediction.optimization_solver import run_receding_horizon_policy
from src.optimization_prediction.parameter_builder import build_parameter_artifacts, prepare_real_cluster_demand_panels


BERKELEY_BLUE = "#003262"
SLATE = "#4A5568"
GOLD = "#FDB515"
RED = "#B83227"
TEAL = "#2C7A7B"


CANDIDATE_FILES = [
    ("test_predictions.csv", "proposed_xgboost_b"),
    ("LightGBMB_prediction.csv", "lightgbm_b"),
    ("LightGBMC_prediction.csv", "lightgbm_c"),
    ("RandomForestB_prediction.csv", "random_forest_b"),
    ("xgboostC_prediction.csv", "xgboost_c"),
]


def _save_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _rank_frame(frame: pd.DataFrame, metric: str, ascending: bool, rank_col: str) -> pd.DataFrame:
    ranked = frame.sort_values(metric, ascending=ascending).reset_index(drop=True).copy()
    ranked[rank_col] = range(1, len(ranked) + 1)
    return ranked


def _build_optimization_summary(policy_summaries: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(policy_summaries)
    frame = frame.sort_values(["realized_total_cost", "realized_shortage_units", "realized_service_level"], ascending=[True, True, False]).reset_index(drop=True)
    frame["optimization_cost_rank"] = range(1, len(frame) + 1)
    return frame


def _create_figures(
    forecast_summary: pd.DataFrame,
    optimization_summary: pd.DataFrame,
    figures_dir: Path,
) -> dict[str, str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    plot = forecast_summary.sort_values("wape", ascending=True).copy()
    colors = [SLATE if x == "Lag-1 Baseline" else BERKELEY_BLUE for x in plot["candidate_label"]]
    bars = ax.barh(plot["candidate_label"], plot["wape"], color=colors)
    ax.set_title("Forecast WAPE Across Candidate Models", fontsize=13, weight="bold")
    ax.set_ylabel("WAPE")
    ax.grid(axis="x", alpha=0.22)
    ax.invert_yaxis()
    for bar, value in zip(bars, plot["wape"]):
        ax.text(value + 0.0012, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center", fontsize=8.8)
    fig.tight_layout()
    path = figures_dir / "figure1_candidate_forecast_wape.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths["forecast_wape"] = str(path)

    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    plot = optimization_summary.sort_values("realized_total_cost", ascending=True).copy()
    colors = [SLATE if x == "Lag-1 Baseline" else BERKELEY_BLUE for x in plot["candidate_label"]]
    bars = ax.barh(plot["candidate_label"], plot["realized_total_cost"], color=colors)
    ax.set_title("Realized Weekly Cost Across Candidate Forecasts", fontsize=13, weight="bold")
    ax.set_ylabel("Realized total cost")
    ax.grid(axis="x", alpha=0.22)
    ax.invert_yaxis()
    for bar, value in zip(bars, plot["realized_total_cost"]):
        ax.text(value + 120, bar.get_y() + bar.get_height() / 2, f"{value:,.0f}", va="center", fontsize=8.8)
    fig.tight_layout()
    path = figures_dir / "figure2_candidate_optimization_cost.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths["optimization_cost"] = str(path)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8))
    plot = optimization_summary.sort_values("realized_service_level", ascending=False).copy()
    colors = [BERKELEY_BLUE if x != "Lag-1 Baseline" else SLATE for x in plot["candidate_label"]]
    axes[0].bar(plot["candidate_label"], plot["realized_service_level"], color=colors)
    axes[0].set_title("Service Level", fontsize=12, weight="bold")
    axes[0].set_ylim(0, min(1.02, plot["realized_service_level"].max() * 1.05))
    axes[0].grid(axis="y", alpha=0.22)
    axes[0].tick_params(axis="x", rotation=30)
    axes[1].bar(plot["candidate_label"], plot["realized_shortage_units"], color=colors)
    axes[1].set_title("Shortage Units", fontsize=12, weight="bold")
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].tick_params(axis="x", rotation=30)
    fig.suptitle("Service and Shortage Across Candidate Forecasts", fontsize=13, weight="bold", y=1.02)
    fig.tight_layout()
    path = figures_dir / "figure3_candidate_service_shortage.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths["service_shortage"] = str(path)

    fig, ax = plt.subplots(figsize=(8.4, 5.5))
    merged = forecast_summary.merge(
        optimization_summary[["candidate_label", "realized_total_cost", "realized_service_level"]],
        on="candidate_label",
        how="inner",
    )
    for row in merged.itertuples(index=False):
        color = SLATE if row.candidate_label == "Lag-1 Baseline" else BERKELEY_BLUE
        size = 90 if row.candidate_label == "Lag-1 Baseline" else 80
        ax.scatter(row.wape, row.realized_total_cost, color=color, s=size)
        ax.text(row.wape + 0.0008, row.realized_total_cost + 250, row.candidate_label, fontsize=8.6)
    ax.set_title("WAPE vs Realized Weekly Cost", fontsize=13, weight="bold")
    ax.set_xlabel("WAPE")
    ax.set_ylabel("Realized total cost")
    ax.grid(alpha=0.22)
    fig.tight_layout()
    path = figures_dir / "figure4_wape_vs_optimization_cost.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths["wape_vs_cost"] = str(path)
    return paths


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

    candidate_artifacts = []
    diagnostics = []
    for filename, role in CANDIDATE_FILES:
        path = ROOT / filename
        artifact = load_candidate_prediction_file(path, actual_test, benchmark_role=role)
        candidate_artifacts.append(artifact.aligned_predictions.copy())
        diagnostics.append(artifact.diagnostics)

    lag1_panel, lag1_diag = build_lag1_baseline(demand.train_panel, demand.test_panel)
    lag1_predictions = lag1_panel.drop(columns=["demand"]).copy()
    lag1_predictions["source_file"] = "lag1_baseline"
    candidate_artifacts.append(lag1_predictions)
    diagnostics.append({"source_file": "lag1_baseline", **lag1_diag})

    all_predictions = pd.concat(candidate_artifacts, ignore_index=True)

    evaluation = evaluate_model_predictions(
        actual_pair_panel=actual_test,
        predicted_pair_panel=all_predictions.drop(columns=["actual_demand"], errors="ignore"),
        cluster_prices=parameters.cluster_prices,
        holding_cost=parameters.constants["h"],
        purchasing_cost=parameters.constants["u"],
    )
    forecast_summary = evaluation.model_summary.copy()
    label_map = (
        all_predictions[["model_id", "model_name", "feature_set", "source_file"]]
        .drop_duplicates("model_id")
        .assign(
            candidate_label=lambda frame: frame.apply(
                lambda row: "Lag-1 Baseline"
                if row["source_file"] == "lag1_baseline"
                else f"{row['model_name']} {row['feature_set']}".strip(),
                axis=1,
            )
        )[["model_id", "candidate_label", "source_file"]]
    )
    forecast_summary = forecast_summary.merge(label_map, on="model_id", how="left")
    forecast_summary["total_proxy_cost"] = forecast_summary["purchase_cost_proxy"] + forecast_summary["holding_cost_proxy"]
    forecast_summary = _rank_frame(forecast_summary, "wape", True, "forecast_wape_rank")
    forecast_summary["forecast_proxy_cost_rank"] = forecast_summary["total_proxy_cost"].rank(method="dense").astype(int)

    policy_summaries: list[dict] = []
    policy_daily_frames: list[pd.DataFrame] = []
    for row in label_map.itertuples(index=False):
        candidate_predictions = all_predictions[all_predictions["model_id"] == row.model_id].copy()
        policy = run_receding_horizon_policy(
            policy_name=row.model_id,
            planning_basis=row.candidate_label,
            planning_demand_panel=candidate_predictions.rename(columns={"predicted_demand": "demand"})[
                ["date", "sku_cluster_ID", "warehouse", "demand"]
            ],
            actual_demand_panel=actual_test[["date", "sku_cluster_ID", "warehouse", "demand"]],
            initial_inventory=parameters.initial_inventory_test,
            cluster_prices_frame=parameters.cluster_prices,
            procurement_eligibility_frame=parameters.procurement_eligibility,
            capacity_frame=parameters.capacity,
            route_matrix=parameters.route_matrix,
            delivery_time_matrix=parameters.delivery_time_matrix,
            constants=parameters.constants,
        )
        week = policy.weekly_summary.iloc[0].to_dict()
        week["model_id"] = row.model_id
        week["candidate_label"] = row.candidate_label
        week["source_file"] = row.source_file
        policy_summaries.append(week)
        policy_daily_frames.append(policy.daily_summary.assign(model_id=row.model_id, candidate_label=row.candidate_label))

    optimization_summary = _build_optimization_summary(policy_summaries)
    optimization_summary["best_balance_rank"] = (
        optimization_summary["optimization_cost_rank"] + optimization_summary["realized_service_level"].rank(ascending=False, method="dense")
    )
    optimization_summary = optimization_summary.sort_values(
        ["optimization_cost_rank", "realized_service_level"], ascending=[True, False]
    ).reset_index(drop=True)

    policy_daily_summary = pd.concat(policy_daily_frames, ignore_index=True)
    figures = _create_figures(forecast_summary, optimization_summary, ROOT / "report_figures")

    comparison_summary = forecast_summary[
        [
            "candidate_label",
            "source_file",
            "mae",
            "rmse",
            "wape",
            "bias",
            "underprediction_units",
            "overprediction_units",
            "purchase_cost_proxy",
            "holding_cost_proxy",
            "total_proxy_cost",
            "forecast_wape_rank",
            "forecast_proxy_cost_rank",
        ]
    ].merge(
        optimization_summary[
            [
                "candidate_label",
                "realized_total_cost",
                "realized_service_level",
                "realized_shortage_units",
                "realized_overflow_units",
                "realized_transfer_units",
                "realized_procurement_units",
                "realized_ending_inventory_units",
                "optimization_cost_rank",
            ]
        ],
        on="candidate_label",
        how="left",
    )
    comparison_summary["best_balance_score"] = (
        comparison_summary["optimization_cost_rank"] + comparison_summary["realized_service_level"].rank(ascending=False, method="dense")
    )
    comparison_summary = comparison_summary.sort_values(
        ["optimization_cost_rank", "forecast_wape_rank"], ascending=[True, True]
    ).reset_index(drop=True)

    files_dir = paths.tables_dir
    _save_table(all_predictions, files_dir / "candidate_model_predictions_harmonized.csv")
    _save_table(pd.DataFrame(diagnostics), files_dir / "candidate_prediction_file_diagnostics.csv")
    _save_table(forecast_summary, files_dir / "candidate_forecast_metrics.csv")
    _save_table(optimization_summary, files_dir / "candidate_optimization_metrics.csv")
    _save_table(policy_daily_summary, files_dir / "candidate_policy_daily_summary.csv")
    _save_table(comparison_summary, files_dir / "candidate_model_ranking_summary.csv")

    best_wape = forecast_summary.sort_values("wape").iloc[0]
    best_cost = optimization_summary.sort_values("realized_total_cost").iloc[0]
    best_balance = comparison_summary.sort_values(["best_balance_score", "optimization_cost_rank", "forecast_wape_rank"]).iloc[0]

    summary = {
        "models_tested": comparison_summary["candidate_label"].tolist(),
        "best_forecast_by_wape": {
            "candidate_label": str(best_wape["candidate_label"]),
            "wape": float(best_wape["wape"]),
        },
        "best_model_by_optimization_cost": {
            "candidate_label": str(best_cost["candidate_label"]),
            "realized_total_cost": float(best_cost["realized_total_cost"]),
        },
        "best_balance_between_cost_and_service": {
            "candidate_label": str(best_balance["candidate_label"]),
            "realized_total_cost": float(best_balance["realized_total_cost"]),
            "realized_service_level": float(best_balance["realized_service_level"]),
        },
        "does_best_wape_match_best_operational_model": bool(best_wape["candidate_label"] == best_cost["candidate_label"]),
        "figure_paths": figures,
    }
    (files_dir / "candidate_model_benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
