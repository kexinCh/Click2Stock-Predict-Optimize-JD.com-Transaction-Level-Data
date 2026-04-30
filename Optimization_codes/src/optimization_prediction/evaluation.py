from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class EvaluationArtifacts:
    merged_predictions: pd.DataFrame
    model_summary: pd.DataFrame
    cluster_metrics: pd.DataFrame
    warehouse_metrics: pd.DataFrame
    daily_totals: pd.DataFrame
    best_model_summary: pd.DataFrame
    best_cluster_metrics: pd.DataFrame
    best_warehouse_metrics: pd.DataFrame
    best_daily_totals: pd.DataFrame


def _metric_series(frame: pd.DataFrame) -> pd.Series:
    actual = frame["demand"].astype(float)
    predicted = frame["predicted_demand"].astype(float)
    error = predicted - actual
    abs_error = error.abs()
    return pd.Series(
        {
            "actual_units": float(actual.sum()),
            "predicted_units": float(predicted.sum()),
            "mae": float(abs_error.mean()),
            "rmse": float(np.sqrt((error**2).mean())),
            "wape": float(abs_error.sum() / actual.sum()) if actual.sum() else float("nan"),
            "bias": float(error.sum() / actual.sum()) if actual.sum() else float("nan"),
            "underprediction_units": float((-error.clip(upper=0)).sum()),
            "overprediction_units": float(error.clip(lower=0).sum()),
            "price_weighted_abs_error": float(frame["price_weighted_abs_error"].sum()),
            "purchase_cost_proxy": float(frame["purchase_cost_proxy"].sum()),
            "holding_cost_proxy": float(frame["holding_cost_proxy"].sum()),
        }
    )


def evaluate_model_predictions(
    actual_pair_panel: pd.DataFrame,
    predicted_pair_panel: pd.DataFrame,
    cluster_prices: pd.DataFrame,
    holding_cost: float,
    purchasing_cost: float,
    best_model_id: str | None = None,
) -> EvaluationArtifacts:
    metadata_cols = ["model_id", "approach", "prediction_level", "model_name", "family"]
    merged = actual_pair_panel.merge(
        predicted_pair_panel,
        on=["date", "warehouse", "sku_cluster_ID"],
        how="left",
    )
    for column in metadata_cols:
        merged[column] = merged[column].fillna("unknown")
    merged["predicted_demand"] = merged["predicted_demand"].fillna(0)
    merged = merged.merge(cluster_prices, on="sku_cluster_ID", how="left")
    merged["p_s"] = merged["p_s"].fillna(0)
    merged["prediction_error"] = merged["predicted_demand"] - merged["demand"]
    merged["abs_error"] = merged["prediction_error"].abs()
    merged["underprediction"] = (-merged["prediction_error"].clip(upper=0)).astype(float)
    merged["overprediction"] = merged["prediction_error"].clip(lower=0).astype(float)
    merged["price_weighted_abs_error"] = merged["abs_error"] * merged["p_s"]
    merged["purchase_cost_proxy"] = merged["underprediction"] * merged["p_s"] * purchasing_cost
    merged["holding_cost_proxy"] = merged["overprediction"] * merged["p_s"] * holding_cost

    model_summary = (
        merged.groupby(metadata_cols)
        .apply(_metric_series, include_groups=False)
        .reset_index()
        .sort_values(["wape", "rmse", "mae"], ascending=[True, True, True])
        .reset_index(drop=True)
    )
    if best_model_id is None and not model_summary.empty:
        best_model_id = str(model_summary.iloc[0]["model_id"])

    cluster_metrics = (
        merged.groupby(metadata_cols + ["sku_cluster_ID"])
        .apply(_metric_series, include_groups=False)
        .reset_index()
        .sort_values(["model_id", "actual_units"], ascending=[True, False])
        .reset_index(drop=True)
    )

    warehouse_metrics = (
        merged.groupby(metadata_cols + ["warehouse"])
        .apply(_metric_series, include_groups=False)
        .reset_index()
        .sort_values(["model_id", "actual_units"], ascending=[True, False])
        .reset_index(drop=True)
    )

    daily_totals = (
        merged.groupby(metadata_cols + ["date"], as_index=False)[
            [
                "demand",
                "predicted_demand",
                "purchase_cost_proxy",
                "holding_cost_proxy",
                "underprediction",
                "overprediction",
            ]
        ]
        .sum()
        .rename(columns={"demand": "actual_total_demand", "predicted_demand": "predicted_total_demand"})
    )
    daily_totals["error"] = daily_totals["predicted_total_demand"] - daily_totals["actual_total_demand"]
    daily_totals["abs_error"] = daily_totals["error"].abs()

    best_model_summary = model_summary[model_summary["model_id"] == best_model_id].copy()
    best_cluster_metrics = cluster_metrics[cluster_metrics["model_id"] == best_model_id].copy()
    best_warehouse_metrics = warehouse_metrics[warehouse_metrics["model_id"] == best_model_id].copy()
    best_daily_totals = daily_totals[daily_totals["model_id"] == best_model_id].copy()

    return EvaluationArtifacts(
        merged_predictions=merged,
        model_summary=model_summary,
        cluster_metrics=cluster_metrics,
        warehouse_metrics=warehouse_metrics,
        daily_totals=daily_totals,
        best_model_summary=best_model_summary,
        best_cluster_metrics=best_cluster_metrics,
        best_warehouse_metrics=best_warehouse_metrics,
        best_daily_totals=best_daily_totals,
    )


def save_figures(
    evaluation: EvaluationArtifacts,
    figures_dir: Path,
    best_model_id: str,
) -> dict:
    figures_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = figures_dir / "model_comparison_wape.png"
    comparison = evaluation.model_summary.copy()
    comparison["label"] = comparison["approach"].str.replace("_", " ") + " | " + comparison["model_name"]
    comparison = comparison.sort_values("wape").head(12)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(comparison["label"], comparison["wape"], color="#003262")
    ax.set_title("Test-Week WAPE by Model")
    ax.set_xlabel("WAPE")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(comparison_path, dpi=180)
    plt.close(fig)

    daily_path = figures_dir / "best_model_daily_total_demand.png"
    best_daily = evaluation.best_daily_totals.sort_values("date")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(best_daily["date"], best_daily["actual_total_demand"], marker="o", label="Actual")
    ax.plot(best_daily["date"], best_daily["predicted_total_demand"], marker="o", label="Predicted")
    ax.set_title(f"Best Model Daily Totals ({best_model_id})")
    ax.set_ylabel("Units")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(daily_path, dpi=180)
    plt.close(fig)

    cluster_path = figures_dir / "best_model_top_clusters_wape.png"
    top_clusters = evaluation.best_cluster_metrics.sort_values("actual_units", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(top_clusters["sku_cluster_ID"].astype(str), top_clusters["wape"], color="#FDB515")
    ax.set_title("Best Model: Top 10 Clusters by Volume")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("WAPE")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(cluster_path, dpi=180)
    plt.close(fig)

    warehouse_path = figures_dir / "best_model_top_warehouses_wape.png"
    top_warehouses = evaluation.best_warehouse_metrics.sort_values("actual_units", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(top_warehouses["warehouse"].astype(str), top_warehouses["wape"], color="#3E7CB1")
    ax.set_title("Best Model: Top 10 Warehouses by Volume")
    ax.set_xlabel("Warehouse")
    ax.set_ylabel("WAPE")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(warehouse_path, dpi=180)
    plt.close(fig)

    residual_hist_path = figures_dir / "best_model_residual_histogram.png"
    best_rows = evaluation.merged_predictions[evaluation.merged_predictions["model_id"] == best_model_id].copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(best_rows["prediction_error"], bins=40, color="#003262", alpha=0.85)
    ax.axvline(0, color="#FDB515", linewidth=2)
    ax.set_title("Best Model Residual Distribution")
    ax.set_xlabel("Predicted - Actual")
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(residual_hist_path, dpi=180)
    plt.close(fig)

    daily_error_path = figures_dir / "best_model_daily_error.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(best_daily["date"], best_daily["error"], color="#C0392B")
    ax.axhline(0, color="#1F2933", linewidth=1.0)
    ax.set_title("Best Model Daily Forecast Error")
    ax.set_ylabel("Predicted - Actual")
    ax.grid(axis="y", alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(daily_error_path, dpi=180)
    plt.close(fig)

    return {
        "model_comparison_wape": str(comparison_path),
        "best_model_daily_totals": str(daily_path),
        "best_model_cluster_wape": str(cluster_path),
        "best_model_warehouse_wape": str(warehouse_path),
        "best_model_residual_histogram": str(residual_hist_path),
        "best_model_daily_error": str(daily_error_path),
    }
