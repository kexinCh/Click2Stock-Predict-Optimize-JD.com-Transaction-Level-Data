#!/usr/bin/env python3
"""Build and compare demand prediction models on prepared cluster-warehouse panels."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from project_paths import DATABASE_DIR, MODEL_OUTPUTS_DIR, PROCESSED_DATA_DIR, ensure_dir

SEED = 42
TARGET_COL = "demand"
DATE_COL = "date"
CLUSTER_COL = "cluster_id"
WAREHOUSE_COL = "warehouse_id"
SPLIT_COL = "dataset_split"


def load_dataset(base_dir: Path) -> tuple[pd.DataFrame, bool]:
    print("Loading prepared train/test demand datasets...")
    train_path = base_dir / "sku_warehouse_train_test_clusters_train_warehouse_daily_demand.csv"
    test_path = base_dir / "sku_warehouse_train_test_clusters_test_warehouse_daily_demand.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    rename_map = {
        "order_date": DATE_COL,
        "sku_cluster_ID": CLUSTER_COL,
        "dc_des": WAREHOUSE_COL,
    }
    train_df = train_df.rename(columns=rename_map)
    test_df = test_df.rename(columns=rename_map)

    for name, frame in [("train", train_df), ("test", test_df)]:
        missing = {DATE_COL, CLUSTER_COL, WAREHOUSE_COL, TARGET_COL} - set(frame.columns)
        if missing:
            raise ValueError(f"{name} dataset missing required columns: {sorted(missing)}")

    train_df[SPLIT_COL] = "train"
    test_df[SPLIT_COL] = "test"

    df = pd.concat([train_df, test_df], ignore_index=True)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce").dt.normalize()
    df = df.dropna(subset=[DATE_COL]).copy()
    df[CLUSTER_COL] = df[CLUSTER_COL].astype(str)
    df[WAREHOUSE_COL] = df[WAREHOUSE_COL].astype(str)
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0.0)
    df = df.sort_values([CLUSTER_COL, WAREHOUSE_COL, DATE_COL]).reset_index(drop=True)

    click_available = (
        (DATABASE_DIR / "click_orders_new.db").exists()
        and (base_dir / "sku_warehouse_train_test_clusters_features.csv").exists()
    )
    print(f"Loaded {len(df):,} rows across {df[CLUSTER_COL].nunique()} clusters and {df[WAREHOUSE_COL].nunique()} warehouses.")
    return df, click_available


def create_demand_history_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Creating demand history features...")
    out = df.copy()
    series_key = [CLUSTER_COL, WAREHOUSE_COL]
    grp = out.groupby(series_key, sort=False)

    out["lag_1_demand"] = grp[TARGET_COL].shift(1)
    out["lag_2_demand"] = grp[TARGET_COL].shift(2)
    out["lag_3_demand"] = grp[TARGET_COL].shift(3)
    out["rolling_mean_3_demand"] = grp[TARGET_COL].transform(
        lambda s: s.shift(1).rolling(window=3, min_periods=3).mean()
    )
    out["rolling_std_3_demand"] = grp[TARGET_COL].transform(
        lambda s: s.shift(1).rolling(window=3, min_periods=3).std()
    )
    out["demand_growth"] = (out["lag_1_demand"] - out["lag_2_demand"]) / (out["lag_2_demand"] + 1.0)
    return out


def create_cluster_popularity_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Creating cluster popularity features...")
    out = df.copy()
    cluster_daily = (
        out.groupby([CLUSTER_COL, DATE_COL], as_index=False)[TARGET_COL]
        .sum()
        .rename(columns={TARGET_COL: "cluster_total_demand_day"})
        .sort_values([CLUSTER_COL, DATE_COL])
    )
    grp = cluster_daily.groupby(CLUSTER_COL, sort=False)["cluster_total_demand_day"]
    cluster_daily["cluster_total_demand_lag1"] = grp.shift(1)
    cluster_daily["cluster_total_demand_rolling3"] = (
        cluster_daily.groupby(CLUSTER_COL, sort=False)["cluster_total_demand_day"]
        .transform(lambda s: s.shift(1).rolling(window=3, min_periods=3).mean())
    )
    return out.merge(cluster_daily, on=[CLUSTER_COL, DATE_COL], how="left")


def create_click_features_if_available(df: pd.DataFrame, base_dir: Path) -> tuple[pd.DataFrame, bool]:
    print("Creating click behavior features if available...")
    db_path = DATABASE_DIR / "click_orders_new.db"
    features_path = base_dir / "sku_warehouse_train_test_clusters_features.csv"
    click_metrics_path = base_dir / "click_to_order_metrics.csv"
    if not db_path.exists() or not features_path.exists():
        print("Click sources not found; skipping click feature generation.")
        return df, False

    assignments = pd.read_csv(features_path, usecols=["sku_ID", "sku_cluster_ID"]).drop_duplicates()
    assignments = assignments.rename(columns={"sku_cluster_ID": CLUSTER_COL})
    assignments[CLUSTER_COL] = assignments[CLUSTER_COL].astype(str)

    click_daily: pd.DataFrame | None = None
    if click_metrics_path.exists():
        try:
            metrics = pd.read_csv(
                click_metrics_path,
                usecols=["user_id", "sku", "first_click_time", "clicks_before_order"],
            )
            metrics["first_click_time"] = pd.to_datetime(metrics["first_click_time"], errors="coerce")
            metrics = metrics.dropna(subset=["first_click_time"]).rename(columns={"sku": "sku_ID"})
            metrics[DATE_COL] = metrics["first_click_time"].dt.normalize()
            metrics["clicks_before_order"] = pd.to_numeric(metrics["clicks_before_order"], errors="coerce").fillna(0.0)
            metrics = metrics.merge(assignments, on="sku_ID", how="inner")
            if not metrics.empty:
                click_daily = (
                    metrics.groupby([CLUSTER_COL, DATE_COL], as_index=False)
                    .agg(
                        click_count=("clicks_before_order", "sum"),
                        unique_click_users=("user_id", "nunique"),
                    )
                    .sort_values([CLUSTER_COL, DATE_COL])
                )
                print("Using click_to_order_metrics.csv as the click feature source.")
        except Exception as exc:
            print(f"Unable to use click_to_order_metrics.csv ({exc}); trying SQLite.")

    if click_daily is None:
        min_date = df[DATE_COL].min().strftime("%Y-%m-%d")
        max_date = df[DATE_COL].max().strftime("%Y-%m-%d")
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            conn.execute("PRAGMA temp_store=MEMORY")
            clicks = pd.read_sql_query(
                f"""
                SELECT
                    sku_ID,
                    date(request_time) AS date,
                    COUNT(*) AS click_count,
                    COUNT(DISTINCT user_ID) AS unique_click_users
                FROM clicks
                WHERE request_time >= '{min_date}' AND request_time < date('{max_date}', '+1 day')
                GROUP BY sku_ID, date(request_time)
                """,
                conn,
            )
            conn.close()
        except Exception as exc:
            print(f"Unable to load click data from SQLite ({exc}); skipping click features.")
            return df, False

        if clicks.empty:
            print("Click table is empty; skipping click features.")
            return df, False

        clicks[DATE_COL] = pd.to_datetime(clicks["date"], errors="coerce").dt.normalize()
        clicks = clicks.drop(columns=["date"]).dropna(subset=[DATE_COL])
        clicks = clicks.merge(assignments, on="sku_ID", how="inner")
        if clicks.empty:
            print("No click rows matched SKU-to-cluster assignments; skipping click features.")
            return df, False

        click_daily = (
            clicks.groupby([CLUSTER_COL, DATE_COL], as_index=False)
            .agg(
                click_count=("click_count", "sum"),
                unique_click_users=("unique_click_users", "sum"),
            )
            .sort_values([CLUSTER_COL, DATE_COL])
        )

    click_daily["click_per_user"] = click_daily["click_count"] / click_daily["unique_click_users"].clip(lower=1)
    click_grp = click_daily.groupby(CLUSTER_COL, sort=False)["click_count"]
    click_daily["lag_1_clicks"] = click_grp.shift(1)
    click_daily["lag_2_clicks"] = click_grp.shift(2)
    click_daily["click_velocity"] = click_daily["lag_1_clicks"] - click_daily["lag_2_clicks"]

    out = df.merge(click_daily, on=[CLUSTER_COL, DATE_COL], how="left")
    out["conversion_rate"] = out["cluster_total_demand_lag1"] / out["lag_1_clicks"].clip(lower=1)
    print("Click features created from cluster-level click aggregates.")
    return out, True


def create_warehouse_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Creating warehouse trend features...")
    out = df.copy()
    warehouse_daily = (
        out.groupby([WAREHOUSE_COL, DATE_COL], as_index=False)[TARGET_COL]
        .sum()
        .rename(columns={TARGET_COL: "warehouse_total_demand_day"})
        .sort_values([WAREHOUSE_COL, DATE_COL])
    )
    grp = warehouse_daily.groupby(WAREHOUSE_COL, sort=False)["warehouse_total_demand_day"]
    warehouse_daily["warehouse_demand_lag1"] = grp.shift(1)
    warehouse_daily["warehouse_demand_rolling3"] = (
        warehouse_daily.groupby(WAREHOUSE_COL, sort=False)["warehouse_total_demand_day"]
        .transform(lambda s: s.shift(1).rolling(window=3, min_periods=3).mean())
    )
    return out.merge(warehouse_daily, on=[WAREHOUSE_COL, DATE_COL], how="left")


def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Creating calendar features...")
    out = df.copy()
    out["day_of_week"] = out[DATE_COL].dt.dayofweek
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["day_of_month"] = out[DATE_COL].dt.day
    return out


def drop_rows_without_history(df: pd.DataFrame) -> pd.DataFrame:
    print("Dropping rows without sufficient lag history...")
    required = ["lag_1_demand", "lag_2_demand", "lag_3_demand"]
    out = df.dropna(subset=required).copy()
    print(f"Retained {len(out):,} rows after lag filtering.")
    return out


def build_feature_sets(click_available: bool) -> dict[str, list[str]]:
    print("Defining feature sets...")
    base_ids = [CLUSTER_COL, WAREHOUSE_COL]
    feature_sets = {
        "A": base_ids + ["lag_1_demand"],
        "B": base_ids + ["lag_1_demand", "lag_2_demand", "lag_3_demand"],
        "C": base_ids + [
            "lag_1_demand",
            "lag_2_demand",
            "lag_3_demand",
            "rolling_mean_3_demand",
            "rolling_std_3_demand",
            "demand_growth",
        ],
        "D": base_ids + [
            "lag_1_demand",
            "lag_2_demand",
            "lag_3_demand",
            "rolling_mean_3_demand",
            "rolling_std_3_demand",
            "demand_growth",
            "day_of_week",
            "is_weekend",
            "day_of_month",
        ],
        "E": base_ids + [
            "lag_1_demand",
            "lag_2_demand",
            "lag_3_demand",
            "rolling_mean_3_demand",
            "rolling_std_3_demand",
            "demand_growth",
            "day_of_week",
            "is_weekend",
            "day_of_month",
            "cluster_total_demand_lag1",
            "cluster_total_demand_rolling3",
            "warehouse_demand_lag1",
            "warehouse_demand_rolling3",
        ],
    }
    if click_available:
        feature_sets["F"] = feature_sets["E"] + [
            "click_count",
            "unique_click_users",
            "click_per_user",
            "lag_1_clicks",
            "lag_2_clicks",
            "click_velocity",
            "conversion_rate",
        ]
    return feature_sets


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    denom = float(y_true.sum())
    wape = float(np.abs(y_true - y_pred).sum() / denom) if denom > 0 else np.nan
    nonzero_mask = y_true > 0
    if nonzero_mask.any():
        mape = float((np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])).mean())
    else:
        mape = np.nan
    return {"MAE": mae, "RMSE": rmse, "WAPE": wape, "MAPE": mape}


def make_preprocessor(feature_cols: list[str]) -> tuple[ColumnTransformer, list[str], list[str]]:
    categorical_cols = [col for col in [CLUSTER_COL, WAREHOUSE_COL] if col in feature_cols]
    numeric_cols = [col for col in feature_cols if col not in categorical_cols]

    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            )
        )
    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor, numeric_cols, categorical_cols


def get_optional_model_factories() -> dict[str, Callable[[], object]]:
    factories: dict[str, Callable[[], object]] = {}
    try:
        from lightgbm import LGBMRegressor

        factories["LightGBM"] = lambda: LGBMRegressor(
            random_state=SEED,
            n_estimators=150,
            learning_rate=0.05,
            num_leaves=31,
        )
    except ImportError:
        print("LightGBM not installed; skipping.")

    try:
        from xgboost import XGBRegressor

        factories["XGBoost"] = lambda: XGBRegressor(
            random_state=SEED,
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
        )
    except ImportError:
        print("XGBoost not installed; skipping.")

    return factories


def build_model_factories() -> dict[str, Callable[[], object]]:
    factories: dict[str, Callable[[], object]] = {
        "LinearRegression": LinearRegression,
        "Ridge": lambda: Ridge(alpha=1.0, random_state=SEED),
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=120,
            random_state=SEED,
            n_jobs=-1,
            min_samples_leaf=2,
        ),
    }
    factories.update(get_optional_model_factories())
    return factories


def extract_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    feature_names: list[str] = []
    for _, transformer, columns in preprocessor.transformers_:
        if transformer == "drop":
            continue
        if hasattr(transformer, "named_steps") and "onehot" in transformer.named_steps:
            onehot = transformer.named_steps["onehot"]
            feature_names.extend(onehot.get_feature_names_out(columns).tolist())
        else:
            feature_names.extend(list(columns))
    return feature_names


def train_and_evaluate_models(df: pd.DataFrame, feature_sets: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Training and evaluating models...")
    train_df = df[df[SPLIT_COL] == "train"].copy()
    test_df = df[df[SPLIT_COL] == "test"].copy()

    results: list[dict[str, object]] = []
    predictions_frames: list[pd.DataFrame] = []
    feature_importance_rows: list[dict[str, object]] = []
    model_factories = build_model_factories()

    for feature_set_name, feature_cols in feature_sets.items():
        print(f"Evaluating feature set {feature_set_name}...")
        X_train = train_df[feature_cols]
        X_test = test_df[feature_cols]
        y_train = train_df[TARGET_COL]
        y_test = test_df[TARGET_COL]

        baseline_pred = test_df["lag_1_demand"].to_numpy()
        baseline_metrics = evaluate_predictions(y_test, baseline_pred)
        results.append({"model_name": "BaselineLag1", "feature_set": feature_set_name, **baseline_metrics})
        predictions_frames.append(
            pd.DataFrame(
                {
                    DATE_COL: test_df[DATE_COL],
                    CLUSTER_COL: test_df[CLUSTER_COL],
                    WAREHOUSE_COL: test_df[WAREHOUSE_COL],
                    "actual_demand": y_test,
                    "predicted_demand": baseline_pred,
                    "model_name": "BaselineLag1",
                    "feature_set": feature_set_name,
                }
            )
        )

        for model_name, model_factory in model_factories.items():
            print(f"  Fitting {model_name} on feature set {feature_set_name}...")
            preprocessor, _, _ = make_preprocessor(feature_cols)
            model = model_factory()
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", model),
                ]
            )
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            metrics = evaluate_predictions(y_test, y_pred)
            results.append({"model_name": model_name, "feature_set": feature_set_name, **metrics})
            predictions_frames.append(
                pd.DataFrame(
                    {
                        DATE_COL: test_df[DATE_COL],
                        CLUSTER_COL: test_df[CLUSTER_COL],
                        WAREHOUSE_COL: test_df[WAREHOUSE_COL],
                        "actual_demand": y_test,
                        "predicted_demand": y_pred,
                        "model_name": model_name,
                        "feature_set": feature_set_name,
                    }
                )
            )

            if model_name in {"RandomForest", "LightGBM", "XGBoost"} and hasattr(pipeline.named_steps["model"], "feature_importances_"):
                feature_names = extract_feature_names(pipeline.named_steps["preprocessor"])
                importances = pipeline.named_steps["model"].feature_importances_
                top_idx = np.argsort(importances)[::-1]
                for idx in top_idx:
                    feature_importance_rows.append(
                        {
                            "model_name": model_name,
                            "feature_set": feature_set_name,
                            "feature": feature_names[idx],
                            "importance": float(importances[idx]),
                        }
                    )

    results_df = pd.DataFrame(results).sort_values(["WAPE", "RMSE", "MAE"], ascending=[True, True, True]).reset_index(drop=True)
    predictions_df = pd.concat(predictions_frames, ignore_index=True)
    feature_importance_df = pd.DataFrame(feature_importance_rows)
    return results_df, predictions_df, feature_importance_df


def save_outputs(
    results_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    feature_importance_df: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Saving outputs...")
    output_dir.mkdir(parents=True, exist_ok=True)

    best_row = results_df.iloc[0]
    best_predictions = predictions_df[
        (predictions_df["model_name"] == best_row["model_name"])
        & (predictions_df["feature_set"] == best_row["feature_set"])
    ].copy()
    best_importance = feature_importance_df[
        (feature_importance_df["model_name"] == best_row["model_name"])
        & (feature_importance_df["feature_set"] == best_row["feature_set"])
    ].copy()

    results_df.to_csv(output_dir / "model_comparison.csv", index=False)
    best_predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    feature_importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

    plt.figure(figsize=(7, 7))
    plt.scatter(best_predictions["actual_demand"], best_predictions["predicted_demand"], alpha=0.45, edgecolor="none")
    max_val = float(max(best_predictions["actual_demand"].max(), best_predictions["predicted_demand"].max()))
    plt.plot([0, max_val], [0, max_val], linestyle="--", color="black", linewidth=1)
    plt.xlabel("Actual demand")
    plt.ylabel("Predicted demand")
    plt.title(f"Actual vs Predicted: {best_row['model_name']} / {best_row['feature_set']}")
    plt.tight_layout()
    plt.savefig(output_dir / "actual_vs_predicted_scatter.png", dpi=150)
    plt.close()

    daily_compare = (
        best_predictions.groupby(DATE_COL, as_index=False)
        .agg(actual_demand=("actual_demand", "sum"), predicted_demand=("predicted_demand", "sum"))
        .sort_values(DATE_COL)
    )
    plt.figure(figsize=(9, 5))
    plt.plot(daily_compare[DATE_COL], daily_compare["actual_demand"], marker="o", label="Actual")
    plt.plot(daily_compare[DATE_COL], daily_compare["predicted_demand"], marker="o", label="Predicted")
    plt.xticks(rotation=45)
    plt.ylabel("Daily total demand")
    plt.title(f"Daily Total Actual vs Predicted: {best_row['model_name']} / {best_row['feature_set']}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "daily_total_actual_vs_predicted.png", dpi=150)
    plt.close()

    return best_predictions, best_importance, daily_compare


def main() -> None:
    base_dir = PROCESSED_DATA_DIR
    output_dir = ensure_dir(MODEL_OUTPUTS_DIR)

    df, click_source_available = load_dataset(base_dir)
    df = create_demand_history_features(df)
    df = create_cluster_popularity_features(df)
    df, click_features_available = create_click_features_if_available(df, base_dir) if click_source_available else (df, False)
    df = create_warehouse_features(df)
    df = create_calendar_features(df)
    df = drop_rows_without_history(df)

    feature_sets = build_feature_sets(click_features_available)
    results_df, predictions_df, feature_importance_df = train_and_evaluate_models(df, feature_sets)
    best_predictions, best_importance, _ = save_outputs(results_df, predictions_df, feature_importance_df, output_dir)

    best_row = results_df.iloc[0]
    top_features = best_importance.sort_values("importance", ascending=False).head(10)

    print("\nExperiment summary:")
    print(results_df.to_string(index=False))
    print("\nBest model:", best_row["model_name"])
    print("Best feature set:", best_row["feature_set"])
    print("Best WAPE:", round(float(best_row["WAPE"]), 6))
    if not top_features.empty:
        print("Top important features:")
        print(top_features[["feature", "importance"]].to_string(index=False))
    else:
        print("Top important features: not available for the best model.")

    assumptions = [
        "Mapped order_date -> date, sku_cluster_ID -> cluster_id, and dc_des -> warehouse_id from the prepared panel files.",
        "Built lag and rolling features on the combined chronological train+test panel so first test days can use prior train history without retraining splits.",
        "Aggregated optional click features at cluster_id + date because the click sources have sku_ID and timestamps but no warehouse identifier.",
        "Created same-day cluster_total_demand_day and warehouse_total_demand_day for completeness, but excluded them from model feature sets to avoid leakage.",
        "Feature set F uses same-day click_count, unique_click_users, and click_per_user under a nowcasting-style assumption that those click aggregates are available at scoring time.",
        "When SQLite aggregation is not practical, click_to_order_metrics.csv is used as a smaller proxy source based on first-click dates and clicks_before_order.",
    ]
    print("\nAssumptions:")
    for item in assumptions:
        print(f"- {item}")


if __name__ == "__main__":
    main()
