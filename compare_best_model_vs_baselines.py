#!/usr/bin/env python3
"""Compare the winning XGBoost setup against naive demand baselines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import build_compare_demand_models as demand_models
from project_paths import BASELINE_OUTPUTS_DIR, PROCESSED_DATA_DIR, ensure_dir

SEED = 42


def create_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Creating baseline comparison features...")
    out = demand_models.create_demand_history_features(df)
    series_key = [demand_models.CLUSTER_COL, demand_models.WAREHOUSE_COL]
    out["rolling_mean_7_demand"] = (
        out.groupby(series_key, sort=False)[demand_models.TARGET_COL]
        .transform(lambda s: s.shift(1).rolling(window=7, min_periods=7).mean())
    )
    return out


def prepare_model_frame(base_dir: Path) -> pd.DataFrame:
    df, _ = demand_models.load_dataset(base_dir)
    df = create_baseline_features(df)
    print("Dropping rows without baseline history...")
    df = df.dropna(subset=["lag_1_demand", "lag_2_demand", "lag_3_demand", "rolling_mean_7_demand"]).copy()
    print(f"Retained {len(df):,} rows for baseline comparison.")
    return df


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


def build_xgboost_pipeline() -> Pipeline:
    from xgboost import XGBRegressor

    feature_cols = [
        demand_models.CLUSTER_COL,
        demand_models.WAREHOUSE_COL,
        "lag_1_demand",
        "lag_2_demand",
        "lag_3_demand",
    ]
    numeric_cols = ["lag_1_demand", "lag_2_demand", "lag_3_demand"]
    categorical_cols = [demand_models.CLUSTER_COL, demand_models.WAREHOUSE_COL]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    model = XGBRegressor(
        random_state=SEED,
        n_estimators=150,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    ), feature_cols


def compare_models(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Training XGBoost feature set B and scoring baselines...")
    train_df = df[df[demand_models.SPLIT_COL] == "train"].copy()
    test_df = df[df[demand_models.SPLIT_COL] == "test"].copy()

    pipeline, feature_cols = build_xgboost_pipeline()
    pipeline.fit(train_df[feature_cols], train_df[demand_models.TARGET_COL])
    xgb_pred = pipeline.predict(test_df[feature_cols])

    experiments = {
        "XGBoost_FeatureSet_B": xgb_pred,
        "Baseline_YesterdayDemand": test_df["lag_1_demand"].to_numpy(),
        "Baseline_Past7DayAverage": test_df["rolling_mean_7_demand"].to_numpy(),
    }

    rows: list[dict[str, float | str]] = []
    prediction_frames: list[pd.DataFrame] = []
    y_test = test_df[demand_models.TARGET_COL]
    for model_name, y_pred in experiments.items():
        metrics = evaluate_predictions(y_test, y_pred)
        rows.append({"model_name": model_name, **metrics})
        prediction_frames.append(
            pd.DataFrame(
                {
                    demand_models.DATE_COL: test_df[demand_models.DATE_COL],
                    demand_models.CLUSTER_COL: test_df[demand_models.CLUSTER_COL],
                    demand_models.WAREHOUSE_COL: test_df[demand_models.WAREHOUSE_COL],
                    "actual_demand": y_test,
                    "predicted_demand": y_pred,
                    "model_name": model_name,
                }
            )
        )

    results_df = pd.DataFrame(rows).sort_values(["WAPE", "RMSE", "MAE"]).reset_index(drop=True)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    return results_df, predictions_df


def save_outputs(results_df: pd.DataFrame, predictions_df: pd.DataFrame, base_dir: Path) -> None:
    print("Saving baseline comparison outputs...")
    output_dir = ensure_dir(base_dir)
    results_df.to_csv(output_dir / "best_model_vs_baselines.csv", index=False)
    predictions_df.to_csv(output_dir / "best_model_vs_baselines_predictions.csv", index=False)


def main() -> None:
    data_dir = PROCESSED_DATA_DIR
    df = prepare_model_frame(data_dir)
    results_df, predictions_df = compare_models(df)
    save_outputs(results_df, predictions_df, BASELINE_OUTPUTS_DIR)

    print("\nBest-model vs baseline summary:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
