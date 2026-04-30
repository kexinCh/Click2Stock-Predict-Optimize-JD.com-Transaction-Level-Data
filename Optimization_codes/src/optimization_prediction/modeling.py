from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import INNER_TRAIN_END, INNER_VALIDATION_START, TRAIN_END_DATE


AUTOREGRESSIVE_LAGS = [1, 2, 3, 7]
ROLLING_WINDOWS = [3, 7]


@dataclass(frozen=True)
class ModelCandidate:
    name: str
    family: str
    params: dict


@dataclass
class ModelingArtifacts:
    validation_summary: pd.DataFrame
    all_validation_predictions: pd.DataFrame
    all_test_predictions: pd.DataFrame
    all_cluster_predictions: pd.DataFrame
    allocation_tables: pd.DataFrame
    final_estimators: dict[str, Pipeline | None]
    best_model_id: str
    best_model_name: str
    best_approach: str
    best_test_predictions: pd.DataFrame
    best_cluster_predictions: pd.DataFrame
    best_allocation_table: pd.DataFrame


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _metric_frame(actual: pd.Series, predicted: pd.Series) -> dict:
    actual = actual.astype(float)
    predicted = predicted.astype(float)
    error = predicted - actual
    abs_error = error.abs()
    return {
        "mae": float(abs_error.mean()),
        "rmse": float(np.sqrt((error**2).mean())),
        "wape": float(abs_error.sum() / actual.sum()) if actual.sum() else float("nan"),
        "bias": float(error.sum() / actual.sum()) if actual.sum() else float("nan"),
        "underprediction_units": float((-error.clip(upper=0)).sum()),
        "overprediction_units": float(error.clip(lower=0).sum()),
        "actual_units": float(actual.sum()),
        "predicted_units": float(predicted.sum()),
    }


def _prepare_order_features(orders: pd.DataFrame) -> pd.DataFrame:
    enriched = orders.copy()
    enriched["discount_rate"] = (
        (
            enriched["original_unit_price"].astype(float)
            - enriched["final_unit_price"].astype(float)
        )
        / enriched["original_unit_price"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0)
    enriched["gift_flag"] = (
        pd.to_numeric(enriched["gift_item"], errors="coerce").fillna(0).gt(0).astype(float)
    )
    return enriched


def _add_group_history_features(
    frame: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    prefix: str,
) -> pd.DataFrame:
    output = frame.sort_values(group_cols + ["date"]).copy() if group_cols else frame.sort_values("date").copy()
    if group_cols:
        grouped = output.groupby(group_cols, sort=False)[value_col]
        for lag in AUTOREGRESSIVE_LAGS:
            output[f"{prefix}_lag_{lag}"] = grouped.shift(lag)
        for window in ROLLING_WINDOWS:
            output[f"{prefix}_roll_{window}"] = grouped.transform(
                lambda series, rolling_window=window: series.shift(1).rolling(rolling_window).mean()
            )
    else:
        series = output[value_col]
        for lag in AUTOREGRESSIVE_LAGS:
            output[f"{prefix}_lag_{lag}"] = series.shift(lag)
        for window in ROLLING_WINDOWS:
            output[f"{prefix}_roll_{window}"] = series.shift(1).rolling(window).mean()
    return output


def _add_target_history_features(
    frame: pd.DataFrame,
    group_cols: list[str],
    target_col: str,
) -> pd.DataFrame:
    output = frame.sort_values(group_cols + ["date"]).copy() if group_cols else frame.sort_values("date").copy()
    if group_cols:
        grouped = output.groupby(group_cols, sort=False)[target_col]
        for lag in AUTOREGRESSIVE_LAGS:
            output[f"lag_{lag}"] = grouped.shift(lag)
        for window in ROLLING_WINDOWS:
            output[f"rolling_mean_{window}"] = grouped.transform(
                lambda series, rolling_window=window: series.shift(1).rolling(rolling_window).mean()
            )
    else:
        series = output[target_col]
        for lag in AUTOREGRESSIVE_LAGS:
            output[f"lag_{lag}"] = series.shift(lag)
        for window in ROLLING_WINDOWS:
            output[f"rolling_mean_{window}"] = series.shift(1).rolling(window).mean()
    return output


def build_global_context(
    orders: pd.DataFrame,
    clicks: pd.DataFrame,
    all_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    orders_enriched = _prepare_order_features(orders)
    global_orders = (
        orders_enriched.groupby("order_date", as_index=False)
        .agg(
            global_discount=("discount_rate", "mean"),
            global_units=("quantity", "sum"),
            global_order_count=("order_ID", "nunique"),
            global_gift_rate=("gift_flag", "mean"),
        )
        .rename(columns={"order_date": "date"})
    )
    global_clicks = (
        clicks.assign(date=clicks["request_time"].dt.floor("D"))
        .groupby("date", as_index=False)
        .agg(
            global_clicks=("sku_ID", "size"),
            global_click_users=("user_ID", lambda series: series[series.ne("-")].nunique()),
        )
    )

    context = pd.DataFrame({"date": all_dates})
    context = context.merge(global_orders, on="date", how="left")
    context = context.merge(global_clicks, on="date", how="left")
    base_columns = [
        "global_discount",
        "global_units",
        "global_order_count",
        "global_gift_rate",
        "global_clicks",
        "global_click_users",
    ]
    for column in base_columns:
        context[column] = context[column].fillna(0)
        context = _add_group_history_features(context, [], column, column)
    return context


def build_warehouse_context(
    orders: pd.DataFrame,
    warehouse_universe: list[int],
    all_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    orders_enriched = _prepare_order_features(orders)
    warehouse_features = (
        orders_enriched.groupby(["order_date", "dc_des"], as_index=False)
        .agg(
            wh_discount=("discount_rate", "mean"),
            wh_units=("quantity", "sum"),
            wh_order_count=("order_ID", "nunique"),
            wh_gift_rate=("gift_flag", "mean"),
        )
        .rename(columns={"order_date": "date", "dc_des": "warehouse"})
    )

    full_index = pd.MultiIndex.from_product(
        [all_dates, warehouse_universe],
        names=["date", "warehouse"],
    ).to_frame(index=False)
    context = full_index.merge(warehouse_features, on=["date", "warehouse"], how="left")
    base_columns = ["wh_discount", "wh_units", "wh_order_count", "wh_gift_rate"]
    for column in base_columns:
        context[column] = context[column].fillna(0)
        context = _add_group_history_features(context, ["warehouse"], column, column)
    return context


def build_cluster_level_panel(full_panel: pd.DataFrame) -> pd.DataFrame:
    return (
        full_panel.groupby(["date", "sku_cluster_ID"], as_index=False)["demand"]
        .sum()
        .sort_values(["date", "sku_cluster_ID"])
    )


def build_training_matrix(
    base_panel: pd.DataFrame,
    group_cols: list[str],
    target_col: str,
    context_df: pd.DataFrame,
    context_keys: list[str],
) -> pd.DataFrame:
    features = _add_target_history_features(base_panel, group_cols, target_col)
    features = features.merge(context_df, on=context_keys, how="left")
    features["dow"] = features["date"].dt.day_name()
    features["is_weekend"] = features["date"].dt.dayofweek.isin([5, 6]).astype(int)
    return features


def _candidate_models() -> list[ModelCandidate]:
    return [
        ModelCandidate("lag_1", "baseline", {}),
        ModelCandidate("moving_average_3", "baseline", {}),
        ModelCandidate("moving_average_7", "baseline", {}),
        ModelCandidate("blend_50_50", "baseline", {}),
        ModelCandidate("linear_regression", "linear", {}),
        ModelCandidate("ridge_alpha_1.0", "linear", {"alpha": 1.0}),
        ModelCandidate("lasso_alpha_0.0100", "linear", {"alpha": 0.0100}),
        ModelCandidate(
            "random_forest",
            "tree",
            {"n_estimators": 40, "max_depth": 10, "min_samples_leaf": 4, "max_features": "sqrt", "max_samples": 0.5},
        ),
        ModelCandidate(
            "gradient_boosting",
            "tree",
            {"n_estimators": 60, "learning_rate": 0.05, "max_depth": 3, "min_samples_leaf": 20, "subsample": 0.9},
        ),
    ]


def _build_estimator(candidate: ModelCandidate, feature_columns: list[str]) -> Pipeline | None:
    if candidate.family == "baseline":
        return None

    categorical_cols = [column for column in feature_columns if column in {"warehouse", "sku_cluster_ID", "dow"}]
    numeric_cols = [column for column in feature_columns if column not in set(categorical_cols)]

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", _make_one_hot_encoder()),
        ]
    )

    if candidate.family == "linear":
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                ("scaler", StandardScaler()),
            ]
        )
        if candidate.name == "linear_regression":
            estimator = LinearRegression()
        elif candidate.name.startswith("ridge"):
            estimator = Ridge(alpha=candidate.params["alpha"])
        else:
            estimator = Lasso(alpha=candidate.params["alpha"], max_iter=20000)
    else:
        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="constant", fill_value=0))]
        )
        if candidate.name.startswith("random_forest"):
            estimator = RandomForestRegressor(
                n_estimators=candidate.params["n_estimators"],
                max_depth=candidate.params["max_depth"],
                min_samples_leaf=candidate.params["min_samples_leaf"],
                max_features=candidate.params["max_features"],
                max_samples=candidate.params["max_samples"],
                random_state=42,
                n_jobs=1,
            )
        else:
            estimator = GradientBoostingRegressor(
                n_estimators=candidate.params["n_estimators"],
                learning_rate=candidate.params["learning_rate"],
                max_depth=candidate.params["max_depth"],
                min_samples_leaf=candidate.params["min_samples_leaf"],
                subsample=candidate.params["subsample"],
                random_state=42,
            )

    return Pipeline(
        steps=[
            (
                "features",
                ColumnTransformer(
                    transformers=[
                        ("categorical", categorical_transformer, categorical_cols),
                        ("numeric", numeric_transformer, numeric_cols),
                    ],
                    remainder="drop",
                ),
            ),
            ("estimator", estimator),
        ]
    )


def _baseline_prediction(candidate: ModelCandidate, feature_frame: pd.DataFrame) -> np.ndarray:
    if candidate.name == "lag_1":
        return feature_frame["lag_1"].fillna(0).to_numpy()
    if candidate.name == "moving_average_3":
        return (
            feature_frame["rolling_mean_3"]
            .fillna(feature_frame["lag_1"])
            .fillna(0)
            .to_numpy()
        )
    if candidate.name == "moving_average_7":
        return (
            feature_frame["rolling_mean_7"]
            .fillna(feature_frame["rolling_mean_3"])
            .fillna(feature_frame["lag_1"])
            .fillna(0)
            .to_numpy()
        )
    if candidate.name == "blend_50_50":
        lag = feature_frame["lag_1"].fillna(0).to_numpy()
        ma3 = feature_frame["rolling_mean_3"].fillna(feature_frame["lag_1"]).fillna(0).to_numpy()
        return 0.5 * lag + 0.5 * ma3
    raise ValueError(f"Unknown baseline model: {candidate.name}")


def _current_feature_snapshot(
    history: pd.DataFrame,
    current_rows: pd.DataFrame,
    group_cols: list[str],
    target_col: str,
    context_df: pd.DataFrame,
    context_keys: list[str],
) -> pd.DataFrame:
    placeholder = current_rows[group_cols + ["date"]].copy() if group_cols else current_rows[["date"]].copy()
    placeholder[target_col] = np.nan
    history_columns = group_cols + ["date", target_col] if group_cols else ["date", target_col]
    temp = pd.concat([history[history_columns], placeholder], ignore_index=True)
    temp = _add_target_history_features(temp, group_cols, target_col)
    current = temp[temp[target_col].isna()].drop(columns=[target_col])
    current = current.merge(context_df, on=context_keys, how="left")
    current["dow"] = current["date"].dt.day_name()
    current["is_weekend"] = current["date"].dt.dayofweek.isin([5, 6]).astype(int)
    return current


def _recursive_forecast(
    candidate: ModelCandidate,
    estimator: Pipeline | None,
    train_history: pd.DataFrame,
    future_base: pd.DataFrame,
    group_cols: list[str],
    target_col: str,
    context_df: pd.DataFrame,
    context_keys: list[str],
    feature_columns: list[str],
) -> pd.DataFrame:
    history = train_history[group_cols + ["date", target_col]].copy() if group_cols else train_history[["date", target_col]].copy()
    predictions: list[pd.DataFrame] = []
    for current_date in sorted(pd.to_datetime(future_base["date"].unique())):
        current_rows = future_base[future_base["date"] == current_date].copy()
        feature_frame = _current_feature_snapshot(
            history=history,
            current_rows=current_rows,
            group_cols=group_cols,
            target_col=target_col,
            context_df=context_df,
            context_keys=context_keys,
        )
        if estimator is None:
            prediction = _baseline_prediction(candidate, feature_frame)
        else:
            prediction = estimator.predict(feature_frame[feature_columns])
        prediction = np.clip(prediction, 0, None)
        out = current_rows.copy()
        out["predicted_demand"] = prediction
        predictions.append(out)

        history_add = out[group_cols + ["date"]].copy() if group_cols else out[["date"]].copy()
        history_add[target_col] = prediction
        history = pd.concat([history, history_add], ignore_index=True)
    return pd.concat(predictions, ignore_index=True)


def estimate_empirical_allocation(train_panel: pd.DataFrame) -> pd.DataFrame:
    allocation = train_panel.copy()
    cluster_totals = (
        allocation.groupby(["date", "sku_cluster_ID"], as_index=False)["demand"]
        .sum()
        .rename(columns={"demand": "cluster_total"})
    )
    allocation = allocation.merge(cluster_totals, on=["date", "sku_cluster_ID"], how="left")
    allocation = allocation[allocation["cluster_total"] > 0].copy()
    allocation["dow"] = allocation["date"].dt.day_name()
    allocation["share"] = allocation["demand"] / allocation["cluster_total"]

    dow_share = (
        allocation.groupby(["sku_cluster_ID", "warehouse", "dow"], as_index=False)["share"]
        .mean()
        .rename(columns={"share": "share_dow"})
    )
    overall_share = (
        allocation.groupby(["sku_cluster_ID", "warehouse"], as_index=False)["share"]
        .mean()
        .rename(columns={"share": "share_overall"})
    )
    return dow_share.merge(overall_share, on=["sku_cluster_ID", "warehouse"], how="outer")


def allocate_cluster_predictions(
    cluster_predictions: pd.DataFrame,
    allocation_table: pd.DataFrame,
    warehouse_universe: list[int],
    cluster_universe: list[int],
) -> pd.DataFrame:
    base = pd.MultiIndex.from_product(
        [sorted(pd.to_datetime(cluster_predictions["date"].unique())), warehouse_universe, cluster_universe],
        names=["date", "warehouse", "sku_cluster_ID"],
    ).to_frame(index=False)
    base["dow"] = base["date"].dt.day_name()
    allocated = base.merge(allocation_table, on=["sku_cluster_ID", "warehouse", "dow"], how="left")
    allocated = allocated.merge(
        allocation_table[["sku_cluster_ID", "warehouse", "share_overall"]].drop_duplicates(),
        on=["sku_cluster_ID", "warehouse"],
        how="left",
        suffixes=("", "_fallback"),
    )
    allocated["share"] = allocated["share_dow"].fillna(allocated["share_overall"]).fillna(0)
    allocated["share_sum"] = allocated.groupby(["date", "sku_cluster_ID"])["share"].transform("sum")
    equal_share = 1.0 / len(warehouse_universe) if warehouse_universe else 0.0
    allocated["share"] = np.where(
        allocated["share_sum"] > 0,
        allocated["share"] / allocated["share_sum"],
        equal_share,
    )
    allocated = allocated.merge(
        cluster_predictions[["date", "sku_cluster_ID", "predicted_demand"]].rename(
            columns={"predicted_demand": "predicted_cluster_demand"}
        ),
        on=["date", "sku_cluster_ID"],
        how="left",
    )
    allocated["predicted_demand"] = allocated["predicted_cluster_demand"].fillna(0) * allocated["share"]
    return allocated[["date", "warehouse", "sku_cluster_ID", "predicted_demand"]].sort_values(
        ["date", "warehouse", "sku_cluster_ID"]
    )


def _train_final_estimator(
    candidate: ModelCandidate,
    feature_frame: pd.DataFrame,
    target_col: str,
    feature_columns: list[str],
) -> Pipeline | None:
    estimator = _build_estimator(candidate, feature_columns)
    if estimator is None:
        return None
    estimator.fit(feature_frame[feature_columns], feature_frame[target_col])
    return estimator


def _direct_feature_columns() -> list[str]:
    return [
        "warehouse",
        "sku_cluster_ID",
        "dow",
        "is_weekend",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_7",
        "rolling_mean_3",
        "rolling_mean_7",
        "wh_discount_lag_1",
        "wh_discount_roll_3",
        "wh_discount_roll_7",
        "wh_units_lag_1",
        "wh_units_roll_3",
        "wh_units_roll_7",
        "wh_order_count_lag_1",
        "wh_order_count_roll_7",
        "wh_gift_rate_lag_1",
        "wh_gift_rate_roll_7",
        "global_discount_lag_1",
        "global_discount_roll_7",
        "global_units_lag_1",
        "global_units_roll_7",
        "global_order_count_lag_1",
        "global_order_count_roll_7",
        "global_clicks_lag_1",
        "global_clicks_roll_7",
        "global_click_users_lag_1",
        "global_click_users_roll_7",
        "global_gift_rate_lag_1",
        "global_gift_rate_roll_7",
    ]


def _cluster_feature_columns() -> list[str]:
    return [
        "sku_cluster_ID",
        "dow",
        "is_weekend",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_7",
        "rolling_mean_3",
        "rolling_mean_7",
        "global_discount_lag_1",
        "global_discount_roll_7",
        "global_units_lag_1",
        "global_units_roll_7",
        "global_order_count_lag_1",
        "global_order_count_roll_7",
        "global_clicks_lag_1",
        "global_clicks_roll_7",
        "global_click_users_lag_1",
        "global_click_users_roll_7",
        "global_gift_rate_lag_1",
        "global_gift_rate_roll_7",
    ]


def run_modeling_experiments(
    train_panel: pd.DataFrame,
    test_panel: pd.DataFrame,
    orders: pd.DataFrame,
    clicks: pd.DataFrame,
    warehouse_universe: list[int],
    cluster_universe: list[int],
) -> ModelingArtifacts:
    all_dates = pd.date_range(train_panel["date"].min(), test_panel["date"].max(), freq="D")
    global_context = build_global_context(orders, clicks, all_dates)
    warehouse_context = build_warehouse_context(orders, warehouse_universe, all_dates)
    direct_context = warehouse_context.merge(global_context, on="date", how="left")
    full_panel = pd.concat([train_panel, test_panel], ignore_index=True).sort_values(
        ["date", "warehouse", "sku_cluster_ID"]
    )

    validation_rows: list[dict] = []
    all_validation_predictions: list[pd.DataFrame] = []
    all_test_predictions: list[pd.DataFrame] = []
    all_cluster_predictions: list[pd.DataFrame] = []
    final_estimators: dict[str, Pipeline | None] = {}

    direct_feature_frame = build_training_matrix(
        base_panel=full_panel,
        group_cols=["warehouse", "sku_cluster_ID"],
        target_col="demand",
        context_df=direct_context,
        context_keys=["date", "warehouse"],
    )
    direct_subtrain = direct_feature_frame[direct_feature_frame["date"] <= INNER_TRAIN_END].copy()
    direct_full_train = direct_feature_frame[direct_feature_frame["date"] <= TRAIN_END_DATE].copy()
    direct_validation_base = train_panel[
        (train_panel["date"] >= INNER_VALIDATION_START) & (train_panel["date"] <= TRAIN_END_DATE)
    ][["date", "warehouse", "sku_cluster_ID"]].copy()
    direct_validation_actual = train_panel[
        (train_panel["date"] >= INNER_VALIDATION_START) & (train_panel["date"] <= TRAIN_END_DATE)
    ][["date", "warehouse", "sku_cluster_ID", "demand"]].copy()
    direct_test_base = test_panel[["date", "warehouse", "sku_cluster_ID"]].copy()
    direct_features = _direct_feature_columns()

    for candidate in _candidate_models():
        model_id = f"direct_warehouse__{candidate.name}"
        validation_estimator = _train_final_estimator(candidate, direct_subtrain, "demand", direct_features)
        direct_validation_columns = list(dict.fromkeys(["date", "warehouse", "sku_cluster_ID", "demand"] + direct_features))
        validation_feature_rows = direct_feature_frame[
            (direct_feature_frame["date"] >= INNER_VALIDATION_START)
            & (direct_feature_frame["date"] <= TRAIN_END_DATE)
        ][direct_validation_columns].copy()
        validation_predictions = validation_feature_rows[["date", "warehouse", "sku_cluster_ID"]].copy()
        if validation_estimator is None:
            validation_predictions["predicted_demand"] = _baseline_prediction(candidate, validation_feature_rows)
        else:
            validation_predictions["predicted_demand"] = np.clip(
                validation_estimator.predict(validation_feature_rows[direct_features]),
                0,
                None,
            )
        validation_merged = direct_validation_actual.merge(
            validation_predictions,
            on=["date", "warehouse", "sku_cluster_ID"],
            how="left",
        )
        validation_merged["predicted_demand"] = validation_merged["predicted_demand"].fillna(0)
        validation_predictions["model_id"] = model_id
        validation_predictions["approach"] = "direct_warehouse"
        validation_predictions["prediction_level"] = "direct warehouse-cluster"
        validation_predictions["model_name"] = candidate.name
        validation_predictions["family"] = candidate.family
        all_validation_predictions.append(validation_predictions)
        validation_rows.append(
            {
                "model_id": model_id,
                "approach": "direct_warehouse",
                "prediction_level": "direct warehouse-cluster",
                "model_name": candidate.name,
                "family": candidate.family,
                **_metric_frame(validation_merged["demand"], validation_merged["predicted_demand"]),
            }
        )

        test_estimator = _train_final_estimator(candidate, direct_full_train, "demand", direct_features)
        final_estimators[model_id] = test_estimator
        direct_test_columns = list(dict.fromkeys(["date", "warehouse", "sku_cluster_ID", "demand"] + direct_features))
        test_feature_rows = direct_feature_frame[direct_feature_frame["date"] >= TRAIN_END_DATE + pd.Timedelta(days=1)][
            direct_test_columns
        ].copy()
        test_predictions = test_feature_rows[["date", "warehouse", "sku_cluster_ID"]].copy()
        if test_estimator is None:
            test_predictions["predicted_demand"] = _baseline_prediction(candidate, test_feature_rows)
        else:
            test_predictions["predicted_demand"] = np.clip(
                test_estimator.predict(test_feature_rows[direct_features]),
                0,
                None,
            )
        test_predictions["model_id"] = model_id
        test_predictions["approach"] = "direct_warehouse"
        test_predictions["prediction_level"] = "direct warehouse-cluster"
        test_predictions["model_name"] = candidate.name
        test_predictions["family"] = candidate.family
        all_test_predictions.append(test_predictions)

    cluster_train = build_cluster_level_panel(train_panel)
    cluster_test = build_cluster_level_panel(test_panel)
    cluster_full = build_cluster_level_panel(full_panel)
    cluster_feature_frame = build_training_matrix(
        base_panel=cluster_full,
        group_cols=["sku_cluster_ID"],
        target_col="demand",
        context_df=global_context,
        context_keys=["date"],
    )
    cluster_subtrain = cluster_feature_frame[cluster_feature_frame["date"] <= INNER_TRAIN_END].copy()
    cluster_full_train = cluster_feature_frame[cluster_feature_frame["date"] <= TRAIN_END_DATE].copy()
    cluster_validation_base = cluster_train[
        (cluster_train["date"] >= INNER_VALIDATION_START) & (cluster_train["date"] <= TRAIN_END_DATE)
    ][["date", "sku_cluster_ID"]].copy()
    cluster_test_base = cluster_test[["date", "sku_cluster_ID"]].copy()
    validation_allocation = estimate_empirical_allocation(train_panel[train_panel["date"] <= INNER_TRAIN_END].copy())
    final_allocation = estimate_empirical_allocation(train_panel.copy())
    cluster_features = _cluster_feature_columns()

    for candidate in _candidate_models():
        model_id = f"cluster_to_warehouse__{candidate.name}"
        validation_estimator = _train_final_estimator(candidate, cluster_subtrain, "demand", cluster_features)
        cluster_validation_columns = list(dict.fromkeys(["date", "sku_cluster_ID", "demand"] + cluster_features))
        validation_cluster_rows = cluster_feature_frame[
            (cluster_feature_frame["date"] >= INNER_VALIDATION_START)
            & (cluster_feature_frame["date"] <= TRAIN_END_DATE)
        ][cluster_validation_columns].copy()
        validation_cluster_predictions = validation_cluster_rows[["date", "sku_cluster_ID"]].copy()
        if validation_estimator is None:
            validation_cluster_predictions["predicted_demand"] = _baseline_prediction(candidate, validation_cluster_rows)
        else:
            validation_cluster_predictions["predicted_demand"] = np.clip(
                validation_estimator.predict(validation_cluster_rows[cluster_features]),
                0,
                None,
            )
        validation_pair_predictions = allocate_cluster_predictions(
            cluster_predictions=validation_cluster_predictions,
            allocation_table=validation_allocation,
            warehouse_universe=warehouse_universe,
            cluster_universe=cluster_universe,
        )
        validation_merged = direct_validation_actual.merge(
            validation_pair_predictions,
            on=["date", "warehouse", "sku_cluster_ID"],
            how="left",
        )
        validation_merged["predicted_demand"] = validation_merged["predicted_demand"].fillna(0)
        validation_pair_predictions["model_id"] = model_id
        validation_pair_predictions["approach"] = "cluster_to_warehouse"
        validation_pair_predictions["prediction_level"] = "cluster total then warehouse allocation"
        validation_pair_predictions["model_name"] = candidate.name
        validation_pair_predictions["family"] = candidate.family
        all_validation_predictions.append(validation_pair_predictions)
        validation_rows.append(
            {
                "model_id": model_id,
                "approach": "cluster_to_warehouse",
                "prediction_level": "cluster total then warehouse allocation",
                "model_name": candidate.name,
                "family": candidate.family,
                **_metric_frame(validation_merged["demand"], validation_merged["predicted_demand"]),
            }
        )

        test_estimator = _train_final_estimator(candidate, cluster_full_train, "demand", cluster_features)
        final_estimators[model_id] = test_estimator
        cluster_test_columns = list(dict.fromkeys(["date", "sku_cluster_ID", "demand"] + cluster_features))
        test_cluster_rows = cluster_feature_frame[cluster_feature_frame["date"] >= TRAIN_END_DATE + pd.Timedelta(days=1)][
            cluster_test_columns
        ].copy()
        test_cluster_predictions = test_cluster_rows[["date", "sku_cluster_ID"]].copy()
        if test_estimator is None:
            test_cluster_predictions["predicted_demand"] = _baseline_prediction(candidate, test_cluster_rows)
        else:
            test_cluster_predictions["predicted_demand"] = np.clip(
                test_estimator.predict(test_cluster_rows[cluster_features]),
                0,
                None,
            )
        test_cluster_predictions["model_id"] = model_id
        test_cluster_predictions["approach"] = "cluster_to_warehouse"
        test_cluster_predictions["prediction_level"] = "cluster total then warehouse allocation"
        test_cluster_predictions["model_name"] = candidate.name
        test_cluster_predictions["family"] = candidate.family
        all_cluster_predictions.append(test_cluster_predictions)

        test_pair_predictions = allocate_cluster_predictions(
            cluster_predictions=test_cluster_predictions[["date", "sku_cluster_ID", "predicted_demand"]],
            allocation_table=final_allocation,
            warehouse_universe=warehouse_universe,
            cluster_universe=cluster_universe,
        )
        test_pair_predictions["model_id"] = model_id
        test_pair_predictions["approach"] = "cluster_to_warehouse"
        test_pair_predictions["prediction_level"] = "cluster total then warehouse allocation"
        test_pair_predictions["model_name"] = candidate.name
        test_pair_predictions["family"] = candidate.family
        all_test_predictions.append(test_pair_predictions)

    validation_summary = pd.DataFrame(validation_rows).sort_values(
        ["wape", "rmse", "mae", "bias"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)
    best_row = validation_summary.iloc[0]
    best_model_id = str(best_row["model_id"])

    all_test_predictions_df = pd.concat(all_test_predictions, ignore_index=True)
    all_cluster_predictions_df = (
        pd.concat(all_cluster_predictions, ignore_index=True) if all_cluster_predictions else pd.DataFrame()
    )

    return ModelingArtifacts(
        validation_summary=validation_summary,
        all_validation_predictions=pd.concat(all_validation_predictions, ignore_index=True),
        all_test_predictions=all_test_predictions_df,
        all_cluster_predictions=all_cluster_predictions_df,
        allocation_tables=final_allocation.assign(
            model_id="cluster_to_warehouse__shared_allocation",
            approach="cluster_to_warehouse",
        ),
        final_estimators=final_estimators,
        best_model_id=best_model_id,
        best_model_name=str(best_row["model_name"]),
        best_approach=str(best_row["approach"]),
        best_test_predictions=all_test_predictions_df[all_test_predictions_df["model_id"] == best_model_id].copy(),
        best_cluster_predictions=(
            all_cluster_predictions_df[all_cluster_predictions_df["model_id"] == best_model_id].copy()
            if not all_cluster_predictions_df.empty
            else pd.DataFrame()
        ),
        best_allocation_table=final_allocation.copy(),
    )


def extract_feature_importance(
    model_id: str,
    modeling_artifacts: ModelingArtifacts,
    top_n: int = 15,
) -> pd.DataFrame:
    estimator = modeling_artifacts.final_estimators.get(model_id)
    if estimator is None:
        if model_id.endswith("blend_50_50"):
            return pd.DataFrame(
                [
                    {"feature": "lag_1", "importance": 0.5, "importance_type": "rule_weight"},
                    {"feature": "rolling_mean_3", "importance": 0.5, "importance_type": "rule_weight"},
                ]
            )
        if model_id.endswith("moving_average_3"):
            return pd.DataFrame(
                [{"feature": "rolling_mean_3", "importance": 1.0, "importance_type": "rule_weight"}]
            )
        if model_id.endswith("moving_average_7"):
            return pd.DataFrame(
                [{"feature": "rolling_mean_7", "importance": 1.0, "importance_type": "rule_weight"}]
            )
        if model_id.endswith("lag_1"):
            return pd.DataFrame(
                [{"feature": "lag_1", "importance": 1.0, "importance_type": "rule_weight"}]
            )
        return pd.DataFrame(columns=["feature", "importance", "importance_type"])

    feature_transformer = estimator.named_steps["features"]
    feature_names = feature_transformer.get_feature_names_out()
    fitted_model = estimator.named_steps["estimator"]

    if hasattr(fitted_model, "feature_importances_"):
        values = fitted_model.feature_importances_
        importance_type = "feature_importance"
    elif hasattr(fitted_model, "coef_"):
        values = np.abs(np.ravel(fitted_model.coef_))
        importance_type = "abs_coefficient"
    else:
        return pd.DataFrame(columns=["feature", "importance", "importance_type"])

    feature_importance = pd.DataFrame(
        {"feature": feature_names, "importance": values, "importance_type": importance_type}
    )
    feature_importance = feature_importance.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
    return feature_importance
