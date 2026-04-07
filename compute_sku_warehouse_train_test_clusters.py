#!/usr/bin/env python3
"""Train SKU clusters on the first 24 days and evaluate them on the last 7 days by destination warehouse."""

import argparse
import os
import sqlite3
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from project_paths import DATABASE_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, resolve_path

EPS = 1e-6
SPARSITY_THRESHOLD = 0.70
CV_THRESHOLD = 0.10
DEFAULT_TRAIN_DAYS = 24
DEFAULT_TEST_DAYS = 7
REQUIRED_ORDER_COLS = {
    "sku_ID",
    "order_date",
    "quantity",
    "final_unit_price",
    "original_unit_price",
    "direct_discount_per_unit",
    "quantity_discount_per_unit",
    "bundle_discount_per_unit",
    "coupon_discount_per_unit",
    "gift_item",
    "dc_des",
}


def sqlite_table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cur.fetchall()}


def load_orders(db_path: str, csv_path: str) -> tuple[pd.DataFrame, str]:
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "orders" in tables:
                cols = sqlite_table_columns(conn, "orders")
                if REQUIRED_ORDER_COLS.issubset(cols):
                    query = """
                        SELECT
                            sku_ID, order_date, quantity,
                            final_unit_price, original_unit_price,
                            direct_discount_per_unit, quantity_discount_per_unit,
                            bundle_discount_per_unit, coupon_discount_per_unit,
                            gift_item, dc_des
                        FROM orders
                    """
                    orders = pd.read_sql_query(query, conn)
                    conn.close()
                    return orders, "sqlite:orders"
            conn.close()
        except Exception:
            pass

    orders = pd.read_csv(csv_path)
    missing = REQUIRED_ORDER_COLS - set(orders.columns)
    if missing:
        raise ValueError(f"Order source missing required columns: {sorted(missing)}")
    return orders, f"csv:{csv_path}"


def load_clicks_daily(db_path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(db_path):
        return None

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if "clicks" not in tables:
            conn.close()
            return None

        cols = sqlite_table_columns(conn, "clicks")
        if not {"sku_ID", "request_time"}.issubset(cols):
            conn.close()
            return None

        query = """
            SELECT
                sku_ID,
                date(request_time) AS order_date,
                COUNT(*) AS daily_clicks
            FROM clicks
            GROUP BY sku_ID, date(request_time)
        """
        clicks = pd.read_sql_query(query, conn)
        conn.close()
        return clicks
    except Exception as exc:
        print(f"warning: unable to load clicks from sqlite ({exc}); skipping click features")
        return None


def safe_numeric_fill(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def normalize_orders(orders: pd.DataFrame) -> pd.DataFrame:
    df = orders.copy()
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce").dt.normalize()
    df = df[df["order_date"].notna()].copy()

    numeric_cols = [
        "quantity",
        "final_unit_price",
        "original_unit_price",
        "direct_discount_per_unit",
        "quantity_discount_per_unit",
        "bundle_discount_per_unit",
        "coupon_discount_per_unit",
    ]
    df = safe_numeric_fill(df, numeric_cols)
    df[numeric_cols] = df[numeric_cols].fillna(0.0)
    df["quantity"] = df["quantity"].clip(lower=0.0)

    df["gift_item"] = df["gift_item"].astype(str)
    df.loc[df["gift_item"].isin(["nan", "None", "NaN"]), "gift_item"] = ""
    df["dc_des"] = df["dc_des"].astype(str)

    df["total_discount_per_unit"] = (
        df["direct_discount_per_unit"]
        + df["quantity_discount_per_unit"]
        + df["bundle_discount_per_unit"]
        + df["coupon_discount_per_unit"]
    )
    df["promo_line_flag"] = (
        (df["total_discount_per_unit"] > 0)
        | (df["gift_item"].str.strip() != "")
    ).astype(int)
    return df


def split_orders_train_test(
    orders: pd.DataFrame,
    train_days: int,
    test_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex, pd.DatetimeIndex, dict]:
    unique_dates = pd.DatetimeIndex(sorted(orders["order_date"].drop_duplicates()))
    if len(unique_dates) < train_days + test_days:
        raise ValueError(
            "Not enough distinct order dates for the requested split: "
            f"{len(unique_dates)} available, need at least {train_days + test_days}"
        )

    train_dates = unique_dates[:train_days]
    test_dates = unique_dates[-test_days:]
    train_orders = orders[orders["order_date"].isin(train_dates)].copy()
    test_orders = orders[orders["order_date"].isin(test_dates)].copy()

    meta = {
        "train_start": train_dates.min().date(),
        "train_end": train_dates.max().date(),
        "train_days": int(len(train_dates)),
        "test_start": test_dates.min().date(),
        "test_end": test_dates.max().date(),
        "test_days": int(len(test_dates)),
        "unused_days": int(len(unique_dates) - len(train_dates) - len(test_dates)),
        "train_rows": int(len(train_orders)),
        "test_rows": int(len(test_orders)),
    }
    return train_orders, test_orders, train_dates, test_dates, meta


def filter_clicks_to_dates(
    clicks_daily: Optional[pd.DataFrame],
    date_index: pd.DatetimeIndex,
) -> Optional[pd.DataFrame]:
    if clicks_daily is None or clicks_daily.empty:
        return None

    clicks = clicks_daily.copy()
    clicks["order_date"] = pd.to_datetime(clicks["order_date"], errors="coerce").dt.normalize()
    clicks = clicks[clicks["order_date"].isin(date_index)].copy()
    if clicks.empty:
        return None
    return clicks


def build_sku_features(
    orders: pd.DataFrame,
    all_dates: pd.DatetimeIndex,
    clicks_daily: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if orders.empty:
        raise ValueError("Cannot build features from an empty order frame.")

    sku_ids = orders["sku_ID"].drop_duplicates().sort_values().reset_index(drop=True)
    full_grid = pd.MultiIndex.from_product(
        [sku_ids, all_dates],
        names=["sku_ID", "order_date"],
    ).to_frame(index=False)

    daily_sku = (
        orders.groupby(["sku_ID", "order_date"], as_index=False)
        .agg(
            daily_demand=("quantity", "sum"),
            avg_final_price=("final_unit_price", "mean"),
            avg_discount=("total_discount_per_unit", "mean"),
            promo_flag=("promo_line_flag", "max"),
            daily_warehouses=("dc_des", "nunique"),
        )
    )

    demand_panel = full_grid.merge(daily_sku, on=["sku_ID", "order_date"], how="left")
    demand_panel["daily_demand"] = demand_panel["daily_demand"].fillna(0.0)
    demand_panel["promo_flag"] = demand_panel["promo_flag"].fillna(0.0)
    demand_panel["daily_warehouses"] = demand_panel["daily_warehouses"].fillna(0.0)

    demand_feats = (
        demand_panel.groupby("sku_ID", as_index=False)
        .agg(
            mean_daily_demand=("daily_demand", "mean"),
            std_daily_demand=("daily_demand", "std"),
            zero_rate=("daily_demand", lambda x: float((x == 0).mean())),
            mean_daily_warehouses=("daily_warehouses", "mean"),
        )
    )
    demand_feats["std_daily_demand"] = demand_feats["std_daily_demand"].fillna(0.0)
    demand_feats["cv_daily_demand"] = (
        demand_feats["std_daily_demand"]
        / demand_feats["mean_daily_demand"].replace(0, np.nan)
    ).fillna(0.0)

    active_only = orders.groupby("sku_ID", as_index=False).agg(
        mean_price=("final_unit_price", "mean"),
        std_price=("final_unit_price", "std"),
        mean_discount=("total_discount_per_unit", "mean"),
        promo_rate_active=("promo_line_flag", "mean"),
        warehouse_count=("dc_des", "nunique"),
    )
    active_only["std_price"] = active_only["std_price"].fillna(0.0)

    feats = demand_feats.merge(active_only, on="sku_ID", how="left")

    if clicks_daily is not None and not clicks_daily.empty:
        click_panel = full_grid.merge(
            clicks_daily[["sku_ID", "order_date", "daily_clicks"]],
            on=["sku_ID", "order_date"],
            how="left",
        )
        click_panel["daily_clicks"] = pd.to_numeric(
            click_panel["daily_clicks"], errors="coerce"
        ).fillna(0.0)
        click_feats = click_panel.groupby("sku_ID", as_index=False).agg(
            mean_daily_clicks=("daily_clicks", "mean"),
            zero_click_rate=("daily_clicks", lambda x: float((x == 0).mean())),
        )
        feats = feats.merge(click_feats, on="sku_ID", how="left")

    numeric_cols = [col for col in feats.columns if col != "sku_ID"]
    feats[numeric_cols] = feats[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return feats


def scale_train_features(
    feats: pd.DataFrame,
) -> tuple[pd.DataFrame, StandardScaler, np.ndarray, list[str]]:
    out = feats.copy()
    feature_cols = [col for col in out.columns if col != "sku_ID"]
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(out[feature_cols])
    return out, scaler, x_scaled, feature_cols


def transform_features(
    feats: pd.DataFrame,
    scaler: StandardScaler,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    out = feats.copy()
    for col in feature_cols:
        if col not in out.columns:
            out[col] = 0.0
    out = out[["sku_ID", *feature_cols]].copy()
    out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x_scaled = scaler.transform(out[feature_cols])
    return out, x_scaled


def compute_demand_grid(
    orders: pd.DataFrame,
    assignments: pd.DataFrame,
    all_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    df = orders.merge(assignments[["sku_ID", "sku_cluster_ID"]], on="sku_ID", how="left")
    df = df.dropna(subset=["sku_cluster_ID"]).copy()
    df["sku_cluster_ID"] = df["sku_cluster_ID"].astype(int)

    demand = (
        df.groupby(["order_date", "dc_des", "sku_cluster_ID"], as_index=False)["quantity"]
        .sum()
        .rename(columns={"quantity": "demand"})
    )

    warehouses = sorted(df["dc_des"].dropna().astype(str).unique())
    clusters = sorted(df["sku_cluster_ID"].unique())
    full_index = pd.MultiIndex.from_product(
        [all_dates, warehouses, clusters],
        names=["order_date", "dc_des", "sku_cluster_ID"],
    )

    demand_full = (
        demand.set_index(["order_date", "dc_des", "sku_cluster_ID"])
        .reindex(full_index, fill_value=0.0)
        .reset_index()
    )
    return demand_full


def summarize_demand_grid(demand_full: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    grp = demand_full.groupby(["sku_cluster_ID", "dc_des"])["demand"]
    metrics = grp.agg(
        total_days="size",
        zero_days=lambda x: int((x == 0).sum()),
        mean_demand="mean",
        std_demand="std",
    ).reset_index()
    metrics["std_demand"] = metrics["std_demand"].fillna(0.0)
    metrics["zero_rate"] = metrics["zero_days"] / metrics["total_days"]
    metrics["cv"] = (metrics["std_demand"] / metrics["mean_demand"].replace(0, np.nan)).fillna(0.0)
    metrics["high_sparsity"] = metrics["zero_rate"] >= SPARSITY_THRESHOLD
    metrics["low_variance"] = (metrics["std_demand"] == 0.0) | (metrics["cv"] < CV_THRESHOLD)

    cluster_summary = (
        metrics.groupby("sku_cluster_ID", as_index=False)
        .agg(
            warehouses=("dc_des", "nunique"),
            series=("dc_des", "size"),
            high_sparsity_series=("high_sparsity", "sum"),
            low_variance_series=("low_variance", "sum"),
            avg_zero_rate=("zero_rate", "mean"),
            avg_cv=("cv", "mean"),
        )
    )
    cluster_summary["high_sparsity_share"] = (
        cluster_summary["high_sparsity_series"] / cluster_summary["series"]
    )
    cluster_summary["low_variance_share"] = (
        cluster_summary["low_variance_series"] / cluster_summary["series"]
    )

    overall = {
        "clusters": int(cluster_summary["sku_cluster_ID"].nunique()),
        "avg_zero_rate": float(cluster_summary["avg_zero_rate"].mean()),
        "avg_cv": float(cluster_summary["avg_cv"].mean()),
        "avg_high_sparsity_share": float(cluster_summary["high_sparsity_share"].mean()),
        "avg_low_variance_share": float(cluster_summary["low_variance_share"].mean()),
        "clusters_high_sparsity_majority": int((cluster_summary["high_sparsity_share"] >= 0.5).sum()),
        "clusters_low_variance_majority": int((cluster_summary["low_variance_share"] >= 0.5).sum()),
    }
    overall["objective"] = (
        overall["avg_high_sparsity_share"] + overall["avg_low_variance_share"]
    )
    return cluster_summary, overall


def evaluate_candidate_k(
    x_scaled: np.ndarray,
    sku_ids: pd.Series,
    train_orders: pd.DataFrame,
    train_dates: pd.DatetimeIndex,
    k: int,
) -> tuple[KMeans, pd.DataFrame, dict]:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(x_scaled)
    assignments = pd.DataFrame(
        {"sku_ID": sku_ids.to_numpy(), "sku_cluster_ID": labels.astype(int)}
    )
    demand_full = compute_demand_grid(train_orders, assignments, train_dates)
    _, overall = summarize_demand_grid(demand_full)
    overall["k"] = int(k)
    return model, assignments, overall


def choose_best_k(
    train_feats_scaled: np.ndarray,
    train_feats: pd.DataFrame,
    train_orders: pd.DataFrame,
    train_dates: pd.DatetimeIndex,
    k_min: int,
    k_max: int,
) -> tuple[KMeans, pd.DataFrame, dict]:
    n_skus = train_feats.shape[0]
    if n_skus < 3:
        raise ValueError("Need at least 3 SKUs in the training split to cluster.")

    lo = max(2, k_min)
    hi = min(k_max, n_skus - 1)
    if lo > hi:
        lo = 2
        hi = max(2, n_skus - 1)

    best_model = None
    best_assignments = None
    best_metrics = None
    best_key = None

    for k in range(lo, hi + 1):
        model, assignments, metrics = evaluate_candidate_k(
            train_feats_scaled,
            train_feats["sku_ID"],
            train_orders,
            train_dates,
            k,
        )
        rank_key = (
            metrics["objective"],
            metrics["avg_zero_rate"],
            -metrics["avg_cv"],
            metrics["clusters_high_sparsity_majority"],
            metrics["clusters_low_variance_majority"],
        )
        if best_key is None or rank_key < best_key:
            best_key = rank_key
            best_model = model
            best_assignments = assignments
            best_metrics = metrics

    return best_model, best_assignments, best_metrics


def assign_test_clusters(
    test_feats: pd.DataFrame,
    train_assignments: pd.DataFrame,
    scaler: StandardScaler,
    model: KMeans,
    feature_cols: list[str],
) -> pd.DataFrame:
    seen_assignments = train_assignments.copy()
    seen_assignments["assignment_source"] = "train_fit"

    unseen = test_feats[~test_feats["sku_ID"].isin(train_assignments["sku_ID"])].copy()
    if unseen.empty:
        return seen_assignments

    unseen_prepared, unseen_scaled = transform_features(unseen, scaler, feature_cols)
    unseen_labels = model.predict(unseen_scaled)
    unseen_assignments = unseen_prepared[["sku_ID"]].copy()
    unseen_assignments["sku_cluster_ID"] = unseen_labels.astype(int)
    unseen_assignments["assignment_source"] = "test_predict_unseen"
    return pd.concat([seen_assignments, unseen_assignments], ignore_index=True)


def write_split_outputs(
    out_prefix: Path,
    split_name: str,
    demand_full: pd.DataFrame,
    cluster_summary: pd.DataFrame,
) -> None:
    demand_path = out_prefix.parent / f"{out_prefix.name}_{split_name}_warehouse_daily_demand.csv"
    summary_path = out_prefix.parent / f"{out_prefix.name}_{split_name}_warehouse_summary.csv"
    demand_full.to_csv(demand_path, index=False)
    cluster_summary.to_csv(summary_path, index=False)
    print(f"wrote={demand_path}")
    print(f"wrote={summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DATABASE_DIR / "click_orders_new.db"), help="Path to SQLite DB")
    parser.add_argument(
        "--order-csv",
        default=str(RAW_DATA_DIR / "JD_order_data.csv"),
        help="Fallback order CSV path if DB lacks required order columns",
    )
    parser.add_argument(
        "--train-days",
        type=int,
        default=DEFAULT_TRAIN_DAYS,
        help="Use the first N distinct order dates as the training window",
    )
    parser.add_argument(
        "--test-days",
        type=int,
        default=DEFAULT_TEST_DAYS,
        help="Use the last N distinct order dates as the test window",
    )
    parser.add_argument("--k-min", type=int, default=20, help="Minimum candidate cluster count")
    parser.add_argument("--k-max", type=int, default=50, help="Maximum candidate cluster count")
    parser.add_argument(
        "--include-clicks",
        action="store_true",
        help="Include optional click-derived features",
    )
    parser.add_argument(
        "--out-prefix",
        default=str(PROCESSED_DATA_DIR / "sku_warehouse_train_test_clusters"),
        help="Prefix for output CSV files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    args.db = str(resolve_path(args.db, base_dir=root))
    order_csv_path = resolve_path(args.order_csv, base_dir=root)

    orders, order_source = load_orders(args.db, str(order_csv_path))
    orders = normalize_orders(orders)
    train_orders, test_orders, train_dates, test_dates, split_meta = split_orders_train_test(
        orders,
        train_days=args.train_days,
        test_days=args.test_days,
    )

    clicks_daily = load_clicks_daily(args.db) if args.include_clicks else None
    train_clicks = filter_clicks_to_dates(clicks_daily, train_dates)
    test_clicks = filter_clicks_to_dates(clicks_daily, test_dates)

    train_feats = build_sku_features(train_orders, train_dates, train_clicks)
    train_feats_clean, scaler, train_scaled, feature_cols = scale_train_features(train_feats)
    model, train_assignments, selected_metrics = choose_best_k(
        train_scaled,
        train_feats_clean,
        train_orders,
        train_dates,
        k_min=args.k_min,
        k_max=args.k_max,
    )

    train_feat_output = train_feats_clean.merge(train_assignments, on="sku_ID", how="left")
    train_feat_output["dataset_role"] = "train"
    train_feat_output["assignment_source"] = "train_fit"

    test_feats = build_sku_features(test_orders, test_dates, test_clicks)
    test_assignments = assign_test_clusters(
        test_feats,
        train_assignments,
        scaler,
        model,
        feature_cols,
    )
    test_feat_output = test_feats.merge(
        test_assignments[["sku_ID", "sku_cluster_ID", "assignment_source"]],
        on="sku_ID",
        how="left",
    )
    test_feat_output["dataset_role"] = "test"

    assignment_catalog = pd.concat(
        [
            train_assignments.assign(
                assignment_source="train_fit",
                seen_in_train=1,
                seen_in_test=lambda x: x["sku_ID"].isin(test_orders["sku_ID"]).astype(int),
            ),
            test_assignments[
                ~test_assignments["sku_ID"].isin(train_assignments["sku_ID"])
            ].assign(
                seen_in_train=0,
                seen_in_test=1,
            ),
        ],
        ignore_index=True,
    )

    out_prefix = resolve_path(args.out_prefix, base_dir=root)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    clusters_path = out_prefix.parent / f"{out_prefix.name}_assignments.csv"
    features_path = out_prefix.parent / f"{out_prefix.name}_features.csv"
    selected_path = out_prefix.parent / f"{out_prefix.name}_selected_model_summary.csv"

    pd.concat([train_feat_output, test_feat_output], ignore_index=True).to_csv(features_path, index=False)
    assignment_catalog.to_csv(clusters_path, index=False)

    train_demand = compute_demand_grid(train_orders, train_assignments, train_dates)
    train_summary, train_overall = summarize_demand_grid(train_demand)
    write_split_outputs(out_prefix, "train", train_demand, train_summary)

    test_order_assignments = test_assignments[["sku_ID", "sku_cluster_ID"]].drop_duplicates()
    test_demand = compute_demand_grid(test_orders, test_order_assignments, test_dates)
    test_summary, test_overall = summarize_demand_grid(test_demand)
    write_split_outputs(out_prefix, "test", test_demand, test_summary)

    model_summary = pd.DataFrame(
        [
            {
                "order_source": order_source,
                "train_start": split_meta["train_start"],
                "train_end": split_meta["train_end"],
                "test_start": split_meta["test_start"],
                "test_end": split_meta["test_end"],
                "unused_days": split_meta["unused_days"],
                "train_skus": int(train_orders["sku_ID"].nunique()),
                "test_skus": int(test_orders["sku_ID"].nunique()),
                "new_test_skus": int(
                    test_orders.loc[
                        ~test_orders["sku_ID"].isin(train_orders["sku_ID"]),
                        "sku_ID",
                    ].nunique()
                ),
                "chosen_k": int(selected_metrics["k"]),
                "train_objective": float(train_overall["objective"]),
                "test_objective": float(test_overall["objective"]),
                "train_avg_zero_rate": float(train_overall["avg_zero_rate"]),
                "test_avg_zero_rate": float(test_overall["avg_zero_rate"]),
                "train_avg_cv": float(train_overall["avg_cv"]),
                "test_avg_cv": float(test_overall["avg_cv"]),
                "train_avg_high_sparsity_share": float(train_overall["avg_high_sparsity_share"]),
                "test_avg_high_sparsity_share": float(test_overall["avg_high_sparsity_share"]),
                "train_avg_low_variance_share": float(train_overall["avg_low_variance_share"]),
                "test_avg_low_variance_share": float(test_overall["avg_low_variance_share"]),
            }
        ]
    )
    model_summary.to_csv(selected_path, index=False)

    print(f"wrote={clusters_path}")
    print(f"wrote={features_path}")
    print(f"wrote={selected_path}")
    print(f"order_source={order_source}")
    print(
        "train_test_split="
        f"train:{split_meta['train_start']} to {split_meta['train_end']} "
        f"({split_meta['train_days']} days, rows={split_meta['train_rows']}), "
        f"test:{split_meta['test_start']} to {split_meta['test_end']} "
        f"({split_meta['test_days']} days, rows={split_meta['test_rows']}), "
        f"unused_days={split_meta['unused_days']}"
    )
    print(f"chosen_k={selected_metrics['k']}")
    print(f"train_objective={train_overall['objective']:.6f}")
    print(f"test_objective={test_overall['objective']:.6f}")
    print(f"train_avg_zero_rate={train_overall['avg_zero_rate']:.6f}")
    print(f"test_avg_zero_rate={test_overall['avg_zero_rate']:.6f}")
    print(f"train_avg_cv={train_overall['avg_cv']:.6f}")
    print(f"test_avg_cv={test_overall['avg_cv']:.6f}")
    print(
        "new_test_skus_assigned="
        f"{int((assignment_catalog['assignment_source'] == 'test_predict_unseen').sum())}"
    )


if __name__ == "__main__":
    main()
