#!/usr/bin/env python3
"""Export cluster-by-warehouse daily demand grids for train/test and cluster-level train features."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import PROCESSED_DATA_DIR, RAW_DATA_DIR, resolve_path

TRAIN_START = "2018-03-01"
TRAIN_END = "2018-03-24"
TEST_START = "2018-03-25"
TEST_END = "2018-03-31"
SPARSITY_THRESHOLD = 0.70
CV_THRESHOLD = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", default=str(RAW_DATA_DIR / "JD_order_data.csv"), help="Order CSV path")
    parser.add_argument(
        "--assignments",
        default=str(PROCESSED_DATA_DIR / "sku_warehouse_train_test_clusters_assignments.csv"),
        help="Cluster assignment CSV produced by compute_sku_warehouse_train_test_clusters.py",
    )
    parser.add_argument("--train-start", default=TRAIN_START, help="Train start date, inclusive")
    parser.add_argument("--train-end", default=TRAIN_END, help="Train end date, inclusive")
    parser.add_argument("--test-start", default=TEST_START, help="Test start date, inclusive")
    parser.add_argument("--test-end", default=TEST_END, help="Test end date, inclusive")
    parser.add_argument(
        "--out-prefix",
        default=str(PROCESSED_DATA_DIR / "cluster_warehouse_daily_demand"),
        help="Output prefix for the exported CSV files",
    )
    return parser.parse_args()


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
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[numeric_cols] = df[numeric_cols].fillna(0.0)

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
    df["gift_rate_flag"] = (df["gift_item"].str.strip() != "").astype(int)
    return df


def load_inputs(root: Path, orders_name: str, assignments_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    orders_path = resolve_path(orders_name, base_dir=root)
    assignments_path = resolve_path(assignments_name, base_dir=root)
    if not orders_path.exists():
        raise FileNotFoundError(f"Missing orders file: {orders_path}")
    if not assignments_path.exists():
        raise FileNotFoundError(f"Missing assignments file: {assignments_path}")

    orders = pd.read_csv(
        orders_path,
        usecols=[
            "order_date",
            "sku_ID",
            "quantity",
            "dc_des",
            "final_unit_price",
            "original_unit_price",
            "direct_discount_per_unit",
            "quantity_discount_per_unit",
            "bundle_discount_per_unit",
            "coupon_discount_per_unit",
            "gift_item",
        ],
    )
    assignments = pd.read_csv(assignments_path, usecols=["sku_ID", "sku_cluster_ID"])
    assignments = assignments.drop_duplicates(subset=["sku_ID"]).copy()
    assignments["sku_cluster_ID"] = pd.to_numeric(
        assignments["sku_cluster_ID"], errors="coerce"
    )
    assignments = assignments.dropna(subset=["sku_cluster_ID"]).copy()
    assignments["sku_cluster_ID"] = assignments["sku_cluster_ID"].astype(int)
    return normalize_orders(orders), assignments


def filter_date_range(df: pd.DataFrame, start: str, end: str) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    subset = df[(df["order_date"] >= start_ts) & (df["order_date"] <= end_ts)].copy()
    dates = pd.date_range(start_ts, end_ts, freq="D")
    return subset, dates


def build_demand_grid(
    orders: pd.DataFrame,
    assignments: pd.DataFrame,
    all_dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, float]:
    df = orders.merge(assignments, on="sku_ID", how="left")
    missing_cluster_rate = float(df["sku_cluster_ID"].isna().mean()) if len(df) else 0.0
    df = df.dropna(subset=["sku_cluster_ID"]).copy()
    df["sku_cluster_ID"] = df["sku_cluster_ID"].astype(int)

    demand = (
        df.groupby(["order_date", "dc_des", "sku_cluster_ID"], as_index=False)["quantity"]
        .sum()
        .rename(columns={"quantity": "demand"})
    )

    warehouses = sorted(df["dc_des"].dropna().astype(str).unique())
    clusters = sorted(assignments["sku_cluster_ID"].unique())
    full_index = pd.MultiIndex.from_product(
        [all_dates, warehouses, clusters],
        names=["order_date", "dc_des", "sku_cluster_ID"],
    )

    demand_full = (
        demand.set_index(["order_date", "dc_des", "sku_cluster_ID"])
        .reindex(full_index, fill_value=0.0)
        .reset_index()
    )
    return demand_full, missing_cluster_rate


def build_cluster_feature_summary(
    train_orders: pd.DataFrame,
    assignments: pd.DataFrame,
    train_demand_full: pd.DataFrame,
) -> pd.DataFrame:
    train_with_cluster = train_orders.merge(assignments, on="sku_ID", how="left")
    train_with_cluster = train_with_cluster.dropna(subset=["sku_cluster_ID"]).copy()
    train_with_cluster["sku_cluster_ID"] = train_with_cluster["sku_cluster_ID"].astype(int)

    order_level = (
        train_with_cluster.groupby("sku_cluster_ID", as_index=False)
        .agg(
            sku_count=("sku_ID", "nunique"),
            warehouse_count=("dc_des", "nunique"),
            order_lines=("sku_ID", "size"),
            total_quantity=("quantity", "sum"),
            avg_price=("final_unit_price", "mean"),
            std_price=("final_unit_price", "std"),
            avg_original_price=("original_unit_price", "mean"),
            avg_discount=("total_discount_per_unit", "mean"),
            promotion_rate=("promo_line_flag", "mean"),
            gift_rate=("gift_rate_flag", "mean"),
        )
    )
    order_level["std_price"] = order_level["std_price"].fillna(0.0)

    cluster_wh_metrics = (
        train_demand_full.groupby(["sku_cluster_ID", "dc_des"])["demand"]
        .agg(
            total_days="size",
            zero_days=lambda x: int((x == 0).sum()),
            mean_demand="mean",
            std_demand="std",
        )
        .reset_index()
    )
    cluster_wh_metrics["std_demand"] = cluster_wh_metrics["std_demand"].fillna(0.0)
    cluster_wh_metrics["zero_rate"] = (
        cluster_wh_metrics["zero_days"] / cluster_wh_metrics["total_days"]
    )
    cluster_wh_metrics["cv"] = (
        cluster_wh_metrics["std_demand"]
        / cluster_wh_metrics["mean_demand"].replace(0, np.nan)
    ).fillna(0.0)
    cluster_wh_metrics["high_sparsity"] = cluster_wh_metrics["zero_rate"] >= SPARSITY_THRESHOLD
    cluster_wh_metrics["low_variance"] = (
        (cluster_wh_metrics["std_demand"] == 0.0)
        | (cluster_wh_metrics["cv"] < CV_THRESHOLD)
    )

    demand_shape = (
        cluster_wh_metrics.groupby("sku_cluster_ID", as_index=False)
        .agg(
            avg_cluster_warehouse_demand=("mean_demand", "mean"),
            avg_cluster_warehouse_std=("std_demand", "mean"),
            avg_zero_rate=("zero_rate", "mean"),
            avg_cv=("cv", "mean"),
            high_sparsity_share=("high_sparsity", "mean"),
            low_variance_share=("low_variance", "mean"),
        )
    )

    return order_level.merge(demand_shape, on="sku_cluster_ID", how="left")


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent

    orders, assignments = load_inputs(root, args.orders, args.assignments)
    train_orders, train_dates = filter_date_range(orders, args.train_start, args.train_end)
    test_orders, test_dates = filter_date_range(orders, args.test_start, args.test_end)

    train_demand, train_missing = build_demand_grid(train_orders, assignments, train_dates)
    test_demand, test_missing = build_demand_grid(test_orders, assignments, test_dates)
    cluster_features = build_cluster_feature_summary(train_orders, assignments, train_demand)

    out_prefix = resolve_path(args.out_prefix, base_dir=root)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    train_path = out_prefix.parent / f"{out_prefix.name}_train.csv"
    test_path = out_prefix.parent / f"{out_prefix.name}_test.csv"
    features_path = out_prefix.parent / f"{out_prefix.name}_features.csv"

    train_demand.to_csv(train_path, index=False)
    test_demand.to_csv(test_path, index=False)
    cluster_features.to_csv(features_path, index=False)

    print(f"WROTE {train_path}")
    print(f"WROTE {test_path}")
    print(f"WROTE {features_path}")
    print(
        "train_split="
        f"{pd.Timestamp(args.train_start).date()} to {pd.Timestamp(args.train_end).date()} "
        f"({len(train_dates)} days, rows={len(train_orders)}), "
        f"missing_cluster_rate={train_missing:.6f}"
    )
    print(
        "test_split="
        f"{pd.Timestamp(args.test_start).date()} to {pd.Timestamp(args.test_end).date()} "
        f"({len(test_dates)} days, rows={len(test_orders)}), "
        f"missing_cluster_rate={test_missing:.6f}"
    )
    print(f"clusters={cluster_features['sku_cluster_ID'].nunique()}")


if __name__ == "__main__":
    main()
