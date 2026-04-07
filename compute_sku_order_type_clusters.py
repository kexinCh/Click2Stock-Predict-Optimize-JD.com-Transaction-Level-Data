#!/usr/bin/env python3
"""Cluster SKUs into structural order types to reduce sparsity."""

import argparse
import os
import sqlite3
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from project_paths import DATABASE_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, resolve_path

EPS = 1e-6
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
        raise ValueError(
            f"Order source missing required columns: {sorted(missing)}"
        )
    return orders, "csv:JD_order_data.csv"


def load_clicks_daily(db_path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(db_path):
        return None

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
        conn.execute("PRAGMA temp_store=MEMORY")
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
        cur = conn.execute(query)
        rows = cur.fetchall()
        clicks = pd.DataFrame(rows, columns=["sku_ID", "order_date", "daily_clicks"])
        conn.close()
        return clicks
    except Exception as exc:
        print(f"warning: unable to load clicks from sqlite ({exc}); skipping click features")
        return None


def safe_numeric_fill(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def split_orders_train_test(
    orders: pd.DataFrame,
    train_days: int = DEFAULT_TRAIN_DAYS,
    test_days: int = DEFAULT_TEST_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = orders.copy()
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce").dt.normalize()
    df = df[df["order_date"].notna()].copy()

    unique_dates = sorted(df["order_date"].drop_duplicates())
    if len(unique_dates) < train_days + test_days:
        raise ValueError(
            "Not enough distinct order dates for the requested split: "
            f"{len(unique_dates)} available, need at least {train_days + test_days}"
        )

    train_dates = pd.DatetimeIndex(unique_dates[:train_days])
    test_dates = pd.DatetimeIndex(unique_dates[-test_days:])

    train_orders = df[df["order_date"].isin(train_dates)].copy()
    test_orders = df[df["order_date"].isin(test_dates)].copy()

    metadata = {
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
    return train_orders, test_orders, metadata


def compute_daily_sku_aggregates(orders: pd.DataFrame) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    df = orders.copy()

    #df = df[df['type'] ==1].copy()  

    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce").dt.date
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

    df["gift_item"] = df["gift_item"].astype(str)
    df.loc[df["gift_item"].isin(["nan", "None", "NaN"]), "gift_item"] = ""

    df["total_discount_per_unit"] = (
        df["direct_discount_per_unit"]
        + df["quantity_discount_per_unit"]
        + df["bundle_discount_per_unit"]
        + df["coupon_discount_per_unit"]
    )

    df["promo_line_flag"] = (
        (df["total_discount_per_unit"] > 0)
        | (df["gift_item"].astype(str).str.strip() != "")
    ).astype(int)

    daily_sku = (
        df.groupby(["sku_ID", "order_date"], as_index=False)
        .agg(
            daily_demand=("quantity", "sum"),
            avg_final_price=("final_unit_price", "mean"),
            avg_original_price=("original_unit_price", "mean"),
            avg_discount=("total_discount_per_unit", "mean"),
            promo_flag=("promo_line_flag", "max"),
        )
    )

    all_dates = pd.date_range(
        pd.to_datetime(daily_sku["order_date"]).min(),
        pd.to_datetime(daily_sku["order_date"]).max(),
        freq="D",
    )

    daily_sku["order_date"] = pd.to_datetime(daily_sku["order_date"]) 
    return daily_sku, all_dates


def compute_elasticity(group: pd.DataFrame) -> float:
    x = np.log(np.maximum(group["avg_final_price"].to_numpy(dtype=float), 0.0) + EPS)
    y = np.log(group["daily_demand"].to_numpy(dtype=float) + EPS)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size < 2:
        return 0.0
    if np.isclose(np.var(x), 0.0):
        return 0.0

    slope = np.polyfit(x, y, 1)[0]
    if not np.isfinite(slope):
        return 0.0
    return float(slope)


def build_sku_features(
    daily_sku: pd.DataFrame,
    all_dates: pd.DatetimeIndex,
    clicks_daily: Optional[pd.DataFrame],
) -> pd.DataFrame:
    sku_ids = daily_sku["sku_ID"].drop_duplicates().sort_values().reset_index(drop=True)

    price_promo_feats = (
        daily_sku.groupby("sku_ID", as_index=False)
        .agg(
            mean_price=("avg_final_price", "mean"),
            std_price=("avg_final_price", "std"),
            mean_discount=("avg_discount", "mean"),
            promo_rate=("promo_flag", "mean"),
        )
    )
    price_promo_feats["std_price"] = price_promo_feats["std_price"].fillna(0.0)
    feats = price_promo_feats.copy()

    if clicks_daily is not None and not clicks_daily.empty:
        # Build full grid to keep zero-click days in the features.
        full_grid = pd.MultiIndex.from_product(
            [sku_ids, all_dates], names=["sku_ID", "order_date"]
        ).to_frame(index=False)
        clicks = clicks_daily.copy()
        clicks["order_date"] = pd.to_datetime(clicks["order_date"], errors="coerce")
        clicks = clicks[clicks["order_date"].notna()].copy()

        click_panel = full_grid.merge(
            clicks,
            on=["sku_ID", "order_date"],
            how="left",
        )
        click_panel["daily_clicks"] = pd.to_numeric(
            click_panel["daily_clicks"], errors="coerce"
        ).fillna(0.0)

        click_feats = (
            click_panel.groupby("sku_ID", as_index=False)
            .agg(
                mean_daily_clicks=("daily_clicks", "mean"),
            )
        )

        feats = feats.merge(click_feats, on="sku_ID", how="left")

    return feats


def clean_and_scale_features(feats: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    out = feats.copy()

    numeric_cols = [c for c in out.columns if c != "sku_ID"]
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(out[numeric_cols])

    return out, x_scaled, numeric_cols


def pick_kmeans(x_scaled: np.ndarray, k_min: int = 20, k_max: int = 50) -> tuple[int, float, np.ndarray]:
    n = x_scaled.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 SKUs for clustering.")

    lo = max(2, k_min)
    hi = min(k_max, n - 1)
    if lo > hi:
        lo = 2
        hi = n - 1

    best_k = lo
    best_score = -1.0
    best_labels = None
    sample_size = min(5000, n)

    for k in range(lo, hi + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(x_scaled)
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(
            x_scaled,
            labels,
            sample_size=sample_size if sample_size < n else None,
            random_state=42,
        )
        if score > best_score:
            best_score = float(score)
            best_k = k
            best_labels = labels

    if best_labels is None:
        model = KMeans(n_clusters=2, random_state=42, n_init=10)
        best_labels = model.fit_predict(x_scaled)
        best_k = 2
        best_score = float(silhouette_score(x_scaled, best_labels))

    return best_k, best_score, best_labels


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default=str(DATABASE_DIR / "click_orders_new.db"), help="Path to SQLite DB")
    p.add_argument(
        "--order-csv",
        default=str(RAW_DATA_DIR / "JD_order_data.csv"),
        help="Fallback order CSV path if DB lacks required order columns",
    )
    p.add_argument(
        "--out-clusters",
        default=str(PROCESSED_DATA_DIR / "sku_clusters.csv"),
        help="Output sku cluster CSV",
    )
    p.add_argument(
        "--out-features",
        default=str(PROCESSED_DATA_DIR / "sku_cluster_features.csv"),
        help="Output feature + cluster CSV",
    )
    p.add_argument(
        "--include-clicks",
        action="store_true",
        help="Include optional click-derived features (can be slow on large click tables)",
    )
    p.add_argument(
        "--train-days",
        type=int,
        default=DEFAULT_TRAIN_DAYS,
        help="Number of earliest distinct order dates reserved for the training split summary",
    )
    p.add_argument(
        "--test-days",
        type=int,
        default=DEFAULT_TEST_DAYS,
        help="Number of latest distinct order dates reserved for the test split summary",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.db = str(resolve_path(args.db))
    args.order_csv = str(resolve_path(args.order_csv))
    args.out_clusters = str(resolve_path(args.out_clusters))
    args.out_features = str(resolve_path(args.out_features))
    Path(args.out_clusters).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_features).parent.mkdir(parents=True, exist_ok=True)

    orders, order_source = load_orders(args.db, args.order_csv)
    _, _, split_meta = split_orders_train_test(
        orders,
        train_days=args.train_days,
        test_days=args.test_days,
    )
    clicks_daily = load_clicks_daily(args.db) if args.include_clicks else None

    daily_sku, all_dates = compute_daily_sku_aggregates(orders)
    feats = build_sku_features(daily_sku, all_dates, clicks_daily)

    feats_clean, x_scaled, _ = clean_and_scale_features(feats)
    best_k, best_silhouette, labels = pick_kmeans(x_scaled, k_min=20, k_max=50)
    feats_clean["sku_cluster_ID"] = labels

    clusters = feats_clean[["sku_ID", "sku_cluster_ID"]].copy()

    clusters.to_csv(args.out_clusters, index=False)
    feats_clean.to_csv(args.out_features, index=False)

    cluster_sizes = feats_clean.groupby("sku_cluster_ID").size()
    avg_size = float(cluster_sizes.mean())
    min_size = int(cluster_sizes.min())
    max_size = int(cluster_sizes.max())

    print(f"order_source={order_source}")
    print(
        "train_test_split="
        f"train:{split_meta['train_start']} to {split_meta['train_end']} "
        f"({split_meta['train_days']} days, rows={split_meta['train_rows']}), "
        f"test:{split_meta['test_start']} to {split_meta['test_end']} "
        f"({split_meta['test_days']} days, rows={split_meta['test_rows']}), "
        f"unused_days={split_meta['unused_days']}"
    )
    print(f"total_skus_clustered={feats_clean['sku_ID'].nunique()}")
    print(f"chosen_k={best_k}")
    print(f"silhouette_score={best_silhouette:.6f}")
    print(f"average_cluster_size={avg_size:.2f}")
    print(f"min_cluster_size={min_size}")
    print(f"max_cluster_size={max_size}")


if __name__ == "__main__":
    main()
