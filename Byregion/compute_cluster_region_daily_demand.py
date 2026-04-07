import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from project_paths import PROCESSED_DATA_DIR, RAW_DATA_DIR, REGION_PROCESSED_DIR, resolve_path

SPARSITY_THRESHOLD = 0.70
CV_THRESHOLD = 0.10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare cluster sets via region-day demand sparsity/variance.")
    p.add_argument("--orders", default=str(RAW_DATA_DIR / "JD_order_data.csv"), help="Order CSV path")
    p.add_argument("--network", default=str(RAW_DATA_DIR / "JD_network_data.csv"), help="Network CSV path")
    p.add_argument("--clusters-a", default=str(REGION_PROCESSED_DIR / "sku_clusters_demand_only.csv"), help="First cluster CSV")
    p.add_argument("--clusters-b", default=str(REGION_PROCESSED_DIR / "sku_clusters_price_promo.csv"), help="Second cluster CSV")
    p.add_argument("--out-prefix-a", default="demand_only", help="Output prefix for clusters-a")
    p.add_argument("--out-prefix-b", default="price_promo", help="Output prefix for clusters-b")
    return p.parse_args()


def compute_for_clusters(
    orders: pd.DataFrame,
    clusters: pd.DataFrame,
    network: pd.DataFrame,
    out_path: Path,
    summary_path: Path,
) -> dict:
    df = orders.merge(clusters, on="sku_ID", how="left")
    missing_cluster = df["sku_cluster_ID"].isna().mean()

    df = df.merge(network, left_on="dc_des", right_on="dc_ID", how="left")
    missing_region = df["region_ID"].isna().mean()

    df = df.dropna(subset=["sku_cluster_ID", "region_ID"]).copy()
    df["sku_cluster_ID"] = df["sku_cluster_ID"].astype(int)
    df["region_ID"] = df["region_ID"].astype(int)
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df = df.dropna(subset=["order_date"])  # drop invalid dates

    # Aggregate demand
    demand = (
        df.groupby(["order_date", "region_ID", "sku_cluster_ID"], as_index=False)["quantity"]
        .sum()
        .rename(columns={"quantity": "demand"})
    )

    # Build full daily grid for sparsity analysis
    all_dates = pd.date_range(demand["order_date"].min(), demand["order_date"].max(), freq="D")
    all_regions = sorted(demand["region_ID"].unique())
    all_clusters = sorted(demand["sku_cluster_ID"].unique())

    full_index = pd.MultiIndex.from_product(
        [all_dates, all_regions, all_clusters],
        names=["order_date", "region_ID", "sku_cluster_ID"],
    )

    demand_full = (
        demand.set_index(["order_date", "region_ID", "sku_cluster_ID"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    demand_full.to_csv(out_path, index=False)

    # Metrics by cluster-region
    grp = demand_full.groupby(["sku_cluster_ID", "region_ID"])["demand"]
    metrics = grp.agg(
        total_days="size",
        zero_days=lambda x: int((x == 0).sum()),
        mean_demand="mean",
        std_demand="std",
    ).reset_index()
    metrics["zero_rate"] = metrics["zero_days"] / metrics["total_days"]
    metrics["cv"] = metrics["std_demand"] / metrics["mean_demand"].replace(0, pd.NA)

    metrics["high_sparsity"] = metrics["zero_rate"] >= SPARSITY_THRESHOLD
    metrics["low_variance"] = (metrics["std_demand"].fillna(0) == 0) | (
        metrics["cv"].fillna(0) < CV_THRESHOLD
    )

    # Summaries by cluster
    cluster_summary = (
        metrics.groupby("sku_cluster_ID", as_index=False)
        .agg(
            regions=("region_ID", "nunique"),
            series=("region_ID", "size"),
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

    cluster_summary.to_csv(summary_path, index=False)

    overall = {
        "missing_cluster_rate": float(missing_cluster),
        "missing_region_rate": float(missing_region),
        "date_start": all_dates.min().date(),
        "date_end": all_dates.max().date(),
        "days": int(len(all_dates)),
        "clusters": int(cluster_summary["sku_cluster_ID"].nunique()),
        "avg_high_sparsity_share": float(cluster_summary["high_sparsity_share"].mean()),
        "avg_low_variance_share": float(cluster_summary["low_variance_share"].mean()),
        "clusters_high_sparsity_majority": int((cluster_summary["high_sparsity_share"] >= 0.5).sum()),
        "clusters_low_variance_majority": int((cluster_summary["low_variance_share"] >= 0.5).sum()),
    }
    return overall


def main():
    args = parse_args()
    root = ROOT_DIR

    orders_path = resolve_path(args.orders, base_dir=root)
    clusters_a_path = resolve_path(args.clusters_a, base_dir=root)
    clusters_b_path = resolve_path(args.clusters_b, base_dir=root)
    network_path = resolve_path(args.network, base_dir=root)

    if not orders_path.exists():
        raise FileNotFoundError(f"Missing orders file: {orders_path}")
    if not clusters_a_path.exists():
        raise FileNotFoundError(f"Missing clusters file: {clusters_a_path}")
    if not clusters_b_path.exists():
        raise FileNotFoundError(f"Missing clusters file: {clusters_b_path}")
    if not network_path.exists():
        raise FileNotFoundError(f"Missing network file: {network_path}")

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
            "type",
        ],
    )
    clusters_a = pd.read_csv(clusters_a_path)
    clusters_b = pd.read_csv(clusters_b_path)
    network = pd.read_csv(network_path)

    REGION_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_a = REGION_PROCESSED_DIR / f"cluster_region_daily_demand_{args.out_prefix_a}.csv"
    sum_a = REGION_PROCESSED_DIR / f"cluster_region_daily_demand_summary_{args.out_prefix_a}.csv"
    out_b = REGION_PROCESSED_DIR / f"cluster_region_daily_demand_{args.out_prefix_b}.csv"
    sum_b = REGION_PROCESSED_DIR / f"cluster_region_daily_demand_summary_{args.out_prefix_b}.csv"

    stats_a = compute_for_clusters(orders, clusters_a, network, out_a, sum_a)
    stats_b = compute_for_clusters(orders, clusters_b, network, out_b, sum_b)

    print("WROTE", out_a)
    print("WROTE", sum_a)
    print("WROTE", out_b)
    print("WROTE", sum_b)

    def fmt(stats: dict) -> str:
        return (
            f"missing_cluster_rate={stats['missing_cluster_rate']:.4f}, "
            f"missing_region_rate={stats['missing_region_rate']:.4f}, "
            f"date_range={stats['date_start']} to {stats['date_end']} ({stats['days']} days), "
            f"clusters={stats['clusters']}, "
            f"avg_high_sparsity_share={stats['avg_high_sparsity_share']:.4f}, "
            f"avg_low_variance_share={stats['avg_low_variance_share']:.4f}, "
            f"clusters_high_sparsity_majority={stats['clusters_high_sparsity_majority']}, "
            f"clusters_low_variance_majority={stats['clusters_low_variance_majority']}"
        )

    print("A:", fmt(stats_a))
    print("B:", fmt(stats_b))

    score_a = stats_a["avg_high_sparsity_share"] + stats_a["avg_low_variance_share"]
    score_b = stats_b["avg_high_sparsity_share"] + stats_b["avg_low_variance_share"]
    if score_a < score_b:
        winner = "A"
    elif score_b < score_a:
        winner = "B"
    else:
        winner = "tie"
    print("RECOMMENDATION=" + winner + " (lower avg_high_sparsity_share + avg_low_variance_share)")


if __name__ == "__main__":
    main()
