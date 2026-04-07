import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from project_paths import EXPLORATION_OUTPUTS_DIR, RAW_DATA_DIR, ensure_dir

def load_data(order_path: str, network_path: str):
    orders = pd.read_csv(order_path)
    network = pd.read_csv(network_path)

    # Normalize column names for join
    network = network.rename(columns={"region_ID": "region_id", "dc_ID": "dc_des"})

    # Parse date
    orders["date"] = pd.to_datetime(orders["order_date"], errors="coerce").dt.date

    # Attach region info via destination DC
    orders = orders.merge(network, on="dc_des", how="left")

    return orders, network


def build_daily_tables(orders: pd.DataFrame):
    # Daily demand by sku + region
    daily_by_region = (
        orders.groupby(["sku_ID", "date", "region_id"], dropna=False)["quantity"]
        .sum()
        .reset_index()
        .rename(columns={"quantity": "demand"})
    )

    # Daily demand by sku + destination DC
    daily_by_dc = (
        orders.groupby(["sku_ID", "date", "dc_des"], dropna=False)["quantity"]
        .sum()
        .reset_index()
        .rename(columns={"quantity": "demand"})
    )

    return daily_by_region, daily_by_dc


def sparsity_metrics(df: pd.DataFrame, dim_cols):
    # Compute total possible combinations from observed distincts
    sizes = [df[c].nunique(dropna=False) for c in dim_cols]
    total_possible = int(np.prod(sizes))
    nonzero = len(df)
    density = nonzero / total_possible if total_possible else 0
    sparsity = 1 - density
    return {
        "dimensions": dict(zip(dim_cols, sizes)),
        "total_possible": total_possible,
        "nonzero_rows": nonzero,
        "density": density,
        "sparsity": sparsity,
    }


def variance_metrics(df: pd.DataFrame, group_cols):
    # Daily demand variance per group over time
    stats = (
        df.groupby(group_cols)["demand"]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    stats["cv"] = stats["std"] / stats["mean"].replace(0, np.nan)
    return stats


def plot_distributions(df: pd.DataFrame, prefix: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Demand distribution
    plt.figure(figsize=(8, 4.5))
    plt.hist(df["demand"], bins=50, color="#2f5d8a", alpha=0.85)
    plt.title(f"Daily Demand Distribution - {prefix}")
    plt.xlabel("Demand")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_demand_hist.png", dpi=150)
    plt.close()

    # Demand distribution (log scale, add 1)
    plt.figure(figsize=(8, 4.5))
    plt.hist(np.log1p(df["demand"]), bins=50, color="#6b8e23", alpha=0.85)
    plt.title(f"Daily Demand Distribution (log1p) - {prefix}")
    plt.xlabel("log1p(Demand)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_demand_hist_log1p.png", dpi=150)
    plt.close()


def plot_variance(stats: pd.DataFrame, prefix: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # CV distribution
    cv = stats["cv"].replace([np.inf, -np.inf], np.nan).dropna()
    plt.figure(figsize=(8, 4.5))
    plt.hist(cv, bins=50, color="#8a2f2f", alpha=0.85)
    plt.title(f"Coefficient of Variation (CV) - {prefix}")
    plt.xlabel("CV")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_cv_hist.png", dpi=150)
    plt.close()


def main():
    orders_path = RAW_DATA_DIR / "JD_order_data.csv"
    network_path = RAW_DATA_DIR / "JD_network_data.csv"
    out_dir = ensure_dir(EXPLORATION_OUTPUTS_DIR)

    orders, _ = load_data(str(orders_path), str(network_path))

    daily_by_region, daily_by_dc = build_daily_tables(orders)

    # Save tables
    daily_by_region.to_csv(out_dir / "daily_demand_by_sku_region.csv", index=False)
    daily_by_dc.to_csv(out_dir / "daily_demand_by_sku_dc.csv", index=False)

    # Sparsity metrics
    sparsity_region = sparsity_metrics(daily_by_region, ["sku_ID", "date", "region_id"])
    sparsity_dc = sparsity_metrics(daily_by_dc, ["sku_ID", "date", "dc_des"])

    # Variance metrics
    var_region = variance_metrics(daily_by_region, ["sku_ID", "region_id"])
    var_dc = variance_metrics(daily_by_dc, ["sku_ID", "dc_des"])

    # Save variance stats
    var_region.to_csv(out_dir / "variance_stats_by_sku_region.csv", index=False)
    var_dc.to_csv(out_dir / "variance_stats_by_sku_dc.csv", index=False)

    # Plots
    plot_distributions(daily_by_region, "sku_region", out_dir)
    plot_distributions(daily_by_dc, "sku_dc", out_dir)
    plot_variance(var_region, "sku_region", out_dir)
    plot_variance(var_dc, "sku_dc", out_dir)

    # Summaries
    def summarize(name, sparsity, var_stats):
        cv = var_stats["cv"].replace([np.inf, -np.inf], np.nan)
        low_cv = (cv < 0.1).mean()
        print(f"== {name} ==")
        print(f"dimensions: {sparsity['dimensions']}")
        print(f"total_possible: {sparsity['total_possible']}")
        print(f"nonzero_rows: {sparsity['nonzero_rows']}")
        print(f"density: {sparsity['density']:.6f}")
        print(f"sparsity: {sparsity['sparsity']:.6f}")
        print(f"share_cv_lt_0.1: {low_cv:.4f}")
        print()

    summarize("Daily demand by SKU+Region", sparsity_region, var_region)
    summarize("Daily demand by SKU+DC", sparsity_dc, var_dc)


if __name__ == "__main__":
    main()
