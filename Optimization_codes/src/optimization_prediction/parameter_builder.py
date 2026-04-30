from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import H_VALUE, K_VALUE, TEST_END_DATE, TEST_START_DATE, TRAIN_END_DATE, U_VALUE


@dataclass
class DemandArtifacts:
    train_panel: pd.DataFrame
    test_panel: pd.DataFrame
    full_panel: pd.DataFrame
    cluster_universe: list[int]
    warehouse_universe: list[int]
    diagnostics: dict


@dataclass
class ParameterArtifacts:
    constants: dict
    sku_prices: pd.DataFrame
    cluster_mapping: pd.DataFrame
    cluster_prices: pd.DataFrame
    price_summary: pd.DataFrame
    delivery_time_matrix: pd.DataFrame
    route_matrix: pd.DataFrame
    capacity: pd.DataFrame
    capacity_summary: pd.DataFrame
    procurement_eligibility: pd.DataFrame
    initial_inventory_test: pd.DataFrame
    train_inventory_parameter: pd.DataFrame
    test_inventory_proxy: pd.DataFrame
    train_demand_parameter: pd.DataFrame
    test_demand_actual: pd.DataFrame
    diagnostics: dict


def prepare_real_cluster_demand_panels(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> DemandArtifacts:
    train = train_df.copy()
    test = test_df.copy()
    train["order_date"] = pd.to_datetime(train["order_date"])
    test["order_date"] = pd.to_datetime(test["order_date"])

    cluster_universe = sorted(train["sku_cluster_ID"].astype(int).unique().tolist())
    warehouse_universe = sorted(set(train["dc_des"].astype(int)).union(test["dc_des"].astype(int)))
    full_test_index = pd.MultiIndex.from_product(
        [
            pd.date_range(TEST_START_DATE, TEST_END_DATE, freq="D"),
            warehouse_universe,
            cluster_universe,
        ],
        names=["order_date", "dc_des", "sku_cluster_ID"],
    )
    test = (
        test.set_index(["order_date", "dc_des", "sku_cluster_ID"])
        .reindex(full_test_index, fill_value=0)
        .reset_index()
    )

    for frame in [train, test]:
        frame["dc_des"] = frame["dc_des"].astype(int)
        frame["sku_cluster_ID"] = frame["sku_cluster_ID"].astype(int)
        frame["demand"] = frame["demand"].astype(float)

    train = train.rename(columns={"order_date": "date", "dc_des": "warehouse"})
    test = test.rename(columns={"order_date": "date", "dc_des": "warehouse"})
    full_panel = pd.concat([train, test], ignore_index=True).sort_values(["date", "warehouse", "sku_cluster_ID"])

    diagnostics = {
        "train_date_min": str(train["date"].min().date()),
        "train_date_max": str(train["date"].max().date()),
        "test_date_min": str(test["date"].min().date()),
        "test_date_max": str(test["date"].max().date()),
        "cluster_universe": cluster_universe,
        "missing_test_clusters_filled_with_zero_rows": sorted(set(cluster_universe) - set(test_df["sku_cluster_ID"].astype(int))),
    }
    return DemandArtifacts(
        train_panel=train,
        test_panel=test,
        full_panel=full_panel,
        cluster_universe=cluster_universe,
        warehouse_universe=warehouse_universe,
        diagnostics=diagnostics,
    )


def build_warehouse_universe(
    capacity_df: pd.DataFrame,
    demand_artifacts: DemandArtifacts,
    daily_sku_dc_summary: pd.DataFrame | None = None,
) -> list[int]:
    capacity_warehouses = set(capacity_df["dc_des"].astype(int))
    summary_warehouses: set[int] = set()
    if daily_sku_dc_summary is not None and "dc_des" in daily_sku_dc_summary.columns:
        summary_warehouses = set(daily_sku_dc_summary["dc_des"].dropna().astype(int))
    return sorted(set(demand_artifacts.warehouse_universe).union(capacity_warehouses).union(summary_warehouses))


def compute_cluster_prices_from_unit_price(
    orders: pd.DataFrame,
    cluster_mapping: pd.DataFrame,
    train_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    train_orders = orders[(orders["order_date"] >= train_panel["date"].min()) & (orders["order_date"] <= TRAIN_END_DATE)].copy()
    train_orders = train_orders.dropna(subset=["sku_ID", "original_unit_price", "quantity"]).copy()
    train_orders["quantity"] = train_orders["quantity"].astype(float)
    train_orders["original_unit_price"] = train_orders["original_unit_price"].astype(float)
    train_orders = train_orders[(train_orders["quantity"] > 0) & (train_orders["original_unit_price"] > 0)].copy()

    sku_prices = (
        train_orders.groupby("sku_ID", as_index=False)["original_unit_price"]
        .mean()
        .rename(columns={"original_unit_price": "sku_price"})
    )

    cluster_mapping_clean = cluster_mapping.copy()
    required_cols = {"sku_ID", "sku_cluster_ID"}
    if cluster_mapping_clean.empty or not required_cols.issubset(cluster_mapping_clean.columns):
        raise ValueError("Cluster mapping is required to build corrected cluster prices.")
    cluster_mapping_clean = cluster_mapping_clean.loc[:, ["sku_ID", "sku_cluster_ID"]].dropna().drop_duplicates()
    cluster_mapping_clean["sku_cluster_ID"] = cluster_mapping_clean["sku_cluster_ID"].astype(int)

    cluster_price_input = cluster_mapping_clean.merge(sku_prices, on="sku_ID", how="left")
    global_mean_sku_price = float(sku_prices["sku_price"].mean()) if not sku_prices.empty else 0.0
    cluster_prices = (
        cluster_price_input.groupby("sku_cluster_ID", as_index=False)["sku_price"]
        .mean()
        .rename(columns={"sku_price": "p_s"})
    )
    cluster_universe = sorted(train_panel["sku_cluster_ID"].astype(int).unique().tolist())
    cluster_prices = (
        pd.DataFrame({"sku_cluster_ID": cluster_universe})
        .merge(cluster_prices, on="sku_cluster_ID", how="left")
        .sort_values("sku_cluster_ID")
        .reset_index(drop=True)
    )
    missing_clusters = cluster_prices.loc[cluster_prices["p_s"].isna(), "sku_cluster_ID"].astype(int).tolist()
    cluster_prices["p_s"] = cluster_prices["p_s"].fillna(global_mean_sku_price)

    priced_order_skus = set(sku_prices["sku_ID"].astype(str))
    mapped_order_skus = set(cluster_mapping_clean["sku_ID"].astype(str)) & set(train_orders["sku_ID"].astype(str))
    weighted_covered_units = float(
        train_orders.loc[train_orders["sku_ID"].astype(str).isin(set(cluster_mapping_clean["sku_ID"].astype(str))), "quantity"].sum()
    )
    total_units = float(train_orders["quantity"].sum())
    price_summary = pd.DataFrame(
        [
            {"metric": "price_method", "value": "mean original_unit_price by SKU, then mean sku_price within cluster"},
            {"metric": "cluster_mapping_rows", "value": int(cluster_mapping_clean.shape[0])},
            {"metric": "unique_mapped_skus", "value": int(cluster_mapping_clean["sku_ID"].nunique())},
            {"metric": "priced_skus", "value": int(sku_prices["sku_ID"].nunique())},
            {"metric": "global_mean_sku_price", "value": global_mean_sku_price},
            {"metric": "missing_clusters_filled_with_global_mean", "value": ", ".join(map(str, missing_clusters)) if missing_clusters else "none"},
            {"metric": "order_sku_mapping_coverage", "value": float(len(mapped_order_skus) / len(priced_order_skus)) if priced_order_skus else 0.0},
            {"metric": "order_unit_mapping_coverage", "value": float(weighted_covered_units / total_units) if total_units else 0.0},
        ]
    )
    diagnostics = {
        "price_method": "mean original_unit_price by SKU, then mean sku_price within cluster",
        "cluster_mapping_rows": int(cluster_mapping_clean.shape[0]),
        "unique_mapped_skus": int(cluster_mapping_clean["sku_ID"].nunique()),
        "priced_skus": int(sku_prices["sku_ID"].nunique()),
        "global_mean_sku_price": global_mean_sku_price,
        "missing_clusters_filled_with_global_mean": missing_clusters,
        "order_sku_mapping_coverage": float(len(mapped_order_skus) / len(priced_order_skus)) if priced_order_skus else 0.0,
        "order_unit_mapping_coverage": float(weighted_covered_units / total_units) if total_units else 0.0,
    }
    return sku_prices, cluster_mapping_clean, cluster_prices, price_summary, diagnostics


def build_route_matrices(
    order_mart: pd.DataFrame,
    warehouse_universe: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train_routes = order_mart.loc[order_mart["order_time_dt"].dt.floor("D") <= TRAIN_END_DATE].copy()
    train_routes = train_routes.dropna(subset=["dc_ori", "dc_des", "lead_time_hours"])
    train_routes = train_routes[train_routes["lead_time_hours"] > 0].copy()
    train_routes["dc_ori"] = train_routes["dc_ori"].astype(int)
    train_routes["dc_des"] = train_routes["dc_des"].astype(int)
    train_routes["route_exists"] = 1

    all_route_warehouses = sorted(
        set(train_routes["dc_ori"]).union(train_routes["dc_des"]).union(warehouse_universe)
    )
    t_matrix = train_routes.pivot_table(
        index="dc_ori",
        columns="dc_des",
        values="lead_time_hours",
        aggfunc="mean",
    ).reindex(index=all_route_warehouses, columns=all_route_warehouses)
    r_matrix = (
        train_routes.pivot_table(
            index="dc_ori",
            columns="dc_des",
            values="route_exists",
            aggfunc="max",
        )
        .reindex(index=all_route_warehouses, columns=all_route_warehouses)
        .fillna(0)
        .astype(int)
    )
    return t_matrix, r_matrix, {
        "n_route_warehouses": len(all_route_warehouses),
        "n_observed_routes_in_training": int(train_routes[["dc_ori", "dc_des"]].drop_duplicates().shape[0]),
    }


def build_capacity_parameter(
    capacity_df: pd.DataFrame,
    daily_sku_dc_summary: pd.DataFrame,
    warehouse_universe: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    capacity = capacity_df.copy()
    capacity["dc_des"] = capacity["dc_des"].astype(int)
    capacity["capacity"] = capacity["capacity"].astype(float)
    capacity = capacity.rename(columns={"dc_des": "warehouse", "capacity": "legacy_capacity"})

    summary = daily_sku_dc_summary.copy()
    summary["dc_des"] = summary["dc_des"].astype(int)
    summary["order_day"] = pd.to_datetime(summary["order_day"])
    summary["inventory"] = pd.to_numeric(summary["inventory"], errors="coerce").fillna(0.0)
    corrected_capacity = (
        summary.groupby(["dc_des", "order_day"], as_index=False)["inventory"]
        .sum()
        .rename(columns={"dc_des": "warehouse", "inventory": "daily_inventory_total"})
        .groupby("warehouse", as_index=False)["daily_inventory_total"]
        .max()
        .rename(columns={"daily_inventory_total": "corrected_capacity"})
    )
    corrected_capacity["corrected_capacity"] = corrected_capacity["corrected_capacity"].where(
        corrected_capacity["corrected_capacity"] > 0,
        np.nan,
    )

    capacity_summary = (
        pd.DataFrame({"warehouse": warehouse_universe})
        .merge(capacity, on="warehouse", how="left")
        .merge(corrected_capacity, on="warehouse", how="left")
        .sort_values("warehouse")
    )
    capacity_summary["capacity_source"] = np.where(
        capacity_summary["corrected_capacity"].notna(),
        "JD_daily_sku_dc_summary.inventory max over warehouse-day",
        "inventory_capacity.xlsx fallback",
    )
    capacity_summary["C_j"] = capacity_summary["corrected_capacity"].fillna(capacity_summary["legacy_capacity"]).fillna(0.0)
    final_capacity = capacity_summary[["warehouse", "C_j"]].copy()

    diagnostics = {
        "capacity_method": "max observed warehouse-day inventory from JD_daily_sku_dc_summary.inventory, with nonpositive values treated as missing and inventory_capacity.xlsx used as fallback",
        "capacity_summary_rows": int(capacity_summary.shape[0]),
        "warehouses_using_corrected_capacity": int(capacity_summary["corrected_capacity"].notna().sum()),
        "warehouses_using_legacy_fallback": int(capacity_summary["corrected_capacity"].isna().sum()),
        "legacy_fallback_warehouses": capacity_summary.loc[
            capacity_summary["corrected_capacity"].isna(), "warehouse"
        ].astype(int).tolist(),
        "legacy_capacity_mean": float(capacity_summary["legacy_capacity"].dropna().mean()) if capacity_summary["legacy_capacity"].notna().any() else 0.0,
        "corrected_capacity_mean": float(capacity_summary["corrected_capacity"].dropna().mean()) if capacity_summary["corrected_capacity"].notna().any() else 0.0,
    }
    return final_capacity, capacity_summary, diagnostics


def build_procurement_eligibility(
    network_df: pd.DataFrame,
    warehouse_universe: list[int],
) -> tuple[pd.DataFrame, dict]:
    network = network_df.copy()
    network["dc_ID"] = network["dc_ID"].astype(int)
    network["region_ID"] = network["region_ID"].astype(int)
    network["W_j"] = (network["dc_ID"] == network["region_ID"]).astype(int)
    procurement = (
        network.groupby("dc_ID", as_index=False)["W_j"]
        .max()
        .rename(columns={"dc_ID": "warehouse"})
    )
    procurement = pd.DataFrame({"warehouse": warehouse_universe}).merge(procurement, on="warehouse", how="left")
    missing = procurement.loc[procurement["W_j"].isna(), "warehouse"].astype(int).tolist()
    procurement["W_j"] = procurement["W_j"].fillna(0).astype(int)
    return procurement.sort_values("warehouse"), {
        "warehouses_missing_network_mapping_assigned_zero": missing,
        "n_large_warehouses": int(procurement["W_j"].sum()),
        "n_total_warehouses": int(procurement.shape[0]),
    }


def build_inventory_proxy(
    inventory_df: pd.DataFrame,
    daily_sku_dc_summary: pd.DataFrame,
    demand_artifacts: DemandArtifacts,
    warehouse_universe: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    inventory = inventory_df.copy()
    inventory["date"] = pd.to_datetime(inventory["date"])
    inventory = inventory[(inventory["date"] >= demand_artifacts.train_panel["date"].min()) & (inventory["date"] <= TEST_END_DATE)]
    raw_inventory = (
        inventory.groupby(["date", "dc_ID"], as_index=False)["sku_ID"]
        .count()
        .rename(columns={"dc_ID": "warehouse", "sku_ID": "raw_inventory_units"})
    )

    summary_inventory = daily_sku_dc_summary.copy()
    summary_inventory["order_day"] = pd.to_datetime(summary_inventory["order_day"])
    summary_inventory = summary_inventory[
        (summary_inventory["order_day"] >= demand_artifacts.train_panel["date"].min())
        & (summary_inventory["order_day"] <= TEST_END_DATE)
    ].copy()
    summary_inventory["dc_des"] = summary_inventory["dc_des"].astype(int)
    summary_inventory["inventory"] = pd.to_numeric(summary_inventory["inventory"], errors="coerce").fillna(0.0)
    warehouse_daily_inventory = (
        summary_inventory.groupby(["order_day", "dc_des"], as_index=False)["inventory"]
        .sum()
        .rename(columns={"order_day": "date", "dc_des": "warehouse", "inventory": "summary_inventory_units"})
    )
    full_inventory_index = pd.MultiIndex.from_product(
        [
            pd.date_range(demand_artifacts.train_panel["date"].min(), TEST_END_DATE, freq="D"),
            warehouse_universe,
        ],
        names=["date", "warehouse"],
    )
    warehouse_daily_inventory = (
        warehouse_daily_inventory.set_index(["date", "warehouse"])
        .reset_index()
    )
    warehouse_daily_inventory = (
        pd.DataFrame(index=full_inventory_index)
        .reset_index()
        .merge(warehouse_daily_inventory, on=["date", "warehouse"], how="left")
        .merge(raw_inventory, on=["date", "warehouse"], how="left")
    )
    warehouse_daily_inventory["warehouse_inventory_units"] = (
        warehouse_daily_inventory["summary_inventory_units"]
        .fillna(warehouse_daily_inventory["raw_inventory_units"])
        .fillna(0.0)
    )

    share = (
        demand_artifacts.train_panel.groupby(["warehouse", "sku_cluster_ID"], as_index=False)["demand"]
        .sum()
    )
    warehouse_totals = share.groupby("warehouse", as_index=False)["demand"].sum().rename(columns={"demand": "warehouse_total"})
    share = share.merge(warehouse_totals, on="warehouse", how="left")
    share["inventory_share_proxy"] = np.where(
        share["warehouse_total"] > 0,
        share["demand"] / share["warehouse_total"],
        0.0,
    )
    full_share_index = pd.MultiIndex.from_product(
        [warehouse_universe, demand_artifacts.cluster_universe],
        names=["warehouse", "sku_cluster_ID"],
    )
    share = (
        share.set_index(["warehouse", "sku_cluster_ID"])
        .reindex(full_share_index, fill_value=0)
        .reset_index()
    )

    cluster_inventory_daily = warehouse_daily_inventory.merge(share[["warehouse", "sku_cluster_ID", "inventory_share_proxy"]], on="warehouse", how="left")
    cluster_inventory_daily["cluster_inventory_proxy"] = (
        cluster_inventory_daily["warehouse_inventory_units"] * cluster_inventory_daily["inventory_share_proxy"]
    )
    train_inventory = (
        cluster_inventory_daily[cluster_inventory_daily["date"] <= TRAIN_END_DATE]
        .groupby(["sku_cluster_ID", "warehouse"], as_index=False)["cluster_inventory_proxy"]
        .mean()
        .rename(columns={"cluster_inventory_proxy": "I0_sj_train_proxy"})
    )
    initial_inventory_test = (
        cluster_inventory_daily[cluster_inventory_daily["date"] == TEST_START_DATE]
        .groupby(["sku_cluster_ID", "warehouse"], as_index=False)["cluster_inventory_proxy"]
        .sum()
        .rename(columns={"cluster_inventory_proxy": "I0_sj_test_initial"})
    )
    test_inventory = (
        cluster_inventory_daily[cluster_inventory_daily["date"] >= TEST_START_DATE]
        .groupby(["sku_cluster_ID", "warehouse"], as_index=False)["cluster_inventory_proxy"]
        .mean()
        .rename(columns={"cluster_inventory_proxy": "I0_sj_test_proxy"})
    )
    diagnostics = {
        "inventory_proxy_source": "warehouse-day inventory from JD_daily_sku_dc_summary apportioned by training demand share, with raw inventory-count fallback when summary is missing",
        "n_inventory_rows": int(len(inventory)),
        "summary_inventory_days": int(summary_inventory["order_day"].nunique()),
        "warehouse_days_using_raw_fallback": int(warehouse_daily_inventory["summary_inventory_units"].isna().sum()),
    }
    return train_inventory, initial_inventory_test, test_inventory, diagnostics


def average_demand_parameter(panel: pd.DataFrame, output_column: str) -> pd.DataFrame:
    return (
        panel.groupby(["sku_cluster_ID", "warehouse"], as_index=False)["demand"]
        .mean()
        .rename(columns={"demand": output_column})
        .sort_values(["sku_cluster_ID", "warehouse"])
    )


def build_parameter_artifacts(
    orders: pd.DataFrame,
    order_mart: pd.DataFrame,
    inventory: pd.DataFrame,
    network: pd.DataFrame,
    capacity: pd.DataFrame,
    daily_sku_dc_summary: pd.DataFrame,
    cluster_mapping: pd.DataFrame,
    demand_artifacts: DemandArtifacts,
) -> ParameterArtifacts:
    warehouse_universe = build_warehouse_universe(capacity, demand_artifacts, daily_sku_dc_summary)
    sku_prices, cluster_mapping_clean, cluster_prices, price_summary, price_diagnostics = compute_cluster_prices_from_unit_price(
        orders=orders,
        cluster_mapping=cluster_mapping,
        train_panel=demand_artifacts.train_panel,
    )
    delivery_time_matrix, route_matrix, route_diag = build_route_matrices(order_mart, warehouse_universe)
    capacity_df, capacity_summary, capacity_diag = build_capacity_parameter(capacity, daily_sku_dc_summary, warehouse_universe)
    procurement_df, procurement_diag = build_procurement_eligibility(network, warehouse_universe)
    train_inventory_proxy, initial_inventory_test, test_inventory_proxy, inventory_diag = build_inventory_proxy(
        inventory_df=inventory,
        daily_sku_dc_summary=daily_sku_dc_summary,
        demand_artifacts=demand_artifacts,
        warehouse_universe=warehouse_universe,
    )

    diagnostics = {
        "demand": demand_artifacts.diagnostics,
        "price": price_diagnostics,
        "routes": route_diag,
        "capacity": capacity_diag,
        "procurement": procurement_diag,
        "inventory": inventory_diag,
    }
    return ParameterArtifacts(
        constants={"k": K_VALUE, "h": H_VALUE, "u": U_VALUE},
        sku_prices=sku_prices,
        cluster_mapping=cluster_mapping_clean,
        cluster_prices=cluster_prices,
        price_summary=price_summary,
        delivery_time_matrix=delivery_time_matrix,
        route_matrix=route_matrix,
        capacity=capacity_df,
        capacity_summary=capacity_summary,
        procurement_eligibility=procurement_df,
        initial_inventory_test=initial_inventory_test,
        train_inventory_parameter=train_inventory_proxy,
        test_inventory_proxy=test_inventory_proxy,
        train_demand_parameter=average_demand_parameter(demand_artifacts.train_panel, "D_sj_train"),
        test_demand_actual=average_demand_parameter(demand_artifacts.test_panel, "D_sj_test_actual"),
        diagnostics=diagnostics,
    )
