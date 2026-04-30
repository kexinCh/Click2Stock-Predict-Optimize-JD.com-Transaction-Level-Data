from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import ProjectPaths


@dataclass
class RawData:
    orders: pd.DataFrame
    clicks: pd.DataFrame
    deliveries: pd.DataFrame
    inventory: pd.DataFrame
    network: pd.DataFrame
    sku: pd.DataFrame
    users: pd.DataFrame
    order_mart: pd.DataFrame
    cross_fill: pd.DataFrame
    capacity: pd.DataFrame
    daily_sku_dc_summary: pd.DataFrame
    train_cluster_demand: pd.DataFrame
    test_cluster_demand: pd.DataFrame
    cluster_mapping: pd.DataFrame


def load_raw_data(paths: ProjectPaths) -> RawData:
    orders = pd.read_csv(paths.data_dir / "JD_order_data.csv")
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    orders["order_time"] = pd.to_datetime(orders["order_time"], format="mixed", errors="coerce")

    clicks = pd.read_csv(paths.data_dir / "JD_click_data.csv")
    clicks["request_time"] = pd.to_datetime(clicks["request_time"])

    deliveries = pd.read_csv(paths.data_dir / "JD_delivery_data.csv")
    for col in ["ship_out_time", "arr_station_time", "arr_time"]:
        deliveries[col] = pd.to_datetime(deliveries[col])

    inventory = pd.read_csv(paths.data_dir / "JD_inventory_data.csv")
    inventory["date"] = pd.to_datetime(inventory["date"])

    network = pd.read_csv(paths.data_dir / "JD_network_data.csv")
    sku = pd.read_csv(paths.data_dir / "JD_sku_data.csv")
    users = pd.read_csv(paths.data_dir / "JD_user_data.csv")

    order_mart = pd.read_csv(paths.optimization_dir / "JD_order_mart.csv")
    for col in ["order_time_dt", "ship_out_dt", "arr_station_dt", "arr_dt", "date_dt"]:
        order_mart[col] = pd.to_datetime(order_mart[col], errors="coerce")

    cross_fill = pd.read_excel(paths.optimization_dir / "JD_sku_dc_cross_filling_hours.xlsx")
    capacity = pd.read_excel(paths.optimization_dir / "inventory_capacity.xlsx")
    daily_sku_dc_summary = pd.read_excel(paths.optimization_dir / "JD_daily_sku_dc_summary.xlsx")
    daily_sku_dc_summary["order_day"] = pd.to_datetime(daily_sku_dc_summary["order_day"])
    train_cluster_demand = pd.read_csv(paths.train_demand_path)
    test_cluster_demand = pd.read_csv(paths.test_demand_path)
    for frame in [train_cluster_demand, test_cluster_demand]:
        frame["order_date"] = pd.to_datetime(frame["order_date"])

    cluster_mapping_candidates = [
        paths.dataset_root / "sku_warehouse_train_test_clusters_assignments.csv",
        paths.root / "sku_warehouse_train_test_clusters_assignments.csv",
        paths.tables_dir / "cluster_mapping.csv",
    ]
    cluster_mapping_path = next((path for path in cluster_mapping_candidates if path.exists()), None)
    if cluster_mapping_path is None:
        cluster_mapping = pd.DataFrame(columns=["sku_ID", "sku_cluster_ID"])
    else:
        if cluster_mapping_path.suffix.lower() == ".xlsx":
            cluster_mapping = pd.read_excel(cluster_mapping_path)
        else:
            cluster_mapping = pd.read_csv(cluster_mapping_path)

    for frame in [
        orders,
        clicks,
        deliveries,
        inventory,
        network,
        sku,
        users,
        order_mart,
        cross_fill,
        capacity,
        daily_sku_dc_summary,
        train_cluster_demand,
        test_cluster_demand,
        cluster_mapping,
    ]:
        frame.columns = frame.columns.str.strip()

    return RawData(
        orders=orders,
        clicks=clicks,
        deliveries=deliveries,
        inventory=inventory,
        network=network,
        sku=sku,
        users=users,
        order_mart=order_mart,
        cross_fill=cross_fill,
        capacity=capacity,
        daily_sku_dc_summary=daily_sku_dc_summary,
        train_cluster_demand=train_cluster_demand,
        test_cluster_demand=test_cluster_demand,
        cluster_mapping=cluster_mapping,
    )
