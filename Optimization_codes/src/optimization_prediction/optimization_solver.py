from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix


@dataclass
class PolicyRunArtifacts:
    policy_name: str
    planning_basis: str
    daily_summary: pd.DataFrame
    procurement_decisions: pd.DataFrame
    transfer_decisions: pd.DataFrame
    planned_inventory: pd.DataFrame
    planned_shortages: pd.DataFrame
    realized_inventory: pd.DataFrame
    realized_shortages: pd.DataFrame
    weekly_summary: pd.DataFrame


@dataclass
class OptimizationArtifacts:
    predicted_policy: PolicyRunArtifacts
    oracle_policy: PolicyRunArtifacts
    weekly_comparison: pd.DataFrame
    parameter_usage_audit: pd.DataFrame


def _series_from_frame(frame: pd.DataFrame, key_col: str, value_col: str) -> dict:
    return {
        int(key): float(value)
        for key, value in frame[[key_col, value_col]].itertuples(index=False, name=None)
    }


def _inventory_state_from_frame(
    frame: pd.DataFrame,
    cluster_universe: list[int],
    warehouse_universe: list[int],
    value_col: str,
) -> dict[tuple[int, int], float]:
    index = pd.MultiIndex.from_product(
        [cluster_universe, warehouse_universe],
        names=["sku_cluster_ID", "warehouse"],
    )
    state = (
        frame.set_index(["sku_cluster_ID", "warehouse"])[value_col]
        .reindex(index, fill_value=0.0)
        .astype(float)
    )
    return {(int(s), int(j)): float(value) for (s, j), value in state.items()}


def _daily_panel_to_dict(
    frame: pd.DataFrame,
    value_col: str,
    cluster_universe: list[int],
    warehouse_universe: list[int],
) -> dict[tuple[int, int], float]:
    index = pd.MultiIndex.from_product(
        [cluster_universe, warehouse_universe],
        names=["sku_cluster_ID", "warehouse"],
    )
    series = (
        frame.set_index(["sku_cluster_ID", "warehouse"])[value_col]
        .reindex(index, fill_value=0.0)
        .astype(float)
    )
    return {(int(s), int(j)): float(value) for (s, j), value in series.items()}


def _prepare_route_inputs(
    route_matrix: pd.DataFrame,
    delivery_time_matrix: pd.DataFrame,
    warehouse_universe: list[int],
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], float], list[tuple[int, int]]]:
    route_subset = (
        route_matrix.reindex(index=warehouse_universe, columns=warehouse_universe)
        .fillna(0)
        .astype(int)
    )
    time_subset = delivery_time_matrix.reindex(index=warehouse_universe, columns=warehouse_universe)
    time_fallback = float(np.nanmedian(time_subset.to_numpy())) if not np.isnan(time_subset.to_numpy()).all() else 24.0
    time_subset = time_subset.fillna(time_fallback)

    route_map: dict[tuple[int, int], int] = {}
    time_map: dict[tuple[int, int], float] = {}
    route_pairs: list[tuple[int, int]] = []
    for n in warehouse_universe:
        for m in warehouse_universe:
            if n == m:
                continue
            route_map[(n, m)] = int(route_subset.loc[n, m])
            time_map[(n, m)] = float(time_subset.loc[n, m])
            route_pairs.append((n, m))
    return route_map, time_map, route_pairs


def solve_daily_inventory_optimization(
    date: pd.Timestamp,
    demand_map: dict[tuple[int, int], float],
    current_inventory: dict[tuple[int, int], float],
    cluster_universe: list[int],
    warehouse_universe: list[int],
    cluster_prices: dict[int, float],
    warehouse_capacity: dict[int, float],
    procurement_eligibility: dict[int, float],
    route_feasibility: dict[tuple[int, int], int],
    route_time: dict[tuple[int, int], float],
    route_pairs: list[tuple[int, int]],
    constants: dict,
) -> dict:
    pair_keys = [(s, j) for s in cluster_universe for j in warehouse_universe]
    transfer_keys = [(s, n, m) for s in cluster_universe for (n, m) in route_pairs]

    n_procure = len(pair_keys)
    n_inventory = len(pair_keys)
    n_shortage = len(pair_keys)
    n_transfer = len(transfer_keys)

    idx_procure = {key: idx for idx, key in enumerate(pair_keys)}
    idx_inventory = {key: n_procure + idx for idx, key in enumerate(pair_keys)}
    idx_shortage = {key: n_procure + n_inventory + idx for idx, key in enumerate(pair_keys)}
    idx_transfer = {key: n_procure + n_inventory + n_shortage + idx for idx, key in enumerate(transfer_keys)}
    n_variables = n_procure + n_inventory + n_shortage + n_transfer

    c = np.zeros(n_variables, dtype=float)
    bounds: list[tuple[float, float | None]] = [(0.0, None)] * n_variables

    for s, j in pair_keys:
        price = float(cluster_prices.get(s, 0.0))
        demand = float(demand_map[(s, j)])
        capacity = float(warehouse_capacity[j])
        procurement_upper = float((capacity + demand) * procurement_eligibility[j])

        c[idx_inventory[(s, j)]] = constants["h"] * price
        c[idx_shortage[(s, j)]] = constants["u"] * price
        bounds[idx_inventory[(s, j)]] = (0.0, capacity)
        bounds[idx_shortage[(s, j)]] = (0.0, max(demand, 0.0))
        bounds[idx_procure[(s, j)]] = (0.0, procurement_upper)

    for s, n, m in transfer_keys:
        price = float(cluster_prices.get(s, 0.0))
        transfer_upper = float((warehouse_capacity[n] + warehouse_capacity[m] + demand_map[(s, n)]) * route_feasibility[(n, m)])
        c[idx_transfer[(s, n, m)]] = constants["k"] * price * float(route_time[(n, m)])
        bounds[idx_transfer[(s, n, m)]] = (0.0, transfer_upper)

    a_eq = lil_matrix((len(pair_keys), n_variables), dtype=float)
    b_eq = np.zeros(len(pair_keys), dtype=float)
    for row_idx, (s, j) in enumerate(pair_keys):
        a_eq[row_idx, idx_inventory[(s, j)]] = 1.0
        a_eq[row_idx, idx_shortage[(s, j)]] = -1.0
        a_eq[row_idx, idx_procure[(s, j)]] = -1.0

        for n in warehouse_universe:
            if n != j:
                a_eq[row_idx, idx_transfer[(s, n, j)]] = -1.0
        for m in warehouse_universe:
            if m != j:
                a_eq[row_idx, idx_transfer[(s, j, m)]] = 1.0

        b_eq[row_idx] = float(current_inventory[(s, j)] - demand_map[(s, j)])

    a_ub = lil_matrix((len(warehouse_universe), n_variables), dtype=float)
    b_ub = np.zeros(len(warehouse_universe), dtype=float)
    for row_idx, j in enumerate(warehouse_universe):
        for s in cluster_universe:
            a_ub[row_idx, idx_inventory[(s, j)]] = 1.0
        b_ub[row_idx] = float(warehouse_capacity[j])

    solution = linprog(
        c=c,
        A_ub=a_ub.tocsr(),
        b_ub=b_ub,
        A_eq=a_eq.tocsr(),
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not solution.success:
        raise RuntimeError(f"Optimization failed on {date.date()}: {solution.message}")

    vector = solution.x
    procurement_rows = []
    inventory_rows = []
    shortage_rows = []
    transfer_rows = []

    for s, j in pair_keys:
        procurement_rows.append(
            {
                "date": date,
                "sku_cluster_ID": s,
                "warehouse": j,
                "procurement_units": float(vector[idx_procure[(s, j)]]),
            }
        )
        inventory_rows.append(
            {
                "date": date,
                "sku_cluster_ID": s,
                "warehouse": j,
                "planned_ending_inventory": float(vector[idx_inventory[(s, j)]]),
            }
        )
        shortage_rows.append(
            {
                "date": date,
                "sku_cluster_ID": s,
                "warehouse": j,
                "planned_shortage_units": float(vector[idx_shortage[(s, j)]]),
            }
        )

    for s, n, m in transfer_keys:
        units = float(vector[idx_transfer[(s, n, m)]])
        if units <= 1e-8:
            continue
        transfer_rows.append(
            {
                "date": date,
                "sku_cluster_ID": s,
                "source_warehouse": n,
                "destination_warehouse": m,
                "transfer_units": units,
                "route_feasible": route_feasibility[(n, m)],
                "lead_time_hours": route_time[(n, m)],
                "transfer_cost": constants["k"] * cluster_prices.get(s, 0.0) * route_time[(n, m)] * units,
            }
        )

    procurement_df = pd.DataFrame(procurement_rows)
    inventory_df = pd.DataFrame(inventory_rows)
    shortage_df = pd.DataFrame(shortage_rows)
    transfer_df = pd.DataFrame(transfer_rows)

    shortage_cost = float(
        shortage_df.merge(
            pd.DataFrame({"sku_cluster_ID": list(cluster_prices.keys()), "p_s": list(cluster_prices.values())}),
            on="sku_cluster_ID",
            how="left",
        )
        .assign(cost=lambda frame: frame["planned_shortage_units"] * frame["p_s"].fillna(0) * constants["u"])["cost"]
        .sum()
    )
    holding_cost = float(
        inventory_df.merge(
            pd.DataFrame({"sku_cluster_ID": list(cluster_prices.keys()), "p_s": list(cluster_prices.values())}),
            on="sku_cluster_ID",
            how="left",
        )
        .assign(cost=lambda frame: frame["planned_ending_inventory"] * frame["p_s"].fillna(0) * constants["h"])["cost"]
        .sum()
    )
    transfer_cost = float(transfer_df["transfer_cost"].sum()) if not transfer_df.empty else 0.0
    demand_total = float(sum(demand_map.values()))
    shortage_units = float(shortage_df["planned_shortage_units"].sum())
    fulfilled_units = demand_total - shortage_units

    return {
        "objective_value": float(solution.fun),
        "procurement": procurement_df,
        "inventory": inventory_df,
        "shortage": shortage_df,
        "transfer": transfer_df,
        "planned_metrics": {
            "date": date,
            "planned_demand_units": demand_total,
            "planned_shortage_units": shortage_units,
            "planned_ending_inventory_units": float(inventory_df["planned_ending_inventory"].sum()),
            "planned_transfer_units": float(transfer_df["transfer_units"].sum()) if not transfer_df.empty else 0.0,
            "planned_procurement_units": float(procurement_df["procurement_units"].sum()),
            "planned_service_level": float(fulfilled_units / demand_total) if demand_total else 1.0,
            "planned_shortage_cost": shortage_cost,
            "planned_holding_cost": holding_cost,
            "planned_transfer_cost": transfer_cost,
            "planned_total_cost": shortage_cost + holding_cost + transfer_cost,
        },
    }


def simulate_realized_day(
    date: pd.Timestamp,
    current_inventory: dict[tuple[int, int], float],
    actual_demand_map: dict[tuple[int, int], float],
    procurement_df: pd.DataFrame,
    transfer_df: pd.DataFrame,
    cluster_prices: dict[int, float],
    warehouse_capacity: dict[int, float],
    route_time: dict[tuple[int, int], float],
    constants: dict,
    cluster_universe: list[int],
    warehouse_universe: list[int],
) -> dict:
    procurement_map = _daily_panel_to_dict(
        procurement_df.rename(columns={"procurement_units": "value"}),
        "value",
        cluster_universe,
        warehouse_universe,
    )
    incoming = {(s, j): 0.0 for s in cluster_universe for j in warehouse_universe}
    outgoing = {(s, j): 0.0 for s in cluster_universe for j in warehouse_universe}
    transfer_cost = 0.0
    if not transfer_df.empty:
        for row in transfer_df.itertuples(index=False):
            key_source = (int(row.sku_cluster_ID), int(row.source_warehouse))
            key_dest = (int(row.sku_cluster_ID), int(row.destination_warehouse))
            units = float(row.transfer_units)
            outgoing[key_source] += units
            incoming[key_dest] += units
            transfer_cost += constants["k"] * cluster_prices.get(int(row.sku_cluster_ID), 0.0) * route_time[
                (int(row.source_warehouse), int(row.destination_warehouse))
            ] * units

    realized_inventory_rows = []
    realized_shortage_rows = []
    next_inventory: dict[tuple[int, int], float] = {}
    shortage_cost = 0.0
    holding_cost = 0.0
    overflow_cost = 0.0

    for s in cluster_universe:
        price = float(cluster_prices.get(s, 0.0))
        for j in warehouse_universe:
            key = (s, j)
            available = (
                float(current_inventory[key])
                + float(procurement_map.get(key, 0.0))
                + float(incoming[key])
                - float(outgoing[key])
            )
            demand = float(actual_demand_map[key])
            shortage = max(demand - available, 0.0)
            ending_inventory = max(available - demand, 0.0)
            next_inventory[key] = ending_inventory
            shortage_cost += shortage * price * constants["u"]
            holding_cost += ending_inventory * price * constants["h"]
            realized_inventory_rows.append(
                {
                    "date": date,
                    "sku_cluster_ID": s,
                    "warehouse": j,
                    "realized_ending_inventory": ending_inventory,
                    "realized_capacity_overflow": 0.0,
                }
            )
            realized_shortage_rows.append(
                {
                    "date": date,
                    "sku_cluster_ID": s,
                    "warehouse": j,
                    "realized_shortage_units": shortage,
                }
            )

    realized_inventory_df = pd.DataFrame(realized_inventory_rows)
    for j in warehouse_universe:
        warehouse_capacity_limit = float(warehouse_capacity.get(j, 0.0))
        warehouse_mask = realized_inventory_df["warehouse"] == j
        warehouse_inventory = realized_inventory_df.loc[warehouse_mask, "realized_ending_inventory"].sum()
        overflow_units = max(warehouse_inventory - warehouse_capacity_limit, 0.0)
        if overflow_units <= 1e-8:
            continue

        # When actual demand comes in below the planning demand, the realized state can end
        # above on-site capacity even though the planned LP respected C_j. To keep the next
        # day's optimization feasible, trim the excess from the lowest-value cluster inventory first.
        warehouse_rows = (
            realized_inventory_df.loc[warehouse_mask, ["sku_cluster_ID", "realized_ending_inventory"]]
            .assign(price=lambda frame: frame["sku_cluster_ID"].map(cluster_prices).fillna(0.0))
            .sort_values(["price", "sku_cluster_ID"], ascending=[True, True])
        )
        remaining_overflow = overflow_units
        for row in warehouse_rows.itertuples(index=False):
            if remaining_overflow <= 1e-8:
                break
            s = int(row.sku_cluster_ID)
            removable = min(float(row.realized_ending_inventory), remaining_overflow)
            if removable <= 1e-8:
                continue
            key = (s, j)
            next_inventory[key] = max(float(next_inventory[key]) - removable, 0.0)
            row_mask = warehouse_mask & (realized_inventory_df["sku_cluster_ID"] == s)
            realized_inventory_df.loc[row_mask, "realized_ending_inventory"] = next_inventory[key]
            realized_inventory_df.loc[row_mask, "realized_capacity_overflow"] += removable
            overflow_cost += removable * float(cluster_prices.get(s, 0.0)) * constants["h"]
            remaining_overflow -= removable

    realized_shortage_df = pd.DataFrame(realized_shortage_rows)
    actual_demand_total = float(sum(actual_demand_map.values()))
    actual_shortage_units = float(realized_shortage_df["realized_shortage_units"].sum())
    fulfilled_units = actual_demand_total - actual_shortage_units
    realized_overflow_units = float(realized_inventory_df["realized_capacity_overflow"].sum())

    return {
        "next_inventory": next_inventory,
        "realized_inventory": realized_inventory_df,
        "realized_shortage": realized_shortage_df,
        "realized_metrics": {
            "date": date,
            "actual_demand_units": actual_demand_total,
            "realized_shortage_units": actual_shortage_units,
            "realized_ending_inventory_units": float(realized_inventory_df["realized_ending_inventory"].sum()),
            "realized_overflow_units": realized_overflow_units,
            "realized_transfer_units": float(transfer_df["transfer_units"].sum()) if not transfer_df.empty else 0.0,
            "realized_procurement_units": float(procurement_df["procurement_units"].sum()),
            "realized_service_level": float(fulfilled_units / actual_demand_total) if actual_demand_total else 1.0,
            "realized_shortage_cost": shortage_cost,
            "realized_holding_cost": holding_cost,
            "realized_overflow_cost": overflow_cost,
            "realized_transfer_cost": transfer_cost,
            "realized_total_cost": shortage_cost + holding_cost + overflow_cost + transfer_cost,
        },
    }


def run_receding_horizon_policy(
    policy_name: str,
    planning_basis: str,
    planning_demand_panel: pd.DataFrame,
    actual_demand_panel: pd.DataFrame,
    initial_inventory: pd.DataFrame,
    cluster_prices_frame: pd.DataFrame,
    procurement_eligibility_frame: pd.DataFrame,
    capacity_frame: pd.DataFrame,
    route_matrix: pd.DataFrame,
    delivery_time_matrix: pd.DataFrame,
    constants: dict,
) -> PolicyRunArtifacts:
    cluster_universe = sorted(cluster_prices_frame["sku_cluster_ID"].astype(int).tolist())
    warehouse_universe = sorted(capacity_frame["warehouse"].astype(int).tolist())
    cluster_prices = _series_from_frame(cluster_prices_frame, "sku_cluster_ID", "p_s")
    warehouse_capacity = _series_from_frame(capacity_frame, "warehouse", "C_j")
    procurement_eligibility = _series_from_frame(procurement_eligibility_frame, "warehouse", "W_j")
    route_feasibility, route_time, route_pairs = _prepare_route_inputs(
        route_matrix=route_matrix,
        delivery_time_matrix=delivery_time_matrix,
        warehouse_universe=warehouse_universe,
    )
    current_inventory = _inventory_state_from_frame(
        initial_inventory,
        cluster_universe=cluster_universe,
        warehouse_universe=warehouse_universe,
        value_col="I0_sj_test_initial",
    )

    planning_by_date = {
        date: frame[["sku_cluster_ID", "warehouse", "demand"]].copy()
        for date, frame in planning_demand_panel.groupby("date")
    }
    actual_by_date = {
        date: frame[["sku_cluster_ID", "warehouse", "demand"]].copy()
        for date, frame in actual_demand_panel.groupby("date")
    }

    daily_rows = []
    procurement_rows = []
    transfer_rows = []
    planned_inventory_rows = []
    planned_shortage_rows = []
    realized_inventory_rows = []
    realized_shortage_rows = []

    for date in sorted(actual_by_date):
        planning_day = planning_by_date[date]
        actual_day = actual_by_date[date]
        planning_demand_map = _daily_panel_to_dict(planning_day, "demand", cluster_universe, warehouse_universe)
        actual_demand_map = _daily_panel_to_dict(actual_day, "demand", cluster_universe, warehouse_universe)

        solved = solve_daily_inventory_optimization(
            date=date,
            demand_map=planning_demand_map,
            current_inventory=current_inventory,
            cluster_universe=cluster_universe,
            warehouse_universe=warehouse_universe,
            cluster_prices=cluster_prices,
            warehouse_capacity=warehouse_capacity,
            procurement_eligibility=procurement_eligibility,
            route_feasibility=route_feasibility,
            route_time=route_time,
            route_pairs=route_pairs,
            constants=constants,
        )
        realized = simulate_realized_day(
            date=date,
            current_inventory=current_inventory,
            actual_demand_map=actual_demand_map,
            procurement_df=solved["procurement"],
            transfer_df=solved["transfer"],
            cluster_prices=cluster_prices,
            warehouse_capacity=warehouse_capacity,
            route_time=route_time,
            constants=constants,
            cluster_universe=cluster_universe,
            warehouse_universe=warehouse_universe,
        )
        current_inventory = realized["next_inventory"]

        daily_rows.append(
            {
                "policy_name": policy_name,
                "planning_basis": planning_basis,
                **solved["planned_metrics"],
                **realized["realized_metrics"],
                "cost_gap_vs_plan": realized["realized_metrics"]["realized_total_cost"] - solved["planned_metrics"]["planned_total_cost"],
            }
        )
        procurement_rows.append(solved["procurement"].assign(policy_name=policy_name, planning_basis=planning_basis))
        planned_inventory_rows.append(solved["inventory"].assign(policy_name=policy_name, planning_basis=planning_basis))
        planned_shortage_rows.append(solved["shortage"].assign(policy_name=policy_name, planning_basis=planning_basis))
        realized_inventory_rows.append(realized["realized_inventory"].assign(policy_name=policy_name, planning_basis=planning_basis))
        realized_shortage_rows.append(realized["realized_shortage"].assign(policy_name=policy_name, planning_basis=planning_basis))
        if not solved["transfer"].empty:
            transfer_rows.append(solved["transfer"].assign(policy_name=policy_name, planning_basis=planning_basis))

    daily_summary = pd.DataFrame(daily_rows).sort_values("date").reset_index(drop=True)
    procurement_df = pd.concat(procurement_rows, ignore_index=True)
    transfer_df = pd.concat(transfer_rows, ignore_index=True) if transfer_rows else pd.DataFrame(
        columns=[
            "date",
            "sku_cluster_ID",
            "source_warehouse",
            "destination_warehouse",
            "transfer_units",
            "route_feasible",
            "lead_time_hours",
            "transfer_cost",
            "policy_name",
            "planning_basis",
        ]
    )
    planned_inventory_df = pd.concat(planned_inventory_rows, ignore_index=True)
    planned_shortage_df = pd.concat(planned_shortage_rows, ignore_index=True)
    realized_inventory_df = pd.concat(realized_inventory_rows, ignore_index=True)
    realized_shortage_df = pd.concat(realized_shortage_rows, ignore_index=True)

    weekly_summary = pd.DataFrame(
        [
            {
                "policy_name": policy_name,
                "planning_basis": planning_basis,
                "planned_total_cost": float(daily_summary["planned_total_cost"].sum()),
                "realized_total_cost": float(daily_summary["realized_total_cost"].sum()),
                "planned_shortage_cost": float(daily_summary["planned_shortage_cost"].sum()),
                "realized_shortage_cost": float(daily_summary["realized_shortage_cost"].sum()),
                "planned_holding_cost": float(daily_summary["planned_holding_cost"].sum()),
                "realized_holding_cost": float(daily_summary["realized_holding_cost"].sum()),
                "realized_overflow_cost": float(daily_summary["realized_overflow_cost"].sum()),
                "planned_transfer_cost": float(daily_summary["planned_transfer_cost"].sum()),
                "realized_transfer_cost": float(daily_summary["realized_transfer_cost"].sum()),
                "actual_demand_units": float(daily_summary["actual_demand_units"].sum()),
                "realized_shortage_units": float(daily_summary["realized_shortage_units"].sum()),
                "realized_ending_inventory_units": float(daily_summary["realized_ending_inventory_units"].sum()),
                "realized_overflow_units": float(daily_summary["realized_overflow_units"].sum()),
                "realized_transfer_units": float(daily_summary["realized_transfer_units"].sum()),
                "realized_procurement_units": float(daily_summary["realized_procurement_units"].sum()),
                "realized_service_level": float(
                    1.0 - daily_summary["realized_shortage_units"].sum() / daily_summary["actual_demand_units"].sum()
                )
                if daily_summary["actual_demand_units"].sum()
                else 1.0,
            }
        ]
    )

    return PolicyRunArtifacts(
        policy_name=policy_name,
        planning_basis=planning_basis,
        daily_summary=daily_summary,
        procurement_decisions=procurement_df,
        transfer_decisions=transfer_df,
        planned_inventory=planned_inventory_df,
        planned_shortages=planned_shortage_df,
        realized_inventory=realized_inventory_df,
        realized_shortages=realized_shortage_df,
        weekly_summary=weekly_summary,
    )


def run_optimization_comparison(
    predicted_demand_panel: pd.DataFrame,
    actual_demand_panel: pd.DataFrame,
    parameter_artifacts,
) -> OptimizationArtifacts:
    predicted_policy = run_receding_horizon_policy(
        policy_name="predicted_policy",
        planning_basis="predicted_demand",
        planning_demand_panel=predicted_demand_panel.rename(columns={"predicted_demand": "demand"}),
        actual_demand_panel=actual_demand_panel[["date", "sku_cluster_ID", "warehouse", "demand"]].copy(),
        initial_inventory=parameter_artifacts.initial_inventory_test,
        cluster_prices_frame=parameter_artifacts.cluster_prices,
        procurement_eligibility_frame=parameter_artifacts.procurement_eligibility,
        capacity_frame=parameter_artifacts.capacity,
        route_matrix=parameter_artifacts.route_matrix,
        delivery_time_matrix=parameter_artifacts.delivery_time_matrix,
        constants=parameter_artifacts.constants,
    )
    oracle_policy = run_receding_horizon_policy(
        policy_name="oracle_policy",
        planning_basis="actual_demand",
        planning_demand_panel=actual_demand_panel[["date", "sku_cluster_ID", "warehouse", "demand"]].copy(),
        actual_demand_panel=actual_demand_panel[["date", "sku_cluster_ID", "warehouse", "demand"]].copy(),
        initial_inventory=parameter_artifacts.initial_inventory_test,
        cluster_prices_frame=parameter_artifacts.cluster_prices,
        procurement_eligibility_frame=parameter_artifacts.procurement_eligibility,
        capacity_frame=parameter_artifacts.capacity,
        route_matrix=parameter_artifacts.route_matrix,
        delivery_time_matrix=parameter_artifacts.delivery_time_matrix,
        constants=parameter_artifacts.constants,
    )

    weekly_comparison = predicted_policy.weekly_summary.merge(
        oracle_policy.weekly_summary,
        how="cross",
        suffixes=("_predicted_policy", "_oracle_policy"),
    )
    weekly_comparison["realized_cost_gap_vs_oracle"] = (
        weekly_comparison["realized_total_cost_predicted_policy"] - weekly_comparison["realized_total_cost_oracle_policy"]
    )
    weekly_comparison["service_level_gap_vs_oracle"] = (
        weekly_comparison["realized_service_level_predicted_policy"] - weekly_comparison["realized_service_level_oracle_policy"]
    )

    parameter_usage_audit = pd.DataFrame(
        [
            {"parameter": "D_{s,j}", "used_in": "inventory balance constraint", "mathematical_role": "daily demand term subtracted from available inventory"},
            {"parameter": "I0_{s,j}", "used_in": "initial condition / inventory balance", "mathematical_role": "starting inventory state for each cluster-warehouse pair"},
            {"parameter": "W_j", "used_in": "procurement eligibility constraint", "mathematical_role": "upper bound on procurement variables"},
            {"parameter": "C_j", "used_in": "capacity constraint", "mathematical_role": "upper bound on total ending inventory per warehouse"},
            {"parameter": "r_{n,m}", "used_in": "route feasibility constraint", "mathematical_role": "upper bound on transfer variables for each route"},
            {"parameter": "t_{n,m}", "used_in": "objective function", "mathematical_role": "transfer cost multiplier inside k * t_{n,m} * transfer"},
            {"parameter": "p_s", "used_in": "objective function", "mathematical_role": "value weighting for shortage, holding, and transfer components"},
            {"parameter": "u", "used_in": "objective function", "mathematical_role": "shortage penalty coefficient"},
            {"parameter": "h", "used_in": "objective function", "mathematical_role": "holding cost coefficient"},
            {"parameter": "k", "used_in": "objective function", "mathematical_role": "delivery-time penalty coefficient on transfers"},
        ]
    )

    return OptimizationArtifacts(
        predicted_policy=predicted_policy,
        oracle_policy=oracle_policy,
        weekly_comparison=weekly_comparison,
        parameter_usage_audit=parameter_usage_audit,
    )


def save_optimization_figures(optimization: OptimizationArtifacts, figures_dir: Path) -> dict:
    figures_dir.mkdir(parents=True, exist_ok=True)

    cost_path = figures_dir / "optimization_cost_comparison.png"
    cost_df = pd.DataFrame(
        [
            {
                "policy": "Predicted Policy",
                "cost": float(optimization.predicted_policy.weekly_summary.iloc[0]["realized_total_cost"]),
            },
            {
                "policy": "Oracle Policy",
                "cost": float(optimization.oracle_policy.weekly_summary.iloc[0]["realized_total_cost"]),
            },
        ]
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(cost_df["policy"], cost_df["cost"], color=["#003262", "#FDB515"])
    ax.set_title("Realized Weekly Cost: Predicted Policy vs Oracle")
    ax.set_ylabel("Cost")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(cost_path, dpi=180)
    plt.close(fig)

    service_path = figures_dir / "optimization_daily_service_level.png"
    service_df = pd.concat(
        [
            optimization.predicted_policy.daily_summary[["date", "realized_service_level"]].assign(policy="Predicted Policy"),
            optimization.oracle_policy.daily_summary[["date", "realized_service_level"]].assign(policy="Oracle Policy"),
        ],
        ignore_index=True,
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    for policy, frame in service_df.groupby("policy"):
        ax.plot(frame["date"], frame["realized_service_level"], marker="o", label=policy)
    ax.set_title("Daily Service Level")
    ax.set_ylabel("Service Level")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(service_path, dpi=180)
    plt.close(fig)

    shortage_path = figures_dir / "optimization_daily_shortage.png"
    shortage_df = pd.concat(
        [
            optimization.predicted_policy.daily_summary[["date", "realized_shortage_units"]].assign(policy="Predicted Policy"),
            optimization.oracle_policy.daily_summary[["date", "realized_shortage_units"]].assign(policy="Oracle Policy"),
        ],
        ignore_index=True,
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    for policy, frame in shortage_df.groupby("policy"):
        ax.plot(frame["date"], frame["realized_shortage_units"], marker="o", label=policy)
    ax.set_title("Daily Shortage Units")
    ax.set_ylabel("Units")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(shortage_path, dpi=180)
    plt.close(fig)

    return {
        "optimization_cost_comparison": str(cost_path),
        "optimization_daily_service_level": str(service_path),
        "optimization_daily_shortage": str(shortage_path),
    }
