from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .optimization_solver import (
    PolicyRunArtifacts,
    _daily_panel_to_dict,
    _inventory_state_from_frame,
    _prepare_route_inputs,
    _series_from_frame,
    simulate_realized_day,
)


@dataclass
class RuleBasedDayPlan:
    procurement: pd.DataFrame
    inventory: pd.DataFrame
    shortage: pd.DataFrame
    transfer: pd.DataFrame
    planned_metrics: dict


def _build_rule_based_day_plan(
    date: pd.Timestamp,
    demand_map: dict[tuple[int, int], float],
    current_inventory: dict[tuple[int, int], float],
    cluster_universe: list[int],
    warehouse_universe: list[int],
    cluster_prices: dict[int, float],
    procurement_eligibility: dict[int, float],
    route_feasibility: dict[tuple[int, int], int],
    route_time: dict[tuple[int, int], float],
    constants: dict,
) -> RuleBasedDayPlan:
    incoming = {(s, j): 0.0 for s in cluster_universe for j in warehouse_universe}
    outgoing = {(s, j): 0.0 for s in cluster_universe for j in warehouse_universe}
    procurement_map = {(s, j): 0.0 for s in cluster_universe for j in warehouse_universe}
    transfer_rows: list[dict] = []

    for s in cluster_universe:
        deficits = {
            j: max(float(demand_map[(s, j)]) - float(current_inventory[(s, j)]), 0.0)
            for j in warehouse_universe
        }
        surpluses = {
            j: max(float(current_inventory[(s, j)]) - float(demand_map[(s, j)]), 0.0)
            for j in warehouse_universe
        }

        deficit_order = sorted(
            [j for j, deficit in deficits.items() if deficit > 1e-8],
            key=lambda j: (-deficits[j], j),
        )
        for destination in deficit_order:
            remaining_deficit = deficits[destination]
            if remaining_deficit <= 1e-8:
                continue

            donor_order = sorted(
                [
                    source
                    for source in warehouse_universe
                    if source != destination
                    and surpluses[source] > 1e-8
                    and int(route_feasibility[(source, destination)]) == 1
                ],
                key=lambda source: (float(route_time[(source, destination)]), source),
            )
            for source in donor_order:
                if remaining_deficit <= 1e-8:
                    break
                transfer_units = min(remaining_deficit, surpluses[source])
                if transfer_units <= 1e-8:
                    continue
                surpluses[source] -= transfer_units
                remaining_deficit -= transfer_units
                incoming[(s, destination)] += transfer_units
                outgoing[(s, source)] += transfer_units
                transfer_rows.append(
                    {
                        "date": date,
                        "sku_cluster_ID": s,
                        "source_warehouse": source,
                        "destination_warehouse": destination,
                        "transfer_units": transfer_units,
                        "route_feasible": int(route_feasibility[(source, destination)]),
                        "lead_time_hours": float(route_time[(source, destination)]),
                        "transfer_cost": float(constants["k"])
                        * float(cluster_prices.get(s, 0.0))
                        * float(route_time[(source, destination)])
                        * transfer_units,
                    }
                )

    procurement_rows: list[dict] = []
    inventory_rows: list[dict] = []
    shortage_rows: list[dict] = []
    shortage_cost = 0.0
    holding_cost = 0.0

    for s in cluster_universe:
        price = float(cluster_prices.get(s, 0.0))
        for j in warehouse_universe:
            key = (s, j)
            available_before_procurement = (
                float(current_inventory[key]) + float(incoming[key]) - float(outgoing[key])
            )
            demand = float(demand_map[key])
            remaining_deficit = max(demand - available_before_procurement, 0.0)
            if float(procurement_eligibility.get(j, 0.0)) >= 1.0:
                procurement_units = remaining_deficit
            else:
                procurement_units = 0.0
            procurement_map[key] = procurement_units

            available = available_before_procurement + procurement_units
            planned_shortage = max(demand - available, 0.0)
            planned_ending_inventory = max(available - demand, 0.0)

            shortage_cost += planned_shortage * price * float(constants["u"])
            holding_cost += planned_ending_inventory * price * float(constants["h"])

            procurement_rows.append(
                {
                    "date": date,
                    "sku_cluster_ID": s,
                    "warehouse": j,
                    "procurement_units": procurement_units,
                }
            )
            inventory_rows.append(
                {
                    "date": date,
                    "sku_cluster_ID": s,
                    "warehouse": j,
                    "planned_ending_inventory": planned_ending_inventory,
                }
            )
            shortage_rows.append(
                {
                    "date": date,
                    "sku_cluster_ID": s,
                    "warehouse": j,
                    "planned_shortage_units": planned_shortage,
                }
            )

    procurement_df = pd.DataFrame(procurement_rows)
    inventory_df = pd.DataFrame(inventory_rows)
    shortage_df = pd.DataFrame(shortage_rows)
    transfer_df = pd.DataFrame(transfer_rows)

    demand_total = float(sum(demand_map.values()))
    shortage_units = float(shortage_df["planned_shortage_units"].sum())
    fulfilled_units = demand_total - shortage_units
    transfer_cost = float(transfer_df["transfer_cost"].sum()) if not transfer_df.empty else 0.0

    return RuleBasedDayPlan(
        procurement=procurement_df,
        inventory=inventory_df,
        shortage=shortage_df,
        transfer=transfer_df,
        planned_metrics={
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
    )


def run_rule_based_policy(
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
    route_feasibility, route_time, _ = _prepare_route_inputs(
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

        planned = _build_rule_based_day_plan(
            date=date,
            demand_map=planning_demand_map,
            current_inventory=current_inventory,
            cluster_universe=cluster_universe,
            warehouse_universe=warehouse_universe,
            cluster_prices=cluster_prices,
            procurement_eligibility=procurement_eligibility,
            route_feasibility=route_feasibility,
            route_time=route_time,
            constants=constants,
        )
        realized = simulate_realized_day(
            date=date,
            current_inventory=current_inventory,
            actual_demand_map=actual_demand_map,
            procurement_df=planned.procurement,
            transfer_df=planned.transfer,
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
                **planned.planned_metrics,
                **realized["realized_metrics"],
                "cost_gap_vs_plan": realized["realized_metrics"]["realized_total_cost"]
                - planned.planned_metrics["planned_total_cost"],
            }
        )
        procurement_rows.append(planned.procurement.assign(policy_name=policy_name, planning_basis=planning_basis))
        planned_inventory_rows.append(planned.inventory.assign(policy_name=policy_name, planning_basis=planning_basis))
        planned_shortage_rows.append(planned.shortage.assign(policy_name=policy_name, planning_basis=planning_basis))
        realized_inventory_rows.append(realized["realized_inventory"].assign(policy_name=policy_name, planning_basis=planning_basis))
        realized_shortage_rows.append(realized["realized_shortage"].assign(policy_name=policy_name, planning_basis=planning_basis))
        if not planned.transfer.empty:
            transfer_rows.append(planned.transfer.assign(policy_name=policy_name, planning_basis=planning_basis))

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
