from typing import Dict


def production_cost(produced_by_factory: Dict[str, float], unit_costs: Dict[str, float]) -> float:
    return sum(max(0.0, produced_by_factory.get(fid, 0.0)) * unit_costs[fid] for fid in unit_costs)


def holding_cost(inventory_by_wh: Dict[str, float], unit_holding_costs: Dict[str, float]) -> float:
    return sum(max(0.0, inventory_by_wh.get(wid, 0.0)) * unit_holding_costs[wid] for wid in unit_holding_costs)


def transport_cost(shipments: Dict[tuple, float], unit_transport_costs: Dict[tuple, float]) -> float:
    total = 0.0
    for edge, qty in shipments.items():
        unit = unit_transport_costs.get(edge, 0.0)
        total += max(0.0, qty) * unit
    return total


def unmet_demand_penalty(unmet_by_retailer: Dict[str, float], penalty_per_unit: Dict[str, float]) -> float:
    return sum(max(0.0, unmet_by_retailer.get(rid, 0.0)) * penalty_per_unit[rid] for rid in penalty_per_unit)


def total_cost(prod_cost: float, hold_cost: float, trans_cost: float, penalty_cost: float) -> float:
    return prod_cost + hold_cost + trans_cost + penalty_cost


def reward_from_cost(total: float) -> float:
    return -total


