from __future__ import annotations

from typing import Dict, Tuple
import numpy as np


def greedy_lp_baseline(env) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]:
    # Heuristic: produce to meet total demand of this step, ship directly if possible via warehouses
    demand = env.demand_schedule[env.step_index]
    total_demand = float(sum(max(0.0, d) for d in demand.values()))
    # Distribute production across factories up to capacity
    prod = {}
    remaining = total_demand
    for f in env.factories:
        qty = float(min(remaining, f.production_capacity))
        prod[f.id] = qty
        remaining -= qty
    # Ship evenly from first factory to all warehouses, then to retailers proportionally to demand
    ship_fw = {}
    if env.edges_fw:
        first_factory = env.edges_fw[0].source_id
        per_wh = total_demand / max(1, len(env.warehouses))
        for e in env.edges_fw:
            if e.source_id == first_factory:
                ship_fw[(e.source_id, e.target_id)] = float(per_wh)
    ship_wr = {}
    total_wh = max(1.0, float(len(env.warehouses)))
    for e in env.edges_wr:
        demand_ratio = max(0.0, demand.get(e.target_id, 0.0)) / max(1e-6, total_demand)
        ship_wr[(e.source_id, e.target_id)] = float(demand_ratio * (total_demand / total_wh))
    return prod, ship_fw, ship_wr


