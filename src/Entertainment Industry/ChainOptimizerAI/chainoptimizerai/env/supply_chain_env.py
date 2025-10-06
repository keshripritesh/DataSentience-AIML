from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json
import argparse
import numpy as np

from .entities import Factory, Warehouse, Retailer, TransportEdge, build_lookup_by_id
from .cost_functions import (
    production_cost,
    holding_cost,
    transport_cost,
    unmet_demand_penalty,
    total_cost,
    reward_from_cost,
)


@dataclass
class SupplyChainState:
    step: int
    inventory_warehouse: Dict[str, float]
    inventory_retailer: Dict[str, float]
    demand_this_step: Dict[str, float]


class SupplyChainEnv:
    def __init__(
        self,
        factories: List[Factory],
        warehouses: List[Warehouse],
        retailers: List[Retailer],
        edges_fw: List[TransportEdge],
        edges_wr: List[TransportEdge],
        demand_schedule: List[Dict[str, float]],
        random_seed: Optional[int] = 0,
    ) -> None:
        self.factories = factories
        self.warehouses = warehouses
        self.retailers = retailers
        self.edges_fw = edges_fw
        self.edges_wr = edges_wr
        self.demand_schedule = demand_schedule
        self.random = np.random.default_rng(random_seed)

        self.fact_by_id = build_lookup_by_id(factories)
        self.wh_by_id = build_lookup_by_id(warehouses)
        self.ret_by_id = build_lookup_by_id(retailers)

        self.unit_prod_cost = {f.id: f.unit_production_cost for f in factories}
        self.unit_hold_cost = {w.id: w.unit_holding_cost for w in warehouses}
        self.penalty_unmet = {r.id: r.penalty_unmet_demand for r in retailers}
        self.unit_transport_cost_fw = {(e.source_id, e.target_id): e.unit_transport_cost for e in edges_fw}
        self.unit_transport_cost_wr = {(e.source_id, e.target_id): e.unit_transport_cost for e in edges_wr}

        self.max_steps = len(demand_schedule)
        self.reset()

    def reset(self) -> SupplyChainState:
        self.step_index = 0
        self.inventory_warehouse = {w.id: 0.0 for w in self.warehouses}
        self.inventory_retailer = {r.id: 0.0 for r in self.retailers}
        demand = self.demand_schedule[self.step_index] if self.max_steps > 0 else {r.id: 0.0 for r in self.retailers}
        return SupplyChainState(
            step=self.step_index,
            inventory_warehouse=self.inventory_warehouse.copy(),
            inventory_retailer=self.inventory_retailer.copy(),
            demand_this_step=demand.copy(),
        )

    def step(
        self,
        production_by_factory: Dict[str, float],
        shipments_fw: Dict[Tuple[str, str], float],
        shipments_wr: Dict[Tuple[str, str], float],
    ) -> Tuple[SupplyChainState, float, bool, Dict]:
        # Clamp production by capacity and non-negativity
        for fid, qty in list(production_by_factory.items()):
            capacity = self.fact_by_id[fid].production_capacity
            production_by_factory[fid] = float(max(0.0, min(qty, capacity)))

        # Inflow to warehouses from factories
        inflow_wh: Dict[str, float] = {w.id: 0.0 for w in self.warehouses}
        for (src, dst), qty in list(shipments_fw.items()):
            qty = max(0.0, float(qty))
            if (src, dst) not in self.unit_transport_cost_fw:
                continue
            inflow_wh[dst] += qty

        # Update warehouse inventory with capacity constraints
        for wid, add_qty in inflow_wh.items():
            cap = self.wh_by_id[wid].storage_capacity
            new_inv = min(self.inventory_warehouse[wid] + add_qty, cap)
            self.inventory_warehouse[wid] = new_inv

        # Shipments from warehouses to retailers limited by on-hand inventory
        inflow_r: Dict[str, float] = {r.id: 0.0 for r in self.retailers}
        for (src, dst), qty in list(shipments_wr.items()):
            if (src, dst) not in self.unit_transport_cost_wr:
                continue
            qty = max(0.0, float(qty))
            send_qty = min(qty, self.inventory_warehouse[src])
            self.inventory_warehouse[src] -= send_qty
            inflow_r[dst] += send_qty

        # Retailer inventory updated
        for rid, add_qty in inflow_r.items():
            self.inventory_retailer[rid] += add_qty

        # Demand satisfaction
        demand = self.demand_schedule[self.step_index]
        unmet: Dict[str, float] = {}
        for rid, d in demand.items():
            on_hand = self.inventory_retailer[rid]
            sold = min(on_hand, max(0.0, d))
            self.inventory_retailer[rid] -= sold
            unmet[rid] = max(0.0, d - sold)

        # Costs
        prod_c = production_cost(production_by_factory, self.unit_prod_cost)
        # Collect transport costs
        trans_fw_c = transport_cost(shipments_fw, self.unit_transport_cost_fw)
        trans_wr_c = transport_cost(shipments_wr, self.unit_transport_cost_wr)
        trans_c = trans_fw_c + trans_wr_c
        hold_c = holding_cost(self.inventory_warehouse, self.unit_hold_cost)
        pen_c = unmet_demand_penalty(unmet, self.penalty_unmet)
        tot_c = total_cost(prod_c, hold_c, trans_c, pen_c)
        reward = reward_from_cost(tot_c)

        # Next step
        self.step_index += 1
        done = self.step_index >= self.max_steps
        next_demand = (
            self.demand_schedule[self.step_index] if not done else {r.id: 0.0 for r in self.retailers}
        )
        state = SupplyChainState(
            step=self.step_index,
            inventory_warehouse=self.inventory_warehouse.copy(),
            inventory_retailer=self.inventory_retailer.copy(),
            demand_this_step=next_demand.copy(),
        )
        info = {
            "cost_breakdown": {
                "production": prod_c,
                "transport": trans_c,
                "holding": hold_c,
                "penalty": pen_c,
                "total": tot_c,
            },
            "unmet_demand": unmet,
        }
        return state, reward, done, info


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_env_from_config(config: Dict, random_seed: Optional[int] = 0) -> SupplyChainEnv:
    factories = [Factory(**f) for f in config["factories"]]
    warehouses = [Warehouse(**w) for w in config["warehouses"]]
    retailers = [Retailer(**r) for r in config["retailers"]]
    edges_fw = [TransportEdge(**e) for e in config["edges_fw"]]
    edges_wr = [TransportEdge(**e) for e in config["edges_wr"]]
    demand_schedule = config["demand_schedule"]
    return SupplyChainEnv(
        factories=factories,
        warehouses=warehouses,
        retailers=retailers,
        edges_fw=edges_fw,
        edges_wr=edges_wr,
        demand_schedule=demand_schedule,
        random_seed=random_seed,
    )


def run_random_agent(env: SupplyChainEnv, episodes: int = 1) -> None:
    for ep in range(episodes):
        state = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            # Random feasible actions
            prod = {f.id: float(env.random.uniform(0, f.production_capacity)) for f in env.factories}
            ship_fw = {}
            for e in env.edges_fw:
                ship_fw[(e.source_id, e.target_id)] = float(max(0.0, env.random.uniform(0, 5)))
            ship_wr = {}
            for e in env.edges_wr:
                ship_wr[(e.source_id, e.target_id)] = float(max(0.0, env.random.uniform(0, 5)))
            state, reward, done, info = env.step(prod, ship_fw, ship_wr)
            ep_reward += reward
        print(f"Episode {ep+1} total reward: {ep_reward:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SupplyChainEnv with a random policy")
    parser.add_argument("--config", type=str, default="chainoptimizerai/examples/small_chain.json")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()
    cfg = load_config(args.config)
    env = build_env_from_config(cfg)
    run_random_agent(env, episodes=args.episodes)


if __name__ == "__main__":
    main()


