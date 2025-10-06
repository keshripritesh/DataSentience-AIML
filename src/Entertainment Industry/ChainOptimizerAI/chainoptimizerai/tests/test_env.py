import json
from chainoptimizerai.env.supply_chain_env import build_env_from_config


def build_env():
    cfg = {
        "factories": [{"id": "F1", "production_capacity": 10, "unit_production_cost": 2.0}],
        "warehouses": [{"id": "W1", "storage_capacity": 50, "unit_holding_cost": 0.2}],
        "retailers": [{"id": "R1", "penalty_unmet_demand": 5.0}],
        "edges_fw": [{"source_id": "F1", "target_id": "W1", "distance": 10, "unit_transport_cost": 0.5}],
        "edges_wr": [{"source_id": "W1", "target_id": "R1", "distance": 5, "unit_transport_cost": 0.3}],
        "demand_schedule": [{"R1": 5}, {"R1": 5}]
    }
    return build_env_from_config(cfg)


def test_reset_and_step_shapes():
    env = build_env()
    s = env.reset()
    assert s.step == 0
    prod = {"F1": 5}
    ship_fw = {("F1", "W1"): 5}
    ship_wr = {("W1", "R1"): 5}
    s2, reward, done, info = env.step(prod, ship_fw, ship_wr)
    assert isinstance(reward, float)
    assert "cost_breakdown" in info
    assert s2.step == 1


def test_costs_positive():
    env = build_env()
    env.reset()
    s2, reward, done, info = env.step({"F1": 0}, {}, {})
    # No production, no transport, but unmet demand penalty
    assert info["cost_breakdown"]["penalty"] > 0
    assert info["cost_breakdown"]["total"] > 0
    assert reward < 0


