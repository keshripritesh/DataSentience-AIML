from dataclasses import dataclass
from typing import Dict


@dataclass
class Factory:
    id: str
    production_capacity: float
    unit_production_cost: float


@dataclass
class Warehouse:
    id: str
    storage_capacity: float
    unit_holding_cost: float


@dataclass
class Retailer:
    id: str
    penalty_unmet_demand: float
    # demand is provided per step, not stored here


@dataclass
class TransportEdge:
    source_id: str
    target_id: str
    distance: float
    unit_transport_cost: float


def build_lookup_by_id(items) -> Dict[str, object]:
    return {item.id: item for item in items}


