from dataclasses import dataclass

@dataclass
class Plant(frozen=True, slots=True):
    name: str
    capacity_mw: float
    fuel_type: str
    marginal_cost: float
