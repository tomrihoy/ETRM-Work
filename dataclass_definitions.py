from dataclasses import dataclass, replace
from enum import StrEnum


# Dispatch
class FuelType(StrEnum):
    NUCLEAR = "nuclear"
    WIND = "wind"
    SOLAR = "solar"
    OCGT = "ocgt"
    CCGT_NEW = "ccgt_new"
    CCGT_OLD = "ccgt_old"

@dataclass(frozen=True, slots=True)
class Plant:
    name: str
    fuel_type: FuelType
    capacity_mw: float
    marginal_cost: float

@dataclass(frozen=True)
class DispatchedPlant:
    plant: Plant
    dispatched_mw: float

@dataclass(frozen=True)
class DispatchedResult:
    dispatched_plant: list[DispatchedPlant]
    clearing_price: float
    unmet_demand_mw: float
    total_cost: float


class OutputFormat(StrEnum):
    STDOUT = "stdout"
    CSV = "csv"

