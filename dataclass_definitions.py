from dataclasses import dataclass, replace
from enum import StrEnum

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
    capacity_mw: float
    fuel_type: FuelType
    marginal_cost: float


class OutputFormat(StrEnum):
    STDOUT = "stdout"
    CSV = "csv"

