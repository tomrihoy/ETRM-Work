import pytest
import numpy as np
from dataclass_definitions import Plant, FuelType, DispatchedPlant, DispatchedResult
from dispatch import dispatch_merit_order, find_merit_order


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_stack() -> list[Plant]:
    """Minimal stack with one cheap and one expensive plant."""
    return [
        Plant("Nuclear", FuelType.NUCLEAR, 50.0, 10.0),
        Plant("OCGT",    FuelType.OCGT,    50.0, 90.0),
    ]

@pytest.fixture
def wind_stack() -> list[Plant]:
    """Stack with wind and a peaker to cover remaining demand."""
    return [
        Plant("Wind",  FuelType.WIND, 40.0, 0.0),
        Plant("CCGT",  FuelType.CCGT_NEW, 60.0, 55.0),
    ]

@pytest.fixture
def full_uk_stack() -> list[Plant]:
    return [
        Plant("Solar",    FuelType.SOLAR,    10.0, 0.0),
        Plant("Wind 1",   FuelType.WIND,     20.0, 0.0),
        Plant("Wind 2",   FuelType.WIND,     15.0, 0.0),
        Plant("Nuclear",  FuelType.NUCLEAR,  30.0, 8.0),
        Plant("CCGT New", FuelType.CCGT_NEW, 20.0, 60.0),
        Plant("CCGT Old", FuelType.CCGT_NEW, 15.0, 75.0),
        Plant("OCGT 1",   FuelType.OCGT,     10.0, 90.0),
        Plant("OCGT 2",   FuelType.OCGT,     10.0, 92.0),
        Plant("OCGT 3",   FuelType.OCGT,     10.0, 96.0),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# dispatch_merit_order
# ══════════════════════════════════════════════════════════════════════════════

class TestDispatchMeritOrderReturnType:
    def test_returns_dispatched_result(self, simple_stack):
        result = dispatch_merit_order(simple_stack, 30.0)
        assert isinstance(result, DispatchedResult)

    def test_dispatched_plants_are_dispatched_plant_instances(self, simple_stack):
        result = dispatch_merit_order(simple_stack, 30.0)
        assert all(isinstance(dp, DispatchedPlant) for dp in result.dispatched_plant)


class TestDispatchMeritOrderMeritOrder:
    def test_cheaper_plant_dispatched_first(self, simple_stack):
        result = dispatch_merit_order(simple_stack, 30.0)
        # Only nuclear should be needed for 30 MW
        assert len(result.dispatched_plant) == 1
        assert result.dispatched_plant[0].plant.name == "Nuclear"

    def test_expensive_plant_dispatched_when_needed(self, simple_stack):
        result = dispatch_merit_order(simple_stack, 80.0)
        names = [dp.plant.name for dp in result.dispatched_plant]
        assert "Nuclear" in names
        assert "OCGT" in names

    def test_clearing_price_is_most_expensive_dispatched(self, simple_stack):
        result = dispatch_merit_order(simple_stack, 80.0)
        assert result.clearing_price == 90.0

    def test_clearing_price_when_only_cheap_plant_needed(self, simple_stack):
        result = dispatch_merit_order(simple_stack, 30.0)
        assert result.clearing_price == 10.0


class TestDispatchMeritOrderCapacity:
    def test_cheap_plant_fully_dispatched_before_expensive(self, simple_stack):
        result = dispatch_merit_order(simple_stack, 80.0)
        nuclear = next(dp for dp in result.dispatched_plant if dp.plant.name == "Nuclear")
        assert nuclear.dispatched_mw == 50.0

    def test_partial_dispatch_of_marginal_plant(self, simple_stack):
        # Nuclear (50 MW) is cheap, demand is 70 MW so OCGT covers last 20 MW
        result = dispatch_merit_order(simple_stack, 70.0)
        ocgt = next(dp for dp in result.dispatched_plant if dp.plant.name == "OCGT")
        assert ocgt.dispatched_mw == 20.0

    def test_demand_exactly_met_unmet_is_zero(self, simple_stack):
        result = dispatch_merit_order(simple_stack, 100.0)
        assert result.unmet_demand_mw == pytest.approx(0.0)

    def test_unmet_demand_when_capacity_insufficient(self, simple_stack):
        result = dispatch_merit_order(simple_stack, 150.0)
        assert result.unmet_demand_mw == pytest.approx(50.0)


class TestDispatchMeritOrderEdgeCases:
    def test_zero_demand_returns_no_dispatched_plants(self, simple_stack):
        result = dispatch_merit_order(simple_stack, 0.0)
        assert result.dispatched_plant == []

    def test_zero_demand_clearing_price_is_zero(self, simple_stack):
        result = dispatch_merit_order(simple_stack, 0.0)
        assert result.clearing_price == 0.0

    def test_total_cost_is_correct(self, simple_stack):
        # Nuclear 50 MW + OCGT 30 MW, both paid at clearing price of £90
        result = dispatch_merit_order(simple_stack, 80.0)
        expected_cost = 80.0 * 90.0  # total dispatched MW * clearing price
        assert result.total_cost == pytest.approx(expected_cost)

    def test_single_plant_stack(self):
        stack = [Plant("Nuclear", FuelType.NUCLEAR, 100.0, 10.0)]
        result = dispatch_merit_order(stack, 50.0)
        assert result.dispatched_plant[0].dispatched_mw == 50.0
        assert result.clearing_price == 10.0


# ══════════════════════════════════════════════════════════════════════════════
# find_merit_order
# ══════════════════════════════════════════════════════════════════════════════

class TestFindMeritOrder:
    def test_returns_list_of_dispatched_results(self, simple_stack):
        demand = np.array([30.0, 50.0, 80.0])
        wind_cf = np.ones(3)
        results = find_merit_order(simple_stack, demand, wind_cf)
        assert all(isinstance(r, DispatchedResult) for r in results)

    def test_result_length_matches_demand(self, simple_stack):
        demand = np.array([30.0, 50.0, 80.0])
        wind_cf = np.ones(3)
        results = find_merit_order(simple_stack, demand, wind_cf)
        assert len(results) == len(demand)

    def test_higher_demand_gives_higher_clearing_price(self, full_uk_stack):
        demand = np.array([30.0, 100.0])
        wind_cf = np.zeros(2)
        results = find_merit_order(full_uk_stack, demand, wind_cf)
        assert results[1].clearing_price >= results[0].clearing_price


class TestFindMeritOrderWindCF:
    def test_wind_cf_zero_reduces_wind_capacity_to_zero(self, wind_stack):
        demand = np.array([20.0])
        wind_cf = np.array([0.0])
        results = find_merit_order(wind_stack, demand, wind_cf)
        wind_plants = [
            dp for dp in results[0].dispatched_plant
            if dp.plant.fuel_type == FuelType.WIND
        ]
        assert all(dp.dispatched_mw == 0.0 for dp in wind_plants)

    def test_wind_cf_one_uses_full_wind_capacity(self, wind_stack):
        demand = np.array([40.0])
        wind_cf = np.array([1.0])
        results = find_merit_order(wind_stack, demand, wind_cf)
        wind_plants = [
            dp for dp in results[0].dispatched_plant
            if dp.plant.fuel_type == FuelType.WIND
        ]
        assert all(dp.dispatched_mw == pytest.approx(40.0) for dp in wind_plants)

    def test_non_wind_plants_unaffected_by_wind_cf(self, wind_stack):
        # Run with two different wind CFs, CCGT capacity should be unchanged
        demand = np.array([80.0])
        results_low  = find_merit_order(wind_stack, demand, np.array([0.0]))
        results_high = find_merit_order(wind_stack, demand, np.array([1.0]))
        ccgt_low  = next(dp for dp in results_low[0].dispatched_plant  if dp.plant.fuel_type == FuelType.CCGT_NEW)
        ccgt_high = next(dp for dp in results_high[0].dispatched_plant if dp.plant.fuel_type == FuelType.CCGT_NEW)
        # With high wind CF, CCGT should dispatch less
        assert ccgt_high.dispatched_mw < ccgt_low.dispatched_mw

    def test_wind_cf_varies_per_period(self, wind_stack):
        demand = np.array([40.0, 40.0])
        wind_cf = np.array([0.0, 1.0])
        results = find_merit_order(wind_stack, demand, wind_cf)
        wind_0 = next(dp for dp in results[0].dispatched_plant if dp.plant.fuel_type == FuelType.WIND)
        wind_1 = next(dp for dp in results[1].dispatched_plant if dp.plant.fuel_type == FuelType.WIND)
        assert wind_0.dispatched_mw < wind_1.dispatched_mw