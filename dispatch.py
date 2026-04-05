from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

from dataclass_definitions import DispatchedPlant, DispatchedResult, FuelType, Plant


def dispatch_merit_order(plants: list[Plant], demand_mw: float)->DispatchedResult:

    sorted_uk_stack = sorted(plants, key = lambda p: p.marginal_cost)

    dispatched_plants = []
    remaining_demand = demand_mw
    clearing_price: float = 0

    for plant in sorted_uk_stack:
        if remaining_demand<=0:
            break

        dispatched_mw = min(plant.capacity_mw, remaining_demand)

        dispatched_plants.append(DispatchedPlant(plant=plant, dispatched_mw=dispatched_mw))

        remaining_demand -=dispatched_mw
        clearing_price = plant.marginal_cost

    dispatched_mw = sum(dp.dispatched_mw for dp in dispatched_plants)
    total_cost = dispatched_mw*clearing_price


    dispatched_result = DispatchedResult(dispatched_plants, clearing_price, remaining_demand, total_cost)

    return dispatched_result


def find_merit_order(plant_stack: list[Plant], demand: np.ndarray, wind_cf: np.ndarray)->list[DispatchedResult]:
    dispatched_results = []
    for demand_val, wind_cf_val in zip(demand, wind_cf, strict=True):
        adjusted_stack = [
        replace(plant, capacity_mw=plant.capacity_mw * wind_cf_val)
        if plant.fuel_type == FuelType.WIND
        else plant
        for plant in plant_stack
        ]
        dispatched_results.append(dispatch_merit_order(plants=adjusted_stack, demand_mw=demand_val))
    return dispatched_results


if __name__=='__main__':
    UK_STACK =[
    Plant("Solar", FuelType.SOLAR, 10, 0),
    Plant("Wind 1", FuelType.WIND, 20, 0),
    Plant("Wind 2", FuelType.WIND, 15, 0),
    Plant("Nuclear", FuelType.NUCLEAR, 30, 8),
    Plant("CCGT New", FuelType.CCGT_NEW, 20, 60),
    Plant("CCGT Old", FuelType.CCGT_NEW, 15, 75),
    Plant("OCGT 1", FuelType.OCGT, 10, 90),
    Plant("OCGT 2", FuelType.OCGT, 10, 92),
    Plant("OCGT 3", FuelType.OCGT, 10, 96)]

    def generate_autocorrelated(n: int, phi: float = 0.95, seed: int | None = None) -> np.ndarray:

        rng = np.random.default_rng(seed)

        noise = rng.normal(0, 1, n)
        series = np.zeros(n)
        series[0] = noise[0]

        for i in range(1, n):
            series[i] = phi * series[i-1] + np.sqrt(1 - phi**2) * noise[i]

        # Normalise to [0, 1]
        series = (series - series.min()) / (series.max() - series.min())

        return series

    x=np.arange(1,49)
    demand = np.sin(x*(2*np.pi/48))*40 + 50
    wind_cf = generate_autocorrelated(len(x), 0.8)

    results = find_merit_order(UK_STACK, demand, wind_cf)

    clearing_prices = [r.clearing_price for r in results]
    fig, ax = plt.subplots(3,1)
    ax[0].plot(x, clearing_prices, label='Clearing Price (£/MWh)', c='r')
    ax[0].set_ylabel('Clearing Price (£/MWh)')
    ax[0].legend()

    ax[1].plot(x, demand, label='Power Demand', c='g')
    ax[1].set_ylabel('Demand (MW)')
    ax[1].legend()

    ax[2].plot(x, wind_cf, label='Wind Capacity Factor (%)', c='g')
    ax[2].set_ylabel('Wind Capacity Factor (%)')
    ax[2].legend()


    plt.show()
