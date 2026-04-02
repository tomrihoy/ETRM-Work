import numpy as np
import matplotlib.pyplot as plt
from dataclass_definitions import Plant, FuelType, DispatchedPlant, DispatchedResult


def dispatch_merit_order(plants: list[Plant], demand_mw: np.ndarray):
    
    sorted_uk_stack = sorted(plants, key = lambda p: p.marginal_cost)

    dispatched_plants = []
    remaining_demand = demand_mw
    clearing_price = 0

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


def find_merit_order(plant_stack: list[Plant], demand: np.ndarray):
    dispatched_results = []
    for demand_val in demand:
        dispatched_results.append(dispatch_merit_order(plants=plant_stack, demand_mw=demand_val))
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

    x=np.arange(1,49)
    DEMAND = np.sin(x*(2*np.pi/48))*60 + 70
    results = find_merit_order(UK_STACK, DEMAND)

    clearing_prices = [r.clearing_price for r in results]

    plt.plot(x, clearing_prices)
    plt.show()
