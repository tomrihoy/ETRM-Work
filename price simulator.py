# Step 2: Initial Python Project - Price Curve Generator

# Let's evaluate your Python level and start building the foundation for your market data simulator. Publish this in your GitHub repo. 

# Build a simple Python 3.14 script that generates something that looks like a "price curve" for a given day and granularity (hourly, half-hourly, 15min). At this stage, it's essentially random numbers - we'll make it realistic later. 

# Requirements: 

# 1. A function which takes as arguments a date  (e.g. '2026-01-29'), a granularity (e.g. 'hourly', 'half-hourly', '15min') and returns a list of random prices (e.g. could be floats between 50.0 and 150.0 - not so important for now)

# 2. Explore 2-3 different approaches to generating random numbers. E.g.
#     - Python stdlib random module                                                                                                        
#     - NumPy's random number generation                                                                                            
#     - Perhaps another method you can think of 

# 3. Compare performance: Write a simple benchmark that generates 30,000 price curves using each method and measures execution time. Which is fastest? Why do you think that is? How does performance compare for a small number of curves (e.g. just 10 curves or fewer) Why do you think that is? 

# 4. Output: Print the curve to console (or write to CSV) so you can visually inspect it 

# 5. Commit and push to your repo with a clear commit message 

# What I'm looking for at this stage is:                                                                                                    

#   - Clean, readable code                                                                                                                      
#   - Proper function structure                                                                                                                 
#   - Performance measurement approach                                                                                             
#   - Your analysis of the results                                                                                                              
                                                                                                                                                           
# Don't overthink it - this is deliberately simple. We can later iterate in Pull Requests to add: more realistic intraday patterns, commodity-specific behaviour, geographic variations, supply/demand factors, eventually a full generation stack simulation. One step at a time. 

import random
import numpy as np
import time
import pandas as pd
from datetime import datetime
from time import perf_counter


def generate_price_curves(start_date, end_date, granularity, method, 
                          lower_bound=70, upper_bound=110, save_to_csv=False,
                           output_dir=None):
    """
    Generate price curves for a given date range and granularity using the specified method.
    
    Args:
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.
        granularity (int): The granularity of the price curve in minutes.
        method (str): The method to use for generating prices.

    Output:
        price_curve (dataframe): Dataframe with columns for time and price.
    """

    if method not in [random_curve_generator, numpy_curve_generator, autoc_curve_generator]:
        raise ValueError('method must be empty or Equal to "random", "numpy" or "autoc"')

    # Generate time series
    start_date_dt = pd.to_datetime(start_date, format='%Y-%m-%d')
    end_date_dt = pd.to_datetime(end_date, format='%Y-%m-%d')
    granularity_dt = granularity+'min'
    datetimes = pd.date_range(start=start_date_dt, end=end_date_dt, freq = granularity_dt)

    len_datapoints = len(datetimes)

    price_curve = method(lower_bound, upper_bound, len_datapoints, datetimes)
    
    if save_to_csv:
        save_price_curve(price_curve, output_dir, method)
    else:
        return price_curve
    

def random_curve_generator(lower_bound, upper_bound, len_datapoints, datetimes):
    prices = [random.uniform(lower_bound, upper_bound) for _ in range(len_datapoints)]
    price_curve=pd.DataFrame(zip(datetimes, prices), columns=['Datetime', 'Prices'])
    return price_curve

def numpy_curve_generator(lower_bound, upper_bound, len_datapoints, datetimes):
    prices = np.random.uniform(low = lower_bound, high = upper_bound,size = len_datapoints)
    price_curve=pd.DataFrame(zip(datetimes, prices), columns=['Datetime', 'Prices']) 
    return price_curve

def autoc_curve_generator(lower_bound, upper_bound, len_datapoints, datetimes):
    start_val = (lower_bound+upper_bound)/2
    prices = np.full(len_datapoints, np.nan)

    prices[0]=start_val
    for idx in range(len_datapoints-1):
        val = np.random.normal(loc=prices[idx], scale=0.1)
        # Prevents random walk from straying out of bounds
        val = np.clip(val, lower_bound, upper_bound)
        prices[idx+1]=val
    price_curve=pd.DataFrame(zip(datetimes, prices), columns=['Datetime', 'Prices'])
    return price_curve

def save_price_curve(price_curve, output_dir, method):
    if output_dir:
        filename = output_dir+'/'+method.__name__+str(datetime.now())+'.csv'
    else:
        filename = method.__name__+str(datetime.now())+'.csv'
    price_curve.to_csv(filename)



def benchmark(method, iterations):
      start = perf_counter()
      for _ in range(iterations):
          df = generate_price_curves(
              start_date='2020-01-01', end_date='2020-01-02',
              granularity='30', method=method
          )
      elapsed = perf_counter() - start
      return {'method': method.__name__, 'iterations': iterations, 'elapsed': elapsed}
    

if __name__=='__main__':
    results_random_10 = benchmark(random_curve_generator, 10)
    # results_numpy_10 = benchmark('numpy', 10)
    # results_autoc_10 = benchmark('autoc', 10)
    # results_random_100 = benchmark('random', 100)
    # results_numpy_100 = benchmark('numpy', 100)
    # results_autoc_100 = benchmark('autoc', 100)
    # results_random_30000 = benchmark('random', 30000)
    # results_numpy_30000 = benchmark('numpy', 30000)
    # results_autoc_30000 = benchmark('autoc', 30000)

    # Results:
    # Random, 10 iterations: 0.029s
    # Numpy, 10 iterations: 0.077s
    # Autoc, 10 iterations: 0.033s
    # Random, 10 iterations: 0.445s
    # Numpy, 10 iterations: 0.243s
    # Autoc, 10 iterations: 0.214s
    # Random, 10 iterations: 58.973s
    # Numpy, 10 iterations: 39.678s
    # Autoc, 10 iterations: 41.655s

    # Answers:
    # Numpy is fastest for larger numbers of iterations as it uses vectorised 
    # operations which are quicker to do at scale.
    # Random seem to be fastest for lower numbers of iterations - not sure 
    # why this is, perhaps that many of iterations is insufficient for the trends
    # to show themselves (i.e. law of large numbers) when combined with other 
    # random factors which affect runtime performance.

            
    df = generate_price_curves(start_date = '2020-01-01', end_date =  '2021-12-31', 
                            granularity = '30', method=autoc_curve_generator)

