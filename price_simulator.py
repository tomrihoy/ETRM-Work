import random
from datetime import datetime
from time import perf_counter
import numpy as np
import pandas as pd

def generate_price_curves(start_date, end_date, granularity, method, 
                          lower_bound=70, upper_bound=110, save_to_csv=False,
                           output_dir=None, **method_kwargs):
    """
    Generate price curves for a given date range and granularity using the specified method.
    """

    # Generate time series
    start_date_dt = pd.to_datetime(start_date, format='%Y-%m-%d')
    end_date_dt = pd.to_datetime(end_date, format='%Y-%m-%d')
    granularity_dt = granularity+'min'
    datetimes = pd.date_range(start=start_date_dt, end=end_date_dt, freq = granularity_dt)

    len_datapoints = len(datetimes)

    price_curve = method(lower_bound, upper_bound, len_datapoints, datetimes, **method_kwargs)
    
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

def autoc_curve_generator(lower_bound, upper_bound, len_datapoints, datetimes, scale=0.1):
    
    start_val = (lower_bound+upper_bound)/2
    prices = np.full(len_datapoints, np.nan)

    prices[0]=start_val
    for idx in range(len_datapoints-1):
        val = np.random.normal(loc=prices[idx], scale=scale)
        # Prevents random walk from straying out of bounds
        val = np.clip(val, lower_bound, upper_bound)
        prices[idx+1]=val
    price_curve=pd.DataFrame(zip(datetimes, prices), columns=['Datetime', 'Prices'])
    return price_curve

def ornstein_ulenbeck(lower_bound, upper_bound, len_datapoints, datetimes, scale=0.1, mu=None, theta=0.1):
    if not mu:
        mu =(upper_bound-lower_bound)/2
    prices = np.full(len_datapoints, np.nan)
    prices[0]=mu
    diff = datetimes.to_series().diff()
    delta_T = diff.mode()[0] / pd.Timedelta(hours=1)
    for t in range(1,len_datapoints):
        dW = np.sqrt(delta_T) * np.random.normal(0, 1)
        prices[t] = prices[t-1] + theta * (mu - prices[t-1]) * delta_T + scale * dW
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
            
    df = generate_price_curves(start_date = '2020-01-01', end_date =  '2021-12-31', 
                            granularity = '30', method=ornstein_ulenbeck, scale=0.2, theta=0.001)
    df['Prices'].plot()

