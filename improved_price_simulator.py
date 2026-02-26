import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def intraday_curve(
    sp_to_model,
    wde_multiplier,
    a_1=10,
    a_2=15,
    sigma_1=4,
    sigma_2=4,
    mean_1=16,
    mean_2=32):
    ''' Model intraday curve with gtwo gaussian peaks'''
    peak_1 = a_1 * np.exp(-(sp_to_model - mean_1) ** 2 / (2 * sigma_1 ** 2))
    peak_2 = a_2 * np.exp(-(sp_to_model - mean_2) ** 2 / (2 * sigma_2 ** 2))

    return wde_multiplier * (peak_1 + peak_2)


def seasonality_curve(day_of_year, A=30):
    ''' Model seasonal baseline variation with cosine'''
    return A * np.cos((2* np.pi / 365) * day_of_year)


def settlement_period(index):
    '''Convert datetimes to settlement periods'''
    return (index.hour * 60 + index.minute) // 30 + 1

def week_day_end(
    index,
    wkd_mult=1,
    we_mult=0.6,
    wkd_sigma_1=4,
    wkd_sigma_2=4,
    we_sigma_1=5,
    we_sigma_2=5):
    is_weekend = index.dayofweek.isin([5, 6])
    ''' Adjust weekday/weekend peaks and peak spreads'''
    wde_mult = np.where(is_weekend, we_mult, wkd_mult)
    wde_sigma_1 = np.where(is_weekend, we_sigma_1, wkd_sigma_1)
    wde_sigma_2 = np.where(is_weekend, we_sigma_2, wkd_sigma_2)

    return wde_mult, wde_sigma_1, wde_sigma_2

def seasonal_sigma(time_series, wde_sigma, sigma_month_list):
    ''' Adjust peak spreads according to the month'''
    month_factors = np.array([sigma_month_list[m - 1] for m in time_series.month])
    return wde_sigma * month_factors


def ornstein_uhlenbeck(
    det_curve,
    dt=1/48,          
    theta=4.0,                   
    sigma_mult=5):
    ''' OU process for stochastic noise, volatility scales with determinstic price'''
    n_steps=len(det_curve)
    x = np.zeros(n_steps)
    x[0] = det_curve[0]
    print(sigma_mult)
    dW = np.random.normal(0, np.sqrt(dt), size=n_steps)

    for t in range(1, n_steps):
        x[t] = (
            x[t-1]
            + theta * (det_curve[t] - x[t-1]) * dt
            + sigma_mult*det_curve[t]/np.mean(det_curve) * dW[t]
        )

    return x


def generate_price_curve(
    start_date,
    end_date,
    intraday_kwargs=None,
    seasonality_kwargs=None,
    weekday_kwargs=None,
    sigma_month_list_1=None,
    sigma_month_list_2=None,
    ou_kwargs=None,
    base_price=70):
    ''' Generate synthetic prices with a detrminstic and stochastic component'''
    
    if intraday_kwargs is None:
        intraday_kwargs = {}

    if seasonality_kwargs is None:
        seasonality_kwargs = {}

    if weekday_kwargs is None:
        weekday_kwargs = {}

    if sigma_month_list_1 is None:
        sigma_month_list_1 = [1,1,1,1.1,1.2,1.2,1.2,1.2,1.1,1,1,1]

    if sigma_month_list_2 is None:
        sigma_month_list_2 = [1,1,1,1.2,1.3,1.3,1.3,1.3,1.2,1.1,1,1]

    if ou_kwargs is None:
        ou_kwargs={}
    time_series = pd.date_range(start_date, end_date, freq="30min")

    sp_array = settlement_period(time_series)

    wde_multiplier, wde_sigma_1, wde_sigma_2 = week_day_end(
        time_series,
        **weekday_kwargs)

    wde_seasonal_sigma_1 = seasonal_sigma(
        time_series,
        wde_sigma_1,
        sigma_month_list_1)

    wde_seasonal_sigma_2 = seasonal_sigma(
        time_series,
        wde_sigma_2,
        sigma_month_list_2)

    itd_curve = intraday_curve(
        sp_array,
        wde_multiplier,
        sigma_1=wde_seasonal_sigma_1,
        sigma_2=wde_seasonal_sigma_2,
        **intraday_kwargs)

    ssn_curve = seasonality_curve(
        time_series.day_of_year,
        **seasonality_kwargs)

    deterministic_curve = ssn_curve + itd_curve + base_price
    price_curve=ornstein_uhlenbeck(det_curve=deterministic_curve, **ou_kwargs) 
    return price_curve



if __name__ == "__main__":

    x = generate_price_curve(
        start_date="2024-01-04 00:00:00",
        end_date="2024-01-06 02:00:00", ou_kwargs={'sigma_mult':5,'theta':7})


    plt.figure(figsize=(12,5))
    plt.plot(x)
    plt.title("Synthetic GB Price Curve")
    plt.show()