import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from datetime import datetime, date
from pathlib import Path
import matplotlib.dates as mdates
from typing import Tuple, Dict, Union, Optional
import typer 
from datetime import datetime, date
from pandas import Timestamp




class PowerPrices:
    """Generate realistic synthetic wholesale power prices"""
    
    DEFAULT_CONFIG = {
        "intraday": {
            "a_1": 10,
            "a_2": 15,
            "mean_1": 16,
            "mean_2": 37
        },
        "seasonality": {
            "A": 30
        },
        "weekday": {
            "wkd_mult": 1,
            "we_mult": 0.6,
            "wkd_sigma_1": 4,
            "wkd_sigma_2": 4,
            "we_sigma_1": 5,
            "we_sigma_2": 5
        },
        "sigma_month_list_1": [1,1,1,1.1,1.2,1.2,1.2,1.2,1.1,1,1,1],
        "sigma_month_list_2": [1,1,1,1.2,1.3,1.3,1.3,1.3,1.2,1.1,1,1],
        "ou": {
            "dt": 1/48,
            "theta": 4.0,
            "sigma_mult": 5
        },
        "base_price": 70
    }
    
    def __init__(self, config: Optional[dict] = None) -> None:
        """Initialize class and merge user config with defaults"""
        self.config = self.deep_update(self.DEFAULT_CONFIG, config or {})
        self.curve_dict: dict[str, pd.DataFrame] = {}


    @staticmethod
    def deep_update(default, override: Dict)-> Dict:
        """Recursively update default config with override dict"""
        result = copy.deepcopy(default)
        for k, v in override.items():
            if isinstance(v, dict) and k in result:
                result[k] = PowerPrices.deep_update(result[k], v)
            else:
                result[k] = v
        return result

    @staticmethod
    def settlement_period(index: pd.DatetimeIndex)->pd.Index:
        """Convert datetimes to settlement periods"""
        return (index.hour * 60 + index.minute) // 30 + 1

    def week_day_end(self, index: pd.DatetimeIndex)->Tuple[np.ndarray,np.ndarray,np.ndarray]:
        """Adjust weekday/weekend multipliers and sigmas"""
        weekday_cfg = self.config['weekday']
        is_weekend = index.dayofweek.isin([5,6])
        wde_mult = np.where(is_weekend, weekday_cfg['we_mult'], weekday_cfg['wkd_mult'])
        wde_sigma_1 = np.where(is_weekend, weekday_cfg['we_sigma_1'], weekday_cfg['wkd_sigma_1'])
        wde_sigma_2 = np.where(is_weekend, weekday_cfg['we_sigma_2'], weekday_cfg['wkd_sigma_2'])
        return wde_mult, wde_sigma_1, wde_sigma_2
    
    def seasonal_sigma(self,time_series: pd.DatetimeIndex, wde_sigma: np.ndarray, sigma_month_list_key: str)->np.ndarray:
        '''Vary spread of two daily peaks by month.'''
        sigma_month_list = self.config[sigma_month_list_key]
        month_factors = np.array([sigma_month_list[m-1] for m in time_series.month])
        return wde_sigma * month_factors

    
    def intraday_curve(self, sp_to_model: pd.Index, wde_multiplier: np.ndarray, sigma_1: np.ndarray, sigma_2: np.ndarray)->pd.Index:
        '''Model intraday curve with gaussian peaks.'''
        intraday_cfg=self.config['intraday']
        peak_1 = intraday_cfg['a_1'] * np.exp(-(sp_to_model - intraday_cfg['mean_1'])**2 / (2 * sigma_1**2))
        peak_2 = intraday_cfg['a_2'] * np.exp(-(sp_to_model - intraday_cfg['mean_2'])**2 / (2 * sigma_2**2))
        return wde_multiplier * (peak_1 + peak_2)

    def seasonality_curve(self,day_of_year: pd.Index)->pd.Index:
        '''Seasonal component of curve modelled by cosine.'''
        seasonality_cfg=self.config['seasonality']
        return seasonality_cfg['A'] * np.cos((2*np.pi / 365) * day_of_year)

    def ornstein_uhlenbeck(self, det_curve: pd.Index)->np.ndarray:
        '''Stochastic mean reverting component of price curve. Volatility
           is scaled to detrminstic curve size.'''
        ou_cfg=self.config['ou']
        n_steps = len(det_curve)
        x = np.zeros(n_steps)
        x[0] = det_curve[0]
        dW = np.random.normal(0, np.sqrt(ou_cfg['dt']), size=n_steps)
        for t in range(1, n_steps):
            x[t] = (x[t-1]
                    + ou_cfg['theta'] * (det_curve[t]-x[t-1]) * ou_cfg['dt']
                    + ou_cfg['sigma_mult'] * det_curve[t]/np.mean(det_curve) * dW[t])
        return x
    
    @staticmethod
    def save_to_csv(df: pd.DataFrame, filepath: Optional[str] = None, filename: Optional[str] = None) -> None:
  
        '''Save power curves to csv.'''
        full_file_path=PowerPrices.generate_filepath(filepath, filename)
        df.to_csv(full_file_path)

    @staticmethod
    def generate_filepath(filepath: Optional[str], filename: Optional[str]) -> Path:
        '''Generate filepath for csv saving'''
        filename = PowerPrices.generate_filename(filename)
        if filepath is not None:
            full_file_path=Path(filepath)/filename
        else:
            full_file_path=Path(filename)
        return full_file_path

    @staticmethod
    def generate_filename(filename: Optional[str]) -> str: 
        '''Generate filename for saving curve to csv and as a key for a dictionary'''
        if filename is None:
            current_time=datetime.now()
            filename=f'{current_time.year}_{current_time.month}_{current_time.day}_{current_time.hour}_{current_time.minute}_{current_time.second}_{current_time.microsecond}_synthetic_prices.csv'
        return filename

    def generate_price_curve(self,
                         start_date: str,
                         end_date: str,
                         save_to_csv: bool = False,
                         filepath: Optional[str] = None,
                         filename: Optional[str] = None) -> pd.DataFrame:
        '''Generate stochastic price curve'''
        ts = pd.date_range(start_date, end_date, freq="30min")
        sp_array = self.settlement_period(ts)
        
        # weekday/weekend adjustments
        wde_mult, wde_sigma_1, wde_sigma_2 = self.week_day_end(ts)
        wde_seasonal_sigma_1 = self.seasonal_sigma(ts, wde_sigma_1, 'sigma_month_list_1')
        wde_seasonal_sigma_2 = self.seasonal_sigma(ts, wde_sigma_2, 'sigma_month_list_2')
        
        # intraday curve
        itd_curve = self.intraday_curve(sp_array, wde_mult, wde_seasonal_sigma_1, wde_seasonal_sigma_2)
        
        # seasonality
        ssn_curve = self.seasonality_curve(ts.day_of_year)
        
        # deterministic 
        det_curve = ssn_curve + itd_curve + self.config['base_price']
        
        # add stochasticity
        price_curve = self.ornstein_uhlenbeck(det_curve)
        
        # construct dataframe

        prices_df = pd.DataFrame(list(zip(ts, price_curve)), columns=['datetime', 'power_prices'])
        
        if save_to_csv:    
            self.save_to_csv(prices_df, filepath, filename)

        curve_name = PowerPrices.generate_filename(filename).removesuffix('.csv')     
        self.curve_dict[curve_name] = prices_df

        return prices_df
    
    def plot_curves(self, curve_to_plot: Optional[str] = None) -> None:
        '''Plot power prices stored in curve_dict. The x coordinates are auto formatted.'''
        fig, ax = plt.subplots()
        if curve_to_plot is None:
            df_length = []
            for key in self.curve_dict.keys():
                prices = self.curve_dict[key]
                df_length.append(len(prices))
                ax.plot(prices['datetime'], prices['power_prices'], label=key)

        
        else:
            prices = self.curve_dict[curve_to_plot]
            ax.plot(prices['datetime'], prices['power_prices'], label=curve_to_plot)
        locator = mdates.AutoDateLocator()
        formatter = mdates.AutoDateFormatter(locator)

        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (£/MWh)')
        fig.autofmt_xdate()
        ax.legend()
        plt.show()

    @staticmethod
    def _compute_stats(prices: pd.Series) -> dict:
        '''Compute summary statistics for a single price series.'''
        return {
            'max':    prices.max(),
            'min':    prices.min(),
            'std':    prices.std(),
            'median': prices.median(),
            'mean':   prices.mean()
        }

    def analyse_curves(self) -> dict:
        '''Return summary statistics for all stored price curves.'''
        return {
            key: self._compute_stats(df['power_prices'])
            for key, df in self.curve_dict.items()
        }
    
if __name__=='__main__':
    custom_cfg = {
    "ou": {"theta": 7, "sigma_mult": 5},
    "intraday": {"a_1": 12}
    }
    pp = PowerPrices(config=custom_cfg)
    prices = pp.generate_price_curve('2026-01-01', 
                                     '2026-01-08', 
                                     save_to_csv=True, 
                                     filepath='price_curve_data')

    pp.generate_price_curve('2026-01-01', '2026-01-08')
    pp.generate_price_curve('2026-01-01', '2026-01-09')
    pp.plot_curves()
    
   
