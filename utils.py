import pandas as pd
from datetime import datetime
from pathlib import Path
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

def save_to_csv(df: pd.DataFrame, filepath: str | None = None, filename: str | None = None) -> None:
    '''Save power curves to csv.'''
    full_file_path = generate_filepath(filepath, filename)
    df.to_csv(full_file_path)

def generate_filepath(filepath: str | None, filename: str | None) -> Path:
    '''Generate filepath for csv saving'''
    filename = generate_filename(filename)
    if filepath is not None:
        full_file_path=Path(filepath)/filename
    else:
        full_file_path=Path(filename)
    return full_file_path

def generate_filename(filename: str | None) -> str:
    '''Generate filename for saving curve to csv and as a key for a dictionary'''
    if filename is None:
        current_time=datetime.now()
        filename=f'{current_time.year}_{current_time.month}_{current_time.day}_{current_time.hour}_{current_time.minute}_{current_time.second}_{current_time.microsecond}_synthetic_prices.csv'
    return filename

def cli_plot_curve(curve_path: str):
    ''' Plot power curve stored in csv file. '''
    price_df = pd.read_csv(curve_path)
    fig, ax = plt.subplots()
    filename = Path(curve_path).stem
    ax.plot(price_df['datetime'], price_df['power_prices'], label=filename)
    locator = mdates.AutoDateLocator()
    formatter = mdates.AutoDateFormatter(locator)

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (£/MWh)')
    fig.autofmt_xdate()
    ax.legend()
    plt.show()