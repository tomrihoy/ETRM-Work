import pandas as pd
from datetime import datetime
from pathlib import Path

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