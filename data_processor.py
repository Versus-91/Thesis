from datetime import datetime, timedelta
from utils.portfolio_trainer import PortfolioOptimization
from pandas import read_csv
from utils.feature_engineer import FeatureEngineer
from utils.helpers import data_split
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")




def add_volatility(df, periods=21):
    rolling_volatility = df.groupby(
        'tic')['log_return'].rolling(window=periods).std()
    rolling_volatility = rolling_volatility.reset_index(level=0, drop=True)
    # Assign the annualized volatility back to the original DataFrame
    df['volatility'] = rolling_volatility

    return df


def get_data(df, train_start='2014-01-01', train_end='2019-12-30', validation_start='2020-01-01', validation_end='2020-12-30', test_start='2021-01-01', test_end='2024-10-01'):

    date_format = '%Y-%m-%d'

    # Convert the string to a datetime object
    start_date = datetime.strptime(train_start, date_format)

    # Subtract one year
    # Use timedelta(days=365) for a rough estimate, or handle leap years properly
    try:
        start_date_year_before = start_date.replace(year=start_date.year - 1)
    except ValueError:  # Handles February 29 cases
        start_date_year_before = start_date.replace(
            month=2, day=28, year=start_date.year - 1)
    INDICATORS = [
        "close_21_ema",
        "close_62_ema"
    ]

    fe = FeatureEngineer(use_technical_indicator=True,
                         tech_indicator_list=INDICATORS,
                         use_turbulence=False,
                         user_defined_feature=True)

    processed_dax = fe.preprocess_data(
        df.query(f'date>"{start_date_year_before}"'))
    cleaned_data = processed_dax.copy()
    cleaned_data = add_volatility(cleaned_data)
    cleaned_data = cleaned_data.fillna(0)
    cleaned_data = cleaned_data.replace(np.inf, 0)
    train_data = data_split(cleaned_data, train_start, train_end)
    test_data = data_split(cleaned_data, test_start, test_end)
    validation_data = data_split(
        cleaned_data, validation_start, validation_end)
    stock_dimension = len(train_data.tic.unique())
    print(f"Stock Dimension: {stock_dimension}")
    return train_data, test_data, validation_data
