from datetime import datetime, timedelta
import math
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



    # df = df_dax[df_dax.tic.isin([ 'AXP', 'DIS', 'GS', 'MMM', 'UNH','MCD','CAT','CRM','V','AMGN','TRV','MSFT'])]
    df = df[df.tic.isin(['ADS.DE', 'ALV.DE', 'BAS.DE', 'BAYN.DE', 'BMW.DE', 'CON.DE', 'DBK.DE', 'DTE.DE', 'EOAN.DE', 'FME.DE', 'VOW3.DE'])]
    TRAIN_START_DATE = '2010-01-01'
    TRAIN_END_DATE = '2019-12-30'

    VALIDATION_START_DATE = '2020-01-01'
    VALIDATION_END_DATE = '2020-12-30'

    TEST_START_DATE = '2021-01-01'
    TEST_END_DATE = '2024-01-01'
    INDICATORS = [
        "macd",
        "rsi_30",
    ]

    fe = FeatureEngineer(use_technical_indicator=True,
                        tech_indicator_list=INDICATORS,
                        use_turbulence=False,
                        user_defined_feature=True)

    processed_prcies = fe.preprocess_data(df.query('date>"2000-01-01"'))
    cleaned_data = processed_prcies.copy()
    cleaned_data = cleaned_data.fillna(0)
    cleaned_data = cleaned_data.replace(np.inf, 0)
    stock_dimension = len(cleaned_data.tic.unique())
    state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
    # Compute exponentially weighted std of log returns
    for window in [21, 42, 63]:
        cleaned_data[f'std_return_{window}'] = cleaned_data.groupby('tic')['log_return'] \
            .transform(lambda x: x.ewm(span=window, min_periods=10, adjust=False).std())
    # Compute exponentially weighted std of closing prices for MACD normalization
    cleaned_data['ewma_std_price_63'] = cleaned_data.groupby('tic')['close'] \
        .transform(lambda x: x.ewm(span=63, min_periods=10, adjust=False).std())

    # Normalize MACD by price volatility
    cleaned_data['macd_normal'] = cleaned_data['macd'] / \
        cleaned_data['ewma_std_price_63']

    # Rolling cumulative log returns over different periods
    for window in [21, 42, 63]:
        cleaned_data[f'return_sum_{window}'] = cleaned_data.groupby('tic')['log_return'] \
            .transform(lambda x: x.rolling(window=window, min_periods=10).sum())

    # Normalize rolling log returns by their respective volatilities
    for window in [21, 42, 63]:
        cleaned_data[f'r_{window}'] = cleaned_data[f'return_sum_{window}'] / \
            cleaned_data['std_return_63']
    cleaned_data['rsi_30'] = cleaned_data['rsi_30'] / 100

    TRAIN_START_DATE = '2010-01-01'
    TRAIN_END_DATE = '2020-12-30'

    VALIDATION_START_DATE = '2021-01-01'
    VALIDATION_END_DATE = '2021-12-30'

    TEST_START_DATE = '2022-01-01'
    TEST_END_DATE = '2024-12-30'
    train_data = data_split(cleaned_data, TRAIN_START_DATE, TRAIN_END_DATE)
    test_data = data_split(cleaned_data, TEST_START_DATE, TEST_END_DATE)
    validation_data = data_split(
        cleaned_data, VALIDATION_START_DATE, VALIDATION_END_DATE)
    stock_dimension = len(train_data.tic.unique())

    return train_data, test_data, validation_data
