from utils.portfolio_trainer import PortfolioOptimization
from pandas import read_csv
from utils.feature_engineer import FeatureEngineer
from utils.helpers import data_split
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
df_dow = read_csv('./data/dow.csv')
df_nasdaq = read_csv('./data/nasdaq.csv')
df_hsi = read_csv('./data/hsi.csv')
df_dax = read_csv('./data/dax.csv')
df_sp500 = read_csv('./data/sp500.csv')


def get_data():
    df_dow = read_csv('./data/dow.csv')
    df = df_dow.copy()
    df = df_dow[df_dow.tic.isin(
        ['AAPL', 'AXP', 'DIS', 'GS', 'IBM', 'MMM', 'WBA'])]
    TRAIN_START_DATE = '2014-01-01'
    TRAIN_END_DATE = '2019-12-30'

    VALIDATION_START_DATE = '2020-01-01'
    VALIDATION_END_DATE = '2020-12-30'

    TEST_START_DATE = '2021-01-01'
    TEST_END_DATE = '2024-10-01'
    INDICATORS = [
        "close_21_ema",
        "close_62_ema"
    ]

    fe = FeatureEngineer(use_technical_indicator=True,
                         tech_indicator_list=INDICATORS,
                         use_turbulence=False,
                         user_defined_feature=True)

    processed_dax = fe.preprocess_data(df.query('date>"2013-01-01"'))
    cleaned_data = processed_dax.copy()
    cleaned_data = cleaned_data.fillna(0)
    cleaned_data = cleaned_data.replace(np.inf, 0)
    train_data = data_split(cleaned_data, TRAIN_START_DATE, TRAIN_END_DATE)
    test_data = data_split(cleaned_data, TEST_START_DATE, TEST_END_DATE)
    validation_data = data_split(
        cleaned_data, VALIDATION_START_DATE, VALIDATION_END_DATE)
    stock_dimension = len(train_data.tic.unique())
    print(f"Stock Dimension: {stock_dimension}")
    return train_data, test_data, validation_data
