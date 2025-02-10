import torch
import math
import warnings
from utils.plotting_helpers import plot_weights
from environements.portfolio_optimization_env import PortfolioOptimizationEnv
import matplotlib as mpl
from utils.portfolio_trainer import PortfolioOptimization
from pandas import read_csv
from utils.feature_engineer import FeatureEngineer
from utils.helpers import data_split
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import scienceplots
from utils.plotting_helpers import plot_mvo_weights
import utils.mean_variance_optimization as mvo
df_dow = read_csv('./data/dow.csv')
# mpl.rcParams['figure.dpi'] = 300
df_hsi = read_csv('./data/hsi.csv')
df_dax = read_csv('./data/dax.csv')
df_sp500 = read_csv('./data/sp500.csv')
warnings.filterwarnings("ignore")


df = df_dow.copy()
df = df_dow[df_dow.tic.isin(
    ['MSFT', 'UNH', 'DIS', 'GS', 'HD', 'V', "AXP", "MCD", "CAT", "AMGN", "TRV"])]
TRAIN_START_DATE = '2015-01-01'
TRAIN_END_DATE = '2019-12-30'

VALIDATION_START_DATE = '2020-01-01'
VALIDATION_END_DATE = '2020-12-30'

TEST_START_DATE = '2021-01-01'
TEST_END_DATE = '2022-01-01'
INDICATORS = [
    "macd",
    "rsi_30",
    "close_5_ema",
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
state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
print(f"Stock Dimension: {stock_dimension}")


cleaned_data['std_return_60'] = cleaned_data.groupby('tic')['log_return'].ewm(span=60, ignore_na=False,
                                                                              min_periods=1).std().reset_index(level=0, drop=True)
cleaned_data['ewma_std_price_63'] = cleaned_data.groupby('tic')['close'].ewm(span=63, ignore_na=False,
                                                                             min_periods=1).std().reset_index(level=0, drop=True)

cleaned_data['macd_normalized'] = cleaned_data['macd'] / \
    cleaned_data['ewma_std_price_63']
cleaned_data['macd_std'] = cleaned_data.groupby('tic')['macd_normalized'].ewm(span=252, ignore_na=False,
                                                                              min_periods=1).std().reset_index(level=0, drop=True)

cleaned_data['macd_normal'] = cleaned_data['macd_normalized'] / \
    cleaned_data['macd_std']
cleaned_data['rsi_normal'] = cleaned_data['rsi_30'] / 100
cleaned_data['price_lag_5'] = cleaned_data.groupby('tic')['log_return'].rolling(
    window=5, min_periods=5).sum().reset_index(level=0, drop=True)
cleaned_data['price_lag_21'] = cleaned_data.groupby('tic')['log_return'].rolling(
    window=21, min_periods=21).sum().reset_index(level=0, drop=True)
cleaned_data['price_lag_42'] = cleaned_data.groupby('tic')['log_return'].rolling(
    window=42, min_periods=42).sum().reset_index(level=0, drop=True)
cleaned_data['price_lag_63'] = cleaned_data.groupby('tic')['log_return'].rolling(
    window=63, min_periods=63).sum().reset_index(level=0, drop=True)
cleaned_data['price_lag_252'] = cleaned_data.groupby('tic')['log_return'].rolling(
    window=252, min_periods=252).sum().reset_index(level=0, drop=True)

cleaned_data['momentum_return_21_normal'] = cleaned_data['price_lag_21'] / \
    (cleaned_data['std_return_60'] * math.sqrt(252))
cleaned_data['momentum_return_42_normal'] = cleaned_data['price_lag_42'] / \
    (cleaned_data['std_return_60'] * math.sqrt(252))
cleaned_data['momentum_return_63_normal'] = cleaned_data['price_lag_63'] / \
    (cleaned_data['std_return_60'] * math.sqrt(252))
cleaned_data['momentum_return_252_normal'] = cleaned_data['price_lag_252'] / \
    (cleaned_data['std_return_60'] * math.sqrt(252))


train_data = data_split(cleaned_data, TRAIN_START_DATE, TRAIN_END_DATE)
test_data = data_split(cleaned_data, TEST_START_DATE, TEST_END_DATE)
validation_data = data_split(
    cleaned_data, VALIDATION_START_DATE, VALIDATION_END_DATE)
stock_dimension = len(train_data.tic.unique())
state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
print(f"Stock Dimension: {stock_dimension}")
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

optimizer = PortfolioOptimization(
    transaction_fee=0.00, comission_fee_model=None, vectorize=False, normalize=None,
    tag="ppo_alternative_state_11_asset", sharp_reward=False, last_weight=False, remove_close=True,
    add_cash=False, env=PortfolioOptimizationEnv
)
optimizer.train_model(train_data,
                      validation_data,
                      features=["close", "log_return", "momentum_return_21_normal",
                                "momentum_return_42_normal", "momentum_return_63_normal", "macd_normal", "rsi_normal"
                                ],
                      model_name="ppo",
                      args={"n_steps":  256, "batch_size": 128, 'learning_rate': 1e-4,
                            'gamma': 0.90, 'gae_lambda': 0.85, 'ent_coef': 0.05},
                      window_size=60,
                      iterations=1000_000)
