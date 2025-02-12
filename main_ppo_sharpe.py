from sklearn.preprocessing import StandardScaler
from torch import nn
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


def linear_schedule(initial_value):
    def scheduler(progress_remaining):
        return initial_value * progress_remaining
    return scheduler


df = df_dow.copy()
df = df_dow[df_dow.tic.isin(
    ['AXP', 'DIS', 'GS', 'MMM', 'UNH', 'MCD'])]
TRAIN_START_DATE = '2015-01-01'
TRAIN_END_DATE = '2019-12-30'

VALIDATION_START_DATE = '2020-01-01'
VALIDATION_END_DATE = '2020-12-30'

TEST_START_DATE = '2021-01-01'
TEST_END_DATE = '2022-01-01'
INDICATORS = [
    "macd",
    "rsi_30",
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


cleaned_data['std_return_63'] = cleaned_data.groupby('tic')['log_return'].ewm(span=63, ignore_na=False,
                                                                              min_periods=1).std().reset_index(level=0, drop=True)

cleaned_data['std_return_21'] = cleaned_data.groupby('tic')['log_return'].ewm(span=21, ignore_na=False,
                                                                              min_periods=1).std().reset_index(level=0, drop=True)
cleaned_data['std_return_42'] = cleaned_data.groupby('tic')['log_return'].ewm(span=42, ignore_na=False,
                                                                              min_periods=1).std().reset_index(level=0, drop=True)
cleaned_data['std_return_256'] = cleaned_data.groupby('tic')['log_return'].ewm(span=256, ignore_na=False,
                                                                              min_periods=1).std().reset_index(level=0, drop=True)
cleaned_data['ewma_std_price_63'] = cleaned_data.groupby('tic')['close'].ewm(span=63, ignore_na=False,
                                                                             min_periods=1).std().reset_index(level=0, drop=True)

cleaned_data['macd_normal'] = cleaned_data['macd'] / \
    cleaned_data['ewma_std_price_63']

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
    (cleaned_data['std_return_60'] * math.sqrt(21))
cleaned_data['momentum_return_42_normal'] = cleaned_data['price_lag_42'] / \
    (cleaned_data['std_return_60'] * math.sqrt(42))
cleaned_data['momentum_return_63_normal'] = cleaned_data['price_lag_63'] / \
    (cleaned_data['std_return_60'] * math.sqrt(63))
cleaned_data['momentum_return_252_normal'] = cleaned_data['price_lag_252'] / \
    (cleaned_data['std_return_60'] * math.sqrt(252))

scaler = StandardScaler()
train_data = data_split(cleaned_data, TRAIN_START_DATE, TRAIN_END_DATE)
test_data = data_split(cleaned_data, TEST_START_DATE, TEST_END_DATE)

train_data["price_normalized"] = scaler.fit_transform(train_data[["close"]])
test_data["price_normalized"] = scaler.transform(test_data[["close"]])
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
    tag="ppo_abs", sharp_reward=False, last_weight=False, remove_close=True,
    add_cash=False, env=PortfolioOptimizationEnv
)
optimizer.train_model(train_data,
                      validation_data,
                      features=["close", "log_return", "momentum_return_21_normal",
                                "momentum_return_42_normal", "momentum_return_63_normal", "momentum_return_252_normal", "macd_normal", "rsi_normal"
                                ],
                      model_name="ppo",
                      args={"n_steps":  256, "batch_size": 64, 'learning_rate': linear_schedule(2e-4),
                            'gamma': 0.92, 'gae_lambda': 0.85, 'ent_coef': 0.05},
                      policy_kwargs=dict(
                          # log_std_init=log_std_init,
                          activation_fn=nn.LeakyReLU,
                      ),
                      window_size=60,
                      iterations=1000_000)
