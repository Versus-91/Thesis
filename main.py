import math
from environements.portfolio_optimization_env import PortfolioOptimizationEnv
from environements.portfolio_optimization_env_flat import PortfolioOptimizationEnvFlat
from utils.portfolio_trainer import PortfolioOptimization
from pandas import read_csv
from utils.feature_engineer import FeatureEngineer
from utils.helpers import data_split
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta
from torch import nn
import torch
warnings.filterwarnings("ignore")


def linear_schedule(initial_value):
    def scheduler(progress_remaining):
        return initial_value * progress_remaining
    return scheduler


df_dow = read_csv('./data/dow.csv')

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
    transaction_fee=0, vectorize=False, sharp_reward=False, remove_close=True, decay_rate=0.0015,
    last_weight=False, tag="lsmt_alternate_state", comission_fee_model=None, env=PortfolioOptimizationEnv)
optimizer.train_model(train_data,
                      validation_data,
                      features=["close", "log_return", "momentum_return_21_normal",
                                "momentum_return_42_normal", "momentum_return_63_normal", "momentum_return_252_normal", "macd_normal", "rsi_normal"
                                ],
                      policy_network="MlpLstmPolicy",
                      model_name="RecurrentPPO",
                      args={'learning_rate': 1e-4,
                            "gamma": 0.95, "gae_lambda": 0.9, 'batch_size': 64, 'ent_coef': 0.03},
                      window_size=5,
                      iterations=1000_000,
                      )
# model = optimizer.load_from_file(
#     'ppo', path="data/RecurrentPPO_close_log_return_window_size_5_0.003_lsmt/RecurrentPPO_10000_steps",env=None)
# test_result = optimizer.DRL_prediction(
#     model, test_data, ["close", "log_return"])
# from utils.plotting_helpers import plot_weights
# plot_weights(test_result[0].weights, test_result[0].date, test_result[1])
