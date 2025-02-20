import pickle
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
    ['AXP', 'DIS', 'GS', 'MMM', 'UNH', 'MCD', 'CAT', 'CRM', 'V', 'AMGN', 'TRV', 'MSFT'])]
TRAIN_START_DATE = '2010-01-01'
TRAIN_END_DATE = '2019-12-30'

VALIDATION_START_DATE = '2020-01-01'
VALIDATION_END_DATE = '2020-12-30'

TEST_START_DATE = '2021-01-01'
TEST_END_DATE = '2023-01-01'
with open('./data/dow_processed.pkl', 'rb') as file:
    cleaned_data = pickle.load(file)

cleaned_data = cleaned_data.fillna(0)
train_data = data_split(cleaned_data, TRAIN_START_DATE, TRAIN_END_DATE)
test_data = data_split(cleaned_data, TEST_START_DATE, TEST_END_DATE)
validation_data = data_split(
    cleaned_data, VALIDATION_START_DATE, VALIDATION_END_DATE)
stock_dimension = len(train_data.tic.unique())


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

optimizer = PortfolioOptimization(
    transaction_fee=0.002, comission_fee_model=None,
    tag="state_corr", sharp_reward=False, last_weight=False, remove_close=True, flatten_state=True,
    add_cash=False, env=PortfolioOptimizationEnv
)
optimizer.train_model(train_data,
                      validation_data,
                      features=["close", "log_return", "r_21", "r_42", "r_63",
                                "macd", "rsi_30", "corr_list"
                                ],
                      model_name="td3",
                      args={"n_steps":  1024, "batch_size": 64, 'learning_rate': 1e-4,
                            'gamma': 0.90},
                      window_size=5,
                      policy_kwargs=dict(
                          log_std_init=True,
                          activation_fn=nn.SiLU,
                      ),
                      iterations=2000_000)
