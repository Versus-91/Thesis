from stable_baselines3.common.noise import NormalActionNoise
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

warnings.filterwarnings("ignore")


def linear_schedule(initial_value):
    def scheduler(progress_remaining):
        return initial_value * progress_remaining
    return scheduler


TRAIN_START_DATE = '2010-01-01'
TRAIN_END_DATE = '2020-12-30'

VALIDATION_START_DATE = '2021-01-01'
VALIDATION_END_DATE = '2021-12-30'

TEST_START_DATE = '2022-01-01'
TEST_END_DATE = '2024-12-30'
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
    transaction_fee=0.001, comission_fee_model=None, flatten_state=False,
    tag="td3__dax_tanh", sharp_reward=False, last_weight=False, remove_close=True,
    add_cash=False, env=PortfolioOptimizationEnv
)
optimizer.train_model(train_data,
                      validation_data,
                      features=["close", "log_return", "r_21", "r_42", "r_63",
                                "macd", "rsi_30"
                                ],
                      model_name="td3",
                      args={'gamma': 0.90, 'learning_rate': 1e-4,
                            "buffer_size": 300_000, "batch_size": 124,
                            "action_noise": "normal"},
                      window_size=21,
                      policy_kwargs=dict(
                          activation_fn=nn.Tanh,
                      ),
                      iterations=2000_000)
