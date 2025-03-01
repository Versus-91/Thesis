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
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")


def linear_schedule(initial_value):
    def scheduler(progress_remaining):
        return initial_value * progress_remaining
    return scheduler


if __name__ == "__main__":
    from argparse import ArgumentParser
    try:
        parser = ArgumentParser()
        parser.add_argument("--sharpe-reward", action="store_true")
        parser.add_argument("activation_function")
        args = parser.parse_args()
        use_sharpe_reward = args.sharpe_reward
        af = str(args.activation_function)
    except:
        use_sharpe_reward = False
        af = 'silu'
        print('An exception occurred')

    TRAIN_START_DATE = '2010-01-01'
    TRAIN_END_DATE = '2020-12-30'

    VALIDATION_START_DATE = '2021-01-01'
    VALIDATION_END_DATE = '2021-12-30'

    TEST_START_DATE = '2022-01-01'
    TEST_END_DATE = '2024-12-30'
    with open('./data/dow_normal_processed.pkl', 'rb') as file:
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
    tag = 'rppo_dow'

    if use_sharpe_reward:
        tag += '_sharpe'
    if af:
        tag += '_'+af
    if af == 'silu':
        activ_func = nn.SiLU
    if af == 'tanh':
        activ_func = nn.Tanh
    if af == 'sigmoid':
        activ_func = nn.Sigmoid
    if af == 'leaky_relu':
        activ_func = nn.LeakyReLU
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
        transaction_fee=0.001, comission_fee_model=None, flatten_state=True,
        tag=tag, sharp_reward=use_sharpe_reward, last_weight=False, remove_close=True,
        add_cash=False, env=PortfolioOptimizationEnv
    )
    optimizer.train_model(train_data,
                          validation_data,
                          features=["close", "log_return", "r_21", "r_42", "r_63",
                                    "rsi_30", "macd", "corr_list"
                                    ],
                          policy_network="MlpLstmPolicy",
                          model_name="RecurrentPPO",
                          args={"n_steps":  256, "batch_size": 64, 'learning_rate': 1e-4,
                                'gamma': 0.90, "gae_lambda": 0.9, "n_epochs": 4, "ent_coef": 0.01},
                          window_size=63,
                          policy_kwargs=dict(
                              activation_fn=activ_func,
                              net_arch=dict(
                                  pi=[64, 32], vf=[64, 32])
                          ),
                          iterations=1000_000)
