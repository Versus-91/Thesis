
from finrl.agents.stablebaselines3.models import DRLAgent, DRLEnsembleAgent
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline, trx_plot
from utils.feature_engineer import FeatureEngineer
from models import DRLAgent
from environements.portfolio_optimization_env import PortfolioOptimizationEnv
import pandas as pd
import numpy as np
from finrl.meta.preprocessor.preprocessors import data_split
from agents.evn_mvo import StockPortfolioEnv
from agents.mvo_agent import MarkowitzAgent
from pypfopt import expected_returns


def mvo_data(data, TEST_START_DATE, TEST_END_DATE):
    df = data.sort_values(['date', 'tic'], ignore_index=True).copy()
    df.index = df.date.factorize()[0]
    cov_list = []
    mu = []
    # look back is one year
    lookback = 252
    for i in range(lookback, len(df.index.unique())):
        data_lookback = df.loc[i-lookback:i, :]
        price_lookback = data_lookback.pivot_table(
            index='date', columns='tic', values='close')
        return_lookback = price_lookback.pct_change().dropna()
        covs = return_lookback.cov().values
        mu.append(expected_returns.mean_historical_return(price_lookback))
        cov_list.append(covs)
    df_cov = pd.DataFrame(
        {'time': df.date.unique()[lookback:], 'cov_list': cov_list, 'returns': mu})
    df = df.merge(df_cov, left_on='date', right_on='time')

    test_df = data_split(
        df,
        start=TEST_START_DATE,
        end=TEST_END_DATE
    )
    return test_df


def mvo(data, solver='OSQP', window=1, rf=0.02, pct=0.001, objective='min_variance', multi_objective=False):
    result = {}
    stock_dimension = len(data.tic.unique())
    state_space = stock_dimension
    env_kwargs = {
        "hmax": 100,
        "initial_amount": 50_000,
        "transaction_cost_pct": pct,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": [],
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
        "window": window

    }
    test_env = StockPortfolioEnv(df=data, **env_kwargs)
    agent = MarkowitzAgent(test_env, rf=rf, objective=objective, cost=pct,multi_objective=multi_objective)
    mvo_min_variance = agent.prediction(test_env)
    mvo_min_variance["method"] = "markowitz"
    mvo_min_variance.columns = ['date', 'account', 'return', 'method']
    result["test"] = mvo_min_variance
    result["name"] = 'Min Variance Portfolio'
    result["action"] = test_env.actions_memory
    result["date"] = test_env.date_memory

    return result
