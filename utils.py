
from finrl.agents.stablebaselines3.models import DRLAgent, DRLEnsembleAgent
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline, trx_plot
from feature_engineer import FeatureEngineer
from models import DRLAgent
from portfolio_optimization_env import PortfolioOptimizationEnv
import pandas as pd
import numpy as np
from finrl.meta.preprocessor.preprocessors import data_split
from agents.evn_mvo import StockPortfolioEnv
from agents.mvo_agent import MarkowitzAgent
from pypfopt import expected_returns


def DRL_prediction(model, environment, time_window, deterministic=True):
    """make a prediction and get results"""
    test_env, test_obs = environment.get_sb_env()
    account_memory = None  # This help avoid unnecessary list creation
    actions_memory = None  # optimize memory consumption
    # state_memory=[] #add memory pool to store states

    test_env.reset()
    max_steps = len(environment._df.index.unique()) - time_window - 1

    for i in range(len(environment._df.index.unique())):
        action, _states = model.predict(test_obs, deterministic=deterministic)
        # account_memory = test_env.env_method(method_name="save_asset_memory")
        # actions_memory = test_env.env_method(method_name="save_action_memory")
        test_obs, rewards, dones, info = test_env.step(action)
        if i == max_steps:  # more descriptive condition for early termination to clarify the logic
            date_list = environment._date_memory
            portfolio_return = environment._portfolio_return_memory
            # print(len(date_list))
            # print(len(asset_list))
            df_account_value = pd.DataFrame(
                {"date": date_list, "daily_return": portfolio_return,
                    'account':  environment._asset_memory["final"], 'weights': environment._final_weights}
            )
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            action_list = environment._actions_memory
            df_actions = pd.DataFrame(action_list)
            tiks = environment._tic_list
            df_actions.columns = np.insert(tiks, 0, 'POS')
            df_actions.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
            account_memory = df_account_value
            actions_memory = df_actions
        # add current state to state memory
        # state_memory=test_env.env_method(method_name="save_state_memory")

        if dones[0]:
            print("hit end!")
            break
    return account_memory, actions_memory, test_obs


def benchmark(train_data, test_data, iterations, t, features):
    final_result = []
    models = [
        {'name': 'ppo', 'args': {
            "n_steps": 2048,
            "ent_coef": 0.01,
            "learning_rate": 0.0003,
            "batch_size": 128,
        }},
        {'name': 'a2c', 'args': None},
        # {'name': 'sac', 'args': None}
    ]
    env_kwargs = {
        "initial_amount": 100_0000,
        "normalize_df": None,
        "features": features,
        'comission_fee_pct': 0.001,
        'time_window': t
    }


    for i, m in enumerate(models):
        result = {}
        train_environment = PortfolioOptimizationEnv(df=train_data, **env_kwargs)
        test_environment = PortfolioOptimizationEnv(df=test_data, **env_kwargs)
        agent = DRLAgent(env=train_environment)
        model = agent.get_model(
            m['name'], model_kwargs=m['args'], tensorboard_log='./data/tb')
        ppo_model = agent.train_model(model=model,
                                      tb_log_name=m['name'],
                                      total_timesteps=iterations)
        training_summary = pd.DataFrame(
            {
                "date": train_environment._date_memory,
                "actions": train_environment._actions_memory,
                "weights": train_environment._final_weights,
                "returns": train_environment._portfolio_return_memory,
                "rewards": train_environment._portfolio_reward_memory,
                "portfolio_values": train_environment._asset_memory["final"],
            }
        )
        prediction_summary = DRL_prediction(ppo_model, test_environment, t)
        result["train"] = training_summary
        result["test"] = prediction_summary
        result["name"] = m['name']
        final_result.append(result)
    return final_result


def baseline(data, INDICATORS, TEST_START_DATE, TEST_END_DATE):
    final_result = []
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
    stock_dimension = len(test_df.tic.unique())
    state_space = stock_dimension

    result = {}

    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "transaction_cost_pct": 0.001,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4

    }
    e_test_gym = StockPortfolioEnv(df=test_df, **env_kwargs)
    agent = MarkowitzAgent(e_test_gym)
    mvo_min_variance = agent.prediction(e_test_gym)
    mvo_min_variance["method"] = "markowitz"
    mvo_min_variance.columns = ['date', 'account', 'return', 'method']
    result["test"] = [mvo_min_variance]
    result["name"] = 'Min Variance Portfolio'
    final_result.append(result)

    result = {}
    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "transaction_cost_pct": 0.001,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4

    }

    e_test_gym = StockPortfolioEnv(df=test_df, **env_kwargs)
    agent = MarkowitzAgent(e_test_gym, objective='sharpe')
    try:
        mvo_max_sharpe = agent.prediction(e_test_gym)
    except:
        agent = MarkowitzAgent(e_test_gym, objective='sharpe', solver='SCS')
        mvo_max_sharpe = agent.prediction(e_test_gym)
    mvo_max_sharpe = agent.prediction(e_test_gym)
    mvo_max_sharpe["method"] = "markowitz"
    mvo_max_sharpe.columns = ['date', 'account', 'return', 'method']
    result["test"] = [mvo_max_sharpe]
    result["name"] = 'Max Sharpe Ratio'
    final_result.append(result)
    return final_result
