from utils.helpers import data_split
from agents.evn_mvo import StockPortfolioEnv
from agents.mvo_agent import MarkowitzAgent
from pypfopt import expected_returns
import pandas as pd


from utils.helpers import data_split
from agents.evn_mvo import StockPortfolioEnv
from agents.mvo_agent import MarkowitzAgent
from pypfopt import EfficientFrontier, expected_returns, risk_models
import pandas as pd
from environements.portfolio_optimization_env import PortfolioOptimizationEnv
import numpy as np


def mvo_data(data, returns_model='ema_historical_return', risk_model='sample_cov'):
    df = data.sort_values(['date', 'tic'], ignore_index=True).copy()
    df.index = df.date.factorize()[0]
    cov_list = []
    corr_list = []
    mu = []
    # look back is one year
    lookback = 252
    for i in range(lookback, len(df.index.unique())):
        data_lookback = df.loc[i-lookback:i, :]
        price_lookback = data_lookback.pivot_table(
            index='date', columns='tic', values='close')
        covs = risk_models.risk_matrix(price_lookback, method=risk_model)
        mu.append(expected_returns.return_model(
            price_lookback, method=returns_model, compounding=False))
        cov_list.append(covs)
        corr_list.append(price_lookback.pct_change().dropna().corr())

    df_cov = pd.DataFrame(
        {'time': df.date.unique()[lookback:], 'cov_list': cov_list, 'corr_list': corr_list, 'returns': mu})
    df = df.merge(df_cov, left_on='date', right_on='time')

    return df


def mean_variance_optimization(test_data, solver='OSQP', window=1, commission_fee=0, objective='min_variance'):
    z = test_data.copy()
    z.sort_values(by=['date', 'tic'])
    environment = PortfolioOptimizationEnv(
        test_data,
        initial_amount=1000000,
        comission_fee_pct=commission_fee,
        time_window=window,
        features=["close", "return"],
        normalize_df=None,
        add_cash=False,
        use_softmax=False
    )

    variances = []
    environment.reset()
    terminated = False
    environment.reset()
    weights_list = []

    for index in z.index.unique():
        mean_returns = z.loc[index].iloc[0].returns
        cov = z.loc[index].iloc[0].cov_list
        ef = EfficientFrontier(mean_returns, cov, solver=solver)
        if objective == 'min_variance':
            ef.min_volatility()
            weights = ef.clean_weights()

        else:
            ef.max_sharpe()
            weights = ef.clean_weights()
        weights = list(weights.values())
        weights_list.append(weights)
        w = np.array(weights)
        _, _, terminated, _, _ = environment.step(weights)
        variances.append(np.dot(w.T, np.dot(cov, w)))
        if terminated:
            break
    date_list = environment._date_memory
    portfolio_return = environment._portfolio_return_memory
    result_df = pd.DataFrame(
        {"date": date_list, "daily_return": portfolio_return,
            'account':  environment._asset_memory["final"], 'weights': environment._final_weights}
    )
    return result_df, environment._tic_list.tolist(), variances, pd.DataFrame(weights_list)
