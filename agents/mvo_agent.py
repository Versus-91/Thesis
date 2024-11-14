# example markowitz agent using info dict from environment to do optimization at each time step

import cvxpy as cp
import numpy as np
import pandas as pd
from collections import defaultdict
from pypfopt import EfficientFrontier


class MarkowitzAgent:
    """Provides implementations for Markowitz agent (mean-variance optimization)
    Attributes
    ----------
        env: gym environment class
            user-defined class
    Methods
    -------
        prediction()
            make a prediction in a test dataset and get result history
    """

    def __init__(
            self,
            env,
            solver = 'OSQP',
            risk_aversion=10,
            objective='min_variance',
            annual_risk_free_rate=0.03  # disregard risk free rate since RL disregards
    ):
        super().__init__()
        self.risk_aversion = risk_aversion
        self.env = env
        self.solver = solver
        self.objective = objective
        # compute daily risk free rate from annual risk free rate
        # self.risk_free_rate = (1 + annual_risk_free_rate) ** (1 / 365) - 1
        # disable risk free rate for now
        self.risk_free_rate = -1

    def get_model(self, model_name, model_kwargs):
        raise NotImplementedError()

    def train_model(self, model, cwd, total_timesteps=5000):
        raise NotImplementedError()

    def act(self, state, info):
        """
        This is the core of markowitz portfolio optimization
        it maximizes the éxpected_return - risk_aversion * risk
        with expected_return = mean_returns @ portfolio_weights
        and risk = portfolio_weights.T @ cov @ portfolio_weights
        The constraints say that the weights must be positive and sum to 1

        returns the action as the weights of the portfolio
        """
        # unpack state to get covariance and means
        data = info["data"].copy()
        # from the data estimate returns and covariances
        cov = data.iloc[-1].cov_list
        mean_returns = data.iloc[-1].returns
        ef = EfficientFrontier(mean_returns, cov, solver=self.solver)
        if self.objective == 'min_variance':
            ef.min_volatility()
        else:
            ef.max_sharpe()

        weights = ef.clean_weights()
        list_weights = list(weights.values())
        # get action. if using risk free rate then integrate it into the action
        action = list_weights
        # action = np.concatenate([weights, risk_free_weight.value])
        action = np.maximum(action, 0)
        action = action / np.sum(action)
        return action

    def prediction(self, environment):
        # args = Arguments(env=environment)
        # args.if_off_policy
        # args.env = environment

        # test on the testing env
        state, info = environment.reset()
        day = environment.sorted_times[environment.time_index]
        history = defaultdict(list)

        total_asset = environment.portfolio_value
        history["date"].append(day)
        history["total_assets"].append(total_asset)
        history["episode_return"].append(0)
        # episode_total_assets.append(environment.initial_amount)
        done = False
        while not done:
            action = self.act(state, info)
            state, reward, done, trunc, info = environment.step(action)
            day = environment.sorted_times[environment.time_index]

            total_asset = environment.portfolio_value
            episode_return = total_asset / environment.initial_amount
            history["date"].append(day)
            history["total_assets"].append(total_asset)
            history["episode_return"].append(episode_return)
        print("Test Finished!")
        # return episode total_assets on testing data
        print("episode_return", episode_return)
        return pd.DataFrame(history)
