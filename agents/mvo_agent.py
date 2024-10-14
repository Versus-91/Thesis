# example markowitz agent using info dict from environment to do optimization at each time step

import cvxpy as cp
import numpy as np
import pandas as pd
from collections import defaultdict


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
            risk_aversion=10,
            annual_risk_free_rate=0.03  # disregard risk free rate since RL disregards
    ):
        super().__init__()
        self.risk_aversion = risk_aversion
        self.env = env
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
        mean_returns = data[
            data["time"] == data["time"].max()
        ]["ewm_returns"].to_numpy()

        # solve markowitz model with cvxpy
        # initialize model
        num_stocks = len(mean_returns)
        weights = cp.Variable(num_stocks)
        risk_free_weight = cp.Variable(1)
        # define constraints
        # constraints = [cp.sum(weights) + risk_free_weight ==
        #                1, weights >= 0, risk_free_weight >= 0]
        constraints = [cp.sum(weights) == 1,
                       weights >= 0,
                       #    risk_free_weight >= 0
                       ]
        # define objective
        # + risk_free_weight*self.risk_free_rate
        portfolio_return = mean_returns @ weights
        portfolio_risk = cp.quad_form(weights, cov)
        # define objective
        objective = cp.Maximize(
            portfolio_return - self.risk_aversion * portfolio_risk)
        # define problem
        problem = cp.Problem(objective, constraints)
        # solve problem
        problem.solve()
        # get weights
        weights = weights.value
        # get action. if using risk free rate then integrate it into the action
        action = weights
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
