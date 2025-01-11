from pathlib import Path
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import gym
import matplotlib
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding

matplotlib.use("Agg")


class StockPortfolioEnv(gym.Env):
    """A single stock trading environment for OpenAI gym

    Attributes
    ----------
        df: DataFrame
            input data
        stock_dim : int
            number of unique stocks
        hmax : int
            maximum number of shares to trade
        initial_amount : int
            start money
        transaction_cost_pct: float
            transaction cost percentage per trade
        reward_scaling: float
            scaling factor for reward, good for training
        state_space: int
            the dimension of input features
        action_space: int
            equals stock dimension
        tech_indicator_list: list
            a list of technical indicator names
        turbulence_threshold: int
            a threshold to control risk aversion
        day: int
            an increment number to control date

    Methods
    -------
    _sell_stock()
        perform sell action based on the sign of the action
    _buy_stock()
        perform buy action based on the sign of the action
    step()
        at each step the agent will return actions, then
        we will calculate the reward, and return the next observation.
    reset()
        reset the environment
    render()
        use render to return other functions
    save_asset_memory()
        return account value at each time step
    save_action_memory()
        return actions/positions at each time step


    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        stock_dim,
        hmax,
        initial_amount,
        transaction_cost_pct,
        reward_scaling,
        state_space,
        action_space,
        tech_indicator_list,
        turbulence_threshold=None,
        lookback=252,
        time_index=0,
        window=1,
        cwd="./",
    ):
        # super(StockEnv, self).__init__()
        # money = 10 , scope = 1
        self.time_index = time_index
        self.lookback = lookback
        self.window = window
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.cwd = Path(cwd)
        self.results_file = self.cwd / "results" / "rl"
        self.results_file.mkdir(parents=True, exist_ok=True)

        self.df.time = pd.to_datetime(self.df.time)
        self.sorted_times = sorted(set(self.df.time))

        # action_space normalization and shape is self.stock_dim
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.action_space,))
        # Shape = (34, 30)
        # covariance matrix + technical indicators
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                self.state_space + len(self.tech_indicator_list),
                self.state_space,
            ),
        )

        # load data from a pandas dataframe
        day = self.sorted_times[self.time_index]
        # self.data = self.df[self.df["time"] == day]
        # self.covs = self.data["cov_list"].values[0]
        # self.state = np.append(
        #     np.array(self.covs),
        #     [self.data[tech].values.tolist() for tech in self.tech_indicator_list],
        #     axis=0,
        # )
        self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
        self.state, self.info = self.get_state_and_info_from_day(day)
        self.terminal = False
        self.turbulence_threshold = turbulence_threshold
        # initalize state: inital portfolio return + individual stock return + individual weights
        self.portfolio_value = self.initial_amount

        # memorize portfolio value each step
        self.asset_memory = [self.initial_amount]
        # memorize portfolio return each step
        self.portfolio_return_memory = [0]
        self.date_memory = [day]

    def step(self, actions):
        self.terminal = self.time_index >= len(self.sorted_times) - self.window

        if self.terminal:
            df = pd.DataFrame(
                {"date": self.date_memory, "daily_return": self.portfolio_return_memory}
            )
            df.set_index("date", inplace=True)
            plt.plot((1 + df.daily_return).cumprod()
                     * self.initial_amount, "r")
            plt.savefig(self.results_file / "cumulative_reward.png")
            plt.close()

            plt.plot(self.portfolio_return_memory, "r")
            plt.savefig(self.results_file / "rewards.png")
            plt.close()

            print("=================================")
            print(f"begin_total_asset:{self.asset_memory[0]}")
            print(f"end_total_asset:{self.portfolio_value}")

            df_daily_return = pd.DataFrame(self.portfolio_return_memory)
            df_daily_return.columns = ["daily_return"]
            if df_daily_return["daily_return"].std() != 0:
                sharpe = (
                    (252**0.5)
                    * df_daily_return["daily_return"].mean()
                    / df_daily_return["daily_return"].std()
                )
                print("Sharpe: ", sharpe)
            print("=================================")
            return self.state, self.reward, self.terminal, False, self.info

        else:
            if np.sum(actions) == 1 and np.min(actions) >= 0:
                weights = actions
            else:
                weights = self.softmax_normalization(actions)
            self.actions_memory.append(weights)
            last_day_memory = self.data

            # load next state
            self.time_index += self.window
            day = self.sorted_times[self.time_index]
            fees = 0
            self.state, self.info = self.get_state_and_info_from_day(day)
            if self.transaction_cost_pct != 0:
                delta_weights = weights - self.actions_memory[-2]
                # delta_assets = delta_weights[1:]  # disconsider
                # calculate fees considering weights modification
                diff = np.sum(np.abs(delta_weights * self.portfolio_value))
                if diff != 0:
                    fees = diff * self.transaction_cost_pct

            portfolio_return = sum(
                ((self.data.close.values / last_day_memory.close.values) - 1) * weights
            )
            # update portfolio value
            new_portfolio_value = self.portfolio_value * (1 + portfolio_return)
            self.portfolio_value = new_portfolio_value - fees

            # save into memory
            self.portfolio_return_memory.append(portfolio_return)
            self.date_memory.append(day)
            self.asset_memory.append(new_portfolio_value)

            # the reward is the new portfolio value or end portfolo value
            self.reward = new_portfolio_value
            # print("Step reward: ", self.reward)
            # self.reward = self.reward*self.reward_scaling

        return self.state, self.reward, self.terminal, False, self.info

    def reset(self):
        self.time_index = self.window
        self.asset_memory = [self.initial_amount]
        day = self.sorted_times[self.time_index]
        self.state, self.info = self.get_state_and_info_from_day(day)
        # self.data = self.df[self.df["time"] == day]
        # self.covs = self.data["cov_list"].values[0]
        # self.state = np.append(
        #     np.array(self.covs),
        #     [self.data[tech].values.tolist() for tech in self.tech_indicator_list],
        #     axis=0,
        # )
        self.portfolio_value = self.initial_amount
        # self.cost = 0
        # self.trades = 0
        self.terminal = False
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
        self.date_memory = [day]
        return self.state, self.info

    def get_state_and_info_from_day(self, day):
        self.data = self.df[self.df["time"] == day]
        covs = self.data["cov_list"].values[0]
        state = np.array(covs)
        info = {
            "data": self.df[self.df.time <= day],
            'before_last_weight': self.actions_memory[-2] if len(self.actions_memory) >= 2 else self.actions_memory[-1],
            'last_weight': self.actions_memory[-1]
        }
        return state, info

    def render(self, mode="human"):
        return self.state

    def softmax_normalization(self, actions):
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator / denominator
        return softmax_output

    def save_asset_memory(self):
        date_list = self.date_memory
        portfolio_return = self.portfolio_return_memory
        # print(len(date_list))
        # print(len(asset_list))
        df_account_value = pd.DataFrame(
            {"date": date_list, "daily_return": portfolio_return}
        )
        return df_account_value

    def save_action_memory(self):
        # date and close price length must match actions length
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ["date"]

        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.tic.values
        df_actions.index = df_date.date
        # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
