import numpy as np
import pandas as pd
from environements.portfolio_optimization_env import PortfolioOptimizationEnv
from models import MODELS, DRLAgent
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from utils.helpers import data_split
from utils.model_helpers import DRL_prediction
from stable_baselines3.common.vec_env import DummyVecEnv
from datetime import datetime, timedelta


class PortfolioOptimization:
    def __init__(
        self,
        transaction_fee=0.001,
        starting_capital=250_000,
        comission_fee_model="wvm",
        normalize=None,
        last_weight=True,
        sharp_reward=False,
        remove_close=True,
        add_cash=True,
        env_num=4,
        decay_rate=0.01,
        tag='',
        vectorize=False,
        env=None
    ):
        self.transaction_fee = transaction_fee
        self.starting_capital = starting_capital
        self.comission_fee_model = comission_fee_model
        self.normalize = normalize
        self.last_weight = last_weight
        self.remove_close = remove_close
        self.env_num = env_num
        self.sharp_reward = sharp_reward
        self.decay_rate = decay_rate
        self.vectorize = vectorize
        self.tag = tag
        self.env = env
        self.add_cash = add_cash

    def make_env(self, rank, data, args):
        def _f():
            # Replace with your custom environment
            env = PortfolioOptimizationEnv(df=data, **args)
            env._seed(rank)  # Ensure each environment has a different seed
            return env
        return _f

    def create_environment(self, data, features, window, seed=142, validate=None):
        env_kwargs = {
            "initial_amount": self.starting_capital,
            "features": features,
            "comission_fee_pct": self.transaction_fee,
            "time_window": window,
            "sharpe_reward": self.sharp_reward,
            "normalize_df": self.normalize,
            "comission_fee_model": self.comission_fee_model,
            "return_last_action": self.last_weight,
            "add_cash": self.add_cash,
            "sr_decay_rate": self.decay_rate,
            "validate": validate,
            "remove_close_from_state": self.remove_close,
        }
        if self.vectorize:
            start = datetime.strptime(data.iloc[0].date, "%Y-%m-%d")
            end = datetime.strptime(
                data.iloc[data.shape[0]-1].date, "%Y-%m-%d")
            delta = end - start
            segments_number = delta / self.env_num
            segments = []
            for i in range(self.env_num):
                segment_start = start + (i * segments_number)
                segment_end = start + ((i+1) * segments_number)
                segments.append((segment_start, segment_end))
            data_splits = []
            for i in range(self.env_num):
                data_splits.append(data_split(
                    data, str(segments[i][0]), str(segments[i][1])))
            env_fns = [self.make_env(i, data_splits[i], env_kwargs)
                       for i in range(self.env_num)]
            env = DummyVecEnv(env_fns)
            return env
        else:
            environment = self.env(df=data, **env_kwargs)
            environment._seed(seed)
            return environment

    def train_model(
        self,
        train_data,
        evaluation_data,
        model_name="a2c",
        iterations=100_000,
        features=["close", "log_return"],
        policy_network="MlpPolicy",
        args=None,
        policy_kwargs=None,
        window_size=5
    ):

        train_environment = self.create_environment(
            train_data, features, window=window_size)
        evaluation_environment = self.create_environment(
            evaluation_data, features, window=window_size, validate=True)
        path_post_fix = "_".join(features) + "_window_size_"+str(window_size)+"_" + \
            str(self.transaction_fee) + "_"+self.tag + "/"
        agent = DRLAgent(env=train_environment)
        model_agent = agent.get_model(
            model_name,
            policy=policy_network,
            policy_kwargs=policy_kwargs,
            tensorboard_log="./tensorboard_log/" + path_post_fix,
            model_kwargs=args,
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path="./data/"+model_name+"_" + path_post_fix,
            name_prefix=model_name,
        )

        eval_callback = EvalCallback(
            evaluation_environment,
            best_model_save_path="./data/"+model_name+"_"+path_post_fix + "best",
            log_path="./tensorboard_log/" + path_post_fix,
            eval_freq=5000,
            deterministic=True,
            n_eval_episodes=1,
            render=False,
        )
        model = agent.train_model(model=model_agent,
                                  tb_log_name=model_name,
                                  total_timesteps=iterations, checkpoint_callback=checkpoint_callback, eval_callbakc=eval_callback)
        # model.save('./data/trained_models/'+model_name+'_'+path_post_fix)
        # model_performance_train = pd.DataFrame(
        #     {
        #         "date": train_environment._date_memory,
        #         "actions": train_environment._actions_memory,
        #         "weights": train_environment._final_weights,
        #         "returns": train_environment._portfolio_return_memory,
        #         "rewards": train_environment._portfolio_reward_memory,
        #         "portfolio_values": train_environment._asset_memory["final"],
        #     }
        # )
        # model_performance_test = DRL_prediction(
        #     model, test_environment, window_size)
        # results = {'train': model_performance_train,
        #            'test': model_performance_test, 'model': model}
        return model

    def load_from_file(self, model_name, path):
        if model_name not in MODELS:
            raise ValueError(
                f"Model '{model_name}' not found in MODELS."
            )  # this is more informative than NotImplementedError("NotImplementedError")
        try:
            # load agent
            model = MODELS[model_name].load(path)
            print("Successfully load model", path)
        except BaseException as error:
            raise ValueError(
                f"Failed to load agent. Error: {str(error)}") from error

        return model

    def DRL_prediction(self, model, test_data, features, deterministic=True, t=5):

        test_environment = self.create_environment(
            test_data, features, window=t)
        """make a prediction and get results"""
        test_env, test_obs = test_environment.get_sb_env()
        account_memory = None  # This help avoid unnecessary list creation
        actions_memory = None  # optimize memory consumption
        # state_memory=[] #add memory pool to store states

        test_env.reset()
        max_steps = len(test_environment._df.index.unique()) - 1 - t

        for i in range(len(test_environment._df.index.unique())):
            action, _states = model.predict(
                test_obs, deterministic=deterministic)
            # account_memory = test_env.env_method(method_name="save_asset_memory")
            # actions_memory = test_env.env_method(method_name="save_action_memory")
            test_obs, rewards, dones, info = test_env.step(action)

            if (
                i == max_steps
            ):  # more descriptive condition for early termination to clarify the logic
                date_list = test_environment._date_memory
                portfolio_return = test_environment._portfolio_return_memory
                result_df = pd.DataFrame(
                    {"date": date_list, "daily_return": portfolio_return,
                        'account':  test_environment._asset_memory["final"], 'weights': test_environment._final_weights}
                )
                tiks = test_environment._tic_list.tolist()

            if dones[0]:
                print("hit end!")
                break
        return result_df, tiks
