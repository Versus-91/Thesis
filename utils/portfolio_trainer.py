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
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement


class PortfolioOptimization:
    def __init__(
        self,
        transaction_fee=0.001,
        starting_capital=250_000,
        comission_fee_model="wvm",
        normalize=None,
        last_weight=True,
        sharp_reward=False,
        remove_close=False,
        add_cash=False,
        flatten_state=False,
        clip_range=0.04,
        seed=42,
        env_num=4,
        decay_rate=0.01,
        tag='',
        vectorize=False,
        env=None
    ):
        self.transaction_fee = transaction_fee
        self.clip_range = clip_range
        self.flatten_state = flatten_state
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
        self.seed = seed
        self.add_cash = add_cash

    def make_env(self, rank, data, args):
        def _f():
            # Replace with your custom environment
            env = PortfolioOptimizationEnv(df=data, **args)
            env._seed(rank)  # Ensure each environment has a different seed
            return env
        return _f

    def create_environment(self, data, features, window, validate=None):
        env_kwargs = {
            "initial_amount": self.starting_capital,
            "features": features,
            "comission_fee_pct": self.transaction_fee,
            "time_window": window,
            "flatten_state": self.flatten_state,
            "clip_range": self.clip_range,
            "sharpe_reward": self.sharp_reward,
            "normalize_df": self.normalize,
            "comission_fee_model": self.comission_fee_model,
            "return_last_action": self.last_weight,
            "add_cash": self.add_cash,
            "is_validation": validate,
            "sr_decay_rate": self.decay_rate,
            "remove_close_from_state": self.remove_close,
        }

        environment = self.env(df=data, **env_kwargs)
        environment._seed(self.seed)
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
        stop_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=50, min_evals=100, verbose=1)
        eval_callback = EvalCallback(
            evaluation_environment,
            best_model_save_path="./data/"+model_name+"_"+path_post_fix + "best",
            log_path="./tensorboard_log/" + path_post_fix,
            eval_freq=2500,
            deterministic=True,
            n_eval_episodes=3,
            render=False,
            callback_after_eval=stop_callback
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

    def load_from_file(self, model_name, environment, cwd, deterministic=True):
        if model_name not in MODELS:
            raise ValueError(
                f"Model '{model_name}' not found in MODELS."
            )  # this is more informative than NotImplementedError("NotImplementedError")
        try:
            # load agent
            model = MODELS[model_name].load(cwd)
            print("Successfully load model", cwd)
        except BaseException as error:
            raise ValueError(
                f"Failed to load agent. Error: {str(error)}") from error

        # test on the testing env
        state = environment.reset()[0]
        done = False
        tiks = environment._tic_list.tolist()
        while not done:
            action = model.predict(state, deterministic=deterministic)[0]
            date_list = environment._date_memory
            portfolio_return = environment._portfolio_return_memory
            result_df = pd.DataFrame(
                {"date": date_list, "daily_return": portfolio_return,
                    'account':  environment._asset_memory["final"], 'weights': environment._final_weights}
            )
            state, reward, done, _, _ = environment.step(action)
        return result_df, tiks
