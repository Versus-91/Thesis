import pandas as pd
from environements.portfolio_optimization_env import PortfolioOptimizationEnv
from models import MODELS, DRLAgent
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from utils.model_helpers import DRL_prediction


class PortfolioOptimization:
    def __init__(
        self,
        transaction_fee=0.001,
        starting_capital=250_000,
        comission_fee_model="trf",
        normalize=None,
        last_weight=True,
        remove_close=True,
    ):
        self.transaction_fee = transaction_fee
        self.starting_capital = starting_capital
        self.comission_fee_model = comission_fee_model
        self.normalize = normalize
        self.last_weight = last_weight
        self.remove_close = remove_close

    def create_environment(self, data, features, window, seed=142):
        env_kwargs = {
            "initial_amount": self.starting_capital,
            "features": features,
            "comission_fee_pct": self.transaction_fee,
            "time_window": window,
            "normalize_df": self.normalize,
            "comission_fee_model": self.comission_fee_model,
            "return_last_action": self.last_weight,
            "remove_close_from_state": self.remove_close,
        }
        environment = PortfolioOptimizationEnv(df=data, **env_kwargs)
        environment._seed(seed)
        return environment

    def train_model(
        self,
        train_data,
        test_data,
        evaluation_data,
        model_name="a2c",
        iterations=100_000,
        features=["close", "log_return"],
        policy_network="MlpPolicy",
        args=None,
        window_size=5
    ):

        train_environment = self.create_environment(
            train_data, features, window=window_size)
        evaluation_environment = self.create_environment(
            evaluation_data, features, window=window_size)
        test_environment = self.create_environment(
            test_data, features, window=window_size)
        path_post_fix = "_".join(features) + "_" + str(self.transaction_fee) + "/"
        agent = DRLAgent(env=train_environment)
        model_agent = agent.get_model(
            model_name,
            policy=policy_network,
            tensorboard_log="./tensorboardlog_" + path_post_fix,
            model_kwargs=args,
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path="./data/history_main_" + path_post_fix,
            name_prefix=model_name,
        )

        eval_callback = EvalCallback(
            evaluation_environment,
            best_model_save_path="./data/history_main_"+path_post_fix + "best",
            log_path="./tensorboard_log/tensorboardlog_" + path_post_fix ,
            eval_freq=10000,
            deterministic=True,
            render=False,
        )
        model = agent.train_model(model=model_agent,
                                  tb_log_name=model_name,
                                  total_timesteps=iterations, checkpoint_callback=checkpoint_callback, eval_callbakc=eval_callback)
        agent.train_model(model=model,
                          tb_log_name=model_name,
                          total_timesteps=iterations, checkpoint_callback=checkpoint_callback, eval_callbakc=eval_callback)
        model.save('./data/trained_models/'+model_name+'_'+path_post_fix)
        model_performance_train = pd.DataFrame(
            {
                "date": train_environment._date_memory,
                "actions": train_environment._actions_memory,
                "weights": train_environment._final_weights,
                "returns": train_environment._portfolio_return_memory,
                "rewards": train_environment._portfolio_reward_memory,
                "portfolio_values": train_environment._asset_memory["final"],
            }
        )
        model_performance_test = DRL_prediction(
            model, test_environment, window_size)
        results = {'train': model_performance_train,
                   'test': model_performance_test, 'model': model}
        return results

    def load_from_file(model_name, environment, path):
        if model_name not in MODELS:
            raise ValueError(
                f"Model '{model_name}' not found in MODELS."
            )  # this is more informative than NotImplementedError("NotImplementedError")
        try:
            # load agent
            model = MODELS[model_name].load(path, environment)
            print("Successfully load model", path)
        except BaseException as error:
            raise ValueError(
                f"Failed to load agent. Error: {str(error)}") from error

        return model
