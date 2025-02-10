from functools import partial
import data_processor
from stable_baselines3 import TD3
from stable_baselines3 import SAC
from stable_baselines3 import DDPG
from stable_baselines3 import A2C
from sb3_contrib import RecurrentPPO
from environements.portfolio_optimization_env import PortfolioOptimizationEnv
from utils.portfolio_trainer import PortfolioOptimization
from utils.optuna.trial_eval_callback import TrialEvalCallback
from utils.optuna.ppo import sample_ppo_params
from environements.portfolio_optimization_env_flat import PortfolioOptimizationEnvFlat
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement
from stable_baselines3 import PPO
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from absl import flags
import optuna
import os
import pickle as pkl
import random
import sys
import time
from pprint import pprint

from pandas import read_csv
import warnings
warnings.filterwarnings("ignore")

FLAGS = flags.FLAGS
FLAGS(sys.argv)
df_dax = read_csv('./dataset/dax.csv')

df_dow = read_csv('./data/dow.csv')

study_path = "./studies/s3"


def objective(trial: optuna.Trial, sharpe_reward=False, commission=0, window_size=5, save_path="./studies/s3") -> float:
    print(
        f"sharpe reward: {sharpe_reward},commission:{commission}, window: {window_size}, path: {save_path}")
    time.sleep(random.random() * 16)

    # step_mul = trial.suggest_categorical("step_mul", [4, 8, 16, 32, 64])
    # env_kwargs = {"step_mul": step_mul}

    sampled_hyperparams, lr = sample_ppo_params(trial)
    df = df_dow.copy()
    df = df_dow[df_dow.tic.isin(['MSFT', 'UNH', 'DIS', 'GS', 'HD','V',"AXP","MCD","CAT","AMGN","TRV"])]

    if sharpe_reward:
        portfolio_optimizer = PortfolioOptimization(
            transaction_fee=0.00, comission_fee_model=None, vectorize=False, normalize=None,
            tag="ppo_alternative_state_11_asset", sharp_reward=False, last_weight=False, remove_close=True,
            add_cash=False, env=PortfolioOptimizationEnv)
    else:
        portfolio_optimizer = PortfolioOptimization(
            transaction_fee=0.00, comission_fee_model=None, vectorize=False, normalize=None,
            tag="ppo_alternative_state_11_asset", sharp_reward=False, last_weight=False, remove_close=True,
            add_cash=False, env=PortfolioOptimizationEnv)

    train_data, test_data, eval_data = data_processor.get_data(df)
    env_train = portfolio_optimizer.create_environment(
        train_data, ["close", "log_return", "momentum_return_21_normal",
                     "momentum_return_42_normal", "momentum_return_63_normal", "macd_normal", "rsi_normal"
                     ], window=window_size)
    env_evaluation = portfolio_optimizer.create_environment(
        eval_data, ["close", "log_return", "momentum_return_21_normal",
                    "momentum_return_42_normal", "momentum_return_63_normal", "macd_normal", "rsi_normal"
                    ], window=window_size)

    path = f"{save_path}/trial_{str(trial.number)}"
    os.makedirs(path, exist_ok=True)

    # env = MoveToBeaconEnv(**env_kwargs)
    env_train = Monitor(env_train)
    env_evaluation = Monitor(env_evaluation)
    model = PPO("MlpPolicy", env=env_train, seed=142, verbose=0, device='cpu',
                tensorboard_log=path, **sampled_hyperparams)

    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10, min_evals=30, verbose=1)
    eval_callback = TrialEvalCallback(
        env_evaluation, trial, best_model_save_path=path, log_path=path,
        n_eval_episodes=5, eval_freq=1500, deterministic=True, callback_after_eval=stop_callback
    )
    sampled_hyperparams['lr_log'] = lr
    params = sampled_hyperparams
    with open(f"{path}/params.txt", "w") as f:
        f.write(str(params))

    try:
        model.learn(1000_000, callback=eval_callback)
        env_train.close()
    except (AssertionError, ValueError) as e:
        env_train.close()
        print(e)
        print("============")
        print("Sampled params:")
        pprint(params)
        raise optuna.exceptions.TrialPruned()

    is_pruned = eval_callback.is_pruned
    reward = eval_callback.best_mean_reward

    del model.env
    del model

    if is_pruned:
        raise optuna.exceptions.TrialPruned()

    return reward


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("sharpe_reward")
    parser.add_argument("commission")
    parser.add_argument("window_size")
    parser.add_argument("save_path")

    args = parser.parse_args()
    if args.sharpe_reward:
        print(args.sharpe_reward)

    sampler = TPESampler(n_startup_trials=10, multivariate=True)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=10)

    study = optuna.create_study(
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
        direction="maximize",
    )
    objective = partial(objective, sharpe_reward=args.sharpe_reward,
                        commission=int(args.commission), window_size=int(args.window_size),  save_path=args.save_path)
    try:
        study.optimize(objective, n_jobs=20, n_trials=128)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    trial = study.best_trial
    print(f"Best trial: {trial.number}")
    print("Value: ", trial.value)

    print("Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    study.trials_dataframe().to_csv(f"{study_path}/report.csv")

    with open(f"{study_path}/study.pkl", "wb+") as f:
        pkl.dump(study, f)

    try:
        fig1 = plot_optimization_history(study)
        fig2 = plot_param_importances(study)
        fig3 = plot_parallel_coordinate(study)

        fig1.show()
        fig2.show()
        fig3.show()

    except (ValueError, ImportError, RuntimeError) as e:
        print("Error during plotting")
        print(e)
