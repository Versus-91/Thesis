import os
import pickle as pkl
import random
import sys
import time
from pprint import pprint

import optuna
from absl import flags
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor

from utils.optuna.ppo import sample_ppo_params
from utils.optuna.trial_eval_callback import TrialEvalCallback
from utils.portfolio_trainer import PortfolioOptimization
from sb3_contrib import RecurrentPPO
from stable_baselines3 import A2C
from stable_baselines3 import DDPG
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import TD3
import data_processor
FLAGS = flags.FLAGS
FLAGS(sys.argv)


study_path = "./studies/sp500_log+return"


def objective(trial: optuna.Trial) -> float:

    time.sleep(random.random() * 16)

    # step_mul = trial.suggest_categorical("step_mul", [4, 8, 16, 32, 64])
    # env_kwargs = {"step_mul": step_mul}

    sampled_hyperparams = sample_ppo_params(trial)

    portfolio_optimizer = PortfolioOptimization(
        transaction_fee=0.003, last_weight=False, vectorize=False)
    train_data, test_data, eval_data = data_processor.get_data()
    env_train = portfolio_optimizer.create_environment(
        train_data, ["close", "log_return"], window=5)
    env_evaluation = portfolio_optimizer.create_environment(
        test_data, ["close", "log_return"], window=5)

    path = f"{study_path}/trial_{str(trial.number)}"
    os.makedirs(path, exist_ok=True)

    # env = MoveToBeaconEnv(**env_kwargs)
    env_train = Monitor(env_train)
    env_evaluation = Monitor(env_evaluation)
    model = PPO("MlpPolicy", env=env_train, seed=142, verbose=0, device='cpu',
                tensorboard_log=path, **sampled_hyperparams)

    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=30, min_evals=50, verbose=1)
    eval_callback = TrialEvalCallback(
        env_evaluation, trial, best_model_save_path=path, log_path=path,
        n_eval_episodes=2, eval_freq=10000, deterministic=False, callback_after_eval=stop_callback
    )

    params = sampled_hyperparams
    with open(f"{path}/params.txt", "w") as f:
        f.write(str(params))

    try:
        model.learn(500_000, callback=eval_callback)
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

    sampler = TPESampler(n_startup_trials=10, multivariate=True)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=10)

    study = optuna.create_study(
        storage="sqlite:///db.sqlite3",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
        direction="maximize",
    )

    try:
        study.optimize(objective, n_jobs=-3, n_trials=128)
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
