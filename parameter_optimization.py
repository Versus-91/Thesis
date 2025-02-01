import data_processor
from stable_baselines3 import TD3
from stable_baselines3 import SAC
from stable_baselines3 import DDPG
from stable_baselines3 import A2C
from sb3_contrib import RecurrentPPO
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


study_path = "./studies/s1"


def objective(trial: optuna.Trial) -> float:

    time.sleep(random.random() * 16)

    # step_mul = trial.suggest_categorical("step_mul", [4, 8, 16, 32, 64])
    # env_kwargs = {"step_mul": step_mul}

    sampled_hyperparams = sample_ppo_params(trial)
    df = df_dax.copy()
    df = df[df.tic.isin(['ADS.DE', 'ALV.DE', 'BAS.DE', 'BAYN.DE',
                        'BMW.DE', 'CON.DE', 'DBK.DE', 'DTE.DE', 'EOAN.DE'])]
    portfolio_optimizer = PortfolioOptimization(
        transaction_fee=0.003, env=PortfolioOptimizationEnvFlat, last_weight=False)
    train_data, test_data, eval_data = data_processor.get_data(df)
    env_train = portfolio_optimizer.create_environment(
        train_data, ["close", "log_return", "volatility"], window=21)
    env_evaluation = portfolio_optimizer.create_environment(
        eval_data, ["close", "log_return", "volatility"], window=21)

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
        n_eval_episodes=1, eval_freq=1000, deterministic=False, callback_after_eval=stop_callback
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
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
        direction="maximize",
    )

    try:
        study.optimize(objective, n_jobs=-1, n_trials=128)
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
