from typing import Dict, Any, Union, Callable

import optuna
from torch import nn


def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """

        return progress_remaining * initial_value

    def __str__(self):
        return f'The linear scheduele learning rate is {self.name} and the age is {self.age}'
    return func


def sample_sac_params(trial: optuna.Trial) -> Dict[str, Any]:

    # Hyperparameters and ranges chosen for tuning
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 0.01)
    train_freq = trial.suggest_int("train_freq", 1, 10)
    gradient_steps = trial.suggest_int("gradient_steps", 1, 4)
    learning_starts = trial.suggest_int("learning_starts", 0, 1000)
    tau = trial.suggest_float("tau", 0.01, 1.0)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    buffer_size = trial.suggest_int("buffer_size", 20000, 100000)
    gamma = trial.suggest_categorical(
        "gamma", [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    suggested_learning_rate = trial.suggest_loguniform(
        "learning_rate", 1e-5, 0.01)
    lr_schedule = trial.suggest_categorical(
        'lr_schedule', ['linear', 'constant'])
    activation_fn = trial.suggest_categorical(
        'activation_fn', ['tanh', 'relu', 'swish', 'leaky_relu'])

    if lr_schedule == "linear":
        learning_rate = linear_schedule(suggested_learning_rate)
    else:
        learning_rate = suggested_learning_rate

    # Independent networks usually work best
    # when not working with images
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])
    net_arch = {
        "small": [dict(pi=[64, 64], vf=[64, 64])],
        "medium": [dict(pi=[128, 128], vf=[128, 128])],
    }[net_arch]
    activation_fn = {"swish": nn.SELU,
                     "leaky_relu": nn.LeakyReLU}[activation_fn]

    return ({
        "tau": tau,
        "batch_size": batch_size,
        "gamma": gamma,
        "buffer_size": buffer_size,
        "learning_rate": learning_rate,
        "learning_starts": learning_starts,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        # "sde_sample_freq": sde_sample_freq,
        "policy_kwargs": dict(
            # log_std_init=log_std_init,
            net_arch=net_arch,
            activation_fn=activation_fn,
        ),
    }, suggested_learning_rate)
