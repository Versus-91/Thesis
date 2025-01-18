
import pandas as pd
from environements.portfolio_optimization_env import PortfolioOptimizationEnv
from models import DRLAgent
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback


def DRL_prediction(model, environment, time_window, deterministic=True):
    """make a prediction and get results"""
    test_env, test_obs = environment.get_sb_env()
    account_memory = None  # This help avoid unnecessary list creation
    actions_memory = None  # optimize memory consumption
    # state_memory=[] #add memory pool to store states

    test_env.reset()
    max_steps = len(environment._df.date.unique()) - (time_window) - 1

    for i in range(len(environment._df.index.unique())):
        action, _states = model.predict(test_obs, deterministic=deterministic)
        test_obs, rewards, dones, info = test_env.step(action)
        if i == max_steps:
            date_list = environment._date_memory
            portfolio_return = environment._portfolio_return_memory
            result = pd.DataFrame(
                {"date": date_list, "daily_return": portfolio_return,
                    'account':  environment._asset_memory["final"], 'weights': environment._final_weights}
            )
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            action_list = environment._actions_memory
            df_actions = pd.DataFrame(action_list)
            tiks = environment._tic_list
            df_actions.index = df_date.date
            result_df = result
            actions_memory = df_actions

        if dones[0]:
            print("hit end!")
            break
    return result_df, actions_memory


def train_model(train_data, test_data, evaluation_data, transaction_fee=0.001, use_sharpe=False, use_dsr=False, use_sortino=False, model_name='a2c', iterations=100_000, save=True, load=False, tag='tag', features=["close", "log_return"], sharpe_reward=False, t=5, args=None, starting_capital=250_000, model_path=None,
                comission_fee_model='wvm', normalize="by_previous_time", load_continue=False, last_weight=True, remove_close=True, policy_network="MlpPolicy"):
    env_kwargs = {
        "initial_amount": starting_capital,
        "features": features,
        'comission_fee_pct': transaction_fee,
        'time_window': t,
        "normalize_df": normalize,
        'comission_fee_model': comission_fee_model,
        "return_last_action": last_weight,
        "remove_close_from_state": remove_close,
        # 'use_sortino':use_sortino,
        # 'use_differentail_sharpe_ratio':use_dsr,
    }
    train_environment = PortfolioOptimizationEnv(df=train_data, **env_kwargs)
    train_environment._seed(142)

    evaluation_environment = PortfolioOptimizationEnv(
        df=evaluation_data, **env_kwargs)
    evaluation_environment._seed(142)

    test_environment = PortfolioOptimizationEnv(df=test_data, **env_kwargs)
    test_environment._seed(142)
    agent = DRLAgent(env=train_environment)
    model_agent = agent.get_model(model_name,
                                  policy=policy_network,
                                  tensorboard_log='./tensorboardlog_' +
                                  '_'.join(features)+'_'+tag+'/',
                                  model_kwargs=args)

    checkpoint_callback = CheckpointCallback(save_freq=10000,
                                             save_path='./data/history_main_' +
                                             '.'.join(features)+'_'+tag+'/',
                                             name_prefix=model_name)
    eval_callback = EvalCallback(
        evaluation_environment,
        best_model_save_path='./data/history_main_' +
        '.'.join(features)+'_'+tag+'/best',
        # Path to save evaluation logs
        log_path='./tensorboardlog_'+'_'.join(features)+'/',
        eval_freq=5000,                            # Evaluate every 5000 steps
        # Use deterministic actions during evaluation
        deterministic=True,
        # Don't render the environment during evaluation
        render=False
    )
    if not load:
        if load_continue == False:
            model = agent.train_model(model=model_agent,
                                      tb_log_name=model_name,
                                      total_timesteps=iterations, checkpoint_callback=checkpoint_callback, eval_callbakc=eval_callback)
        else:
            model = model_agent.load(model_path, env=train_environment)
            agent.train_model(model=model,
                              tb_log_name=model_name,
                              total_timesteps=iterations, checkpoint_callback=checkpoint_callback, eval_callbakc=eval_callback)
    else:
        print('loading model')
        if model_path == None:
            model = model_agent.load(
                './data/trained_models_2025/'+str(model_name)+'_'+str(iterations)+'_' + tag)
        else:
            model = model_agent.load(model_path)
    if save and not load:
        model.save('./data/trained_models/'+str(model_name) +
                   '_'+str(iterations)+'_' + tag)
    metrics_df_dax = pd.DataFrame(
        {
            "date": train_environment._date_memory,
            "actions": train_environment._actions_memory,
            "weights": train_environment._final_weights,
            "returns": train_environment._portfolio_return_memory,
            "rewards": train_environment._portfolio_reward_memory,
            "portfolio_values": train_environment._asset_memory["final"],
        }
    )
    ppo_predictions = DRL_prediction(model, test_environment, t)
    results = {'train': metrics_df_dax,
               'test': ppo_predictions, 'model': model_agent}
    return results
