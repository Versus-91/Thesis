import random
from matplotlib import pyplot as plt
import pandas as pd
import scienceplots
import pandas as pd

def plot_weights(weights, dates, tics, add_cash=True):
    w = pd.DataFrame(weights.tolist())
    columns = tics
    columns.append('date')
    w['date'] = dates
    if add_cash:
        w.columns = ['cash']+columns
    else:
        w.columns = columns

    with plt.style.context('science', 'ieee'):
        fig, (ax_main, ax_legend) = plt.subplots(
            ncols=2,
            gridspec_kw={'width_ratios': [10, 1]},  # Adjust width ratios
            figsize=(12, 6)
        )
        w.plot(x='date', figsize=(12, 4), kind='area', stacked=True,
               colormap="icefire", ax=ax_main, alpha=0.8)
        ax_main.set_xlim(w.date.min(), w.date.max())
        ax_main.set_ylim(0, 1)
        ax_main.set_ylabel('Weights')
        ax_main.set_xlabel('Dates')
        ax_legend.axis('off')
        handles, labels = ax_main.get_legend_handles_labels()
        ax_main.legend().remove()
        # Adjust ncol for horizontal legend
        ax_legend.legend(handles, labels, loc='center', ncol=1)
        plt.tight_layout()
        plt.savefig('./figures/'+str(random.randint(0, 1_000_000))+'.png')


def plot_mvo_weights(mvo_result, test_data):
    w = pd.DataFrame(mvo_result['action'])
    unique_tics = test_data.tic.unique().tolist()
    unique_tics.append('date')
    w['date'] = mvo_result['date']
    w.columns = unique_tics
    with plt.style.context('science', 'ieee'):
        fig, (ax_main, ax_legend) = plt.subplots(
            ncols=2,
            gridspec_kw={'width_ratios': [10, 1]},  # Adjust width ratios
            figsize=(12, 6)
        )
        w.plot(x='date', figsize=(12, 4), kind='area', stacked=True,
               colormap="icefire", ax=ax_main, alpha=0.8)
        ax_main.set_xlim(w.date.min(), w.date.max())
        ax_main.set_ylim(0, 1)
        ax_main.set_ylabel('Weights')
        ax_main.set_xlabel('Dates')
        ax_legend.axis('off')
        handles, labels = ax_main.get_legend_handles_labels()
        ax_main.legend().remove()
        # Adjust ncol for horizontal legend
        ax_legend.legend(handles, labels, loc='center', ncol=1)
        plt.tight_layout()
        plt.show()


def plot_buy_and_hold_weights(env, test_data):
    w = pd.DataFrame(env._final_weights)
    columns = test_data.tic.unique().tolist()
    columns.append('date')
    w['date'] = env._date_memory
    w.columns = ['Cash']+columns
    with plt.style.context('science', 'ieee'):
        fig, (ax_main, ax_legend) = plt.subplots(
            ncols=2,
            gridspec_kw={'width_ratios': [10, 1]},  # Adjust width ratios
            figsize=(12, 6)
        )
        w.plot(x='date', figsize=(12, 4), kind='area', stacked=True,
               colormap="icefire", ax=ax_main, alpha=0.8)
        ax_main.set_xlim(w.date.min(), w.date.max())
        ax_main.set_ylim(0, 1)
        ax_main.set_ylabel('Weights')
        ax_main.set_xlabel('Dates')
        ax_legend.axis('off')
        handles, labels = ax_main.get_legend_handles_labels()
        ax_main.legend().remove()
        # Adjust ncol for horizontal legend
        ax_legend.legend(handles, labels, loc='center', ncol=1)
    plt.tight_layout()
