import matplotlib.pyplot as plt
import matplotlib
import scienceplots
import pandas as pd
import random
from matplotlib import pyplot as plt
matplotlib.use('Agg')


def plot_weights(weights, dates, tics, add_cash=True):
    w = pd.DataFrame(weights.tolist())
    columns = tics
    columns.append('date')
    w['date'] = dates
    if add_cash:
        w.columns = ['CASH']+columns
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


plt.rcdefaults()
plt.style.use(['science', 'ieee'])


def plot_mvo_weights(mvo_result, test_data, include_variance=True, figsize=(11, 8.5), weights_figsize=(4, 4), dpi=500, save_path=''):
    w = pd.DataFrame(mvo_result['action'])
    unique_tics = test_data.tic.unique().tolist()
    unique_tics.append('date')
    w['date'] = mvo_result['date']
    w.columns = unique_tics
    with plt.style.context('science', 'ieee'):
        fig, ((ax_main, ax_legend), (ax_below, ax_empty)) = plt.subplots(
            nrows=2, ncols=2,
            gridspec_kw={'width_ratios': [10, 1], 'height_ratios': [3, 1]},
            figsize=figsize,
            dpi=dpi
        )

        # Main plot
        w.plot(
            x='date', kind='area', stacked=True,
            colormap="icefire", ax=ax_main, alpha=0.8
        )
        ax_main.set_xlim(w.date.min(), w.date.max())
        ax_main.set_ylim(0, 1)
        ax_main.set_ylabel('Weights')

        # Legend
        ax_legend.axis('off')
        handles, labels = ax_main.get_legend_handles_labels()
        ax_main.legend().remove()
        ax_legend.legend(handles, labels, loc='center', ncol=1)

        # Variance plot (optional)
        if include_variance:
            ax_below.plot(w['date'], mvo_result['variance'],
                        label='Portfolio Variance')
            ax_below.set_xlim(w.date.min(), w.date.max())
            ax_below.set_ylabel('Variance')
            ax_below.tick_params(axis='x', rotation=30)
        else:
            fig.delaxes(ax_below)

        # Remove empty subplot
        fig.delaxes(ax_empty)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=500)
        plt.show()
        # Save or show


def plot_buy_and_hold_weights(env, test_data, ad_cash=True):
    w = pd.DataFrame(env._final_weights)
    columns = test_data.tic.unique().tolist()
    columns.append('date')
    w['date'] = env._date_memory
    if ad_cash:
        w.columns = ['Cash']+columns
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
