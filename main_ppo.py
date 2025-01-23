from utils.portfolio_trainer import PortfolioOptimization
from pandas import read_csv
from utils.feature_engineer import FeatureEngineer
from utils.helpers import data_split
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    def add_volatility(df, periods=21):
        rolling_volatility = df.groupby(
            'tic')['log_return'].rolling(window=periods).std()
        rolling_volatility = rolling_volatility.reset_index(level=0, drop=True)
        # Assign the annualized volatility back to the original DataFrame
        df['volatility'] = rolling_volatility

        # Fill missing values with 0 (first periods will have NaN)
        df['volatility'].fillna(0, inplace=True)

        return df

    df_dow = read_csv('./data/dow.csv')
    df_nasdaq = read_csv('./data/nasdaq.csv')
    df_hsi = read_csv('./data/hsi.csv')
    df_dax = read_csv('./data/dax.csv')
    df_sp500 = read_csv('./data/sp500.csv')
    df = df_dow.copy()
    TRAIN_START_DATE = '2014-01-01'
    TRAIN_END_DATE = '2019-12-30'

    VALIDATION_START_DATE = '2020-01-01'
    VALIDATION_END_DATE = '2020-12-30'

    TEST_START_DATE = '2021-01-01'
    TEST_END_DATE = '2024-10-01'
    INDICATORS = [
        "close_21_ema",
        "close_62_ema"
    ]

    fe = FeatureEngineer(use_technical_indicator=True,
                         tech_indicator_list=INDICATORS,
                         use_turbulence=False,
                         user_defined_feature=True)

    processed_dax = fe.preprocess_data(df.query('date>"2013-01-01"'))
    cleaned_data = processed_dax.copy()
    cleaned_data = cleaned_data.fillna(0)
    cleaned_data = cleaned_data.replace(np.inf, 0)
    train_data = data_split(cleaned_data, TRAIN_START_DATE, TRAIN_END_DATE)
    test_data = data_split(cleaned_data, TEST_START_DATE, TEST_END_DATE)
    validation_data = data_split(
        cleaned_data, VALIDATION_START_DATE, VALIDATION_END_DATE)
    stock_dimension = len(train_data.tic.unique())
    state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
    print(f"Stock Dimension: {stock_dimension}")
    pv = train_data.pivot(columns='tic', values='close')
    corr = pv.corr()
    data = corr.copy()
    high_corr_pairs = set()
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > 0.9:
                col_name = corr.columns[i]
                high_corr_pairs.add(col_name)

    # Step 3: Drop one of the correlated columns
    reduced_data = data.drop(columns=high_corr_pairs)
    print("Original columns:", data.columns.tolist())
    print("Columns retained after filtering:", reduced_data.columns.tolist())
    reduced_data = reduced_data.drop(high_corr_pairs)
    reduced_data.style.background_gradient(cmap='coolwarm')

    columns = reduced_data.columns.tolist()
    cleaned_data = cleaned_data[cleaned_data.tic.isin(columns)]

    cleaned_data = add_volatility(cleaned_data)
    train_data = data_split(cleaned_data, TRAIN_START_DATE, TRAIN_END_DATE)
    test_data = data_split(cleaned_data, TEST_START_DATE, TEST_END_DATE)
    validation_data = data_split(
        cleaned_data, VALIDATION_START_DATE, VALIDATION_END_DATE)
    stock_dimension = len(train_data.tic.unique())
    state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
    print(f"Stock Dimension: {stock_dimension}")

    optimizer = PortfolioOptimization(
        transaction_fee=0.003, vectorize=False, tag="with_weight_state_128_128_net", sharp_reward=True)
    optimizer.train_model(train_data,
                          validation_data,
                          features=["close", "log_return"],
                          policy_network="MultiInputPolicy",
                          model_name="ppo",
                          window_size=5,
                          policy_kwargs=dict(
                              net_arch=[
                                  dict(pi=[128, 128], vf=[128, 128])
                              ]
                          ),
                          args={
                              "n_steps": 2048,
                              "ent_coef": 0.02,
                              "learning_rate": 3e-4,
                              "batch_size": 256,
                          },
                          iterations=1000_000)
    # model = optimizer.load_from_file(
    #     'ppo', path="data\RecurrentPPO_close_log_return_window_size_5_0.003_wth_weith_state\RecurrentPPO_500000_steps")
    # test_result = optimizer.DRL_prediction(
    #     model, test_data, ["close", "log_return"])
    # from utils.plotting_helpers import plot_weights
    # plot_weights(test_result[0].weights, test_result[0].date, test_result[1])
