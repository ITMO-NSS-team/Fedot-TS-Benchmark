import os
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error

from pylab import rcParams
rcParams['figure.figsize'] = 12, 6

import warnings
warnings.filterwarnings('ignore')

from time_series_case_1.analysis.metric_tools import _create_report_dataframe,\
    get_ts_short_names, get_ts_long_names


def plot_results(actual_time_series, predicted_values, len_train_data,
                 y_name='Parameter'):
    """
    Function for drawing plot with predictions
    :param actual_time_series: the entire array with one-dimensional data
    :param predicted_values: array with predicted values
    :param len_train_data: number of elements in the training sample
    :param y_name: name of the y axis
    """

    plt.plot(np.arange(0, len(actual_time_series)),
             actual_time_series, label='Actual values', c='green')
    plt.plot(np.arange(len_train_data, len_train_data + len(predicted_values)),
             predicted_values, label='Predicted', c='blue')
    # Plot black line which divide our array into train and test
    plt.plot([len_train_data, len_train_data],
             [min(actual_time_series), max(actual_time_series)], c='black',
             linewidth=1)
    plt.ylabel(y_name, fontsize=15)
    plt.xlabel('Time index', fontsize=15)
    plt.legend(fontsize=15)
    plt.grid()
    plt.show()


def plot_mape_vs_len(path: str, mode: str):
    """
    Function plot simple plot MAPE vs forecast lengths for chosen time series
    """

    if mode == 'short':
        l_forecasts = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        ts_names = get_ts_short_names()
    elif mode == 'long':
        l_forecasts = np.arange(10, 1010, 10)
        ts_names = get_ts_long_names()
    else:
        raise NotImplementedError(f'Mode "{mode}" is not available')

    # Get metrics per time series for all short time series
    for ts_name in ts_names:
        dataframe = _create_report_dataframe(folder=path,
                                             time_series=ts_name)

        local_mapes = []
        for len_forecast in l_forecasts:
            local_df = dataframe[dataframe['Forecast len'] == len_forecast]
            mape_metric = float(local_df['MAPE'])

            local_mapes.append(mape_metric)

        plt.plot(l_forecasts, local_mapes, alpha=0.8, label=ts_name)

    plt.grid()
    plt.legend()
    plt.ylabel('MAPE', fontsize=15)
    plt.xlabel('Forecast length', fontsize=15)
    plt.show()


def plot_forecast(path: str, ts_label: str, forecast_len: int):

    acceptable_name = ''.join((str(forecast_len), '.csv'))
    forecast_df = pd.read_csv(os.path.join(path, acceptable_name))

    # Get predictions only for one time series
    ts_forecast_df = forecast_df[forecast_df['series_id'] == ts_label]
    ts_forecast_df['datetime'] = pd.to_datetime(ts_forecast_df['datetime'])
    dates = list(ts_forecast_df['datetime'])

    actual_data = np.array(ts_forecast_df['value'])
    predicted = np.array(ts_forecast_df['Predicted'])

    plt.plot(dates, actual_data, c='blue', alpha=0.5, label='Actual')
    plt.plot(dates[-(forecast_len + 1):], predicted[-(forecast_len + 1):], c='green', alpha=1.0, label='Predicted')
    # Plot black line which divide our array into train and test
    plt.plot([dates[-(forecast_len + 1)], dates[-(forecast_len + 1)]],
             [min(ts_forecast_df['value']), max(ts_forecast_df['value'])], c='black', linewidth=1)
    plt.grid()
    plt.legend()
    plt.show()





