from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics

import os
import timeit
import itertools

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt

from pylab import rcParams
rcParams['figure.figsize'] = 11, 4

# Custom metric functions and function for visualisation
from analysis.metric_tools import mean_absolute_percentage_error
from analysis.visualisation_tools import plot_results


def make_forecast(df, len_forecast: int):
    """
    Function for making time series forecasting with Prophet library

    :param df: dataframe to process
    :param len_forecast: forecast length

    :return predicted_values: forecast
    :return model_name: name of the model (always 'AutoTS')
    """

    df['ds'] = df['datetime']
    df['y'] = df['value']

    best_params = find_best_params(df, len_forecast)
    prophet_model = Prophet(**best_params)
    prophet_model.fit(df)

    future = prophet_model.make_future_dataframe(periods=len_forecast,
                                                 include_history=False)
    forecast = prophet_model.predict(future)

    predicted_values = np.array(forecast['yhat'])
    model_name = 'Prophet'
    return predicted_values, model_name


def find_best_params(df, len_forecast):
    """ Function find optimal changepoint_prior_scale value for data """

    # Split dataset for validation
    df_test = df.tail(len_forecast)
    actual_values = np.array(df_test['y'])
    df_train = df.head(len(df) - len_forecast)

    changepoint_range = [0.001, 0.05, 0.5]
    seasonality_range = [0.05, 10]
    mapes = []
    changepoints = []
    seasonality = []
    for changepoint_prior_scale_param in changepoint_range:
        for seasonality_prior_scale_param in seasonality_range:
            params = {'changepoint_prior_scale': changepoint_prior_scale_param,
                      'seasonality_prior_scale': seasonality_prior_scale_param}
            m = Prophet(**params).fit(df_train)
            future = m.make_future_dataframe(periods=len_forecast,
                                             include_history=False)
            forecast = m.predict(future)
            predicted_values = np.array(forecast['yhat'])

            # Calculate MAPE metric
            current_mape = mean_absolute_percentage_error(predicted_values, actual_values)

            # Update lists with info
            mapes.append(current_mape)
            changepoints.append(changepoint_prior_scale_param)
            seasonality.append(seasonality_prior_scale_param)

    # Find index of the smallest metric value
    mapes = np.array(mapes)
    min_id = np.argmin(mapes)

    # Find the best parameters
    best_params = {'changepoint_prior_scale': changepoints[min_id],
                   'seasonality_prior_scale': seasonality[min_id]}
    return best_params


def run_experiment(path, folder_to_save, l_forecasts, vis: bool = False):
    """ Function start the experiment

    :param path: path to the file
    :param folder_to_save: path to the folder where to save reports
    :param l_forecasts: list with forecast lengths
    :param vis: is visualisations needed or not
    """
    # Read dataframe with data
    df = pd.read_csv(path)
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Forecast lengths
    for len_forecast in l_forecasts:
        print(f'\nThe considering forecast length {len_forecast} elements')

        all_models = []
        all_maes = []
        all_mapes = []
        all_labels = []
        all_times = []
        for index, time_series_label in enumerate(df['series_id'].unique()):
            time_series_df = df[df['series_id'] == time_series_label]
            forecast_df = time_series_df.copy()
            true_values = np.array(time_series_df['value'])
            predicted_array = np.array(time_series_df['value'])

            # Got train, test parts, and the entire data
            dates = time_series_df['datetime']
            train_dates = dates[:-len_forecast]
            train_array = true_values[:-len_forecast]
            test_array = true_values[-len_forecast:]

            dataframe_process = pd.DataFrame({'datetime': train_dates,
                                              'value': train_array})
            print(len(np.argwhere(np.isnan(train_array))))
            if len(np.argwhere(np.isnan(train_array))) != 0:
                raise ValueError()

            dataframe_process['datetime'] = pd.to_datetime(
                dataframe_process['datetime'])

            start = timeit.default_timer()
            predicted_values, model_name = make_forecast(dataframe_process,
                                                         len_forecast=len_forecast)
            time_launch = timeit.default_timer() - start

            if vis:
                plot_results(actual_time_series=true_values,
                             predicted_values=predicted_values,
                             len_train_data=len(train_array),
                             y_name=time_series_label)

            # Metrics
            mae = mean_absolute_error(test_array, predicted_values)
            mape = mean_absolute_percentage_error(test_array, predicted_values)

            print(f'Model {model_name}, forecast length {len_forecast}, MAPE {mape:.2f}')

            all_models.append(model_name)
            all_maes.append(mae)
            all_mapes.append(mape)
            all_labels.append(time_series_label)
            all_times.append(time_launch)

            # Create dataframe with new column
            predicted_array[-len_forecast:] = predicted_values
            forecast_df['Predicted'] = predicted_array

            if index == 0:
                main_forecast_df = forecast_df
            else:
                frames = [main_forecast_df, forecast_df]
                main_forecast_df = pd.concat(frames)

        result = pd.DataFrame({'Models': all_models,
                               'MAE': all_maes,
                               'MAPE': all_mapes,
                               'Time series label': all_labels,
                               'Time': all_times})
        report_name = ''.join((str(len_forecast), '_report_', '.csv'))
        result.to_csv(os.path.join(folder_to_save, report_name), index=False)

        results_name = ''.join((str(len_forecast), '.csv'))
        main_forecast_df.to_csv(os.path.join(folder_to_save, results_name), index=False)


if __name__ == '__main__':
    ##########################################################################
    #                           Comparison with                              #
    #         Prophet library - https://facebook.github.io/prophet           #
    ##########################################################################

    # Paths to the files
    path_to_the_short_file = 'data/ts_short.csv'
    path_to_the_long_file = 'data/ts_long.csv'

    # Paths to the folders with report csv files
    path_to_save_short = 'results/prophet/short'
    path_to_save_long = 'results/prophet/long'

    # Lists with forecasts lengths
    l_forecasts_short = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    l_forecasts_long = np.arange(10, 1010, 10)

    # Launch for short time series
    run_experiment(path_to_the_short_file,
                   path_to_save_short,
                   l_forecasts_short,
                   vis=False)

    # Launch for short time series
    run_experiment(path_to_the_long_file,
                   path_to_save_long,
                   l_forecasts_long,
                   vis=False)
