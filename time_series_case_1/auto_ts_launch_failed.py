from auto_pip ts import auto_timeseries

import os
import timeit
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt

from pylab import rcParams
rcParams['figure.figsize'] = 11, 4

# Custom metric functions and function for visualisation
from time_series_case_1.analysis.metric_tools import mean_absolute_percentage_error
from time_series_case_1.analysis.visualisation_tools import plot_results


def run_experiment(dataframe, folder_to_save, l_forecasts, make_forecast_func,
                   vis: bool = False):
    """ Function start the experiment

    :param dataframe: dataframe to process
    :param folder_to_save: path to the folder where to save reports
    :param l_forecasts: list with forecast lengths
    :param make_forecast_func: function which can make forecast
    :param vis: is visualisations needed or not
    """

    # Forecast lengths
    for len_forecast in l_forecasts:
        print(f'\nThe considering forecast length {len_forecast} elements')

        all_models = []
        all_maes = []
        all_mapes = []
        all_labels = []
        all_times = []
        for index, time_series_label in enumerate(dataframe['series_id'].unique()):
            time_series_df = dataframe[dataframe['series_id'] == time_series_label]
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

            dataframe_process['datetime'] = pd.to_datetime(dataframe_process['datetime'])

            start = timeit.default_timer()
            predicted_values, model_name = make_forecast_func(dataframe_process,
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

            print(
                f'Model {model_name}, forecast length {len_forecast}, MAPE {mape:.2f}')

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
        main_forecast_df.to_csv(os.path.join(folder_to_save, results_name),
                                index=False)


def make_forecast_monthly(df, len_forecast: int):
    """
    Function for making time series forecasting with auto_timeseries library

    :param df: dataframe to process
    :param len_forecast: forecast length

    :return predicted_values: forecast
    :return model_name: name of the model (always 'AutoTS')
    """

    # Get year and month
    df['Month'] = df['datetime'].dt.to_period('M')
    df['Month'] = pd.Series(df['Month'], dtype=str)
    df.drop(columns=['datetime'], inplace=True)

    model = auto_timeseries(score_type='rmse',
                            time_interval='M',
                            non_seasonal_pdq=None,
                            seasonality=False,
                            seasonal_period=12,
                            model_type=['best'],
                            verbose=2)
    model.fit(traindata=df,
              ts_column='Month',
              target='value',
              cv=3,
              sep=',')

    try:
        future_predictions = model.predict(testdata=len_forecast)
        predicted_values = np.array(future_predictions['yhat'])
    except Exception:
        series_to_predict = df.tail(len_forecast)
        future_predictions = model.predict(testdata=series_to_predict)
        predicted_values = np.array(future_predictions['yhat'])
    model_name = 'Auto_Timeseries'

    """
    TODO 
    Problem 1 - for different models that turn out to be the best, the forecast 
    length is different (sometimes it takes 5 elements - default values)
    
    Problem 2 - for for unknown reasons, matplotlib crashes with an error (when 
    trying to plot graphs by auto_ts (is it possible not to draw these graphs?))
    """
    return predicted_values, model_name


def run_monthly_data_experiment(path, folder_to_save, vis):
    l_forecasts = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # Read dataframe with data
    df = pd.read_csv(path, engine='python')
    df['datetime'] = pd.to_datetime(df['datetime'])

    run_experiment(dataframe=df, folder_to_save=folder_to_save,
                   l_forecasts=l_forecasts,
                   make_forecast_func=make_forecast_monthly, vis=vis)


if __name__ == '__main__':
    ###########################################################################
    #                             Comparison with                             #
    #      Auto_TimeSeries library - https://github.com/AutoViML/Auto_TS      #
    ###########################################################################

    # Paths to the files
    path_to_the_short_file = 'data/ts_short.csv'
    path_to_the_long_file = 'data/ts_long.csv'

    # Paths to the folders with report csv files
    path_to_save_short = 'results/auto_ts/short'
    path_to_save_long = 'results/auto_ts/long'

    # Make forecasts on "short" time series (monthly discreteness)
    run_monthly_data_experiment(path_to_the_short_file, path_to_save_short,
                                vis=True)
