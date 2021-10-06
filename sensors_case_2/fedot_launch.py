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

from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.repository.dataset_types import DataTypesEnum

# Custom metric functions and function for visualisation
from time_series_case_1.analysis.metric_tools import mean_absolute_percentage_error
from time_series_case_1.analysis.visualisation_tools import plot_results


def make_forecast(df, len_forecast: int, time_series_label: str):
    """
    Function for making time series forecasting with Prophet library

    :param df: dataframe to process
    :param len_forecast: forecast length
    :param time_series_label: name of time series to process

    :return predicted_values: forecast
    :return model_name: name of the model (always 'AutoTS')
    """

    # Define parameters
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=len_forecast))

    # Init model for the time series forecasting
    model = Fedot(problem='ts_forecasting', task_params=task.task_params,
                  composer_params={'timeout': 1, 'preset': 'ultra_light_tun'},
                  preset='ultra_light_tun')

    input_data = InputData(idx=np.arange(0, len(df)),
                           features=np.array(df[time_series_label]),
                           target=np.array(df[time_series_label]),
                           task=task,
                           data_type=DataTypesEnum.ts)

    start_forecast = len(df)
    end_forecast = start_forecast + len_forecast
    predict_input = InputData(idx=np.arange(start_forecast, end_forecast),
                              features=np.array(df[time_series_label]),
                              target=np.array(df[time_series_label]),
                              task=task,
                              data_type=DataTypesEnum.ts)
    # Run AutoML model design in the same way
    pipeline = model.fit(features=input_data)
    predicted_values = model.predict(predict_input)

    model_name = 'FEDOT'
    return predicted_values, model_name


def run_experiment(path, folder_to_save, l_forecasts, vis: bool = False):
    """ Function start the experiment

    :param path: path to the file
    :param folder_to_save: path to the folder where to save reports
    :param l_forecasts: list with forecast lengths
    :param vis: is visualisations needed or not
    """
    if os.path.isdir(folder_to_save) is False:
        os.makedirs(folder_to_save)
    # Read dataframe with data
    df = pd.read_csv(path)
    # Generate datetime column for Prophet model
    df['datetime'] = pd.date_range(end='1/1/2021', periods=len(df), freq='min')
    df = df.dropna()

    # Forecast lengths
    for len_forecast in l_forecasts:
        print(f'\nThe considering forecast length {len_forecast} elements')

        all_models = []
        all_maes = []
        all_mapes = []
        all_labels = []
        all_times = []
        for index, time_series_label in enumerate(df.columns[1:24]):
            if time_series_label in ['icon', 'summary', 'Kitchen 38 [kW]']:
                pass
            else:
                time_series_df = df[['datetime', time_series_label]]
                time_series_df = time_series_df.dropna()
                # Clip time series
                time_series_df = time_series_df.tail(3000)

                forecast_df = time_series_df.copy()
                true_values = np.array(time_series_df[time_series_label])
                predicted_array = np.array(time_series_df[time_series_label])

                # Got train, test parts, and the entire data
                dates = time_series_df['datetime']
                train_dates = dates[:-len_forecast]
                train_array = true_values[:-len_forecast]
                test_array = true_values[-len_forecast:]

                dataframe_process = pd.DataFrame({'datetime': train_dates,
                                                  time_series_label: train_array})

                print(len(np.argwhere(np.isnan(train_array))))
                if len(np.argwhere(np.isnan(train_array))) != 0:
                    raise ValueError()

                dataframe_process['datetime'] = pd.to_datetime(dataframe_process['datetime'])

                start = timeit.default_timer()
                predicted_values, model_name = make_forecast(dataframe_process,
                                                             len_forecast=len_forecast,
                                                             time_series_label=time_series_label)
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

                # Rename columns before concatenation
                forecast_df = forecast_df.rename(columns={time_series_label: 'value'})
                forecast_df['series_id'] = [time_series_label]*len(forecast_df)
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
    # Paths to the files
    path_to_file = 'data/home_sensors.csv'

    # Paths to the folders with report csv files
    path_to_save = 'results/fedot'

    # Lists with forecasts lengths
    l_forecasts = [90, 100]

    # Launch for short time series
    run_experiment(path_to_file,
                   path_to_save,
                   l_forecasts,
                   vis=False)
