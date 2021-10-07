import os
import timeit
import itertools
import datetime

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_absolute_error
import timeit
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from fedot.core.chains.tuning.unified import ChainTuner
from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.composer.gp_composer.gp_composer import \
    GPComposerBuilder, GPComposerRequirements
from fedot.core.composer.optimisers.gp_comp.gp_optimiser import GPChainOptimiserParameters
from fedot.core.composer.optimisers.gp_comp.operators.mutation import MutationTypesEnum
from fedot.core.composer.visualisation import ChainVisualiser
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import \
    MetricsRepository, RegressionMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

from pylab import rcParams
rcParams['figure.figsize'] = 11, 4

# Custom metric functions and function for visualisation
from analysis.metric_tools import mean_absolute_percentage_error
from analysis.visualisation_tools import plot_results


def get_source_chain():
    """
    Return chain with the following structure:
    lagged - ridge \
                    -> ridge
    lagged - ridge /
    """

    # First level
    node_lagged_1 = PrimaryNode('lagged')
    node_lagged_2 = PrimaryNode('lagged')

    # Second level
    node_ridge_1 = SecondaryNode('ridge', nodes_from=[node_lagged_1])
    node_ridge_2 = SecondaryNode('ridge', nodes_from=[node_lagged_2])

    # Third level - root node
    node_final = SecondaryNode('ridge', nodes_from=[node_ridge_1, node_ridge_2])
    chain = Chain(node_final)

    return chain


def display_chain_info(chain):
    """ Function print info about chain """

    print('\nObtained chain:')
    list_with_nodes = []
    for node in chain.nodes:
        print(f'{node.operation.operation_type}, params: {node.custom_params}')
        list_with_nodes.append(node.operation.operation_type)
    depth = int(chain.depth)
    print(f'Chain depth {depth}\n')

    return list_with_nodes


def get_available_operations():
    """ Function returns available operations for primary and secondary nodes """
    primary_operations = ['lagged', 'smoothing', 'gaussian_filter', 'ar']
    secondary_operations = ['lagged', 'ridge', 'lasso', 'knnreg', 'linear',
                            'scaling', 'ransac_lin_reg']
    return primary_operations, secondary_operations


def prepare_input_data(len_forecast, train_data_features, train_data_target,
                       test_data_features):
    """ Function return prepared data for fit and predict

    :param len_forecast: forecast length
    :param train_data_features: time series which can be used as predictors for train
    :param train_data_target: time series which can be used as target for train
    :param test_data_features: time series which can be used as predictors for prediction

    :return train_input: Input Data for fit
    :return predict_input: Input Data for predict
    :return task: Time series forecasting task with parameters
    """

    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=len_forecast))

    train_input = InputData(idx=np.arange(0, len(train_data_features)),
                            features=train_data_features,
                            target=train_data_target,
                            task=task,
                            data_type=DataTypesEnum.ts)

    start_forecast = len(train_data_features)
    end_forecast = start_forecast + len_forecast
    predict_input = InputData(idx=np.arange(start_forecast, end_forecast),
                              features=test_data_features,
                              target=None,
                              task=task,
                              data_type=DataTypesEnum.ts)

    return train_input, predict_input, task


def fit_predict_for_chain(chain, train_input, predict_input):
    """ Function apply fit and predict operations

    :param chain: chain to process
    :param train_input: InputData for fit
    :param predict_input: InputData for predict

    :return preds: prediction of the chain
    """
    # Fit it
    chain.fit_from_scratch(train_input)

    # Predict
    predicted_values = chain.predict(predict_input)
    # Convert to one dimensional array
    preds = np.ravel(np.array(predicted_values.predict))

    return preds


def make_forecast(df, len_forecast: int):
    """
    Function for making time series forecasting with AutoTS library

    :param df: dataframe to process
    :param len_forecast: forecast length

    :return predicted_values: forecast
    :return model_name: name of the model (always 'AutoTS')
    """

    time_series = np.array(df['value'])
    train_input, predict_input, task = prepare_input_data(len_forecast=len_forecast,
                                                          train_data_features=time_series,
                                                          train_data_target=time_series,
                                                          test_data_features=time_series)

    # Get chain with pre-defined structure
    init_chain = get_source_chain()

    # Init check
    preds = fit_predict_for_chain(chain=init_chain,
                                  train_input=train_input,
                                  predict_input=predict_input)

    # Get available_operations type
    primary_operations, secondary_operations = get_available_operations()

    # Composer parameters
    composer_requirements = GPComposerRequirements(
        primary=primary_operations,
        secondary=secondary_operations, max_arity=3,
        max_depth=7, pop_size=10, num_of_generations=10,
        crossover_prob=0.8, mutation_prob=0.8,
        max_lead_time=datetime.timedelta(minutes=5),
        allow_single_operations=False)

    mutation_types = [MutationTypesEnum.parameter_change,
                      MutationTypesEnum.simple,
                      MutationTypesEnum.reduce]
    optimiser_parameters = GPChainOptimiserParameters(
        mutation_types=mutation_types)

    metric_function = MetricsRepository().metric_by_id(
        RegressionMetricsEnum.MAE)
    builder = GPComposerBuilder(task=task). \
        with_optimiser_parameters(optimiser_parameters). \
        with_requirements(composer_requirements). \
        with_metrics(metric_function).with_initial_chain(init_chain)
    composer = builder.build()

    obtained_chain = composer.compose_chain(data=train_input,
                                            is_visualise=False)

    chain_tuner = ChainTuner(chain=obtained_chain,
                             task=task,
                             iterations=10)
    tuned_chain = chain_tuner.tune_chain(input_data=train_input,
                                         loss_function=mean_squared_error,
                                         loss_params={'squared': False})

    preds = fit_predict_for_chain(chain=tuned_chain,
                                  train_input=train_input,
                                  predict_input=predict_input)

    list_with_nodes = display_chain_info(obtained_chain)

    model_name = str(list_with_nodes)
    return preds, model_name


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
    # Comparison with AutoTs library - https://github.com/winedarksea/AutoTS #
    ##########################################################################

    # Paths to the files
    path_to_the_short_file = 'data/ts_short.csv'
    path_to_the_long_file = 'data/ts_long.csv'

    # Paths to the folders with report csv files
    path_to_save_short = 'results/fedot_new/short'
    path_to_save_long = 'results/fedot_new/long'

    # Lists with forecasts lengths
    l_forecasts_short = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    l_forecasts_long = np.arange(280, 1010, 10)

    # Launch for short time series
    # run_experiment(path_to_the_short_file, path_to_save_short, l_forecasts_short,
    #                vis=False)

    # Launch for short time series
    run_experiment(path_to_the_long_file, path_to_save_long, l_forecasts_long,
                   vis=False)

