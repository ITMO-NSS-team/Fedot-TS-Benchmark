import numpy as np
import pandas as pd

from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.tuning.unified import PipelineTuner
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.api.main import Fedot
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

from exchange_case.exchange_rate import EXCHANGE_LABELS, top_correlated_series


def basic_pipeline(lagged_value=10):
    lagged = PrimaryNode('lagged')
    lagged.custom_params = {'window_size': lagged_value}
    secondary = SecondaryNode('ridge', nodes_from=[lagged])

    pipeline = Pipeline(secondary)

    return pipeline


def plot_results(actual_time_series, predicted_values, len_train_data, y_name='Parameter'):
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
             [min(actual_time_series), max(actual_time_series)], c='black', linewidth=1)
    plt.ylabel(y_name, fontsize=15)
    plt.xlabel('Time index', fontsize=15)
    plt.legend(fontsize=15, loc='upper left')
    plt.grid()
    plt.show()


def forecast_by_fedot(forecast_length, train_input, predict_input):
    # Define parameters
    task_parameters = TsForecastingParams(forecast_length=forecast_length)

    # Init model for the time series forecasting
    model = Fedot(problem='ts_forecasting', task_params=task_parameters)

    # Run AutoML model design in the same way
    pipeline = model.fit(features=train_input)

    # Use model to obtain forecast
    forecast = model.predict(features=predict_input)

    return forecast


if __name__ == '__main__':
    exchange = pd.read_csv('../exchange_rate/exchange_rate.txt', header=None)
    exchange.columns = EXCHANGE_LABELS

    top_5_correlated = top_correlated_series(exchange, currency='Canada', num_variables=5)
    canada = top_5_correlated[['Canada']]

    forecast_length = 10

    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=forecast_length))

    # Get time series from dataframe
    rate_values = np.array(canada).ravel()[-100:]
    input_data = InputData(idx=np.arange(0, len(rate_values)),
                           features=rate_values,
                           target=rate_values,
                           task=task,
                           data_type=DataTypesEnum.ts)

    # Split data into train and test
    train_input, predict_input = train_test_data_setup(input_data)

    pipeline = basic_pipeline(lagged_value=10)
    pipeline.fit(train_input)

    chain_tuner = PipelineTuner(pipeline=pipeline,
                                task=task,
                                iterations=50)

    tuned_pipeline = chain_tuner.tune_pipeline(input_data=train_input,
                                               loss_function=mean_absolute_error,
                                               loss_params=None)

    output = pipeline.predict(predict_input)
    forecast = np.ravel(np.array(output.predict))

    plot_results(actual_time_series=rate_values,
                 predicted_values=forecast,
                 len_train_data=len(rate_values) - forecast_length)

    # Print MAE metric
    print(f'Mean absolute error: {mean_absolute_error(predict_input.target, forecast):.3f}')

    output = forecast_by_fedot(forecast_length, train_input, predict_input)
    # forecast = np.ravel(np.array(output.predict))

    plot_results(actual_time_series=rate_values,
                 predicted_values=output,
                 len_train_data=len(rate_values) - forecast_length)
