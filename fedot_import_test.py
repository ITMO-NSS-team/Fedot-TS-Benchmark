import datetime

import numpy as np
import pandas as pd
from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.composer.gp_composer.gp_composer import GPComposerRequirements, GPComposerBuilder
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import MetricsRepository, \
    RegressionMetricsEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


def get_chain_info(chain):
    """ Function print info about chain and return operations in it and depth

    :param chain: chain to process
    :return obtained_operations: operations in the nodes
    :return depth: depth of the chain
    """

    print('\nObtained chain for current iteration')
    obtained_operations = []
    for node in chain.nodes:
        print(str(node))
        obtained_operations.append(str(node))
    depth = int(chain.depth)
    print(f'Chain depth {depth}\n')

    return obtained_operations, depth


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
    preds = predicted_values.predict

    return preds


def run_experiment(file_path, init_chain):
    # Read dataframe and prepare train and test data
    df = pd.read_csv(file_path)
    features = np.array(df[['level_station_1', 'mean_temp', 'month', 'precip']])
    target = np.array(df['level_station_2'])
    x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(
        features, target, test_size=0.2, shuffle=True, random_state=10)
    y_data_test = np.ravel(y_data_test)

    # Define regression task
    task = Task(TaskTypesEnum.regression)

    # Prepare data to train the model
    train_input = InputData(idx=np.arange(0, len(x_data_train)),
                            features=x_data_train,
                            target=y_data_train,
                            task=task,
                            data_type=DataTypesEnum.table)

    predict_input = InputData(idx=np.arange(0, len(x_data_test)),
                              features=x_data_test,
                              target=None,
                              task=task,
                              data_type=DataTypesEnum.table)

    available_operations_types = ['ridge', 'lasso', 'dtreg',
                                  'xgbreg', 'adareg', 'knnreg',
                                  'linear', 'svr', 'poly_features',
                                  'scaling', 'ransac_lin_reg', 'rfe_lin_reg',
                                  'pca', 'ransac_non_lin_reg',
                                  'rfe_non_lin_reg', 'normalization']

    composer_requirements = GPComposerRequirements(
        primary=['one_hot_encoding'],
        secondary=available_operations_types, max_arity=3,
        max_depth=8, pop_size=10, num_of_generations=5,
        crossover_prob=0.8, mutation_prob=0.8,
        max_lead_time=datetime.timedelta(minutes=5),
        allow_single_operations=True)

    metric_function = MetricsRepository().metric_by_id(
        RegressionMetricsEnum.MAE)
    builder = GPComposerBuilder(task=task).with_requirements(
        composer_requirements).with_metrics(metric_function).with_initial_chain(
        init_chain)
    composer = builder.build()

    obtained_chain = composer.compose_chain(data=train_input, is_visualise=False)

    # Display info about obtained chain
    obtained_models, depth = get_chain_info(chain=obtained_chain)

    preds = fit_predict_for_chain(chain=obtained_chain,
                                  train_input=train_input,
                                  predict_input=predict_input)

    mse_value = mean_squared_error(y_data_test, preds, squared=False)
    mae_value = mean_absolute_error(y_data_test, preds)
    print(f'RMSE - {mse_value:.2f}')
    print(f'MAE - {mae_value:.2f}\n')


if __name__ == '__main__':
    # Define chain to start composing with it
    node_encoder = PrimaryNode('one_hot_encoding')
    node_rans = SecondaryNode('ransac_lin_reg', nodes_from=[node_encoder])
    node_scaling = SecondaryNode('scaling', nodes_from=[node_rans])
    node_final = SecondaryNode('linear', nodes_from=[node_scaling])

    init_chain = Chain(node_final)

    # Available tuners for application: ChainTuner, NodesTuner
    run_experiment(file_path='data/river_levels/station_levels.csv',
                   init_chain=init_chain)
