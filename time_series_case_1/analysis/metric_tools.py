import os
import numpy as np
import pandas as pd


def mean_absolute_percentage_error(y_true, y_pred):
    """ Calculate MAPE metric """
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    zero_indexes = np.argwhere(y_true == 0.0)
    for index in zero_indexes:
        y_true[index] = 0.01
    value = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return value


def _create_report_dataframe(folder, time_series='all'):
    """ The function creates a dataframe with metric value, time and forecast
    length columns from report files
    :param folder: path, where are located all necessary csv files
    :param time_series: for which time series we need to prepare dataframe
    if 'all' - mean value will be taken
    """

    # Files in the filder
    files = os.listdir(folder)
    files.sort()

    # Find all repost files in the folder
    reports = []
    for file in files:
        if file.endswith("report_.csv"):
            reports.append(file)

    # Read all files
    for index, report_file in enumerate(reports):
        report_df = pd.read_csv(os.path.join(folder, report_file))

        if time_series != 'all':
            # Take data only for one time series
            time_series_df = report_df[report_df['Time series label'] == time_series]
        else:
            # Take mean value for all launches
            time_series_df = report_df.groupby(['Models']).agg({'MAE': 'mean',
                                                                'MAPE': 'mean',
                                                                'Time': 'mean'})
            time_series_df = time_series_df.reset_index()

        splitted = report_file.split('_')
        forecast_len = int(splitted[0])
        result_df = time_series_df[['MAPE', 'Time']]
        len_df = len(result_df)
        result_df['Forecast len'] = [int(forecast_len)] * len_df

        if index == 0:
            response_df = result_df
        else:
            frames = [response_df, result_df]
            response_df = pd.concat(frames)

    response_df = response_df.sort_values(by=['Forecast len'])
    return response_df


def get_ts_short_names():
    ts_names = ['GS10', 'EXCHUS', 'EXCAUS',
                'Weekly U.S. Refiner and Blender Adjusted Net Production of Finished Motor Gasoline  (Thousand Barrels per Day)',
                'Weekly Minnesota Midgrade Conventional Retail Gasoline Prices  (Dollars per Gallon)',
                'Weekly U.S. Percent Utilization of Refinery Operable Capacity (Percent)',
                'Weekly U.S. Exports of Crude Oil and Petroleum Products  (Thousand Barrels per Day)',
                'Weekly U.S. Field Production of Crude Oil  (Thousand Barrels per Day)',
                'Weekly U.S. Ending Stocks of Crude Oil and Petroleum Products  (Thousand Barrels)',
                'Weekly U.S. Product Supplied of Finished Motor Gasoline  (Thousand Barrels per Day)']
    return ts_names


def get_ts_long_names():
    ts_names = ['temp', 'traffic_volume']
    return ts_names


def print_metrics_by_folder(path: str, mode: str):
    """ Display MAPE metric and time from report files """
    if mode == 'short':
        ts_names = get_ts_short_names()
    elif mode == 'long':
        ts_names = get_ts_long_names()
    else:
        raise NotImplementedError(f'Mode "{mode}" is not available')

    # Get metrics per time series for all short time series
    for ts_name in ts_names:
        dataframe = _create_report_dataframe(folder=path,
                                             time_series=ts_name)
        print(f'Time series name {ts_name}')
        print(f'Mean MAPE value - {dataframe["MAPE"].mean():.2f}')
        print(f'Mean time - {dataframe["Time"].mean():.2f}\n')

    # Get metric for all short time series (mean value)
    dataframe = _create_report_dataframe(folder=path,
                                         time_series='all')
    print('MEAN METRICS AND TIME FOR ALL TIME SERIES')
    print(f'Mean MAPE value - {dataframe["MAPE"].mean():.2f}')
    print(f'Mean time - {dataframe["Time"].mean():.2f}\n')
