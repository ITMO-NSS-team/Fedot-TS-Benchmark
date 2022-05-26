import pandas as pd
import os
import scikit_posthocs as ph
import scipy as sp
import numpy as np
from sklearn.metrics import mean_absolute_error


def perform_comparison(path: str):
    """ Make comparison by predictions """

    fedot_library_path = os.path.join(path, 'fedot')
    report_files = os.listdir(fedot_library_path)

    forecast_files = list(filter(lambda x: 'report' not in x, report_files))
    all_actual = []
    all_fedot = []
    all_autots = []
    all_prophet = []
    for forecast_file in forecast_files:
        fedot_df = pd.read_csv(os.path.join(path, 'fedot', forecast_file), parse_dates=['datetime'])
        ts_labels = list(fedot_df['series_id'].unique())
        forecast_horizon = int(forecast_file.split('.csv')[0])

        # Compare with other libraries
        autots_df = pd.read_csv(os.path.join(path, 'autots', forecast_file), parse_dates=['datetime'])
        prophet_df = pd.read_csv(os.path.join(path, 'prophet', forecast_file), parse_dates=['datetime'])

        # Calculate fedot vs library
        for ts in ts_labels:
            fedot_ts = fedot_df[fedot_df['series_id'] == ts]
            autots_ts = autots_df[autots_df['series_id'] == ts]
            prophet_ts = prophet_df[prophet_df['series_id'] == ts]

            actual, fedot_forecast, autots_forecast, prophet_forecast = get_true_predicted_values(forecast_horizon,
                                                                                                  fedot_ts,
                                                                                                  autots_ts,
                                                                                                  prophet_ts)
            all_actual.extend(actual)
            all_fedot.extend(fedot_forecast)
            all_autots.extend(autots_forecast)
            all_prophet.extend(prophet_forecast)

    all_actual = np.array(all_actual)
    all_fedot = np.array(all_fedot)
    all_autots = np.array(all_autots)
    all_prophet = np.array(all_prophet)

    intervals = 180
    actual_chunks = np.array_split(all_actual, intervals)
    fedot_chunks = np.array_split(all_fedot, intervals)
    autots_chunks = np.array_split(all_autots, intervals)
    prophet_chunks = np.array_split(all_prophet, intervals)

    fedot_metrics = []
    autots_metrics = []
    prophet_metrics = []
    for i in range(len(actual_chunks)):
        actual_chunk = actual_chunks[i]
        fedot_chunk = fedot_chunks[i]
        autots_chunk = autots_chunks[i]
        prophet_chunk = prophet_chunks[i]

        fedot_metrics.append(mean_absolute_error(np.array(actual_chunk, dtype=float),
                                                 np.array(fedot_chunk, dtype=float)))
        autots_metrics.append(mean_absolute_error(np.array(actual_chunk, dtype=float),
                                                  np.array(autots_chunk, dtype=float)))
        prophet_metrics.append(mean_absolute_error(np.array(actual_chunk, dtype=float),
                                                   np.array(prophet_chunk, dtype=float)))

    fedot_metrics = np.array(fedot_metrics)
    autots_metrics = np.array(autots_metrics)
    prophet_metrics = np.array(prophet_metrics)
    friedman_results = sp.stats.friedmanchisquare(fedot_metrics, autots_metrics, prophet_metrics)
    p_value = friedman_results.pvalue
    result = ph.posthoc_nemenyi_friedman(pd.DataFrame({'fedot': fedot_metrics,
                                                       'autots': autots_metrics,
                                                       'prophet': prophet_metrics}))
    statistic_vs_autots = result.loc['fedot', 'autots']
    statistic_vs_prophet = result.loc['fedot', 'prophet']

    print(f'FEDOT vs AutoTS {statistic_vs_autots < p_value}, {statistic_vs_autots}, {p_value}')
    print(f'FEDOT vs prophet {statistic_vs_prophet < p_value}, {statistic_vs_prophet}, {p_value}')


def get_true_predicted_values(forecast_horizon: int, fedot_ts: pd.DataFrame,
                              autots_ts: pd.DataFrame, prophet_ts: pd.DataFrame):
    fedot_part = fedot_ts.tail(forecast_horizon)
    autots_part = autots_ts.tail(forecast_horizon)
    prophet_part = prophet_ts.tail(forecast_horizon)

    fedot_forecast = list(fedot_part['Predicted'])
    actual = list(fedot_part['value'])
    autots_forecast = list(autots_part['Predicted'])
    prophet_forecast = list(prophet_part['Predicted'])
    return actual, fedot_forecast, autots_forecast, prophet_forecast


if __name__ == '__main__':
    print('TEP')
    perform_comparison('results')
