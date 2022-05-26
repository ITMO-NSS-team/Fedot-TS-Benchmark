import pandas as pd
import os
import scikit_posthocs as ph
import scipy as sp

import numpy as np


def perform_comparison(path: str):
    """ Make comparison by predictions """

    fedot_library_path = os.path.join(path, 'fedot')
    report_files = os.listdir(fedot_library_path)

    calc_dict = {'autots': 0, 'prophet': 0}
    all_cases = 0
    forecast_files = list(filter(lambda x: 'report' not in x, report_files))
    for forecast_file in forecast_files:
        real_df = pd.read_csv('tep_data.csv')
        forecast_df = pd.read_csv(os.path.join(path, 'fedot', forecast_file), parse_dates=['datetime'])
        autots_df = pd.read_csv(os.path.join(path, 'autots', forecast_file),
                                parse_dates=['datetime'])
        prophet_df = pd.read_csv(os.path.join(path, 'prophet', forecast_file),
                                 parse_dates=['datetime'])
        ts_labels = list(forecast_df['series_id'].unique())
        forecast_horizon = int(forecast_file.split('.csv')[0])
        for ts in ts_labels:
            # Calculate fedot vs library
            fedot_ts = forecast_df[forecast_df['series_id'] == ts]
            real_ts = real_df.iloc[:, ts]
            autots_ts = autots_df[autots_df['series_id'] == ts]
            prophet_ts = prophet_df[prophet_df['series_id'] == ts]
            autots_r, prophet_r = is_ts_different(forecast_horizon, fedot_ts, autots_ts, prophet_ts, real_ts)
            calc_dict['autots'] += autots_r
            calc_dict['prophet'] += prophet_r
            all_cases += 1

    autots_ratio = round(calc_dict.get('autots') / all_cases, 2)
    print(f'For autots: {autots_ratio:.2f}')

    prophet_ratio = round(calc_dict.get('prophet') / all_cases, 2)
    print(f'For prophet: {prophet_ratio:.2f}')


def is_ts_different(forecast_horizon: int, fedot_ts: pd.DataFrame, autots_ts: pd.DataFrame, prophet_ts: pd.DataFrame,
                    real_ts: pd.DataFrame):
    real_part = real_ts.tail(forecast_horizon)
    forecasted_part = fedot_ts.tail(forecast_horizon)

    autots_part = autots_ts.tail(forecast_horizon)
    prophet_part = prophet_ts.tail(forecast_horizon)

    fedot_metric = np.abs(np.array(forecasted_part['Predicted']) - real_part).values
    autots_metric = np.abs(np.array(autots_part['Predicted']) - real_part).values
    prophet_metric = np.abs(np.array(prophet_part['Predicted']) - real_part).values
    fedot_group = []
    autots_group = []
    prophet_group = []

    for i in range(0, fedot_metric.shape[0], 10):
        fedot_group.append(np.mean(fedot_metric[i:i+10:]))

    for i in range(0, autots_metric.shape[0], 10):
        autots_group.append(np.mean(autots_metric[i:i+10:]))

    for i in range(0, prophet_metric.shape[0], 10):
        prophet_group.append(np.mean(prophet_metric[i:i+10:]))

    fedot_group = np.array(fedot_group)
    autots_group = np.array(autots_group)
    prophet_group = np.array(prophet_group)
    friedman_results = sp.stats.friedmanchisquare(fedot_group, autots_group, prophet_group)
    p_value = friedman_results.pvalue
    if p_value >= 0.005:
        return (0, 0)

    result = ph.posthoc_nemenyi_friedman(pd.DataFrame({'fedot': fedot_group,
                                                       'autots': autots_group,
                                                       'prophet': prophet_group}))
    statistic_vs_autots = result.loc['fedot', 'autots']
    statistic_vs_prophet = result.loc['fedot', 'prophet']

    return statistic_vs_autots < 0.05, statistic_vs_prophet < 0.05


if __name__ == '__main__':
    print('TEP')
    perform_comparison('results')
