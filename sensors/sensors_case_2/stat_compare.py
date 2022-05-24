import pandas as pd
import os
import scikit_posthocs as sp
import numpy as np


def perform_comparison(path: str):
    """ Make comparison by predictions """

    fedot_library_path = os.path.join(path, 'fedot')
    report_files = os.listdir(fedot_library_path)

    calc_dict = {'autots': 0, 'prophet': 0}
    all_cases = 0
    forecast_files = list(filter(lambda x: 'report' not in x, report_files))
    for forecast_file in forecast_files:
        forecast_df = pd.read_csv(os.path.join(path, 'fedot', forecast_file), parse_dates=['datetime'])
        ts_labels = list(forecast_df['series_id'].unique())
        forecast_horizon = int(forecast_file.split('.csv')[0])

        for library in ['autots', 'prophet']:
            # Compare with other libraries
            compare_df = pd.read_csv(os.path.join(path, library, forecast_file),
                                     parse_dates=['datetime'])

            # Calculate fedot vs library
            for ts in ts_labels:
                fedot_ts = forecast_df[forecast_df['series_id'] == ts]
                compare_ts = compare_df[compare_df['series_id'] == ts]

                if is_ts_different(forecast_horizon, fedot_ts, compare_ts):
                    # There is different forecasts
                    current_number = calc_dict.get(library)
                    current_number += 1
                    calc_dict.update({library: current_number})

                if library == 'autots':
                    # Calculate only once
                    all_cases += 1

    autots_ratio = round(calc_dict.get('autots') / all_cases, 2)
    print(f'For autots: {autots_ratio:.2f}')

    prophet_ratio = round(calc_dict.get('prophet') / all_cases, 2)
    print(f'For prophet: {prophet_ratio:.2f}')


def is_ts_different(forecast_horizon: int, fedot_ts: pd.DataFrame, compare_ts: pd.DataFrame):
    forecasted_part = fedot_ts.tail(forecast_horizon)
    compare_part = compare_ts.tail(forecast_horizon)

    fedot_forecast = np.array(forecasted_part['Predicted'])
    compare_forecast = np.array(compare_part['Predicted'])
    result = sp.posthoc_nemenyi_friedman(pd.DataFrame({'FEDOT': fedot_forecast,
                                                       'Competitor': compare_forecast}))
    statistic = result['FEDOT'][-1]
    return statistic < 0.05


if __name__ == '__main__':
    print('SMART')
    perform_comparison('results')
