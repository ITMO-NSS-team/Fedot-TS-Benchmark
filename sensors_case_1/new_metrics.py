from time_series_case_1.analysis.metric_tools import calculate_new_metric, smape, mean_absolute_percentage_error

if __name__ == '__main__':
    for path in ['results/prophet', 'results/autots', 'results/fedot']:
        print(f'\n{path}')
        calculate_new_metric(metric_func=smape,
                             path=path,
                             mode='tep',
                             forecast_thr={'patch_min': [10],
                                           'patch_max': [10]})
