from analysis.metric_tools import calculate_new_metric, smape


if __name__ == '__main__':
    for path in ['results/prophet', 'results/autots', 'results/fedot']:
        print(f'\n{path}')
        calculate_new_metric(metric_func=smape,
                             path=path,
                             mode='smart',
                             forecast_thr={'patch_min': [10, 20, 30, 40, 50],
                                           'patch_max': [60, 70, 80, 90, 100]})
