from analysis.metric_tools import calculate_quantiles_metric, smape


if __name__ == '__main__':
    for path in ['results/autots', 'results/prophet', 'results/fedot']:
        print(f'\n{path}')
        calculate_quantiles_metric(metric_func=smape,
                                   path=path,
                                   mode='smart',
                                   forecast_thr={'patch_min': [10, 20, 30, 40, 60, 70, 80, 90, 100],
                                                 'patch_max': [50]})
