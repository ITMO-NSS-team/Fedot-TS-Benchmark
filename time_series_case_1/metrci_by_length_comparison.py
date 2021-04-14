from analysis.metric_tools import *

if __name__ == '__main__':
    for path in ['results/fedot_new', 'results/prophet', 'results/autots']:
        print(f'\nProcessing path {path}')
        make_comparison_for_different_horizons(mode='short',
                                               path=path,
                                               forecast_thr={'patch_min': [10, 20, 30, 40, 50],
                                                             'patch_max': [60, 70, 80, 90, 100]})