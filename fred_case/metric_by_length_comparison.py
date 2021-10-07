from analysis.metric_tools import make_comparison_for_different_horizons

if __name__ == '__main__':
    for path in ['results/fedot_new', 'results/prophet', 'results/autots']:
        print(f'\nProcessing path {path}')

        print('\nLong time series:')
        patch_min = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        patch_max = [110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
        make_comparison_for_different_horizons(mode='long',
                                               path=path,
                                               forecast_thr={'patch_min': patch_min,
                                                             'patch_max': patch_max})

        print('\nShort time series:')
        patch_min = [10, 20, 30, 40, 50]
        patch_max = [50, 60, 70, 80, 90, 100]
        make_comparison_for_different_horizons(mode='short',
                                               path=path,
                                               forecast_thr={'patch_min': patch_min,
                                                             'patch_max': patch_max})
