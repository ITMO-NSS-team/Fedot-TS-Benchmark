from time_series_case_1.analysis.metric_tools import print_metrics_by_folder, make_comparison_for_different_horizons
from time_series_case_1.analysis.visualisation_tools import *

if __name__ == '__main__':
    path = 'results/autots_0'
    # Plot forecast for 100 forecast horizon
    plot_forecast(path=path, ts_label='use [kW]', forecast_len=10)

    # print('------ Info about time series processing ------')
    print_metrics_by_folder(path, mode='smart')
    plot_mape_vs_len(path, mode='smart')

    make_comparison_for_different_horizons(mode='smart',
                                           path=path,
                                           forecast_thr={'patch_min': [10, 20, 30, 40, 50],
                                                         'patch_max': [60, 70, 80, 90, 100]})
