from time_series_case_1.analysis.metric_tools import print_metrics_by_folder
from time_series_case_1.analysis.visualisation_tools import *

if __name__ == '__main__':
    path_short = 'results/fedot_new/short'
    path_long = 'results/fedot_new/long'

    print('------ Info about short time series processing ------')
    print_metrics_by_folder(path_short, mode='short')
    plot_mape_vs_len(path_short, mode='short')

    print('------ Info about long time series processing ------')
    print_metrics_by_folder(path_long, mode='long')
    plot_mape_vs_len(path_long, mode='long')

    # Plot forecast for 100 forecast horizon
    plot_forecast(path=path_long, ts_label='traffic_volume', forecast_len=10)
