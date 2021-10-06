from analysis.metric_tools import print_metrics_by_folder
from analysis.visualisation_tools import plot_mape_vs_len, plot_forecast

if __name__ == '__main__':
    path_short = 'results/autots/short'
    path_long = 'results/autots/long'

    print('------ Info about short time series processing ------')
    print_metrics_by_folder(path_short, mode='short')
    plot_mape_vs_len(path_short, mode='short')

    print('------ Info about long time series processing ------')
    print_metrics_by_folder(path_long, mode='long')
    plot_mape_vs_len(path_long, mode='long')

    # Plot forecast for 100 forecast horizon
    plot_forecast(path=path_long, ts_label='traffic_volume', forecast_len=30)
