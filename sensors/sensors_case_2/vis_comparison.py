from analysis.visualisation_tools import compare_forecasts
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    compare_forecasts(mode='smart', forecast_len=50)
