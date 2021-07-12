import gzip
import shutil
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

EXCHANGE_LABELS = ['Australia', 'British', 'Canada', 'Switzerland',
                   'China', 'Japan', 'New Zealand', 'Singapore']


def extract_tar(file_in, file_out):
    with gzip.open(file_in, 'rb') as f_in:
        with open(file_out, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def top_correlated_series(multi_time_series, currency, num_variables):
    '''
    Returns the top _num_variables_ series for a given _currency_ (including itself)
    '''

    corr_values_of_currency = multi_time_series.corr().abs()[currency].sort_values(kind="quicksort",
                                                                                   ascending=False)
    to_pick = corr_values_of_currency.index[:num_variables]
    return multi_time_series[to_pick]


def visualize_rates(data_frame):
    plot = sns.lineplot(data=data_frame, dashes=False, )
    plot.set_xlabel('Days')
    plot.set_ylabel('Rate')
    plt.show()


def low_scale_only(source_df):
    resulted = source_df[['New Zealand', 'Singapore']]

    return resulted


if __name__ == '__main__':
    exchange = pd.read_csv('../exchange_rate/exchange_rate.txt', header=None)
    exchange.columns = EXCHANGE_LABELS
    visualize_rates(exchange)

    top_5_correlated = top_correlated_series(exchange, currency='Canada', num_variables=5)
    # low_scale = low_scale_only(exchange)
    visualize_rates(top_5_correlated)
