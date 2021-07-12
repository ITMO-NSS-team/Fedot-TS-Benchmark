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


def visualize_rates(file_path):
    exchange = pd.read_csv(file_path, header=None)
    exchange.columns = EXCHANGE_LABELS
    plot = sns.lineplot(data=exchange, dashes=False, )
    plot.set_xlabel('Days')
    plot.set_ylabel('Rate')
    plt.show()


if __name__ == '__main__':
    visualize_rates(file_path='../exchange_rate/exchange_rate.txt')
