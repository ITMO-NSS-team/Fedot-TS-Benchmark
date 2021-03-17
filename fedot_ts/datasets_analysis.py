import matplotlib.pyplot as plt
import pandas as pd


def plot_time_series(ts_frame):
    series_ids = ts_frame['series_id'].unique()

    for ts_id in series_ids:
        time_series = ts_frame[ts_frame['series_id'] == ts_id]
        print(f'Длина временного ряда {len(time_series)}')
        plt.plot(time_series['datetime'], time_series['value'])
        plt.title(ts_id)
        plt.grid()
        plt.show()


if __name__ == '__main__':
    short_df = pd.read_csv('../data/time_series_data/ts_short.csv')
    short_df['datetime'] = pd.to_datetime(short_df['datetime'])
    long_df = pd.read_csv('../data/time_series_data/ts_long.csv')
    long_df['datetime'] = pd.to_datetime(long_df['datetime'])

    print('Short time_series:')
    plot_time_series(short_df)
    print('Long time_series:')
    plot_time_series(long_df)
