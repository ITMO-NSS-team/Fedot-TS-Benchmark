import os

from autots.datasets.fred import get_fred_data

if __name__ == '__main__':
    fred_api_key = os.environ['FRED_API_KEY']

    ts = get_fred_data(fredkey=fred_api_key)
    print(ts.head(50))
