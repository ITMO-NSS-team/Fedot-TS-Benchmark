import gzip
import shutil


def extract_tar(file_in, file_out):
    with gzip.open(file_in, 'rb') as f_in:
        with open(file_out, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


if __name__ == '__main__':
    extract_tar('../exchange_rate/exchange_rate.txt.gz',
                '../exchange_rate/exchange_rate.txt')
