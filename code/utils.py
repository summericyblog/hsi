import yfinance as yf
import pandas as pd
import numpy as np
import datetime


def date_delay(date, n):
    day = datetime.datetime.strptime(date, "%Y-%m-%d")
    ret = day + datetime.timedelta(days=n)
    ret = ret.strftime("%Y-%m-%d")
    return ret


def get_daily_data():
    fp = 'data/hsi.csv'
    # check if historical data exists
    try:
        data = pd.read_csv(fp, encoding='utf8')
    except FileNotFoundError:
        data = pd.DataFrame()

    shape = data.shape
    if shape[0] != 0:
        data.sort_values('Date', ascending=True, inplace=True)
        data.drop(data.tail(1).index, inplace=True) 
        last_day = data.iloc[-1]['Date']
    else:
        last_day = "1999-12-31"

    # set start and end date
    today = datetime.datetime.today()
    start_date = date_delay(last_day, 1)
    end_date = today.strftime("%Y-%m-%d")
    end_date = date_delay(end_date, 1)

    print("Collect data from {0} to {1}".format(start_date, end_date))

    hsi = yf.Ticker("^HSI")
    print("Collecting......", end='')
    new_df = hsi.history(start=start_date, end=end_date, interval="1d")
    print("Finished.")
    new_df = new_df[["Open", "High", "Low", "Close", "Volume"]]
    new_df.reset_index(drop=False, inplace=True)
    new_df['Date'] = new_df['Date'].dt.strftime("%Y-%m-%d")
    data = pd.concat([data, new_df], axis=0)
    print("Start writing.")
    data.to_csv(fp, index=False, encoding='utf8')
    return 0


def ts_mean(d: pd.Series, period, nan=False):
    if nan:
        ret = d.rolling(period, min_periods=1).apply(lambda x: np.nanmean(x))
    else:
        ret = d.rolling(period).mean()
    return ret


def ts_std(d: pd.Series, period, nan=False):
    if nan:
        ret = d.rolling(period, min_periods=1).apply(lambda x: np.nanstd(x))
    else:
        ret = d.rolling(period).std()
    return ret


def append_ma():
    intervals = [5, 10, 20, 60, 180]
    fp = 'data/hsi.csv'
    out = 'data/hsi_ma.csv'
    data = pd.read_csv(fp, encoding='utf8')
    print("Reading completed.")
    data['Volume'].replace(0, np.nan, inplace=True)
    volume_std = ts_std(data['Volume'], 30, True)
    volume_mean = ts_mean(data['Volume'], 30, True)
    data['vol'] = (data['Volume'] - volume_mean) / volume_std
    data['vol'].fillna(0, inplace=True)
    yesterday_close = data['Close'].shift(1)
    data['open_ret'] = (data['Open'] - yesterday_close) / yesterday_close
    data['day_ret'] = (data['Close'] - data['Open']) / data['Open']
    for i in intervals:
        open_name = 'ma_close_' + str(i)
        data[open_name] = ts_mean(data['Close'], i)
        close_name = 'ma_open_' + str(i)
        data[close_name] = ts_mean(data['Open'], i)

        open_std = ts_std(data['Open'], i)
        close_std = ts_std(data['Close'], i)

        close_dev = 'ma_close_z_' + str(i)
        data[close_dev] = (data['Close'] - data[close_name]) / close_std
        open_dev = 'ma_open_z_' + str(i)
        data[open_dev] = (data['Open'] - data[open_name]) / open_std

        daystrong = (data['High'] - data['Open']) / (data['High'] - data['Low'])
        strong_name = 'strong_' + str(i)
        data[strong_name] = ts_mean(daystrong, i)
    print("Start writing.")
    data.to_csv(out, index=False, encoding='utf8')
    return 0


if __name__ == '__main__':
    # get_daily_data()
    # fp = 'data/hsi.csv'
    # data = pd.read_csv(fp, encoding='utf8')
    # print(ts_mean(data['Open'], 3))
    append_ma()
