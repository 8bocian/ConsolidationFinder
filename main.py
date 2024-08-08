import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import mplfinance as mpf
from scipy.spatial import ConvexHull

def check_reversal(previous_candle, current_candle):
    is_previous_green = previous_candle['Close'] > previous_candle['Open']
    is_current_green = current_candle['Close'] > current_candle['Open']
    is_reversal = is_previous_green ^ is_current_green

    if is_reversal:
        if is_previous_green:
            peak_price = max(previous_candle['High'], current_candle['High'])
        else:
            peak_price = min(previous_candle['Low'], current_candle['Low'])
        return is_reversal, peak_price
    else:
        return False, None

df = pd.read_csv('BTCUSDT-1s-2024-08-07.csv')
df.columns = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore"
]
df = df[
        [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]
    ]

df['Date'] = pd.to_datetime(df['Date'], unit='ms')
# df = df[df['Date'] > '2024-08-07 17:11:00']
# df = df[df['Date'] < '2024-08-07 17:13:00']
df = df[df['Date'] > '2024-08-07 14:30:41']
df = df[df['Date'] < '2024-08-07 14:32:41']
# 18:46:01 18:47:21
# 18:33:21 18:34:50
# 14:30:10 14:33:01
# mpf.plot(df.set_index('Date'), type='candle', style='charles', title='BTC Candlestick Chart', ylabel='Price', datetime_format='%H:%M:%S')


max_window_width = 2 * 60
min_window_width = 119 #10
df.reset_index(inplace=True)
for idx in range(len(df)):
    for i in range(min_window_width, max_window_width):
        candles_range = df.loc[idx:idx+i]
        # mpf.plot(candles_range.set_index('Date'), type='candle', style='charles', title=f'BTC Candlestick Chart {idx, i}', ylabel='Price', datetime_format='%H:%M:%S')
        candles_range = candles_range[(candles_range['High'] - candles_range['Low']) > 1]
        candles_range.reset_index(inplace=True)
        peaks = []

        for idx_, candle in candles_range[1:].iterrows():
            previous_candle = candles_range.loc[idx_ - 1]
            is_reversal, peak_price = check_reversal(previous_candle=previous_candle, current_candle=candle)
            if is_reversal:
                peaks.append((candle['Date'], peak_price))
        peaks = pd.DataFrame(peaks, columns=['Date', 'Peak'])

        top = max(peaks['Peak'])
        bottom = min(peaks['Peak'])
        print(top, bottom)
        midpoint = (top + bottom) / 2

        low_peaks = peaks[peaks['Peak'] <= midpoint]
        high_peaks = peaks[peaks['Peak'] > midpoint]
        print(low_peaks)
        print(high_peaks)
        plt.scatter(x=low_peaks['Date'], y=low_peaks['Peak'], c='red')
        plt.scatter(x=high_peaks['Date'], y=high_peaks['Peak'], c='blue')

        plt.plot(peaks['Date'], peaks['Peak'])
        plt.show()

