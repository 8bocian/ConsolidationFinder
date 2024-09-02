import time
from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import mplfinance as mpf
import os
import warnings


def check_reversal(candles_range):
    peaks = []
    is_previous_green = None
    for idx, candle in candles_range.iterrows():
        is_current_green = candle['Close'] > candle['Open']
        if is_previous_green is None:
            peaks.append((candle['Date'], candle['Open'], candle['Order']))
        if idx == len(candles_range) - 1:
            peaks.append((candle['Date'], candle['Close'], candle['Order']))
        else:
            if is_previous_green != is_current_green:
                peaks.append((candle['Date'], candle['Open'], candle['Order']))
        is_previous_green = is_current_green
    peaks = pd.DataFrame(columns=['Date', 'Peak', 'Order'], data=peaks)

    return peaks


df = pd.read_csv('BTCUSDT-1m-2024-06.csv')
df.columns = ["Date", "Open", "High", "Low", "Close", "Volume", "close_time", "quote_volume", "count",
              "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

df['Date'] = pd.to_datetime(df['Date'], unit='ms')
# df = df[(df['Date'] > '2024-06-07 15:22:15') & (df['Date'] < '2024-07-07 15:23:44')]
candles_range = df.copy()
candles_range = candles_range[(candles_range['High'] - candles_range['Low']) > 1]
candles_range['Order'] = range(len(candles_range))
candles_range.reset_index(inplace=True)

peaks = check_reversal(candles_range)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

mpf.plot(candles_range[::-1].set_index('Date'), type='candle', style='charles', ylabel='Price',
         datetime_format='%H:%M:%S', ax=ax1, show_nontrading=True)

# ax2.fill_between(grouped_peaks['Order'], top_border_max,
#                  top_border_min, color='blue', alpha=0.3)
# ax2.fill_between(grouped_peaks['Order'], bottom_border_max,
#                  bottom_border_min, color='red', alpha=0.3)

# ax2.axhline(mean_top, color='blue')
# ax2.axhline(mean_bottom, color='red')

ax2.plot(peaks['Date'], peaks['Peak'])

fig.suptitle('BTC Candlestick Chart and Peak Analysis')

plt.tight_layout()


plt.show()
