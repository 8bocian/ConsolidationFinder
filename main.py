import time

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


def get_extreme_row(group):
    if group['Half'].iloc[0]:
        return group.loc[group['Peak'].idxmax()]
    else:
        return group.loc[group['Peak'].idxmin()]


def total_deviation(values, allowed_percentage):
    mean = np.mean(values)

    deviations = np.abs(values - mean)

    sum_of_deviations = np.sum(deviations)

    sum_of_data_values = np.sum(values)

    allowed_max_deviation_sum = (allowed_percentage / 100) * sum_of_data_values

    return sum_of_deviations <= allowed_max_deviation_sum


def max_deviation(values):
    mean = np.mean(values)
    deviations = np.abs(values - mean)
    max_deviation_percent = (np.max(deviations) / mean) * 100
    return max_deviation_percent, mean


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
mpf.plot(df.set_index('Date'), type='candle', style='charles', title='BTC Candlestick Chart', ylabel='Price', datetime_format='%H:%M:%S')

max_window_width = 2 * 60
min_window_width = 119 #10
df.reset_index(inplace=True)
for idx in range(len(df)):
    for i in range(min_window_width, max_window_width):
        t = time.time()

        candles_range = df.loc[idx:idx+i]
        candles_range = candles_range[(candles_range['High'] - candles_range['Low']) > 1]
        candles_range.reset_index(inplace=True)
        peaks = []

        for idx_, candle in candles_range[1:].iterrows():
            previous_candle = candles_range.loc[idx_ - 1]
            is_reversal, peak_price = check_reversal(previous_candle=previous_candle, current_candle=candle)
            if is_reversal:
                peaks.append((candle['Date'], peak_price))
        peaks = pd.DataFrame(peaks, columns=['Date', 'Peak'])
        peaks['Order'] = [i for i in range(len(peaks))]
        # peaks.sort_values(by='Peak', ascending=False, inplace=True)

        top = max(peaks['Peak'])
        bottom = min(peaks['Peak'])
        midpoint = (top + bottom) / 2

        middle_threshold = 0.4

        diff = top - midpoint

        top_border = midpoint + (diff * middle_threshold)
        bottom_border = midpoint - (diff * middle_threshold)

        peaks = peaks[(peaks['Peak'] >= top_border) | (peaks['Peak'] <= bottom_border)]
        peaks['Half'] = peaks['Peak'] >= midpoint

        peaks['Group'] = (peaks['Half'] != peaks['Half'].shift()).cumsum()

        grouped_peaks = peaks.groupby(['Group']).apply(get_extreme_row).reset_index(drop=True)

        grouped_peaks = grouped_peaks.drop(columns=["Group"])

        tops = grouped_peaks[grouped_peaks['Half'] == True]
        bottoms = grouped_peaks[grouped_peaks['Half'] == False]

        mean_top = np.mean(tops['Peak'])
        mean_bottom = np.mean(bottoms['Peak'])

        a = 0.02

        allowed_cumulative_percentage_top = a * (1 - np.power(np.e, -len(tops))) #0.007 * len(tops)
        allowed_cumulative_percentage_bottom = a * (1 - np.power(np.e, -len(bottoms))) #0.007 * len(bottoms)

        top_border_max = mean_top * (1 - (allowed_cumulative_percentage_top / 100))
        top_border_min = mean_top * (1 + (allowed_cumulative_percentage_top / 100))

        bottom_border_max = mean_bottom * (1 - (allowed_cumulative_percentage_bottom / 100))
        bottom_border_min = mean_bottom * (1 + (allowed_cumulative_percentage_bottom / 100))

        if len(tops) >= 2 and len(bottoms) > 2:

            print("CONSOLIDATION FOUND")
            print(candles_range.iloc[1]['Date'], i)
        print(time.time() - t)

        plt.fill_between(grouped_peaks['Date'], mean_top * (1 - (allowed_cumulative_percentage_top / 100)), mean_top * (1 + (allowed_cumulative_percentage_top / 100)), color='blue', alpha=0.3,
                         label=f'Channel around {mean_top}')
        plt.fill_between(grouped_peaks['Date'], mean_bottom * (1 - (allowed_cumulative_percentage_bottom / 100)), mean_bottom * (1 + (allowed_cumulative_percentage_bottom / 100)), color='red', alpha=0.3,
                         label=f'Channel around {mean_top}')

        plt.axhline(mean_top, color='blue')
        plt.axhline(mean_bottom, color='red')

        plt.plot(grouped_peaks['Date'], grouped_peaks['Peak'])
        plt.show()

# DODAC POTWIERDZENIE KONSOLI tzn brac srednia z topow i bottomow i patrzec na maksymalne odchylenia I PRZYSPIESZYC