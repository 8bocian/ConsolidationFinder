import time
from datetime import timedelta

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import mplfinance as mpf


# def check_reversal(candles_range):
#     # Calculate the necessary booleans directly
#     previous_close = candles_range['Close'].shift(1)
#     previous_open = candles_range['Open'].shift(1)
#
#     current_close = candles_range['Close']
#     current_open = candles_range['Open']
#
#     # Determine the color of the current and previous candles
#     is_previous_green = previous_close > previous_open
#     is_current_green = current_close > current_open
#
#     # Check for reversal (change in candle color)
#     is_reversal = is_previous_green != is_current_green
#
#     # Determine peak prices based on the direction of the previous candle
#     peak_prices = np.where(is_previous_green,
#                            candles_range['High'].shift(1),
#                            candles_range['Low'].shift(1))
#     return is_reversal, peak_prices
def check_reversal(candles_range):
    open_prices = candles_range['Open'].values
    close_prices = candles_range['Close'].values
    high_prices = candles_range['High'].values
    low_prices = candles_range['Low'].values

    # Calculate previous and current candle colors
    is_previous_green = close_prices[:-1] > open_prices[:-1]
    is_current_green = close_prices[1:] > open_prices[1:]

    # Detect reversals
    is_reversal = is_previous_green != is_current_green

    # Calculate peak prices
    peak_prices = np.where(is_previous_green,
                           high_prices[1:],
                           low_prices[1:])

    # Apply the reversal mask to get only the relevant dates and peak prices
    peaks = pd.DataFrame({
        'Date': candles_range['Date'].values[1:][is_reversal],
        'Peak': peak_prices[is_reversal]
    })

    return peaks


def get_extreme_row(group):
    return group.loc[group['Peak'].idxmax()] if group['Half'].iloc[0] else group.loc[group['Peak'].idxmin()]


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


def values_in_range(values, lower_border, upper_border):
    for value in values:
        if not (lower_border <= value <= upper_border):
            return False
    return True


def detect_consolidation(candles_range):
    peaks = check_reversal(candles_range[1:])
    peaks['Order'] = range(len(peaks))

    top = peaks['Peak'].max()
    bottom = peaks['Peak'].min()
    midpoint = (top + bottom) / 2

    middle_threshold = 0.5
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

    diff = mean_top - mean_bottom
    g = diff/2000


    top_border_min = mean_top * (1 - (g / 100))
    top_border_max = mean_top * (1 + (g / 100))

    bottom_border_min = mean_bottom * (1 - (g / 100))
    bottom_border_max = mean_bottom * (1 + (g / 100))

    found = False
    if len(tops) + len(bottoms) > 4 \
            and \
            values_in_range(tops['Peak'],
                            top_border_min,
                            top_border_max) \
            and \
            values_in_range(bottoms['Peak'],
                            bottom_border_min,
                            bottom_border_max):
        found = True
    return grouped_peaks, found, \
           bottom_border_min, mean_bottom, bottom_border_max, \
           top_border_min, mean_top, top_border_max


def overlaps(list1, list2):
    set1, set2 = set(list1), set(list2)
    common_items = list(set1.intersection(set2))
    return bool(len(common_items))


df = pd.read_csv('BTCUSDT-1s-2024-08-07.csv')
df.columns = ["Date", "Open", "High", "Low", "Close", "Volume", "close_time", "quote_volume", "count",
              "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

df['Date'] = pd.to_datetime(df['Date'], unit='ms')
df = df[(df['Date'] > '2024-08-07 14:30:41') & (df['Date'] < '2024-08-07 23:59:41')]
# mpf.plot(df.set_index('Date'), type='candle', style='charles', title='BTC Candlestick Chart', ylabel='Price',
#          datetime_format='%H:%M:%S')

max_window_width = 2 * 60
min_window_width = 10#20

df.reset_index(inplace=True)
previous_consolidation_dates = []
for idx in range(0, len(df), 1):
    start_date = df.loc[idx]['Date'].to_pydatetime()
    print(start_date)
    t = time.time()
    found_in_window = False
    detection_time = 0
    range_extraction_time = 0
    for i in range(min_window_width, max_window_width, 1):
        t1 = time.time()
        candles_range = df.loc[idx:idx + i].copy()
        candles_range = candles_range[(candles_range['High'] - candles_range['Low']) > 1]
        candles_range.reset_index(inplace=True)
        range_extraction_time += time.time() - t1

        t1 = time.time()
        grouped_peaks, found, \
        bottom_border_min, mean_bottom, bottom_border_max, \
        top_border_min, mean_top, top_border_max = detect_consolidation(candles_range)
        detection_time += time.time() - t1
        # quit()
        if found:
            found_in_window = True
            if len(previous_consolidation_dates) < 2:
                previous_consolidation_dates.append((idx, i))
            else:
                previous_consolidation_dates[-1] = (idx, i)

    print(
        "DET", detection_time,
        "EXTR", range_extraction_time
    )
    if found_in_window == False and len(previous_consolidation_dates) != 0:
        consolidation_start = previous_consolidation_dates[0][0]
        consolidation_end = previous_consolidation_dates[-1][0] + previous_consolidation_dates[-1][-1]
        candles_range = df.loc[consolidation_start:consolidation_end].copy()
        candles_range = candles_range[(candles_range['High'] - candles_range['Low']) > 1]
        candles_range.reset_index(inplace=True)

        grouped_peaks, found, \
        bottom_border_min, mean_bottom, bottom_border_max, \
        top_border_min, mean_top, top_border_max = detect_consolidation(candles_range)
        if found:
            plt.fill_between(grouped_peaks['Date'], top_border_max,
                             top_border_min, color='blue', alpha=0.3)
            plt.fill_between(grouped_peaks['Date'], bottom_border_max,
                             bottom_border_min, color='red', alpha=0.3)

            plt.axhline(mean_top, color='blue')
            plt.axhline(mean_bottom, color='red')

            plt.plot(grouped_peaks['Date'], grouped_peaks['Peak'])
            plt.savefig(f'consolidations_new/{str(candles_range["Date"].values[0]).replace(":", "-").replace(" ", "_")}-{str(candles_range["Date"].values[-1]).replace(":", "-").replace(" ", "_")}.png')
            plt.show()

            plt.cla()
            plt.clf()
            with open("results.txt", 'w+') as f:
                f.write(f"{str(start_date)}\n")
        previous_consolidation_dates = []
    print(time.time() - t)
