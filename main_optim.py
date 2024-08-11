import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import mplfinance as mpf


def check_reversal(candles_range):
    is_previous_green = candles_range['Close'].shift(1) > candles_range['Open'].shift(1)
    is_current_green = candles_range['Close'] > candles_range['Open']
    is_reversal = is_previous_green != is_current_green

    peak_prices = np.where(is_previous_green,
                           candles_range[['High', 'High']].max(axis=1),
                           candles_range[['Low', 'Low']].min(axis=1))

    return is_reversal, peak_prices


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


df = pd.read_csv('BTCUSDT-1s-2024-08-07.csv')
df.columns = ["Date", "Open", "High", "Low", "Close", "Volume", "close_time", "quote_volume", "count",
              "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

df['Date'] = pd.to_datetime(df['Date'], unit='ms')
df = df[(df['Date'] > '2024-08-07 01:26:00') & (df['Date'] < '2024-08-07 23:59:41')]
# mpf.plot(df.set_index('Date'), type='candle', style='charles', title='BTC Candlestick Chart', ylabel='Price',
#          datetime_format='%H:%M:%S')

max_window_width = 2 * 60
min_window_width = 20#20

df.reset_index(inplace=True)

for idx in range(0, len(df), 1):
    start_date = df.loc[idx]['Date']
    print(start_date)
    t = time.time()
    for i in range(min_window_width, max_window_width, 2):

        candles_range = df.loc[idx:idx + i].copy()
        candles_range = candles_range[(candles_range['High'] - candles_range['Low']) > 1]
        candles_range.reset_index(inplace=True)

        is_reversal, peak_prices = check_reversal(candles_range[1:])

        peaks = pd.DataFrame({
            'Date': candles_range['Date'][1:][is_reversal],
            'Peak': peak_prices[is_reversal]
        })
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

        a = 0.005
        allowed_cumulative_percentage_top = a * (1 - np.exp(-len(tops)))
        allowed_cumulative_percentage_bottom = a * (1 - np.exp(-len(bottoms)))

        top_border_min = mean_top * (1 - (allowed_cumulative_percentage_top / 100))
        top_border_max = mean_top * (1 + (allowed_cumulative_percentage_top / 100))

        bottom_border_min = mean_bottom * (1 - (allowed_cumulative_percentage_bottom / 100))
        bottom_border_max = mean_bottom * (1 + (allowed_cumulative_percentage_bottom / 100))

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
        if found:
            print("CONSOLIDATION FOUND")
            print(candles_range.iloc[1]['Date'], i)

            print(values_in_range(tops['Peak'],
                                  top_border_min,
                                  top_border_max),
                  values_in_range(bottoms['Peak'],
                                  bottom_border_min,
                                  bottom_border_max))

            plt.fill_between(grouped_peaks['Date'], top_border_max,
                             top_border_min, color='blue', alpha=0.3)
            plt.fill_between(grouped_peaks['Date'], bottom_border_max,
                             bottom_border_min, color='red', alpha=0.3)

            plt.axhline(mean_top, color='blue')
            plt.axhline(mean_bottom, color='red')

            plt.plot(grouped_peaks['Date'], grouped_peaks['Peak'])
            # plt.show()
            plt.savefig(f'consolidations/{str(start_date).replace(":", "-").replace(" ", "_")}.png')
            plt.cla()
            plt.clf()
            with open("results.txt", 'w+') as f:
                f.write(f"{str(start_date)}\n")
            # idx += 10
            #
            # mpf.plot(candles_range.set_index('Date'), type='candle', style='charles', title='BTC Candlestick Chart',
            #          ylabel='Price',
            #          datetime_format='%H:%M:%S')
            # candles_range.set_index('Date', inplace=True)

            # Plot the candlestick chart
            # fig, ax = mpf.plot(candles_range, type='candle', style='charles', title='BTC Candlestick Chart',
            #                    ylabel='Price', datetime_format='%H:%M:%S', returnfig=True)
            # ax[0].fill_between(candles_range.index, top_border_max, top_border_min, color='blue', alpha=0.3)
            # ax[0].fill_between(candles_range.index, bottom_border_max, bottom_border_min, color='red', alpha=0.3)

            # Draw the horizontal lines for mean values
            # ax[0].axhline(mean_top, color='blue')
            # ax[0].axhline(mean_bottom, color='red')
            # ax[0].scatter(x=tops['Date'], y=tops['Peak'], color='blue')
            # ax[0].scatter(x=bottoms['Date'], y=bottoms['Peak'], color='red')
            # Display the plot
            # plt.show()
    print(time.time() - t)
