import time
from datetime import timedelta

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import mplfinance as mpf
import os
import warnings

# Suppress a specific warning category
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Suppress warnings with a specific message
warnings.filterwarnings("ignore", message="some specific warning message")

warnings.filterwarnings("ignore")
directory = "consolidations_new"

# Check if the directory exists
if not os.path.exists(directory):
    # Create the directory
    os.makedirs(directory)

def check_reversal(candles_range):
    peaks = []
    is_previous_green = None
    for idx, candle in candles_range.iterrows():
        is_current_green = candle['Close'] > candle['Open']
        # if is_previous_green is None:
        #     peaks.append((candle['Date'], candle['Open']))
        if idx == len(candles_range) - 1:
            peaks.append((candle['Date'], candle['Close']))
        else:
            if is_previous_green != is_current_green:
                peaks.append((candle['Date'], candle['Open']))
        is_previous_green = is_current_green
    peaks = pd.DataFrame(columns=['Date', 'Peak'], data=peaks)

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
    peaks = check_reversal(candles_range)
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
    g = diff/6000


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


class Trade:
    def __init__(self, entry_price, stop_loss, take_profit, date, size=10000):
        self.size = size
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.type = "LONG" if stop_loss < entry_price else "SHORT"
        print(f"{self.type} TRADE_OPENED AT {entry_price} ON {date}")
        self.open_date = date
        self.percentage_loss = 1 - abs(entry_price - stop_loss) / entry_price
        self.percentage_profit = 1 + abs(entry_price - take_profit) / entry_price
        self.is_open = True
        self.is_in_game = False
        self.is_profit = None

    def check_trade(self, current_candle):
        # if self.type == "LONG" and self.open_date.minute == 32 and self.open_date.second == 15:
        #     print(current_candle['Low'], self.stop_loss, current_candle['High'])
        if self.is_in_game:
            if self.type == "SHORT":
                if current_candle['High'] >= self.stop_loss:
                    self.is_open = False
                    self.is_profit = False
                    self.is_in_game = False
                    self.trade_return = self.percentage_loss * self.size
                    print(f"{self.type} {self.open_date} TRADE CLOSED ON LOSS: {self.trade_return:.2f} AT {current_candle['Date']}")
                    return self.is_profit, self.trade_return
                elif current_candle['Low'] <= self.take_profit:
                    self.is_open = False
                    self.is_profit = True
                    self.is_in_game = False
                    self.trade_return = self.percentage_profit * self.size
                    print(f"{self.type} {self.open_date} TRADE CLOSED ON PROFIT: {self.trade_return:.2f} AT {current_candle['Date']}")
                    return self.is_profit, self.trade_return
            else:
                if current_candle['Low'] <= self.stop_loss:
                    self.is_open = False
                    self.is_profit = False
                    self.is_in_game = False
                    self.trade_return = self.percentage_loss * self.size
                    print(f"{self.type} {self.open_date} TRADE CLOSED ON LOSS: {self.trade_return:.2f} AT {current_candle['Date']}")
                    return self.is_profit, self.trade_return
                elif current_candle['High'] >= self.take_profit:
                    self.is_open = False
                    self.is_profit = True
                    self.is_in_game = False
                    self.trade_return = self.percentage_profit * self.size
                    print(f"{self.type} {self.open_date} TRADE CLOSED ON PROFIT: {self.trade_return:.2f} AT {current_candle['Date']}")
                    return self.is_profit, self.trade_return
        else:
            if self.type == "SHORT":
                if self.entry_price <= current_candle['High']:
                    self.is_in_game = True
                    print(f"{self.type} {self.open_date} {self.entry_price:.2f} TRADE IS IN GAME")
            else:
                if self.entry_price >= current_candle['Low']:
                    self.is_in_game = True
                    print(f"{self.type} {self.open_date} {self.entry_price:.2f} TRADE IS IN GAME")
        return None, None

    def cancel_trade(self):
        self.is_open = False
        self.is_in_game = False

class Trader:
    def __init__(self):
        self.trades = []
        self.trades_history = []
        self.total_money = 0
        self.trades_closed = 0

    def show_stats(self):
        avg_return = 0 if self.trades_closed == 0 else self.total_money / self.trades_closed
        print(f"AVG RETURN PER TRADE: {avg_return:.2f}, TOTAL: {(self.total_money - (10000 * self.trades_closed)):.2f}")

    def check_trade(self, current_candle):
        change = False
        for trade in self.trades:
            if trade.is_open:
                status, trade_return = trade.check_trade(current_candle)
                if status is not None:
                    self.total_money += trade_return
                    self.trades_closed += 1
                    change = True
        return change

    def cancel_unopened_trades(self):
        closed = 0
        for idx, trade in enumerate(self.trades):
            if trade.is_open and not trade.is_in_game:
                trade.cancel_trade()
                print(f"CLOSED TRADE: {trade.open_date} {trade.type} {trade.entry_price}")
                closed += 1


    # DECIDE WHAT IS THE ENTRY POINT
    def open_trade(self, entry_price, stop_loss, take_profit, date):
        # self.cancel_unopened_trades()
        new_trade = Trade(entry_price, stop_loss, take_profit, date)
        self.trades.append(new_trade)
        return new_trade

trader = Trader()

df = pd.read_csv('BTCUSDT-1s-2024-07.csv')
df.columns = ["Date", "Open", "High", "Low", "Close", "Volume", "close_time", "quote_volume", "count",
              "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

df['Date'] = pd.to_datetime(df['Date'], unit='ms')
df = df[(df['Date'] > '2024-07-04 00:00:01') & (df['Date'] < '2024-07-05 23:59:59')]
mpf.plot(df.set_index('Date'), type='candle', style='charles', title='BTC Candlestick Chart', ylabel='Price',
         datetime_format='%H:%M:%S')

max_window_width = 2 * 60
min_window_width = 10#10

df.reset_index(inplace=True)

previous_consolidation_peak = None
for idx in range(max_window_width - 1, len(df), 1):
    current_candle = df.loc[idx]
    start_date = current_candle['Date'].to_pydatetime()
    if start_date.second == 0:
        print(start_date)
    t = time.time()
    found_in_window = False

    show = False
    max_window_size = 0

    is_change = trader.check_trade(current_candle)
    if is_change:
        show = True
        trader.show_stats()

    for i in range(min_window_width, max_window_width, 1):
        candles_range = df.loc[idx - i:idx].copy()
        # candles_range = candles_range[::-1]
        candles_range = candles_range[(candles_range['High'] - candles_range['Low']) > 1]
        candles_range.reset_index(inplace=True)

        grouped_peaks, found, \
        bottom_border_min, mean_bottom, bottom_border_max, \
        top_border_min, mean_top, top_border_max = detect_consolidation(candles_range)

        if found:
            max_window_size = i
    if max_window_size != 0:
        candles_range = df.loc[idx - max_window_size:idx].copy()
        # candles_range = candles_range[::-1]
        candles_range = candles_range[(candles_range['High'] - candles_range['Low']) > 1]
        candles_range.reset_index(inplace=True)
        grouped_peaks, found, \
        bottom_border_min, mean_bottom, bottom_border_max, \
        top_border_min, mean_top, top_border_max = detect_consolidation(candles_range)

        #CHECK IF FOUND CONSECUTIVE CONSOLIDATIONS, IF ARE CONSECUTIVE CHECK IF THE PEAK IS TOP OR BOTTOM, IF IS THE SAME AS PREVIOUS THEN DO NOTHING ELSE OPEN A TRADE

        if found:
            last_peak = grouped_peaks.iloc[-1]
            opened_trade = None
            if previous_consolidation_peak is None:
                # print(last_peak['Half'], previous_consolidation_peak)
                show = True
                # DO A TRADE
                # ENTRY PRICE IS TOP MID BOTTOM OF CHANNEL?
                if last_peak['Half']:
                    opened_trade = trader.open_trade(entry_price=bottom_border_max, stop_loss=bottom_border_min, take_profit=top_border_min, date=start_date)
                    # IF WE ARE ON TOP THEN TRY TO FIND A LONG FROM BOTTOM
                else:
                    opened_trade = trader.open_trade(entry_price=top_border_min, stop_loss=top_border_max, take_profit=bottom_border_max, date=start_date)
                    # IF WE ARE ON BOTTOM THEN TRY TO FIND A SHORT FROM TOP
            else:
                # print(last_peak['Half'], previous_consolidation_peak['Half'])
                if previous_consolidation_peak['Half'] != last_peak['Half']:
                    show = True
                    if last_peak['Half']:
                        opened_trade = trader.open_trade(entry_price=bottom_border_max, stop_loss=bottom_border_min, take_profit=top_border_min, date=start_date)
                        # IF WE ARE ON TOP THEN TRY TO FIND A LONG FROM BOTTOM
                    else:
                        opened_trade = trader.open_trade(entry_price=top_border_min, stop_loss=top_border_max, take_profit=bottom_border_max, date=start_date)
                        # IF WE ARE ON BOTTOM THEN TRY TO FIND A SHORT FROM TOP
            if opened_trade is not None:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

                mpf.plot(candles_range[::-1].set_index('Date'), type='candle', style='charles', ylabel='Price',
                         datetime_format='%H:%M:%S', ax=ax1, show_nontrading=True)

                ax2.fill_between(grouped_peaks['Date'], top_border_max,
                                 top_border_min, color='blue', alpha=0.3)
                ax2.fill_between(grouped_peaks['Date'], bottom_border_max,
                                 bottom_border_min, color='red', alpha=0.3)

                ax2.axhline(mean_top, color='blue')
                ax2.axhline(mean_bottom, color='red')

                ax2.plot(grouped_peaks['Date'], grouped_peaks['Peak'])

                fig.suptitle('BTC Candlestick Chart and Peak Analysis')

                plt.tight_layout()
                plt.savefig(f'trades/{opened_trade.open_date} {max_window_size} {opened_trade.type} {opened_trade.entry_price}.png')
                if show and False:

                    plt.show()

                # plt.cla()
                # plt.clf()
            previous_consolidation_peak = last_peak

            # print(previous_consolidation_peak)
        else:
            previous_consolidation_peak = None
    else:
        trader.cancel_unopened_trades()

    # if found_in_window == False and len(previous_consolidation_dates) != 0:
    #     print(previous_consolidation_dates[0], previous_consolidation_dates[-1])
    #
    #     consolidation_start = previous_consolidation_dates[0][0]
    #     consolidation_end = previous_consolidation_dates[-1][0] + previous_consolidation_dates[-1][-1]
    #     candles_range = df.loc[consolidation_start:consolidation_end].copy()
    #     candles_range = candles_range[(candles_range['High'] - candles_range['Low']) > 1]
    #     candles_range = candles_range[::-1]
    #     candles_range.reset_index(inplace=True)
    #     mpf.plot(candles_range[::-1].set_index('Date'), type='candle', style='charles', title='BTC Candlestick Chart', ylabel='Price',
    #              datetime_format='%H:%M:%S')
    #     grouped_peaks, found, \
    #     bottom_border_min, mean_bottom, bottom_border_max, \
    #     top_border_min, mean_top, top_border_max = detect_consolidation(candles_range)
    #     if found:
    #         plt.fill_between(grouped_peaks['Date'], top_border_max,
    #                          top_border_min, color='blue', alpha=0.3)
    #         plt.fill_between(grouped_peaks['Date'], bottom_border_max,
    #                          bottom_border_min, color='red', alpha=0.3)
    #
    #         plt.axhline(mean_top, color='blue')
    #         plt.axhline(mean_bottom, color='red')
    #
    #         plt.plot(grouped_peaks['Date'], grouped_peaks['Peak'])
    #         # plt.savefig(f'consolidations_new/{str(candles_range["Date"].values[0]).replace(":", "-").replace(" ", "_")}-{str(candles_range["Date"].values[-1]).replace(":", "-").replace(" ", "_")}.png')
    #         plt.show()
    #
    #         plt.cla()
    #         plt.clf()
    #         # with open("results.txt", 'w+') as f:
    #         #     f.write(f"{str(start_date)}\n")
    #     previous_consolidation_dates = []
    # print(time.time() - t)


