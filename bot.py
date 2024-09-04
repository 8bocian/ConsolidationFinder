import copy
import datetime
import time

import binance
import numpy as np
import websockets
import asyncio
import json
import pandas as pd
from matplotlib import pyplot as plt
import mplfinance as mpf
import warnings

# Suppress a specific warning category
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Suppress warnings with a specific message
warnings.filterwarnings("ignore", message="some specific warning message")

warnings.filterwarnings("ignore")


def values_in_range(values, lower_border, upper_border):
    for value in values:
        if not (lower_border <= value <= upper_border):
            return False
    return True


def get_extreme_row(group):
    return group.loc[group['Peak'].idxmax()] if group['Half'].iloc[0] else group.loc[group['Peak'].idxmin()]


class Trade:
    def __init__(self, entry_price, stop_loss, take_profit, date, size=10_000):
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


class Trader:
    def __init__(self):
        self.trades = []

    def open_trade(self, entry_price, stop_loss, take_profit, date):
        # self.cancel_unopened_trades()
        new_trade = Trade(entry_price, stop_loss, take_profit, date)
        self.trades.append(new_trade)
        return new_trade

    def detect_consolidation(self, peaks):
        found = False

        top = peaks['Peak'].max()
        bottom = peaks['Peak'].min()

        midpoint = (top + bottom) / 2
        middle_threshold = 0.7
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

        if len(tops) == 0 or len(bottoms) == 0 or len(grouped_peaks) <= 4:
            return None, found, \
                   None, None, None, \
                   None, None, None

        mean_top = np.mean(tops['Peak'])
        mean_bottom = np.mean(bottoms['Peak'])

        diff = mean_top - mean_bottom
        g = diff / 6000

        top_border_min = mean_top * (1 - (g / 100))
        top_border_max = mean_top * (1 + (g / 100))

        bottom_border_min = mean_bottom * (1 - (g / 100))
        bottom_border_max = mean_bottom * (1 + (g / 100))

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

    def detect_peaks(self, candles_range):
        candles_range = candles_range[abs(candles_range['Open'] - candles_range['Close']) >= 1]

        peaks = []
        is_previous_green = None
        candles_range.reset_index(inplace=True)
        previous_candle = None
        for idx, candle in candles_range.iterrows():
            is_current_green = candle['Close'] > candle['Open']
            T = 0
            if is_previous_green != is_current_green and is_previous_green is not None:
                peaks.append((previous_candle['Date'], candle['Open'], previous_candle['Order']))
                T += 1
            if is_previous_green is None:
                peaks.append((candle['Date'], candle['Open'], candle['Order']))
                T += 1
            if idx == len(candles_range) - 1:
                peaks.append((candle['Date'], candle['Close'], candle['Order']))
                T += 1

            is_previous_green = is_current_green
            previous_candle = candle
        peaks = pd.DataFrame(columns=['Date', 'Peak', 'Order'], data=peaks)

        return peaks

    async def run(self, pair):
        url = f"wss://fstream.binance.com/ws/{pair}@markPrice@1s"

        candles_range = pd.DataFrame(columns=['Date', 'Open', 'Close'])
        current_candle = None
        timer = 0
        max_window_width = 120
        min_window_width = 10

        async with websockets.connect(url) as ws:
            print("RECEIVING DATA")
            while True:
                t = time.time()
                message = await ws.recv()
                data = json.loads(message)
                price = float(data['p'])
                date = datetime.datetime.fromtimestamp(data['E']/1000)

                timer += 1
                if current_candle is None:
                    current_candle = {"Open": price}
                if timer == 3:
                    timer = 0
                    current_candle['Close'] = price
                    current_candle['Date'] = date
                    new_row = pd.DataFrame([current_candle])

                    candles_range = pd.concat([candles_range, new_row], ignore_index=True)
                    if len(candles_range) > max_window_width:
                        candles_range = candles_range.iloc[-max_window_width:]

                    #check if have enough candles

                    if len(candles_range) >= min_window_width:
                        max_window_size = 0
                        start_date = copy.copy(date)
                        print(start_date)
                        candles_range['Order'] = range(len(candles_range))
                        total_reversals = self.detect_peaks(candles_range)
                        for i in range(min_window_width, len(candles_range), 1):
                            candles_sub_range = candles_range[-i:]

                            peaks = total_reversals[total_reversals['Date'].isin(candles_sub_range['Date'])]

                            grouped_peaks, found, \
                            bottom_border_min, mean_bottom, bottom_border_max, \
                            top_border_min, mean_top, top_border_max = self.detect_consolidation(peaks)

                            if found:
                                max_window_size = i

                        if max_window_size != 0:
                            peaks = total_reversals[total_reversals['Order'].isin(candles_sub_range['Order'])]

                            candles_sub_range.reset_index(inplace=True)

                            grouped_peaks, found, \
                            bottom_border_min, mean_bottom, bottom_border_max, \
                            top_border_min, mean_top, top_border_max = self.detect_consolidation(peaks)

                            if found:
                                last_peak = grouped_peaks.iloc[-1]
                                opened_trade = None

                                long_entry = current_candle['Close'] + (0.1 * abs(current_candle['Close'] - top_border_min))
                                short_entry = current_candle['Close'] - (0.1 * abs(current_candle['Close'] - bottom_border_max))

                                sl_long = current_candle['Close'] - (top_border_max - top_border_min)
                                sl_short = current_candle['Close'] + (top_border_max - top_border_min)
                                if previous_consolidation_peak is None:
                                    show = True
                                    if last_peak['Half']:
                                        opened_trade = self.open_trade(entry_price=short_entry, stop_loss=sl_short,
                                                                         take_profit=bottom_border_max, date=start_date)
                                    else:
                                        opened_trade = self.open_trade(entry_price=long_entry, stop_loss=sl_long,
                                                                         take_profit=top_border_min, date=start_date)
                                else:
                                    if previous_consolidation_peak['Half'] != last_peak['Half']:
                                        show = True
                                        if last_peak['Half']:
                                            opened_trade = self.open_trade(entry_price=short_entry, stop_loss=sl_short,
                                                                             take_profit=bottom_border_max, date=start_date)
                                        else:
                                            opened_trade = self.open_trade(entry_price=long_entry, stop_loss=sl_long,
                                                                             take_profit=top_border_min, date=start_date)
                                if opened_trade is not None:
                                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

                                    mpf.plot(candles_range[::-1].set_index('Date'), type='candle', style='charles',
                                             ylabel='Price',
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
                                    plt.savefig(
                                        f'trades/{str(opened_trade.open_date).replace(":", "-")} {max_window_size} {opened_trade.type} {opened_trade.entry_price}.png')

                                previous_consolidation_peak = last_peak

                            else:
                                previous_consolidation_peak = None
                        else:
                            ...
                            # candel unopened trades?
                    current_candle = None

if __name__ == "__main__":
    trader = Trader()
    pair = 'btcusdt'
    asyncio.get_event_loop().run_until_complete(trader.run(pair))

