import copy
import datetime
import os
import time
from concurrent.futures import ThreadPoolExecutor

import binance
import numpy as np
import websockets
import asyncio
import json
import pandas as pd
from matplotlib import pyplot as plt
import mplfinance as mpf
import warnings
from binance_api import BinanceApi


warnings.filterwarnings("ignore")
directory = "trades"
if not os.path.exists(directory):
    os.makedirs(directory)


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


class Trader():
    def __init__(self, binance_api, symbol="BTCUSDT", dollars_per_trade=5, price_file="test.csv"):
        self.trades = []
        self.binance_api = binance_api
        self.price_saver = ThreadPoolExecutor(max_workers=1)
        self.price_file = price_file
        self.dollars_per_trade = dollars_per_trade
        self.symbol = symbol

    def open_trade(self, entry_price, stop_loss, take_profit, quantity, side, price):
        # self.cancel_unopened_trades()
        print(f"{side} TRADE OPENED AT {entry_price}, {price} ")
        self.binance_api.trade(symbol=self.symbol, side=side, quantity=round(quantity, 3), price=round(entry_price, 1), stop_loss_price=round(stop_loss, 1), take_profit_price=round(take_profit, 1))


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
        if len(peaks) <= 4:
            return None, False, None, None, None, None, None, None
        peaks['Half'] = peaks['Peak'] >= midpoint
        peaks['Group'] = (peaks['Half'] != peaks['Half'].shift()).cumsum()
        grouped_peaks = peaks.groupby(['Group']).apply(get_extreme_row).reset_index(drop=True)
        grouped_peaks = grouped_peaks.drop(columns=["Group"])

        tops = grouped_peaks[grouped_peaks['Half'] == True]
        bottoms = grouped_peaks[grouped_peaks['Half'] == False]

        mean_top = np.mean(tops['Peak'])
        mean_bottom = np.mean(bottoms['Peak'])

        diff = mean_top - mean_bottom
        g = diff / 6000

        top_border_min = mean_top * (1 - (g / 100))
        top_border_max = mean_top * (1 + (g / 100))

        bottom_border_min = mean_bottom * (1 - (g / 100))
        bottom_border_max = mean_bottom * (1 + (g / 100))

        if len(tops) + len(bottoms) > 4 and values_in_range(
                tops['Peak'],
                top_border_min,
                top_border_max) \
                and \
                values_in_range(
                    bottoms['Peak'],
                    bottom_border_min,
                    bottom_border_max
                ):
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
            if is_previous_green != is_current_green and is_previous_green is not None:
                peaks.append((previous_candle['Date'], candle['Open'], previous_candle['Order']))
            if is_previous_green is None:
                peaks.append((candle['Date'], candle['Open'], candle['Order']))
            if idx == len(candles_range) - 1:
                peaks.append((candle['Date'], candle['Close'], candle['Order']))

            is_previous_green = is_current_green
            previous_candle = candle
        peaks = pd.DataFrame(columns=['Date', 'Peak', 'Order'], data=peaks)

        return peaks

    async def save_candle_to_file(self, candle):
        with open(self.price_file, "a") as f:
            f.write(candle['Date'] + "," + candle['Open'] + "," + candle['Close'])

    async def run(self, pair):
        url = f"wss://fstream.binance.com/ws/{pair.lower()}@markPrice@1s"
        with open(self.price_file, "w") as file:
            pass
        file = open(self.price_file, "a")
        candles_range = pd.DataFrame(columns=['Date', 'Open', 'Close'])
        current_candle = None

        file.write(','.join(candles_range.columns) + "\n")

        timer = 0
        max_window_width = 120
        min_window_width = 10
        previous_consolidation_peak = None

        async with websockets.connect(url) as ws:
            print("RECEIVING DATA")
            while True:
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
                        saved = None
                        start_date = copy.copy(date)
                        candles_range['Order'] = range(len(candles_range))
                        total_reversals = self.detect_peaks(candles_range)
                        print(start_date, current_candle['Close'])
                        for i in range(min_window_width, len(candles_range), 1):
                            candles_sub_range = candles_range[-i:]

                            peaks = total_reversals[total_reversals['Date'].isin(candles_sub_range['Date'])]
                            grouped_peaks, found, \
                            bottom_border_min, mean_bottom, bottom_border_max, \
                            top_border_min, mean_top, top_border_max = self.detect_consolidation(peaks)

                            if found:
                                saved = (
                                    grouped_peaks, found, bottom_border_min, mean_bottom, bottom_border_max,
                                    top_border_min, mean_top, top_border_max, candles_sub_range, peaks
                                )

                        if saved is not None:
                            grouped_peaks, found, \
                            bottom_border_min, mean_bottom, bottom_border_max, \
                            top_border_min, mean_top, top_border_max, candles_sub_range, peaks = saved

                            if found:
                                last_peak = grouped_peaks.iloc[-1]

                                long_entry = current_candle['Close'] + (0.1 * abs(current_candle['Close'] - top_border_min))
                                short_entry = current_candle['Close'] - (0.1 * abs(current_candle['Close'] - bottom_border_max))

                                sl_long = current_candle['Close'] - (top_border_max - top_border_min)
                                sl_short = current_candle['Close'] + (top_border_max - top_border_min)
                                short_trade_quantity = self.dollars_per_trade * self.binance_api.leverage / short_entry
                                long_trade_quantity = self.dollars_per_trade * self.binance_api.leverage / long_entry

                                if previous_consolidation_peak is None:
                                    if last_peak['Half']:
                                        self.open_trade(entry_price=short_entry, stop_loss=sl_short,
                                                        take_profit=bottom_border_max, quantity=short_trade_quantity, side="SELL", price=current_candle['Close'])
                                    else:
                                        self.open_trade(entry_price=long_entry, stop_loss=sl_long,
                                                        take_profit=top_border_min, quantity=long_trade_quantity, side="BUY", price=current_candle['Close'])
                                else:
                                    if previous_consolidation_peak['Half'] != last_peak['Half']:
                                        if last_peak['Half']:
                                            self.open_trade(entry_price=short_entry, stop_loss=sl_short,
                                                            take_profit=bottom_border_max, quantity=short_trade_quantity, side="SELL", price=current_candle['Close'])
                                        else:
                                            self.open_trade(entry_price=long_entry, stop_loss=sl_long,
                                                            take_profit=top_border_min, quantity=long_trade_quantity, side="BUY", price=current_candle['Close'])
                                if found:

                                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

                                    candles_sub_range[['High', 'Low']] = [
                                        [open_price, close_price] if open_price > close_price else [close_price,
                                                                                                    open_price] for
                                        idx, (close_price, open_price) in candles_sub_range[['Close', 'Open']].iterrows()]
                                    mpf.plot(candles_sub_range[::-1].set_index('Date'), type='candle', style='charles',
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
                                        f'trades/{current_candle["Close"]}_{str(current_candle["Date"]).replace(":", "-").replace(" ", "_")}.png')

                                previous_consolidation_peak = last_peak

                            else:
                                previous_consolidation_peak = None
                        else:
                            try:
                                self.binance_api.cancel_all_orders()
                            except:
                                quit()
                    asyncio.get_event_loop().run_in_executor(self.price_saver, self.save_candle_to_file, current_candle)
                    current_candle = None

if __name__ == "__main__":
    pair = 'BTCUSDT'
    binance_api = BinanceApi(
        binance_api_key=os.getenv("BINANCE_API_KEY"),
        binance_api_private_key=os.getenv("BINANCE_API_SECRET"),
        binance_ws_key=os.getenv("BINANCE_API_KEY_WS"),
        binance_ws_private_key_path="Private_key.pem"
    )
    binance_api.set_leverage(symbol=pair, leverage=100)
    trader = Trader(binance_api=binance_api, dollars_per_trade=10)

    asyncio.get_event_loop().run_until_complete(trader.run(pair))

