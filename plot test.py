from main_optim import *

df = pd.read_csv('BTCUSDT-1s-2024-07.csv')
df.columns = ["Date", "Open", "High", "Low", "Close", "Volume", "close_time", "quote_volume", "count",
              "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

df['Date'] = pd.to_datetime(df['Date'], unit='ms')
df = df[(df['Date'] > '2024-07-04 05:17:52') & (df['Date'] < '2024-07-04 05:22:20')]
candles_range = df.copy()
candles_range = candles_range[(candles_range['High'] - candles_range['Low']) > 1]
candles_range.reset_index(inplace=True)
grouped_peaks, found, \
bottom_border_min, mean_bottom, bottom_border_max, \
top_border_min, mean_top, top_border_max = detect_consolidation(candles_range)

#CHECK IF FOUND CONSECUTIVE CONSOLIDATIONS, IF ARE CONSECUTIVE CHECK IF THE PEAK IS TOP OR BOTTOM, IF IS THE SAME AS PREVIOUS THEN DO NOTHING ELSE OPEN A TRADE

if found:
    last_peak = grouped_peaks.iloc[-1]
    # print(last_peak['Half'], previous_consolidation_peak)
    show = True
    # DO A TRADE
    # ENTRY PRICE IS TOP MID BOTTOM OF CHANNEL?
    if last_peak['Half']:
        # opened_trade = trader.open_trade(entry_price=mean_bottom, stop_loss=bottom_border_min, take_profit=top_border_min, date=start_date)
        print("LONG")
        # IF WE ARE ON TOP THEN TRY TO FIND A LONG FROM BOTTOM
    else:
        # opened_trade = trader.open_trade(entry_price=mean_top, stop_loss=top_border_max, take_profit=bottom_border_max, date=start_date)
        print("SHORT")
        # IF WE ARE ON BOTTOM THEN TRY TO FIND A SHORT FROM TOP
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


    plt.show()