from main_optim import *

df = pd.read_csv('BTCUSDT-3s-2024-07.csv', index_col=0)

# df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
# df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

df['Date'] = pd.to_datetime(df['Date'])
df = df[(df['Date'] >= '2024-07-08 15:48:33') & (df['Date'] <= '2024-07-08 15:50:51')]
df['Order'] = range(len(df))

candles_range = df.copy()
mpf.plot(candles_range[::-1].set_index('Date'), type='candle', style='charles', ylabel='Price',
         datetime_format='%H:%M:%S', show_nontrading=True)
peaks = check_reversal(candles_range)

candles_range = candles_range[(candles_range['High'] - candles_range['Low']) > 1]
candles_range.reset_index(inplace=True)

grouped_peaks, found, \
bottom_border_min, mean_bottom, bottom_border_max, \
top_border_min, mean_top, top_border_max = detect_consolidation(peaks)

#CHECK IF FOUND CONSECUTIVE CONSOLIDATIONS, IF ARE CONSECUTIVE CHECK IF THE PEAK IS TOP OR BOTTOM, IF IS THE SAME AS PREVIOUS THEN DO NOTHING ELSE OPEN A TRADE
# 15% z kazdej strony lub mniej
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

# 5 17:54 23:13
# 7 30:29 32:52
# 7 58:07 8 00:45
# 8 57:38 59:05
# 9 19:27 20:32
# 9 31:56 32:39
# 9 40:47 42:49 MULTI
# 10 27:31 28:29