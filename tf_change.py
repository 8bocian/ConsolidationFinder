import pandas as pd

df = pd.read_csv('BTCUSDT-1s-2024-07.csv')
df.columns = ["Date", "Open", "High", "Low", "Close", "Volume", "close_time", "quote_volume", "count",
              "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

df['Date'] = pd.to_datetime(df['Date'], unit='ms')

df.set_index('Date', inplace=True)

df_res = df.resample('3S').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
})
df_res.reset_index(inplace=True)
df_res.index = range(len(df_res))
df_res.to_csv('BTCUSDT-3s-2024-07.csv')
print(df_res)
