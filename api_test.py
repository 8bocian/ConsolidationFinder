import datetime
import time

import binance
import websockets
import asyncio
import json
import pandas as pd


class Candle():
    def __init__(self, open_price, high_price, low_price, close_price):
        self.open = open_price
        self.high = high_price
        self.low = low_price
        self.close = close_price



async def run(pair):
    url = f"wss://fstream.binance.com/ws/{pair}@markPrice@1s"

    async with websockets.connect(url) as ws:
        while True:
            t = time.time()
            message = await ws.recv()
            data = json.loads(message)
            print(data['p'], datetime.datetime.fromtimestamp(data['E']/1000), datetime.datetime.now())


if __name__ == "__main__":
    pair = 'btcusdt'
    asyncio.get_event_loop().run_until_complete(run(pair))

# MAKE IT TO BE 3 second candle from 1s price ticker