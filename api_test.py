import datetime
import time

import binance
import websockets
import asyncio
import json
import pandas as pd

async def run(pair):
    url = f"wss://fstream.binance.com/ws/{pair}@markPrice@1s"

    with open("log.log", "w") as f:
        ...

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