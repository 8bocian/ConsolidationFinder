import hmac
import time
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from cryptography.hazmat.primitives.serialization import load_pem_private_key
import base64
import requests
import json
import websocket
import dotenv
import os
dotenv.load_dotenv()


class BinanceApi():
    def __init__(self, binance_api_key, binance_api_private_key, binance_ws_key, binance_ws_private_key_path="Private_key.pem", ws_base_url="wss://ws-fapi.binance.com/ws-fapi/v1", api_base_url="https://fapi.binance.com"):
        self.ws_base_url = ws_base_url
        self.api_base_url = api_base_url
        self._binance_api_key = binance_api_key
        self._binance_api_private_key = binance_api_private_key
        self._binance_ws_key = binance_ws_key

        with open(binance_ws_private_key_path, 'rb') as f:
            self._binance_ws_private_key = load_pem_private_key(data=f.read(),
                                               password=None)

        self.ws = websocket.create_connection(self.ws_base_url)

        login_body = {
            "id": "1",
            "method": "session.logon",
            "params": {
                "timestamp": int(time.time() * 1000),
                "apiKey": self._binance_ws_key
            }
        }

        login_body['params']['signature'] = self.signature_ws(login_body['params'])

        self.ws.send(json.dumps(login_body))
        result = json.loads(self.ws.recv())
        if str(result['status'])[0] == "2":
            print("Logged in ")
        else:
            print("Failed to log in")

    def signature_ws(self, payload):
        payload = '&'.join([f'{param}={value}' for param, value in sorted(payload.items())])
        signature = base64.b64encode(self._binance_ws_private_key.sign(payload.encode('ASCII')))
        return signature.decode('ASCII')

    def signature_api(self, payload):
        payload = '&'.join([f'{param}={value}' for param, value in sorted(payload.items())])
        return hmac.new(self._binance_api_private_key.encode('utf-8'), payload.encode('utf-8'), sha256).hexdigest()

    def set_leverage(self, symbol, leverage):
        params = {
            'symbol': symbol,
            'leverage': leverage,
            'timestamp': int(time.time() * 1000)
        }

        signature = self.signature_api(params)
        params['signature'] = signature

        headers = {
            'X-MBX-APIKEY': self._binance_api_key
        }

        response = requests.post(f'{self.api_base_url}/fapi/v1/leverage', headers=headers, params=params)
        return response.json()

    def place_limit_order(self, symbol, side, quantity, price):
        body_order = {
            'id': "2321",
            'method': 'order.place',
            'params': {
                # 'apiKey': self._binance_ws_key,
                'symbol': symbol,
                'side': side,
                'type': 'LIMIT',
                'quantity': quantity,
                'price': price,
                'workingType': "MARK_PRICE",
                'timeInForce': 'GTC',
                'timestamp': int(time.time() * 1000)
            }
        }

        # body_order['params']['signature'] = self.signature_ws(body_order['params'])

        self.ws.send(json.dumps(body_order))
        response = json.loads(self.ws.recv())
        update_time = int(response['result']['updateTime']) / 1000
        took = time.time() - update_time
        return took


    def place_stop_order(self, symbol, side, type, quantity, stop_price):
        close_side = 'SELL' if side == 'BUY' else 'BUY'

        body_stop_loss = {
            'id': "2321",
            'method': 'order.place',
            'params': {
                # 'apiKey': self._binance_ws_key,
                'symbol': symbol,
                'side': close_side,
                'type': type,
                'quantity': quantity,
                'stopPrice': stop_price,
                'workingType': "MARK_PRICE",
                'timestamp': int(time.time() * 1000)
            }
        }

        # body_stop_loss['params']['signature'] = self.signature_ws(body_stop_loss['params'])
        self.ws.send(json.dumps(body_stop_loss))
        response_stop_order = json.loads(self.ws.recv())
        update_time = int(response_stop_order['result']['updateTime']) / 1000
        took = time.time() - update_time
        return took

    def trade_with_leverage(self, symbol, side, quantity, price, stop_loss_price, take_profit_price):
        def run_limit_order():
            return self.place_limit_order(symbol, side, quantity, price)

        def run_stop_loss():
            return self.place_stop_order(symbol=symbol, side=side, type='STOP_MARKET', quantity=quantity, stop_price=stop_loss_price)

        def run_take_profit():
            return self.place_stop_order(symbol=symbol, side=side, type='TAKE_PROFIT_MARKET', quantity=quantity, stop_price=take_profit_price)

        t = time.time()
        with ThreadPoolExecutor(max_workers=3) as executor:

            limit_order_future = executor.submit(run_limit_order)
            stop_loss_future = executor.submit(run_stop_loss)
            take_profit_future = executor.submit(run_take_profit)

        print(time.time() - t)


def main():
    binance_api = BinanceApi(
        binance_api_key=os.getenv("BINANCE_API_KEY"),
        binance_api_private_key=os.getenv("BINANCE_API_SECRET"),
        binance_ws_key=os.getenv("BINANCE_API_KEY_WS"),
        binance_ws_private_key_path="Private_key.pem"
    )

    symbol = "BTCUSDT"

    binance_api.set_leverage(symbol, 10)
    binance_api.trade_with_leverage(symbol, "SELL", 0.002, 59000, 59200, 51000)

if __name__ == "__main__":
    main()