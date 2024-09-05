import threading
import time
import hmac
import hashlib
from urllib.parse import urlencode
import requests
import json
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization
import websocket


class BinanceApi():
    def __init__(self, binance_api_key, binance_api_secret, base_url="wss://ws-fapi.binance.com/ws-fapi/v1"):
        self.base_url = base_url
        self._binance_api_key = binance_api_key
        self._binance_api_secert = binance_api_secret


        self.ws = websocket.WebSocketApp(self.base_url, on_open=self.on_open, on_message=self.on_message)

        wst = threading.Thread(target=self.ws.run_forever)
        wst.start()

    def on_open(self, ws):
        params = {
            "timestamp": int(time.time() * 1000),
            "apiKey": self._binance_api_key
        }

        signature = self.create_ed25519_signature(json.dumps(params, separators=(',', ':')))
        params['signature'] = signature

        login_body = {
            "id": "1",
            "method": "session.logon",
            "params": params
        }
        print("login")
        ws.send(json.dumps(login_body))

    def on_message(self, ws, message):
        data = json.loads(message)
        print("Received Message:", data)


    def create_signature(self, query_string, secret):
        return hmac.new(secret.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()


    def create_ed25519_signature(self, message):
        if isinstance(message, dict):
            message = json.dumps(message, sort_keys=True)
        elif not isinstance(message, str):
            raise TypeError("Message must be a dictionary or a string.")

        message_bytes = message.encode('utf-8')
        private_key_bytes = bytes.fromhex(self._binance_api_secert)

        ed25519_private_key = Ed25519PrivateKey.from_private_bytes(private_key_bytes)

        signature = ed25519_private_key.sign(message_bytes)

        return signature.hex()


    def set_leverage(self, symbol, leverage):
        params = {
            'symbol': symbol,
            'leverage': leverage,
            'timestamp': int(time.time() * 1000)
        }

        query_string = '&'.join([f"{key}={value}" for key, value in params.items()])
        signature = self.create_signature(query_string, self._binance_api_secert)
        params['signature'] = signature

        headers = {
            'X-MBX-APIKEY': self._binance_api_key
        }

        response = requests.post(f'{self.base_url}/fapi/v1/leverage', headers=headers, params=params)
        return response.json()

    def place_limit_order(self, symbol, side, quantity, price):
        # body = {
        #     'id': "2321",
        #     'method': 'order.place',
        #     'params': {
        #         'apiKey': self._binance_api_key,
        #         'symbol': symbol,
        #         'side': side,
        #         'type': 'LIMIT',
        #         'quantity': quantity,
        #         'price': price,
        #         'timeInForce': 'GTC',
        #         'timestamp': int(time.time() * 1000)
        #     }
        # }
        #
        # query_string = '&'.join([f"{key}={value}" for key, value in params.items()])
        # signature = self.create_signature(query_string, self._binance_api_secert)
        # params['signature'] = signature
        #
        # headers = {
        #     'X-MBX-APIKEY': self._binance_api_key
        # }
        #
        # response = requests.post(f'{self.base_url}/fapi/v1/order', headers=headers, params=params)
        # return response.json()
        ...

    def place_stop_loss_take_profit(self, symbol, side, quantity, stop_price, take_profit_price):
        close_side = 'SELL' if side == 'BUY' else 'BUY'

        params_stop_loss = {
            'symbol': symbol,
            'side': close_side,
            'type': 'STOP_MARKET',
            'quantity': quantity,
            'stopPrice': stop_price,
            'timestamp': int(time.time() * 1000)
        }

        query_string_stop_loss = '&'.join([f"{key}={value}" for key, value in params_stop_loss.items()])
        signature_stop_loss = self.create_signature(query_string_stop_loss, self._binance_api_secert)
        params_stop_loss['signature'] = signature_stop_loss

        params_take_profit = {
            'symbol': symbol,
            'side': close_side,
            'type': 'TAKE_PROFIT_MARKET',
            'quantity': quantity,
            'stopPrice': take_profit_price,
            'timestamp': int(time.time() * 1000)
        }

        query_string_take_profit = '&'.join([f"{key}={value}" for key, value in params_take_profit.items()])
        signature_take_profit = self.create_signature(query_string_take_profit, self._binance_api_secert)
        params_take_profit['signature'] = signature_take_profit

        headers = {
            'X-MBX-APIKEY': self._binance_api_key
        }

        response_stop_loss = requests.post(f'{self.base_url}/fapi/v1/order', headers=headers, params=params_stop_loss)

        response_take_profit = requests.post(f'{self.base_url}/fapi/v1/order', headers=headers, params=params_take_profit)

        return response_stop_loss.json(), response_take_profit.json()

    def trade_with_leverage(self, symbol, side, leverage, quantity, price, stop_price, take_profit_price):
        t = time.time()
        # Step 1: Set leverage
        leverage_response = self.set_leverage(symbol, leverage)
        print("Leverage Set Response:", leverage_response)

        # Step 2: Place the initial limit order
        order_response = self.place_limit_order(symbol, side, quantity, price)
        print("Limit Order Response:", order_response)

        # Step 3: Place stop loss and take profit orders

        stop_loss_response, take_profit_response = self.place_stop_loss_take_profit(symbol, side, quantity, stop_price,
                                                                               take_profit_price)
        print("Stop Loss Order Response:", stop_loss_response)
        print("Take Profit Order Response:", take_profit_response)
        print("Took: ", time.time() - t)

import dotenv
import os
dotenv.load_dotenv()

binance_api = BinanceApi(
    binance_api_key=os.getenv("BINANCE_API_KEY"),
    binance_api_secret=os.getenv("BINANCE_API_SECRET")
)

# binance_api.trade_with_leverage("BTCUSDT", "SELL", 10, 0.002, 59000, 59200, 51000)