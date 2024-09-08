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
        self.leverage = 1
        self.opened_trades = []

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
            print(result)

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
        self.leverage = leverage
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
        return response


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
        return response_stop_order

    def cancel_order(self, orderId):
        body_cancel_order = {
            "id": "90",
            "method": "order.cancel",
            "params": {
                "orderId": orderId,
                "symbol": "BTCUSDT",
                "timestamp": int(time.time() * 1000)
            }
        }

        self.ws.send(json.dumps(body_cancel_order))
        response_stop_order = json.loads(self.ws.recv())
        return response_stop_order

    def close_order(self, order):
        close_side = 'SELL' if order['result']['side'] == 'BUY' else 'BUY'
        print(order)
        body_close_order = {
            "id": "91",
            "method": "order.place",
            "params": {
                'symbol': order['result']['symbol'],
                'side': close_side,
                'type': 'MARKET',
                'quantity': order['result']['origQty'],
                'workingType': "MARK_PRICE",
                'timestamp': int(time.time() * 1000),
            }
        }

        self.ws.send(json.dumps(body_close_order))
        response_stop_order = json.loads(self.ws.recv())
        return response_stop_order

    def cancel_all_orders(self, orders_=None):
        if orders_ is None:
            orders_ = self.opened_trades
        with ThreadPoolExecutor() as executor:
            futures = []
            for orders in orders_:
                for order in orders:
                    order_id = order['result']['orderId']
                    futures.append(executor.submit(self.cancel_order, order_id))

                    if order['result']['type'] == "LIMIT":
                        futures.append(executor.submit(self.close_order, order))

            for future in futures:
                future.result()
        self.opened_trades = []

    def trade(self, symbol, side, quantity, price, stop_loss_price, take_profit_price):

        def run_limit_order():
            return self.place_limit_order(symbol, side, quantity, price)

        def run_stop_loss():
            return self.place_stop_order(symbol=symbol, side=side, type='STOP_MARKET', quantity=quantity, stop_price=stop_loss_price)

        def run_take_profit():
            return self.place_stop_order(symbol=symbol, side=side, type='TAKE_PROFIT_MARKET', quantity=quantity, stop_price=take_profit_price)

        def run_cancel_all_orders():
            return self.cancel_all_orders()

        t = time.time()
        with ThreadPoolExecutor(max_workers=3) as executor:
            run_cancel_all_orders_future = executor.submit(run_cancel_all_orders)
            limit_order_future = executor.submit(run_limit_order)
            stop_loss_future = executor.submit(run_stop_loss)
            take_profit_future = executor.submit(run_take_profit)

            orders = [
                limit_order_future.result(),
                stop_loss_future.result(),
                take_profit_future.result()
            ]
            if not all([str(order['status'])[0] == "2" for order in orders]):
                self.cancel_all_orders(orders)
            else:
                self.opened_trades.append(orders)
        print(time.time() - t)
def main():
    binance_api = BinanceApi(
        binance_api_key=os.getenv("BINANCE_API_KEY"),
        binance_api_private_key=os.getenv("BINANCE_API_SECRET"),
        binance_ws_key=os.getenv("BINANCE_API_KEY_WS"),
        binance_ws_private_key_path="Private_key.pem"
    )

    symbol = "BTCUSDT"

    # binance_api.set_leverage(symbol, 100)
    # print(10 * 100 / 53000)
    # binance_api.trade(symbol, "BUY", 0.002, 53820, 50500, 59000)
    # binance_api.cancel_all_orders()
if __name__ == "__main__":
    main()