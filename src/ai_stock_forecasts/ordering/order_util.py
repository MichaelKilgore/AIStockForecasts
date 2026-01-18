
from dotenv import load_dotenv

from ai_stock_forecasts.models.order import Order, OrderItem

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

import os
import base64
import json
import requests


class OrderUtil:
    def __init__(self):
        self._alpaca_key = os.getenv("ALPACA_KEY")
        self._alpaca_secret = os.getenv("ALPACA_SECRET")

        bytes = (self._alpaca_key + ":" + self._alpaca_secret).encode('utf-8')
        self.base64_password = base64.b64encode(bytes).decode("ascii")

        self.region = os.getenv("REGION_NAME")

        self.trading_client = TradingClient(self._alpaca_key, self._alpaca_secret, paper=True)

    def place_order(self, order: Order): 
        for order_item in order.order_items:
            market_order_data = MarketOrderRequest(
                symbol=order_item.symbol,
                qty=order_item.quantity,
                side=order_item.order_type,
                time_in_force=TimeInForce.DAY
            )

            print(f'placing order on symbol: {order_item.symbol}, qty: {order_item.quantity}, side: {order_item.order_type}')
            _ = self.trading_client.submit_order(
                order_data=market_order_data
            )

    def close_all_positions(self):
        print('closing all positions')
        self.trading_client.close_all_positions(cancel_orders=True)

    def is_stock_market_open(self):
        url = "https://broker-api.alpaca.markets/v1/clock"

        headers = {
            "accept": "application/json",
            "authorization": f"Basic {self.base64_password}"
        }

        response = requests.get(url, headers=headers)

        print(response.text)
        obj = json.loads(response.text)

        return obj['is_open'] == 'true'

