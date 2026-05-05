from datetime import datetime
from typing import Optional

from alpaca.trading.enums import OrderSide

class OrderItem:
    def __init__(self, symbol: str, quantity: float, order_type: OrderSide, limit_price: Optional[float] = None):
        self.symbol = symbol
        self.quantity = quantity
        self.order_type = order_type
        self.limit_price = limit_price

class Order:

    def __init__(self, model_id: str, order_timestamp: datetime, order_items: list[OrderItem]):
        self.model_id: str = model_id
        self.order_timestamp: datetime = order_timestamp
        self.order_items = order_items
