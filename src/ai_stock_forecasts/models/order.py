from datetime import datetime

from alpaca.trading.enums import OrderSide

class OrderItem:
    def __init__(self, symbol: str, quantity: float, order_type: OrderSide):
        self.symbol = symbol
        self.quantity = quantity 
        self.order_type = order_type

class Order:

    def __init__(self, model_id: str, order_timestamp: datetime,
                 total_money_invested: float, order_items: list[OrderItem]):
        self.model_id: str = model_id
        self.order_timestamp: datetime = order_timestamp
        self.total_money_invested: float = total_money_invested
        self.order_items = order_items
