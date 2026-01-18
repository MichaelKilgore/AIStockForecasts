from typing import Optional
import boto3
import os

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from dotenv import load_dotenv

from ai_stock_forecasts.models.order import Order, OrderItem

from datetime import datetime

from decimal import Decimal

from boto3.dynamodb.conditions import Key

class DynamoDBUtil:
    def __init__(self):
        load_dotenv()

        self._alpaca_key = os.getenv("ALPACA_KEY")
        self._alpaca_secret = os.getenv("ALPACA_SECRET")

        self.region = os.getenv("REGION_NAME")

        access_key = os.getenv("ACCESS_KEY")
        secret_access_key = os.getenv("SECRET_ACCESS_KEY")

        self.db = boto3.resource(
            'dynamodb',
            region_name=self.region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_access_key,
        )
        self.order_table = self.db.Table('orders_table')

        self.trading_client = TradingClient(self._alpaca_key, self._alpaca_secret, paper=True)


    """ order structure:
            pk: model_id
            sk: order_timestamp
            total_money_invested,
            orders: [
                {
                    symbol,
                    quantity,
                    order_type
                },
                ...
            ]
    """
    def upload_order(self, order: Order):
        print(f'uploading order to dynamodb for model_id: {order.model_id}, order_timestamp: {order.order_timestamp}')
        self.order_table.put_item(
            Item={
                'model_id': order.model_id,
                'order_timestamp': str(order.order_timestamp),
                'total_money_invested': Decimal(str(order.total_money_invested)),
                'orders': [ { 'symbol': order_item.symbol, 'quantity': Decimal(str(order_item.quantity)), 'order_type': order_item.order_type } for order_item in order.order_items ]
            }
        )

    def get_latest_order(self, model_id: str) -> Optional[Order]:
        resp = self.order_table.query(
            KeyConditionExpression=Key('model_id').eq(model_id),
            ScanIndexForward=False,
            Limit=1
        )

        latest = resp['Items'][0] if resp.get('Items') else None

        if latest == None:
            return None

        order_items = [ OrderItem(order['symbol'], order['quantity'], OrderSide(order['order_type'])) for order in latest['orders']]

        return Order(latest['model_id'], datetime.strptime(latest['order_timestamp'], "%Y-%m-%d %H:%M:%S.%f"), latest['total_money_invested'], order_items)


if __name__ == "__main__":
    order_item = OrderItem('AAPL', 5.3, OrderSide.BUY)
    order_item_2 = OrderItem('AMD', 2.8, OrderSide.BUY)

    order = Order('test_model_id', datetime.now(), 25000, [ order_item, order_item_2 ])

    db = DynamoDBUtil()

    # db.upload_order(order)
    order = db.get_latest_order('test_model_id')

    print(order.model_id, order.order_timestamp, order.total_money_invested, order.order_items[0].symbol)


