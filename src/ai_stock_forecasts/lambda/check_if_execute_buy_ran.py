import logging
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from boto3.dynamodb.conditions import Key

from ai_stock_forecasts.utils.dynamodb_util import DynamoDBUtil
from ai_stock_forecasts.utils.telegram_bot_util import TelegramBotUtil


logging.basicConfig(level=logging.INFO)


def lambda_handler(event: dict, context: Any) -> dict:
    model_id = event['model_id']
    date_override = event.get('date')
    today_prefix = date_override if date_override else datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d')

    db_util = DynamoDBUtil()
    telegram = TelegramBotUtil()
    resp = db_util.order_table.query(
        KeyConditionExpression=Key('model_id').eq(model_id) & Key('order_timestamp').begins_with(today_prefix),
    )

    items = resp.get('Items', [])
    if not items:
        msg = f'execute_buy did NOT run for model_id={model_id} on {today_prefix} (no dynamodb records)'
        logging.warning(msg)
        telegram.send_message(msg)
        return {'symbols': []}

    buy_symbols = [
        order['symbol']
        for item in items
        if item['orders'] and all(order['order_type'] == 'buy' for order in item['orders'])
        for order in item['orders']
    ]

    if not buy_symbols:
        msg = f'execute_buy ran for model_id={model_id} on {today_prefix} but no buy orders were placed, which means there was probably some type of failure mid run'
        logging.warning(msg)
        telegram.send_message(msg)
        return {'symbols': []}

    msg = f'execute_buy SUCCEEDED for model_id={model_id} on {today_prefix}. Purchased: {", ".join(buy_symbols)}'
    logging.info(msg)
    telegram.send_message(msg)
    return {'symbols': buy_symbols}

if __name__ == '__main__':
    lambda_handler({ 'model_id': 'ubuntu-with-even-more-recent-training', 'date': '2026-05-06' }, context=None)

