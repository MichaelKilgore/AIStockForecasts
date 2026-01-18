import json
from ai_stock_forecasts.model.orchestration.orchestration import Orchestration

def lambda_handler(event, context):
    orc = Orchestration(['AAPL'], 'm1-medium-high-with-less-features-and-earnings-calendar-features', './src/ai_stock_forecasts/constants/configs.yaml')

    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }


