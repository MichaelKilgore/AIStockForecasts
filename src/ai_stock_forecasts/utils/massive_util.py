
from massive import RESTClient

client = RESTClient("EFhFkyh8w1YkWtK9ZITbfI38axIejVa2")

def get_tickers():
    tickers = []
    for t in client.list_tickers(
        market="stocks",
        active="true",
        order="asc",
        limit="100",
        sort="ticker",
        ):
        tickers.append(t)

    print(tickers)

def get_daily(date: str='2025-04-28'):
    request = client.get_daily_open_close_agg(
        "AACB",
        date,
        adjusted="true",
    )

    print(request)

get_daily('2025-04-28')
get_daily('2025-04-29')
get_daily('2025-04-30')
get_daily('2025-05-01')
get_daily('2025-05-02')
get_daily('2025-05-03')
get_daily('2025-05-04')
get_daily('2025-05-05')
get_daily('2025-05-06')
