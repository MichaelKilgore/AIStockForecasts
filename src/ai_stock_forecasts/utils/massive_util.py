
from massive import RESTClient

client = RESTClient("EFhFkyh8w1YkWtK9ZITbfI38axIejVa2")

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

