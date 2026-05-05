import databento as db

client = db.Historical('db-mVqGfaqEfSubTWabLchstVFSJdcqq')  # uses DATABENTO_API_KEY env var

data = client.timeseries.get_range(
    dataset="XNAS.ITCH",      # Nasdaq TotalView-ITCH; example equities dataset
    schema="mbp-1",           # top-of-book bid/ask updates
    symbols=["AAPL"],
    stype_in="raw_symbol",
    start="2024-01-02T14:30:00",
    end="2024-01-02T21:00:00",
)

df = data.to_df()
print(df.head())

# day   bid     ask         return_lag_week
# mon   187.10  187.20
# tue   187.30  187.50      PROFITED
# wed   190.00  190.50

# ask has to be less than bid on day we sell for it to be considered a profit

