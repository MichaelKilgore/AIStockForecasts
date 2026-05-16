## Data Migration Notes

### historical_data -> historical_data_v2

historical_data had the data partitioned by symbol. But aggregating the data into that many separate files made it take a long time to pull the data locally so I migrated to v2 which partitions data by feature only.

## historical_data_v2 -> historical_data_v3

historical_data_v2 contains only 500 s&p symbols worth of data. historical_data_v3 contains 5000+ symbols

## historical_data_v3 -> historical_data_v4

historical_data_v4 pulls all the price history data from yfinance instead of alpaca as data in alpaca is often not accurate.

