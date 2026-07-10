# massive

## Download Stock OHLCV Data
```
python scripts/stocks_aggs_download.py --tickers_file data/spy_tickers/tickers_combined_unique.csv --year 2025
```


## Download Stock Fundamentals and Reference Data
```
python scripts/fundamentals_download.py --tickers_file data/spy_tickers/tickers_combined_unique.csv
python scripts/financial_statements_download.py --tickers_file data/spy_tickers/tickers_combined_unique.csv
python scripts/corporate_actions_download.py --tickers_file data/spy_tickers/tickers_combined_unique.csv
python scripts/reference_download.py --tickers_file data/spy_tickers/tickers_combined_unique.csv

ubuntu@ple:~/projects/massive$ ll data/
total 88
drwxrwxr-x 11 ubuntu ubuntu  4096 Jul  7 17:54 ./
drwxrwxr-x  7 ubuntu ubuntu  4096 Jul 10 07:39 ../
drwxrwxr-x  3 ubuntu ubuntu  4096 Jul  7 13:13 corporate_actions/
drwxrwxr-x  3 ubuntu ubuntu 32768 Jul  7 13:57 financials/
drwxrwxr-x  3 ubuntu ubuntu  4096 Jul  7 13:45 fundamentals/
drwxrwxr-x  3 ubuntu ubuntu 20480 Jul  7 12:34 fundamentals.bak/
drwxrwxr-x  3 ubuntu ubuntu  4096 Jul  7 13:06 reference/
drwxrwxr-x  5 ubuntu ubuntu  4096 Jul  6 13:34 SPY/
drwxrwxr-x 28 ubuntu ubuntu  4096 Jul  9 09:43 spy_tickers/
drwxrwxr-x  7 ubuntu ubuntu  4096 Jul 10 07:41 trades/
drwxrwxr-x  3 ubuntu ubuntu  4096 Jul  7 17:54 trades_raw/

```

## Download Stock Trades Data
```
# Individual
python scripts/trades_enrichment_download.py --tickers AAPL,NVDA --year 2025
python scripts/trades_enrichment_download.py --tickers TWTR --year 2022 --aggregate 1min --logs

# Parallel
python scripts/trades_enrichment_parallel_download.py --ohlcv_tickers --year 2010 --spawn 12

# Monitor
python scripts/trades_enrichment_parallel_status.py --year 2010 --watch
    --watch: refresh every 5 seconds (live monitoring)

python scripts/trades_enrichment_parallel_status.py --year 2010 --kill
    --kill: kill all running processes (ps aux | grep trades_enrichment_download.py)
```

## Download Stock Quotes Data
```
python scripts/quotes_download.py --tickers NVDA --year 2025
```
