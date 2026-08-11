# massive

# Create list of tickers from filenames in folder
```
{ echo "ticker"; ls data/quotes/1min/2024/processing/ | cut -d'_' -f1; } > /tmp/processing_2024_tickers.csv
{ echo "ticker"; ls data/combined/trades/1min/2024/processing/ | cut -d'_' -f1; } > /tmp/processing_2024_tickers.csv
```

## Download Stock OHLCV Data
```
python scripts/stocks_aggs_download.py --tickers AAPL --year 2010 --aggregate 1sec &
python scripts/quotes_download.py --tickers AAPL --year 2010 --aggregate 1sec &
python scripts/trades_enrichment_download.py --tickers AAPL --year 2010 --aggregate 1sec &

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
python scripts/quotes_parallel_download.py --ohlcv_tickers --year 2025 --spawn 100 --logs --delay 1.1

python scripts/quotes_download.py --tickers NVDA --year 2025
```

## Resuming execution
```
# collect together all missing tickers
python scripts/find_missing_tickers.py --reference data/SPY/1min/2022 --target data/quotes/1min/2022 --output /tmp/missing_2022_tickers.txt
Wrote 248 missing tickers to /tmp/missing_2022_tickers.txt

python scripts/find_missing_tickers.py --reference data/SPY/1min/2025 --target data/SPY/1sec/2025 --output /tmp/missing_2025_tickers.csv

# clear the state
rm data/quotes/.parallel_state_2022_1min.json

# Run tickers (re-running from where scripts died/failed)
python scripts/quotes_parallel_download.py --tickers_file /tmp/missing_2022_tickers.txt --year 2022 --spawn 100 --smart_resume --resume &

# Monitor

python scripts/quotes_parallel_status.py --year 2022 --watch

tail -f data/quotes/1min/2022/processing/AAPL_2022_1min_quotes.csv
```


# combined unique
```
# Stocks
python scripts/stocks_aggs_parallel_download.py --tickers_file data/universes/2008/combined_unique.csv --year 2008 --output data/combined --spawn 100 &

python scripts/stocks_aggs_parallel_status.py --year 2008 --output data/combined --watch


# Trades
python scripts/trades_enrichment_parallel_download.py --tickers_file data/universes/2025/combined_unique.csv  --year 2025 --output data/combined --spawn 100 --check data/quotes --delay 1.1 &

{ echo "ticker"; ls data/combined/trades/1min/2024/processing/ | cut -d'_' -f1; } > /tmp/processing_2024_tickers.csv
python scripts/trades_enrichment_parallel_download.py --tickers_file /tmp/processing_2025_tickers.csv  --year 2025 --output data/combined --spawn 30 --smart_resume --resume --check data/trades --delay 1.1 &

python scripts/trades_enrichment_parallel_status.py --year 2025 --output data/combined --watch

tail -f data/combined/trades/1min/2025/processing/XYZ_2025_1min_trades.csv
```

# 1sec aggregate stocks
```
{ echo "ticker"; ls data/SPY/1min/2025 | cut -d'_' -f1; } > /tmp/processing_2025_tickers.csv
python scripts/stocks_aggs_parallel_download.py --tickers_file /tmp/processing_2025_tickers.csv --year 2025 --spawn 50 --aggregate 1sec &


python scripts/stocks_aggs_parallel_status.py --year 2025 --watch


ll data/SPY/1sec/2025/processing/
ll data/SPY/1sec/2025/01


# files smaller than 12.5MB
{ echo "ticker"; find data/quotes/1min/2025/ -maxdepth 1 -name "*.csv" -size -12500000c -printf "%f\n" | cut -d'_' -f1 | sort; } > /tmp/subset_2025_tickers.csv

# files smaller than 6.1MB
{ echo "ticker"; find data/SPY/1min/2025/ -maxdepth 1 -name "*.csv" -size -6100000c -printf "%f\n" | cut -d'_' -f1 | sort; } | wc -l
234

# files smaller than 6.15MB
{ echo "ticker"; find data/SPY/1min/2025/ -maxdepth 1 -name "*.csv" -size -6150000c -printf "%f\n" | cut -d'_' -f1 | sort; } | wc -l
247


python scripts/stocks_aggs_parallel_download.py --tickers_file /tmp/missing_2020_tickers.csv --year 2020 --spawn 40 --aggregate 1sec &
python scripts/stocks_aggs_parallel_status.py --year 2020 --watch

python scripts/trades_enrichment_parallel_download.py --tickers_file /tmp/subset_2025_tickers.csv --year 2025 --spawn 40 --aggregate 1sec &
python scripts/trades_enrichment_parallel_status.py --year 2025 --watch

$ use the same file
python scripts/quotes_parallel_download.py --tickers_file /tmp/subset_2025_tickers.csv --year 2025 --spawn 40 --aggregate 1sec &
python scripts/quotes_parallel_download.py --tickers_file /tmp/subset_2025_tickers.csv --year 2024 --spawn 40 --aggregate 1sec &
python scripts/quotes_parallel_download.py --tickers_file /tmp/subset_2025_tickers.csv --year 2023 --spawn 40 --aggregate 1sec &
python scripts/quotes_parallel_status.py --year 2025 --watch

```

```
python scripts/quotes_parallel_download.py --tickers_file data/universes/etf_tickers.csv --year 2025 --spawn 22  --output data/etf &

python scripts/trades_enrichment_parallel_download.py --tickers_file data/universes/etf_tickers.csv --year 2025 --spawn 22  --output data/etf &


python scripts/quotes_parallel_download.py --tickers_file data/universes/etf_tickers.csv --year 2025 --spawn 22  --output data/etf &
python scripts/quotes_parallel_download.py --tickers_file data/universes/etf_tickers.csv --year 2024 --spawn 22  --output data/etf &
python scripts/quotes_parallel_download.py --tickers_file data/universes/etf_tickers.csv --year 2023 --spawn 22  --output data/etf &
python scripts/quotes_parallel_download.py --tickers_file data/universes/etf_tickers.csv --year 2022 --spawn 22  --output data/etf &
python scripts/quotes_parallel_download.py --tickers_file data/universes/etf_tickers.csv --year 2021 --spawn 22  --output data/etf &
python scripts/quotes_parallel_download.py --tickers_file data/universes/etf_tickers.csv --year 2020 --spawn 22  --output data/etf &
python scripts/quotes_parallel_download.py --tickers_file data/universes/etf_tickers.csv --year 2019 --spawn 22  --output data/etf &

ps aux | grep download | wc -l
ll data/etf/quotes/1min/2025/


python scripts/quotes_parallel_download.py --tickers_file /tmp/missing_sub_2020_tickers.txt --year 2020 --spawn 40 --aggregate 1sec --delay 1.0 &
python scripts/quotes_parallel_download.py --tickers_file /tmp/missing_sub_2021_tickers.txt --year 2021 --spawn 40 --aggregate 1sec --delay 1.0 &

python scripts/trades_enrichment_parallel_download.py --tickers_file data/universes/subset_2025_tickers.csv --year 2022 --spawn 40 --aggregate 1sec --delay 1.0 &
python scripts/trades_enrichment_parallel_download.py --tickers_file data/universes/subset_2025_tickers.csv --year 2021 --spawn 40 --aggregate 1sec --delay 1.0 &
  

python scripts/trades_enrichment_parallel_download.py --tickers_file data/universes/etf_tickers.csv --year 2025 --spawn 40  --output data/etf &
python scripts/trades_enrichment_parallel_download.py --tickers_file /tmp/missing_etf_2025_tickers.txt --year 2025 --spawn 40  --output data/etf --smart_resume --resume &
``


# Stocks Options
```
python scripts/options/stock_options_from_flatfiles_download.py --tickers UPS --year 2025 --aggregate 1min &
python scripts/options/stock_options_from_flatfiles_download.py --tickers UPS --year 2025 --aggregate 1min --smart_resume --resume &

python scripts/options/stock_options_from_flatfiles_parallel_download.py --ohlcv_tickers --year 2025 --spawn 100 --aggregate 1min &
python scripts/options/stock_options_from_flatfiles_parallel_download.py --ohlcv_tickers --year 2025 --spawn 100 --aggregate 1min --smart_resume --resume &
```



# Default: 20-day batches with concurrent download
python scripts/options/stock_options_from_flatfiles_parallel_download.py --ohlcv_tickers --year 2025 --spawn 100 --aggregate 1min &
                                                                                                                                                                                        
# Larger batches: 60 days each (fewer round trips) 
python scripts/options/stock_options_from_flatfiles_parallel_download.py --ohlcv_tickers --year 2025 --spawn 100 --pre_download 60 & 
                                                                                                                                                                                        
# All-at-once (original behavior): 261 days downloaded first
python scripts/options/stock_options_from_flatfiles_parallel_download.py --ohlcv_tickers --year 2025 --spawn 100 --pre_download 0 &

---
python scripts/options/stock_options_from_flatfiles_parallel_download.py --ohlcv_tickers --year 2025 --spawn 100 --resume --smart_resume
python scripts/options/stock_options_from_flatfiles_parallel_status.py --year 2025 --watch
ll tmp/options_cache_2025/
ps aux | grep _download | wc -l
ps aux | grep aws | more

-----

aws configure set aws_access_key_id bc4815fc-fe98-40a6-aeb4-af6adf27d0e6
aws configure set aws_secret_access_key MA2RPkwqWuSYxke1mP1ECpWp4G4l263e
aws s3 ls s3://flatfiles/us_options_opra/ --endpoint-url https://files.massive.com
                           PRE day_aggs_v1/
                           PRE minute_aggs_v1/
                           PRE quotes_v1/
                           PRE trades_v1/


# Download all 2025 flat files to cache (100 parallel downloads)
python scripts/options/stock_options_from_flatfiles_parallel_download.py --download_only --year 2025 --spawn 100
                                                                                                                                                                                        
# Resume: re-runs only missing files (cached ones skipped) 
python scripts/options/stock_options_from_flatfiles_parallel_download.py --download_only --year 2025 --spawn 100
                                                                                                                                                                                        
# Then process with the cache already populated 
python scripts/options/stock_options_from_flatfiles_parallel_download.py --ohlcv_tickers --year 2025 --spawn 100 --smart_resume --resume &

python scripts/options/stock_options_from_flatfiles_parallel_download.py --ohlcv_tickers --year 2024 --spawn 100 --use_local_cache &
python scripts/options/stock_options_from_flatfiles_parallel_download.py --tickers_file /tmp/opt_2024_tickers.csv --year 2024 --spawn 16 --use_local_cache --smart_resume --resume &


gunzip -k tmp/options_cache_2024/*.csv.gz
gzip tmp/options_cache_2024/*.csv 

cd data/trades/1sec
gzip 2018/*.csv && gzip 2019/*.csv && gzip 2020/*.csv && gzip 2021/*.csv && gzip 2022/*.csv

cd data/quotes/1sec
gzip 2018/*.csv && gzip 2019/*.csv && gzip 2020/*.csv && gzip 2021/*.csv && gzip 2022/*.csv

cd data/SPY/1sec
gzip 2018/*.csv && gzip 2019/*.csv && gzip 2020/*.csv && gzip 2021/*.csv && gzip 2022/*.csv

cd data/SPY/1min
gzip 2003/*.csv && gzip 2004/*.csv && gzip 2005/*.csv && gzip 2006/*.csv && gzip 2007/*.csv && gzip 2008/*.csv && gzip 2009/*.csv && gzip 2010/*.csv && gzip 2011/*.csv && gzip 2012/*.csv && gzip 2013/*.csv && gzip 2014/*.csv && gzip 2015/*.csv && gzip 2016/*.csv && gzip 2017/*.csv && gzip 2018/*.csv && gzip 2019/*.csv && gzip 2020/*.csv &&

cd data/trades/1min
gzip 2003/*.csv && gzip 2004/*.csv && gzip 2005/*.csv && gzip 2006/*.csv && gzip 2007/*.csv && gzip 2008/*.csv && gzip 2009/*.csv && gzip 2010/*.csv && gzip 2011/*.csv && gzip 2012/*.csv && gzip 2013/*.csv && gzip 2014/*.csv && gzip 2015/*.csv && gzip 2016/*.csv && gzip 2017/*.csv && gzip 2018/*.csv && gzip 2019/*.csv && gzip 2020/*.csv &&

cd data/quotes/1min
gzip 2003/*.csv && gzip 2004/*.csv && gzip 2005/*.csv && gzip 2006/*.csv && gzip 2007/*.csv && gzip 2008/*.csv && gzip 2009/*.csv && gzip 2010/*.csv && gzip 2011/*.csv && gzip 2012/*.csv && gzip 2013/*.csv && gzip 2014/*.csv && gzip 2015/*.csv && gzip 2016/*.csv && gzip 2017/*.csv && gzip 2018/*.csv && gzip 2019/*.csv && gzip 2020/*.csv &&


python scripts/options/stock_options_from_flatfiles_parallel_download.py --tickers_file data/universes/2025/combined_unique.csv --year 2025 --spawn 16 --use_local_cache --output data/combined &




----
python scripts/options/options_chain_discover.py --ohlcv_tickers --year 2025
python scripts/options/stock_options_from_flatfiles_parallel_download.py --download_only --year 2025 --spawn 100
python scripts/options/stock_options_from_flatfiles_parallel_download.py --download_only --year 2025 --spawn 100 --aggregate 1sec
python scripts/options/options_chain_parallel_status.py --year 2025 --watch
ll data/options/chains/2025/

python scripts/options/options_chain_download.py --tickers CLF --year 2025 --aggregate 1sec --delay 0.1 --no_rename

------
# Split Adjust the stock Options
python scripts/options/adjust_options_for_splits.py --year 2014
ll data/options/stocks/1min/2025/split-unadjusted/


-----------
Treasury Yields
python scripts/options/download_treasury_yields.py

Greeks
python scripts/options/compute_chain_greeks_parallel_download.py --tickers AAPL --year 2025 --aggregate 1sec --spawn 1
python scripts/options/compute_chain_greeks_parallel_download.py --ohlcv_tickers --year 2025 --aggregate 1sec --spawn 16
python scripts/options/compute_chain_greeks_parallel_status.py --year 2025 --aggregate --watch
     - Input: data/options/chains/<agg>/<year>/<ticker>_<year>_<agg>_chains.csv
     - Output: ..._chains_greeks.csv (13 cols: ticker, timestamp, contract_symbol, strike, call_put, expiry, DTE, iv, delta, gamma, theta, vega, rho)
     - Per-contract IV via Newton-Raphson (20 iterations), risk-free rate interpolated from yield curve

python scripts/options/compute_chain_greeks_parallel_download.py --ohlcv_tickers --year 2022 --spawn 24 --aggregate 1sec --exclude_tickers data/universes/excluded_tickers.txt &
python scripts/options/compute_chain_greeks_parallel_status.py --year 2022 --aggregate 1sec --watch

python scripts/options/compute_stocks_greeks_parallel_download.py --ohlcv_tickers --year 2025 --aggregate 1sec --spawn 16
python scripts/options/compute_stocks_greeks_parallel_status.py --year 2025 --aggregate 1sec --watch
     - Input: data/options/stocks/<agg>/<year>/<ticker>_<year>_<agg>_options.csv
     - Output: ..._options_greeks.csv (26 cols: ticker, timestamp, 6 greeks × 4 contracts — atm_call, atm_put, iv30d_call, iv30d_put)
     - Skips rows with no time premium (close ≤ intrinsic)

python scripts/options/compute_stocks_greeks_parallel_download.py --ohlcv_tickers --year 2025 --spawn 24 --aggregate 1min --exclude_tickers data/universes/excluded_tickers.txt &
python scripts/options/compute_stocks_greeks_parallel_status.py --year 2025 --aggregate 1min --watch

----
# VIX
python scripts/stocks_aggs_download.py --tickers "I:VIX,I:VXN,I:VVIX,I:RVX" --year 2025 --aggregate 1min --output data/vix --UTC
python scripts/stocks_aggs_download.py --tickers_file data/universes/idx_tickers.txt --year 2025 --aggregate 1min --output data/vix --UTC
ll data/vix/1min/2025


