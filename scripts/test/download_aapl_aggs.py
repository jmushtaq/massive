import os
import sys
import csv
import datetime
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(env_path)

api_key = os.getenv("APIKEY")
if not api_key:
    raise ValueError("APIKEY not found in .env")

os.environ["MASSIVE_API_KEY"] = api_key

from massive import RESTClient
from massive.rest.models import Agg

client = RESTClient(trace=True)

ticker = "AAPL"
year = "2025"
from_date = f"{year}-01-01"
to_date = f"{year}-12-31"

aggs = []
for a in client.list_aggs(
    ticker,
    1,
    "minute",
    from_date,
    to_date,
    adjusted=True,
    limit=50000,
):
    aggs.append(a)

out_dir = Path("data") / "SPY" / "1min" / ticker / year
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"{ticker}_1min.csv"

headers = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "transactions",
    "otc",
]

count = 0
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    for agg in aggs:
        if isinstance(agg, Agg) and isinstance(agg.timestamp, int):
            writer.writerow({
                "timestamp": datetime.datetime.fromtimestamp(agg.timestamp / 1000).isoformat(),
                "open": agg.open,
                "high": agg.high,
                "low": agg.low,
                "close": agg.close,
                "volume": agg.volume,
                "vwap": agg.vwap,
                "transactions": agg.transactions,
                "otc": agg.otc,
            })
            count += 1

print(f"Downloaded {count} aggregates -> {out_path}")
