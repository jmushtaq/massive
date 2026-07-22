"""
Download NBBO quote data for a list of tickers from the Massive REST API and
compute per-minute enriched quote aggregates (spreads, bid/ask stats, quote
imbalance, microprice, etc.) saved to CSV/Parquet files.

Processes one trading day at a time — only one day's quotes are held in
memory at once, keeping memory utilisation low.

save tickers from dir
{ echo "ticker"; ls data/quotes/1min/2024/processing/ | cut -d'_' -f1; } > /tmp/processing_2024_tickers.csv


Usage:
    python scripts/quotes_download.py --tickers AAPL,NVDA --year 2025
    python scripts/quotes_download.py --tickers AAPL --year 2025 --resume
    python scripts/quotes_download.py --tickers AAPL --year 2025 --parquet
    python scripts/quotes_download.py --tickers AAPL,NVDA --year 2025 --aggregate 1H

To Resume from failed last date in csv file:
    python scripts/quotes_download.py --tickers AMZN --year 2025 --aggregate 1min --start_date 2025-12-03 &

Output layout:
    data/quotes/<aggregate>/<year>/<ticker>_<year>_<aggregate>_quotes.csv
    data/quotes/<aggregate>/<year>/<ticker>_<year>_<aggregate>_quotes.parquet  (with --parquet)

Features per aggregate window:
  - Average/Max/Min Spread  (ask_price - bid_price)
  - Average Bid / Average Ask
  - Quote Imbalance  ((bid_size - ask_size) / (bid_size + ask_size))
  - Bid Size / Ask Size
  - Spread Volatility  (stddev of spreads within window)
  - Mid Price  ((bid_price + ask_price) / 2)
  - Microprice  ((bid_price * ask_size + ask_price * bid_size) / (bid_size + ask_size))

One of --tickers or --tickers_file must be specified.
"""

import argparse
import csv
import datetime
import json
import logging
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

api_key = os.getenv("APIKEY")
if not api_key:
    raise ValueError("APIKEY not found in .env")
os.environ["MASSIVE_API_KEY"] = api_key

from massive import RESTClient

SCRIPT_NAME = Path(__file__).resolve().stem

AGGREGATE_MAP = {
    "1min": (1, "minute", "1min"),
    "5min": (5, "minute", "5min"),
    "15min": (15, "minute", "15min"),
    "1H": (1, "hour", "1H"),
    "4H": (4, "hour", "4H"),
    "1D": (1, "day", "1D"),
}

CSV_HEADERS = [
    "ticker",
    "timestamp",
    "quote_count",
    "avg_spread",
    "max_spread",
    "min_spread",
    "avg_bid",
    "avg_ask",
    "quote_imbalance",
    "bid_size",
    "ask_size",
    "spread_volatility",
    "mid_price",
    "microprice",
]

NANOS_PER_MINUTE = 60_000_000_000


def clean_ticker(raw: str) -> str:
    return raw.strip().upper().split("-")[0]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download NBBO quote data and compute enriched aggregates from Massive API"
    )
    parser.add_argument(
        "--aggregate",
        choices=list(AGGREGATE_MAP.keys()),
        default="1min",
        help="Aggregate window size (default: 1min)",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        help="Comma-separated list of ticker symbols (e.g. AAPL,TSLA,NVDA)",
    )
    parser.add_argument(
        "--tickers_file",
        type=str,
        help="Path to CSV with ticker list (one ticker per row, header 'ticker')",
    )
    parser.add_argument(
        "--year",
        type=str,
        default=str(datetime.date.today().year),
        help="Year to download (default: current year)",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default=None,
        help="Start from this date instead of Jan 1 (YYYY-MM-DD). Useful for resuming partial downloads.",
    )
    parser.add_argument(
        "--smart_resume",
        action="store_true",
        default=False,
        help="Read each ticker's processing file and start from the day after its last row. Implies --resume semantics.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip tickers that already have a non-empty output file",
    )
    parser.add_argument(
        "--parquet",
        action="store_true",
        help="Write output as Parquet instead of CSV",
    )
    parser.add_argument(
        "--logs",
        action="store_true",
        default=False,
        help="Save detailed per-ticker log files (default: False)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Seconds to sleep after each trading day's fetch (default: 0.1)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Base output directory (default: data/). Inferred aggregate/year subdirs are appended.",
    )
    return parser.parse_args()


def load_tickers(args) -> list[str]:
    tickers = []
    if args.tickers:
        tickers.extend(clean_ticker(t) for t in args.tickers.split(",") if t.strip())
    if args.tickers_file:
        with open(args.tickers_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                t = row.get("ticker", "").strip()
                if t:
                    tickers.append(clean_ticker(t))
    if not tickers:
        raise SystemExit("Error: specify at least one of --tickers or --tickers_file")
    return tickers


def output_path(ticker: str, year: str, agg: str, parquet: bool = False, subdir: str | None = None, output_dir: str | None = None) -> Path:
    folder = AGGREGATE_MAP[agg][2]
    ext = "parquet" if parquet else "csv"
    base = (Path(output_dir) if output_dir else Path("data")) / "quotes" / folder / year
    if subdir:
        base = base / subdir
    return base / f"{ticker}_{year}_{folder}_quotes.{ext}"


def is_ticker_complete(ticker: str, year: str, agg: str, parquet: bool = False, output_dir: str | None = None) -> bool:
    path = output_path(ticker, year, agg, parquet, output_dir=output_dir)
    if not path.exists() or path.stat().st_size == 0:
        return False
    if parquet:
        reader = pq.ParquetFile(path)
        return reader.metadata.num_rows > 0
    with open(path) as f:
        return sum(1 for _ in f) > 1


def last_row_date(ticker: str, year: str, agg: str, output_dir: str | None = None) -> str | None:
    path = output_path(ticker, year, agg, subdir="processing", output_dir=output_dir)
    if not path.exists() or path.stat().st_size == 0:
        return None
    with open(path) as f:
        last_line = None
        for line in f:
            line = line.strip()
            if line:
                last_line = line
    if not last_line:
        return None
    parts = last_line.split(",")
    if len(parts) < 2:
        return None
    try:
        ts = datetime.datetime.fromisoformat(parts[1])
        return ts.date().isoformat()
    except (ValueError, IndexError):
        return None


def trading_days(year: str):
    start = datetime.date(int(year), 1, 1)
    end = datetime.date(int(year), 12, 31)
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += datetime.timedelta(days=1)


def fmt_bytes(size: int) -> str:
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}GB"


def microprice(bid_price: float, ask_price: float, bid_size: float, ask_size: float) -> float:
    total = bid_size + ask_size
    if total == 0:
        return (bid_price + ask_price) / 2.0
    return (bid_price * ask_size + ask_price * bid_size) / total


def _append_rows(path: Path, rows: list[dict], parquet: bool) -> None:
    if parquet:
        existing = pq.read_table(path) if path.exists() else None
        new_table = pa.Table.from_pylist(rows)
        if existing is not None:
            table = pa.concat_tables([existing, new_table])
        else:
            table = new_table
        pq.write_table(table, path)
    else:
        write_header = not path.exists()
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)


def process_day(ticker: str, day: datetime.date, client, delay: float = 0.1) -> list[dict]:
    bucket: dict[int, list] = defaultdict(list)

    date_str = day.isoformat()
    try:
        for q in client.list_quotes(ticker, date_str, limit=50000):
            if (q.sip_timestamp is not None
                    and q.bid_price is not None
                    and q.ask_price is not None
                    and q.bid_size is not None
                    and q.ask_size is not None):
                bucket[q.sip_timestamp // NANOS_PER_MINUTE].append(
                    (q.sip_timestamp, q.bid_price, q.ask_price, q.bid_size, q.ask_size)
                )
    except Exception:
        return []
    time.sleep(delay)

    if not bucket:
        return []

    day_rows = []

    for bucket_key in sorted(bucket):
        b = bucket[bucket_key]
        b.sort(key=lambda x: x[0])

        ts_ns = bucket_key * NANOS_PER_MINUTE
        ts_sec = ts_ns / 1_000_000_000
        ts_iso = datetime.datetime.fromtimestamp(ts_sec, tz=datetime.timezone.utc).isoformat()

        spreads = [x[2] - x[1] for x in b]
        bid_prices = [x[1] for x in b]
        ask_prices = [x[2] for x in b]
        bid_sizes = [x[3] for x in b]
        ask_sizes = [x[4] for x in b]

        quote_count = len(b)

        avg_spread = statistics.mean(spreads) if spreads else 0.0
        max_spread = max(spreads) if spreads else 0.0
        min_spread = min(spreads) if spreads else 0.0
        avg_bid = statistics.mean(bid_prices) if bid_prices else 0.0
        avg_ask = statistics.mean(ask_prices) if ask_prices else 0.0

        total_bid_size = sum(bid_sizes)
        total_ask_size = sum(ask_sizes)

        imbalance_num = total_bid_size - total_ask_size
        imbalance_den = total_bid_size + total_ask_size
        quote_imbalance = imbalance_num / imbalance_den if imbalance_den > 0 else 0.0

        spread_vol = statistics.stdev(spreads) if len(spreads) > 1 else 0.0

        mid = (avg_bid + avg_ask) / 2.0

        mp = microprice(
            sum(bid_prices) / len(bid_prices) if bid_prices else 0.0,
            sum(ask_prices) / len(ask_prices) if ask_prices else 0.0,
            total_bid_size,
            total_ask_size,
        )

        day_rows.append({
            "ticker": ticker,
            "timestamp": ts_iso,
            "quote_count": quote_count,
            "avg_spread": round(avg_spread, 6),
            "max_spread": round(max_spread, 6),
            "min_spread": round(min_spread, 6),
            "avg_bid": round(avg_bid, 4),
            "avg_ask": round(avg_ask, 4),
            "quote_imbalance": round(quote_imbalance, 6),
            "bid_size": total_bid_size,
            "ask_size": total_ask_size,
            "spread_volatility": round(spread_vol, 6),
            "mid_price": round(mid, 4),
            "microprice": round(mp, 4),
        })

    return day_rows


def rollup_rows(rows: list[dict], multiplier: int, timespan: str) -> list[dict]:
    if multiplier == 1 and timespan == "minute":
        return rows
    window_ns = multiplier * NANOS_PER_MINUTE
    if timespan == "hour":
        window_ns = multiplier * 60 * NANOS_PER_MINUTE
    elif timespan == "day":
        window_ns = multiplier * 24 * 60 * NANOS_PER_MINUTE

    groups: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        ts_ns = int(datetime.datetime.fromisoformat(r["timestamp"]).timestamp() * 1_000_000_000)
        groups[ts_ns // window_ns].append(r)

    rolled = []
    for bucket_key in sorted(groups):
        g = groups[bucket_key]
        ts_ns = bucket_key * window_ns
        ts_sec = ts_ns / 1_000_000_000
        ts_iso = datetime.datetime.fromtimestamp(ts_sec, tz=datetime.timezone.utc).isoformat()

        total_quotes = sum(r["quote_count"] for r in g)
        avg_spread = sum(r["avg_spread"] * r["quote_count"] for r in g) / total_quotes if total_quotes > 0 else 0.0
        max_spread = max(r["max_spread"] for r in g)
        min_spread = min(r["min_spread"] for r in g)
        avg_bid = sum(r["avg_bid"] * r["quote_count"] for r in g) / total_quotes if total_quotes > 0 else 0.0
        avg_ask = sum(r["avg_ask"] * r["quote_count"] for r in g) / total_quotes if total_quotes > 0 else 0.0
        bid_size = sum(r["bid_size"] for r in g)
        ask_size = sum(r["ask_size"] for r in g)
        imbalance_num = bid_size - ask_size
        imbalance_den = bid_size + ask_size
        quote_imbalance = imbalance_num / imbalance_den if imbalance_den > 0 else 0.0
        spread_vol = statistics.stdev([r["avg_spread"] for r in g]) if len(g) > 1 else max(r["spread_volatility"] for r in g)
        mid = (avg_bid + avg_ask) / 2.0
        mp = sum(r["microprice"] * (r["bid_size"] + r["ask_size"]) for r in g) / (bid_size + ask_size) if (bid_size + ask_size) > 0 else 0.0

        rolled.append({
            "ticker": g[0]["ticker"],
            "timestamp": ts_iso,
            "quote_count": total_quotes,
            "avg_spread": round(avg_spread, 6),
            "max_spread": round(max_spread, 6),
            "min_spread": round(min_spread, 6),
            "avg_bid": round(avg_bid, 4),
            "avg_ask": round(avg_ask, 4),
            "quote_imbalance": round(quote_imbalance, 6),
            "bid_size": bid_size,
            "ask_size": ask_size,
            "spread_volatility": round(spread_vol, 6),
            "mid_price": round(mid, 4),
            "microprice": round(mp, 4),
        })

    return rolled


def process_ticker(ticker: str, year: str, agg: str, parquet: bool, client, delay: float = 0.1, start_date: str | None = None, output_dir: str | None = None) -> int:
    multiplier, timespan, _ = AGGREGATE_MAP[agg]
    trade_dates = list(trading_days(year))
    if start_date:
        start = datetime.date.fromisoformat(start_date)
        trade_dates = [d for d in trade_dates if d >= start]
    total_rows = 0

    out = output_path(ticker, year, agg, parquet, subdir="processing", output_dir=output_dir)
    out.parent.mkdir(parents=True, exist_ok=True)

    for d in trade_dates:
        day_rows = process_day(ticker, d, client, delay=delay)
        if day_rows:
            rolled = rollup_rows(day_rows, multiplier, timespan)
            _append_rows(out, rolled, parquet)
            total_rows += len(rolled)

    return total_rows


def main():
    args = parse_args()
    overall_start = time.time()
    tickers = load_tickers(args)
    year = args.year
    agg = args.aggregate
    parquet = args.parquet
    folder = AGGREGATE_MAP[agg][2]

    output_dir = args.output
    output_base = (Path(output_dir) if output_dir else Path("data")) / "quotes" / folder
    log_ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(logging.INFO)
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    stream_handler.setFormatter(log_formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(stream_handler)
    logger = logging.getLogger(SCRIPT_NAME)

    client = RESTClient(trace=False)

    missing: list[str] = []
    results: list[dict] = []
    downloaded = 0
    log_fh = None

    for i, ticker in enumerate(tickers, 1):
        if args.logs:
            log_dir = Path("data") / "quotes" / folder / year / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{SCRIPT_NAME}_{ticker}.log"
            if log_fh:
                log_fh.close()
            for h in list(logger.handlers):
                if isinstance(h, logging.FileHandler):
                    h.close()
                    logger.removeHandler(h)
            log_fh = open(log_path, "w")
            fh = logging.FileHandler(log_path)
            fh.setFormatter(log_formatter)
            logger.addHandler(fh)
            logger.info("Logging to %s", log_path)

        if args.smart_resume:
            last_date = last_row_date(ticker, year, agg, output_dir)
            if last_date:
                ticker_start = last_date
                logger.info("[%d/%d] %s -> smart-resuming from %s", i, len(tickers), ticker, ticker_start)
            else:
                logger.info("[%d/%d] %s -> no partial data found, starting from Jan 1", i, len(tickers), ticker)
                ticker_start = None
        else:
            ticker_start = args.start_date

        if args.resume and is_ticker_complete(ticker, year, agg, parquet, output_dir):
            logger.info("[%d/%d] %s -> already complete, skipping", i, len(tickers), ticker)
            results.append({"ticker": ticker, "status": "skipped"})
            continue

        logger.info("[%d/%d] %s -> downloading quotes ...", i, len(tickers), ticker)
        t0 = time.time()

        try:
            total_rows = process_ticker(ticker, year, agg, parquet, client, delay=args.delay, start_date=ticker_start, output_dir=output_dir)
        except Exception as e:
            elapsed = time.time() - t0
            logger.error("[%d/%d] %s -> FAILED after %.1fs: %s", i, len(tickers), ticker, elapsed, e)
            missing.append(ticker)
            results.append({"ticker": ticker, "status": "failed", "error": str(e), "elapsed_s": round(elapsed, 1)})
            proc_path = output_path(ticker, year, agg, parquet, subdir="processing", output_dir=output_dir)
            if proc_path.exists():
                err_dir = output_path(ticker, year, agg, parquet, subdir="errors", output_dir=output_dir).parent
                err_dir.mkdir(parents=True, exist_ok=True)
                proc_path.rename(err_dir / proc_path.name)
            continue

        elapsed = time.time() - t0

        if total_rows == 0:
            logger.warning("[%d/%d] %s -> no quote data (%.1fs)", i, len(tickers), ticker, elapsed)
            missing.append(ticker)
            results.append({"ticker": ticker, "status": "no_data", "elapsed_s": round(elapsed, 1)})
            proc_path = output_path(ticker, year, agg, parquet, subdir="processing", output_dir=output_dir)
            if proc_path.exists():
                err_dir = output_path(ticker, year, agg, parquet, subdir="errors", output_dir=output_dir).parent
                err_dir.mkdir(parents=True, exist_ok=True)
                proc_path.rename(err_dir / proc_path.name)
            continue

        out = output_path(ticker, year, agg, parquet, output_dir=output_dir)
        proc_path = output_path(ticker, year, agg, parquet, subdir="processing", output_dir=output_dir)
        out.parent.mkdir(parents=True, exist_ok=True)
        proc_path.rename(out)
        size = out.stat().st_size
        logger.info(
            "[%d/%d] %s -> %d %s bars (%s, %.1fs) -> %s",
            i, len(tickers), ticker, total_rows, agg, fmt_bytes(size), elapsed, out,
        )
        downloaded += 1
        results.append({
            "ticker": ticker, "status": "ok", "rows": total_rows,
            "size_bytes": size, "path": str(out), "elapsed_s": round(elapsed, 1),
        })

    total_time = time.time() - overall_start

    summary = {
        "script": SCRIPT_NAME,
        "timestamp": log_ts,
        "year": year,
        "aggregate": agg,
        "format": "parquet" if parquet else "csv",
        "total_tickers": len(tickers),
        "downloaded": downloaded,
        "missing": sorted(set(missing)),
        "missing_count": len(set(missing)),
        "duration_s": round(total_time, 1),
        "results": results,
    }

    if log_fh:
        log_fh.close()

    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler):
            logger.removeHandler(h)

    logger.info("=" * 60)
    logger.info("SUMMARY REPORT")
    logger.info("  Aggregate:      %s", agg)
    logger.info("  Duration:       %.1fs", total_time)
    logger.info("  Downloaded:     %d", downloaded)
    no_data_count = len([r for r in results if r["status"] == "no_data"])
    if no_data_count:
        logger.info("  No data:        %d", no_data_count)
    skips = len([r for r in results if r["status"] == "skipped"])
    if skips:
        logger.info("  Skipped:        %d", skips)
    logger.info("  Missing/failed: %d", len(set(missing)))
    missing_unique = sorted(set(missing))
    if missing_unique:
        logger.info("  Missing tickers: %s", ", ".join(missing_unique))

    if len(tickers) == 1:
        status = "no_data" if results and results[0]["status"] == "no_data" else \
                 "skipped" if results and results[0]["status"] == "skipped" else \
                 "failed" if results and results[0]["status"] == "failed" else \
                 "ok" if results else "unknown"
    else:
        status = "ok" if downloaded > 0 else "no_data" if any(r["status"] == "no_data" for r in results) else "unknown"
    print("PARALLEL_RESULT:{\"status\": \"%s\", \"downloaded\": %d}" % (status, downloaded), flush=True)


if __name__ == "__main__":
    main()
