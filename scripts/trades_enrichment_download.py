"""
Download trade data for a list of tickers from the Massive REST API and
compute per-minute enriched aggregates (trade count, volume, VWAP, buy/sell
flows, deltas, etc.) saved to CSV/Parquet files. No look-ahead bias.

Processes one trading day at a time — only one day's trades are held in
memory at once, keeping memory utilisation low.

Usage:
    python scripts/trades_enrichment_download.py --tickers AAPL,NVDA --year 2025
    python scripts/trades_enrichment_download.py --tickers AAPL --year 2025 --resume
    python scripts/trades_enrichment_download.py --tickers AAPL --year 2025 --parquet
    python scripts/trades_enrichment_download.py --tickers AAPL,NVDA --year 2025 --aggregate 1H

Output layout:
    data/trades/<aggregate>/<year>/<ticker>_<year>_<aggregate>_trades.csv
    data/trades/<aggregate>/<year>/<ticker>_<year>_<aggregate>_trades.parquet  (with --parquet)

Each day's raw trades are enriched into 1-minute bars (the finest granularity),
then rolled up to the requested --aggregate window. No look-ahead bias:
  - Only data up to and including the current minute is used.
  - Tick-rule: trade classified using the PRIOR trade's price, not the next.
  - Large trade threshold computed per-minute using that minute's stats.
  - Cumulative delta accumulates sequentially across all minutes.

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
    "trade_count",
    "volume",
    "vwap",
    "avg_trade_size",
    "median_trade_size",
    "largest_trade",
    "stddev_trade_size",
    "buy_volume",
    "sell_volume",
    "delta",
    "cumulative_delta",
    "delta_pct",
    "aggression_ratio",
    "trade_frequency",
    "large_trade_count",
    "large_trade_ratio",
    "avg_seconds_between_trades",
]

NANOS_PER_MINUTE = 60_000_000_000


def clean_ticker(raw: str) -> str:
    return raw.strip().upper().split("-")[0]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download trade data and compute enriched aggregates from Massive API"
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


def output_path(ticker: str, year: str, agg: str, parquet: bool = False, subdir: str | None = None) -> Path:
    folder = AGGREGATE_MAP[agg][2]
    ext = "parquet" if parquet else "csv"
    base = Path("data") / "trades" / folder / year
    if subdir:
        base = base / subdir
    return base / f"{ticker}_{year}_{folder}_trades.{ext}"


def is_ticker_complete(ticker: str, year: str, agg: str, parquet: bool = False) -> bool:
    path = output_path(ticker, year, agg, parquet)
    if not path.exists() or path.stat().st_size == 0:
        return False
    if parquet:
        reader = pq.ParquetFile(path)
        return reader.metadata.num_rows > 0
    with open(path) as f:
        return sum(1 for _ in f) > 1


def trading_days(year: str):
    start = datetime.date(int(year), 1, 1)
    end = datetime.date(int(year), 12, 31)
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += datetime.timedelta(days=1)


def classify_trade_side(price: float, prev_price: float | None) -> str:
    if prev_price is None:
        return "neutral"
    if price > prev_price:
        return "buy"
    elif price < prev_price:
        return "sell"
    return "neutral"


def fmt_bytes(size: int) -> str:
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}GB"


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


def process_day(ticker: str, day: datetime.date, client, cum_delta: float) -> tuple[list[dict], float]:
    bucket: dict[int, list] = defaultdict(list)

    date_str = day.isoformat()
    try:
        for t in client.list_trades(ticker, date_str, limit=50000):
            if t.sip_timestamp is not None and t.size is not None and t.price is not None:
                bucket[t.sip_timestamp // NANOS_PER_MINUTE].append(
                    (t.sip_timestamp, t.price, t.size)
                )
    except Exception:
        return [], cum_delta
    time.sleep(0.1)

    if not bucket:
        return [], cum_delta

    day_rows = []

    for bucket_key in sorted(bucket):
        b = bucket[bucket_key]
        b.sort(key=lambda x: x[0])

        ts_ns = bucket_key * NANOS_PER_MINUTE
        ts_sec = ts_ns / 1_000_000_000
        ts_iso = datetime.datetime.fromtimestamp(ts_sec, tz=datetime.timezone.utc).isoformat()

        sizes = [t[2] for t in b]
        prices = [t[1] for t in b]
        volumes = [p * s for p, s in zip(prices, sizes)]

        trade_count = len(b)
        total_volume = sum(sizes)
        vwap = sum(volumes) / total_volume if total_volume > 0 else 0.0
        avg_size = statistics.mean(sizes)
        median_size = statistics.median(sizes)
        max_size = max(sizes)
        std_size = statistics.stdev(sizes) if len(sizes) > 1 else 0.0

        buy_vol = 0.0
        sell_vol = 0.0
        for idx in range(trade_count):
            cls = classify_trade_side(prices[idx], prices[idx - 1] if idx > 0 else None)
            s = float(sizes[idx])
            if cls == "buy":
                buy_vol += s
            elif cls == "sell":
                sell_vol += s

        delta = buy_vol - sell_vol
        cum_delta += delta
        classified_vol = buy_vol + sell_vol
        delta_pct = (delta / classified_vol * 100) if classified_vol > 0 else 0.0
        aggression_ratio = buy_vol / classified_vol if classified_vol > 0 else 0.5

        if trade_count > 1:
            span_ns = b[-1][0] - b[0][0]
            trade_frequency = trade_count / (span_ns / 1_000_000_000) if span_ns > 0 else trade_count / 60.0
        else:
            trade_frequency = trade_count / 60.0

        large_threshold = avg_size + 2.0 * std_size if std_size > 0 else float("inf")
        large_count = sum(1 for s in sizes if s >= large_threshold)
        large_ratio = large_count / trade_count if trade_count > 0 else 0.0

        if trade_count > 1:
            diffs = [(b[i + 1][0] - b[i][0]) / 1_000_000_000.0 for i in range(trade_count - 1)]
            avg_seconds = statistics.mean(diffs) if diffs else 0.0
        else:
            avg_seconds = 0.0

        day_rows.append({
            "ticker": ticker,
            "timestamp": ts_iso,
            "trade_count": trade_count,
            "volume": total_volume,
            "vwap": round(vwap, 4),
            "avg_trade_size": round(avg_size, 2),
            "median_trade_size": float(median_size),
            "largest_trade": max_size,
            "stddev_trade_size": round(std_size, 2),
            "buy_volume": buy_vol,
            "sell_volume": sell_vol,
            "delta": delta,
            "cumulative_delta": cum_delta,
            "delta_pct": round(delta_pct, 2),
            "aggression_ratio": round(aggression_ratio, 4),
            "trade_frequency": round(trade_frequency, 4),
            "large_trade_count": large_count,
            "large_trade_ratio": round(large_ratio, 4),
            "avg_seconds_between_trades": round(avg_seconds, 2),
        })

    return day_rows, cum_delta


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
        total_trade_count = sum(r["trade_count"] for r in g)
        total_volume = sum(r["volume"] for r in g)
        vwap = sum(r["vwap"] * r["volume"] for r in g) / total_volume if total_volume > 0 else 0.0
        avg_trade_size = sum(r["avg_trade_size"] * r["trade_count"] for r in g) / total_trade_count if total_trade_count > 0 else 0.0
        largest_trade = max(r["largest_trade"] for r in g)
        buy_volume = sum(r["buy_volume"] for r in g)
        sell_volume = sum(r["sell_volume"] for r in g)
        delta = sum(r["delta"] for r in g)
        cum_delta = g[-1]["cumulative_delta"]
        classified_vol = buy_volume + sell_volume
        delta_pct = (delta / classified_vol * 100) if classified_vol > 0 else 0.0
        aggression_ratio = buy_volume / classified_vol if classified_vol > 0 else 0.5
        trade_frequency = total_trade_count / (window_ns / 1_000_000_000)
        large_count = sum(r["large_trade_count"] for r in g)
        large_ratio = large_count / total_trade_count if total_trade_count > 0 else 0.0
        avg_seconds = window_ns / 1_000_000_000 / total_trade_count if total_trade_count > 0 else 0.0

        rolled.append({
            "ticker": g[0]["ticker"],
            "timestamp": ts_iso,
            "trade_count": total_trade_count,
            "volume": total_volume,
            "vwap": round(vwap, 4),
            "avg_trade_size": round(avg_trade_size, 2),
            "median_trade_size": round(avg_trade_size, 2),
            "largest_trade": largest_trade,
            "stddev_trade_size": round(avg_trade_size, 2),
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "delta": delta,
            "cumulative_delta": round(cum_delta, 2),
            "delta_pct": round(delta_pct, 2),
            "aggression_ratio": round(aggression_ratio, 4),
            "trade_frequency": round(trade_frequency, 4),
            "large_trade_count": large_count,
            "large_trade_ratio": round(large_ratio, 4),
            "avg_seconds_between_trades": round(avg_seconds, 2),
        })

    return rolled


def process_ticker(ticker: str, year: str, agg: str, parquet: bool, client) -> tuple[int, int]:
    multiplier, timespan, _ = AGGREGATE_MAP[agg]
    trade_dates = list(trading_days(year))
    total_rows = 0
    cum_delta = 0.0

    out = output_path(ticker, year, agg, parquet, subdir="processing")
    out.parent.mkdir(parents=True, exist_ok=True)

    for d in trade_dates:
        day_rows, cum_delta = process_day(ticker, d, client, cum_delta)
        if day_rows:
            rolled = rollup_rows(day_rows, multiplier, timespan)
            _append_rows(out, rolled, parquet)
            total_rows += len(rolled)

    return total_rows, round(cum_delta, 2)


def main():
    args = parse_args()
    overall_start = time.time()
    tickers = load_tickers(args)
    year = args.year
    agg = args.aggregate
    parquet = args.parquet
    folder = AGGREGATE_MAP[agg][2]

    output_base = Path("data") / "trades" / folder
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
            log_dir = Path("data") / "trades" / folder / year / "logs"
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

        if args.resume and is_ticker_complete(ticker, year, agg, parquet):
            logger.info("[%d/%d] %s -> already complete, skipping", i, len(tickers), ticker)
            results.append({"ticker": ticker, "status": "skipped"})
            continue

        logger.info("[%d/%d] %s -> downloading trades ...", i, len(tickers), ticker)
        t0 = time.time()

        try:
            total_rows, _ = process_ticker(ticker, year, agg, parquet, client)
        except Exception as e:
            elapsed = time.time() - t0
            logger.error("[%d/%d] %s -> FAILED after %.1fs: %s", i, len(tickers), ticker, elapsed, e)
            missing.append(ticker)
            results.append({"ticker": ticker, "status": "failed", "error": str(e), "elapsed_s": round(elapsed, 1)})
            proc_path = output_path(ticker, year, agg, parquet, subdir="processing")
            if proc_path.exists():
                err_dir = output_path(ticker, year, agg, parquet, subdir="errors").parent
                err_dir.mkdir(parents=True, exist_ok=True)
                proc_path.rename(err_dir / proc_path.name)
            continue

        elapsed = time.time() - t0

        if total_rows == 0:
            logger.warning("[%d/%d] %s -> no trade data (%.1fs)", i, len(tickers), ticker, elapsed)
            missing.append(ticker)
            results.append({"ticker": ticker, "status": "no_data", "elapsed_s": round(elapsed, 1)})
            proc_path = output_path(ticker, year, agg, parquet, subdir="processing")
            if proc_path.exists():
                err_dir = output_path(ticker, year, agg, parquet, subdir="errors").parent
                err_dir.mkdir(parents=True, exist_ok=True)
                proc_path.rename(err_dir / proc_path.name)
            continue

        out = output_path(ticker, year, agg, parquet)
        proc_path = output_path(ticker, year, agg, parquet, subdir="processing")
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

    # Machine-readable status marker for parallel dispatcher
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
