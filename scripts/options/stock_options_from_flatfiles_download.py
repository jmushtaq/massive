"""
Download options-derived features for a ticker from S3 flat files and
underlying OHLCV data. One row per aggregate window with options activity.

Sources:
  S3 flat files (minute_aggs_v1) — downloaded per day, deleted after use
  Underlying 1min OHLCV — from stocks_aggs_download.py (data/SPY/1min/<year>/)

Quotes (quotes_v1 ~95 GiB/day) are not processed — quote columns are present
but left empty. Minute-aggs data is resampled for --aggregate values > 1min.

Features (Priority 1 — no Black-Scholes, v2 will add IV/greeks):
  ATM strike, call/put/straddle price, expected move
  Put/call volume, ratio, contract count
  Underlying OHLCV (open, high, low, close, vwap)
  Raw inputs for v2 BS: iv30d strike/call/put close, days to expiry

Output layout:
  data/options/stocks/<aggregate>/<year>/<ticker>_<year>_<aggregate>_options.csv

Usage:
  python scripts/options/stock_options_from_flatfiles_download.py --tickers AAPL --year 2025
  python scripts/options/stock_options_from_flatfiles_download.py --tickers AAPL,NVDA --year 2025 --aggregate 1D
  python scripts/options/stock_options_from_flatfiles_download.py --tickers UPS --year 2025 --smart_resume --resume
"""

import argparse
import csv
import datetime
import gzip
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(env_path)

api_key = os.getenv("APIKEY")
if not api_key:
    raise ValueError("APIKEY not found in .env")
os.environ["MASSIVE_API_KEY"] = api_key

SCRIPT_NAME = Path(__file__).resolve().stem

S3_ENDPOINT = "https://files.massive.com"
S3_BASE = "s3://flatfiles/us_options_opra"

OPTION_RE = re.compile(r"^O:([A-Z]+)(\d{6})([CP])(\d{8})$")

AGGREGATE_MAP = {
    "1sec": (1, "second", "1sec"),
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
    "underlying_price",
    "atm_strike",
    "atm_call_close",
    "atm_put_close",
    "atm_straddle_price",
    "expected_move",
    "put_volume",
    "call_volume",
    "put_call_ratio",
    "contract_count",
    "avg_bid_ask_spread",
    "avg_bid_size",
    "avg_ask_size",
    "quote_imbalance",
    "atm_days_to_expiry",
    "iv30d_strike",
    "iv30d_call_close",
    "iv30d_put_close",
    "iv30d_days_to_expiry",
    "open",
    "high",
    "low",
    "close",
    "vwap",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download 1min options features from S3 flat files + underlying OHLCV"
    )
    parser.add_argument(
        "--aggregate",
        choices=list(AGGREGATE_MAP.keys()),
        default="1min",
        help="Aggregate window size (default: 1min). 1min reads flat files directly; larger windows resample from 1min.",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        help="Comma-separated ticker symbols (e.g. AAPL,TSLA,NVDA)",
    )
    parser.add_argument(
        "--tickers_file",
        type=str,
        help="Path to CSV with ticker list (header 'ticker')",
    )
    parser.add_argument(
        "--year",
        type=str,
        required=True,
        help="Year or year range (e.g. 2025 or 2022-2025)",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default=None,
        help="Start date YYYY-MM-DD (default: <year>-01-01)",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default=None,
        help="End date YYYY-MM-DD (default: <year>-12-31). Limits the date range processed.",
    )
    parser.add_argument(
        "--no_rename",
        action="store_true",
        default=False,
        help="Leave output in processing/ directory — dispatcher handles final rename (used internally by parallel runner).",
    )
    parser.add_argument(
        "--smart_resume",
        action="store_true",
        default=False,
        help="Read processing file and start from the day after its last row.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip tickers that already have a non-empty output file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Base output directory (default: data/)",
    )
    parser.add_argument(
        "--downloads_dir",
        type=str,
        default=None,
        help="Directory with pre-downloaded minute_aggs files (YYYY-MM-DD.csv or .csv.gz). When set, skips S3 download and reads from this dir.",
    )
    parser.add_argument(
        "--use_unzipped",
        type=lambda s: s.lower() == "true",
        default=True,
        help="When using --downloads_dir, expect .csv files (True) or .csv.gz files (False). Default: True.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Sleep seconds between trading days (default: 0)",
    )
    return parser.parse_args()


def clean_ticker(raw: str) -> str:
    return raw.strip().upper().split("-")[0]


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


def parse_years(year_arg: str) -> list[str]:
    parts = year_arg.split("-")
    if len(parts) == 1:
        y = parts[0].strip()
        if not y.isdigit():
            raise SystemExit(f"Error: invalid year '{year_arg}'")
        return [y]
    elif len(parts) == 2:
        start, end = parts[0].strip(), parts[1].strip()
        if not start.isdigit() or not end.isdigit():
            raise SystemExit(f"Error: invalid year range '{year_arg}'")
        return [str(y) for y in range(int(start), int(end) + 1)]
    else:
        raise SystemExit(f"Error: invalid year format '{year_arg}' (use YYYY or YYYY-YYYY)")


def output_base(agg: str, output_dir: str | None = None) -> Path:
    base = Path(output_dir) if output_dir else Path("data")
    return base / "options" / "stocks" / AGGREGATE_MAP[agg][2]


def output_path(ticker: str, year: str, agg: str, output_dir: str | None = None, subdir: str | None = None) -> Path:
    folder = AGGREGATE_MAP[agg][2]
    ext = "csv"
    base = output_base(agg, output_dir) / year
    if subdir:
        base = base / subdir
    return base / f"{ticker}_{year}_{folder}_options.{ext}"


def is_ticker_complete(ticker: str, year: str, agg: str, output_dir: str | None = None) -> bool:
    p = output_path(ticker, year, agg, output_dir)
    if not p.exists() or p.stat().st_size == 0:
        return False
    with open(p) as f:
        return sum(1 for _ in f) > 1


def last_row_date(ticker: str, year: str, agg: str, output_dir: str | None = None) -> str | None:
    path = output_path(ticker, year, agg, output_dir, subdir="processing")
    if not path.exists() or path.stat().st_size == 0:
        return None
    last_ts = None
    with open(path) as f:
        for line in f:
            line = line.strip().strip("\x00").strip()
            if not line or "," not in line:
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            try:
                ts = datetime.datetime.fromisoformat(parts[1])
                last_ts = ts
            except (ValueError, IndexError):
                continue
    if last_ts is None:
        return None
    return last_ts.date().isoformat()


def trading_days(year: str) -> list[datetime.date]:
    start = datetime.date(int(year), 1, 1)
    end = datetime.date(int(year), 12, 31)
    return [start + datetime.timedelta(days=i) for i in range((end - start).days + 1) if (start + datetime.timedelta(days=i)).weekday() < 5]


def download_s3_file(remote_path: str, local_path: str) -> bool:
    cmd = [
        "aws", "s3", "cp", remote_path, local_path,
        "--endpoint-url", S3_ENDPOINT,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
        return os.path.exists(local_path) and os.path.getsize(local_path) > 0
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or b"").decode(errors="replace") if isinstance(e.stderr, bytes) else str(e.stderr or "")
        if "404" in stderr or "Not Found" in stderr:
            logging.getLogger(SCRIPT_NAME).debug("S3 file not found (non-trading day?): %s", remote_path)
        else:
            logging.getLogger(SCRIPT_NAME).warning("S3 download failed: %s", stderr.strip() if stderr else e)
        return False
    except Exception as e:
        logging.getLogger(SCRIPT_NAME).warning("S3 download error: %s", e)
        return False


def ns_to_minute_key(ns: int) -> int:
    return ns // 60_000_000_000


def dt_to_minute_key(dt: datetime.datetime) -> int:
    return int(dt.timestamp() // 60)


def parse_option_symbol(symbol: str) -> dict | None:
    m = OPTION_RE.match(symbol)
    if not m:
        return None
    ticker = m.group(1)
    expiry_str = "20" + m.group(2)
    opt_type = m.group(3)  # C or P
    strike = int(m.group(4)) / 1000.0
    try:
        expiry_date = datetime.date(int(expiry_str[:4]), int(expiry_str[4:6]), int(expiry_str[6:8]))
    except ValueError:
        return None
    return {
        "underlying": ticker,
        "expiry": expiry_date,
        "type": opt_type,
        "strike": strike,
    }


def load_underlying_ohlcv(underlying_dir: Path, ticker: str, year: int, day_str: str) -> dict[str, dict]:
    bars: dict[str, dict] = {}
    for ext in (".csv", ".csv.gz"):
        path = underlying_dir / f"{ticker}_{year}_1min{ext}"
        if path.exists():
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(path, "rt", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ts = row.get("timestamp", "")
                    if not ts.startswith(day_str):
                        continue
                    try:
                        dt = datetime.datetime.fromisoformat(ts)
                        mk = str(dt_to_minute_key(dt))
                    except (ValueError, TypeError):
                        continue
                    bars[mk] = {
                        "timestamp": ts,
                        "open": float(row.get("open", 0) or 0),
                        "high": float(row.get("high", 0) or 0),
                        "low": float(row.get("low", 0) or 0),
                        "close": float(row.get("close", 0) or 0),
                        "volume": float(row.get("volume", 0) or 0),
                        "vwap": float(row.get("vwap", 0) or 0),
                    }
            break
    return bars


def extract_option_bars(ticker: str, gz_path: str) -> dict[int, list[dict]]:
    bars: dict[int, list[dict]] = defaultdict(list)
    prefix = f"O:{ticker}"
    try:
        if gz_path.endswith(".gz"):
            opener = gzip.open
        else:
            opener = open
        with opener(gz_path, "rt", errors="replace") as f:
            f.readline()  # header
            for line in f:
                if not line.startswith(prefix):
                    continue
                parts = line.strip().split(",")
                if len(parts) < 7:
                    continue
                info = parse_option_symbol(parts[0])
                if info is None or info["underlying"] != ticker:
                    continue
                try:
                    window_start = int(parts[6])
                    mk = ns_to_minute_key(window_start)
                except (ValueError, IndexError):
                    continue
                bar = {
                    "underlying": ticker,
                    "opt_type": info["type"],
                    "strike": info["strike"],
                    "expiry": info["expiry"],
                    "volume": float(parts[1]) if parts[1] else 0.0,
                    "open": float(parts[2]) if parts[2] else None,
                    "close": float(parts[3]) if parts[3] else None,
                    "high": float(parts[4]) if parts[4] else None,
                    "low": float(parts[5]) if parts[5] else None,
                }
                bars[mk].append(bar)
    except (OSError, gzip.BadGzipFile) as e:
        logging.getLogger(SCRIPT_NAME).warning("Error reading %s: %s", gz_path, e)
    return dict(bars)


def stream_quotes_from_s3(ticker: str, s3_path: str) -> dict[int, list[dict]]:
    quotes: dict[int, list[dict]] = defaultdict(list)
    prefix = f"O:{ticker}"

    cmd = (
        f"aws s3 cp {s3_path} - --endpoint-url {S3_ENDPOINT} 2>/dev/null"
        f" | zcat 2>/dev/null"
        f" | grep '^{prefix}'"
    )
    try:
        proc = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
        )
    except Exception as e:
        logging.getLogger(SCRIPT_NAME).warning("Failed to stream quotes: %s", e)
        return dict(quotes)

    try:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 8:
                continue
            info = parse_option_symbol(parts[0])
            if info is None or info["underlying"] != ticker:
                continue
            try:
                sip_ts = int(parts[7])
                mk = ns_to_minute_key(sip_ts)
                bid_price = float(parts[5]) if parts[5] and parts[5] != "0" else None
                ask_price = float(parts[2]) if parts[2] and parts[2] != "0" else None
                bid_size = float(parts[6]) if parts[6] else 0
                ask_size = float(parts[3]) if parts[3] else 0
            except (ValueError, IndexError):
                continue
            if bid_price is None or ask_price is None:
                continue
            quotes[mk].append({
                "bid_price": bid_price,
                "ask_price": ask_price,
                "bid_size": bid_size,
                "ask_size": ask_size,
                "strike": info["strike"],
                "opt_type": info["type"],
                "expiry": info["expiry"],
            })
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
    return dict(quotes)


def compute_quote_features(quotes_list: list[dict], atm_strike: float) -> tuple:
    if not quotes_list:
        return (None, None, None, None)
    spreads = []
    bid_sizes = []
    ask_sizes = []
    imbalances = []
    for q in quotes_list:
        spread = q["ask_price"] - q["bid_price"]
        if spread >= 0:
            spreads.append(spread)
        bid_sizes.append(q["bid_size"])
        ask_sizes.append(q["ask_size"])
        total = q["bid_size"] + q["ask_size"]
        if total > 0:
            imbalances.append((q["bid_size"] - q["ask_size"]) / total)
    avg_spread = sum(spreads) / len(spreads) if spreads else None
    avg_bid_size = sum(bid_sizes) / len(bid_sizes) if bid_sizes else None
    avg_ask_size = sum(ask_sizes) / len(ask_sizes) if ask_sizes else None
    avg_imbalance = sum(imbalances) / len(imbalances) if imbalances else None
    return (avg_spread, avg_bid_size, avg_ask_size, avg_imbalance)


def compute_30d_contract(bars_list: list[dict], ref_date: datetime.date, atm_strike: float | None) -> dict | None:
    if not atm_strike:
        return None
    candidates = []
    for b in bars_list:
        if b["close"] is None:
            continue
        days = (b["expiry"] - ref_date).days
        if days < 1:
            continue
        strike_diff = abs(b["strike"] - atm_strike)
        if strike_diff / atm_strike > 0.02:
            continue
        candidates.append((days, b))
    if not candidates:
        return None
    target_days = 30
    best = min(candidates, key=lambda x: abs(x[0] - target_days))
    days, bar = best
    return {
        "strike": bar["strike"],
        "call_close": bar["close"] if bar["opt_type"] == "C" else None,
        "put_close": bar["close"] if bar["opt_type"] == "P" else None,
        "days_to_expiry": days,
        "expiry": bar["expiry"].isoformat(),
    }


def compute_row(ticker: str, mk: int, underlying: dict, bars_list: list[dict], quotes_list: list[dict], ref_date: datetime.date) -> dict | None:
    if not bars_list:
        return None

    underlying_price = underlying["close"]
    if not underlying_price or underlying_price <= 0:
        return None

    strikes = {}
    for b in bars_list:
        s = b["strike"]
        if s not in strikes:
            strikes[s] = {"C": None, "P": None}
        strikes[s][b["opt_type"]] = b

    if not strikes:
        return None

    atm_strike = min(strikes.keys(), key=lambda s: abs(s - underlying_price))
    atm_data = strikes[atm_strike]

    atm_call = atm_data.get("C")
    atm_put = atm_data.get("P")

    atm_call_close = atm_call["close"] if atm_call else None
    atm_put_close = atm_put["close"] if atm_put else None

    straddle = None
    if atm_call_close is not None and atm_put_close is not None:
        straddle = atm_call_close + atm_put_close

    put_vol = sum(b["volume"] for b in bars_list if b["opt_type"] == "P")
    call_vol = sum(b["volume"] for b in bars_list if b["opt_type"] == "C")
    put_call_ratio = put_vol / call_vol if call_vol > 0 else None

    contract_count = len(strikes)

    atm_expiry = None
    atm_days = None
    if atm_call:
        atm_expiry = atm_call["expiry"]
    elif atm_put:
        atm_expiry = atm_put["expiry"]
    if atm_expiry:
        atm_days = (atm_expiry - ref_date).days

    spread, bid_sz, ask_sz, imbalance = compute_quote_features(quotes_list, atm_strike)

    iv30d_info = compute_30d_contract(bars_list, ref_date, atm_strike)

    try:
        mk_dt = datetime.datetime.fromtimestamp(mk * 60, tz=datetime.timezone.utc)
        ts_str = mk_dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    except (ValueError, OSError):
        ts_str = ""

    return {
        "ticker": ticker,
        "timestamp": ts_str,
        "underlying_price": underlying_price,
        "atm_strike": atm_strike,
        "atm_call_close": atm_call_close,
        "atm_put_close": atm_put_close,
        "atm_straddle_price": straddle,
        "expected_move": straddle,
        "put_volume": put_vol,
        "call_volume": call_vol,
        "put_call_ratio": put_call_ratio,
        "contract_count": contract_count,
        "avg_bid_ask_spread": spread,
        "avg_bid_size": bid_sz,
        "avg_ask_size": ask_sz,
        "quote_imbalance": imbalance,
        "atm_days_to_expiry": atm_days,
        "iv30d_strike": iv30d_info["strike"] if iv30d_info else None,
        "iv30d_call_close": iv30d_info["call_close"] if iv30d_info else None,
        "iv30d_put_close": iv30d_info["put_close"] if iv30d_info else None,
        "iv30d_days_to_expiry": iv30d_info["days_to_expiry"] if iv30d_info else None,
        "open": underlying.get("open"),
        "high": underlying.get("high"),
        "low": underlying.get("low"),
        "close": underlying.get("close"),
        "vwap": underlying.get("vwap"),
    }


def process_ticker_day(ticker: str, day: datetime.date, minute_aggs_gz: str, underlying_dir: Path, agg: str) -> list[dict]:
    day_str = day.strftime("%Y-%m-%d")

    option_bars = extract_option_bars(ticker, minute_aggs_gz)
    underlying_bars = load_underlying_ohlcv(underlying_dir, ticker, day.year, day_str)

    multiplier, _, _ = AGGREGATE_MAP[agg]
    window_minutes = 1
    if agg != "1min" and agg != "1sec":
        if agg == "1D":
            window_minutes = 390
        elif agg in ("5min", "15min"):
            window_minutes = multiplier
        elif agg == "1H":
            window_minutes = 60
        elif agg == "4H":
            window_minutes = 240

    if window_minutes > 1:
        option_bars = resample_bars(option_bars, window_minutes)
        underlying_bars = resample_underlying(underlying_bars, window_minutes)

    rows = []
    for mk_str, ubar in sorted(underlying_bars.items()):
        mk = int(mk_str)
        bars_list = option_bars.get(mk, [])
        if not bars_list:
            continue
        row = compute_row(ticker, mk, ubar, bars_list, [], day)
        if row:
            rows.append(row)

    return rows


def resample_bars(bars: dict[int, list[dict]], window_min: int) -> dict[int, list[dict]]:
    result: dict[int, list[dict]] = defaultdict(list)
    for mk, bar_list in bars.items():
        bucket = (mk // window_min) * window_min
        for b in bar_list:
            result[bucket].append(b)
    return dict(result)


def resample_underlying(bars: dict[str, dict], window_min: int) -> dict[str, dict]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for mk_str, bar in bars.items():
        mk = int(mk_str)
        bucket = (mk // window_min) * window_min
        grouped[bucket].append(bar)
    result = {}
    for bucket, group in sorted(grouped.items()):
        opens = [b["open"] for b in group if b["open"] > 0]
        highs = [b["high"] for b in group if b["high"] > 0]
        lows = [b["low"] for b in group if b["low"] > 0]
        closes = [b["close"] for b in group if b["close"] > 0]
        vols = [b["volume"] for b in group]
        vwaps = [b["vwap"] for b in group if b["vwap"] > 0]
        if not opens:
            continue
        result[str(bucket)] = {
            "timestamp": group[0]["timestamp"],
            "open": opens[0],
            "high": max(highs) if highs else opens[0],
            "low": min(lows) if lows else opens[0],
            "close": closes[-1],
            "volume": sum(vols),
            "vwap": sum(vwaps) / len(vwaps) if vwaps else 0.0,
        }
    return result


def download_ticker(client_unused, ticker: str, year: str, agg: str, output_dir: str | None = None, start_date: str | None = None, end_date: str | None = None, delay: float = 0.0, downloads_dir: str | None = None, use_unzipped: bool = True) -> int:
    ext = ".csv" if use_unzipped else ".csv.gz"
    if downloads_dir:
        cache_files = sorted(f for f in os.listdir(downloads_dir) if f.endswith(ext))
        days = []
        for cf in cache_files:
            try:
                name = cf.replace(ext, "")
                d = datetime.date.fromisoformat(name)
                if d.year == int(year):
                    days.append(d)
            except ValueError:
                continue
    else:
        days = trading_days(year)

    if start_date:
        sd = datetime.date.fromisoformat(start_date)
        days = [d for d in days if d >= sd]
    if end_date:
        ed = datetime.date.fromisoformat(end_date)
        days = [d for d in days if d <= ed]
    if not days:
        return 0

    proc_path = output_path(ticker, year, agg, output_dir, subdir="processing")
    proc_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(SCRIPT_NAME)
    written = 0

    # Append mode if file already exists (for re-spawn / smart-resume), write mode for fresh start
    file_exists = proc_path.exists() and proc_path.stat().st_size > 0
    mode = "a" if file_exists else "w"
    f = open(proc_path, mode, newline="")
    writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
    if not file_exists:
        writer.writeheader()

    use_cache = downloads_dir is not None

    try:
        for day in days:
            day_str = day.strftime("%Y-%m-%d")
            year_str = day.strftime("%Y")
            month_str = day.strftime("%m")

            tmpdir: str | None = None
            local_minute: str

            if use_cache:
                cached = os.path.join(downloads_dir, f"{day_str}{ext}")
                if not os.path.exists(cached) or os.path.getsize(cached) == 0:
                    continue
                local_minute = cached
            else:
                tmpdir = f"/tmp/options_flatfiles_{ticker}_{day_str}"
                os.makedirs(tmpdir, exist_ok=True)
                s3_minute = f"{S3_BASE}/minute_aggs_v1/{year_str}/{month_str}/{day_str}.csv.gz"
                local_minute = os.path.join(tmpdir, "minute_aggs.csv.gz")
                if not download_s3_file(s3_minute, local_minute):
                    logger.warning("  %s %s: failed to download minute_aggs", ticker, day_str)
                    _cleanup(tmpdir)
                    continue

            underlying_dir = Path(output_dir) / "SPY" / "1min" / year_str if output_dir else Path("data") / "SPY" / "1min" / year_str

            rows = process_ticker_day(ticker, day, local_minute, underlying_dir, agg)

            for row in rows:
                writer.writerow(row)
                written += 1

            f.flush()

            if not use_cache and tmpdir:
                _cleanup(tmpdir)

            if delay > 0:
                time.sleep(delay)

    finally:
        f.flush()
        f.close()

    if written == 0:
        return 0

    return written


def _cleanup(tmpdir: str):
    import shutil
    try:
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:
        pass


def main():
    args = parse_args()
    overall_start = time.time()
    tickers = load_tickers(args)
    years = parse_years(args.year)
    agg = args.aggregate
    out_dir = args.output
    _, _, folder = AGGREGATE_MAP[agg]

    log_dir = output_base(agg, out_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    log_path = log_dir / f"{SCRIPT_NAME}_{log_ts}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger(SCRIPT_NAME)

    logger.info("Starting options-from-flatfiles for %d tickers, years=%s, aggregate=%s", len(tickers), years, agg)
    logger.info("Resume mode: %s", args.resume)
    if args.start_date:
        logger.info("Start date: %s", args.start_date)
    logger.info("Output base: %s", output_base(agg, out_dir).resolve())

    all_missing: list[str] = []
    all_results: list[dict] = []
    all_downloaded = 0
    all_skipped = 0

    for year in years:
        downloaded = 0
        skipped = 0

        for i, ticker in enumerate(tickers, 1):
            if args.resume and is_ticker_complete(ticker, year, agg, out_dir):
                logger.info("[%d/%d] %s (%s) -> already complete, skipping", i, len(tickers), ticker, year)
                skipped += 1
                all_results.append({"ticker": ticker, "year": year, "status": "skipped"})
                continue

            logger.info("[%d/%d] %s (%s) -> downloading ...", i, len(tickers), ticker, year)

            ticker_start = args.start_date
            if args.smart_resume:
                last_date = last_row_date(ticker, year, agg, out_dir)
                if last_date:
                    ticker_start = last_date
                    logger.info("  smart-resuming from %s", last_date)
                else:
                    logger.info("  no partial data found, starting from beginning")

            t0 = time.time()
            try:
                count = download_ticker(None, ticker, year, agg, out_dir, ticker_start, args.end_date, args.delay, args.downloads_dir, args.use_unzipped)
            except Exception as e:
                elapsed = time.time() - t0
                logger.error("[%d/%d] %s (%s) -> FAILED after %.1fs: %s", i, len(tickers), ticker, year, elapsed, e)
                all_missing.append(ticker)
                all_results.append({"ticker": ticker, "year": year, "status": "failed", "error": str(e), "elapsed_s": round(elapsed, 1)})
                proc_path = output_path(ticker, year, agg, out_dir, subdir="processing")
                if proc_path.exists():
                    err_dir = output_path(ticker, year, agg, out_dir, subdir="errors").parent
                    err_dir.mkdir(parents=True, exist_ok=True)
                    proc_path.rename(err_dir / proc_path.name)
                continue

            elapsed = time.time() - t0

            if count == 0:
                proc_path = output_path(ticker, year, agg, out_dir, subdir="processing")
                already_had_data = proc_path.exists() and proc_path.stat().st_size > 0
                if already_had_data:
                    logger.info("[%d/%d] %s (%s) -> no new days (%.1fs) — existing data preserved in %s", i, len(tickers), ticker, year, elapsed, proc_path)
                    downloaded += 1
                    size = proc_path.stat().st_size
                    all_results.append({
                        "ticker": ticker, "year": year, "status": "ok",
                        "rows": -1, "size_bytes": size, "path": str(proc_path),
                        "elapsed_s": round(elapsed, 1),
                    })
                    logger.info("PARALLEL_RESULT:{\"ticker\":\"%s\",\"year\":\"%s\",\"status\":\"ok\",\"rows\":-1}", ticker, year)
                    sys.stderr.write("PARALLEL_RESULT:{\"ticker\":\"%s\",\"year\":\"%s\",\"status\":\"ok\",\"rows\":-1}\n" % (ticker, year))
                    sys.stderr.flush()
                else:
                    logger.warning("[%d/%d] %s (%s) -> no data returned (%.1fs)", i, len(tickers), ticker, year, elapsed)
                    all_missing.append(ticker)
                    all_results.append({"ticker": ticker, "year": year, "status": "no_data", "elapsed_s": round(elapsed, 1)})
                    if proc_path.exists():
                        no_data_dir = output_path(ticker, year, agg, out_dir, subdir="no_data").parent
                        no_data_dir.mkdir(parents=True, exist_ok=True)
                        try:
                            proc_path.rename(no_data_dir / proc_path.name)
                        except OSError:
                            pass
            else:
                proc_path = output_path(ticker, year, agg, out_dir, subdir="processing")
                size = proc_path.stat().st_size
                logger.info(
                    "[%d/%d] %s (%s) -> %d rows (%.1fs) -> %s",
                    i, len(tickers), ticker, year, count, elapsed, proc_path,
                )
                downloaded += 1
                all_results.append({
                    "ticker": ticker, "year": year, "status": "ok",
                    "rows": count, "size_bytes": size, "path": str(proc_path),
                    "elapsed_s": round(elapsed, 1),
                })
                logger.info("PARALLEL_RESULT:{\"ticker\":\"%s\",\"year\":\"%s\",\"status\":\"ok\",\"rows\":%d}", ticker, year, count)
                sys.stderr.write("PARALLEL_RESULT:{\"ticker\":\"%s\",\"year\":\"%s\",\"status\":\"ok\",\"rows\":%d}\n" % (ticker, year, count))
                sys.stderr.flush()

        all_downloaded += downloaded
        all_skipped += skipped

    total_time = time.time() - overall_start

    summary = {
        "script": SCRIPT_NAME,
        "timestamp": log_ts,
        "years": years,
        "aggregate": agg,
        "total_tickers": len(tickers),
        "downloaded": all_downloaded,
        "skipped": all_skipped,
        "missing": sorted(set(all_missing)),
        "missing_count": len(set(all_missing)),
        "duration_s": round(total_time, 1),
        "results": all_results,
    }

    report_path = log_dir / f"{SCRIPT_NAME}_{log_ts}_report.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("SUMMARY REPORT")
    logger.info("  Duration:       %.1fs", total_time)
    logger.info("  Years:          %s", years)
    logger.info("  Downloaded:     %d", all_downloaded)
    logger.info("  Skipped:        %d", all_skipped)
    logger.info("  Missing/failed: %d", len(set(all_missing)))
    missing_unique = sorted(set(all_missing))
    if missing_unique:
        logger.info("  Missing tickers: %s", ", ".join(missing_unique))
    logger.info("  Report:         %s", report_path)

    # Rename processing files to final (standalone mode — skipped when dispatcher manages finalization)
    if not args.no_rename:
        for year in years:
            for ticker in tickers:
                proc = output_path(ticker, year, agg, out_dir, subdir="processing")
                if not proc.exists() or proc.stat().st_size == 0:
                    continue
                with open(proc) as f:
                    has_rows = sum(1 for _ in f) > 1
                if has_rows:
                    final = output_path(ticker, year, agg, out_dir)
                    final.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        proc.rename(final)
                    except OSError:
                        pass
                else:
                    no_data_dir = output_path(ticker, year, agg, out_dir, subdir="no_data").parent
                    no_data_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        proc.rename(no_data_dir / proc.name)
                    except OSError:
                        pass


if __name__ == "__main__":
    main()
