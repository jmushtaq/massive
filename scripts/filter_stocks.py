"""
Filter stocks by configurable criteria using local data (fundamentals, reference)
or the Massive API (snapshots, ticker details).

Two modes:
  --use_local : Filter using pre-downloaded fundamentals.csv and reference.csv
  --use_api   : Fetch data live from the Massive API

In --use_api mode, the script first calls get_snapshot_all("stocks") (1 API call)
to bulk-evaluate price, volume, and relative-volume filters. Tickers that pass
those filters then get individual get_ticker_details() calls for market_cap,
sector, and exchange. This avoids 13K+ sequential API calls.

Filters are read from a JSON config file (--config or built-in defaults).

Config format (all fields optional -- omitted fields are skipped):
{
  "market_cap": {"min": 2e9, "max": 6e9},
  "price": {"min": 10, "max": 15},
  "volume": {"min": 1.5e6, "max": 3.5e6},
  "rvol": {"min": 1.0, "max": 1.5},
  #"sector_keywords": ["TECHNOLOGY", "SEMICONDUCTORS"],
  #"exchange": "XNAS",
  "active": true
}

Usage:
    python scripts/filter_stocks.py --use_api --config my_filters.json --date 2025-01-15
    python scripts/filter_stocks.py --use_api --config my_filters.json --startdate 2020-01-01 --enddate 2020-12-31
    python scripts/filter_stocks.py --use_local --config my_filters.json
    python scripts/filter_stocks.py --tickers AAPL,NVDA,TSLA --use_api --config my_filters.json
"""

import argparse
import csv
import datetime
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

api_key = os.getenv("APIKEY")
if not api_key:
    raise ValueError("APIKEY not found in .env")
os.environ["MASSIVE_API_KEY"] = api_key

from massive import RESTClient

DEFAULT_CONFIG = {
    "market_cap": {},
    "price": {},
    "volume": {},
    "rvol": {},
    "sector_keywords": [],
    "exchange": "",
    "active": True,
}


def clean_ticker(raw: str) -> str:
    return raw.strip().upper().split("-")[0]


def load_tickers(tickers_arg: str | None, tickers_file: str | None) -> list[str]:
    tickers = []
    if tickers_arg:
        tickers.extend(clean_ticker(t) for t in tickers_arg.split(",") if t.strip())
    if tickers_file:
        with open(tickers_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                t = row.get("ticker", "").strip()
                if t:
                    tickers.append(clean_ticker(t))
    return tickers


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter stocks by configurable criteria"
    )
    parser.add_argument("--tickers", type=str, help="Comma-separated ticker symbols")
    parser.add_argument("--tickers_file", type=str, help="CSV with ticker list (header 'ticker')")
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    parser.add_argument("--use_local", action="store_true", default=False,
                        help="Use pre-downloaded fundamentals.csv and reference.csv")
    parser.add_argument("--use_api", action="store_true", default=False,
                        help="Fetch data live from the Massive API")
    parser.add_argument("--output", type=str, help="Save filtered tickers to CSV")
    parser.add_argument("--date", type=str, default=None,
                        help="Single date to evaluate filters (YYYY-MM-DD)")
    parser.add_argument("--startdate", type=str, default=None,
                        help="Start of date range (YYYY-MM-DD)")
    parser.add_argument("--enddate", type=str, default=None,
                        help="End of date range (YYYY-MM-DD)")
    return parser.parse_args()


def resolve_dates(args) -> list[str]:
    if args.date:
        return [args.date]
    if args.startdate and args.enddate:
        start = datetime.date.fromisoformat(args.startdate)
        end = datetime.date.fromisoformat(args.enddate)
        dates = []
        d = start
        while d <= end:
            if d.weekday() < 5:
                dates.append(d.isoformat())
            d += datetime.timedelta(days=1)
        return dates
    if args.startdate:
        return [args.startdate]
    if args.enddate:
        return [args.enddate]
    return [datetime.date.today().isoformat()]


def load_config(config_path: str | None) -> dict:
    if config_path:
        raw = []
        with open(config_path) as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("#"):
                    raw.append("\n")
                else:
                    raw.append(line)
        cfg = json.loads("".join(raw))
    else:
        cfg = {}
    for key, default_val in DEFAULT_CONFIG.items():
        cfg.setdefault(key, default_val)
    return cfg


def passes_range_filter(value: float | None, bounds: dict) -> bool:
    if value is None:
        return False
    if "min" in bounds and value < bounds["min"]:
        return False
    if "max" in bounds and value > bounds["max"]:
        return False
    return True


def passes_snapshot_filters(info: dict, config: dict) -> tuple[bool, list[str]]:
    reasons = []

    price = info.get("close_price")
    if config.get("price") and not passes_range_filter(price, config["price"]):
        reasons.append(f"price={price} outside {config['price']}")
        return False, reasons

    vol = info.get("volume")
    if config.get("volume") and not passes_range_filter(vol, config["volume"]):
        reasons.append(f"volume={vol} outside {config['volume']}")
        return False, reasons

    rvol = info.get("rvol")
    if config.get("rvol") and not passes_range_filter(rvol, config["rvol"]):
        reasons.append(f"rvol={rvol} outside {config['rvol']}")
        return False, reasons

    return True, reasons


def passes_detail_filters(info: dict, config: dict) -> tuple[bool, list[str]]:
    reasons = []

    mc = info.get("market_cap")
    if config.get("market_cap") and not passes_range_filter(mc, config["market_cap"]):
        reasons.append(f"market_cap={mc} outside {config['market_cap']}")
        return False, reasons

    sector = (info.get("sector") or "").upper()
    keywords = config.get("sector_keywords", [])
    if keywords:
        if not any(kw.upper() in sector for kw in keywords):
            reasons.append(f"sector='{sector}' has none of {keywords}")
            return False, reasons

    exchange = info.get("exchange") or ""
    req_exchange = config.get("exchange", "")
    if req_exchange and exchange.upper() != req_exchange.upper():
        reasons.append(f"exchange={exchange} != {req_exchange}")
        return False, reasons

    active = info.get("active")
    req_active = config.get("active")
    if req_active is not None and active is not None and active != req_active:
        reasons.append(f"active={active} != {req_active}")
        return False, reasons

    return True, reasons


def fetch_from_local(tickers: list[str], config: dict) -> list[dict]:
    fund_path = Path("data") / "fundamentals" / "fundamentals.csv"
    ref_path = Path("data") / "reference" / "reference.csv"

    fund_data = {}
    if fund_path.exists():
        with open(fund_path) as f:
            for row in csv.DictReader(f):
                t = row.get("ticker", "").strip().upper()
                if t:
                    mc_str = row.get("market_cap", "").strip()
                    fund_data[t] = {
                        "market_cap": float(mc_str) if mc_str else None,
                        "sector": row.get("sector", "").strip(),
                    }

    ref_data = {}
    if ref_path.exists():
        with open(ref_path) as f:
            for row in csv.DictReader(f):
                t = row.get("ticker", "").strip().upper()
                if t:
                    ref_data[t] = {
                        "exchange": row.get("exchange", "").strip(),
                        "sector": row.get("sector_industry", row.get("sector", "")).strip(),
                    }

    if not tickers:
        tickers = sorted(set(fund_data.keys()) | set(ref_data.keys()))

    results = []
    for ticker in tickers:
        info = {"ticker": ticker}
        f = fund_data.get(ticker, {})
        r = ref_data.get(ticker, {})
        info["market_cap"] = f.get("market_cap")
        info["sector"] = f.get("sector") or r.get("sector", "")
        info["exchange"] = r.get("exchange", "")

        passes, _ = passes_detail_filters(info, config)
        if passes:
            results.append(info)

    return results


def fetch_from_api(tickers: list[str], config: dict, dates: list[str]) -> list[dict]:
    client = RESTClient(trace=False)
    results = []

    if tickers:
        candidates = tickers
    else:
        sys.stderr.write("Fetching snapshot for all stocks (1 API call) ...\n")
        try:
            snapshots = client.get_snapshot_all("stocks", include_otc=False)
        except Exception as e:
            raise SystemExit(f"Error fetching snapshot: {e}")

        candidates = []
        for snap in snapshots:
            if not snap or not snap.ticker:
                continue
            close = None
            volume = None
            prev_vol = None
            if snap.prev_day:
                close = snap.prev_day.close
                volume = snap.prev_day.volume
                prev_vol = volume
            if snap.day and not close:
                close = snap.day.close
            if snap.day and not volume:
                volume = snap.day.volume

            src_price = snap.last_trade.price if snap.last_trade else None
            price = src_price or close

            rvol = None
            if volume and prev_vol and prev_vol > 0:
                rvol = volume / prev_vol

            info = {
                "ticker": snap.ticker,
                "close_price": price,
                "volume": volume,
                "rvol": rvol,
            }
            passes, _ = passes_snapshot_filters(info, config)
            if passes:
                candidates.append(snap.ticker)

        sys.stderr.write(f"  {len(candidates)}/{len(snapshots)} passed snapshot filters\n")

    needs_details = bool(config.get("market_cap") or config.get("sector_keywords") or config.get("exchange") or config.get("active") is not None)

    for i, ticker in enumerate(candidates):
        info = {"ticker": ticker}

        if needs_details:
            sys.stderr.write(f"[{i+1}/{len(candidates)}] {ticker} ...\n")
            try:
                details = client.get_ticker_details(ticker)
                if details:
                    info["market_cap"] = details.market_cap
                    info["sector"] = details.sic_description or ""
                    info["exchange"] = details.primary_exchange or ""
                    info["active"] = details.active
            except Exception as e:
                sys.stderr.write(f"  ticker_details failed: {e}\n")
                time.sleep(0.25)
                continue

            if config.get("market_cap") and not info.get("market_cap"):
                try:
                    for r in client.list_financials_ratios(ticker=ticker, limit=1):
                        if r and r.market_cap:
                            info["market_cap"] = r.market_cap
                        break
                except Exception:
                    pass

            passes, reasons = passes_detail_filters(info, config)
        else:
            passes = True
            reasons = []

        if passes:
            if not needs_details:
                info = {"ticker": ticker}
            results.append(info)

        if needs_details:
            sys.stderr.write(f"  passes={passes}" + (f" reasons={reasons}" if not passes else "") + "\n")
            time.sleep(0.2)

    return results


def main():
    args = parse_args()
    config = load_config(args.config)
    dates = resolve_dates(args)

    if args.tickers or args.tickers_file:
        tickers = load_tickers(args.tickers, args.tickers_file)
    else:
        tickers = []

    if args.use_local:
        matches = fetch_from_local(tickers, config)
    elif args.use_api:
        matches = fetch_from_api(tickers, config, dates)
    else:
        raise SystemExit("Error: specify --use_local or --use_api")

    print(f"\nMatches: {len(matches)}")
    for m in matches:
        parts = [m["ticker"]]
        if m.get("market_cap"):
            parts.append(f"mc=${m['market_cap']:,.0f}")
        if m.get("close_price"):
            parts.append(f"price=${m['close_price']:.2f}")
        if m.get("volume"):
            parts.append(f"vol={m['volume']:,.0f}")
        if m.get("rvol"):
            parts.append(f"rvol={m['rvol']:.2f}")
        if m.get("sector"):
            parts.append(f"sector={m['sector']}")
        print("  " + " | ".join(parts))

    if args.output:
        out_path = Path(args.output)
        fieldnames = ["ticker", "market_cap", "close_price", "volume", "rvol", "sector", "exchange", "active"]
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in matches:
                writer.writerow({k: m.get(k, "") for k in fieldnames})
        print(f"\nSaved {len(matches)} matches to {out_path}")


if __name__ == "__main__":
    main()
