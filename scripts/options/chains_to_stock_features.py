"""
Aggregate options-chain Greeks (per-contract, per-second) into per-second
stock options features.

Reads data/options/chains/<agg>/<year>/greeks/<T>_<year>_<agg>_chains.csv
(per-contract rows), groups by timestamp, and emits the standard 50-column
options-features file to data/options/stocks/<agg>/<year>/<T>_<year>_<agg>_options.csv.

Selection rules (target schema = stocks/.../greeks/<T>_..._options.csv):
  - atm_strike: strike nearest to underlying_price (backfilled from 1min OHLCV when missing)
  - ATM contract: call/put at atm_strike on the NEAREST expiry (shortest days_to_expiry >= 1)
  - iv30d: expiry nearest 30 days-to-expiry; strike nearest atm_strike within 2%
  - Greeks: copied directly from the selected call/put rows (no Black-Scholes recompute)
  - Underlying open/high/low/close/vwap: joined from data/SPY/1sec if present,
    otherwise synthesized from the chain's underlying_price observations
  - Quote columns avg_bid_size/avg_ask_size/quote_imbalance left empty (chain has no
    sizes); avg_bid_ask_spread averaged from the chain's spread column

Usage:
  python scripts/options/chains_to_stock_features.py --tickers AAP --year 2014 --aggregate 1sec
  python scripts/options/chains_to_stock_features.py --ohlcv_tickers --year 2014 --aggregate 1sec
  python scripts/options/chains_to_stock_features.py --tickers_file data/universes/2014/combined_unique.csv --year 2014 --aggregate 1sec
"""

import argparse
import csv
import datetime
import gzip
import math
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(env_path)

SCRIPT_NAME = Path(__file__).resolve().stem
STDOUT_MARKER = "PARALLEL_RESULT:"

AGGREGATE_MAP = {
    "1sec": "1sec", "1min": "1min", "5min": "5min",
    "15min": "15min", "1H": "1H", "4H": "4H", "1D": "1D",
}

_GREEK_NAMES = ["iv", "delta", "gamma", "theta", "vega", "rho"]
_GREEK_CHAIN_COLS = {
    "iv": "implied_volatility",
    "delta": "delta",
    "gamma": "gamma",
    "theta": "theta",
    "vega": "vega",
    "rho": "rho",
}

CSV_HEADERS = [
    "ticker", "timestamp", "underlying_price", "atm_strike",
    "atm_call_close", "atm_put_close", "atm_straddle_price", "expected_move",
    "put_volume", "call_volume", "put_call_ratio", "contract_count",
    "avg_bid_ask_spread", "avg_bid_size", "avg_ask_size", "quote_imbalance",
    "atm_days_to_expiry", "iv30d_strike", "iv30d_call_close", "iv30d_put_close",
    "iv30d_days_to_expiry", "open", "high", "low", "close", "vwap",
] + [f"{prefix}_{greek}"
     for prefix in ["atm_call", "atm_put", "iv30d_call", "iv30d_put"]
     for greek in _GREEK_NAMES]

AWST = datetime.timezone(datetime.timedelta(hours=8))


def clean_ticker(raw: str) -> str:
    return raw.strip().upper().split("-")[0]


def parse_years(year_arg: str) -> list[str]:
    parts = year_arg.split("-")
    if len(parts) == 1:
        if not parts[0].strip().isdigit():
            raise SystemExit("Error: invalid year '%s'" % year_arg)
        return [parts[0].strip()]
    if len(parts) == 2:
        start, end = parts[0].strip(), parts[1].strip()
        if not start.isdigit() or not end.isdigit():
            raise SystemExit("Error: invalid year range '%s'" % year_arg)
        return [str(y) for y in range(int(start), int(end) + 1)]
    raise SystemExit("Error: invalid year format '%s'" % year_arg)


def parse_args():
    p = argparse.ArgumentParser(description="Chain greeks -> stock options features")
    p.add_argument("--tickers", type=str, default=None)
    p.add_argument("--tickers_file", type=str, default=None)
    p.add_argument("--ohlcv_tickers", action="store_true", default=False)
    p.add_argument("--year", type=str, required=True)
    p.add_argument("--aggregate", choices=list(AGGREGATE_MAP.keys()), default="1sec")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--resume", action="store_true",
                   help="Skip tickers whose output file already exists")
    p.add_argument("--no_rename", action="store_true", default=False,
                   help="Leave output in processing/ (dispatcher handles final rename)")
    return p.parse_args()


def load_tickers(args) -> list[str]:
    tickers = []
    if args.tickers:
        tickers.extend(clean_ticker(t) for t in args.tickers.split(",") if t.strip())
    if args.tickers_file:
        with open(args.tickers_file) as f:
            for row in csv.DictReader(f):
                t = row.get("ticker", "").strip()
                if t:
                    tickers.append(clean_ticker(t))
    if args.ohlcv_tickers:
        year = args.year.split("-")[0]
        src = Path("data") / "SPY" / "1min" / year
        if not src.exists():
            raise SystemExit("Error: OHLCV dir not found: %s" % src)
        for f in sorted(src.glob(f"*_{year}_1min.csv*")):
            name = f.name.replace(".csv.gz", "").replace(".csv", "")
            tickers.append(clean_ticker(name.split("_")[0]))
    if not tickers:
        raise SystemExit("Error: specify --tickers, --tickers_file, or --ohlcv_tickers")
    return list(dict.fromkeys(tickers))


def _num(v):
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        x = float(s)
    except ValueError:
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def _num_int(v):
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _parse_ts(ts: str) -> datetime.datetime:
    dt = datetime.datetime.fromisoformat(ts.strip())
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=AWST)
    return dt.astimezone(datetime.timezone.utc)


def _sec_key(ts: str) -> str:
    return _parse_ts(ts).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def _min_key(ts: str) -> str:
    return _parse_ts(ts).strftime("%Y-%m-%dT%H:%M:00+00:00")


def chain_path(ticker: str, year: str, agg: str, output_dir: str | None = None) -> Path | None:
    folder = AGGREGATE_MAP[agg]
    base = (Path(output_dir) if output_dir else Path("data")) / "options" / "chains" / folder / year
    for sub in ("greeks", None):
        d = base if sub is None else base / sub
        for ext in ("", ".gz"):
            p = d / f"{ticker}_{year}_{folder}_chains.csv{ext}"
            if p.exists():
                return p
    return None


def out_path(ticker: str, year: str, agg: str, output_dir: str | None = None,
             subdir: str | None = None) -> Path:
    folder = AGGREGATE_MAP[agg]
    base = (Path(output_dir) if output_dir else Path("data")) / "options" / "stocks" / folder / year
    if subdir:
        base = base / subdir
    return base / f"{ticker}_{year}_{folder}_options.csv"


def load_1min_close_cache(underlying_dir: Path, ticker: str, year: str) -> dict[str, float]:
    cache: dict[str, float] = {}
    for ext in (".csv", ".csv.gz"):
        p = underlying_dir / f"{ticker}_{year}_1min{ext}"
        if p.exists():
            opener = gzip.open if ext.endswith(".gz") else open
            with opener(p, "rt", errors="replace") as f:
                for row in csv.DictReader(f):
                    ts = row.get("timestamp", "")
                    cv = row.get("close", "")
                    if not ts or not cv:
                        continue
                    try:
                        cache[_min_key(ts)] = float(cv)
                    except (ValueError, TypeError):
                        pass
            break
    return cache


def load_1sec_ohlcv_cache(underlying_dir: Path, ticker: str, year: str) -> dict[str, tuple]:
    cache: dict[str, tuple] = {}
    for ext in (".csv", ".csv.gz"):
        p = underlying_dir / f"{ticker}_{year}_1sec{ext}"
        if p.exists():
            opener = gzip.open if ext.endswith(".gz") else open
            with opener(p, "rt", errors="replace") as f:
                for row in csv.DictReader(f):
                    ts = row.get("timestamp", "")
                    if not ts:
                        continue
                    try:
                        o = float(row.get("open") or 0)
                        h = float(row.get("high") or 0)
                        l = float(row.get("low") or 0)
                        c = float(row.get("close") or 0)
                        vw = float(row.get("vwap") or 0)
                    except (ValueError, TypeError):
                        continue
                    cache[_sec_key(ts)] = (o, h, l, c, vw)
            break
    return cache


def _parse_contract(row: dict) -> dict:
    return {
        "strike": _num(row.get("strike")),
        "close": _num(row.get("close")),
        "dte": _num_int(row.get("days_to_expiry")),
        "cp": (row.get("call_put", "") or "").strip().upper(),
        "volume": _num(row.get("volume")) or 0.0,
        "spread": _num(row.get("spread")),
        "up": _num(row.get("underlying_price")),
        "greeks": {g: (row.get(c) or "").strip() for g, c in _GREEK_CHAIN_COLS.items()},
    }


def _select_iv30d(valid: list[dict], atm_strike: float) -> dict | None:
    target = 30
    dtes = sorted({c["dte"] for c in valid if c["dte"] is not None})
    if not dtes:
        return None
    nearest_dte = min(dtes, key=lambda d: abs(d - target))
    same_expiry = [c for c in valid if c["dte"] == nearest_dte and c["strike"] is not None]
    within_band = [c for c in same_expiry
                   if abs(c["strike"] - atm_strike) / atm_strike <= 0.02]
    if not within_band:
        within_band = same_expiry
    strike = min({c["strike"] for c in within_band}, key=lambda s: abs(s - atm_strike))
    call = next((c for c in within_band if c["strike"] == strike and c["cp"] == "C"), None)
    put = next((c for c in within_band if c["strike"] == strike and c["cp"] == "P"), None)
    return {"strike": strike, "dte": nearest_dte, "call": call, "put": put}


def build_feature_row(ticker: str, ts_key: str, rows: list[dict],
                      close_1min: dict[str, float],
                      ohlcv_1sec: dict[str, tuple]) -> dict | None:
    contracts = [c for c in (_parse_contract(r) for r in rows) if c["strike"] is not None]
    if not contracts:
        return None

    ups = [c["up"] for c in contracts if c["up"] is not None]
    underlying = ups[-1] if ups else None
    if underlying is None:
        underlying = close_1min.get(_min_key(ts_key))
    if underlying is None or underlying <= 0:
        return None

    valid = [c for c in contracts
             if c["close"] is not None and c["close"] > 0
             and c["dte"] is not None and c["dte"] >= 1]
    if not valid:
        return None

    atm_strike = min({c["strike"] for c in valid}, key=lambda s: abs(s - underlying))
    atm_cands = [c for c in valid if c["strike"] == atm_strike]
    nearest_dte = min(c["dte"] for c in atm_cands)
    atm_call = next((c for c in atm_cands if c["dte"] == nearest_dte and c["cp"] == "C"), None)
    atm_put = next((c for c in atm_cands if c["dte"] == nearest_dte and c["cp"] == "P"), None)

    atm_call_close = atm_call["close"] if atm_call else None
    atm_put_close = atm_put["close"] if atm_put else None
    straddle = (atm_call_close + atm_put_close) if (atm_call_close is not None
                                                    and atm_put_close is not None) else None

    put_vol = sum(c["volume"] for c in contracts if c["cp"] == "P")
    call_vol = sum(c["volume"] for c in contracts if c["cp"] == "C")
    pcr = put_vol / call_vol if call_vol > 0 else None
    contract_count = len({c["strike"] for c in contracts})

    spreads = [c["spread"] for c in contracts if c["spread"] is not None and c["spread"] >= 0]
    avg_spread = sum(spreads) / len(spreads) if spreads else None

    iv30d = _select_iv30d(valid, atm_strike)

    if ohlcv_1sec and ts_key in ohlcv_1sec:
        o, h, l, c, vw = ohlcv_1sec[ts_key]
    else:
        o = ups[0] if ups else underlying
        h = max(ups) if ups else underlying
        l = min(ups) if ups else underlying
        c = underlying
        vw = sum(ups) / len(ups) if ups else underlying

    feat = {
        "ticker": ticker,
        "timestamp": ts_key,
        "underlying_price": underlying,
        "atm_strike": atm_strike,
        "atm_call_close": atm_call_close,
        "atm_put_close": atm_put_close,
        "atm_straddle_price": straddle,
        "expected_move": straddle,
        "put_volume": put_vol,
        "call_volume": call_vol,
        "put_call_ratio": pcr,
        "contract_count": contract_count,
        "avg_bid_ask_spread": avg_spread,
        "avg_bid_size": None,
        "avg_ask_size": None,
        "quote_imbalance": None,
        "atm_days_to_expiry": nearest_dte,
        "iv30d_strike": iv30d["strike"] if iv30d else None,
        "iv30d_call_close": (iv30d["call"]["close"] if iv30d and iv30d["call"] else None),
        "iv30d_put_close": (iv30d["put"]["close"] if iv30d and iv30d["put"] else None),
        "iv30d_days_to_expiry": iv30d["dte"] if iv30d else None,
        "open": o, "high": h, "low": l, "close": c, "vwap": vw,
    }

    for prefix, ctr in (("atm_call", atm_call), ("atm_put", atm_put),
                        ("iv30d_call", iv30d["call"] if iv30d else None),
                        ("iv30d_put", iv30d["put"] if iv30d else None)):
        for g in _GREEK_NAMES:
            feat[f"{prefix}_{g}"] = ctr["greeks"].get(g, "") if ctr is not None else ""

    return feat


def process_ticker(ticker: str, year: str, agg: str,
                   output_dir: str | None) -> tuple[Path | None, int]:
    in_path = chain_path(ticker, year, agg, output_dir)
    if in_path is None:
        return None, 0

    base = Path(output_dir) if output_dir else Path("data")
    close_1min = load_1min_close_cache(base / "SPY" / "1min" / year, ticker, year)
    ohlcv_1sec = load_1sec_ohlcv_cache(base / "SPY" / "1sec" / year, ticker, year)

    groups: dict[str, list[dict]] = {}
    opener = gzip.open if in_path.name.endswith(".gz") else open
    with opener(in_path, "rt", errors="replace") as f:
        for row in csv.DictReader(f):
            ts = row.get("timestamp", "")
            if not ts:
                continue
            try:
                key = _sec_key(ts)
            except (ValueError, TypeError):
                continue
            groups.setdefault(key, []).append(row)

    proc_path = out_path(ticker, year, agg, output_dir, subdir="processing")
    proc_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(proc_path, "w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=CSV_HEADERS)
        writer.writeheader()
        for key in sorted(groups):
            feat = build_feature_row(ticker, key, groups[key], close_1min, ohlcv_1sec)
            if feat is None:
                continue
            writer.writerow({k: ("" if v is None else v) for k, v in feat.items()})
            written += 1

    return proc_path, written


def is_complete(ticker: str, year: str, agg: str, output_dir: str | None = None) -> bool:
    p = out_path(ticker, year, agg, output_dir)
    if not p.exists() or p.stat().st_size == 0:
        return False
    with open(p) as f:
        reader = csv.DictReader(f)
        if "atm_call_iv" not in (reader.fieldnames or []):
            return False
        for _ in reader:
            return True
    return False


def emit_result(ticker: str, year: str, status: str, rows: int):
    line = (STDOUT_MARKER + '{"ticker":"%s","year":"%s","status":"%s","rows":%d}'
            % (ticker, year, status, rows))
    sys.stderr.write(line + "\n")
    sys.stderr.flush()


def main():
    args = parse_args()
    tickers = load_tickers(args)
    years = parse_years(args.year)
    agg = args.aggregate
    overall_start = time.time()

    ok = no_data = failed = skipped = 0
    for year in years:
        for i, ticker in enumerate(tickers, 1):
            if args.resume and is_complete(ticker, year, agg, args.output):
                print("[%d/%d] %s (%s) -> complete, skipping" % (i, len(tickers), ticker, year))
                emit_result(ticker, year, "skipped", 0)
                skipped += 1
                continue

            t0 = time.time()
            try:
                proc_path, written = process_ticker(ticker, year, agg, args.output)
            except Exception as e:
                print("[%d/%d] %s (%s) -> FAILED: %s" % (i, len(tickers), ticker, year, e))
                emit_result(ticker, year, "failed", 0)
                failed += 1
                continue
            elapsed = time.time() - t0

            if proc_path is None:
                print("[%d/%d] %s (%s) -> no chain source file (%.1fs)"
                      % (i, len(tickers), ticker, year, elapsed))
                emit_result(ticker, year, "no_data", 0)
                no_data += 1
                continue

            if written == 0:
                if proc_path.exists():
                    try:
                        proc_path.unlink()
                    except OSError:
                        pass
                print("[%d/%d] %s (%s) -> no feature rows (%.1fs)"
                      % (i, len(tickers), ticker, year, elapsed))
                emit_result(ticker, year, "no_data", 0)
                no_data += 1
                continue

            if not args.no_rename:
                final = out_path(ticker, year, agg, args.output)
                final.parent.mkdir(parents=True, exist_ok=True)
                try:
                    proc_path.replace(final)
                except OSError:
                    pass

            print("[%d/%d] %s (%s) -> %d rows (%.1fs)"
                  % (i, len(tickers), ticker, year, written, elapsed))
            emit_result(ticker, year, "ok", written)
            ok += 1

    total = time.time() - overall_start
    print("=" * 60)
    print("SUMMARY: %d ok, %d no-data, %d failed, %d skipped (%.1fs)"
          % (ok, no_data, failed, skipped, total))


if __name__ == "__main__":
    main()
