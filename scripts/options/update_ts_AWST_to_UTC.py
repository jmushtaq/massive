"""
Update the timestamp column of OHLCV files from AWST (naive local time) to UTC
(timezone-aware), in place.

Files live at data/SPY/<aggregate>/<year>/<ticker>_<year>_<aggregate>.csv (or
.csv.gz). Timestamps written by stocks_aggs_download.py without --UTC are naive
and assume the machine's local timezone, Australia/Perth (AWST, UTC+8, no DST).
This script re-interprets each naive timestamp as AWST and rewrites it in UTC
with a +00:00 offset, matching what stocks_aggs_download.py --UTC produces.

Files that already contain timezone-aware timestamps (e.g. ...+00:00) are left
untouched and reported as "ignored".

Usage:
    python scripts/options/update_ts_AWST_to_UTC.py --tickers AAPL --year 2025 --aggregate 1min
    python scripts/options/update_ts_AWST_to_UTC.py --tickers_file data/universes/2025/combined_unique.csv --year 2025 --aggregate 1sec
"""

import argparse
import csv
import datetime
import gzip
import os
import sys
import time
from pathlib import Path

AGGREGATE_MAP = {
    "1sec": "1sec", "1min": "1min", "5min": "5min",
    "15min": "15min", "1H": "1H", "4H": "4H", "1D": "1D",
}

AWST = datetime.timezone(datetime.timedelta(hours=8))
UTC = datetime.timezone.utc

STDOUT_MARKER = "PARALLEL_RESULT:"


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
    raise SystemExit("Error: invalid year format '%s' (use YYYY or YYYY-YYYY)" % year_arg)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Update OHLCV timestamps from AWST to UTC (in place)"
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated list of ticker symbols (e.g. AAPL,TSLA,NVDA)",
    )
    parser.add_argument(
        "--tickers_file",
        type=str,
        default=None,
        help="Path to CSV with ticker list (header 'ticker')",
    )
    parser.add_argument(
        "--year",
        type=str,
        required=True,
        help="Year or year range (e.g. 2025 or 2003-2025)",
    )
    parser.add_argument(
        "--aggregate",
        choices=list(AGGREGATE_MAP.keys()),
        default="1min",
        help="Aggregate window (default: 1min)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Base output directory (default: data/SPY)",
    )
    return parser.parse_args()


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
    if not tickers:
        raise SystemExit("Error: specify at least one of --tickers or --tickers_file")
    return tickers


def data_dir(agg: str, output_dir: str | None = None) -> Path:
    base = Path(output_dir) if output_dir else Path("data") / "SPY"
    return base / AGGREGATE_MAP[agg]


def find_file(ticker: str, year: str, agg: str, output_dir: str | None = None) -> Path | None:
    d = data_dir(agg, output_dir) / year
    folder = AGGREGATE_MAP[agg]
    for ext in ("", ".gz"):
        p = d / f"{ticker}_{year}_{folder}.csv{ext}"
        if p.exists():
            return p
    return None


def timestamp_aware(ts: str) -> bool:
    try:
        return datetime.datetime.fromisoformat(ts.strip()).tzinfo is not None
    except ValueError:
        return True


def convert_timestamp(ts: str) -> str:
    dt = datetime.datetime.fromisoformat(ts.strip())
    if dt.tzinfo is not None:
        return dt.isoformat()
    return dt.replace(tzinfo=AWST).astimezone(UTC).isoformat()


def convert_line(line: str, ts_idx: int) -> str:
    parts = line.split(",")
    parts[ts_idx] = convert_timestamp(parts[ts_idx])
    return ",".join(parts)


def verify_converted(tmp_path: Path, expected_header: str, ts_idx: int,
                     expected_rows: int, gz: bool) -> bool:
    """Re-open the staged tmp file and confirm it is a valid, fully timezone-aware
    (UTC) copy with the same header and row count before replacing the original.
    Returns False on any discrepancy so the original is left untouched."""
    try:
        if gz:
            f = gzip.open(tmp_path, "rt")
        else:
            f = open(tmp_path, "r", newline="")
        with f:
            header = f.readline().rstrip("\r\n")
            if header != expected_header:
                return False
            cols = header.split(",")
            if not (0 <= ts_idx < len(cols)) or cols[ts_idx] != "timestamp":
                return False
            rows = 0
            for line in f:
                s = line.rstrip("\r\n")
                if not s.strip():
                    continue
                parts = s.split(",")
                if len(parts) != len(cols):
                    return False
                try:
                    dt = datetime.datetime.fromisoformat(parts[ts_idx].strip())
                except ValueError:
                    return False
                if dt.tzinfo is None or dt.utcoffset() != datetime.timedelta(0):
                    return False
                rows += 1
        return rows == expected_rows
    except (OSError, ValueError):
        return False


def process_file(path: Path) -> tuple[str, int]:
    gz = path.name.endswith(".gz")

    if gz:
        in_f = gzip.open(path, "rt")
    else:
        in_f = open(path, "r")

    with in_f:
        header = in_f.readline().rstrip("\r\n")
        if not header:
            return "no_data", 0
        cols = header.split(",")
        try:
            ts_idx = cols.index("timestamp")
        except ValueError:
            return "failed", 0

        first = in_f.readline().rstrip("\r\n")
        if not first.strip():
            return "no_data", 0
        if timestamp_aware(first.split(",")[ts_idx]):
            return "ignored", 0

        tmp = path.with_name(path.name + ".tmp")
        count = 0
        try:
            if gz:
                out_f = gzip.open(tmp, "wt", compresslevel=6)
            else:
                out_f = open(tmp, "w", newline="")
            with out_f:
                out_f.write(header + "\n")
                out_f.write(convert_line(first, ts_idx) + "\n")
                count = 1
                for line in in_f:
                    s = line.rstrip("\r\n")
                    if not s.strip():
                        continue
                    out_f.write(convert_line(s, ts_idx) + "\n")
                    count += 1
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise

    if not verify_converted(tmp, header, ts_idx, count, gz):
        if tmp.exists():
            tmp.unlink()
        return "failed", 0

    os.replace(tmp, path)
    return "ok", count


def emit_result(ticker: str, year: str, status: str, rows: int):
    line = (STDOUT_MARKER + '{"ticker":"%s","year":"%s","status":"%s","rows":%d}'
            % (ticker, year, status, rows))
    sys.stderr.write(line + "\n")
    sys.stderr.flush()


def main():
    args = parse_args()
    tickers = load_tickers(args)
    years = parse_years(args.year)
    overall_start = time.time()

    ok = ignored = no_data = failed = 0

    for year in years:
        for i, ticker in enumerate(tickers, 1):
            path = find_file(ticker, year, args.aggregate, args.output)
            if path is None:
                print("[%d/%d] %s (%s) -> no file found" % (i, len(tickers), ticker, year))
                emit_result(ticker, year, "no_data", 0)
                no_data += 1
                continue

            t0 = time.time()
            try:
                status, rows = process_file(path)
            except Exception as e:
                print("[%d/%d] %s (%s) -> FAILED: %s" % (i, len(tickers), ticker, year, e))
                emit_result(ticker, year, "failed", 0)
                failed += 1
                continue
            elapsed = time.time() - t0

            if status == "ignored":
                print("[%d/%d] %s (%s) -> [ignored] %s already timezone-aware, skipping"
                      % (i, len(tickers), ticker, year, path))
                emit_result(ticker, year, "ignored", 0)
                ignored += 1
            elif status == "no_data":
                print("[%d/%d] %s (%s) -> empty file" % (i, len(tickers), ticker, year))
                emit_result(ticker, year, "no_data", 0)
                no_data += 1
            else:
                print("[%d/%d] %s (%s) -> %d rows updated in %.1fs" % (i, len(tickers), ticker, year, rows, elapsed))
                emit_result(ticker, year, "ok", rows)
                ok += 1

    total_time = time.time() - overall_start
    print("=" * 60)
    print("SUMMARY: %d updated, %d ignored, %d no-data, %d failed (%.1fs)"
          % (ok, ignored, no_data, failed, total_time))


if __name__ == "__main__":
    main()
