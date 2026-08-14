"""
Parallel dispatcher for update_ts_AWST_to_UTC.py (in-place AWST→UTC conversion).

Spawns N worker subprocesses (each running update_ts_AWST_to_UTC.py) and
distributes tickers across them. One ticker per worker at a time.

State file: data/SPY/.parallel_state_<year>_<aggregate>_utc.json
  Records completed, in-progress, and timing metrics per ticker.
  Used by update_ts_AWST_to_UTC_parallel_status.py for live monitoring.

Usage:
    python scripts/options/update_ts_AWST_to_UTC_parallel.py --ohlcv_tickers --year 2025 --aggregate 1sec --spawn 16
    python scripts/options/update_ts_AWST_to_UTC_parallel.py --ohlcv_tickers --year 2003-2025 --aggregate 1sec --spawn 16
    python scripts/options/update_ts_AWST_to_UTC_parallel.py --tickers_file data/universes/2025/combined_unique.csv --year 2025 --aggregate 1min --spawn 16
"""

import argparse
import csv
import datetime
import gzip
import json
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
WORKER_SCRIPT = SCRIPT_DIR / "update_ts_AWST_to_UTC.py"
STDOUT_MARKER = "PARALLEL_RESULT:"
SCRIPT_NAME = Path(__file__).resolve().stem

AGGREGATE_MAP = {
    "1sec": "1sec", "1min": "1min", "5min": "5min",
    "15min": "15min", "1H": "1H", "4H": "4H", "1D": "1D",
}


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
        description="Parallel update of OHLCV timestamps from AWST to UTC"
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
        help="Path to CSV with ticker list (header 'ticker'; may also have market_cap,rank columns)",
    )
    parser.add_argument(
        "--ohlcv_tickers",
        action="store_true",
        default=False,
        help="Derive ticker list from saved OHLCV files in data/SPY/<aggregate>/<year>/",
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
        "--spawn",
        type=int,
        required=True,
        help="Number of parallel worker processes to spawn",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip tickers whose file is already timezone-aware",
    )
    parser.add_argument(
        "--logs",
        action="store_true",
        default=False,
        help="Save a dispatcher log file (default: False)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Base output directory passed to workers via --output (default: data/SPY)",
    )
    return parser.parse_args()


def load_tickers(tickers_file: str) -> list[str]:
    tickers = []
    with open(tickers_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = row.get("ticker", "").strip()
            if t:
                tickers.append(clean_ticker(t))
    if not tickers:
        raise SystemExit("Error: no tickers found in %s" % tickers_file)
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


def load_ohlcv_tickers(year: str, agg: str, output_dir: str | None = None) -> list[str]:
    src_dir = data_dir(agg, output_dir) / year
    if not src_dir.exists():
        return []
    suffix = f"_{year}_{AGGREGATE_MAP[agg]}.csv"
    tickers = []
    for f in sorted(src_dir.iterdir()):
        name = f.name
        if ".tmp" in name:
            continue
        if name.endswith(".gz"):
            name = name[:-3]
        if not name.endswith(suffix):
            continue
        ticker = name[:-len(suffix)]
        if ticker:
            tickers.append(clean_ticker(ticker))
    return tickers


def is_converted(ticker: str, year: str, agg: str, output_dir: str | None = None) -> bool:
    path = find_file(ticker, year, agg, output_dir)
    if path is None:
        return False
    try:
        if path.name.endswith(".gz"):
            f = gzip.open(path, "rt")
        else:
            f = open(path, "r")
        with f:
            f.readline()
            line = f.readline().rstrip("\r\n")
            if not line.strip():
                return False
            ts = line.split(",")[0].strip()
            return datetime.datetime.fromisoformat(ts).tzinfo is not None
    except (ValueError, OSError):
        return False


def tx_key(ticker: str, year: str) -> str:
    return f"{ticker}_{year}"


def load_state(state_path: Path) -> dict:
    if state_path.exists():
        with open(state_path) as f:
            return json.load(f)
    return {"completed": {}, "in_progress": {}, "all_tickers": [], "stats": {}}


def save_state(state_path: Path, state: dict):
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    tmp.replace(state_path)


def parse_worker_status(proc, ret: int, result_line: str | None = None) -> str:
    if ret != 0:
        return "failed"
    if result_line and STDOUT_MARKER in result_line:
        try:
            result = json.loads(result_line.split(STDOUT_MARKER)[1].strip())
            return result.get("status", "ok")
        except Exception:
            pass
    return "ok"


def _drain_worker_stderr(entry):
    proc = entry["proc"]
    if not proc or not proc.stderr:
        return
    while select.select([proc.stderr], [], [], 0)[0]:
        line = proc.stderr.readline()
        if not line:
            break
        entry["stderr_file"].write(line)
        entry["stderr_file"].flush()
        if STDOUT_MARKER in line:
            entry["_result_line"] = line.strip()


def fmt_duration(s: float) -> str:
    if s < 60:
        return "%.1fs" % s
    m = int(s // 60)
    sec = s % 60
    if m < 60:
        return "%dm %02.0fs" % (m, sec)
    h = m // 60
    m = m % 60
    return "%dh %02dm" % (h, m)


def run_year(year: str, args) -> dict:
    year_start = time.time()
    agg = args.aggregate
    folder = AGGREGATE_MAP[agg]

    state_base = Path(args.output) if args.output else Path("data") / "SPY"
    state_path = state_base / f".parallel_state_{year}_{folder}_utc.json"
    state = load_state(state_path)

    if args.ohlcv_tickers:
        raw_tickers = load_ohlcv_tickers(year, agg, args.output)
    elif args.tickers_file:
        raw_tickers = load_tickers(args.tickers_file)
    elif args.tickers:
        raw_tickers = [clean_ticker(t) for t in args.tickers.split(",") if t.strip()]
    else:
        raise SystemExit("Error: specify one of --tickers, --tickers_file, or --ohlcv_tickers")

    log_fh = None
    if args.logs:
        log_dir = state_base / folder / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        log_fh = open(log_dir / f"parallel_utc_{year}_{folder}_{log_ts}.log", "w")

    def log(msg: str, end: str = "\n"):
        print(msg, end=end, flush=True)
        if log_fh:
            log_fh.write(msg + (end if end else "\n"))
            log_fh.flush()

    if not raw_tickers:
        log("No OHLCV files for year %s (aggregate %s). Skipping." % (year, agg))
        if log_fh:
            log_fh.close()
        return {"year": year, "total": 0, "completed": 0, "ok": 0, "ignored": 0,
                "no_data": 0, "failed": 0, "elapsed_s": 0, "skipped": True}

    pre_skipped = 0
    filtered = []
    for t in raw_tickers:
        if is_converted(t, year, agg, args.output):
            pre_skipped += 1
            path = find_file(t, year, agg, args.output)
            log("[IGNORED] %s (%s): %s already timezone-aware, skipping"
                % (t, year, path.name if path else "file"))
            continue
        filtered.append(t)
    all_tickers = filtered

    state["all_tickers"] = all_tickers

    ticker_queue = []
    for t in all_tickers:
        key = tx_key(t, year)
        if key in state.get("completed", {}):
            continue
        if key in state.get("in_progress", {}):
            continue
        ticker_queue.append(t)

    total = len(all_tickers)
    remaining = len(ticker_queue)
    completed_count = total - remaining + pre_skipped

    state["config"] = {"workers": args.spawn, "year": year, "aggregate": agg}
    ticker_source = "ohlcv_tickers" if args.ohlcv_tickers else (args.tickers_file or args.tickers)
    log("=" * 60)
    log("PARALLEL AWST->UTC TIMESTAMP UPDATE  [year %s, aggregate %s]" % (year, agg))
    log("  Workers:      %d" % args.spawn)
    log("  Ticker src:   %s" % ticker_source)
    log("  Already UTC:  %d skipped" % pre_skipped)
    log("  Total:        %d tickers" % total)
    log("  Already done: %d" % completed_count)
    log("  Remaining:    %d" % remaining)
    log("  State file:   %s" % state_path)
    log("=" * 60)

    if remaining == 0:
        log("All tickers for %s already converted. Nothing to do." % year)
        if log_fh:
            log_fh.close()
        return {"year": year, "total": total, "completed": 0, "ok": 0, "ignored": 0,
                "no_data": 0, "failed": 0, "elapsed_s": 0, "skipped": True}

    state["in_progress"] = {}
    save_state(state_path, state)

    queue = list(ticker_queue)
    queue.reverse()

    active_workers: list[dict] = []

    def spawn_worker(ticker: str) -> subprocess.Popen | None:
        key = tx_key(ticker, year)
        cmd = [
            sys.executable,
            str(WORKER_SCRIPT),
            "--tickers", ticker,
            "--year", year,
            "--aggregate", agg,
        ]
        if args.output:
            cmd.extend(["--output", args.output])
        try:
            stderr_file = open(f"/tmp/worker_utc_{ticker}_{year}.log", "w")
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as e:
            log("[ERROR] Failed to spawn worker for %s: %s" % (ticker, e))
            return None
        entry = {
            "ticker": ticker,
            "pid": proc.pid,
            "proc": proc,
            "start_time": time.time(),
            "stderr_file": stderr_file,
        }
        state["in_progress"][key] = {"pid": proc.pid, "start_time": entry["start_time"]}
        state["stats"] = {"elapsed_s": round(time.time() - year_start, 1),
                          "total_tickers": total, "completed": len(state["completed"]),
                          "running": len(state["in_progress"]), "workers": args.spawn}
        save_state(state_path, state)
        active_workers.append(entry)
        return proc

    for _ in range(min(args.spawn, len(queue))):
        ticker = queue.pop()
        spawn_worker(ticker)

    def reap_finished():
        still_active = []
        for entry in active_workers:
            proc = entry["proc"]
            ret = proc.poll()
            if ret is not None:
                ticker = entry["ticker"]
                key = tx_key(ticker, year)
                duration = time.time() - entry["start_time"]

                _drain_worker_stderr(entry)
                worker_status = parse_worker_status(proc, ret, entry.get("_result_line"))

                entry["stderr_file"].close()
                if proc.stderr:
                    proc.stderr.close()
                state["completed"][key] = {
                    "ticker": ticker,
                    "duration_s": round(duration, 1),
                    "returncode": ret,
                    "status": worker_status,
                    "timestamp": datetime.datetime.now().isoformat(),
                }
                if key in state["in_progress"]:
                    del state["in_progress"][key]
                state["stats"] = {"elapsed_s": round(time.time() - year_start, 1),
                                  "total_tickers": total, "completed": len(state["completed"]),
                                  "running": len(state["in_progress"]), "workers": args.spawn}
                save_state(state_path, state)
                if worker_status == "ignored":
                    log("[IGNORED] %s (%s): file already timezone-aware, skipped" % (ticker, year))
                else:
                    log("[%s] %s (%s) finished in %.1fs" % (worker_status.upper(), ticker, year, duration))
            else:
                still_active.append(entry)
        active_workers[:] = still_active

    try:
        while active_workers or queue:
            for entry in active_workers:
                _drain_worker_stderr(entry)
            reap_finished()
            while len(active_workers) < args.spawn and queue:
                ticker = queue.pop()
                spawn_worker(ticker)
            if active_workers:
                time.sleep(1)
    except KeyboardInterrupt:
        log("\n[INTERRUPT] Terminating workers ...")
        for entry in active_workers:
            try:
                os.kill(entry["proc"].pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for entry in active_workers:
            entry["proc"].wait(timeout=5)
        log("All workers terminated.")
        sys.exit(130)

    year_elapsed = time.time() - year_start

    completed_ok = [k for k, v in state["completed"].items() if v.get("status") == "ok"]
    completed_ignored = [k for k, v in state["completed"].items() if v.get("status") == "ignored"]
    completed_no_data = [k for k, v in state["completed"].items() if v.get("status") == "no_data"]
    completed_fail = [k for k, v in state["completed"].items() if v.get("status") == "failed"]

    all_durations = [v["duration_s"] for v in state["completed"].values()]

    stats = {
        "total_tickers": total,
        "completed": len(state["completed"]),
        "ok": len(completed_ok),
        "ignored": len(completed_ignored),
        "no_data": len(completed_no_data),
        "failed": len(completed_fail),
        "elapsed_s": round(year_elapsed, 1),
        "workers": args.spawn,
    }
    if all_durations:
        stats["avg_time_s"] = round(sum(all_durations) / len(all_durations), 1)
        stats["min_time_s"] = round(min(all_durations), 1)
        stats["max_time_s"] = round(max(all_durations), 1)
    state["stats"] = stats
    save_state(state_path, state)

    log("=" * 60)
    log("SUMMARY  [year %s, aggregate %s]" % (year, agg))
    log("  Total:        %d" % total)
    log("  Updated:      %d" % len(completed_ok))
    log("  Ignored:      %d" % len(completed_ignored))
    log("  No data:      %d" % len(completed_no_data))
    log("  Failed:       %d" % len(completed_fail))
    log("  Duration:     %s" % fmt_duration(year_elapsed))
    if all_durations:
        log("  Avg/ticker:   %.1fs" % stats["avg_time_s"])
    log("=" * 60)

    if log_fh:
        log_fh.close()

    return {
        "year": year,
        "total": total,
        "completed": len(state["completed"]),
        "ok": len(completed_ok),
        "ignored": len(completed_ignored),
        "no_data": len(completed_no_data),
        "failed": len(completed_fail),
        "elapsed_s": round(year_elapsed, 1),
        "skipped": False,
    }


def main():
    args = parse_args()
    overall_start = time.time()
    years = parse_years(args.year)

    all_year_results = []
    for year in years:
        result = run_year(year, args)
        all_year_results.append(result)

    total_elapsed = time.time() - overall_start
    total_ok = sum(r["ok"] for r in all_year_results)
    total_ignored = sum(r["ignored"] for r in all_year_results)
    total_no_data = sum(r["no_data"] for r in all_year_results)
    total_failed = sum(r["failed"] for r in all_year_results)

    print()
    print("=" * 60)
    print("OVERALL SUMMARY (%d year(s): %s)" % (len(years), args.year))
    print("  Total elapsed:    %s" % datetime.timedelta(seconds=int(total_elapsed)))
    print("  Updated:          %d" % total_ok)
    print("  Ignored:          %d" % total_ignored)
    print("  No data:          %d" % total_no_data)
    print("  Failed:           %d" % total_failed)
    print("=" * 60)


if __name__ == "__main__":
    main()
