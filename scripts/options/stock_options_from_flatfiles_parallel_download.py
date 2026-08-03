"""
Parallel dispatcher for stock_options_from_flatfiles_download.py.

Spawns N worker subprocesses and distributes tickers from a CSV file or from
saved OHLCV filenames across them. One ticker per worker at a time.

Streaming model: a background download thread populates a shared cache dir.
Workers are spawned as soon as enough days are ready, and re-spawned in waves
as more days become available.  Each worker uses --smart_resume so it only
processes newly-cached days.

State file: data/options/stocks/.parallel_state_<year>_<aggregate>_options.json

Usage:
    python scripts/options/stock_options_from_flatfiles_parallel_download.py --tickers AAPL,TSLA --year 2025 --spawn 2
    python scripts/options/stock_options_from_flatfiles_parallel_download.py --tickers_file data/universes/2025/combined_unique.csv --year 2025 --spawn 12
    python scripts/options/stock_options_from_flatfiles_parallel_download.py --ohlcv_tickers --year 2025 --spawn 100
    python scripts/options/stock_options_from_flatfiles_parallel_download.py --ohlcv_tickers --year 2025 --spawn 100 --aggregate 1D
    python scripts/options/stock_options_from_flatfiles_parallel_download.py --tickers_file foo.csv --year 2025 --spawn 10 --output data/combined
    python scripts/options/stock_options_from_flatfiles_parallel_download.py --ohlcv_tickers --year 2025 --spawn 40 --smart_resume --resume &
"""

import argparse
import csv
import datetime
import json
import os
import select
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(env_path)

SCRIPT_DIR = Path(__file__).resolve().parent
WORKER_SCRIPT = SCRIPT_DIR / "stock_options_from_flatfiles_download.py"

AGGREGATE_MAP = {
    "1sec": (1, "second", "1sec"),
    "1min": (1, "minute", "1min"),
    "5min": (5, "minute", "5min"),
    "15min": (15, "minute", "15min"),
    "1H": (1, "hour", "1H"),
    "4H": (4, "hour", "4H"),
    "1D": (1, "day", "1D"),
}

STDOUT_MARKER = "PARALLEL_RESULT:"
SCRIPT_NAME = Path(__file__).resolve().stem

S3_ENDPOINT = "https://files.massive.com"
S3_BASE = "s3://flatfiles/us_options_opra"


def trading_days(year: str) -> list[str]:
    import datetime as _dt
    start = _dt.date(int(year), 1, 1)
    end = _dt.date(int(year), 12, 31)
    days = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            days.append(d.strftime("%Y-%m-%d"))
        d += _dt.timedelta(days=1)
    return days


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
        if "404" not in stderr and "Not Found" not in stderr:
            print(f"  [WARN] S3 download failed: {stderr.strip()}" if stderr.strip() else f"  [WARN] S3 download failed: {e}", flush=True)
        return False
    except Exception as e:
        print(f"  [WARN] S3 download error: {e}", flush=True)
        return False


def clean_ticker(raw: str) -> str:
    return raw.strip().upper().split("-")[0]


def parse_years(year_arg: str) -> list[str]:
    parts = year_arg.split("-")
    if len(parts) == 1:
        y = parts[0].strip()
        if not y.isdigit():
            raise SystemExit("Error: invalid year '%s'" % year_arg)
        return [y]
    elif len(parts) == 2:
        start, end = parts[0].strip(), parts[1].strip()
        if not start.isdigit() or not end.isdigit():
            raise SystemExit("Error: invalid year range '%s'" % year_arg)
        return [str(y) for y in range(int(start), int(end) + 1)]
    else:
        raise SystemExit("Error: invalid year format '%s' (use YYYY or YYYY-YYYY)" % year_arg)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parallel download 1min options features from S3 flat files"
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
        default=None,
        help="Comma-separated ticker symbols (e.g. AAPL,TSLA,NVDA)",
    )
    parser.add_argument(
        "--tickers_file",
        type=str,
        default=None,
        help="Path to CSV with ticker list (header 'ticker')",
    )
    parser.add_argument(
        "--ohlcv_tickers",
        action="store_true",
        default=False,
        help="Derive ticker list from saved OHLCV files in data/SPY/1min/<year>/",
    )
    parser.add_argument(
        "--year",
        type=str,
        required=True,
        help="Year to download (e.g. 2025)",
    )
    parser.add_argument(
        "--spawn",
        type=int,
        required=True,
        help="Number of parallel worker processes to spawn",
    )
    parser.add_argument(
        "--smart_resume",
        action="store_true",
        default=False,
        help="Pass --smart_resume to workers: read processing file and resume from last date.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip tickers that already have a non-empty output file",
    )
    parser.add_argument(
        "--logs",
        action="store_true",
        default=False,
        help="Save a dispatcher log file",
    )
    parser.add_argument(
        "--skip_completed",
        action="store_true",
        default=True,
        help="Skip tickers that already have a non-empty output file (default: True)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Base output directory passed to workers via --output (default: data/)",
    )
    parser.add_argument(
        "--check",
        type=str,
        default=None,
        help="Directory to check for existing per-ticker files",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Sleep seconds between trading days in the worker (default: 0)",
    )
    parser.add_argument(
        "--download_only",
        action="store_true",
        default=False,
        help="Only download flat files to cache (no processing). Uses --spawn for parallel S3 downloads.",
    )
    parser.add_argument(
        "--use_local_cache",
        action="store_true",
        default=False,
        help="Use only pre-downloaded files in tmp/options_cache_<year>/ — no S3 calls. Mutually exclusive with --download_only.",
    )
    parser.add_argument(
        "--use_unzipped",
        type=lambda s: s.lower() == "true",
        default=True,
        help="When using --use_local_cache, expect .csv files (True) or .csv.gz files (False). Default: True.",
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
        default=False,
        help="Read the state file and move completed processing/ files to final. Useful after a crashed or interrupted run.",
    )
    parser.add_argument(
        "--min_first_wave",
        type=int,
        default=5,
        help="Start processing after this many day files are cached (default: 5)",
    )
    parser.add_argument(
        "--min_new_wave",
        type=int,
        default=10,
        help="Spawn a new wave of workers after this many new day files appear (default: 10)",
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


def load_ohlcv_tickers(year: str) -> list[str]:
    src_dir = Path("data") / "SPY" / "1min" / year
    if not src_dir.exists():
        raise SystemExit("Error: OHLCV directory not found: %s" % src_dir)
    tickers = []
    for f in sorted(src_dir.glob(f"*_{year}_1min.csv*")):
        name = f.name.replace(".csv.gz", "").replace(".csv", "")
        ticker = name.split("_")[0]
        tickers.append(clean_ticker(ticker))
    if not tickers:
        raise SystemExit("Error: no OHLCV files found in %s" % src_dir)
    return tickers


def output_path(ticker: str, year: str, agg: str, output_dir: str | None = None, subdir: str | None = None) -> Path:
    folder = AGGREGATE_MAP[agg][2]
    base = Path(output_dir) if output_dir else Path("data")
    p = base / "options" / "stocks" / folder / year
    if subdir:
        p = p / subdir
    return p / f"{ticker}_{year}_{folder}_options.csv"


def tx_key(ticker: str, year: str) -> str:
    return f"{ticker}_{year}"


def is_ticker_final(ticker: str, year: str, agg: str, output_dir: str | None = None) -> bool:
    p = output_path(ticker, year, agg, output_dir)
    if not p.exists() or p.stat().st_size == 0:
        return False
    with open(p) as f:
        return sum(1 for _ in f) > 1


def _has_data_rows(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    with open(path) as f:
        return sum(1 for _ in f) > 1


def load_state(state_path: Path):
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


def run_year(agg: str, year: str, args) -> dict:
    year_start = time.time()
    folder = AGGREGATE_MAP[agg][2]

    state_base = Path(args.output) if args.output else Path("data")
    state_path = state_base / "options" / "stocks" / f".parallel_state_{year}_{folder}_options.json"
    state = load_state(state_path)

    # --- Download-only mode (no tickers needed) ---
    if args.download_only:
        cache_dir = f"tmp/options_cache_{year}"
        os.makedirs(cache_dir, exist_ok=True)
        day_list = trading_days(year)
        missing = []
        for day_str in day_list:
            p = os.path.join(cache_dir, f"{day_str}.csv.gz")
            if not os.path.exists(p) or os.path.getsize(p) == 0:
                missing.append(day_str)

        def log(msg: str, end: str = "\n"):
            print(msg, end=end, flush=True)

        log("")
        log("=" * 60)
        log("DOWNLOAD-ONLY  [year %s]" % year)
        log("  Total days:   %d" % len(day_list))
        log("  Cached:       %d" % (len(day_list) - len(missing)))
        log("  To download:  %d" % len(missing))
        log("  Spawn:        %d" % args.spawn)
        log("  Cache dir:    %s" % cache_dir)
        log("=" * 60)

        if missing:
            log("")
            queue = list(missing)
            queue.reverse()
            active_procs: list[tuple] = []
            while queue or active_procs:
                while len(active_procs) < args.spawn and queue:
                    day_str = queue.pop()
                    y, m, d = day_str.split("-")
                    s3_path = f"{S3_BASE}/minute_aggs_v1/{y}/{m}/{day_str}.csv.gz"
                    local_path = os.path.join(cache_dir, f"{day_str}.csv.gz")
                    cmd = ["aws", "s3", "cp", s3_path, local_path, "--endpoint-url", S3_ENDPOINT]
                    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    active_procs.append((proc, day_str))
                    log("  [DL] %d/%d: %s" % (len(missing) - len(queue), len(missing), day_str))
                still_active: list[tuple] = []
                for proc, day_str in active_procs:
                    if proc.poll() is None:
                        still_active.append((proc, day_str))
                active_procs = still_active
                if active_procs:
                    time.sleep(0.5)
            log("")
            log("  All %d files downloaded." % len(missing))
        else:
            log("  All files already cached.")
        elapsed = time.time() - year_start
        return {"year": year, "total": 0, "completed": 0, "successful": 0, "no_data": 0, "failed": 0, "elapsed_s": round(elapsed, 1), "skipped": True}

    # --- Finalize-only mode: move completed processing/ files to final ---
    if args.finalize:
        if not state.get("completed"):
            print("No completed entries in state file %s" % state_path)
            return {"year": year, "total": 0, "completed": 0, "successful": 0, "no_data": 0, "failed": 0, "elapsed_s": 0, "skipped": True}

        print("=" * 60)
        print("FINALIZE  [year %s]" % year)
        print("  State file:   %s" % state_path)
        print("  Completed:    %d" % len(state["completed"]))
        print("=" * 60)
        finalized_ok = 0
        finalized_nodata = 0
        for key, info in state["completed"].items():
            ticker = info.get("ticker", key.rsplit("_", 1)[0])
            if info.get("status") == "failed":
                continue
            proc_path = output_path(ticker, year, agg, args.output, subdir="processing")
            if not proc_path.exists():
                continue
            if _has_data_rows(proc_path):
                final_path = output_path(ticker, year, agg, args.output)
                final_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    proc_path.rename(final_path)
                    finalized_ok += 1
                    print("  [OK]    %s -> %s" % (ticker, final_path.name))
                except OSError as e:
                    print("  [ERR]   %s rename failed: %s" % (ticker, e))
            elif proc_path.stat().st_size > 0:
                no_data_dir = output_path(ticker, year, agg, args.output, subdir="no_data").parent
                no_data_dir.mkdir(parents=True, exist_ok=True)
                try:
                    proc_path.rename(no_data_dir / proc_path.name)
                    finalized_nodata += 1
                    print("  [NODATA] %s -> no_data/%s" % (ticker, proc_path.name))
                except OSError as e:
                    print("  [ERR]   %s rename failed: %s" % (ticker, e))
        print()
        print("  Finalized OK:    %d" % finalized_ok)
        print("  Finalized NODATA: %d" % finalized_nodata)
        print("=" * 60)
        return {"year": year, "total": len(state["completed"]), "completed": finalized_ok + finalized_nodata, "successful": finalized_ok, "no_data": finalized_nodata, "failed": 0, "elapsed_s": 0, "skipped": False}

    # --- Processing mode: need tickers ---
    if args.ohlcv_tickers:
        all_tickers = load_ohlcv_tickers(year)
    elif args.tickers:
        all_tickers = [clean_ticker(t) for t in args.tickers.split(",") if t.strip()]
    elif args.tickers_file:
        all_tickers = load_tickers(args.tickers_file)
    else:
        raise SystemExit("Error: specify one of --tickers, --tickers_file, or --ohlcv_tickers")

    if args.skip_completed:
        pre_filtered = []
        for t in all_tickers:
            if is_ticker_final(t, year, agg, args.output):
                continue
            pre_filtered.append(t)
        skipped_pre = len(all_tickers) - len(pre_filtered)
        all_tickers = pre_filtered
    else:
        skipped_pre = 0

    log_lines: list[str] = []

    def log(msg: str, end: str = "\n"):
        print(msg, end=end, flush=True)
        log_lines.append(msg)
        if log_fh:
            log_fh.write(msg + ("\n" if end == "\n" else end))
            log_fh.flush()

    log_fh = None
    if args.logs:
        log_dir = state_base / "options" / "stocks" / folder / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        log_path = log_dir / f"parallel_{year}_{folder}_options_{log_ts}.log"
        log_fh = open(log_path, "w")

    if args.check:
        check_filtered = []
        for t in all_tickers:
            check_file = Path(args.check) / "options" / "stocks" / folder / year / f"{t}_{year}_{folder}_options.csv"
            if check_file.exists() and check_file.stat().st_size > 0:
                continue
            check_filtered.append(t)
        checked_skipped = len(all_tickers) - len(check_filtered)
        if checked_skipped:
            log("  Check dir skipped: %d tickers (file exists in %s)" % (checked_skipped, args.check))
        all_tickers = check_filtered
        skipped_pre += checked_skipped

    state["all_tickers"] = all_tickers

    ticker_queue = []
    for t in all_tickers:
        key = tx_key(t, year)
        if key in state.get("completed", {}):
            continue
        if args.resume:
            if is_ticker_final(t, year, agg, args.output):
                continue
        if key in state.get("in_progress", {}):
            continue
        ticker_queue.append(t)

    total = len(all_tickers)
    remaining = len(ticker_queue)
    completed_count = total - remaining + skipped_pre

    state["config"] = {"workers": args.spawn, "aggregate": agg, "year": year}
    ticker_source = "ohlcv_tickers" if args.ohlcv_tickers else args.tickers_file
    log("=" * 60)
    log("PARALLEL OPTIONS 1MIN FEATURES (flat files)  [year %s]" % year)
    log("  Workers:      %d" % args.spawn)
    log("  Aggregate:    %s" % agg)
    log("  Resume:       %s" % args.resume)
    log("  Logs:         %s" % ("enabled" if args.logs else "disabled"))
    if args.logs:
        log("  Log path:     %s" % log_path)
    log("  Ticker src:   %s" % ticker_source)
    log("  Skip compl:   %s" % args.skip_completed)
    if args.output:
        log("  Output base:  %s" % args.output)
    if args.check:
        log("  Check dir:    %s" % args.check)
    log("  Total:        %d tickers" % total)
    log("  Already done: %d" % completed_count)
    log("  Remaining:    %d" % remaining)
    log("  State file:   %s" % state_path)
    log("=" * 60)

    if remaining == 0:
        log("All tickers for %s already processed. Nothing to do." % year)
        if log_fh:
            log_fh.close()
        return {"year": year, "total": total, "completed": 0, "successful": 0, "no_data": 0, "failed": 0, "elapsed_s": 0, "skipped": True}

    cache_dir = f"tmp/options_cache_{year}"
    day_list = trading_days(year)

    # --- Local-cache-only mode: no S3 calls, process all tickers against cached files ---
    if args.use_local_cache:
        os.makedirs(cache_dir, exist_ok=True)
        local_ext = ".csv" if args.use_unzipped else ".csv.gz"
        cached_count = sum(
            1 for d in day_list
            if os.path.exists(os.path.join(cache_dir, f"{d}{local_ext}"))
            and os.path.getsize(os.path.join(cache_dir, f"{d}{local_ext}")) > 0
        )
        log("  [LOCAL CACHE] %d/%d days cached, no S3 downloads" % (cached_count, len(day_list)))
        log("  [LOCAL CACHE] Processing %d tickers ..." % remaining)
        log("")
        errors_occurred = False

        pending = list(ticker_queue)
        pending.reverse()
        active = []
        last_status_time = time.time()

        def spawn_local_worker(ticker: str) -> tuple | None:
            nonlocal errors_occurred
            key = tx_key(ticker, year)
            cmd = [
                sys.executable,
                str(WORKER_SCRIPT),
                "--tickers", ticker,
                "--year", year,
                "--aggregate", agg,
                "--downloads_dir", cache_dir,
                "--use_unzipped", str(args.use_unzipped),
                "--no_rename",
            ]
            if args.smart_resume:
                cmd.append("--smart_resume")
            if args.resume:
                cmd.append("--resume")
            if args.output:
                cmd.extend(["--output", args.output])
            if args.delay:
                cmd.extend(["--delay", str(args.delay)])
            try:
                stderr_file = open(f"/tmp/worker_options_localcache_{ticker}_{year}.log", "w")
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            except Exception as e:
                log("[ERROR] Failed to spawn worker for %s: %s" % (ticker, e))
                errors_occurred = True
                return None
            entry = {
                "ticker": ticker,
                "pid": proc.pid,
                "proc": proc,
                "start_time": time.time(),
                "stderr_file": stderr_file,
                "_result_line": None,
            }
            state["in_progress"][key] = {"pid": proc.pid, "start_time": entry["start_time"]}
            state["stats"] = {"elapsed_s": round(time.time() - year_start, 1), "total_tickers": total, "completed": len(state["completed"]), "running": len(state["in_progress"])}
            save_state(state_path, state)
            return proc, entry

        try:
            while pending or active:
                # Spawn up to --spawn workers
                while len(active) < args.spawn and pending:
                    ticker = pending.pop()
                    # Filter already-completed tickers
                    key = tx_key(ticker, year)
                    if key in state.get("completed", {}):
                        continue
                    if key in state.get("in_progress", {}):
                        continue
                    result = spawn_local_worker(ticker)
                    if result:
                        _, entry = result
                        active.append(entry)

                if not active and not pending:
                    break

                # Poll workers
                still_active = []
                for entry in active:
                    proc = entry["proc"]
                    ret = proc.poll()
                    if ret is not None:
                        ticker = entry["ticker"]
                        key = tx_key(ticker, year)
                        duration = time.time() - entry["start_time"]
                        _drain_worker_stderr(entry)
                        worker_status = parse_worker_status(proc, ret, entry.get("_result_line"))
                        if worker_status == "failed":
                            errors_occurred = True
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
                        state["stats"] = {"elapsed_s": round(time.time() - year_start, 1), "total_tickers": total, "completed": len(state["completed"]), "running": len(state["in_progress"])}
                        save_state(state_path, state)
                        # Finalize processing → final immediately
                        proc_path = output_path(ticker, year, agg, args.output, subdir="processing")
                        if proc_path.exists() and _has_data_rows(proc_path):
                            final_path = output_path(ticker, year, agg, args.output)
                            final_path.parent.mkdir(parents=True, exist_ok=True)
                            try:
                                proc_path.rename(final_path)
                            except OSError:
                                pass
                        elif proc_path.exists() and proc_path.stat().st_size > 0:
                            no_data_dir = output_path(ticker, year, agg, args.output, subdir="no_data").parent
                            no_data_dir.mkdir(parents=True, exist_ok=True)
                            try:
                                proc_path.rename(no_data_dir / proc_path.name)
                            except OSError:
                                pass
                        # Log progress
                        completed = len(state["completed"])
                        log("  [%d/%d] %s done (%s, %.1fs)" % (completed, total, ticker, worker_status, duration))
                    else:
                        still_active.append(entry)
                active = still_active

                if active:
                    # Drain stderr periodically
                    now = time.time()
                    if now - last_status_time >= 5:
                        for entry in active:
                            _drain_worker_stderr(entry)
                        last_status_time = now
                    time.sleep(1)

        except KeyboardInterrupt:
            log("\n[INTERRUPT] Terminating workers ...")
            for entry in active:
                try:
                    os.kill(entry["proc"].pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            for entry in active:
                entry["proc"].wait(timeout=5)
            log("All workers terminated.")
            sys.exit(130)

        state["in_progress"] = {}
        save_state(state_path, state)

        # --- Finalize ---
        for t in all_tickers:
            proc = output_path(t, year, agg, args.output, subdir="processing")
            if not proc.exists() or proc.stat().st_size == 0:
                continue
            if _has_data_rows(proc):
                final = output_path(t, year, agg, args.output)
                final.parent.mkdir(parents=True, exist_ok=True)
                try:
                    proc.rename(final)
                except OSError:
                    pass
            else:
                no_data_dir = output_path(t, year, agg, args.output, subdir="no_data").parent
                no_data_dir.mkdir(parents=True, exist_ok=True)
                try:
                    proc.rename(no_data_dir / proc.name)
                except OSError:
                    pass

        year_elapsed = time.time() - year_start
        completed_ok = [k for k, v in state["completed"].items() if v.get("status") == "ok"]
        completed_no_data = [k for k, v in state["completed"].items() if v.get("status") == "no_data"]
        completed_fail = [k for k, v in state["completed"].items() if v.get("status") == "failed"]

        data_durations = [state["completed"][k]["duration_s"] for k in completed_ok]
        all_durations = [v["duration_s"] for v in state["completed"].values()]

        stats = {
            "total_tickers": total,
            "completed": len(state["completed"]),
            "successful": len(completed_ok),
            "no_data": len(completed_no_data),
            "failed": len(completed_fail),
            "elapsed_s": round(year_elapsed, 1),
        }
        if data_durations:
            stats["data_avg_time_s"] = round(sum(data_durations) / len(data_durations), 1)
            stats["data_min_time_s"] = round(min(data_durations), 1)
            stats["data_max_time_s"] = round(max(data_durations), 1)
        if all_durations:
            stats["avg_time_s"] = round(sum(all_durations) / len(all_durations), 1)
            stats["min_time_s"] = round(min(all_durations), 1)
            stats["max_time_s"] = round(max(all_durations), 1)
        state["stats"] = stats
        save_state(state_path, state)

        log("=" * 60)
        log("SUMMARY  [year %s]" % year)
        log("  Total:        %d" % total)
        log("  Successful:   %d" % len(completed_ok))
        log("  No data:      %d" % len(completed_no_data))
        log("  Failed:       %d" % len(completed_fail))
        log("  Duration:     %.1fs" % year_elapsed)
        if data_durations:
            log("  Avg/ticker (with data):   %.1fs" % stats["data_avg_time_s"])
            log("  Min/ticker (with data):   %.1fs" % stats["data_min_time_s"])
            log("  Max/ticker (with data):   %.1fs" % stats["data_max_time_s"])
        if all_durations:
            log("  Avg/ticker (all):         %.1fs" % stats["avg_time_s"])
            log("  Min/ticker (all):         %.1fs" % stats["min_time_s"])
            log("  Max/ticker (all):         %.1fs" % stats["max_time_s"])
        log("  State file:   %s" % state_path)
        if completed_fail:
            log("  Failed keys:  %s" % ", ".join(completed_fail))
        log("=" * 60)

        if log_fh:
            log_fh.close()

        log_ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        report = {
            "script": "stock_options_from_flatfiles_parallel_download",
            "timestamp": log_ts,
            "year": year,
            "workers": args.spawn,
            "resume": args.resume,
            "logs": args.logs,
            "use_local_cache": True,
            "total_tickers": total,
            "successful": len(completed_ok),
            "no_data": len(completed_no_data),
            "failed": len(completed_fail),
            "duration_s": round(year_elapsed, 1),
            "completed": state["completed"],
            "stats": stats,
        }
        report_dir = state_base / "options" / "stocks"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"parallel_report_{year}_{folder}_options.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        log("  Report:       %s" % report_path)

        return {
            "year": year,
            "total": total,
            "completed": len(completed_ok) + len(completed_no_data) + len(completed_fail),
            "successful": len(completed_ok),
            "no_data": len(completed_no_data),
            "failed": len(completed_fail),
            "elapsed_s": round(year_elapsed, 1),
            "skipped": False,
        }

    # --- Streaming pipeline (download + process in waves) ---
    os.makedirs(cache_dir, exist_ok=True)
    day_list = trading_days(year)

    min_first = args.min_first_wave
    min_new = args.min_new_wave

    log("")
    log("  [STREAM] %d days, cache: %s" % (len(day_list), cache_dir))
    log("  [STREAM] First wave after %d days ready, new waves after %d more" % (min_first, min_new))

    errors_occurred = False

    def count_ready() -> int:
        n = 0
        for day_str in day_list:
            p = os.path.join(cache_dir, f"{day_str}.csv.gz")
            if os.path.exists(p) and os.path.getsize(p) > 0:
                n += 1
        return n

    # Background download thread
    dl_done = threading.Event()

    def download_all():
        for day_str in day_list:
            y, m, d = day_str.split("-")
            s3_path = f"{S3_BASE}/minute_aggs_v1/{y}/{m}/{day_str}.csv.gz"
            local_path = os.path.join(cache_dir, f"{day_str}.csv.gz")
            if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                continue
            download_s3_file(s3_path, local_path)
        dl_done.set()

    dl_thread = threading.Thread(target=download_all, daemon=True)
    dl_thread.start()

    # --- Worker spawn / reap helpers ---

    def spawn_worker_stream(ticker: str) -> tuple | None:
        nonlocal errors_occurred
        key = tx_key(ticker, year)
        cmd = [
            sys.executable,
            str(WORKER_SCRIPT),
            "--tickers", ticker,
            "--year", year,
            "--aggregate", agg,
            "--downloads_dir", cache_dir,
            "--smart_resume",
            "--no_rename",
        ]
        if args.resume:
            cmd.append("--resume")
        if args.output:
            cmd.extend(["--output", args.output])
        if args.delay:
            cmd.extend(["--delay", str(args.delay)])
        try:
            stderr_file = open(f"/tmp/worker_options_flatfiles_{ticker}_{year}.log", "w")
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as e:
            log("[ERROR] Failed to spawn worker for %s: %s" % (ticker, e))
            errors_occurred = True
            return None
        entry = {
            "ticker": ticker,
            "pid": proc.pid,
            "proc": proc,
            "start_time": time.time(),
            "stderr_file": stderr_file,
        }
        state["in_progress"][key] = {"pid": proc.pid, "start_time": entry["start_time"]}
        state["stats"] = {"elapsed_s": round(time.time() - year_start, 1), "total_tickers": total, "completed": len(state["completed"]), "running": len(state["in_progress"])}
        save_state(state_path, state)
        return proc, entry

    def spawn_wave(tickers_to_spawn: list[str], active: list[dict]):
        for ticker in tickers_to_spawn:
            result = spawn_worker_stream(ticker)
            if result:
                _, entry = result
                active.append(entry)

    def reap_wave(active: list[dict]) -> bool:
        """Return True if all workers finished."""
        nonlocal errors_occurred
        still_active = []
        for entry in active:
            proc = entry["proc"]
            ret = proc.poll()
            if ret is not None:
                ticker = entry["ticker"]
                key = tx_key(ticker, year)
                duration = time.time() - entry["start_time"]
                _drain_worker_stderr(entry)
                worker_status = parse_worker_status(proc, ret, entry.get("_result_line"))
                if worker_status == "failed":
                    errors_occurred = True
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
                state["stats"] = {"elapsed_s": round(time.time() - year_start, 1), "total_tickers": total, "completed": len(state["completed"]), "running": len(state["in_progress"])}
                save_state(state_path, state)
            else:
                still_active.append(entry)
        active[:] = still_active
        return len(active) == 0

    # --- Main streaming loop ---

    # Wait for first N days
    log("  [STREAM] Waiting for %d days to be ready ..." % min_first)
    while count_ready() < min_first and not dl_done.is_set():
        time.sleep(2)

    active: list[dict] = []
    queue_base = list(ticker_queue)
    last_ready_count = count_ready()
    wave_idx = 0

    def launch_wave():
        nonlocal wave_idx, last_ready_count
        wave_idx += 1
        current = count_ready()
        log("  [WAVE %d] %d days ready, spawning up to %d workers" % (wave_idx, current, args.spawn))
        last_ready_count = current
        q = [t for t in queue_base
             if tx_key(t, year) not in state.get("completed", {})
             and tx_key(t, year) not in state.get("in_progress", {})]
        q.reverse()
        # Spawn initial batch of workers (up to --spawn)
        for _ in range(min(args.spawn, len(q))):
            ticker = q.pop()
            result = spawn_worker_stream(ticker)
            if result:
                _, entry = result
                active.append(entry)
        # Return remaining queue for the polling loop to drain
        return q

    pending_queue = launch_wave()

    try:
        while True:
            # Poll workers and re-fill from pending queue
            for _ in range(5):
                for entry in active:
                    _drain_worker_stderr(entry)
                reap_wave(active)
                # Spawn more workers from the pending queue
                while len(active) < args.spawn and pending_queue:
                    ticker = pending_queue.pop()
                    result = spawn_worker_stream(ticker)
                    if result:
                        _, entry = result
                        active.append(entry)
                if not active and not pending_queue:
                    break
                if active:
                    time.sleep(1)

            # All workers finished this wave and queue is empty
            dl_finished = dl_done.is_set()
            current_ready = count_ready()
            new_days = current_ready - last_ready_count

            if dl_finished and new_days == 0 and not active:
                log("  [STREAM] All days downloaded and all workers finished.")
                break

            if new_days >= min_new or (dl_finished and new_days > 0):
                launch_wave()
            elif not dl_finished:
                time.sleep(3)
            else:
                # Final stragglers
                if new_days > 0:
                    launch_wave()
                else:
                    break

    except KeyboardInterrupt:
        log("\n[INTERRUPT] Terminating workers ...")
        for entry in active:
            try:
                os.kill(entry["proc"].pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for entry in active:
            entry["proc"].wait(timeout=5)
        log("All workers terminated.")
        sys.exit(130)

    state["in_progress"] = {}
    save_state(state_path, state)

    # --- Finalize ---
    for t in all_tickers:
        proc = output_path(t, year, agg, args.output, subdir="processing")
        if not proc.exists() or proc.stat().st_size == 0:
            continue
        if _has_data_rows(proc):
            final = output_path(t, year, agg, args.output)
            final.parent.mkdir(parents=True, exist_ok=True)
            try:
                proc.rename(final)
            except OSError:
                pass
        else:
            no_data_dir = output_path(t, year, agg, args.output, subdir="no_data").parent
            no_data_dir.mkdir(parents=True, exist_ok=True)
            try:
                proc.rename(no_data_dir / proc.name)
            except OSError:
                pass

    year_elapsed = time.time() - year_start

    completed_ok = [k for k, v in state["completed"].items() if v.get("status") == "ok"]
    completed_no_data = [k for k, v in state["completed"].items() if v.get("status") == "no_data"]
    completed_fail = [k for k, v in state["completed"].items() if v.get("status") == "failed"]

    data_durations = [state["completed"][k]["duration_s"] for k in completed_ok]
    all_durations = [v["duration_s"] for v in state["completed"].values()]

    stats = {
        "total_tickers": total,
        "completed": len(state["completed"]),
        "successful": len(completed_ok),
        "no_data": len(completed_no_data),
        "failed": len(completed_fail),
        "elapsed_s": round(year_elapsed, 1),
    }
    if data_durations:
        stats["data_avg_time_s"] = round(sum(data_durations) / len(data_durations), 1)
        stats["data_min_time_s"] = round(min(data_durations), 1)
        stats["data_max_time_s"] = round(max(data_durations), 1)
    if all_durations:
        stats["avg_time_s"] = round(sum(all_durations) / len(all_durations), 1)
        stats["min_time_s"] = round(min(all_durations), 1)
        stats["max_time_s"] = round(max(all_durations), 1)
    state["stats"] = stats
    save_state(state_path, state)

    log("=" * 60)
    log("SUMMARY  [year %s]" % year)
    log("  Total:        %d" % total)
    log("  Successful:   %d" % len(completed_ok))
    log("  No data:      %d" % len(completed_no_data))
    log("  Failed:       %d" % len(completed_fail))
    log("  Duration:     %.1fs" % year_elapsed)
    if data_durations:
        log("  Avg/ticker (with data):   %.1fs" % stats["data_avg_time_s"])
        log("  Min/ticker (with data):   %.1fs" % stats["data_min_time_s"])
        log("  Max/ticker (with data):   %.1fs" % stats["data_max_time_s"])
    if all_durations:
        log("  Avg/ticker (all):         %.1fs" % stats["avg_time_s"])
        log("  Min/ticker (all):         %.1fs" % stats["min_time_s"])
        log("  Max/ticker (all):         %.1fs" % stats["max_time_s"])
    log("  State file:   %s" % state_path)
    if completed_fail:
        log("  Failed keys:  %s" % ", ".join(completed_fail))
    log("=" * 60)

    if log_fh:
        log_fh.close()

    log_ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    report = {
        "script": "stock_options_from_flatfiles_parallel_download",
        "timestamp": log_ts,
        "year": year,
        "workers": args.spawn,
        "resume": args.resume,
        "logs": args.logs,
        "total_tickers": total,
        "successful": len(completed_ok),
        "no_data": len(completed_no_data),
        "failed": len(completed_fail),
        "duration_s": round(year_elapsed, 1),
        "completed": state["completed"],
        "stats": stats,
    }
    report_dir = state_base / "options" / "stocks"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"parallel_report_{year}_{folder}_options.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print("  Report:       %s" % report_path)

    return {
        "year": year,
        "total": total,
        "completed": len(completed_ok) + len(completed_no_data) + len(completed_fail),
        "successful": len(completed_ok),
        "no_data": len(completed_no_data),
        "failed": len(completed_fail),
        "elapsed_s": round(year_elapsed, 1),
        "skipped": False,
    }


def main():
    args = parse_args()
    if args.download_only and args.use_local_cache:
        raise SystemExit("Error: --download_only and --use_local_cache are mutually exclusive")
    overall_start = time.time()
    agg = args.aggregate
    years = parse_years(args.year)

    all_year_results = []
    for year in years:
        result = run_year(agg, year, args)
        all_year_results.append(result)

    total_elapsed = time.time() - overall_start
    total_successful = sum(r["successful"] for r in all_year_results)
    total_no_data = sum(r["no_data"] for r in all_year_results)
    total_failed = sum(r["failed"] for r in all_year_results)

    print()
    print("=" * 60)
    print("OVERALL SUMMARY (%d year(s): %s)" % (len(years), args.year))
    print("  Total elapsed:    %s" % datetime.timedelta(seconds=int(total_elapsed)))
    print("  Successful:       %d" % total_successful)
    print("  No data:          %d" % total_no_data)
    print("  Failed:           %d" % total_failed)
    print("=" * 60)


if __name__ == "__main__":
    main()
