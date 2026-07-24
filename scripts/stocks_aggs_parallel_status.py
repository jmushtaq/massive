"""
Display live status of a running or completed parallel stock aggregate download.

Reads the state file written by stocks_aggs_parallel_download.py.

Usage:
    python scripts/stocks_aggs_parallel_status.py
    python scripts/stocks_aggs_parallel_status.py --year 2010
    python scripts/stocks_aggs_parallel_status.py --year 2010 --aggregate 1H
    python scripts/stocks_aggs_parallel_status.py --year 2010 --watch
        --watch: refresh every 5 seconds (live monitoring)

    python scripts/stocks_aggs_parallel_status.py --year 2010 --kill
        --kill: kill all running processes
"""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

AGGREGATE_MAP = {
    "1min": "1min",
    "5min": "5min",
    "15min": "15min",
    "1H": "1H",
    "4H": "4H",
    "1D": "1D",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Show status of parallel stock aggregate downloads"
    )
    parser.add_argument(
        "--year",
        type=str,
        default=None,
        help="Year filter (default: show all available state files)",
    )
    parser.add_argument(
        "--aggregate",
        choices=list(AGGREGATE_MAP.keys()),
        default=None,
        help="Aggregate window (default: show all)",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously refresh every 5 seconds",
    )
    parser.add_argument(
        "--kill",
        action="store_true",
        help="Kill all running workers and the dispatcher process",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Base output directory where state file lives (e.g. data/combined).",
    )
    return parser.parse_args()


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


def show_status(state_path: Path):
    if not state_path.exists():
        print("  (no state file)")
        return False

    with open(state_path) as f:
        state = json.load(f)

    all_tickers = state.get("all_tickers", [])
    completed = state.get("completed", {})
    in_progress = state.get("in_progress", {})
    stats = state.get("stats", {})
    config = state.get("config", {})

    total = len(all_tickers)
    done = len(completed)
    running = len(in_progress)
    remaining = total - done - running

    completed_ok = sum(1 for v in completed.values() if v.get("status") == "ok")
    completed_no_data = sum(1 for v in completed.values() if v.get("status") == "no_data")
    completed_failed = sum(1 for v in completed.values() if v.get("status") == "failed")

    elapsed = stats.get("elapsed_s", 0)
    elapsed_hours = elapsed / 3600 if elapsed > 0 else 0.001

    pct = (done / total * 100) if total > 0 else 0
    throughput = done / elapsed_hours if elapsed_hours > 0 else 0
    eta_s = (remaining / (done / elapsed)) if done > 0 and elapsed > 0 else 0

    workers = config.get("workers", 0)

    print("  State file:   %s" % state_path.name)
    print("  Total:        %d" % total)
    print("  Completed:    %d (%2.1f%%)" % (done, pct))
    print("    OK:         %d" % completed_ok)
    print("    No data:    %d" % completed_no_data)
    print("    Failed:     %d" % completed_failed)
    print("  Running:      %d" % running)
    print("  Remaining:    %d" % remaining)
    if workers:
        print("  Workers:      %d" % workers)
    print()
    print("  Elapsed:      %s" % fmt_duration(elapsed))
    if done > 0:
        print("  Throughput:   %.0f tickers/hr" % throughput)
        print("  Est. finish:  %s" % fmt_duration(eta_s))

    if stats:
        if "data_avg_time_s" in stats:
            print()
            print("  Duration stats (tickers with data only):")
            print("    Fastest:    %.1fs" % stats["data_min_time_s"])
            print("    Slowest:    %.1fs" % stats["data_max_time_s"])
            print("    Average:    %.1fs" % stats["data_avg_time_s"])

    if in_progress:
        entries = sorted(in_progress.items(), key=lambda x: x[1].get("start_time", 0))
        MAX_RUNNING_DISPLAY = 20
        if len(entries) > MAX_RUNNING_DISPLAY:
            display = entries[:MAX_RUNNING_DISPLAY]
            extra = len(entries) - MAX_RUNNING_DISPLAY
        else:
            display = entries
            extra = 0
        print()
        print("  -- Currently running (%d total%s) --" % (len(entries), "; showing first %d" % MAX_RUNNING_DISPLAY if extra else ""))
        for key, info in display:
            start = info.get("start_time", 0)
            elapsed_running = time.time() - start
            print("    %-12s pid=%-6d %s" % (key.split("_")[0], info.get("pid", 0), fmt_duration(elapsed_running)))

    return running > 0


def kill_workers(state_path: Path):
    if not state_path.exists():
        print("No state file at %s" % state_path)
        return
    with open(state_path) as f:
        state = json.load(f)
    in_progress = state.get("in_progress", {})
    if not in_progress:
        print("No running workers found in %s" % state_path.name)
        return
    pids = []
    for key, info in in_progress.items():
        pid = info.get("pid")
        if pid:
            pids.append((key, pid))
    if not pids:
        print("No PIDs in state.")
        return
    print("Killing %d worker(s) ..." % len(pids))
    for key, pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
            print("  Killed worker %s (pid %d)" % (key.split("_")[0], pid))
        except ProcessLookupError:
            print("  Worker %s (pid %d) already gone" % (key.split("_")[0], pid))
        except PermissionError as e:
            print("  Cannot kill %s (pid %d): %s" % (key.split("_")[0], pid, e))
    dispatcher_pid = None
    for key, info in in_progress.items():
        ppid = info.get("pid")
        if ppid:
            try:
                with open("/proc/%d/status" % ppid) as fh:
                    for line in fh:
                        if line.startswith("PPid:"):
                            candidate = int(line.split()[1])
                            if candidate > 1:
                                dispatcher_pid = candidate
                            break
            except (IOError, ValueError, IndexError):
                pass
            break
    if dispatcher_pid:
        try:
            os.kill(dispatcher_pid, signal.SIGKILL)
            print("  Killed dispatcher (pid %d)" % dispatcher_pid)
        except ProcessLookupError:
            print("  Dispatcher (pid %d) already gone" % dispatcher_pid)
        except PermissionError as e:
            print("  Cannot kill dispatcher (pid %d): %s" % (dispatcher_pid, e))


def main():
    args = parse_args()

    search_dirs = [Path("data"), Path("data") / "SPY"]
    if args.output:
        search_dirs.insert(0, Path(args.output))
    state_files = []
    for data_dir in search_dirs:
        pattern = ".parallel_state_"
        if args.year:
            pattern += args.year
        if args.aggregate:
            pattern += "_" + AGGREGATE_MAP[args.aggregate]
        else:
            pattern += "*"
        state_files.extend(data_dir.glob(pattern + ".json"))
    state_files = sorted(set(state_files))

    if not state_files:
        dirs_str = ", ".join(str(d) for d in search_dirs)
        print("No matching state files found in " + dirs_str)
        sys.exit(1)

    if args.kill:
        for sf in state_files:
            print()
            print("Killing workers for %s" % sf.name)
            print("-" * 50)
            kill_workers(sf)
        return

    for sf in state_files:
        print()
        label = sf.stem.replace(".parallel_state_", "state: ")
        print(label)
        print("-" * 50)
        has_active = show_status(sf)
        print()

    if args.watch:
        try:
            while True:
                time.sleep(5)
                print("\033[2J\033[H", end="")
                for sf in state_files:
                    label = sf.stem.replace(".parallel_state_", "state: ")
                    print(label)
                    print("-" * 50)
                    has_active = show_status(sf)
                    print()
        except KeyboardInterrupt:
            print("\nExiting.")


if __name__ == "__main__":
    main()
