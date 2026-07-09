"""
Display live status of a running or completed parallel trade enrichment run.

Reads the state file written by trades_enrichment_parallel_download.py.

Usage:
    python scripts/trades_enrichment_parallel_status.py
    python scripts/trades_enrichment_parallel_status.py --year 2010
    python scripts/trades_enrichment_parallel_status.py --year 2010 --aggregate 1H
    python scripts/trades_enrichment_parallel_status.py --year 2010 --watch
        --watch: refresh every 5 seconds (live monitoring)
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
        description="Show status of parallel trade enrichment downloads"
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

    total = len(all_tickers)
    done = len(completed)
    running = len(in_progress)
    remaining = total - done - running

    completed_ok = sum(1 for v in completed.values() if v.get("status") == "ok")
    completed_no_data = sum(1 for v in completed.values() if v.get("status") == "no_data")
    completed_failed = sum(1 for v in completed.values() if v.get("status") == "failed")

    print("  State file:   %s" % state_path.name)
    print("  Total:        %d" % total)
    print("  Completed:    %d" % done)
    print("    OK:         %d" % completed_ok)
    print("    No data:    %d" % completed_no_data)
    print("    Failed:     %d" % completed_failed)
    print("  Running:      %d" % running)
    print("  Remaining:    %d" % remaining)

    if stats:
        elapsed = stats.get("elapsed_s", 0)
        print("  Elapsed:      %s" % fmt_duration(elapsed))
        if "data_avg_time_s" in stats:
            print("  Avg/ticker (with data):     %.1fs" % stats["data_avg_time_s"])
            print("  Min/ticker (with data):     %.1fs" % stats["data_min_time_s"])
            print("  Max/ticker (with data):     %.1fs" % stats["data_max_time_s"])
        if "avg_time_s" in stats:
            print("  Avg/ticker (all):           %.1fs" % stats["avg_time_s"])
            print("  Min/ticker (all):           %.1fs" % stats["min_time_s"])
            print("  Max/ticker (all):           %.1fs" % stats["max_time_s"])
        if "successful" in stats:
            print("  Successful:   %d" % stats["successful"])
        if "no_data" in stats:
            print("  No data:      %d" % stats["no_data"])
        if "failed" in stats:
            print("  Failed:       %d" % stats["failed"])

    if in_progress:
        print()
        print("  -- Currently running --")
        entries = sorted(in_progress.items(), key=lambda x: x[1].get("start_time", 0))
        for key, info in entries:
            start = info.get("start_time", 0)
            elapsed_running = time.time() - start
            print("    %-12s pid=%-6d %s" % (key.split("_")[0], info.get("pid", 0), fmt_duration(elapsed_running)))

    if done > 0:
        data_durations = [v.get("duration_s", 0) for v in completed.values() if v.get("status") == "ok"]
        if data_durations:
            print()
            print("  Duration stats (tickers with data only):")
            print("    Fastest:    %.1fs" % min(data_durations))
            print("    Slowest:    %.1fs" % max(data_durations))
            print("    Average:    %.1fs" % (sum(data_durations) / len(data_durations)))

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
    # Also kill the dispatcher (parent of these workers)
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

    data_dir = Path("data") / "trades"
    pattern = ".parallel_state_"
    if args.year:
        pattern += args.year
    if args.aggregate:
        pattern += "_" + AGGREGATE_MAP[args.aggregate]
    else:
        pattern += "*"

    state_files = sorted(data_dir.glob(pattern + ".json"))

    if not state_files:
        print("No matching state files found in %s" % data_dir)
        # List available
        available = list(data_dir.glob(".parallel_state_*.json"))
        if available:
            print("Available state files:")
            for sf in available:
                print("  %s" % sf.name)
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
                print("\033[2J\033[H", end="")  # clear screen
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
