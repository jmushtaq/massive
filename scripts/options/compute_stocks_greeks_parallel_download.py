"""
Parallel dispatcher for compute_stocks_greeks.py (in-place update).

Usage:
    python scripts/options/compute_stocks_greeks_parallel_download.py --ohlcv_tickers --year 2025 --spawn 16
    python scripts/options/compute_stocks_greeks_parallel_download.py --tickers AAPL --year 2025 --spawn 4 --exclude_tickers /tmp/skip.txt
"""

import argparse, csv, datetime, json, os, select, signal, subprocess, sys, time
from pathlib import Path
from dotenv import load_dotenv
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(env_path)

SCRIPT_DIR = Path(__file__).resolve().parent
WORKER_SCRIPT = SCRIPT_DIR / "compute_stocks_greeks.py"
STDOUT_MARKER = "PARALLEL_RESULT:"
SCRIPT_NAME = Path(__file__).resolve().stem

AGGREGATE_MAP = {
    "1sec": "1sec", "1min": "1min", "5min": "5min",
    "15min": "15min", "1H": "1H", "4H": "4H", "1D": "1D",
}


def clean_ticker(raw: str) -> str:
    return raw.strip().upper().split("-")[0]


def parse_args():
    p = argparse.ArgumentParser(description="Parallel compute stocks greeks")
    p.add_argument("--tickers", type=str, default=None)
    p.add_argument("--tickers_file", type=str, default=None)
    p.add_argument("--ohlcv_tickers", action="store_true", default=False)
    p.add_argument("--exclude_tickers", type=str, default=None)
    p.add_argument("--year", type=str, required=True)
    p.add_argument("--aggregate", choices=list(AGGREGATE_MAP.keys()), default="1min")
    p.add_argument("--spawn", type=int, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--logs", action="store_true", default=False)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--check", type=str, default=None)
    p.add_argument("--inplace", type=lambda s: s.lower() == "true", default=False,
                   help="Overwrite original (True) or write to greeks/ subfolder (False). Default: False.")
    return p.parse_args()


def load_tickers(f: str) -> list[str]:
    tickers = []
    with open(f) as fh:
        for row in csv.DictReader(fh):
            t = row.get("ticker", "").strip()
            if t:
                tickers.append(clean_ticker(t))
    return tickers or sys.exit("Error: no tickers found")


def load_ohlcv_tickers(year: str) -> list[str]:
    d = Path("data") / "SPY" / "1min" / year
    if not d.exists():
        sys.exit("Error: OHLCV dir not found: %s" % d)
    tickers = []
    for f in sorted(d.glob(f"*_{year}_1min.csv*")):
        name = f.name.replace(".csv.gz", "").replace(".csv", "")
        tickers.append(clean_ticker(name.split("_")[0]))
    return tickers or sys.exit("Error: no OHLCV files")


def load_exclude(filepath: str) -> set[str]:
    if not filepath or not os.path.exists(filepath):
        return set()
    exclude = set()
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = row.get("ticker", "").strip()
            if t:
                exclude.add(clean_ticker(t))
        if not exclude:
            f.seek(0)
            for line in f:
                t = line.strip()
                if t and not t.startswith("ticker"):
                    exclude.add(clean_ticker(t))
    return exclude


def out_path(ticker, year, agg, odir=None, subdir=None):
    folder = AGGREGATE_MAP[agg]
    base = Path(odir) if odir else Path("data")
    p = base / "options" / "stocks" / folder / year
    if subdir:
        p = p / subdir
    return p / f"{ticker}_{year}_{folder}_options.csv"


def tx_key(ticker, year):
    return f"{ticker}_{year}"


def is_final(ticker, year, agg, odir=None, inplace=False):
    """Check if atm_call_iv column is populated."""
    p = out_path(ticker, year, agg, odir, subdir=None if inplace else "greeks")
    if not p.exists() or p.stat().st_size == 0:
        return False
    with open(p) as f:
        reader = csv.DictReader(f)
        if "atm_call_iv" not in (reader.fieldnames or []):
            return False
        for row in reader:
            iv = row.get("atm_call_iv", "")
            return bool(iv and iv.strip())
    return False


def load_state(sp):
    if sp.exists():
        return json.loads(sp.read_text())
    return {"completed": {}, "in_progress": {}, "all_tickers": [], "stats": {}}


def save_state(sp, state):
    sp.parent.mkdir(parents=True, exist_ok=True)
    tmp = sp.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(sp)


def worker_status(proc, ret, rl=None):
    if ret != 0:
        return "failed"
    if rl and STDOUT_MARKER in rl:
        try:
            return json.loads(rl.split(STDOUT_MARKER)[1].strip()).get("status", "ok")
        except Exception:
            pass
    return "ok"


def drain(entry):
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


def main():
    args = parse_args()
    t0 = time.time()
    year, agg = args.year, args.aggregate
    folder = AGGREGATE_MAP[agg]

    sb = Path(args.output) if args.output else Path("data")
    sp = sb / "options" / "stocks" / f".parallel_state_{year}_{folder}_options_greeks.json"
    state = load_state(sp)

    if args.ohlcv_tickers:
        all_t = load_ohlcv_tickers(year)
    elif args.tickers_file:
        all_t = load_tickers(args.tickers_file)
    elif args.tickers:
        all_t = [clean_ticker(x) for x in args.tickers.split(",") if x.strip()]
    else:
        sys.exit("Error: specify --tickers, --tickers_file, or --ohlcv_tickers")

    exclude = load_exclude(args.exclude_tickers)
    if exclude:
        all_t = [t for t in all_t if t not in exclude]
        print(f"  Excluded {len(exclude)} tickers, {len(all_t)} remaining")

    all_t = [t for t in all_t if not is_final(t, year, agg, args.output, args.inplace)]

    if args.check:
        chk = [t for t in all_t
               if not (Path(args.check) / "options" / "stocks" / folder / year
                       / f"{t}_{year}_{folder}_options.csv").exists()]
        all_t = chk

    state["all_tickers"] = all_t
    queue = [x for x in all_t
             if tx_key(x, year) not in state.get("completed", {})
             and not (args.resume and is_final(x, year, agg, args.output, args.inplace))
             and tx_key(x, year) not in state.get("in_progress", {})]

    total, remaining = len(all_t), len(queue)
    done_cnt = total - remaining

    lfh = None
    if args.logs:
        ld = sb / "options" / "stocks" / "logs"
        ld.mkdir(parents=True, exist_ok=True)
        lfh = open(ld / f"parallel_{year}_{folder}_greeks_{datetime.datetime.now():%Y%m%dT%H%M%S}.log", "w")

    def log(msg, end="\n"):
        print(msg, end=end, flush=True)
        if lfh:
            lfh.write(msg + (end if end else "\n"))
            lfh.flush()

    log("=" * 60)
    log(f"PARALLEL STOCKS GREEKS (in-place)  [year {year}]")
    log(f"  Workers:      {args.spawn}")
    log(f"  Aggregate:    {agg}")
    log(f"  Resume:       {args.resume}")
    log(f"  Total:        {total}  Done: {done_cnt}  Remaining: {remaining}")
    log("=" * 60)

    if not remaining:
        log("Nothing to do.")
        if lfh:
            lfh.close()
        return

    log("")
    pending = list(queue)
    pending.reverse()
    active = []

    def spawn(ticker):
        k = tx_key(ticker, year)
        cmd = [sys.executable, str(WORKER_SCRIPT), "--tickers", ticker,
               "--year", year, "--aggregate", agg, "--no_rename"]
        if args.resume:
            cmd.append("--resume")
        if args.output:
            cmd.extend(["--output", args.output])
        if args.exclude_tickers:
            cmd.extend(["--exclude_tickers", args.exclude_tickers])
        cmd.extend(["--inplace", str(args.inplace)])
        try:
            sf = open(f"/tmp/worker_stocks_greeks_{ticker}_{year}.log", "w")
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.PIPE, text=True)
        except Exception as e:
            log(f"[ERROR] spawn {ticker}: {e}")
            return None
        entry = {"ticker": ticker, "pid": proc.pid, "proc": proc,
                 "start_time": time.time(), "stderr_file": sf, "_result_line": None}
        state["in_progress"][k] = {"pid": proc.pid, "start_time": entry["start_time"]}
        state["stats"] = {"elapsed_s": round(time.time() - t0, 1), "total_tickers": total,
                          "completed": len(state["completed"]), "running": len(state["in_progress"]),
                          "workers": args.spawn}
        save_state(sp, state)
        return entry

    try:
        while pending or active:
            while len(active) < args.spawn and pending:
                tkr = pending.pop()
                if tx_key(tkr, year) in state.get("completed", {}) or tx_key(tkr, year) in state.get("in_progress", {}):
                    continue
                e = spawn(tkr)
                if e:
                    active.append(e)
            if not active and not pending:
                break
            still = []
            for e in active:
                proc = e["proc"]
                ret = proc.poll()
                if ret is not None:
                    tkr = e["ticker"]
                    k = tx_key(tkr, year)
                    dur = time.time() - e["start_time"]
                    drain(e)
                    ws = worker_status(proc, ret, e.get("_result_line"))
                    e["stderr_file"].close()
                    if proc.stderr:
                        proc.stderr.close()
                    state["completed"][k] = {"ticker": tkr, "duration_s": round(dur, 1),
                                             "returncode": ret, "status": ws,
                                             "timestamp": datetime.datetime.now().isoformat()}
                    if k in state["in_progress"]:
                        del state["in_progress"][k]
                    state["stats"] = {"elapsed_s": round(time.time() - t0, 1),
                                      "total_tickers": total, "completed": len(state["completed"]),
                                      "running": len(state["in_progress"]), "workers": args.spawn}
                    save_state(sp, state)
                    if ws == "ok":
                        proc_p = out_path(tkr, year, agg, args.output, subdir="processing")
                        if args.inplace:
                            final_p = out_path(tkr, year, agg, args.output)
                        else:
                            final_p = out_path(tkr, year, agg, args.output, subdir="greeks")
                        if proc_p.exists() and proc_p.stat().st_size > 0:
                            final_p.parent.mkdir(parents=True, exist_ok=True)
                            try:
                                proc_p.replace(final_p)
                            except OSError:
                                pass
                    log(f"  [{len(state['completed'])}/{total}] {tkr} done ({ws}, {fmt_duration(dur)})")
                else:
                    still.append(e)
            active = still
            if active:
                time.sleep(1)
    except KeyboardInterrupt:
        log("\n[INTERRUPT] Terminating ...")
        for e in active:
            try:
                os.kill(e["proc"].pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for e in active:
            e["proc"].wait(timeout=5)
        sys.exit(130)

    state["in_progress"] = {}
    save_state(sp, state)
    ye = time.time() - t0
    ok = [k for k, v in state["completed"].items() if v.get("status") == "ok"]
    fail = [k for k, v in state["completed"].items() if v.get("status") == "failed"]
    state["stats"] = {"total_tickers": total, "completed": len(state["completed"]),
                      "successful": len(ok), "failed": len(fail), "elapsed_s": round(ye, 1),
                      "workers": args.spawn}
    save_state(sp, state)
    log("=" * 60)
    log(f"SUMMARY  [year {year}]")
    log(f"  Successful:   {len(ok)}  Failed: {len(fail)}  Duration: {fmt_duration(ye)}")
    log("=" * 60)
    if lfh:
        lfh.close()


if __name__ == "__main__":
    main()
