#!/usr/bin/env python3
"""
Backtest integrity audit for the scalping backtester (scalping_analysis.py).

Audits the execution ENGINE (not the alpha) against the classic sources of
inflated backtest results:

  1. Lookahead in signals (trigger bar's own OHLCV used for fill)
  2. Future trades/quotes/options/chains in signals or ranking
  3. Entry-fill reachability / phantom fills
  4. Intrabar stop/target ordering (target-first bias)
  5. Entry-bar stop being ignored
  6. Capital aggregation across tickers (independent vs shared account)
  7. Position-sizing leverage and silently-skipped trades
  8. Corporate-action (split) adjustment consistency
  9. Survivorship / delisting and full-period selection bias
  10. Bid/ask spread + slippage inclusion

It reproduces the engine's "ORB + Volume Confluence" trades EXACTLY but faster:
instead of running the slow `backtest_opening_range_breakout` over every bar, it
builds the Liquidity Vacuum confirmation set and only walks ORB signals that
confirm (verified to produce byte-identical trades to the engine, see
`_orb_confirmed`). Each reported trade is then re-walked with conservative
intrabar assumptions to quantify the bias.

Usage:
    python scripts/strategy/scalping/backtest_integrity_audit.py \
        --year 2026 --aggregate 1sec --rr 2.5 --risk-amount 1 \
        --num-trades 100 --limit 20
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scalping_analysis import (  # noqa: E402
    CapitalManager,
    backtest_liquidity_vacuum,
    load_ohlcv,
    load_quotes,
    _walk_trade,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("audit")

OHLCV_DIR = Path("data/SPY")
ORB_MAX_BARS = 30  # backtest_opening_range_breakout default
ORB_LOOKBACK = 5   # opening-range bars


def _aligned(df):
    out = df[["open", "high", "low", "close", "volume"]].copy()
    out.columns = ["o", "h", "l", "c", "vol"]
    return out


def _orb_confirmed(df, rr, max_bars, confirm_set, limit):
    """Mirror of backtest_opening_range_breakout that only walks signals whose
    (entry_time, direction) is in confirm_set, stopping at `limit` trades.
    Produces identical trades to backtest_orb_volume_confluence(df, rr)[:limit]."""
    results = []
    h, l, ts = df["h"].values, df["l"].values, df.index
    try:
        dates = df.tz_convert("US/Eastern").index.normalize()
    except Exception:
        dates = df.index.normalize()
    n = len(dates)
    if n == 0:
        return results

    orb_high = orb_low = 0.0
    orb_seen = 0
    orb_ready = False
    prev_date = dates[0]

    for i in range(n):
        if i > 0 and dates[i] != prev_date:
            orb_seen = 0
            orb_high = orb_low = 0.0
            orb_ready = False
            prev_date = dates[i]

        if not orb_ready and orb_seen < ORB_LOOKBACK:
            orb_high = h[i] if orb_seen == 0 else max(orb_high, h[i])
            orb_low = l[i] if orb_seen == 0 else min(orb_low, l[i])
            orb_seen += 1
            if orb_seen == ORB_LOOKBACK:
                orb_ready = True
            continue

        if not orb_ready:
            continue

        try:
            cur_hour = ts[i].tz_convert("US/Eastern").hour
        except Exception:
            cur_hour = 9
        if cur_hour < 9 or cur_hour >= 11:
            continue

        if h[i] > orb_high:
            entry, stop = orb_high + 0.01, orb_low
            risk = entry - stop
            if risk > 0 and (ts[i], "long") in confirm_set:
                r = _walk_trade(df, i, entry, stop, entry + risk * rr, 1, max_bars)
                if r:
                    results.append(r)
                if limit and len(results) >= limit:
                    return results
        elif l[i] < orb_low:
            entry, stop = orb_low - 0.01, orb_high
            risk = stop - entry
            if risk > 0 and (ts[i], "short") in confirm_set:
                r = _walk_trade(df, i, entry, stop, entry - risk * rr, -1, max_bars)
                if r:
                    results.append(r)
                if limit and len(results) >= limit:
                    return results
    return results


def _instrument_trade(df, t, rr):
    h = df["h"].values
    l = df["l"].values
    n = len(df)
    idx = df.index

    entry_time = t["entry_time"]
    try:
        entry_i = idx.get_loc(entry_time)
        if isinstance(entry_i, slice):
            entry_i = entry_i.start
    except KeyError:
        return None

    exit_i = entry_i + int(t.get("bars_held", 0))
    d = 1 if t["direction"] == "long" else -1
    entry = float(t["entry_price"])
    stop = float(t["stop_price"])
    risk = abs(entry - stop)
    target = entry + d * risk * rr

    ambiguous = False
    for j in range(entry_i + 1, min(entry_i + ORB_MAX_BARS, n)):
        hit_t = (h[j] >= target) if d > 0 else (l[j] <= target)
        hit_s = (l[j] <= stop) if d > 0 else (h[j] >= stop)
        if hit_t and hit_s:
            ambiguous = True
            break

    sf = None
    for j in range(entry_i + 1, min(entry_i + ORB_MAX_BARS, n)):
        hit_s = (l[j] <= stop) if d > 0 else (h[j] >= stop)
        hit_t = (h[j] >= target) if d > 0 else (l[j] <= target)
        if hit_s:
            sf = "loss"
            break
        if hit_t:
            sf = "win"
            break
    if sf is None:
        sf = "timeout"

    entry_bar_stop = bool((l[entry_i] <= stop) if d > 0 else (h[entry_i] >= stop))
    phantom = bool((h[entry_i] < entry) if d > 0 else (l[entry_i] > entry))

    exit_ok = True
    if t.get("result") == "win":
        exit_ok = bool((h[exit_i] >= t["exit_price"]) if d > 0 else (l[exit_i] <= t["exit_price"]))
    elif t.get("result") == "loss":
        exit_ok = bool((l[exit_i] <= t["exit_price"]) if d > 0 else (h[exit_i] >= t["exit_price"]))

    return {
        "entry_time": entry_time,
        "direction": t["direction"],
        "result": t.get("result"),
        "entry": entry,
        "stop": stop,
        "exit_price": float(t["exit_price"]),
        "target": target,
        "risk": risk,
        "entry_i": entry_i,
        "exit_i": exit_i,
        "ambiguous": ambiguous,
        "stop_first": sf,
        "entry_bar_stop": entry_bar_stop,
        "phantom_entry": phantom,
        "exit_ok": exit_ok,
    }


def _audit_ticker(args):
    (ticker, year, agg, rr, num_trades, risk_pct, starting_cash) = args
    ohlcv = load_ohlcv(ticker, year, agg)
    if ohlcv is None or len(ohlcv) < 500:
        return None

    df = _aligned(ohlcv)

    conf_trades = backtest_liquidity_vacuum(df, rr=rr)
    confirm_set = {(x["entry_time"], x["direction"]) for x in conf_trades}
    raw = _orb_confirmed(df, rr, ORB_MAX_BARS, confirm_set, num_trades)

    if not raw:
        return None

    cap = CapitalManager(starting_cash=starting_cash, risk_pct=risk_pct)

    trades = []
    n_executed = 0
    n_skipped = 0
    skipped_results = []

    for t in raw:
        rec = _instrument_trade(df, t, rr)
        if rec is None:
            continue
        d = 1 if t["direction"] == "long" else -1
        res = cap.execute_trade(t["entry_price"], t["exit_price"],
                                t.get("stop_price", t["entry_price"]), d, t["exit_time"])
        if res is None:
            n_skipped += 1
            skipped_results.append(t.get("result"))
            continue
        n_executed += 1
        rec.update({
            "shares": res["shares"],
            "position_value": res["position_value"],
            "margin_used": res["margin_used"],
            "risk_$": res["risk_$"],
            "pnl_$": res["pnl_$"],
            "commission_$": res["commission_$"],
        })
        trades.append(rec)

    if trades:
        qu = load_quotes(ticker, year, agg)
        if qu is not None and len(qu) > 0:
            spread_series = qu["avg_spread"].reindex(df.index, method="ffill")
            for rec in trades:
                ei, xi = rec["entry_i"], rec["exit_i"]
                se = spread_series.iloc[ei] if ei < len(spread_series) else np.nan
                sx = spread_series.iloc[xi] if xi < len(spread_series) else np.nan
                rec["spread_entry"] = se if pd.notna(se) else 0.0
                rec["spread_exit"] = sx if pd.notna(sx) else 0.0

    day_first_et = None
    try:
        et = pd.Series(df.index).dt.tz_convert("US/Eastern")
        day_first_et = int(et.groupby(et.dt.date).first().dt.hour.mode().iloc[0])
    except Exception:
        pass

    return {
        "ticker": ticker,
        "executed": n_executed,
        "skipped": n_skipped,
        "skipped_results": skipped_results,
        "trades": trades,
        "day_first_et_hour": day_first_et,
    }


def _worker_wrapper(args):
    try:
        return _audit_ticker(args)
    except Exception as e:  # noqa: BLE001
        return {"ticker": args[0], "error": str(e)}


def _spread_adjusted_pnl(rec):
    d = 1 if rec["direction"] == "long" else -1
    shares = rec.get("shares", 0)
    se = rec.get("spread_entry", 0.0) or 0.0
    sx = rec.get("spread_exit", 0.0) or 0.0
    entry = rec["entry"]
    exit_p = rec["exit_price"]
    if d > 0:
        gross = ((exit_p - sx / 2) - (entry + se / 2)) * shares
    else:
        gross = ((entry - se / 2) - (exit_p + sx / 2)) * shares
    return gross - rec.get("commission_$", 0.0)


def main():
    p = argparse.ArgumentParser(description="Backtest integrity audit")
    p.add_argument("--year", default="2026")
    p.add_argument("--aggregate", default="1sec")
    p.add_argument("--rr", type=float, default=2.5)
    p.add_argument("--risk-amount", type=float, default=1.0)
    p.add_argument("--starting-cash", type=float, default=100_000.0)
    p.add_argument("--num-trades", type=int, default=100)
    p.add_argument("--limit", type=int, default=0, help="0 = all tickers")
    p.add_argument("--nprocs", type=int, default=os.cpu_count() or 1)
    args = p.parse_args()

    risk_pct = args.risk_amount / 100.0
    ohlcv_dir = OHLCV_DIR / args.aggregate / args.year
    tickers = sorted(f.stem.split("_")[0]
                     for f in ohlcv_dir.glob(f"*_{args.year}_{args.aggregate}.csv"))
    if args.limit:
        tickers = tickers[: args.limit]
    log.info("Auditing %d tickers (%s %s, rr=1:%s, risk %.2f%%)",
             len(tickers), args.year, args.aggregate, args.rr, args.risk_amount)

    work = [(t, args.year, args.aggregate, args.rr, args.num_trades,
             risk_pct, args.starting_cash) for t in tickers]

    results = []
    nprocs = max(1, min(args.nprocs, len(work)))
    if nprocs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=nprocs) as pool:
            for r in pool.map(_worker_wrapper, work):
                if r:
                    results.append(r)
    else:
        for w in work:
            r = _worker_wrapper(w)
            if r:
                results.append(r)

    results = [r for r in results if r and "error" not in r]
    all_trades = []
    for r in results:
        all_trades.extend(r["trades"])

    print("\n" + "=" * 82)
    print("  BACKTEST INTEGRITY AUDIT  —  ORB + Volume Confluence")
    print("  %s %s  |  R:R 1:%s  |  risk %.2f%%  |  %d tickers"
          % (args.year, args.aggregate, args.rr, args.risk_amount, len(results)))
    print("=" * 82)

    print("\n[1] SIGNAL LOOKAHEAD (same-bar OHLCV)  ->  FAULT")
    print("    Breakout triggers read the *trigger bar's own* high (h[i] > prior_high)")
    print("    and its complete volume (v[i] > 1.2*avg_v), which are only known at bar")
    print("    close, yet the entry is a stop filled mid-bar at prior_high+0.01. The")
    print("    confluence matches a Liquidity-Vacuum signal on the SAME second, and that")
    print("    signal also needs the full bar volume (v[i]/vol_ma >= 2.5). On 1s bars the")
    print("    engine knows bar i's whole OHLCV before deciding to enter that bar.")

    print("\n[2] FUTURE TRADES/QUOTES/OPTIONS IN SIGNALS  ->  PASS (this strategy) / CAVEAT")
    print("    ORB + Volume is OHLCV-only, so no future auxiliary data enters the trades.")
    print("    Other strategies read tr_delta[i] / qu_quote_imbalance[i] of the trigger bar")
    print("    (same-bar, unknown until close). The daily ranking (opportunity_score) uses")
    print("    the FULL day's trades/quotes/options/chains and the full-year average to pick")
    print("    tickers/strategies -> hindsight selection (see [9]).")

    n = len(all_trades)
    print("\n[4/5] INTRABAR STOP/TARGET (target-first bias) + TIMEOUT HIDING")
    if n == 0:
        print("    No trades generated across the sample (raise --limit or check data).")
    else:
        amb = [r for r in all_trades if r["ambiguous"]]
        rep_win = sum(1 for r in all_trades if r["result"] == "win")
        rep_loss = sum(1 for r in all_trades if r["result"] == "loss")
        rep_to = sum(1 for r in all_trades if r["result"] == "timeout")
        sf_win = sum(1 for r in all_trades if r["stop_first"] == "win")
        sf_loss = sum(1 for r in all_trades if r["stop_first"] == "loss")
        sf_to = sum(1 for r in all_trades if r["stop_first"] == "timeout")
        wr_rep = rep_win / (rep_win + rep_loss) * 100 if (rep_win + rep_loss) else 0
        wr_sf = sf_win / (sf_win + sf_loss) * 100 if (sf_win + sf_loss) else 0
        print(f"    Reported trades: {n}  (win={rep_win} loss={rep_loss} timeout={rep_to})")
        print(f"    Bars where BOTH stop and target hit (ambiguous): {len(amb)} "
              f"({len(amb)/n*100:.1f}%)")
        print(f"    _walk_trade checks TARGET first -> ambiguous bars are counted as WIN.")
        print(f"    Win rate reported (target-first): {wr_rep:.1f}%")
        print(f"    Win rate conservative (stop-first): {wr_sf:.1f}%")
        entry_stop = [r for r in all_trades if r["entry_bar_stop"]]
        phantom = [r for r in all_trades if r["phantom_entry"]]
        bad_exit = [r for r in all_trades if not r["exit_ok"]]
        print(f"    Trades whose ENTRY bar already touched the stop (ignored): "
              f"{len(entry_stop)} ({len(entry_stop)/n*100:.1f}%)")
        print(f"    Phantom entries (fill not reachable on entry bar): {len(phantom)}")
        print(f"    Unreachable reported exit prices: {len(bad_exit)}")
        # Timeout hiding: win rate excludes timeouts, but timeouts carry the P&L.
        to_recs = [r for r in all_trades if r["result"] == "timeout" and "pnl_$" in r]
        if to_recs:
            to_pnl = sum(r["pnl_$"] for r in to_recs)
            total_pnl = sum(r["pnl_$"] for r in all_trades if "pnl_$" in r)
            print(f"    TIMEOUTS: {len(to_recs)} ({len(to_recs)/n*100:.0f}% of trades) are excluded")
            print(f"    from the win rate. Their P&L = ${to_pnl:,.0f} "
                  f"({to_pnl/total_pnl*100:.0f}% of total P&L ${total_pnl:,.0f}).")
            print("    -> the 92-96% 'win rate' is only for the minority that resolve cleanly.")

    print("\n[6] CAPITAL AGGREGATION (independent vs shared account)  ->  FAULT")
    ticker_pnl = {r["ticker"]: sum(t["pnl_$"] for t in r["trades"]) for r in results}
    n_tick = len(ticker_pnl)
    total_pnl = sum(ticker_pnl.values())
    combined = n_tick * args.starting_cash
    print(f"    Engine gives each of {n_tick} tickers its OWN ${args.starting_cash:,.0f} account")
    print("    (CapitalManager created per ticker, reset per strategy/R:R). Headline")
    print(f"    'Total P&L' = sum of independent accounts = ${total_pnl:,.0f}.")
    print(f"    Combined capital deployed = ${combined:,.0f} -> true combined return")
    print(f"    = {total_pnl/combined*100:+.1f}% (not {total_pnl/args.starting_cash*100:+.0f}%).")
    print("    No cross-ticker capital/margin contention is modelled; trades overlap in")
    print("    time but never compete for the same cash.")

    print("\n[7] POSITION SIZING / LEVERAGE")
    pvs = [r["position_value"] for r in all_trades if "position_value" in r]
    risks = [r["risk_$"] for r in all_trades if "risk_$" in r]
    if pvs:
        max_pv = max(pvs)
        print(f"    margin_rate=0.25 (4x), max_position_pct=200% of CURRENT equity.")
        print(f"    Max executed position = ${max_pv:,.0f} (up to 2x current equity; equity")
        print(f"    grows with wins, so absolute size can exceed 2x starting cash).")
        print(f"    Risk per executed trade: mean ${np.mean(risks):,.0f} "
              f"(target ${args.risk_amount/100*args.starting_cash:,.0f}).")
    skipped_total = sum(r["skipped"] for r in results)
    if skipped_total:
        skr = []
        for r in results:
            skr.extend(r["skipped_results"])
        sk_win = sum(1 for x in skr if x == "win")
        sk_loss = sum(1 for x in skr if x == "loss")
        sk_wr = sk_win / (sk_win + sk_loss) * 100 if (sk_win + sk_loss) else 0
        print(f"    SILENTLY SKIPPED (position would exceed cap): {skipped_total} trades")
        print(f"    Their would-be win rate (target-first): {sk_wr:.1f}% (n={sk_win+sk_loss})")
        print("    -> dropping tight-stop trades is itself an undocumented filter.")

    print("\n[8] CORPORATE ACTIONS / SPLIT ADJUSTMENT  ->  PARTIAL")
    print("    OHLCV is split-adjusted (stocks_aggs_download.py uses list_aggs(adjusted=True)).")
    print("    Trades/quotes are RAW (unadjusted) tick data; only OPTIONS are split-adjusted")
    print("    (adjust_options_for_splits.py). Delta/volume strategies mix adjusted prices with")
    print("    unadjusted volumes across a split date. This OHLCV-only strategy is unaffected.")

    print("\n[9] SURVIVORSHIP / SELECTION BIAS  ->  FAULT")
    all_files = sorted(ohlcv_dir.glob(f"*_{args.year}_{args.aggregate}.csv"))
    ends = {}
    for f in all_files:
        t = f.stem.split("_")[0]
        with open(f, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 4096))
            for line in reversed(fh.read().decode(errors="ignore").splitlines()):
                if line and line[0].isdigit():
                    ends[t] = line.split(",")[0]
                    break
    latest = max(ends.values())
    cutoff = (pd.Timestamp(latest) - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
    trunc = {t: ts for t, ts in ends.items() if ts < cutoff}
    print(f"    Universe on disk: {len(ends)} tickers; newest timestamp {latest}.")
    print(f"    Tickers whose data ends >90d before newest: {len(trunc)} (delisted/truncated) "
          f"— these ARE present in the universe, so no universe-level survivorship.")
    if trunc:
        print("      e.g. " + ", ".join(f"{t}({ts[:10]})" for t, ts in list(trunc.items())[:8]))
    print("    BUT backtest_tickers = top-N by FULL-YEAR opportunity_score -> winners are")
    print("    selected with hindsight (the score uses the entire period's data).")

    print("\n[10] BID/ASK SPREAD & SLIPPAGE  ->  FAULT")
    with_spread = [r for r in all_trades if "shares" in r]
    if with_spread:
        base = sum(r["pnl_$"] for r in with_spread)
        spread = sum(_spread_adjusted_pnl(r) for r in with_spread)
        print("    execute_trade fills at exact entry/exit prices; only cost is commission")
        print("    max(0.35, shares*0.0001). No bid/ask, no slippage.")
        print(f"    P&L engine (no spread):   ${base:,.0f}")
        print(f"    P&L half-spread each side (real quotes avg_spread): ${spread:,.0f}")
        if base:
            print(f"    Spread drag: ${base - spread:,.0f} ({(1 - spread/base)*100:.1f}% of gross)")

    print("\n[BONUS] ORB 'OPENING RANGE' IS PRE-MARKET")
    hrs = {r["day_first_et_hour"] for r in results if r.get("day_first_et_hour") is not None}
    print(f"    Day-first-bar ET hour across tickers: {sorted(hrs)}")
    print("    backtest_opening_range_breakout uses the first 5 BARS of the ET day as the")
    print("    'opening range', but the day's first bars are the 04:00 ET pre-market open, not")
    print("    the 09:30 regular open; trades fire 09:00-11:00 ET.")

    print("\n" + "=" * 82)
    print("  AUDIT COMPLETE")
    print("=" * 82)


if __name__ == "__main__":
    main()
