#!/usr/bin/env python3
"""
gex_fracdiff/strategy.py  -  GEX-FracDiff Level-Momentum (GFLM) Strategy

Methodology from nq-fracdiff-optimizer:
  - Fractional differencing: d=0.55|0.71, N=130 lags -> stationary Z-score
  - Z extreme + PERSISTENT (>=persist bars) + snapback = quality entry
  - Directional efficiency (Kaufman ER) regime filter
  - Inter-trade cooldown prevents over-trading

Extended with structural reversal levels as targets:
  - PDH/PDL, PWH/PWL (5d), IBH/IBL, Round-50 multiples
  - Target = nearest level in snap direction -> much bigger R vs plain fracdiff Z=0 exit

Walk-forward: optimize 2024 -> test 2025 (no 2025 data seen during optimization)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.signal import lfilter
import json, random, warnings
warnings.filterwarnings("ignore")

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE   = os.path.join(ROOT, "gex_profile", "hdata", "NQ_1m_cache.parquet")
CSV     = os.path.join(ROOT, "..", "1Min_NQ.csv")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT_DIR, exist_ok=True)

OPT_START  = "2022-01-01";  OPT_END  = "2024-12-31"   # 3-year window -> more trades for reliable optimization
TEST_START = "2025-01-01";  TEST_END = "2025-12-31"
MIN_TRADES = 80


# ── data ──────────────────────────────────────────────────────────────────────
def load_nq():
    if os.path.exists(CACHE):
        df = pd.read_parquet(CACHE)
        df.columns = [c.lower() for c in df.columns]
        if "date" in df.columns:
            df = df.set_index("date")
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize("America/New_York",
                                             ambiguous="NaT", nonexistent="NaT")
        elif "New_York" not in str(df.index.tz):
            df.index = df.index.tz_convert("America/New_York")
    else:
        df = _load_csv()
    return df.dropna().sort_index()[["open","high","low","close"]]

def _load_csv():
    df = pd.read_csv(CSV, sep=";", decimal=",",
                     names=["ts","open","high","low","close","vol"], header=0)
    for c in ["open","high","low","close"]:
        if df[c].dtype == object:
            df[c] = (df[c].astype(str).str.replace(".", "", regex=False)
                          .str.replace(",", ".", regex=False).astype(float))
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.set_index("ts")
    df.index = df.index.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
    return df.dropna()


# ── fracdiff + Z-score ────────────────────────────────────────────────────────
def _fd_weights(d, n):
    w = [1.0]
    for k in range(1, n):
        w.append(-w[-1] * (d - k + 1) / k)
    return np.array(w, dtype=np.float64)

def compute_z(log_px, d, n_lags=130, z_len=100):
    """Causal FIR fracdiff -> rolling Z-score."""
    w  = _fd_weights(d, n_lags)
    fd = lfilter(w, [1.0], log_px.astype(np.float64))
    fd[:n_lags] = np.nan
    s  = pd.Series(fd)
    mu = s.rolling(z_len, min_periods=z_len).mean().values
    sg = s.rolling(z_len, min_periods=z_len).std(ddof=0).values
    return np.where((sg > 1e-10) & ~np.isnan(fd), (fd - mu) / sg, np.nan)


# ── structural levels (walk-forward, no leakage) ─────────────────────────────
def build_levels(df):
    """PDH/PDL, PWH/PWL (5d), IBH/IBL. All values NaN until enough history."""
    dates = df.index.date

    daily = df.resample("1D").agg(dh=("high","max"), dl=("low","min")).dropna()
    daily = daily[daily.index.dayofweek < 5]
    di    = daily.index

    pdh_map = {di[i].date(): daily["dh"].iloc[i-1] for i in range(1, len(di))}
    pdl_map = {di[i].date(): daily["dl"].iloc[i-1] for i in range(1, len(di))}
    pwh_map = {di[i].date(): daily["dh"].iloc[max(0,i-5):i].max() for i in range(5, len(di))}
    pwl_map = {di[i].date(): daily["dl"].iloc[max(0,i-5):i].min() for i in range(5, len(di))}

    ib_mask = (
        ((df.index.hour == 9)  & (df.index.minute >= 30)) |
        ((df.index.hour == 10) & (df.index.minute == 0))
    )
    ib = df[ib_mask].copy(); ib["_d"] = ib.index.date
    ibg     = ib.groupby("_d").agg(ibh=("high","max"), ibl=("low","min"))
    ibh_map = ibg["ibh"].to_dict(); ibl_map = ibg["ibl"].to_dict()

    return pd.DataFrame({
        "pdh": [pdh_map.get(d, np.nan) for d in dates],
        "pdl": [pdl_map.get(d, np.nan) for d in dates],
        "pwh": [pwh_map.get(d, np.nan) for d in dates],
        "pwl": [pwl_map.get(d, np.nan) for d in dates],
        "ibh": [ibh_map.get(d, np.nan) for d in dates],
        "ibl": [ibl_map.get(d, np.nan) for d in dates],
    }, index=df.index)


# ── directional efficiency ────────────────────────────────────────────────────
def dir_eff(close, period=14):
    """Kaufman ER: |net n-bar move| / sum(|1-bar moves|). 0=chop, 1=trend."""
    diff1 = np.abs(np.diff(close, prepend=close[0]))
    net   = pd.Series(close).diff(period).abs().values
    path  = pd.Series(diff1).rolling(period, min_periods=period).sum().values
    return np.where(path > 1e-6, net / path, np.nan)


# ── vectorized target precomputation ─────────────────────────────────────────
def precompute_targets(close, lvl_arr, min_tgt, max_tgt):
    """
    Per-bar: nearest level above (tgt_up) and below (tgt_dn) within [min_tgt, max_tgt] pts.
    Includes structural levels + round-50 multiples.
    """
    # structural
    dist_up  = lvl_arr - close[:, None]
    ok_up    = (dist_up >= min_tgt) & (dist_up <= max_tgt)
    best_up  = np.where(ok_up, dist_up, np.inf).min(axis=1)

    dist_dn  = close[:, None] - lvl_arr
    ok_dn    = (dist_dn >= min_tgt) & (dist_dn <= max_tgt)
    best_dn  = np.where(ok_dn, dist_dn, np.inf).min(axis=1)

    # round-50
    rnd = 50.0
    rnd_up   = np.ceil((close + min_tgt) / rnd) * rnd
    rd_up    = rnd_up - close
    best_up  = np.minimum(best_up,  np.where((rd_up >= min_tgt) & (rd_up <= max_tgt), rd_up, np.inf))

    rnd_dn   = np.floor((close - min_tgt) / rnd) * rnd
    rd_dn    = close - rnd_dn
    best_dn  = np.minimum(best_dn,  np.where((rd_dn >= min_tgt) & (rd_dn <= max_tgt), rd_dn, np.inf))

    tgt_up = np.where(np.isinf(best_up), np.nan, close + best_up)
    tgt_dn = np.where(np.isinf(best_dn), np.nan, close - best_dn)
    return tgt_up, tgt_dn


# ── backtest loop ─────────────────────────────────────────────────────────────
def _run_loop(close, high, low, z, tgt_up, tgt_dn, de,
              hour, minute, z_entry, z_confirm, stop_pts,
              de_min, de_max, persist, cooldown,
              trend_dir=None, trend_gate=False):
    """
    State-machine backtest.
    persist:  consecutive bars Z must stay extreme before entry (quality gate)
    cooldown: bars between exit and next entry (prevents over-trading)
    Returns list of (entry_px, exit_px, pnl, dir, bars, exit_type, entry_bar_idx).
    """
    n = len(close)
    trades = []

    in_trade      = False
    trade_dir     = 0
    entry_px      = 0.0
    stop_px       = 0.0
    target_px     = 0.0
    entry_bar     = 0
    last_exit_bar = -9999

    neg_ext_cnt   = 0  # consecutive bars at negative extreme
    pos_ext_cnt   = 0

    for i in range(1, n):
        zi = z[i]
        if zi != zi:
            continue

        if zi < -z_entry:
            neg_ext_cnt += 1
        else:
            neg_ext_cnt = 0
        if zi > z_entry:
            pos_ext_cnt += 1
        else:
            pos_ext_cnt = 0

        if in_trade:
            if hour[i] > 15 or (hour[i] == 15 and minute[i] >= 30):
                pnl = (close[i] - entry_px) * trade_dir
                trades.append((entry_px, close[i], pnl, trade_dir, i-entry_bar, "time", entry_bar))
                in_trade = False; last_exit_bar = i
                neg_ext_cnt = pos_ext_cnt = 0
                continue

            if trade_dir == 1:
                if low[i] <= stop_px:
                    trades.append((entry_px, stop_px, stop_px-entry_px, 1, i-entry_bar, "stop", entry_bar))
                    in_trade = False; last_exit_bar = i
                elif high[i] >= target_px:
                    trades.append((entry_px, target_px, target_px-entry_px, 1, i-entry_bar, "tgt", entry_bar))
                    in_trade = False; last_exit_bar = i
            else:
                if high[i] >= stop_px:
                    trades.append((entry_px, stop_px, entry_px-stop_px, -1, i-entry_bar, "stop", entry_bar))
                    in_trade = False; last_exit_bar = i
                elif low[i] <= target_px:
                    trades.append((entry_px, target_px, entry_px-target_px, -1, i-entry_bar, "tgt", entry_bar))
                    in_trade = False; last_exit_bar = i
            continue

        # cooldown
        if (i - last_exit_bar) < cooldown:
            continue

        # entry window 9:45-15:30
        if hour[i] < 9 or (hour[i] == 9 and minute[i] < 45):
            continue
        if hour[i] > 15 or (hour[i] == 15 and minute[i] > 30):
            continue

        dei = de[i]
        if dei != dei or dei < de_min or dei > de_max:
            continue

        # trend gate: skip direction that fights the 20d EMA trend
        td_i = trend_dir[i] if (trend_gate and trend_dir is not None) else 0

        # LONG: crossed back above -z_entry after persistent negative extreme
        prev_neg_cnt = sum(1 for j in range(max(1, i-persist), i) if z[j] == z[j] and z[j] < -z_entry)
        if neg_ext_cnt == 0 and prev_neg_cnt >= persist and zi > -z_confirm and zi < 0.5:
            if not (trend_gate and td_i == -1):  # skip if trend is down and gate active
                tu = tgt_up[i]
                if tu == tu:
                    entry_px = close[i]; stop_px = entry_px - stop_pts
                    target_px = tu
                    in_trade = True; trade_dir = 1; entry_bar = i

        prev_pos_cnt = sum(1 for j in range(max(1, i-persist), i) if z[j] == z[j] and z[j] > z_entry)
        if not in_trade and pos_ext_cnt == 0 and prev_pos_cnt >= persist and zi < z_confirm and zi > -0.5:
            if not (trend_gate and td_i == 1):  # skip if trend is up and gate active
                td = tgt_dn[i]
                if td == td:
                    entry_px = close[i]; stop_px = entry_px + stop_pts
                    target_px = td
                    in_trade = True; trade_dir = -1; entry_bar = i

    return trades


# ── metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(trades, n_trading_days=252):
    if len(trades) < 5:
        return {"sharpe":-99,"trades":len(trades),"wr":0.0,"rr":0.0,
                "total_pts":0.0,"avg_trade":0.0,"max_dd_pts":0.0,
                "expectancy":0.0,"n_tgt":0,"n_stop":0,"n_time":0}

    pnls   = np.array([t[2] for t in trades], dtype=np.float64)
    wins   = pnls[pnls > 0]; losses = pnls[pnls <= 0]
    wr     = len(wins) / len(pnls)
    avg_w  = wins.mean()        if len(wins)   > 0 else 0.0
    avg_l  = abs(losses.mean()) if len(losses) > 0 else 1.0
    rr     = avg_w / avg_l      if avg_l > 0 else 0.0

    trades_per_year = len(pnls) / max(n_trading_days / 252.0, 0.1)
    std_t  = pnls.std(ddof=1)
    sharpe = (pnls.mean() / std_t * np.sqrt(trades_per_year)) if std_t > 1e-8 else 0.0

    eq     = np.cumsum(pnls)
    max_dd = (np.maximum.accumulate(eq) - eq).max()

    return {
        "sharpe":     round(sharpe, 3),
        "trades":     len(trades),
        "wr":         round(wr, 4),
        "rr":         round(rr, 3),
        "avg_w":      round(avg_w, 2),
        "avg_l":      round(avg_l, 2),
        "total_pts":  round(pnls.sum(), 1),
        "avg_trade":  round(pnls.mean(), 2),
        "max_dd_pts": round(max_dd, 1),
        "expectancy": round(pnls.mean(), 2),
        "n_tgt":      sum(1 for t in trades if t[5] == "tgt"),
        "n_stop":     sum(1 for t in trades if t[5] == "stop"),
        "n_time":     sum(1 for t in trades if t[5] == "time"),
    }


# ── optimization ──────────────────────────────────────────────────────────────
PARAM_SPACE = {
    "d":          [0.55, 0.71],
    "z_entry":    [1.5, 1.8, 2.2],
    "z_confirm":  [0.5, 0.8],
    "stop_pts":   [10.0, 15.0, 20.0],
    "min_tgt":    [15.0, 20.0, 25.0],
    "max_tgt":    [35.0, 45.0, 60.0],
    "de_min":     [0.10, 0.20],
    "de_max":     [0.70, 1.00],
    "persist":    [2, 3, 5],
    "cooldown":   [15, 30, 45],
    "trend_gate": [True, False],  # True = only trade direction aligned with 20d EMA slope
}

def _score(m):
    if m["trades"] < MIN_TRADES or m["sharpe"] < 0:
        return -99.0
    # Pure Sharpe * sqrt(trade count) — penalizes low sample configs
    return m["sharpe"] * min(np.sqrt(m["trades"] / 100.0), 1.5)

def optimize(close, high, low, z_dict, lvl_arr, de, trend_dir, hour, minute,
             n_configs=150, n_trading_days=252):
    print(f"  Optimizing {n_configs} random configs on {n_trading_days} trading days...")
    rng     = random.Random(42)
    results = []
    tgt_cache = {}

    for ci in range(n_configs):
        cfg = {k: rng.choice(v) for k, v in PARAM_SPACE.items()}
        if cfg["z_confirm"] >= cfg["z_entry"]: continue
        if cfg["min_tgt"]   >= cfg["max_tgt"]:  continue

        key = (cfg["min_tgt"], cfg["max_tgt"])
        if key not in tgt_cache:
            tgt_cache[key] = precompute_targets(close, lvl_arr, cfg["min_tgt"], cfg["max_tgt"])
        tu, td = tgt_cache[key]

        z = z_dict[cfg["d"]]
        trades = _run_loop(close, high, low, z, tu, td, de, hour, minute,
                           cfg["z_entry"], cfg["z_confirm"], cfg["stop_pts"],
                           cfg["de_min"], cfg["de_max"],
                           cfg["persist"], cfg["cooldown"],
                           trend_dir=trend_dir, trend_gate=cfg.get("trend_gate", False))
        m = compute_metrics(trades, n_trading_days)
        results.append((_score(m), cfg, m))

        if (ci + 1) % 50 == 0:
            valid = sum(1 for r in results if r[0] > -99)
            best  = max((r[2]["sharpe"] for r in results if r[0] > -99), default=0)
            print(f"    [{ci+1}/{n_configs}]  valid={valid}  best_sharpe={best:.2f}")

    results.sort(key=lambda x: -x[0])
    valid = sum(1 for r in results if r[0] > -99)
    print(f"  Done. Valid configs: {valid}/{len(results)}")
    return results


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  GEX-FracDiff Level-Momentum (GFLM) Strategy")
    print("=" * 65)

    # load
    print("\nLoading NQ 1m data...")
    df = load_nq()
    print(f"  {len(df):,} bars  {df.index[0].date()} -> {df.index[-1].date()}")

    # fracdiff Z
    print("\nComputing fracdiff Z (d=0.55 and d=0.71)...")
    log_px = np.log(df["close"].values)
    z_full = {0.55: compute_z(log_px, 0.55), 0.71: compute_z(log_px, 0.71)}

    # structural levels
    print("Building walk-forward structural levels...")
    levels = build_levels(df)

    # directional efficiency
    de_full = dir_eff(df["close"].values)

    # 20-day EMA of close for trend direction
    ema20d = df["close"].resample("1D").last().ewm(span=20, adjust=False).mean()
    ema20d_1m = ema20d.shift(1).reindex(df.index, method="ffill")
    # +1 = close > ema (uptrend), -1 = close < ema (downtrend)
    trend_dir_full = np.where(df["close"].values > ema20d_1m.values, 1,
                     np.where(df["close"].values < ema20d_1m.values, -1, 0))

    # RTH slice helper
    def rth_slice(start, end):
        mask = (
            (df.index >= start) & (df.index <= end) &
            ((df.index.hour > 9) | ((df.index.hour == 9) & (df.index.minute >= 30))) &
            (df.index.hour < 16)
        )
        idx = np.where(mask)[0]
        sub = df.iloc[idx]
        return {
            "close":      df["close"].values[idx],
            "high":       df["high"].values[idx],
            "low":        df["low"].values[idx],
            "z_dict":     {d: z_full[d][idx] for d in z_full},
            "lvl":        levels.values[idx],
            "de":         de_full[idx],
            "trend_dir":  trend_dir_full[idx],
            "hour":       sub.index.hour,
            "minute":     sub.index.minute,
            "index":      sub.index,
            "n_days":     int(len(sub.index.normalize().unique())),
        }

    print("\nSlicing RTH periods...")
    opt  = rth_slice(OPT_START,  OPT_END)
    test = rth_slice(TEST_START, TEST_END)
    print(f"  Optimize (2024): {len(opt['close']):,} RTH bars  {opt['n_days']} trading days")
    print(f"  Test     (2025): {len(test['close']):,} RTH bars  {test['n_days']} trading days")

    # optimize on 2024
    print("\n" + "-"*65)
    print("  PHASE 1: OPTIMIZATION  (2024 data, not seen by 2025 test)")
    print("-"*65)
    opt_results = optimize(opt["close"], opt["high"], opt["low"],
                           opt["z_dict"], opt["lvl"], opt["de"],
                           opt["trend_dir"], opt["hour"], opt["minute"],
                           n_configs=200, n_trading_days=opt["n_days"])

    valid = [(s,c,m) for s,c,m in opt_results if s > -99]
    if not valid:
        print("  No valid configs. Try lowering MIN_TRADES.")
        return

    print(f"\n  Top 5 configs on 2024:")
    hdr = f"  {'d':>4}  {'ze':>4}  {'zc':>4}  {'stp':>4}  {'mt':>3}  {'xt':>3}  {'dem':>4}  {'deM':>4}  {'per':>3}  {'cdwn':>4}  Sharpe  WR%   R:R  Trades"
    print(hdr)
    for sc, cfg, m in valid[:5]:
        print(f"  {cfg['d']:>4.2f}  {cfg['z_entry']:>4.1f}  {cfg['z_confirm']:>4.1f}"
              f"  {cfg['stop_pts']:>4.0f}  {cfg['min_tgt']:>3.0f}  {cfg['max_tgt']:>3.0f}"
              f"  {cfg['de_min']:>4.2f}  {cfg['de_max']:>4.2f}  {cfg['persist']:>3}  {cfg['cooldown']:>4}"
              f"  {m['sharpe']:>6.2f}  {m['wr']*100:>4.1f}  {m['rr']:>4.2f}  {m['trades']:>6}")

    best_score, best_cfg, best_opt_m = valid[0]
    print(f"\n  Best config: Sharpe={best_opt_m['sharpe']:.2f}  WR={best_opt_m['wr']*100:.1f}%"
          f"  R:R={best_opt_m['rr']:.2f}  Trades={best_opt_m['trades']}")

    # test on 2025
    print("\n" + "-"*65)
    print("  PHASE 2: OUT-OF-SAMPLE TEST  (2025)")
    print("-"*65)

    key    = (best_cfg["min_tgt"], best_cfg["max_tgt"])
    tu, td = precompute_targets(test["close"], test["lvl"], best_cfg["min_tgt"], best_cfg["max_tgt"])
    z_t    = test["z_dict"][best_cfg["d"]]

    test_trades = _run_loop(
        test["close"], test["high"], test["low"],
        z_t, tu, td, test["de"], test["hour"], test["minute"],
        best_cfg["z_entry"], best_cfg["z_confirm"], best_cfg["stop_pts"],
        best_cfg["de_min"], best_cfg["de_max"],
        best_cfg["persist"], best_cfg["cooldown"],
        trend_dir=test["trend_dir"], trend_gate=best_cfg.get("trend_gate", False)
    )

    m = compute_metrics(test_trades, test["n_days"])

    print(f"\n  Best config:")
    for k, v in best_cfg.items():
        print(f"    {k:12s}: {v}")

    print(f"\n  ======================================================")
    print(f"  2025 OUT-OF-SAMPLE RESULTS  (GFLM)")
    print(f"  ======================================================")
    print(f"  Trades          : {m['trades']}")
    print(f"  Win Rate        : {m['wr']*100:.1f}%")
    print(f"  Avg Win         : +{m['avg_w']:.1f} pts")
    print(f"  Avg Loss        : -{m['avg_l']:.1f} pts")
    print(f"  R:R             : {m['rr']:.2f}")
    print(f"  Expectancy      : {m['expectancy']:.2f} pts/trade")
    print(f"  Total pts       : {m['total_pts']:.1f}")
    print(f"  Max DD          : {m['max_dd_pts']:.1f} pts")
    print(f"  Per-trade Sharpe: {m['sharpe']:.2f}")
    print(f"  Exit: tgt={m['n_tgt']}  stop={m['n_stop']}  time={m['n_time']}")
    print(f"  ======================================================")

    # monthly breakdown
    if test_trades:
        rows = []
        for t in test_trades:
            bar_idx = min(t[6], len(test["index"])-1)
            rows.append({
                "month": test["index"][bar_idx].strftime("%Y-%m"),
                "pnl": t[2], "win": t[2] > 0,
            })
        mdf = pd.DataFrame(rows)
        mg  = mdf.groupby("month")
        print(f"\n  Monthly breakdown (2025):")
        print(f"  {'Month':>7}  {'Trd':>4}  {'W':>3}  {'WR%':>5}  {'Total':>8}")
        tot_t = tot_w = tot_p = 0
        for mth, g in mg:
            n  = len(g); w = g["win"].sum(); p = g["pnl"].sum()
            tot_t += n; tot_w += w; tot_p += p
            flag = " +" if p > 0 else "  "
            print(f"  {mth:>7}  {n:>4}  {w:>3}  {w/n*100:>5.1f}  {p:>8.1f}{flag}")
        print(f"  {'TOTAL':>7}  {tot_t:>4}  {tot_w:>3}  {tot_w/tot_t*100:>5.1f}  {tot_p:>8.1f}")

    # direction breakdown
    if test_trades:
        longs  = [t for t in test_trades if t[3] == 1]
        shorts = [t for t in test_trades if t[3] == -1]
        print(f"\n  Direction breakdown:")
        for side, tlist in [("LONG", longs), ("SHORT", shorts)]:
            if tlist:
                p = [t[2] for t in tlist]; w = sum(1 for x in p if x > 0)
                print(f"    {side:5s}: n={len(tlist):3d}  wr={w/len(tlist)*100:.1f}%  "
                      f"avg={np.mean(p):.1f}  total={sum(p):.1f}")

    # baseline comparison
    print(f"\n  -- Baseline (fracdiff only, fixed {best_cfg['stop_pts']*1.5:.0f}pt target) --")
    base_trades = _baseline_run(
        test["close"], test["high"], test["low"], z_t,
        test["de"], test["trend_dir"], test["hour"], test["minute"],
        best_cfg["z_entry"], best_cfg["z_confirm"], best_cfg["stop_pts"],
        best_cfg["de_min"], best_cfg["de_max"],
        best_cfg["persist"], best_cfg["cooldown"],
        trend_gate=best_cfg.get("trend_gate", False),
        fixed_tgt=best_cfg["stop_pts"] * 1.5
    )
    bm = compute_metrics(base_trades, test["n_days"])
    print(f"  Baseline: trades={bm['trades']}  wr={bm['wr']*100:.1f}%  "
          f"rr={bm['rr']:.2f}  sharpe={bm['sharpe']:.2f}  total={bm['total_pts']:.1f}")
    print(f"  GFLM:     trades={m['trades']}  wr={m['wr']*100:.1f}%  "
          f"rr={m['rr']:.2f}  sharpe={m['sharpe']:.2f}  total={m['total_pts']:.1f}")
    print(f"  Lift from level targets: {m['total_pts']-bm['total_pts']:+.1f} pts")

    # save
    out = {
        "config": best_cfg,
        "opt_2024": best_opt_m,
        "test_2025": m,
    }
    with open(os.path.join(OUT_DIR, "best_config.json"), "w") as f:
        json.dump(out, f, indent=2)

    if test_trades:
        rows2 = []
        for t in test_trades:
            bar_idx = min(t[6], len(test["index"])-1)
            rows2.append({
                "ts":      str(test["index"][bar_idx]),
                "entry":   t[0], "exit": t[1], "pnl": t[2],
                "dir":     "L" if t[3]==1 else "S",
                "bars":    t[4], "how":  t[5],
            })
        pd.DataFrame(rows2).to_csv(os.path.join(OUT_DIR, "trades_2025.csv"), index=False)
        print(f"\n  Saved: results/best_config.json  results/trades_2025.csv")

    print("\n  Done.")


def _baseline_run(close, high, low, z, de, trend_dir, hour, minute,
                  z_entry, z_confirm, stop_pts, de_min, de_max,
                  persist, cooldown, trend_gate=False, fixed_tgt=20.0):
    """Same logic but fixed target (no levels)."""
    n = len(close); trades = []
    in_trade = False; trade_dir = 0
    entry_px = 0.0; stop_px = 0.0; target_px = 0.0; entry_bar = 0
    last_exit = -9999; neg_cnt = 0; pos_cnt = 0

    for i in range(1, n):
        zi = z[i]
        if zi != zi: continue
        if zi < -z_entry: neg_cnt += 1
        else: neg_cnt = 0
        if zi > z_entry: pos_cnt += 1
        else: pos_cnt = 0

        if in_trade:
            if hour[i] > 15 or (hour[i] == 15 and minute[i] >= 30):
                pnl = (close[i] - entry_px) * trade_dir
                trades.append((entry_px, close[i], pnl, trade_dir, i-entry_bar, "time", entry_bar))
                in_trade = False; last_exit = i; continue
            if trade_dir == 1:
                if low[i] <= stop_px:
                    trades.append((entry_px, stop_px, stop_px-entry_px, 1, i-entry_bar, "stop", entry_bar))
                    in_trade = False; last_exit = i
                elif high[i] >= target_px:
                    trades.append((entry_px, target_px, target_px-entry_px, 1, i-entry_bar, "tgt", entry_bar))
                    in_trade = False; last_exit = i
            else:
                if high[i] >= stop_px:
                    trades.append((entry_px, stop_px, entry_px-stop_px, -1, i-entry_bar, "stop", entry_bar))
                    in_trade = False; last_exit = i
                elif low[i] <= target_px:
                    trades.append((entry_px, target_px, entry_px-target_px, -1, i-entry_bar, "tgt", entry_bar))
                    in_trade = False; last_exit = i
            continue

        if (i - last_exit) < cooldown: continue
        if hour[i] < 9 or (hour[i] == 9 and minute[i] < 45): continue
        if hour[i] > 15 or (hour[i] == 15 and minute[i] > 30): continue
        dei = de[i]
        if dei != dei or dei < de_min or dei > de_max: continue
        td_i = trend_dir[i] if (trend_gate and trend_dir is not None) else 0

        prev_neg = sum(1 for j in range(max(1,i-persist),i) if z[j]==z[j] and z[j]<-z_entry)
        if neg_cnt == 0 and prev_neg >= persist and zi > -z_confirm and zi < 0.5:
            if not (trend_gate and td_i == -1):
                entry_px = close[i]; stop_px = entry_px - stop_pts
                target_px = entry_px + fixed_tgt
                in_trade = True; trade_dir = 1; entry_bar = i

        prev_pos = sum(1 for j in range(max(1,i-persist),i) if z[j]==z[j] and z[j]>z_entry)
        if not in_trade and pos_cnt == 0 and prev_pos >= persist and zi < z_confirm and zi > -0.5:
            if not (trend_gate and td_i == 1):
                entry_px = close[i]; stop_px = entry_px + stop_pts
                target_px = entry_px - fixed_tgt
                in_trade = True; trade_dir = -1; entry_bar = i

    return trades


if __name__ == "__main__":
    main()
