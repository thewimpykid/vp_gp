"""Fractional-differencing mean-reversion backtest on NQ (port of FCVECM Pine indicator).

Logic (matches Pine):
  - fracdiff weights: w[0]=1, w[k]=w[k-1]*(k-1-d)/k, truncated at N lags
  - fd_t = sum_k w[k] * src[t-k], src = hlc3
  - z = (fd - SMA(fd, z_len)) / STDEV(fd, z_len)
  - LONG  when z crosses under -band_w
  - SHORT when z crosses over  +band_w
  - EXIT  when z crosses zero

Execution: signal evaluated on bar close, filled at next bar open.
One position at a time; entries ignored while in position.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "nq_amd"))
from loader import load_nq  # noqa: E402

D = 0.45
N = 100
Z_LEN = 100
BAND = 2.618
POINT_VALUE = 20.0      # NQ full contract $/pt
COST_PTS = 0.75         # round-trip slippage + commission in points


def frac_weights(d: float, n: int) -> np.ndarray:
    w = np.empty(n + 1)
    w[0] = 1.0
    for k in range(1, n + 1):
        w[k] = w[k - 1] * (k - 1 - d) / k
    return w


def run_tf(df: pd.DataFrame, label: str) -> dict:
    o = df["Open"].to_numpy()
    src = ((df["High"] + df["Low"] + df["Close"]) / 3.0).to_numpy()

    w = frac_weights(D, N)
    fd = lfilter(w, [1.0], src)
    fd[: N] = np.nan

    fds = pd.Series(fd)
    mean = fds.rolling(Z_LEN).mean()
    std = fds.rolling(Z_LEN).std()
    z = ((fds - mean) / std.replace(0.0, np.nan)).to_numpy()

    zp = np.roll(z, 1); zp[0] = np.nan
    long_sig = (zp >= -BAND) & (z < -BAND)
    short_sig = (zp <= BAND) & (z > BAND)
    exit_sig = ((zp < 0) & (z >= 0)) | ((zp > 0) & (z <= 0))

    ts = df.index
    n_bars = len(df)
    trades = []
    pos = 0          # +1 long, -1 short
    entry_px = 0.0
    entry_i = 0

    for i in range(N + Z_LEN, n_bars - 1):
        if np.isnan(z[i]):
            continue
        if pos == 0:
            if long_sig[i]:
                pos, entry_px, entry_i = 1, o[i + 1], i + 1
            elif short_sig[i]:
                pos, entry_px, entry_i = -1, o[i + 1], i + 1
        else:
            if exit_sig[i]:
                exit_px = o[i + 1]
                pts = (exit_px - entry_px) * pos
                trades.append({
                    "entry_ts": ts[entry_i], "exit_ts": ts[i + 1],
                    "dir": pos, "pts": pts, "net_pts": pts - COST_PTS,
                    "bars": i + 1 - entry_i,
                })
                pos = 0
    # force-flat at end
    if pos != 0:
        pts = (df["Close"].iloc[-1] - entry_px) * pos
        trades.append({"entry_ts": ts[entry_i], "exit_ts": ts[-1], "dir": pos,
                       "pts": pts, "net_pts": pts - COST_PTS, "bars": n_bars - 1 - entry_i})

    tr = pd.DataFrame(trades)
    out = {"tf": label, "bars": n_bars, "trades": len(tr)}
    if tr.empty:
        return out

    net = tr["net_pts"]
    out["wr"] = (net > 0).mean() * 100
    out["avg_pts"] = net.mean()
    out["med_pts"] = net.median()
    out["tot_pts"] = net.sum()
    out["tot_usd"] = net.sum() * POINT_VALUE
    out["gross_pts"] = tr["pts"].sum()
    out["pt_sharpe"] = net.mean() / net.std() if net.std() > 0 else np.nan
    out["avg_hold_bars"] = tr["bars"].mean()

    # daily pnl -> annualized sharpe + max dd
    daily = tr.set_index("exit_ts")["net_pts"].resample("1D").sum()
    daily = daily[daily.index.dayofweek < 5]
    yrs = (ts[-1] - ts[0]).days / 365.25
    out["trades_yr"] = len(tr) / yrs
    if daily.std() > 0:
        out["ann_sharpe"] = daily.mean() / daily.std() * np.sqrt(252)
    eq = net.cumsum() * POINT_VALUE
    out["max_dd_usd"] = (eq - eq.cummax()).min()

    # long/short split
    for d_, name in [(1, "long"), (-1, "short")]:
        sub = tr[tr["dir"] == d_]["net_pts"]
        out[f"{name}_n"] = len(sub)
        out[f"{name}_avg"] = sub.mean() if len(sub) else np.nan

    # yearly
    out["_yearly"] = tr.set_index("exit_ts")["net_pts"].resample("YE").agg(["sum", "count"])
    return out


def main():
    print("Loading 1Min_NQ.csv ...", flush=True)
    df1 = load_nq(ROOT / "1Min_NQ.csv", start="2021-06-09", end="2026-06-09")
    print(f"{len(df1):,} 1m bars  {df1.index.min()} -> {df1.index.max()}\n", flush=True)

    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    tfs = {
        "1m": df1,
        "5m": df1.resample("5min").agg(agg).dropna(),
        "15m": df1.resample("15min").agg(agg).dropna(),
        "1h": df1.resample("60min").agg(agg).dropna(),
        "1D": df1.resample("1D").agg(agg).dropna(),
    }

    results = []
    for label, df in tfs.items():
        r = run_tf(df, label)
        results.append(r)
        print(f"--- {label} ({r['bars']:,} bars) ---")
        if r["trades"] == 0:
            print("no trades\n")
            continue
        print(f"trades {r['trades']}  ({r['trades_yr']:.0f}/yr)   WR {r['wr']:.1f}%   "
              f"avg {r['avg_pts']:+.2f} pts  med {r['med_pts']:+.2f}")
        print(f"net total {r['tot_pts']:+,.0f} pts = ${r['tot_usd']:+,.0f}  "
              f"(gross {r['gross_pts']:+,.0f} pts, cost {COST_PTS} pt/RT)")
        print(f"per-trade Sharpe {r['pt_sharpe']:.3f}   ann Sharpe {r.get('ann_sharpe', float('nan')):.2f}   "
              f"maxDD ${r['max_dd_usd']:+,.0f}   hold {r['avg_hold_bars']:.0f} bars")
        print(f"long  n={r['long_n']}  avg {r['long_avg']:+.2f} | "
              f"short n={r['short_n']}  avg {r['short_avg']:+.2f}")
        print("yearly net pts:")
        print(r["_yearly"].rename(columns={"sum": "net_pts", "count": "n"}).round(1).to_string())
        print()


if __name__ == "__main__":
    main()
