"""Fracdiff strategy optimizer — walk-forward, repo conventions.

Train 2018-2022 (coarse grid) -> Val 2023 (regime filters + direction) ->
Test 2024 holdout + OOS 2025-26 (final configs only, never optimized on).

Anti-leak: fracdiff/z are causal rolling ops. Daily regime series shifted
on the DAILY grid before reindex to TF grid (repo rule #1).
"""
from __future__ import annotations
import sys, time
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from scipy.signal import lfilter

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "nq_amd"))
from loader import load_nq  # noqa: E402

COST_PTS = 0.75
POINT_VALUE = 20.0
NLAGS = 100
MAX_HOLD = 400

TRAIN = ("2018-01-01", "2022-12-31")
VAL = ("2023-01-01", "2023-12-31")
TEST = ("2024-01-01", "2024-12-31")
OOS = ("2025-01-01", "2026-12-31")

CACHE = Path(__file__).parent / "nq1m_2018.parquet"


def frac_weights(d, n):
    w = np.empty(n + 1)
    w[0] = 1.0
    for k in range(1, n + 1):
        w[k] = w[k - 1] * (k - 1 - d) / k
    return w


def load_1m():
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    df = load_nq(ROOT / "1Min_NQ.csv", start="2018-01-01")
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.to_parquet(CACHE)
    return df


class TFData:
    """Per-timeframe precomputed arrays + regime masks."""

    def __init__(self, df: pd.DataFrame, df1: pd.DataFrame):
        self.ts = df.index
        self.o = df["Open"].to_numpy()
        self.h = df["High"].to_numpy()
        self.l = df["Low"].to_numpy()
        self.c = df["Close"].to_numpy()
        self.src = ((df["High"] + df["Low"] + df["Close"]) / 3.0).to_numpy()
        # ATR(14) on this TF, known at bar close (uses bars <= i)
        tr = np.maximum(df["High"] - df["Low"],
                        np.maximum((df["High"] - df["Close"].shift(1)).abs(),
                                   (df["Low"] - df["Close"].shift(1)).abs()))
        self.atr = tr.rolling(14).mean().to_numpy()
        # --- daily regime series (shift on DAILY grid, then reindex) ---
        dc = df1["Close"].resample("1D").last().dropna()
        ema50_d = dc.ewm(span=50).mean().shift(1)
        trend_up_d = (dc.shift(1) > ema50_d)
        dh = df1["High"].resample("1D").max().dropna()
        dl = df1["Low"].resample("1D").min().dropna()
        dtr = np.maximum(dh - dl, np.maximum((dh - dc.shift(1)).abs(), (dl - dc.shift(1)).abs()))
        datr = dtr.rolling(14).mean()
        datr_pct = datr.rolling(252, min_periods=100).rank(pct=True).shift(1)
        self.trend_up = trend_up_d.reindex(self.ts, method="ffill").fillna(False).to_numpy()
        self.vol_pct = datr_pct.reindex(self.ts, method="ffill").to_numpy()
        # session mask: RTH 09:30-16:00 ET (index is ET)
        mins = self.ts.hour * 60 + self.ts.minute
        self.rth = (mins >= 570) & (mins < 960)
        # split boundaries
        self.splits = {}
        for name, (s, e) in dict(train=TRAIN, val=VAL, test=TEST, oos=OOS).items():
            i0 = self.ts.searchsorted(pd.Timestamp(s))
            i1 = self.ts.searchsorted(pd.Timestamp(e) + pd.Timedelta(days=1))
            self.splits[name] = (i0, i1)
        self._fd = {}
        self._z = {}

    def z(self, d, z_len):
        key = (d, z_len)
        if key not in self._z:
            if d not in self._fd:
                fd = lfilter(frac_weights(d, NLAGS), [1.0], self.src)
                fd[:NLAGS] = np.nan
                self._fd[d] = pd.Series(fd)
            fds = self._fd[d]
            mean = fds.rolling(z_len).mean()
            std = fds.rolling(z_len).std().replace(0.0, np.nan)
            self._z[key] = ((fds - mean) / std).to_numpy()
        return self._z[key]


def signals(z, band, mode):
    zp = np.roll(z, 1)
    zp[0] = np.nan
    cross_dn = (zp >= -band) & (z < -band)   # z crosses under -band
    cross_up = (zp <= band) & (z > band)     # z crosses over +band
    zc = ((zp < 0) & (z >= 0)) | ((zp > 0) & (z <= 0))
    if mode == "revert":
        return cross_dn, cross_up, zc        # long below, short above
    return cross_up, cross_dn, zc            # momentum: trade with break


def backtest(tf: TFData, long_e, short_e, zc, split, stop_atr=None, exit_mode="zc",
             rr=2.0, ent_mask=None):
    """Event-driven. Entry: signal bar close -> next open. Exits:
    zc: zero-cross (next open), optional ATR stop intrabar.
    rr: ATR stop + rr*stop target, intrabar. MAX_HOLD time stop both modes."""
    i0, i1 = tf.splits[split]
    o, h, l, c = tf.o, tf.h, tf.l, tf.c
    le, se = long_e.copy(), short_e.copy()
    if ent_mask is not None:
        le &= ent_mask
        se &= ent_mask
    cand = np.where(le | se)[0]
    cand = cand[(cand >= i0) & (cand < i1 - 2)]
    zc_idx = np.where(zc)[0]
    trades = []
    nxt = i0
    for i in cand:
        if i < nxt or np.isnan(tf.atr[i]):
            continue
        d_ = 1 if le[i] else -1
        ei = i + 1
        ep = o[ei]
        stop = ep - d_ * stop_atr * tf.atr[i] if stop_atr else None
        tgt = ep + d_ * rr * stop_atr * tf.atr[i] if (stop_atr and exit_mode == "rr") else None
        # zero-cross exit bar (fill next open)
        if exit_mode == "zc":
            k = np.searchsorted(zc_idx, ei)
            zc_fill = zc_idx[k] + 1 if k < len(zc_idx) else i1 - 1
        else:
            zc_fill = i1 - 1
        end = min(ei + MAX_HOLD, zc_fill, i1 - 1)
        xp, xi = None, None
        if stop is not None:
            for j in range(ei, end):
                lo, hi = l[j], h[j]
                if d_ == 1:
                    if lo <= stop:
                        xp, xi = stop, j
                        break
                    if tgt is not None and hi >= tgt:
                        xp, xi = tgt, j
                        break
                else:
                    if hi >= stop:
                        xp, xi = stop, j
                        break
                    if tgt is not None and lo <= tgt:
                        xp, xi = tgt, j
                        break
        if xp is None:
            xi = end
            xp = o[xi] if xi < len(o) else c[-1]
        pts = (xp - ep) * d_ - COST_PTS
        trades.append((ei, xi, d_, pts))
        nxt = xi + 1
    return trades


def metrics(trades, tf: TFData, split):
    if len(trades) < 5:
        return dict(n=len(trades))
    tr = pd.DataFrame(trades, columns=["ei", "xi", "dir", "pts"])
    net = tr["pts"]
    i0, i1 = tf.splits[split]
    yrs = max((tf.ts[i1 - 1] - tf.ts[i0]).days / 365.25, 0.1)
    daily = pd.Series(net.values, index=tf.ts[tr["xi"].values]).resample("1D").sum()
    daily = daily[daily.index.dayofweek < 5]
    ann = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else np.nan
    eq = net.cumsum() * POINT_VALUE
    return dict(n=len(tr), wr=(net > 0).mean() * 100, avg=net.mean(),
                tot=net.sum(), usd=net.sum() * POINT_VALUE,
                pt_sh=net.mean() / net.std() if net.std() > 0 else np.nan,
                ann_sh=ann, dd=(eq - eq.cummax()).min(), tpy=len(tr) / yrs,
                long_avg=net[tr["dir"] == 1].mean(), short_avg=net[tr["dir"] == -1].mean(),
                long_n=int((tr["dir"] == 1).sum()), short_n=int((tr["dir"] == -1).sum()))


def fmt(m):
    if m.get("n", 0) < 5:
        return f"n={m.get('n', 0)} (too few)"
    return (f"n={m['n']:>4} ({m['tpy']:>4.0f}/yr) WR {m['wr']:4.1f}% avg {m['avg']:+6.2f} "
            f"tot {m['tot']:+8.0f}pts ${m['usd']:+9,.0f} ptSh {m['pt_sh']:+.3f} "
            f"annSh {m['ann_sh']:+5.2f} DD ${m['dd']:+9,.0f}")


def main():
    t0 = time.time()
    print("Loading 1m 2018+ ...", flush=True)
    df1 = load_1m()
    print(f"{len(df1):,} bars {df1.index.min()} -> {df1.index.max()}  [{time.time()-t0:.0f}s]", flush=True)

    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    tfd = {}
    for label, rule in [("5m", "5min"), ("15m", "15min"), ("1h", "60min")]:
        tfd[label] = TFData(df1.resample(rule).agg(agg).dropna(), df1)
    print(f"TF data built [{time.time()-t0:.0f}s]", flush=True)

    # ---------------- Stage 1: coarse grid on TRAIN ----------------
    grid = list(product(
        ["5m", "15m", "1h"],
        [0.3, 0.45, 0.6],          # d
        [50, 100, 200],            # z_len
        [1.5, 2.0, 2.5, 3.0],      # band
        ["revert", "momo"],
        [("zc", None), ("zc", 1.5), ("zc", 3.0), ("rr", 1.5)],
    ))
    rows = []
    for gi, (tfl, d, zl, band, mode, (exm, stp)) in enumerate(grid):
        tf = tfd[tfl]
        z = tf.z(d, zl)
        le, se, zc = signals(z, band, mode)
        tr = backtest(tf, le, se, zc, "train", stop_atr=stp, exit_mode=exm)
        m = metrics(tr, tf, "train")
        m.update(tf_=tfl, d=d, zl=zl, band=band, mode=mode, exm=exm, stp=stp)
        rows.append(m)
        if (gi + 1) % 96 == 0:
            print(f"  grid {gi+1}/{len(grid)} [{time.time()-t0:.0f}s]", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(Path(__file__).parent / "fracdiff_grid_train.csv", index=False)
    ok = res[(res["n"] >= 60) & res["ann_sh"].notna()].sort_values("ann_sh", ascending=False)
    print(f"\n=== Stage 1 (train 2018-2022): {len(res)} configs, {len(ok)} with n>=60 ===")
    print(f"configs with positive train annSh: {(ok['ann_sh'] > 0).sum()}")
    cols = ["tf_", "d", "zl", "band", "mode", "exm", "stp", "n", "wr", "avg", "tot", "pt_sh", "ann_sh"]
    print(ok.head(20)[cols].round(3).to_string(index=False))

    # ---------------- Stage 2: top-15 -> VAL with filters ----------------
    top = ok.head(15)
    filters = ["none", "with_trend", "counter_trend", "vol_hi", "vol_lo", "rth"]
    dirs = ["both", "long", "short"]
    rows2 = []
    print(f"\n=== Stage 2 (val 2023): top {len(top)} x {len(filters)} filters x {len(dirs)} dirs ===", flush=True)
    for _, cfg in top.iterrows():
        tf = tfd[cfg["tf_"]]
        z = tf.z(cfg["d"], cfg["zl"])
        le0, se0, zc = signals(z, cfg["band"], cfg["mode"])
        for filt, dr in product(filters, dirs):
            le, se = le0.copy(), se0.copy()
            if filt == "with_trend":
                le, se = le & tf.trend_up, se & ~tf.trend_up
            elif filt == "counter_trend":
                le, se = le & ~tf.trend_up, se & tf.trend_up
            elif filt == "vol_hi":
                mask = tf.vol_pct > 0.5
                le, se = le & mask, se & mask
            elif filt == "vol_lo":
                mask = tf.vol_pct <= 0.5
                le, se = le & mask, se & mask
            elif filt == "rth":
                le, se = le & tf.rth, se & tf.rth
            if dr == "long":
                se &= False
            elif dr == "short":
                le &= False
            mv = metrics(backtest(tf, le, se, zc, "val", stop_atr=cfg["stp"], exit_mode=cfg["exm"]), tf, "val")
            mt = metrics(backtest(tf, le, se, zc, "train", stop_atr=cfg["stp"], exit_mode=cfg["exm"]), tf, "train")
            rows2.append(dict(tf_=cfg["tf_"], d=cfg["d"], zl=cfg["zl"], band=cfg["band"],
                              mode=cfg["mode"], exm=cfg["exm"], stp=cfg["stp"], filt=filt, dir=dr,
                              tr_n=mt.get("n"), tr_ann=mt.get("ann_sh"),
                              v_n=mv.get("n"), v_wr=mv.get("wr"), v_avg=mv.get("avg"),
                              v_tot=mv.get("tot"), v_ann=mv.get("ann_sh"), v_dd=mv.get("dd")))
    r2 = pd.DataFrame(rows2)
    r2.to_csv(Path(__file__).parent / "fracdiff_grid_val.csv", index=False)
    good = r2[(r2["v_n"] >= 20) & (r2["tr_ann"] > 0) & r2["v_ann"].notna()]
    good = good.sort_values("v_ann", ascending=False)
    print(good.head(20).round(3).to_string(index=False))

    # ---------------- Final: top-3 robust -> TEST + OOS ----------------
    # dedupe by underlying config, require both train and val positive
    final = good[good["v_ann"] > 0].drop_duplicates(
        subset=["tf_", "d", "zl", "band", "mode", "exm", "stp"]).head(3)
    print(f"\n=== FINAL (test 2024 holdout + OOS 2025-26) — {len(final)} configs ===")
    for _, cfg in final.iterrows():
        tf = tfd[cfg["tf_"]]
        z = tf.z(cfg["d"], cfg["zl"])
        le, se, zc = signals(z, cfg["band"], cfg["mode"])
        if cfg["filt"] == "with_trend":
            le, se = le & tf.trend_up, se & ~tf.trend_up
        elif cfg["filt"] == "counter_trend":
            le, se = le & ~tf.trend_up, se & tf.trend_up
        elif cfg["filt"] == "vol_hi":
            le, se = le & (tf.vol_pct > 0.5), se & (tf.vol_pct > 0.5)
        elif cfg["filt"] == "vol_lo":
            le, se = le & (tf.vol_pct <= 0.5), se & (tf.vol_pct <= 0.5)
        elif cfg["filt"] == "rth":
            le, se = le & tf.rth, se & tf.rth
        if cfg["dir"] == "long":
            se = se & False
        elif cfg["dir"] == "short":
            le = le & False
        print(f"\nCONFIG {cfg['tf_']} d={cfg['d']} zl={cfg['zl']} band={cfg['band']} "
              f"mode={cfg['mode']} exit={cfg['exm']}/stop={cfg['stp']} filt={cfg['filt']} dir={cfg['dir']}")
        for split in ["train", "val", "test", "oos"]:
            m = metrics(backtest(tf, le, se, zc, split, stop_atr=cfg["stp"], exit_mode=cfg["exm"]), tf, split)
            print(f"  {split:5}: {fmt(m)}")
    print(f"\ndone [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
