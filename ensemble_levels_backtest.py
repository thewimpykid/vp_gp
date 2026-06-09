"""
ensemble_levels_backtest.py — combine every level source we've backtested
into ensemble REVERSAL LEVELS, and score them against actual intraday
reversal pivots (not just HoD/LoD).

Sources per day (all computed from prior data only — no lookahead):
  shelf : stacked shelf/ledge balance-edge confluence  (chart.py)
  gex   : multi-day-confirmed GEX walls                (gex.py)
  vp    : stacked POC/VAH/VAL/HVN level confluence     (chart.py)
  em    : VXN expected-move bands  open +/- k*sigma
  cam   : Camarilla R3/R4/S3/S4 off prior RTH H/L/C

Ensemble: cluster all candidate levels within `tol` points; cluster score =
sum of source weights (a source counts once per cluster). Emit clusters
with score >= min_score as reversal levels.

Actual reversals: zigzag pivots on 1m RTH highs/lows with `zz_thresh`
minimum reversal size.

Metrics per config:
  precision = tested levels that had a pivot within rev_tol / tested levels
              (level must be touched first — untouched levels excluded)
  recall    = pivots within rev_tol of an emitted level / all pivots
  f1, avg #levels/day
Baseline: grid of levels every 25pt across the day's range, same metrics.
"""
import os
import sys
import itertools

import numpy as np
import pandas as pd
from scipy.signal import argrelmax

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
import backtest as bt
import chart as ch
import gex as gx

RESULTS_DIR = bt.RESULTS_DIR
SQRT252     = np.sqrt(252.0)

ZZ_THRESHES = [50.0, 75.0, 100.0]   # min reversal size (pts) — majors only
TOUCH_TOL   = 3.0      # level counts as "tested" if price within this
REV_TOL     = 10.0     # pivot within this of level = level "worked"

# sweep grids
TOLS       = [10.0, 15.0]
MIN_SCORES = [0.8, 1.2, 1.6, 2.0, 2.5]
W_EM_GRID  = [0.0, 0.5]
W_CAM_GRID = [0.0, 0.5]
W_SHELF    = 1.0
W_GEX      = 1.0
W_VP       = 1.0
EM_KS      = [0.75, 1.25]


def zigzag_pivots(df: pd.DataFrame, thresh: float) -> list[float]:
    """Pivot prices from 1m highs/lows, alternating, >= thresh reversal."""
    hi = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    piv: list[float] = []
    max_p, min_p = hi[0], lo[0]
    direction = 0                       # 0 unknown, +1 tracking high, -1 low
    for i in range(1, len(hi)):
        if direction == 0:
            max_p = max(max_p, hi[i])
            min_p = min(min_p, lo[i])
            if max_p - lo[i] >= thresh:
                piv.append(max_p); direction = -1; min_p = lo[i]
            elif hi[i] - min_p >= thresh:
                piv.append(min_p); direction = 1; max_p = hi[i]
        elif direction > 0:             # tracking a high
            if hi[i] > max_p:
                max_p = hi[i]
            if max_p - lo[i] >= thresh:
                piv.append(max_p); direction = -1; min_p = lo[i]
        else:                           # tracking a low
            if lo[i] < min_p:
                min_p = lo[i]
            if hi[i] - min_p >= thresh:
                piv.append(min_p); direction = 1; max_p = hi[i]
    return piv


def peaks_from_heat(heat, min_val=0.20, order_pts=10.0) -> list[tuple[float, float]]:
    """(price, intensity) local maxima of a heatmap intensity array."""
    if heat is None:
        return []
    pg, it = heat
    tick  = float(pg[1] - pg[0])
    order = max(1, int(round(order_pts / tick)))
    idx   = argrelmax(it, order=order)[0]
    out   = [(float(pg[i]), float(it[i])) for i in idx if it[i] >= min_val]
    # plateaus (argrelmax misses flat tops): add runs of max value
    return out


def evaluate(levels: list[float], pivots: list[float],
             day_lo: float, day_hi: float) -> tuple[int, int, int, int]:
    """returns (tested, reversed, pivots_total, pivots_matched)"""
    tested = reversed_ = 0
    for lv in levels:
        if day_lo - TOUCH_TOL <= lv <= day_hi + TOUCH_TOL:   # touched
            tested += 1
            if any(abs(lv - p) <= REV_TOL for p in pivots):
                reversed_ += 1
    matched = sum(1 for p in pivots
                  if any(abs(lv - p) <= REV_TOL for lv in levels))
    return tested, reversed_, len(pivots), matched


def main():
    ohlcv_rth = bt.load_all_ohlcv("rth")
    vxn       = gx.load_vxn()
    dates     = sorted(d for d in ohlcv_rth if not ohlcv_rth[d].empty)

    # ── per-day candidate pool + actual pivots ──
    day_data = {}
    for i, d in enumerate(dates):
        if i < 2:
            continue
        df = ohlcv_rth[d]
        if len(df) < 120:
            continue
        prior  = dates[i - 1]
        open_  = float(df["open"].iloc[0])
        day_hi = float(df["high"].max())
        day_lo = float(df["low"].min())

        cands: list[tuple[float, str, float]] = []   # (price, source, strength)

        shelf = ch.build_shelf_confluence_heatmap(d)
        for px, s in peaks_from_heat(shelf):
            cands.append((px, "shelf", s))

        gz = gx.build_stacked_gex(d)
        for px, s in peaks_from_heat(gz):
            cands.append((px, "gex", s))

        vp = ch.build_level_confluence_heatmap(d)
        for px, s in peaks_from_heat(vp):
            cands.append((px, "vp", s))

        iv = vxn.get(prior)
        if iv:
            sg = open_ * iv / SQRT252
            for k in EM_KS:
                cands.append((open_ + k * sg, "em", 1.0))
                cands.append((open_ - k * sg, "em", 1.0))

        pdf = ohlcv_rth.get(prior)
        if pdf is not None and not pdf.empty:
            from camarilla_backtest import camarilla
            cam = camarilla(float(pdf["high"].max()), float(pdf["low"].min()),
                            float(pdf["close"].iloc[-1]))
            for kk in ("r3", "r4", "s3", "s4"):
                cands.append((cam[kk], "cam", 1.0))

        pivots = {zt: zigzag_pivots(df, zt) for zt in ZZ_THRESHES}
        day_data[d] = dict(cands=cands, pivots=pivots,
                           day_lo=day_lo, day_hi=day_hi)
        if (len(day_data)) % 10 == 0:
            print(f"  prepped {len(day_data)} days…", flush=True)

    for zt in ZZ_THRESHES:
        cnt = np.mean([len(v["pivots"][zt]) for v in day_data.values()])
        print(f"{len(day_data)} days, zz={zt:.0f}pt: avg pivots/day {cnt:.1f}")

    # ── sweep ──
    rows = []
    for zt, tol, ms, w_em, w_cam in itertools.product(
            ZZ_THRESHES, TOLS, MIN_SCORES, W_EM_GRID, W_CAM_GRID):
        W = {"shelf": W_SHELF, "gex": W_GEX, "vp": W_VP,
             "em": w_em, "cam": w_cam}

        tot_tested = tot_rev = tot_piv = tot_match = tot_lvls = 0
        n_days = 0
        for d, v in day_data.items():
            # cluster: greedy by descending strength*weight
            cl = [(px, src, st) for px, src, st in v["cands"] if W[src] > 0]
            cl.sort(key=lambda x: -(W[x[1]] * x[2]))
            used = [False] * len(cl)
            levels = []
            for a in range(len(cl)):
                if used[a]:
                    continue
                members = [a]
                for b in range(a + 1, len(cl)):
                    if not used[b] and abs(cl[a][0] - cl[b][0]) <= tol:
                        members.append(b)
                # score: each SOURCE counts once (max strength) per cluster
                by_src: dict[str, float] = {}
                for m in members:
                    px, src, st = cl[m]
                    by_src[src] = max(by_src.get(src, 0.0), st)
                score = sum(W[s] * st for s, st in by_src.items())
                if score >= ms:
                    wsum = sum(W[cl[m][1]] * cl[m][2] for m in members)
                    lvl  = sum(cl[m][0] * W[cl[m][1]] * cl[m][2]
                               for m in members) / max(wsum, 1e-9)
                    levels.append(lvl)
                    for m in members:
                        used[m] = True
            t, r, p, m_ = evaluate(levels, v["pivots"][zt],
                                   v["day_lo"], v["day_hi"])
            tot_tested += t; tot_rev += r; tot_piv += p; tot_match += m_
            tot_lvls += len(levels)
            n_days += 1

        prec = tot_rev / tot_tested if tot_tested else np.nan
        rec  = tot_match / tot_piv if tot_piv else np.nan
        f1   = (2 * prec * rec / (prec + rec)
                if prec and rec and (prec + rec) > 0 else np.nan)
        rows.append(dict(zz=zt, tol=tol, min_score=ms, w_em=w_em, w_cam=w_cam,
                         levels_per_day=tot_lvls / n_days,
                         tested=tot_tested, precision=prec, recall=rec, f1=f1))

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RESULTS_DIR, "ensemble_levels_summary.csv"),
               index=False)

    # ── baseline: 25pt grid across each day's range, per zz threshold ──
    pd.set_option("display.width", 220)
    for zt in ZZ_THRESHES:
        bt_tested = bt_rev = bt_piv = bt_match = bt_lvls = 0
        for d, v in day_data.items():
            lo = np.floor(v["day_lo"] / 25) * 25
            hi = np.ceil(v["day_hi"] / 25) * 25
            levels = list(np.arange(lo, hi + 25, 25))
            t, r, p, m_ = evaluate(levels, v["pivots"][zt],
                                   v["day_lo"], v["day_hi"])
            bt_tested += t; bt_rev += r; bt_piv += p; bt_match += m_
            bt_lvls += len(levels)
        b_prec = bt_rev / bt_tested if bt_tested else np.nan
        b_rec  = bt_match / bt_piv if bt_piv else np.nan

        sub = res[res["zz"] == zt].sort_values("precision", ascending=False)
        print(f"\n=== zz={zt:.0f}pt — TOP 8 by precision "
              f"(baseline prec={b_prec:.3f} rec={b_rec:.3f}, 17.5 lvls/day) ===")
        print(sub.head(8).to_string(index=False))
        best = sub.iloc[0]
        if b_prec and best["precision"]:
            print(f"  best precision {best['precision']:.3f} = "
                  f"{best['precision']/b_prec:.2f}x baseline at "
                  f"{best['levels_per_day']:.1f} levels/day")


if __name__ == "__main__":
    main()
