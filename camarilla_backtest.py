"""
camarilla_backtest.py — backtest Camarilla pivots + VIX(VXN) expected-move
levels + IB extensions (the Pine "mu +/- k*sigma" indicator math) as daily
high/low reversal predictors on NQ tick-derived 1m data.

Level families tested:
  em_k          : cashOpen +/- k*sigma           (known at open, pre-IB)
  ib_ext        : IBH + IBrange / IBL - IBrange  (known at IB completion)
  ib_em_k_off   : Pine blend — midpoint of (IB ext, EM level) -/+ offset
  cam_r3s3/r4s4/r5s5 : Camarilla off prior-day H/L/C (rth or full session)
  cam_em_k      : midpoint of (Camarilla R4/S4, EM level)

sigma = cashOpen * (VXN_prior_close / 100) / sqrt(252)   — no lookahead.
Rounding variant: levels snapped to nearest 25pt (NQ option strikes).

Scored two ways:
  full-day RTH high/low   (comparable to VP sweep results)
  post-IB high/low        (reversal window after levels are known)
"""
import os
import sys
import itertools

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
import backtest as bt
import gex as gx

RESULTS_DIR = bt.RESULTS_DIR
IB_MINS     = 60
SQRT252     = np.sqrt(252.0)

# ── sweep grids ──
K_GRID      = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
OFF_GRID    = [0.0, 7.875, 15.75, 23.625, 31.5]
ROUND_GRID  = [False, True]          # snap to nearest 25pt
CAM_SESS    = ["rth", "full"]

HIT_WINDOWS = (10.0, 20.0)


def camarilla(h: float, l: float, c: float) -> dict:
    rng = h - l
    r3 = c + rng * 1.1 / 4.0
    r4 = c + rng * 1.1 / 2.0
    s3 = c - rng * 1.1 / 4.0
    s4 = c - rng * 1.1 / 2.0
    r5 = r4 + 1.168 * (r4 - r3)
    s5 = s4 - 1.168 * (s3 - s4)
    return {"r3": r3, "r4": r4, "r5": r5, "s3": s3, "s4": s4, "s5": s5}


def maybe_round(x: float | None, do: bool) -> float | None:
    if x is None or not do:
        return x
    return round(x / 25.0) * 25.0


def score(pred_top, pred_bot, hi, lo) -> dict:
    out = {}
    te = abs(pred_top - hi) if pred_top is not None else np.nan
    be = abs(pred_bot - lo) if pred_bot is not None else np.nan
    out["top_err"] = te
    out["bot_err"] = be
    for wdw in HIT_WINDOWS:
        out[f"top_hit{int(wdw)}"] = (te <= wdw) if not np.isnan(te) else np.nan
        out[f"bot_hit{int(wdw)}"] = (be <= wdw) if not np.isnan(be) else np.nan
    return out


def main():
    ohlcv_rth  = bt.load_all_ohlcv("rth")
    ohlcv_full = bt.load_all_ohlcv("full")
    vxn        = gx.load_vxn()            # date → IV decimal (close that day)
    dates      = sorted(d for d in ohlcv_rth if not ohlcv_rth[d].empty)

    # per-day precompute
    day_info = {}
    for i, d in enumerate(dates):
        if i == 0:
            continue
        df = ohlcv_rth[d]
        if len(df) < IB_MINS + 30:        # need IB + meaningful post-IB session
            continue
        prior = dates[i - 1]

        open_  = float(df["open"].iloc[0])
        hi     = float(df["high"].max())
        lo     = float(df["low"].min())
        ib     = df.iloc[:IB_MINS]
        post   = df.iloc[IB_MINS:]
        ib_h   = float(ib["high"].max())
        ib_l   = float(ib["low"].min())
        ib_rng = ib_h - ib_l
        post_h = float(post["high"].max())
        post_l = float(post["low"].min())

        iv = vxn.get(prior)               # prior day VXN close, no lookahead
        sigma = open_ * iv / SQRT252 if iv else None

        cams = {}
        for sess, src in (("rth", ohlcv_rth), ("full", ohlcv_full)):
            pdf = src.get(prior)
            if pdf is None or pdf.empty:
                continue
            cams[sess] = camarilla(float(pdf["high"].max()),
                                   float(pdf["low"].min()),
                                   float(pdf["close"].iloc[-1]))

        day_info[d] = dict(
            open=open_, hi=hi, lo=lo, sigma=sigma,
            ib_h=ib_h, ib_l=ib_l, ib_rng=ib_rng,
            post_h=post_h, post_l=post_l, cams=cams,
        )

    print(f"{len(day_info)} test days, VXN coverage "
          f"{sum(1 for v in day_info.values() if v['sigma'])}/{len(day_info)}")

    # ── strategy configs ──
    configs = []
    for rnd in ROUND_GRID:
        for k in K_GRID:
            configs.append(("em", dict(k=k), rnd))
        configs.append(("ib_ext", dict(), rnd))
        for k, off in itertools.product(K_GRID, OFF_GRID):
            configs.append(("ib_em", dict(k=k, off=off), rnd))
        for sess, lvl in itertools.product(CAM_SESS, ["r3s3", "r4s4", "r5s5"]):
            configs.append((f"cam_{lvl}", dict(sess=sess), rnd))
        for sess, k in itertools.product(CAM_SESS, K_GRID):
            configs.append(("cam_em", dict(sess=sess, k=k), rnd))

    rows = []
    for strat, prm, rnd in configs:
        for d, v in day_info.items():
            top = bot = None
            sg  = v["sigma"]
            if strat == "em" and sg:
                top = v["open"] + prm["k"] * sg
                bot = v["open"] - prm["k"] * sg
            elif strat == "ib_ext":
                top = v["ib_h"] + v["ib_rng"]
                bot = v["ib_l"] - v["ib_rng"]
            elif strat == "ib_em" and sg:
                em_u = v["open"] + prm["k"] * sg
                em_d = v["open"] - prm["k"] * sg
                top  = (v["ib_h"] + v["ib_rng"] + em_u) / 2 - prm["off"]
                bot  = (v["ib_l"] - v["ib_rng"] + em_d) / 2 + prm["off"]
            elif strat.startswith("cam_") and strat != "cam_em":
                cam = v["cams"].get(prm["sess"])
                if cam:
                    lvl = strat.split("_")[1]        # r3s3 etc.
                    top = cam[lvl[:2]]
                    bot = cam[lvl[2:]]
            elif strat == "cam_em" and sg:
                cam = v["cams"].get(prm["sess"])
                if cam:
                    top = (cam["r4"] + v["open"] + prm["k"] * sg) / 2
                    bot = (cam["s4"] + v["open"] - prm["k"] * sg) / 2
            if top is None and bot is None:
                continue
            top = maybe_round(top, rnd)
            bot = maybe_round(bot, rnd)

            row = dict(strategy=strat, rounded=rnd, date=d,
                       **{f"p_{k_}": v_ for k_, v_ in prm.items()},
                       pred_top=top, pred_bot=bot,
                       hi=v["hi"], lo=v["lo"])
            row.update(score(top, bot, v["hi"], v["lo"]))
            post_sc = score(top, bot, v["post_h"], v["post_l"])
            row.update({f"pib_{k_}": v_ for k_, v_ in post_sc.items()})
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "camarilla_results.csv"), index=False)

    # ── aggregate ──
    gcols = ["strategy", "rounded", "p_k", "p_off", "p_sess"]
    for c in gcols:
        if c not in df.columns:
            df[c] = np.nan
    df["_g"] = df[gcols].astype(str).agg("|".join, axis=1)

    aggs = {}
    for w in (10, 20):
        aggs[f"hit{w}"]     = ((df[f"top_hit{w}"].astype(float) +
                                df[f"bot_hit{w}"].astype(float)) / 2)
        aggs[f"pib_hit{w}"] = ((df[f"pib_top_hit{w}"].astype(float) +
                                df[f"pib_bot_hit{w}"].astype(float)) / 2)
    for k_, v_ in aggs.items():
        df[k_] = v_
    df["mae"]     = (df["top_err"] + df["bot_err"]) / 2
    df["pib_mae"] = (df["pib_top_err"] + df["pib_bot_err"]) / 2

    summ = (df.groupby(["strategy", "rounded", "p_k", "p_off", "p_sess"],
                       dropna=False)
              .agg(n=("date", "count"),
                   hit10=("hit10", "mean"), hit20=("hit20", "mean"),
                   mae=("mae", "mean"),
                   pib_hit10=("pib_hit10", "mean"),
                   pib_hit20=("pib_hit20", "mean"),
                   pib_mae=("pib_mae", "mean"))
              .reset_index()
              .sort_values("hit10", ascending=False))
    summ.to_csv(os.path.join(RESULTS_DIR, "camarilla_summary.csv"), index=False)

    pd.set_option("display.width", 200)
    print("\n=== TOP 20 by full-day hit@10 ===")
    print(summ.head(20).to_string(index=False))
    print("\n=== TOP 15 by post-IB hit@10 ===")
    print(summ.sort_values("pib_hit10", ascending=False)
              .head(15).to_string(index=False))
    print("\n=== per strategy family (mean / best full-day hit@10) ===")
    fam = (summ.groupby("strategy")
               .agg(mean_hit10=("hit10", "mean"), best_hit10=("hit10", "max"),
                    best_hit20=("hit20", "max"), best_mae=("mae", "min"),
                    best_pib_hit10=("pib_hit10", "max"))
               .sort_values("best_hit10", ascending=False))
    print(fam.to_string())


if __name__ == "__main__":
    main()
