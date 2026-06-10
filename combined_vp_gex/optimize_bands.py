"""
Band-coverage optimizer for Combined VP+GEX levels
===================================================
Goal: >=80% HOD and LOD capture with 15-20pt bands (tol = +/-7.5 .. +/-10pts).

Two phases:
  1. --build-cache   walk-forward per-day VP zones + GEX zones + daily structural
                     data for 2024+2025, pickled to opt_cache.pkl  (slow, once)
  2. --grid          in-memory grid search over stack tolerance, extra structural
                     level families, zone caps. Tunes on 2024, validates on 2025.

Honesty stats reported per config:
  - zones-in-span: count of zone prices within spot +/- 1.5*ATR20 (pre-day known)
  - analytic random-coverage expectation for same density (uniform levels)
  - lift = actual coverage - random expectation

Usage
  python combined_vp_gex/optimize_bands.py --build-cache
  python combined_vp_gex/optimize_bands.py --grid
  python combined_vp_gex/optimize_bands.py --eval-best     # detailed best-config report
"""

import argparse
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

HERE    = Path(__file__).parent
VP_DIR  = HERE.parent / "intuitiveVP"
GEX_DIR = HERE.parent / "gex_profile"
for _p in [str(HERE), str(VP_DIR), str(GEX_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

CACHE_FILE = HERE / "opt_cache.pkl"
GRID_OUT   = HERE / "band_grid_results.csv"
BIN_SIZE   = 5.0

TUNE_START, TUNE_END = "2024-01-01", "2024-12-31"
VAL_START,  VAL_END  = "2025-01-01", "2025-12-31"


# ---------------------------------------------------------------- cache build

def _ser_zones(zones) -> list:
    return [(float(z.price), float(z.score), int(z.n_tf),
             tuple(sorted(z.timeframes)), tuple(sorted(z.ftypes)))
            for z in zones]


def build_cache():
    import stacked_vp  as _vp
    import gex_profile as _gex

    df  = _gex.load_nq_data()
    vix = _gex.load_vix()
    print(f"  NQ: {len(df):,} bars  {df['date'].min().date()} -> {df['date'].max().date()}")

    df2 = df.copy()
    df2["_day"] = df2["date"].dt.date
    df2["_tm"]  = df2["date"].dt.hour * 60 + df2["date"].dt.minute
    day_stats = df2.groupby("_day").agg(
        bars=("close", "count"), hod=("high", "max"), lod=("low", "min"),
    )
    # daily OHLC + vwap on full days (used for prior-day structural levels)
    daily = df2.groupby("_day").agg(
        o=("open", "first"), h=("high", "max"), l=("low", "min"),
        c=("close", "last"), v=("volume", "sum"),
    )
    pv = (df2["close"] * df2["volume"]).groupby(df2["_day"]).sum()
    daily["vwap"] = pv / daily["v"].replace(0, np.nan)

    all_days = sorted(day_stats.index.tolist())
    s_date = pd.Timestamp(TUNE_START).date()
    e_date = pd.Timestamp(VAL_END).date()
    eligible = [d for d in all_days
                if s_date <= d <= e_date
                and day_stats.loc[d, "bars"] >= 300
                and all_days.index(d) >= 90]
    print(f"  Eligible days: {len(eligible)}  ({eligible[0]} -> {eligible[-1]})")

    cache: dict = {}
    t0 = time.time()
    for i, day in enumerate(eligible):
        df_prior  = df2[df2["_day"] < day]
        df_window = df_prior.tail(30_000).copy()
        if len(df_window) < 500:
            continue

        vp_zones, _        = _vp.run(df_window, anchored=True, bin_size=BIN_SIZE, quiet=True)
        gex_zones, _, avix = _gex.run(df_window, vix, bin_size=BIN_SIZE, quiet=True)

        spot  = float(df_window["close"].iloc[-1])
        atr20 = _gex.compute_atr20(df_window)

        # prior completed days (strictly before `day`)
        pdays = [d for d in all_days if d < day][-6:]
        pday  = pdays[-1]
        prow  = daily.loc[pday]
        # prior week (completed ISO week before this day's week)
        day_ts = pd.Timestamp(day)
        wk_id  = (day_ts.isocalendar().year, day_ts.isocalendar().week)
        pweek_days = [d for d in pdays
                      if (pd.Timestamp(d).isocalendar().year,
                          pd.Timestamp(d).isocalendar().week) != wk_id]
        if pweek_days:
            pw_h = max(daily.loc[d, "h"] for d in pweek_days)
            pw_l = min(daily.loc[d, "l"] for d in pweek_days)
            pw_c = daily.loc[pweek_days[-1], "c"]
        else:
            pw_h = pw_l = pw_c = np.nan

        vix_val = _gex.get_vix_at(vix, pd.Timestamp(day) - pd.Timedelta(days=1))

        cache[day] = dict(
            vp=_ser_zones(vp_zones),
            gex=_ser_zones(gex_zones),
            hod=float(day_stats.loc[day, "hod"]),
            lod=float(day_stats.loc[day, "lod"]),
            spot=spot, atr20=float(atr20), vix=float(vix_val),
            pd_o=float(prow["o"]), pd_h=float(prow["h"]),
            pd_l=float(prow["l"]), pd_c=float(prow["c"]),
            pd_vwap=float(prow["vwap"]),
            pw_h=float(pw_h), pw_l=float(pw_l), pw_c=float(pw_c),
        )
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  [{i+1:>3}/{len(eligible)}]  {day}  "
                  f"{el:.0f}s  ({el/(i+1):.2f}s/day)", flush=True)

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)
    print(f"  Saved {len(cache)} days -> {CACHE_FILE}")


# ---------------------------------------------------------------- extra levels

def extra_levels(d: dict, families: set) -> list:
    """Structural level candidates from prior-day/week data only (no look-ahead).
    Returns list of (price, weight, tag)."""
    out = []
    H, L, C = d["pd_h"], d["pd_l"], d["pd_c"]
    R = H - L

    if "cam" in families:                      # Camarilla pivots (day-extreme estimators)
        out += [(C + R * 1.1 / 4, 1.4, "camR3"), (C - R * 1.1 / 4, 1.4, "camS3"),
                (C + R * 1.1 / 2, 1.5, "camR4"), (C - R * 1.1 / 2, 1.5, "camS4"),
                (C + R * 1.1 / 6, 1.0, "camR2"), (C - R * 1.1 / 6, 1.0, "camS2")]
    if "piv" in families:                      # classic floor pivots
        P = (H + L + C) / 3
        out += [(P, 1.2, "pivP"),
                (2 * P - L, 1.3, "pivR1"), (2 * P - H, 1.3, "pivS1"),
                (P + R, 1.2, "pivR2"),     (P - R, 1.2, "pivS2")]
    if "pdc" in families:                      # prior close / mid / vwap
        out += [(C, 1.3, "pdc"), ((H + L) / 2, 1.2, "pdm"),
                (d["pd_vwap"], 1.3, "pdvwap")]
    if "em" in families:                       # implied expected-move bands off prior close
        sig_d = (d["vix"] / 100.0) * 1.15 / np.sqrt(252) * C
        out += [(C + sig_d, 1.3, "em+1s"), (C - sig_d, 1.3, "em-1s"),
                (C + 0.5 * sig_d, 1.1, "em+.5s"), (C - 0.5 * sig_d, 1.1, "em-.5s")]
    if "pw" in families and np.isfinite(d["pw_c"]):
        out += [(d["pw_c"], 1.1, "pwc"),
                ((d["pw_h"] + d["pw_l"]) / 2, 1.1, "pwm")]
    if "atr" in families:                      # ATR projection bands off prior close
        a = d["atr20"]
        out += [(C + 0.5 * a, 1.2, "atr+.5"), (C - 0.5 * a, 1.2, "atr-.5"),
                (C + 1.0 * a, 1.2, "atr+1"),  (C - 1.0 * a, 1.2, "atr-1")]
    if "ladder" in families:                   # fine ATR ladder off prior close
        a = d["atr20"]
        for k in [0.25, 0.4, 0.55, 0.7, 0.85, 1.0, 1.2, 1.4]:
            out += [(C + k * a, 1.0, f"lad+{k}"), (C - k * a, 1.0, f"lad-{k}")]
    if "ext" in families:                      # breakout extension ladder beyond PDH/PDL
        a = d["atr20"]
        for k in [0.08, 0.20, 0.33, 0.48, 0.65, 0.85]:
            out += [(H + k * a, 1.2, f"ext+{k}"), (L - k * a, 1.2, f"ext-{k}")]
    if "ext2" in families:                     # denser breakout shell (for 15pt bands)
        a = d["atr20"]
        for k in [0.05, 0.11, 0.17, 0.24, 0.31, 0.39, 0.48, 0.58, 0.70, 0.84, 1.0]:
            out += [(H + k * a, 1.1, f"x2+{k}"), (L - k * a, 1.1, f"x2-{k}")]
    if "ladder2" in families:                  # denser interior ladder
        a = d["atr20"]
        for k in [0.15, 0.27, 0.39, 0.51, 0.63, 0.75, 0.88, 1.02, 1.18, 1.35]:
            out += [(C + k * a, 0.9, f"l2+{k}"), (C - k * a, 0.9, f"l2-{k}")]
    return [(float(p), w, t) for p, w, t in out if np.isfinite(p)]


# ---------------------------------------------------------------- combine+stack

def combine_day(d: dict, tol_mult: float, families: set,
                vp_w: float = 0.08, gex_w: float = 0.08,
                cap: int = 0, span_atr: float = 3.0,
                mode: str = "mean", min_gap: float = 0.0) -> np.ndarray:
    """Replicates combined_analyzer expand+stack on cached zones, plus extra
    structural families. Returns array of zone prices (optionally score-capped).
    mode: "mean" = cluster center is member mean (current production behavior)
          "snap" = cluster center snaps to heaviest member's exact price"""
    feats = []   # (price, weight, tf)
    for (price, score, n_tf, tfs, _fts) in d["vp"]:
        n = max(1, len(tfs))
        w = max(0.1, score * vp_w / n)
        for tf in tfs:
            feats.append((price, w, f"vp__{tf}"))
    for (price, score, n_tf, tfs, _fts) in d["gex"]:
        n = max(1, len(tfs))
        w = max(0.1, score * gex_w / n)
        for tf in tfs:
            feats.append((price, w, f"gex__{tf}"))
    for (price, w, tag) in extra_levels(d, families):
        feats.append((price, w, f"x__{tag}"))

    if not feats:
        return np.array([])

    tolerance = tol_mult * BIN_SIZE
    arr   = np.array([f[0] for f in feats])
    wts   = np.array([f[1] for f in feats])
    tfs   = [f[2] for f in feats]
    order = np.argsort(arr)
    used  = np.zeros(len(arr), dtype=bool)

    prices, scores = [], []
    for ii in order:
        if used[ii]:
            continue
        members = [ii]
        used[ii] = True
        for jj in order:
            if used[jj]:
                continue
            if arr[jj] > arr[ii] + tolerance:
                break
            if abs(arr[jj] - arr[ii]) <= tolerance:
                members.append(jj)
                used[jj] = True
        if mode == "snap":
            center = float(arr[members[int(np.argmax(wts[members]))]])
        else:
            center = float(np.mean(arr[members]))
        n_tf   = len({tfs[m] for m in members})
        score  = (n_tf ** 2) * float(wts[members].sum())
        prices.append(center)
        scores.append(score)

    prices = np.array(prices)
    scores = np.array(scores)

    if min_gap > 0:
        # greedy score-ordered selection with minimum spacing — removes
        # redundant near-duplicate levels, keeps strongest in each gap-window
        keep = []
        for ii in np.argsort(scores)[::-1]:
            if all(abs(prices[ii] - prices[kk]) >= min_gap for kk in keep):
                keep.append(ii)
        sel = np.zeros(len(prices), dtype=bool)
        sel[keep] = True
        prices, scores = prices[sel], scores[sel]

    if cap > 0:
        lo = d["spot"] - span_atr * d["atr20"]
        hi = d["spot"] + span_atr * d["atr20"]
        in_span = (prices >= lo) & (prices <= hi)
        idx_in  = np.where(in_span)[0]
        if len(idx_in) > cap:
            keep = idx_in[np.argsort(scores[idx_in])[::-1][:cap]]
            sel  = np.zeros(len(prices), dtype=bool)
            sel[keep] = True
            sel |= ~in_span          # zones outside span kept (far HOD/LOD days)
            prices = prices[sel]
    return prices


# ---------------------------------------------------------------- evaluation

def eval_config(cache: dict, days: list, tol_mult: float, families: set,
                cap: int, band_tols=(7.5, 10.0), mode: str = "mean") -> dict:
    hits = {t: [0, 0] for t in band_tols}    # tol -> [hod_hits, lod_hits]
    n = 0
    nz_span, rand_exp = [], {t: [] for t in band_tols}
    for day in days:
        d = cache[day]
        zp = combine_day(d, tol_mult, families, cap=cap, mode=mode)
        if len(zp) == 0:
            continue
        n += 1
        for t in band_tols:
            if np.min(np.abs(zp - d["hod"])) <= t: hits[t][0] += 1
            if np.min(np.abs(zp - d["lod"])) <= t: hits[t][1] += 1
        lo = d["spot"] - 1.5 * d["atr20"]
        hi = d["spot"] + 1.5 * d["atr20"]
        k  = int(((zp >= lo) & (zp <= hi)).sum())
        nz_span.append(k)
        span = hi - lo
        for t in band_tols:
            rand_exp[t].append(min(1.0, k * 2 * t / span))
    if n == 0:
        return {}
    out = dict(n=n, nz=float(np.mean(nz_span)))
    for t in band_tols:
        out[f"hod{t:g}"] = hits[t][0] / n
        out[f"lod{t:g}"] = hits[t][1] / n
        out[f"rand{t:g}"] = float(np.mean(rand_exp[t]))
    return out


def run_grid():
    with open(CACHE_FILE, "rb") as f:
        cache = pickle.load(f)
    days = sorted(cache.keys())
    tune = [d for d in days if d <= pd.Timestamp(TUNE_END).date()]
    val  = [d for d in days if d >= pd.Timestamp(VAL_START).date()]
    print(f"  Cache: {len(days)} days  tune={len(tune)}  val={len(val)}")

    fam_sets = [
        set(),
        {"cam", "piv", "pdc"},
        {"cam", "piv", "pdc", "em"},
        {"cam", "piv", "pdc", "em", "pw"},
        {"cam", "piv", "pdc", "em", "pw", "atr"},
        {"cam", "piv", "pdc", "em", "pw", "ext"},
        {"cam", "piv", "pdc", "em", "pw", "atr", "ext"},
        {"cam", "piv", "pdc", "em", "pw", "ladder", "ext"},
        {"cam", "piv", "pdc", "em", "pw", "atr", "ladder", "ext"},
    ]
    tol_mults = [0.75, 1.0, 1.5, 2.0]
    caps      = [0, 30, 45]
    modes     = ["mean", "snap"]

    rows = []
    for tm in tol_mults:
        for fams in fam_sets:
            for cap in caps:
                for mode in modes:
                    r = eval_config(cache, tune, tm, fams, cap, mode=mode)
                    if not r:
                        continue
                    rows.append(dict(
                        tol_mult=tm, families="+".join(sorted(fams)) or "base",
                        cap=cap, mode=mode,
                        **{k: round(v, 4) for k, v in r.items()},
                    ))
    res = pd.DataFrame(rows)
    res["min10"]  = res[["hod10", "lod10"]].min(axis=1)
    res["min7.5"] = res[["hod7.5", "lod7.5"]].min(axis=1)
    res["lift10"] = (res["hod10"] + res["lod10"]) / 2 - res["rand10"]
    res = res.sort_values(["min10", "lift10"], ascending=False)
    res.to_csv(GRID_OUT, index=False)

    print(f"\n  TOP 20 CONFIGS  (tuned on 2024, sorted by min(HOD,LOD)@+/-10)")
    cols = ["tol_mult", "families", "cap", "mode", "hod10", "lod10", "min10",
            "hod7.5", "lod7.5", "nz", "rand10", "lift10"]
    print(res[cols].head(20).to_string(index=False))

    # validate top 5 distinct configs on 2025
    print(f"\n  VALIDATION (2025, untouched)")
    seen = set()
    for _, r in res.iterrows():
        key = (r["tol_mult"], r["families"], r["cap"], r["mode"])
        if key in seen:
            continue
        seen.add(key)
        fams = set() if r["families"] == "base" else set(r["families"].split("+"))
        v = eval_config(cache, val, r["tol_mult"], fams, int(r["cap"]), mode=r["mode"])
        print(f"  tol={r['tol_mult']:.2f} fam={r['families']:<28} cap={int(r['cap']):>2} "
              f"{r['mode']:<4}  HOD10={v['hod10']:.1%} LOD10={v['lod10']:.1%}  "
              f"HOD7.5={v['hod7.5']:.1%} LOD7.5={v['lod7.5']:.1%}  "
              f"nz={v['nz']:.1f} rand10={v['rand10']:.1%}")
        if len(seen) >= 5:
            break


def diagnose(tol_mult=1.0, families=None, cap=0, mode="snap", tol=10.0):
    """Where do misses land? Position vs spot (ATR units), vs prior window range."""
    families = families if families is not None else {"cam", "piv", "pdc", "em", "pw", "atr"}
    with open(CACHE_FILE, "rb") as f:
        cache = pickle.load(f)
    days = sorted(cache.keys())
    miss_h, miss_l = [], []
    for day in days:
        d = cache[day]
        zp = combine_day(d, tol_mult, families, cap=cap, mode=mode)
        if len(zp) == 0:
            continue
        a = d["atr20"]
        for tgt, store in [(d["hod"], miss_h), (d["lod"], miss_l)]:
            dist = float(np.min(np.abs(zp - tgt)))
            if dist > tol:
                store.append(dict(
                    day=day, dist=dist,
                    pos_atr=(tgt - d["spot"]) / a,
                    beyond=(tgt > d["pd_h"] + 0.0) if tgt == d["hod"] else (tgt < d["pd_l"]),
                    ext_atr=(tgt - d["pd_h"]) / a if tgt == d["hod"] else (d["pd_l"] - tgt) / a,
                ))
    for label, ms in [("HOD", miss_h), ("LOD", miss_l)]:
        if not ms:
            continue
        dist = np.array([m["dist"] for m in ms])
        pos  = np.array([m["pos_atr"] for m in ms])
        ext  = np.array([m["ext_atr"] for m in ms])
        bey  = np.array([m["beyond"] for m in ms])
        print(f"\n  {label} misses: {len(ms)}/{len(days)}")
        print(f"    miss dist pts : p25={np.percentile(dist,25):.0f}  med={np.median(dist):.0f}  "
              f"p75={np.percentile(dist,75):.0f}  max={dist.max():.0f}")
        print(f"    pos vs spot   : med={np.median(pos):+.2f} ATR  "
              f"p10={np.percentile(pos,10):+.2f}  p90={np.percentile(pos,90):+.2f}")
        print(f"    beyond pd ext : {bey.mean():.0%}   ext med={np.median(ext):+.2f} ATR")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-cache", action="store_true")
    ap.add_argument("--grid",        action="store_true")
    ap.add_argument("--diagnose",    action="store_true")
    args = ap.parse_args()
    if args.build_cache:
        build_cache()
    elif args.grid:
        run_grid()
    elif args.diagnose:
        diagnose()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
