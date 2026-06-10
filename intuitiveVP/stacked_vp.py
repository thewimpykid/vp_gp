"""
NQ Stacked Volume Profile Analyzer
===================================
Builds 4 VP windows (weekly / 30d / 60d / 90d) and stacks their structural
features (VAH, VAL, POC, HVN shelves, HVN ledges, LVNs) to identify
high-probability reversal zones.

Usage
-----
  python stacked_vp.py                  # rolling mode (default)
  python stacked_vp.py --anchored       # calendar-anchored profiles
  python stacked_vp.py --compare        # run both and compare
  python stacked_vp.py --bins 2.5       # bin size in NQ points (default 5)
  python stacked_vp.py --display 1000   # recent bars in chart (default 600)
  python stacked_vp.py --top 25         # print top N zones (default 20)

Shelf  = HVN boundary with gradual volume taper  (gentle gradient)
Ledge  = HVN boundary with abrupt volume cliff   (steep gradient)
"""

import argparse
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

# Ensure UTF-8 output on Windows (box-drawing chars, arrows in comments/labels)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths, savgol_filter

# Magenta confidence colormap: light (low) → dark (high)
MAGENTA_CMAP = LinearSegmentedColormap.from_list(
    "magenta_conf", ["#FF99FF", "#880088"]
)

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
# Sierra Chart CSV lives one level above this script's directory
CSV_PATH   = Path(__file__).parent.parent / "1Min_NQ.csv"
CACHE_PATH = Path(__file__).parent / "hdata" / "NQ_1m_sierra_cache.parquet"

# ─── Parameters ───────────────────────────────────────────────────────────────
BIN_SIZE         = 5.0    # NQ points per bin (tick = 0.25; 5 pts = ~20 ticks)
VALUE_AREA_PCT   = 0.70   # fraction of total volume that defines the VA
HVN_PROMINENCE   = 0.35   # HVN peak must exceed this × max_volume in prominence (optimized)
LVN_DEPTH        = 0.40   # LVN valley must be < this × mean volume (optimized)
SHELF_SLOPE_MAX  = 0.08   # shelf detection threshold (retained for reference)
LEDGE_SLOPE_MIN  = 0.25   # ledge detection threshold (retained for reference)
SLOPE_BINS       = 5      # bins used to measure boundary gradient
STACK_TOLERANCE  = 3.0    # confluence radius = this × BIN_SIZE (optimized: wider clusters)
DISPLAY_BARS     = 600    # recent 1-min bars shown in price chart
# shelf/ledge features hurt aggregate performance (solo rates 6.5% vs 11.9% baseline)
INCLUDE_SHELF    = False
INCLUDE_LEDGE    = False

LOOKBACKS = {             # approximate trading days per window (weekly dropped — adds noise)
    "30d":    21,
    "60d":    42,
    "90d":    63,
}
MINS_PER_DAY = 390        # ~6.5 h × 60


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class VolumeProfile:
    name:  str
    bins:  np.ndarray   # left edge of each price bin
    vols:  np.ndarray   # volume assigned to each bin
    poc:   float        # point of control (highest-volume bin)
    vah:   float        # value area high
    val:   float        # value area low


@dataclass
class VPFeature:
    price:     float
    ftype:     str      # vah | val | poc | shelf | ledge | lvn
    timeframe: str
    weight:    float = 1.0


@dataclass
class Zone:
    price:     float
    score:     float    # confluence strength  (n_tf² × Σ weights)
    n_tf:      int      # unique timeframes contributing
    timeframes: set
    ftypes:    set
    features:  list


# ─── Data Loading ─────────────────────────────────────────────────────────────

def _load_sierra_csv(path: Path) -> pd.DataFrame:
    """
    Parse Sierra Chart 1m CSV.
    Format: Date;Symbol;Open;High;Low;Close;Volume
    Numbers use EU notation: thousands='.', decimal=','  (e.g. 3.575,75 → 3575.75)
    Dates are US format: M/D/YYYY H:MM AM/PM
    """
    df = pd.read_csv(path, sep=";", thousands=".", decimal=",")
    df = df.rename(columns={
        "Date":   "date",
        "Open":   "open",
        "High":   "high",
        "Low":    "low",
        "Close":  "close",
        "Volume": "volume",
    })
    df["date"] = pd.to_datetime(df["date"], dayfirst=False)
    return df[["date", "open", "high", "low", "close", "volume"]].copy()


def load_data() -> pd.DataFrame:
    if CACHE_PATH.exists():
        print(f"  cache  → {CACHE_PATH}")
        df = pd.read_parquet(CACHE_PATH)
        if "date" not in df.columns and "ts" in df.columns:
            df = df.rename(columns={"ts": "date"})
        elif "date" not in df.columns:
            df = df.reset_index().rename(columns={df.index.name or "index": "date"})
    else:
        print(f"  csv    → {CSV_PATH}  (first load — caching to parquet)")
        df = _load_sierra_csv(CSV_PATH)
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(CACHE_PATH)

    df = df[["date", "open", "high", "low", "close", "volume"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= "2020-01-01") & (df["date"] <= "2026-12-31")]
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ─── Volume Profile Construction ──────────────────────────────────────────────

def build_profile(bars: pd.DataFrame, name: str, bin_size: float) -> VolumeProfile:
    """
    Distribute each bar's volume uniformly across price bins spanning [low, high].
    This is the standard OHLCV approximation for volume-at-price.
    """
    lo_arr  = bars["low"].values
    hi_arr  = bars["high"].values
    vol_arr = bars["volume"].values.astype(np.float64)

    price_min = np.floor(lo_arr.min() / bin_size) * bin_size
    price_max = np.ceil(hi_arr.max()  / bin_size) * bin_size
    n_bins    = int(round((price_max - price_min) / bin_size)) + 1
    profile   = np.zeros(n_bins, dtype=np.float64)

    lo_idx = np.floor((lo_arr - price_min) / bin_size).astype(int).clip(0, n_bins - 1)
    hi_idx = np.floor((hi_arr - price_min) / bin_size).astype(int).clip(0, n_bins - 1)

    for i in range(len(bars)):
        span = hi_idx[i] - lo_idx[i] + 1
        profile[lo_idx[i]: hi_idx[i] + 1] += vol_arr[i] / span

    bins = price_min + np.arange(n_bins) * bin_size

    # Point of Control
    poc = bins[np.argmax(profile)]

    # Value Area: greedily add highest-volume bins until 70 % threshold
    total     = profile.sum()
    order     = np.argsort(profile)[::-1]
    cumsum    = 0.0
    va_idx    = set()
    for idx in order:
        cumsum += profile[idx]
        va_idx.add(int(idx))
        if cumsum >= total * VALUE_AREA_PCT:
            break
    va_sorted = sorted(va_idx)
    vah = bins[va_sorted[-1]] + bin_size  # top of the highest VA bin
    val = bins[va_sorted[0]]

    return VolumeProfile(name=name, bins=bins, vols=profile, poc=poc, vah=vah, val=val)


# ─── Feature Detection ────────────────────────────────────────────────────────

def detect_features(vp: VolumeProfile,
                    shelf_max: float = None,
                    ledge_min: float = None,
                    hvn_prominence: float = None,
                    lvn_depth: float = None) -> list:
    shelf_max      = SHELF_SLOPE_MAX  if shelf_max      is None else shelf_max
    ledge_min      = LEDGE_SLOPE_MIN  if ledge_min      is None else ledge_min
    hvn_prominence = HVN_PROMINENCE   if hvn_prominence is None else hvn_prominence
    lvn_depth      = LVN_DEPTH        if lvn_depth      is None else lvn_depth

    feats: list[VPFeature] = []
    bins, vols = vp.bins, vp.vols
    tf = vp.name

    feats += [
        VPFeature(vp.vah, "vah", tf, weight=1.2),
        VPFeature(vp.val, "val", tf, weight=1.2),
        VPFeature(vp.poc, "poc", tf, weight=1.1),
    ]

    # Smooth the raw profile before structural analysis
    n   = len(vols)
    win = max(5, n // 15)
    if win % 2 == 0:
        win += 1
    win = min(win, n - 1 if n % 2 == 0 else n)
    try:
        smooth = savgol_filter(vols, window_length=win, polyorder=2).clip(0)
    except Exception:
        smooth = vols.copy()

    peak_vol = smooth.max()
    mean_vol = smooth.mean()
    if peak_vol < 1e-9:
        return feats

    # ── HVN: peaks with significant prominence ────────────────────────────────
    peaks, _ = find_peaks(smooth, prominence=hvn_prominence * peak_vol)

    for p in peaks:
        try:
            _, _, lo_ips, hi_ips = peak_widths(smooth, [p], rel_height=0.5)
        except Exception:
            continue
        l_idx = int(np.clip(lo_ips[0], 0, n - 1))
        r_idx = int(np.clip(hi_ips[0], 0, n - 1))

        _classify_boundary(smooth, bins, l_idx, "lower", peak_vol, tf, feats,
                            shelf_max, ledge_min)
        _classify_boundary(smooth, bins, r_idx, "upper", peak_vol, tf, feats,
                            shelf_max, ledge_min)

    # ── LVN: valleys significantly below mean ─────────────────────────────────
    valleys, _ = find_peaks(-smooth, prominence=0.08 * peak_vol)
    for v in valleys:
        if smooth[v] < lvn_depth * mean_vol:
            feats.append(VPFeature(bins[v], "lvn", tf, weight=0.9))

    return feats


def _classify_boundary(smooth, bins, edge_idx, side, peak_vol, tf, feats,
                        shelf_max, ledge_min):
    """
    Measure the average normalised volume slope moving outward from an HVN edge.
    slope ≤ shelf_max  →  Shelf (gradual taper)
    slope ≥ ledge_min  →  Ledge (abrupt cliff)
    """
    n = len(smooth)
    if side == "lower":
        indices = list(range(edge_idx, max(-1, edge_idx - SLOPE_BINS - 1), -1))
    else:
        indices = list(range(edge_idx, min(n, edge_idx + SLOPE_BINS + 1)))

    seg = smooth[indices]
    if len(seg) < 2:
        return

    drops     = np.diff(seg)
    avg_slope = np.mean(np.abs(drops)) / peak_vol

    price = bins[edge_idx]
    if avg_slope <= shelf_max and INCLUDE_SHELF:
        feats.append(VPFeature(price, "shelf", tf, weight=0.8))
    elif avg_slope >= ledge_min and INCLUDE_LEDGE:
        feats.append(VPFeature(price, "ledge", tf, weight=1.3))


# ─── Confluence Stacking ──────────────────────────────────────────────────────

def stack_features(all_features: list, bin_size: float,
                   tol_mult: float = None, score_exp: float = 2.0) -> list:
    """
    Greedy clustering: group features within tol_mult × bin_size of each other.
    Score = n_unique_timeframes^score_exp × Σ feature_weights.
    score_exp=2 (default) strongly rewards multi-TF alignment.
    """
    if not all_features:
        return []

    tol_mult  = STACK_TOLERANCE if tol_mult is None else tol_mult
    tolerance = tol_mult * bin_size
    prices    = np.array([f.price for f in all_features])
    sort_idx  = np.argsort(prices)
    used      = np.zeros(len(all_features), dtype=bool)
    zones: list[Zone] = []

    for i in sort_idx:
        if used[i]:
            continue
        cluster = [all_features[i]]
        used[i] = True
        for j in sort_idx:
            if used[j]:
                continue
            if prices[j] > prices[i] + tolerance:
                break
            if abs(prices[j] - prices[i]) <= tolerance:
                cluster.append(all_features[j])
                used[j] = True

        center = float(np.mean([f.price for f in cluster]))
        tfs    = {f.timeframe for f in cluster}
        ftypes = {f.ftype    for f in cluster}
        n_tf   = len(tfs)
        score  = (n_tf ** score_exp) * sum(f.weight for f in cluster)

        zones.append(Zone(
            price=center, score=score,
            n_tf=n_tf, timeframes=tfs, ftypes=ftypes, features=cluster,
        ))

    return sorted(zones, key=lambda z: z.score, reverse=True)


# ─── Lookback Cutoff ──────────────────────────────────────────────────────────

def get_cutoff(df: pd.DataFrame, name: str, anchored: bool) -> pd.Timestamp:
    last = df["date"].max()
    td   = LOOKBACKS[name]
    if not anchored:
        idx = max(0, len(df) - td * MINS_PER_DAY)
        return df.iloc[idx]["date"]
    # Anchored: calendar period boundaries
    if name == "weekly":
        return last - pd.Timedelta(days=last.weekday())
    months_back = {"30d": 1, "60d": 2, "90d": 3}[name]
    m, y = last.month - months_back, last.year
    while m <= 0:
        m += 12; y -= 1
    return pd.Timestamp(y, m, 1)


# ─── Analysis Driver ──────────────────────────────────────────────────────────

def run(df: pd.DataFrame, anchored: bool, bin_size: float, quiet: bool = False):
    all_features: list[VPFeature] = []
    profiles: dict[str, VolumeProfile] = {}

    for name in LOOKBACKS:
        cutoff = get_cutoff(df, name, anchored)
        subset = df[df["date"] >= cutoff]
        if len(subset) < 20:
            if not quiet:
                print(f"  [{name}] insufficient data, skipping")
            continue

        vp = build_profile(subset, name, bin_size)
        profiles[name] = vp
        feats = detect_features(vp)
        all_features.extend(feats)

        if not quiet:
            print(f"  [{name}]  bars={len(subset):>7,}  "
                  f"VAH={vp.vah:>9.2f}  VAL={vp.val:>9.2f}  "
                  f"feats={len(feats)}")

    all_features.extend(prior_day_features(df))
    all_features.extend(session_level_features(df))
    atr20 = compute_atr20(df)
    all_features.extend(round_number_features(df, atr20))
    zones = stack_features(all_features, bin_size)
    return zones, profiles


# ─── Console Report ───────────────────────────────────────────────────────────

def print_report(zones: list, show_n: int = 20):
    sep = "-" * 74
    print(f"\n{sep}")
    print(f"  TOP {min(show_n, len(zones))} CONFLUENCE ZONES")
    print(sep)
    print(f"  {'PRICE':>9}  {'SCORE':>6}  {'TF':>2}  TIMEFRAMES            FEATURE TYPES")
    print(sep)
    for z in zones[:show_n]:
        tfs = ",".join(sorted(z.timeframes))
        fts = "|".join(sorted(z.ftypes))
        print(f"  {z.price:>9.2f}  {z.score:>6.1f}  {z.n_tf:>2}  {tfs:<20}  {fts}")
    print(sep)
    print(f"  3+ timeframe zones : {sum(1 for z in zones if z.n_tf >= 3)}")
    print(f"  4  timeframe zones : {sum(1 for z in zones if z.n_tf == 4)}")


# ─── Daily Levels ─────────────────────────────────────────────────────────────

def compute_atr20(df: pd.DataFrame) -> float:
    """20-day ATR from daily OHLC aggregated from 1m bars."""
    df2 = df.copy()
    df2["_d"] = df2["date"].dt.date
    day_bars = (
        df2.groupby("_d")
        .agg(hi=("high", "max"), lo=("low", "min"), cl=("close", "last"))
        .reset_index()
        .tail(25)
        .reset_index(drop=True)
    )
    if len(day_bars) < 2:
        return 300.0
    hi = day_bars["hi"].values
    lo = day_bars["lo"].values
    cl = day_bars["cl"].values
    prev_cl = np.concatenate([[cl[0]], cl[:-1]])
    tr = np.maximum(hi - lo, np.maximum(np.abs(hi - prev_cl), np.abs(lo - prev_cl)))
    return float(tr[-20:].mean())


def prior_day_features(df: pd.DataFrame, n_prior: int = 10) -> list:
    """
    PDH/PDL for prior n_prior sessions (timeframe = "d-1", "d-2", ... so that
    multiple sessions stacking at same price get multi-TF score boost).
    PWH/PWL for prior completed week.
    """
    df2 = df.copy()
    df2["_d"] = df2["date"].dt.date
    daily = (
        df2.groupby("_d")
        .agg(hi=("high", "max"), lo=("low", "min"))
        .reset_index()
    )
    daily = daily.iloc[:-1].tail(n_prior)  # exclude today
    feats: list[VPFeature] = []
    for i, (_, row) in enumerate(daily.iloc[::-1].iterrows()):  # most recent = i=0
        tf = f"d-{i+1}"
        feats.append(VPFeature(float(row.hi), "pdh", tf, weight=1.4))
        feats.append(VPFeature(float(row.lo), "pdl", tf, weight=1.4))

    # prior completed week high / low
    df2["_w"] = df2["date"].dt.isocalendar().week
    df2["_y"] = df2["date"].dt.isocalendar().year
    weekly = (
        df2.groupby(["_y", "_w"])
        .agg(hi=("high", "max"), lo=("low", "min"))
        .reset_index()
    )
    if len(weekly) >= 2:
        pw = weekly.iloc[-2]
        feats.append(VPFeature(float(pw.hi), "pwh", "prior_week", weight=1.2))
        feats.append(VPFeature(float(pw.lo), "pwl", "prior_week", weight=1.2))
    return feats


def session_level_features(df: pd.DataFrame, n_prior: int = 3) -> list:
    """
    Session-derived structural features for the last n_prior completed days.
    All times assumed Eastern Time (matches Sierra Chart NQ export):
      ibh/ibl   Initial Balance H/L (09:30-10:30 ET)  weight 1.6 — strongest intraday level
      onh/onl   Overnight pre-RTH H/L (00:00-09:29 ET) weight 1.2
      gap       Prior RTH close = gap reference level   weight 1.1
      pmh/pml   Prior month H/L                         weight 1.3
    """
    feats: list[VPFeature] = []
    df2 = df.copy()
    df2["_d"]  = df2["date"].dt.date
    df2["_tm"] = df2["date"].dt.hour * 60 + df2["date"].dt.minute

    all_days = sorted(df2["_d"].unique())
    if len(all_days) < 2:
        return feats

    prior_days = all_days[:-1][-n_prior:]

    for i, day in enumerate(reversed(prior_days)):   # i=0 → most recent (d-1)
        tf       = f"d-{i+1}"
        day_bars = df2[df2["_d"] == day]

        # IB: 09:30–10:30 ET = minutes 570–630
        ib = day_bars[(day_bars["_tm"] >= 570) & (day_bars["_tm"] <= 630)]
        if len(ib) >= 5:
            feats.append(VPFeature(float(ib["high"].max()), "ibh", tf, weight=1.6))
            feats.append(VPFeature(float(ib["low"].min()),  "ibl", tf, weight=1.6))

        # Overnight pre-RTH: 00:00–09:29 ET
        overnight = day_bars[day_bars["_tm"] < 570]
        if len(overnight) >= 5:
            feats.append(VPFeature(float(overnight["high"].max()), "onh", tf, weight=1.2))
            feats.append(VPFeature(float(overnight["low"].min()),  "onl", tf, weight=1.2))

        # RTH close = gap reference for next session
        rth = day_bars[(day_bars["_tm"] >= 570) & (day_bars["_tm"] <= 975)]
        if len(rth) >= 5:
            feats.append(VPFeature(float(rth["close"].iloc[-1]), "gap", tf, weight=1.1))

    # Prior month H/L
    df2["_ym"] = df2["date"].dt.to_period("M")
    months = sorted(df2["_ym"].unique())
    if len(months) >= 2:
        pm_bars = df2[df2["_ym"] == months[-2]]
        feats.append(VPFeature(float(pm_bars["high"].max()), "pmh", "prior_month", weight=1.3))
        feats.append(VPFeature(float(pm_bars["low"].min()),  "pml", "prior_month", weight=1.3))

    return feats


def round_number_features(df: pd.DataFrame, atr20: float) -> list:
    """100pt and 500pt psychological levels within ±4 ATR of current price."""
    last_close = float(df["close"].iloc[-1])
    lo = last_close - 4 * atr20
    hi = last_close + 4 * atr20
    feats: list[VPFeature] = []
    for p in np.arange(np.ceil(lo / 500) * 500, hi + 1, 500):
        feats.append(VPFeature(float(p), "round", "psychological", weight=1.5))
    for p in np.arange(np.ceil(lo / 100) * 100, hi + 1, 100):
        if p % 500 != 0:
            feats.append(VPFeature(float(p), "round", "psychological", weight=1.0))
    return feats


def hod_lod_coverage(df: pd.DataFrame, n_days: int = 60,
                     tolerance_pct: float = 0.0012,
                     bin_size: float = BIN_SIZE,
                     seed: int = 42) -> dict:
    """
    Walk-forward HOD/LOD coverage test (zero look-ahead).
    For each sampled day: compute zones using only prior data, then check
    whether any zone is within tolerance_pct*price of that day's actual HOD/LOD.
    Tolerance 0.12% ≈ 25pt at 21k — a zone that close to HOD/LOD is a real hit.
    """
    rng = np.random.default_rng(seed)
    df2 = df.copy()
    df2["_day"] = df2["date"].dt.date

    day_stats = df2.groupby("_day").agg(
        bars=("close", "count"),
        hod=("high",  "max"),
        lod=("low",   "min"),
    )
    all_days_list = sorted(day_stats.index.tolist())
    eligible = [d for d in all_days_list
                if day_stats.loc[d, "bars"] >= 300
                and all_days_list.index(d) >= 90]
    if not eligible:
        print("  No eligible days"); return {}

    sample = sorted(rng.choice(eligible,
                               size=min(n_days, len(eligible)),
                               replace=False).tolist())

    hod_hits = lod_hits = both_hits = total = 0
    rows = []
    sep = "-" * 68
    print(f"\n  HOD/LOD coverage  ({len(sample)} days, tol={tolerance_pct*100:.2f}% of price)")
    print(sep)
    print(f"  {'DATE':<12}  {'HOD':>8}  {'LOD':>8}  {'dHOD':>6}  {'dLOD':>6}  HOD  LOD")
    print(sep)

    for day in sample:
        df_prior  = df2[df2["_day"] < day]
        df_window = df_prior.tail(30_000).copy()
        if len(df_window) < 2000:
            continue

        zones, _ = run(df_window, anchored=True, bin_size=bin_size, quiet=True)
        if not zones:
            continue

        zp   = np.array([z.price for z in zones])
        hod  = float(day_stats.loc[day, "hod"])
        lod  = float(day_stats.loc[day, "lod"])
        tol  = ((hod + lod) / 2) * tolerance_pct

        hod_dist = float(np.min(np.abs(zp - hod)))
        lod_dist = float(np.min(np.abs(zp - lod)))
        hod_hit  = hod_dist <= tol
        lod_hit  = lod_dist <= tol

        if hod_hit: hod_hits += 1
        if lod_hit: lod_hits += 1
        if hod_hit and lod_hit: both_hits += 1
        total += 1

        h_flag = "✓" if hod_hit else " "
        l_flag = "✓" if lod_hit else " "
        print(f"  {str(day):<12}  {hod:>8.1f}  {lod:>8.1f}  "
              f"{hod_dist:>6.1f}  {lod_dist:>6.1f}   {h_flag}    {l_flag}")
        rows.append(dict(day=str(day), hod=hod, lod=lod,
                         hod_dist=round(hod_dist, 1), lod_dist=round(lod_dist, 1),
                         hod_hit=hod_hit, lod_hit=lod_hit))

    if total == 0:
        print("  No days processed"); return {}

    print(sep)
    print(f"  HOD captured : {hod_hits/total:>5.1%}  ({hod_hits}/{total})")
    print(f"  LOD captured : {lod_hits/total:>5.1%}  ({lod_hits}/{total})")
    print(f"  Both captured: {both_hits/total:>5.1%}  ({both_hits}/{total})")
    avg_hd = float(np.mean([r["hod_dist"] for r in rows]))
    avg_ld = float(np.mean([r["lod_dist"] for r in rows]))
    print(f"  Avg dist HOD : {avg_hd:.1f} pts   Avg dist LOD: {avg_ld:.1f} pts")
    print(sep)

    pd.DataFrame(rows).to_csv("hod_lod_coverage.csv", index=False)
    print("  Saved → hod_lod_coverage.csv")
    return dict(hod_rate=hod_hits/total, lod_rate=lod_hits/total,
                both_rate=both_hits/total, avg_hod_dist=avg_hd,
                avg_lod_dist=avg_ld, n_days=total)


def daily_levels(df: pd.DataFrame, zones: list, n: int = 5) -> tuple:
    """
    Return n zones (4–6) balanced above and below current close.

    Selection rules:
      - Require n_tf >= 2 (drop single-timeframe noise zones)
      - Expand ATR multiplier until >= n qualifying zones found
      - Split ceil(n/2) above price, floor(n/2) below — fill deficit from other side
      - Final list sorted by price ascending
    Returns (levels, last_close, atr20).
    """
    atr20      = compute_atr20(df)
    last_close = float(df["close"].iloc[-1])
    qualified  = [z for z in zones if z.n_tf >= 2]

    in_range: list = []
    for mult in np.arange(2.0, 5.1, 0.5):
        lo_b = last_close - mult * atr20
        hi_b = last_close + mult * atr20
        in_range = [z for z in qualified if lo_b <= z.price <= hi_b]
        if len(in_range) >= n:
            break

    # zones already score-sorted desc
    above = [z for z in in_range if z.price > last_close]
    below = [z for z in in_range if z.price <= last_close]

    n_above = math.ceil(n / 2)
    n_below = n - n_above
    sel_above = above[:n_above]
    sel_below = below[:n_below]

    # fill deficit from the other side
    if len(sel_above) < n_above:
        sel_below = below[:n - len(sel_above)]
    elif len(sel_below) < n_below:
        sel_above = above[:n - len(sel_below)]

    return sorted(sel_above + sel_below, key=lambda z: z.price), last_close, atr20


def print_daily_levels(levels: list, last_close: float, atr20: float):
    req_move = atr20 * 0.40          # ~0.4 ATR = meaningful swing, not noise
    sep = "=" * 76
    print(f"\n{sep}")
    print(f"  TODAY'S HIGH-PROBABILITY REVERSAL LEVELS  ({len(levels)} zones)")
    print(f"  Last close: {last_close:.2f}   20-day ATR: {atr20:.1f} pts   "
          f"Req swing: >{req_move:.0f} pts")
    print(sep)
    print(f"  {'#':>2}  {'PRICE':>9}  {'ROLE':>6}  {'SCORE':>6}  {'TF':>2}  "
          f"{'DIST':>8}  FEATURE TYPES")
    print(sep)
    for i, z in enumerate(levels, 1):
        dist = z.price - last_close
        role = "RESIST" if dist > 0 else "SUPPRT"
        side = "▲" if dist > 0 else "▼"
        fts  = "|".join(sorted(z.ftypes))
        print(f"  {i:>2}. {z.price:>9.2f}  {role:>6}  {z.score:>6.1f}  {z.n_tf:>2}  "
              f"  {side}{abs(dist):>5.1f} pts  {fts}")
    print(sep)


# ─── Visualization ────────────────────────────────────────────────────────────

TF_COLORS = {
    "weekly": "#FF6B6B",
    "30d":    "#FFA042",
    "60d":    "#FFD966",
    "90d":    "#4EC9B0",
}


def _draw_ohlc(ax, window: pd.DataFrame):
    """Render OHLC candlestick bars using LineCollections (fast, no extra deps)."""
    n = len(window)
    x = np.arange(n)
    o = window["open"].values
    h = window["high"].values
    l = window["low"].values
    c = window["close"].values
    up = c >= o

    # Wicks: one thin line per bar, low → high
    wick_segs = [[(i, l[i]), (i, h[i])] for i in range(n)]
    wick_cols = ["#3fb950" if up[i] else "#f85149" for i in range(n)]
    ax.add_collection(LineCollection(wick_segs, colors=wick_cols,
                                     linewidths=0.55, zorder=1))

    # Bodies: thick line open → close, coloured by direction
    up_segs = [[(i, o[i]), (i, c[i])] for i in range(n) if up[i]]
    dn_segs = [[(i, c[i]), (i, o[i])] for i in range(n) if not up[i]]
    if up_segs:
        ax.add_collection(LineCollection(up_segs, colors="#3fb950",
                                         linewidths=3.0, zorder=2))
    if dn_segs:
        ax.add_collection(LineCollection(dn_segs, colors="#f85149",
                                         linewidths=3.0, zorder=2))


def plot(df: pd.DataFrame, zones: list, profiles: dict,
         anchored: bool, bin_size: float, show_n: int = 15,
         random_window: bool = False):

    # ── Select display window ─────────────────────────────────────────────────
    if random_window:
        # Only pick windows whose price range overlaps with at least one zone.
        # Retry up to 200 times, then fall back to latest bars.
        zone_prices = np.array([z.price for z in zones]) if zones else np.array([])
        max_start   = max(0, len(df) - DISPLAY_BARS - 1)
        window      = None
        for _ in range(200):
            si   = int(np.random.randint(0, max_start + 1))
            seg  = df.iloc[si: si + DISPLAY_BARS]
            lo_w = seg["low"].min()
            hi_w = seg["high"].max()
            if len(zone_prices) == 0 or np.any((zone_prices >= lo_w) & (zone_prices <= hi_w)):
                window = seg.copy().reset_index(drop=True)
                break
        if window is None:
            window = df.tail(DISPLAY_BARS).copy().reset_index(drop=True)
    else:
        window = df.tail(DISPLAY_BARS).copy().reset_index(drop=True)

    date_start = window["date"].iloc[0]
    date_end   = window["date"].iloc[-1]

    price_lo = window["low"].min()
    price_hi = window["high"].max()
    spread   = price_hi - price_lo
    pad      = spread * 0.06
    y_lo, y_hi = price_lo - pad, price_hi + pad

    dark = "#0D1117"
    fig  = plt.figure(figsize=(22, 10), facecolor=dark)
    gs   = gridspec.GridSpec(1, 4, figure=fig,
                             width_ratios=[3.8, 0.03, 0.7, 0.03],
                             wspace=0.025)
    ax_p  = fig.add_subplot(gs[0])
    ax_c  = fig.add_subplot(gs[1])
    ax_vp = fig.add_subplot(gs[2])
    ax_c2 = fig.add_subplot(gs[3])

    for ax in [ax_p, ax_c, ax_vp, ax_c2]:
        ax.set_facecolor(dark)
        for sp in ax.spines.values():
            sp.set_color("#21262D")

    x = np.arange(len(window))

    # OHLC candlesticks
    _draw_ohlc(ax_p, window)

    # ── Confluence heatmap bands ──────────────────────────────────────────────
    visible = [z for z in zones if y_lo <= z.price <= y_hi]

    if visible:
        max_score = max(z.score for z in visible)
        cmap      = plt.cm.YlOrRd

        for zone in visible:
            ns     = zone.score / max_score
            colour = cmap(0.15 + 0.85 * ns)
            alpha  = 0.10 + 0.52 * ns
            hw     = bin_size * 1.3
            ax_p.axhspan(zone.price - hw, zone.price + hw,
                         color=colour, alpha=alpha, linewidth=0)

        # Annotate high-score zones
        top_vis = [z for z in visible if z.score > 0.50 * max_score][:show_n]
        for z in top_vis:
            ns     = z.score / max_score
            colour = cmap(0.15 + 0.85 * ns)
            tfs    = ",".join(sorted(z.timeframes))
            fts    = "|".join(sorted(z.ftypes))
            ax_p.annotate(
                f"{z.score:.1f} [{tfs}]\n{fts}",
                xy=(x[-1], z.price),
                xytext=(10, 0), textcoords="offset points",
                color=colour, fontsize=5.5, va="center",
                fontfamily="monospace",
            )

    # VAH / VAL reference lines per timeframe (POC removed)
    for tf, vp in profiles.items():
        c = TF_COLORS[tf]
        ax_p.axhline(vp.vah, color=c, lw=0.7, ls="--", alpha=0.45, label=f"VAH/VAL {tf}")
        ax_p.axhline(vp.val, color=c, lw=0.7, ls="--", alpha=0.45)

    # ── Date x-axis ───────────────────────────────────────────────────────────
    n_bars = len(window)
    dates  = window["date"].values   # numpy datetime64 array

    # Pick ~8 evenly spaced tick positions
    n_ticks  = 8
    tick_idx = np.linspace(0, n_bars - 1, n_ticks, dtype=int)

    def fmt_date(idx):
        ts = pd.Timestamp(dates[int(idx)])
        return ts.strftime("%b %d\n%H:%M")

    ax_p.set_xticks(tick_idx)
    ax_p.set_xticklabels([fmt_date(i) for i in tick_idx],
                          fontsize=7, color="#8B949E")

    ax_p.set_xlim(0, n_bars - 1)
    ax_p.set_ylim(y_lo, y_hi)
    ax_p.tick_params(axis="y", colors="#8B949E", labelsize=8)
    ax_p.tick_params(axis="x", colors="#8B949E", labelsize=7, length=3)
    ax_p.set_ylabel("Price  (NQ)", color="#8B949E", fontsize=9)
    ax_p.legend(loc="upper left", fontsize=7, framealpha=0.25,
                labelcolor="white", facecolor="#161B22", edgecolor="#21262D")

    mode    = "Anchored" if anchored else "Rolling"
    win_tag = "random" if random_window else "latest"
    d_start = date_start.strftime("%Y-%m-%d %H:%M")
    d_end   = date_end.strftime("%Y-%m-%d %H:%M")
    ax_p.set_title(
        f"NQ Stacked VP  ·  {mode}  ·  {d_start}  →  {d_end}  ({win_tag})",
        color="#F0F6FC", fontsize=10, pad=10, fontweight="bold",
    )

    # Colorbar for bands
    sm1 = plt.cm.ScalarMappable(cmap="YlOrRd", norm=plt.Normalize(0, 1))
    sm1.set_array([])
    cb1 = plt.colorbar(sm1, cax=ax_c)
    cb1.set_label("Confluence", color="#8B949E", fontsize=7)
    cb1.ax.tick_params(colors="#8B949E", labelsize=6)

    # ── Composite VP heatmap (right strip) ───────────────────────────────────
    tf_list  = list(profiles.keys())
    ref_bins = profiles[tf_list[0]].bins if tf_list else np.array([])
    nb       = len(ref_bins)

    # Stack normalised volume profiles side-by-side, then average
    hmap = np.zeros((nb, len(tf_list)))
    for j, tf in enumerate(tf_list):
        vp   = profiles[tf]
        col  = np.interp(ref_bins, vp.bins, vp.vols, left=0, right=0)
        mx   = col.max()
        hmap[:, j] = col / mx if mx > 0 else col

    composite = hmap.mean(axis=1)

    # Blend with normalised confluence scores
    cf_layer = np.zeros(nb)
    if visible:
        mx_s = max(z.score for z in visible)
        for z in visible:
            ci = int(np.argmin(np.abs(ref_bins - z.price)))
            cf_layer[ci] = max(cf_layer[ci], z.score / mx_s)

    final_hmap = 0.30 * composite + 0.70 * cf_layer

    extent = [0, 1,
              float(ref_bins[0])  if nb else 0,
              float(ref_bins[-1]) if nb else 1]
    ax_vp.imshow(final_hmap.reshape(-1, 1), aspect="auto", origin="lower",
                 cmap="YlOrRd", extent=extent, vmin=0, vmax=1)
    ax_vp.set_xlim(0, 1)
    ax_vp.set_ylim(y_lo, y_hi)
    ax_vp.set_xticks([])
    ax_vp.yaxis.set_label_position("right")
    ax_vp.yaxis.tick_right()
    ax_vp.tick_params(colors="#8B949E", labelsize=7)
    ax_vp.set_title("VP\nHeat", color="#F0F6FC", fontsize=8, pad=6)

    sm2 = plt.cm.ScalarMappable(cmap="YlOrRd", norm=plt.Normalize(0, 1))
    sm2.set_array([])
    cb2 = plt.colorbar(sm2, cax=ax_c2)
    cb2.ax.tick_params(colors="#8B949E", labelsize=6)

    plt.tight_layout(pad=0.6)

    mode_tag = "anchored" if anchored else "rolling"
    win_slug  = date_start.strftime("%Y%m%d_%H%M")
    out_file  = f"stacked_vp_{mode_tag}_{win_slug}.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight", facecolor=dark)
    print(f"\n  Saved → {out_file}")
    plt.show()


# ─── Backtest ─────────────────────────────────────────────────────────────────

def backtest(
    df: pd.DataFrame,
    zones: list,
    tolerance: float,
    years: int = 3,
    lookback_bars: int = 60,
    forward_bars: int = 390,      # full session (6.5 h) — HOD/LOD made hours after touch
    min_sep_bars: int = 240,      # 4 h separation — one touch per session direction
    reversal_pct: float = 0.006,  # 0.6% ≈ 150 pts @ 25k ≈ 0.4 ATR — meaningful swing
    quiet: bool = False,
) -> pd.DataFrame:
    """
    For each confluence zone, scan the past `years` years of 1-minute data and
    find every qualifying touch.

    Touch requirements (both must be true):
      1. Bar's [low, high] range overlaps zone ± tolerance  (wick enters zone)
      2. Bar's close stays within 4 × tolerance of zone     (loose gate — swing highs
         often wick through before closing back near zone)

    Approach direction:
      - 'resistance': close lookback_bars before touch was above zone + tolerance
      - 'support':    close lookback_bars before touch was below zone - tolerance
      - 'ambiguous':  price was already inside the zone

    Reversal: price moves ≥ reversal_pct (or rev_atr fraction of ATR20) in the
              approach-opposite direction within the next forward_bars bars.
    """
    last_date  = df["date"].max()
    start_date = last_date - pd.Timedelta(days=int(years * 365.25))
    df3 = df[df["date"] >= start_date].reset_index(drop=True)

    hi  = df3["high"].values.astype(np.float64)
    lo  = df3["low"].values.astype(np.float64)
    cl  = df3["close"].values.astype(np.float64)
    n   = len(df3)
    if not quiet:
        print(f"  Backtest window: {df3['date'].iloc[0].date()} → {df3['date'].iloc[-1].date()}  ({n:,} bars)")

    rows = []
    for zone in zones:
        p   = zone.price
        tol = tolerance

        # Touch = wick enters zone AND close stays within 4× tol (swing highs
        # wick through before closing back — 2× was filtering them out)
        close_near  = np.abs(cl - p) <= tol * 4
        raw_touches = np.where(
            (lo <= p + tol) & (hi >= p - tol) & close_near
        )[0]

        # Deduplicate: keep first touch in any run of min_sep_bars
        deduped: list[int] = []
        last_t = -min_sep_bars
        for t in raw_touches:
            if t - last_t >= min_sep_bars:
                deduped.append(t)
                last_t = t

        total     = len(deduped)
        reversals = 0
        rev_moves: list[float] = []

        for t in deduped:
            touch_price = cl[t]
            threshold   = reversal_pct * touch_price

            # Approach direction
            prev_idx   = max(0, t - lookback_bars)
            prev_close = cl[prev_idx]
            if prev_close > p + tol:
                direction = "resistance"
            elif prev_close < p - tol:
                direction = "support"
            else:
                direction = "ambiguous"

            # Forward window
            fwd_end = min(n, t + forward_bars + 1)
            if fwd_end <= t:
                continue
            fwd_hi = hi[t:fwd_end].max()
            fwd_lo = lo[t:fwd_end].min()

            if direction == "resistance":
                # Expect price to fall away: down-move must exceed threshold
                move = touch_price - fwd_lo
            elif direction == "support":
                # Expect price to rise away: up-move must exceed threshold
                move = fwd_hi - touch_price
            else:
                # Ambiguous: accept either direction
                move = max(touch_price - fwd_lo, fwd_hi - touch_price)

            if move >= threshold:
                reversals += 1
                rev_moves.append(move / touch_price * 100)

        rows.append({
            "price":        round(zone.price, 2),
            "score":        round(zone.score, 1),
            "n_tf":         zone.n_tf,
            "timeframes":   ",".join(sorted(zone.timeframes)),
            "ftypes":       "|".join(sorted(zone.ftypes)),
            "touches":      total,
            "reversals":    reversals,
            "reversal_rate": round(reversals / total, 3) if total > 0 else 0.0,
            "avg_rev_pct":  round(float(np.mean(rev_moves)), 3) if rev_moves else 0.0,
        })

    if not rows:
        return pd.DataFrame(columns=["price","score","n_tf","timeframes","ftypes",
                                     "touches","reversals","reversal_rate","avg_rev_pct"])
    return pd.DataFrame(rows).sort_values("reversal_rate", ascending=False)


def _baseline_rate(df: pd.DataFrame, n_samples: int, tolerance: float,
                   forward_bars: int, min_sep_bars: int,
                   reversal_pct: float, years: int) -> float:
    """
    Sample n_samples random price levels from the backtest window and compute
    their average touch-weighted reversal rate using identical methodology.
    This is the 'random level' baseline our zones must beat.
    """
    last_date  = df["date"].max()
    start_date = last_date - pd.Timedelta(days=int(years * 365.25))
    df3 = df[df["date"] >= start_date].reset_index(drop=True)
    hi  = df3["high"].values.astype(np.float64)
    lo  = df3["low"].values.astype(np.float64)
    cl  = df3["close"].values.astype(np.float64)
    n   = len(df3)

    # Sample random prices from the data's own range so they're realistic levels
    all_rates, all_touches = [], []
    rng = np.random.default_rng(seed=42)
    prices = rng.uniform(lo.min(), hi.max(), size=n_samples * 3)
    # Keep only prices that have at least 10 touches (otherwise too noisy)
    tested = 0
    for p in prices:
        if tested >= n_samples:
            break
        close_near  = np.abs(cl - p) <= tolerance * 4   # match backtest() gate
        raw = np.where((lo <= p + tolerance) & (hi >= p - tolerance) & close_near)[0]
        deduped = []
        last_t = -min_sep_bars
        for t in raw:
            if t - last_t >= min_sep_bars:
                deduped.append(t); last_t = t
        if len(deduped) < 10:
            continue
        revs = 0
        for t in deduped:
            tp  = cl[t]
            thr = reversal_pct * tp
            prev_cl = cl[max(0, t - 30)]
            fwd_end = min(n, t + forward_bars + 1)
            fwd_hi  = hi[t:fwd_end].max()
            fwd_lo  = lo[t:fwd_end].min()
            if prev_cl > p + tolerance:
                move = tp - fwd_lo
            elif prev_cl < p - tolerance:
                move = fwd_hi - tp
            else:
                move = max(tp - fwd_lo, fwd_hi - tp)
            if move >= thr:
                revs += 1
        rate = revs / len(deduped)
        all_rates.append(rate)
        all_touches.append(len(deduped))
        tested += 1

    if not all_rates:
        return 0.0
    total_t = sum(all_touches)
    return float(sum(r * t for r, t in zip(all_rates, all_touches)) / total_t)


def print_backtest_report(df: pd.DataFrame, bt: pd.DataFrame,
                          reversal_pct: float = 0.0035,
                          tolerance: float = BIN_SIZE,
                          forward_bars: int = 390,
                          min_sep_bars: int = 240):
    pct_str  = f"{reversal_pct*100:.2f}%"
    win_str  = f"{forward_bars}-bar window"

    print("  Computing baseline (30 random price levels)...", end="", flush=True)
    baseline = _baseline_rate(df, n_samples=30, tolerance=tolerance,
                              forward_bars=forward_bars, min_sep_bars=min_sep_bars,
                              reversal_pct=reversal_pct, years=3)
    print(f"  baseline = {baseline:.1%}")

    sep = "-" * 92
    print(f"\n{sep}")
    print(f"  BACKTEST — threshold {pct_str}  |  {win_str}  |  close-confirmed  |  baseline {baseline:.1%}")
    print(sep)
    print(f"  {'PRICE':>9}  {'SCORE':>5}  {'TF':>2}  {'TOUCHES':>7}  "
          f"{'REVS':>5}  {'RATE':>6}  {'vs BASE':>7}  {'AVG_REV%':>8}  TYPES")
    print(sep)
    for _, r in bt.iterrows():
        edge = r["reversal_rate"] - baseline
        flag = " *" if edge >= 0.08 else "  "
        print(f"  {r['price']:>9.2f}  {r['score']:>5.1f}  {r['n_tf']:>2}  "
              f"{r['touches']:>7}  {r['reversals']:>5}  "
              f"{r['reversal_rate']:>5.1%}  {edge:>+6.1%}  "
              f"{r['avg_rev_pct']:>7.3f}%  {r['ftypes']}{flag}")
    print(sep)

    zone_rate = float((bt["reversal_rate"] * bt["touches"]).sum() / bt["touches"].sum())
    lift      = zone_rate - baseline
    above     = bt[bt["reversal_rate"] - baseline >= 0.05]
    print(f"\n  Random baseline rate        : {baseline:.1%}")
    print(f"  All-zone weighted rate      : {zone_rate:.1%}")
    print(f"  Lift over baseline          : {lift:+.1%}")
    print(f"  Zones beating baseline +5%  : {len(above)} of {len(bt)}")
    print(f"  Zones beating baseline +8%  : {sum(1 for _, r in bt.iterrows() if r['reversal_rate'] - baseline >= 0.08)}  (* in table)")


# ─── Gradient Tuning Sweep ────────────────────────────────────────────────────

def tune_shelf_gradient(
    df: pd.DataFrame,
    bin_size: float,
    reversal_pct: float = 0.0035,
    shelf_steps: int = 20,
) -> pd.DataFrame:
    """
    Sweep SHELF_SLOPE_MAX from 0.01 → 0.30 while holding LEDGE_SLOPE_MIN fixed.
    For each value: rebuild features, restack zones, run the full backtest.
    Reports touch-weighted reversal rate and zone counts so we can find the
    inflection point where tighter/looser shelf classification helps most.
    """
    years      = 3
    start_date = df["date"].max() - pd.Timedelta(days=int(years * 365.25))
    df_train   = df[df["date"] < start_date]
    _, profiles = run(df_train, anchored=False, bin_size=bin_size, quiet=True)

    shelf_vals = np.round(np.linspace(0.01, 0.30, shelf_steps), 4)

    sep = "-" * 80
    print(f"\n{sep}")
    print(f"  SHELF_SLOPE_MAX sweep  (LEDGE_SLOPE_MIN fixed at {LEDGE_SLOPE_MIN})")
    print(f"  reversal threshold: {reversal_pct*100:.2f}%  |  {shelf_steps} steps")
    print(sep)
    print(f"  {'SHELF_MAX':>9}  {'SHELVES':>7}  {'ZONES':>5}  "
          f"{'W_RATE':>7}  {'AVG_RATE':>8}  {'MAX_RATE':>8}  {'>=50%':>5}  {'>=60%':>5}")
    print(sep)

    rows = []
    for sh in shelf_vals:
        # Re-detect features with this shelf threshold, LVN/VAH/VAL unchanged
        all_features = []
        for vp in profiles.values():
            all_features.extend(detect_features(vp, shelf_max=sh))
        zones = stack_features(all_features, bin_size)

        n_shelf = sum(1 for f in all_features if f.ftype == "shelf")

        bt = backtest(df, zones, tolerance=bin_size,
                      reversal_pct=reversal_pct, quiet=True)

        if len(bt) == 0 or bt["touches"].sum() == 0:
            w_rate = avg_rate = max_rate = n50 = n60 = 0.0
        else:
            total_t = bt["touches"].sum()
            w_rate  = float((bt["reversal_rate"] * bt["touches"]).sum() / total_t)
            avg_rate = float(bt["reversal_rate"].mean())
            max_rate = float(bt["reversal_rate"].max())
            n50 = int((bt["reversal_rate"] >= 0.50).sum())
            n60 = int((bt["reversal_rate"] >= 0.60).sum())

        print(f"  {sh:>9.4f}  {n_shelf:>7}  {len(zones):>5}  "
              f"{w_rate:>7.1%}  {avg_rate:>8.1%}  {max_rate:>8.1%}  {n50:>5}  {n60:>5}")

        rows.append({
            "shelf_max":  sh,
            "n_shelf":    n_shelf,
            "n_zones":    len(zones),
            "w_rate":     round(w_rate,   4),
            "avg_rate":   round(avg_rate, 4),
            "max_rate":   round(max_rate, 4),
            "n_50pct":    n50,
            "n_60pct":    n60,
        })

    print(sep)
    result = pd.DataFrame(rows)

    # Highlight the best row by weighted rate
    best = result.loc[result["w_rate"].idxmax()]
    print(f"\n  Best shelf_max by touch-weighted rate: {best['shelf_max']:.4f}  "
          f"(w_rate={best['w_rate']:.1%}, >=50%: {int(best['n_50pct'])})")

    out = "gradient_sweep.csv"
    result.to_csv(out, index=False)
    print(f"  Full sweep saved → {out}")
    return result


# ─── Optimization Sweep ──────────────────────────────────────────────────────

def _run_cfg(df: pd.DataFrame, cfg: dict, anchored: bool = False) -> tuple:
    """
    Run VP analysis with parameter overrides in cfg.
    cfg keys (all optional, fall back to module-level defaults):
      hvn_prominence, lvn_depth, shelf_max, ledge_min,
      tol_mult, score_exp,
      lookbacks  → set/list of TF names to include (e.g. {"weekly","30d"})
      ftypes     → set of feature ftypes to keep (None = all)
      incl_prior → bool, include PDH/PDL/PWH/PWL features (default True)
    Returns (zones, all_features).
    """
    bin_size   = cfg.get("bin_size",        BIN_SIZE)
    hvn_prom   = cfg.get("hvn_prominence",  HVN_PROMINENCE)
    lvn_dep    = cfg.get("lvn_depth",       LVN_DEPTH)
    s_max      = cfg.get("shelf_max",       SHELF_SLOPE_MAX)
    l_min      = cfg.get("ledge_min",       LEDGE_SLOPE_MIN)
    tol_mult   = cfg.get("tol_mult",        STACK_TOLERANCE)
    score_exp  = cfg.get("score_exp",       2.0)
    allowed_tfs= set(cfg.get("lookbacks",   list(LOOKBACKS.keys())))
    ftypes_keep= cfg.get("ftypes",          None)
    incl_prior = cfg.get("incl_prior",      True)

    all_features: list[VPFeature] = []

    for name in LOOKBACKS:
        if name not in allowed_tfs:
            continue
        cutoff = get_cutoff(df, name, anchored)
        subset = df[df["date"] >= cutoff]
        if len(subset) < 20:
            continue
        vp    = build_profile(subset, name, bin_size)
        feats = detect_features(vp, shelf_max=s_max, ledge_min=l_min,
                                hvn_prominence=hvn_prom, lvn_depth=lvn_dep)
        all_features.extend(feats)

    if incl_prior:
        all_features.extend(prior_day_features(df))

    if ftypes_keep is not None:
        all_features = [f for f in all_features if f.ftype in ftypes_keep]

    zones = stack_features(all_features, bin_size,
                           tol_mult=tol_mult, score_exp=score_exp)
    return zones, all_features


def _bt_metrics(df: pd.DataFrame, zones: list,
                tolerance: float, rev_pct: float,
                years: int = 3,
                forward_bars: int = 90,
                min_sep_bars: int = 60) -> dict:
    """Run backtest with given window params and return compact metrics dict."""
    if not zones:
        return dict(n_zones=0, w_rate=0.0, avg_rate=0.0,
                    max_rate=0.0, n5=0, n8=0, touches=0)
    bt = backtest(df, zones, tolerance=tolerance, reversal_pct=rev_pct,
                  years=years, quiet=True,
                  forward_bars=forward_bars, min_sep_bars=min_sep_bars)
    if bt.empty or bt["touches"].sum() == 0:
        return dict(n_zones=len(zones), w_rate=0.0, avg_rate=0.0,
                    max_rate=0.0, n5=0, n8=0, touches=0)
    total_t = int(bt["touches"].sum())
    w_rate  = float((bt["reversal_rate"] * bt["touches"]).sum() / total_t)
    return dict(
        n_zones  = len(zones),
        w_rate   = round(w_rate, 4),
        avg_rate = round(float(bt["reversal_rate"].mean()), 4),
        max_rate = round(float(bt["reversal_rate"].max()), 4),
        n5       = 0,   # caller fills in vs baseline
        touches  = total_t,
    )


def optimize_sweep(df: pd.DataFrame, bin_size: float = BIN_SIZE,
                   rev_pct: float = 0.006, anchored: bool = False):
    """
    Systematic sweep across 5 dimensions:
      1. Feature type ablation   — which types drive the edge
      2. Timeframe contribution  — which VP windows matter most
      3. HVN/LVN detection params — prominence & depth thresholds
      4. Shelf/Ledge classification — slope thresholds
      5. Stacking config         — tolerance × score exponent grid
    Also compares rolling vs anchored.
    Results saved to optimize_results.csv + printed summary.
    """
    tolerance  = bin_size
    years      = 3
    start_date = df["date"].max() - pd.Timedelta(days=int(years * 365.25))
    df_train   = df[df["date"] < start_date]
    # Tighter window for optimization — maximises discrimination between configs.
    # Full-session (390-bar) window is too noisy: random baseline hits ~30%
    # because NQ moves 150 pts from almost any level within 6.5 hours.
    # 90-bar (1.5h) window produces ~10-15% baseline with clear zone lift.
    OPT_FWD = 90
    OPT_SEP = 60

    # ── compute baseline once ────────────────────────────────────────────────
    print("  Computing random baseline...", end="", flush=True)
    baseline = _baseline_rate(df, n_samples=30, tolerance=tolerance,
                              forward_bars=OPT_FWD, min_sep_bars=OPT_SEP,
                              reversal_pct=rev_pct, years=years)
    print(f"  {baseline:.1%}")

    base_cfg = {}   # all defaults

    def _run(label: str, cfg: dict) -> dict:
        zones, _ = _run_cfg(df_train, cfg, anchored=anchored)
        if not zones:
            return dict(label=label, n_zones=0, w_rate=0.0, avg_rate=0.0,
                        max_rate=0.0, touches=0, lift=-baseline, n5=0, n8=0)
        bt = backtest(df, zones, tolerance=tolerance,
                      reversal_pct=rev_pct, years=years, quiet=True,
                      forward_bars=OPT_FWD, min_sep_bars=OPT_SEP)
        if bt.empty or bt["touches"].sum() == 0:
            return dict(label=label, n_zones=len(zones), w_rate=0.0,
                        avg_rate=0.0, max_rate=0.0, touches=0,
                        lift=-baseline, n5=0, n8=0)
        total_t = int(bt["touches"].sum())
        w_rate  = float((bt["reversal_rate"] * bt["touches"]).sum() / total_t)
        n5 = int((bt["reversal_rate"] - baseline >= 0.05).sum())
        n8 = int((bt["reversal_rate"] - baseline >= 0.08).sum())
        return dict(
            label    = label,
            n_zones  = len(zones),
            w_rate   = round(w_rate, 4),
            avg_rate = round(float(bt["reversal_rate"].mean()), 4),
            max_rate = round(float(bt["reversal_rate"].max()), 4),
            touches  = total_t,
            lift     = round(w_rate - baseline, 4),
            n5       = n5,
            n8       = n8,
        )

    all_rows = []
    sep = "-" * 78

    # ─── 1. Feature type ablation ─────────────────────────────────────────────
    ALL_FTYPES = {"vah", "val", "shelf", "ledge", "lvn",
                  "pdh", "pdl", "pwh", "pwl"}

    print(f"\n{'='*78}")
    print("  1. FEATURE TYPE ABLATION  (remove one type at a time)")
    print(f"{'='*78}")
    print(f"  {'LABEL':<28}  {'ZONES':>5}  {'W_RATE':>7}  {'LIFT':>7}  {'N≥5%':>5}  {'N≥8%':>5}")
    print(sep)

    # baseline — all features
    m = _run("ALL features", base_cfg)
    all_rows.append({**m, "dimension": "feature_ablation"})
    print(f"  {'ALL features':<28}  {m['n_zones']:>5}  {m['w_rate']:>6.1%}  "
          f"{m['lift']:>+6.1%}  {m['n5']:>5}  {m['n8']:>5}  ← baseline")

    for ftype in sorted(ALL_FTYPES):
        keep = ALL_FTYPES - {ftype}
        m = _run(f"No {ftype}", {**base_cfg, "ftypes": keep, "incl_prior": True})
        # re-apply incl_prior logic: if removing pdh/pdl, also disable incl_prior
        if ftype in {"pdh", "pdl", "pwh", "pwl"}:
            m = _run(f"No {ftype}", {**base_cfg, "ftypes": keep})
        all_rows.append({**m, "dimension": "feature_ablation"})
        print(f"  {f'No {ftype}':<28}  {m['n_zones']:>5}  {m['w_rate']:>6.1%}  "
              f"{m['lift']:>+6.1%}  {m['n5']:>5}  {m['n8']:>5}")

    print(sep)
    print("  Solo feature types:")
    for ftype in sorted(ALL_FTYPES):
        m = _run(f"Only {ftype}", {**base_cfg, "ftypes": {ftype}})
        all_rows.append({**m, "dimension": "feature_solo"})
        print(f"  {'Only '+ftype:<28}  {m['n_zones']:>5}  {m['w_rate']:>6.1%}  "
              f"{m['lift']:>+6.1%}  {m['n5']:>5}  {m['n8']:>5}")

    # ─── 2. Timeframe contribution ────────────────────────────────────────────
    ALL_TFS = set(LOOKBACKS.keys())
    print(f"\n{'='*78}")
    print("  2. TIMEFRAME CONTRIBUTION  (remove / isolate one TF at a time)")
    print(f"{'='*78}")
    print(f"  {'LABEL':<28}  {'ZONES':>5}  {'W_RATE':>7}  {'LIFT':>7}  {'N≥5%':>5}  {'N≥8%':>5}")
    print(sep)

    m = _run("ALL timeframes", base_cfg)
    all_rows.append({**m, "dimension": "tf_contribution"})
    print(f"  {'ALL timeframes':<28}  {m['n_zones']:>5}  {m['w_rate']:>6.1%}  "
          f"{m['lift']:>+6.1%}  {m['n5']:>5}  {m['n8']:>5}  ← baseline")

    for tf in sorted(ALL_TFS):
        m = _run(f"No {tf}", {**base_cfg, "lookbacks": ALL_TFS - {tf}})
        all_rows.append({**m, "dimension": "tf_contribution"})
        print(f"  {f'No {tf}':<28}  {m['n_zones']:>5}  {m['w_rate']:>6.1%}  "
              f"{m['lift']:>+6.1%}  {m['n5']:>5}  {m['n8']:>5}")
    for tf in sorted(ALL_TFS):
        m = _run(f"Only {tf}", {**base_cfg, "lookbacks": {tf}})
        all_rows.append({**m, "dimension": "tf_solo"})
        print(f"  {'Only '+tf:<28}  {m['n_zones']:>5}  {m['w_rate']:>6.1%}  "
              f"{m['lift']:>+6.1%}  {m['n5']:>5}  {m['n8']:>5}")

    # ─── 3. HVN prominence sweep ──────────────────────────────────────────────
    print(f"\n{'='*78}")
    print("  3. HVN PROMINENCE  (how prominent a peak must be to classify as HVN)")
    print(f"{'='*78}")
    print(f"  {'HVN_PROM':>8}  {'ZONES':>5}  {'W_RATE':>7}  {'LIFT':>7}  {'N≥5%':>5}  {'N≥8%':>5}")
    print(sep)
    for val in np.round(np.linspace(0.08, 0.35, 10), 3):
        m = _run(f"hvn={val:.3f}", {**base_cfg, "hvn_prominence": float(val)})
        m["param_val"] = float(val)
        all_rows.append({**m, "dimension": "hvn_prominence"})
        marker = " ◄" if abs(val - HVN_PROMINENCE) < 0.01 else ""
        print(f"  {val:>8.3f}  {m['n_zones']:>5}  {m['w_rate']:>6.1%}  "
              f"{m['lift']:>+6.1%}  {m['n5']:>5}  {m['n8']:>5}{marker}")

    # ─── 4. LVN depth sweep ───────────────────────────────────────────────────
    print(f"\n{'='*78}")
    print("  4. LVN DEPTH  (valley must be < this × mean_vol to count as LVN)")
    print(f"{'='*78}")
    print(f"  {'LVN_DEPTH':>9}  {'ZONES':>5}  {'W_RATE':>7}  {'LIFT':>7}  {'N≥5%':>5}  {'N≥8%':>5}")
    print(sep)
    for val in np.round(np.linspace(0.20, 0.80, 10), 3):
        m = _run(f"lvn={val:.3f}", {**base_cfg, "lvn_depth": float(val)})
        m["param_val"] = float(val)
        all_rows.append({**m, "dimension": "lvn_depth"})
        marker = " ◄" if abs(val - LVN_DEPTH) < 0.03 else ""
        print(f"  {val:>9.3f}  {m['n_zones']:>5}  {m['w_rate']:>6.1%}  "
              f"{m['lift']:>+6.1%}  {m['n5']:>5}  {m['n8']:>5}{marker}")

    # ─── 5. Shelf/Ledge slope sweep ───────────────────────────────────────────
    print(f"\n{'='*78}")
    print("  5. SHELF_MAX / LEDGE_MIN  (boundary slope classification thresholds)")
    print(f"{'='*78}")
    print(f"  {'SH_MAX':>6}  {'LD_MIN':>6}  {'ZONES':>5}  {'W_RATE':>7}  {'LIFT':>7}  {'N≥5%':>5}  {'N≥8%':>5}")
    print(sep)
    for sh in np.round(np.linspace(0.03, 0.20, 6), 3):
        for ld in np.round(np.linspace(0.15, 0.45, 4), 3):
            if sh >= ld:
                continue
            m = _run(f"sh={sh:.2f}/ld={ld:.2f}",
                     {**base_cfg, "shelf_max": float(sh), "ledge_min": float(ld)})
            m["param_val"] = float(sh)
            all_rows.append({**m, "dimension": "shelf_ledge"})
            marker = (" ◄" if abs(sh - SHELF_SLOPE_MAX) < 0.01
                              and abs(ld - LEDGE_SLOPE_MIN) < 0.02 else "")
            print(f"  {sh:>6.3f}  {ld:>6.3f}  {m['n_zones']:>5}  "
                  f"{m['w_rate']:>6.1%}  {m['lift']:>+6.1%}  "
                  f"{m['n5']:>5}  {m['n8']:>5}{marker}")

    # ─── 6. Stacking: tolerance × score exponent ─────────────────────────────
    print(f"\n{'='*78}")
    print("  6. STACK TOLERANCE × SCORE EXPONENT  (confluence radius & TF-alignment reward)")
    print(f"{'='*78}")
    print(f"  {'TOL_MULT':>8}  {'SCORE_EXP':>9}  {'ZONES':>5}  {'W_RATE':>7}  {'LIFT':>7}  {'N≥5%':>5}  {'N≥8%':>5}")
    print(sep)
    for tol in [0.8, 1.0, 1.5, 2.0, 2.5, 3.0]:
        for exp in [1.0, 1.5, 2.0, 2.5, 3.0]:
            m = _run(f"tol={tol}/exp={exp}",
                     {**base_cfg, "tol_mult": tol, "score_exp": exp})
            all_rows.append({**m, "dimension": "stacking",
                             "tol_mult": tol, "score_exp_val": exp})
            marker = (" ◄" if abs(tol - STACK_TOLERANCE) < 0.1
                              and abs(exp - 2.0) < 0.1 else "")
            print(f"  {tol:>8.1f}  {exp:>9.1f}  {m['n_zones']:>5}  "
                  f"{m['w_rate']:>6.1%}  {m['lift']:>+6.1%}  "
                  f"{m['n5']:>5}  {m['n8']:>5}{marker}")

    # ─── 7. Rolling vs Anchored ───────────────────────────────────────────────
    print(f"\n{'='*78}")
    print("  7. ROLLING vs ANCHORED PROFILES")
    print(f"{'='*78}")
    for anc in [False, True]:
        label = "Anchored" if anc else "Rolling"
        m = _run(label, {**base_cfg})   # _run already uses OPT_FWD/OPT_SEP
        # re-run with correct anchored flag, still using df_train for zones
        zones_a, _ = _run_cfg(df_train, base_cfg, anchored=anc)
        bt = backtest(df, zones_a, tolerance=tolerance,
                      reversal_pct=rev_pct, years=years, quiet=True,
                      forward_bars=OPT_FWD, min_sep_bars=OPT_SEP)
        if not bt.empty and bt["touches"].sum() > 0:
            total_t = int(bt["touches"].sum())
            w_rate  = float((bt["reversal_rate"] * bt["touches"]).sum() / total_t)
            n5 = int((bt["reversal_rate"] - baseline >= 0.05).sum())
            n8 = int((bt["reversal_rate"] - baseline >= 0.08).sum())
        else:
            w_rate = n5 = n8 = 0
        m = dict(label=label, n_zones=len(zones_a),
                 w_rate=round(w_rate, 4), lift=round(w_rate - baseline, 4),
                 n5=n5, n8=n8)
        all_rows.append({**m, "dimension": "mode"})
        print(f"  {label:<12}  zones={m['n_zones']:>3}  "
              f"w_rate={m['w_rate']:.1%}  lift={m['lift']:+.1%}  "
              f"N≥5%={m['n5']}  N≥8%={m['n8']}")

    # ─── Summary: best config per dimension ───────────────────────────────────
    result_df = pd.DataFrame(all_rows)
    result_df.to_csv("optimize_results.csv", index=False)

    print(f"\n{'='*78}")
    print("  BEST CONFIG PER DIMENSION  (by lift over baseline)")
    print(f"{'='*78}")
    for dim in result_df["dimension"].unique():
        sub  = result_df[result_df["dimension"] == dim]
        best = sub.loc[sub["lift"].idxmax()]
        print(f"  {dim:<22}  best={best['label']:<30}  "
              f"lift={best['lift']:+.1%}  zones={int(best['n_zones'])}")

    print(f"\n  Full results → optimize_results.csv")
    print(f"  Baseline: {baseline:.1%}")
    return result_df


# ─── Sample Days ──────────────────────────────────────────────────────────────

def _select_day_zones(zones: list, day_lo: float, day_hi: float,
                      day_open: float, n_min: int = 4, n_max: int = 6) -> list:
    """
    Pick 4-6 zones for a single day's price range, balanced above/below day_open.
    Zones must already be score-sorted descending.
    Falls back to single-TF zones if multi-TF count < n_min.
    """
    n_target = 5
    qual = [z for z in zones if z.n_tf >= 2 and day_lo <= z.price <= day_hi]
    if len(qual) < n_min:
        qual = [z for z in zones if day_lo <= z.price <= day_hi]
    if not qual:
        return []
    if len(qual) <= n_max:
        return sorted(qual, key=lambda z: z.price)

    above    = [z for z in qual if z.price > day_open]
    below    = [z for z in qual if z.price <= day_open]
    n_above  = math.ceil(n_target / 2)
    n_below  = n_target - n_above
    sel_abv  = above[:n_above]
    sel_blw  = below[:n_below]
    if len(sel_abv) < n_above:
        sel_blw = below[:n_target - len(sel_abv)]
    elif len(sel_blw) < n_below:
        sel_abv = above[:n_target - len(sel_blw)]

    return sorted(sel_abv + sel_blw, key=lambda z: z.price)


def plot_sample_days(df: pd.DataFrame, zones: list, n: int = 15,
                     bin_size: float = BIN_SIZE, seed: int = None,
                     df_sample_pool: pd.DataFrame = None):
    """
    Generate n full-day OHLC charts with zero look-ahead.
    For each sample day, VP zones are recomputed using ONLY data prior to that
    session — exactly what would have been available to a live trader that morning.
    Zone bands: magenta, dark = high confidence, light = low confidence.

    df_sample_pool: if provided, sample candidate days only from this subset
                    (e.g. a 2024-2025 slice) while still using full df for
                    per-day prior-history VP computation.
    """
    if seed is not None:
        np.random.seed(seed)

    df2 = df.copy()
    df2["_day"] = df2["date"].dt.date

    # Use pool to restrict which days are candidates, but keep full df for history
    pool = df_sample_pool.copy() if df_sample_pool is not None else df2
    pool["_day"] = pool["date"].dt.date

    day_stats = pool.groupby("_day").agg(
        bars=("close", "count"),
        lo=("low", "min"), hi=("high", "max"), op=("open", "first"),
    )

    # Need >= 300 bars (real session) and enough prior history for VP windows
    all_days_full = sorted(df2["_day"].unique().tolist())
    all_days  = day_stats.index.tolist()
    full_days = [d for d in all_days
                 if day_stats.loc[d, "bars"] >= 300
                 and all_days_full.index(d) >= 90]   # 90 prior days minimum in full df

    if not full_days:
        print("  No eligible days found")
        return

    # Sample extra candidates; skip any that yield < 4 zones after per-day compute
    rng        = np.random.default_rng(seed=seed)
    candidates = sorted(rng.choice(full_days,
                                   size=min(n * 4, len(full_days)),
                                   replace=False).tolist())

    dark      = "#0D1117"
    generated = 0
    chart_idx = 0

    print(f"\n  Generating {n} leak-free day charts (zones computed per-day)...")

    for day in candidates:
        if generated >= n:
            break
        chart_idx += 1

        # All 1m bars strictly before this session — no future data
        df_prior = df2[df2["_day"] < day]
        # Only last 30 k bars needed (covers 90-day VP window + PDH/PDL)
        df_window = df_prior.tail(30_000).copy()
        if len(df_window) < 2000:
            continue

        day_zones, _ = run(df_window, anchored=True,
                           bin_size=bin_size, quiet=True)
        if not day_zones:
            continue

        bars     = df2[df2["_day"] == day].copy().reset_index(drop=True)
        price_lo = bars["low"].min()
        price_hi = bars["high"].max()
        spread   = price_hi - price_lo
        pad      = spread * 0.10
        y_lo, y_hi = price_lo - pad, price_hi + pad
        day_open   = float(bars["open"].iloc[0])

        selected = _select_day_zones(day_zones, y_lo, y_hi, day_open)
        if len(selected) < 4:
            continue   # skip days where prior structure gives too few levels

        max_score = max(z.score for z in day_zones)

        fig, ax = plt.subplots(figsize=(18, 7), facecolor=dark)
        ax.set_facecolor(dark)
        for sp in ax.spines.values():
            sp.set_color("#21262D")

        _draw_ohlc(ax, bars)

        # Mark actual HOD and LOD in gold so we can see if zones caught them
        hod_price = float(bars["high"].max())
        lod_price = float(bars["low"].min())
        ax.axhline(hod_price, color="#FFD700", lw=1.3, ls=":", alpha=0.90, zorder=6)
        ax.axhline(lod_price, color="#FFD700", lw=1.3, ls=":", alpha=0.90, zorder=6)
        ax.annotate("HOD", xy=(2, hod_price), xytext=(0, 3),
                    textcoords="offset points",
                    color="#FFD700", fontsize=6, fontfamily="monospace", zorder=7)
        ax.annotate("LOD", xy=(2, lod_price), xytext=(0, -9),
                    textcoords="offset points",
                    color="#FFD700", fontsize=6, fontfamily="monospace", zorder=7)

        for zone in selected:
            ns     = zone.score / max_score
            colour = MAGENTA_CMAP(0.20 + 0.80 * ns)
            hw     = zone.price * 0.0005  # 0.1% of price total band (~9pt@18k, ~13pt@25k)
            ax.axhspan(zone.price - hw, zone.price + hw,
                       color=colour, alpha=0.07 + 0.40 * ns, linewidth=0)
            ax.axhline(zone.price, color=colour,
                       lw=0.8 + 1.2 * ns, alpha=0.60 + 0.30 * ns, ls="--")
            role = "R" if zone.price > day_open else "S"
            fts  = "|".join(sorted(zone.ftypes))
            ax.annotate(
                f"{zone.price:.0f} [{role}] {fts}  n_tf={zone.n_tf}",
                xy=(len(bars) - 1, zone.price),
                xytext=(8, 0), textcoords="offset points",
                color=colour, fontsize=6.2, va="center",
                fontfamily="monospace", clip_on=False,
            )

        nb       = len(bars)
        ts_vals  = bars["date"].values
        n_ticks  = min(10, nb)
        tick_idx = np.linspace(0, nb - 1, n_ticks, dtype=int)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(
            [pd.Timestamp(ts_vals[i]).strftime("%H:%M") for i in tick_idx],
            fontsize=7, color="#8B949E",
        )
        ax.set_xlim(0, nb - 1)
        ax.set_ylim(y_lo, y_hi)
        ax.tick_params(axis="y", colors="#8B949E", labelsize=8)
        ax.tick_params(axis="x", colors="#8B949E", labelsize=7, length=3)
        ax.set_ylabel("Price  (NQ)", color="#8B949E", fontsize=9)

        day_str = str(day)
        ax.set_title(
            f"NQ  {day_str}  ·  {nb} bars  ·  {len(selected)} zones  "
            f"(no look-ahead · darker = higher confidence)",
            color="#F0F6FC", fontsize=10, pad=8, fontweight="bold",
        )

        plt.tight_layout(pad=0.5)
        out = f"day_{day_str}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=dark)
        plt.close(fig)
        generated += 1
        print(f"  [{generated:02d}/{n}]  {day_str}  bars={nb:4d}  "
              f"zones={len(selected)}  → {out}")

    print(f"  Done.  ({generated}/{n} generated, {chart_idx} candidates tried)")


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    global DISPLAY_BARS
    parser = argparse.ArgumentParser(
        description="NQ Stacked Volume Profile — confluence zone detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--anchored", action="store_true",
                        help="Calendar-anchored profiles (default: rolling)")
    parser.add_argument("--compare",  action="store_true",
                        help="Run both rolling and anchored, compare results")
    parser.add_argument("--bins",     type=float, default=BIN_SIZE,
                        help=f"Bin size in NQ points (default {BIN_SIZE})")
    parser.add_argument("--display",  type=int,   default=DISPLAY_BARS,
                        help=f"Recent 1-min bars in chart (default {DISPLAY_BARS})")
    parser.add_argument("--top",      type=int,   default=20,
                        help="Top N zones in console report (default 20)")
    parser.add_argument("--backtest", action="store_true",
                        help="Run 3-year reversal backtest on detected zones")
    parser.add_argument("--rev-pct",  type=float, default=0.60,
                        help="Reversal threshold in %% of price (default 0.60)")
    parser.add_argument("--rev-atr",  type=float, default=None,
                        help="Reversal threshold as fraction of 20-day ATR (e.g. 0.40); "
                             "overrides --rev-pct when set")
    parser.add_argument("--random",         action="store_true",
                        help="Show a random historical window instead of latest bars")
    parser.add_argument("--seed",           type=int,   default=None,
                        help="Random seed for --random (for reproducibility)")
    parser.add_argument("--tune-gradients", action="store_true",
                        help="Sweep SHELF_SLOPE_MAX and report backtest metrics per step")
    parser.add_argument("--sample-days",   type=int, default=0,
                        help="Generate N full-day OHLC charts with zone overlays")
    parser.add_argument("--daily",         type=int, default=5,
                        help="Print top N reversal levels for today (default 5, range 4-6)")
    parser.add_argument("--optimize",     action="store_true",
                        help="Run full 7-dimension optimization sweep (feature ablation, "
                             "TF contribution, detection params, stacking config, mode)")
    parser.add_argument("--hod-lod",      action="store_true",
                        help="Walk-forward HOD/LOD coverage test (60 random days, no look-ahead)")
    parser.add_argument("--hod-days",     type=int, default=60,
                        help="Number of days to sample for --hod-lod test (default 60)")
    args = parser.parse_args()

    DISPLAY_BARS = args.display
    if args.seed is not None:
        np.random.seed(args.seed)

    print("=" * 52)
    print("  NQ STACKED VOLUME PROFILE ANALYZER")
    print("=" * 52)

    print("\nLoading data...")
    df = load_data()
    print(f"  {len(df):,} bars  |  {df['date'].min().date()} → {df['date'].max().date()}")

    if args.optimize:
        rev_threshold = (args.rev_pct / 100.0)
        print(f"\nRunning optimization sweep  (rev_pct={args.rev_pct}%)...")
        optimize_sweep(df, bin_size=args.bins,
                       rev_pct=rev_threshold, anchored=args.anchored)
        return

    if args.hod_lod:
        print(f"\nRunning HOD/LOD coverage test  ({args.hod_days} days)...")
        hod_lod_coverage(df, n_days=args.hod_days, bin_size=args.bins)
        return

    modes = [False, True] if args.compare else [args.anchored]

    results = {}
    for anchored in modes:
        label = "ANCHORED" if anchored else "ROLLING"
        print(f"\n{'-'*52}")
        print(f"  {label} profiles")
        print(f"{'-'*52}")
        zones, profiles = run(df, anchored=anchored, bin_size=args.bins)
        print_report(zones, show_n=args.top)

        n_daily = max(4, min(6, args.daily))
        levels, last_close, atr20 = daily_levels(df, zones, n=n_daily)
        print_daily_levels(levels, last_close, atr20)

        results[label] = zones

        if args.sample_days > 0:
            plot_sample_days(df, zones, n=args.sample_days,
                             bin_size=args.bins, seed=args.seed)

        if args.tune_gradients:
            tune_shelf_gradient(df, bin_size=args.bins,
                                reversal_pct=args.rev_pct / 100.0)

        if args.backtest:
            if args.rev_atr is not None:
                atr_val = compute_atr20(df)
                rev_threshold = args.rev_atr * atr_val / df["close"].iloc[-1]
                thresh_label  = f"{args.rev_atr}×ATR ({atr_val*args.rev_atr:.0f} pts)"
            else:
                rev_threshold = args.rev_pct / 100.0
                thresh_label  = f"{args.rev_pct}%"
            # Use same fwd/sep window in both backtest() and baseline computation
            BT_FWD = 390
            BT_SEP = 240
            print(f"\nRunning backtest (threshold={thresh_label})...")
            bt_start  = df["date"].max() - pd.Timedelta(days=int(3 * 365.25))
            zones_bt, _ = run(df[df["date"] < bt_start], anchored=anchored,
                              bin_size=args.bins, quiet=True)
            bt = backtest(
                df, zones_bt,
                tolerance=args.bins,
                reversal_pct=rev_threshold,
                forward_bars=BT_FWD,
                min_sep_bars=BT_SEP,
            )
            print_backtest_report(df, bt, reversal_pct=rev_threshold,
                                      tolerance=args.bins,
                                      forward_bars=BT_FWD,
                                      min_sep_bars=BT_SEP)
            bt_file = f"backtest_{'anchored' if anchored else 'rolling'}.csv"
            bt.to_csv(bt_file, index=False)
            print(f"  Full results saved → {bt_file}")

        plot(df, zones, profiles, anchored=anchored,
             bin_size=args.bins, show_n=args.top,
             random_window=args.random)

    if args.compare and len(results) == 2:
        print("\n  -- COMPARISON ------------------------------------------")
        for label, zones in results.items():
            high = sum(1 for z in zones if z.n_tf >= 3)
            top4 = sum(1 for z in zones if z.n_tf == 4)
            print(f"  {label:<10}  3+tf={high:>3}  4tf={top4:>3}")


if __name__ == "__main__":
    main()
