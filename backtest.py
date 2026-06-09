"""
backtest.py  —  VP top/bottom tick prediction sweep

Requires prep_cache.py to have run first.

Three-stage sweep:
  Stage 1  strategy horse race         (fixed params, all strategies)
  Stage 2  param tuning on best strats (vary lookback/detection/VA/ref)
  Stage 3  full grid on top combos     (best strategy × best params)

Run:
    python backtest.py --stage 1          # strategy horse race (~minutes)
    python backtest.py --stage 2          # param tuning on stage-1 winners
    python backtest.py --stage 3          # full grid
    python backtest.py --stage all        # all stages sequentially
    python backtest.py --plot-only        # replot from existing CSVs
"""

import argparse
import itertools
import os
import sys
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import argrelmin, argrelmax

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vol_profile as vp

BASE        = os.path.dirname(os.path.abspath(__file__))
CACHE       = os.path.join(BASE, "cache")
PROFILE_DIR = os.path.join(CACHE, "profiles")
OHLCV_DIR   = os.path.join(CACHE, "ohlcv")
GRID_FILE   = os.path.join(CACHE, "price_grid.npy")
RESULTS_DIR = os.path.join(BASE, "results")

THRESHOLDS = [5, 10, 20, 30]
VA_PCT_DEFAULT = 0.70
BG = "#080808"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING  (all tiny, instant)
# ═══════════════════════════════════════════════════════════════════════════════

def load_price_grid() -> np.ndarray:
    return np.load(GRID_FILE).astype(np.float64)


def load_all_profiles(session: str = "full") -> dict[str, np.ndarray]:
    """Load daily profile arrays. session='full' or 'rth'."""
    out = {}
    for f in sorted(os.listdir(PROFILE_DIR)):
        if f.endswith(f"_{session}.npy"):
            date = f.replace(f"_{session}.npy", "")
            out[date] = np.load(os.path.join(PROFILE_DIR, f)).astype(np.float64)
    return out


def load_all_ohlcv(session: str = "rth") -> dict[str, pd.DataFrame]:
    out = {}
    for f in sorted(os.listdir(OHLCV_DIR)):
        if not f.endswith(".parquet"):
            continue
        d  = f[:-8]
        df = pd.read_parquet(os.path.join(OHLCV_DIR, f))
        out[d] = df[df["session"] == session].drop(columns=["session"])
    return out


def daily_summary(ohlcv: pd.DataFrame) -> dict | None:
    if ohlcv.empty:
        return None
    return {
        "high":  float(ohlcv["high"].max()),
        "low":   float(ohlcv["low"].min()),
        "open":  float(ohlcv["open"].iloc[0]),
        "close": float(ohlcv["close"].iloc[-1]),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PROFILE AGGREGATION  (numpy only, no raw data)
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_profiles(
    daily: dict[str, np.ndarray],
    prior_dates: list[str],
    anchored_modes: list[str],
    rolling_windows: list[int],
    weighting: str,                    # "equal" | "recency"
    week_profiles:  dict[str, np.ndarray],
    month_profiles: dict[str, np.ndarray],
) -> list[np.ndarray]:
    """
    Return list of profile arrays to feed into stacking.
    Each array is one "session" (day, week, month, or rolling window).
    weighting='recency': day profiles weighted by position index (older = less weight).
    """
    profiles = []
    n = len(prior_dates)

    # day profiles
    if "day" in anchored_modes:
        for idx, d in enumerate(prior_dates):
            if d not in daily:
                continue
            p = daily[d].copy()
            if weighting == "recency":
                w = (idx + 1) / n        # older=lower, newer=higher
                p = p * w
            profiles.append(p)

    # week cumulative
    if "week" in anchored_modes:
        seen = set()
        for d in prior_dates:
            wk = pd.Timestamp(d).strftime("%GW%V")
            if wk not in seen and wk in week_profiles:
                seen.add(wk)
                profiles.append(week_profiles[wk])

    # month cumulative
    if "month" in anchored_modes:
        seen = set()
        for d in prior_dates:
            mo = d[:6]
            if mo not in seen and mo in month_profiles:
                seen.add(mo)
                profiles.append(month_profiles[mo])

    # rolling windows
    for w in rolling_windows:
        roll = prior_dates[-w:]
        if len(roll) >= 2:
            rp = sum(daily[d] for d in roll if d in daily)
            if isinstance(rp, np.ndarray):
                profiles.append(rp)

    return profiles


def build_week_month_profiles(
    daily: dict[str, np.ndarray],
    sorted_dates: list[str],
) -> tuple[dict, dict]:
    week_p: dict[str, np.ndarray] = {}
    month_p: dict[str, np.ndarray] = {}
    for d in sorted_dates:
        if d not in daily:
            continue
        p  = daily[d]
        wk = pd.Timestamp(d).strftime("%GW%V")
        mo = d[:6]
        week_p[wk]  = week_p.get(wk,  np.zeros_like(p)) + p
        month_p[mo] = month_p.get(mo, np.zeros_like(p)) + p
    return week_p, month_p


# ═══════════════════════════════════════════════════════════════════════════════
# LVN / HVN DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_lvn_fast(
    profile: np.ndarray,
    smooth:    int   = 5,
    pct:       float = 25.0,
    depth_pct: float = 60.0,
    order:     int   = 2,
) -> np.ndarray:
    """Boolean mask — True = LVN bin."""
    if profile.sum() == 0:
        return np.zeros(len(profile), dtype=bool)
    s  = uniform_filter1d(profile, size=smooth)
    nz = s[s > 0]
    if len(nz) == 0:
        return np.zeros(len(profile), dtype=bool)
    thr       = np.percentile(nz, pct)
    mins_idx  = argrelmin(s, order=order)[0]
    mins_idx  = mins_idx[s[mins_idx] < thr]
    peaks_idx = argrelmax(s, order=order)[0]
    lvn = np.zeros(len(profile), dtype=bool)
    for mi in mins_idx:
        lp = peaks_idx[peaks_idx < mi]
        rp = peaks_idx[peaks_idx > mi]
        lv = s[lp[-1]] if len(lp) else None
        rv = s[rp[0]]  if len(rp) else None
        if lv is not None and rv is not None:
            if s[mi] < (depth_pct / 100) * ((lv + rv) / 2):
                lvn[mi] = True
        elif lv is not None or rv is not None:
            lvn[mi] = True
    return lvn


def detect_hvn_fast(
    profile: np.ndarray,
    smooth: int   = 5,
    pct:    float = 70.0,   # must be ABOVE this percentile to be HVN
    order:  int   = 2,
) -> np.ndarray:
    """Boolean mask — True = HVN bin (local max above pct percentile)."""
    if profile.sum() == 0:
        return np.zeros(len(profile), dtype=bool)
    s   = uniform_filter1d(profile, size=smooth)
    nz  = s[s > 0]
    if len(nz) == 0:
        return np.zeros(len(profile), dtype=bool)
    thr      = np.percentile(nz, pct)
    max_idx  = argrelmax(s, order=order)[0]
    max_idx  = max_idx[s[max_idx] > thr]
    hvn = np.zeros(len(profile), dtype=bool)
    hvn[max_idx] = True
    return hvn


def label_shelves_fast(lvn_mask: np.ndarray, min_run: int = 2) -> np.ndarray:
    result = np.zeros(len(lvn_mask), dtype=bool)
    i = 0
    while i < len(lvn_mask):
        if lvn_mask[i]:
            j = i
            while j < len(lvn_mask) and lvn_mask[j]:
                j += 1
            if (j - i) >= min_run:
                result[i:j] = True
            i = j
        else:
            i += 1
    return result


def stack_lvn_masks(
    profiles: list[np.ndarray],
    lvn_params: dict,
    min_stack: int,
    min_shelf_ticks: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stack LVN masks across all profile arrays.
    Returns: (stack_count, stacked_mask, shelf_mask, stacked_hvn_mask)
    HVN stack uses equal weighting over same profile set.
    """
    if not profiles:
        n = 1
        return np.zeros(n,dtype=np.int16), np.zeros(n,dtype=bool), np.zeros(n,dtype=bool), np.zeros(n,dtype=bool)

    n     = len(profiles[0])
    stack = np.zeros(n, dtype=np.int16)
    hvn_stack = np.zeros(n, dtype=np.int16)

    for p in profiles:
        if p.sum() == 0:
            continue
        stack     += detect_lvn_fast(p, **lvn_params).astype(np.int16)
        hvn_stack += detect_hvn_fast(p, smooth=lvn_params.get("smooth", 5)).astype(np.int16)

    stacked_mask = stack     >= min_stack
    hvn_mask     = hvn_stack >= max(1, min_stack // 2)   # HVN gate = half LVN gate
    shelf_mask   = label_shelves_fast(stacked_mask, min_run=min_shelf_ticks)

    return stack, stacked_mask, shelf_mask, hvn_mask


# ═══════════════════════════════════════════════════════════════════════════════
# POC / VAL / VAH
# ═══════════════════════════════════════════════════════════════════════════════

def compute_va(
    profile: np.ndarray,
    price_grid: np.ndarray,
    va_pct: float = VA_PCT_DEFAULT,
) -> dict:
    total = profile.sum()
    if total == 0:
        return {"poc": None, "val": None, "vah": None}
    poc_idx     = int(np.argmax(profile))
    lo, hi      = poc_idx, poc_idx
    accumulated = profile[poc_idx]
    target      = total * va_pct

    while accumulated < target:
        can_lo = lo > 0
        can_hi = hi < len(profile) - 1
        if not can_lo and not can_hi:
            break
        v_lo = profile[lo - 1] if can_lo else -1
        v_hi = profile[hi + 1] if can_hi else -1
        if v_hi >= v_lo:
            hi += 1; accumulated += profile[hi]
        else:
            lo -= 1; accumulated += profile[lo]

    return {
        "poc": float(price_grid[poc_idx]),
        "val": float(price_grid[lo]),
        "vah": float(price_grid[hi]),
    }


def build_va_cache(
    daily: dict[str, np.ndarray],
    price_grid: np.ndarray,
    va_pct: float,
) -> dict[str, dict]:
    return {d: compute_va(p, price_grid, va_pct) for d, p in daily.items()}


def stacked_va_level(
    va_cache: dict[str, dict],
    prior_dates: list[str],
    key: str,
    tolerance: float = 5.0,
) -> float | None:
    vals = [va_cache[d][key] for d in prior_dates
            if d in va_cache and va_cache[d][key] is not None]
    if not vals:
        return None
    arr     = np.array(vals)
    buckets: dict[float, list] = {}
    for v in arr:
        b = round(v / tolerance) * tolerance
        buckets.setdefault(b, []).append(v)
    best = max(buckets, key=lambda k: len(buckets[k]))
    return float(np.mean(buckets[best]))


def rolled_va(
    daily: dict[str, np.ndarray],
    prior_dates: list[str],
    price_grid: np.ndarray,
    va_pct: float,
) -> dict:
    if not prior_dates:
        return {"poc": None, "val": None, "vah": None}
    rp = sum(daily[d] for d in prior_dates if d in daily)
    if not isinstance(rp, np.ndarray):
        return {"poc": None, "val": None, "vah": None}
    return compute_va(rp, price_grid, va_pct)


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════════

def predict(
    stack:        np.ndarray,
    stacked_mask: np.ndarray,
    shelf_mask:   np.ndarray,
    hvn_mask:     np.ndarray,
    price_grid:   np.ndarray,
    ref_price:    float,
    strategy:     str,
    va_prior:     dict,
    va_rolled:    dict,
    va_stk_vah:   float | None,
    va_stk_val:   float | None,
    search_range: float = 300.0,
) -> tuple[float | None, float | None]:

    lo_b  = ref_price - search_range
    hi_b  = ref_price + search_range
    in_r  = (price_grid >= lo_b) & (price_grid <= hi_b)
    above = in_r & (price_grid > ref_price)
    below = in_r & (price_grid < ref_price)

    def first_above(mask) -> float | None:
        idx = np.where(mask)[0]
        return float(price_grid[idx[0]])  if len(idx) else None

    def first_below(mask) -> float | None:
        idx = np.where(mask)[0]
        return float(price_grid[idx[-1]]) if len(idx) else None

    def max_stack_in(mask) -> float | None:
        idx = np.where(mask)[0]
        if len(idx) == 0: return None
        return float(price_grid[idx[np.argmax(stack[idx])]])

    def nearest_to(target: float | None, mask) -> float | None:
        if target is None: return None
        idx = np.where(mask)[0]
        if len(idx) == 0: return None
        return float(price_grid[idx[np.argmin(np.abs(price_grid[idx] - target))]])

    def hvn_beyond_shelf_top() -> float | None:
        """First HVN above the topmost LVN shelf boundary above ref."""
        shelf_above = np.where(above & shelf_mask)[0]
        if len(shelf_above) == 0:
            return first_above(above & hvn_mask)
        shelf_top = shelf_above[-1]   # highest shelf bin above ref
        hvn_above_shelf = np.where(hvn_mask)[0]
        hvn_above_shelf = hvn_above_shelf[hvn_above_shelf > shelf_top]
        return float(price_grid[hvn_above_shelf[0]]) if len(hvn_above_shelf) else None

    def hvn_beyond_shelf_bot() -> float | None:
        """First HVN below the bottommost LVN shelf boundary below ref."""
        shelf_below = np.where(below & shelf_mask)[0]
        if len(shelf_below) == 0:
            return first_below(below & hvn_mask)
        shelf_bot = shelf_below[0]   # lowest shelf bin below ref
        hvn_below_shelf = np.where(hvn_mask)[0]
        hvn_below_shelf = hvn_below_shelf[hvn_below_shelf < shelf_bot]
        return float(price_grid[hvn_below_shelf[-1]]) if len(hvn_below_shelf) else None

    def va_gate(price: float | None, want_above: bool) -> float | None:
        if price is None: return None
        if want_above and price <= ref_price: return None
        if not want_above and price >= ref_price: return None
        return price

    # ── LVN strategies ──
    if strategy == "nearest_lvn":
        top = first_above(above & stacked_mask)
        bot = first_below(below & stacked_mask)

    elif strategy == "nearest_shelf":
        top = first_above(above & shelf_mask)
        bot = first_below(below & shelf_mask)

    elif strategy == "max_stack_lvn":
        top = max_stack_in(above & stacked_mask)
        bot = max_stack_in(below & stacked_mask)

    elif strategy == "max_stack_shelf":
        top = max_stack_in(above & shelf_mask)
        bot = max_stack_in(below & shelf_mask)

    elif strategy == "confluence_top25":
        if stacked_mask.any():
            thr = np.percentile(stack[stacked_mask], 75)
            top = first_above(above & (stack >= thr))
            bot = first_below(below & (stack >= thr))
        else:
            top = bot = None

    # ── HVN strategies ──
    elif strategy == "nearest_hvn":
        top = first_above(above & hvn_mask)
        bot = first_below(below & hvn_mask)

    elif strategy == "strongest_hvn":
        top = max_stack_in(above & hvn_mask)
        bot = max_stack_in(below & hvn_mask)

    elif strategy == "hvn_beyond_shelf":
        top = hvn_beyond_shelf_top()
        bot = hvn_beyond_shelf_bot()

    # ── VA strategies ──
    elif strategy == "prior_va":
        top = va_gate(va_prior.get("vah"), True)
        bot = va_gate(va_prior.get("val"), False)

    elif strategy == "stacked_va":
        top = va_gate(va_stk_vah, True)
        bot = va_gate(va_stk_val, False)

    elif strategy == "rolled_va":
        top = va_gate(va_rolled.get("vah"), True)
        bot = va_gate(va_rolled.get("val"), False)

    elif strategy == "poc_extension":
        poc = va_prior.get("poc")
        vah = va_prior.get("vah")
        val = va_prior.get("val")
        if poc and vah and val:
            half = (vah - val) / 2
            top  = va_gate(poc + half, True)
            bot  = va_gate(poc - half, False)
        else:
            top = bot = None

    # ── confluence strategies ──
    elif strategy == "lvn_near_vah":
        top = nearest_to(va_prior.get("vah"), above & stacked_mask)
        bot = nearest_to(va_prior.get("val"), below & stacked_mask)

    elif strategy == "lvn_near_stacked_va":
        top = nearest_to(va_stk_vah, above & stacked_mask)
        bot = nearest_to(va_stk_val, below & stacked_mask)

    elif strategy == "va_lvn_confluence":
        tol = int(5.0 / vp.TICK)
        def near_level(lvl: float | None, direction_mask) -> float | None:
            if lvl is None: return None
            li = int(np.argmin(np.abs(price_grid - lvl)))
            zone = np.zeros(len(price_grid), dtype=bool)
            zone[max(0,li-tol):min(len(price_grid)-1,li+tol)+1] = True
            r = first_above(direction_mask & stacked_mask & zone) if direction_mask is above \
                else first_below(direction_mask & stacked_mask & zone)
            return r or (first_above(direction_mask & stacked_mask) if direction_mask is above
                         else first_below(direction_mask & stacked_mask))
        top = near_level(va_prior.get("vah"), above)
        bot = near_level(va_prior.get("val"), below)

    else:
        top = bot = None

    return top, bot


# ═══════════════════════════════════════════════════════════════════════════════
# SCORING
# ═══════════════════════════════════════════════════════════════════════════════

def score_day(pred_top, pred_bot, actual_high, actual_low) -> dict:
    s = {}
    for side, pred, actual in [("top", pred_top, actual_high),
                                ("bot", pred_bot, actual_low)]:
        if pred is None:
            s[f"{side}_err"] = np.nan
            s[f"{side}_bias"] = np.nan
            for t in THRESHOLDS:
                s[f"{side}_hit{t}"] = np.nan
        else:
            err  = abs(pred - actual)
            bias = pred - actual
            s[f"{side}_err"]  = err
            s[f"{side}_bias"] = bias
            for t in THRESHOLDS:
                s[f"{side}_hit{t}"] = int(err <= t)
    return s


# ═══════════════════════════════════════════════════════════════════════════════
# SWEEP CONFIGS (3 stages)
# ═══════════════════════════════════════════════════════════════════════════════

ALL_STRATEGIES = [
    # LVN
    "nearest_lvn", "nearest_shelf", "max_stack_lvn", "max_stack_shelf", "confluence_top25",
    # HVN
    "nearest_hvn", "strongest_hvn", "hvn_beyond_shelf",
    # VA
    "prior_va", "stacked_va", "rolled_va", "poc_extension",
    # confluence
    "lvn_near_vah", "lvn_near_stacked_va", "va_lvn_confluence",
]

@dataclass
class Stage1Config:
    """Strategy horse race — everything else held at sensible defaults."""
    strategies:      list  = field(default_factory=lambda: ALL_STRATEGIES)
    lookbacks:       list  = field(default_factory=lambda: [10])
    min_stacks:      list  = field(default_factory=lambda: [3])
    sessions:        list  = field(default_factory=lambda: ["full"])
    anchored_combos: list  = field(default_factory=lambda: [["day","week","month"]])
    rolling_combos:  list  = field(default_factory=lambda: [[5,10,20]])
    lvn_params_list: list  = field(default_factory=lambda: [
        {"smooth":5, "pct":25.0, "depth_pct":60.0}])
    va_pcts:         list  = field(default_factory=lambda: [0.70])
    ref_types:       list  = field(default_factory=lambda: ["prior_close"])
    weightings:      list  = field(default_factory=lambda: ["equal"])
    min_shelf_ticks: list  = field(default_factory=lambda: [2])
    search_range:    float = 300.0
    va_tolerance:    float = 5.0


@dataclass
class Stage2Config:
    """
    Param tuning — top-5 strategies from stage 1, targeted param variation.
    Designed to run in ~10-20 min on 57 days of cached data.

    Key axes:
      lookback × session × anchoring × rolling × lvn_params × va_pct × ref_type × weighting
    Shelf/min_stack kept narrow since shelf strategies had 0 coverage in stage 1.
    """
    strategies:      list  = field(default_factory=lambda: [])  # filled after stage 1
    lookbacks:       list  = field(default_factory=lambda: [5, 10, 20])
    min_stacks:      list  = field(default_factory=lambda: [2, 3, 5])
    sessions:        list  = field(default_factory=lambda: ["rth", "full"])
    anchored_combos: list  = field(default_factory=lambda: [
        ["day"],
        ["day", "week", "month"],
        ["week", "month"],
    ])
    rolling_combos:  list  = field(default_factory=lambda: [
        [],
        [5, 10, 20],
    ])
    lvn_params_list: list  = field(default_factory=lambda: [
        {"smooth": 3, "pct": 20.0, "depth_pct": 60.0},
        {"smooth": 5, "pct": 25.0, "depth_pct": 60.0},
        {"smooth": 5, "pct": 25.0, "depth_pct": 70.0},
        {"smooth": 7, "pct": 30.0, "depth_pct": 60.0},
    ])
    va_pcts:         list  = field(default_factory=lambda: [0.68, 0.70, 0.75])
    ref_types:       list  = field(default_factory=lambda: ["prior_close", "today_open"])
    weightings:      list  = field(default_factory=lambda: ["equal", "recency"])
    min_shelf_ticks: list  = field(default_factory=lambda: [1, 2])
    search_range:    float = 300.0
    va_tolerance:    float = 5.0


@dataclass
class Stage3Config:
    """
    Fine grid — locked to Stage 2 winners to keep memory manageable.
    today_open + recency + full fixed (Stage 2 showed these dominate).
    Stack configs = 1×2×2×1×4×2 = 32 → ~50MB cache, no OOM.
    """
    strategies:      list  = field(default_factory=lambda: [])
    lookbacks:       list  = field(default_factory=lambda: [5, 10, 15, 20])
    min_stacks:      list  = field(default_factory=lambda: [2, 3, 4, 5, 6])
    sessions:        list  = field(default_factory=lambda: ["full"])      # full wins for HVN+today_open
    anchored_combos: list  = field(default_factory=lambda: [
        ["day"], ["day","week","month"]])
    rolling_combos:  list  = field(default_factory=lambda: [[]])          # rolling adds noise
    lvn_params_list: list  = field(default_factory=lambda: [])            # filled from stage 2
    va_pcts:         list  = field(default_factory=lambda: [0.70])
    ref_types:       list  = field(default_factory=lambda: ["today_open"]) # 2.5× winner
    weightings:      list  = field(default_factory=lambda: ["recency"])   # marginal winner
    min_shelf_ticks: list  = field(default_factory=lambda: [2])
    search_range:    float = 300.0
    va_tolerance:    float = 5.0


# ═══════════════════════════════════════════════════════════════════════════════
# SWEEP ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_sweep(cfg, label: str = "") -> pd.DataFrame:
    """
    Optimised sweep: pre-compute LVN stacks for every unique
    (session, anchored, rolling, lookback, lvn_params, date) combination,
    then fan-out cheaply across (strategy, min_stack, va_pct, ref_type,
    weighting, shelf_ticks).

    Stack computation is the bottleneck (scipy on 9886-bin arrays).
    Pre-caching reduces calls from N_combos×N_dates to
    N_stack_configs×N_dates — typically 50-100× speedup.
    """
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    price_grid = load_price_grid()

    session_profiles: dict[str, dict] = {}
    for sess in cfg.sessions:
        session_profiles[sess] = load_all_profiles(sess)

    ohlcv_data   = load_all_ohlcv("rth")
    sorted_dates = sorted(next(iter(session_profiles.values())).keys())

    wk_mo_cache: dict[str, tuple] = {}
    for sess, daily in session_profiles.items():
        wk_mo_cache[sess] = build_week_month_profiles(daily, sorted_dates)

    daily_stats = {d: daily_summary(ohlcv_data.get(d, pd.DataFrame()))
                   for d in sorted_dates}
    daily_stats = {d: s for d, s in daily_stats.items() if s}

    # ── VA caches (per session × va_pct) ──────────────────────────────────
    va_cache_store: dict[tuple, dict] = {}
    for sess in cfg.sessions:
        for vp_pct in cfg.va_pcts:
            key = (sess, vp_pct)
            va_cache_store[key] = build_va_cache(
                session_profiles[sess], price_grid, vp_pct
            )

    # ── pre-compute stacked arrays ────────────────────────────────────────
    # Key: (sess, anchored_key, rolling_key, lb, pi, weighting, date)
    # Value: (stack, stacked_mask, hvn_mask)  — shelf derived per shelf_ticks later
    #
    # weighting is included because recency-weighted profiles differ from equal.
    # shelf_ticks is cheap (label_shelves_fast is instant) so applied at prediction time.
    print("  Pre-computing LVN stacks…", flush=True)

    stack_cache: dict[tuple, tuple] = {}
    stack_configs = list(itertools.product(
        cfg.sessions,
        range(len(cfg.anchored_combos)),
        range(len(cfg.rolling_combos)),
        cfg.lookbacks,
        range(len(cfg.lvn_params_list)),
        cfg.weightings,
    ))
    n_stack = len(stack_configs)

    for si, (sess, ai, ri, lb, pi, weighting) in enumerate(stack_configs):
        anchored = cfg.anchored_combos[ai]
        rolling  = cfg.rolling_combos[ri]
        lvn_p    = cfg.lvn_params_list[pi]
        daily    = session_profiles[sess]
        wk_prof, mo_prof = wk_mo_cache[sess]

        if (si + 1) % max(1, n_stack // 10) == 0:
            print(f"    stacks {si+1}/{n_stack}  sess={sess} "
                  f"lb={lb} anch={'|'.join(anchored)} "
                  f"roll={rolling} pct={lvn_p['pct']} w={weighting}", flush=True)

        for i, test_date in enumerate(sorted_dates):
            if i < 2:
                continue
            prior = sorted_dates[max(0, i - lb): i]
            if len(prior) < 2 or test_date not in daily_stats:
                continue

            profile_list = aggregate_profiles(
                daily, prior, anchored, rolling, weighting, wk_prof, mo_prof
            )
            # Accumulate raw counts; don't apply gates here — gates vary by ms in fan-out
            n = len(profile_list[0]) if profile_list else 1
            stack_raw = np.zeros(n, dtype=np.int16)
            hvn_raw   = np.zeros(n, dtype=np.int16)
            for p in profile_list:
                if p.sum() == 0:
                    continue
                stack_raw += detect_lvn_fast(p, **lvn_p).astype(np.int16)
                hvn_raw   += detect_hvn_fast(p, smooth=lvn_p.get("smooth", 5)).astype(np.int16)

            cache_key = (sess, ai, ri, lb, pi, weighting, test_date)
            stack_cache[cache_key] = (stack_raw, hvn_raw)

    print(f"  {len(stack_cache):,} stacks cached.\n", flush=True)

    # ── pre-cache rolled_va per (sess, lb, date, va_pct) ─────────────────
    # rolled_va only depends on raw daily profiles + lookback, not on
    # anchored/rolling/lvn_params — pre-compute once to avoid 2.5M redundant calls.
    print("  Pre-computing rolled_va cache…", flush=True)
    rolled_va_cache: dict[tuple, dict] = {}
    for sess in cfg.sessions:
        daily = session_profiles[sess]
        for lb in cfg.lookbacks:
            for va_pct in cfg.va_pcts:
                for i, test_date in enumerate(sorted_dates):
                    if i < 2 or test_date not in daily_stats:
                        continue
                    prior = sorted_dates[max(0, i - lb): i]
                    if len(prior) < 2:
                        continue
                    rolled_va_cache[(sess, lb, test_date, va_pct)] = rolled_va(
                        daily, prior, price_grid, va_pct
                    )
    print(f"  {len(rolled_va_cache):,} rolled_va entries cached.\n", flush=True)

    # ── fan-out across all combos ─────────────────────────────────────────
    combos = list(itertools.product(
        cfg.strategies,
        cfg.lookbacks,
        cfg.min_stacks,
        cfg.sessions,
        range(len(cfg.anchored_combos)),
        range(len(cfg.rolling_combos)),
        range(len(cfg.lvn_params_list)),
        cfg.va_pcts,
        cfg.ref_types,
        cfg.weightings,
        cfg.min_shelf_ticks,
    ))
    total_combos = len(combos)
    print(f"  {total_combos:,} prediction combos × ~{len(daily_stats)} test days\n",
          flush=True)

    all_rows = []

    for ci, (strat, lb, ms, sess, ai, ri, pi, va_pct, ref_type, weighting, shelf_ticks) in enumerate(combos):
        anchored = cfg.anchored_combos[ai]
        rolling  = cfg.rolling_combos[ri]
        lvn_p    = cfg.lvn_params_list[pi]
        va_cache = va_cache_store[(sess, va_pct)]
        daily    = session_profiles[sess]

        if (ci + 1) % max(1, total_combos // 20) == 0:
            print(f"  [{ci+1:>6}/{total_combos}] {strat:<22} lb={lb} ms={ms} "
                  f"sess={sess} vp={va_pct} {ref_type}", flush=True)

        for i, test_date in enumerate(sorted_dates):
            if i < 2:
                continue
            prior = sorted_dates[max(0, i - lb): i]
            if len(prior) < 2 or test_date not in daily_stats:
                continue

            cache_key = (sess, ai, ri, lb, pi, weighting, test_date)
            if cache_key not in stack_cache:
                continue

            stack_raw, hvn_raw = stack_cache[cache_key]

            # apply ms gates + shelf label (instant — no scipy)
            stack        = stack_raw
            stacked_mask = stack >= ms
            hvn_mask     = hvn_raw >= max(1, ms // 2)
            shelf_mask   = label_shelves_fast(stacked_mask, min_run=shelf_ticks)

            ds          = daily_stats[test_date]
            prior_date  = sorted_dates[i - 1]
            prior_stats = daily_stats.get(prior_date, {})
            va_prior    = va_cache.get(prior_date, {})

            if ref_type == "prior_close":
                ref = prior_stats.get("close", ds["open"])
            elif ref_type == "prior_poc":
                ref = va_prior.get("poc") or prior_stats.get("close", ds["open"])
            else:
                ref = ds["open"]

            va_rolled_va = rolled_va_cache.get((sess, lb, test_date, va_pct),
                                               {"poc": None, "val": None, "vah": None})
            va_stk_vah   = stacked_va_level(va_cache, prior, "vah", cfg.va_tolerance)
            va_stk_val   = stacked_va_level(va_cache, prior, "val", cfg.va_tolerance)

            pred_top, pred_bot = predict(
                stack, stacked_mask, shelf_mask, hvn_mask,
                price_grid, ref, strat,
                va_prior, va_rolled_va, va_stk_vah, va_stk_val,
                cfg.search_range,
            )

            scores = score_day(pred_top, pred_bot, ds["high"], ds["low"])

            all_rows.append({
                "date":        test_date,
                "strategy":    strat,
                "lookback":    lb,
                "min_stack":   ms,
                "session":     sess,
                "anchored":    "+".join(sorted(anchored)),
                "rolling":     "+".join(str(w) for w in sorted(rolling)) or "none",
                "smooth":      lvn_p["smooth"],
                "pct":         lvn_p["pct"],
                "depth_pct":   lvn_p["depth_pct"],
                "va_pct":      va_pct,
                "ref_type":    ref_type,
                "weighting":   weighting,
                "shelf_ticks": shelf_ticks,
                "ref_price":   ref,
                "pred_top":    pred_top,
                "pred_bot":    pred_bot,
                "actual_high": ds["high"],
                "actual_low":  ds["low"],
                **scores,
            })

    return pd.DataFrame(all_rows)


# ═══════════════════════════════════════════════════════════════════════════════
# AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════════

GROUP_COLS = [
    "strategy","lookback","min_stack","session","anchored","rolling",
    "smooth","pct","depth_pct","va_pct","ref_type","weighting","shelf_ticks",
]

def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    hit_cols  = [c for c in df.columns if "_hit"  in c]
    err_cols  = [c for c in df.columns if "_err"  == c[-4:]]
    bias_cols = [c for c in df.columns if "_bias" in c]
    agg = {c: "mean" for c in hit_cols + err_cols + bias_cols}
    agg["pred_top"] = lambda x: x.notna().mean()
    agg["pred_bot"] = lambda x: x.notna().mean()
    agg["date"]     = "count"

    gcols = [c for c in GROUP_COLS if c in df.columns]
    s = df.groupby(gcols).agg(agg).reset_index()
    s.rename(columns={"date":"n_days","pred_top":"top_cov","pred_bot":"bot_cov"}, inplace=True)

    for t in THRESHOLDS:
        tc, bc = f"top_hit{t}", f"bot_hit{t}"
        if tc in s and bc in s:
            s[f"combined_hit{t}"] = (s[tc] + s[bc]) / 2

    if "top_err" in s and "bot_err" in s:
        s["combined_mae"] = (s["top_err"] + s["bot_err"]) / 2

    return s.sort_values("combined_hit10", ascending=False).reset_index(drop=True)


def top_n_strategies(summary: pd.DataFrame, n: int = 5) -> list[str]:
    best = (summary.groupby("strategy")["combined_hit10"].mean()
            .sort_values(ascending=False))
    return list(best.head(n).index)


def top_n_lvn_params(summary: pd.DataFrame, n: int = 3) -> list[dict]:
    cols = ["smooth","pct","depth_pct"]
    best = (summary.groupby(cols)["combined_hit10"].mean()
            .sort_values(ascending=False).head(n))
    return [{"smooth": int(r[0]), "pct": float(r[1]), "depth_pct": float(r[2])}
            for r in best.index]


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_strategy_horse_race(summary: pd.DataFrame, out_dir: str, stage_label: str = ""):
    import matplotlib.pyplot as plt

    # best params per strategy
    best = (summary.groupby("strategy")
            .apply(lambda x: x.nlargest(1, "combined_hit10"), include_groups=False)
            .reset_index(level=0).reset_index(drop=True))

    strats  = best["strategy"].tolist()
    hit10   = best["combined_hit10"].fillna(0).tolist()
    hit20   = best["combined_hit20"].fillna(0).tolist() if "combined_hit20" in best else [0]*len(strats)
    mae     = best["combined_mae"].fillna(999).tolist() if "combined_mae"  in best else [999]*len(strats)

    va_set  = {"prior_va","stacked_va","rolled_va","poc_extension",
               "lvn_near_vah","lvn_near_stacked_va","va_lvn_confluence"}
    hvn_set = {"nearest_hvn","strongest_hvn","hvn_beyond_shelf"}
    colors  = ["#cc00ff" if s in va_set else "#00ccaa" if s in hvn_set else "#0088ff"
               for s in strats]

    x = np.arange(len(strats))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=BG)
    fig.suptitle(f"Strategy horse race {stage_label}\n"
                 f"■ purple=VA  ■ teal=HVN  ■ blue=LVN",
                 color="#ccc", fontsize=10)

    for ax, vals, lbl, fmt in [
        (axes[0], hit10, "Hit rate @ 10pt",  ".2f"),
        (axes[1], hit20, "Hit rate @ 20pt",  ".2f"),
        (axes[2], mae,   "MAE (pts)",         ".1f"),
    ]:
        bars = ax.bar(x, vals, color=colors, width=0.7, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(strats, rotation=45, ha="right", fontsize=8, color="#aaa")
        ax.set_title(lbl, color="#cc88ff", fontsize=9)
        ax.set_facecolor("#0d0d0d")
        ax.tick_params(colors="#666", labelsize=7)
        ax.yaxis.grid(True, color="#1a1a1a", lw=0.5, zorder=0)
        for sp in ax.spines.values(): sp.set_edgecolor("#222")
        for bar, v in zip(bars, vals):
            if not np.isnan(v) and v > 0:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                        f"{v:{fmt}}", ha="center", va="bottom", fontsize=7, color="white")

    fig.patch.set_facecolor(BG)
    fig.tight_layout(rect=[0,0,1,0.88])
    fp = os.path.join(out_dir, f"s1_horse_race.png")
    fig.savefig(fp, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  → {fp}")


def plot_param_heatmap(summary: pd.DataFrame, out_dir: str, metric: str = "combined_hit10"):
    import matplotlib.pyplot as plt

    strats = summary["strategy"].unique()
    fig, axes = plt.subplots(
        1, len(strats),
        figsize=(max(14, len(strats)*3.5), 5),
        facecolor=BG,
    )
    if len(strats) == 1: axes = [axes]
    label = {"combined_hit10":"Hit@10", "combined_hit20":"Hit@20", "combined_mae":"MAE"}
    fig.suptitle(f"{label.get(metric, metric)} — lookback × min_stack per strategy",
                 color="#ccc", fontsize=10)

    cmap = "magma" if "hit" in metric else "magma_r"

    for ax, strat in zip(axes, strats):
        sub = summary[summary["strategy"] == strat]
        if sub.empty: ax.set_visible(False); continue
        pivot = sub.pivot_table(index="lookback", columns="min_stack",
                                values=metric, aggfunc="mean")
        vals = pivot.values
        vmin = np.nanmin(vals); vmax = np.nanmax(vals)
        im = ax.imshow(vals, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=7, color="#999")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=7, color="#999")
        ax.set_xlabel("min_stack", color="#666", fontsize=7)
        ax.set_ylabel("lookback",  color="#666", fontsize=7)
        ax.set_title(strat, color="#cc88ff", fontsize=8, pad=3)
        ax.set_facecolor("#111")
        for sp in ax.spines.values(): sp.set_edgecolor("#222")
        for ri in range(vals.shape[0]):
            for ci in range(vals.shape[1]):
                v = vals[ri, ci]
                if not np.isnan(v):
                    ax.text(ci, ri, f"{v:.2f}" if v < 5 else f"{v:.1f}",
                            ha="center", va="center", fontsize=6, color="white")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(colors="#666", labelsize=6)
        cb.outline.set_edgecolor("#333")

    fig.patch.set_facecolor(BG)
    fig.tight_layout(rect=[0,0,1,0.92])
    fp = os.path.join(out_dir, f"s2_heatmap_{metric}.png")
    fig.savefig(fp, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  → {fp}")


def plot_best_daily(df: pd.DataFrame, summary: pd.DataFrame, out_dir: str):
    import matplotlib.pyplot as plt

    best = summary.iloc[0]
    mask = True
    for col in ["strategy","lookback","min_stack","session","anchored","rolling","ref_type"]:
        if col in df.columns and col in best.index:
            mask = mask & (df[col] == best[col])
    sub = df[mask].dropna(subset=["pred_top","pred_bot"]).sort_values("date")
    if sub.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), facecolor=BG)
    title = (f"Best: {best.get('strategy','')}  lb={int(best.get('lookback',0))} "
             f"ms={int(best.get('min_stack',0))} sess={best.get('session','')} "
             f"anch={best.get('anchored','')} ref={best.get('ref_type','')}\n"
             f"hit@10={best.get('combined_hit10',0):.1%}  "
             f"hit@20={best.get('combined_hit20',0):.1%}  "
             f"MAE={best.get('combined_mae',0):.1f}pt  "
             f"n={int(best.get('n_days',0))} days")
    fig.suptitle(title, color="#ccc", fontsize=9)

    x = range(len(sub))
    for ax, (pc, ac, lbl, cp, ca) in zip(axes, [
        ("pred_top","actual_high","TOP",    "#ff00ff","#26a69a"),
        ("pred_bot","actual_low", "BOTTOM", "#ff00ff","#ef5350"),
    ]):
        ax.fill_between(x, sub[pc].values, sub[ac].values, alpha=0.10, color=cp)
        ax.plot(x, sub[ac].values, color=ca, lw=1.2, label=f"Actual {lbl}")
        ax.plot(x, sub[pc].values, color=cp, lw=0.9, ls="--", label=f"Predicted {lbl}")
        ax.set_facecolor("#0d0d0d")
        ax.tick_params(colors="#777", labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#222")
        ax.legend(fontsize=8, labelcolor="#ccc", facecolor="#111", edgecolor="#333")
        ax.yaxis.grid(True, color="#1a1a1a", lw=0.4)
        step = max(1, len(sub) // 12)
        ax.set_xticks(list(range(0, len(sub), step)))
        ax.set_xticklabels([sub["date"].iloc[ii] for ii in range(0, len(sub), step)],
                           rotation=30, ha="right", color="#777", fontsize=7)

    fig.patch.set_facecolor(BG)
    fig.tight_layout(rect=[0,0,1,0.91])
    fp = os.path.join(out_dir, "best_daily_predictions.png")
    fig.savefig(fp, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  → {fp}")


def plot_error_dist(df: pd.DataFrame, summary: pd.DataFrame, out_dir: str):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
    fig.suptitle("Error distribution — top 6 combos", color="#ccc", fontsize=10)
    colors = ["#ff00ff","#bb00dd","#8800aa","#00ccaa","#0088ff","#ff8800"]

    for ax, (ecol, title) in zip(axes, [
        ("top_err", "Top tick error (pts)"),
        ("bot_err", "Bottom tick error (pts)"),
    ]):
        for i, (_, row) in enumerate(summary.head(6).iterrows()):
            mask = (df["strategy"] == row["strategy"]) & (df["lookback"] == row["lookback"]) & \
                   (df["min_stack"] == row["min_stack"])
            errs = df[mask][ecol].dropna()
            if errs.empty: continue
            lbl = f"{row['strategy'][:12]} lb={int(row['lookback'])} ms={int(row['min_stack'])}"
            ax.hist(errs, bins=20, alpha=0.5, color=colors[i % len(colors)],
                    label=lbl, density=True)
        ax.set_facecolor("#0d0d0d")
        ax.tick_params(colors="#777", labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#222")
        ax.set_xlabel(title, color="#777", fontsize=8)
        ax.set_ylabel("Density", color="#777", fontsize=8)
        ax.legend(fontsize=6, labelcolor="#ccc", facecolor="#111", edgecolor="#333")
        for xv, c in [(10,"#ff4444"),(20,"#ff8800"),(30,"#ffcc00")]:
            ax.axvline(xv, color=c, lw=0.8, ls="--", alpha=0.6, label=f"{xv}pt")

    fig.patch.set_facecolor(BG)
    fig.tight_layout()
    fp = os.path.join(out_dir, "error_distribution.png")
    fig.savefig(fp, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  → {fp}")


def print_top(summary: pd.DataFrame, n: int = 15):
    cols = ["strategy","lookback","min_stack","session","anchored","rolling",
            "ref_type","weighting","combined_hit10","combined_hit20",
            "combined_mae","top_cov","bot_cov","n_days"]
    disp = [c for c in cols if c in summary.columns]
    print(f"\n{'─'*120}")
    print(f"Top {n} combos (hit@10):")
    print(summary[disp].head(n).to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="1", choices=["1","2","3","all"])
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args()

    stages_to_run = ["1","2","3"] if args.stage == "all" else [args.stage]

    s1_csv = os.path.join(RESULTS_DIR, "stage1_results.csv")
    s2_csv = os.path.join(RESULTS_DIR, "stage2_results.csv")
    s3_csv = os.path.join(RESULTS_DIR, "stage3_results.csv")
    s1_sum_csv = os.path.join(RESULTS_DIR, "stage1_summary.csv")
    s2_sum_csv = os.path.join(RESULTS_DIR, "stage2_summary.csv")
    s3_sum_csv = os.path.join(RESULTS_DIR, "stage3_summary.csv")

    s1_sum = s2_sum = s3_sum = None
    s1_df  = s2_df  = s3_df  = None

    # pre-load prior summaries so stage 3 can inherit from stage 2 even standalone
    if os.path.exists(s1_sum_csv):
        try: s1_sum = pd.read_csv(s1_sum_csv)
        except Exception: pass
    if os.path.exists(s2_sum_csv):
        try: s2_sum = pd.read_csv(s2_sum_csv)
        except Exception: pass

    # ── Stage 1: horse race ──
    if "1" in stages_to_run:
        if args.plot_only and os.path.exists(s1_csv):
            s1_df  = pd.read_csv(s1_csv)
            s1_sum = aggregate(s1_df)
        else:
            cfg   = Stage1Config()
            s1_df = run_sweep(cfg, "STAGE 1 — Strategy horse race")
            s1_df.to_csv(s1_csv, index=False)
            s1_sum = aggregate(s1_df)
            s1_sum.to_csv(s1_sum_csv, index=False)

        print_top(s1_sum)
        print("\nPlotting Stage 1…")
        plot_strategy_horse_race(s1_sum, RESULTS_DIR, "(stage 1)")
        if not s1_df.empty:
            plot_best_daily(s1_df, s1_sum, RESULTS_DIR)
            plot_error_dist(s1_df, s1_sum, RESULTS_DIR)

    # ── Stage 2: param tuning ──
    if "2" in stages_to_run:
        # pick top-5 strategies from stage 1
        top_strats = []
        if s1_sum is not None:
            top_strats = top_n_strategies(s1_sum, n=5)
        elif os.path.exists(s1_csv):
            top_strats = top_n_strategies(aggregate(pd.read_csv(s1_csv)), n=5)

        if not top_strats:
            print("Stage 1 results not found — running all strategies in stage 2.")
            top_strats = ALL_STRATEGIES

        print(f"\nStage 2 strategies: {top_strats}")
        cfg = Stage2Config()
        cfg.strategies = top_strats

        if args.plot_only and os.path.exists(s2_csv):
            s2_df  = pd.read_csv(s2_csv)
            s2_sum = aggregate(s2_df)
        else:
            s2_df  = run_sweep(cfg, "STAGE 2 — Param tuning")
            s2_df.to_csv(s2_csv, index=False)
            s2_sum = aggregate(s2_df)
            s2_sum.to_csv(s2_sum_csv, index=False)

        print_top(s2_sum)
        print("\nPlotting Stage 2…")
        for metric in ["combined_hit10", "combined_hit20", "combined_mae"]:
            plot_param_heatmap(s2_sum, RESULTS_DIR, metric)
        if not s2_df.empty:
            plot_best_daily(s2_df, s2_sum, RESULTS_DIR)

    # ── Stage 3: full grid ──
    if "3" in stages_to_run:
        top_strats = []
        best_params = []
        # prefer stage 2 strategies; fall back to stage 1
        if s2_sum is not None:
            top_strats  = top_n_strategies(s2_sum, n=3)
        elif s1_sum is not None:
            top_strats  = top_n_strategies(s1_sum, n=3)
        if s2_sum is not None:
            best_params = top_n_lvn_params(s2_sum, n=3)
        if not top_strats:
            top_strats  = ALL_STRATEGIES[:5]
        if not best_params:
            best_params = [{"smooth":5,"pct":25.0,"depth_pct":60.0}]

        print(f"\nStage 3 strategies: {top_strats}")
        print(f"Stage 3 LVN params: {best_params}")
        cfg = Stage3Config()
        cfg.strategies      = top_strats
        cfg.lvn_params_list = best_params

        if args.plot_only and os.path.exists(s3_csv):
            s3_df  = pd.read_csv(s3_csv)
            s3_sum = aggregate(s3_df)
        else:
            s3_df  = run_sweep(cfg, "STAGE 3 — Full grid")
            s3_df.to_csv(s3_csv, index=False)
            s3_sum = aggregate(s3_df)
            s3_sum.to_csv(s3_sum_csv, index=False)

        print_top(s3_sum, n=20)
        print("\nPlotting Stage 3…")
        for metric in ["combined_hit10", "combined_hit20", "combined_mae"]:
            plot_param_heatmap(s3_sum, RESULTS_DIR, metric)
        if not s3_df.empty:
            plot_best_daily(s3_df, s3_sum, RESULTS_DIR)

    print("\nDone → results/")


if __name__ == "__main__":
    main()
