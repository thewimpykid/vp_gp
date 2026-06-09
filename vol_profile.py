"""
vol_profile.py
--------------
Volume-profile LVN stacking utility for NQ futures tick data.

Data: daily parquet files under vp_gp/**/YYYYMMDD.parquet
      Columns: raw_timestamp (str), msg_type (int8), book_level (int8),
               price (float32), size (int32)

Trade extraction: msg_type==2, book_level==0, size>0
  - msg_type=1 book_level=0 = VP snapshots (cumulative, skip to avoid double-count)
  - msg_type=2 book_level=0 = incremental traded-volume updates (use these)
  - book_level>=1 = resting order-book depth (not traded volume)

Entry point: run() — returns (stacked_lvns_df, heatmap_array, price_bins)
Do NOT run this file directly; import and call run().
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import argrelmin

warnings.filterwarnings("ignore", category=FutureWarning)

# ── constants ────────────────────────────────────────────────────────────────
TICK = 0.25          # NQ minimum tick
TRADE_MSG  = 2       # incremental update
TRADE_LVL  = 0       # book_level=0 = trade event

# LVN detection defaults
LVN_SMOOTH    = 5    # bins to smooth profile before finding minima (1.25 pts)
LVN_PCT       = 25   # volume must be below this percentile of session to qualify
LVN_DEPTH_PCT = 60   # valley must be < 60% of average of its two flanking peaks

# Stacking
MIN_STACK_DEFAULT = 3   # levels agreed by this many profiles = "stacked LVN"

# Rolling windows (trading days)
ROLL_WINDOWS = [5, 10, 20]

# ── file discovery ────────────────────────────────────────────────────────────

def _discover_files(base_dir: str) -> dict[str, str]:
    """
    Scan all subdirs of base_dir for YYYYMMDD.parquet files.
    Returns {date_str: filepath}. If a date appears in multiple part-folders,
    last one wins (parts are complementary, same schema).
    """
    mapping: dict[str, str] = {}
    for fp in glob.glob(os.path.join(base_dir, "**", "*.parquet"), recursive=True):
        fname = os.path.splitext(os.path.basename(fp))[0]
        if fname.isdigit() and len(fname) == 8:
            mapping[fname] = fp
    return dict(sorted(mapping.items()))


def _price_grid(lo: float, hi: float) -> np.ndarray:
    """Return array of price levels on TICK grid spanning [lo, hi]."""
    lo_snapped = np.floor(lo / TICK) * TICK
    hi_snapped = np.ceil(hi  / TICK) * TICK
    return np.arange(lo_snapped, hi_snapped + TICK, TICK)


# ── load ─────────────────────────────────────────────────────────────────────

def load_trades(filepath: str) -> pd.DataFrame:
    """
    Load one day's parquet and return only trade rows.
    Parses raw_timestamp to a proper datetime (UTC, ns precision).
    """
    df = pd.read_parquet(filepath)
    trades = df[(df["msg_type"] == TRADE_MSG) &
                (df["book_level"] == TRADE_LVL) &
                (df["size"] > 0)].copy()

    # raw_timestamp format: YYYYMMDDHHMMSS + sub-second digits (variable length)
    # Truncate to 14 chars (YYYYMMDDHHMMSS) for pandas parsing
    trades["ts"] = pd.to_datetime(
        trades["raw_timestamp"].astype(str).str[:14],
        format="%Y%m%d%H%M%S",
        utc=True,
    )
    trades.drop(columns=["raw_timestamp", "msg_type", "book_level"], inplace=True)
    trades.set_index("ts", inplace=True)
    return trades[["price", "size"]]


def load_range(date_list: list[str], base_dir: str) -> dict[str, pd.DataFrame]:
    """
    Load trades for each date in date_list.
    Returns {date_str: trades_df}.  Missing files are skipped silently.
    """
    files = _discover_files(base_dir)
    result = {}
    for d in date_list:
        if d in files:
            result[d] = load_trades(files[d])
        else:
            pass  # weekend / holiday
    return result


# ── volume profile ────────────────────────────────────────────────────────────

def build_profile(
    trades: pd.DataFrame,
    price_bins: np.ndarray,
) -> np.ndarray:
    """
    Aggregate traded size into a histogram over price_bins.
    Returns array of same length as price_bins.
    """
    if trades.empty:
        return np.zeros(len(price_bins))

    idx = np.round((trades["price"].values - price_bins[0]) / TICK).astype(int)
    in_range = (idx >= 0) & (idx < len(price_bins))
    profile = np.zeros(len(price_bins))
    np.add.at(profile, idx[in_range], trades.loc[in_range.values if hasattr(in_range, 'values') else in_range, "size"].values)
    return profile


def _build_profile_from_concat(frames: list[pd.DataFrame], price_bins: np.ndarray) -> np.ndarray:
    if not frames:
        return np.zeros(len(price_bins))
    combined = pd.concat(frames)
    return build_profile(combined, price_bins)


# ── LVN detection ─────────────────────────────────────────────────────────────

def detect_lvn(
    profile: np.ndarray,
    price_bins: np.ndarray,
    smooth: int = LVN_SMOOTH,
    pct: float = LVN_PCT,
    depth_pct: float = LVN_DEPTH_PCT,
    min_width_ticks: int = 1,
) -> np.ndarray:
    """
    Find Low Volume Nodes in a single profile.

    Method:
      1. Smooth with uniform filter (reduce single-tick noise).
      2. Find local minima (argrelmin).
      3. Keep only minima below `pct` percentile of session volume.
      4. Keep only minima whose value < depth_pct% of the mean of their
         two nearest flanking peaks (ensures it's a real valley, not flat tail).
      5. Expand each minimum ± min_width_ticks to form a shelf.

    Returns boolean array (True = LVN bin).
    """
    if profile.sum() == 0:
        return np.zeros(len(price_bins), dtype=bool)

    smoothed = uniform_filter1d(profile.astype(float), size=smooth)
    threshold = np.percentile(smoothed[smoothed > 0], pct)

    # local minima (order=2 means checks 2 neighbours each side)
    mins_idx = argrelmin(smoothed, order=2)[0]
    mins_idx = mins_idx[smoothed[mins_idx] < threshold]

    # depth check: valley < depth_pct% of flanking peaks
    peaks_idx = _flanking_peaks(smoothed)
    qualified = []
    for mi in mins_idx:
        left_pk  = _nearest_peak(peaks_idx, mi, direction="left",  profile=smoothed)
        right_pk = _nearest_peak(peaks_idx, mi, direction="right", profile=smoothed)
        if left_pk is not None and right_pk is not None:
            avg_flank = (smoothed[left_pk] + smoothed[right_pk]) / 2
            if avg_flank > 0 and smoothed[mi] < (depth_pct / 100) * avg_flank:
                qualified.append(mi)
        elif left_pk is not None or right_pk is not None:
            # edge of profile — relax to percentile-only gate
            qualified.append(mi)

    lvn = np.zeros(len(price_bins), dtype=bool)
    for mi in qualified:
        lo = max(0, mi - min_width_ticks)
        hi = min(len(price_bins) - 1, mi + min_width_ticks)
        lvn[lo:hi + 1] = True

    return lvn


def _flanking_peaks(smoothed: np.ndarray) -> np.ndarray:
    from scipy.signal import argrelmax
    return argrelmax(smoothed, order=2)[0]


def _nearest_peak(peaks: np.ndarray, idx: int, direction: str, profile: np.ndarray):
    if len(peaks) == 0:
        return None
    if direction == "left":
        cands = peaks[peaks < idx]
        return int(cands[-1]) if len(cands) else None
    else:
        cands = peaks[peaks > idx]
        return int(cands[0]) if len(cands) else None


# ── profile factories ─────────────────────────────────────────────────────────

def build_anchored_profiles(
    daily: dict[str, pd.DataFrame],
    price_bins: np.ndarray,
    modes: list[str] = ("day", "week", "month"),
) -> dict[str, np.ndarray]:
    """
    Build volume profiles anchored to the start of each day / week / month.

    'day'   → one profile per trading day (fresh each session)
    'week'  → cumulative from Monday of the week through each Friday
    'month' → cumulative from the 1st trading day of the month

    Returns dict keyed like "day_20251006", "week_2025W41", "month_202510".
    Each value is a 1-D volume array over price_bins.
    """
    dates = sorted(daily.keys())
    profiles: dict[str, np.ndarray] = {}

    if "day" in modes:
        for d in dates:
            profiles[f"day_{d}"] = build_profile(daily[d], price_bins)

    if "week" in modes:
        # group by ISO week
        week_map: dict[str, list[str]] = {}
        for d in dates:
            dt = pd.Timestamp(d)
            wk = dt.strftime("%GW%V")   # ISO week e.g. "2025W41"
            week_map.setdefault(wk, []).append(d)
        for wk, ds in week_map.items():
            frames = [daily[d] for d in sorted(ds)]
            profiles[f"week_{wk}"] = _build_profile_from_concat(frames, price_bins)

    if "month" in modes:
        month_map: dict[str, list[str]] = {}
        for d in dates:
            mo = d[:6]   # "202510"
            month_map.setdefault(mo, []).append(d)
        for mo, ds in month_map.items():
            frames = [daily[d] for d in sorted(ds)]
            profiles[f"month_{mo}"] = _build_profile_from_concat(frames, price_bins)

    return profiles


def build_rolling_profiles(
    daily: dict[str, pd.DataFrame],
    price_bins: np.ndarray,
    windows: list[int] = ROLL_WINDOWS,
) -> dict[str, np.ndarray]:
    """
    Build rolling N-day cumulative volume profiles ending on each trading day.

    For window=5: the profile for day D = sum of trades from (D-4) through D.
    Produces one profile per (window, end_date) pair.
    Keys: "roll5_20251010", "roll10_20251010", "roll20_20251010"
    """
    dates = sorted(daily.keys())
    profiles: dict[str, np.ndarray] = {}

    for w in windows:
        for i, d in enumerate(dates):
            if i < w - 1:
                continue   # not enough history yet
            window_dates = dates[i - w + 1: i + 1]
            frames = [daily[dd] for dd in window_dates]
            profiles[f"roll{w}_{d}"] = _build_profile_from_concat(frames, price_bins)

    return profiles


# ── stacking ──────────────────────────────────────────────────────────────────

def stack_lvns(
    all_profiles: dict[str, np.ndarray],
    price_bins: np.ndarray,
    lvn_kwargs: dict | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    For every profile, detect LVNs, then count how many profiles flag each
    price bin as an LVN.

    Returns:
      stacked_df  — DataFrame with columns:
                    price | stack_count | profiles (list of profile names) | is_shelf
        'is_shelf': contiguous run of >=2 stacked-LVN bins (ledge/shelf structure)
      heatmap     — 2-D bool array shape (n_price_bins, n_profiles)
                    rows=price_bins, cols=sorted(all_profiles.keys())
    """
    kwargs = lvn_kwargs or {}
    profile_names = sorted(all_profiles.keys())
    n_bins   = len(price_bins)
    n_prof   = len(profile_names)

    heatmap = np.zeros((n_bins, n_prof), dtype=bool)
    for j, name in enumerate(profile_names):
        heatmap[:, j] = detect_lvn(all_profiles[name], price_bins, **kwargs)

    stack_count = heatmap.sum(axis=1)                          # per-bin count
    sources = [
        [profile_names[j] for j in np.where(heatmap[i])[0]]
        for i in range(n_bins)
    ]

    df = pd.DataFrame({
        "price":       price_bins,
        "stack_count": stack_count,
        "profiles":    sources,
    })

    # mark contiguous shelf runs (>=2 adjacent stacked bins)
    df["is_shelf"] = _label_shelves(stack_count > 0)
    return df, heatmap


def _label_shelves(lvn_mask: np.ndarray, min_run: int = 2) -> np.ndarray:
    """True for every bin that belongs to a contiguous LVN run of length >= min_run."""
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


# ── top-level entry point ────────────────────────────────────────────────────

def run(
    base_dir: str | None = None,
    start: str = "2025-10-01",
    end:   str = "2025-11-30",
    anchored_modes: list[str] = ("day", "week", "month"),
    rolling_windows: list[int] = ROLL_WINDOWS,
    min_stack: int = MIN_STACK_DEFAULT,
    lvn_smooth:    int   = LVN_SMOOTH,
    lvn_pct:       float = LVN_PCT,
    lvn_depth_pct: float = LVN_DEPTH_PCT,
    price_lo: float | None = None,
    price_hi: float | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Full pipeline.

    Parameters
    ----------
    base_dir        Root dir containing the month/part subfolders.
                    Defaults to the directory of this script.
    start / end     Date range to analyse (inclusive). Format: 'YYYY-MM-DD'.
    anchored_modes  Subset of ('day', 'week', 'month').
    rolling_windows List of rolling-window sizes in trading days.
    min_stack       Minimum profile agreement to include in output.
    lvn_smooth      Smoothing kernel width (bins = ticks).
    lvn_pct         Volume percentile ceiling for LVN qualification.
    lvn_depth_pct   Max valley/flank-peak ratio to keep as genuine valley.
    price_lo/hi     Override price range (auto-detected from data if None).

    Returns
    -------
    stacked_df      DataFrame of all price bins with stack_count >= min_stack,
                    sorted descending. Columns: price, stack_count, profiles, is_shelf.
    heatmap         (n_bins, n_profiles) bool array.
    price_bins      1-D float array of price levels corresponding to heatmap rows.
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    # ── discover dates in range ──
    all_files = _discover_files(base_dir)
    start_str = start.replace("-", "")
    end_str   = end.replace("-",   "")
    dates_in_range = [d for d in sorted(all_files) if start_str <= d <= end_str]

    if not dates_in_range:
        raise ValueError(f"No parquet files found between {start} and {end} in {base_dir}")

    print(f"[vol_profile] Loading {len(dates_in_range)} trading days …")
    daily = load_range(dates_in_range, base_dir)
    loaded_dates = sorted(daily.keys())
    print(f"[vol_profile] Loaded {len(loaded_dates)} days: {loaded_dates[0]} → {loaded_dates[-1]}")

    # ── global price grid ──
    all_prices = pd.concat([df["price"] for df in daily.values()])
    lo = price_lo if price_lo is not None else float(all_prices.min())
    hi = price_hi if price_hi is not None else float(all_prices.max())
    price_bins = _price_grid(lo, hi)
    print(f"[vol_profile] Price grid: {lo:.2f} – {hi:.2f}  ({len(price_bins)} bins, tick={TICK})")

    # ── build profiles ──
    print(f"[vol_profile] Building anchored profiles: {anchored_modes}")
    anch = build_anchored_profiles(daily, price_bins, modes=anchored_modes)

    print(f"[vol_profile] Building rolling profiles: windows={rolling_windows}")
    roll = build_rolling_profiles(daily, price_bins, windows=rolling_windows)

    all_profiles = {**anch, **roll}
    print(f"[vol_profile] Total profiles: {len(all_profiles)}")

    # ── detect + stack LVNs ──
    lvn_kw = dict(smooth=lvn_smooth, pct=lvn_pct, depth_pct=lvn_depth_pct)
    print("[vol_profile] Detecting LVNs and stacking …")
    stacked_df, heatmap = stack_lvns(all_profiles, price_bins, lvn_kwargs=lvn_kw)

    # ── filter to min_stack ──
    out = (
        stacked_df[stacked_df["stack_count"] >= min_stack]
        .sort_values("stack_count", ascending=False)
        .reset_index(drop=True)
    )
    print(f"[vol_profile] Stacked LVNs (>= {min_stack} profiles): {len(out)} bins")
    print(f"[vol_profile]   of which shelves (contiguous runs): "
          f"{out['is_shelf'].sum()} bins, "
          f"{_count_runs(out['is_shelf'].values)} distinct shelves")

    return out, heatmap, price_bins


def _count_runs(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    return int(np.diff(np.concatenate([[False], mask, [False]]).astype(int)).clip(0).sum())


# ── convenience: summary table ────────────────────────────────────────────────

def summarise_shelves(stacked_df: pd.DataFrame, min_stack: int = MIN_STACK_DEFAULT) -> pd.DataFrame:
    """
    Collapse contiguous stacked-LVN runs into shelf records.

    Returns DataFrame: shelf_lo | shelf_hi | width_pts | max_stack | mean_stack | profiles
    """
    df = stacked_df[stacked_df["stack_count"] >= min_stack].copy()
    if df.empty:
        return pd.DataFrame()

    prices  = df["price"].values
    counts  = df["stack_count"].values
    shelves = []
    i = 0
    while i < len(prices):
        j = i
        # group prices within 1 tick of each other (contiguous on grid)
        while j + 1 < len(prices) and prices[j + 1] - prices[j] <= TICK * 1.5:
            j += 1
        run_prices = prices[i:j + 1]
        run_counts = counts[i:j + 1]
        all_profs  = set()
        for row in df.iloc[i:j + 1]["profiles"]:
            all_profs.update(row)
        shelves.append({
            "shelf_lo":   float(run_prices[0]),
            "shelf_hi":   float(run_prices[-1]),
            "width_pts":  float(run_prices[-1] - run_prices[0]),
            "max_stack":  int(run_counts.max()),
            "mean_stack": float(run_counts.mean()),
            "n_profiles": len(all_profs),
            "profiles":   sorted(all_profs),
        })
        i = j + 1

    result = pd.DataFrame(shelves).sort_values("max_stack", ascending=False).reset_index(drop=True)
    return result
