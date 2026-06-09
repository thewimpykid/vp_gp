"""
chart.py — NQ HVN zone heatmap + 1-min candlestick viewer

Shows:
  - Horizontal magenta bands = High Volume Nodes (where price stalls/reverses)
  - Black gaps between = Low Volume Nodes (price moves through fast)
  - Band intensity = how strongly that zone is stacked across prior sessions
  - Optional predicted high / low lines from the backtest model

Usage:
    python chart.py --date 2025-11-13
    python chart.py --date 2025-11-13 --start 09:30 --end 16:00
    python chart.py --date 2025-11-13 --lookback 10 --min-stack 2 --session full
    python chart.py --date 2025-11-13 --pred-high 21500 --pred-low 21200
"""

import argparse
import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.ticker
from matplotlib.collections import PatchCollection
from scipy.ndimage import uniform_filter1d
from zoneinfo import ZoneInfo

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vol_profile as vp

ET  = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR   = os.path.join(BASE_DIR, "cache")
PROFILE_DIR = os.path.join(CACHE_DIR, "profiles")
TRADES_DIR  = os.path.join(CACHE_DIR, "trades")
OHLCV_DIR   = os.path.join(CACHE_DIR, "ohlcv")
GRID_FILE   = os.path.join(CACHE_DIR, "price_grid.npy")

# ── colour palette ─────────────────────────────────────────────────────────────
BG_COLOR    = "#000000"
PANEL_COLOR = "#050505"
GRID_COLOR  = "#111111"
TEXT_COLOR  = "#666666"
UP_COLOR    = "#ffffff"      # white candles
DOWN_COLOR  = "#aaaaaa"      # grey candles
WICK_COLOR  = "#555555"

HVN_DIM  = np.array(mcolors.to_rgb("#3a0060"))   # dark purple
HVN_BRIGHT = np.array(mcolors.to_rgb("#cc00cc")) # bright magenta

# continuous confluence colormap: black (LVN gap) → purple → bright magenta (max confluence)
HEAT_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "hvn_heat",
    [(0.00, "#000000"),
     (0.15, "#0d0015"),
     (0.35, "#2a0045"),
     (0.55, "#4a0075"),
     (0.75, "#8800aa"),
     (1.00, "#dd22dd")],
)

PRED_HIGH_COLOR = "#00ffcc"   # cyan-green line = predicted top
PRED_LOW_COLOR  = "#ff6600"   # orange line = predicted bottom


def _hvn_color(intensity: float) -> tuple:
    """intensity ∈ [0,1] → RGBA magenta."""
    rgb = HVN_DIM + intensity * (HVN_BRIGHT - HVN_DIM)
    alpha = 0.55 + 0.40 * intensity
    return (*rgb, alpha)


# ── cache loaders ──────────────────────────────────────────────────────────────

def _load_price_grid() -> np.ndarray | None:
    if os.path.exists(GRID_FILE):
        return np.load(GRID_FILE).astype(np.float64)
    return None


def _load_cached_profile(date_str: str, session: str) -> np.ndarray | None:
    fp = os.path.join(PROFILE_DIR, f"{date_str}_{session}.npy")
    return np.load(fp).astype(np.float64) if os.path.exists(fp) else None


def _load_cached_ohlcv(date_str: str) -> pd.DataFrame | None:
    fp = os.path.join(OHLCV_DIR, f"{date_str}.parquet")
    if not os.path.exists(fp):
        return None
    df = pd.read_parquet(fp)
    return df


def _load_cached_trades(date_str: str) -> pd.DataFrame | None:
    fp = os.path.join(TRADES_DIR, f"{date_str}.parquet")
    if not os.path.exists(fp):
        return None
    return pd.read_parquet(fp)


# ── HVN zone detection ─────────────────────────────────────────────────────────

def build_stacked_profile(
    prior_dates: list[str],
    session: str,
    weighting: str = "recency",
) -> np.ndarray | None:
    """Sum prior-session profiles into one stacked profile array."""
    arrays = []
    for d in prior_dates:
        p = _load_cached_profile(d, session)
        if p is not None:
            arrays.append(p)
    if not arrays:
        return None
    if weighting == "recency":
        weights = np.linspace(0.5, 1.0, len(arrays))
        stacked = sum(w * a for w, a in zip(weights, arrays))
    else:
        stacked = sum(arrays)
    return stacked


def detect_hvn_zones(
    profile: np.ndarray,
    price_grid: np.ndarray,
    smooth: int = 5,
    pct: float = 60.0,
) -> list[tuple]:
    """
    Find contiguous HVN regions in profile.
    Returns list of (price_lo, price_hi, intensity) where intensity ∈ [0,1].
    """
    if profile is None or profile.sum() == 0:
        return []

    smoothed = uniform_filter1d(profile.astype(float), size=smooth)
    nonzero = smoothed[smoothed > 0]
    if len(nonzero) == 0:
        return []

    threshold = np.percentile(nonzero, pct)
    hvn_mask = smoothed >= threshold
    max_val = smoothed.max()

    zones = []
    in_zone = False
    zone_start = 0
    for i in range(len(hvn_mask)):
        if hvn_mask[i] and not in_zone:
            in_zone = True
            zone_start = i
        elif not hvn_mask[i] and in_zone:
            in_zone = False
            intensity = float(smoothed[zone_start:i].mean()) / max_val
            lo = price_grid[zone_start] - vp.TICK / 2
            hi = price_grid[i - 1]      + vp.TICK / 2
            zones.append((lo, hi, intensity))
    if in_zone:
        intensity = float(smoothed[zone_start:].mean()) / max_val
        lo = price_grid[zone_start] - vp.TICK / 2
        hi = price_grid[-1]         + vp.TICK / 2
        zones.append((lo, hi, intensity))

    return zones


def build_confluence_heatmap(
    date_nodash: str,
    session:  str   = "full",
    lookback: int   = 15,
    smooth:   int   = 5,
    hvn_pct:  float = 70.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Per-bin HVN confluence intensity in [0,1].

    Stage-3 winner config: anchored day+week+month profiles from prior
    `lookback` sessions, recency-weighted. Each profile votes: bins whose
    smoothed volume >= hvn_pct percentile are that profile's HVN region.
    Intensity = weighted fraction of profiles voting HVN for the bin.
    Black gaps (intensity 0) = LVN areas no profile flags.

    Returns (price_grid, intensity) or None if insufficient prior data.
    """
    import backtest as bt
    daily = bt.load_all_profiles(session)
    sorted_dates = sorted(daily.keys())
    prior = [d for d in sorted_dates if d < date_nodash][-lookback:]
    if len(prior) < 1:
        return None

    wk, mo = bt.build_week_month_profiles(daily, prior)   # prior-only, no leak
    profiles = bt.aggregate_profiles(
        daily, prior, ["day", "week", "month"], [], "recency", wk, mo)
    if not profiles:
        return None

    n = len(profiles[0])
    votes  = np.zeros(n, dtype=np.float64)
    w_sum  = 0.0
    for p in profiles:
        tot = p.sum()
        if tot == 0:
            continue
        s  = uniform_filter1d(p, size=smooth)
        nz = s[s > 0]
        if len(nz) == 0:
            continue
        thr = np.percentile(nz, hvn_pct)
        # graded vote: 0 below threshold, ramps with volume above it
        v = np.clip((s - thr) / max(s.max() - thr, 1e-9), 0, 1)
        v[s < thr] = 0.0
        votes += np.sqrt(v)        # sqrt → wide bands like reference image
        w_sum += 1.0

    if w_sum == 0:
        return None
    intensity = votes / w_sum
    intensity = uniform_filter1d(intensity, size=3)        # soften band edges
    if intensity.max() > 0:
        intensity = intensity / intensity.max()

    price_grid = _load_price_grid()
    return price_grid, intensity


def build_level_confluence_heatmap(
    date_nodash: str,
    session:  str   = "full",
    lookback: int   = 15,
    band_pts: float = 6.0,
    va_pct:   float = 0.70,
    hvn_pct:  float = 70.0,
    smooth:   int   = 5,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Stacked-level confluence heatmap.

    Each prior profile (every prior day, recency-weighted, plus week and
    month aggregates) contributes discrete levels:
        POC, VAH, VAL  (value area at va_pct)
        HVN peaks      (local maxima above hvn_pct percentile)
    Every level paints a flat band of ±band_pts around itself.
    Intensity = stacked vote weight per bin → crisp bands, brighter where
    many independent profiles agree. Bins no level touches stay black (LVN
    gaps / low-confluence air).

    Returns (price_grid, intensity in [0,1]) or None.
    """
    import backtest as bt
    daily = bt.load_all_profiles(session)
    sorted_dates = sorted(daily.keys())
    prior = [d for d in sorted_dates if d < date_nodash][-lookback:]
    if len(prior) < 1:
        return None

    price_grid = _load_price_grid()
    if price_grid is None:
        return None
    n    = len(price_grid)
    tick = float(price_grid[1] - price_grid[0])
    half = max(1, int(round(band_pts / tick)))

    # profile list with explicit weights: prior days (recency) + week + month
    wk, mo = bt.build_week_month_profiles(daily, prior)
    weighted_profiles: list[tuple[np.ndarray, float]] = []
    n_prior = len(prior)
    for idx, d in enumerate(prior):
        if d in daily:
            weighted_profiles.append((daily[d], (idx + 1) / n_prior))
    for p in wk.values():
        weighted_profiles.append((p, 1.0))
    for p in mo.values():
        weighted_profiles.append((p, 1.0))

    votes = np.zeros(n, dtype=np.float64)
    for p, w in weighted_profiles:
        if p.sum() == 0:
            continue
        levels: list[float] = []
        va = bt.compute_va(p, price_grid, va_pct)
        for k in ("poc", "vah", "val"):
            if va[k] is not None:
                levels.append(va[k])
        hvn_mask = bt.detect_hvn_fast(p, smooth=smooth, pct=hvn_pct)
        levels.extend(price_grid[hvn_mask].tolist())

        for lv in levels:
            i  = int(round((lv - price_grid[0]) / tick))
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            votes[lo:hi] += w

    if votes.max() == 0:
        return None
    intensity = votes / votes.max()
    return price_grid, intensity


def _shelf_ledge_edges(s: np.ndarray, tick: float) -> list[tuple[int, float]]:
    """
    Detect balance-edge extremes in one smoothed profile (auction-market
    definition):

      balance = high-volume region (acceptance), bins >= 30% of profile max
      ledge   = SHARP drop-off at the balance edge (volume dies within
                ~2.5pt) — hard boundary, weighted 1.0
      shelf   = GRADUAL taper at the balance edge (volume fades over up to
                ~15pt) — soft boundary, weighted 0.6

    Returns list of (bin_idx_of_edge, weight).
    """
    if s.max() <= 0:
        return []
    n        = len(s)
    bal      = s >= 0.20 * s.max()
    min_run  = max(1, int(round(5.0 / tick)))    # balance >= 5pt thick
    sharp_w  = max(1, int(round(8.0 / tick)))    # ledge: dead within 8pt
    taper_w  = max(1, int(round(40.0 / tick)))   # shelf: fades within 40pt

    out: list[tuple[int, float]] = []
    i = 0
    while i < n:
        if not bal[i]:
            i += 1
            continue
        j = i
        while j < n and bal[j]:
            j += 1
        if j - i >= min_run:
            # bottom edge (i, walking down) and top edge (j-1, walking up)
            for edge, step in ((i, -1), (j - 1, +1)):
                ev = s[edge]
                if ev <= 0:
                    continue
                # how far outside the balance until volume < 15% of edge?
                k, dist = edge + step, 0
                while 0 <= k < n and s[k] >= 0.15 * ev and dist <= taper_w:
                    k += step
                    dist += 1
                if dist <= sharp_w:
                    out.append((edge, 1.0))          # ledge — clean rejection
                elif dist <= taper_w:
                    out.append((edge, 0.6))          # shelf — soft taper
                # else: volume keeps going — not a balance extreme
        i = j
    return out


def build_shelf_confluence_heatmap(
    date_nodash: str,
    session:   str   = "full",
    lookback:  int   = 15,
    smooth:    int   = 5,
    band_pts:  float = 6.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Stacked shelf/ledge confluence (balance-edge extremes).

    Each prior profile (days recency-weighted + week + month aggregates)
    votes its balance edges: ledges (sharp cliff, weight 1.0) and shelves
    (gradual taper, weight 0.6). Each edge paints a +/-band_pts band.
    Intensity = stacked vote weight — bright where many profiles place a
    balance extreme at the same price.

    Returns (price_grid, intensity in [0,1]) or None.
    """
    import backtest as bt
    daily = bt.load_all_profiles(session)
    sorted_dates = sorted(daily.keys())
    prior = [d for d in sorted_dates if d < date_nodash][-lookback:]
    if len(prior) < 1:
        return None

    price_grid = _load_price_grid()
    if price_grid is None:
        return None
    n    = len(price_grid)
    tick = float(price_grid[1] - price_grid[0])
    half = max(1, int(round(band_pts / tick)))

    wk, mo = bt.build_week_month_profiles(daily, prior)
    weighted: list[tuple[np.ndarray, float]] = []
    n_prior = len(prior)
    for idx, d in enumerate(prior):
        if d in daily:
            weighted.append((daily[d], (idx + 1) / n_prior))
    for p in wk.values():
        weighted.append((p, 1.0))
    for p in mo.values():
        weighted.append((p, 1.0))

    votes = np.zeros(n, dtype=np.float64)
    smooth_bins = max(3, int(round(2.0 / tick)))   # 2pt profile smoothing
    for p, w_p in weighted:
        if p.sum() == 0:
            continue
        s = uniform_filter1d(p.astype(float), size=smooth_bins)
        for edge_idx, w_e in _shelf_ledge_edges(s, tick):
            lo = max(0, edge_idx - half)
            hi = min(n, edge_idx + half + 1)
            votes[lo:hi] += w_p * w_e

    if votes.max() == 0:
        return None
    return price_grid, votes / votes.max()


def draw_dual_heatmap(ax, price_grid: np.ndarray,
                      vp_int: np.ndarray, gex_int: np.ndarray,
                      x_lo: float, x_hi: float,
                      price_lo: float, price_hi: float,
                      gamma: float = 0.75):
    """
    Blend VP confluence (magenta) and stacked GEX (green) into one RGB image.
    Overlap zones glow white — both maps agree price reacts there.
    """
    mask = (price_grid >= price_lo) & (price_grid <= price_hi)
    if not mask.any():
        return

    def _prep(a, quantize=True):
        s = a[mask].astype(float)
        if s.max() > 0:
            s = s / s.max()
        s = s ** gamma
        if quantize:
            s = np.round(s * 6) / 6         # quantized tiers
        return s

    v = _prep(vp_int)                       # crisp banded VP shelves
    g = _prep(gex_int)                      # crisp GEX reversal zones

    # VP → magenta (R+B), GEX → green. Overlap → white-hot.
    R = 0.85 * v + 0.15 * g
    G = 0.80 * g + 0.10 * v
    B = 0.85 * v + 0.15 * g
    img = np.clip(np.stack([R, G, B], axis=-1), 0, 1)[:, None, :]

    ax.imshow(
        img,
        extent=(x_lo, x_hi, price_grid[mask][0], price_grid[mask][-1]),
        origin="lower", aspect="auto",
        interpolation="nearest",
        zorder=0,
    )


def draw_heatmap(ax, price_grid: np.ndarray, intensity: np.ndarray,
                 x_lo: float, x_hi: float,
                 price_lo: float, price_hi: float,
                 gamma: float = 0.75, alpha: float = 1.0):
    """Render confluence intensity as full-width horizontal gradient bands."""
    mask = (price_grid >= price_lo) & (price_grid <= price_hi)
    if not mask.any():
        return
    sub = intensity[mask].astype(float)
    if sub.max() > 0:
        sub = sub / sub.max()               # renormalize to visible window
    sub = sub ** gamma                      # gamma <1 lifts mid tones
    sub = np.round(sub * 6) / 6             # quantize → distinct brightness tiers
    img = sub[:, None]                      # (n_bins, 1) column image
    ax.imshow(
        img,
        cmap=HEAT_CMAP, vmin=0.0, vmax=1.0,
        extent=(x_lo, x_hi, price_grid[mask][0], price_grid[mask][-1]),
        origin="lower", aspect="auto",
        interpolation="nearest",            # crisp band edges
        zorder=0, alpha=alpha,
    )


# ── OHLCV helpers ─────────────────────────────────────────────────────────────

def _resample_ohlcv(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    price = trades["price"]
    size  = trades["size"]
    o = price.resample("1min").first()
    h = price.resample("1min").max()
    l = price.resample("1min").min()
    c = price.resample("1min").last()
    v = size.resample("1min").sum()
    df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v})
    return df.dropna(subset=["open"])


# ── drawing ───────────────────────────────────────────────────────────────────

def draw_hvn_zones(ax, zones: list[tuple], price_lo: float, price_hi: float):
    """Draw filled HVN bands spanning full chart width."""
    for lo, hi, intensity in zones:
        if hi < price_lo or lo > price_hi:
            continue
        lo = max(lo, price_lo)
        hi = min(hi, price_hi)
        ax.axhspan(lo, hi, xmin=0, xmax=1,
                   color=_hvn_color(intensity), zorder=1, linewidth=0)


def draw_candles(ax, ohlcv: pd.DataFrame, x_nums: np.ndarray, alpha: float = 0.7):
    bar_w  = 0.55
    wick_w = 0.07
    up_bodies, dn_bodies, wicks = [], [], []

    for xi, (_, row) in zip(x_nums, ohlcv.iterrows()):
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        is_up = c >= o
        body_lo = min(o, c)
        body_hi = max(o, c)
        body_h  = max(body_hi - body_lo, vp.TICK)

        body = mpatches.FancyBboxPatch(
            (xi - bar_w / 2, body_lo), bar_w, body_h,
            boxstyle="square,pad=0", linewidth=0,
        )
        for wy, wh in [(l, body_lo - l), (body_hi, h - body_hi)]:
            if wh > 0:
                wicks.append(mpatches.FancyBboxPatch(
                    (xi - wick_w / 2, wy), wick_w, wh,
                    boxstyle="square,pad=0", linewidth=0,
                ))
        (up_bodies if is_up else dn_bodies).append(body)

    if up_bodies:
        ax.add_collection(PatchCollection(up_bodies,
            facecolor=UP_COLOR, linewidth=0, zorder=3, alpha=alpha))
    if dn_bodies:
        ax.add_collection(PatchCollection(dn_bodies,
            facecolor=DOWN_COLOR, linewidth=0, zorder=3, alpha=alpha))
    if wicks:
        ax.add_collection(PatchCollection(wicks,
            facecolor=WICK_COLOR, linewidth=0, zorder=2, alpha=alpha * 0.7))


def _format_x_ticks(ax, ohlcv_et: pd.DataFrame, max_ticks: int = 10):
    n = len(ohlcv_et)
    if n == 0:
        return
    step = max(1, n // max_ticks)
    positions = list(range(0, n, step))
    labels = [ohlcv_et.index[i].strftime("%H:%M") for i in positions]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, color=TEXT_COLOR, fontsize=8)


# ── main ─────────────────────────────────────────────────────────────────────

def make_chart(
    date: str,
    start_time:   str   = "00:00",
    end_time:     str   = "23:59",
    lookback:     int   = 15,
    min_stack:    int   = 2,
    session:      str   = "full",
    smooth:       int   = 5,
    hvn_pct:      float = 60.0,
    pred_high:    float | None = None,
    pred_low:     float | None = None,
    save_path:    str   | None = None,
    show_candles: bool  = True,
    show_gex:     bool  = False,
):
    date_nodash = date.replace("-", "")
    price_grid  = _load_price_grid()

    # ── find prior dates ──
    if price_grid is not None:
        # use cache
        all_dates = sorted(
            f.replace(f"_{session}.npy", "")
            for f in os.listdir(PROFILE_DIR)
            if f.endswith(f"_{session}.npy")
        )
    else:
        all_files = vp._discover_files(BASE_DIR)
        all_dates = sorted(all_files.keys())
        price_grid = None

    prior_dates = [d for d in all_dates if d < date_nodash][-lookback:]
    if not prior_dates:
        raise ValueError(f"No prior sessions before {date}.")

    # ── load OHLCV for today ──
    ohlcv = None
    cached_ohlcv = _load_cached_ohlcv(date_nodash)
    if cached_ohlcv is not None:
        sess_filter = "rth" if start_time == "09:30" else "full"
        sub = cached_ohlcv[cached_ohlcv["session"] == sess_filter]
        if not sub.empty:
            ohlcv = sub.drop(columns=["session"])

    if ohlcv is None:
        # fall back to cached trades
        trades = _load_cached_trades(date_nodash)
        if trades is None:
            all_files = vp._discover_files(BASE_DIR)
            if date_nodash not in all_files:
                raise ValueError(f"No data for {date}")
            trades = vp.load_trades(all_files[date_nodash])
        trades_et = trades.tz_convert(ET)
        start_et  = pd.Timestamp(f"{date} {start_time}", tz=ET)
        end_et    = pd.Timestamp(f"{date} {end_time}",   tz=ET)
        window    = trades_et[(trades_et.index >= start_et) & (trades_et.index <= end_et)]
        ohlcv     = _resample_ohlcv(window)

    if ohlcv is None or ohlcv.empty:
        raise ValueError(f"No OHLCV data for {date} {start_time}–{end_time}")

    # filter to time window if index is datetime
    if hasattr(ohlcv.index, 'tz') and ohlcv.index.tz is not None:
        ohlcv_et = ohlcv.copy()
        ohlcv_et.index = ohlcv.index.tz_convert(ET)
        start_et = pd.Timestamp(f"{date} {start_time}", tz=ET)
        end_et   = pd.Timestamp(f"{date} {end_time}",   tz=ET)
        ohlcv_et = ohlcv_et[(ohlcv_et.index >= start_et) & (ohlcv_et.index <= end_et)]
        ohlcv    = ohlcv_et.copy()
        ohlcv.index = ohlcv.index.tz_convert("UTC")
    else:
        ohlcv_et = ohlcv.copy()

    price_lo = float(ohlcv["low"].min())  - 30
    price_hi = float(ohlcv["high"].max()) + 30

    # ── stacked shelf/ledge confluence heatmap (day+week+month) ──
    heat = build_shelf_confluence_heatmap(
        date_nodash, session=session, lookback=lookback, smooth=smooth)

    gex_heat = None
    if show_gex:
        import gex as gx
        gex_heat = gx.build_stacked_gex(
            date_nodash, session=session, lookback=lookback)

    # ── auto-extend y-range if no bands fall inside the view (breakout /
    # price-discovery days trade beyond all prior structure — pull the
    # nearest bands into view for context, capped at 250pt) ──
    nz_mask = None
    for h in (heat, gex_heat):
        if h is not None:
            pg_h, it_h = h
            m = it_h > 0.05
            nz_mask = m if nz_mask is None else (nz_mask | m)
    if nz_mask is not None and nz_mask.any():
        pg_h = (heat or gex_heat)[0]
        in_view = nz_mask & (pg_h >= price_lo) & (pg_h <= price_hi)
        if not in_view.any():
            below = pg_h[nz_mask & (pg_h < price_lo)]
            above = pg_h[nz_mask & (pg_h > price_hi)]
            if len(below):
                price_lo = max(float(below.max()) - 15.0, price_lo - 250.0)
            if len(above):
                price_hi = min(float(above.min()) + 15.0, price_hi + 250.0)

    # ── figure ──
    fig, (ax_main, ax_vol) = plt.subplots(
        2, 1,
        figsize=(18, 11),
        gridspec_kw={"height_ratios": [5, 1]},
        facecolor=BG_COLOR,
    )
    fig.subplots_adjust(hspace=0.0, left=0.06, right=0.97, top=0.93, bottom=0.07)

    for ax in [ax_main, ax_vol]:
        ax.set_facecolor(BG_COLOR)
        ax.tick_params(colors=TEXT_COLOR, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#111111")
        ax.yaxis.grid(False)
        ax.xaxis.grid(False)

    x_nums = np.arange(len(ohlcv))

    # confluence heatmap first (background)
    if heat is not None and gex_heat is not None:
        pg_full, vp_intensity = heat
        _, gex_intensity = gex_heat
        draw_dual_heatmap(ax_main, pg_full, vp_intensity, gex_intensity,
                          x_lo=-1, x_hi=len(ohlcv),
                          price_lo=price_lo, price_hi=price_hi)
    elif heat is not None:
        pg_full, intensity = heat
        draw_heatmap(ax_main, pg_full, intensity,
                     x_lo=-1, x_hi=len(ohlcv),
                     price_lo=price_lo, price_hi=price_hi)

    # candles on top
    if show_candles:
        draw_candles(ax_main, ohlcv, x_nums, alpha=0.65)

    # predicted high / low lines
    if pred_high is not None:
        ax_main.axhline(pred_high, color=PRED_HIGH_COLOR, linewidth=1.5,
                        linestyle="--", zorder=5, alpha=0.9)
        ax_main.annotate(f"Pred High  {pred_high:.2f}",
                         xy=(len(ohlcv) - 1, pred_high),
                         xytext=(4, 2), textcoords="offset points",
                         color=PRED_HIGH_COLOR, fontsize=8,
                         annotation_clip=False)

    if pred_low is not None:
        ax_main.axhline(pred_low, color=PRED_LOW_COLOR, linewidth=1.5,
                        linestyle="--", zorder=5, alpha=0.9)
        ax_main.annotate(f"Pred Low   {pred_low:.2f}",
                         xy=(len(ohlcv) - 1, pred_low),
                         xytext=(4, 2), textcoords="offset points",
                         color=PRED_LOW_COLOR, fontsize=8,
                         annotation_clip=False)

    ax_main.set_xlim(-1, len(ohlcv))
    ax_main.set_ylim(price_lo, price_hi)
    ax_main.yaxis.tick_right()
    ax_main.yaxis.set_tick_params(labelcolor=TEXT_COLOR)

    ohlcv_et2 = ohlcv.copy()
    if hasattr(ohlcv.index, 'tz') and ohlcv.index.tz is not None:
        ohlcv_et2.index = ohlcv.index.tz_convert(ET)
    _format_x_ticks(ax_main, ohlcv_et2)
    ax_main.set_xticklabels([])

    # ── volume panel ──
    vol_colors = [
        "#444444" if row["close"] >= row["open"] else "#333333"
        for _, row in ohlcv.iterrows()
    ]
    ax_vol.bar(x_nums, ohlcv["volume"], color=vol_colors, width=0.8, zorder=2)
    ax_vol.set_facecolor(BG_COLOR)
    ax_vol.set_xlim(-1, len(ohlcv))
    ax_vol.set_ylim(0, ohlcv["volume"].max() * 1.3)
    _format_x_ticks(ax_vol, ohlcv_et2)
    ax_vol.set_ylabel("Vol", color=TEXT_COLOR, fontsize=7)
    ax_vol.yaxis.tick_right()
    ax_vol.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(
            lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))
        )
    )

    # ── title ──
    kind = ("VP shelves (magenta) + GEX reversal zones (green) — overlap = white"
            if (show_gex and gex_heat is not None)
            else "Shelf/ledge confluence heatmap")
    ax_main.set_title(
        f"NQ  {date}  {start_time}–{end_time} ET  |  "
        f"{len(ohlcv)} bars  |  "
        f"{kind}  (lb={lookback}d, day+week+month, recency, sess={session})",
        color=TEXT_COLOR, fontsize=9, pad=6,
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
        print(f"[chart] Saved → {save_path}")
        plt.close(fig)
    else:
        plt.show()

    return fig


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="NQ HVN heatmap + predicted high/low")
    p.add_argument("--date",       required=True,  help="YYYY-MM-DD")
    p.add_argument("--start",      default="00:00")
    p.add_argument("--end",        default="23:59")
    p.add_argument("--lookback",   type=int,   default=15)
    p.add_argument("--min-stack",  dest="min_stack", type=int, default=2)
    p.add_argument("--session",    default="full", choices=["full","rth"])
    p.add_argument("--hvn-pct",    dest="hvn_pct", type=float, default=60.0,
                   help="Percentile threshold for HVN detection (default 60)")
    p.add_argument("--smooth",     type=int, default=5)
    p.add_argument("--pred-high",  dest="pred_high", type=float, default=None,
                   help="Predicted top tick price (draws cyan dashed line)")
    p.add_argument("--pred-low",   dest="pred_low",  type=float, default=None,
                   help="Predicted bottom tick price (draws orange dashed line)")
    p.add_argument("--save",       default=None)
    p.add_argument("--no-candles", dest="no_candles", action="store_true")
    p.add_argument("--gex",        action="store_true",
                   help="Overlay stacked GEX heatmap (green); overlap with VP = white")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    make_chart(
        date         = args.date,
        start_time   = args.start,
        end_time     = args.end,
        lookback     = args.lookback,
        session      = args.session,
        hvn_pct      = args.hvn_pct,
        smooth       = args.smooth,
        pred_high    = args.pred_high,
        pred_low     = args.pred_low,
        save_path    = args.save,
        show_candles = not args.no_candles,
        show_gex     = args.gex,
    )
