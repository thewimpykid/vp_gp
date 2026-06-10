"""
GEX Profile Analyzer for NQ Futures
=====================================
Builds Gamma Exposure profiles by weighting historical volume-at-price
with Black-Scholes gamma keyed to VIX-implied IV.

  GEX(K) = VP_volume(K) × Γ(S=spot, K=bin, σ=σ_NQ, T=DTE)

VP_volume(K) : historical volume traded near price K  (OHLCV uniform distribution)
BS Gamma     : options dealer sensitivity at strike K given today's spot and IV
σ_NQ         : VIX/100 × 1.15  (NQ IV historically ~15 % above SPX implied vol)

Three DTE windows create independent GEX profiles:
  5d  (weekly options)    — immediate positioning
  21d (monthly options)   — standard monthly expiry
  63d (quarterly options) — institutional hedges

GEX HVN  → high dealer sensitivity → price stalls / reverses there
GEX LVN  → dealer-neutral zone    → price moves freely (fast travel)
gwall    → steepest GEX gradient  → abrupt dealer response boundary
gflip    → GEX inflection point   → dealer behaviour changes mode

Usage
-----
  python gex_profile.py                   # daily levels + chart
  python gex_profile.py --sample-days 15  # 15 no-look-ahead sample charts
  python gex_profile.py --backtest        # 3-year reversal backtest
  python gex_profile.py --hod-lod         # walk-forward HOD/LOD coverage test
  python gex_profile.py --compare-vp      # GEX vs plain VP level comparison
"""

import argparse
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths, savgol_filter

warnings.filterwarnings("ignore")

# Magenta confidence colormap (same as stacked_vp)
MAGENTA_CMAP = LinearSegmentedColormap.from_list(
    "magenta_conf", ["#FF99FF", "#880088"]
)

# ─── Paths ────────────────────────────────────────────────────────────────────
CSV_PATH   = Path(__file__).parent.parent / "1Min_NQ.csv"
CACHE_PATH = Path(__file__).parent / "hdata" / "NQ_1m_cache.parquet"
VIX_CACHE  = Path(__file__).parent / "hdata" / "vix_daily.parquet"

# ─── Parameters ───────────────────────────────────────────────────────────────
BIN_SIZE        = 5.0    # NQ points per bin (matches stacked_vp)
VALUE_AREA_PCT  = 0.70   # 70% value area
HVN_PROMINENCE  = 0.30   # GEX HVN peak prominence (slightly looser than VP)
LVN_DEPTH       = 0.40   # GEX LVN valley depth fraction
SHELF_SLOPE_MAX = 0.08
LEDGE_SLOPE_MIN = 0.20   # tighter than VP (GEX curves have steeper cliffs)
SLOPE_BINS      = 5
STACK_TOLERANCE = 3.0    # confluence radius = 3 × BIN_SIZE = 15 pts
GWALL_PROMINENCE= 0.25   # gradient-wall peak prominence (fraction of max |dG/dK|)
GFLIP_MIN_GEX   = 0.20   # inflection only kept if local GEX > 20% of max
NQ_IV_MULT      = 1.15   # NQ implied vol ≈ VIX × 1.15
INCLUDE_SHELF   = True   # GEX shelf enabled (bell-curve shoulders meaningful)
INCLUDE_LEDGE   = True   # GEX ledge enabled
MINS_PER_DAY    = 390

# GEX DTE windows: name → (lookback_trading_days, dte_trading_days)
GEX_WINDOWS = {
    "gex_5d":  (21, 5),
    "gex_21d": (42, 21),
    "gex_63d": (63, 63),
}


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class VolumeProfile:
    name: str
    bins: np.ndarray   # left edge of each 5-pt bin
    vols: np.ndarray   # gamma-weighted volume (GEX) per bin
    poc:  float
    vah:  float
    val:  float


@dataclass
class VPFeature:
    price:     float
    ftype:     str     # vah|val|poc|hvn|lvn|shelf|ledge|gwall|gflip|pdh|pdl|...
    timeframe: str
    weight:    float = 1.0


@dataclass
class Zone:
    price:      float
    score:      float
    n_tf:       int
    timeframes: set
    ftypes:     set
    features:   list


# ─── Data Loading ─────────────────────────────────────────────────────────────

def _load_sierra_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", thousands=".", decimal=",")
    df = df.rename(columns={
        "Date": "date", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume",
    })
    df["date"] = pd.to_datetime(df["date"], dayfirst=False)
    return df[["date", "open", "high", "low", "close", "volume"]].copy()


def load_nq_data() -> pd.DataFrame:
    if CACHE_PATH.exists():
        print(f"  NQ cache  → {CACHE_PATH}")
        df = pd.read_parquet(CACHE_PATH)
        if "date" not in df.columns:
            df = df.reset_index().rename(columns={df.index.name or "index": "date"})
    else:
        print(f"  NQ csv    → {CSV_PATH}  (caching...)")
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


def load_vix() -> pd.Series:
    """
    Load VIX daily closes. Tries yfinance first, then CBOE CSV if present,
    then falls back to 20-day annualised realised vol computed from NQ prices.
    Returns a Series indexed by date (daily, forward-filled to cover weekends).
    """
    if VIX_CACHE.exists():
        s = pd.read_parquet(VIX_CACHE)["vix"]
        print(f"  VIX cache → {VIX_CACHE}  ({len(s)} days)")
        return s

    vix_series = None

    # Try yfinance
    try:
        import yfinance as yf
        raw = yf.download("^VIX", start="2019-01-01", end="2026-12-31",
                          progress=False, auto_adjust=True)
        if len(raw) > 100:
            vix_series = raw["Close"].rename("vix")
            vix_series.index = pd.to_datetime(vix_series.index).tz_localize(None)
            print(f"  VIX yfinance: {len(vix_series)} days")
    except Exception as e:
        print(f"  yfinance unavailable ({e}), falling back to realised vol")

    if vix_series is None:
        # Fallback: compute 20-day realised vol from NQ
        df_nq = load_nq_data()
        df_nq["_d"] = df_nq["date"].dt.date
        daily_cl = df_nq.groupby("_d")["close"].last()
        daily_cl.index = pd.to_datetime(daily_cl.index)
        log_ret = np.log(daily_cl / daily_cl.shift(1)).dropna()
        rvol = log_ret.rolling(20).std() * np.sqrt(252) * 100
        vix_series = rvol.rename("vix").dropna()
        print(f"  VIX fallback (realised vol): {len(vix_series)} days")

    df_v = pd.DataFrame({"vix": vix_series})
    df_v.index = pd.to_datetime(df_v.index)
    VIX_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df_v.to_parquet(VIX_CACHE)
    return df_v["vix"]


def get_vix_at(vix_series: pd.Series, as_of_date) -> float:
    """
    Return VIX level on or before as_of_date (no look-ahead).
    Falls back to 20.0 if series has no data before that date.
    """
    as_of = pd.Timestamp(as_of_date)
    prior = vix_series[vix_series.index <= as_of]
    if prior.empty:
        return 20.0
    return float(prior.iloc[-1])


# ─── Black-Scholes Gamma ──────────────────────────────────────────────────────

def bs_gamma(S: float, K: float, T: float, sigma: float, r: float = 0.03) -> float:
    """
    Black-Scholes gamma for a European option.
    S : spot price
    K : strike price
    T : time to expiry in years
    sigma: annualised implied vol (decimal, e.g. 0.18 for 18%)
    Returns gamma (per point of spot move).
    """
    if T <= 1e-9 or sigma <= 1e-9 or S <= 0 or K <= 0:
        return 0.0
    try:
        sqT = math.sqrt(T)
        d1  = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqT)
        # Standard normal PDF of d1
        nd1 = math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
        return nd1 / (S * sigma * sqT)
    except (ValueError, ZeroDivisionError, OverflowError):
        return 0.0


# ─── Volume Profile (raw) ─────────────────────────────────────────────────────

def _build_raw_vp(bars: pd.DataFrame, bin_size: float) -> tuple:
    """
    Uniform OHLCV distribution into price bins.
    Returns (bins, vols) arrays — same algo as stacked_vp.py.
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
    return bins, profile


# ─── GEX Profile Construction ─────────────────────────────────────────────────

def build_gex_profile(bars: pd.DataFrame, spot: float, sigma_nq: float,
                       dte_days: int, name: str, bin_size: float) -> VolumeProfile:
    """
    Build a gamma-weighted volume profile (GEX profile).

    Steps:
      1. Build raw VP from OHLCV bars (uniform distribution per bar)
      2. Compute BS gamma for each price bin given spot, σ_NQ, DTE
      3. GEX(K) = VP_vol(K) × Gamma(S=spot, K, σ, T)
      4. Rescale GEX to same magnitude as raw VP (for feature detection consistency)
      5. Compute POC, VAH, VAL on GEX-weighted distribution

    Why this works:
      - VP captures WHERE volume historically traded
      - Gamma kernel weights by HOW SENSITIVE dealers are AT EACH LEVEL NOW
      - Levels near spot + high historical volume → high GEX (dealers must hedge there)
      - Levels far from spot → low GEX even if historically heavy (OTM gamma ≈ 0)
      - High VIX → wide gamma bell → more distant levels get weight
      - Low VIX → narrow gamma bell → only ATM levels matter
    """
    bins, vp_vols = _build_raw_vp(bars, bin_size)

    T = max(dte_days / 252.0, 1 / 252.0)
    gammas = np.array([bs_gamma(spot, float(K), T, sigma_nq) for K in bins])

    gex_raw = vp_vols * gammas

    # Rescale to original VP magnitude for comparable feature detection
    vp_max  = vp_vols.max()
    gex_max = gex_raw.max()
    if gex_max > 1e-9 and vp_max > 1e-9:
        gex_vols = gex_raw * (vp_max / gex_max)
    else:
        gex_vols = gex_raw.copy()

    # POC on GEX distribution
    poc_idx = int(np.argmax(gex_vols))
    poc     = float(bins[poc_idx])

    # Value Area: 70% of GEX volume
    total  = gex_vols.sum()
    order  = np.argsort(gex_vols)[::-1]
    cumsum = 0.0
    va_idx = set()
    for idx in order:
        cumsum += gex_vols[idx]
        va_idx.add(int(idx))
        if cumsum >= total * VALUE_AREA_PCT:
            break
    va_sorted = sorted(va_idx)
    vah = float(bins[va_sorted[-1]]) + bin_size
    val = float(bins[va_sorted[0]])

    return VolumeProfile(name=name, bins=bins, vols=gex_vols, poc=poc, vah=vah, val=val)


# ─── Feature Detection (same scipy logic as stacked_vp) ─────────────────────

def detect_features(vp: VolumeProfile, bin_size: float = BIN_SIZE) -> list:
    """
    Detect HVN, LVN, shelf, ledge, VAH, VAL, POC on the GEX profile.
    Same methodology as stacked_vp.detect_features — works on any volume array.
    """
    feats: list[VPFeature] = []
    bins, vols = vp.bins, vp.vols
    tf = vp.name

    feats += [
        VPFeature(vp.vah, "vah", tf, weight=1.2),
        VPFeature(vp.val, "val", tf, weight=1.2),
        VPFeature(vp.poc, "poc", tf, weight=1.1),
    ]

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

    # HVN peaks
    peaks, _ = find_peaks(smooth, prominence=HVN_PROMINENCE * peak_vol)
    for p in peaks:
        try:
            _, _, lo_ips, hi_ips = peak_widths(smooth, [p], rel_height=0.5)
        except Exception:
            continue
        l_idx = int(np.clip(lo_ips[0], 0, n - 1))
        r_idx = int(np.clip(hi_ips[0], 0, n - 1))
        _classify_boundary(smooth, bins, l_idx, "lower", peak_vol, tf, feats, bin_size)
        _classify_boundary(smooth, bins, r_idx, "upper", peak_vol, tf, feats, bin_size)

    # LVN valleys
    valleys, _ = find_peaks(-smooth, prominence=0.08 * peak_vol)
    for v in valleys:
        if smooth[v] < LVN_DEPTH * mean_vol:
            feats.append(VPFeature(float(bins[v]), "lvn", tf, weight=0.9))

    return feats


def _classify_boundary(smooth, bins, edge_idx, side, peak_vol, tf, feats, bin_size):
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

    price = float(bins[edge_idx])
    if avg_slope <= SHELF_SLOPE_MAX and INCLUDE_SHELF:
        feats.append(VPFeature(price, "shelf", tf, weight=0.8))
    elif avg_slope >= LEDGE_SLOPE_MIN and INCLUDE_LEDGE:
        feats.append(VPFeature(price, "ledge", tf, weight=1.3))


def detect_gex_features(vp: VolumeProfile, bin_size: float = BIN_SIZE) -> list:
    """
    GEX-specific structural features not present in plain VP:

    gwall (weight 1.1):
      Local maximum of |dGEX/dK| — where gamma exposure changes most sharply.
      Physically: the price at which dealer hedging activity transitions most
      abruptly from heavy to light (or vice versa). Price tends to stall at
      these gradient walls as dealers pile into hedges.
      Only kept if local GEX > 15% of max (avoid noise at tails).

    gflip (weight 1.3):
      Inflection point of the GEX curve (d²GEX/dK² = 0).
      Marks where the curvature of dealer sensitivity reverses.
      Below the lower gflip: GEX is increasing toward ATM (dealers getting more
      sensitive as price approaches). Above the upper gflip: same from above.
      The band between the two gflips is the "gamma maximum" region — maximum
      dealer sensitivity = maximum mean-reversion pressure.
      Only the 2 most prominent inflection points (on main GEX hump) are kept.
    """
    feats: list[VPFeature] = []
    bins, vols = vp.bins, vp.vols
    tf = vp.name
    n  = len(vols)

    if n < 15:
        return feats

    win = max(7, n // 12)
    if win % 2 == 0:
        win += 1
    win = min(win, n - 1 if n % 2 == 0 else n)
    try:
        smooth = savgol_filter(vols, window_length=win, polyorder=3).clip(0)
    except Exception:
        smooth = vols.copy()

    peak_vol = smooth.max()
    if peak_vol < 1e-9:
        return feats

    # GEX Walls: local maxima of |dGEX/dK|
    grad     = np.gradient(smooth, bin_size)
    abs_grad = np.abs(grad)
    ag_max   = abs_grad.max()
    if ag_max > 1e-12:
        wall_peaks, props = find_peaks(
            abs_grad, prominence=GWALL_PROMINENCE * ag_max,
            distance=max(3, n // 20)
        )
        # Sort by prominence, keep top 3
        if len(wall_peaks) > 0:
            proms = props["prominences"]
            order = np.argsort(proms)[::-1][:3]
            for idx in order:
                p = wall_peaks[idx]
                if smooth[p] >= 0.15 * peak_vol:  # only if GEX is meaningful there
                    feats.append(VPFeature(float(bins[p]), "gwall", tf, weight=1.1))

    # GEX Flip: zero crossings of d²GEX/dK² (inflection points)
    d2 = np.gradient(grad, bin_size)
    sign_ch = np.where(np.diff(np.sign(d2)))[0]

    # Score each inflection by local GEX magnitude + gradient magnitude
    candidates = []
    for i in sign_ch:
        local_gex = float(smooth[i])
        local_grad = float(abs_grad[i])
        if local_gex >= GFLIP_MIN_GEX * peak_vol:
            score = local_gex * local_grad
            candidates.append((score, i))

    # Keep top 2 inflection points (shoulders of main GEX bell)
    candidates.sort(reverse=True)
    for _, i in candidates[:2]:
        feats.append(VPFeature(float(bins[i]), "gflip", tf, weight=1.3))

    return feats


# ─── Structural Price Levels (same as stacked_vp) ────────────────────────────

def compute_atr20(df: pd.DataFrame) -> float:
    df2 = df.copy()
    df2["_d"] = df2["date"].dt.date
    day_bars = (
        df2.groupby("_d")
        .agg(hi=("high", "max"), lo=("low", "min"), cl=("close", "last"))
        .reset_index().tail(25).reset_index(drop=True)
    )
    if len(day_bars) < 2:
        return 300.0
    hi, lo, cl = day_bars["hi"].values, day_bars["lo"].values, day_bars["cl"].values
    prev_cl = np.concatenate([[cl[0]], cl[:-1]])
    tr = np.maximum(hi - lo, np.maximum(np.abs(hi - prev_cl), np.abs(lo - prev_cl)))
    return float(tr[-20:].mean())


def prior_day_features(df: pd.DataFrame, n_prior: int = 10) -> list:
    """PDH/PDL per-session timeframe IDs (d-1..d-10), PWH/PWL."""
    df2 = df.copy()
    df2["_d"] = df2["date"].dt.date
    daily = (
        df2.groupby("_d")
        .agg(hi=("high", "max"), lo=("low", "min"))
        .reset_index()
    )
    daily = daily.iloc[:-1].tail(n_prior)
    feats: list[VPFeature] = []
    for i, (_, row) in enumerate(daily.iloc[::-1].iterrows()):
        tf = f"d-{i+1}"
        feats.append(VPFeature(float(row.hi), "pdh", tf, weight=1.4))
        feats.append(VPFeature(float(row.lo), "pdl", tf, weight=1.4))

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
    """IBH/IBL (wt 1.6), ONH/ONL (1.2), gap (1.1), PMH/PML (1.3)."""
    feats: list[VPFeature] = []
    df2 = df.copy()
    df2["_d"]  = df2["date"].dt.date
    df2["_tm"] = df2["date"].dt.hour * 60 + df2["date"].dt.minute

    all_days  = sorted(df2["_d"].unique())
    if len(all_days) < 2:
        return feats

    prior_days = all_days[:-1][-n_prior:]
    for i, day in enumerate(reversed(prior_days)):
        tf       = f"d-{i+1}"
        day_bars = df2[df2["_d"] == day]

        ib = day_bars[(day_bars["_tm"] >= 570) & (day_bars["_tm"] <= 630)]
        if len(ib) >= 5:
            feats.append(VPFeature(float(ib["high"].max()), "ibh", tf, weight=1.6))
            feats.append(VPFeature(float(ib["low"].min()),  "ibl", tf, weight=1.6))

        overnight = day_bars[day_bars["_tm"] < 570]
        if len(overnight) >= 5:
            feats.append(VPFeature(float(overnight["high"].max()), "onh", tf, weight=1.2))
            feats.append(VPFeature(float(overnight["low"].min()),  "onl", tf, weight=1.2))

        rth = day_bars[(day_bars["_tm"] >= 570) & (day_bars["_tm"] <= 975)]
        if len(rth) >= 5:
            feats.append(VPFeature(float(rth["close"].iloc[-1]), "gap", tf, weight=1.1))

    df2["_ym"] = df2["date"].dt.to_period("M")
    months = sorted(df2["_ym"].unique())
    if len(months) >= 2:
        pm_bars = df2[df2["_ym"] == months[-2]]
        feats.append(VPFeature(float(pm_bars["high"].max()), "pmh", "prior_month", weight=1.3))
        feats.append(VPFeature(float(pm_bars["low"].min()),  "pml", "prior_month", weight=1.3))

    return feats


def round_number_features(df: pd.DataFrame, atr20: float) -> list:
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


# ─── Confluence Stacking ──────────────────────────────────────────────────────

def stack_features(all_features: list, bin_size: float,
                   tol_mult: float = STACK_TOLERANCE,
                   score_exp: float = 2.0) -> list:
    """
    Greedy clustering identical to stacked_vp.py.
    score = n_unique_timeframes^score_exp × Σ weights
    """
    if not all_features:
        return []

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


# ─── Analysis Driver ──────────────────────────────────────────────────────────

def get_cutoff(df: pd.DataFrame, lookback_days: int) -> pd.Timestamp:
    """Return timestamp at approximately lookback_days trading days before last bar."""
    idx = max(0, len(df) - lookback_days * MINS_PER_DAY)
    return df.iloc[idx]["date"]


def run(df: pd.DataFrame, vix_series: pd.Series,
        bin_size: float = BIN_SIZE, quiet: bool = False):
    """
    Build all three GEX profiles, detect features, add structural levels, stack zones.
    Uses last bar's date for VIX lookup (no look-ahead in per-day mode).
    """
    spot      = float(df["close"].iloc[-1])
    last_date = df["date"].max()

    vix_raw   = get_vix_at(vix_series, last_date)
    sigma_nq  = (vix_raw / 100.0) * NQ_IV_MULT

    if not quiet:
        print(f"  Spot={spot:.2f}  VIX={vix_raw:.1f}  σ_NQ={sigma_nq:.3f}")

    all_features: list[VPFeature] = []
    profiles: dict = {}

    for name, (lookback_d, dte_d) in GEX_WINDOWS.items():
        cutoff = get_cutoff(df, lookback_d)
        subset = df[df["date"] >= cutoff]
        if len(subset) < 200:
            if not quiet:
                print(f"  [{name}] insufficient data, skipping")
            continue

        gex_vp = build_gex_profile(subset, spot, sigma_nq, dte_d, name, bin_size)
        profiles[name] = gex_vp

        feats  = detect_features(gex_vp, bin_size)
        feats += detect_gex_features(gex_vp, bin_size)
        all_features.extend(feats)

        if not quiet:
            n_hvn = sum(1 for f in feats if f.ftype in {"hvn", "shelf", "ledge"})
            print(f"  [{name}]  dte={dte_d:>3}d  σ_NQ={sigma_nq:.3f}  "
                  f"POC={gex_vp.poc:.1f}  VAH={gex_vp.vah:.1f}  VAL={gex_vp.val:.1f}  "
                  f"feats={len(feats)}  (gwall/gflip="
                  f"{sum(1 for f in feats if f.ftype=='gwall')}/"
                  f"{sum(1 for f in feats if f.ftype=='gflip')})")

    # Structural levels (same as stacked_vp)
    all_features.extend(prior_day_features(df))
    all_features.extend(session_level_features(df))
    atr20 = compute_atr20(df)
    all_features.extend(round_number_features(df, atr20))

    zones = stack_features(all_features, bin_size)
    return zones, profiles, sigma_nq


# ─── Console Reports ──────────────────────────────────────────────────────────

def print_report(zones: list, show_n: int = 20, sigma_nq: float = 0.0):
    sep = "-" * 78
    print(f"\n{sep}")
    print(f"  TOP {min(show_n, len(zones))} GEX CONFLUENCE ZONES   (σ_NQ={sigma_nq:.3f})")
    print(sep)
    print(f"  {'PRICE':>9}  {'SCORE':>6}  {'TF':>2}  TIMEFRAMES            FEATURE TYPES")
    print(sep)
    for z in zones[:show_n]:
        tfs = ",".join(sorted(z.timeframes))
        fts = "|".join(sorted(z.ftypes))
        print(f"  {z.price:>9.2f}  {z.score:>6.1f}  {z.n_tf:>2}  {tfs:<20}  {fts}")
    print(sep)
    print(f"  3+ TF zones : {sum(1 for z in zones if z.n_tf >= 3)}")
    print(f"  gwall zones : {sum(1 for z in zones if 'gwall' in z.ftypes)}")
    print(f"  gflip zones : {sum(1 for z in zones if 'gflip' in z.ftypes)}")


def daily_levels(df: pd.DataFrame, zones: list, n: int = 5) -> tuple:
    """5 balanced GEX zones above/below current close. Same logic as stacked_vp."""
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

    above   = [z for z in in_range if z.price > last_close]
    below   = [z for z in in_range if z.price <= last_close]
    n_above = math.ceil(n / 2)
    n_below = n - n_above
    sel_abv = above[:n_above]
    sel_blw = below[:n_below]
    if len(sel_abv) < n_above:
        sel_blw = below[:n - len(sel_abv)]
    elif len(sel_blw) < n_below:
        sel_abv = above[:n - len(sel_blw)]

    return sorted(sel_abv + sel_blw, key=lambda z: z.price), last_close, atr20


def print_daily_levels(levels: list, last_close: float, atr20: float):
    req_move = atr20 * 0.40
    sep = "=" * 78
    print(f"\n{sep}")
    print(f"  TODAY'S GEX REVERSAL LEVELS  ({len(levels)} zones)")
    print(f"  Last close: {last_close:.2f}   ATR20: {atr20:.1f} pts   Req swing: >{req_move:.0f} pts")
    print(sep)
    print(f"  {'#':>2}  {'PRICE':>9}  {'ROLE':>6}  {'SCORE':>6}  {'TF':>2}  {'DIST':>8}  TYPES")
    print(sep)
    for i, z in enumerate(levels, 1):
        dist = z.price - last_close
        role = "RESIST" if dist > 0 else "SUPPRT"
        side = "▲" if dist > 0 else "▼"
        fts  = "|".join(sorted(z.ftypes))
        print(f"  {i:>2}. {z.price:>9.2f}  {role:>6}  {z.score:>6.1f}  {z.n_tf:>2}  "
              f"  {side}{abs(dist):>5.1f} pts  {fts}")
    print(sep)


# ─── Backtest ─────────────────────────────────────────────────────────────────

def backtest(df: pd.DataFrame, zones: list,
             tolerance: float = BIN_SIZE,
             years: int = 3,
             lookback_bars: int = 60,
             forward_bars: int = 390,
             min_sep_bars: int = 240,
             reversal_pct: float = 0.006,
             quiet: bool = False) -> pd.DataFrame:
    """
    3-year reversal backtest. Identical methodology to stacked_vp.backtest():
    touch = wick enters zone ± tol AND close within 4×tol.
    Reversal = price moves ≥ reversal_pct in approach-opposite direction
               within forward_bars bars.
    """
    last_date  = df["date"].max()
    start_date = last_date - pd.Timedelta(days=int(years * 365.25))
    df3 = df[df["date"] >= start_date].reset_index(drop=True)

    hi  = df3["high"].values.astype(np.float64)
    lo  = df3["low"].values.astype(np.float64)
    cl  = df3["close"].values.astype(np.float64)
    n   = len(df3)
    if not quiet:
        print(f"  Backtest: {df3['date'].iloc[0].date()} → {df3['date'].iloc[-1].date()}  ({n:,} bars)")

    rows = []
    for zone in zones:
        p   = zone.price
        tol = tolerance

        close_near  = np.abs(cl - p) <= tol * 4
        raw_touches = np.where((lo <= p + tol) & (hi >= p - tol) & close_near)[0]

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
            tp        = cl[t]
            threshold = reversal_pct * tp
            prev_cl   = cl[max(0, t - lookback_bars)]

            if prev_cl > p + tol:
                direction = "resistance"
            elif prev_cl < p - tol:
                direction = "support"
            else:
                direction = "ambiguous"

            fwd_end = min(n, t + forward_bars + 1)
            if fwd_end <= t:
                continue
            fwd_hi = hi[t:fwd_end].max()
            fwd_lo = lo[t:fwd_end].min()

            if direction == "resistance":
                move = tp - fwd_lo
            elif direction == "support":
                move = fwd_hi - tp
            else:
                move = max(tp - fwd_lo, fwd_hi - tp)

            if move >= threshold:
                reversals += 1
                rev_moves.append(move / tp * 100)

        rows.append({
            "price":         round(zone.price, 2),
            "score":         round(zone.score, 1),
            "n_tf":          zone.n_tf,
            "timeframes":    ",".join(sorted(zone.timeframes)),
            "ftypes":        "|".join(sorted(zone.ftypes)),
            "touches":       total,
            "reversals":     reversals,
            "reversal_rate": round(reversals / total, 3) if total > 0 else 0.0,
            "avg_rev_pct":   round(float(np.mean(rev_moves)), 3) if rev_moves else 0.0,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("reversal_rate", ascending=False)


def _baseline_rate(df, n_samples, tolerance, forward_bars, min_sep_bars,
                   reversal_pct, years):
    last_date  = df["date"].max()
    start_date = last_date - pd.Timedelta(days=int(years * 365.25))
    df3 = df[df["date"] >= start_date].reset_index(drop=True)
    hi  = df3["high"].values.astype(np.float64)
    lo  = df3["low"].values.astype(np.float64)
    cl  = df3["close"].values.astype(np.float64)
    n   = len(df3)

    all_rates, all_touches = [], []
    rng    = np.random.default_rng(seed=42)
    prices = rng.uniform(lo.min(), hi.max(), size=n_samples * 3)
    tested = 0
    for p in prices:
        if tested >= n_samples:
            break
        close_near = np.abs(cl - p) <= tolerance * 4
        raw = np.where((lo <= p + tolerance) & (hi >= p - tolerance) & close_near)[0]
        deduped = []
        last_t  = -min_sep_bars
        for t in raw:
            if t - last_t >= min_sep_bars:
                deduped.append(t); last_t = t
        if len(deduped) < 10:
            continue
        revs = 0
        for t in deduped:
            tp  = cl[t]; thr = reversal_pct * tp
            pc  = cl[max(0, t - 30)]
            fwd_end = min(n, t + forward_bars + 1)
            fwd_hi  = hi[t:fwd_end].max()
            fwd_lo  = lo[t:fwd_end].min()
            if pc > p + tolerance:
                move = tp - fwd_lo
            elif pc < p - tolerance:
                move = fwd_hi - tp
            else:
                move = max(tp - fwd_lo, fwd_hi - tp)
            if move >= thr:
                revs += 1
        all_rates.append(revs / len(deduped))
        all_touches.append(len(deduped))
        tested += 1

    if not all_rates:
        return 0.0
    total_t = sum(all_touches)
    return float(sum(r * t for r, t in zip(all_rates, all_touches)) / total_t)


def print_backtest_report(df, bt, reversal_pct=0.006, tolerance=BIN_SIZE,
                           forward_bars=390, min_sep_bars=240):
    print("  Computing baseline (30 random levels)...", end="", flush=True)
    baseline = _baseline_rate(df, 30, tolerance, forward_bars, min_sep_bars,
                              reversal_pct, years=3)
    print(f"  {baseline:.1%}")

    sep = "-" * 96
    print(f"\n{sep}")
    print(f"  GEX BACKTEST  threshold={reversal_pct*100:.2f}%  |  {forward_bars}-bar window  |  baseline={baseline:.1%}")
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
    print(f"\n  Random baseline         : {baseline:.1%}")
    print(f"  All-zone weighted rate  : {zone_rate:.1%}")
    print(f"  Lift over baseline      : {lift:+.1%}")
    print(f"  Zones beating +5%       : {len(above)} of {len(bt)}")
    print(f"  Zones beating +8%       : {sum(1 for _, r in bt.iterrows() if r['reversal_rate'] - baseline >= 0.08)}  (* in table)")

    # Breakdown by feature type
    print(f"\n  Feature type breakdown:")
    for ftype in ["gflip", "gwall", "ledge", "shelf", "vah", "val", "poc", "lvn",
                  "ibh", "ibl", "pdh", "pdl", "round"]:
        sub = bt[bt["ftypes"].str.contains(ftype, na=False)]
        if len(sub) == 0:
            continue
        t_total = sub["touches"].sum()
        if t_total == 0:
            continue
        wr = float((sub["reversal_rate"] * sub["touches"]).sum() / t_total)
        lift_f = wr - baseline
        print(f"    {ftype:<8}  zones={len(sub):>2}  touches={t_total:>5}  "
              f"rate={wr:.1%}  lift={lift_f:+.1%}")


# ─── HOD/LOD Coverage ────────────────────────────────────────────────────────

def hod_lod_coverage(df: pd.DataFrame, vix_series: pd.Series,
                     n_days: int = 60, tolerance_pct: float = 0.0012,
                     bin_size: float = BIN_SIZE, seed: int = 42) -> dict:
    """
    Walk-forward HOD/LOD coverage test. Identical to stacked_vp methodology.
    Zones computed from strictly prior data for each sample day.
    """
    rng  = np.random.default_rng(seed)
    df2  = df.copy()
    df2["_day"] = df2["date"].dt.date

    day_stats = df2.groupby("_day").agg(
        bars=("close", "count"), hod=("high", "max"), lod=("low", "min"),
    )
    all_days_list = sorted(day_stats.index.tolist())
    eligible = [d for d in all_days_list
                if day_stats.loc[d, "bars"] >= 300
                and all_days_list.index(d) >= 90]
    if not eligible:
        print("  No eligible days"); return {}

    sample = sorted(rng.choice(eligible, size=min(n_days, len(eligible)),
                               replace=False).tolist())

    hod_hits = lod_hits = both_hits = total = 0
    rows = []
    sep = "-" * 72
    print(f"\n  GEX HOD/LOD coverage  ({len(sample)} days, tol={tolerance_pct*100:.2f}% of price)")
    print(sep)
    print(f"  {'DATE':<12}  {'HOD':>8}  {'LOD':>8}  {'VIX':>5}  {'dHOD':>6}  {'dLOD':>6}  HOD  LOD")
    print(sep)

    for day in sample:
        df_prior  = df2[df2["_day"] < day]
        df_window = df_prior.tail(30_000).copy()
        if len(df_window) < 2000:
            continue

        zones, _, _ = run(df_window, vix_series, bin_size=bin_size, quiet=True)
        if not zones:
            continue

        vix_val = get_vix_at(vix_series, pd.Timestamp(day) - pd.Timedelta(days=1))
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
        print(f"  {str(day):<12}  {hod:>8.1f}  {lod:>8.1f}  {vix_val:>5.1f}  "
              f"{hod_dist:>6.1f}  {lod_dist:>6.1f}   {h_flag}    {l_flag}")
        rows.append(dict(day=str(day), hod=hod, lod=lod, vix=round(vix_val, 1),
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

    pd.DataFrame(rows).to_csv(
        Path(__file__).parent / "hod_lod_coverage.csv", index=False
    )
    print("  Saved → hod_lod_coverage.csv")
    return dict(hod_rate=hod_hits/total, lod_rate=lod_hits/total,
                both_rate=both_hits/total, avg_hod_dist=avg_hd,
                avg_lod_dist=avg_ld, n_days=total)


# ─── Charts ───────────────────────────────────────────────────────────────────

def _draw_ohlc(ax, window: pd.DataFrame):
    n = len(window)
    x = np.arange(n)
    o = window["open"].values
    h = window["high"].values
    l = window["low"].values
    c = window["close"].values
    up = c >= o

    wick_segs = [[(i, l[i]), (i, h[i])] for i in range(n)]
    wick_cols = ["#3fb950" if up[i] else "#f85149" for i in range(n)]
    ax.add_collection(LineCollection(wick_segs, colors=wick_cols,
                                     linewidths=0.55, zorder=1))

    up_segs = [[(i, o[i]), (i, c[i])] for i in range(n) if up[i]]
    dn_segs = [[(i, c[i]), (i, o[i])] for i in range(n) if not up[i]]
    if up_segs:
        ax.add_collection(LineCollection(up_segs, colors="#3fb950",
                                         linewidths=3.0, zorder=2))
    if dn_segs:
        ax.add_collection(LineCollection(dn_segs, colors="#f85149",
                                         linewidths=3.0, zorder=2))


def _select_day_zones(zones, day_lo, day_hi, day_open,
                      n_min=4, n_max=6) -> list:
    n_target = 5
    qual = [z for z in zones if z.n_tf >= 2 and day_lo <= z.price <= day_hi]
    if len(qual) < n_min:
        qual = [z for z in zones if day_lo <= z.price <= day_hi]
    if not qual:
        return []
    if len(qual) <= n_max:
        return sorted(qual, key=lambda z: z.price)

    above   = [z for z in qual if z.price > day_open]
    below   = [z for z in qual if z.price <= day_open]
    n_above = math.ceil(n_target / 2)
    n_below = n_target - n_above
    sel_abv = above[:n_above]
    sel_blw = below[:n_below]
    if len(sel_abv) < n_above:
        sel_blw = below[:n_target - len(sel_abv)]
    elif len(sel_blw) < n_below:
        sel_abv = above[:n_target - len(sel_blw)]

    return sorted(sel_abv + sel_blw, key=lambda z: z.price)


def plot_sample_days(df: pd.DataFrame, vix_series: pd.Series,
                     n: int = 15, bin_size: float = BIN_SIZE,
                     seed: int = None, df_sample_pool=None,
                     output_dir: Path = None):
    """
    Generate n no-look-ahead GEX day charts.
    Cyan colormap (vs VP's magenta) so charts are visually distinct.
    Gold dotted lines = actual HOD/LOD.
    """
    CYAN_CMAP = LinearSegmentedColormap.from_list(
        "cyan_conf", ["#99FFFF", "#006688"]
    )

    if seed is not None:
        np.random.seed(seed)

    out_dir = output_dir or Path(__file__).parent
    df2 = df.copy()
    df2["_day"] = df2["date"].dt.date

    pool = df_sample_pool.copy() if df_sample_pool is not None else df2
    pool["_day"] = pool["date"].dt.date

    day_stats = pool.groupby("_day").agg(
        bars=("close", "count"),
        lo=("low", "min"), hi=("high", "max"), op=("open", "first"),
    )
    all_days_full = sorted(df2["_day"].unique().tolist())
    all_days      = day_stats.index.tolist()
    full_days     = [d for d in all_days
                     if day_stats.loc[d, "bars"] >= 300
                     and all_days_full.index(d) >= 90]

    if not full_days:
        print("  No eligible days found"); return

    rng        = np.random.default_rng(seed=seed)
    candidates = sorted(rng.choice(full_days,
                                   size=min(n * 4, len(full_days)),
                                   replace=False).tolist())

    dark      = "#0D1117"
    generated = 0

    print(f"\n  Generating {n} GEX day charts (no look-ahead)...")

    for day in candidates:
        if generated >= n:
            break

        df_prior  = df2[df2["_day"] < day]
        df_window = df_prior.tail(30_000).copy()
        if len(df_window) < 2000:
            continue

        day_zones, _, sigma_nq = run(df_window, vix_series,
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
            continue

        vix_val   = get_vix_at(vix_series, pd.Timestamp(day) - pd.Timedelta(days=1))
        max_score = max(z.score for z in day_zones)

        fig, ax = plt.subplots(figsize=(18, 7), facecolor=dark)
        ax.set_facecolor(dark)
        for sp in ax.spines.values():
            sp.set_color("#21262D")

        _draw_ohlc(ax, bars)

        # HOD / LOD gold reference lines
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
            colour = CYAN_CMAP(0.20 + 0.80 * ns)
            hw     = zone.price * 0.0005   # 0.05% of price, same as stacked_vp
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
            f"NQ GEX  {day_str}  ·  {nb} bars  ·  {len(selected)} zones  "
            f"·  VIX={vix_val:.1f}  σ_NQ={sigma_nq:.3f}  "
            f"(no look-ahead · darker = higher confidence)",
            color="#F0F6FC", fontsize=9.5, pad=8, fontweight="bold",
        )

        plt.tight_layout(pad=0.5)
        out = out_dir / f"gex_{day_str}.png"
        fig.savefig(str(out), dpi=130, bbox_inches="tight", facecolor=dark)
        plt.close(fig)
        generated += 1
        print(f"  [{generated:02d}/{n}]  {day_str}  VIX={vix_val:.1f}  "
              f"bars={nb:4d}  zones={len(selected)}  → gex_{day_str}.png")

    print(f"  Done.  ({generated}/{n} charts)")


# ─── GEX Profile Curve Plot ───────────────────────────────────────────────────

def plot_gex_curves(profiles: dict, zones: list, last_close: float,
                    sigma_nq: float):
    """Show raw GEX curves for all three DTE windows side by side."""
    dark = "#0D1117"
    n_profiles = len(profiles)
    if n_profiles == 0:
        return

    fig, axes = plt.subplots(1, n_profiles, figsize=(6 * n_profiles, 8),
                              facecolor=dark, sharey=True)
    if n_profiles == 1:
        axes = [axes]

    DTE_COLORS = {
        "gex_5d":  "#FF6B6B",
        "gex_21d": "#FFD966",
        "gex_63d": "#4EC9B0",
    }

    for ax, (name, vp) in zip(axes, profiles.items()):
        ax.set_facecolor(dark)
        for sp in ax.spines.values():
            sp.set_color("#21262D")

        n   = len(vp.vols)
        win = max(5, n // 15)
        if win % 2 == 0: win += 1
        win = min(win, n - 1 if n % 2 == 0 else n)
        try:
            smooth = savgol_filter(vp.vols, win, 2).clip(0)
        except Exception:
            smooth = vp.vols.copy()

        colour = DTE_COLORS.get(name, "#AAAAAA")
        ax.fill_betweenx(vp.bins, smooth, alpha=0.30, color=colour)
        ax.plot(smooth, vp.bins, color=colour, lw=1.2, alpha=0.85)

        # Zone markers
        visible = [z for z in zones if vp.bins[0] <= z.price <= vp.bins[-1]]
        for z in visible:
            ns = z.score / max((zz.score for zz in zones), default=1)
            ax.axhline(z.price, color="#FF88FF", lw=0.7 + 0.8 * ns,
                       alpha=0.4 + 0.4 * ns, ls="--")

        ax.axhline(last_close, color="#FFFFFF", lw=0.8, alpha=0.5, ls=":")
        ax.axhline(vp.vah, color=colour, lw=0.6, alpha=0.4, ls="--")
        ax.axhline(vp.val, color=colour, lw=0.6, alpha=0.4, ls="--")
        ax.axhline(vp.poc, color=colour, lw=1.0, alpha=0.6, ls="-")

        dte_label = name.replace("gex_", "DTE=").replace("d", "d ")
        ax.set_title(f"{dte_label}  (σ_NQ={sigma_nq:.3f})",
                     color="#F0F6FC", fontsize=9)
        ax.tick_params(colors="#8B949E", labelsize=7)
        ax.set_xlabel("GEX Volume", color="#8B949E", fontsize=8)

    axes[0].set_ylabel("Price  (NQ)", color="#8B949E", fontsize=9)
    fig.suptitle(f"NQ GEX Profiles — spot={last_close:.0f}  VIX-implied σ_NQ={sigma_nq:.3f}",
                 color="#F0F6FC", fontsize=11, fontweight="bold")
    plt.tight_layout(pad=0.8)
    out = Path(__file__).parent / "gex_curves.png"
    fig.savefig(str(out), dpi=130, bbox_inches="tight", facecolor=dark)
    print(f"\n  GEX curve plot → {out}")
    plt.show()


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NQ GEX Profile Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--backtest",    action="store_true",
                        help="3-year reversal backtest on GEX zones")
    parser.add_argument("--hod-lod",     action="store_true",
                        help="Walk-forward HOD/LOD coverage test (60 days)")
    parser.add_argument("--hod-days",    type=int, default=60)
    parser.add_argument("--sample-days", type=int, default=0,
                        help="Generate N no-look-ahead day charts")
    parser.add_argument("--seed",        type=int, default=None)
    parser.add_argument("--top",         type=int, default=20)
    parser.add_argument("--bins",        type=float, default=BIN_SIZE)
    parser.add_argument("--rev-pct",     type=float, default=0.60)
    parser.add_argument("--curves",      action="store_true",
                        help="Plot raw GEX curve profiles")
    parser.add_argument("--daily",       type=int, default=5)
    args = parser.parse_args()

    print("=" * 56)
    print("  NQ GEX PROFILE ANALYZER")
    print("=" * 56)

    print("\nLoading data...")
    df         = load_nq_data()
    vix_series = load_vix()
    print(f"  NQ: {len(df):,} bars  {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  VIX: {len(vix_series)} days")

    if args.hod_lod:
        print(f"\nRunning GEX HOD/LOD coverage test ({args.hod_days} days)...")
        seed = args.seed if args.seed else 42
        hod_lod_coverage(df, vix_series, n_days=args.hod_days,
                         bin_size=args.bins, seed=seed)
        return

    print("\nBuilding GEX profiles...")
    zones, profiles, sigma_nq = run(df, vix_series, bin_size=args.bins)
    print_report(zones, show_n=args.top, sigma_nq=sigma_nq)

    n_daily = max(4, min(6, args.daily))
    levels, last_close, atr20 = daily_levels(df, zones, n=n_daily)
    print_daily_levels(levels, last_close, atr20)

    if args.curves:
        plot_gex_curves(profiles, zones, last_close, sigma_nq)

    if args.sample_days > 0:
        plot_sample_days(df, vix_series, n=args.sample_days,
                         bin_size=args.bins, seed=args.seed)

    if args.backtest:
        rev_threshold = args.rev_pct / 100.0
        BT_FWD = 390
        BT_SEP = 240
        print(f"\nRunning GEX backtest (threshold={args.rev_pct}%)...")
        bt = backtest(df, zones, tolerance=args.bins,
                      reversal_pct=rev_threshold,
                      forward_bars=BT_FWD, min_sep_bars=BT_SEP)
        if not bt.empty:
            print_backtest_report(df, bt, reversal_pct=rev_threshold,
                                  tolerance=args.bins,
                                  forward_bars=BT_FWD, min_sep_bars=BT_SEP)
            bt.to_csv(Path(__file__).parent / "gex_backtest.csv", index=False)
            print("  Saved → gex_backtest.csv")


if __name__ == "__main__":
    main()
