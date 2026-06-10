"""
0DTE GEX Profile Analyzer for NQ Futures
==========================================
Builds stacked 0-DTE Gamma Exposure profiles from the past 5 sessions.

Core idea
---------
For each of the last N=5 prior sessions independently:
  1. Take that session's intraday 1-minute bars (volume at price)
  2. Build raw VP (uniform OHLCV distribution into 5-pt bins)
  3. Apply 0DTE gamma kernel: GEX(K) = VP_vol(K) × Γ(S=session_close, K, σ_NQ, T=1/252)
  4. Detect HVN / LVN / shelf / ledge / gwall / gflip on the GEX curve
  5. Assign timeframe = "0dte_d-1" … "0dte_d-5"

Why 0DTE?
  T=1/252 → gamma is tightest, most concentrated near ATM.
  Any VP level within ±1σ daily move of that session's close gets high weight.
  Levels farther out get near-zero weight (0DTE gamma decays steeply).

Why stack across 5 sessions?
  A level with high 0DTE GEX on d-1 only = single data point (score × 1² = low).
  Same level appearing on d-1, d-2, d-3 → n_tf=3 → score × 3² = 9× boost.
  "Sticky" levels that matter repeatedly to 0DTE dealers = highest confidence.
  This is identical logic to stacked_vp's multi-TF stacking, applied across time.

Feature types
  vah / val  weight 1.2   70% GEX value area boundaries
  poc        weight 1.1   highest-GEX bin per session
  lvn        weight 0.9   thin zone — fast travel or reversal magnet
  shelf      weight 0.8   gradual GEX boundary (enabled — GEX bell-curve shoulder)
  ledge      weight 1.3   abrupt GEX cliff (strongest 0DTE boundary)
  gwall      weight 1.1   steepest |dGEX/dK| — fastest dealer response boundary
  gflip      weight 1.3   GEX inflection — dealer behaviour mode change

Structural levels (same as stacked_vp)
  pdh / pdl  weight 1.4   prior day H/L  (timeframe d-1..d-10)
  ibh / ibl  weight 1.6   initial balance H/L
  onh / onl  weight 1.2   overnight H/L
  gap        weight 1.1   prior RTH close
  pwh / pwl  weight 1.2   prior week H/L
  pmh / pml  weight 1.3   prior month H/L
  round      weight 1.5/1.0  500pt / 100pt psychological levels

Usage
-----
  python gex_profile.py                    # daily levels
  python gex_profile.py --sample-days 15  # 15 no-look-ahead charts
  python gex_profile.py --backtest        # 3-year reversal backtest
  python gex_profile.py --hod-lod        # walk-forward HOD/LOD coverage (60 days)
  python gex_profile.py --curves         # plot raw GEX curves for each session
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
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths, savgol_filter

warnings.filterwarnings("ignore")

CYAN_CMAP = LinearSegmentedColormap.from_list("cyan_conf", ["#99FFFF", "#006688"])

# ─── Paths ────────────────────────────────────────────────────────────────────
CSV_PATH   = Path(__file__).parent.parent / "1Min_NQ.csv"
CACHE_PATH = Path(__file__).parent / "hdata" / "NQ_1m_cache.parquet"
VIX_CACHE  = Path(__file__).parent / "hdata" / "vix_daily.parquet"

# ─── Parameters ───────────────────────────────────────────────────────────────
BIN_SIZE          = 5.0
VALUE_AREA_PCT    = 0.70
HVN_PROMINENCE    = 0.30   # fraction of peak GEX
LVN_DEPTH         = 0.40   # fraction of mean GEX
SHELF_SLOPE_MAX   = 0.08
LEDGE_SLOPE_MIN   = 0.20
SLOPE_BINS        = 5
STACK_TOLERANCE   = 3.0    # cluster radius = 3 × 5 = 15 pts
GWALL_PROMINENCE  = 0.25
GFLIP_MIN_GEX     = 0.20
NQ_IV_MULT        = 1.15   # NQ IV ≈ VIX × 1.15
INCLUDE_SHELF     = True
INCLUDE_LEDGE     = True
USE_GWALL         = True
USE_GFLIP         = True
N_SESSIONS        = 5      # prior sessions for 0DTE stack (1 calendar week)
MINS_PER_DAY      = 390
T_0DTE            = 1.0 / 252.0   # 0DTE = 1 trading day in years


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class VolumeProfile:
    name:  str
    bins:  np.ndarray
    vols:  np.ndarray   # GEX-weighted volume per bin
    poc:   float
    vah:   float
    val:   float
    spot:  float = 0.0  # session's closing spot (center of gamma kernel)
    sigma: float = 0.0  # σ_NQ used for this session


@dataclass
class VPFeature:
    price:     float
    ftype:     str
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
    if VIX_CACHE.exists():
        s = pd.read_parquet(VIX_CACHE)["vix"]
        print(f"  VIX cache → {len(s)} days")
        return s

    vix_series = None
    try:
        import yfinance as yf
        raw = yf.download("^VIX", start="2019-01-01", end="2026-12-31",
                          progress=False, auto_adjust=True)
        if len(raw) > 100:
            vix_series = raw["Close"].rename("vix")
            vix_series.index = pd.to_datetime(vix_series.index).tz_localize(None)
            print(f"  VIX yfinance: {len(vix_series)} days")
    except Exception as e:
        print(f"  yfinance unavail ({e}), using realised vol fallback")

    if vix_series is None:
        df_nq = load_nq_data()
        df_nq["_d"] = df_nq["date"].dt.date
        daily_cl = df_nq.groupby("_d")["close"].last()
        daily_cl.index = pd.to_datetime(daily_cl.index)
        log_ret = np.log(daily_cl / daily_cl.shift(1)).dropna()
        rvol = log_ret.rolling(20).std() * np.sqrt(252) * 100
        vix_series = rvol.rename("vix").dropna()
        print(f"  VIX fallback (realised vol 20d): {len(vix_series)} days")

    df_v = pd.DataFrame({"vix": vix_series})
    df_v.index = pd.to_datetime(df_v.index)
    VIX_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df_v.to_parquet(VIX_CACHE)
    return df_v["vix"]


def get_vix_at(vix_series: pd.Series, as_of_date) -> float:
    as_of = pd.Timestamp(as_of_date)
    prior = vix_series[vix_series.index <= as_of]
    return float(prior.iloc[-1]) if not prior.empty else 20.0


# ─── Black-Scholes Gamma ──────────────────────────────────────────────────────

def bs_gamma(S: float, K: float, T: float, sigma: float, r: float = 0.03) -> float:
    if T <= 1e-9 or sigma <= 1e-9 or S <= 0 or K <= 0:
        return 0.0
    try:
        sqT = math.sqrt(T)
        d1  = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqT)
        nd1 = math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
        return nd1 / (S * sigma * sqT)
    except (ValueError, ZeroDivisionError, OverflowError):
        return 0.0


# ─── Raw VP Construction ──────────────────────────────────────────────────────

def _build_raw_vp(bars: pd.DataFrame, bin_size: float) -> tuple:
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


# ─── 0DTE GEX Profile per Session ────────────────────────────────────────────

def build_0dte_session(session_bars: pd.DataFrame, spot_close: float,
                        sigma_nq: float, session_name: str,
                        bin_size: float) -> VolumeProfile:
    """
    0DTE GEX profile for one prior session.

    Uses that session's own intraday bars as the volume base — capturing exactly
    where 0DTE contracts were most actively hedged during that session.

    Gamma kernel T = 1/252 (1 trading day): extremely tight bell centered on
    that session's closing price. Volume at K matters only if K is within
    roughly ±2σ_daily of spot_close (σ_daily = σ_NQ / sqrt(252)).

    At NQ=22000, VIX=18 (σ_NQ≈0.207):
      σ_daily ≈ 22000 × 0.207 / sqrt(252) ≈ 287 pts
      1σ range: [21713, 22287] — nearly full gamma weight
      2σ range: [21426, 22574] — ~13% gamma weight
      3σ range: [21139, 22861] — ~0.3% gamma weight
    """
    bins, vp_vols = _build_raw_vp(session_bars, bin_size)

    gammas = np.array([bs_gamma(spot_close, float(K), T_0DTE, sigma_nq)
                       for K in bins])
    gex_raw = vp_vols * gammas

    vp_max  = vp_vols.max()
    gex_max = gex_raw.max()
    if gex_max > 1e-12 and vp_max > 1e-12:
        gex_vols = gex_raw * (vp_max / gex_max)
    else:
        gex_vols = gex_raw.copy()

    if gex_vols.sum() < 1e-9:
        gex_vols = vp_vols.copy()  # fallback: use raw VP if gamma is degenerate

    poc_idx = int(np.argmax(gex_vols))
    poc     = float(bins[poc_idx])

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

    return VolumeProfile(name=session_name, bins=bins, vols=gex_vols,
                         poc=poc, vah=vah, val=val,
                         spot=spot_close, sigma=sigma_nq)


# ─── Feature Detection ────────────────────────────────────────────────────────

def detect_features(vp: VolumeProfile, bin_size: float = BIN_SIZE) -> list:
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
    if win % 2 == 0: win += 1
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
        _classify_boundary(smooth, bins, l_idx, "lower", peak_vol, tf, feats)
        _classify_boundary(smooth, bins, r_idx, "upper", peak_vol, tf, feats)

    # LVN valleys
    valleys, _ = find_peaks(-smooth, prominence=0.08 * peak_vol)
    for v in valleys:
        if smooth[v] < LVN_DEPTH * mean_vol:
            feats.append(VPFeature(float(bins[v]), "lvn", tf, weight=0.9))

    return feats


def _classify_boundary(smooth, bins, edge_idx, side, peak_vol, tf, feats):
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
    price     = float(bins[edge_idx])

    if avg_slope <= SHELF_SLOPE_MAX and INCLUDE_SHELF:
        feats.append(VPFeature(price, "shelf", tf, weight=0.8))
    elif avg_slope >= LEDGE_SLOPE_MIN and INCLUDE_LEDGE:
        feats.append(VPFeature(price, "ledge", tf, weight=1.3))


def detect_gex_features(vp: VolumeProfile, bin_size: float = BIN_SIZE) -> list:
    """
    gwall: steepest |dGEX/dK| peaks — fastest dealer hedging transitions
    gflip: GEX curve inflection points — dealer mode changes (up to 2 per session)
    """
    feats: list[VPFeature] = []
    bins, vols = vp.bins, vp.vols
    tf = vp.name
    n  = len(vols)
    if n < 15:
        return feats

    win = max(7, n // 12)
    if win % 2 == 0: win += 1
    win = min(win, n - 1 if n % 2 == 0 else n)
    try:
        smooth = savgol_filter(vols, window_length=win, polyorder=3).clip(0)
    except Exception:
        smooth = vols.copy()

    peak_vol = smooth.max()
    if peak_vol < 1e-9:
        return feats

    grad     = np.gradient(smooth, bin_size)
    abs_grad = np.abs(grad)
    ag_max   = abs_grad.max()

    if USE_GWALL and ag_max > 1e-12:
        wall_peaks, props = find_peaks(
            abs_grad, prominence=GWALL_PROMINENCE * ag_max,
            distance=max(3, n // 20)
        )
        if len(wall_peaks) > 0:
            proms = props["prominences"]
            order = np.argsort(proms)[::-1][:3]
            for idx in order:
                p = wall_peaks[idx]
                if smooth[p] >= 0.15 * peak_vol:
                    feats.append(VPFeature(float(bins[p]), "gwall", tf, weight=1.1))

    if USE_GFLIP:
        d2      = np.gradient(grad, bin_size)
        sign_ch = np.where(np.diff(np.sign(d2)))[0]
        cands   = []
        for i in sign_ch:
            local_gex  = float(smooth[i])
            local_grad = float(abs_grad[i])
            if local_gex >= GFLIP_MIN_GEX * peak_vol:
                cands.append((local_gex * local_grad, i))
        cands.sort(reverse=True)
        for _, i in cands[:2]:
            feats.append(VPFeature(float(bins[i]), "gflip", tf, weight=1.3))

    return feats


# ─── Structural Levels (same as stacked_vp) ──────────────────────────────────

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
    feats: list[VPFeature] = []
    df2 = df.copy()
    df2["_d"]  = df2["date"].dt.date
    df2["_tm"] = df2["date"].dt.hour * 60 + df2["date"].dt.minute

    all_days   = sorted(df2["_d"].unique())
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
        pm = df2[df2["_ym"] == months[-2]]
        feats.append(VPFeature(float(pm["high"].max()), "pmh", "prior_month", weight=1.3))
        feats.append(VPFeature(float(pm["low"].min()),  "pml", "prior_month", weight=1.3))

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


# ─── Main Analysis Driver ─────────────────────────────────────────────────────

def run(df: pd.DataFrame, vix_series: pd.Series,
        bin_size: float = BIN_SIZE, quiet: bool = False,
        n_sessions: int = N_SESSIONS):
    """
    Build N 0DTE GEX profiles (one per prior session) + structural levels,
    then stack all features. Sessions get timeframes "0dte_d-1"…"0dte_d-N".
    A level repeated across 3 sessions scores 9× more than a single-session level.
    """
    df2 = df.copy()
    df2["_d"] = df2["date"].dt.date
    all_days  = sorted(df2["_d"].unique())

    # Exclude current (potentially partial) session — use only completed prior sessions
    prior_days = all_days[:-1][-n_sessions:]
    if len(prior_days) == 0:
        return [], {}, 0.0

    all_features: list[VPFeature] = []
    profiles: dict = {}

    if not quiet:
        print(f"  0DTE sessions: {[str(d) for d in prior_days]}")

    for i, day in enumerate(reversed(prior_days)):   # i=0 = most recent
        session_name = f"0dte_d-{i+1}"
        day_bars = df2[df2["_d"] == day].copy()
        if len(day_bars) < 50:
            continue

        spot_close = float(day_bars["close"].iloc[-1])
        day_vix    = get_vix_at(vix_series, pd.Timestamp(day))
        sigma_nq   = (day_vix / 100.0) * NQ_IV_MULT

        gex_vp = build_0dte_session(day_bars, spot_close, sigma_nq,
                                     session_name, bin_size)
        profiles[session_name] = gex_vp

        feats  = detect_features(gex_vp, bin_size)
        feats += detect_gex_features(gex_vp, bin_size)
        all_features.extend(feats)

        if not quiet:
            gw = sum(1 for f in feats if f.ftype == "gwall")
            gf = sum(1 for f in feats if f.ftype == "gflip")
            print(f"  [{session_name}]  {day}  close={spot_close:.1f}  "
                  f"VIX={day_vix:.1f}  σ={sigma_nq:.3f}  "
                  f"feats={len(feats)}  gwall={gw}  gflip={gf}")

    # Structural levels
    all_features.extend(prior_day_features(df))
    all_features.extend(session_level_features(df))
    atr20 = compute_atr20(df)
    all_features.extend(round_number_features(df, atr20))

    zones    = stack_features(all_features, bin_size)
    avg_vix  = float(np.mean([
        get_vix_at(vix_series, pd.Timestamp(d)) for d in prior_days
    ]))
    return zones, profiles, avg_vix


# ─── Console Output ───────────────────────────────────────────────────────────

def print_report(zones: list, show_n: int = 20, avg_vix: float = 0.0):
    sep = "-" * 80
    print(f"\n{sep}")
    print(f"  TOP {min(show_n, len(zones))} 0DTE-GEX CONFLUENCE ZONES   "
          f"(avg_VIX={avg_vix:.1f})")
    print(sep)
    print(f"  {'PRICE':>9}  {'SCORE':>6}  {'TF':>2}  0DTE_SESS  FEATURE TYPES")
    print(sep)
    for z in zones[:show_n]:
        dte_sess = sum(1 for t in z.timeframes if t.startswith("0dte"))
        tfs_str  = "|".join(t for t in sorted(z.timeframes) if t.startswith("0dte"))
        fts      = "|".join(sorted(z.ftypes))
        print(f"  {z.price:>9.2f}  {z.score:>6.1f}  {z.n_tf:>2}  "
              f"{dte_sess} sess  {fts}  [{tfs_str}]")
    print(sep)
    multi_dte = sum(1 for z in zones
                    if sum(1 for t in z.timeframes if t.startswith("0dte")) >= 2)
    print(f"  Zones with 2+ 0DTE sessions  : {multi_dte}")
    print(f"  Zones with 3+ 0DTE sessions  : "
          f"{sum(1 for z in zones if sum(1 for t in z.timeframes if t.startswith('0dte')) >= 3)}")
    print(f"  gwall-bearing zones          : "
          f"{sum(1 for z in zones if 'gwall' in z.ftypes)}")
    print(f"  gflip-bearing zones          : "
          f"{sum(1 for z in zones if 'gflip' in z.ftypes)}")
    print(f"  ledge-bearing zones          : "
          f"{sum(1 for z in zones if 'ledge' in z.ftypes)}")


def daily_levels(df: pd.DataFrame, zones: list, n: int = 5) -> tuple:
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


def print_daily_levels(levels, last_close, atr20):
    req = atr20 * 0.40
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  0DTE-GEX REVERSAL LEVELS  ({len(levels)} zones)")
    print(f"  Last close: {last_close:.2f}   ATR20: {atr20:.1f}   Req swing: >{req:.0f} pts")
    print(sep)
    print(f"  {'#':>2}  {'PRICE':>9}  {'ROLE':>6}  {'SCORE':>6}  {'TF':>2}  "
          f"{'0DTE_N':>6}  {'DIST':>8}  TYPES")
    print(sep)
    for i, z in enumerate(levels, 1):
        dist     = z.price - last_close
        role     = "RESIST" if dist > 0 else "SUPPRT"
        side     = "▲" if dist > 0 else "▼"
        fts      = "|".join(sorted(z.ftypes))
        dte_n    = sum(1 for t in z.timeframes if t.startswith("0dte"))
        print(f"  {i:>2}. {z.price:>9.2f}  {role:>6}  {z.score:>6.1f}  {z.n_tf:>2}  "
              f"  {dte_n:>2}sess  {side}{abs(dist):>5.1f} pts  {fts}")
    print(sep)


# ─── Backtest ─────────────────────────────────────────────────────────────────

def backtest(df, zones, tolerance=BIN_SIZE, years=3,
             lookback_bars=60, forward_bars=390, min_sep_bars=240,
             reversal_pct=0.006, quiet=False):
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
        p           = zone.price
        close_near  = np.abs(cl - p) <= tolerance * 4
        raw_touches = np.where((lo <= p + tolerance) & (hi >= p - tolerance) & close_near)[0]

        deduped: list[int] = []
        last_t = -min_sep_bars
        for t in raw_touches:
            if t - last_t >= min_sep_bars:
                deduped.append(t); last_t = t

        total = len(deduped); reversals = 0; rev_moves: list[float] = []
        for t in deduped:
            tp  = cl[t]; thr = reversal_pct * tp
            pc  = cl[max(0, t - lookback_bars)]
            if pc > p + tolerance:
                direction = "resistance"
            elif pc < p - tolerance:
                direction = "support"
            else:
                direction = "ambiguous"

            fwd_end = min(n, t + forward_bars + 1)
            if fwd_end <= t: continue
            fwd_hi = hi[t:fwd_end].max(); fwd_lo = lo[t:fwd_end].min()

            if direction == "resistance":   move = tp - fwd_lo
            elif direction == "support":    move = fwd_hi - tp
            else:                           move = max(tp - fwd_lo, fwd_hi - tp)

            if move >= thr:
                reversals += 1
                rev_moves.append(move / tp * 100)

        rows.append({
            "price":         round(zone.price, 2),
            "score":         round(zone.score, 1),
            "n_tf":          zone.n_tf,
            "n_0dte":        sum(1 for t in zone.timeframes if t.startswith("0dte")),
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
    rng = np.random.default_rng(seed=42)
    prices = rng.uniform(lo.min(), hi.max(), size=n_samples * 3)
    tested = 0
    for p in prices:
        if tested >= n_samples: break
        close_near = np.abs(cl - p) <= tolerance * 4
        raw = np.where((lo <= p + tolerance) & (hi >= p - tolerance) & close_near)[0]
        deduped = []; last_t = -min_sep_bars
        for t in raw:
            if t - last_t >= min_sep_bars:
                deduped.append(t); last_t = t
        if len(deduped) < 10: continue
        revs = 0
        for t in deduped:
            tp = cl[t]; thr = reversal_pct * tp
            pc = cl[max(0, t - 30)]
            fwd_end = min(n, t + forward_bars + 1)
            fwd_hi = hi[t:fwd_end].max(); fwd_lo = lo[t:fwd_end].min()
            if pc > p + tolerance: move = tp - fwd_lo
            elif pc < p - tolerance: move = fwd_hi - tp
            else: move = max(tp - fwd_lo, fwd_hi - tp)
            if move >= thr: revs += 1
        all_rates.append(revs / len(deduped))
        all_touches.append(len(deduped))
        tested += 1

    if not all_rates: return 0.0
    total_t = sum(all_touches)
    return float(sum(r * t for r, t in zip(all_rates, all_touches)) / total_t)


def print_backtest_report(df, bt, reversal_pct=0.006, tolerance=BIN_SIZE,
                           forward_bars=390, min_sep_bars=240):
    print("  Computing baseline (30 random levels)...", end="", flush=True)
    baseline = _baseline_rate(df, 30, tolerance, forward_bars, min_sep_bars,
                              reversal_pct, 3)
    print(f"  {baseline:.1%}")

    sep = "-" * 100
    print(f"\n{sep}")
    print(f"  0DTE-GEX BACKTEST  threshold={reversal_pct*100:.2f}%  "
          f"fwd={forward_bars}bars  baseline={baseline:.1%}")
    print(sep)
    print(f"  {'PRICE':>9}  {'SCORE':>5}  {'TF':>2}  {'0DTE':>4}  "
          f"{'TOUCH':>5}  {'REV':>4}  {'RATE':>6}  {'vs BASE':>7}  TYPES")
    print(sep)
    for _, r in bt.iterrows():
        edge = r["reversal_rate"] - baseline
        flag = " **" if edge >= 0.10 else (" *" if edge >= 0.05 else "   ")
        print(f"  {r['price']:>9.2f}  {r['score']:>5.1f}  {r['n_tf']:>2}  "
              f"{int(r['n_0dte']):>4}  {r['touches']:>5}  {r['reversals']:>4}  "
              f"{r['reversal_rate']:>5.1%}  {edge:>+6.1%}  {r['ftypes']}{flag}")
    print(sep)

    zone_rate = float((bt["reversal_rate"] * bt["touches"]).sum() / bt["touches"].sum())
    lift      = zone_rate - baseline
    print(f"\n  Random baseline           : {baseline:.1%}")
    print(f"  All-zone weighted rate    : {zone_rate:.1%}")
    print(f"  Lift over baseline        : {lift:+.1%}")
    print(f"  Zones beating +5%         : {int((bt['reversal_rate'] - baseline >= 0.05).sum())} / {len(bt)}")
    print(f"  Zones beating +10%        : {int((bt['reversal_rate'] - baseline >= 0.10).sum())} / {len(bt)}")

    print(f"\n  Feature-type breakdown (touch-weighted reversal rate vs baseline {baseline:.1%}):")
    ftypes_to_check = ["gflip", "gwall", "ledge", "shelf", "vah", "val", "poc",
                        "lvn", "ibh", "ibl", "pdh", "pdl", "round"]
    rows_out = []
    for ftype in ftypes_to_check:
        sub = bt[bt["ftypes"].str.contains(ftype, na=False)]
        if len(sub) == 0: continue
        t_total = sub["touches"].sum()
        if t_total == 0: continue
        wr     = float((sub["reversal_rate"] * sub["touches"]).sum() / t_total)
        lift_f = wr - baseline
        rows_out.append((lift_f, ftype, len(sub), t_total, wr))
    rows_out.sort(reverse=True)
    for lift_f, ftype, nz, tt, wr in rows_out:
        bar = "#" * int(max(0, lift_f) * 100) + ("." * int(max(0, -lift_f) * 20))
        print(f"    {ftype:<8}  zones={nz:>2}  touches={tt:>5}  "
              f"rate={wr:.1%}  lift={lift_f:+.1%}  {bar}")

    # 0DTE session count breakdown
    print(f"\n  0DTE session count vs reversal rate:")
    for n_dte in range(0, N_SESSIONS + 1):
        sub = bt[bt["n_0dte"] == n_dte]
        if len(sub) == 0: continue
        t_total = sub["touches"].sum()
        if t_total == 0: continue
        wr     = float((sub["reversal_rate"] * sub["touches"]).sum() / t_total)
        lift_f = wr - baseline
        print(f"    {n_dte} 0DTE sessions  zones={len(sub):>2}  "
              f"rate={wr:.1%}  lift={lift_f:+.1%}")


# ─── HOD/LOD Coverage ────────────────────────────────────────────────────────

def hod_lod_coverage(df: pd.DataFrame, vix_series: pd.Series,
                     n_days: int = 60, tolerance_pct: float = 0.0012,
                     bin_size: float = BIN_SIZE, seed: int = 42) -> dict:
    rng  = np.random.default_rng(seed)
    df2  = df.copy()
    df2["_day"] = df2["date"].dt.date

    day_stats = df2.groupby("_day").agg(
        bars=("close", "count"), hod=("high", "max"), lod=("low", "min"),
    )
    all_days_list = sorted(day_stats.index.tolist())
    eligible = [d for d in all_days_list
                if day_stats.loc[d, "bars"] >= 300
                and all_days_list.index(d) >= 10]  # need 5+ prior sessions
    if not eligible:
        print("  No eligible days"); return {}

    sample = sorted(rng.choice(eligible, size=min(n_days, len(eligible)),
                               replace=False).tolist())

    hod_hits = lod_hits = both_hits = total = 0
    rows = []
    sep = "-" * 76
    print(f"\n  0DTE-GEX HOD/LOD coverage  ({len(sample)} days, "
          f"tol={tolerance_pct*100:.2f}% ≈ {tolerance_pct*21000:.0f}pts@21k)")
    print(sep)
    print(f"  {'DATE':<12}  {'HOD':>8}  {'LOD':>8}  {'VIX':>5}  "
          f"{'dHOD':>6}  {'dLOD':>6}  HOD  LOD")
    print(sep)

    for day in sample:
        # Use data strictly before this day
        df_prior  = df2[df2["_day"] < day]
        df_window = df_prior.tail(30_000).copy()
        if len(df_window) < 500:
            continue

        zones, _, avg_vix = run(df_window, vix_series, bin_size=bin_size, quiet=True)
        if not zones:
            continue

        vix_val  = get_vix_at(vix_series, pd.Timestamp(day) - pd.Timedelta(days=1))
        zp       = np.array([z.price for z in zones])
        hod      = float(day_stats.loc[day, "hod"])
        lod      = float(day_stats.loc[day, "lod"])
        tol      = ((hod + lod) / 2) * tolerance_pct

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

    out = Path(__file__).parent / "hod_lod_coverage.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  Saved → {out}")
    return dict(hod_rate=hod_hits/total, lod_rate=lod_hits/total,
                both_rate=both_hits/total, avg_hod_dist=avg_hd,
                avg_lod_dist=avg_ld, n_days=total)


# ─── Sample Day Charts ────────────────────────────────────────────────────────

def _draw_ohlc(ax, window):
    n  = len(window)
    o  = window["open"].values
    h  = window["high"].values
    l  = window["low"].values
    c  = window["close"].values
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


def _select_day_zones(zones, day_lo, day_hi, day_open, n_min=4, n_max=6):
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
    full_days = [d for d in day_stats.index.tolist()
                 if day_stats.loc[d, "bars"] >= 300
                 and all_days_full.index(d) >= 10]

    if not full_days:
        print("  No eligible days"); return

    rng        = np.random.default_rng(seed=seed)
    candidates = sorted(rng.choice(full_days,
                                   size=min(n * 4, len(full_days)),
                                   replace=False).tolist())
    dark      = "#0D1117"
    generated = 0

    print(f"\n  Generating {n} 0DTE-GEX day charts (no look-ahead)...")

    for day in candidates:
        if generated >= n:
            break

        df_prior  = df2[df2["_day"] < day]
        df_window = df_prior.tail(30_000).copy()
        if len(df_window) < 500:
            continue

        day_zones, _, avg_vix = run(df_window, vix_series,
                                    bin_size=bin_size, quiet=True)
        if not day_zones:
            continue

        bars     = df2[df2["_day"] == day].copy().reset_index(drop=True)
        price_lo = bars["low"].min(); price_hi = bars["high"].max()
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
            hw     = zone.price * 0.0005
            ax.axhspan(zone.price - hw, zone.price + hw,
                       color=colour, alpha=0.07 + 0.40 * ns, linewidth=0)
            ax.axhline(zone.price, color=colour,
                       lw=0.8 + 1.2 * ns, alpha=0.60 + 0.30 * ns, ls="--")
            role  = "R" if zone.price > day_open else "S"
            fts   = "|".join(sorted(zone.ftypes))
            n_dte = sum(1 for t in zone.timeframes if t.startswith("0dte"))
            ax.annotate(
                f"{zone.price:.0f} [{role}] {fts}  n_tf={zone.n_tf}  0dte={n_dte}",
                xy=(len(bars) - 1, zone.price),
                xytext=(8, 0), textcoords="offset points",
                color=colour, fontsize=6.2, va="center",
                fontfamily="monospace", clip_on=False,
            )

        nb = len(bars); ts_vals = bars["date"].values
        tick_idx = np.linspace(0, nb - 1, min(10, nb), dtype=int)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(
            [pd.Timestamp(ts_vals[i]).strftime("%H:%M") for i in tick_idx],
            fontsize=7, color="#8B949E",
        )
        ax.set_xlim(0, nb - 1); ax.set_ylim(y_lo, y_hi)
        ax.tick_params(axis="y", colors="#8B949E", labelsize=8)
        ax.tick_params(axis="x", colors="#8B949E", labelsize=7, length=3)
        ax.set_ylabel("Price  (NQ)", color="#8B949E", fontsize=9)
        ax.set_title(
            f"NQ 0DTE-GEX  {day}  ·  {nb} bars  ·  {len(selected)} zones  "
            f"·  VIX={vix_val:.1f}  (no look-ahead · darker = higher confidence)",
            color="#F0F6FC", fontsize=9.5, pad=8, fontweight="bold",
        )
        plt.tight_layout(pad=0.5)
        out = out_dir / f"gex_{day}.png"
        fig.savefig(str(out), dpi=130, bbox_inches="tight", facecolor=dark)
        plt.close(fig)
        generated += 1
        print(f"  [{generated:02d}/{n}]  {day}  VIX={vix_val:.1f}  "
              f"bars={nb:4d}  zones={len(selected)}  → gex_{day}.png")

    print(f"  Done.  ({generated}/{n} charts)")


# ─── Optimization & Full Coverage ────────────────────────────────────────────

def _apply_cfg(cfg: dict) -> dict:
    """Temporarily override module-level constants. Returns originals for restore."""
    import sys as _sys
    mod = _sys.modules[__name__]
    originals = {}
    for k, v in cfg.items():
        if hasattr(mod, k):
            originals[k] = getattr(mod, k)
            setattr(mod, k, v)
    return originals


def _restore_cfg(originals: dict):
    import sys as _sys
    mod = _sys.modules[__name__]
    for k, v in originals.items():
        setattr(mod, k, v)


def _fast_hod_lod(df: pd.DataFrame, vix_series: pd.Series,
                  days_list: list, bin_size: float = BIN_SIZE,
                  tol_pct: float = 0.0012) -> dict:
    """Quiet HOD/LOD test on a fixed day list. Returns stats dict."""
    df2 = df.copy()
    df2["_day"] = df2["date"].dt.date
    day_stats = df2.groupby("_day").agg(
        bars=("close", "count"), hod=("high", "max"), lod=("low", "min"),
    )
    hod_hits = lod_hits = both_hits = total = 0
    hod_dists: list[float] = []
    lod_dists: list[float] = []

    for day in days_list:
        if day not in day_stats.index:
            continue
        df_prior  = df2[df2["_day"] < day]
        df_window = df_prior.tail(30_000).copy()
        if len(df_window) < 500:
            continue
        zones, _, _ = run(df_window, vix_series, bin_size=bin_size, quiet=True)
        if not zones:
            continue
        zp      = np.array([z.price for z in zones])
        hod     = float(day_stats.loc[day, "hod"])
        lod     = float(day_stats.loc[day, "lod"])
        tol     = ((hod + lod) / 2) * tol_pct
        hd      = float(np.min(np.abs(zp - hod)))
        ld      = float(np.min(np.abs(zp - lod)))
        hod_hit = hd <= tol
        lod_hit = ld <= tol
        if hod_hit: hod_hits += 1
        if lod_hit: lod_hits += 1
        if hod_hit and lod_hit: both_hits += 1
        total += 1
        hod_dists.append(hd)
        lod_dists.append(ld)

    if total == 0:
        return {"hod_rate": 0.0, "lod_rate": 0.0, "both_rate": 0.0,
                "avg_hod_dist": 999.0, "avg_lod_dist": 999.0, "n_days": 0}
    return {
        "hod_rate":     hod_hits / total,
        "lod_rate":     lod_hits / total,
        "both_rate":    both_hits / total,
        "avg_hod_dist": float(np.mean(hod_dists)),
        "avg_lod_dist": float(np.mean(lod_dists)),
        "n_days":       total,
    }


def optimize_gex(df: pd.DataFrame, vix_series: pd.Series,
                 n_sample: int = 40, seed: int = 42,
                 bin_size: float = BIN_SIZE) -> tuple:
    """
    Greedy single-parameter sweep to maximize HOD+LOD+Both coverage.
    Each parameter swept independently; best value locked before next sweep.
    Uses a fixed random sample of n_sample days (reproducible via seed).
    """
    rng = np.random.default_rng(seed)
    df2 = df.copy()
    df2["_day"] = df2["date"].dt.date
    day_stats   = df2.groupby("_day").agg(bars=("close", "count"))
    all_days    = sorted(day_stats.index.tolist())
    eligible    = [d for d in all_days
                   if day_stats.loc[d, "bars"] >= 300
                   and all_days.index(d) >= 10]
    sample = sorted(rng.choice(eligible,
                               size=min(n_sample, len(eligible)),
                               replace=False).tolist())

    print(f"\n  Optimization: {len(sample)} sample days  seed={seed}")
    print(f"  Range: {sample[0]} → {sample[-1]}")

    def _score(r): return r["hod_rate"] + r["lod_rate"] + r["both_rate"] * 0.5

    base  = _fast_hod_lod(df, vix_series, sample, bin_size)
    print(f"\n  Baseline  HOD={base['hod_rate']:.1%}  LOD={base['lod_rate']:.1%}  "
          f"Both={base['both_rate']:.1%}  score={_score(base):.3f}\n")

    best_cfg   = {}
    best_score = _score(base)

    sweeps = [
        ("N_SESSIONS",       [3, 4, 5, 6, 7]),
        ("STACK_TOLERANCE",  [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]),
        ("HVN_PROMINENCE",   [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]),
        ("LVN_DEPTH",        [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]),
        ("INCLUDE_SHELF",    [True, False]),
        ("INCLUDE_LEDGE",    [True, False]),
        ("USE_GWALL",        [True, False]),
        ("USE_GFLIP",        [True, False]),
        ("GWALL_PROMINENCE", [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]),
    ]

    for param, values in sweeps:
        print(f"  ─ {param}")
        phase_best_val   = None
        phase_best_score = best_score

        for val in values:
            cfg  = {**best_cfg, param: val}
            orig = _apply_cfg(cfg)
            try:
                res = _fast_hod_lod(df, vix_series, sample, bin_size)
                s   = _score(res)
                mark = " ◄" if s > phase_best_score else ""
                print(f"    {str(val):<8}  HOD={res['hod_rate']:.1%}  "
                      f"LOD={res['lod_rate']:.1%}  Both={res['both_rate']:.1%}  "
                      f"score={s:.3f}{mark}")
                if s > phase_best_score:
                    phase_best_score = s
                    phase_best_val   = val
            finally:
                _restore_cfg(orig)

        if phase_best_val is not None and phase_best_val != {**best_cfg}.get(param):
            best_cfg[param]  = phase_best_val
            best_score       = phase_best_score
            print(f"  → updated: {param} = {phase_best_val!r}  score={best_score:.3f}")

    print(f"\n  ══════ OPTIMAL CONFIG ══════")
    for k, v in best_cfg.items():
        print(f"    {k} = {v!r}")

    orig = _apply_cfg(best_cfg)  # apply permanently for rest of session
    print(f"\n  Verifying on sample ({n_sample} days)...")
    final = _fast_hod_lod(df, vix_series, sample, bin_size)
    print(f"  HOD={final['hod_rate']:.1%}  LOD={final['lod_rate']:.1%}  "
          f"Both={final['both_rate']:.1%}  "
          f"avg_HOD={final['avg_hod_dist']:.1f}pts  avg_LOD={final['avg_lod_dist']:.1f}pts")
    return best_cfg, final


def full_coverage_date_range(df: pd.DataFrame, vix_series: pd.Series,
                             start_date: str, end_date: str,
                             tol_pct: float = 0.0012,
                             bin_size: float = BIN_SIZE) -> dict:
    """
    HOD/LOD coverage on ALL eligible trading days in [start_date, end_date].
    No sampling — every qualifying day is tested.
    """
    df2 = df.copy()
    df2["_day"] = df2["date"].dt.date
    day_stats   = df2.groupby("_day").agg(
        bars=("close", "count"), hod=("high", "max"), lod=("low", "min"),
    )
    all_days  = sorted(day_stats.index.tolist())
    start     = pd.Timestamp(start_date).date()
    end       = pd.Timestamp(end_date).date()
    eligible  = [d for d in all_days
                 if start <= d <= end
                 and day_stats.loc[d, "bars"] >= 300
                 and all_days.index(d) >= 10]

    print(f"  Full coverage: {start} → {end}  ({len(eligible)} eligible days)")
    hod_hits = lod_hits = both_hits = total = 0
    hod_dists: list[float] = []
    lod_dists: list[float] = []

    for i, day in enumerate(eligible):
        df_prior  = df2[df2["_day"] < day]
        df_window = df_prior.tail(30_000).copy()
        if len(df_window) < 500:
            continue
        zones, _, _ = run(df_window, vix_series, bin_size=bin_size, quiet=True)
        if not zones:
            continue
        zp      = np.array([z.price for z in zones])
        hod     = float(day_stats.loc[day, "hod"])
        lod     = float(day_stats.loc[day, "lod"])
        tol     = ((hod + lod) / 2) * tol_pct
        hd      = float(np.min(np.abs(zp - hod)))
        ld      = float(np.min(np.abs(zp - lod)))
        hod_hit = hd <= tol
        lod_hit = ld <= tol
        if hod_hit: hod_hits += 1
        if lod_hit: lod_hits += 1
        if hod_hit and lod_hit: both_hits += 1
        total += 1
        hod_dists.append(hd)
        lod_dists.append(ld)

        if (i + 1) % 50 == 0:
            print(f"    [{i+1:>3}/{len(eligible)}]  {day}  "
                  f"HOD={hod_hits}/{total}={hod_hits/total:.1%}  "
                  f"LOD={lod_hits}/{total}={lod_hits/total:.1%}")

    if total == 0:
        print("  No days processed")
        return {}

    avg_hd = float(np.mean(hod_dists))
    avg_ld = float(np.mean(lod_dists))
    print(f"\n  ─── Results ({start} → {end}) ───")
    print(f"  HOD: {hod_hits/total:.1%}  ({hod_hits}/{total})")
    print(f"  LOD: {lod_hits/total:.1%}  ({lod_hits}/{total})")
    print(f"  Both: {both_hits/total:.1%}  ({both_hits}/{total})")
    print(f"  avg dist HOD={avg_hd:.1f}pts  LOD={avg_ld:.1f}pts")
    return dict(hod_rate=hod_hits/total, lod_rate=lod_hits/total,
                both_rate=both_hits/total, avg_hod_dist=avg_hd,
                avg_lod_dist=avg_ld, n_days=total)


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NQ 0DTE GEX Profile Analyzer")
    parser.add_argument("--backtest",      action="store_true")
    parser.add_argument("--hod-lod",       action="store_true")
    parser.add_argument("--hod-days",      type=int, default=60)
    parser.add_argument("--optimize",      action="store_true",
                        help="Sweep params to maximise HOD/LOD coverage")
    parser.add_argument("--opt-sample",    type=int, default=40,
                        help="Days to sample per config during optimization")
    parser.add_argument("--full-coverage", action="store_true",
                        help="Run HOD/LOD on ALL days in --start / --end range")
    parser.add_argument("--start",         type=str, default="2024-01-01")
    parser.add_argument("--end",           type=str, default="2025-12-31")
    parser.add_argument("--sample-days",   type=int, default=0)
    parser.add_argument("--seed",          type=int, default=None)
    parser.add_argument("--top",           type=int, default=20)
    parser.add_argument("--bins",          type=float, default=BIN_SIZE)
    parser.add_argument("--rev-pct",       type=float, default=0.60)
    parser.add_argument("--sessions",      type=int, default=N_SESSIONS,
                        help=f"Prior sessions to stack (default {N_SESSIONS})")
    parser.add_argument("--daily",         type=int, default=5)
    args = parser.parse_args()

    print("=" * 56)
    print("  NQ 0DTE GEX PROFILE ANALYZER")
    print("=" * 56)
    print("\nLoading data...")
    df         = load_nq_data()
    vix_series = load_vix()
    print(f"  NQ: {len(df):,} bars  {df['date'].min().date()} → {df['date'].max().date()}")

    if args.optimize:
        print(f"\nRunning GEX parameter optimization ({args.opt_sample} sample days)...")
        seed = args.seed if args.seed is not None else 42
        best_cfg, final = optimize_gex(df, vix_series, n_sample=args.opt_sample,
                                       seed=seed, bin_size=args.bins)
        return

    if args.full_coverage:
        print(f"\nFull HOD/LOD coverage: {args.start} → {args.end}")
        full_coverage_date_range(df, vix_series,
                                 start_date=args.start, end_date=args.end,
                                 bin_size=args.bins)
        return

    if args.hod_lod:
        print(f"\nRunning 0DTE HOD/LOD coverage ({args.hod_days} days)...")
        seed = args.seed if args.seed else 42
        hod_lod_coverage(df, vix_series, n_days=args.hod_days,
                         bin_size=args.bins, seed=seed)
        return

    print("\nBuilding 0DTE GEX stacked profiles...")
    zones, profiles, avg_vix = run(df, vix_series, bin_size=args.bins,
                                   n_sessions=args.sessions)
    print_report(zones, show_n=args.top, avg_vix=avg_vix)

    n_daily = max(4, min(6, args.daily))
    levels, last_close, atr20 = daily_levels(df, zones, n=n_daily)
    print_daily_levels(levels, last_close, atr20)

    if args.sample_days > 0:
        plot_sample_days(df, vix_series, n=args.sample_days,
                         bin_size=args.bins, seed=args.seed)

    if args.backtest:
        rev_thr = args.rev_pct / 100.0
        BT_FWD = 390; BT_SEP = 240
        print(f"\nRunning 3-year reversal backtest (threshold={args.rev_pct}%)...")
        bt = backtest(df, zones, tolerance=args.bins,
                      reversal_pct=rev_thr,
                      forward_bars=BT_FWD, min_sep_bars=BT_SEP)
        if not bt.empty:
            print_backtest_report(df, bt, reversal_pct=rev_thr,
                                  tolerance=args.bins,
                                  forward_bars=BT_FWD, min_sep_bars=BT_SEP)
            out = Path(__file__).parent / "gex_backtest.csv"
            bt.to_csv(out, index=False)
            print(f"  Saved → {out}")


if __name__ == "__main__":
    main()
