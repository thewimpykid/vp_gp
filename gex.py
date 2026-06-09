"""
gex.py — synthetic stacked Gamma Exposure (GEX) profile for NQ.

No options chain available, so GEX is proxied:
  - IV: daily ^VXN close (Nasdaq-100 vol index, cached cache/vxn_daily.csv)
  - Strikes: every STRIKE_STEP points across the active price range
  - OI proxy: traded futures volume near each strike (from cached volume
    profiles) — where futures volume clusters, options open interest
    concentrates too
  - Gamma: Black-Scholes gamma at each strike, spot = day VWAP,
    T = T_DAYS/252 (weekly option horizon)
  - GEX(K) = gamma(S,K,T,iv) * OI(K) * S

Stacked across `lookback` prior days with recency weighting, painted as
±band_pts bands on the 0.25pt price grid → intensity in [0,1], same shape
as the VP confluence heatmap so the two can be blended.
"""
import os
import sys

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

VXN_CSV     = os.path.join(BASE_DIR, "cache", "vxn_daily.csv")
STRIKE_STEP = 25.0     # NQ option strikes
T_DAYS      = 5        # weekly option horizon
RISK_FREE   = 0.04
DEFAULT_IV  = 0.20

_vxn_cache: dict | None = None


def load_vxn() -> dict[str, float]:
    """date_nodash → IV (decimal). Forward-fills non-trading days."""
    global _vxn_cache
    if _vxn_cache is not None:
        return _vxn_cache
    _vxn_cache = {}
    if not os.path.exists(VXN_CSV):
        return _vxn_cache
    # yfinance multiindex csv: rows 0-2 are Price/Ticker/Date headers
    df = pd.read_csv(VXN_CSV, skiprows=3,
                     names=["Date", "Close", "High", "Low", "Open", "Volume"])
    df["Date"] = pd.to_datetime(df["Date"])
    ser = df.set_index("Date")["Close"].astype(float)
    # reindex to calendar days and ffill so weekend/holiday sessions resolve
    full = ser.reindex(pd.date_range(ser.index.min(), ser.index.max())).ffill()
    _vxn_cache = {ts.strftime("%Y%m%d"): float(v) / 100.0
                  for ts, v in full.items() if pd.notna(v)}
    return _vxn_cache


def bs_gamma(S: float, K: np.ndarray, T: float, sigma: float,
             r: float = RISK_FREE) -> np.ndarray:
    """Black-Scholes gamma (same for calls and puts), vectorized over K."""
    sqT = np.sqrt(T)
    d1  = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqT)
    pdf = np.exp(-0.5 * d1**2) / np.sqrt(2.0 * np.pi)
    return pdf / (S * sigma * sqT)


def build_stacked_gex(
    date_nodash: str,
    session:  str   = "full",
    lookback: int   = 15,
    band_pts: float = 6.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Stacked GEX intensity on the cached price grid.

    Each prior day contributes GEX(K) at every strike, painted as a flat
    ±band_pts band, recency-weighted. Returns (price_grid, intensity in
    [0,1]) or None.
    """
    import backtest as bt
    daily = bt.load_all_profiles(session)
    sorted_dates = sorted(daily.keys())
    prior = [d for d in sorted_dates if d < date_nodash][-lookback:]
    if len(prior) < 1:
        return None

    price_grid = bt.load_price_grid()
    if price_grid is None:
        return None
    n    = len(price_grid)
    tick = float(price_grid[1] - price_grid[0])
    half = max(1, int(round(band_pts / tick)))

    vxn = load_vxn()
    T   = T_DAYS / 252.0

    from scipy.ndimage import uniform_filter1d
    from scipy.signal import argrelmax

    # ── per-day gamma walls, then STACK the walls across days.
    # A zone only survives if it shows up on multiple days (cumulative
    # exposure, recency-weighted) — one-day walls are noise.
    votes   = np.zeros(n, dtype=np.float64)   # recency-weighted exposure
    count   = np.zeros(n, dtype=np.int32)     # how many days flag the bin
    n_prior = len(prior)
    order   = max(1, int(round(15.0 / tick)))
    sm_bins = max(3, int(round(25.0 / tick)))

    for idx, d in enumerate(prior):
        p = daily.get(d)
        if p is None or p.sum() == 0:
            continue
        w     = (idx + 1) / n_prior            # recency weight
        spot  = float((p * price_grid).sum() / p.sum())   # day VWAP
        sigma = vxn.get(d, DEFAULT_IV)

        # continuous day GEX curve: BS gamma per bin × volume there (OI proxy)
        active = p > 0
        gex_day = np.zeros(n, dtype=np.float64)
        gex_day[active] = (
            bs_gamma(spot, price_grid[active], T, sigma) * p[active] * spot
        )
        gex_day = uniform_filter1d(gex_day, size=sm_bins)
        if gex_day.max() == 0:
            continue
        s = gex_day / gex_day.max()

        # this day's gamma walls
        peaks = argrelmax(s, order=order)[0]
        peaks = peaks[s[peaks] >= 0.30]
        if len(peaks) == 0:
            continue

        day_band = np.zeros(n, dtype=np.float64)
        for pk in peaks:
            lo = max(0, pk - half)
            hi = min(n, pk + half + 1)
            day_band[lo:hi] = np.maximum(day_band[lo:hi], s[pk])

        votes += w * day_band
        count += (day_band > 0)

    # reversal zones = walls confirmed on >= 2 separate days
    zones = np.where(count >= 2, votes, 0.0)
    if zones.max() == 0:
        return None
    return price_grid, zones / zones.max()
