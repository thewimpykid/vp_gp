"""
Combined VP + GEX Level Analyzer
=================================
Merges Volume Profile multi-timeframe zones with 0DTE GEX zones.
When VP and GEX agree on a price, that zone receives a cross-system n_tf^2 boost.

Architecture
  1. stacked_vp.run()  → VP zones (30/60/90d multi-TF profiles + structural)
  2. gex_profile.run() → GEX zones (5 × 0DTE sessions + structural)
  3. Expand each zone → per-timeframe VPFeature objects (preserves internal n_tf)
  4. Re-stack all features → combined zones
     Solo VP zone:  score  ∝  vp_internal_n_tf^2  (unchanged)
     Solo GEX zone: score  ∝  gex_0dte_n^2        (unchanged)
     VP+GEX agree:  score  ∝  (vp_n_tf + gex_n_tf)^2  → large cross-system boost
  5. Test combined zones: HOD/LOD coverage + reversal rate

Usage
  python combined_vp_gex/combined_analyzer.py              # today's levels
  python combined_vp_gex/combined_analyzer.py --hod-lod    # walk-forward coverage (60 days)
  python combined_vp_gex/combined_analyzer.py --compare    # full 3-way VP/GEX/Combined (2024+2025)
  python combined_vp_gex/combined_analyzer.py --backtest   # reversal backtest
  python combined_vp_gex/combined_analyzer.py --sample-days 10   # no-look-ahead charts
"""

import argparse
import math
import sys
import warnings
from dataclasses import dataclass, field
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

# ─── Parameters ───────────────────────────────────────────────────────────────
# Band-coverage optimized (optimize_bands.py, tuned 2024 / validated 2025):
#   ±10pt band: HOD 88.4% / LOD 86.8%   ±7.5pt band: HOD 81.7% / LOD 78.5%
#   Lift vs equal-density random levels: +10-12%
BIN_SIZE         = 5.0
STACK_TOLERANCE  = 0.75   # cluster radius = 0.75 × 5pt = 3.75pt (tight, was 3.0)
SNAP_CENTER      = True   # zone center = heaviest member price (not cluster mean)
TOL_PCT          = 0.0012 # HOD/LOD tolerance 0.12% of mid ≈ 25pts@21k
VP_WEIGHT_SCALE  = 0.08   # VP zone score → feature weight (scores ~5-80 → 0.4-6.4)
GEX_WEIGHT_SCALE = 0.08   # GEX zone score → feature weight (same scale)
HOD_LOD_TOL      = 7.5    # default coverage band half-width (15pt full band)

# Extra structural families (all validated to add coverage):
#   cam=Camarilla pivots  piv=floor pivots  pdc=prior close/mid/vwap
#   em=implied expected-move bands  pw=prior week close/mid  atr=ATR projections
#   ladder/ladder2=interior ATR ladders  ext/ext2=breakout extension ladders
EXTRA_FAMILIES   = {"cam", "piv", "pdc", "em", "pw", "atr",
                    "ladder", "ladder2", "ext", "ext2"}
NQ_IV_MULT       = 1.15


# ─── Data Classes ─────────────────────────────────────────────────────────────

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
    systems:    set = field(default_factory=set)


# ─── Zone expansion + stacking ────────────────────────────────────────────────

def _zones_to_features(zones: list, system: str,
                        weight_scale: float) -> list[VPFeature]:
    """
    Expand pre-stacked zones back into per-timeframe VPFeature objects.
    A VP zone from 3 TFs → 3 features at the same price (preserves n_tf signal).
    Timeframes prefixed with system name so VP/GEX TFs never collide.
    """
    feats: list[VPFeature] = []
    for zone in zones:
        tfs = zone.timeframes if zone.timeframes else {system}
        n   = max(1, len(tfs))
        w   = max(0.1, zone.score * weight_scale / n)
        pf  = min(zone.ftypes) if zone.ftypes else system
        for tf in tfs:
            feats.append(VPFeature(
                price=zone.price,
                ftype=pf,
                timeframe=f"{system}__{tf}",
                weight=w,
            ))
    return feats


def _extra_struct_features(df: pd.DataFrame, vix_series: pd.Series,
                           families: set = None) -> list[VPFeature]:
    """
    Extra structural level families computed from prior completed sessions only
    (df is the walk-forward window — everything in it is prior data).
    Validated by optimize_bands.py: breakout extension ladders (ext/ext2) fix
    the dominant miss mode (73% of HOD misses were breakouts beyond PDH,
    median extension only +0.22 ATR).
    """
    import gex_profile as _gex

    families = EXTRA_FAMILIES if families is None else families
    df2 = df.copy()
    df2["_d"] = df2["date"].dt.date
    daily = df2.groupby("_d").agg(
        h=("high", "max"), l=("low", "min"), c=("close", "last"),
        v=("volume", "sum"),
    )
    pv = (df2["close"] * df2["volume"]).groupby(df2["_d"]).sum()
    daily["vwap"] = pv / daily["v"].replace(0, np.nan)
    if len(daily) < 2:
        return []

    prow = daily.iloc[-1]
    H, L, C = float(prow["h"]), float(prow["l"]), float(prow["c"])
    R = H - L
    a = _gex.compute_atr20(df)
    last_dt = df2["date"].iloc[-1]
    vix_val = _gex.get_vix_at(vix_series, last_dt)

    # prior completed week relative to the upcoming session (next calendar day)
    nxt   = last_dt + pd.Timedelta(days=1)
    wk_id = (nxt.isocalendar().year, nxt.isocalendar().week)
    pdays = daily.index.tolist()[-6:]
    pweek = [d for d in pdays
             if (pd.Timestamp(d).isocalendar().year,
                 pd.Timestamp(d).isocalendar().week) != wk_id]

    out: list[tuple] = []
    if "cam" in families:
        out += [(C + R * 1.1 / 4, 1.4, "camR3"), (C - R * 1.1 / 4, 1.4, "camS3"),
                (C + R * 1.1 / 2, 1.5, "camR4"), (C - R * 1.1 / 2, 1.5, "camS4"),
                (C + R * 1.1 / 6, 1.0, "camR2"), (C - R * 1.1 / 6, 1.0, "camS2")]
    if "piv" in families:
        P = (H + L + C) / 3
        out += [(P, 1.2, "pivP"),
                (2 * P - L, 1.3, "pivR1"), (2 * P - H, 1.3, "pivS1"),
                (P + R, 1.2, "pivR2"),     (P - R, 1.2, "pivS2")]
    if "pdc" in families:
        out += [(C, 1.3, "pdc"), ((H + L) / 2, 1.2, "pdm"),
                (float(prow["vwap"]), 1.3, "pdvwap")]
    if "em" in families:
        sig_d = (vix_val / 100.0) * NQ_IV_MULT / np.sqrt(252) * C
        out += [(C + sig_d, 1.3, "em+1s"), (C - sig_d, 1.3, "em-1s"),
                (C + 0.5 * sig_d, 1.1, "em+.5s"), (C - 0.5 * sig_d, 1.1, "em-.5s")]
    if "pw" in families and pweek:
        pw_h = max(daily.loc[d, "h"] for d in pweek)
        pw_l = min(daily.loc[d, "l"] for d in pweek)
        pw_c = daily.loc[pweek[-1], "c"]
        out += [(float(pw_c), 1.1, "pwc"), ((pw_h + pw_l) / 2, 1.1, "pwm")]
    if "atr" in families:
        out += [(C + 0.5 * a, 1.2, "atr+.5"), (C - 0.5 * a, 1.2, "atr-.5"),
                (C + 1.0 * a, 1.2, "atr+1"),  (C - 1.0 * a, 1.2, "atr-1")]
    if "ladder" in families:
        for k in [0.25, 0.4, 0.55, 0.7, 0.85, 1.0, 1.2, 1.4]:
            out += [(C + k * a, 1.0, f"lad+{k}"), (C - k * a, 1.0, f"lad-{k}")]
    if "ladder2" in families:
        for k in [0.15, 0.27, 0.39, 0.51, 0.63, 0.75, 0.88, 1.02, 1.18, 1.35]:
            out += [(C + k * a, 0.9, f"l2+{k}"), (C - k * a, 0.9, f"l2-{k}")]
    if "ext" in families:
        for k in [0.08, 0.20, 0.33, 0.48, 0.65, 0.85]:
            out += [(H + k * a, 1.2, f"ext+{k}"), (L - k * a, 1.2, f"ext-{k}")]
    if "ext2" in families:
        for k in [0.05, 0.11, 0.17, 0.24, 0.31, 0.39, 0.48, 0.58, 0.70, 0.84, 1.0]:
            out += [(H + k * a, 1.1, f"x2+{k}"), (L - k * a, 1.1, f"x2-{k}")]

    return [VPFeature(price=float(p), ftype=tag, timeframe=f"x__{tag}", weight=w)
            for p, w, tag in out if np.isfinite(p)]


def _stack(all_features: list[VPFeature], bin_size: float,
           tol_mult: float = STACK_TOLERANCE) -> list[Zone]:
    """Greedy price-proximity clustering. Score = n_tf^2 × Σweights."""
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

        if SNAP_CENTER:
            center = float(max(cluster, key=lambda f: f.weight).price)
        else:
            center = float(np.mean([f.price for f in cluster]))
        tfs     = {f.timeframe for f in cluster}
        ftypes  = {f.ftype for f in cluster}
        systems = {f.timeframe.split("__")[0] for f in cluster}
        n_tf    = len(tfs)
        score   = (n_tf ** 2) * sum(f.weight for f in cluster)

        zones.append(Zone(
            price=center, score=score, n_tf=n_tf,
            timeframes=tfs, ftypes=ftypes,
            features=cluster, systems=systems,
        ))
    return sorted(zones, key=lambda z: z.score, reverse=True)


# ─── Main driver ──────────────────────────────────────────────────────────────

def run(df: pd.DataFrame, vix_series: pd.Series,
        bin_size: float = BIN_SIZE, quiet: bool = False) -> tuple:
    """
    Build combined VP+GEX zones from prior data.
    Returns (zones, avg_vix).
    """
    import stacked_vp  as _vp
    import gex_profile as _gex

    vp_zones,  _            = _vp.run(df, anchored=True, bin_size=bin_size, quiet=True)
    gex_zones, _, avg_vix   = _gex.run(df, vix_series, bin_size=bin_size, quiet=True)

    vp_feats  = _zones_to_features(vp_zones,  "vp",  VP_WEIGHT_SCALE)
    gex_feats = _zones_to_features(gex_zones, "gex", GEX_WEIGHT_SCALE)
    x_feats   = _extra_struct_features(df, vix_series)
    combined  = _stack(vp_feats + gex_feats + x_feats, bin_size)

    if not quiet:
        cross = sum(1 for z in combined if len(z.systems) >= 2)
        print(f"  VP {len(vp_zones)} zones + GEX {len(gex_zones)} zones "
              f"+ {len(x_feats)} struct feats → {len(combined)} combined  "
              f"({cross} cross-system boosted)")
    return combined, avg_vix


# ─── HOD/LOD coverage ────────────────────────────────────────────────────────

def hod_lod_coverage(df: pd.DataFrame, vix_series: pd.Series,
                     n_days: int = 60, tolerance_pts: float = HOD_LOD_TOL,
                     bin_size: float = BIN_SIZE, seed: int = 42,
                     quiet_days: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    df2 = df.copy()
    df2["_day"] = df2["date"].dt.date
    day_stats   = df2.groupby("_day").agg(
        bars=("close", "count"), hod=("high", "max"), lod=("low", "min"),
    )
    all_days = sorted(day_stats.index.tolist())
    eligible = [d for d in all_days
                if day_stats.loc[d, "bars"] >= 300
                and all_days.index(d) >= 90]
    if not eligible:
        print("  No eligible days"); return {}

    sample = sorted(rng.choice(eligible, size=min(n_days, len(eligible)),
                               replace=False).tolist())

    hod_hits = lod_hits = both_hits = total = 0
    rows: list[dict] = []
    sep = "-" * 76
    if not quiet_days:
        print(f"\n  Combined VP+GEX HOD/LOD  ({len(sample)} days, "
              f"tol=±{tolerance_pts:.0f}pts fixed)")
        print(sep)
        print(f"  {'DATE':<12}  {'HOD':>8}  {'LOD':>8}  {'dHOD':>6}  {'dLOD':>6}  HOD  LOD")
        print(sep)

    for day in sample:
        df_prior  = df2[df2["_day"] < day]
        df_window = df_prior.tail(30_000).copy()
        if len(df_window) < 500:
            continue
        zones, _ = run(df_window, vix_series, bin_size=bin_size, quiet=True)
        if not zones:
            continue
        zp      = np.array([z.price for z in zones])
        hod     = float(day_stats.loc[day, "hod"])
        lod     = float(day_stats.loc[day, "lod"])
        tol     = tolerance_pts
        hd      = float(np.min(np.abs(zp - hod)))
        ld      = float(np.min(np.abs(zp - lod)))
        hod_hit = hd <= tol
        lod_hit = ld <= tol
        if hod_hit: hod_hits += 1
        if lod_hit: lod_hits += 1
        if hod_hit and lod_hit: both_hits += 1
        total += 1
        rows.append(dict(day=str(day), hod=hod, lod=lod,
                         hod_dist=round(hd, 1), lod_dist=round(ld, 1),
                         hod_hit=hod_hit, lod_hit=lod_hit))
        if not quiet_days:
            h_f = "✓" if hod_hit else " "
            l_f = "✓" if lod_hit else " "
            print(f"  {str(day):<12}  {hod:>8.1f}  {lod:>8.1f}  "
                  f"{hd:>6.1f}  {ld:>6.1f}   {h_f}    {l_f}")

    if total == 0:
        print("  No days processed"); return {}

    avg_hd = float(np.mean([r["hod_dist"] for r in rows]))
    avg_ld = float(np.mean([r["lod_dist"] for r in rows]))
    if not quiet_days:
        print(sep)
        print(f"  HOD: {hod_hits/total:.1%}  ({hod_hits}/{total})")
        print(f"  LOD: {lod_hits/total:.1%}  ({lod_hits}/{total})")
        print(f"  Both: {both_hits/total:.1%}  ({both_hits}/{total})")
        print(f"  Avg dist HOD={avg_hd:.1f}pts  LOD={avg_ld:.1f}pts")
    return dict(hod_rate=hod_hits/total, lod_rate=lod_hits/total,
                both_rate=both_hits/total, avg_hod_dist=avg_hd,
                avg_lod_dist=avg_ld, n_days=total)


def full_coverage_date_range(df: pd.DataFrame, vix_series: pd.Series,
                              start_date: str, end_date: str,
                              tolerance_pts: float = HOD_LOD_TOL,
                              bin_size: float = BIN_SIZE) -> dict:
    """All-days HOD/LOD in date range. No sampling."""
    df2 = df.copy()
    df2["_day"] = df2["date"].dt.date
    day_stats   = df2.groupby("_day").agg(
        bars=("close", "count"), hod=("high", "max"), lod=("low", "min"),
    )
    all_days = sorted(day_stats.index.tolist())
    s_date   = pd.Timestamp(start_date).date()
    e_date   = pd.Timestamp(end_date).date()
    eligible = [d for d in all_days
                if s_date <= d <= e_date
                and day_stats.loc[d, "bars"] >= 300
                and all_days.index(d) >= 90]

    print(f"  [COMB] {start_date} → {end_date}  ({len(eligible)} eligible days)")
    hod_hits = lod_hits = both_hits = total = 0
    hod_dists: list[float] = []
    lod_dists: list[float] = []

    for i, day in enumerate(eligible):
        df_prior  = df2[df2["_day"] < day]
        df_window = df_prior.tail(30_000).copy()
        if len(df_window) < 500:
            continue
        zones, _ = run(df_window, vix_series, bin_size=bin_size, quiet=True)
        if not zones:
            continue
        zp      = np.array([z.price for z in zones])
        hod     = float(day_stats.loc[day, "hod"])
        lod     = float(day_stats.loc[day, "lod"])
        tol     = tolerance_pts
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
            print(f"    COMB [{i+1:>3}/{len(eligible)}]  {day}  "
                  f"HOD={hod_hits/total:.1%}  LOD={lod_hits/total:.1%}")

    if total == 0:
        print("  No days processed"); return {}
    avg_hd = float(np.mean(hod_dists))
    avg_ld = float(np.mean(lod_dists))
    return dict(hod_rate=hod_hits/total, lod_rate=lod_hits/total,
                both_rate=both_hits/total, avg_hod_dist=avg_hd,
                avg_lod_dist=avg_ld, n_days=total)


# ─── Reversal backtest ────────────────────────────────────────────────────────

def _cap_daily(indices: list, dates_arr, max_per_day: int = 2) -> list:
    counts: dict = {}
    out: list = []
    for t in indices:
        d = pd.Timestamp(dates_arr[t]).date()
        c = counts.get(d, 0)
        if c < max_per_day:
            counts[d] = c + 1
            out.append(t)
    return out


def _reversal_test(df: pd.DataFrame, zones: list,
                   tolerance: float = 15.0,
                   forward_bars: int = 390,
                   min_sep_bars:  int = 240,
                   reversal_pct:  float = 0.006,
                   lookback_bars: int = 60) -> pd.DataFrame:
    hi = df["high"].values.astype(np.float64)
    lo = df["low"].values.astype(np.float64)
    cl = df["close"].values.astype(np.float64)
    dt = df["date"].values
    n  = len(df)
    rows = []
    for zone in zones:
        p          = zone.price
        close_near = np.abs(cl - p) <= tolerance * 4
        raw_t      = np.where((lo <= p + tolerance) & (hi >= p - tolerance) & close_near)[0]
        deduped: list[int] = []
        last_t = -min_sep_bars
        for t in raw_t:
            if t - last_t >= min_sep_bars:
                deduped.append(t); last_t = t
        deduped = _cap_daily(deduped, dt)
        total = len(deduped); revs = 0; moves: list[float] = []
        for t in deduped:
            tp  = cl[t]; thr = reversal_pct * tp
            pc  = cl[max(0, t - lookback_bars)]
            if   pc > p + tolerance: direction = "resistance"
            elif pc < p - tolerance: direction = "support"
            else:                    direction = "ambiguous"
            fwd_end = min(n, t + forward_bars + 1)
            if fwd_end <= t: continue
            fwd_hi = hi[t:fwd_end].max(); fwd_lo = lo[t:fwd_end].min()
            if   direction == "resistance": move = tp - fwd_lo
            elif direction == "support":    move = fwd_hi - tp
            else:                           move = max(tp - fwd_lo, fwd_hi - tp)
            if move >= thr:
                revs += 1; moves.append(move / tp * 100)
        sys_set = getattr(zone, "systems", set())
        cross   = {"vp", "gex"} <= sys_set
        rows.append(dict(
            price=round(zone.price, 2),
            score=round(zone.score, 2),
            n_tf=zone.n_tf,
            cross_system=cross,
            systems="|".join(sorted(sys_set)),
            ftypes="|".join(sorted(zone.ftypes)),
            touches=total, reversals=revs,
            rev_rate=round(revs / total, 3) if total > 0 else 0.0,
            avg_rev_pct=round(float(np.mean(moves)), 3) if moves else 0.0,
        ))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("rev_rate", ascending=False)


def _baseline_reversal(df: pd.DataFrame, n_samples: int = 40,
                        tolerance: float = 15.0,
                        forward_bars: int = 390,
                        min_sep_bars: int = 240,
                        reversal_pct: float = 0.006) -> float:
    hi = df["high"].values.astype(np.float64)
    lo = df["low"].values.astype(np.float64)
    cl = df["close"].values.astype(np.float64)
    dt = df["date"].values
    n  = len(df)
    rng = np.random.default_rng(42)
    prices = rng.uniform(lo.min(), hi.max(), size=n_samples * 3)
    all_rates = []; all_t = []
    tested = 0
    for p in prices:
        if tested >= n_samples: break
        cn  = np.abs(cl - p) <= tolerance * 4
        raw = np.where((lo <= p + tolerance) & (hi >= p - tolerance) & cn)[0]
        ded = []; lt = -min_sep_bars
        for t in raw:
            if t - lt >= min_sep_bars: ded.append(t); lt = t
        ded = _cap_daily(ded, dt)
        if len(ded) < 10: continue
        revs = 0
        for t in ded:
            tp = cl[t]; thr = reversal_pct * tp
            pc = cl[max(0, t - 60)]
            fe = min(n, t + forward_bars + 1)
            fh = hi[t:fe].max(); fl = lo[t:fe].min()
            if pc > p + tolerance: mv = tp - fl
            elif pc < p - tolerance: mv = fh - tp
            else: mv = max(tp - fl, fh - tp)
            if mv >= thr: revs += 1
        all_rates.append(revs / len(ded)); all_t.append(len(ded)); tested += 1
    if not all_rates: return 0.30
    return float(sum(r * t for r, t in zip(all_rates, all_t)) / sum(all_t))


def run_backtest(df: pd.DataFrame, vix_series: pd.Series,
                 years: float = 2.0, reversal_pct: float = 0.006) -> dict:
    """
    Static reversal backtest on last `years` of data.
    Builds combined zones from the full df, tests against historical prices.
    Also runs VP-only and GEX-only for 3-way comparison.
    """
    import stacked_vp  as _vp
    import gex_profile as _gex

    last_date  = df["date"].max()
    start_date = last_date - pd.Timedelta(days=int(years * 365.25))
    df_bt = df[df["date"] >= start_date].reset_index(drop=True)
    n_bars = len(df_bt)

    print(f"  Backtest window: {df_bt['date'].iloc[0].date()} → "
          f"{df_bt['date'].iloc[-1].date()}  ({n_bars:,} bars)")

    print("  Building zones...", end="", flush=True)
    vp_zones,  _            = _vp.run(df, anchored=True, bin_size=BIN_SIZE, quiet=True)
    gex_zones, _, avg_vix   = _gex.run(df, vix_series, bin_size=BIN_SIZE, quiet=True)
    comb_zones, _           = run(df, vix_series, quiet=True)
    print(f"  VP={len(vp_zones)}  GEX={len(gex_zones)}  Comb={len(comb_zones)}")

    print("  Computing baseline...", end="", flush=True)
    baseline = _baseline_reversal(df_bt, n_samples=40,
                                   reversal_pct=reversal_pct)
    print(f"  {baseline:.1%}")

    def _run_test(zones, label):
        bt = _reversal_test(df_bt, zones, reversal_pct=reversal_pct)
        if bt.empty:
            return None
        zone_rate = float((bt["rev_rate"] * bt["touches"]).sum() /
                          bt["touches"].sum()) if bt["touches"].sum() > 0 else 0.0
        lift = zone_rate - baseline
        n_beat5  = int((bt["rev_rate"] - baseline >= 0.05).sum())
        n_beat10 = int((bt["rev_rate"] - baseline >= 0.10).sum())
        avg_rev  = float(bt[bt["avg_rev_pct"] > 0]["avg_rev_pct"].mean())
        print(f"\n  ─── {label} ───")
        print(f"  Zones: {len(zones)}  Baseline: {baseline:.1%}  "
              f"Weighted rate: {zone_rate:.1%}  Lift: {lift:+.1%}")
        print(f"  Zones beating +5%: {n_beat5}/{len(zones)}  "
              f"Zones beating +10%: {n_beat10}/{len(zones)}")
        print(f"  Avg reversal size: {avg_rev:.2f}%")
        return dict(system=label, baseline=baseline, zone_rate=zone_rate,
                    lift=lift, n_zones=len(zones),
                    n_beat5=n_beat5, n_beat10=n_beat10, avg_rev_pct=avg_rev,
                    bt=bt)

    vp_res   = _run_test(vp_zones,   "VP   (volume profile)")
    gex_res  = _run_test(gex_zones,  "GEX  (0DTE gamma)")
    comb_res = _run_test(comb_zones, "COMB (VP+GEX combined)")

    # Cross-system zone breakdown
    if comb_res:
        bt = comb_res["bt"]
        cross = bt[bt["cross_system"] == True]
        solo  = bt[bt["cross_system"] == False]
        if len(cross) > 0 and cross["touches"].sum() > 0:
            cr = float((cross["rev_rate"]*cross["touches"]).sum()/cross["touches"].sum())
            print(f"\n  Cross-system zones (VP+GEX): {len(cross)}  "
                  f"rate={cr:.1%}  lift={cr-baseline:+.1%}")
        if len(solo) > 0 and solo["touches"].sum() > 0:
            sr = float((solo["rev_rate"]*solo["touches"]).sum()/solo["touches"].sum())
            print(f"  Solo zones (VP or GEX only): {len(solo)}  "
                  f"rate={sr:.1%}  lift={sr-baseline:+.1%}")

    sep = "=" * 76
    print(f"\n{sep}")
    print(f"  3-WAY COMPARISON  (baseline {baseline:.1%})")
    print(sep)
    print(f"  {'System':<24}  {'Rate':>6}  {'Lift':>7}  {'Beat+5%':>7}  "
          f"{'Beat+10%':>8}  {'AvgRev':>7}")
    print(sep)
    for res in [vp_res, gex_res, comb_res]:
        if res:
            print(f"  {res['system']:<24}  {res['zone_rate']:>5.1%}  "
                  f"{res['lift']:>+6.1%}  {res['n_beat5']:>7}  "
                  f"{res['n_beat10']:>8}  {res['avg_rev_pct']:>6.2f}%")
    print(sep)
    return dict(vp=vp_res, gex=gex_res, comb=comb_res, baseline=baseline)


# ─── 3-way HOD/LOD comparison ────────────────────────────────────────────────

def compare_three_way(df: pd.DataFrame, vix_series: pd.Series,
                       start: str = "2024-01-01", end: str = "2025-12-31") -> None:
    """Full 2-year HOD/LOD comparison: VP alone, GEX alone, VP+GEX combined."""
    import stacked_vp  as _vp
    import gex_profile as _gex

    df2 = df.copy()
    df2["_day"] = df2["date"].dt.date
    day_stats   = df2.groupby("_day").agg(
        bars=("close", "count"), hod=("high", "max"), lod=("low", "min"),
    )
    all_days = sorted(day_stats.index.tolist())
    s_date   = pd.Timestamp(start).date()
    e_date   = pd.Timestamp(end).date()
    eligible = [d for d in all_days
                if s_date <= d <= e_date
                and day_stats.loc[d, "bars"] >= 300
                and all_days.index(d) >= 90]

    print(f"\n  3-way comparison: {start} → {end}  ({len(eligible)} days)")

    tolerance_pts = HOD_LOD_TOL
    results: dict[str, dict] = {
        "vp": dict(hod=0, lod=0, both=0, n=0, hd=[], ld=[]),
        "gex": dict(hod=0, lod=0, both=0, n=0, hd=[], ld=[]),
        "comb": dict(hod=0, lod=0, both=0, n=0, hd=[], ld=[]),
    }

    for i, day in enumerate(eligible):
        df_prior  = df2[df2["_day"] < day]
        df_window = df_prior.tail(30_000).copy()
        if len(df_window) < 500:
            continue

        vp_zones,  _          = _vp.run(df_window, anchored=True, bin_size=BIN_SIZE, quiet=True)
        gex_zones, _, _       = _gex.run(df_window, vix_series, bin_size=BIN_SIZE, quiet=True)
        comb_zones, _         = run(df_window, vix_series, quiet=True)

        hod = float(day_stats.loc[day, "hod"])
        lod = float(day_stats.loc[day, "lod"])
        tol = tolerance_pts

        for sys_name, zones in [("vp", vp_zones), ("gex", gex_zones), ("comb", comb_zones)]:
            if not zones:
                continue
            zp  = np.array([z.price for z in zones])
            hd  = float(np.min(np.abs(zp - hod)))
            ld  = float(np.min(np.abs(zp - lod)))
            hh  = hd <= tol
            lh  = ld <= tol
            r   = results[sys_name]
            if hh: r["hod"] += 1
            if lh: r["lod"] += 1
            if hh and lh: r["both"] += 1
            r["n"] += 1
            r["hd"].append(hd); r["ld"].append(ld)

        if (i + 1) % 50 == 0:
            n = results["comb"]["n"]
            if n > 0:
                c = results["comb"]
                print(f"  [{i+1:>3}/{len(eligible)}]  {day}  "
                      f"COMB HOD={c['hod']/c['n']:.1%}  LOD={c['lod']/c['n']:.1%}")

    # Print comparison table
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  3-WAY HOD/LOD COMPARISON  {start} → {end}  (tol=±{tolerance_pts:.0f}pts fixed)")
    print(sep)
    print(f"  {'System':<12}  {'HOD':>6}  {'LOD':>6}  {'Both':>6}  "
          f"{'avgHOD':>8}  {'avgLOD':>8}  {'n':>5}")
    print(sep)

    labels = [("VP",      "vp"),
              ("GEX",     "gex"),
              ("COMBINED","comb")]
    rows_out = []
    for label, key in labels:
        r = results[key]
        if r["n"] == 0:
            continue
        hr  = r["hod"] / r["n"]
        lr  = r["lod"] / r["n"]
        br  = r["both"] / r["n"]
        ahd = float(np.mean(r["hd"])) if r["hd"] else 0.0
        ald = float(np.mean(r["ld"])) if r["ld"] else 0.0
        print(f"  {label:<12}  {hr:>5.1%}  {lr:>5.1%}  {br:>5.1%}  "
              f"{ahd:>7.1f}pts  {ald:>7.1f}pts  {r['n']:>5}")
        rows_out.append(dict(system=label, start=start, end=end,
                             hod_rate=round(hr,3), lod_rate=round(lr,3),
                             both_rate=round(br,3),
                             avg_hod_dist=round(ahd,1), avg_lod_dist=round(ald,1),
                             n_days=r["n"]))

    # Lift vs VP
    vp_r = results["vp"]
    if vp_r["n"] > 0:
        print(sep)
        for label, key in [("GEX vs VP", "gex"), ("COMB vs VP", "comb")]:
            r = results[key]
            if r["n"] == 0: continue
            dh = r["hod"]/r["n"] - vp_r["hod"]/vp_r["n"]
            dl = r["lod"]/r["n"] - vp_r["lod"]/vp_r["n"]
            db = r["both"]/r["n"] - vp_r["both"]/vp_r["n"]
            dad_h = float(np.mean(r["hd"])) - float(np.mean(vp_r["hd"])) if r["hd"] else 0
            dad_l = float(np.mean(r["ld"])) - float(np.mean(vp_r["ld"])) if r["ld"] else 0
            print(f"  {label:<12}  HOD {dh:>+5.1%}  LOD {dl:>+5.1%}  Both {db:>+5.1%}  "
                  f"dist {dad_h:>+5.1f}pts  {dad_l:>+5.1f}pts")
    print(sep)

    out = HERE / "three_way_results.csv"
    pd.DataFrame(rows_out).to_csv(out, index=False)
    print(f"  Saved → {out}")


# ─── Sample charts ────────────────────────────────────────────────────────────

def plot_sample_days(df: pd.DataFrame, vix_series: pd.Series,
                     n: int = 10, bin_size: float = BIN_SIZE,
                     seed: int = None, df_sample_pool=None,
                     output_dir: Path = None) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap

    CMAP = LinearSegmentedColormap.from_list("combined", ["#DD33DD", "#DD00DD"])
    DARK = "#0D1117"

    if seed is not None:
        np.random.seed(seed)

    out_dir = output_dir or HERE
    df2 = df.copy()
    df2["_day"] = df2["date"].dt.date
    pool = df_sample_pool.copy() if df_sample_pool is not None else df2
    pool["_day"] = pool["date"].dt.date

    day_stats = pool.groupby("_day").agg(
        bars=("close", "count"), lo=("low", "min"), hi=("high", "max"), op=("open", "first"),
    )
    all_full = sorted(df2["_day"].unique().tolist())
    eligible = [d for d in day_stats.index.tolist()
                if day_stats.loc[d, "bars"] >= 300
                and all_full.index(d) >= 90]
    if not eligible:
        print("  No eligible days"); return

    rng        = np.random.default_rng(seed=seed)
    candidates = sorted(rng.choice(eligible, size=min(n * 4, len(eligible)),
                                   replace=False).tolist())
    generated  = 0
    print(f"\n  Generating {n} Combined VP+GEX charts...")

    def _draw_ohlc(ax, bars):
        nb = len(bars)
        o, h, l, c = bars["open"].values, bars["high"].values, bars["low"].values, bars["close"].values
        up = c >= o
        wicks = [[(i, l[i]), (i, h[i])] for i in range(nb)]
        wcols = ["#3fb950" if up[i] else "#f85149" for i in range(nb)]
        ax.add_collection(LineCollection(wicks, colors=wcols, linewidths=0.55, zorder=1))
        up_s = [[(i, o[i]), (i, c[i])] for i in range(nb) if up[i]]
        dn_s = [[(i, c[i]), (i, o[i])] for i in range(nb) if not up[i]]
        if up_s: ax.add_collection(LineCollection(up_s, colors="#3fb950", linewidths=3, zorder=2))
        if dn_s: ax.add_collection(LineCollection(dn_s, colors="#f85149", linewidths=3, zorder=2))

    for day in candidates:
        if generated >= n:
            break
        df_prior  = df2[df2["_day"] < day]
        df_window = df_prior.tail(30_000).copy()
        if len(df_window) < 500:
            continue
        day_zones, avg_vix = run(df_window, vix_series, bin_size=bin_size, quiet=True)
        if not day_zones:
            continue

        bars     = df2[df2["_day"] == day].copy().reset_index(drop=True)
        p_lo, p_hi = bars["low"].min(), bars["high"].max()
        spread = p_hi - p_lo; pad = spread * 0.10
        y_lo, y_hi = p_lo - pad, p_hi + pad
        day_open = float(bars["open"].iloc[0])

        # Select zones visible in day range
        qual = [z for z in day_zones if z.n_tf >= 2 and y_lo <= z.price <= y_hi]
        if not qual:
            qual = [z for z in day_zones if y_lo <= z.price <= y_hi]
        if len(qual) < 4:
            continue

        # Balance above/below
        above   = [z for z in qual if z.price > day_open]
        below   = [z for z in qual if z.price <= day_open]
        sel_abv = above[:3]; sel_blw = below[:3]
        if len(sel_abv) < 2: sel_blw = below[:5 - len(sel_abv)]
        elif len(sel_blw) < 2: sel_abv = above[:5 - len(sel_blw)]
        selected = sorted(sel_abv + sel_blw, key=lambda z: z.price)
        if len(selected) < 4:
            continue

        max_score = max(z.score for z in day_zones)

        fig, ax = plt.subplots(figsize=(18, 7), facecolor=DARK)
        ax.set_facecolor(DARK)
        for sp in ax.spines.values(): sp.set_color("#21262D")

        _draw_ohlc(ax, bars)

        hod_p = bars["high"].max(); lod_p = bars["low"].min()
        ax.axhline(hod_p, color="#FFD700", lw=1.2, ls=":", alpha=0.9, zorder=6)
        ax.axhline(lod_p, color="#FFD700", lw=1.2, ls=":", alpha=0.9, zorder=6)
        ax.annotate("HOD", xy=(2, hod_p), xytext=(0, 3),
                    textcoords="offset points", color="#FFD700", fontsize=6,
                    fontfamily="monospace", zorder=7)
        ax.annotate("LOD", xy=(2, lod_p), xytext=(0, -9),
                    textcoords="offset points", color="#FFD700", fontsize=6,
                    fontfamily="monospace", zorder=7)

        for zone in selected:
            ns     = zone.score / max_score
            colour = CMAP(0.20 + 0.80 * ns)
            hw     = zone.price * 0.0005
            cross  = {"vp", "gex"} <= zone.systems
            ax.axhspan(zone.price - hw, zone.price + hw,
                       color=colour, alpha=(0.10 + 0.40 * ns) * (1.5 if cross else 1.0),
                       linewidth=0)
            lw = (0.8 + 1.2 * ns) * (1.3 if cross else 1.0)
            ax.axhline(zone.price, color=colour, lw=lw,
                       alpha=0.60 + 0.30 * ns, ls="--" if not cross else "-")
            role  = "R" if zone.price > day_open else "S"
            sys_s = "VP+GEX" if cross else "|".join(sorted(zone.systems))
            fts   = "|".join(sorted(zone.ftypes))
            ax.annotate(
                f"{zone.price:.0f} [{role}] {sys_s}  {fts}  n_tf={zone.n_tf}",
                xy=(len(bars) - 1, zone.price),
                xytext=(8, 0), textcoords="offset points",
                color=colour, fontsize=6.2, va="center",
                fontfamily="monospace", clip_on=False,
            )

        nb = len(bars); ts = bars["date"].values
        tick_idx = np.linspace(0, nb - 1, min(10, nb), dtype=int)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([pd.Timestamp(ts[i]).strftime("%H:%M") for i in tick_idx],
                           fontsize=7, color="#8B949E")
        ax.set_xlim(0, nb - 1); ax.set_ylim(y_lo, y_hi)
        ax.tick_params(axis="y", colors="#8B949E", labelsize=8)
        ax.tick_params(axis="x", colors="#8B949E", labelsize=7, length=3)
        ax.set_ylabel("Price  (NQ)", color="#8B949E", fontsize=9)
        cross_n = sum(1 for z in selected if {"vp", "gex"} <= z.systems)
        ax.set_title(
            f"NQ Combined VP+GEX  {day}  ·  {len(selected)} zones  "
            f"({cross_n} cross-system)  ·  VIX≈{avg_vix:.1f}  (no look-ahead · orange=solid = VP+GEX overlap)",
            color="#F0F6FC", fontsize=9.5, pad=8, fontweight="bold",
        )
        plt.tight_layout(pad=0.5)
        out = out_dir / f"combined_{day}.png"
        fig.savefig(str(out), dpi=130, bbox_inches="tight", facecolor=DARK)
        plt.close(fig)
        generated += 1
        print(f"  [{generated:02d}/{n}]  {day}  bars={nb}  "
              f"zones={len(selected)}  cross={cross_n}  → combined_{day}.png")

    print(f"  Done. ({generated}/{n} charts)")


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Combined VP+GEX Level Analyzer")
    parser.add_argument("--hod-lod",     action="store_true",
                        help="Walk-forward HOD/LOD coverage (sampled)")
    parser.add_argument("--hod-days",    type=int, default=60)
    parser.add_argument("--compare",     action="store_true",
                        help="Full 3-way VP/GEX/Combined over 2024+2025")
    parser.add_argument("--start",       type=str, default="2024-01-01")
    parser.add_argument("--end",         type=str, default="2025-12-31")
    parser.add_argument("--backtest",    action="store_true",
                        help="Reversal backtest (VP vs GEX vs Combined)")
    parser.add_argument("--years",       type=float, default=2.0,
                        help="Years of history for reversal backtest")
    parser.add_argument("--sample-days", type=int, default=0)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--top",         type=int, default=20)
    args = parser.parse_args()

    print("=" * 60)
    print("  COMBINED VP + GEX LEVEL ANALYZER")
    print("=" * 60)
    print("\nLoading data...")

    from gex_profile import load_nq_data, load_vix
    df         = load_nq_data()
    vix_series = load_vix()
    print(f"  NQ: {len(df):,} bars  {df['date'].min().date()} → {df['date'].max().date()}")

    if args.compare:
        print(f"\nRunning 3-way comparison ({args.start} → {args.end})...")
        compare_three_way(df, vix_series, start=args.start, end=args.end)
        return

    if args.hod_lod:
        print(f"\nRunning combined HOD/LOD coverage ({args.hod_days} days)...")
        hod_lod_coverage(df, vix_series, n_days=args.hod_days, seed=args.seed)
        return

    if args.backtest:
        print(f"\nRunning reversal backtest ({args.years} years)...")
        run_backtest(df, vix_series, years=args.years)
        return

    if args.sample_days > 0:
        pool = df[(df["date"] >= f"{args.start}") & (df["date"] <= f"{args.end}")].copy()
        pool.reset_index(drop=True, inplace=True)
        plot_sample_days(df, vix_series, n=args.sample_days,
                         seed=args.seed, df_sample_pool=pool)
        return

    # Default: today's levels
    print("\nBuilding combined VP+GEX zones...")
    zones, avg_vix = run(df, vix_series)

    sep = "-" * 80
    print(f"\n{sep}")
    print(f"  TOP {min(args.top, len(zones))} COMBINED ZONES  (avg_VIX={avg_vix:.1f})")
    print(sep)
    print(f"  {'PRICE':>9}  {'SCORE':>7}  {'TF':>3}  {'SYSTEMS':<10}  FEATURE TYPES")
    print(sep)
    for z in zones[:args.top]:
        sys_s = "VP+GEX" if {"vp", "gex"} <= z.systems else "|".join(sorted(z.systems))
        fts   = "|".join(sorted(z.ftypes))
        print(f"  {z.price:>9.2f}  {z.score:>7.2f}  {z.n_tf:>3}  {sys_s:<10}  {fts}")
    print(sep)
    cross = sum(1 for z in zones if {"vp", "gex"} <= z.systems)
    print(f"  Cross-system zones (VP+GEX): {cross} / {len(zones)}")


if __name__ == "__main__":
    main()
