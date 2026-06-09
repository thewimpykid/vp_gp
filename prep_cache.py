"""
prep_cache.py  —  one-time preprocessing

Builds under cache/:
  price_grid.npy              global 0.25pt grid (9886 bins, ~40KB)
  profiles/YYYYMMDD_full.npy  full-session (24h) volume profile
  profiles/YYYYMMDD_rth.npy   RTH-only (09:30-16:00 ET) volume profile
  ohlcv/YYYYMMDD.parquet      1-min OHLCV for rth + full sessions (~45KB each)
  trades/YYYYMMDD.parquet     filtered trade rows only (~8MB vs ~200MB raw)

Run once:
    python prep_cache.py
    python prep_cache.py --force          # rebuild everything
    python prep_cache.py --skip-trades    # skip large trade files, rebuild profiles only
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vol_profile as vp

BASE        = os.path.dirname(os.path.abspath(__file__))
CACHE       = os.path.join(BASE, "cache")
TRADES_DIR  = os.path.join(CACHE, "trades")
OHLCV_DIR   = os.path.join(CACHE, "ohlcv")
PROFILE_DIR = os.path.join(CACHE, "profiles")
GRID_FILE   = os.path.join(CACHE, "price_grid.npy")

ET = ZoneInfo("America/New_York")
RTH_START = "09:30"
RTH_END   = "16:00"


def make_dirs():
    for d in [TRADES_DIR, OHLCV_DIR, PROFILE_DIR]:
        os.makedirs(d, exist_ok=True)


def extract_trades(raw_fp: str, out_fp: str, force: bool) -> pd.DataFrame:
    if os.path.exists(out_fp) and not force:
        return pd.read_parquet(out_fp)
    raw = pd.read_parquet(raw_fp)
    t = raw[
        (raw["msg_type"]   == 2) &
        (raw["book_level"] == 0) &
        (raw["size"]       > 0)
    ][["raw_timestamp", "price", "size"]].copy()
    t["ts"] = pd.to_datetime(
        t["raw_timestamp"].astype(str).str[:14],
        format="%Y%m%d%H%M%S", utc=True,
    )
    t = t.drop(columns=["raw_timestamp"]).set_index("ts").sort_index()
    t.to_parquet(out_fp)
    return t


def resample_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["open","high","low","close","volume"])
    o = df["price"].resample("1min").first()
    h = df["price"].resample("1min").max()
    l = df["price"].resample("1min").min()
    c = df["price"].resample("1min").last()
    v = df["size"].resample("1min").sum()
    out = pd.DataFrame({"open":o,"high":h,"low":l,"close":c,"volume":v})
    return out.dropna(subset=["open"])


def build_ohlcv(trades: pd.DataFrame, date_str: str, out_fp: str, force: bool):
    if os.path.exists(out_fp) and not force:
        return
    et = trades.tz_convert(ET)

    rth_start = pd.Timestamp(f"{date_str} {RTH_START}", tz=ET)
    rth_end   = pd.Timestamp(f"{date_str} {RTH_END}",   tz=ET)

    full = resample_ohlcv(et)
    rth  = resample_ohlcv(et[(et.index >= rth_start) & (et.index <= rth_end)])

    full["session"] = "full"
    rth["session"]  = "rth"
    pd.concat([full, rth]).to_parquet(out_fp)


def build_profiles(
    trades: pd.DataFrame,
    date_str: str,
    price_grid: np.ndarray,
    force: bool,
):
    """Build both full-session and RTH profiles."""
    fp_full = os.path.join(PROFILE_DIR, f"{date_str}_full.npy")
    fp_rth  = os.path.join(PROFILE_DIR, f"{date_str}_rth.npy")

    needs_full = not os.path.exists(fp_full) or force
    needs_rth  = not os.path.exists(fp_rth)  or force

    if not needs_full and not needs_rth:
        return False   # nothing to do

    et = trades.tz_convert(ET)

    if needs_full:
        prof = vp.build_profile(trades, price_grid)
        np.save(fp_full, prof.astype(np.float32))

    if needs_rth:
        rth_start = pd.Timestamp(f"{date_str} {RTH_START}", tz=ET)
        rth_end   = pd.Timestamp(f"{date_str} {RTH_END}",   tz=ET)
        rth_trades = trades[(et.index >= rth_start) & (et.index <= rth_end)]
        prof = vp.build_profile(rth_trades, price_grid)
        np.save(fp_rth, prof.astype(np.float32))

    return True


def discover_global_range(files: dict) -> tuple[float, float]:
    mins, maxs = [], []
    sampled = list(files.values())[::4]
    print(f"  Sampling {len(sampled)} files for price range…", flush=True)
    for fp in sampled:
        raw = pd.read_parquet(fp, columns=["msg_type","book_level","price","size"])
        t = raw[(raw["msg_type"]==2)&(raw["book_level"]==0)&(raw["size"]>0)]
        if not t.empty:
            mins.append(float(t["price"].min()))
            maxs.append(float(t["price"].max()))
    return min(mins) - 50, max(maxs) + 50


def load_or_build_grid(files: dict, force: bool) -> np.ndarray:
    if os.path.exists(GRID_FILE) and not force:
        grid = np.load(GRID_FILE)
        print(f"  Price grid loaded: {grid[0]:.2f}–{grid[-1]:.2f} ({len(grid)} bins)", flush=True)
        return grid.astype(np.float64)
    lo, hi = discover_global_range(files)
    grid = vp._price_grid(lo, hi)
    np.save(GRID_FILE, grid.astype(np.float32))
    print(f"  Price grid built: {lo:.2f}–{hi:.2f} ({len(grid)} bins)", flush=True)
    return grid


def run(force: bool = False, skip_trades: bool = False):
    make_dirs()
    raw_files = vp._discover_files(BASE)
    print(f"Found {len(raw_files)} raw parquet files.\n", flush=True)

    print("=== Price grid ===", flush=True)
    price_grid = load_or_build_grid(raw_files, force)
    print(flush=True)

    for i, (date_str, raw_fp) in enumerate(sorted(raw_files.items())):
        trades_fp = os.path.join(TRADES_DIR,  f"{date_str}.parquet")
        ohlcv_fp  = os.path.join(OHLCV_DIR,   f"{date_str}.parquet")
        fp_full   = os.path.join(PROFILE_DIR, f"{date_str}_full.npy")
        fp_rth    = os.path.join(PROFILE_DIR, f"{date_str}_rth.npy")

        all_cached = (
            (os.path.exists(trades_fp) or skip_trades) and
            os.path.exists(ohlcv_fp) and
            os.path.exists(fp_full) and
            os.path.exists(fp_rth)
        )
        if all_cached and not force:
            print(f"[{i+1:02d}/{len(raw_files)}] {date_str}  cached ✓", flush=True)
            continue

        raw_mb = os.path.getsize(raw_fp) / 1e6
        print(f"[{i+1:02d}/{len(raw_files)}] {date_str}  ({raw_mb:.0f}MB)", flush=True)

        # load trades (from cache or raw)
        if skip_trades and os.path.exists(trades_fp):
            trades = pd.read_parquet(trades_fp)
        else:
            trades = extract_trades(raw_fp, trades_fp, force)
            sz = os.path.getsize(trades_fp) / 1e6 if os.path.exists(trades_fp) else 0
            print(f"  trades: {len(trades):,} rows  →  {sz:.1f}MB", flush=True)

        ds_fmt = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"

        if not os.path.exists(ohlcv_fp) or force:
            build_ohlcv(trades, ds_fmt, ohlcv_fp, force)
            sz = os.path.getsize(ohlcv_fp) / 1e3 if os.path.exists(ohlcv_fp) else 0
            print(f"  ohlcv: saved ({sz:.0f}KB)", flush=True)

        built = build_profiles(trades, date_str, price_grid, force)
        if built:
            sz_f = os.path.getsize(fp_full) / 1e3
            sz_r = os.path.getsize(fp_rth)  / 1e3
            print(f"  profiles: full={sz_f:.0f}KB  rth={sz_r:.0f}KB", flush=True)

    print("\n=== Cache summary ===", flush=True)
    for label, d, ext in [
        ("Trades",   TRADES_DIR,  "parquet"),
        ("OHLCV",    OHLCV_DIR,   "parquet"),
        ("Profiles", PROFILE_DIR, "npy"),
    ]:
        fs   = [f for f in os.listdir(d) if f.endswith(ext)]
        size = sum(os.path.getsize(os.path.join(d, f)) for f in fs) / 1e6
        print(f"  {label}: {len(fs)} files, {size:.1f}MB", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force",       action="store_true")
    p.add_argument("--skip-trades", action="store_true")
    args = p.parse_args()
    run(force=args.force, skip_trades=args.skip_trades)
