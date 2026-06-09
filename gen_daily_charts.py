"""
gen_daily_charts.py — render HVN confluence heatmap PNG for every cached day.

Output: results/daily_charts/YYYY-MM-DD.png
Predictions auto-filled from Stage-3 best combo (nearest_hvn lb=15 ms=6).
"""
import argparse
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chart as ch

_ap = argparse.ArgumentParser()
_ap.add_argument("--gex", action="store_true",
                 help="VP+GEX dual heatmap → results/daily_charts_gex/")
ARGS = _ap.parse_args()

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROFILE_DIR = os.path.join(BASE_DIR, "cache", "profiles")
OUT_DIR     = os.path.join(
    BASE_DIR, "results", "daily_charts_gex" if ARGS.gex else "daily_charts")
os.makedirs(OUT_DIR, exist_ok=True)

LOOKBACK = 15
SESSION  = "full"
HVN_PCT  = 60.0
SMOOTH   = 5

# ── load Stage-3 best-combo predictions ──
preds: dict[str, tuple] = {}
fp = os.path.join(BASE_DIR, "results", "stage3_results.csv")
if os.path.exists(fp):
    df = pd.read_csv(fp, dtype={"date": str})
    best = df[
        (df.strategy == "nearest_hvn") & (df.lookback == 15) &
        (df.min_stack == 6) & (df.anchored == "day+month+week") &
        (df.pct == 25.0) & (df.depth_pct == 60.0) & (df.smooth == 5)
    ]
    for _, r in best.iterrows():
        pt = float(r["pred_top"]) if pd.notna(r["pred_top"]) else None
        pb = float(r["pred_bot"]) if pd.notna(r["pred_bot"]) else None
        preds[str(r["date"])] = (pt, pb)
print(f"Loaded predictions for {len(preds)} days")

# ── all cached dates ──
dates = sorted(
    f.replace("_full.npy", "")
    for f in os.listdir(PROFILE_DIR)
    if f.endswith("_full.npy")
)
print(f"{len(dates)} cached days\n")

ok, skipped, failed = 0, 0, 0
for d in dates:
    date_iso = f"{d[:4]}-{d[4:6]}-{d[6:]}"
    out_fp   = os.path.join(OUT_DIR, f"{date_iso}.png")
    pt, pb   = preds.get(d, (None, None))
    try:
        ch.make_chart(
            date       = date_iso,
            start_time = "09:30",
            end_time   = "16:00",
            lookback   = LOOKBACK,
            session    = SESSION,
            hvn_pct    = HVN_PCT,
            smooth     = SMOOTH,
            pred_high  = pt,
            pred_low   = pb,
            save_path  = out_fp,
            show_gex   = ARGS.gex,
        )
        ok += 1
    except ValueError as e:
        # first few days have no prior sessions / weekend stubs
        print(f"  skip {date_iso}: {e}")
        skipped += 1
    except Exception as e:
        print(f"  FAIL {date_iso}: {e}")
        failed += 1

print(f"\nDone: {ok} charts, {skipped} skipped, {failed} failed")
print(f"→ {OUT_DIR}")
