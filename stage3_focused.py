"""
Stage 3 focused sweep.
Locked to today_open + recency + full session (Stage 2 winners).
Only varies: lookback, min_stack, anchored.
~48 stack configs → fast, no OOM.
"""
import os, sys, itertools
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest import (
    load_price_grid, load_all_profiles, load_all_ohlcv,
    build_week_month_profiles, aggregate_profiles,
    detect_lvn_fast, detect_hvn_fast, label_shelves_fast,
    build_va_cache, rolled_va, stacked_va_level,
    predict, score_day, daily_summary, aggregate,
    print_top, plot_param_heatmap, plot_best_daily, RESULTS_DIR,
)

STRATEGIES   = ["nearest_hvn", "hvn_beyond_shelf"]
LOOKBACKS    = [5, 10, 15, 20]
MIN_STACKS   = [2, 3, 4, 5, 6]
SESSION      = "full"
ANCHORED     = [["day"], ["day","week","month"], ["week","month"]]
ROLLING      = [[]]          # no rolling (Stage 2 showed it adds noise)
LVN_PARAMS   = [
    {"smooth": 5, "pct": 25.0, "depth_pct": 60.0},
    {"smooth": 5, "pct": 25.0, "depth_pct": 70.0},
    {"smooth": 7, "pct": 30.0, "depth_pct": 60.0},
]
VA_PCTS      = [0.70]
REF_TYPES    = ["today_open"]
WEIGHTINGS   = ["recency"]
SHELF_TICKS  = [2]
SEARCH_RANGE = 300.0
VA_TOL       = 5.0

price_grid  = load_price_grid()
daily       = load_all_profiles(SESSION)
ohlcv_data  = load_all_ohlcv("rth")
sorted_dates = sorted(daily.keys())
wk_prof, mo_prof = build_week_month_profiles(daily, sorted_dates)

daily_stats = {d: daily_summary(ohlcv_data.get(d, pd.DataFrame()))
               for d in sorted_dates}
daily_stats = {d: s for d, s in daily_stats.items() if s}

# VA cache
va_cache_store = {}
for va_pct in VA_PCTS:
    va_cache_store[va_pct] = build_va_cache(daily, price_grid, va_pct)

# ── pre-compute stacks ──
print("Pre-computing stacks…", flush=True)
stack_cache = {}
stack_configs = list(itertools.product(
    range(len(ANCHORED)), LOOKBACKS, range(len(LVN_PARAMS))
))
n_sc = len(stack_configs)
print(f"  {n_sc} unique stack configs × ~{len(daily_stats)} test days", flush=True)

for si, (ai, lb, pi) in enumerate(stack_configs):
    anchored = ANCHORED[ai]
    lvn_p    = LVN_PARAMS[pi]
    if (si+1) % max(1, n_sc // 6) == 0:
        print(f"  {si+1}/{n_sc}  anch={'|'.join(anchored)} lb={lb} pct={lvn_p['pct']}", flush=True)

    for i, test_date in enumerate(sorted_dates):
        if i < 2 or test_date not in daily_stats:
            continue
        prior = sorted_dates[max(0, i-lb): i]
        if len(prior) < 2:
            continue
        profile_list = aggregate_profiles(daily, prior, anchored, [], "recency", wk_prof, mo_prof)
        n = len(profile_list[0]) if profile_list else 1
        stack_raw = np.zeros(n, dtype=np.int16)
        hvn_raw   = np.zeros(n, dtype=np.int16)
        for p in profile_list:
            if p.sum() == 0:
                continue
            stack_raw += detect_lvn_fast(p, **lvn_p).astype(np.int16)
            hvn_raw   += detect_hvn_fast(p, smooth=lvn_p.get("smooth", 5)).astype(np.int16)
        stack_cache[(ai, lb, pi, test_date)] = (stack_raw, hvn_raw)

print(f"  {len(stack_cache):,} stacks cached.", flush=True)

# ── rolled_va cache ──
print("Pre-computing rolled_va…", flush=True)
rv_cache = {}
for lb in LOOKBACKS:
    for va_pct in VA_PCTS:
        for i, test_date in enumerate(sorted_dates):
            if i < 2 or test_date not in daily_stats:
                continue
            prior = sorted_dates[max(0, i-lb): i]
            if len(prior) < 2:
                continue
            rv_cache[(lb, test_date, va_pct)] = rolled_va(daily, prior, price_grid, va_pct)
print(f"  {len(rv_cache):,} entries.", flush=True)

# ── fan-out ──
combos = list(itertools.product(
    STRATEGIES, LOOKBACKS, MIN_STACKS,
    range(len(ANCHORED)), range(len(LVN_PARAMS)),
    VA_PCTS, REF_TYPES, WEIGHTINGS, SHELF_TICKS,
))
print(f"\n{len(combos):,} prediction combos…", flush=True)

all_rows = []
for ci, (strat, lb, ms, ai, pi, va_pct, ref_type, weighting, shelf_ticks) in enumerate(combos):
    anchored = ANCHORED[ai]
    lvn_p    = LVN_PARAMS[pi]
    va_cache = va_cache_store[va_pct]

    if (ci+1) % max(1, len(combos)//10) == 0:
        print(f"  [{ci+1:>5}/{len(combos)}] {strat:<22} lb={lb} ms={ms} anch={'|'.join(anchored)}", flush=True)

    for i, test_date in enumerate(sorted_dates):
        if i < 2 or test_date not in daily_stats:
            continue
        prior = sorted_dates[max(0, i-lb): i]
        if len(prior) < 2:
            continue

        key = (ai, lb, pi, test_date)
        if key not in stack_cache:
            continue

        stack_raw, hvn_raw = stack_cache[key]
        stack        = stack_raw
        stacked_mask = stack >= ms
        hvn_mask     = hvn_raw >= max(1, ms // 2)
        shelf_mask   = label_shelves_fast(stacked_mask, min_run=shelf_ticks)

        ds         = daily_stats[test_date]
        prior_date = sorted_dates[i-1]
        prior_stats = daily_stats.get(prior_date, {})
        va_prior   = va_cache.get(prior_date, {})

        # today_open ref
        ref = ds["open"]

        va_rolled = rv_cache.get((lb, test_date, va_pct), {"poc":None,"val":None,"vah":None})
        va_stk_vah = stacked_va_level(va_cache, prior, "vah", VA_TOL)
        va_stk_val = stacked_va_level(va_cache, prior, "val", VA_TOL)

        pred_top, pred_bot = predict(
            stack, stacked_mask, shelf_mask, hvn_mask,
            price_grid, ref, strat,
            va_prior, va_rolled, va_stk_vah, va_stk_val, SEARCH_RANGE,
        )

        scores = score_day(pred_top, pred_bot, ds["high"], ds["low"])

        all_rows.append({
            "date": test_date, "strategy": strat,
            "lookback": lb, "min_stack": ms, "session": SESSION,
            "anchored": "+".join(sorted(anchored)), "rolling": "none",
            "smooth": lvn_p["smooth"], "pct": lvn_p["pct"], "depth_pct": lvn_p["depth_pct"],
            "va_pct": va_pct, "ref_type": ref_type, "weighting": weighting,
            "shelf_ticks": shelf_ticks,
            "ref_price": ref, "pred_top": pred_top, "pred_bot": pred_bot,
            "actual_high": ds["high"], "actual_low": ds["low"],
            **scores,
        })

df = pd.DataFrame(all_rows)
df.to_csv(os.path.join(RESULTS_DIR, "stage3_results.csv"), index=False)
s3_sum = aggregate(df)
s3_sum.to_csv(os.path.join(RESULTS_DIR, "stage3_summary.csv"), index=False)

print("\n" + "="*60)
print_top(s3_sum, n=15)

print("\nPlotting…")
for metric in ["combined_hit10", "combined_hit20", "combined_mae"]:
    plot_param_heatmap(s3_sum, RESULTS_DIR, metric)
if not df.empty:
    plot_best_daily(df, s3_sum, RESULTS_DIR)

print(f"\nDone → {RESULTS_DIR}")
