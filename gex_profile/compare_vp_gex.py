"""
VP vs GEX HOD/LOD Coverage Comparison (2024 + 2025)
=====================================================
Runs BOTH stacked_vp and gex_profile over all eligible trading days in
2024 and 2025 (no sampling), reports side-by-side results per year.

Usage:
  python gex_profile/compare_vp_gex.py
  python gex_profile/compare_vp_gex.py --no-vp      # GEX only (faster)
  python gex_profile/compare_vp_gex.py --no-gex      # VP only
  python gex_profile/compare_vp_gex.py --year 2024   # single year
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE    = Path(__file__).parent
VP_DIR  = HERE.parent / "intuitiveVP"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(VP_DIR))

import warnings
warnings.filterwarnings("ignore")


# ─── VP coverage runner ───────────────────────────────────────────────────────

def _vp_coverage(df_full: pd.DataFrame, start: str, end: str,
                 tolerance_pts: float = 15.0, bin_size: float = 5.0) -> dict:
    """Walk-forward VP HOD/LOD on all eligible days in [start, end]."""
    from stacked_vp import run as vp_run

    df2 = df_full.copy()
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

    print(f"  [VP]  {start} → {end}  ({len(eligible)} eligible days)")
    hod_hits = lod_hits = both_hits = total = 0
    hod_dists: list[float] = []
    lod_dists: list[float] = []

    for i, day in enumerate(eligible):
        df_prior  = df2[df2["_day"] < day]
        df_window = df_prior.tail(30_000).copy()
        if len(df_window) < 2000:
            continue
        zones, _ = vp_run(df_window, anchored=True, bin_size=bin_size, quiet=True)
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
            print(f"    VP [{i+1:>3}/{len(eligible)}]  {day}  "
                  f"HOD={hod_hits/total:.1%}  LOD={lod_hits/total:.1%}")

    if total == 0:
        return {}
    return dict(hod_rate=hod_hits/total, lod_rate=lod_hits/total,
                both_rate=both_hits/total,
                avg_hod_dist=float(np.mean(hod_dists)),
                avg_lod_dist=float(np.mean(lod_dists)),
                n_days=total)


# ─── GEX coverage runner ──────────────────────────────────────────────────────

def _gex_coverage(df_full: pd.DataFrame, vix_series: pd.Series,
                  start: str, end: str,
                  tolerance_pts: float = 15.0, bin_size: float = 5.0) -> dict:
    """Walk-forward GEX HOD/LOD on all eligible days in [start, end]."""
    from gex_profile import run as gex_run

    df2 = df_full.copy()
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
                and all_days.index(d) >= 10]

    print(f"  [GEX] {start} → {end}  ({len(eligible)} eligible days)")
    hod_hits = lod_hits = both_hits = total = 0
    hod_dists: list[float] = []
    lod_dists: list[float] = []

    for i, day in enumerate(eligible):
        df_prior  = df2[df2["_day"] < day]
        df_window = df_prior.tail(30_000).copy()
        if len(df_window) < 500:
            continue
        zones, _, _ = gex_run(df_window, vix_series, bin_size=bin_size, quiet=True)
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
            print(f"    GEX [{i+1:>3}/{len(eligible)}]  {day}  "
                  f"HOD={hod_hits/total:.1%}  LOD={lod_hits/total:.1%}")

    if total == 0:
        return {}
    return dict(hod_rate=hod_hits/total, lod_rate=lod_hits/total,
                both_rate=both_hits/total,
                avg_hod_dist=float(np.mean(hod_dists)),
                avg_lod_dist=float(np.mean(lod_dists)),
                n_days=total)


# ─── Reporting ────────────────────────────────────────────────────────────────

def _fmt(r: dict) -> str:
    if not r:
        return "  (no data)"
    return (f"  HOD {r['hod_rate']:>5.1%}  LOD {r['lod_rate']:>5.1%}  "
            f"Both {r['both_rate']:>5.1%}  "
            f"avgHOD {r['avg_hod_dist']:>5.1f}pts  "
            f"avgLOD {r['avg_lod_dist']:>5.1f}pts  "
            f"n={r['n_days']}")


def print_comparison(year: str, vp_res: dict, gex_res: dict):
    sep = "=" * 76
    print(f"\n{sep}")
    print(f"  {year}  HOD / LOD COVERAGE COMPARISON")
    print(sep)
    print(f"  VP  {_fmt(vp_res)}")
    print(f"  GEX {_fmt(gex_res)}")
    if vp_res and gex_res:
        dh = gex_res["hod_rate"] - vp_res["hod_rate"]
        dl = gex_res["lod_rate"] - vp_res["lod_rate"]
        db = gex_res["both_rate"] - vp_res["both_rate"]
        print(f"  GEX vs VP:  HOD {dh:+.1%}  LOD {dl:+.1%}  Both {db:+.1%}")
    print(sep)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VP vs GEX HOD/LOD comparison")
    parser.add_argument("--no-vp",  action="store_true")
    parser.add_argument("--no-gex", action="store_true")
    parser.add_argument("--year",   type=str, default=None,
                        help="Single year to test (e.g. 2024). Default: 2024 + 2025")
    parser.add_argument("--tol",    type=float, default=15.0,
                        help="Tolerance in points on each side (default 15 pts)")
    args = parser.parse_args()

    print("=" * 56)
    print("  VP vs GEX HOD/LOD COVERAGE  2024 / 2025")
    print("=" * 56)
    print("\nLoading data...")

    from stacked_vp import load_data as vp_load
    from gex_profile import load_nq_data as gex_load, load_vix

    vp_df  = vp_load()
    gex_df = gex_load()
    vix    = load_vix()

    years = [args.year] if args.year else ["2024", "2025"]
    results: dict[str, dict] = {}

    for yr in years:
        start = f"{yr}-01-01"
        end   = f"{yr}-12-31"
        print(f"\n{'─'*56}\n  {yr}\n{'─'*56}")

        vp_r = {}
        if not args.no_vp:
            print(f"\n  Running VP ({yr})...")
            vp_r = _vp_coverage(vp_df, start, end, tolerance_pts=args.tol)

        gex_r = {}
        if not args.no_gex:
            print(f"\n  Running GEX ({yr})...")
            gex_r = _gex_coverage(gex_df, vix, start, end, tolerance_pts=args.tol)

        results[yr] = {"vp": vp_r, "gex": gex_r}
        print_comparison(yr, vp_r, gex_r)

    # Combined 2-year summary
    if len(years) == 2 and not args.no_vp and not args.no_gex:
        all_vp  = [r["vp"]  for r in results.values() if r["vp"]]
        all_gex = [r["gex"] for r in results.values() if r["gex"]]
        if all_vp and all_gex:
            def _weighted(res_list, key):
                n_total = sum(r["n_days"] for r in res_list)
                return sum(r[key] * r["n_days"] for r in res_list) / n_total

            vp_comb = {k: _weighted(all_vp, k)
                       for k in ("hod_rate","lod_rate","both_rate",
                                 "avg_hod_dist","avg_lod_dist")}
            vp_comb["n_days"] = sum(r["n_days"] for r in all_vp)
            gex_comb = {k: _weighted(all_gex, k)
                        for k in ("hod_rate","lod_rate","both_rate",
                                  "avg_hod_dist","avg_lod_dist")}
            gex_comb["n_days"] = sum(r["n_days"] for r in all_gex)
            print_comparison("2024+2025 COMBINED", vp_comb, gex_comb)

    # Save CSV
    rows = []
    for yr, res in results.items():
        for sys_name, r in res.items():
            if r:
                rows.append({"year": yr, "system": sys_name, **r})
    if rows:
        out = HERE / "compare_results.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\n  Saved → {out}")


if __name__ == "__main__":
    main()
