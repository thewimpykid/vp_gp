"""Render Combined VP+GEX charts for every trading day in 2025."""
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "gex_profile"))
sys.path.insert(0, str(HERE.parent / "intuitiveVP"))

from gex_profile import load_nq_data, load_vix
from combined_analyzer import plot_sample_days

out_dir = HERE / "charts_2025"
out_dir.mkdir(exist_ok=True)

print("Loading data...")
df  = load_nq_data()
vix = load_vix()

pool = df[(df["date"] >= "2025-01-01") & (df["date"] <= "2025-12-31")].copy()
pool.reset_index(drop=True, inplace=True)
print(f"  2025 pool: {len(pool):,} bars  "
      f"{pool['date'].min().date()} → {pool['date'].max().date()}")

# n=400 > trading days in 2025 (~252) → tries every eligible day
plot_sample_days(df, vix, n=400, bin_size=5.0, seed=42,
                 df_sample_pool=pool, output_dir=out_dir)
