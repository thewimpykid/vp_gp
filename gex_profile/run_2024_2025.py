"""Generate GEX sample charts from 2024-2025 only."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from gex_profile import load_nq_data, load_vix, plot_sample_days

df         = load_nq_data()
vix_series = load_vix()

df_pool = df[(df["date"] >= "2024-01-01") & (df["date"] <= "2025-12-31")].copy()
df_pool.reset_index(drop=True, inplace=True)
print(f"  2024-2025 pool: {len(df_pool):,} bars  "
      f"{df_pool['date'].min().date()} → {df_pool['date'].max().date()}")

plot_sample_days(df, vix_series, n=30, bin_size=5.0, seed=77,
                 df_sample_pool=df_pool)
