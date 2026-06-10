"""Generate 15 sample day charts from 2024-2025 only."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from stacked_vp import load_data, plot_sample_days

df = load_data()
df25 = df[(df["date"] >= "2025-01-01") & (df["date"] <= "2025-12-31")].copy()
df25.reset_index(drop=True, inplace=True)
print(f"  2025 subset: {len(df25):,} bars  "
      f"{df25['date'].min().date()} → {df25['date'].max().date()}")

plot_sample_days(df, zones=[], n=30, bin_size=5.0, seed=77,
                 df_sample_pool=df25)
