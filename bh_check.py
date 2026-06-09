import pandas as pd, numpy as np
df = pd.read_parquet(r"C:\Users\kirti\Desktop\claudebt\vp_gp\nq1m_2018.parquet")
dc = df["Close"].resample("1D").last().dropna()
for s, e, lbl in [("2024-01-01", "2024-12-31", "2024"), ("2025-01-01", "2026-01-30", "2025-26")]:
    seg = dc.loc[s:e]
    r = seg.pct_change().dropna()
    print(lbl, "buy-hold annSh", round(r.mean() / r.std() * np.sqrt(252), 2),
          "ret%", round((seg.iloc[-1] / seg.iloc[0] - 1) * 100, 1))
