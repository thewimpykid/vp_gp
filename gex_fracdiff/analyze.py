import pandas as pd, numpy as np

df = pd.read_csv("gex_fracdiff/results/trades_2025.csv")
print("TRADE DISTRIBUTION  (2025 OOS)")
print(f"  Total trades  : {len(df)}")
print(f"  Winners       : {(df.pnl>0).sum()}  (avg={df[df.pnl>0].pnl.mean():.1f} pts)")
print(f"  Losers        : {(df.pnl<=0).sum()}  (avg={df[df.pnl<=0].pnl.mean():.1f} pts)")
print()
print("PnL distribution (pts):")
for p in [5,10,25,50,75,90,95,99]:
    print(f"  p{p:>2}: {np.percentile(df.pnl, p):>7.1f}")
print(f"  min: {df.pnl.min():.1f}   max: {df.pnl.max():.1f}")
print()
print("Largest 10 winners:")
print(df.nlargest(10,"pnl")[["ts","dir","pnl","bars","how"]].to_string(index=False))
print()
print("Largest 5 losers:")
print(df.nsmallest(5,"pnl")[["ts","dir","pnl","bars","how"]].to_string(index=False))
total = df.pnl.sum()
print()
print("Dollar value per contract:")
print(f"  NQ  (5/pt)    : ${total*5:.0f}")
print(f"  MNQ (0.5/pt)  : ${total*0.5:.0f}")
print(f"  3 MNQ         : ${total*0.5*3:.0f}")
print()
# long vs short
longs  = df[df.dir=="L"]
shorts = df[df.dir=="S"]
print("Long-only 2025:")
print(f"  n={len(longs)}  wr={( longs.pnl>0).mean()*100:.1f}%  avg={longs.pnl.mean():.1f}  total={longs.pnl.sum():.1f}")
print("Short-only 2025:")
print(f"  n={len(shorts)}  wr={(shorts.pnl>0).mean()*100:.1f}%  avg={shorts.pnl.mean():.1f}  total={shorts.pnl.sum():.1f}")
