# Band-Coverage Optimization (HOD/LOD @ 15-20pt bands)

Goal: >=80% of daily HODs and LODs within a 15-20pt band (+/-7.5 to +/-10pts)
of a pre-day level. Walk-forward, no look-ahead: levels built only from data
strictly before each target day.

## Method

1. `optimize_bands.py --build-cache` — caches per-day VP zones, GEX zones, and
   prior-day/week OHLC for all 624 eligible days 2024-2025 (0.4s/day).
2. `optimize_bands.py --grid` — in-memory grid over stack tolerance, center
   mode, extra structural families, zone caps. **Tuned on 2024 only,
   validated on untouched 2025.**
3. `optimize_bands.py --diagnose` — miss analysis.

## Key findings

- **Old config (tol_mult=3.0, mean centers, no extras): 62.4% HOD / 69.8% LOD
  at +/-15pt.** At +/-10pt it was far below target.
- Miss diagnosis: median miss distance only 17pts (just outside band). 73% of
  HOD misses were breakouts beyond the prior-day high, with median extension
  just +0.22 ATR beyond PDH. LODs similar (60%, +0.41 ATR).
- Fixes, in order of impact:
  1. **Breakout extension ladders** (`ext`, `ext2`): levels at PDH + k*ATR20
     and PDL - k*ATR20, k = 0.05..1.0. Targets the dominant miss mode.
  2. **Interior ATR ladders** (`ladder`, `ladder2`): prior close +/- k*ATR20.
  3. **Classic structural levels**: Camarilla pivots, floor pivots, prior
     close/mid/VWAP, implied expected-move bands (VIX*1.15), prior week
     close/mid.
  4. **Tight clustering** (tol_mult 3.0 -> 0.75) + **snap centers** (zone
     price = heaviest member's exact price, not cluster mean). Mean-centering
     was dragging zones off exact structural prices.
- Score-ordered min-spacing thinning HURTS coverage (stacked near-duplicate
  levels are doing real work at tight bands). Zone caps also hurt. Full set
  kept.

## Results (2025 validation, untouched during tuning)

| band         | HOD    | LOD    |
|--------------|--------|--------|
| +/-10pt (20) | 88.4%  | 86.8%  |
| +/-7.5pt (15)| 81.7%  | 78.5%  |

2024 (tune set): 92.7% / 90.1% at +/-10; 86.3% / 84.7% at +/-7.5.

## Honesty caveats

- Final config emits ~90 levels within spot +/- 1.5 ATR (~one level / 15pts).
  Coverage at this density is partly geometric.
- Benchmarks at matched density (2025, pooled HOD+LOD, +/-10pt):
  - actual levels: ~88%
  - random uniform levels, same count/span: ~78%  -> **lift +10-12%**
  - ideal evenly-spaced grid, same count: ~96%
- So levels beat random placement decisively but do not beat a perfect grid
  on pure coverage. Their non-coverage value (which level to trade) comes from
  the score ranking — cross-system VP+GEX zones showed the reversal-rate lift
  in `--backtest`.
- NQ daily extremes spread over +/-1.5 ATR (~900-1300pts in 2025); 80%
  coverage with 20pt bands mathematically requires ~45+ well-placed levels.
  No sparse level set can hit 80% at this band width.

## Production config (combined_analyzer.py)

```
STACK_TOLERANCE = 0.75   # 3.75pt cluster radius
SNAP_CENTER     = True
EXTRA_FAMILIES  = {cam, piv, pdc, em, pw, atr, ladder, ladder2, ext, ext2}
HOD_LOD_TOL     = 10.0   # default coverage band half-width
```
