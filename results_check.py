import pandas as pd, numpy as np, os

s3_sum = pd.read_csv('results/stage3_summary.csv')
print('Shape:', s3_sum.shape)

cols = ['strategy','lookback','min_stack','session','anchored','rolling',
        'ref_type','weighting','combined_hit10','combined_hit20','combined_mae','top_cov','bot_cov']

print('\nTOP 15 by hit@10:')
print(s3_sum.nlargest(15,'combined_hit10')[cols].to_string())

print('\n=== PER STRATEGY ===')
grp = s3_sum.groupby('strategy').agg(
    mean_hit10=('combined_hit10','mean'),
    best_hit10=('combined_hit10','max'),
    mean_hit20=('combined_hit20','mean'),
    mean_mae=('combined_mae','mean'),
    mean_topcov=('top_cov','mean'),
    mean_botcov=('bot_cov','mean'),
).sort_values('best_hit10', ascending=False)
print(grp.to_string())

print('\n=== ref_type ===')
print(s3_sum.groupby('ref_type')[['combined_hit10','combined_hit20','combined_mae','top_cov','bot_cov']].mean().to_string())

print('\n=== BEST SINGLE COMBO ===')
best = s3_sum.iloc[0]
for c in cols:
    print(f'  {c}: {best[c]}')

print('\n=== lookback effect (nearest_hvn only) ===')
hvn = s3_sum[s3_sum['strategy']=='nearest_hvn']
print(hvn.groupby('lookback')[['combined_hit10','combined_hit20','combined_mae','top_cov','bot_cov']].mean().to_string())

print('\n=== min_stack effect (nearest_hvn only) ===')
print(hvn.groupby('min_stack')[['combined_hit10','combined_hit20','combined_mae','top_cov','bot_cov']].mean().to_string())

print('\n=== today_open nearest_hvn top combos ===')
hvn_open = hvn[hvn['ref_type']=='today_open'].nlargest(8,'combined_hit10')
print(hvn_open[cols].to_string())
