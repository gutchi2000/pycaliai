import pandas as pd

df = pd.read_csv(r'E:\PyCaLiAI\reports\backtest_results_train.csv', encoding='utf-8-sig')
df['date'] = pd.to_datetime(df['日付'].astype(str), format='%Y%m%d')
df['曜日'] = df['date'].dt.dayofweek
df['土日'] = df['曜日'].isin([5, 6])
rc = df.groupby(['日付','場所'])['race_id'].nunique().reset_index().rename(columns={'race_id':'R数'})
df = df.merge(rc, on=['日付','場所'], how='left')
df = df[df['土日'] & (df['R数'] >= 10)]

s = df[(df['クラス'] == '新馬') & (df['馬券種'] == '馬連')]

print('=== 全会場 新馬馬連 Kelly分析（10年）===')
for place, grp in s.groupby('場所'):
    cost    = grp['購入額'].sum()
    pay     = grp['実払戻額'].sum()
    roi     = pay / cost * 100
    p       = grp['的中'].mean()
    hit_grp = grp[grp['的中'] == 1]
    if hit_grp.empty or p <= 0:
        print(f'{place}: 的中なし')
        continue
    avg_b  = (hit_grp['実配当(100円)'] / 100).mean() - 1
    if avg_b <= 0:
        print(f'{place}: 配当データなし')
        continue
    f_full = max((p * avg_b - (1-p)) / avg_b, 0)
    f_half = f_full / 2
    f_qtr  = f_full / 4
    n_r    = grp['race_id'].nunique()
    print(f'{place}: 回収率{roi:.1f}% 的中率{p*100:.1f}% 平均配当{avg_b+1:.1f}倍 Half-Kelly{f_half*100:.1f}% Quarter-Kelly{f_qtr*100:.1f}% ({n_r}R)')
