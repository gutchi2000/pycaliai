import pandas as pd

df = pd.read_csv(r'E:\PyCaLiAI\reports\backtest_results_train.csv', encoding='utf-8-sig')
df['date'] = pd.to_datetime(df['日付'].astype(str), format='%Y%m%d')
df['曜日'] = df['date'].dt.dayofweek
df['土日'] = df['曜日'].isin([5, 6])
rc = df.groupby(['日付','場所'])['race_id'].nunique().reset_index().rename(columns={'race_id':'R数'})
df = df.merge(rc, on=['日付','場所'], how='left')
df = df[df['土日'] & (df['R数'] >= 10)]

s12 = df[
    (df['クラス'] == '新馬') &
    (df['馬券種'] == '馬連') &
    (df['場所'].isin(['東京','中山','中京','小倉']))
].copy()

print('=== 会場別 Kelly分析（10年分）===')
for place, grp in s12.groupby('場所'):
    cost    = grp['購入額'].sum()
    pay     = grp['実払戻額'].sum()
    roi     = pay / cost * 100
    p       = grp['的中'].mean()
    hit_grp = grp[grp['的中'] == 1]
    avg_b   = (hit_grp['実配当(100円)'] / 100).mean() - 1
    f_full  = max((p * avg_b - (1-p)) / avg_b, 0)
    f_half  = f_full / 2
    print(f'{place}: 的中率{p*100:.1f}% 平均配当{avg_b+1:.1f}倍 回収率{roi:.1f}% Half-Kelly{f_half*100:.1f}% 200万/{int(f_half*2000000/100)*100:,}円')
