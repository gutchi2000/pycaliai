import pandas as pd

df = pd.read_csv(r'E:\PyCaLiAI\reports\backtest_results_train.csv', encoding='utf-8-sig')
df['year'] = df['日付'].astype(str).str[:4].astype(int)

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

rows = []
for year, grp in s12.groupby('year'):
    cost = grp['購入額'].sum()
    pay  = grp['実払戻額'].sum()
    roi  = pay / cost * 100 if cost > 0 else 0
    hits = grp['的中'].sum()
    n    = len(grp)
    rows.append({'年':year,'レース数':grp['race_id'].nunique(),
                 '点数':n,'投資':int(cost),'払戻':int(pay),
                 '純収支':int(pay-cost),'回収率':round(roi,1),
                 '的中率':round(hits/n*100,1)})

res = pd.DataFrame(rows)
print(res.to_string(index=False))
print()
print('10年平均回収率:', round(res['回収率'].mean(),1), '%')
print('100%超え年数:', (res['回収率']>=100).sum(), '/10年')
print('最低回収率:', res['回収率'].min(), '%')
print('最高回収率:', res['回収率'].max(), '%')
