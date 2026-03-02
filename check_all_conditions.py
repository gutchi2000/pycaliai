import pandas as pd

df = pd.read_csv(r'E:\PyCaLiAI\reports\backtest_results_train.csv', encoding='utf-8-sig')
df['date'] = pd.to_datetime(df['日付'].astype(str), format='%Y%m%d')
df['曜日'] = df['date'].dt.dayofweek
df['土日'] = df['曜日'].isin([5, 6])
rc = df.groupby(['日付','場所'])['race_id'].nunique().reset_index().rename(columns={'race_id':'R数'})
df = df.merge(rc, on=['日付','場所'], how='left')
df = df[df['土日'] & (df['R数'] >= 10)]

rows = []
for (place, cls, bet), grp in df.groupby(['場所','クラス','馬券種']):
    n_r  = grp['race_id'].nunique()
    if n_r < 20:
        continue
    cost = grp['購入額'].sum()
    pay  = grp['実払戻額'].sum()
    roi  = pay / cost * 100
    if roi < 100:
        continue
    p       = grp['的中'].mean()
    hit_grp = grp[grp['的中'] == 1]
    if hit_grp.empty or p <= 0:
        continue
    avg_b  = (hit_grp['実配当(100円)'] / 100).mean() - 1
    if avg_b <= 0:
        continue
    f_full = max((p * avg_b - (1-p)) / avg_b, 0)
    f_qtr  = f_full / 4
    rows.append({
        '場所': place, 'クラス': cls, '馬券種': bet,
        'レース数': n_r,
        '回収率': round(roi, 1),
        '的中率': round(p*100, 1),
        '平均配当': round(avg_b+1, 1),
        'Quarter-Kelly(%)': round(f_qtr*100, 1),
        '推奨賭け金(200万)': int(f_qtr*2000000/100)*100,
    })

res = pd.DataFrame(rows).sort_values('回収率', ascending=False)
print(f'条件数: {len(res)}')
print(res.to_string(index=False))
res.to_csv(r'E:\PyCaLiAI\reports\all_profitable_conditions.csv', index=False, encoding='utf-8-sig')
print('CSV保存: all_profitable_conditions.csv')
