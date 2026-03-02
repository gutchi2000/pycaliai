import pandas as pd

df = pd.read_csv(r'E:\PyCaLiAI\reports\all_profitable_conditions.csv', encoding='utf-8-sig')

# 中山未勝利で例示
example = df[(df['場所']=='中山') & (df['クラス']=='未勝利')].copy()
print('=== 例: 中山未勝利 ===')
print(example[['馬券種','回収率','的中率','平均配当']].to_string(index=False))

# 回収率ウェイトで1万円按分
total_roi = example['回収率'].sum()
example['ウェイト'] = example['回収率'] / total_roi
example['配分額(1万)'] = (example['ウェイト'] * 10000).apply(lambda x: max(int(x//100)*100, 100))
print()
print('回収率ウェイト按分（1万円）:')
print(example[['馬券種','回収率','ウェイト','配分額(1万)']].to_string(index=False))
print(f'合計配分: {example["配分額(1万)"].sum():,}円')

print()
# 全条件でウェイト按分した場合の年間収支
print('=== 全場所クラス馬券種 回収率ウェイト按分シミュレーション ===')
BUDGET_PER_RACE = 10_000

rows = []
for (place, cls), grp in df.groupby(['場所','クラス']):
    total_roi_grp = grp['回収率'].sum()
    for _, row in grp.iterrows():
        weight = row['回収率'] / total_roi_grp
        bet    = max(int(weight * BUDGET_PER_RACE // 100) * 100, 100)
        rows.append({
            '場所': place, 'クラス': cls, '馬券種': row['馬券種'],
            '回収率': row['回収率'],
            'ウェイト': round(weight, 4),
            '配分額': bet,
            'レース数/年': row['レース数'] / 10,
        })

res = pd.DataFrame(rows)
res['年間投資'] = res['配分額'] * res['レース数/年']
res['年間払戻'] = res['年間投資'] * res['回収率'] / 100

total_inv = res['年間投資'].sum()
total_pay = res['年間払戻'].sum()
roi_all   = total_pay / total_inv * 100

print(f'年間総投資額: {total_inv:>12,.0f}円')
print(f'年間総払戻額: {total_pay:>12,.0f}円')
print(f'年間純収支  : {total_pay-total_inv:>+12,.0f}円')
print(f'加重平均回収率: {roi_all:.1f}%')
print(f'月間投資額  : {total_inv/12:>12,.0f}円')
print()
print('=== 場所クラス別 1レース予算按分TOP10 ===')
chk = res.groupby(['場所','クラス'])['配分額'].sum().reset_index().sort_values('配分額',ascending=False)
print(chk.head(10).to_string(index=False))

res.to_csv(r'E:\PyCaLiAI\reports\weighted_allocation.csv', index=False, encoding='utf-8-sig')
print()
print('CSV保存: weighted_allocation.csv')
