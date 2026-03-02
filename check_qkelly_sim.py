import pandas as pd

df = pd.read_csv(r'E:\PyCaLiAI\reports\all_profitable_conditions.csv', encoding='utf-8-sig')

# Quarter-Kelly賭け金をそのまま使用
# 年間投資額 = 賭け金  年間レース数（10年平均）
df['年間レース数'] = df['レース数'] / 10
df['年間投資額'] = df['推奨賭け金(200万)'] * df['年間レース数']

total_inv  = df['年間投資額'].sum()
total_pay  = (df['年間投資額'] * df['回収率'] / 100).sum()
total_net  = total_pay - total_inv
roi        = total_pay / total_inv * 100

print('=== Quarter-Kelly独立賭け金 年間シミュレーション ===')
print(f'年間総投資額: {total_inv:>15,.0f}円')
print(f'年間総払戻額: {total_pay:>15,.0f}円')
print(f'年間純収支  : {total_net:>+15,.0f}円')
print(f'加重平均回収率: {roi:.1f}%')
print()

# 月間投資額
print(f'月間投資額: {total_inv/12:,.0f}円')
print()

# 1レース最大投資額（同一レースで重なる条件の合計）
grp = df.groupby(['場所','クラス'])['推奨賭け金(200万)'].sum().reset_index()
grp = grp.sort_values('推奨賭け金(200万)', ascending=False)
print('=== 1レースあたり最大投資額TOP10（同条件重複時）===')
print(grp.head(10).to_string(index=False))
print()

# Quarter-Kelly閾値別の集計
print('=== Quarter-Kelly閾値別 年間収支 ===')
for thr in [0.0, 0.5, 1.0, 2.0, 3.0]:
    sub = df[df['Quarter-Kelly(%)'] >= thr]
    inv = sub['年間投資額'].sum()
    pay = (sub['年間投資額'] * sub['回収率'] / 100).sum()
    net = pay - inv
    r   = pay / inv * 100 if inv > 0 else 0
    print(f'  {thr:.1f}%以上({len(sub):3d}条件): 投資{inv/10000:,.0f}万 純収支{net/10000:+,.0f}万 回収率{r:.1f}%')
