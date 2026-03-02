import pandas as pd
import numpy as np

df = pd.read_csv(r'E:\PyCaLiAI\reports\all_profitable_conditions.csv', encoding='utf-8-sig')

# 回収率ウェイトで予算按分
# 1レース予算: 10,000円
BUDGET = 10_000

# ウェイト = 回収率 / 全条件の回収率合計
df['ウェイト'] = df['回収率'] / df['回収率'].sum()
df['配分額'] = (df['ウェイト'] * BUDGET).apply(lambda x: max(int(x // 100) * 100, 100))

# 同一レースに何条件重なるか（場所クラスで重なる馬券種数）
overlap = df.groupby(['場所','クラス'])['馬券種'].count().reset_index().rename(columns={'馬券種':'重複条件数'})
df = df.merge(overlap, on=['場所','クラス'], how='left')

print('=== 回収率上位20条件の配分額 ===')
print(df[['場所','クラス','馬券種','回収率','ウェイト','配分額']].head(20).to_string(index=False))

print()
print('=== 配分額の分布 ===')
print(df['配分額'].describe().apply(lambda x: f'{x:,.0f}'))

print()
print('=== 1レース最大投資額（重複条件が全部該当した場合）===')
grp = df.groupby(['場所','クラス'])['配分額'].sum().reset_index().sort_values('配分額', ascending=False)
print(grp.head(10).to_string(index=False))

print()
# 実際の年間投資額シミュレーション（10年平均レース数から）
total_races_per_year = df['レース数'].sum() / 10
total_investment = (df['配分額'] * df['レース数'] / 10).sum()
print(f'年間想定レース数: {total_races_per_year:,.0f}R')
print(f'年間想定投資額: {total_investment:,.0f}円')
print(f'月間想定投資額: {total_investment/12:,.0f}円')
