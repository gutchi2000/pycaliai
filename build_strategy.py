import pandas as pd
import json

df = pd.read_csv(r'E:\PyCaLiAI\reports\weighted_allocation.csv', encoding='utf-8-sig')

# {場所: {クラス: {馬券種: {weight, bet_ratio}}}} の形式で保存
BUDGET = 10_000
strategy = {}

for (place, cls), grp in df.groupby(['場所','クラス']):
    if place not in strategy:
        strategy[place] = {}
    strategy[place][cls] = {}
    for _, row in grp.iterrows():
        strategy[place][cls][row['馬券種']] = {
            'roi':       round(row['回収率'], 1),
            'weight':    round(row['ウェイト'], 4),
            'bet_ratio': round(row['ウェイト'], 4),  # 予算に乗じる比率
        }

out = r'E:\PyCaLiAI\data\strategy_weights.json'
with open(out, 'w', encoding='utf-8') as f:
    json.dump(strategy, f, ensure_ascii=False, indent=2)

print(f'保存: {out}')
print(f'会場数: {len(strategy)}')
print(f'総条件数: {sum(len(v2) for v in strategy.values() for v2 in v.values())}')

# 確認: 中山未勝利
print()
print('=== 確認: 中山未勝利 ===')
for bt, info in strategy['中山']['未勝利'].items():
    print(f'  {bt}: ROI{info["roi"]}% ウェイト{info["weight"]:.4f} 1万円で{int(info["bet_ratio"]*10000//100)*100:,}円')
