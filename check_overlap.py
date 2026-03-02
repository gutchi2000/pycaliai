import pandas as pd

df = pd.read_csv(r'E:\PyCaLiAI\reports\all_profitable_conditions.csv', encoding='utf-8-sig')

print('=== Quarter-Kelly閾値別 条件数 ===')
for thr in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
    n = (df['Quarter-Kelly(%)'] >= thr).sum()
    avg_bet = int(df[df['Quarter-Kelly(%)'] >= thr]['推奨賭け金(200万)'].mean()) if n > 0 else 0
    print(f'  {thr:.1f}%以上: {n:3d}条件  平均賭け金{avg_bet:,}円/R')

print()
print('=== 馬券種別 条件数（全361）===')
print(df.groupby('馬券種')['回収率'].agg(['count','mean']).rename(columns={'count':'条件数','mean':'平均回収率'}).round(1))

print()
print('=== クラス別 条件数（Quarter-Kelly 1%以上）===')
sub = df[df['Quarter-Kelly(%)'] >= 1.0]
print(sub.groupby('クラス')['回収率'].agg(['count','mean']).rename(columns={'count':'条件数','mean':'平均回収率'}).sort_values('条件数', ascending=False).round(1))

print()
print('=== 同一レースに重なる可能性がある条件数の最大値 ===')
overlap = df.groupby(['場所','クラス'])['馬券種'].count().reset_index().rename(columns={'馬券種':'馬券種数'})
print(overlap.sort_values('馬券種数', ascending=False).head(10).to_string(index=False))
