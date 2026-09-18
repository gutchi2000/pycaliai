# EXP04 環境をまたいで残る情報の選別

共通基盤は `analysis/mcond/README.md`。本番には接続しない。
結果は `REPORT.md`、先行研究との比較は `PRIOR_ART_AUDIT.md`、データ監査は `DATA_AUDIT.md`、事前固定の仕様は `spec.json`。

## 実行順
```
# 前提: analysis/mcond の market / rebuild_oof_c1 / v6base、exp01/exp02/exp03 の build が生成済み
python -m analysis.mcond.exp04_invariant_info_dev.data_audit     # C1候補の年別被覆率監査 (数分)
python -m analysis.mcond.exp04_invariant_info_dev.features       # C1+C2+C3 候補行列の生成 (数分)
python -m pytest analysis/mcond/exp04_invariant_info_dev/test_environment_split.py -q
python -m analysis.mcond.exp04_invariant_info_dev.test_time_safety
python -m analysis.mcond.exp04_invariant_info_dev.run            # Gate 0-3, LOEO診断 (~30-60分)
```
中間データ: `data/_research/mcond/exp04_candidates.parquet`（gitignore）

## 定義の要点
- 環境: E1年度(2016-2025) / E2競馬場(10場) / E3芝ダ×距離帯(8セル)。TRAIN(2016-2021)で1000レース未満の水準は統合(今回は該当なし)
- 候補特徴: C1=v6の120特徴から機械監査で105→133列(数値化後) / C2=EXP01の生の選択8列 / C3=EXP02能力2列+EXP03経験2列。計145列
- offset: v6+市場(発走31-38分前)のロジット和。M1として先に通常フィットし、その予測をoffsetとして固定した残差モデルをM2-M5が学習
- I1: 特徴ごとに年度・競馬場・芝ダ距離帯別の効果方向の一致率+開催日ブロックbootstrap200回の符号安定性+BH-FDRで選別
- I2: Group-DRO(交互最適化)。3軸の環境水準を1つのグループ集合として管理し、損失の大きいグループの重みを指数的に上げながら再学習
- 主比較: M3(安定特徴) vs M1(追加情報の有無) / vs M2(通常一括学習との比較) / vs M5(同数の通常選択との比較)
