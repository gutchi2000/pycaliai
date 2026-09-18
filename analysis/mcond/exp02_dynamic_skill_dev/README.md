# EXP02 動的対戦能力と不確実性

共通基盤は `analysis/mcond/README.md`（市場確率・v6 OOF・評価器・データの事実表）。本番には接続しない。
結果は `REPORT.md`、先行研究との比較は `PRIOR_ART_AUDIT.md`、事前固定の仕様は `spec.json`。

## 実行順
```
# 前提: analysis/mcond の market / rebuild_oof_c1 / v6base と exp01 の build_features が生成済み
python -m pytest analysis/mcond/exp02_dynamic_skill_dev/test_boundary.py -q     # 7 passed
python -m analysis.mcond.exp02_dynamic_skill_dev.test_time_safety               # 4本 PASS (数分)
python -m analysis.mcond.exp02_dynamic_skill_dev.build                          # 格子選択 + 特徴生成 (約10分)
python -m analysis.mcond.exp02_dynamic_skill_dev.run                            # Gate 0-3
python -m analysis.mcond.exp02_dynamic_skill_dev.supplement                     # 探索分析
```
中間データ: `data/_research/mcond/exp02_features.parquet`（gitignore）

## 実行ログ（2026-09-18）
- test_boundary: 7 passed
- test_time_safety: test_future_deletion / test_same_day_later_race_deletion / test_same_day_earlier_result_not_used / test_target_result_not_used すべて PASS
- build: 選択 β=2.0833, τ²/日=0.005（2016-2021 次走勝ち対数尤度 −2.54928）
- run: 仕様固定コミット 02084801、spec 未変更で実行

## 定義の要点
- 更新: Weng & Lin (2011) の Plackett–Luce 版。1レースの全着順を1観測として閉形式更新（γ=1）
- 時間変化: 予測時点で 分散 += τ² × 前回更新からの日数
- 同日: 全レースを前日までの状態で予測し、日の終わりにまとめて更新
- T2: 全体能力＋芝ダ効果＋距離帯効果（4帯）、効果は N(0, (25/9)²) から始めて分散比で更新を配分
- 着差は使わない。取消・除外・中止は参加者外、同着は同順位
