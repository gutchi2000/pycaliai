# EXP03 恒常的能力と短期状態の分離

共通基盤は `analysis/mcond/README.md`。本番には接続しない。
結果は `REPORT.md`、先行研究との比較は `PRIOR_ART_AUDIT.md`、事前固定の仕様は `spec.json`。

**結論（詳細はREPORT.md）: Gate 0/1 PASS、Gate 2 主判定（M4 vs M3）FAIL → 仮説終了。Gate 3 未実施。**

## 実行順
```
# 前提: analysis/mcond の market / rebuild_oof_c1 / v6base、exp02 の build が生成済み
python -m pytest analysis/mcond/exp03_latent_state_dev/test_state_identification.py -q   # 4 passed
python -m analysis.mcond.exp03_latent_state_dev.test_time_safety                          # 5本 PASS (数分)
python -m analysis.mcond.exp03_latent_state_dev.build                                      # 格子選択 + 特徴生成 (約1.5時間)
python -m analysis.mcond.exp03_latent_state_dev.run                                        # Gate 0-3
```
中間データ: `data/_research/mcond/exp03_features.parquet`（gitignore）

## 実行ログ（2026-09-18/19）
- test_state_identification: 4 passed
- test_time_safety: 5本 PASS
- build: 選択 半減期=180日, τa²=0.0025, qs=2.0, β=1.0417（innovation定義=候補B）
- run: 初回実行で Gate 1 の相関計算にバグがあり誤FAIL（`groupby.shift()` の行順と元フレームの行順の不一致）。
  `gate1()` 冒頭で `sort_values(["hid","date"]).reset_index(drop=True)` してから計算するよう修正し再実行。
  修正後 Gate 1 は全項目 PASS。詳細は REPORT.md「実装バグの発見と修正」節。
- 仕様固定コミット `3716d370`、spec 未変更で実行

## 定義の要点
- 観測: 走りの強さ = 長期能力 a ＋ 短期状態 s ＋ ノイズ。EXP02と同じ多頭数PL更新(γ=1)をa+sに適用し、分散比で2成分へ配分
- 時間変化: a は τa²×経過日数で分散のみ増加（ランダムウォーク）。s は半減期で0へ指数減衰＋定常分散qs²のAR(1)
- innovation: 相手構成込みの期待からのずれ。採用は期待順位分位−実順位分位（候補B）
- 同日: EXP02と同じく全レースを前日までの状態で予測し、日の終わりにまとめて更新
