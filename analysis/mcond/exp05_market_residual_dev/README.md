# EXP05 市場条件付き第2段確率モデル

共通基盤は `analysis/mcond/README.md`。本番には接続しない。
結果は `REPORT.md`、先行研究との比較は `PRIOR_ART_AUDIT.md`、データ監査は `DATA_AUDIT.md`、
2026ロック可否は `LOCKED_PERIOD_AUDIT.md`、事前固定の仕様は `spec.json`。

## 実行順
```
# 前提: analysis/mcond の market / rebuild_oof_c1 / v6base、exp04 の features (exp04_candidates.parquet) が生成済み
python -m analysis.mcond.exp05_market_residual_dev.build_features   # 設計行列 (数分)
python -m analysis.mcond.exp05_market_residual_dev.tie_diagnosis    # p_win同値診断
python -m pytest analysis/mcond/exp05_market_residual_dev/test_time_safety.py -q
python -m analysis.mcond.exp05_market_residual_dev.run              # Gate0/1/2, 条件付き較正, 探索的経済評価 (数分)
```
中間データ: `data/_research/mcond/exp05_design.parquet`（gitignore）

## 定義の要点
- calibrated_v6_probability: 本番pl_calibrators_v6.pkl(valid=2023 fit)は2023年評価に使うとリークするため、
  train(2016-2021)のみでisotonicを自前再fit
- F-full=145 (EXP04のC1+C2+C3をそのまま流用) / F-serve=117 (`data/serve_feature_baseline.json`のserve_dead由来28列を除外)
- M0-M3: 自由fitのロジスティック回帰 (v6+市場の情報圧縮診断を兼ねる、spec本文のB1/B2/B3と同一定義)
- M4/M5: offset(M3) + 表特徴 (F-serve/F-full) の残差L2回帰
- M6: Benter型対照 (w=exp(α log v6_pwin + β log market_pi) をtrainのwin尤度でMLE fit)
- 主比較: M5 vs M1 (Gate1, EXP04再現確認) / M4 vs M3 (Gate2主判定, 探索的) / M5 vs M4 (実運用可能性) / M4 vs M6 (Benter対照)
- **2026年はロックできない** (LOCKED_PERIOD_AUDIT.md)。Gate2以降は2023-2025を使う探索的評価であり最終確証ではない
