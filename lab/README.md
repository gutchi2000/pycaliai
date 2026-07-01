# lab/ — 実験・研究スクリプトアーカイブ

2026-07-01 のルート整理で、ルート直下に散らばっていた実験・研究スクリプト **97 本**をここへ集約した。
本番パイプライン（週次フロー / T-10 / 静的サイト生成）が参照するスクリプトと、現用の手動ツールは
**ルートに残している**（`compute_bets.py` / `export_weekly_marks.py` / `build_site.py` / `build_bet_plan.py` /
`place_weekly.py` / `participation_analyzer.py` / `gutchi_brain.py` など）。

## 実行方法（重要）
移動したスクリプトは **リポジトリルートから `-m` で** 実行する：

```bash
# 例: lab/bet_type_lab/wide_lab.py を動かす
python -m lab.bet_type_lab.wide_lab
```

理由: これらは `import utils` / `import pl_probs` のようにルート直下のモジュールを参照する。
`python -m lab.<theme>.<name>` なら cwd（=ルート）が `sys.path` に乗るので解決する。
`python lab/<theme>/<name>.py` の**直叩きはルートが path に乗らず ImportError になる**。

## テーマ構成
| フォルダ | 中身 | 代表（記憶の再現性アンカー） |
|---|---|---|
| `experiments/` | 予測層 FE / 特徴量実験（exp_*, race_*_exp, traj_*, task_*, test_*, distortion, multi_index 等） | `exp_havoc_m1.py`, `exp_feateng_v7.py` 系 |
| `betting_lab/` | 馬券学習・清算実験（learn_bet_v1..5, learn_max, settle_* 等） | `learn_bet_v5_clv.py` |
| `bet_type_lab/` | 券種別ラボ（wide_*, trio_*, umami_*, crosspool_umaren） | `wide_lab.py`, `wide_holdout_2026.py` |
| `physics_gates/` | 物理/統計力学ゲート・EVT（gate1..4, physics_*, evt_*） | `gate2_3body.py`, `evt_eval.py` |
| `backtest/` | 単発バックテスト各種（backtest_ev, backtest_exotic_*, backtest_frontier, backtest_v2 等） | `backtest_exotic_weapons.py` |
| `train/` | モデル学習の実験版（train_v1000/v5000/v6_multiseed/v7, train_lgbm_*, optuna_v9/v10 等） | `train_v1000.py`, `train_v5000.py` |
| `features_dead/` | 死亡ルートの特徴量ビルダー（build_evt/elo/glicko/pedigree/feats_serious 等） | `build_feats_serious.py` |
| `audits/` | 監査・評価・検証の単発（audit_*, eval_*, validate_*, check_*, xai_marks 等） | `serve_skew_exotic.py` |
| `pipelines_old/` | 旧世代の一括再構築（run_v3/v4/v7/v8_pipeline） | — |
| `sims/` | シミュレーション（sim_tournament_terminal） | — |

## 移動判定の根拠（安全性）
- import 依存グラフ・subprocess/文字列 `X.py` 参照・`*.ps1` 参照・`*.md` の `python X.py` 記載を機械解析
- **どれかに引っかかるものはルート固定**（＝移動後も本番の import / 呼び出しは無傷）
- 移動後の検証: 移動モジュールを import する非 lab ファイル **0 件**、全 97 本 `py_compile` 通過
- 生成物（大容量 parquet / 実験モデル / スクラッチ）は `.gitignore` 済みでディスクに残置
