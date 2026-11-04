# legacy/ — 旧系統 (Streamlit / 8 モデルアンサンブル) の隔離先

2026-11-04、`docs/LEGACY_RETIREMENT_PLAN_20261004.md` の段階 3 (縮小版)。本番 (v6 印 → a95 → 静的サイト) は
このディレクトリを import しない。削除はしていない (移動のみ)。

| ファイル | 旧役割 | 実行 |
|---|---|---|
| `predict_weekly.py` | 旧 8 モデルアンサンブル + strategy_weights の買い目 (Streamlit 用)。読み込み部 `parse_csv` は root の `weekly_parse.py` へ切り出し済み | `python -m legacy.predict_weekly --csv data/weekly/YYYYMMDD.csv` (`weekly_nicegui.ps1 -WithPredict`) |
| `ev_filter.py` | 旧 EV フィルタ / 較正 (predict_weekly 専用) | — |
| `precompute_buylist.py` | 旧買い目の事前計算 (呼び出し元なし) | `python -m legacy.precompute_buylist` |
| `build_strategy_walkforward.py` / `build_strategy_stable.py` / `simulate_patterns.py` | `data/strategy_weights.json` (旧 rule-based 戦略、Streamlit のみ参照) の再構築 | `python -m legacy.<name>` |

root の `predict_weekly.py` は shim: 読み込み部の名前は `weekly_parse` から再 export し、それ以外の名前
(`ensemble_predict` 等) は参照されたときだけ `legacy.predict_weekly` を読む (PEP 562)。研究スクリプトの
`from predict_weekly import parse_csv` は無修正で動く。

root に残っている旧学習スクリプト (optuna_lgbm / optuna_catboost / train_lgbm / stacking / calibrate /
optuna_transformer* / train_transformer / optimize_weights / train_value_model / retrain_value_model) は
`backtest.py` などの root モジュールが import しているため今回は動かしていない (段階 3 の残)。
`retrain_value_model.py` の月次自動実行は `weekly_post.ps1 -WithValueModel` のときだけ (既定 OFF)。
Streamlit `app.py` は Streamlit Cloud が root を見るため root に残置。
