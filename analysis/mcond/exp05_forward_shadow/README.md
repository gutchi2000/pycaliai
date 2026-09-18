# EXP05-F 市場条件付き第2段モデルの前向き検証基盤

共通基盤は `analysis/mcond/README.md`。本番の印・買い目・資金配分には一切接続しない。
データ在庫監査は `DATA_SOURCE_AUDIT.md`、凍結モデルの詳細は `MODEL_FREEZE.md`、
事前登録の仕様は `spec.json`。

## 証拠水準 (必ず守ること)
2023-2025年はEXP04で仮説発見・EXP05で再評価済みであり「未使用期間」ではない。EXP05の結果は
「頑健な開発結果」。**唯一の最終確認は、これから前向きに保存する未来データ。**
2026年の過去開催分は31-38分前の市場スナップショットが1件も存在しないため(`DATA_SOURCE_AUDIT.md`)、
「準確認的な時系列OOS」にすらできない。EXP05-Fは前向き観測ゼロから開始する。

## 今回のセッションで完了したこと (spec §23)
- [x] 2026年データ在庫監査 (`DATA_SOURCE_AUDIT.md`)
- [x] EXP05モデル凍結 (`MODEL_FREEZE.md`, `out/frozen_model.joblib`)
- [x] 市場スナップショット取得コード (`market_snapshot.py`, `jvlink_odds.py`に`exp05fs_t35` stage追加)
- [x] serve-safe 117特徴の週次生成 (`feature_snapshot.py`, 2026-09-19実データで109/117列確認)
- [x] append-only予測保存 + 重複防止 (`predict_and_store.py`, `test_forward_shadow.py`)
- [x] 仮想買い目(R1/R2)を結果確定前に保存
- [x] 結果結合を別処理で実装 (`join_results.py`)
- [x] テスト用レースでend-to-end確認 (合成市場データ、2026091906040501)
- [x] 本番非干渉であることを確認 (専用ディレクトリ・専用stage・既存テスト全通過)
- [ ] **タスクスケジューラへの本登録は未実施** (下記参照、ユーザー確認してから)
- [ ] 実データ(本物のJV-Linkオッズ)でのT-35取得は次回開催で要確認 (JV-Link稼働時間内での動作確認が必要)

この時点では「前向き検証成功」を報告しない (spec §23)。

## 実行順 (前提: exp05_market_residual_dev の build_features.py が実行済み)
```
# 一度だけ (凍結)
python -m analysis.mcond.exp05_forward_shadow.horse_identity      # 2025年末dyn_skill状態
python -m analysis.mcond.exp05_forward_shadow.freeze_model        # M1/M3/M4凍結
python -m pytest analysis/mcond/exp05_forward_shadow/test_forward_shadow.py -q

# 週次 (土曜朝、bundle生成後)
python -m analysis.mcond.exp05_forward_shadow.feature_snapshot --date YYYYMMDD

# レース毎 (発走31-38分前を狙う、通常はタスクスケジューラ経由)
python -m analysis.mcond.exp05_forward_shadow.market_snapshot --once <rid16> --date YYYYMMDD

# 結果確定後 (日曜夜)
python -m analysis.mcond.exp05_forward_shadow.join_results --date YYYYMMDD
```

## タスクスケジューラへの登録 (未実施、ユーザー確認事項)
`t35_shadow.ps1` は `t20_site.ps1` と全く同じ「レース毎タスク」方式で実装済み。

```powershell
.\t35_shadow.ps1 -Schedule     # 今すぐ今日のレース毎タスクを登録
.\t35_shadow.ps1 -Once <rid16> -Dry   # 動作確認のみ(保存しない)
```

土日9:00に自動で`-Schedule`を起動する常設タスク("PyCaLiAI_EXP05FS_T35"、`PyCaLiAI_T20_Site`と
同型)を登録するかはユーザーに確認する。理由: 無人で毎週JV-Linkに継続アクセスする新しい
常設ジョブを追加することになるため、既存のT-10/T-20タスクとは別に一言確認したい。

## 非干渉の設計
- 生成物は全て専用ディレクトリ: `reports/exp05fs_odds/`, `data/_research/mcond/exp05fs_*/`
- `jvlink_odds.py`と`forward_prices.py`への変更は「新しいstage文字列を1つ追加」のみ
  (既存t10/t20/vote/close/manualの動作は不変、`tests/test_forward_prices.py`で確認済み)
- `weekly_nicegui.ps1` / `compute_bets.py` / `t10_runner.py` / `t20_site_bets.py` は未変更
- 例外は`logs/exp05fs_errors.log`に記録し、他の週次タスクをブロックしない

## 必要サンプル数
6,600レース (`spec.json`の`required_sample_size`参照、EXP05実測の効果量とばらつきから
race-level 1標本z検定で算出、両側α=1%・検出力80%)。1開催週約250-300レースなので概ね
22-26開催週(5-6ヶ月)。途中経過でのROI判断は禁止 (spec §16)。

## 次の一手 (未着手)
- 本物のJV-Linkオッズで`market_snapshot.py --once`が31-38分ウィンドウ内に収まるかを実開催で確認
- dyn_skill_mu等の2026年内更新の仕組み (MODEL_FREEZE.md #2の限界を解消)
- タスクスケジューラへの本登録可否をユーザーと決める
