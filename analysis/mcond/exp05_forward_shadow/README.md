# EXP05-F 市場条件付き第2段モデルの前向き検証基盤

共通基盤は `analysis/mcond/README.md`。本番の印・買い目・資金配分には一切接続しない。
データ在庫監査は `DATA_SOURCE_AUDIT.md`、特徴パリティ監査は `FEATURE_PARITY.csv`、
凍結モデルの詳細は `MODEL_FREEZE.md`、定義同値性チェックは `SERVE_PARITY.md`、
必要サンプル数は `POWER_ANALYSIS.md`、実データ試験は `LIVE_PREFLIGHT.md`、
スケジューラ設計は `SCHEDULER_PLAN.md`、混在コミットの記録は `MIXED_COMMIT_NOTE.md`、
事前登録の仕様は `spec.json`。

## 証拠水準 (必ず守ること)
2023-2025年はEXP04で仮説発見・EXP05で再評価済みであり「未使用期間」ではない。EXP05の結果は
「頑健な開発結果」。**唯一の最終確認は、これから前向きに保存する未来データ。**
2026年の過去開催分は31-38分前の市場スナップショットが1件も存在しないため(`DATA_SOURCE_AUDIT.md`)、
「準確認的な時系列OOS」にすらできない。EXP05-Fは前向き観測ゼロから開始する
(2026-09-19の実データ試験で2件のみ生成、うち主評価に使えるのは1件。詳細は`LIVE_PREFLIGHT.md`)。

## 登録前Gate (spec §11、状況)
- [x] `FEATURE_PARITY.csv` 完成 (F-serve117列を全分類、除外0件)
- [x] 動的能力の馬名固定持ち越しを解消 (`live_history.py`、2025年末凍結→2026年確定分を逐次反映)
- [x] C2未算出特徴の扱い確定 (8/8列を時点安全に算出、旧v1は4/8列のみだった)
- [x] F-forward固定 (=F-serveの117列、`spec.json`)
- [x] serve-parity replay 実施 (`SERVE_PARITY.md`。**真のreplayは2023-2025週次CSVアーカイブが
  無く不可能**、代替として定義同値性チェックを実施し7/8列+dyn_skill_muで完全一致、
  1件のバグ(raw_jt_pair)を発見・修正)
- [x] 必要サンプル数の単位修正 (`POWER_ANALYSIS.md`。旧「6600レース≈5-6ヶ月」の根拠不明な
  換算を修正し「6600レース≈約99開催週≈約1.9年」と訂正)
- [x] 実JV-LinkでのT-35取得成功 (`LIVE_PREFLIGHT.md`。2026-09-19実開催で2回実行、
  ウィンドウ内外の判定が正しく機能することを確認)
- [x] append-only保存成功 (実データでrevision機構も確認)
- [x] 本番処理への非干渉確認 (`reports/live_odds`・`reports/site_odds`とも無汚染)
- [x] スケジューラ設定内容の文書化 (`SCHEDULER_PLAN.md`、**登録自体は未実施**)
- [x] 全テストPASS (`pytest tests/ -q` 172件、EXP05-F固有 5件、計180件)

**登録前Gateは形式上すべて満たしたが、タスクスケジューラへの自動登録(`PyCaLiAI_EXP05FS_T35`)は
このセッションでは行わない。** 無人で毎週JV-Linkへ継続アクセスする新しい常設ジョブになるため、
ユーザーに一言確認してから登録する(`SCHEDULER_PLAN.md`の登録コマンド参照)。

## 実行順
```
# 一度だけ (凍結、既に実行済み)
python -m analysis.mcond.exp05_forward_shadow.horse_identity      # 2025年末dyn_skill状態 (参考値、live_history.pyが実際には使う)
python -m analysis.mcond.exp05_forward_shadow.freeze_model        # M1/M3/M4凍結
python -m pytest analysis/mcond/exp05_forward_shadow/test_forward_shadow.py -q

# 監査 (再実行可能、結果は都度FEATURE_PARITY.csv等を上書き)
python -m analysis.mcond.exp05_forward_shadow.feature_parity_audit
python -m analysis.mcond.exp05_forward_shadow.serve_parity_check

# 週次 (土曜朝、bundle生成後)
python -m analysis.mcond.exp05_forward_shadow.feature_snapshot --date YYYYMMDD

# レース毎 (発走31-38分前を狙う、通常はタスクスケジューラ経由)
python -m analysis.mcond.exp05_forward_shadow.market_snapshot --once <rid16> --date YYYYMMDD

# 結果確定後 (日曜夜)
python -m analysis.mcond.exp05_forward_shadow.join_results --date YYYYMMDD
```

## タスクスケジューラへの登録 (未実施、ユーザー確認事項)
`t35_shadow.ps1` は `t20_site.ps1` と全く同じ「レース毎タスク」方式で実装済み・実データで動作確認済み。

```powershell
.\t35_shadow.ps1 -Schedule     # 今すぐ今日のレース毎タスクを登録
.\t35_shadow.ps1 -Once <rid16> -Dry   # 動作確認のみ(保存しない)
```

土日9:00に自動で`-Schedule`を起動する常設タスク("PyCaLiAI_EXP05FS_T35"、`PyCaLiAI_T20_Site`と
同型)を登録するかはユーザーに確認する。

## 非干渉の設計
- 生成物は全て専用ディレクトリ: `reports/exp05fs_odds/`, `data/_research/mcond/exp05fs_*/`
- `jvlink_odds.py`と`forward_prices.py`への変更は「新しいstage文字列を1つ追加」のみ
  (既存t10/t20/vote/close/manualの動作は不変、`tests/test_forward_prices.py`で確認済み。
  このコミットに他セッションの未コミット変更が混入した経緯は`MIXED_COMMIT_NOTE.md`参照)
- `weekly_nicegui.ps1` / `compute_bets.py` / `t10_runner.py` / `t20_site_bets.py` は未変更
- 例外は`logs/exp05fs_errors.log`に記録し、他の週次タスクをブロックしない

## 必要サンプル数
**6,600レース ≈ 約99開催週 ≈ 約1.9年**(`POWER_ANALYSIS.md`参照、race-level 1標本z検定、
両側α=1%・検出力80%)。途中経過でのROI判断は禁止 (spec §16)。

## 独立に見つかった重要な指摘事項 (EXP05-Fのスコープ外、本番コードは未変更)
週次CSVのカテゴリ表記(`芝・ダ`="ダート"等)が学習時の表記(`master_v2.csv`="ダ"等)と
食い違っており、本番`export_weekly_marks.py`の`apply_encoders()`がこれを正規化しないまま
`models/unified_rank_v6.pkl`のLabelEncoderへ渡している。ダート戦(大多数)で`芝・ダ`特徴が
毎週`__NaN__`(未知カテゴリ)扱いになっている可能性が高い。ユーザーへ別途報告済み、
このセッションでは本番コードを変更していない。EXP05-F自身は`frozen_encode.normalize_categorical`
でこの表記ゆれを吸収済み。

## 次の一手 (未着手)
- タスクスケジューラへの本登録可否をユーザーと決める
- 本番`export_weekly_marks.py`のカテゴリ表記不一致(上記)の修正可否をユーザーと決める
- 2026年デビュー馬の同姓同名衝突リスク低減 (MODEL_FREEZE.md #2、血統登録番号を取得できる
  代替データ源があれば解消可能)
