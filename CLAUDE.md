# PyCaLiAI — Claude Code 引き継ぎメモ

> **このファイルの位置づけ**: 次に触る人（含む将来の自分）が 5 分で全体像を掴めるようにする。
> 詳細仕様は `docs/` 配下、運用フローは `WORKFLOW.md` を参照。
>
> 最終更新: 2026-11-04（**A〜S スプリント**: serve 特徴復元 `serve_feature_recovery.py`（間隔の列対応バグ発見、CB_SERVE_RECOVER で有効化、A/B は下記）／週次フローの WARNING を Discord 転送／収集待ち台帳の Gate 自動判定／T-10・T-20 起動監視／較正ドリフト監視／旧 NiceGUI 同期と hosei 生成を既定 OFF／サイト: 騎手・調教師ページ、通信簿の週次推移と見送り成績、馬ページチャート、開催日カード、sitemap/JSON-LD、馬ページ OG（Worker）、相談AI の回顧参照、Discord ナビ枠）
> 旧: 2026-07-01（**ルート整理**: 実験・研究スクリプト 97 本を `lab/<theme>/` へ集約し root .py を 237→139 に。巨大生成物/スクラッチを `.gitignore`。詳細は下記「✅ ルート整理済み」節と `lab/README.md`）
> 旧: 2026-06-24（**現状整理**: cowork_results は修復済み＝壊れていない／Cowork 78% 表は v5 期の歴史値で現在値ではない／「これ以上強くする」の本筋は予測 v7 や EV 閾値ではなく **検証済み未配線 ROI の回収**＝下記「強化の本筋」参照）
> 旧: 2026-06-16（**本番表示を静的サイト pycaliai-umami に移行**、週次フローに sync-hf-umami 組込み、predict デフォルト skip / 見送りガードを反映）
> 旧: 2026-05-20（NiceGUI + Cowork パラダイムに全面改訂、**v6 本番投入**、cowork_results 集計修復）

---

## 📘 完全仕様書（2026-08-23 新設）

外部 AI へのレビュー依頼・新規参画者の把握には **`docs/SPEC/`** を使う（本ファイルは要約）。

| 巻 | 内容 |
|---|---|
| [Vol. I](docs/SPEC/VOL1_SYSTEM.md) | データ / 特徴量（120 特徴の gain×serve 三元表）/ モデル / 確率 / 印 / serve 層 |
| [Vol. II](docs/SPEC/VOL2_BETTING_OPS.md) | compute_bets 完全仕様 / ガード / T-10 / 週次フロー / 決済集計 / 公開層 |
| [Vol. III](docs/SPEC/VOL3_VALIDATION_AND_OPEN_PROBLEMS.md) | 死亡ルート台帳 / **欠陥台帳 P0-P3** / ガバナンス / 研究アジェンダ |
| [Vol. IV](docs/SPEC/VOL4_CODE_REFERENCE.md) | 関数レベルのコードリファレンス |
| [結合版](docs/SPEC/ALL_IN_ONE.md) | 外部 AI への一括投入用 |

**新発見（要対応、2026-09-10更新）**: 本番 v6 は学習 gain の **28.15%** を serve で失っていた（2026-06時点）。
P0-1（騎手/調教師stats）・P0-2（着度数CSV列数）・P0-4（Ｒ全角）は **2026-09-10 実データ再検証で解消確認済み**
（Vol. III §5.1a、2026-07-29コミット済みの修正が本番経路に既に乗っていた）。旧見積りの「約20.8%回収可能」は
その前の baseline を基準にした値のため要再計算。**2026-11-04**: baseline を直近4開催日で再生成（serve 死 gain **7.42%**/22特徴、prev_hosei は推定器で 81%）し P0-3 の較正器マスクが同期していることを確認（Vol. III §5.1y、再 fit 不要）。同時に 9/27 の研究コミットで baseline が 7 月週へ巻き戻り学習基準（train_cov）が消えて絶対 canary が沈黙していた事故を修復（`measure_serve_coverage.py --weeks/--out`、train_cov 引き継ぎ、古い baseline / train_cov 欠落は WARN）。

---

## Claude への作業指示

### コンテキスト上限（重要）
コンテキスト使用率が 95% に近づいたら作業を止めて以下を bullet でまとめてからコンパクションすること：
- やったこと
- 未完了のタスク
- 次の一手

### 自律性（重要）
ファイル編集・読み込み・モデル学習・バックテスト・スクリプト実行・コミットなどの通常作業では、いちいち許可を求めない。情報をもらったら自分で判断して自律的に進める。

確認が必要なのは以下のみ：
- `git push` のうち **HuggingFace Spaces への反映を伴うもの**（`sync-hf.ps1` / `sync-hf-umami.ps1` の単独実行、および `weekly_nicegui.ps1` の HF 同期フェーズ）
- データ・モデルの**削除**（不可逆操作）
- `data/master*.csv` の再生成（数十分〜時間規模）
- スコープ外の大規模リファクタ

「やりますか？」「進めてもいいですか？」は言わない。やる。

### 収集待ち台帳（継続更新）

研究・検証でデータ蓄積待ちが発生したら、`docs/research/COLLECTION_WAIT_REGISTRY.md`へ必ず追加・更新する。完了・閉鎖後も行を削除せず履歴を残す。通常のデータ増加だけで自動研究や定期報告を開始せず、Gate到達・失敗・異常・ユーザー明示依頼のときだけ通知する。

### 1ファイル書き直し系タスクで TaskCreate は不要
ドキュメント書き直し・1関数修正のような単発作業に TaskCreate/TaskUpdate を使う必要はない。
複数日にまたがるモデル再学習や Sprint 単位の作業のときだけ使う。

---

## プロジェクトの現在地

**JRA 中央競馬の AI 予想システム。役割分担を明確に分けている：**

| レイヤー | 担当 | 出力 |
|---|---|---|
| **PyCaLiAI**（このリポジトリのモデル部分） | 印付け（◎〇▲△△）と Plackett-Luce 確率の算出 | `reports/cowork_input/{date}_bundle.json` |
| **Cowork**（Anthropic Claude Desktop App） | 馬券種・点数・予算配分・見送り判断 | `reports/cowork_output/{date}_bets.json` |
| **静的サイト**（HF Docker Space `pycaliai-umami`、本番）／NiceGUI（旧）／Streamlit（Cloud） | 表示専用（推論済みデータを可視化） | 画面 |

**重要**：かつての rule-based betting（`strategy_weights.json`）から **Cowork-driven betting** に主軸が移っている。
`strategy_weights.json` は Streamlit 版が今も読むが、運用上の意思決定は Cowork 側で行われる。

- **ターゲット変数**: `fukusho_flag`（3着以内 = 1、正例率 ≈ 21.9%）
- **予算**: 1R = 1万円目安。実額は Cowork が race ごとに配分
- **除外レース**: 障害・新馬中心（旧 `EXCLUDE_PLACES` の東京/小倉除外は v5 移行時に解除済み）
- **三連単**: 廃止済み

---

## 環境

- Python 3.11（`venv311\`）
- Windows 11 / `E:\PyCaLiAI`
- 仮想環境: `venv311\Scripts\activate`
- **静的サイト（本番表示, 2026-06-14〜）**: https://gutchi15300-pycaliai-umami.hf.space ← クリーンURL。`site/` を Docker Space `gutchi15300/pycaliai-umami` で配信。ソースは `site/`、データ生成は `build_site.py`、デプロイは `sync-hf-umami.ps1`
- **pycaliai.com（Cloudflare Workers）**: `wrangler.jsonc` で `site/` を ASSETS 配信 + `worker/index.js` が `/api/soudan`（相談AI）。デプロイは `sync-hf-umami.ps1` 内の `npx wrangler deploy`（単独でも可）。`package.json`（`@anthropic-ai/sdk`）は `npm install` が必要、`node_modules/` は .gitignore
- **NiceGUI（旧本番）**: `python nicegui_app.py` → `http://localhost:8080` / HF: https://gutchi15300-pycaliai.hf.space （`sync-hf.ps1`、まだ併行更新中）
- **Streamlit（副）**: `streamlit run app.py`
- **GPU / torch**: CUDA 12.8 動作中（unified_rank_v5 自体は LightGBM のみで torch 不要）

---

## 時系列分割（全モデル共通）

| セット | 期間 |
|---|---|
| Train | 〜 2022-12-31 |
| Valid | 2023-01-01 〜 2023-12-31 |
| Test  | 2024-01-01 〜 |

---

## 現役モデル（**v6 marks stack が本番**）

### NiceGUI / Cowork 主導の本番ライン（unified_rank_v6）

| ファイル | 役割 |
|---|---|
| `models/unified_rank_v6.pkl` | ✅ **本番**: LGBM LambdaRank（Optuna 目的関数 = mark composite - 0.5 × ECE_high_p）。v5 比で複勝/馬連 ECE -32〜34% |
| `models/pl_calibrators_v6.pkl` | ✅ **本番**: PL スコア → 実勝率 の Isotonic（valid=2023 で fit） |
| `data/pl_payout_curve_v6.pkl` | ✅ **本番**: 期待払戻カーブ |
| `models/unified_rank_v5.pkl` | 🗄️ **退役**: 旧本番。比較・rollback 用に保持 |
| `models/pl_calibrators_v5.pkl` | 🗄️ **退役**: 旧本番 calibrator |

**v6 本番化の根拠（2026-05-20）**:
- 印精度: ◎ 連対率 +0.48pt、▲ 複勝圏率 +1.22pt（微改善）
- Calibration: ECE 複勝(◎) **-32%**、ECE 馬連(◎-〇) **-34%**（大幅改善）
- 機械買い ROI は v5 と同等（±0.01）だが、Cowork (LLM) の判断材料として calibration の質が効くと判断
- Cowork 実績（4/18-5/17、292 bets、ROI 78.0%）の弱点である複勝/馬連の判断改善を期待

### Streamlit 版 predict_weekly.py が使う旧アンサンブル

| ファイル | 役割 |
|---|---|
| `lgbm_optuna_v1.pkl` / `catboost_optuna_v1.pkl` / `catboost_rank_v1.pkl` | 基本 3 モデル |
| `lgbm_fukusho_v1.pkl` / `catboost_fukusho_v1.pkl` | 複勝特化 |
| `lgbm_rank_v1.pkl` / `lgbm_regression_v1.pkl` | Phase 5 で追加 |
| `transformer_pl_v2.pkl` | Transformer + Plackett-Luce |
| `stacking_meta_v1.pkl` / `stacking_calibrator_v1.pkl` | スタッキング（WARNING 出るがエンサンブルへフォールバック中） |
| `ensemble_calibrator_v4.pkl` | Test 2024-fit Isotonic（v3/v2/v1 はフォールバック） |
| `ensemble_weights.json` | Nelder-Mead で求めた最適化重み |
| `value_model_v2.pkl` / `order_model_v1.pkl` | EV 補正・着順 3 クラス（HALO formation 用） |

⚠️ **モデル 2 系統が並走している状態**。v5 統一が今後の整理方針。

---

## 週次運用フロー（NiceGUI 中心の 3 フェーズ）

すべて `weekly_nicegui.ps1` 1 本で完結。`weekly_pre.ps1` / `weekly_post.ps1` は内部で呼ばれる旧版。

### Phase A — 土曜朝（TARGET 出走表エクスポート後）
```powershell
.\weekly_nicegui.ps1                  # 最新 data/weekly/*.csv を自動検出
# または .\weekly_nicegui.ps1 20260516
```
自動で（デフォルト model=**v6**）：
1. `make_weekly_hosei.py` → **デフォルト SKIP**（2026-11-04。TARGET の補正タイム供給は 2026-09-19 で恒久停止しヘッダだけを量産していた。prev_hosei は推定器 `CB_HOSEI_PROXY=1`。再開は `-WithHosei`）
2. `predict_weekly.py` → **デフォルト SKIP**（2026-06-11 audit P1-1。旧8モデルは重く NiceGUI/Cowork に不要）。Streamlit 用に欲しい週だけ `-WithPredict`
3. `export_weekly_marks.py --model v6` → `reports/cowork_input/{date}_bundle.json` **★これが Cowork のキー入力**（2026-11-04〜 `scripts/run_notify.py` 経由: 出力を `logs/export_{date}.log` に残し、WARNING 以上・serve canary 行だけ Discord へ。exit code はそのまま）
4. `build_course_stats.py` → `data/course_stats.json`（コース分析タブ用）
5. `git push origin master` → `sync-hf-umami.ps1`（**本番 静的サイト**、`build_site.py` で再生成して push。旧 NiceGUI Space の `sync-hf.ps1` は 2026-11-04 退役＝`-WithNiceGUI` のときだけ）
6. `[4/3]` 今日の見どころ 生成（`scripts/kaiko_warm.py --discord`、best-effort。Worker が発走前記事を書き Discord へ投稿。`--x` を足すと X 下書きを X へ自動投稿できる（2026-11-04 実装、鍵は `notify_config.json` の x_api_key 等）が、**週次スクリプトには付けていない**＝X 投稿はユーザーが明示的に有効化する）

### Phase B — 土曜昼（Cowork が買い目を返してきたら）
```
1. Claude Desktop に reports/cowork_input/{date}_bundle.json を投入
   （プロンプトは docs/cowork_prompt.md を貼る）
2. Cowork のレスポンスを reports/cowork_output/{date}_bets.json として保存
3. .\weekly_nicegui.ps1 -BetsOnly
```
内部で：
1. `validate_cowork_bets.py --apply` で**見送りガード**を強制（絶対禁則の4条件。Cowork が見送るべき race に買い目を付けても bets:[] に矯正）。ガード自体が実行不能なら **fail-closed で停止**（未検証 bets を本番に出さない）。どうしても push したい時のみ `-Force`
2. `git push origin master` → `sync-hf-umami.ps1`（旧 NiceGUI は `-WithNiceGUI` のみ）

### Phase C — 日曜夜（TARGET 結果エクスポート後）
```powershell
# data/kekka/{date}.csv を配置
.\weekly_nicegui.ps1 -Post
```
内部で：
1. `weekly_post.ps1`（失敗したら **fail-hard** して HF 同期を中止。2026-11-04〜 出力は `logs/weekly_post_{date}.log` に残り、終了後に `scripts/run_notify.py --scan` が WARNING/Error 行を Discord へ）
   - `generate_results.py` → `data/results.json` / `data/cowork_results.json`（**集計凍結対策で必ず commit**）
   - `update_live_results.py` → `data/live_results_2026.csv`
   - `git push` + `reports/cowork_bets/{date}/`
   - `cowork_results.json` の `generated_at` が当日かチェック（凍結検知）
   - Step 2.8 `wide_kekka.csv` 鮮度 / Step 2.9 `analysis/collection_wait_gate.py --notify`（収集待ち台帳 CW-01/02/03/09/10/11 を正本から数え直し、Gate 到達・減少・消失・停滞のときだけ Discord。現況 `docs/research/COLLECTION_WAIT_STATUS.md` 自動生成）
   - settle 直後に `analysis/calibration_drift_check.py --notify`（現行較正器の直近 8 週 ECE を前全週の参照と比較、超えたときだけ Discord。再 fit は手動）
2. `sync-hf-umami.ps1`（**本番 静的サイト**。旧 NiceGUI `sync-hf.ps1` は `-WithNiceGUI` のみ）
3. 日曜 + 月初 1〜7 日なら `retrain_value_model.py` 自動実行
4. 日曜なら `run_audit.ps1` 自動実行（週次監査）
5. `[1e/3]` 出世予測 AI 更新（`analysis.shusse.extend_2026` → `predict_live`、best-effort。失敗は Warn のみで本番に影響なし。`build_site.py` が `site/data/shusse/` へ反映）
6. `[1f/3]` 回顧記事 生成（HF 同期の後に `scripts/kaiko_warm.py` が pycaliai.com の Worker に当週分の生成を依頼。best-effort、鍵はローカルに無い。失敗しても月曜 03:00 の cron とページ閲覧時に埋まる）

---

## モデル再学習・キャリブレータ再構築（四半期〜半期）

### v5 系の再構築（calibrator/curve のみ）
```bash
python run_v5_pipeline.py
# → pl_calibrators_v5.pkl / pl_payout_curve_v5.pkl 更新
# → audit_marks v5 ログ + backtest_fixed v5 ROI
```

### v6 系の再構築（calibrator/curve + class_prior）
```bash
python run_v6_pipeline.py
# → pl_calibrators_v6.pkl / pl_payout_curve_v6.pkl 更新
# → audit_marks v6 / backtest_fixed v6 / audit_v6_vs_v5 ログ
# → data/class_prior_v6.json (bundle.json 埋込用、クラス×印 経験率)
```

### class_prior 単体再生成
```bash
python scripts/audit_marks_by_class.py --model v6
# → reports/audit_marks_by_class_v6.{json,log}
# → data/class_prior_v6.json (export_weekly_marks.py が読む)
```

### v7 等の新版を試したい場合
```bash
python optuna_v6_marks.py            # 1. v6 と同等の Optuna スクリプトを派生作成
python run_v6_pipeline.py            # 2. calibrator/curve 生成 + audit
# 良ければ weekly_nicegui.ps1 のデフォルト Model を更新
```

### 旧アンサンブル（Streamlit 用）の再学習
```bash
python build_dataset.py        # master CSV
python optuna_lgbm.py
python optuna_catboost.py
python optuna_transformer.py   # torch 必要
python stacking.py
python calibrate.py            # ensemble_calibrator_v4 / stacking_calibrator_v1
python backtest.py --no_strategy --period valid --output_suffix _valid
```

### rule-based 戦略を再構築したい場合（Streamlit 表示用）
```bash
python build_strategy_walkforward.py  # walk-forward × OOS 二段階フィルタ
python build_strategy_stable.py       # strict: valid+test 両方黒字のみ
```

---

## アーキテクチャ

### v6 marks stack（本番、NiceGUI/Cowork 経路）
```
出走表 CSV
    ↓
export_weekly_marks.py --model v6
    ↓ unified_rank_v6.pkl → 生スコア
    ↓ pl_calibrators_v6.pkl → 実勝率（calibration 改善版）
    ↓ Plackett-Luce → 全馬の P(着順)
    ↓ 印付け（◎〇▲△△）+ race_confidence
    ↓ data/class_prior_v6.json → race_meta.class_prior 埋込
        (◎〇▲△△ のクラス別経験的中率を Cowork に与える)
reports/cowork_input/{date}_bundle.json   ← Cowork 投入
    ↓ （Cowork が race ごとに馬券を組む、class_prior を判断材料に）
reports/cowork_output/{date}_bets.json    ← NiceGUI が読む
```

### 旧アンサンブル（Streamlit 用、predict_weekly.py）
```
出走表 CSV
    ↓
ensemble_predict()  ←── lgbm/cat/rank_cat/rank_lgbm/regression/transformer/fuku_lgbm/fuku_cat
                    ←── ensemble_weights.json で加重
                    ↓ （stacking 試行 → 失敗時フォールバック）
stacking_meta_v1.pkl → ensemble_calibrator_v4.pkl
                    ↓
strategy_weights.json で会場×クラス×馬券種フィルタ
                    ↓
Kelly 金額計算 → 買い目 CSV
```

---

## データファイル

```
data/
  master_v2_20130105-20251228.csv     ★本番マスター（515MB、約62万行）
  master_20130105-20251228.csv         旧マスター（412MB、互換性のため残置）
  master_kako5.csv                     過去5走特徴量入り版
  kekka_20130105-20251228.csv          払戻マスター
  pl_payout_curve_v5.pkl               ★v5 期待払戻カーブ
  payout_table.parquet                 wide / 三連複 / 三連単 payout
  wide_payouts_2016-2025.parquet       ワイド払戻

  weekly/{YYYYMMDD}.csv                週次入力（TARGET 出走表）
  kekka/{YYYYMMDD}.csv                 週次払戻（TARGET 結果）
  hosei/H_{YYYYMMDD}.csv               補正タイム週次（make_weekly_hosei.py 生成）
  kako5/{YYYYMMDD}.csv                 過去5走詳細
  training/{H|W}-*.csv                 坂路 / WC 調教 週次

  strategy_weights.json                ⚠️ 旧 rule-based 戦略（Streamlit のみ参照、現状 13条件 5会場）
  course_trend.json                    コース傾向
  course_stats.json                    NiceGUI コース分析タブ用
  jockey_stats.csv / trainer_stats.csv 騎手・厩舎統計
  results.json                         結果集計（Streamlit 表示用）
  live_results_2026.csv                ★2026 シーズン実績（5,000+ 行）
  cowork_results.json                  ✅ 修復済み（2026-06-22: 620R/ROI 69.5%、by_type に CI95 と roi_verdict 完備）。generate_results.py が毎回 commit で凍結対策

E:\競馬過去走データ\                  プロジェクト外（TARGET フロンティア出力）
  H-20150401-20260313.csv             坂路調教マスター（520万行、cp932）
  W-20150401-20260313.csv             WC 調教マスター（70万行、cp932）
```

### 調教データ仕様
- **坂路（H）列**: 場所, 年月日, 馬名, Time1(4F合計), Lap4, Lap3, Lap2, Lap1(最終200m)
- **WC（W）列**: 場所, コース, 回り, 年月日, 馬名, 5F, 4F, 3F, Lap3, Lap2, Lap1
- **JOIN**: 馬名 + 年月日（レース 14日前以内の最終追い切り）で `merge_chukyo()`
- **特徴量**: `trn_hanro_4f`, `trn_hanro_lap1`, `trn_hanro_days`, `trn_wc_3f`, `trn_wc_lap1`, `trn_wc_days`
- **カバレッジ**: 坂路 80%（2015〜）、WC 25%（2021〜、2022 以降 67%）

---

## ファイル役割マップ

### ★ NiceGUI / Cowork ライン（本番、最重要）
| ファイル | 役割 |
|---|---|
| `build_site.py` | ★**本番 静的サイト**のデータ生成（bundle/cowork_output/kekka 等 → `site/data/*.json`） |
| `site/` | ★**本番 静的サイト**ソース（index.html / css / js、vanilla JS + ECharts CDN） |
| `sync-hf-umami.ps1` | ★**本番**デプロイ: `build_site.py` → Docker Space `pycaliai-umami` へ push |
| `site/shusse.html` + `analysis/shusse/site_export.py` | 出世予測 AI（ファン向け読み物、馬券非転用、2026-10-31 公開）。データ `site/data/shusse/{index,YYYY}.json`。更新は Phase C の [1e/3]。詳細 `analysis/shusse/README.md` |
| `worker/index.js` + `worker/tools.js` + `worker/prompt.js` | **相談AI（2026-11-03 公開）**: pycaliai.com の Cloudflare Worker。`POST /api/soudan` が Claude（Sonnet 5.5、全体 1 日 150 問超で Haiku 5.5、IP 別 30 問/日、月 $90 で停止、KV カウンタ）を中継。数字は公開済み `site/data/{date}.json` だけ（オッズ・EV 不在）。UI は `app.js renderSoudan()` の「💬 相談AI」タブ。鍵は `npx wrangler secret put ANTHROPIC_API_KEY`（Max の月次 API クレジット）。確認 `GET /api/soudan/health`。テスト `node --test worker/tools.test.mjs`。`assistant/chat.py`（Ollama 版）は未改修 |
| `worker/kaiko.js` + `site/kaiko.html` + `scripts/kaiko_warm.py` | **回顧記事「負けた理由まで書く AI」（2026-11-04 公開）**: 結果のある開催日の全レースを「予想→結果→ズレた理由→教訓」で Claude が文章化、日の総括と X 下書き付き。**今日の見どころ**（`race_id="preview"`、発走前: 堅い 3 鞍・AI と人気が割れる鞍・見送り寄り・重賞・注目 1 鞍）も同じ経路。facts は公開 JSON だけ、記事は KV に永続（1 件 1 回）。生成経路: Phase A `[4/3]`（見どころ）／Phase C `[1f/3]`（回顧）＝`kaiko_warm.py --discord` が Worker に依頼し新規分を Discord へ（**2026-11-05〜 公開サーバー**: 見どころ→`discord_midokoro_webhook`、回顧→`discord_kaiko_webhook`、私用 `discord_webhook` には要約 1 行。鍵は `notify_config.json`。接続テスト `kaiko_warm.py --test-discord`）／cron 土日 05:00・月曜 03:00 JST／ページ閲覧時の lazy 生成。一覧 `GET /api/kaiko?date=`、生成 `POST /api/kaiko/gen`、RSS `GET /kaiko.xml` |
| `build_report_card.py` + `site/tsushinbo.html` | **AI の通信簿（2026-11-04 公開）**: 公開済み `site/data/{date}.json` から条件別（クラス・芝ダ距離・混戦度百分位・◎の朝人気・◎勝率帯・頭数・競馬場・月）の ◎勝率/複勝圏率（Wilson 95% CI）と表示買い目の回収率を `site/data/report_card.json` に集計（`build_site.py` 末尾で毎ビルド）。LLM 不使用。相談AI のツール `ai_track_record` が同じ JSON をレース条件で引く |
| `build_share_cards.py` + `site/r/` + `site/og/` | **レース共有カード（2026-11-04）**: 公開 JSON から各レースの固定 URL `https://pycaliai.com/r/{race_id16}`（OG/Twitter カード meta → `/#race={rid}` へリダイレクト）と OG 画像 `site/og/{rid}.png`（結果待ち + 結果ありの直近 2 開催日だけ生成、他は削除）。`app.js` は `#race=` で表紙を飛ばして直接開き、レースヘッダに「𝕏 で共有」「URL をコピー」。`build_site.py` 末尾で毎ビルド |
| `build_jt_pages.py` + `site/jt.html` + `site/js/jt.js` | **騎手・調教師ページ（2026-11-04）**: `jt.html?j=騎手名` / `?t=調教師名`。2026 年の出走・KPI（勝率/3着内率/AI◎時/AI1位時）・月別・クラス別・検索。データ `site/data/jt/{jockeys,trainers,_index}.json`（コンパクト配列）。全ページのナビ「騎手」 |
| `build_sitemap.py` | **sitemap.xml 生成（2026-11-04）**: 静的ページ + `/r/{rid}` + `/r/d/{date}` + 馬ページ（`horse?name=`）+ 騎手/調教師。約 11,600 URL。`build_site.py` 末尾 |
| `worker/horse.js` | **馬ページ OG（2026-11-04）**: `/horse?name=` の応答に HTMLRewriter でその馬の title/description/og/canonical を差し込む（静的 9,339 ファイルは Workers assets 上限）。`/h/{id16}` → 302。相談AI ツール `horse_kaiko`（直近 3 走の回顧記事からその馬の段落）。`wrangler.jsonc` run_worker_first に `/horse`, `/horse.html`, `/h/*`。テスト `node --test worker/horse.test.mjs` |
| `worker_mail/` | **問い合わせメール通知（2026-11-05）**: `contact@pycaliai.com` 宛てを Email Routing → 別 Worker `pycaliai-mail` が受け、先に転送先 Gmail へ forward してから Discord の**私用サーバー**（`notify_config.json` の `discord_webhook`）に差出人・件名を投稿。サイト用 Worker とは別なので、デプロイ（`npx wrangler deploy -c worker_mail/wrangler.jsonc`）は site/ に触れない。秘密 `FORWARD_TO` / `DISCORD_WEBHOOK`。テスト `node --test worker_mail/index.test.mjs` |
| `site/js/site_config.js` | **Discord ナビ枠（2026-11-04）**: `discord_invite` に公開サーバーの招待 URL を入れると全ページのナビに「Discord」が出る（空なら非表示）。X の代わりの無料配信チャネル |
| `build_report_card.py` + `report_card_miokuri.py` + `site/tsushinbo.html` | **通信簿 拡張（2026-11-04）**: 週ごとの ◎勝率/複勝圏率（Wilson CI）と回収率の推移チャート、見送りレースを「買っていたら」の比較（参戦 vs 見送り、race-cluster bootstrap CI、`a95_all` 由来、集計値のみ） |
| `build_horse_pages.py` + `site/horse.html` + `site/js/horse.js` | **馬ページ（2026-11-04）**: `horse.html?name=馬名`。2026 年の出走ごとの印・AI順・勝率・3着内率・人気・着順、近走、血統、馬レベル、出世予測リンク。データは `site/data/h/{id[:2]}.json`（id = 馬名の FNV-1a×2、256 シャード）+ 検索用 `_index.json`。出走表の詳細パネルと全ページのナビ「馬」から |
| `export_weekly_marks.py` | unified_rank_v6 → bundle.json（**2026-11-04〜 予測経路 `sp-20261004-1`**: serve-repro-sp1 反映。順序キー全順序の騎手/調教師集計、day-start 馬同定、`serve_path` と `input_quality`（馬同定が判別不能のレースは下流が見送り、2% 超の日は不合格）を bundle に刻む。`data/production_policy.json` の `serve_path_version` と版が違う bundle は compute_bets / t20 / shadow_a95 が見送る＝bundle は必ず現行コードで作り直す） |
| `weekly_parse.py` + `training_join.py` | 週次 CSV の読み込み部 `parse_csv` と調教 JOIN（2026-11-04 に predict_weekly / optuna_lgbm から無改変で切り出し。本番はこちらを import） |
| `serve_feature_recovery.py` | **serve 特徴復元（2026-11-04）**: `CB_SERVE_RECOVER=1/2` のときだけ有効（既定 OFF）。間隔（週次 CSV の列対応が「1走前の前走間隔」で誤っていた→kako5 の現走間隔で置換）／前走日付／休み明け～戦目／血統タイプ（`data/pedigree_type_map.json`）／L2 で 斤量体重比 代理。`export_weekly_marks.py` と `analysis/measure_serve_coverage.py` が同じ関数を呼ぶ。**2026-11-04 A/B（2026 全 59 開催日 1,809R、OFF vs ON、同一レース）**: AI1位 3着内 +0.17pt [-1.23,+1.56]、◎3着内 +0.44pt [-0.75,+1.74] = 害なし。定義の正しさで **L1 を 2026-11-07 から本番有効**（`weekly_nicegui.ps1` が `CB_SERVE_RECOVER=1`）。baseline も同条件で再生成（dead gain 7.42%→下記）。L2 は未採用。較正器は再 fit せず `calibration_drift_check` で監視 |
| `scripts/run_notify.py` | 週次フローの出力保存 + WARNING 以上/serve canary 行の Discord 転送（exec モード / `--scan` モード） |
| `scripts/task_watchdog.py` + `watchdog_register.ps1` | 土日 09:10 に `PyCaLiAI_T10` / `PyCaLiAI_T20_Site` の起動・結果・レース毎タスク・bundle を確認し失敗時だけ Discord。タスク登録は `watchdog_register.ps1`（**未登録、ユーザー実行**） |
| `analysis/collection_wait_gate.py` | 収集待ち台帳の Gate 自動判定（Phase C Step 2.9） |
| `analysis/calibration_drift_check.py` | 較正ドリフト監視（Phase C settle 直後） |
| `analysis/replay_ab_compare.py` | replay 作業ディレクトリ 2 つを同一レース集合で比較（AI1位 1着/3着内、◎3着内、開催日 bootstrap CI） |
| `build_course_stats.py` | コース分析タブ用統計 |
| `power_rating.py` | **強さスコア（2026-11-12、非公開・ローカル専用）**: 全レースの着順だけで多頭数 Elo を逐次更新し、現役最上位=100 の線形スケールにしたレース横断の能力値（「この馬とこの馬どっちが強い?」用）。**芝とダートは別レーティング（2026-11-12、芝の現役最上位=100／ダートの現役最上位=100。直接対戦がほぼ無く 1 本に載せるとダート OP 馬が芝 G1 馬と並んだため。芝スコアとダートスコアの数字同士は比べられない）**。`build` / `compare 馬A 馬B` / `top` / `card YYYYMMDD` / `tune` / `selftest`。入力は master（あれば 2013〜）+ `data/kekka` + `data/weekly`。出力 `reports/power_rating/`（.gitignore）。**サイト・bundle・買い目には未配線**、馬券用ではない。`card` は 3着内率（Harville）と **好走気配/凡走注意**（各走の上振れ S−E を、間隔・昇降級の段数・芝ダ替わり・距離変更・近走上振れ・キャリア・乗り替わり・斤量増減・その馬の芝ダ実績の区分で説明する ridge。予測上振れの上位/下位 15%、理由付き）も出す。build のたびにラベルの前向き検証（レース単位 CI）を表示。事前評価は `build --until YYYYMMDD`、条件の取捨は `ablate`（既定は全 12 条件。選び直しは `ablate --eval-from 2023-01-01 --until 20240101`）、推移図は `chart 馬名…`（要 matplotlib）。**ローカル閲覧用ページ**: `power_rating.py site`（本体 `power_rating_site.py`）が `reports/power_rating/site/index.html` + `data.js` を生成（出走表・比較・ランキング・検証の 4 タブ、公開サイトと同じ配色、サーバー不要でダブルクリックで開く）。**`site/` の外に出すので sync-hf-umami / wrangler では配られない**（出力先に `site/` 配下を指定するとエラーで止まる）。**週次に組込み済み（2026-11-12）**: `weekly_nicegui.ps1` の `Invoke-PowerRating` が Phase A で `build --until {date}` → `reports/power_rating/card_{date}.txt`、Phase C で `build`、どちらも最後に `site` でローカルページを更新（best-effort、失敗は Warn のみ）。**既定値は master 込みで確定（2026-11-12）**: `k0=128, floor_ratio=0.25, tau=6`（2023 年のペア logloss 最小 0.6217。最適域は平坦で k0=96〜160・tau=3〜6 の差は 0.002 以内）、好走/凡走モデルは全 12 条件（2023 年で選定: 予測上振れと実績の相関 基本 0.162 → 全条件 0.197、2024 年以降でも 0.163 → 0.198。効きの大半は hs＝その芝ダでの実績）。**前向き検証（2024-01-01〜2026-11-01、4,461 レース）**: スコア 1 位の勝率 21.2%、3着内率 51.4%（CI95 49.9〜52.8%、無作為 24.2%）、2 頭の先着的中率 65.6%。v6 の ◎3着内 約 62% より下＝予測用ではなく比較用のものさし。好走気配 3着内 見込み 23.5% → 実際 30.2%（上振れ +0.067、レース単位 CI95 +0.064〜+0.071）、凡走注意 18.5% → 12.8%（−0.092、−0.097〜−0.089）。**読み方の注意**: ラベルの中身の多くは「スコア自体の癖の補正」（その芝ダが初の馬は相手平均から始めるため過大＝凡走注意、下振れ続きの馬はスコアが下がり過ぎ＝反動で好走気配）で、調子の良し悪しそのものではない。理由欄は条件ごとに平均的な馬との差（`form_model.json` の `coef_c`）で選ぶ。芝は距離を分けていないので上位は短距離〜マイルの G1 馬が並ぶ（サトノレーヴ 100 / ソウルラッシュ 97.5 / ナムラクレア 97.2、ダートは コスタノヴァ 100 / ウィルソンテソーロ 96.8）。build は約 2 分。master に `騎手` 列は無く `騎手コード` を使う（2026 年の出走表は騎手名なので、年またぎの 1 走だけ乗り替わり「不明」） |
| `weekly_nicegui.ps1` | 週次ワークフロー 3 フェーズ統合（sync-hf + sync-hf-umami 両方を叩く） |
| `nicegui_app.py` | NiceGUI 本体（v11、旧本番）。`sync-hf.ps1` でまだ併行更新 |
| `sync-hf.ps1` | （旧）master → hf-spaces orphan → NiceGUI Space へ push |
| `docs/cowork_prompt.md` | Cowork に投げるプロンプト |
| `docs/marks_schema.md` | bundle.json のスキーマ仕様 |

### Streamlit / 旧 predict ライン（副系統）
| ファイル | 役割 |
|---|---|
| `app.py` | Streamlit UI（Cloud 用） |
| `predict_weekly.py` | 旧 8 モデルアンサンブル + strategy_weights ベースの買い目 |
| `weekly_pre.ps1` / `weekly_post.ps1` | NiceGUI ps1 から呼ばれる旧 ps1 |
| `betting.py` / `kelly.py` / `ev_filter.py` | 旧買い目生成・Kelly・EV フィルタ |

### モデル学習・キャリブレータ
| ファイル | 役割 |
|---|---|
| `train_unified_rank.py` | unified_rank_v1〜 学習 |
| `optuna_v5_marks.py` / `optuna_v6_marks.py` | v5/v6 用 Optuna |
| `build_pl_calibrators.py` | PL calibrator fit |
| `build_payout_curve.py` | 期待払戻カーブ生成 |
| `run_v5_pipeline.py` | v5 系を一括再構築 |
| `audit_marks.py` / `audit_v6_vs_v5.py` | 印精度監査 |
| `calibrate.py` | 旧アンサンブル用 calibrator |
| `train_lgbm.py` / `train_catboost.py` / `train_transformer.py` 等 | 旧モデル学習群 |
| `stacking.py` | 旧スタッキング |

### バックテスト
| ファイル | 役割 |
|---|---|
| `backtest_pl_kelly.py` / `backtest_pl_ev.py` / `backtest_pl_formation.py` | v5 系（Plackett-Luce ベース） |
| `backtest_fixed.py` | 馬連 7 点流し固定戦略 |
| `backtest.py` / `backtest_v2.py` | 旧アンサンブル用 |

### 共通ユーティリティ
| ファイル | 役割 |
|---|---|
| `utils.py` | `add_meta()`, `parse_time_str()`, `backup_model()` 等 |
| `parse_kako5.py` | 過去5走パース |
| `parse_training.py` | 調教パース |
| `make_weekly_hosei.py` | 補正タイム週次生成 |
| `update_live_results.py` | live_results_2026.csv 更新 |
| `generate_results.py` | results.json 集計 |

### docs/ 重要ファイル
| ファイル | 中身 |
|---|---|
| `docs/cowork_prompt.md` | Cowork への投入プロンプト（絶対禁則 4 含む） |
| `docs/marks_schema.md` | bundle.json スキーマ |
| `docs/operation_roadmap.md` | 運用ロードマップ |
| `docs/PROJECT_OVERVIEW.md` | プロジェクト全体像 |
| `docs/hypothesis_registry.md` | 検証中の仮説リスト |

### ✅ ルート整理済み（2026-07-01）
実験・研究スクリプト **97 本を `lab/<theme>/` へ移動**（root .py 237→139）。移動対象は import 依存グラフ＋subprocess/文字列参照＋ps1/doc 参照を機械解析し、**本番が参照するものは root 固定**（import 破壊 0・全 py_compile 通過を検証済み）。詳細は `lab/README.md`。
- テーマ: `lab/experiments/ betting_lab/ bet_type_lab/ physics_gates/ backtest/ train/ audits/ features_dead/ pipelines_old/ sims/`
- **移動スクリプトの再実行は root から** `python -m lab.<theme>.<name>`（cwd=root が sys.path に乗り `import utils` 等が解決。`python lab/…/x.py` の直叩きは不可）
- 巨大生成物（`feats_1000.parquet` 686MB 等）・実験モデル・スクラッチは `.gitignore` 済み（**ディスクには残す**）
- 現用ツール（`place_weekly.py` `participation_analyzer.py` `keiba_card*.py` `build_bet_plan.py` 等）と本番エントリは **root 維持**（`gutchi_brain.py` は 2026-08-09 退役・削除、TACT は topdown へ置換）
- 注: 上の「ファイル役割マップ」内の一部実験スクリプト（`backtest_v2.py` `train_catboost.py` 等）は `lab/` 配下へ移動済み

**残タスク**：
- ~~`models/` 配下の日付付き pkl（`*_20260313_*.pkl` 等）→ `models/archive/` へ~~ **2026-11-04 完了**（残っていた `pl_calibrators_v6_serve_2026*` 3 本も archive へ。`models/` 直下に日付付き pkl は 0）
- 互換用旧マスター `data/master_20130105-20251228.csv` は v5 が `master_v2_*` を使うので削除候補
- `models/expert_*_rejected.pkl`（命名通り不採用）/ `models/lgbm_*_20260319_*.pkl` 中間 dump → archive 行き

---

## Cowork 実績（2026-04-18〜05-17、9 開催 / 292 bets / v5 期間）

| 馬券種 | 件数 | 的中 | hit 率 | 投資 | 払戻 | 収支 | **ROI** |
|---|---|---|---|---|---|---|---|
| 単勝 | 30 | 5 | 16.7% | 118,000 | 142,580 | **+24,580** | **120.8% ⭐** |
| ワイド | 95 | 29 | 30.5% | 398,200 | 389,431 | -8,769 | 97.8% |
| 複勝 | 48 | 26 | 54.2% | 238,100 | 170,100 | -68,000 | 71.4% |
| 馬連 | 119 | 14 | 11.8% | 525,700 | 295,657 | -230,043 | 56.2% |
| **合計** | 292 | 74 | 25.3% | 1,280,000 | 997,767 | -282,233 | **78.0%** |

控除率 80% に対し総合 78.0% で **ほぼ期待値中立**。単勝・ワイドは強み、複勝・馬連は要改善。
v6 + cowork_prompt 改修（節 11）でこの弱点改善を狙う。

> ⚠️ **上表は v5 期（4-5月）の歴史値であり現在地ではない**。最新累積（`data/cowork_results.json` 2026-06-22）は **620R / ROI 69.5%**（ワイド 78.0%・単勝 86.6%・各 by_type に CI95/roi_verdict 付き）。「現状の Cowork 実績」を語るときは v5 表でなくこの json を見る。

---

## 強化の本筋（「これ以上強くする」への回答, 2026-06-24）

外部 AI（sakana 等）の汎用提案は **予測 v7 / 特徴量追加 / EV 閾値で買い目を絞れ / ROI 85-90%** を推しがちだが、これらは**全て敵対的検証で死亡済みのルート**：

- **予測層は天井（◎top3≈62%）で閉店**。物理/EVT/統計力学/クラス変動+1400特徴ブルートフォースを v6 本番土俵で検定し採用ゼロ（規模を上げても天井不動）。独立 AI も同 58-62% 天井。
- **EV による銘柄選抜は -13pt の有害手**。`compute_bets.py:408-419` は既に prob-first に統一済み（馬連も prob-first 再有効化で test ROI 79%>控除）。**EV 閾値の再追加＝逆走**。
- **ROI 85-90% は無理筋**。罠を全切りしても残 ROI ≈ 79.9% ≈ 控除率＝期待値中立が床。CLV は pari-mutuel で換金不能。目標は「儲け」でなく「**最も負けへん線**」。

**残る本当のレバー＝検証済みだが本番未配線の ROI 回収**（詳細は記憶 `project_unwired_roi_audit` / `project_roi_max_doctrine`）：
1. **participation ゲートのバイパス解消（最有力, 実測 +5.31pt）**: clean-band ゲートは `compute_bets.py:246` にあるが主経路 `:580-581 force_floor=True` で常時殺している。配線＝二段化（クリーン帯外は見送りでなく消化枠降格）。
2. **UMAMI 正配線**（馬連 cap10=84%/ワイド cap3-4=82.8% は測定済だが本番の印スロット流しに帰属不可）。
3. **joint_m1 umaren**（ECE-33% / +0.3〜0.7pt、本番 export は素の PL 積のまま）。
4. **トリガミ床 + chalk-cap**（ユーザー自身のルールが ML を dominate した唯一の頑健改善、未配線）。

**鉄則**: どの点推定も単独で主軸化せず、**配線より「最新期間での CI 付き再検証」に手間を割く**（in-sample 蜃気楼回避）。

**2026-08-09 構築層4修正を配線済み**（commit fe96e5127d、`compute_bets.py`）: 4-8月全ベット解剖で慢性出血が馬券構築層に局在と特定（妙味絡みペア60.9%／EV配分で最厚点が最悪／非妙味◎単勝14%／6点目以降60%以下）。①穴overlayのvb-◎ペア撤去 ②◎単勝は妙味時のみ ③cap6→5 ④EVサイジング→p×boost配分（`CB_ALLOC` env で旧挙動可）。リプレイA/B（4/18-8/9・506R同条件）: 旧74.1%→新78.2%（全月プラス）。⚠同期間in-sample検証につき前向き監視で確認すること。

**2026-08-09 完全トップダウンエンジン配線・既定化**（commit ee17596c46、`CB_ENGINE=topdown` 既定／`shape` で旧経路）: 印スロット・shape・妙味ヒューリスティクスをバイパスし、全馬 p_win→λ補正PL（`data/harville_lambda.json`）で全ペア確率→確率順候補→p比例配分→**適応トリガミ床**。リプレイ 82.8%（全月78%以上、旧比+8.7pt、ペアCI[-0.5,+12.8] P(改善)=0.961）。実態は「最高確率馬の複勝厚+単勝+少数ワイド、平均1.9点/R」。§0/§0b 参戦ガードは維持。

---

**2026-09-30 本線エンジンを a95_equal_v1 へ切替**（`data/production_policy.json` engine=`a95`、policy `a95-equal-v1-serve34-p667-20260930`、起算 2026-10-03）: 生の p_win 順の上位馬で、AI1位の単勝オッズ帯ごとに「2023-25 OOS 表で回収率 95% 以上だった 1 点券種」を買い、1R 1万円を均等配分（方策の正本 `data/shadow_policies/a95_equal_v1.json`、生成は `a95_engine.py`）。T-10（`compute_bets.py`）・T-20 サイト速報（`t20_site_bets.py`）・Discord（`t10_runner.py`）・サイト表示/決済（`build_site.py`）へ配線。
- **参戦ガードは従来と同一**（混戦度 percentile・頭数・◎p_win・◎オッズ・障害 P0 gate）。紙上台帳 `shadow_a95.py`（CW-10）は全レース、本線（CW-11）はガード通過レースのみ。ガードで見送るレースにも T-10 オッズで「買っていたら」の買い目を `a95_all` として記録し、Discord に「🧪 全レース版（検証用）」と併記する（買い目ではない。`bets` は空のまま・サイトには出ない。集計は `python -m analysis.a95_all_t10_report`）。
- **`real_money.enabled` は false のまま**＝買い目は表示専用の扱い。実弾再開はこのフィールドと `docs/hypothesis_registry.md` の追記が必要。
- 馬単・枠連・三連単は engine=a95 のときだけ検証ガードが許可する。wide residual shadow v3 は親 policy 変更で cohort 停止。
- 控除率を超える証拠は無い（OOS 98〜110%、CI は 100 を跨ぐ、上位 10 払戻除外で 100 未満）。500R でレース的中率により初回棄却判定、ROI は記述のみ。旧経路へ戻すには policy の engine を `topdown` に戻す（旧 policy id `topdown-serve34-p667-20260825`）。

---

## 既知の問題

### 🟡 P1
1. **モデル 2 系統並走**: 退役計画 `docs/LEGACY_RETIREMENT_PLAN_20261004.md`。**2026-11-04 実施**: 段階 1（読み込み部を `weekly_parse.py` / `training_join.py` へ切り出し、本番は predict_weekly に依存しなくなった）、段階 2 の一部（value_model 月次再学習と旧 NiceGUI 同期を既定 OFF）、段階 3 縮小（predict_weekly 本体・ev_filter・precompute_buylist・strategy 系 3 本を `legacy/`、root は shim）。残: app.py・旧学習群・pkl（同文書 §6）
2. **`models/` ディレクトリ肥大化**: 41 ファイル (dated pkl 退避済み)、まだ整理余地あり
3. **ルート Python 93 本**: 整理進めたが `backtest_*` `train_*` の重複版残あり
4. **wide_kekka.csv 週次更新の運用化**: TARGET からのエクスポート自体は手動のまま。**2026-11-04**: `weekly_post.ps1` Step 2.8 が最終日付を検査し今週より古ければ警告（10/24〜11/01 の 4 開催日が未反映のまま公開されていた）、変更があれば Phase C の commit に同乗（それまで誰も commit しておらず 240 行が未コミットだった）

### 🟢 P2
5. **`strategy_weights.json` の位置づけ曖昧**: Streamlit 用なのか、廃止予定なのか書面化されていない
6. **`catboost_info/` を `.gitignore` に追加**
7. **`docs/` 配下のドキュメント間で記述が重複**: ROADMAP / WORKFLOW / PROJECT_OVERVIEW の役割分離
8. **v6 効果検証**: 2〜4 週運用後、v5 期間 (4/18-5/17) と v6 期間で Cowork ROI を比較

---

## 馬券種と印の対応（v5 では Cowork が最終決定）

| 馬券種 | デフォルト指針 |
|---|---|
| 単勝 | ◎〇△ のいずれか（Cowork が EV で選ぶ） |
| 複勝 | ◎〇 中心 |
| 馬連 | ◎軸 or ◎〇▲ ボックス |
| ワイド | ◎〇▲ ボックス |
| 三連複 | ◎〇▲ + 軸流し |
| 三連単 | **廃止** |

詳細ルールは `docs/cowork_prompt.md`。

---

## 注意事項

- 当日情報（馬体重・オッズ等）は学習特徴量から除外済み
- `単勝オッズ`（今走）は完全リーク（kekka は勝ち馬のみ収録）。EV 補正でのみ使う
- `前走単勝オッズ` は使用 OK
- `COL_RACE_ID` のカラム名は **`"レースID(新/馬番無)"`** に統一（旧 `"レースID(新)"` 揺れ注意）
- `master_v2_*.csv` は HF Spaces には未配置（~390MB、ローカルのみ）

---

## クイックリファレンス

```powershell
# 週次 Phase A（土曜朝、default Model=v6）
.\weekly_nicegui.ps1

# 週次 Phase B（Cowork 返答後）
.\weekly_nicegui.ps1 -BetsOnly

# 当日 T-10 自動馬券ライン（各レース発走10分前に JV-Linkオッズ取得→compute_bets
# →validate を自動実行し買い目表示。投票は人間IPAT。詳細: docs/compute_bets_spec.md）
# ★土日 9:00 はタスクスケジューラ「PyCaLiAI_T10」が自動起動 (bundle 待機つき) — 手動起動は不要。
#   祝日(月)開催のみ手動: .\t10.ps1   / テスト: .\t10.ps1 20260614 -Dry
# サイト反映(2026-10-03): T-10 確定の直後に t10_runner.py が sync-hf-umami.ps1 を切り離し起動
#   (publish_site_async)。以前は次レースの T-20 同期待ちで、発走に間に合わないレースがあった。
#   -Dry では反映しない。ログ: logs\t10_publish_{date}.log
# 補正印(オッズblend, 2026-07-03): T-10 で log(p_win)+λ·log(市場π) 再ランクの印を表示。
#   OOS実証 ◎top3 61.7%→65.1% (test2024-25, Δ+3.41pt CI[+2.5,+4.3])。表示専用・買い目不干渉。
#   λ再fit: python -m analysis.fit_t10_blend → data/t10_blend.json

# サイト公開用 T-20 買い目プレビュー（2026-09-10新設、2026-09-11 堅牢化）: 各レース
#   発走20分前に大会仕様ロジック(masters_vote.aite_switch_tickets)で計算しTACTへ
#   反映(オッズ非公開)。実投票(masters_vote, T-4)が入ればそちらに自動置換、大会側が
#   最終的に見送り/投票失敗と決めた後はT-20速報も出さない(load_masters_vote_final_rids)。
#   validate_market()がok/race_id一致/鮮度(10分)/現存馬全頭の単勝オッズ充足を検査、
#   不備があれば判定自体をスキップし取得失敗として扱う(bundleの朝オッズが黙って
#   混ざる経路を遮断)。判定結果はレース単位ファイル reports/masters_vote_site/{date}/{rid}.json
#   (同時刻帯の複数レース処理が競合しないよう分離)。公開はsync-hf-umami.ps1のみを
#   呼ぶ(内部にファイルロックreports/.locks/を追加、同スクリプトを呼ぶ全経路
#   [t20/baba_daily/weekly_nicegui]を直列化)。T-10/T-4の実ラインには一切干渉しない
#   (専用dir reports/site_odds、forward_prices stage="t20"でT-10履歴と分離)。
# ★土日 9:00 はタスクスケジューラ「PyCaLiAI_T20_Site」が自動起動 — 手動起動は不要。
#   テスト: .\t20_site.ps1 -Once <rid16> -Dry

# 週次 Phase C（日曜夜、kekka 配置後）
.\weekly_nicegui.ps1 -Post

# NiceGUI ローカル起動
python nicegui_app.py

# Streamlit 起動
streamlit run app.py

# v6 系再構築 (本番)
python run_v6_pipeline.py

# v5 系再構築 (rollback 用)
python run_v5_pipeline.py

# 印監査（v6 vs v5）
python audit_marks.py --model v6

# Cowork 集計再生成 (kekka 追加後)
python generate_results.py
```
