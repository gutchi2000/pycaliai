# PyCaLiAI — Claude Code 引き継ぎメモ

> **このファイルの位置づけ**: 次に触る人（含む将来の自分）が 5 分で全体像を掴めるようにする。
> 詳細仕様は `docs/` 配下、運用フローは `WORKFLOW.md` を参照。
>
> 最終更新: 2026-07-01（**ルート整理**: 実験・研究スクリプト 97 本を `lab/<theme>/` へ集約し root .py を 237→139 に。巨大生成物/スクラッチを `.gitignore`。詳細は下記「✅ ルート整理済み」節と `lab/README.md`）
> 旧: 2026-06-24（**現状整理**: cowork_results は修復済み＝壊れていない／Cowork 78% 表は v5 期の歴史値で現在値ではない／「これ以上強くする」の本筋は予測 v7 や EV 閾値ではなく **検証済み未配線 ROI の回収**＝下記「強化の本筋」参照）
> 旧: 2026-06-16（**本番表示を静的サイト pycaliai-umami に移行**、週次フローに sync-hf-umami 組込み、predict デフォルト skip / 見送りガードを反映）
> 旧: 2026-05-20（NiceGUI + Cowork パラダイムに全面改訂、**v6 本番投入**、cowork_results 集計修復）

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
1. `make_weekly_hosei.py` → `data/hosei/H_{date}.csv`
2. `predict_weekly.py` → **デフォルト SKIP**（2026-06-11 audit P1-1。旧8モデルは重く NiceGUI/Cowork に不要）。Streamlit 用に欲しい週だけ `-WithPredict`
3. `export_weekly_marks.py --model v6` → `reports/cowork_input/{date}_bundle.json` **★これが Cowork のキー入力**
4. `build_course_stats.py` → `data/course_stats.json`（コース分析タブ用）
5. `git push origin master` → `sync-hf.ps1`（旧 NiceGUI Space）→ `sync-hf-umami.ps1`（**本番 静的サイト**、`build_site.py` で再生成して push）

### Phase B — 土曜昼（Cowork が買い目を返してきたら）
```
1. Claude Desktop に reports/cowork_input/{date}_bundle.json を投入
   （プロンプトは docs/cowork_prompt.md を貼る）
2. Cowork のレスポンスを reports/cowork_output/{date}_bets.json として保存
3. .\weekly_nicegui.ps1 -BetsOnly
```
内部で：
1. `validate_cowork_bets.py --apply` で**見送りガード**を強制（絶対禁則の4条件。Cowork が見送るべき race に買い目を付けても bets:[] に矯正）。ガード自体が実行不能なら **fail-closed で停止**（未検証 bets を本番に出さない）。どうしても push したい時のみ `-Force`
2. `git push origin master` → `sync-hf.ps1` → `sync-hf-umami.ps1`

### Phase C — 日曜夜（TARGET 結果エクスポート後）
```powershell
# data/kekka/{date}.csv を配置
.\weekly_nicegui.ps1 -Post
```
内部で：
1. `weekly_post.ps1`（失敗したら **fail-hard** して HF 同期を中止）
   - `generate_results.py` → `data/results.json` / `data/cowork_results.json`（**集計凍結対策で必ず commit**）
   - `update_live_results.py` → `data/live_results_2026.csv`
   - `git push` + `reports/cowork_bets/{date}/`
   - `cowork_results.json` の `generated_at` が当日かチェック（凍結検知）
2. `sync-hf.ps1`（旧 NiceGUI）→ `sync-hf-umami.ps1`（**本番 静的サイト**）
3. 日曜 + 月初 1〜7 日なら `retrain_value_model.py` 自動実行
4. 日曜なら `run_audit.ps1` 自動実行（週次監査）

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
| `export_weekly_marks.py` | unified_rank_v6 → bundle.json |
| `build_course_stats.py` | コース分析タブ用統計 |
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
- 現用ツール（`place_weekly.py` `participation_analyzer.py` `gutchi_brain.py` `keiba_card*.py` `build_bet_plan.py` 等）と本番エントリは **root 維持**
- 注: 上の「ファイル役割マップ」内の一部実験スクリプト（`backtest_v2.py` `train_catboost.py` 等）は `lab/` 配下へ移動済み

**残タスク**：
- `models/` 配下の日付付き pkl（`*_20260313_*.pkl` 等）→ `models/archive/` へ
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

---

## 既知の問題

### 🟡 P1
1. **モデル 2 系統並走**: v6 marks stack と 旧アンサンブル両方を `weekly_nicegui.ps1` で生成中
   - `predict_weekly.py` は重く、`-SkipPredict` 推奨
   - 中長期は v6 統一して旧側を `legacy/` に隔離
2. **`models/` ディレクトリ肥大化**: 41 ファイル (dated pkl 退避済み)、まだ整理余地あり
3. **ルート Python 93 本**: 整理進めたが `backtest_*` `train_*` の重複版残あり
4. **wide_kekka.csv 週次更新の運用化**: 現状ユーザー手動配置、weekly_post.ps1 への組込み検討

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
# 補正印(オッズblend, 2026-07-03): T-10 で log(p_win)+λ·log(市場π) 再ランクの印を表示。
#   OOS実証 ◎top3 61.7%→65.1% (test2024-25, Δ+3.41pt CI[+2.5,+4.3])。表示専用・買い目不干渉。
#   λ再fit: python -m analysis.fit_t10_blend → data/t10_blend.json

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
