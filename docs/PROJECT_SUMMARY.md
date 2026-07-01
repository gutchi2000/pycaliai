# PyCaLiAI 技術サマリ（外部競馬MLアドバイザー向け）

> 目的: リポジトリの現状を、推測でなく**実ソース・設定・git管理ファイルの根拠付き**でまとめる。
> 各節末に参照ファイルパスを併記。確証のない箇所は `[推定]`、確認できない箇所は `不明` と明記する。
>
> 作成日: 2026-06-01 / 対象コミット時点の実コードを読んで作成。
> 重要な前提: 本リポジトリには**2つのモデル系統が並走**している。
> - **本番ライン (v6 marks stack)**: LightGBM LambdaRank → Plackett-Luce → 印JSON → Cowork(LLM)が馬券決定
> - **レガシーライン (Streamlit用)**: 8モデルAUC加重アンサンブル → strategy_weights.json ルール → Kelly
>
> 以下、特記なき限り「本番ライン」を主に記述する。

---

## 0. ドキュメントと実態の乖離（先に共有すべき注意点）

実コード/データを読んだ結果、`CLAUDE.md` の記述と現物がずれている点が複数見つかった。アドバイザーが数字を扱う前に把握しておくべき：

| 項目 | CLAUDE.md の記述 | 実態（根拠ファイル） |
|---|---|---|
| `cowork_results.json` | 「集計が全0で壊れている。要修正」 | **壊れていない**。2026-05-26 再生成済みで全集計が非ゼロ（445 bets / ROI 75.9%）。`data/cowork_results.json` |
| v6 の本番採用 | 「v6 本番投入」 | **採否は文書間で矛盾**。`reports/audit_v6_vs_v5_20260520.md` は「❌ v6 採用見送り」（高EV単勝ROIが基準未達）と結論。一方 calibration 改善は実測で確認できる。さらに v7/v8/v9 の Optuna 成果物が存在し開発継続中。 |
| ECE 改善幅 | 「複勝-32% / 馬連-34%」 | 実測の audit ファイルでは 3年集計で**複勝◎ -44% / 馬連◎〇 -43%**（より大きい）。`reports/audit_marks_v5.json` `audit_marks_v6.json` |
| strategy_weights | 「13条件 5会場」 | 実ファイルは**13条件 / 4会場**（中京・中山・新潟・福島）。`data/strategy_weights.json` |
| Cowork実績 | 「292 bets / ROI 78.0% / 4-18〜5-17」 | これは**古いスナップショット**。現物は445 bets / 75.9% / 6週（〜5-24）。292/78.0%を持つファイルはディスク上に存在しない。 |

---

## 1. パイプライン全体

### 本番ライン（土〜日の週次運用、3フェーズ）

統合スクリプト: `weekly_nicegui.ps1`（内部で `weekly_pre.ps1` / `weekly_post.ps1` を呼ぶ）

```
[Phase A 土曜朝]
TARGET出走表 CSV (data/weekly/{YYYYMMDD}.csv)
  ├─ make_weekly_hosei.py  → data/hosei/H_{date}.csv（補正タイム週次）
  ├─ predict_weekly.py     → reports/pred_{date}.csv（Streamlit用、-SkipPredict で省略可）
  ├─ export_weekly_marks.py --model v5/v6  → reports/cowork_input/{date}_bundle.json  ★主出力
  ├─ build_course_stats.py → data/course_stats.json
  └─ git push → sync-hf.ps1（HF Spaces 反映）

[Phase B 土曜昼]
bundle.json を Claude Desktop(Cowork) に投入（docs/cowork_prompt.md を貼付）
  → Cowork が馬券構築（馬券種・点数・予算配分・見送り）
  → reports/cowork_output/{date}_bets.json として保存
  → weekly_nicegui.ps1 -BetsOnly（validate_cowork_bets.py で見送り条件をコード強制）

[Phase C 日曜夜]
TARGET結果 CSV (data/kekka/{date}.csv)
  ├─ generate_results.py       → data/results.json / data/cowork_results.json
  ├─ update_live_results.py    → data/live_results_2026.csv
  ├─ sync-hf.ps1
  └─（日曜/月初）retrain_value_model.py / run_audit.ps1 自動実行
```

### 各段の入出力と主要関数

| 段 | 入力 | 出力 | 主要ファイル / 関数 |
|---|---|---|---|
| データ取得 | TARGETフロンティア手動エクスポート | `data/weekly/*.csv`, `data/kekka/*.csv`, `data/kako5/*.csv`, 調教 `H/W-*.csv` | （手動） |
| マスター生成 | 上記 + 調教マスター | `data/master_v2_*.csv`（132列） | `build_master_v2.py`（`_latest_before()` で `merge_asof`） |
| 特徴量（推論時付加） | master / weekly | race内 z-score・rank 等 | `race_relative_feats.py: add_race_relative_feats()` |
| 週次パース | weekly CSV | DataFrame | `predict_weekly.py: parse_csv()` |
| 印付け・確率 | model + calibrator | bundle.json | `export_weekly_marks.py: main()` → `export_marks_json.py: export_race()` |
| 意思決定 | bundle.json | bets.json | **Cowork (Claude Desktop, LLM)**、プロンプト = `docs/cowork_prompt.md` |
| 買い目検証 | bets.json | 検証済 bets | `validate_cowork_bets.py`（見送り4条件をコード強制） |
| 表示 | bets.json + results | 画面 | `nicegui_app.py`（HF Spaces本番）/ `app.py`（Streamlit） |

> 参照: `weekly_nicegui.ps1`, `export_weekly_marks.py`, `export_marks_json.py`, `build_master_v2.py`, `predict_weekly.py`, `docs/cowork_prompt.md`, CLAUDE.md「週次運用フロー」

**重要な役割分担**: PyCaLiAI（コード）は**印付けと確率算出のみ**。馬券種選択・点数・予算配分・見送り判断は**Cowork(LLM)が最終決定**する。機械的ルール（strategy_weights.json）はレガシーStreamlit側のみ。

---

## 2. モデル本体

### 2.1 目的関数

- **本番 v6**: LightGBM **LambdaRank**（`objective="lambdarank"`, `lambdarank_truncation_level=5`, `metric="ndcg"`, `eval_at=[5]`）。
- **Optuna の最適化目的**（モデル選択の指標）は LambdaRank の内部 loss ではなく、独自 composite：
  ```
  composite = 0.30·NDCG@5 + 0.25·◎top3率 + 0.20·(実top3⊂予測top5)率
            + 0.15·(勝馬∈予測top5)率 + 0.10·◎top2率
  目的       = composite − 0.5·ECE_high_p
  ECE_high_p = | mean(p_win) − mean(actual) |   (p_win ≥ 0.10 の高確率帯のみ)
  ```
  > 参照: `optuna_v6_marks.py:82-90, 261-270, 303-344`

- **理由（コード内コメント根拠）**: v5 監査で「`alpha=1.325` だと穴馬の raw_score が系統的に嵩上げされ、高確率帯で calibration が崩壊」が判明。v6 は **calibration awareness を目的関数に組み込み**（ECEペナルティ）、ランキング性能と calibration のトレードオフを Optuna に解かせる設計。
  > 参照: `optuna_v6_marks.py:1-27`（docstring に設計意図）

- **sample_weight**: `1 + alpha·log1p(winner_tansho/100)`（高配当レースを重み付け）。alpha 探索範囲 v6=[0,1.5]（v5=[0,2.0]）。実測 best alpha=**0.031**（ほぼ無効化＝均等重みに近い）。
  > 参照: `optuna_v6_marks.py:148-157, 304-305`、best値は `reports/optuna_v6_marks.json`

### 2.2 教師ラベルの定義

- **本番 v6**: `label = clip(6 − 着順, 0, 5)` → **段階的 relevance**（1着=5, 2着=4, …, 6着以下=0）。**勝ち負け二値ではない**。着順そのものを relevance として LambdaRank に与える。
  > 参照: `optuna_v6_marks.py:113`
- **v1/v2 旧版**: `label = clip(11 − 着順, 0, 10)`（同じく段階 relevance、上位10着まで）。
  > 参照: `train_unified_rank.py:76`
- **レガシーアンサンブル**: ターゲットは `fukusho_flag`（3着以内=1）の**二値**。こちらは着順を二値に潰している。
  > 参照: CLAUDE.md「ターゲット変数: fukusho_flag」、`ensemble_weights.json`（fuku_lgbm 等）

### 2.3 グループ単位（レース）の扱い

- LightGBM の `group` 引数にレースごとの行数配列を渡す。`groupby(COL_RID)` で連続行をグルーピング（学習前に `sort_values(COL_RID)` 必須）。
  ```python
  g = np.array([len(list(gr)) for _, gr in groupby(d[COL_RID])])
  lgb.Dataset(X, label=y, group=g, weight=w)
  ```
  > 参照: `optuna_v6_marks.py:148-157`, `train_unified_rank.py:125-130`
- `COL_RID = "レースID(新/馬番無)"`（馬番を含まないレース単位ID）。

### 2.4 ランキングスコア → 勝率/複勝率 変換

2段構え：

**(a) Plackett-Luce による厳密 joint 確率**（`pl_probs.py`、近似なし）
- 重み `w_i = exp(s_i − max(s))`（softmax的、オーバーフロー回避）
- 単勝 `p_i = w_i / Σw`
- PL定義: `P(順列 h1..hk) = Π_m w_{hm} / (Σw − Σ_{j<m} w_{hj})`
- 複勝(top3)・馬連(top2)・ワイド・三連複・三連単すべて閉形式で計算（`p_fukusho`, `p_umaren`, `p_wide`, `p_sanrenpuku`, `p_sanrentan`）。自己テストで Σ単勝=1, Σ複勝=3, Σ馬連=1 等を検証済み。
  > 参照: `pl_probs.py:25-161, 167-236`

**(b) Isotonic Regression による calibration**（`build_pl_calibrators.py`）
- 馬券種ごと（tansho/fukusho/umaren/wide/umatan/sanrenpuku/sanrentan の7種）に独立に Isotonic 回帰を fit。
- **fit データ = valid (2023) のみ**。各レースで PL確率 vs 実的中(0/1) のペアを集め、`IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)` で単調補正。
  > 参照: `build_pl_calibrators.py:86-174`
- 推論時、`export_race()` 内で `calibrators["tansho"].predict(p_win)` / `["fukusho"].predict(p_sho)` を適用。
  > 参照: `export_marks_json.py:245-249`

### 2.5 アンサンブル構成

- **本番 v6 ライン**: アンサンブルなし。**単一 LightGBM LambdaRank** のみ。
- **レガシー Streamlit ライン**: **8モデルの AUC加重平均**。
  | モデル | 重み |
  |---|---|
  | fuku_lgbm | 0.5736（支配的） |
  | regression | 0.1611 |
  | rank_cat | 0.1115 |
  | catboost | 0.0659 |
  | rank_lgbm | 0.0329 |
  | lgbm | 0.0278 |
  | fuku_cat | 0.0265 |
  | lgbm_win | 0.0007 |
  - valid AUC=0.7767 / test AUC=0.7809。重みは AUC最適化（Nelder-Mead系）で算出。
  - スタッキング（stacking_meta_v1）は IsotonicRegression が定数出力に潰れ、100%アンサンブルへフォールバック中（`docs/hypothesis_registry.md` に廃止記録）。
  > 参照: `models/ensemble_weights.json`, CLAUDE.md「旧アンサンブル」, `docs/hypothesis_registry.md`

---

## 3. 特徴量

### 3.1 特徴量一覧（構成ルール）

- **採用ルール**: `master_v2`（132列）のうち `LEAK_COLS` と `label` を除く全列。明示リストではなく除外方式。
  ```python
  LEAK_COLS = {"着順", "fukusho_flag", "roi_target", "レースID(新)",
               "レースID(新/馬番無)", "馬名", "レース名", "発走時刻",
               "date_dt", "日付", "血統登録番号", "split"}
  ```
  > 参照: `optuna_v6_marks.py:68-73`, `train_unified_rank.py:51-56`
- **特徴量数（実測）**: v1=**120**, v2=**132**（`reports/unified_rank_v1_eval.json` / `v2_eval.json` の `n_features`）。v5/v6 の正確な特徴量数は eval JSON が存在せず **不明**（[推定] 同程度の120前後。v6 は prep() で race_relative を再付加していないため master_v2 列依存）。
- **カテゴリ列（28個）**: 場所/芝・ダ/コース区分/馬場状態/天気/クラス名/種牡馬/父タイプ名/母父馬/母父タイプ名/騎手コード/調教師コード/性別/前走場所 等。`LabelEncoder` で train で fit、未知値は `"__NaN__"` に寄せる。
  > 参照: `optuna_v6_marks.py:74-80, 119-135`
- **特徴量ファミリー（master生成側）**:
  - 補正タイム: `前走補正`, `前走補9`（**今走補正は除外＝リーク**）
  - 調教: `trnH_*`（坂路）, `trnW_*`（WC）
  - コース累積: `course_n_prev`, `course_wins_prev`（馬×コースの過去累積）
  - 騎手/厩舎/馬 30日複勝率: `jockey_fuku30`, `trainer_fuku30`, `horse_fuku30`
  - 過去5走サマリ: `kako5_avg_pos`, `kako5_best_agari3f`, `kako5_std_pos` 等
  > 参照: `build_master_v2.py:5-9, 128-204`, `race_relative_feats.py:29-43`

### 3.2 標準化方法

- **週×クラスの median/MAD ではない**。実装は**レース内（race-relative）標準化**：
  - z-score: `(x − レース内mean) / レース内std`（std=0は NaN→0埋め）
  - rank: `(rank − 1)/(n − 1)`（レース内パーセンタイル、欠損は0.5）
  - v2追加: peer_max/peer_min（同レース内の相手最強値）、`kako5_agari3f_volatility`（上がりブレ幅）、`kako5_pos_volatility`、`peer_count`
  ```python
  g = df.groupby(COL_RID)
  df[f"{c}_race_z"] = ((df[c] − g[c].transform("mean")) / g[c].transform("std")).fillna(0)
  ```
  > 参照: `race_relative_feats.py:46-104`
- 連続値の欠損は最終的に LightGBM 入力で `fillna(-9999)`。
  > 参照: `optuna_v6_marks.py:150`, `export_marks_json.py:230`

### 3.3 リーク防止の実装箇所（具体）

| 対策 | 実装 | 参照 |
|---|---|---|
| 目的変数直系の除外 | `LEAK_COLS`（着順/fukusho_flag/roi_target 等） | `optuna_v6_marks.py:68-73` |
| 当日情報の除外 | 馬体重・今走オッズ・人気を学習特徴量から除外（CLAUDE.md明記） | CLAUDE.md「注意事項」 |
| 補正タイムの as-of | 「今走補正＝目的変数と同時決定→リーク」とコメント。`前走補正` のみ採用 | `build_master_v2.py:7` |
| 調教の point-in-time | `merge_asof` で `train_date < 日付` の直近のみ結合 | `build_master_v2.py:129-151` |
| コース累積の自己除外 | `cumsum() − 当該行` で自レースを引く（過去のみ） | `build_master_v2.py:200-204` |
| コース相性の train限定 fit | 種牡馬/母父/騎手/馬×コースの集計は train(~2022)のみで作成、推論時 JOIN | `build_course_affinity.py:5` |

> **既知のリーク懸念（要改善）**: コース相性（affinity / v8系）は train 期間で集計テーブルを作る際に**自レースを含む in-sample 集計**になっており、train 行で leak の可能性がある（test は train-only テーブル参照なので clean）。as-of / OOF 化が必要。
> 参照: ユーザーメモ `project_v8_affinity_insample_leak.md`, `build_course_affinity.py`

---

## 4. 着順と確率構造【重点】

### 4.1 着順を二値に潰しているか / 順列を使っているか

- **本番ライン: 順列（着順全体）を使う**。
  1. **学習**: LambdaRank に段階 relevance `clip(6−着順,0,5)` を与え、着順上位ほど高評価になるよう NDCG@5 を最適化（`train_unified_rank.py`/`optuna_v6_marks.py`）。二値ではない。
  2. **確率化**: ランキングスコアを **Plackett-Luce で全順列の同時確率**に変換（`pl_probs.py`）。単勝・連対・複勝・馬連・三連系すべて閉形式。
- **レガシーアンサンブル: 二値**。`fukusho_flag`（top3=1）を直接学習。順列構造なし。

### 4.2 Plackett-Luce / 条件付きロジット的「同時確率」を扱う箇所

- **明確に存在する**。`pl_probs.py` が PL の厳密実装：
  - `pl_weights(scores) = exp(s − max)`（条件付きロジットの重み）
  - `p_sanrentan(i,j,k) = w_i/T · w_j/(T−w_i) · w_k/(T−w_i−w_j)`（逐次選択 = 条件付きロジット連鎖）
  - `p_umaren = p_umatan(i,j) + p_umatan(j,i)`、複勝は1〜3着の周辺化
  > 参照: `pl_probs.py:25-118`
- バックテストのベクトル化版も同じ閉形式（`all_umaren_mat`, `all_fukusho_vec_fast`, `all_sanrenpuku_tensor`）。
  > 参照: `backtest_pl_ev.py:112-195`

### 4.3 馬連・三連系の EV を単勝確率からどう導出するか

**2系統で導出方法が異なる**点が重要：

**(a) バックテスト/curve（コード内）**: PL の**厳密 joint 確率**を使う。
```
p_raw = all_umaren_mat(w)[i,j]      # PL厳密
p_cal = calibrators["umaren"].predict(p_raw)   # Isotonic補正
pay   = lookup_pay_vec(payout_curve["umaren"], p_cal)   # 確率→実配当curve
EV    = p_cal × pay
```
- ここで `pay` は**個別馬の実オッズではなく**、prob bin ごとの**実配当中央値カーブ**（`pl_payout_curve_v6.pkl`、train+valid≤2023で作成）。
- → バックテストのEVゲートは「その確率帯は歴史的に+EVか」の判定であり、**特定馬券の市場ミスプライスは検知できない**（curve依存）。
  > 参照: `backtest_pl_ev.py:225-358`, `build_payout_curve.py:1-10, 137-211`

**(b) ライブ/Cowork（意思決定層）**: bundle.json には**各馬の p_win/p_plc/p_sho と実オッズ**しか載らない（PLペア行列は非搭載）。Cowork は p_win から**条件付き近似**で馬連等を計算：
```
単勝 EV   = p_win(◎) × tansho_odds(◎)
複勝 EV   = p_sho(◎) × (fuku_low + fuku_high)/2
馬連 EV   = umaren_matrix["i-j"] × [ p_i·p_j/(1−p_i) + p_j·p_i/(1−p_j) ]
ワイド EV ≈ (umaren_matrix["i-j"]/3) × (p_sho_i · p_sho_j)
```
- 馬連の実オッズは `data/odds/OD*.CSV`（TARGET形式）から取得した `umaren_matrix` を使用。馬単/三連は確定オッズ無しのため Cowork 側で対象外。
  > 参照: `docs/cowork_prompt.md:361-368, 426-433`, `export_marks_json.py:302-312`, `export_weekly_marks.py:86-124`

> **まとめ**: 確率の「同時分布」は PL で厳密に持っているが、**ライブ意思決定では bundle に p_win しか渡さないため、馬連は単勝確率からの近似で再計算**している（PL厳密joint はライブには流れていない）。アドバイザーが改善余地を探すならここ（bundleにPLペア行列を載せる等）。

---

## 5. 市場オッズの扱い【重点】

### 5.1 オッズを特徴量 or 二段目モデルに取り込んでいるか（Benter的構成の有無）

- **一段目（ランキングモデル）の入力特徴量にはオッズを使っていない**。`master_v2` に今走オッズ列が無く、`LEAK_COLS` 方式で除外される設計（今走単勝オッズは完全リークと明記）。
  - `前走単勝オッズ` は「使用OK」と CLAUDE.md にあるが、**master_v2 に実在するかはコードから確認できず 不明**（[推定] 未搭載）。
  > 参照: `optuna_v6_marks.py:68-73`, CLAUDE.md「注意事項」
- **二段目モデルは部分的に存在する（ただし Benter 型の確率ブレンドではない）**：
  - `value_model_v2.pkl`（`train_value_model.py`）: 1段目の base_prob + オッズ系特徴（`tan_odds`, `log_tan_odds`, `odds_rank_ratio` 等）から**期待ROIを予測**する Layer2。確率の指数ブレンド `p_model^a · p_market^b` ではない。
  - `bundle_signals.py`: 馬連ペアのみ `umaren_pair_v1` と v6 PL を **α=0.8 で線形ブレンド**（`p_umaren_stacked`）。勝率には未適用。
  - ai_vs_market: `market_p = 1/odds` と `p_win` を比較し under/fair/over に分類（ブレンドではなく分類タグ）。
  > 参照: `train_value_model.py`, `bundle_signals.py`, `export_marks_json.py:155-178`
- **結論**: 古典的 **Benter 二段（モデル確率 × 市場確率の対数線形ブレンド）は未実装**。市場オッズは (1)Cowork へのEV計算材料、(2)value_model のROI予測入力、(3)馬連ペアの限定的ブレンド、として使われるのみ。

### 5.2 CLV（クロージングラインバリュー）計測

- **未実装**。`CLV` / `closing` / `終値` / `確定オッズ vs 前売り` の差分追跡コードは見つからない。
- オッズは朝〜場内の単一スナップショット（`data/odds/OD*.CSV` / weekly CSV の `単勝`列）を静的に使うのみで、時間変化の追跡なし。
  > 参照: 全 .py grep で該当なし（Explore調査）、`export_marks_json.py:82-114`, `parse_od_csv.py`

### 5.3 補足: 「EV反転」現象の認識

- `ev_filter.py` に「高EVほど実ROIが低い（model過信）」という EV reversal を補正する `EVCalibrator` の記述あり。calibration の崩れがEVに伝播する問題は認識・対処されている。
  > 参照: `ev_filter.py`, `optuna_v6_marks.py:17-22`（v5監査での過信記録）

---

## 6. 検証

### 6.1 分割手法（時系列）

- **時系列固定分割**（walk-forward ではない、単一ホールドアウト）。`build_dataset.py` で日付しきい値により `split` 列を付与：
  ```python
  TRAIN_END = 20221231   # train: ≤ 2022-12-31
  VALID_END = 20231231   # valid: 2023
                          # test : 2024-01-01 〜
  ```
  > 参照: `build_dataset.py:39-41`, `build_master_v2.py`（split列生成）, CLAUDE.md「時系列分割」
- モデル選択（Optuna）は valid=2023 で評価。test=2024-2025 は独立評価。
  - ただし Optuna 内の「5-fold CV」は **valid をレースID単位で5分割して分散推定**するもので、early stopping も valid を使うため、**モデル選択と早期停止が同一 valid 上**で行われる（真の入れ子CVではない）点は留意。
  > 参照: `optuna_v6_marks.py:273-294, 327-333`

### 6.2 strategy_weights.json の「選択と評価が同一データ」循環

- **問題は文書化済み・部分的に解決**。
  - **発見**: `docs/hypothesis_registry.md`（2026-03-25）に「採用判断 ROI_test≥80% と評価が同じ2024-2025データ → 30エントリ全てが循環」と定量証拠付きで記録。
  - **再設計の存在**: `build_strategy_walkforward.py` が循環解消版。
    - 採用判断 = train(2013-2022) walk-forward（複数年で黒字）＋ valid(2023) のみ。
    - test(2024-2025) は**採用基準に使わず独立記録のみ**。
    - 「test で選んで test で評価する循環を解消する」とコード冒頭・57行・339行に明記。
  - **strict版**: `build_strategy_stable.py` は valid+test 両方黒字のみ採用（両期間フィルタ）。
  > 参照: `build_strategy_walkforward.py:1-19, 57, 208-214, 339`, `build_strategy_stable.py:4-6, 121-128`, `docs/hypothesis_registry.md:316-361`
- **現物の状態（`data/strategy_weights.json`）**: 13条件 / 4会場（中京・中山・新潟・福島）。各エントリに `roi_valid` / `roi_test` / `n_races_valid/test` / `profitable_years` を保持。
  - **しかし valid→test の乖離が依然大きいエントリが残る**（選択が valid でも OOS=test を保証していない）:
    - 中山ｵｰﾌﾟﾝ三連複: valid 190.0% → **test 13.8%**（n=14/26 と極小）
    - 福島未勝利三連複: valid 108.7% → test 48.1%
    - 新潟未勝利馬連: valid 83.4% → test 51.7%
  - **位置づけ**: このファイルは**レガシー Streamlit 表示のみ**が参照。本番 Cowork ラインは strategy_weights を読まず、`data/class_prior_v6.json`（クラス×印の経験的中率）を Cowork の判断材料に使う。
  > 参照: `data/strategy_weights.json`, `export_weekly_marks.py:191-211`

### 6.3 事前登録仮説と棄却履歴

`docs/hypothesis_registry.md` に登録済み：

| ID | 内容 | 判定 | 根拠 |
|---|---|---|---|
| H1 | field_spread × 市場歪み（混戦のミスプライス） | **棄却**（2026-03-25） | 独立検証で差+0.7%のみ、閾値未達。市場は混戦を正確に織込み |
| H2 | class_gap × 降級効果 | **棄却寄り** | 3期間中2期間で逆転、n=27で統計不十分 |
| H3 | EV除外ルール（3.0-4.0帯） | **現行維持**（閾値3.0） | 両条件未達 |
| H4 | G1直後ローテ馬のRPCI過小評価 | **検証予定**（2026下半期、n≥200後） | 未検証 |
| （廃止）| 重×16頭除外ルール | 削除（2026-03-25） | 単年n=130 vs 3年n=376 で2024矛盾 |
| （廃止）| スタッキングcalibrator | 廃止（2026-03-28） | Isotonic定数出力、100%フォールバック |

> 参照: `docs/hypothesis_registry.md:140-196, 287-361`

### 6.4 バックテストスクリプト

| スクリプト | 戦略 | ホールドアウト |
|---|---|---|
| `backtest_fixed.py` | 馬連 ◎〇軸-◎〇▲△△ 7点流し（EV閾値なし、¥700/R固定） | valid=2023 表示, 真OOS=2024-2025 |
| `backtest_pl_kelly.py` | EVゲート + fractional Kelly 配分 | 同上 |
| `backtest_pl_ev.py` | EV閾値ゲート（80〜200%）、PLベクトル化 | test=2024-2025（include_valid可） |

> 参照: `backtest_fixed.py:1-12`, `backtest_pl_kelly.py:1-11`, `backtest_pl_ev.py:43, 199-222`

---

## 7. 現状の成績（実測値）

### 7.1 データ規模（実測）

- 行数: **626,774**（`data/master_v2_20130105-20251228.csv`、ヘッダ除く）
- レース数: **44,907**（`レースID(新/馬番無)` のユニーク数）
- 期間: **2013-01-05 〜 2025-12-28**
- 列数: **132**
- 正例率（fukusho_flag）: ≈21.9%（3着以内）
> 参照: 実測 line count + `reports/master_v2_build_log.json`, CLAUDE.md

### 7.2 印精度・順位指標（v6、`reports/audit_marks_v6.json`）

| 指標 | 3年(2023-25) | 真OOS(2024-25) | 2024 | 2025 |
|---|---:|---:|---:|---:|
| レース数 | 10,327 | 6,878 | 3,440 | 3,438 |
| NDCG@5 | 0.6002 | 0.6040 | 0.6088 | 0.5991 |
| ◎単勝的中率 | 30.01% | 30.29% | 30.61% | 29.96% |
| ◎連対率(top2) | 48.86% | 49.52% | 49.94% | 49.10% |
| ◎複勝圏率(top3) | 61.64% | 62.08% | 62.76% | 61.40% |

- Optuna v6: best_composite=**0.5490**, alpha=0.031, ece_high_p=0.0112（`reports/optuna_v6_marks.json`）
- 旧版参考（per-split tansho/fukusho、`reports/unified_rank_v1_eval.json`）: valid 単勝29.9%/複勝60.6%、test(2024+25) 単勝29.9%/複勝61.3%/三連単1.66%。
> 注: v5/v6 の `unified_rank_v*_eval.json` は**存在しない**。v5/v6 の印率は `audit_marks_v{5,6}.json` から取得。

### 7.3 キャリブレーション（ECE、v6 vs v5、3年集計）

| ECE指標 | v5 | v6 | 改善 |
|---|---:|---:|---:|
| 単勝◎ | 0.01582 | 0.01196 | **−24%** |
| 複勝◎ | 0.01544 | 0.00864 | **−44%** |
| 複勝〇 | 0.01990 | 0.01509 | −24% |
| 馬連◎-〇 | 0.01847 | 0.01045 | **−43%** |

> v6 の主たる改善点は calibration（CLAUDE.md は−32/−34%と控えめ、実測はより大きい）。印精度自体は v5≈v6（微差）。
> 参照: `reports/audit_marks_v5.json`, `reports/audit_marks_v6.json`

### 7.4 機械買いバックテスト ROI（馬連7点流し固定、`reports/backtest_fixed_v6.json`）

| 期間 | ROI | 的中率 |
|---|---:|---:|
| 2023(valid) | 76.3% | 45.1% |
| 2024 | 77.4% | 48.6% |
| 2025 | 76.2% | 46.3% |
| 真OOS(2024-25) | 76.8% | 47.4% |

- 控除率80%水準を**下回る**（この機械固定戦略は単体では非収益、calibration の代理指標）。v5 と±0.3pt 程度で同等。

### 7.5 Cowork 実弾実績（`data/cowork_results.json`、2026-05-26再生成）

- 全体: **445 bets / 387 races / 投資¥1,510,000 / 払戻¥1,145,937 / ROI 75.9% / hit率 22.5% / 見送り236**
- 馬券種別:
  | 馬券種 | bets | ROI | hit率 |
  |---|---:|---:|---:|
  | 単勝 | 62 | **96.3%** | 16.1% |
  | ワイド | 157 | **78.4%** | 25.5% |
  | 複勝 | 68 | 68.4% | 51.5% |
  | 馬連 | 158 | 71.5% | 9.5% |
- 週次ROI: 4/19 84.3% / 4/26 44.2% / 5/03 55.3% / 5/10 116.7% / 5/17 70.3% / 5/24 69.2%
- 傾向: **単勝・ワイドが相対的に強く、複勝・馬連が弱い**（控除率80%に対し総合やや負け）。CLAUDE.md の292bets/78.0%は古いスナップショット。
> 参照: `data/cowork_results.json:3-52`

### 7.6 レガシーアンサンブル AUC

- valid AUC=0.7767 / test AUC=0.7809（`models/ensemble_weights.json`）

### 7.7 予測精度の天井（メモ）

- ◎ top3 ≈ 62% は model/新信号で破りにくい「天井」。伸びしろは betting/market 層、という整理がユーザーメモにある。
> 参照: ユーザーメモ `project_prediction_accuracy_ceiling.md`

---

## 8. 直近の課題と TODO

### 8.1 コード内 TODO/FIXME
- 明示的な `TODO`/`FIXME` マーカーは**ほぼ無い**（grep 結果は track condition の「暫定」表記と循環解消コメントが大半）。課題管理は `CLAUDE.md` と `docs/hypothesis_registry.md` に集約されている。
> 参照: 全 .py grep

### 8.2 CLAUDE.md「既知の問題」
- **P1**:
  1. モデル2系統並走（v6 marks stack と旧8モデルアンサンブル両方を週次生成）。v6統一が方針。
  2. `models/` 肥大化（41ファイル）。
  3. ルート Python 93本、`backtest_*`/`train_*` の重複版残存。
  4. `wide_kekka.csv` 週次更新が手動。
- **P2**: strategy_weights の位置づけ曖昧、`catboost_info/` の .gitignore、docs重複、v6効果検証（2-4週運用後の比較）。
- `.git` 14GB肥大（重複pack、auto-gc無効化で停止回避中、完全圧縮は fresh clone 要）。
> 参照: CLAUDE.md「既知の問題」, ユーザーメモ `project_git_bloat_autogc.md`

### 8.3 未解決の設計課題（本調査で確認・補強）
1. **v6 採否が文書間で矛盾**: `audit_v6_vs_v5_20260520.md` は「採用見送り」（高EV単勝ROI基準未達）、CLAUDE.md は「本番」。さらに v7/v8/v9 の Optuna 成果物が存在（`reports/optuna_v7-v9_marks.json`）→ 本番バージョンの確定と文書整合が必要。
2. **strategy_weights 循環の完全解消が未完**: walk-forward 版コードは存在するが、デプロイ済み JSON には valid→test 乖離の大きい少サンプルエントリが残る。再設計トリガー（2026実績 n≥200）待ち。
3. **コース相性(affinity)の in-sample leak**: train集計に自レースが入る。as-of/OOF 化が必要（ユーザーメモに記録）。
4. **ライブ意思決定で PL厳密joint が未活用**: bundle に p_win のみ → 馬連は単勝確率の近似で再計算。PLペア確率を bundle に載せれば精度向上の余地。
5. **CLV 未計測 / Benter二段未実装**: 市場効率の取り込みが弱い。穴(longshot)予測が構造的に苦手（◎大穴帯勝率0%）というメモもあり、market 層の補強余地。
6. **cowork_results.json の「壊れている」記述は古い**: 既に再生成済み。CLAUDE.md の更新が必要。
> 参照: `reports/audit_v6_vs_v5_20260520.md`, `docs/hypothesis_registry.md:316-361`, ユーザーメモ `project_v8_affinity_insample_leak.md` `project_longshot_weakness.md`, `data/cowork_results.json`

---

## 付録: 主要ファイル早見表

| 役割 | ファイル |
|---|---|
| 本番 export | `export_weekly_marks.py` → `export_marks_json.py: export_race()` |
| PL確率エンジン | `pl_probs.py` |
| v6学習 | `optuna_v6_marks.py`（旧: `train_unified_rank.py`） |
| calibrator | `build_pl_calibrators.py` |
| 配当カーブ | `build_payout_curve.py` |
| EVバックテスト | `backtest_pl_ev.py` / `backtest_fixed.py` / `backtest_pl_kelly.py` |
| 特徴量(race相対) | `race_relative_feats.py` |
| マスター生成 | `build_master_v2.py` / `build_dataset.py` |
| コース相性 | `build_course_affinity.py` / `course_affinity_feats.py` |
| 意思決定プロンプト | `docs/cowork_prompt.md` |
| 仮説台帳 | `docs/hypothesis_registry.md` |
| 週次運用 | `weekly_nicegui.ps1` |
| レガシー予測 | `predict_weekly.py` / `models/ensemble_weights.json` |
| 二段目(ROI) | `train_value_model.py`（value_model_v2） |
| モデル本体 | `models/unified_rank_v6.pkl` / `pl_calibrators_v6.pkl` / `data/pl_payout_curve_v6.pkl` |

*作成: 2026-06-01。数値は上記ソースの実測。`不明`/`[推定]` タグ箇所は追加調査または bundle/pkl の直接確認を推奨。*
