# PyCaLiAI 特徴量バックログ（即実装版・確定リスト）

> 作成: 2026-06-24。出典: Claude Code による3段階調査
> （①使用/未使用マップ ②-9999死蔵の真因特定 ③bundle.json/cowork_prompt 配線マッピング）。
> 行番号は調査時点のもの。実装時に該当箇所を `grep` で再確認すること。

## 前提（重要）

- **予測層は天井近傍**: ◎複勝圏率 61-62%。1400特徴ブルートフォースで採用ゼロ（`project_feateng_v7_classmove_dead`）。
  → 新規 raw 予測特徴で精度を伸ばす方向はほぼ死に筋。即効性は **死蔵修復／第二意見（選別層）／安いleak-safe列／Cowork narrative** に寄せる。
- **既に使用済み（候補から除外）**: PCI（前走PCI3/RPCI/前PCI）、前走通過順（前1〜4角→`prev_pos_rel`/`closing_power`）、
  前走上り3F、補正タイム（prev_hosei/prev_hosei9）、坂路/WC調教、父/母父/父タイプ/母父タイプ、ブリンカー、
  kako5統計16個、騎手/調教師 複勝率（fuku30/90）。
- **死蔵確定（-9999）**: 母馬・前走走破タイム・前走着差タイム。真因 = `fillna_overwrite`（下記#4）。

---

## 即効性 Top 9

### 1. ZI乖離フラグ 🟢（最安・今日完結可）
- **取得元**: weekly CSV `ZI` / `ZI順`（`build_site.py` parse_weekly L165-166 で既読、bundleに無いだけ）
- **有用性**: 補正タイムと別系統の指数。`zi_divergence = ai_rank − zi_rank` で「◎なのに指数下位＝妙味or罠」を即可視化。◎の質点検。
  ※補正タイムと相関＝raw予測特徴では priced。価値は解釈フラグ（回収率は中立）。
- **難易度**: 簡単
- **実装ステップ**:
  1. `export_marks_json.py` `export_race()`（L282付近）で per-horse 行 `g.iloc[i]` から `ZI`/`ZI順位` を抽出
     （`predict_weekly.parse_csv` が既に `ZI`→float, `ZI順`→`ZI順位` rename 済み・L185,436周辺）
  2. `horse_record()`（L150-181）に `zi`/`zi_rank` パラメータ追加 → rec に `zi`,`zi_rank`,`zi_divergence`(=`ai_rank-zi_rank`) を格納
  3. `build_site.py` transform_bundle（L699-700）は既に `hext.get("zi")` 抽出済み → `zi_divergence` を bundle 経由で表示に追加
  4. `docs/marks_schema.md` horses[] に追記
- **bundle/Cowork**: `horses[].zi / zi_rank / zi_divergence`。cowork_prompt の `horses[]` 説明に
  「zi_divergence: 負=AIが指数より高評価（妙味/罠を点検）、正=AIが指数より低評価（軽視の妥当性確認）」を追記。

### 2. 乗替（替）＋ 減量（減M）🟢（安い・leak-safe）
- **取得元**: weekly CSV `替`・`減M`（どちらも完全未使用。`make_weekly_hosei.py:42,46` に列定義のみ）
- **有用性**: `替`＝継続騎乗の信頼/乗替リスク。`減M`＝**実効斤量(斤量−減量)** で `斤量体重比` の精度↑（未勝利/条件で効きやすい）。
- **難易度**: 簡単
- **実装ステップ**:
  1. `predict_weekly.py` parse_csv の `HORSE_COLS_xx` で `替`/`減M` を読込（既に列位置にある）→ `pd.to_numeric`(減M) / カテゴリ(替)
  2. `optuna_v6_marks.py` CAT_COLS に `"替"` 追加。`実効斤量 = 斤量 - 減M.fillna(0)` を派生列として feats に追加
  3. `train_unified_rank.py` 側も同様（CAT_COLS + 派生）
  4. serve: `export_weekly_marks.py` の serve rename / 派生計算に `実効斤量` を反映（学習と一致させる）
  5. 再学習（`run_v6_pipeline.py`）
- **bundle/Cowork**: 学習特徴。任意で `horses[].kawari`(継続/乗替) を bundle に出して Cowork narrative（「継続騎乗で手の内」）に。

### 3. 今走マイニング順位 🟢（選別層に効く第二意見）
- **取得元**: **出馬表分析**（TARGET側で「今走マイニング」列を出力設定）。
  ⚠️ weekly内の `マイニング順位` は46列フォーマット38番目＝**前走ブロック**＝前走のもの。今走は別出力が必須。
  発走前公開のJRA-VAN予測＝**leak-safe**。
- **有用性**: PyCaLiAIと別系統AIの評価。◎と一致＝信頼度↑、乖離＝**◎飛び（負けの局在 `project_loss_forensics_842`）を事前警戒**。
  伸びしろのある参戦/選別層に直撃（`project_participation_selectivity_realtool`）。
  ※生の予測特徴にはしない（過去にマイニング学習活用は死蔵）。第二意見＝Cowork/選別用。
- **難易度**: 中（TARGET出力設定＋パーサ列追加）
- **実装ステップ**:
  1. TARGET 出馬表分析テンプレに今走マイニング列を追加 → weekly CSV の列数フォーマット分岐を更新（`predict_weekly.py` HORSE_COLS_xx / `build_site.py` IDX_xx）
  2. `export_marks_json.py` `export_race()` で per-horse `mining_rank`(今走) を抽出 → `horse_record()` に格納
  3. `race_confidence`（L304-310 で構築）に `mining_agree` を追加（◎のmining順位の単調スコア：1位→1.0, ≤3位→0.6, 圏外→0.0 等）
  4. `docs/marks_schema.md` + `docs/cowork_prompt.md` を更新
- **bundle/Cowork**: `horses[].mining_rank` + `race_confidence.mining_agree`。
  cowork_prompt race_confidence 説明に「mining_agree が高い→◎本線を勝負/準勝負へ格上げ、低い→独立筋と割れる→見送り/消化の根拠」。

### 4. 死蔵修復：母馬 / 前走走破タイム・着差 🟢🟡
- **取得元**: master_v2（**生データは健全**: 母馬21,206 distinct・-9999は0件、前走走破タイムは M.SS.T 形式で NaN 9.56%＝初出走のみ）
- **真因（確定）**: `optuna_v6_marks.py:150` の
  `X = d[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999)` 直前で、
  (a) `前走走破タイム`/`前走着差タイム` に `parse_time_str` が未適用（文字列→to_numeric→全NaN→-9999）、
  (b) `母馬` が CAT_COLS 欠落で LabelEncode されず（文字列→全NaN→-9999）。
  同バグが `train_unified_rank.py:127`。`parse_time_str`(utils.py:161-175) 自体は無罪（実データ失敗0件）。
  既に `optuna_v10_marks.py:118-133` に修正(`fix_time_string_cols`)があるが **v10未採用**。
- **有用性**:
  - **母馬 🟢** = 3列で最も**直交的な血統信号**（過去走支配を深めない）。監査docが +0.1〜0.3pt 期待。
  - **前走走破タイム/着差 🟡** = 本来強いが**過去走系→v6 gain60.9%の過去走支配を深める恐れ**（`project_v6_pastform_dominance`）。
- **難易度**: 中（再学習が必要）
- **実装ステップ**:
  1. `optuna_v6_marks.py` CAT_COLS（L74-80）に `"母馬"` 追加。prep内 split後・エンコード前（L117付近）に
     `for c in ["前走走破タイム","前走着差タイム"]: tr[c]=tr[c].map(parse_time_str); vl[c]=...` を挿入。L150の `.fillna(-9999)` は据置（パース後NaN=初出走のみ＝正当な欠損印）。
  2. `train_unified_rank.py` も CAT_COLS（L59-65）に `"母馬"`、L82付近に同 parse ループ。
  3. **serve整合（必須）**: `export_weekly_marks.py` の `to_numeric` 前に同 `parse_time_str` を1回。
     （現状 serve は parse未呼出＝学習で実秒/serveで-9999 になると serve skew 再発 `project_serve_skew_quantified_v6_governance`）
  4. 再構築: `python run_v6_pipeline.py` → `python audit_marks.py --model v6` → `python audit_v6_vs_v5.py`（採用ゲート）
  5. **母馬と時計は別 ablation**。時計は◎飛び率が悪化しないか厳格監視、悪化なら不採用。母馬は高カーディナリティ→未知率を audit で確認、効果薄ければ freq encoding。
- **bundle/Cowork**: 新フィールド不要。修復後 `horses[].why`（SHAP寄与）に「母父系の好相性」「同条件で走破時計最速」等が出て Cowork の判断材料が具体化。

### 5. レース展開予測（決手→逃げ/先行頭数→想定ペース）🟡
- **取得元**: kako5 `決手`（各過去走の脚質、現状 `build_site.py` の脚質チップ`classify_style`でUI表示のみ）
- **有用性**: レース内の逃げ/先行頭数→ハイ/ミドル/スロー想定→「**中人気×展開利**」の妙味抽出（`project_longshot_weakness` と整合）。
  ※個別脚質はpriced（`project_statmech_coupling_3body`）→モデル特徴にしない。
- **難易度**: 中
- **実装ステップ**: `build_site.py` の脚質判定を流用し race 単位で逃げ/先行頭数を集計 → `buy_judgment` か `race_confidence` に `pace_scenario`(逃げ多=ハイ等) を付与。
- **bundle/Cowork**: `race_confidence.pace_scenario` → cowork_prompt で「展開利の差し/前残りの先行」を narrative に。

### 6. 出馬表分析の各種指数（IDM/距離適性/馬場適性/上昇度）🟡
- **取得元**: 出馬表分析CSV（JRA-VAN計算済み指数群）
- **有用性**: ZIと同系の独立指標。複数指数とPyCaLiAI ◎の**合議**で「全筋が◎支持」を確認。
  ※raw特徴では補正タイム/既存特徴と相関＝priced。
- **難易度**: 中（出力設定＋パーサ）
- **実装ステップ**: TARGET出力→parse→`horses[]` に各指数を格納（#1/#3と同じ insertion パターン）。
- **bundle/Cowork**: bundle追加＋Cowork narrative（第二意見の合議）。

### 7. 輸送/遠征フラグ（所属×場所）🟡
- **取得元**: weekly CSV `所属`（栗東/美浦、未使用）× レース `場所`
- **有用性**: 関西馬→関東開催等の輸送ローテ＝状態シグナル、leak-safe。
- **難易度**: 簡単
- **実装ステップ**: `predict_weekly.py` で `所属` 読込→`遠征フラグ = (所属の東西 ≠ 場所の東西)` を派生→CAT/bool特徴。
- **bundle/Cowork**: 学習特徴 + 任意で `horses[].transport` を Cowork narrative。

### 8. kako5の同脚質/同コース成績・脚質トレンド 🟡
- **取得元**: kako5（`決手`×`場所`×`着順`、未集計）
- **有用性**: 「差しに回ると好走」「同コース同脚質で堅実」等、`closing_power` で拾えない離散適性。
- **難易度**: 中（as-of集計）
- **実装ステップ**: kako5パース（`parse_kako5.py`）に脚質別 as-of 集計を追加 → Cowork narrative（学習特徴は慎重に）。
- **bundle/Cowork**: 主に Cowork narrative。

### 9. 前走馬体重・増減トレンド 🟡
- **取得元**: master_v2（前走馬体重/増減は使用済みだが増減トレンドの活用が浅い。今走馬体重は当日情報=リーク不可）
- **有用性**: 馬体増減傾向＝状態の代理。
- **難易度**: 簡単
- **実装ステップ**: 既存の前走馬体重特徴を「直近数走の増減トレンド」に拡張（kako5から）。
- **bundle/Cowork**: 学習特徴。

---

## その他候補（中〜低）

| 特徴量 | 取得元 | 注意 |
|---|---|---|
| 騎手 条件別成績（コース/距離/脚質別） | 出馬表分析 / as-of集計 | ⚠️ v8 affinity が in-sample leak で死亡。as-of/OOF必須でも天井止まり（`project_v8_affinity_insample_leak`） |
| 母父×コース相性 | as-of集計 | v8の地雷、非推奨 |
| ZI印（指数印） | weekly | ZIから導出済み、冗長 |
| 上昇度/調子マーク | 出馬表分析 | narrative補助程度 |
| 1(2)着馬（前走勝ち馬） | weekly | 前走レベル測定の材料 |
| 毛色・馬主・生産者 | weekly/出馬表分析 | 高カーディナリティ＝v8死亡パターン、非推奨 |

## 提案しない（検証済みで死亡・再走防止）

| 死亡項目 | 根拠メモリ |
|---|---|
| 生の予測特徴の量産（1400特徴採用ゼロ） | project_feateng_v7_classmove_dead |
| 血統embedding・affinity（逆効果/leak） | project_v8_affinity_insample_leak |
| 物理/EVT/statmech を raw予測特徴に | project_physics_features_tested / project_evt_extreme_value_tested |
| 当日通過順/Ave-3F/上3F順（当日結果リーク） | torch_csv_builder FORBIDDEN（正） |
| odds_ou等の市場価格を予測特徴に | project_odds_ou_first_gate_pass |
| 馬場/トラックバイアス特徴（~9割priced） | project_baba_cushion_tested |

---

## 推奨着手順（今すぐやるべき順）

1. **今日完結（極小コスト）**: #1 ZI乖離 → bundle配線＋`marks_schema.md`＋cowork_prompt。
2. **安い学習特徴**: #2 替/減M（次の再学習にバンドル）。
3. **選別層の本命**: #3 今走マイニング（TARGET出力設定がボトルネック、並行着手）。← 一番伸びしろのある層。
4. **再学習にバンドル**: #4 母馬（安全に効く）＋前走走破タイム（過去走支配を監視しつつ別ablation）。serve整合を忘れない。
5. **Cowork強化**: #5 展開予測 → #8 脚質適性。

### 検証ゲート（共通）
- 学習特徴を触ったら必ず `audit_marks.py --model v6` で◎top3非劣化、`audit_v6_vs_v5.py` の採用ゲートを通す。
- 第二意見（#1/#3/#6）は前向き実運用ログで CLV/的中を観測（in-sample改善は信じない）。
- 死蔵修復は母馬と時計を**分離して** ablation。◎飛び率の悪化は不採用シグナル。
