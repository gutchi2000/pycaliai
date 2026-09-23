# 再開メモ（2026-09-23 23:xx 時点 / セッション中断）

ユーザー指示 A〜F のうち、**A/C の実測が走行中、B/D の一部が未完、E は対応不要、F は未着手**。
以下だけ読めば再開できる。

---

## 1. 完了済み

- **D-1（文言修正）適用済み**: `REPORT.md` §4 から「残差は v6 の race 重みと seed ノイズに帰属すると見られる」を削除し、
  「今回の事前固定 ablation では、EXP15 で観測した NDCG 差を単一列へ帰属できなかった。差の原因は未解決であり、
  結果後に追加探索しない。」へ置換。
- **E（再エクスポート）**: 不要と確定。`馬主(レース時)` の再 export は time-safe owner feature を作る段階まで延期。
  REPORT.md §8 の「再 export の要望」は**削除または「当面不要」と書き換える**こと（未対応）。
- **EXP16 用の先行調査 2 件**は完了済み（結果は §4・§5 に保存。再実行不要）。

## 2. 走行中（バックグラウンド）

`python -m analysis.mcond.owner_id_time_safety_audit.serve_coverage`
- ログ: `data/_research/mcond/exp15/logs/serve_coverage.log`（`^\[2026` の行数が進捗。全 80 日）
- 出力: `out/serve_coverage.json` / `out/shadow_s0s1s2.json` / `out/serve_coverage_table.md`
- 中断していたら**そのまま再実行**（冪等、production には書き込まない。1 日あたり約 45 秒）

### 途中までに確定した実測（重要）

| 事実 | 値 |
|---|---|
| bunseki **無し**の週（2026-01〜02 の 17 日分） | owner 実効既知 **0.0%**、**S0 と S1 のスコアが完全一致**（= 実 serve は馬主 0%） |
| bunseki **有り**の週（例 20260920） | pre-encoder 非欠損 **95.3%**、encoder 既知（`__NaN__` 含む）87.6% → **実効既知 ≈ 87.6%**、約 8% が語彙外で `__NaN__` |
| 馬名 join（production の実キー） | 失敗 15/322 頭（4.7%）。うち 9 頭は表記正規化（`$` 接頭辞・空白）で回復可能。bunseki 内同名重複 0、同名異 ID 0、切詰め候補 0 |
| weekly CSV 側に血統登録番号 | **無い**（bunseki 側にはある）→ production は馬名 join しか選べない |
| encoder の `__NaN__` コード | 102（2,548 クラス中） |
| S2（他週 bunseki からの復元）で ◎ が変わるレース | bunseki 無し週で 0〜3 R / 11〜35 R |

**注意**: S2 は counterfactual。bunseki 無しの過去日に対して 2026-09 時点のスナップショットを当てるため、
S2 自体が time-unsafe（将来情報）である。性能改善とは呼ばない、と REPORT に明記すること。

## 3. 残作業（この順で）

1. 実測完了を確認し、`out/serve_coverage_table.md` を読む。
2. **B（production 影響の表現訂正）**を REPORT.md に反映。実測は「日によって変わる」ケースなので、
   ユーザー指定の後者の文面を使う:
   - future-overwritten/current owner が**一部だけ live へ届く**
   - encoder unknown も混在する
   - **日ごとに feature contract が変わる**
   - より複雑な serve skew
   加えて、bunseki 無し週については「owner 値は live に届かず、モデルは owner 分岐を学習済みで、
   live は常に missing branch に入る（影響量は未測定）」も併記する。
3. **D-2**: REPORT.md §1 の「serve 被覆率 = `serve_feature_baseline.json` で 0.0」は**誤り**（bunseki 配線前の
   古い baseline）。実測値へ差し替え、「offline の学習・評価に限定」という表現を削除する。
   §5 の「本番の予測時にはこの列の値は届いていない」も同様に訂正。
4. メモリ `project_owner_column_future_overwrite.md` の serve 記述（被覆 0% と書いている箇所）も訂正。
5. A〜D をコミット（`audit(P1): owner serve coverage 実測と production 影響の訂正` 等）。
6. **F: EXP16 Stage 0** 着手。成果物は `analysis/mcond/exp16_ticket_candidates_dev/` に
   `PRIOR_ART_AUDIT.md` / `DATA_AVAILABILITY_AUDIT.md` / `MINIMAL_FALSIFICATION_PLAN.md` / `spec_draft.json`。
   実 API・学習・2024/2025 新規評価・ROI・production 変更は**しない**。

---

## 4. EXP16 先行研究監査の素材（調査済み、再実行不要）

### 4.1 current topdown の候補生成（`compute_bets.py`）
- `compute_race_bets()` :364。候補生成の前に P0 障害 gate :385-405、`MIN_BET=500` :406、T-10 オッズ fail-safe :416-452、
  §0 `hard_skip_reasons()` :474-479、percentile fail-safe :482-489、§0b clean-band :495-504。
  `CLEAN_BAND_MAX` は `load_policy()["chaos_reference"]["skip_percentile"]`（:159）。**旧 p33 clean-band は 2026 as-served で符号反転**
  とコメント :156-158。`DEMOTE_BUDGET=2000` :165 は**未配線**（:「★配線中止 2026-07-23 … clean 77.6% < 帯外 87.9%」）。
- **候補は 4 スロット固定**（:512-557）:
  - 複勝: `p_sho` top-1（`fc[0]`）、`fuku_odds_low/high` 必須 :533-538
  - ワイド: `wc[:2]`（λ補正 PL の `p_wide` 上位 2 ペア）、`o <= 50` :539-543、オッズは live wide 中点、無ければ `umaren/3` :521-525
  - 馬連: `p_umaren` top-1 ペア、`_umo(*k) <= 50` :544-548
  - 単勝: `p_win` top-1（`tansho_odds <= 30`）:549-553
  → **最大 4 点。三連複・三連単・馬単は topdown では生成しない**。
- ペア確率 `pl_pair_probs()` :118-149 は λ 補正 PL（`data/harville_lambda.json`、fallback (0.8405, 0.7542) :114）で**未較正**。
- 配分（生成とは別）: 事前 trim :560-562 → `allocate()` p 比例 :563 → **適応トリガミ床** :564-568、`MIN_BET=500/MAX_BET=7000` :241-269。
- 旧 `shape` エンジン :581-858: 印スロット由来の候補、`ODDS_CAP={"馬連":50,"ワイド":50,"馬単":200}` :611、
  `ANA_TAN_ODDS_CAP=30` :734、妙味 overlay :731-743（**vb-◎ペアは 2026-08-09 撤去**: 妙味絡みワイド ROI 60.9% n=575）、
  ◎単勝は妙味時のみ :699（妙味◎単勝 91.6% n=163 vs 非妙味 14.1% n=31）、相手信頼 `AITE_WEAK_TH=0.252` :199/:766-775、
  `cap = min(5, budget//MIN_BET)` :793（6→5: 6 点目以降 ROI 61.9%）、`CB_PLACE_CAP=3000` :839-841。

### 4.2 EXP05 / EXP07
- EXP05 は**確率モデル実験**で候補生成ではない。経済アームは複勝 1 点固定（prob-first）。
  REPORT.md:30-37「M4=86.35% [85.08,87.72] > M1=85.28% … 統計的に明確な差ではない」「EV 閾値選抜(R1)は prob-first(R2) より一貫して悪い」。
- EXP07 は候補**固定**: `TOP_N=3`（`cal_tansho_p` 上位 3 頭）× {単勝, 複勝} = **6 点固定**、P1..P6 は配分のみ。
  Gate1 FAIL（paired bootstrap +4.876 円 CI95[−17.536,+28.516]、P(改善)=0.6558）。馬連は Gate J1 FAIL（最下位帯 O/E=1.625 > 1.6）。

### 4.3 EV フィルタ（候補選抜としては死亡）
`ev_filter.py`（EV 0.8-1.0 → ROI 83.2% / 2.5+ → 72.1% の逆転）、`ev_gate.py`（`MIN_EV_*`）、
`backtest_ev_grid.py`（289 セル、有意な黒字 0）、`audit_ev_bin_roi.py`（単勝/複勝/馬連のみ、9時 TANPUK/UMAREN）。
CLAUDE.md:401「EV による銘柄選抜は -13pt の有害手／EV 閾値の再追加＝逆走」。

### 4.4 較正・同時分布・変換
- `build_pl_calibrators.py:95` が **7 券種**（tansho/fukusho/umatan/umaren/wide/sanrentan/sanrenpuku）の isotonic を
  **valid=2023 のみ**で fit（:56, :180）。course 版・serve 版あり。**production は `pl_calibrators_v6_serve.pkl`**。
- serve 適用: `export_marks_json.py:253/:255`（tansho/fukusho）、:350-356（wide/umaren/umatan、**印馬 top5 の 10 ペアのみ**）。
  一方 topdown は全ペアを λ-PL で**再計算（未較正）**→ 候補集合に直結する不整合で、未測定。
- `pl_probs.py` の閉形式: `p_tansho` :34 / `p_umatan` :39 / `p_sanrentan` :47 / `p_umaren` :58 / `p_sanrenpuku` :63 /
  `p_place_at` :73 / `p_fukusho` :100 / `p_wide` :108、ベクトル版 `all_*` :124-154。
- `analysis/fit_harville_lambda.py` は 2026-05-31〜07-11 の 349R のみで λ を fit。

### 4.5 決済・評価
- `canonical_settlement.py:58` `BET_TYPES=("win","place","wide","quinella","exacta","trio","trifecta")`、
  `UNORDERED_TYPES={"wide","quinella","trio"}`、:14-19「確定着順 == 0 は取消フラグとして信頼できない → fail closed」、
  :73-75「通常の TARGET 結果 export にワイド配当列が無い」。
- `generate_results.py:185-235` `_bet_cis()`: Wilson hit CI + **race-cluster bootstrap ROI CI**（n_boot=2000, seed=42）、
  `roi_verdict ∈ {above_takeout(CI lo>80), below_takeout(CI hi<80), inconclusive}`。
- 集中度・DD ツールは 3 実装に散在（`analysis/prospective_topdown_eval.py` の `max_drawdown`/`outlier_dependence`/`top_share`、
  `analysis/profit_diagnosis_20260917.py`、exp07 REPORT §8 の top1/5/10% 表）。**ruin 確率の実装は存在しない**。

### 4.6 「繰り返さない」既決事項（候補生成に関して）
1. EV 閾値による銘柄選抜（3 方向から死亡）
2. 点数を広げる（「点数は的中率を買えるがエッジを作らない」）
3. 三連単（廃止）
4. 較正済み候補としての馬連（Gate J1 FAIL、かつ真の T-10 馬連価格が無い）
5. 妙味馬をペア券に絡める（ROI 60.9%）
6. 三連複フォーメーション形状（¥100 均等で全形状測定済み、最良 ≈82.5%）
7. 固定候補集合上の配分ルール入替（exp07 Gate1 FAIL、4 ルールが 81.4-81.6% に収まる）
8. clean-band 参戦ゲート（符号反転、意図的に未配線）

### 4.7 本当の空白（EXP16 の新規性になりうる）
- **確率・判断時点オッズ・賭け金ルールを固定したまま、候補の「券種構成」だけを動かした実験は一度も無い**。
  topdown の 4 スロット（複勝×1 + ワイド×2 + 馬連×1 + 単勝×1）と定数（`wc[:2]`、`<=50`、`<=30`、p_sho top1）は**未検証の定数**。
- topdown のペア確率が未較正である一方、bundle は印馬 10 ペアだけ較正済み — この不整合の影響は未測定。
- ruin / bankroll survival の指標が無い。集中度ツールは散在。

---

## 5. EXP16 データ可用性（調査済み、再実行不要）

| 券種 | 判断時点オッズ 2016-21 / 22 / 23 | 確定オッズ | 決済 | 較正器 |
|---|---|---|---|---|
| **単勝** | ✅ `data/Time _series_odds/TANPUK_*.csv`（区分1、発走 26-30 分前、中央値 28 分） | ✅ 区分4（発走 +6-9 分） | ✅ `data/kekka_20130105-20251228.csv` `単勝配当` | ✅ v6_serve `tansho` |
| **複勝** | 🟡 Lo/Hi 帯のみ（`{n}複Lo/Hi`） | ✅ 区分4 | ✅ `複勝配当` | ✅ `fukusho` |
| **馬連** | ✅ `UMAREN_*.csv`（**153 ペア全部**、T-30 相当） | ✅ 区分4 / `data/_joint/mkt_umaren.parquet` | ✅ `馬連` | ✅ `umaren` |
| ワイド | ❌（2026-08-29 以降の forward のみ） | ❌ | ✅ `data/wide_payouts_2016-2025.parquet` | ✅ |
| 馬単 | ❌（forward 0B34、2026-） | ❌ | ✅ | ✅ |
| 三連複 | ❌（`reports/sanpuku_odds_audit.json` = PHASE0_FAIL、全組合せオッズが**いかなる時点にも存在しない**） | ❌ | ✅ | ✅ |
| 三連単 | ❌ | ❌ | ✅ | ✅ |

- TANPUK/UMAREN のレース数: 2016-2023 各年 3,452〜3,456（train/sel/dev を完全被覆）。encoding cp932。
- `区分`: 1=途中（前売り、1 レース約 3 回）、4=確定。overround 中央値 1.260、de-vig は比例配分。
- 結果ファイル内のオッズ（`kekka_1986_2025.csv` の `単勝オッズ` など）は**タイムスタンプが無く判断時点に使えない**
  （provenance 監査 §5 が `odds_in_result_file` として警告）。
- **束縛条件**: production 較正器は **2023 全体（valid split）で fit** されている。
  `exp07 CALIBRATION_AUDIT.md:6`「以前の Gate J1 評価は較正器にとって in-sample だった」。
  → EXP16 が 2023 を development に使うなら、**H1(2023-01-05〜06-25) で fit → H2(07-01〜12-28) で評価**へ分けるか、
  in-sample 較正であることを明示する。2022 で独立に fit することは**不可能**（v6 自身が 2022 を学習に使用）。
- 2026 forward: `data/forward_prices/{date}/{rid}_{stage}_*.json.gz`（stage = t10/t20/decision/vote/close/exp05fs_t35、
  最古 2026-08-29）。`decision` には `p_win_model`・`p_sho_model`・`p_win_market_devig`・pair 系まで入っている。

---

## 6. EXP16 の設計メモ（ユーザー指示の要点、spec_draft.json に落とす）

- **中心仮説**: 同じ race probability・同じ判断時点オッズから、**どの券種・どの組合せを候補集合に入れるか**。
  EXP07 の CVaR 配分再実験ではない。
- **正式確率 baseline**: R0-clean 111 列（train 2016-2021 / selection・calibration 2022 / development 2023）。
  2024/2025 は Gate 通過まで開けない。current v6 は production reference に限定。
  **leaky owner 列と raw 前走レース ID は正式確率に使わない**。
- **対象券種**: 単勝・複勝・馬連のみ（データ監査を通ったもの）。ワイド・馬単・三連系は対象外。
- **比較概念**: T0 current topdown / T1 calibrated joint-probability / T2 uncertainty-adjusted lower-bound EV /
  T3 market-residual。Stage 0 では定義と必要データの固定のみ。
- **分離する軸**: ①race selection ②bet-type selection ③combination selection ④stake allocation ⑤no-bet。
  **主仮説は②と③**。配分は固定する。
- **最終目的**: OOS ROI / 年度方向一致 / meeting-day bootstrap / max drawdown / ruin probability /
  profit concentration / top1・top5・top10 race contribution / 控除後 lower-bound EV。
  大会結果は tail-risk 戦略が存在する外部証拠としてのみ引用し、目的関数には使わない。
