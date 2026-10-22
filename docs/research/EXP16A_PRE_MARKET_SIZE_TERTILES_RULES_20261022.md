# EXP16A 副腕 `pre_market_size_tertiles` — 補足規則（結果開封前に固定）

- 作成: 2026-10-22（結果 = winner / 着順 / Q1 の分位別評価を**開封する前**に書いた）
- 親: EXP16A spec v0.4（凍結 commit `e7603678`）。新しい EXP 番号は立てない（候補台帳 L-3）
- 本ファイルは spec v0.4 を書き換えない。spec に無い部分だけを補う

## 1. spec v0.4 にある定義（そのまま従う）

- `subsets.learned_boundaries_per_eval_year.items` に `pre_market_size_tertiles` が登録されている
- 境界は評価年 Y ごとに **Y-1 以前のみ**から作る（`retrospective_rolling_crossfit_development.learned_subset_boundaries`）
- `subsets.rule`: 評価年を見て best 分位を選ばない。**全体評価を主**とし、サブセットだけの改善を全体改善として扱わない
- 母集団（正式 race set）・評価期間（2019-2023 crossfit）・5 seed・median seed 主判定・年層化 meeting-day bootstrap 3,000 回・
  実務床 0.005 nats/race・等級（`gate_grade.grade_gate`）・2024/2025 封印は spec v0.4 のとおり

spec に**無い**もの: 票数の定義、分位ごとの判定規則、多重比較の扱い。以下で補う。

## 2. 票数の定義（label-free 段で実装・検証済み）

- 票数 = `data/Time _series_odds/TANPUK_*.csv` の `単勝票数`（race 単位の合計）を、
  **EXP16A の historical_pre_snapshot と同じ行**（区分1・当日・最後の確定行の 15 分以上前の最後の 1 本）から取る
- 同一行であることの証明: その行から作った de-vig が `official_races_all.npz` の `pi_pre` と 2016-2023 の正式 race 全 25,501R で一致（最大誤差 0.0）。評価年 2019-2023 は 15,951R
- 境界: 評価年 Y ごとに 2016..Y-1 の正式 race の pre 票数の 1/3・2/3 分位。`x <= t1` → T1（小）、`x <= t2` → T2、それ以外 T3（大）。
  favorite_pre_odds_tertiles と同じ「2016..Y-1 を pooled」の作り方
- 票数の生値・境界の生値は公開物に書かない（`out/pre_market_size_labelfree.json` には sha256 と 2016 比の年中央値比だけ）

label-free の実測（結果を読まずに確認）:

| 年 | T1 小 | T2 中 | T3 大 |
|---|---|---|---|
| 2019 | 670R / 200 日 | 1,241R / 286 日 | 1,274R / 287 日 |
| 2020 | 579R / 182 日 | 1,157R / 286 日 | 1,433R / 288 日 |
| 2021 | 564R / 186 日 | 1,218R / 286 日 | 1,418R / 287 日 |
| 2022 | 529R / 182 日 | 1,289R / 287 日 | 1,388R / 284 日 |
| 2023 | 578R / 185 日 | 1,260R / 288 日 | 1,353R / 285 日 |
| 計 | 2,920R | 6,165R | 6,866R |

票数は年々増えている（年中央値の 2016 比: 2019 1.27 → 2023 1.50）。spec の「Y-1 以前のみ」規則に従うと
評価年のレースは大きい分位へ偏る（T1 は各年 17〜21%）。**これは規則の帰結として受け入れ、境界の作り方は変えない**。
全年・全分位にレースがあり、年方向条件（5 年中 4 年）は全分位で計算できる。

## 3. 主判定（等級を付けるもの）

**比較**: 各分位 k で Δ_k = LL(Q2a) − LL(Q0)（Gate A と同じ「締切市場残差」、race-level categorical logloss、改善は負）。

- Q2a は **EXP16A 本体と同一の係数**（評価年 Y ごとに 2016..Y-1 の全正式 race で fit、seed ごと）。分位ごとに再 fit しない。
  分位は評価の切り口であり、arm を増やさない
- 本体の再現を先に assert する: 全体の seed 別 pooled Δ(Q2a−Q0) が `out/eval_2019_2023.json` と 1e-9 以内で一致しなければ停止

**等級**: `gate_grade.grade_gate("A", ...)` をそのまま使う（手計算で上書きしない）。各分位で
- median seed = 分位内 pooled Δ の 5 seed 中央値の seed（本体の `delta_block` と同じ選び方）
- CI95 = その分位のレースだけで、年層化 meeting-day bootstrap 3,000 回（seed 20260924）
- 年方向 = median seed の年別 Δ_k < 0 の年数（要 4/5）、seed 方向 = seed 別 pooled Δ_k < 0 の数（要 4/5）
- leave-one-year-out = 各年を除いた 5 通りすべてで CI95 上限 < 0
- placebo = 本体と同じ構成（log Q1 を「芝ダ × 頭数帯 × pre 人気オッズ三分位」セル内で馬単位に置換、20 draw、
  rng 20260925、係数は置換後に 2016..Y-1 で再 fit）。per-race Δ を分位 k に切り出し、
  real Δ_k が placebo Δ_k の 2.5 パーセンタイルより小さければ「超過」
- 実務床 0.005 nats/race、PASS-PRACTICAL / PASS-SIGNAL / FAIL の定義は spec v0.4 のまま

**多重比較（3 分位）**: 分位等級に加えて、Bonferroni 調整 CI（両側 1 − 0.05/3 = 98.33%、同じ bootstrap 分布）を報告する。

**副腕としての結論（3 段）**:

| 結論 | 条件 |
|---|---|
| OPEN-PRACTICAL | ある分位が `grade_gate` で PASS-PRACTICAL、かつその分位の Bonferroni CI 上限 < −0.005 |
| OPEN-SIGNAL | 上を満たさず、ある分位が PASS-SIGNAL 以上、かつその分位の Bonferroni CI 上限 < 0 |
| CLOSED | それ以外 |

- どの結論でも **EXP16A の Gate A（全体 FAIL）は変わらない**（spec `subsets.rule`）
- OPEN-PRACTICAL でも economic checks・候補生成・ROI 評価・production 変更へ進まない。できるのは「別途レビューを経た新規事前登録の提案」まで
- CLOSED なら L-3 は閉鎖。境界の作り方・分位数（三分位 → 五分位など）・指標・期間を変えた再実行はしない（救済禁止）

## 4. 記述のみ（等級を付けない・結論を変えない）

各分位で次を CI 付きで報告する（median seed は各比較の分位内で同じ規則で選ぶ）:

1. Q2b − Q2a（追加 8 残差特徴、close 基準）
2. Q3a − PRE、Q3b − Q3a（pre 基準）
3. Q1 − Q0、PRE − Q0（pre→close gap）
4. 分位間の差 Δ_T1 − Δ_T3（Q2a − Q0、median seed は全体の median seed、年層化 day bootstrap を全体レースで回し両分位を同時に再標本化）
5. **分位内再 fit 版 Q2a**（係数を 2016..Y-1 の同じ分位のレースだけで fit。境界は評価年 Y のもの）の Δ と CI。
   「票数で条件付けた較正」の素朴版としての参考値。等級を付けず、主判定の代わりにも使わない

## 5. 検出力の目安（結果前の記録、simulation ではない）

本体 tier2 では R ≈ 16k で真の効果 0.005 のとき median CI95 ≈ [−0.0064, −0.0033]（半幅 ≈ 0.0015）。
CI 半幅が √R に反比例すると置くと、T1（2,920R）で約 0.0035、T2（6,165R）で約 0.0024、T3（6,866R）で約 0.0023。
したがって T1 で PASS-PRACTICAL を得るには真の効果が概ね 0.0085 以上必要で、T1 の FAIL / SIGNAL は
「0.005 の効果が無い」ことの強い証拠にはならない。この限界は結論文に明記する。

## 6. 禁止事項

2024/2025 の開封、閾値・境界・分位数・指標の事後変更、結果を見てからの分位選択、再実行、本番ファイルへの接触、commit / push。

## 7. 実装

- label-free: `analysis/mcond/exp16a_close_market_residual_dev/pre_market_size_labelfree.py` → `out/pre_market_size_labelfree.json`
- 評価: `analysis/mcond/exp16a_close_market_residual_dev/pre_market_size_tertiles.py` → `out/pre_market_size_tertiles.json`、`PRE_MARKET_SIZE_TERTILES_REPORT.md`
- 評価スクリプトは本ファイルの sha256 を結果 JSON に記録する
