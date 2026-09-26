# EXP18 Stage 0 — 先行研究・等価性監査（S0-A）

**凍結仕様**: v0.4-frozen-stage0（commit `67b874ec`）
**判定**: 停止規律 1（既存研究と同値）には**該当しない**。T0/T1 頭対頭は既実施（anchor 専用）、
較正済み馬連市場への offset 残差と T2 soft 複勝制約は、同値な先行実施が見つからなかった。

---

## 1. 監査の問い

EXP18 の新規部分は次の 2 点に限る（SPEC §4.4、§S0-A）。

- **offset 残差**: 年 Y ごとに ≤Y−1 で温度を自由に fit した terminal 馬連市場 `q_temp ∝ m_cal^a0` に対し、
  非対象プール（単勝・複勝）由来の確率を別係数で足した `q_cross ∝ m_cal^a · q_LOPO^β` が、
  realized 順不同 top2 の race-level logloss を改善するかを入れ子の条件付き logit で検定する。
- **T2**: 単勝 marginal を等式制約、複勝から逆算した τ を soft 制約にした MaxEnt 分布。

「T0/T1 で馬連市場を置き換えられるか」という頭対頭比較は新規性に数えない。

## 2. 対照表

| 先行研究 | 何をしたか | 対象・時点・損失 | EXP18 との関係 | 同値か |
|---|---|---|---|---|
| `analysis/crux_joint.py` | 9 時単勝 de-vig → Harville(λ) の馬連確率 vs 9 時馬連 de-vig の**頭対頭** logloss。EV ゲート ROI・CLV | 馬連 realized top2、λ ≤2023 fit、2024–25 評価。**市場 3.343 < Harville 3.380** | **T0/T1 頭対頭の既実施結果**。EXP18 では anchor としてだけ使い、再実施しない。温度 null も入れ子の offset も無い | T0/T1 のみ同値（既知として扱う） |
| `analysis/build_joint_substrate.py` / `joint_lib.py` | 9 時単勝 marginal と 9 時・確定馬連の per-race / per-pair 基盤の生成 | 基盤のみ（検定なし） | EXP18 は同じ TANPUK/UMAREN を**独自 loader で読み直す**（結果 loader と構造 loader を分離するため）。基盤の parquet は使わない | 非該当 |
| `analysis/crux_fuku.py` / `_joint_m1`〜`_m5` | 単勝 PL の P(top3) vs 市場複勝 implied（Hausch–Ziemba / Dr.Z）。楔 grid、ダッチング、π 較正、本命解剖、独立再現 | 対象は**複勝**（realized top3）。9 時。EV・ROI・頭対頭 logloss | 向きが逆（単勝 → 複勝の頭対頭）。馬連を対象にした入れ子残差ではない。複勝を**入力**側で使う T2 とも別物 | 非同値 |
| `lab/bet_type_lab/crosspool_umaren.py` | 確定単勝 Harville × 確定馬連オッズの EV>τ で馬連を買う ROI | 馬連、確定オッズで判定と決済（自称 oracle 上限版） | proper score の残差検定ではなく EV 選別・ROI。温度 null・入れ子なし | 非同値 |
| `lab/experiments/distortion_exp.py` | 単勝 PL の複勝率と複勝市場の乖離が大きい race を外れ値として賭ける | 複勝・単勝、9 時、分位別 ROI | EV / ROI の外れ値選別。proper score 残差ではない | 非同値 |
| `stage1_benter_blend.py`（Benter 二段） | `softmax(α·log f + β·log π)` を MLE fit | 単勝。モデル f と市場 π | 形は EXP18 の `m^a · q^β` と同型だが、**source はモデル**、対象は**単勝**。非対象プールを source にしていない | 非同値（数式形のみ類似） |
| `pl_stern.py` / `analysis/fit_harville_lambda.py` / `lab/experiments/task_a_stern_fit.py` | λ 割引 Harville（Stern / Lo–Bacon-Shone）の λ fit。モデル確率の較正 | モデル確率の連系較正。市場プール間の残差ではない | T1 の関数形の出典。EXP18 の T1 は**単勝市場**に同じ形を当てるだけ | 非同値（T1 の形の出典） |
| `docs/plans/plan_exotics_real_ev.md` | 実市場オッズでの馬連・ワイド EV 検定の計画（未着手と記載） | EV 選別・ROI | EV 選別の計画。EXP18 は ROI・候補券を扱わない | 非同値 |
| deep bet search（`analysis/deep_bet_search.py`、`reports/deep_bet_search/REPORT.md`） | 8 券種 × 買い方 × 点数 × 条件 × 金額 44,928 セルの総当たり | 本番 v6 の確率 × 実払戻、ROI、3 分割 + FDR | モデル確率の馬券ポリシー探索。市場プール間の整合は検定していない | 非同値 |
| EXP07（`analysis/mcond/exp07_robust_portfolio_dev/`） | 同じ候補馬券に対する配分（P6 ロバスト CVaR vs P1 均等） | 単勝・複勝、配分、ROI | 配分方法の比較。確率の残差ではない | 非同値 |
| EXP16A（`analysis/mcond/exp16a_close_market_residual_dev/`） | terminal 単勝市場に対する**表特徴**の残差情報（条件付き logit offset、温度統制） | 対象**単勝**、source は表特徴、2019–2023 rolling | 検定の枠組み（offset・温度統制・暦日 cluster・3 段 Gate）は同系。source（非対象**市場プール**）と対象（**馬連**）が異なる | 非同値（統計枠組みのみ共有） |
| 学術（Harville 1973、Stern 1990、Lo–Bacon-Shone 1994、Hausch–Ziemba–Rubinstein 1981） | Harville の偏り、λ 割引、place / show の Dr.Z 型非効率 | 連系確率の近似と複勝の裁定 | T0/T1 の形と複勝の非効率の出典。較正済み馬連市場への入れ子 offset 残差は扱っていない | 非同値 |

## 3. 判定

1. **T0/T1 頭対頭**は `crux_joint.py` で 9 時・2024–25 評価まで実施済み（市場優位 Δ ≈ 0.037 nats）。
   EXP18 はこれを新規研究として再実施せず、≤2018 の anchor 再現（§4）にだけ使う。
2. **入れ子の offset 残差**（温度を自由に fit した terminal 馬連市場に対する、非対象プール由来の別係数）は、
   上記のどの先行研究にも同値な実施が無い。Benter 二段は同じ数式の形だが source と対象が異なり、EXP16A は同じ
   統計枠組みだが source と対象が異なる。
3. **T2 soft 複勝制約**は先行実施なし。ただし Stage 0 の複勝往復 Gate が FAIL したため、T2 は未実装で停止した
   （`TOMOGRAPHY_IDENTIFIABILITY_AUDIT.md` §6）。
4. 以上により停止規律 1 は発動しない。単なる EV 選別・配分・PL joint の再実行でもない。

### 事前の期待（記録）

`crux_joint.py` の頭対頭で馬連市場は Harville を上回っており、市場内部裁定の記録（複勝の Dr.Z 型非効率は
実在するが控除に届かない）とも合わせると、較正済み馬連市場に対して非対象プールが残差情報を持つ事前確率は低い。
これは Stage 1 の Gate を変える理由ではなく、結果解釈のための記録である。頭対頭で負けることは、入れ子の
offset 係数 β が 0 であることを意味しない（市場が source より良くても、source が市場と独立な情報を少し
持つことはありうる）ので、offset 残差は同値ではない。

## 4. ≤2018 anchor（9 時 snapshot、結果 loader ≤2018）

`anchor_le2018.py` → `out/anchor_le2018.json`。9 時 = 区分 1・レース当日・09:00 に最も近い 1 本（crux_joint と同じ定義）。
λ は 2013–2016 で格子 fit（0.5〜1.2、0.1 刻み）、評価は 2017–2018。Δ = LL(Harville) − LL(市場)（nats / race、正 = 市場が良い）。

| 項目 | 値 |
|---|---|
| λ*（2013–2016 fit） | 1.1 |
| fit race / 評価 race | 12,389 / 6,289 |
| 評価 LL: Harville(λ*) / 市場 de-vig | 3.4280 / 3.3864 |
| **Δ** | **+0.0416**（市場優位） |
| 要求 | Δ > 0 かつ \|Δ\| ∈ [0.0185, 0.074]（0.037 の 0.5〜2 倍） |
| **判定** | **PASS**（方向一致・同桁） |

anchor 母集団の除外（≤2018）: DNF 871 race、障害 763、9 時馬連格子不完全 305、1着・2着同着 61、
top2 が 9 時 starter 外 27、9 時に両プール無し 17。DNF は「terminal starter（確定単勝 > 1.0）なのに finisher 行が
無い馬がいる race」（EXP16A と同定義、`loaders.dnf_horses`）。

2019〜2023 の着順・払戻・realized top2 はこの監査で一度も読んでいない。

### DNF 修正前後の anchor（v0.5 で追記）

| | 修正前（DNF を着順 NaN で判定 = 常に 0） | 修正後（DNF = starter − finisher） |
|---|---|---|
| λ* | 1.1 | 1.1 |
| fit / 評価 race | 12,955 / 6,569 | 12,389 / 6,289 |
| 評価 LL: Harville / 市場 | 3.43807 / 3.39522 | 3.42805 / 3.38641 |
| Δ | +0.04285（PASS） | +0.04164（PASS） |
| 出典 | Stage 0 セッションの実行ログ（transcript `39ff8c26-…jsonl` のツール出力）。commit されていない | `out/anchor_le2018.json`（commit `3c1deb28`） |

修正前の dry-run γ・λ、anchor の funnel 件数、停止した修正前の検出力計算は**未保存**。推定・再現はしない。

## 5. 大会 1 位コード（keiba-masters-kit、v0.5 で追記）

`https://github.com/sol12378/keiba-masters-kit`、commit `b942a4d3442b65c5d244a672815c9a77e657bff6`、Apache-2.0。

| 事実 | 内容 | 確認 |
|---|---|---|
| モデル | 三連単市場確率 ＋ 単勝 Harville 順序確率の対数線形モデル | README で確認 |
| 係数 | `[0.9265476143601139, 0.0]`（Harville の重みは下限 0） | submission README で確認（再現値 0.9265475951474501、差 1.92e-8） |
| 分割 | train 155 race / 時系列 holdout 46 race | 原典 `submission/phase2/model.json` に `training_races: 155`、`holdout_races: 46` と明記されている。2026-09-26 に原典確認済み（https://github.com/sol12378/keiba-masters-kit/blob/b942a4d3442b65c5d244a672815c9a77e657bff6/submission/phase2/model.json） |
| holdout NLL | 5.8987667 → 5.8957164 | submission README で 5.8988 → 5.8957 を確認 |
| 実運用 | 市場確率へ 5% blend | README で確認 |
| 成績 | 204 race で 4 的中、公式最終残高 5,504,260pt | submission README で確認 |
| 作者の注記 | 較正・positive edge の証拠ではなく、運の寄与が大きい | README で確認 |
| 学習済みモデルの使用 | 最終日の 14 race だけ | README で確認 |

**位置づけ**: EXP18 と同値ではない（別プール・小標本・比例 de-vig・rolling なし・placebo なし・terminal 評価なし）。
構造上は、別プールで行われた UB1 相当の外部小標本例である。新しい arm は追加しない。Stage 1 では UB1 の β を参考値として報告する。
Harville 係数 0 は UB2 の停止理由にしない。市場の温度 null を置く設計を弱く支持する先行例として扱う。大会用の到達確率最大化方策は
PyCaLiAI の長期成長目的へ移植しない。
