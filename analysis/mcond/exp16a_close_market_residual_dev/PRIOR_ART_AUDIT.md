# EXP16A Stage 0 — 先行研究監査（Close-Market Residual Information Bound）

**作成日**: 2026-09-24　**状態**: 監査のみ。学習・2024/2025 開封・ROI 評価・production 変更はしていない。
引用した数値はすべて既存成果物からのもので、本監査で新規に計算したものではない
（新規計算は `provenance.py` の市場素性と 2022 の市場 baseline、`verify_growth.py` の合成例のみ）。

---

## 0. 結論（先に要点）

1. **「表特徴が pre 市場に情報を足すか」は EXP05 が既に答えている**。単勝 win logloss で
   M3 0.20838 → M4 0.20799、top3 では 3 年とも CI が 0 を跨がない（§1）。EXP16A の Q3 は
   **その再実行になりうる**ので、新規性は metric と baseline の置き方で明示的に切り分ける必要がある。
2. **確定オッズを「評価の baseline」にした実験はこのリポジトリに存在しない**（§5）。
   確定オッズは決済・CLV の分母・ORACLE 判定済みの crosspool にしか現れない。**Q0 は本当に空白**。
3. **race-level 多項 logloss を市場確率に対して計算した例も無い**（§6）。既存の 3 実装はすべてモデル対モデル。
4. **市場を確率指標で上回ったものは一つも無い**（§3・§4）。M1（v6+市場）が M0（市場のみ）に勝つ方向は
   何度も観測されているが、それは「市場に足す」であって「市場を超える」ではない。
5. **`-ln(1-takeout)` を閾値として使った実装は存在しない**（§9）。控除の壁は ROI の CI 比較
   （`above_takeout`: CI 下限 > 80%）としてのみ運用されている。
6. **EXP05 の教訓**: 頑健な logloss 改善が、統計的に区別できる ROI 改善を生まなかった（86.35% 対 85.28%、CI 重複、
   いずれも控除の床 80% の内側）。情報の存在とエッジは別物である。

---

## 1. EXP05 market residual（`analysis/mcond/exp05_market_residual_dev/`）

| 項目 | 内容 |
|---|---|
| 市場 | **pre スナップショットのみ**。`market.py:13` に `final : 区分4 (確定)。**特徴量には使わない**` と明記 |
| 主 target | **top3**（`spec.json`）。win は secondary で主判定から除外 |
| 主指標 | **馬単位の二値 logloss** + 開催日ブロック bootstrap 2,000（`evaluate.py:45-47`, `:69-83`） |
| モデル | M0 市場のみ / M1 +v6 PL top3 / M2 +v6 score・rank / M3 +レース内相対4本 / M4 offset(M3)+serve 117列 / M5 +full 145列 / M6 Benter 型 |
| top3 logloss | M0 .41503 / M1 .41303 / M3 .41301 / **M4 .41169**（2023）。2024・2025 も同傾向 |
| win logloss | M1 .20841 → M3 .20838 → **M4 .20799** → M5 .20790、M6 .20851 |
| Gate2（M4 対 M3） | 2023 −0.00132 [−0.00184,−0.00078] / 2024 −0.00138 / 2025 −0.00147。10/10 場、4/4 人気帯、上位100R 除去後も −0.00123 |
| ROI | prob-first 1点: M1 85.28% [83.98,86.54] → M4 86.35% [85.08,87.72]。**100% 超のセルは無し** |
| 結論 | logloss 改善は頑健だが ROI へ変換されない。Gate2-4 は exploratory 扱い |

**EXP16A との関係**: Q3（pre 市場 + R0-clean）は EXP05 M3→M4 と**同じ問いに近い**。違いは
(a) 確率モデルが v6 ではなく **R0-clean 111列**、(b) target が **win を主**に、(c) 指標が **race-level 多項 logloss**、
(d) baseline に **Q0（確定）** が入ること。(a)-(c) だけでは新規性は弱い。**新規性の主張は (d) に置く**。

## 2. mcond ハーネス（再利用できるもの）

- `market.py:99-105`: `pi = (1/odds) / Σ(1/odds)`（比例 de-vig のみ）、`odds<=1.0` は欠損。
- `v6base.py:140-143`: **出走馬で二度目の正規化**をしてから Harville。→ EXP16A では正規化を一度に固定する（§7 罠5）。
- `v6base.py:62-75` `fit_tau`: レース内 softmax の条件付きロジット MLE で温度推定。
- `evaluate.py`: 二値 logloss・10 等幅 bin ECE・Brier・AUC・開催日 bootstrap。**race-level 多項ではない**。
- `exp09/evaluate.py:33-75`: **race-level 多項 logloss + 開催日 paired bootstrap 2,000** が既にある。EXP16A の主指標に最も近い実装。
- `exp07/gate_j1_calibration.py:145-156`: **calibration intercept/slope** と **adaptive-bin ECE**。そのまま使える。

## 3. 市場を条件にした既存実験（すべて pre 市場 + 馬単位二値 logloss）

| 実験 | 表現 | 判定 | 市場を確率で上回ったか |
|---|---|---|---|
| EXP01 | 陣営選択の逸脱 | PASS だが生の選択列で説明でき終了 | いいえ |
| EXP02 | 動的対戦能力 | M3 対 M1 −0.00074 [−0.00116,−0.00032]、効果の3/4は出走回数・休養日数 | いいえ |
| EXP04 | 環境不変な特徴選別 | M1 には勝つが素の M2 に負ける | いいえ |
| EXP06 | LLM 由来 risk_prob | Gate1 PASS / Gate2 FAIL | いいえ |
| EXP08 | 当日馬場状態 Kalman | Δ = −4.00e-08（要求の 1/12,500）、placebo 不合格 | いいえ |
| EXP09 | Conformal abstention | Gate3 FAIL（係数 CI が 0 跨ぎ） | 選抜のみ |
| EXP10 | 大敗リスク head | 2023 CV で FAIL。**市場特徴が result-file odds 由来で素性不明**（要再検証） | いいえ |
| EXP12 | 対戦相手ネットワーク | placebo FAIL | いいえ |

## 4. 「市場に勝つ」直接の試み

| 研究 | スナップショット | baseline | 数値 | 判定 |
|---|---|---|---|---|
| Benter 二段 blend | **9時** de-vig | 市場のみ R²=0.23833 | ΔR²_test **−0.0198**、ROI 70.9% | 控除割れ |
| オッズ軌跡 3成分 | 9時前の軌跡 | 2成分 | γ CI [−0.0097,+0.0269] 非有意、k≥2 は 52.2% のレースのみ | 冗長 |
| odds_ou | 区分1 ≤09:00 | v6 | ΔAUC は上がるが ROI 変わらず | 死亡 |
| `crux_joint.py` | 馬連 9時 | 単勝周辺から Harville | **市場 LL 3.343 < Harville 3.380** | 市場の勝ち |
| Dr.Z 型（複勝プール） | 9時 | 市場の複勝 implied | 非効率は実在（fair 0.4154 < 市場 0.4287）が Kelly dutch ROI 85.5% | PRICED |
| Deep Value Net (T-10) | 9時入力・確定決済 | ◎単勝 | 4 反復とも OOS 下限 > 1.00 に届かず | 死亡 |
| CLV | 9時 対 close | — | 正 CLV 馬 ROI 81.7% 対 負 CLV 81.1% | 換金不能 |
| crosspool | **確定オッズを両脚に使用** | — | ROI 88〜92% | **ORACLE**（賭けられない） |

## 5. 確定オッズを baseline にした例（＝ EXP16A の空白）

**存在しない。** 確定オッズの出現箇所は次の3つだけである。

1. **決済**（`exp_deepvalue_t10.py` の `oc`、`crux_joint.py` の `o_umfin`）
2. **CLV の分母**（`analysis/measure_clv.py`）
3. **ORACLE と判定されて死んだ crosspool**（VOL3 §3.3）

`market.py` は設計として `final` を特徴から除外しており（`:13`）、`v6base.py` が `tan_final_odds` を
運んでいても `build_features.py` は結合していない。

→ **Q0（確定オッズの de-vig を評価 baseline として使う）は未実施**。ただし §7 の罠1・罠2 を厳守すること。

## 6. race-level 多項 logloss

3 実装ある（`exp12/stage1_cv.py:68-79`、`exp09/evaluate.py:33-46`、`exp15/evaluate.py:22`）。
**いずれもモデル対モデル**で、市場確率に対して計算した例は無い。EXP09 の実装が
開催日 paired bootstrap 込みで最も近い。

## 7. EXP16A が踏んではいけない罠（監査で名前がついたもの）

1. **oracle 汚染**: Q0/Q2 は診断専用。確定オッズ由来の量が Q3 の特徴・ハイパラ・サブセット定義に
   一度でも触れたら crosspool と同じ ORACLE になる。**Q0 系と Q3 系を物理的に分離する**。
2. **確定は発走後**: 区分4 は発走 5〜13 分後の記録。「締切価格」ではない（`MARKET_DATA_PROVENANCE.md` §2）。
3. **較正器リーク**: `models/pl_calibrators_v6.pkl` は valid=2023 全体で fit。2023 に当てると較正器にとって in-sample。
   EXP05 は train のみで isotonic を再 fit し、EXP07 は H1(1-6月) fit → H2(7-12月) 評価に分けた。
4. **指標の置き換えを新結果と呼ばない**: race-level 多項 logloss と馬単位二値 logloss は同じ確率の別集約。
   Q2 > Q1 が EXP05 M4 > M3 の言い換えにすぎない可能性を事前に区別する。
5. **二重正規化**: `market.py` と `v6base.py` で 2 回正規化されている。取消集合が pre と確定で違うため、
   EXP16A は正規化を一度に固定し、Q0 と Q3 を同一の馬集合に載せる。
6. **同着の扱い**: 既存実装は勝者が一意でないレースを黙って落とす。除外件数を必ず報告し、アーム間でずらさない。
7. **評価年でのサブセット選択**: clean-band ゲートの前例（VOL3 §3.7）。2024fit→2025eval で選んだ帯が
   2026 as-served で符号反転した。サブセットは 2022 だけで事前固定する。
8. **ROI への飛躍**: EXP05 は頑健な logloss 改善を出しながら ROI で区別できなかった。
   情報の存在を「エッジ」と語らない。
9. **被覆差が静かな選抜になる**: Q0 と Q3 で適格レース集合が違うと別の母集団を比べることになる。共通集合を先に固定する。

## 8. `-ln(1-takeout)` の扱い（§9 の監査結果）

リポジトリ内に **相互情報量・倍加率・Kelly 閾値の実装は無い**。`Kelly` を名乗るコードは
`docs/audit_20260615_full.md:150` により「実態はどこにも未実装（dead code、実走は edge 比例配分）」と記録されている。
控除の壁は `roi_verdict`（CI 下限 > 80% で `above_takeout`）としてのみ運用されている。

EXP16A はこの式を **hard Gate に使わない**。理由と成立条件の導出は
`INFORMATION_TO_GROWTH_DERIVATION.md`（合成例 C1-C5 で確認済み）。

## 9. EXP16A の新規性（主張できる範囲）

1. **確定オッズ de-vig を評価 baseline（Q0）に置くこと**（未実施）。
2. **race-level 多項 logloss を市場確率に対して測ること**（未実施）。
3. **Q0 と Q3 の差を「oracle と実運用可能性の分解」として測ること**
   （pre → 確定の価格移動は記録があるが、情報量として測られたことは無い）。
4. **win を主 target にした mcond 基盤での測定**（EXP05 は top3 が主で win は補助）。

新規性ではないもの: 「表特徴が pre 市場に情報を足す」こと自体（EXP05 が既に示した）、
Benter 型 blend、オッズ軌跡、CLV、crosspool。
