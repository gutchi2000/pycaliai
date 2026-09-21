# EXP09 — Conformal/OODによる保証付き見送り 最終報告

**作成日**: 2026-09-21　**状態**: Stage2本実行完了。**Gate1 PASS・Gate2 PASS・
Gate3 FAIL**。中止規律によりGate4以降(経済評価)は実施せず、EXP09を終了する。

## 0. 結論(先出し)

Conformal(APS-derived abstention score)は、同一participation_rateでの単純比較
(Gate2)では6つの比較方式全て・4つのparticipation_rate点全てで一貫してlogloss
が低かった。しかし、モデル自身の確信度(頭数・最大確率・エントロピー)を完全
統制した回帰(Gate3)では、Conformalの係数の95%CIがゼロを跨ぎ、**統制後の
固有情報量は確認できなかった**。これはEXP06のJev risk_probが同じ関門で
FAILした構造と同型であり、本実験の設計時点で最も警戒していたパターンが
実際に再現した。中止規律に従い、統制変数を減らした再検定は行わず、
Gate4(層別安定性の正式検証)・Gate5(経済評価)には進まない。

## 1. Gate0: データ・時点・モデルhash・eligible race集合

| 項目 | 結果 |
|---|---|
| raw PL OOS provenance | PASS(Git LFS hash一致でtrain<=2022を確認済み、DATA_AUDIT.md §9) |
| eligible race数 | 2023=3,453 / 2024=3,447 / 2025=3,449 |
| 除外(同着) | 2023=3件 / 2024=6件 / 2025=6件 |
| 除外(その他: 重複/NaN確率/少頭数/確率和異常/結果欠損) | 全カテゴリ0件(全年) |

## 2. Gate1: Conformal coverageの実測妥当性

| 年 | empirical_coverage_all_eligible_races | nominal_conformal_coverage |
|---|---|---|
| 2024 | 0.9339 | 0.90 |
| 2025 | 0.9304 | 0.90 |

**両年ともnominal coverage(0.90)を上回り、Gate1 = PASS**。non-randomized APS
は理論通りconservative(nominalをやや上回る)側に振れた。

**診断値(保証の対象外、primary participation_rate=0.75時点)**: 参加サブセット
のcoverage=0.9321、見送りサブセットのcoverage=0.9321(ほぼ同一)。これは
APS-derived abstention scoreが「予測集合が真の勝ち馬を含むかどうか」と強く
相関する形では参加/見送りを分けていないことを示唆する(診断値であり保証
ではないため、この一致自体はGate1判定には影響しない)。

## 3. Gate2: 同一participation_rateでの情報指標比較

主判定はparticipation_rate=0.75。全4点(90/75/50/25%)でConformalが6方式
**全て**を上回った(レース単位logloss、Conformal−他方式の差、負=Conformal優位):

| participation_rate | n_select | vs max_prob | vs entropy | vs OOD | vs feature_missing | vs LR_CONTROL | vs current_gate |
|---|---|---|---|---|---|---|---|
| 0.90 | 6,206 | -0.0024 | -0.0166 | -0.1029 | -0.0402 | -0.0070 | -0.0196 |
| **0.75(主判定)** | **5,172** | **-0.0101** | **-0.0268** | **-0.2339** | **-0.1195** | **-0.0187** | **-0.0306** |
| 0.50 | 3,448 | -0.0439 | -0.0640 | -0.4387 | -0.2855 | -0.0602 | -0.0825 |
| 0.25 | 1,724 | -0.0397 | -0.0877 | -0.6754 | -0.3996 | -0.0687 | -0.1076 |

**Gate2 = PASS**(全参加率・全方式で一貫してConformalが優位)。ただし
max-probability・entropyに対する差は小さい(-0.01〜-0.09)のに対し、
OOD/feature_missingに対する差は大きい(-0.10〜-0.68)。この非対称性は
Gate3の結果と整合的である(下記参照)。

## 4. Gate3: full-control後の固有上積み(★最重要関門、FAIL)

Conformal由来スコアを、n_field・max_prob・entropy(モデル自身の確信度)を
統制した回帰に追加項として入れた結果:

```
conformal_coefficient = 0.0256
95%CI = [-0.1121, +0.1717]  ← ゼロを跨ぐ
survives_full_control = False
```

**Gate3 = FAIL**。Gate2で観測された「Conformalが全方式に勝つ」という結果は、
max_prob・entropyという、Conformalの算出根拠そのものでもある2つの基本的な
確信度指標をコントロールすると消える。**Conformalの参加/見送り判断は、実質的に
max_prob/entropyが既に持っている情報の再表現に留まり、それを超える固有の
情報量を持たない**、という結論になる。Gate2でmax_prob/entropyへの優位差が
他方式より一貫して小さかったこと(§3)は、この結論と整合する。

この関門はEXP06のJev risk_probが失格した関門(モデル自身の確信度を統制すると
消える「難しいレースの代理指標に過ぎない見せかけ」)と構造的に同型であり、
`MINIMAL_FALSIFICATION_PLAN.md`で最初から最重要警戒点として設計していた
通りの結果になった。

## 5. Gate4以降: 実施せず(中止規律)

`spec.json` `stop_discipline`の「Gate1〜4のどれかがFAILならROIを実行しない」
「Gate3(完全統制)でFAILしたら、統制変数を減らして再検定しない」に従い、
Gate4(層別安定性の正式検証)・Gate5(経済評価)は実施しない。参考として
年度別logloss(primary participation_rate、簡易確認のみ)を記録する:

| 年 | n | 平均logloss |
|---|---|---|
| 2024 | 2,585 | 1.9418 |
| 2025 | 2,587 | 1.9615 |

年度間で大きな乖離はないが、Gate3が既に決定的にFAILしているため、この
安定性確認はGate4の正式な層別検証(競馬場・人気帯・芝ダート別)としては
実施しない。

## 6. 仕様書の分類(再掲、Stage0監査結果との整合性確認)

Stage0で「厳密なConformalとして未着手」と分類したものを、本実験で実装・
評価した。coverage保証自体(Gate1)は成立した一方、**この保証を「見送り
判断の質を上げる」という目的に使おうとすると、単純な確信度指標(max_prob/
entropy)を超える価値がないことが判明した** — これはStage0で発見した
EXP06 Jevの教訓が、Conformalという理論的に厳密な枠組みでも同様に成立する
ことを示す。「予測集合が真の結果を含む確率を保証できる」ことと「その保証を
使った参加選別が実務的に優れている」ことは別問題である。

## 7. 仕様書§18相当の結論

1. **厳密なConformal coverageを達成できたか** — できた(Gate1 PASS、
   両年ともnominal 0.90を上回る周辺coverage)。
2. **単純方式に対する優位はあったか(Gate2)** — あった(全4参加率・全6方式に
   対し一貫してlogloss優位)。
3. **モデル自身の確信度を統制した後も優位は残ったか(Gate3)** — **残らなかった**
   (係数95%CIがゼロを跨ぐ)。これが本実験の決定的な結論。
4. **年度・競馬場・人気帯・芝ダートで安定したか(Gate4)** — Gate3 FAILのため
   正式検証せず(中止規律)。
5. **利益へ変換できたか(Gate5)** — 未評価(中止規律によりGate1-4不通過のため
   実施せず)。
6. **この仮説を終了するか** — **終了する**。Conformal predictionによる保証付き
   見送りは、coverage保証自体は理論通り機能したが、参加判断の質という実務的
   目的においては、モデル自身が既に持つ基本的な確信度指標(最大予測確率・
   エントロピー)を超える固有の価値を示せなかった。EXP06のJevと同じ「難しい
   レースの代理指標に過ぎない」という構造的な罠が、より厳密な理論的枠組み
   (Conformal)でも再現した。

## 8. 中止規律の遵守

Gate3 FAILという結果を見た後、統制変数(n_field/max_prob/entropy)を減らして
再検定していない。参加率をずらして都合の良い点を探していない(4点全てで
一貫してGate2はPASSしたが、これはGate3の判定を変えない)。Gate5(経済評価)
は実施していない。APS形式・quantile定義・tie-break規則・例外処理カテゴリの
いずれも結果を見た後に変更していない。

関連: [[project_exp09_conformal_abstention]] [[project_mcond_exp06_jev_decision]]
[[project_unwired_roi_audit]]
