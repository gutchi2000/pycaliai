# EXP07 — 馬券ポートフォリオ・ロバスト最適化 検証

既存の予測確率・候補馬券・判断時点オッズを固定した上で、馬券間の相関・排他的払戻・確率誤差・
確定オッズ変動を考慮するロバストポートフォリオ最適化(CVaR/distributionally robust)が、
現行の資金配分および単純均等配分を上回るか検証する。詳細は`spec.json`（結果を見る前に固定済み）。

## 絶対条件

EXP01-06・EXP05-Fを変更しない。本番の印・買い目・資金配分・`compute_bets.py`へ接続しない。
既存モデルを再学習しない。候補馬券生成と資金配分を最初から同時最適化しない。判断時点より
後のオッズ・結果を最適化入力に使わない。2024〜2025年を完全未使用期間と呼ばない。ROI点推定
だけで採用しない。

## 現在の進捗（2026-09-20夜時点）

| 段階 | 状態 | 成果物 |
|---|---|---|
| Stage 0（既存未コミットコード監査） | **完了** | `PRIOR_ART_AUDIT.md` — 総合判定「再利用可能（3制約明記の上で）」 |
| 有効券種の再監査 | **完了** | `compute_bets.py`のtopdown経路を直読し、現在有効な候補馬券を確定(下記「既知の制約」4参照) |
| データ監査（§5） | **完了** | `DATA_AUDIT.md` — 2024-2025年は真のT-10オッズが存在せず、TANPUKアーカイブの`historical_pre_snapshot`（発走26-30分前、中央値28分前）による「現在コードのreplay」としてのみ評価可能と判明 |
| 共同損益分布 + Gate J0（§6） | **完了、PASS** | `JOINT_DISTRIBUTION_AUDIT.md`, `build_scenarios.py`, `test_build_scenarios.py`（14テスト全通過） — 既存`pl_probs.py`のPL厳密連鎖を土台に1-3着順列状態空間(18頭立て最大4,896状態)を構築、marginal整合性を機械精度で確認 |
| 数理定義 + spec.json（§7） | **完了、凍結** | `spec.json` — 比較政策P0-P7、Gate0-5、CVaR alpha規約の変換規則（**重要**: 仕様書のalpha=0.10[最悪10%平均]と既存コードのcvar_alpha[信頼水準]は逆の慣習なので`policies.py`で変換している）を含む |
| Gate J1（共同分布の実測較正、2023 developmentのみ） | **完了、PASS** | `gate_j1_calibration.py`, `test_gate_j1_calibration.py`（11テスト） — raw PLは低確率帯(穴馬)で的中率を過大評価(tansho最下位帯O/E=0.46、統計的に十分な標本)、既存較正済み確率(`pl_calibrators_v6`)はこれを補正すると実測確認。Stage 2Aへは較正済み確率のみを渡す |
| Solverの全列挙oracleテスト | **完了、全通過** | `test_oracle.py`（8テスト） — 独立に書いた参照実装(solverの内部関数を一切再利用しない)で目的値・同値解・丸め安全性を検証 |
| 不確実性集合の構築 | **完了** | `uncertainty_scenarios.py`, `test_uncertainty_scenarios.py`（7テスト） — PL latent scoreを摂動し共同着順分布をまるごと再計算(独立摂動による確率破綻を構造的に回避)。スコア摂動幅・オッズ縮小分布とも2023年developmentのみで実測・凍結（下記参照） |
| 決済ロジック（evaluate.py） | **完了** | `evaluate.py`, `test_evaluate.py`（16テスト） — 公式確定払戻テーブルをそのまま使うfail-closed設計。同着・複勝レンジは独自計算せず公式値をそのまま通す |
| Stage 1合成データ試験（§8） | **完了、15/15 PASS** | `test_synthetic.py` — 決済ロジック実装により項目15（返還馬券）を有効化、全15項目通過。Stage 2A予算完全消化探索(`full_spend_search`)のテストも追加 |
| Stage 2A（実データ・配分のみ比較） | **未着手** | 次の作業。予算完全消化(`sum(w_j)=B`)を主比較の必須制約とする |
| Stage 2B（選別+配分） | **未着手** | Stage 2A PASS後のみ |
| Gate 0-5 | **Gate 0/J0/J1のみ判定済み(全PASS)、Gate 1-5は未判定** | Stage 2A/2B完了後 |

## ファイル構成

```
analysis/mcond/exp07_robust_portfolio_dev/
├── README.md                    このファイル
├── PRIOR_ART_AUDIT.md            Stage 0: 既存robust_ticket_portfolio.pyの監査
├── DATA_AUDIT.md                 §5: データ監査(必須入力/評価専用/時点安全性/過去再現性/券種スコープ)
├── JOINT_DISTRIBUTION_AUDIT.md   §6: 共同着順分布の監査 + Gate J0結果
├── spec.json                     §7: 数理定義・比較政策・Gate定義(結果を見る前に凍結)
├── build_scenarios.py            共同着順分布(top-3順列状態空間)の構築
├── test_build_scenarios.py       Gate J0関連の単体テスト(14件)
├── gate_j1_calibration.py        Gate J1: 2023 developmentのみで較正を実測
├── test_gate_j1_calibration.py   Gate J1純粋ロジックの単体テスト(11件)
├── uncertainty_scenarios.py      不確実性集合の生成(latent score摂動方式)
├── test_uncertainty_scenarios.py 不確実性集合の単体テスト(7件)
├── evaluate.py                   決済ドライバ(fail-closed、公式払戻テーブル準拠)
├── test_evaluate.py              決済ロジックの単体テスト(16件)
├── policies.py                   P0-P7政策の実装、既存ソルバーへのラッパー、Stage2A予算完全消化探索
├── test_synthetic.py             Stage 1合成データ試験(15/15 PASS)+Stage2A予算完全消化テスト
├── test_oracle.py                Solverの全列挙oracleテスト(独立参照実装、8件)
└── out/                          再生成可能な成果物置き場(Git非登録)
```

`REPORT.md`（最終報告、仕様書§17/§18）はStage 2A/2B完了後に追加する。

## 既知の制約・設計判断（Stage 0〜Gate J1で判明、対応方針込み）

1. **同一馬への露出上限**は既存`Ticket`にidentityフィールドがないため、`policies.py`の
   `robust_cvar_portfolio()`で反復再solveによる近似実装とした（真の制約付き最適化ではない、
   候補数が多い場合の挙動はStage 2Aで要再検証）。
2. **不確実性集合**(`state_probability_scenarios`)は`uncertainty_scenarios.py`で新規実装した。
   PL latent score(v6生スコア)へi.i.d.ガウス摂動を加え共同着順分布をまるごと再計算する方式
   (各馬券確率を独立に上下させない、券種間の確率整合性が構造的に保証される)。摂動幅は
   2023年developmentのtansho raw PL/既存較正済み確率のlogit差から**IQRベースの頑健推定量**
   （σ=0.2513、当初の単純標準偏差2.2010は穴馬の外れ値に支配され不採用、詳細はspec.json
   `uncertainty_set_construction_method`参照）で固定。オッズ縮小分布もTANPUKアーカイブ内の
   同一レース前売り→確定比を2023年のみで実測（p10=0.774を保守下限として使用）。2024-2025年
   を見て調整しない。
3. **2024-2025年の評価はhistorical_pre_snapshot（発走26-30分前、中央値28分前、実測n=6,909
   レース）を使った「現在コードによるreplay」**であり、実際にその時点で配信された真のT-10
   オッズを使った実配信の再現ではない（DATA_AUDIT.md §5.4参照。「T-10 replay」「実配信再現」
   という呼称は使わない、文書・出力・列名はhistorical_pre_snapshot/median_minutes_to_post/
   range_minutes_to_postで統一）。
4. **対象券種は単勝・複勝・馬連に限定**（`compute_bets.py`のtopdown経路で現在有効な5券種
   [単勝/複勝/馬連/ワイド/馬単]のうち、historical_pre_snapshotの判断時点オッズが実在するのは
   単勝・複勝(TANPUKアーカイブ)・馬連(UMARENアーカイブ)の3種のみ。ワイド・馬単は現在有効だが
   専用オッズarchiveが無くStage 2A主評価からは除外[試す場合は別の探索分析、候補追加を配分
   改善として扱わない]。三連複・三連単はそもそも現行topdownで生成されない）。
5. **raw PLは低確率帯(穴馬)で的中率を過大評価する**（Gate J1実測、tansho最下位確率帯で
   observed/expected=0.46、umaren同0.25[ただし後者は期待イベント数<10の統計的ノイズと判明、
   期待イベント数十分な帯ではO/E=0.82と健全]）。既存較正済み確率(`pl_calibrators_v6`、
   fit_split=valid=2023)がこれを実測で補正することを確認済み。**Stage 2Aへは較正済み確率
   のみを渡し、raw PLは最適化に直接使わない**。
6. **Stage 2A主比較は予算完全消化(sum(w_j)=B)を必須制約とする**。既存ソルバー
   (`optimise_portfolio`)はbudget以下の任意額を許容しno-betを積極的に選ぶ設計のため転用せず、
   `policies.full_spend_search()`を独立実装した。ソルバー異常時はno-betで黙って逃げず
   `Stage2ABudgetAnomaly`として異常件数に計上する。
