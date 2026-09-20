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
| calibrator provenance監査 | **完了** | `CALIBRATION_AUDIT.md` — `pl_calibrators_v6.pkl`はvalid=2023(2023年全体)でfitと判明。同じ2023年全体でのGate J1評価はin-sampleだったため時系列blocked split(H1fit→H2評価)へ訂正 |
| Gate J1（共同分布の実測較正、時系列安全版） | **完了、単勝・複勝はPASS、馬連はFAIL** | `gate_j1_calibration.py`, `test_gate_j1_calibration.py`（14テスト） — H1(前半)でIsotonic較正器を新規fitしH2(後半)で評価。単勝・複勝は最下位確率帯のO/Eが許容範囲内だが、**馬連は最下位帯O/E=1.625で許容上限1.6をわずかに超過しFAIL**(期待イベント数14.1件、統計的ノイズの可能性はあるが事前固定基準のため緩和していない)。**Stage 2A対象券種を単勝・複勝の2種に確定**(馬連は除外) |
| Solverの全列挙oracleテスト | **完了、全通過** | `test_oracle.py`（8テスト） — 独立に書いた参照実装(solverの内部関数を一切再利用しない)で目的値・同値解・丸め安全性を検証 |
| 不確実性集合の構築 | **完了** | `uncertainty_scenarios.py`, `test_uncertainty_scenarios.py`（7テスト） — PL latent scoreを摂動し共同着順分布をまるごと再計算(独立摂動による確率破綻を構造的に回避)。スコア摂動幅・オッズ縮小分布とも2023年developmentのみで実測・凍結（下記参照） |
| 決済ロジック（evaluate.py） | **完了** | `evaluate.py`, `test_evaluate.py`（16テスト） — 公式確定払戻テーブルをそのまま使うfail-closed設計。同着・複勝レンジは独自計算せず公式値をそのまま通す |
| Stage 1合成データ試験（§8） | **完了、15/15 PASS** | `test_synthetic.py` — 決済ロジック実装により項目15（返還馬券）を有効化、全15項目通過。Stage 2A予算完全消化探索(`full_spend_search`)のテストも追加 |
| Stage 2Aドライラン | **完了、異常なし** | `stage2a_dry_run.py`, `test_stage2a_dry_run.py`（6テスト） — 2024/2025の結果列(fin/top3/win/fpay)を一切読まずに全パラメータを固定(候補生成・予算1,000円・露出上限600円・λ=1.0・摂動draw数200・seed20260920等)。レース数2024=3,453/2025=3,455、n_field5-18(3頭未満ゼロ)を確認、グリッド規模3,003<上限200,000 |
| P2_PROXY構築可否の検証 | **完了、構築断念** | `spec.json` `primary_comparison_amendment_20260920` — P2(現行topdown)の2024-2025年網羅的配信記録が存在しないと判明(単発サンプル1件のみ)。`compute_bets.py`はexport_weekly_marks.py由来のbundle構造とlive_dir経由T-10ライブオッズの鮮度検証(20分)を必須とする設計で、historical_pre_snapshotへの再実装は単発1サンプルでは正しさを担保できないため構築を断念。**正式主比較をP6 ROBUST_CVAR vs P1 FLATのみに限定**(結果開封前にamendmentとして記録・コミット) |
| Stage 2A本実行（実データ・配分のみ比較） | **未着手** | 次の作業。予算完全消化(`sum(w_j)=B`)を主比較の必須制約とする。正式Gate判定はP6 vs P1のみ |
| Stage 2B（選別+配分） | **未着手** | Stage 2A PASS後のみ |
| Gate 0-5 | **Gate 0/J0/J1判定済み(J1は単勝・複勝PASS、馬連FAIL)、Gate 1-5は未判定** | Stage 2A/2B完了後 |

## ファイル構成

```
analysis/mcond/exp07_robust_portfolio_dev/
├── README.md                    このファイル
├── PRIOR_ART_AUDIT.md            Stage 0: 既存robust_ticket_portfolio.pyの監査
├── DATA_AUDIT.md                 §5: データ監査(必須入力/評価専用/時点安全性/過去再現性/券種スコープ)
├── JOINT_DISTRIBUTION_AUDIT.md   §6: 共同着順分布の監査 + Gate J0結果
├── CALIBRATION_AUDIT.md          pl_calibrators_v6.pkl の provenance監査(fit期間等)
├── spec.json                     §7: 数理定義・比較政策・Gate定義(結果を見る前に凍結)
├── build_scenarios.py            共同着順分布(top-3順列状態空間)の構築
├── test_build_scenarios.py       Gate J0関連の単体テスト(14件)
├── gate_j1_calibration.py        Gate J1: 2023 developmentのみ、時系列blocked split(H1fit→H2評価)で較正を実測
├── test_gate_j1_calibration.py   Gate J1純粋ロジックの単体テスト(14件)
├── uncertainty_scenarios.py      不確実性集合の生成(latent score摂動方式)
├── test_uncertainty_scenarios.py 不確実性集合の単体テスト(7件)
├── evaluate.py                   決済ドライバ(fail-closed、公式払戻テーブル準拠)
├── test_evaluate.py              決済ロジックの単体テスト(16件)
├── policies.py                   P0-P7政策の実装、既存ソルバーへのラッパー、Stage2A予算完全消化探索
├── test_synthetic.py             Stage 1合成データ試験(15/15 PASS)+Stage2A予算完全消化テスト
├── test_oracle.py                Solverの全列挙oracleテスト(独立参照実装、8件)
├── stage2a_dry_run.py            Stage 2Aドライラン(結果列を読まずに全パラメータ固定)
├── test_stage2a_dry_run.py       ドライランの単体テスト(6件)
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
4. **【2026-09-20夜、二段階で訂正】対象券種は最終的に単勝・複勝の2種に限定**。
   `compute_bets.py`のtopdown経路で現在有効な5券種[単勝/複勝/馬連/ワイド/馬単]のうち、
   historical_pre_snapshotの判断時点オッズが実在するのは単勝・複勝(TANPUKアーカイブ)・
   馬連(UMARENアーカイブ)の3種(item1の再監査結果)。しかしcalibrator provenance監査で
   `pl_calibrators_v6.pkl`が2023年全体でfitされていると判明し、Gate J1を時系列安全な
   H1fit→H2評価へ訂正した結果、**馬連はH2での最下位確率帯O/E=1.625が許容上限1.6を
   わずかに超過してFAIL**(CALIBRATION_AUDIT.md§4参照)。馬連はStage 2A主評価から除外し、
   試す場合は別の探索分析に限定する(候補追加を配分改善として扱わない)。ワイド・馬単は
   引き続きhistorical_pre_snapshotデータが無く対象外。三連複・三連単はそもそも現行
   topdownで生成されない。
5. **raw PLは低確率帯(穴馬)で的中率を過大評価する**（Gate J1実測、tansho最下位確率帯で
   observed/expected≈0.68、統計的に十分な標本）。**較正器自体の評価は時系列安全な
   H1fit→H2評価を使う**(pl_calibrators_v6は2023年全体でfitされているため同じ2023年
   全体での評価はin-sampleだったと判明、CALIBRATION_AUDIT.md参照)。H1fit較正器を
   H2へ適用した結果、単勝・複勝は最下位確率帯のO/Eが許容範囲内(0.6-1.6)に収まることを
   確認。**Stage 2Aへは本番のpl_calibrators_v6.pkl(2023年全体でfit、再fitしない、
   2024-2025年の結果は一切未使用)による較正済み確率のみを渡し、raw PLは最適化に
   直接使わない**。
6. **Stage 2A主比較は予算完全消化(sum(w_j)=B)を必須制約とする**。既存ソルバー
   (`optimise_portfolio`)はbudget以下の任意額を許容しno-betを積極的に選ぶ設計のため転用せず、
   `policies.full_spend_search()`を独立実装した。ソルバー異常時はno-betで黙って逃げず
   `Stage2ABudgetAnomaly`として異常件数に計上する。
7. **【2026-09-20夜】正式な主比較はP6 ROBUST_CVAR vs P1 FLATのみに限定**。P2(現行topdown)
   の2024-2025年網羅的配信記録が存在せず、`compute_bets.py`の再実装によるP2_PROXY構築も
   単発サンプル1件では正しさを担保できないため断念した(`spec.json`
   `primary_comparison_amendment_20260920`参照)。**P1に勝っても「現行topdownより優れている」
   「本番配分を置き換えられる」「実配信replayで勝った」とは表現しない**。表現は「同一候補・
   同一予算で均等配分より改善した」に限定する。前向きshadowで実際のtopdown候補・配分を
   同時保存できるようになった時点で、正式なP6 vs P2比較を再開する設計とする。
