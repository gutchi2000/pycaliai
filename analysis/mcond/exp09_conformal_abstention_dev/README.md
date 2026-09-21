# EXP09 — Conformal/OODによる保証付き見送り

## 目的

予測精度の向上ではなく、**外挿・不確実なレースを検知して参加判断(参加/見送り)を
改善できるか**の検証。ROIではなく情報量指標(logloss/Brier/的中率)を主指標とする。

## 絶対条件

EXP01-08・v6・EXP05・EXP05-F・compute_bets.py・EXP06は変更しない。本番の印・
買い目へ接続しない。新規作業は`analysis/mcond/exp09_conformal_abstention_dev/`
へ限定する。

## 現在の進捗(2026-09-21時点)

| 段階 | 状態 | 成果物 |
|---|---|---|
| Stage 0(先行研究・既存実装監査) | **完了** | `PRIOR_ART_AUDIT.md` — 厳密なConformalは未着手。近縁の3系統(chaos gate/participation gate/EXP06 OOD)が既に実装・検証済み。うち participation gateの旗艦数値「+5.31pt」は2026 as-servedで**符号反転・撤回済み**。EXP06のLR_CONTROLはJevに完勝(同一coverage)だが、Jevのrisk_prob自体はモデル自身の確信度を統制すると価値消失(FAIL) |
| Stage 1(データ・定義監査) | **完了** | `DATA_AUDIT.md` — 予測対象=単勝(APS型conformal分類)、nonconformity score=APS累積確率質量(non-randomized、有限標本quantile k=ceil((n+1)(1-α))固定)、確率入力=生PL確率(既存calibratorのvalid=2023 in-sample問題を回避)、予測集合=レース単位・周辺coverageのみ主張、例外処理カテゴリ確定(同着2023=3/2024=6/2025=6件等)、raw PL OOS provenance確認済み(Git LFS hash一致でGate0 PASS)、モデル版hash記録、nominal_conformal_coverage=0.90固定。**2024・2025年の性能・ROIは未開封** |
| spec.json | **凍結済み** | 上記全てを2024-2025結果を見る前にコミット。coverage_scope(all_eligible/participating/abstained分離)・tie-break3段階規則(APS-derived abstention score)・Stage2実装順(9ステップ)を含む |
| `MINIMAL_FALSIFICATION_PLAN.md` | **確定** | coverage/participation_rate/abstention_rateの用語分離、Gate0-5構造(主判定participation_rate=75%)、7方式の同一participation_rate比較、中止規律を確定 |

## Stage 0の中心的発見

1. **「厳密なConformal」は未着手**だが、2023-only fit→2024-25 frozen evaluationの
   設計パターンは3系統(participation_analyzer.py/ood_support.py/LR_CONTROL)で
   既に実証済み。EXP09はこの型を独自実装で踏襲する。
2. **participation gateの+5.31ptは regime-stableでない**(2026 as-servedで
   -10.31ptへ符号反転、配線中止済み)。EXP09で類似の設計をする場合、点推定を
   過信せず情報量指標を主指標にする。
3. **EXP06のLR_CONTROL(2023-only fit)は、Jev(LLM判断層)に同一coverageで完勝**
   (Brier 0.107 vs 0.219、全区分で一貫)。EXP09の必須比較対象「2023年だけで
   学習したLR gate」はこの設計をほぼそのまま踏襲できる。
4. **EXP06のJev risk_probは、単純相関ではPASSしたが、モデル自身の確信度
   (m4_top_prob等)を統制すると価値が消えた(FAIL)**。EXP09のConformal/OOD候補も
   同じ罠(「難しいレースの代理指標に過ぎない」)を踏む可能性が高く、最初から
   完全統制比較を設計に組み込む必要がある。
5. **交換可能性(exchangeability)は年度をまたいで無条件には成立しない**
   (calibrator差し替えでchaos分布が丸ごと平行移動した実測例、v6カテゴリ
   エンコーディング不一致、participation gateの符号反転、他の信頼度ゲートの
   符号反転が複数確認済み)。Conformalのcoverage保証を主張する場合は「モデル・
   calibratorの版が変わらない限り」という射程の明記が必須。

## Stage 2実装(コード完成・合成テスト完了)

| ファイル | 役割 |
|---|---|
| `eligible_races.py` | ステップ1。例外処理カテゴリ(同着・重複・NaN・少頭数・確率和異常・結果欠損)を適用しeligible race集合を確定 |
| `aps.py` | ステップ2-6。APS nonconformity score・有限標本quantile(q_hat)・prediction set・3段階tie-break(APS-derived abstention score)・coverage 3分離出力 |
| `comparators.py` | ステップ7。6方式(max-probability/entropy/OOD support/feature missing/2023-only LR_CONTROL/現行chaos gate)。実装中に発覚したバグ(`score_feature_missing`のyearsフィルタ未適用、全年混入)を修正・回帰テスト追加 |
| `evaluate.py` | ステップ8。レース単位logloss/Brier(全7方式共通の生PL確率)・meeting-day paired bootstrap・Gate3 full-control回帰 |
| `stage2_run.py` | ステップ1-9の統合ドライバ |
| `test_*.py` | 合成テスト40件、全通過(実データ2024-2025は一切使わない) |

## 次の一手

commit後、`stage2_run.py`を実行しGate1-4を評価する(初めて2024・2025年の
結果を開封する段階)。Gate1-4の結果に基づき、通過した場合のみGate5(経済評価)
へ進むかを判断する。
