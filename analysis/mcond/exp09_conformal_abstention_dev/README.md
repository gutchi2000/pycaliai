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
| Stage 1(データ・定義監査) | **完了** | `DATA_AUDIT.md` — 予測対象=単勝(APS型conformal分類)、nonconformity score=APS累積確率質量、確率入力=生PL確率(既存calibratorのvalid=2023 in-sample問題を回避)、予測集合=レース単位・周辺coverageのみ主張、モデル版hash記録、nominal_conformal_coverage=0.90固定。**2024・2025年の性能・ROIは未開封** |
| `MINIMAL_FALSIFICATION_PLAN.md` | **改訂・確定** | coverage/participation_rate/abstention_rateの用語分離、Gate0-5構造(主判定participation_rate=75%)、7方式の同一participation_rate比較、中止規律を確定 |

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

## 次の一手

`MINIMAL_FALSIFICATION_PLAN.md`(§A正式確定版)をユーザーへ再提示済み。
承認後、spec.json凍結→Stage 2(実装)へ進む。
