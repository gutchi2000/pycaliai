# EXP07 — 馬券ポートフォリオ・ロバスト最適化 検証

既存の予測確率・候補馬券・判断時点オッズを固定した上で、馬券間の相関・排他的払戻・確率誤差・
確定オッズ変動を考慮するロバストポートフォリオ最適化(CVaR/distributionally robust)が、
現行の資金配分および単純均等配分を上回るか検証する。詳細は`spec.json`（結果を見る前に固定済み）。

## 絶対条件

EXP01-06・EXP05-Fを変更しない。本番の印・買い目・資金配分・`compute_bets.py`へ接続しない。
既存モデルを再学習しない。候補馬券生成と資金配分を最初から同時最適化しない。判断時点より
後のオッズ・結果を最適化入力に使わない。2024〜2025年を完全未使用期間と呼ばない。ROI点推定
だけで採用しない。

## 現在の進捗（2026-09-20時点）

| 段階 | 状態 | 成果物 |
|---|---|---|
| Stage 0（既存未コミットコード監査） | **完了** | `PRIOR_ART_AUDIT.md` — 総合判定「再利用可能（3制約明記の上で）」 |
| データ監査（§5） | **完了** | `DATA_AUDIT.md` — 2024-2025年は真のT-10データが存在せず、TANPUKアーカイブのT-28相当値による「現在コードのreplay」としてのみ評価可能と判明 |
| 共同損益分布 + Gate J0（§6） | **完了、PASS** | `JOINT_DISTRIBUTION_AUDIT.md`, `build_scenarios.py`, `test_build_scenarios.py`（14テスト全通過） — 既存`pl_probs.py`のPL厳密連鎖を土台に1-3着順列状態空間(18頭立て最大4,896状態)を構築、marginal整合性を機械精度で確認 |
| 数理定義 + spec.json（§7） | **完了、凍結** | `spec.json` — 比較政策P0-P7、Gate0-5、CVaR alpha規約の変換規則（**重要**: 仕様書のalpha=0.10[最悪10%平均]と既存コードのcvar_alpha[信頼水準]は逆の慣習なので`policies.py`で変換している）を含む |
| Stage 1合成データ試験（§8） | **完了** | `test_synthetic.py` — 15項目中14項目PASS、1項目(返還馬券の扱い)は決済ロジック(`evaluate.py`)未実装のため正直にskip |
| Stage 2A（実データ・配分のみ比較） | **未着手** | 次の作業 |
| Stage 2B（選別+配分） | **未着手** | Stage 2A PASS後のみ |
| Gate 0-5 | **未判定** | Stage 2A/2B完了後 |

## ファイル構成

```
analysis/mcond/exp07_robust_portfolio_dev/
├── README.md                    このファイル
├── PRIOR_ART_AUDIT.md            Stage 0: 既存robust_ticket_portfolio.pyの監査
├── DATA_AUDIT.md                 §5: データ監査(必須入力/評価専用/時点安全性/過去再現性)
├── JOINT_DISTRIBUTION_AUDIT.md   §6: 共同着順分布の監査 + Gate J0結果
├── spec.json                     §7: 数理定義・比較政策・Gate定義(結果を見る前に凍結)
├── build_scenarios.py            共同着順分布(top-3順列状態空間)の構築
├── test_build_scenarios.py       Gate J0関連の単体テスト(14件)
├── policies.py                   P0-P7政策の実装、既存ソルバーへのラッパー
├── test_synthetic.py             Stage 1合成データ試験(15項目)
└── out/                          再生成可能な成果物置き場(Git非登録)
```

`evaluate.py`（決済・Stage 2A/2B評価ドライバ）とREPORT.mdはStage 2A着手時に追加する。

## 既知の制約（Stage 0/データ監査で判明、対応方針込み）

1. **同一馬への露出上限**は既存`Ticket`にidentityフィールドがないため、`policies.py`の
   `robust_cvar_portfolio()`で反復再solveによる近似実装とした（真の制約付き最適化ではない、
   候補数が多い場合の挙動はStage 2Aで要再検証）。
2. **不確実性集合**(`state_probability_scenarios`)は呼び出し側が構築する必要があり、
   モジュール自身はキャリブレーション誤差から自動的に不確実性半径を作らない。development
   期間(2023、TANPUKアーカイブT-28相当)だけで半径を固定する設計とする。
3. **2024-2025年の評価は「現在コードによるreplay」**であり、実際にその時点で配信された
   T-10オッズを使った実配信の再現ではない（DATA_AUDIT.md §5.4参照）。
4. **対象券種は単勝・複勝・ワイド・馬単に限定**（データ監査の結果、2024-2025年の馬連は
   TANPUKアーカイブに該当列がなく実質データなしと判明したため除外。三連複・三連単は
   そもそも仕様書の対象外）。
