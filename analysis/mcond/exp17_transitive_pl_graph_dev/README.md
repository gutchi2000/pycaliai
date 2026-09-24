# EXP17 — Dynamic Common-Opponent Hodge–Plackett–Luce

## Status

**Stage 0 完了・終了（2026-09-25）。事前登録した停止規律 1（EXP02 scalar ability との等価）と 2（被覆が凍結床未満）に該当。Stage 1 以降は実施しない。**

2019〜2023 の成績評価、2024/2025 の開封、ROI・払戻、候補馬券生成、資金配分、production 変更は一切していない。

## 結論（範囲限定）

> 事前固定した時点安全な 2-hop 共通対戦馬表現は、対象母集団・期間・統制の下で必要なペア固有情報を追加しなかった。

- **E0-A（代数）**: Hodge 射影の出力は馬ごとのスカラー。pair 固有情報は射影残差にのみ残り、`softmax(s/τ)` には届かない。race-level 確率 arm は EXP02 と同じモデル族（推定量が異なるだけ）。
- **E0-B（2022 label-free 実測）**: 未対戦 pair の `d_ij − b·Δμ_ij` の分散は、EXP02 能力から各対戦を再抽選した null の **0.840 倍**（CI95 [0.825, 0.856]、事前固定 FAIL 条件 上限 < 1.10）。pair 固有状態は抽選ノイズと区別できない。
- **データ構造**: 使用辺の 78.8%（2022 では 84.7%）が対戦 1 回。1 回の対戦は符号 1 ビットで、pair 固有状態を推定する繰り返し観測が存在しない。
- **被覆**: 暫定床 4 本中 1 本 FAIL（80% のレースで全 pair の半数以上 covered: 実測 61.7%）。
- **検出力**: G1 floor 0.001 nats/pair で power 1.000。検出力は停止理由ではない。

PageRank・3-hop 以上・embedding・GNN・identity 結合項・地方海外履歴・他券種が失敗したとは主張しない。

## Files

| ファイル | 内容 |
|---|---|
| `SPEC.md` / `spec.json` | v0.2（Stage 0 完了・終了版） |
| `PRIOR_ART_EQUIVALENCE_AUDIT.md` | 先行研究との式単位比較、E0-A/B/C の判定 |
| `GRAPH_DATA_AUDIT.md` | ID 監査、正式 race set funnel、時点安全性テスト、被覆率（年・頭数帯・芝ダ・年齢・出走数帯）、床照合 |
| `POWER_AUDIT.md` | 推論単位、生成モデル、power 表、機構 floor の固定、placebo 設計と EXP12 比較 |
| `COMPUTE_DRY_RUN.json` | 実測時間・容量と Stage 1 見積り |
| `graph_core.py` | 唯一の実装（ID 検証、履歴ストア、pair 証拠、Hodge 射影、softmax） |
| `test_invariants.py` → `out/invariant_tests.json` | 合成 invariant 17 項目 |
| `coverage_audit.py` → `out/GRAPH_COVERAGE.json`, `out/race_population.json`, `out/*.parquet`, `out/compute_timings.json` | 実データ被覆監査 |
| `equivalence_audit.py` → `out/equivalence_audit.json` | 2022 label-free 等価性監査 |
| `power_audit.py` → `out/power_audit.json` | 検出力監査 |
| `stage0_checks.py` | 成果物間の母集団・期間・Gate・停止条件の整合検査 |

## 再現手順（root から）

```bash
python -m analysis.mcond.exp17_transitive_pl_graph_dev.test_invariants
python -m analysis.mcond.exp17_transitive_pl_graph_dev.coverage_audit      # 約 7 分
python -m analysis.mcond.exp17_transitive_pl_graph_dev.equivalence_audit   # 約 2 分
python -m analysis.mcond.exp17_transitive_pl_graph_dev.power_audit         # 約 3 分
python -m analysis.mcond.exp17_transitive_pl_graph_dev.stage0_checks
```

## Hard boundaries（v0.1 から不変）

- No 2024/2025 result evaluation. No ROI, ticket generation, staking, or production connection.
- No PageRank, embedding, GNN, or paths longer than two hops.
- No claim that all graph methods failed. No claim of novelty after Stage 0 showed equivalence to EXP02.
