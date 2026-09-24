# EXP16A — Close-Market Residual Information Bound（運用メモ / 凍結記録）

このファイルは**管理上の記録だけ**を持つ。結果・Gate・閾値・母集団・arm・placebo・seed 規約は
`spec.json` が正本であり、このファイルの追記で spec を書き換えない。

---

## 1. 凍結記録（FREEZE RECORD）

| 項目 | 値 |
|---|---|
| 凍結対象 | `spec.json` **v0.4**（Stage 0 改訂3、Gate 3 等級） |
| 凍結基準 commit | **`e7603678`**（`research(EXP16A): Stage 0 改訂3 — Gate を3等級化…`、2026-09-24 21:12:27 +0900） |
| frozen_at | **2026-09-24 21:17:12 +0900** |
| 凍結時の検証 | `stage0_checks` 137 項目 PASS（JSON 妥当性 / 文書間整合 / 等級文言一致 / 合成境界テスト 13 件 / Wilson 記録 / C2c 一致 / DNF 注記） |
| Stage 1 着手判定 | power audit tier2 の進行条件（真の効果 0.005 で PASS-PRACTICAL ∪ PASS-SIGNAL ≥ 80%）を満たす（実測 100%、Wilson 下限 0.990） |

### 凍結後の変更ルール

1. 凍結後に `spec.json` を変更する場合は**必ず新しい版番号**を振る（v0.5、v0.6 …）。
2. 変更理由・影響範囲・変更時刻を、**評価結果を開封する前に** commit する。
3. **結果を見た後**の Gate・閾値・母集団・arm・placebo・seed 規約の変更は**禁止**。
4. 管理上の追記（本 README・実行手順・ログ）で spec v0.4 本体を書き換えない。

### 凍結後に行った管理上の追記

| 時刻 | 内容 | spec への影響 |
|---|---|---|
| 2026-09-24 21:17 | 本 README を新設（凍結記録・実行手順） | なし |
| 2026-09-24 21:2x | Stage 1-1: rolling OOF 40 fit と検査を追加（`build_oof.py` / `oof_tests.py`、`race_population.py` に official race artifact の保存を追加） | なし（母集団規則・基準値は不変） |
| 2026-09-24 21:4x | Stage 1-2: 実方向 `log(Q1/π)` の検出力監査を追加（`power_audit_q1.py`） | なし（判定規則・実務床・等級は spec v0.4 のまま） |
| 2026-09-24 21:5x | Stage 1-3: 2019-2023 評価と REPORT を追加（`evaluate.py` / `REPORT.md`） | なし（Gate 判定は `gate_grade.grade_gate` のみ） |

---

## 2. 正式な実行方法

すべて**リポジトリ root（`E:\PyCaLiAI`）から module として**実行する（`python analysis/…/x.py` の直叩きは不可）。

```bash
# Stage 0 の検証（JSON 妥当性・文書間整合・等級の境界テスト）
python -m analysis.mcond.exp16a_close_market_residual_dev.stage0_checks

# 市場データの素性
python -m analysis.mcond.exp16a_close_market_residual_dev.provenance

# 正式 race set の確定と母集団監査
python -m analysis.mcond.exp16a_close_market_residual_dev.race_population

# 情報量→成長率の合成確認（C1-C5, C2b, C2c）
python -m analysis.mcond.exp16a_close_market_residual_dev.verify_growth

# 検出力監査（3 段・等級別）
python -m analysis.mcond.exp16a_close_market_residual_dev.power_audit

# Gate の 3 等級判定と境界テスト（単体）
python -m analysis.mcond.exp16a_close_market_residual_dev.gate_grade

# Stage 0 成果物の組み立て
python -m analysis.mcond.exp16a_close_market_residual_dev.stage0_dry_run
```

Stage 1 のスクリプトも同じ形式で実行する（実行順）。

```bash
# Stage 1-1: rolling OOF 40 fit → out/oof_manifest.json, out/oof_build_checks.json
python -m analysis.mcond.exp16a_close_market_residual_dev.build_oof

# Stage 1-1b: OOF の検査 (行対応・被覆・確率和・再現性・削除不変性・未来行・seed ばらつき)
python -m analysis.mcond.exp16a_close_market_residual_dev.oof_tests

# Stage 1-2: 実方向 log(Q1/π) の検出力監査 (2019-2023 の結果を開ける前に必ず実行)
python -m analysis.mcond.exp16a_close_market_residual_dev.power_audit_q1

# Stage 1-3: 2019-2023 crossfit 評価 (power Gate 通過後のみ)
python -m analysis.mcond.exp16a_close_market_residual_dev.evaluate
```

## 3. ファイルの役割

| ファイル | 役割 |
|---|---|
| `spec.json` | **正本**。Gate・閾値・母集団・arm・placebo・seed 規約 |
| `gate_grade.py` | Gate A/B の 3 等級判定の**唯一の実装**（simulation・本判定・境界テストが同じ関数を通る） |
| `PRIOR_ART_AUDIT.md` | 先行研究監査＋改訂履歴＋DNF 再正規化の注記 |
| `INFORMATION_TO_GROWTH_DERIVATION.md` | `Δ > −log(1−t)` の意味と hard Gate にしない理由、C1-C5 / C2b / C2c |
| `MARKET_DATA_PROVENANCE.md` | TANPUK の素性、pre / terminal_close_market の定義と時刻 |
| `RACE_POPULATION_AUDIT.md` | 正式 race set（平地・DNF なし）と funnel、DNF 感度 |
| `OOF_STACKING_PLAN.md` | rolling OOF と `retrospective_rolling_crossfit_development`、artifact 契約 |
| `POWER_AUDIT.md` | 検出力監査 3 段と等級別 power |
| `STAGE0_DRY_RUN.json` | Stage 0 の集約成果物（母集団・基準値・power・artifact 契約） |
| `build_oof.py` / `oof_tests.py` | Stage 1-1: rolling OOF の構築と検査 |
| `power_audit_q1.py` | Stage 1-2: 実方向 `log(Q1/π)` の検出力監査 |
| `evaluate.py` | Stage 1-3: 2019-2023 crossfit 評価（Gate 判定は `gate_grade` のみ） |
| `REPORT.md` | Stage 1 の結果報告 |
| `out/*.json` | 各スクリプトの実測値 |

## 4. 禁止事項（Stage 1 実行中も継続）

- 2024/2025 の開封
- ROI 本評価・候補馬券生成・資金配分最適化・production 変更
- 結果を見てからの Gate / 閾値 / 母集団 / arm / placebo / seed 規約の変更
- PASS-SIGNAL を PASS-PRACTICAL と言い換えること
- FAIL を他券種・JRA 全体・依存型馬券へ一般化すること
