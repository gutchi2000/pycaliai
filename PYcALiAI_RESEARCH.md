# PyCaLiAI 研究開発憲章 (PYcALiAI_RESEARCH.md)

> **位置づけ**: 生成AI（Claude Code 等）が PyCaLiAI を研究・改善する際に必ず最初に読む憲章。
> 単なる README ではない。**約1年分・数百本の実験の生存/死亡記録**であり、同じ失敗を繰り返さないためのガードレールである。
> 作成: 2026-08-09（リポジトリ全域監査に基づく。エージェント3系統: データ/特徴量・モデル/評価/serve・実験史発掘）
> 運用フロー詳細は `CLAUDE.md`、歴史詳細は `docs/STATUS_AND_HISTORY.md`、監査は `docs/audit_20260615_full.md`、版台帳は `docs/version_ledger.md` を参照。
> 本書で確認できなかった事項は **UNKNOWN** と明記する。推測を事実として書かない。

---

## 0. 最重要の思想（先に読む）

1. **予測層は天井**。◎top3 ≈ 62%（オッズ非使用時）は、1400特徴ブルートフォース・Transformer・物理/EVT/統計力学・独立AI（Keiba-ai 58-62%）の全てで破れなかった。**「特徴量を足す/モデルを変える」提案を無条件にするな**。
2. **戦場は betting/ops 層と市場との関係**。理論床は控除率 ≈80%（ROI）。目標は「儲け」より「最も負けない線」+ 未配線の検証済み ROI の回収。
3. **唯一の予測側の勝利は T-10 オッズブレンド**（◎top3 61.7%→65.1% OOS）。62%天井は「オッズ無し時」の話。
4. **点推定を単独で配線するな**。clean-band ゲート（+5.31pt 実測 → 2026 再検証で符号反転 → 撤回）が教訓。配線より「最新期間での CI 付き再検証」を優先。
5. **test (2024-25) は既に ≥7 回開封され汚染されている**（VAL-02）。新実験は v10 プロトコル（test=2025 封印、valid CI 下限 > 閾値のときのみ 1 回開封）に従う。
6. Fact / Evidence / Hypothesis / Speculation を常に区別する。「競馬では○○が重要」という一般論を「だから PyCaLiAI に効く」に飛躍させない。

---

## 1. Project Overview

JRA 中央競馬の AI 予想システム。役割分担:

| レイヤー | 担当 | 出力 |
|---|---|---|
| PyCaLiAI モデル部 | 印付け（◎〇▲△△）+ Plackett-Luce 確率 | `reports/cowork_input/{date}_bundle.json` |
| compute_bets.py（T-10 自動、本番買い目） | topdown エンジンで買い目生成 | `reports/cowork_output/{date}_bets.json` |
| Cowork (Claude Desktop) | narrative 論評専用（買い目は書かない） | 同上 bets.json の advisor 部 |
| 静的サイト（HF Space pycaliai-umami / pycaliai.com） | 表示専用 | 画面 |

- ターゲット問題: 「3着以内」中心（正例率 ≈21.9%）。三連単・馬単は廃止済み。
- 予算: 1R ≈ ¥10,000。投票は人間 IPAT（自動投票なし）。
- JRA-VAN 投稿ガイドライン準拠: サイトから調教タイム生値/オッズ生値/払戻/EV 等は撤去済み（新データ公開時は必ずこの基準に照らす）。

---

## 2. Current Architecture

```
出走表 CSV (TARGET)
  → export_weekly_marks.py --model v6
      unified_rank_v6.pkl (LGBM lambdarank, 120特徴) → 生スコア
      → PL 重み → p_win / p_plc / p_sho
      → pl_calibrators_v6_serve.pkl（存在時こちらが優先; tansho/fukusho のみ適用, p_plc は生のまま）
      → 印 ◎〇▲△△（生スコア降順 top-5）+ race_confidence（dominance/concentration/chaos/市場一致）
  → bundle.json
  → compute_bets.py（T-10, CB_ENGINE=topdown 既定, ENGINE_VERSION 2026-07-31+）
      §0 参戦ガード（chaos_raw≥0.92 / 頭数≤7 / ◎オッズ欠損 / p_win(◎)<0.05 → skip）
      §0b clean-band は配線撤回済み（機構のみ残置）
      p_win → λ補正 PL（data/harville_lambda.json, λ1=0.8405, λ2=0.7542）→ 全ペア p_umaren/p_wide
      候補: 複勝top1 + ワイドtop2(オッズ≤50) + 馬連top1(≤50) + 単勝top1(≤30)
      p 比例配分（¥500-7000, ¥100刻み）→ 適応トリガミ床（最安払戻<総額なら低p点を削る）
      決済ドリフト補正（SETTLE_DRIFT_*, EV のみ）
  → validate_cowork_bets.py --apply（見送りガード強制, fail-closed）
```

- 旧経路: `CB_ENGINE=shape`（印スロット/shape/妙味ヒューリスティクス）。gutchi_brain は 2026-08-09 退役・削除。
- 表示専用: T-10 補正印（`log(p_win)+λ·log(π)` 再ランク, `data/t10_blend.json`）— 買い目不干渉。
- Streamlit + 旧8モデルアンサンブルは副系統として残存（P1 課題: v6 統一が方針）。

---

## 3. Data Pipeline

生ソース → master 4 段構成（全て確認済み、行数は実測）:

| 段 | スクリプト | 出力 |
|---|---|---|
| 1 | `build_dataset.py`（lgbm/cat/torch/add CSV + kekka を JOIN） | `data/master_20130105-20251228.csv` (412MB) |
| 2 | `parse_kako5.py --mode master` | `data/master_kako5.csv` (469MB) |
| 3 | `build_master_v2.py`（+補正タイム+調教+コース/騎手履歴） | `data/master_v2_20130105-20251228.csv` (516MB) |
| 4（推論のみ） | `make_weekly_hosei.py` / `parse_kako5.py --mode weekly` / `parse_training.py` | `data/hosei/H_{date}.csv` 等 |

- **master_v2 実測: 626,774 行 × 132 列、2013-01-05 〜 2025-12-28**。
- JOIN キー = `["レースID(新)", "馬番"]`。着順 NaN（除外・中止等）は drop。
- ターゲット定義箇所: `build_dataset.py:256` `fukusho_flag = (着順<=3)`、`roi_target = 複勝配当/100`。
- カラム名統一: レースIDは **`"レースID(新/馬番無)"`**（旧 `"レースID(新)"` 揺れ注意）。
- 週次 CSV を Excel で開くと破壊される（レースID指数表記化）→ 修復は `scripts/repair_excel_weekly_csv.py --inplace`。
- 外部: `E:\競馬過去走データ\`（調教マスター H/W 520万/70万行 cp932、全頭確定単勝オッズ）。**このディレクトリは不可侵（読み取りのみ）**。手元に無いデータは探し回らずユーザーに依頼（TARGET から出力可能）。

---

## 4. Feature Inventory（v6 本番 = 120 特徴）

選択方式は**除外リスト方式**: `optuna_v6_marks.py:138` — master_v2 132 列 − LEAK_COLS 12 列 = 120。

LEAK_COLS（`optuna_v6_marks.py:68-73`）: 着順, fukusho_flag, roi_target, レースID×2, 馬名, レース名, 発走時刻, date_dt, 日付, 血統登録番号, split。

カテゴリ内訳（詳細はコード参照）:
- **前走成績** ~23列（前走走破タイム/着差/通過/上り/着順/斤量/馬体重/PCI 系…）
- **kako5 過去5走集約** 16列（avg/std/best_pos, avg_ninki, pos_vs_ninki, agari3f, same_cond 系, pos_trend, hidden_good_count…）
- **全キャリア履歴** 4列（hist_same_cond/place 系）
- **ローリング複勝率** 6列（jockey/trainer/horse × 窓, `shift(1)` 済み）
- **脚質** 2列（prev_pos_rel, closing_power）
- **補正タイム** 2列（prev_hosei, prev_hosei9 — 前走のみ。今走補正はリークとして除外）
- **調教** 16列（trnH_* 坂路 / trnW_* WC、カバレッジ 坂路~80% / WC~25%）
- **コース/騎手履歴** 6列（course_/jockey_ n_prev/win_rate/top3_rate、as-of cumsum）
- **コース・条件・クラス・血統・厩舎・枠情報** 残り（CAT_COLS 27列 = LabelEncoder, train のみ fit, 未知値 `__NaN__`）

**観測していない情報**（= モデル入力に無い）:
- 当日オッズ・人気（EV/賭け側のみで使用。今走単勝オッズは完全リーク=kekka は勝ち馬のみ収録）
- 当日馬体重（master_v2 に列自体が無い。週次 CSV にはあるがモデルに渡らない）
- レース内相対特徴（race_relative_feats は v2/v7 系のみ、v6 未使用）
- 過去走のラップ生値/不利/位置取り詳細（ラップ CSV は評価済み: priced + JOIN 不能, `project_lap_csv_evaluated`）
- 既知の疑似デッド: `母馬`（CAT_COLS 外の文字列 → -9999 化と推定, **UNKNOWN 未実証**）。定数刷り込み等の serve 死は正直計数 36 特徴（大半修復済み）。
- v10 で発見済み・未配線のパースバグ修正（前走走破タイム/着差の -9999 量死）: `lab/train/optuna_v10_marks.py:118-133` に修正コードあり。

---

## 5. Target Definition

- 学習ラベル（v6）: `np.clip(6 - 着順, 0, 5)`（lambdarank 用グレード）。`fukusho_flag` は LEAK_COLS で除外（**ターゲット直接学習ではない**）。
- sample weight: `1 + α·log1p(勝ち馬単勝配当/100)`、α は Optuna 探索（v6 採用値 α=0.0308 ≈ ほぼ無重み。v5 の α=1.325 はテール calibration 崩壊で退役）。
- 評価ターゲット: 印別的中率（◎top3 等）、券種別 ECE、ROI。

---

## 6. Train / Validation / Test

- 定義は **`build_dataset.py:40-41, 273-281` の1箇所のみ**: train ≤2022-12-31 / valid 2023 / test 2024-01-01〜（実データは 2025-12-28 まで）。時系列分割、`split` 列として master_v2 に焼き込み。ランダム分割は存在しない。
- 学習: train fit + valid early stopping。Optuna 目的も valid のみ（valid 内レースID 5-fold KFold — **時系列 CV ではない点に注意**）。
- Calibrator: valid=2023 のみで fit（`build_pl_calibrators.py:56`, メタに `fit_split` 記録）。
- **既知の汚染**: test 2024-25 は版選定で ≥7 回開封済み（VAL-02, `docs/audit_20260615_full.md`）。v6 自体が汚染 test 下で採用された。さらに v6 採用は自ら定めたゲート（単勝 EV≥1.2 ROI +0.05 等）を**満たしていない**（OBJ-04, `docs/version_ledger.md`）。
- **今後の規律（v10 プロトコル）**: test=2025 封印。valid の CI 下限が閾値を超えたときのみ 1 回だけ開封。偽 PASS を製造しない。

---

## 7. Leakage Rules（赤旗一覧）

コードで確認済みの防御:
- ローリング特徴は `shift(1)`、kako5/hist は自レース除外の as-of、コース/騎手履歴は cumsum−自行。
- 今走補正タイム・今走オッズ・当日馬体重は入力から排除。target encoding は**リポジトリに存在しない**。
- LabelEncoder / calibrator / early stopping いずれも test を触らない。
- serve 側: `_SERVE_RENAME`（列名不一致→-9999 化の防止）、bundle 品質ゲート、**serve skew canary**（`data/serve_feature_baseline.json` 対比で特徴の無言死を検知、fail-closed で push 阻止）。

既知の学習/推論非対称（残存リスク）:
- 調教 JOIN: 学習は `merge_asof(allow_exact_matches=False)` 日数制限なし / 推論は 14 日カットオフ + 同日許容（`export_weekly_marks.py:308-313` に既知残差として明記。serve 条件 fit calibrator が吸収する設計）。

過去に実際に踏んだリーク（再発防止）:
- v8 course_affinity: 自レース込み集計の in-sample leak（`build_course_affinity.py:120-123`）。as-of/OOF 必須。
- crosspool 88-92%: 確定オッズ oracle（betable 不可）。
- ラップ CSV 先頭 1678 行の当日 leak。
- realized σ のトレンド汚染（EVT H2 反証）。
- strategy_weights.json: test ROI で採用→同じ test で評価の構造循環（registry L316-361）。

原則: 「結果的に少ししかリークしていない」は許容しない。生成順序をコードで確認するまで leak-safe と主張しない。

---

## 8. Current Baseline（これを超えない提案は無意味）

| 指標 | 値 | 条件 |
|---|---|---|
| ◎top3（複勝圏） | ≈62%（オフライン）/ serve 実測 ≈61.0% | v6, オッズ非使用 |
| ◎top3 T-10 ブレンド後 | 65.1%（OOS Δ+3.41pt CI[+2.5,+4.3]） | 表示専用 |
| ECE | 複勝 0.0108（serve 再fit 後）; 印確率 calibration はほぼ完璧（ECE 0.001-0.003） | |
| ROI（機械買い床） | ≈79.9% ≈ 控除率（罠全切り後の残 ROI） | |
| topdown エンジン replay | 82.8%（4/18-8/9 506R, 旧比+8.7pt, P(改善)=0.961, **in-sample 注意**） | 前向き監視中 |
| 実運用累積 | 620R / ROI 69.5%（`data/cowork_results.json` 2026-06-22 時点） | |
| 市場ベースライン | 群衆 R=0.703、上位N点フラット買いは全券種で控除壁内 | |

v6 Optuna 採用値: lr=0.0508, num_leaves=59, max_depth=12, min_data_in_leaf=197, best_iter=469（`reports/optuna_v6_marks.json`）。

---

## 9. Evaluation Framework

階層で評価する。**ROI 単独でモデルを評価しない / 予測指標改善 = 収益改善ではない**:

```
Prediction quality (NDCG@5, ◎top3, AUC)
  ↓ Calibration (10-bin ECE 券種別: p_tansho◎/p_fukusho◎〇/p_umaren◎-〇)
  ↓ Market comparison (対市場 R², EV-bin ROI, ai_market_agreement)
  ↓ Decision policy (topdown replay, 参戦ゲート, トリガミ)
  ↓ ROI (CI 付き, ペア bootstrap, 期間別)
```

- ツール: `audit_marks.py`（期間: 2023=valid参照 / 2024 / 2025 / 真OOS 2024-25）、`scripts/audit_v6_vs_v5.py`（EV-bin ROI）、`backtest_fixed.py`（馬連7点固定）、`backtest_pl_ev/formation/kelly.py`、`lab/audits/eval_rps_brier.py`。
- 既知の欠陥: v6 の `ECE_high_p` は単一 bin で過小/過大確信を相殺する（OBJ-02/03）。修正版（4-bin）は v10 にあるが未採用。新実験では 10-bin ECE を使うこと。
- 統計基準: 複数期間 + paired bootstrap CI を基本。閉鎖/採用の基準非対称に注意（MDE≈1pt — 「死亡」判定の一部は検出力不足の可能性が既知監査で指摘済み）。

---

## 10-12. Historical / Failed / Successful Experiments

**一次資料**: `docs/STATUS_AND_HISTORY.md` §4「死亡ルート一覧（再走禁止）」が正典。`docs/audit_20260615_full.md`、`docs/version_ledger.md`、`docs/hypothesis_registry.md`、`docs/feature_backlog.md`「提案しない」節、`lab/README.md`（実験 97 本の所在地図。再実行は `python -m lab.<theme>.<name>`）。

### 死亡ルート（再走禁止）— 要約表

**予測層・特徴量**（全て本番 v6 土俵で検定し採用ゼロ）:
| ルート | 死因 | 証拠 |
|---|---|---|
| 特徴量ブルートフォース（+100/+300/+1000/1400計） | gain 寄与 94.8% でも正味 0 | `reports/feat_exam_300_result.json`, `lab/train/train_v1000.py` |
| 格/クラス変動（v7, v11 15特徴フル再学習） | +0.62pt CI 非有意 = 市場吸収済み | `reports/audit_marks_v11.json` |
| 適性系（距離/場所/血統/回り左右） | −0.3〜−0.68pt | `reports/ablation_aptitude.json`, `ablation_direction.json` |
| 特徴プルーニング | 有意悪化 −0.72pt CI 全負（弱特徴も集団で効く） | `reports/ablation_prune.json` |
| Elo/Glicko・血統 embedding・race level | 冗長/逆効果 | `lab/features_dead/` |
| セリ取引価格 | 生 coef 有意→市場統制で 91% 吸収 = priced | `exp_auction_decisive.py` |
| 物理（Keller/pace）・EVT・統計力学（2/3体・自由E） | ΔAUC 微小/符号逆/直交ゼロ | `lab/physics_gates/` |
| 夏専用/regime 専用モデル | 全 6 粒度で否定、プール学習が最良 | `exp_summer_upweight.py` |
| 馬場（クッション/含水/トラックバイアス当日・クロスデイ） | 馬場状態に吸収 / ~9割 priced / ROI 床割れ | `analysis/daybias_within_card_test.py`, `crossday_bias_test.py` |
| 不利代理（hidden_good の次走 ROI） | priced | `analysis/measure_hidden_good_roi.py` |

**モデルアーキテクチャ**:
| ルート | 死因 |
|---|---|
| Transformer / Set Transformer | 汎化ゼロ（配置情報は予測層で exploitable 不可） |
| Stacking meta / MoE 距離別 expert / custom profit loss | Brier +≤0.001 / rejected pkl / ログ棄却 |
| v7(wide-ROI 目的) / v8(affinity=leak) / v9 / v10 / v11 | 全て採用ゲート未達（v10 の封印プロトコルとバグ修正は資産） |
| Deep Value Net（◎単勝 ROI100% 狙い DL） | 確信度と ROI が単調逆相関 |
| 消しモデル×合成印 | 3層全死（消し=強いの逆数） |

**市場・オッズ**:
| ルート | 死因 |
|---|---|
| Benter blend / shrinkage | test ROI 0.709、対市場 ΔR² 負（α≈0.3 のみ REAL_BUT_UNPROFITABLE） |
| crosspool 裁定 | 確定オッズ二重 oracle |
| EV 価格エッジ選抜（gate_q2 等） | ≤2023 の 125-130% が test で 64-67% に崩壊、CLV 負 |
| 市場内部歪み裁定 / Dr.Z | 非効率実在だが控除 > 非効率 |
| オッズ軌跡/マネーフロー/CLV | 9時価格を OOS で超えず / pari-mutuel で CLV 換金不能 |
| **オッズを予測特徴に入れる** | AUC↑は市場価格の写像 = ROI 不変 + serve 不可。**禁止**（T-10 の bet 時ブレンドとは別物） |

**馬券・ポリシー**:
| ルート | 死因 |
|---|---|
| EV 閾値による銘柄選抜 | **有害 −13pt**。289 セル網羅で全滅。prob-first 化済み（EV はフロア/配分に降格） |
| 学習型ベッティング（learn_bet v1-v5, NN） | 9時価格を超えず |
| ポリシー空間ブルートフォース | ユーザーのトリガミ床ルールを dominate 不可 |
| 三連複 value AI / WIN5 / 枠連 / 券種空間全体 | プールオッズ入手不能 / 全史 ROI 0.837 / 全券種で群衆超過≈+7pt一定＝壁内 |
| 穴×under カット / danger-fav / レース選抜 ROI セル | OOS で蜃気楼 |
| clean-band 参戦ゲート | +5.31pt → 2026 as-served で符号反転 → **配線撤回**（点推定配線の戒め） |
| 「市場エッジ=利益」（gutchi 実打 10 例） | leave-one-out で崩壊（2 レース抜くと 84.3%） |

### 生存・採用済み（Successful）
- **T-10 オッズブレンド補正印**（唯一の予測側勝利、表示専用）
- **λ補正 PL（Stern/Lo-Bacon-Shone）+ topdown エンジン**（2026-08-09 既定化、replay 82.8%、前向き監視中）
- **構築層 4 修正**（穴 overlay vb-◎ペア撤去 / ◎単勝妙味時のみ / cap5 / p×boost 配分）
- **適応トリガミ床**（ユーザー発ルール、ML が dominate できなかった唯一の頑健改善。chalk-cap 併用）
- **prob-first**（EV 選抜の廃止）・**馬単/三連単全廃**・**serve skew 修復 + canary（fail-closed）**・**serve 条件 fit calibrator**・決済ドリフト補正・◎圧勝 conf ボーナス（rule mining 唯一の生存）・見送りガード（validate_cowork_bets）
- 診断として価値: 負けは◎飛び 37.9% に局在 / v6 は gain 60.9% 過去走支配 / calibration は完璧（= 問題は予測でも確率でもなく市場との重なり）

### 判定不明（UNKNOWN — 再検証候補）
`exp_recency_sweep` / `exp_window` / `quantile_exp` / `learn_target_compare` / `train_v6_multiseed` の採否記録なし。`analysis/exotics_ev_market_test.py`（SettleAI Phase1, exotics 実市場 EV 初検定）は初回実行のみ。G1 ローテ RPCI バイアス（H4）は n≥200 待ち。

---

## 13. Known Limitations

- 予測天井 ◎top3≈62%（オッズ無し）— 独立 AI も同値 = 情報限界であってモデル容量不足ではない。
- AI 印の構造盲点: 過去走の「格・対戦相手の質・斤量文脈」を平均着順に潰して見れない（`project_marks_grade_blind_pastform`）。◎はスロット粒度で 1 番人気に −4.2pt 負け。
- 穴予測は構造的に苦手（◎大穴帯勝率 0%）。新馬戦は graceful degrade（複勝ワイド軸 OK、単勝深追い不可）。
- CLV は pari-mutuel で換金不能。控除率 ≈20% が常に床。
- 印表示レイヤーが lossy middle layer（mark_composite が目的関数に侵入 + 買い目が印スロットにゲート）— topdown 化で買い目側は解消、印/bundle 側は残存。

## 14. Current Bottlenecks

1. **市場との重なり**（過去走支配 60.9% = 市場と同じものを見ている）が◎飛びと ROI 床の根本原因。
2. **検証の非対称性**: test 汚染（≥7 回開封）+ 閉鎖判定の検出力不足（MDE≈1pt）— 「死亡」の一部は原理死でなく検出不能死。
3. topdown/構築層修正が **in-sample replay 検証のまま**（最大の運用リスク。前向き監視が最優先タスク）。
4. モデル 2 系統並走・governance 矛盾（v6 が自ゲート未達のまま本番、version_ledger 化は途上）。

---

## 15. Residual Analysis（誤差構造の既知事実）

- 負けの局在: **◎飛び（37.9%）**。組み合わせ層は健全（◎来時 ROI 131-139%）。calibration 完璧 → 「確率が悪い」のではなく「市場も同じ確率を出している」。
- entropy/gap による荒れ検知は可能だが priced。「夏は荒れる」は 6 粒度で否定。「夏は牝馬」は実在するが完全 priced。
- 妙味帯: mid-popularity 過小評価は方向として実在（mid 80.9 > fav 76.6 > tail 68.2）だが magnitude はノイズ。
- 新実験での残差分析は `analysis/diag_upset_decomposition.py`, `diag_pastform_dominance.py`, `analysis/build_loss_forensics.py` を出発点にする。

---

## 16. Research Priorities（改善候補の優先順位）

Tier 1: **未観測情報源**（ただし §10-12 の死亡表と衝突しないもののみ）
Tier 2: 系統誤差（◎飛び）を説明する情報
Tier 3: 市場乖離を説明する情報（bet 時のみ。予測特徴化は禁止）
Tier 4: calibration / probability（joint_m1 配線など）
Tier 5: decision / policy（topdown 前向き検証、SettleAI）
Tier 6-7: アーキテクチャ変更 / HPO — **原則却下**（全滅実績）

## 17. Experiment Protocol

1. **提案 → 議論 → 採用 → 実装** の順を厳守。新理論の勝手実装禁止。最終採用判断はユーザー。
2. ONE CHANGE AT A TIME（明示的な ablation/interaction のみ例外）。
3. 新特徴の提案フォーマット必須: Hypothesis / Information / Mechanism / Leakage risk / Expected effect / Failure mode / Experiment / Falsification criterion。
4. test=2025 封印（v10 プロトコル）。valid CI 下限 > 閾値のときだけ 1 回開封。
5. 死亡ルート再走は原則禁止。再走するなら「以前の実験と何が違うか」を明示（検出力不足死の再検定は正当な理由になりうる）。
6. 既存コードの大規模書き換え禁止。実験は新規スクリプト（`analysis/` or `lab/`）+ config 切替 + 固定 seed。
7. 実行前に本書 + `docs/STATUS_AND_HISTORY.md` §4 + `docs/feature_backlog.md`「提案しない」節を照合する。

## 18. Quality Gates

実験前チェック（`CLAUDE.md` 注意事項も参照）:
- Input: 列存在/dtype/欠損/行数/重複/JOIN キー（`レースID(新/馬番無)`）/encoding（cp932 vs utf-8-sig）
- Validity: 単位/符号/レンジ/桁/リーク（§7 赤旗）/分布シフト
- Reproducibility: 入出力パス/コマンド/seed/git commit/ログを実験ログに残す
- Verification: `assert` / `df.shape` / `df.isna().sum()` を惜しまない
- serve に触る変更は canary（`export_weekly_marks.py:536+`）と `analysis/measure_serve_coverage.py` を必ず通す

## 19. Research Log Format

```
Experiment ID: / Date: / Hypothesis: / Baseline: / Change: / Data:
Train period: / Validation period: / Test period: (封印状態を明記)
Metrics: / Result: / OOS result: (CI 付き) / Leakage check: / Decision: / Reason:
```
悪かった結果も必ず残す（死亡の記録こそ本プロジェクト最大の資産）。置き場: `docs/hypothesis_registry.md` 形式 or `reports/` + docs 追記。

## 20. AI Development Rules

- 役割: Researcher / Data auditor / Experiment designer / Statistical analyst / Software engineer / Reproducibility auditor。コード生成器ではない。
- 説明の上手さを採用根拠にしない。もっともらしい未検証案の量産を自制する。
- 汎用 AI の定番提案（予測 v7 / 特徴量追加 / Transformer / アンサンブル / EV 閾値 / ROI 85-90% 目標）は**全て死亡済みルート**。反証は §10-12 と `project_external_ai_proposals_dead_routes`。
- 確認できないことは UNKNOWN と書く。過去会話由来の情報とリポジトリで確認した情報を区別する。
- git push のうち HF 反映を伴うもの・データ/モデル削除・master 再生成は要確認。それ以外の通常作業は自律実行。

---

## 21. Current Research Frontier（次に何を研究すべきか）

### 十分検証済み（閉鎖領域）
予測層特徴量全般 / モデルアーキテクチャ全般 / EV 選抜 / CLV / オッズ軌跡 / 市場内部裁定 / 券種空間（WIN5・枠連・三連複value 含む）/ 馬場バイアス馬券化 / regime 専用化 / 学習型ポリシー。

### Priority 1 — topdown エンジンの前向き検証（最重要・最安）
- 仮説: replay 82.8% は in-sample。前向きでも旧経路+構築層修正を上回る。
- 方法: 週次実運用ログを ENGINE_VERSION 付きで蓄積、n≥300 bets で paired 比較 + CI。反証条件: 前向き ROI が旧経路 CI 内に沈む → shape へロールバック。
- 難易度: 低（配線済み、観測するだけ）。リーク危険: なし。

### Priority 2 — 検証済み未配線 ROI の回収
- **joint_m1 umaren 配線**（ECE−33% / +0.3-0.7pt、本番 export は素の PL 積のまま）。要: 本番土俵での CI 再検証 → export 配線。
- **UMAMI 正配線**（馬連 cap10=84% / ワイド cap3-4=82.8% 測定済みだが topdown 化後の帰属再検証が先）。
- **serve/学習の調教 JOIN 非対称解消**（学習側に 14 日カットオフを揃える再学習は低コスト・低リスク）。

### Priority 3 — SettleAI（決済層）の続き
- 帯別ドリフト再 fit（15-30 倍帯の符号逆は確定済み）→ exotics 実市場 EV 初検定の完遂（`analysis/exotics_ev_market_test.py` は初回のみ）→ per-horse 予測器（optional）。ドリフト方向選別は両方向エッジゼロ実証済みで**毒・禁止**。

### Priority 4 — UNKNOWN 実験の決着（低コスト）
- exp_recency_sweep / exp_window / quantile_exp / learn_target_compare / v6 multiseed の採否を registry に記録して閉じる。
- v10 のパースバグ修正（前走走破タイム等の -9999 量死）だけを v6 に単独移植して ablation（ONE CHANGE）。`母馬` 疑似デッドの実証。
- H4（G1 ローテ RPCI）の n≥200 到達判定。

### Priority 5 — 統計ガバナンスの整備
- 閉鎖判定の検出力（MDE）を registry に明記し、「原理死」（priced/oracle/換金不能）と「検出不能死」を区別ラベル化。後者のみ、データが倍増した将来時点での再検定を許可する。
- test 開封台帳の運用（version_ledger 拡張）。

### 明示的に狙わないこと
ROI 85-90%（無理筋、床は ≈80%）。予測精度の追求への回帰（競合ベンチマーク上も 62% は業界の地の値であり、弱いのはプロダクト/配信側）。

---

## Appendix: 主要ファイル索引

| 目的 | ファイル |
|---|---|
| 分割定義 | `build_dataset.py:273-281` |
| v6 学習 | `optuna_v6_marks.py`（LEAK_COLS :68, ラベル :113, 目的 :261-270） |
| Calibrator | `build_pl_calibrators.py` / `build_pl_calibrators_serve.py` |
| Serve + canary | `export_weekly_marks.py`（rename :313, canary :536+） |
| 買い目エンジン | `compute_bets.py`（topdown :478-535, ガード :436-473, トリガミ床 :521-527） |
| λ fit | `analysis/fit_harville_lambda.py` → `data/harville_lambda.json` |
| 死亡ルート正典 | `docs/STATUS_AND_HISTORY.md` §4 |
| 版台帳 | `docs/version_ledger.md` |
| 仮説登録簿 | `docs/hypothesis_registry.md` |
| 実験置き場地図 | `lab/README.md` |

**UNKNOWN 残項目**: master/master_kako5 の正確な行数、`run_v6_pipeline.py` 内の calibrator 生成パラメータ経路、`serve_history_feats` の as-of 実装の学習側との厳密一致、`serve_feature_baseline.json`/`chaos_quantiles.json`/`t10_blend.json` の現在値、`母馬` の実データ挙動。
