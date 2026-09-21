# EXP10 — 大敗・中止・能力未発揮リスク

## 【2026-09-21】最終結論: Stage1.5で終了（2024・2025年は未開封）

> **既存の時点安全な表特徴だけを用いた、完走馬に対するAI◎のcatastrophic
> downside専用headは、2023年内の時系列CVでB3を上回らなかった。**

R1(最小構成: 過去着順分散・振幅・距離/馬場/競馬場変更の5特徴)は、B3(v6下方
尾部確率+市場確率+人気+頭数+最大確率+entropy+score gap+出走回数+休養日数)に
対し、meeting-day forward-chaining CV(2023年developmentのみ)で**一貫した
限界的寄与を示さなかった**(logloss/Brier/PR-AUC/較正の4指標中3指標が悪化、
信頼区間はいずれもゼロを跨ぐ)。事前登録した停止規律「続行条件を一つでも
満たさなければ2024・2025年を開封せずEXP10を終了する」に従い終了する。

**結論の範囲は上記一文に厳密に限定する**。以下は**未検証**（失敗ではない）
として記録し、EXP10への追加・再実行は禁止、実施する場合は必ず新規の
独立した実験として扱う: DNF・中止・取消・除外の別母集団／raw結果履歴から
新規構築する過去大敗頻度／着順・走破タイム残差の個体内分散／位置取り崩壊
頻度／故障・疾病理由を使うモデル／完走確率そのもの。

詳細は`STAGE1_DESIGN.md` §7-10参照。

## 目的

通常の着順精度向上ではなく、**v6が高く評価した馬が大きく崩れる下方リスク**を、
既存の勝率・entropy・市場人気の後にも検知できるかの検証。

**主仮説**: v6確率、raw score、entropy、市場確率、人気、頭数、出走回数、
休養日数を統制した後にも、専用の下方リスクモデルが大敗を識別できるか。

**主目的変数**: `catastrophic_downside = 1[observed_rank > q90_predicted_rank]`
（AI◎自身のPlackett-Luce予測分布から見て、実際の着順が上位90%分位の外に
あったか＝「v6自身の予測分布からみても異常な下方乖離」）。

## 絶対条件

EXP01-09・v6・compute_bets.pyは変更しない。新規作業は
`analysis/mcond/exp10_downside_risk_dev/`へ限定する。
「負けた馬=陽性」という設計は禁止（v6の逆モデルになるため）。

## 現在の進捗(2026-09-21時点)

| 段階 | 状態 | 成果物 |
|---|---|---|
| Stage 0(先行研究・データ監査) | **完了** | `PRIOR_ART_AUDIT.md` / `DATA_AUDIT.md` |
| Stage 1(定義確定・2023実件数・エンジン検証) | **完了(修正版)** | `STAGE1_DESIGN.md` / `spec.json` |
| Stage 1.5(最小反証実験、meeting-day forward-chaining CV) | **完了・続行条件不成立** | `out/STAGE1_5_REPORT.json` |
| Stage 2 | **実施せず（EXP10終了）** | — |

## 2026-09-21 Stage1修正（ユーザー指摘を反映）

Stage1完了後、ユーザーから3点の修正指示を受け反映した:

1. **q90ラベルをモンテカルロ誤差から保護**: `resolve_q90_label()`を新設し、
   n<=18(JRA平地レース出走可能最大頭数、2023年実測上限と一致)は常に厳密
   bitmask DPを使う設計へ変更。n>18のみWilson信頼区間付きadaptiveモンテカルロ
   梯子(K=50,000→200,000→1,000,000→厳密DP fallback→q90_unresolved)を使う。
   **2023年development全3,378レースは100%厳密DPで確定し、モンテカルロ誤差は
   主ラベルに一切含まれない**。
2. **PL入力の整合性を実データで確認**: Gumbel-maxのlocationがraw score
   (exp化なし)であること、既存`pl_probs.all_tansho()`との最大誤差1.11e-16、
   入力順序不変性の最大誤差5.69e-16を実測(149レース)。
3. **同着件数をEXP09と比較可能なカテゴリへ分解**: EXP09の「2023年3件」は
   EXP10の「1着同着のみ」カテゴリと完全一致(パーサバグではなく定義の広さの
   違い)。新たに「着順番号の欠番」を1件発見し除外規則へ追加、2023年の主
   ラベル件数を554/3,379(16.40%)→551/3,378(16.31%)へ再計算。

詳細は`STAGE1_DESIGN.md`の改訂履歴セクション参照。

## Stage 1.5（実行中）

- **CV設計**: random foldを禁止し、meeting-day(rid[:10]、288単位)の
  forward-chaining(blocked chronological) CV(5fold)へ変更。同一開催日を
  train/testへ分割しない。前処理(imputer/scaler)は各train foldのみでfit。
- **固定ハイパーパラメータ**: LogisticRegression(l2, C=1.0, class_weight=None)、
  事後変更禁止。SMOTE等の合成oversamplingは使わない。
- **R1(最小構成)**: 時点安全監査で「既存列で対応可」と判定した5特徴のみ
  (過去着順分散・振幅・距離/馬場/競馬場変更)。「要構築」の5候補はStage2で追加。
- **判定**: R1 vs B3のOOF予測、開催日単位paired bootstrap。logloss/Brier/PR-AUC/
  較正の4指標、ROC-AUCのみの改善はFAILとする。

## Stage 1.5 実行結果（2026-09-21、確定）

288 meeting_day単位、5fold forward-chaining。**集約結果(OOF)**:

| 指標 | B3 | R1 | 差(R1-B3) |
|---|---|---|---|
| logloss | 0.43652 | 0.43748 | +0.00096(悪化) |
| Brier | 0.13416 | 0.13430 | +0.00014(悪化) |
| PR-AUC | 0.23999 | 0.23819 | -0.00180(悪化) |

開催日単位paired bootstrap: logloss差CI95=[-0.00171,+0.00370]（ゼロを跨ぐ）、
Brier差CI95=[-0.00061,+0.00090]（ゼロを跨ぐ）。較正slope: B3=0.898→R1=0.831
（悪化）。8つの続行条件中**6条件が不成立**。

**判定: 続行条件を満たさないため、2024・2025年を開封せずEXP10を終了する。**
詳細は`STAGE1_DESIGN.md` §7.3-10参照。

## コード構成

| ファイル | 役割 |
|---|---|
| `pl_rank_distribution.py` | PL順位分布計算エンジン(厳密DP/adaptive MC梯子/全順列列挙/resolve_q90_label) |
| `test_pl_rank_distribution.py` | 合成oracleテスト43件 |
| `validate_mc_vs_dp_realdata.py` | 実2023データでのMC-DP突合検証 |
| `pl_input_consistency_audit.py` | PL入力の整合性確認(raw score/pl_probs一致/順序不変性) |
| `eligible_races.py` | 対象馬(◎)選択・tie-break・母集団構築 |
| `dead_heat_audit.py` | 同着・除外カテゴリの内訳分解(EXP09比較用) |
| `labels.py` | 主ラベル(catastrophic_downside)・副次診断計算 |
| `subsidiary_favorite.py` | 1番人気版副次診断 |
| `stage1_5_features.py` | Stage1.5用B0-B3/R1(最小構成)特徴テーブル構築 |
| `stage1_5_cv.py` | meeting-day forward-chaining CV実行・判定 |
| `spec.json` | 全定義の凍結(修正版) |

## 次の一手

なし。EXP10は終了。再走する場合は、少なくとも「要構築」5候補
（タイム残差下方分位・休養明け成績分散・馬体重履歴変動・出遅れ頻度・
騎手調教師変更）を含むフルR1で改めてStage1.5相当の検証を行う必要があるが、
中止規律により本セッションでは実施しない。
