# EXP09 — 最小反証実験・停止条件(提案、実装前・ユーザー承認待ち)

`PRIOR_ART_AUDIT.md`の発見を踏まえた設計。特に(a) EXP06のJev risk_probが
「モデル自身の確信度を統制すると価値が消える」で失格した前例、(b) participation
gateの+5.31ptが2026 as-servedで符号反転した前例、の2つを踏まえ、最初から
これらの罠を検出できるGate構造にする。

## 対象・単位

- **レース単位**の参加/見送り判断(§5監査の結論通り、既存3系統と整合)。
- 主指標: **top3 logloss/Brier(情報量指標)**。ROIは副次(EXP08 Gate2A/2B踏襲、
  participation_analyzer.pyがROIのみだった弱点への対処)。

## 比較する7ゲート(同一coverageで比較、ユーザー指定通り)

1. **Conformal/risk-controlled gate**(新規、本実験の対象)
2. 単純な最大予測確率gate(◎のp_win、較正済み)
3. entropy gate(全馬確率分布のnormalized entropy)
4. OOD/support gate(`ood_support.py`と同型設計を**EXP09で独自実装**、2023-only
   fit・PCA・k-NN距離、EXP06のコード自体は再利用しない)
5. feature-missing gate(特徴欠損率、単体では未検証の次元)
6. **2023年だけで学習したLR gate**(`stage_b_gate_eval.py::_fit_lr_control()`と
   同型設計を独自実装、目的変数は要件定義次第だが「2023年内top3 loglossの
   中央値超」等、結果を見る前に固定)
7. 現行participation gate(`production_policy.py`のchaos percentile、読み取り
   専用の参照として使う、コード変更なし)

## coverage点(事前固定、結果を見て有利な点だけ選ばない)

**90% / 75% / 50% / 25%** の4点で全ゲートを比較する。この4点はStage 1で
データ監査した後、実装前にspec.jsonへ凍結し、2024-2025結果を見た後に変更しない。

## Gate構造(EXP06のGate1→2→3構造を踏襲、EXP08のGate2A/2B踏襲)

### Gate0: データ健全性・時点安全性
- 判断時点で入力が再現可能(§8監査参照、市場系はhistorical_pre_snapshot規約)
- 較正器の版とfit期間の来歴確認(§7監査、in-sample汚染がないか)
- 交換可能性の射程を明記(§6監査、「v6/calibratorの版が固定の期間内」と限定)

### Gate1: 限界情報量(2023 developmentのみ)
Conformal/OOD候補のスコアと、2023年内の予測誤差(Brier/logloss)の単調関係
(Spearman)。EXP06 Gate1と同型。

### Gate2: 完全統制後の固有情報量(★最重要、EXP06 Jevが落ちた関門)
候補スコアを、モデル自身の確信度(m1-m4相当のtop prob・entropy・市場確率・
市場entropy・AI市場乖離・頭数・人気帯・競馬場・芝ダ・距離帯・クラス帯・年度)
**全てを統制した回帰**に追加項として入れ、係数の95%CIがゼロを跨がないことを
要求する。**単純相関でPASSしても、この完全統制でFAILすれば終了**(EXP06と
同じ判定基準)。

### Gate3: 同一coverage比較(7ゲート全部、4coverage点全部)
2023developmentでfitした各ゲートを2024年・2025年へ固定適用(再fitなし)。
各coverage点で、Conformal/risk-controlled gateの平均Brier/loglossが、
他6ゲート**全て**を下回ること(レース単位bootstrap 95%CI上限<0)。
一部のゲートにしか勝てない場合はPASSとしない(EXP06 Gate3の基準を踏襲、
「全区分で一貫」の精神)。

### Gate4: Coverage保証の実測検証(Conformalとして名乗るための固有関門)
Conformal/risk-controlled gateが理論上主張するcoverage(例: 「エラー率X%以下を
保証」)が、2024年・2025年**それぞれ単独で**実測coverageと事前指定した許容誤差
以内で一致すること。2024と2025で乖離が大きい場合は「exchangeability前提が
この期間で崩れている」と解釈し、保証の主張を撤回する(誤魔化して閾値を
事後調整しない)。

### Gate5(Gate1-4通過後のみ): 経済評価
100円均等ベースのROI比較。EXP07のロバストCVaRは再利用しない。**+5.31ptの
教訓により、ここでPASSしても「2026 as-served相当の独立期間で再確認するまでは
本番配線しない」と明記する**(将来課題として記録、本Stageの範囲外)。

## 中止規律(結果を見た後に変更しない)

- coverage点(90/75/50/25%)、7ゲートの定義、Gate1-4の閾値・統制変数リスト、
  LR gateの目的変数定義、OOD gateのk・PCA次元・距離関数、のいずれも2024-2025
  結果を見た後に変更しない。
- Gate2(完全統制)でFAILしたら、統制変数を減らして再検定しない(EXP06と同じ
  「単純相関のPASSは見せかけである可能性」を常に疑う)。
- Gate3で一部のゲートにしか勝てない場合、比較対象から都合の悪いゲートを外して
  再判定しない。
- Gate4でcoverage保証が実測と乖離した場合、許容誤差を広げて通過させない。
- どのGateで落ちても、それ以降のGateには進まない(Gate5経済評価は最後の
  Gate1-4全通過時のみ)。

## 想定される終了条件(事前登録)

- Gate1 FAIL: 候補スコアが予測誤差と全く相関しない → 終了。
- Gate2 FAIL: モデル自身の確信度を統制すると価値が消える(EXP06 Jevと同型) → 終了。
- Gate3 FAIL: 単純ゲート(max-prob/entropy/LR等)のいずれかに負ける → 終了
  (「Conformal機構固有の価値なし、単純ゲートで十分」という結論)。
- Gate4 FAIL: coverage保証が2024/2025で一致しない → 「厳密なconformalとしては
  射程限定的」と結論し、経済評価(Gate5)には進まない。

## Stage 1以降の作業(承認後)

1. データ監査(§8の入力再現性を実データで確認、DATA_AUDIT.md)
2. spec.json凍結(coverage点・7ゲート定義・Gate1-4基準・LR gateの目的変数・
   OOD gateのハイパーパラメータを2024-2025結果を見る前に確定)
3. 実装(各ゲートの構築コード、時点安全テスト)
4. Gate0-4評価(2023developmentでfit、2024-2025でgenuinely OOS)
5. 通過した場合のみGate5(経済評価)

このプランでよいか、coverage点・7ゲート・Gate構造に変更が必要か、ご確認ください。
