# EXP15 Stage 0 — 先行研究監査（PyCaLiAI vNext: Race-as-a-Set Decision System）

**作成日**: 2026-09-23　**状態**: Stage 0 監査のみ。実装・学習・バックテスト・
2024/2025年性能の新規開封・ROI評価は一切行っていない。

**本監査で行った操作（全て読み取り）**: リポジトリ全文検索、既存 `.py` / `.md` /
`.json` / `.log` の閲覧、`models/unified_rank_v6.pkl` の読み込み（特徴量名と
encoder の一覧取得のみ。スコアリングはしていない）、`master_v2` の先頭 2,000 行
（2013年分）の列の型確認。下で引用する 2024-25 年の数値は**全て本監査以前に
作られた既存成果物からの引用**であり、本監査で再計算したものはない。

**監査プロセス上の注記**: 並列の読み取り専用サブエージェントのうち1体が
`python -c "print(1)"` を1回、別の1体が空の `python -` を1回（ハングして停止）
誤って起動した。どちらも既存スクリプトは実行しておらず、ファイル変更もない。

---

## 0. 結論（先に要点）

1. **ユーザー指定の19項目を判定した**（19と明示的に数えた。
   `UNEXPLORED_METHODS_AUDIT_20260920.md` で起きた方向数の数え漏れを繰り返さない
   ため）。内訳は**実施済み 6 / 部分実施 8 / 未着手 5**。詳細は §2。
2. **「レースを1つの集合として見るニューラルモデル」は既に存在した**。
   `train_transformer.py` の `RaceTransformer`（2026-03）は、同一レース全頭を
   `nn.TransformerEncoder` で self-attention 処理する。位置符号化も CLS トークンも
   無いので、実質的に**順序に対して equivariant な集合エンコーダ**になっている。
   PL（ListMLE 型）損失版の `transformer_pl_v2` もある。単独の ◎top3 は **54.36%**
   で、v6 の **62.03%** を 7.67pt 下回る（test 2024-25、6,908R、既存成果物）。
   ただしこの比較は**特徴量48本 対 120本、旧 master 対 master_v2、
   複勝AUCでのモデル選択**という3つの交絡を含む。したがって
   「同じ入力で、レース文脈だけを足した時に何が起きるか」は一度も測られていない。
3. **既存文書の記述が実態とずれている**（§5）。死亡台帳 E7「Transformer /
   Set Transformer は汎化ゼロ」の根拠は `exp_havoc_m1.py` だが、これは2体結合の
   特徴量を GBM に入れた実験であり Transformer ではない。一方、
   `UNEXPLORED_METHODS_AUDIT_20260920.md` §4.4 は RaceTransformer を見落としている。
   **過大主張（死亡扱い）と過小主張（未検証扱い）が同時に起きている**。
4. **レース文脈の「中身」を馬ごとの行に押し込む形では、少なくとも7系統が全て死亡済み**
   （§3.D）。対象は展開の構成、レース内の相対値、出走馬の構成、2体/3体結合、
   順序依存の結合、自由エネルギー、havoc 読み出し。共通する症状は
   **「gain 上位で貪欲に使われるのに test は改善ゼロ〜微悪化」**である。
   - `exp_pace_interaction` v2: `pc_ka_vs_field` が gain 1位で ΔAUC −0.00109
   - `feats_1000` の F1 群: 全体の 44% の特徴で gain の 55% を占めながら ◎top3 −0.25pt
   - M1 の havoc 読み出し: gain 4.82% で ΔAUC −0.0042
5. **真に一度も行われていないもの**:
   - (a) 同じ120入力・同じ fold 手順・同じ容量で、文脈経路だけを外した対照
     （R1-noctx）を置く比較
   - (b) レース文脈に対する placebo 検定
   - (c) DeepSets 型の明示的プーリング
   - (d) 同じアーキテクチャの中での pointwise 損失と race-level 損失の比較
   - (e) 同じモデルの中での LabelEncoder と学習埋め込みの比較
   - (f) differentiable sorting / Sinkhorn / Gumbel-Sinkhorn
   - (g) win/top3/rank のマルチタスク head

   vNext が新規性を主張できるのは**この範囲に限られる**。
   「初めてのレース単位モデル」とは主張できない。
6. **事前確率は低い**。文脈情報が実在しても、次の2つの説明が既存証拠と整合する。
   - 馬ごとの行の特徴と木の交互作用で既に再構成されている
   - レースの確信度（entropy / maxprob）を言い換えているだけ

   このため最小反証計画では、**容量統制（Gate 6）と maxprob/entropy 統制
   （Gate 7）が決定的な Gate になる**。

---

## 1. 方法

- **検索語（英）**: `DeepSets|DeepSet`, `Set ?Transformer|ISAB|PMA`,
  `permutation.?(equi|in)variant`, `ListMLE|ListNet`, `plackett_luce_loss|pl_nll|neural.?Plackett`,
  `Sinkhorn|Gumbel|NeuralSort|SoftSort|differentiable.?sort|torchsort`,
  `pairwise|RankNet|bradley.?terry|head.?to.?head`,
  `YetiRank|PairLogit|QueryRMSE|lambdarank|rank_xendcg`,
  `graph.?attention|GAT|GNN|GraphSAGE|message.?passing|torch_geometric|dgl`,
  `multi.?task|multi.?head|auxiliary.?loss`,
  `race.?(level.?)?encoder|race.?embedding|race_context|mean.?pool`,
  `field.?composition|n_front|pace_pressure|front_count`,
  `within.?race|rank_in_race|field.?mean|race_mean`, `copula|Thurstone|Henery|Stern|Mallows`,
  `import torch|nn.Module|MultiheadAttention|TransformerEncoder|GRU|LSTM|MLPClassifier`,
  `categorical_feature|cat_features|LabelEncoder|OrdinalEncoder|FeatureHasher|target.?encoding`
- **検索語（日）**: `逃げ頭数|先行頭数|同型|脚質.*頭数|レース内相対|レース内順位|相対能力|券種別確率|マルチタスク`
- **範囲**: `*.py`・`*.md`・`reports/*.json|*.log`・`docs/`・`lab/`・`analysis/`・`archive/`・
  auto-memory。
- **除外**: `site/`・`venv311/`・`.git/`・`reports/note/`（HTML）、`*.csv|*.parquet|*.pkl` の中身
  （pkl は v6 の特徴一覧取得のみ例外）。
- 3系統の読み取り専用スイープを並列で行い、監査者本人が主要ファイル
  （`train_transformer.py`・`optuna_transformer_pl.py`・
  `reports/evaluate_transformer_v6_stack.*`・`reports/evaluate_ensemble_v6.log`・
  `analysis/mcond/*`）を直接確認した。

---

## 2. 項目別判定（19項目）

判定の凡例:
- **実施済み**: 実装があり、評価結果が残っている
- **部分実施**: 隣接する狭い版だけ実施済み、または交絡や未検証を含む
- **未着手**: 実装も結果も無い

「生死」列の凡例:
- **本番**: 現在の本番で稼働中
- **死亡**: 不採用が確定している
- **交絡**: 比較に交絡があり、結論が出ていない

| # | 項目 | 判定 | 生死 | 主な根拠 | 既存結果（引用） | vNext との差分 |
|---|---|---|---|---|---|---|
| 1 | DeepSets | **未着手** | — | 検索 0 件 | — | 明示的な ρ(pool φ) で文脈を馬に戻す構造は無い。手作りの集約特徴（§3.D）は「固定した φ・ρ の DeepSets」とみなせる |
| 2 | Set Transformer | **部分実施** | 交絡 | `train_transformer.py` `RaceTransformer`（SAB を積んだ形。ISAB・PMA は無い） | §3.A | 「Set Transformer」という名前のものは実装されていない。E7 の根拠の付け方が誤っている（§5） |
| 3 | permutation-equivariant model | **実施済み** | 交絡 | 同上（位置符号化なし・CLS なし・`src_key_padding_mask`） | §3.A | **equivariance のテストは無い**。入力順は CSV の行順 |
| 4 | listwise ranking | **実施済み** | 本番 | v6 = LambdaRank（NDCG@5 切り詰め）、`exp_v6_settings.py` の `rank_xendcg` | xendcg: Δ◎top3 +0.01pt → `CEILING_CONFIRMED` | 集合エンコーダと組み合わせた例は無い |
| 5 | ListMLE / ListNet | **部分実施** | 交絡 | ListMLE = `optuna_transformer_pl.py`（全順列の PL NLL）。ListNet は GBDT の近縁 `rank_xendcg` のみ | test AUC 0.7304 / ◎top3 単独 54.36% | ニューラルの ListNet、top-k 切り詰めの PL は無い |
| 6 | pairwise ranking | **部分実施** | 死亡 | CatBoost YetiRank（`train_catboost_marks_diag.py`）、PRED-03B（未実装で打ち切り）、`umaren_pair_v1`（ペア分類） | YetiRank は v6 比 −1.81pt（交絡あり） | RankNet・PairLogit・同一レース内のペア順序損失は無い |
| 7 | LambdaRank | **実施済み** | 本番 | `optuna_v6_marks.py`、v9（truncation 3: −0.91pt）、v10、多 seed、PRED-03A | 5 seed 平均 +0.13pt・ROI −1.85pt、PRED-03A Δ+0.00pt | R0 そのもの |
| 8 | neural Plackett-Luce | **実施済み** | 交絡 | `transformer_pl_v2`（PL 損失、選択は複勝 AUC） | valid 0.7231 / test 0.7304、v6 との配合で最良 +0.14pt（α を test で選択） | 同じ入力の対照が無い。温度・較正は未評価 |
| 9 | differentiable sorting | **未着手** | — | 検索 0 件（`UNEXPLORED` §4.15 も E） | — | — |
| 10 | Sinkhorn ranking | **未着手** | — | 検索 0 件（Sinkhorn は最適輸送の検索語としてのみ出現） | — | — |
| 11 | Gumbel-Sinkhorn | **未着手** | — | Gumbel のヒットは全て Gumbel-max による PL サンプリング（EXP10 のエンジン、`crux_fuku`）か EVT の Gumbel 雑音で、**別概念** | — | — |
| 12 | joint finish distribution | **部分実施** | 死亡（校正は生存） | `pl_probs.py`（本番）、Harville λ、`pl_stern.py`、M1/M2/順序/自由エネルギー、`crux_joint.py` | §3.C | 全て「周辺の強さから導出した分布＋結合補正」。学習された同時分布 head は無い |
| 13 | horse-to-horse interaction | **部分実施** | 死亡 | 2体 M1（`gate1_eval1.py`）、3体（`gate2_3body.py`）、RaceTransformer の attention（暗黙） | 馬連 ECE 改善、実馬券 OOS 黒字セル 0/36 | 学習された相互作用を**同一入力の対照付きで**測った例は無い |
| 14 | pace interaction | **実施済み** | 死亡 | `exp_pace_interaction.py` v1/v2、`build_feats_serious.py` の S2、keller、gate1-4 | ΔAUC −0.001 前後 | 手作り特徴としては閉じている |
| 15 | field composition | **実施済み** | 死亡 | S6（14特徴）、`build_feats_1000.py` の F1、`race_relative_feats.py`（v2/v7）、EXP05 の M3 | §3.D | 学習された構成表現は RaceTransformer の中で暗黙に扱われただけ（交絡あり） |
| 16 | race-level encoder | **部分実施** | 交絡／死亡 | RaceTransformer（attention で文脈を各馬に戻す）、havoc_v1（レース単位のスカラー） | havoc_v1 test AUC 0.6265 | 明示的な race vector（pool）を作って馬に戻す構造は無い |
| 17 | graph attention within a race | **部分実施** | 交絡 | 全結合グラフの attention ＝ RaceTransformer | — | 辺特徴（同厩舎・前走同レース・騎手関係など）付きの GAT は無い |
| 18 | multi-task win/top3/rank | **未着手** | — | `REPORT_FOR_AI_FEEDBACK.md` §6.2 に提案があるのみ | — | 単一タスクの別モデル（`fukusho_binary_v1`・`order_model_v1`）は存在する（§3.C） |
| 19 | ticket-probability direct modeling | **部分実施** | 死亡／未検証 | `umaren_pair_v1`（423万ペアの LightGBM）、三連複トリプル（放棄）、`order_model_v1` の3クラス | pair と v6 の配合 81.89%（α を test で選択、CI なし） | 券種確率を集合から直接出す head は無い |

**集計**:
- 実施済み（6）: #3, 4, 7, 8, 14, 15
- 部分実施（8）: #2, 5, 6, 12, 13, 16, 17, 19
- 未着手（5）: #1, 9, 10, 11, 18

合計 19。

---

## 3. 詳細

### 3.A レース集合エンコーダ（#1, 2, 3, 16, 17）

**`train_transformer.py` / `optuna_transformer.py`（BCE 版、`models/transformer_optuna_v1.pkl`）**
- 構造
  - 各馬＝1トークン（CSV の行順）。カテゴリは列ごとの `nn.Embedding`
    （次元 max(4, vocab//4)、padding_idx=0）。
  - 数値特徴と結合し、Linear→LayerNorm→ReLU で d_model に写す。
  - `nn.TransformerEncoder(batch_first)` と `src_key_padding_mask`、
    最後に MLP head（64→1）。
- **位置符号化なし・CLS/race トークンなし**。枠番・馬番は z 化した数値特徴としてだけ入る。
- 18頭で打ち切り、1頭のレースはスキップ。
- 特徴: カテゴリ17（斤量をカテゴリ扱い）＋数値29＋時刻文字列2 = **48本**。
  血統・騎手/調教師 ID・kako5・hosei・調教は**入っていない**。
- データ: **旧 `master_20130105-20251228.csv`**、split は train≤2022 / valid 2023 / test 2024〜。
- 記録されている数値: valid AUC 0.7496（`ensemble.py:69` などにハードコード）。

**`optuna_transformer_pl.py`（PL 版、`models/transformer_pl_v2.pkl`）**
- 着順の全順列に対する PL NLL（ListMLE 型、`logcumsumexp`）。top-k 切り詰めなし、温度なし。
- **学習は PL 損失、モデル選択は複勝 AUC**（目的が一致していない）。
- 30 trials。記録: best valid AUC 0.7255、最終 valid 0.7231 / test 0.7304（`ROADMAP.md`）。

**`archive/experiments/evaluate_transformer_v6_stack.py` →
`reports/evaluate_transformer_v6_stack.{json,log}`（2026-05-25、既存）**
- test 2024-25（94,249行 / 6,908R）で次のように比較した。
  - transformer 単独: ◎top1 24.80% / ◎top3 **54.36%** / 勝ち馬の top5 内率 69.88%
  - v6 単独: 30.24% / **62.03%** / 78.20%
- min-max 正規化したスコアを α で配合したときの最良は α=0.85 で ◎top3 62.17%（+0.14pt）。
  **α を test で選んでおり**、ノイズの範囲内。
- 不採用の理由は torch のコスト（`REPORT_FOR_AI_FEEDBACK` §3.6）。

**比較として解釈できない理由（交絡）**:
- ① 特徴が 48本 対 120本
- ② 旧 master 対 master_v2
- ③ 選択指標が複勝 AUC
- ④ 同じ NN で文脈経路を外した対照が無い
- ⑤ 温度・較正の評価が無い

54% 対 62% の差のうち、何が「NN と GBDT の差」で、何が「特徴の差」で、
何が「文脈の効果」なのかは分解できない。

**旧アンサンブルでの扱い**: Optuna の配合重み（`optimize_ensemble_weights.py`）で
TransPL は 0.103。現行の `models/ensemble_weights*.json` には含まれておらず、
本番経路では死蔵されている（`docs/audit_20260615_full.md` ALG-07）。

**未検証の懸念（旧ラインのみ、本監査の範囲外）**:
`predict_weekly.predict_transformer_local`（1004-1013行付近）は、安定でない
`sort_values` と `groupby(sort=True)` でスコアを元の行に戻している。
一方、データセットは `groupby(sort=False)` で回しているので、馬とスコアの
対応がずれる可能性がある。54% という結果から見て大きくは壊れていないと推測するが、未確認。

**「Set Transformer」という名前の実装は存在しない**。E7 の根拠 `exp_havoc_m1.py` は
M1 の2体結合から作ったレース特徴8本を havoc_v1（GBM）に足した実験
（test AUC 0.62653→0.62234）である。

### 3.B ランキング損失（#4-11）

**v6（`optuna_v6_marks.py`）**
- LGBM `lambdarank`、`lambdarank_truncation_level=5`、ラベル `clip(6-着順,0,5)`。
- race 重み = 1+α·log1p(勝ち馬の単勝/100)、α=0.0308。
- Optuna の目的 = composite − 0.5×ECE_high_p（**valid=2023 で選択**）、515 trees。

**損失・ラベルの派生（`exp_v6_settings.py`）**
- `rank_xendcg`: ΔAUC +0.0023、Δpairwise +0.0029、Δ◎top3 +0.01pt
- binary、regression、ラベル変種（binwin / top3 / margin）、重み・ハイパラの変種は全て Gate 不合格
- 判定 `CEILING_CONFIRMED`

**その他**
- v9（truncation 3）: ◎top3 61.12%（−0.91pt）
- PRED-01（直接 binary / 3クラス）: −1.74pt・−1.55pt（CI が 0 をまたがない）
- PRED-03A（僅差 collapse ラベル）: Δ+0.00pt

**CatBoost YetiRank（native カテゴリ、`lab/train/train_catboost_marks_diag.py`）**:
◎top3 60.22% で v6 比 −1.81pt（`reports/catboost_marks_diag_v1.log`）。
損失（YetiRank 対 LambdaRank）とカテゴリ表現が同時に変わっているので、
カテゴリ単独の効果は分離できない。

**NN × race-level 損失（集合エンコーダではない例）**:
`lab/betting_lab/learn_bet_customloss.py` は、馬ごとの MLP にレース内 softmax の
「ポートフォリオ損失」をかけたもの。1回目は NaN、2回目は 2025 回収 82。
馬券層の実験であり、予測表現の検定ではない。

**differentiable sorting / Sinkhorn / Gumbel-Sinkhorn**: 実装ゼロ。

### 3.C 同時着順分布・券種確率（#12, 19）

**PL と較正**
- `pl_probs.py`: exact Harville。本番で稼働中。
- `joint_calibration_v6.py`: isotonic（**2023 で fit**）。ECE 生→較正後
  - 単勝 0.00630→0.00312
  - 馬連 0.00067→0.00036
  - 三連複 1.01e-4→8.1e-5
- 同スクリプトの "Phase 3（PL-NLL/ListMLE）" は **v6 では未実施**。

**分布の補正**
- **Stern λ**（`pl_stern.py`、λ=0.881 を 2023 で fit）: test の相対 ECE が悪化した
  （馬連 0.0609→0.1116、ワイド 0.031→0.0455、三連複 0.039→0.0556）。
- **Harville λ**（`data/harville_lambda.json`、349R で fit）: 三連複の過大予測は
  +29.2%→+8.2% に減ったが、ROI は 65.6→65.6 で不変。

**結合項**
- **M1/M2/順序/自由エネルギー**（`gate1_eval1.py`、`lab/physics_gates/gate2_3body.py`・
  `gate3_order.py`・`gate4_freeenergy.py`）
  - ECE: M0 1.05e-4 / M1 1.50e-4 / M2 1.41e-4 / M2j 1.27e-4。**全変種が素の PL より悪い**。
  - 順序項: α_pace +0.178 だが OOS は改善しない。
  - 本命崩壊の AUC: base 0.6555 / +freeE 0.6564 / +raw 0.6600。
- `crux_joint.py`: 市場の馬連 LL 3.343 が Harville 3.380 に勝つ。死亡。

**券種確率を直接学習した例**
- **`umaren_pair_v1`**（`build_umaren_pair_dataset.py`）
  - ペア単位の is_top2 を LightGBM で学習（423万ペア）。valid AUC 0.8152。
  - v6 との配合 α=0.8 で ROI 81.89% だが、**α を test で選んでおり CI も無い**。
  - 利用先は `bundle_signals.py:224` の表示用シグナルのみ。
- **三連複トリプル**: 学習ログが 0 byte で放棄。
- **`order_model_v1` / `fukusho_binary_v1`**: PRED-01 で FAIL。
  `order_model_v1` は学習スクリプト内で test を評価している（test 消費済み）。
- **trifecta v1**: リークで ROI 132% と報告されたもの。v2 は 69 / 58%。

### 3.D 相互作用の「中身」（#13, 14, 15）— 馬ごとの行に押し込む形は全て死亡

以下は特記が無い限り v6 harness（train≤2022 / valid 2023 / **test 2024-25 消費済み**）での結果。

| 系統 | ファイル | 何を足したか | 結果 |
|---|---|---|---|
| 展開構成 v1/v2 | `lab/experiments/exp_pace_interaction.py` | 逃げ頭数・前密度・単騎逃げ・自馬の脚質順位・脚質の場平均からの差・場平均 RPCI など10本 | v2: ΔAUC −0.00109、Δpair −0.00048、logloss +0.00317。**`pc_ka_vs_field` が gain 1位** |
| 展開 S2 / 場構造 S6 | `lab/features_dead/build_feats_serious.py` | S2 = n_front・pace_press・レース内脚質順位。S6 = 場の std / range / 最強馬との差 / z / entropy（14本） | S2: ΔAUC −0.00173。S6: ΔAUC −0.0002・Δpair −0.00092 |
| レース内相対の総当たり | `build_feats_1000.py` + `lab/train/train_v1000.py` | F1 = 76列 × (z / 順位pct / 平均差 / 最大差 / 最小差 / range) で **456/1039本** | F1 が **gain の 55%** を占め、◎top3 −0.25pt |
| 1400特徴の総当たり全体 | 100+300+1000 | 約 686/1436本（約48%）がレース内相対 | 採用 0（VOL3 E2） |
| `race_relative_feats.py` | v2 / v7 / v8 で本番系に投入 | z・順位（6列）→ 22列 + peer max/min | v2 は v1 と同等〜悪化。v7 は ROI −1.50pt。v8 はリークで退役 |
| 100 / 300特徴 | `analysis/test_100_feats.py`、`feat_exam_300*` | 43/99本・187/298本がレース内 groupby | 生き残りは kako5z のみ（**test 内の 50/50 分割で評価**）。120特徴の分類器に対しては消滅 |
| EXP05 の M3 | `analysis/mcond/exp05_market_residual_dev` | v6 スコアのレース内差・パーセンタイル | 2023 logloss 0.41303→0.41301（無価値） |
| keller | `lab/physics_gates/physics_keller_feats.py` | 場のペース硬度・単騎逃げ・drafting | v6 に積んで ΔAUC +0.00056 / Δpair −0.00048 |
| 2体結合 M1 | `gate1_eval1.py` + `corr_features.py` | both_front / both_closer / style_gap / both_front×pace / draw_gap を同時分布へ | 馬連 ECE 0.000503→0.000484（both-front 帯 0.00225→0.00152）、三連複の ECE は悪化、実馬券 OOS 黒字セル 0/36 |
| 3体 / 順序 / 自由エネルギー | `gate2_3body.py` ほか | 逃げ3頭×pace、差し3頭、位置重み | 上乗せ +0.07pt 以下。M0 を超える ECE は無し |
| havoc 読み出し | `lab/experiments/exp_havoc_m1.py` | M1 から作ったレース特徴8本 | gain 4.82%、ΔAUC −0.0042 |

**v6 本体が持つレース文脈**
- 場の規模・位置: 出走頭数、フルゲート頭数、枠番、馬番
- レース条件: 場所 / 距離 / 芝・ダ / 馬場 / 天気 / クラス / トラックコード / Ｒ / 開催 / コース区分 / 各種限定 / 重量種別
- **今走の他馬に対する相対値は持っていない**（`docs/SPEC/VOL1_SYSTEM.md:468`、race_relative=None）。
- EXP14 の DATA_AUDIT で、条件特徴は gain は低いが split 回数が多い
  （192-261回）ことが実測されている。

**解釈**: 「文脈情報は実在するが、馬ごとの行の特徴と深さ12・515本の木の
交互作用で既に再構成されている」と、「文脈がやっていることはレースの確信度の
言い換えにすぎない」の2つが同時に整合する。vNext の R1 が手作り特徴と違うのは、
φ と ρ を学習すること、そして文脈を各馬に戻すことである。ただし
**同じ情報源（同じ120入力）から作る表現**である点は変わらない。
これは `project_statmech_coupling_3body` の総括「v6 / オッズの周辺と同じ観測量から
作る物理は薄い」と同じ構図である。

### 3.E マルチタスク（#18）

実装ゼロ。関連するのは単一タスクの別モデルだけである。
- `fukusho_binary_v1`: top3 の2値
- `order_model_v1`: 1着 / 2-3着 / 4着以下の3クラス softmax。単一ヘッドの多クラスで、マルチタスクではない。

### 3.F カテゴリ表現（native categorical の監査）

- **v6 の扱い**
  - 28列を LabelEncoder で整数化し、**普通の数値として** LightGBM に渡している。
    `categorical_feature` は未使用で、欠損は −9999。
  - 高カーディナリティの列: 馬主(最新/仮想) 2548 / 生産者 2164 / 母父馬 1539 /
    種牡馬 1008 / 調教師コード 441 / 騎手コード 393。
- **リポジトリ全体**
  - LightGBM の `categorical_feature`、OrdinalEncoder、target / frequency 符号化、
    FeatureHasher は**ゼロ**。
  - 学習された埋め込みは RaceTransformer の中にしか無い。
- **比較実験**
  - 唯一の比較は CatBoost（native・YetiRank）対 v6（LabelEncoder・LambdaRank）の −1.81pt。
    損失が交絡している。
- **本番と学習の一致**
  - `category_normalize.py` が 芝・ダ の「ダート」→「ダ」など、serve 側の表記揺れを
    正規化している（2026-09-19 修正）。
  - 旧 transformer の serve 経路はこれを呼んでいない。
  - 集合モデルでカテゴリ埋め込みを使う場合も、**学習時と serve 時の表記正規化の
    一致が前提**になる。

---

## 4. 別仮説として扱うもの（同一レース内 interaction の検証済みとして数えない）

| 実験 | 何を扱ったか | 同一レース内 interaction と違う点 | vNext への含意 |
|---|---|---|---|
| **EXP12** 対戦ネットワーク | **過去の**対戦相手を個体追跡し、その相手の現在能力を見る（時系列グラフ・1-hop） | 「過去に誰と走ったか」であり「今誰と走るか」ではない | placebo で FAIL。改善は年齢×経験数の交絡で再現できた → **vNext の placebo も層別化が必須** |
| **EXP08** 当日馬場状態 | 同日に先に行われたレースから、時計・上がり・ペースの当日状態を推定（Kalman） | レースをまたぐ当日状態であり、出走馬の組み合わせではない | 効果量が要求の 1/12,500 → 小さい効果は統計的に有意でも経済的に無価値 |
| day_state_counting / 当日・翌日の track bias | 当日の枠・内外バイアス | 同上 | — |
| `race_level_exp{,_v2}` | **前走**レースのレベル | レースをまたぐ過去走の質 | +0.13 / +0.45pt で誤差の範囲。matched retrain の arm A は 61.48% |
| EXP02 / EXP03、ELO / Glicko | 動的能力・潜在状態 | 時系列の能力推定 | — |
| NNR | 馬ごとの kNN（レースをまたぐ） | 集合ではない | 同じ特徴の LGBM に負けた（AUC 0.7458 対 0.7584） |
| Harville λ / Stern / DR-01A | 分布族や補正パラメータ | 表現学習ではない | 「校正は改善するが選抜は不変」が3回再現 |
| havoc_v1 / 本命崩壊の検出 | レース単位の荒れを当てる target | 馬ごとのスコアではない | 検出はできる（AUC≈0.65）が市場に織り込み済み |

---

## 5. 既存文書の記述に関する訂正候補（本 Stage 0 では編集しない）

VOL3 §3.0.1 の遡及訂正ルール（打ち消し線で旧記述を残す）に従い、
レビュー後に反映する候補。

1. **VOL3 §2.2 E7 / §3.2、`ALL_IN_ONE.md` の対応行**
   - 現状の記述: 「Transformer / Set Transformer は汎化ゼロ（gain 4.8% 使うのに ΔAUC −0.004）」。
     根拠は `exp_havoc_m1.py`。
   - 問題: この根拠は M1 結合の GBM 読み出しであり、Transformer ではない。
   - 正しい Transformer の証拠は `evaluate_transformer_v6_stack`（54.36% 対 62.03%）。
     死因は「汎化ゼロ」ではなく「**交絡した比較（特徴・データ世代・選択指標）で劣後、
     未決着**」。
2. **`docs/research/UNEXPLORED_METHODS_AUDIT_20260920.md` §4.4、
   `NEXT_GEN_RESEARCH_PLAN_20260918.md` §6**
   - 問題: 「Set Transformer 予備検証」は実際には M1 読み出しだった。
     一方、RaceTransformer（BCE / PL の2本）が記載から漏れている。
3. **VOL3 §3.1 / §3.0.1（joint_m1 の訂正）**
   - 問題: 根拠として `analysis/_joint_m1_wedge.py` を挙げているが、これは市場オッズに基づく
     **複勝**の wedge セル表で、馬連の結合モデルとは別の実験。
   - 結論（全 ROI < 100%）は変わらないが、引用が2つの実験を混同している。
4. **`bundle_signals.py:224` の "ROI 81.89% @ test"（umaren_pair）**
   - 問題: α を test で選んだ値で、CI も無い。
5. **（範囲外・未検証）`predict_weekly.predict_transformer_local` のスコア対応ずれ疑い**、
   および `analysis/lupi_gate0_20260907/gate0_lupi.py` が v6 の encoder ではなく
   `astype("category")` でスコアリングしている疑い。

---

## 6. v6 の入力のうち、Race-as-a-Set 実験の前に扱いを決める必要がある列

R1 は「R0 と同じ入力」が前提だが、v6 の120特徴には必須テスト
（ID を記憶させない、as-of であること）とぶつかる列がある
（`models/unified_rank_v6.pkl` の `feature_cols` で実測）。

| 列 | 実態 | 集合モデル特有の問題 |
|---|---|---|
| `前走レースID(新)`・`前走レースID(新/馬番無)` | float の数値（例 2.012120e+17 / 2.012120e+15）。前者は馬番込みで、馬×前走ごとにほぼ一意 | 同じ場の中で値が一致する馬を検出できる → 「前走で同じレースを走った」という**ID ベースの馬間リンク**を集合モデルが拾えてしまう。GBDT の行モデルには無い経路 |
| `馬主(最新/仮想)`（CAT 2548） | `analysis/mcond/README.md` に「キャリア中に値が変わる馬 0.0% ＝ 最新値で上書き → 使用禁止」と記録されている | **as-of 違反**。ソースの段階で上書きされているので、未来削除不変性テストでは検出できない |
| `母馬` | master では文字列（繁殖牝馬名）。`feature_cols` にあるが encoders には無い | v6 が実際にどう扱っているか（定数化していないか）は**要確認** |
| `Ｒ`・`開催`・`前走日付` | レース条件・日付の数値 | 今走のレース ID ではないので使用可。ただし「レース ID を特徴に入れない」テストの許可リストに明記する |

---

## 7. vNext が主張できる新規性と事前確率

**新規と言えるもの（どれも実施例ゼロ）**:
1. 同じ120入力、同じ fold 手順、容量をそろえた NN の行モデル（R1-noctx）を対照に置き、
   **文脈の効果を「NN と GBDT の差」から切り離して**測ること
2. レース文脈の placebo（馬間 shuffle / 別レースとの交換 / 偽レース）
3. DeepSets 型の明示的プーリングで文脈を各馬に戻す構造
4. 同じアーキテクチャの中で pointwise 損失と race-level 損失（R1 対 R2）を比べること
5. 同じモデルの中でのカテゴリ表現の比較（LabelEncoder / frequency・hash / 学習埋め込み）

**新規ではないもの**:
- 手作りの展開・構成・レース内相対特徴
- PL の結合補正
- LambdaRank の派生
- 直接の binary / 多クラス target
- 旧 RaceTransformer の再学習

**事前確率**: レース文脈の中身は、少なくとも7系統の独立した検査
（§3.D）で「gain は取るが test では改善しない」を示してきた。独立に作られた
RaceTransformer も交絡付きとはいえ v6 に 7.67pt 劣後している。したがって
**Gate 1（R1 が R0 を両主要損失で上回る）の事前確率は低い**。
「文脈の効果が placebo を上回る」（Gate 5）は、Gate 1 より事前確率が高い。
NN の行モデルの弱さに影響されない比較だからである。
ただし `project_race_as_sample_settransformer_low_ev` の3件のヌル
（ランキング・波乱・3体）から見て、高いとは言えない。

関連: [[project_race_as_sample_settransformer_low_ev]] [[project_statmech_coupling_3body]]
[[project_direct_outcome_head_prior_art]] [[project_pred03a_neartie_collapse_result]]
[[project_feateng_v7_classmove_dead]] [[project_exp12_opponent_network]]
[[project_exp08_online_track_state]] [[project_exp14_regime_moe]]
[[project_unexplored_methods_audit_20260920]] [[project_dr01a_erratum_policy]]
