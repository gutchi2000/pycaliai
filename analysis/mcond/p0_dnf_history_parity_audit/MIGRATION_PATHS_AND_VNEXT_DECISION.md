# 二つの移行経路と corrected-vNext 判断基準

**状態**: 方針文書。本番master/model・Optuna・EXP再実行はまだ行っていない。

---

## 5. 二つの移行経路

### 5-1. Legacy-compatible経路（現在v6を使い続ける間）

**目的**: 現行`unified_rank_v6.pkl`を維持しながら、live serveの実際のserve skew
（`SEMANTIC_VS_PARITY_CLASSIFICATION.md`の**B分類**のみ）を是正するかを判定する。

**対象はBのみ**: kako5_race_count・kako5_same_td_ratio・kako5_same_dist_ratio・
kako5_same_place_ratioの4特徴。この4特徴は同一の正しい入力を与えてもtrain相当と
serveのコードが異なる値を返す、確認済みの真のserve skewである。

**対象外（A・D・E）**: course/jockey6特徴（A、train=serveで両方誤り）・
course/jockey2026分（D、未計測）・kako5残り9特徴（E、未計測）は、
「semanticには正しくても」現行v6へ**突然新定義の値を入力しない**。
現行モデルは既存の（誤った）分布で学習されているため、A・D・Eの値だけを
先に「正しく」しても、学習時の分布と乖離した未知の入力になり、モデルの
予測が不安定化するリスクがある。

**Legacy-compatible経路での判断手順**:
1. Bの4特徴について、現行live serve値と「もし正しく実装した場合の値」の
   実際の乖離（実データでの発生率・平均乖離幅）を定量化する（現時点では
   `train_serve_same_input_parity.py`の合成シナリオでコードレベルの相違は
   確認済みだが、実データでの発生頻度・規模は未計測）。
2. 乖離が実運用上有意な規模であれば、**serve側`build_from_kako5()`のみ**を
   修正する（`build_from_master()`や`compute_row_feats()`には触れない）。
3. 修正後の値を現行v6へ入力した場合の予測変化を、本番接続前にshadow環境で
   確認する（本監査のshadow手法を流用可能）。
4. 前向きcanary（一定期間、新旧の予測を並行出力して乖離を監視）を経てから
   本番切替を判断する。

**Legacy-compatible経路ではA・D・Eには手を付けない**——これらは
corrected-vNext側の対象。

### 5-2. Corrected-vNext経路

正しい意味定義（`DNF_SEMANTIC_SPEC.md`）で、以下を**一組のバージョン管理単位**
として扱う。v6とは混在させない:

- corrected training master（本監査の`shadow/corrected_19features.parquet`が
  出発点、ただし本番master置換ではなく別バージョンとして管理）
- corrected offline replay（PLソフトマックス分母にDNF馬を正しく含める評価
  パイプライン——本監査`dnf_inclusive_baseline.py`が出発点）
- corrected serve builder（`build_from_kako5()`・`compute_row_feats()`が
  参照するデータソース自体をDNF_SEMANTIC_SPEC.md準拠に是正したもの）
- corrected model（corrected training masterで学習したモデル。本監査の
  `models/unified_rank_v6_shadow_corrected.pkl`が最初のablation版、
  正式なvNextはOptuna再探索を経て別途構築）

これら4点セットは**v6と独立したバージョン**として管理し、本番切替の判断が
下るまでv6と混在させない（例: corrected featureをv6へ入力しない、
v6の予測とcorrected modelの予測を無断で合成しない）。

---

## 6. DNF-inclusive baseline（既存指標の計測訂正）

`dnf_inclusive_baseline.py`の実行結果は別途`out/dnf_inclusive_baseline.json`
および本レポート更新版に記載する。current v6のモデル・ハイパーパラメータは
一切変更していない、既存の「finisher-only」測定方法との比較のための
計測手法の追加のみ。

---

## 7. corrected-vNextへ進む判断

**現時点のsame-hyperparameter shadow retrainは予測性能悪化として記録する**
（◎top3 -1.45pt、NDCG@5 -0.0020、valid=2023）。この結果を「ハイパー
パラメータを再調整すれば解消する」という前提で軽視しない。

**次に進む条件（全て満たすまでOptunaは実行しない）**:
1. 実際のtrain/serve skewが定量化済み（現時点: コードレベルの相違は
   確認済みだが実データでの発生規模は未計測、B分類特徴について優先実施）
2. corrected masterとserveが完全parity（Legacy-compatible経路のB是正、
   および将来のcorrected-vNext serve builderの構築後に確認）
3. DNF-inclusive baselineが完成（本監査で着手、`dnf_inclusive_baseline.json`
   参照）
4. corrected modelの較正改善が実用上意味を持つ（本監査のshadow retrainで
   ECEは改善したが、実運用上のインパクト——見送り閾値・EV計算等への影響——は
   未評価）
5. 予測悪化がデータ定義移行直後の一時的問題か検証する合理的根拠がある
   （現時点では未検証。同一ハイパーパラメータでの悪化がハイパーパラメータの
   ミスマッチによるものか、真にcorrected featureの情報量が現行分布より
   劣るのかを区別する追加検証が必要）

**条件を満たした場合でも、最初の正式な一次判定はOptunaではなく**:
同一ハイパーパラメータ・同一seed・同一学習期間・同一評価期間で
corrected definitionのみを変更した結果（=本監査の`shadow_retrain_and_compare.py`
の結果そのもの）とする。

**その結果が悪い場合、意味的に正しいという理由だけでproduction modelを
差し替えない**。corrected feature定義は次世代モデル（corrected-vNext）用
として保存し、**現行v6はlegacy-compatible定義で維持する**
（§5-1のBのみ是正、それ以外は現状維持）。

関連: `DNF_SEMANTIC_SPEC.md`, `SEMANTIC_VS_PARITY_CLASSIFICATION.md`,
`out/v6_contract_manifest.json`, `out/model_impact_comparison.json`
