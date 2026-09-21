# EXP12 Stage 0 — 先行研究監査（対戦ネットワークによる対戦相手の質）

**作成日**: 2026-09-21　**状態**: Stage 0 監査のみ。実装・学習・バックテストなし。
2024・2025年の性能・ROIは未開封。

## ⚠️ 先に報告: 未追跡・未実行のscratchスクリプトを発見

監査中に、**gitに追跡されておらず（`git ls-files`で非該当）、実行結果も
一切保存されていない**2本のscratchスクリプトを発見した:

- `analysis/_tmp_oppstrength_gate.py`（更新日2026-06-25）: 「相手の強さ」＝
  フィールドの平均`前走補9`(補正タイム)で`opp_prev`/`opp_recent5`/
  `opp_best_beaten`（複勝圏内だった時のfield強度の最大値=「負かした相手の
  強さ」そのもの）を計算し、v6ベースライン・格13特徴に対するgate検定を
  設計している。
- `analysis/_tmp_elo_strict_gate.py`（更新日2026-06-25）: 逐次as-of ELO
  (`pre_elo`)、`field_elo`（レース平均事前ELO）、`opp_best_beaten_fe`
  （複勝圏内だった時のfield ELOの最大値）。

いずれも`reports/`・`analysis/`・`logs/`のどこにも対応するログ/JSONがなく、
**実行されたかどうか、結果がどうだったかは一切不明**。これは私が今回作成した
コードではなく、過去の別セッション（またはユーザー自身）による未完了の
作業と見られる。**Stage 0では実行しない**（実装禁止の範囲）が、Stage 1で
「対戦相手の質」特徴を設計する際、車輪の再発明を避けるため参照候補として
記録する。ユーザーには実行結果の有無・扱いを別途確認いただきたい。

---

## 1. ELO・Glicko・レースレベル・対戦相手の質: 既存研究は4系統、全て死亡確定

### 1.1 ELO（`lab/features_dead/build_elo_feats.py`、死亡）

4アーム（T1逐次/T2年次収束 × M1ペアワイズ margin加重/M2 PL型field同時更新）。
**M1はペアワイズに分解する**（個別対戦相手ごとの期待勝率を計算）が、
**M2は分解しない**（field全体のsoftmax対実着順分布で同時更新）。
出力特徴に`elo_*_level`（レース平均=「レースレベル」そのもの）・
`elo_*_vs`（相対値）を含む。

**検定結果（`reports/race_elo_exp.json`）**: **verdict="LOSE"**。事前登録
Gate（ΔAUC≥+0.008 かつ Δpairwise≥+0.005 かつ Δlogloss≤0）に対し、
4アーム全てΔAUC=[-0.00006, -0.0003, +0.00121, +0.00149]で不合格。

### 1.2 Glicko-2（`glicko_exp.py`/`lab/features_dead/build_glicko_feats.py`、死亡）

標準的なGlicko-2更新、1レース=1レーティング期間として**ペアワイズに分解**
して処理（`build_glicko_feats.py:8`）。μ（点推定）・RD（不確実性、
レース内z化）・μ-2RD（保守的レーティング）を出力。

**検定結果（`reports/glicko_exp.json`）**: baseline再現済み、3アーム全て
WIN gate不合格。**verdict="REDUNDANT"**——「RDはキャリア戦数
(kako5_race_count)の言い換えで冗長」。

### 1.3 レースレベル（field強度のスカラー集約、2件とも死亡）

- `race_level_feats.py`（現存、features_deadではない）: fieldの**後の**
  レースでのtop5率窓平均。**検定結果（`reports/race_level_exp.json`）**:
  verdict="NO_GAIN"（最良窓でも◎top3+0.13pt、ノイズ域）。
- `lab/features_dead/race_level_feats_v2.py`（死亡）: 着順・着差・補正
  タイムの加重strengthスコア版。**検定結果（`reports/race_level_exp_v2.json`）**:
  verdict="REDUNDANT"（重要度11位だが◎top3+0.45ptノイズ域）。

### 1.4 対戦相手の質・格の既存軽量シグナル

- **`race_relative_feats.py`は現存かつ本番稼働中の可能性がある**
  （`optuna_v6_marks.py`等の`race_relative_mode`バンドルパラメータ経由）。
  `{col}_race_max/_race_min`（レース内のjockey_fuku30/horse_fuku30/
  trainer_fuku30の最大・最小=「相手陣営の強さ」）+`peer_count`（頭数）を
  計算する。**騎手・調教師陣営レベルの粗い相手強度シグナルは既に
  v6の入力候補に含まれている可能性がある**（要Stage1で実際のv6バンドル
  設定を確認）。
- `grade_feats.py`（現存）: 15個の`gf_*`特徴。ただしこれは**馬自身の**
  クラス履歴（`gf_cur_crank`・`gf_prev_crank`・`gf_max_class5`等）であり、
  対戦相手の識別・強度ではない。既存の実現値はΔAUC(複勝)+0.0130
  （エコロアルバ事件を契機に配線、docstring記載）。EXP12の対象とは
  独立（馬自身の格 vs 対戦相手の識別）。

### 1.5 血統埋め込み（構造は似るが対戦ではなく血統、死亡）

`lab/features_dead/build_pedigree_emb.py`: 種牡馬・母父・母馬の親関係を
スパース行列化しTruncatedSVDで64次元化（gensimが使えずNode2Vec代替として
PPMI-SVDを採用、と明記）。**検定結果（`reports/race_ped_exp.json`）**:
verdict="REDUNDANT"（ΔAUC=[-0.00133,-0.00147,-0.00219]、label encodingの
言い換え）。**これは血統構造の埋め込みであり、対戦結果に基づくグラフでは
ない**——EXP12の対象と概念的に隣接するが別物。

### 1.6 対戦履歴ビルダーは存在するが表示専用

`matchup_history.py`（212行）は「誰と誰が対戦したか」の履歴構造
(`build_matchup_history()`)を構築するが、**他のどのファイルからも
importされておらず、予測特徴として未使用**（grep確認、既存の
`docs/research/UNEXPLORED_METHODS_AUDIT_20260920.md:133`でも同じ結論を
独立に確認済み: 「対戦履歴の表示専用ユーティリティで予測特徴には未使用」）。
グラフ構築のビルディングブロックとしてStage1で再利用できる可能性がある。

---

## 2. ネットワーク/グラフ特徴（PageRank・中心性・embedding・GNN）: 真に未着手

`PageRank|pagerank|centrality|中心性|node2vec|DeepWalk|graph neural|GNN|
message passing|networkx|torch_geometric|dgl`でリポジトリ全体
（*.py）を検索した結果、**PageRank・中心性・node2vec/DeepWalk・GNN/
message passing・networkx・GNNフレームワークのいずれも0件**。

既存の独立監査文書`docs/research/UNEXPLORED_METHODS_AUDIT_20260920.md:124-146`
（[[project_unexplored_methods_audit_20260920]]に対応）が同じ結論を
既に出している: 「馬・騎手・調教師・競馬場の異種時間グラフ・GNN・
message passing」を**E=完全未着手**と分類し、次の監査候補として推奨
している。**2つの独立した監査（今回のgrepと既存文書）が同じ結論に収束**
——EXP12のグラフ/GNN角度は本物の未踏領域と判定する。

---

## 3. 「強い相手に負けた」vs「弱い相手に勝った」を区別する既存特徴

**本番稼働中・検定済みの特徴としては存在しない**。ただし§0で報告した
2本の未追跡scratchスクリプトが、まさにこの発想（`opp_best_beaten`/
`opp_best_beaten_fe`=「複勝圏内だった時のfield強度・ELOの最大値」）を
コード化している。実行結果不明のため参照候補として記録するに留める。

「着差×対戦相手の質」を1つのスカラーに合成する指標（quality-adjusted
finish margin）は、上記のいずれにも存在しない。

---

## 4. 新馬・地方馬・海外馬・転入馬の扱い（項目6）

**master_v2は構造的にJRA(中央)専用**: `場所`列の値は東京・中山・阪神・
京都・中京・新潟・小倉・福島・函館・札幌の**10場のみ**（実データ確認済み）。
地方・海外のレースは行として一切存在しない。

**含意**: 地方・海外を挟んだ馬の履歴には「見えない前走」が生じる。
EXP01（`build_features.py:7-9`）は既にこの罠に対処済み——「masterの
直前行」が真の前走とは限らない（障害・地方を挟むケース）ため、
`前走日付`列との一致チェックで無効化する設計を採用している。**EXP12が
グラフのエッジ（誰と誰が対戦したか）を構築する際、同じ検証を踏襲する
必要がある**。

**新馬（未出走）の識別**: `build_horse_history.py:267-305`
(`resolve_2026_ped_ids`)が種牡馬+生年一致で名前を血統登録番号へ解決、
不一致は合成ID化。`serve_history_feats.py`のHistoryIndexが学習・配信
両方で同じ規約を使用（未知馬はn_prev=0・NaN履歴）。

---

## 5. 馬の識別子の安定性（項目7）

**血統登録番号は安定**（実データ確認: 1血統登録番号に複数馬名が紐づく
ケースは**0件**）。血統登録番号そのものが変化するという記述はコード上
見当たらない（未確認、リポジトリ内に言及なし）。

**馬名は再利用される、実測で確認・定量化済み**: 実データで**1馬名に
複数血統登録番号が紐づくケース478件**（アイアムイチバン・アイアン
ブランド等）を確認。独立して`analysis/mcond/exp05_forward_shadow/
horse_identity.py`も同じ「478件の名前衝突」を報告（`out/horse_state_2025.json`
のmeta、67,495頭中478件）——2つの独立した実測が完全一致。

**馬名単純一致での解決成功率は65.0%に留まる**ことが`MODEL_FREEZE.md:29-30`
に記録されており、種牡馬+生年での曖昧性解消により100%（287頭のテストで
実測）まで改善したと報告されている。**結論: グラフのノードキーには
必ず血統登録番号を使う。馬名を使ってはならない**（既に複数箇所で実証済みの
規律）。

---

## 6. EXP02との数理的な違い（項目9）

`analysis/mcond/exp02_dynamic_skill_dev/PRIOR_ART_AUDIT.md`の比較表
（「二者対戦へ分解するか」列）によれば、**EXP02は分解しない**——Weng-Lin PLの
尤度はfield全体を同時に扱う（ELO M2・Glickoの一部と同型）。EXP02自身の
REPORT.mdは「強い相手に善戦した価値を少しだけ測れた」と結論しているが、
これは「フィールド全体が強かったか」という集約情報であり、**「具体的に
どの馬に勝った/負けたか」という個体識別を伴う対戦相手情報ではない**。

**EXP12が追加できる可能性がある情報**: (a) 個体単位の対戦相手識別
（fieldの平均ではなく、specific rivalとの勝敗）、(b) 複数ホップの伝播
（この馬が負かした相手が、後に別の強豪に勝った、という推移的な強さの
伝播）——(a)は§1.1のELO M1アームが部分的に試みたが死亡、(b)は
`PageRank`/`centrality`/`GNN`のいずれも未着手（§2）。したがって
**EXP12の新規性は主に(b)の多段伝播にある**が、単段の(a)相当の信号が
既に弱いという実測（下記§7）を踏まえると、(b)へ進む前に単段の信号を
再確認する必要がある。

---

## 7. 実データでの一次診断: O1(単純な過去対戦相手平均)はO0へ追加情報を
持つか（項目10、O4以降へ進む合理性の判断材料）

Stage0の範囲内で、`o1_opponent_quality_check.py`により2023年development
のみで簡易diagnosticを実施した（2024・2025年は不使用、in-sample全数fit
のため参考値、正式なOOS検定はStage1で実施）:

- **設計**: 各馬の前走（`前走レースID(新/馬番無)`で自己結合、時点安全）の
  同一レース内・他馬のv6 raw score平均を`opponent_avg_quality_prev`として
  計算（v6は全期間master_v2に対しpredict()、train期間はin-sampleである
  ことに留意——対戦相手強度の代理としての用途に限定し、予測精度の主張には
  使わない）。
- **単純相関**: `corr(opponent_avg_quality_prev, fukusho_flag)=0.0328`、
  `corr(opponent_avg_quality_prev, v6_p_win_oof)=0.0238`（v6自身の予測との
  相関は低く、独立した情報を持つ可能性は残る）。
- **full-control**: v6 OOF確率+市場確率のみでロジスティック回帰した係数
  `{v6:0.2995, market:0.8068}`に対し、`opponent_avg_quality_prev`を追加
  すると`{v6:0.303, market:0.8024, opponent:0.059}`——**係数はゼロではないが
  市場係数(0.80)に比べ小さい**。in-sample loglossは0.43764→0.43739
  （**-0.00024、微小**）。

**判定**: O1は**小さいが非ゼロの信号**を示す。ただしこの効果量は、
§1.1-1.3で死亡した4つの独立した「field/対戦相手強度のスカラー集約」
実装（ELO・Glicko・レースレベルv1/v2）が示した効果量と同程度の小ささ
であり、**「スカラー集約」という発想自体が、この領域では既に天井に
近い**ことを示唆する。

**O4以降(2-hop/PageRank/embedding/GNN)へ進む合理性の判定**:
1. O1(単純平均)は小さいが非ゼロ——O2(既存ELO/Glicko、ただし死亡確定済み
   につき素朴な再検定は不要)・O3(個体単位の1-hop対戦相手特徴、時点安全に
   構築し直す)は最小限のコストで検証する価値がある。
2. しかし**O1の効果量の小ささは、O4以降(多段伝播・embedding・GNN)へ
   進む根拠としては弱い**。4つの独立した先行研究が「fieldの強さを
   スカラーに集約する」アプローチで軒並み小さい/ゼロの効果しか得られな
   かったという事実は、単純な集約では捉えられない**多段の構造情報**
   （EXP12固有の主張）が存在するかどうかを、O3で個体識別ベースの
   信号を確認してから判断すべきという結論を支持する。
3. **高度なモデル(embedding/GNN)を作ること自体を目的にしない**という
   ユーザー指示に従い、Stage1の最小反証実験は**まずO1-O3の範囲で
   full-controlを通過するかを検定し、通過した場合のみO4への昇格を
   検討する**設計とする。O4-O6を並行して作り込むことはしない。

---

関連: [[project_mcond_exp02_dynamic_skill]] [[project_mcond_exp01_choice_dev]]
[[project_unexplored_methods_audit_20260920]] [[project_longshot_weakness]]
[[project_exp11_hierarchical_bayes]] [[feedback_asof_population_definition]]
