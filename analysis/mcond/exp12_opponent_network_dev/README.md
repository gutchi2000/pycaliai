# EXP12 — 対戦ネットワークによる対戦相手の質

## 目的

馬同士の対戦履歴を二部グラフ（馬×レース）として捉え、対戦相手の識別・
強度・多段伝播が、v6+市場+既存特徴を超える情報を持つかを検証する。

**主眼**: 高度なモデル（embedding・GNN）を作ること自体を目的にしない。
O1（単純平均）→O2（既存ELO/Glicko）→O3（1-hop個体識別）→O4（2-hop/
PageRank）→O5（embedding）→O6（GNN）の順に、各段階でfull-control gateを
通過した場合のみ次へ進む。

## 絶対条件

EXP01-11・v6・compute_bets.pyは変更しない。新規作業は
`analysis/mcond/exp12_opponent_network_dev/`へ限定する。

## ⚠️ 未追跡・未実行のscratchスクリプトを発見

`analysis/_tmp_oppstrength_gate.py`・`analysis/_tmp_elo_strict_gate.py`
（いずれもgit未追跡、2026-06-25更新、実行結果の保存なし）が、EXP12と
ほぼ同じ発想（`opp_best_beaten`=「負かした相手の強さ」）を既にコード化
している。詳細は`PRIOR_ART_AUDIT.md`冒頭参照。過去の未完了作業と見られ、
Stage 0では実行していない。

## 現在の進捗(2026-09-21時点)

| 段階 | 状態 | 成果物 |
|---|---|---|
| Stage 0(先行研究・データ監査) | **完了** | `PRIOR_ART_AUDIT.md` / `DATA_AUDIT.md` |
| Stage 1以降 | **未着手** | ユーザー承認待ち |

## Stage 0 の要点

- **先行研究は4系統、全て死亡確定**: ELO(`build_elo_feats.py`、verdict
  "LOSE")・Glicko-2(`glicko_exp.py`、verdict "REDUNDANT")・レースレベル
  v1/v2(`race_level_feats.py`/`_v2.py`、verdict "NO_GAIN"/"REDUNDANT")。
  いずれも「fieldの強さをスカラーに集約する」アプローチ。
- **グラフ/PageRank/embedding/GNNは真に未着手**: grep 0件、既存の独立監査
  文書(`docs/research/UNEXPLORED_METHODS_AUDIT_20260920.md`)も同じ結論。
- **一次診断(2023年developmentのみ、in-sample参考値)**: 単純な過去対戦
  相手平均(O1)はv6+市場を統制した後も小さいが非ゼロの信号を示す
  (標準化係数0.059、logloss改善-0.00024)。ただし先行4系統の死亡実績と
  同程度の小ささであり、O4以降(多段伝播・GNN)へ進む根拠としては弱い。
  まずO3(個体識別ベースの1-hop、時点安全に厳密化)から着手すべきと判断。
- **馬の識別子**: 血統登録番号は安定(1IDに複数馬名=0件)、馬名は
  再利用される(1馬名に複数ID=478件、実測)。グラフのノードキーは必ず
  血統登録番号を使う。
- **地方・海外馬**: master_v2は構造的にJRA10場のみ、地方・海外を挟んだ
  履歴には「見えない前走」が生じる。EXP01が確立した前走日付検証を
  グラフ構築でも踏襲する必要がある。

詳細は`PRIOR_ART_AUDIT.md`/`DATA_AUDIT.md`参照。

## 次の一手

ユーザー承認待ち。Stage1へ進む場合はO3(1-hop個体識別特徴、EXP01の前走
日付検証を踏襲した厳密as-of実装)から着手し、full-control gateを通過
した場合のみO4以降を検討する。
