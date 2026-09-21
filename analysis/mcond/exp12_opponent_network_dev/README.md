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

## 【2026-09-21】最終結論(ユーザー承認により結論範囲を確定): 完全終了

> **個々の過去対戦相手を対象日まで追跡した1-hop特徴は、2023 meeting-day
> CVで小さな改善を示したが、degree・年齢・経験量を維持したplaceboを
> 上回らず、対戦相手identity固有の情報とは確認できなかった。**

**PageRank(O4)・2-hopグラフ・graph embedding(O5)・GNN(O6)は「失敗」では
ない。未実装・未検証である**。O3が事前登録したGateを通過しなかったため、
現時点ではこれらへ進む実装根拠がないとして保留する。2024・2025年は
一切開封していない。詳細は`STAGE1_DESIGN.md`参照。

これにより、既存表データを使った大型予測研究(EXPシリーズ)は一時停止する。
詳細は`docs/research/RESEARCH_STOPLINE_20260921.md`参照。

## 現在の進捗(2026-09-21時点、最終)

| 段階 | 状態 | 成果物 |
|---|---|---|
| Stage 0(先行研究・データ監査) | 完了 | `PRIOR_ART_AUDIT.md` / `DATA_AUDIT.md` |
| Stage 1(O3最小反証) | **完了・Gate不合格でEXP12終了** | `SCRATCH_PRIOR_ART_AUDIT.md` / `STAGE1_DESIGN.md` / `spec.json` |
| O4以降 | **実施せず** | — |

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

なし。EXP12は終了。再走する場合は、少なくともplacebo検定を通過する
新しい対戦相手強度の定式化（例えば個体識別を維持しつつ交絡層をより
厳密に統制する設計）が必要だが、中止規律により本セッションでは実施しない。
