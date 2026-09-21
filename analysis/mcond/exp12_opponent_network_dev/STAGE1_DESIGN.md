# EXP12 Stage 1 — O3最小反証（対戦ネットワークによる対戦相手の質）

**作成日**: 2026-09-21　**最終改訂**: 2026-09-21（ユーザー承認により終了、
結論範囲を確定）。**状態**: **完全終了**（O3がGateで不合格、O4以降は
未実装・未検証のまま保留）。2024・2025年の性能・ROIは一切開封していない。

## 結論（この一文に厳密に限定する）

> **個々の過去対戦相手を対象日まで追跡した1-hop特徴は、2023 meeting-day
> CVで小さな改善を示したが、degree・年齢・経験量を維持したplaceboを
> 上回らず、対戦相手identity固有の情報とは確認できなかった。**

**O4(PageRank)・2-hopグラフ・graph embedding・GNNは「失敗」ではない**。
これらは**未実装・未検証**である。O3が事前登録したGateを通過しなかった
ため、現時点ではこれらへ進む実装根拠がないとして**保留**する（実装すれば
分かる、という話ではなく、根拠を欠いたまま高度化する理由がないという
判断）。

---

## 1. 未コミットscratchコードの先行監査

`SCRATCH_PRIOR_ART_AUDIT.md`参照。`analysis/_tmp_oppstrength_gate.py`・
`analysis/_tmp_elo_strict_gate.py`（いずれも未追跡、2026-06-25更新、
実行痕跡なし）を全文監査した。両方とも「field-strength平均のas-of
集約」というO1/O2寄りのアプローチで、EXP12のO3が要求する「対戦相手を
個体識別し、対象日までの事後的な能力変化を反映する」という核心機構を
欠くことを確認した。**両スクリプトはO3と等価ではない**。また両方とも
train=2023/test=2024-2025という設計だった（実行された形跡はないため
実害は未確認）が、本Stage1ではこの設計を採用せず、2023年development
のみのmeeting-day forward-chainingへ変更した。

## 2. O3固有差分の実装

`opponent_graph.py`で、単なる「過去レースの平均レベル」ではなく、対象馬hの
過去対戦相手oを血統登録番号で個体追跡し、対象日tの前までに判明した相手o
の**現在**能力（遭遇時点で凍結された値ではない）を使う10特徴量を実装した:
`unique_opponent_count`・`opponent_current_strength_mean`・`_max`・
`_top3_mean`・`beaten_opponent_strength_mean`・`lost_to_opponent_strength_mean`・
`strongest_beaten_opponent`・`recency_weighted_opponent_strength`・
`opponent_strength_dispersion`・`opponent_later_proved_strong_count`。

**核心の合成テストで検証済み**
（`test_opponent_graph.py::test_opponent_current_ability_not_frozen_at_encounter_time`）:
H1がDay1にH2(能力5)を破り、H2がDay2にH3を破って能力が9に上昇した場合、
Day3のH1の対戦相手特徴は遭遇時の5ではなく**現在の9**を反映することを確認。
これがscratchコードとの核心的な違い。

## 3. 時点安全なグラフ

- node key = 血統登録番号を**文字列のまま**使用（数値変換禁止）。実データで
  先頭ゼロ0件・1馬名に複数IDが紐づくケース478件（Stage0確認済み）を踏まえ、
  合成テストで文字列保持とID衝突回避を確認済み。
- 日初時点固定。同日内の先行レース結果は使わない
  （`test_same_day_earlier_race_not_used`で確認）。
- edgeは対象日より前のJRAレースのみ（master_v2は構造的にJRA限定、
  地方・海外は不在）。
- target raceの出走馬同士の当日edgeは追加しない
  （`test_target_race_coparticipants_are_not_edges`で確認）。
- external_history_gap: 前走日付とmaster_v2上の直前行日付の不一致で1
  （EXP01established patternを踏襲、合成テストで確認）。
- **未来削除不変性テスト（実データ、必須要件）**: 2025年を削除しても
  2024年の全特徴が一致することを実データで確認した
  （`verify_deletion_invariance_realdata.py`、46,752行、11特徴列
  （10特徴+external_history_gap）**全て不一致0件で完全一致**）。

## 4. 相手能力の基礎値

3方式を事前固定: **O3b（主方式・固定）= EXP02動的能力**
(`dyn_skill_mu`、`data/_research/mcond/exp02_features.parquet`から取得、
再学習なし)。O3a(既存ELO)・O3c(v6 as-of proxy)は感度分析用として設計・
実装したが、**O3自体がStage1のGateで不合格だったため、感度分析は実施
しなかった**（上位の判定が不合格の場合に感度分析を追加しても結論を覆す
ものではないため）。

## 5. 比較モデル

O0(v6+市場)・O1(単純な過去field-strength平均、Stage0の一次診断を再利用)・
O2(既存ELO`elo_T1M1_horse`+Glicko`g2_mu`+EXP02`dyn_skill_mu`)・
O3(O2+10特徴量)を構築した。**主比較はO3 vs O2**（O3 vs O0だけでは採用
しない、という事前指示を遵守）。

## 6. 代理変数対策

career starts・休養日数・年齢・頭数・クラス・人気・市場確率・v6確率・
entropy・missing_history_rate・unique_opponent_count・
external_history_gapを評価テーブルへ結合済み（full-control回帰の入力
として利用可能な状態）。

## 7. Placebo検定（★決定的FAIL）

**設計**: 層(年齢帯×出走回数帯、2023年単一年のため年度層別化は不要)内で
経験的プール平均・標準偏差から中心極限定理近似のブートストラップを行い、
実際のdegree(unique_opponent_count)は維持したまま対戦相手強度を置換。
1,000回。**簡略化を明記する**: 計算量制約のため、真の識別子完全置換
ではなく、opponent_current_strength_mean・beaten_opponent_strength_mean・
lost_to_opponent_strength_meanの3特徴（mean系）に限定した近似
（`placebo_test.py`のdocstringに詳細）。

**結果**:
```
real_improvement (logloss) = +0.00037
placebo 97.5%ile           = +0.00057
placebo mean                = +0.00045
判定: FAIL (実際の改善がplacebo分布の97.5%ileを超えない)
```

**解釈**: placebo平均(+0.00045)が実際の改善(+0.00037)を**上回っている**。
これは、観測された小さな"改善"の大部分が、対戦相手の個体識別情報とは
無関係に、年齢×経験数という層内の交絡だけで再現できてしまうことを
直接示す。O3の固有価値を支持しない、最も明確な反証。

## 8. Stage 1 最小反証（meeting-day forward-chaining CV）

2023年development、276 meeting_day、6ブロック5fold。

| モデル | logloss | Brier | PR-AUC |
|---|---|---|---|
| O0 | 0.21653 | 0.06001 | 0.30770 |
| O1 | 0.21645 | 0.06000 | 0.30749 |
| O2 | 0.21581 | 0.05991 | 0.30507 |
| O3 | 0.21541 | 0.05987 | 0.30157 |

O3-O2: logloss差=-0.00040(改善)、Brier差=-0.00004(微小改善)、
**PR-AUCはO3の方が悪化**(0.30507→0.30157)。開催日paired bootstrap:
logloss差CI95=[-0.00086,+0.00002]（**上限がわずかにゼロを超える**）。
fold別: 5fold中4foldでO3が同方向に優位。較正slope: O2=0.865→O3=0.867
（ほぼ変化なし）。

### 続行条件の判定（8条件）

| 条件 | 判定 |
|---|---|
| ①logloss改善が両方向で一貫 | ✅ PASS(4/5fold) |
| ②開催日bootstrap CI上限<0 | ❌ **FAIL**(上限+0.00002) |
| ③絶対改善≥0.0005 または相対改善≥0.1% | 相対改善0.185%でOR条件はPASS |
| ④Brierも改善 | ✅ PASS(ただし-0.00004と極小) |
| ⑤placebo 97.5%ileを超える | ❌ **FAIL(決定的)** |
| ⑥degree/count-onlyモデルに勝つ | 未実施(⑤のFAILにより打ち切り) |
| ⑦一部の馬・騎手・競馬場へ集中しない | 未実施(同上) |
| ⑧O3a/O3b/O3cで極端な符号逆転がない | 未実施(Stage1はO3bのみ評価) |

**8条件中2条件が明確に不合格**（②③④は境界線上またはOR条件でかろうじて
PASSする程度の弱さ）。ユーザー指定の停止規律「一つでも満たさなければ
2024・2025年を開封せずEXP12終了」に従い、**終了する**。

## 9. O4以降への条件

**実装しない（保留、失敗ではない）**。2-hop/PageRank(O4)・graph
embedding(O5)・GNN(O6)は未実装・未検証のままである。O3がGateを通過
しなかったため、現時点ではこれらへ進む実装根拠がない、という判断で
保留する。「単純だったから高度化すればよい」という解釈はしない
（ユーザー事前指示通り）。

## 10. 未実施の検証（正直な記録）

以下はplacebo/bootstrapの決定的FAILを受けて実施しなかった:
- degree/count-onlyモデル(次数・対戦数のみ)との比較
- 特定の馬・騎手・競馬場への効果集中確認
- O3a(ELO基礎値)・O3c(v6 as-of基礎値)感度分析

これらを実施しても、上位のGate（placebo検定）が不合格である以上、
続行の結論を覆すものではないと判断した。

---

## まとめ（結論の範囲を厳密に限定）

- 未追跡scratchコード2本を全文監査し、O3とは非等価と確認した。
- O3を「遭遇時に凍結された相手強度」ではなく「対象日まで追跡した相手の
  現在の能力」として実装し、この核心差分を合成テストで検証した。
- 時点安全なas-ofグラフ（日初固定・文字列ID・external_history_gap）を
  実装し、**実データで未来削除不変性を完全一致(不一致0件)で確認**した。
- **確定した結論**: 個々の過去対戦相手を対象日まで追跡した1-hop特徴
  (O3)は、2023 meeting-day CVで小さな改善を示したが、degree・年齢・
  経験量を維持したplaceboを上回らず、対戦相手identity固有の情報とは
  確認できなかった。
- **PageRank(O4)・2-hopグラフ・graph embedding(O5)・GNN(O6)は未実装・
  未検証であり、「失敗」ではない**。O3が事前Gateを通らなかったため、
  現時点では実装根拠なしとして保留する。
- 2024・2025年は一切開封していない。

関連: [[project_exp12_opponent_network]] [[project_mcond_exp02_dynamic_skill]]
[[project_exp10_downside_risk]] [[project_exp11_hierarchical_bayes]]
[[feedback_asof_population_definition]] [[docs/research/RESEARCH_STOPLINE_20260921]]
