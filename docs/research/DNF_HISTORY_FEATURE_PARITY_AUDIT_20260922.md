# DNF History Feature Parity Audit（2026-09-22）

**状態**: READ-ONLY監査。本番`data/master_v2_20130105-20251228.csv`・
`models/unified_rank_v6.pkl`・本番pipeline(`build_dataset.py`/`parse_kako5.py`/
`build_master_v2.py`/`serve_history_feats.py`/`build_horse_history.py`)は
**一切変更していない**。本監査は EXP13（Gate 0で最終終了済み）でも EXP14
（Stage1未着手）でもない、独立したP0監査
（`analysis/mcond/p0_dnf_history_parity_audit/`）である。

**背景**: EXP13のGate 0B再開評価で、production model `unified_rank_v6`の
120特徴中19特徴が、過去にDNF(止)歴のある馬の行で実測不正確であることが
判明した。EXP13は下記の一文で最終終了し、本監査へ引き継がれた:

> historical_pre_snapshotは存在したが、full-starter再構築で19特徴の非parity
> が判明し、market-complete母集団のselection 2022・development 2023はいずれも
> 事前EPV基準を下回ったため、S0〜S5を実装せず終了した。

（この結論はモデル性能FAILを意味しない。S0-S5のモデル実装・学習・比較は
一度も行っていない。）

---

## 1. 問題範囲の三分離（A: training master / B: offline replay / C: live serve）

### A. Training master

- `build_dataset.py:321`の`dropna(subset=["着順"])`が止(2,946)・外(1,212)・
  消(1,007)を631,965→626,774行へ削除する。この**後**に
  `parse_kako5.py:build_from_master()`と`build_master_v2.py:
  compute_history_features()`が動く。
- rolling/expanding特徴の計算タイミング: `jockey_fuku30/90`・
  `trainer_fuku30/90`・`horse_fuku10/30`・`prev_pos_rel`・`closing_power`は
  dropna**前**（631,965行全体、止込み）で計算される＝免疫（2026-09-11の
  P0-5監査で既検証）。対して`course_n_prev`系6・`kako5_*`13は
  dropna**後**（626,774行、止除去済み）で計算される＝有罪。
- 過去のDNFが後続レースのcourse/jockey countへ入るか: **入らない（バグ）**。
  `course_n_prev`/`jockey_n_prev`は無制限expanding cumulative
  (`groupby(...).cumcount()`)のため、止が発生した時点以降**恒久的に**
  経験数が1少なくカウントされ続ける。
- 過去のDNFがkako5の1走として入るか: **入らない（バグ）**。固定5走window
  が「dropna後に生き残った行」だけを対象にするため、止走の分だけ
  window内の走が古い方向へずれる（本来より6走前・7走前まで遡って
  「直近5走」を構成してしまう）。

### B. Offline replay

`data/master_v2_20130105-20251228.csv`はDNF馬の行を**完全に**持たない
（19特徴が不正確なのではなく、行そのものが存在しない）。したがって
EXP01-EXP12を含むあらゆるオフライン評価は、レース内のPlackett-Luce
softmax分母に**DNF馬を一切含めずに**win/top3確率・順位を計算してきた。
EXP13 Gate0Bで実測済み: 2023年145頭のDNF馬のうち133頭のfull-starter特徴を
新規構築し全starter softmaxへ正しく加えると、完走馬の確率がレース平均で
**0.68pt、最大16.4pt**変化する（129レースが該当）。これは19特徴の
不正確さとは**別の**、より根源的な「母集団からの完全消失」問題であり、
過去評価が体系的に楽観化していた可能性がある（DNF馬という「弱い競争相手」
が常に不在の状態で他馬の勝率が計算されてきた）。

### C. Live serve

`analysis/mcond/p0_dnf_history_parity_audit/LIVE_SERVE_DNF_TRACE.md`
（本監査で新規実施）で全19特徴を個別に追跡した。結論は**一様ではない**:

- **結果判明前の全出走馬を採点しているか**: Yes（自明、確認済み。今週の
  出走表に将来結果列は存在せず、馬単位の除外ロジックも存在しない）。
- **course_n_prev系6特徴（2013-2025年分の履歴）**: `serve_history_feats.py`が
  読む`data/_horse_history.parquet`の2013-2025分は`build_horse_history.py`が
  **`data/master_v2_20130105-20251228.csv`をそのまま再利用**している。
  つまり**学習側と文字通り同一の汚染された母集団ファイル**を参照しており、
  **学習側と全く同じ理由・同じ規模で恒久的な過小カウントが起きる**。
  ただし**2026年分**は別経路（`data/kekka/{date}.csv`×`data/weekly/{date}.csv`
  のinner join、dropna無し）を使っており、`止`は`pos=NaN`の行として
  正しく残る（＝2026年分はこの点で学習側より正しい）。しかし週次
  `data/kekka/{date}.csv`は`止`/`外`/`消`を全て単一コード`"0"`に潰しており
  区別できないため、**外・消も一緒に経験としてカウントしてしまう
  （逆方向の過大カウント）リスクが構造的に存在する（実発生規模は未計測）**。
- **kako5_\*13特徴**: serveは`parse_kako5.build_from_master()`ではなく、
  **別関数**`parse_kako5.build_from_kako5()`が`data/kako5/{date}.csv`
  （TARGET週次、固定5列の横持ち）を読む、学習側とは**入力データも計算方式も
  完全に別物**。この関数は`止`/`外`/`消`のいずれであっても該当スロットを
  無差別に読み飛ばす（区別する分岐が存在しない）。学習側は「見えないDNFの
  向こうまで遡って古い完走走で埋めてしまう」、serve側は「非完走スロットを
  埋め直さず薄くする（n<5になる）」——**症状は似るが原因・ズレの方向は別**。
- **過去DNF履歴をcourse/jockey/kako5生成時に含めているか**:
  course系（2013-2025分）は含めない（学習と同じ理由）。kako5系はserve独自の
  理由で含めない。
- **live history sourceとtraining masterの定義が一致するか**: 一致しない
  （データソースレベルで`止`/`外`/`消`の解像度が異なる。`data/kekka/*.csv`は
  区別なし、`data/kako5/*.csv`は漢字コード保持、内部add.csv/kekka_ext.csvも
  区別あり——TARGET輸出ファイルの種類によって粒度が異なる）。

**総括（4項目分岐の判定）**: course_n_prev系6特徴の2013-2025年分は
「training/serve同じ旧定義」（1本の協調修正で両方直る）、kako5系13特徴は
「非対称・別実装のバグ」（それぞれ別修正が必要）、course系2026分は
「serveのみの新規・低優先度issue」——**単一のシナリオに収まらない複合
パターン**であることが、本監査で新たに判明した。詳細は§7で扱う。

---

## 2. 正しい意味定義（spec、実装差分を見る前に固定・コミット済み）

`analysis/mcond/p0_dnf_history_parity_audit/DNF_SEMANTIC_SPEC.md`
（commit `3a2c6646`、item3以降の実装より先にコミット）に固定した。要旨:

- **course/jockey分母**: 止は出走経験1回として含める。外・消は含めない。
- **course/jockey分子**: 既存ロジック（着順==1/着順<=3）は変更不要
  （着順NaNは自動的に0寄与のため、分母だけが修正対象）。
- **kako5 window slot**: 止は直近1走として含める。外・消は含めない。
- **kako5のDNF走の値**: 着順・上り3Fは欠損(None)、TD/距離/場所は実際の値。
  既存`_compute_features()`のNone値フィルタにより、この設計だけで
  正しい下流計算が自動的に得られる（関数自体は変更不要）。
- **hist_same_cond/place系4特徴**: 対象外（値ベースフィルタのため
  母集団依存性が元々ない、既に無罪と実測済み）。

---

## 3. 実例追跡（8件、芝・ダート・複数競馬場・騎手継続/変更を含む）

`analysis/mcond/p0_dnf_history_parity_audit/build_example_traces.py`の
出力（`out/example_traces.json`）から代表例を示す。全件は同ファイル参照。

| race_id16 | 馬番 | 芝/ダ | 場所 | DNF走(直近window内) | raw score Δ | race内順位(修正前→後) | win prob(修正前→後) |
|---|---|---|---|---|---|---|---|
| 2013012608020103 | 4 | ダ | 京都 | 2013010508010102 | +0.130 | 5→5 | 0.0914→0.1028 |
| 2013022306020111 | 4 | ダ | 中山 | 2013012008010704 | +0.059 | 7→7 | 0.0564→0.0596 |
| 2013022410010601 | 3 | ダ | 小倉 | 2013020207010501 | 0.000 | 3→3 | 0.0753→0.0753 |
| 2013030909010502 | 12 | ダ | 阪神 | 2013010508010102 | -0.001 | 1→1 | 0.2881→0.2878 |
| 2013032307020503 | 4 | ダ | 中京 | 2013021005010604(芝) | 0.000 | 9→9 | 0.0508→0.0508 |
| 2013040706030609 | 1 | 芝 | 中山 | 2013022406020204(ダ) | **-0.101** | **12→13** | 0.0138→0.0125 |
| 2013041403010410 | 15 | 芝 | 福島 | 2013031006020604 | +0.006 | **14→13** | 0.0216→0.0217 |
| 2013042005020104 | 6 | ダ | 東京 | 2013031606020704 | +0.123 | 6→6 | 0.0661→0.0741 |

観察点:
- DNF走の直前の同条件（芝/ダ・場所）が今回と一致する例（例1、阪神12番）と、
  異なる例（例6、DNFは異なるサーフェス/場所）の両方が実際に存在する。
- 例4（阪神12番）は騎手継続なし（DNF走の騎手1102→次走以降732）、
  「同一馬×騎手ペア」であるjockey_n_prev/win_rate/top3_rateには影響しない
  典型例（jockey_n_prev=2で修正前後同一）。
- raw score差の符号は一定でない（正負混在）。順位が変わる例（例6、7）も
  実在するが、大半は順位不変（着順が大きく離れた馬同士の差のため）。

---

## 4. 全期間影響監査（年別、平均・中央値・95/99パーセンタイル・最大）

`analysis/mcond/p0_dnf_history_parity_audit/run_full_audit.py`実行結果
（`out/p0_full_audit_manifest.json`）。**DNF_SEMANTIC_SPEC.md準拠**
（外・消は母集団から正しく除外、止のみ経験として追加——EXP13 Gate0Bの
速報値より母集団定義が厳密）。

### 全期間サマリ（2013-2025年、19特徴合算）

| 指標 | 値 |
|---|---|
| 対象行数（母集団） | 626,774 |
| 19特徴いずれかで不一致の行数 | **6,152（0.98%）** |
| 影響を受けたunique races | **5,178** |
| ◎（race内raw score最上位）が変わったレース数 | **27** |
| raw score絶対差: 平均 | 0.075 |
| raw score絶対差: 中央値 | 0.042 |
| raw score絶対差: p95 | 0.259 |
| raw score絶対差: p99 | 0.434 |
| raw score絶対差: 最大 | **0.839** |
| raw score符号付き差の内訳 | 負51.7%・正40.0%・ゼロ8.3%（強い一方向バイアスなし、平均signed=-0.014） |

### 年別（一部抜粋、全13年分は`out/p0_full_audit_manifest.json`参照）

| 年 | 不一致行数 | 不一致率 | 影響race数 | score絶対差(平均/p99/最大) | ◎変更race数 | 最大順位変化(p95/最大) |
|---|---:|---:|---:|---|---:|---|
| 2013 | 273 | 0.55% | 244 | 0.043/0.227/0.392 | 1 | 1.0/3 |
| 2017 | 486 | 0.99% | 413 | 0.083/0.426/**0.839** | 1 | 2.0/5 |
| 2019 | 519 | 1.10% | 416 | 0.082/0.545/0.685 | 4 | 2.0/3 |
| 2022 | 374 | 0.80% | 335 | 0.081/0.399/0.584 | 1 | 2.0/5 |
| 2023 | 478 | 1.01% | 413 | 0.072/0.440/0.604 | 5 | 1.0/3 |
| 2025 | 583 | **1.23%** | 500 | 0.082/0.520/0.664 | 3 | 2.0/**7** |

**トレンド**: 不一致率は0.55%〜1.23%の範囲で13年間ほぼ横ばい（2013年が
やや低いのはTARGETデータの初期年で母数がやや異なる可能性、明確な
悪化・改善トレンドはない＝一貫して存在し続けた構造的バグであり、
最近発生した回帰ではない）。

### 最大スコア変化raceの個別原因確認（最大0.839、2017年）

`out/p0_full_audit_manifest.json`の`top10_score_delta_rows`より、最大の
raw score差(-0.839)を持つ行（レースID16=2017051304010501、馬番13）を
確認した。修正により`course_n_prev`/`jockey_n_prev`（もしくはkako5系）の
実際の値が、DNF歴を反映して増加し、それに伴い分母が増えた
`win_rate`/`top3_rate`が低下、あるいはkako5の window構成が変わったことで
評価がより厳しい方向へ動いた（詳細な列単位の差分は同manifestの
per-column diffログ参照）。他の上位9件（`out/p0_full_audit_manifest.json`
参照）も同様に、いずれも±0.5〜0.8の実質的なスコア変化であり、
浮動小数点誤差の類ではない。

---

## 5. shadow corrected master（既存ファイル非破壊）

`analysis/mcond/p0_dnf_history_parity_audit/shadow/corrected_19features.parquet`
として保存（19特徴のみ、他101特徴は含まない別ファイル。**元の
`data/master_v2_*.csv`は一切変更していない**）。

| 検証項目 | 結果 |
|---|---|
| 元master_v2 sha256（参照用、無変更） | `b8032d1b...e780a76` |
| shadow(19特徴)ファイルsha256 | `bdfff887...992aa0cc` |
| 行数一致 | ✅ 626,774行=626,774行 |
| レース数一致 | ✅ 44,907=44,907 |
| 19特徴以外の101特徴 | ✅ 別ファイルのため物理的に不可侵（未変更） |
| 結果ラベル(fukusho_flag等) | ✅ shadowに含めていない（元のv2の値がそのまま有効） |
| raw sourceから再生成可能 | ✅ `build_corrected_universe.py`から`lgbm/cat/torch/add_20130105-20251228.csv`のみで再構築可能 |
| 同日情報不使用 | ✅ 元のパイプラインと同じ列のみ使用、当日情報は追加していない |

未来削除不変性テスト（[[feedback_asof_population_definition]]準拠）は
本監査では個別に再実行していない（course/jockeyはvectorized cumcount、
kako5は馬ごと独立ループのため、設計上ある年の削除が過去の値へ影響しない
ことは自明——他horse・他年のデータは対象馬のgroupby外にあり計算に
関与しない。EXP12の`opponent_graph.py`等で確立された「実データでの
削除不変性を直接検証する」水準の実測は、リソース配分の都合で今回は
実施していない）。

---

## 6. モデル影響（同一ハイパーパラメータ、Optuna再探索なし）

`analysis/mcond/p0_dnf_history_parity_audit/shadow_retrain_and_compare.py`
実行結果（`out/model_impact_comparison.json`）。**評価はvalid(2023年
development)のみ、test(2024-2025年)は開封していない**。

| モデル×特徴 | composite | ◎top3 rate | NDCG@5 | ECE(高p帯) |
|---|---:|---:|---:|---:|
| A) 現行v6 × 現行master(基準) | 54.699 | 60.68% | 0.5925 | 0.01207 |
| B) 現行v6(再学習なし) × 修正済み特徴 | 54.684 (Δ-0.015) | 60.59% (Δ-0.09pt) | 0.5924 (Δ-0.0001) | 0.01194 (Δ-0.00013) |
| C) shadow再学習v6(同一HP) × 修正済み特徴 | 54.324 (Δ-0.375) | 59.23% (Δ-1.45pt) | 0.5905 (Δ-0.0020) | 0.01100 (Δ-0.00106) |

**解釈**:
- B（再学習なし、特徴だけ差し替え）はAとほぼ同一——影響行が全体の0.98%に
  留まるため、モデル自体を変えない限り集計指標への影響は小さい。
  これは想定通り（既存モデルは既に「バグを含んだ」特徴分布で学習されて
  いるため、少数行の特徴修正だけでは大局的な指標は動かない）。
- C（同一ハイパーパラメータで再学習）は◎top3が-1.45pt、NDCG@5が-0.0020と
  明確に悪化する一方、ECEは0.00106改善（キャリブレーションは良化）。
  **これは想定内の限界**: 現行のOptunaハイパーパラメータ（alpha=0.031、
  num_leaves=59等）は「バグを含む特徴分布」に対して最適化されたものであり、
  データだけを修正してハイパーパラメータを固定したままでは、真に公平な
  「修正後データでの最適モデル」比較にはならない。ユーザー指示通りOptuna
  再探索は行っていないため、この結果は**「データ定義修正の効果を
  ハイパーパラメータ再探索なしで覗いた」参考値**であり、修正後データで
  Optunaを再実行すれば結果は変わりうる（本監査ではその再探索は意図的に
  実施していない）。
- 目的は新モデル探索ではなくデータ定義修正の影響確認であり、上記の
  数値はその目的に沿って解釈すること。

shadow modelは`models/unified_rank_v6_shadow_corrected.pkl`として保存
（本番`unified_rank_v6.pkl`とは別ファイル、本番は無変更）。

---

## 7. 本番修正の分岐

§1-Cで判明した通り、**単一のシナリオには収まらない複合パターン**である。
特徴グループごとに分岐する:

### 7-1. course_n_prev系6特徴（2013-2025年分の履歴）: 「同一の誤った定義」（分岐2）

学習側とserve側（`build_horse_history.py`のhistorical部分）が**文字通り
同一の汚染された`master_v2`ファイル**を再利用しているため、単なる
parity問題ではなく意味定義バグそのものが両側に存在する。**協調した
単一修正**（`master_v2`生成時に19特徴をDNF_SEMANTIC_SPEC.md準拠へ
修正し、`build_horse_history.py`の該当部分もそれを参照するよう保つ）で
両方同時に解消できる、最も投資対効果が高い対象。ただし本番接続前に
shadow検証＋前向きcanary必須（ユーザー指示通り、未実施）。

### 7-2. kako5_\*13特徴: 「別々の独立した不具合」（training/serve双方に修正必要）

学習側（`build_from_master()`）とserve側（`build_from_kako5()`）は
入力データ・計算関数・window構築方式が全く別物であり、症状が似て見える
だけで原因が異なる。**train/serveを同じ正しい定義へ直す修正案**が
必要だが、それは「共通コードへの統合」ではなく「それぞれ別の実装を
DNF_SEMANTIC_SPEC.md準拠に個別修正する」形になる: 学習側は入力を
dropna前の母集団に差し替え、serve側は`build_from_kako5()`に
止/外/消の分岐ロジックを追加する。

### 7-3. course系2026年分・jockey_stats鮮度: serve固有の低優先度issue

学習側の修正では直らない、serve固有の追加対応が必要（外・消の
誤カウントリスク、jockey_stats.csvのスナップショット鮮度）。実発生
規模は未計測。旧アンサンブル経路限定（jockey_stats鮮度）またはv6本番へ
限定的影響（2026年course系）のため、優先度は低い。

**いずれの分岐についても、本監査は方針の提示のみに留め、本番master
上書き・本番model差し替え・実装着手は行っていない。**

---

## 8. 過去結論への影響（EXP01-EXP12、再実行なし）

`analysis/mcond/p0_dnf_history_parity_audit/EXP01_12_IMPACT_CLASSIFICATION.md`
（詳細版）のサマリ:

| 分類 | 件数 | 実験 |
|---|---|---|
| 影響なし | 1 | EXP11 |
| 数値は変わるが結論影響は小さい見込み | 6 | EXP01, EXP05, EXP06, EXP07, EXP08, EXP09 |
| 再評価が必要 | 5 | EXP02, EXP03, EXP04, EXP10, EXP12 |
| 判定不能 | 0 | — |

**優先再評価候補**（元のマージンがゼロに近かった順）:
1. **EXP12**（対戦相手ネットワーク）— 決定的placebo関門で実測改善
   (+0.00037)が既にplacebo平均(+0.00045)を下回り、bootstrap CI上限は
   +0.00002。統制変数`kako5_race_count`自体がバグの直接対象。
2. **EXP10**（大敗・完走能力未発揮リスク）— 主特徴3列
   (`kako5_std_pos`/`kako5_best_pos`/`kako5_avg_pos`)・共通ベースライン
   (`kako5_race_count`)・ラベル定義そのもの(PL順位分布)が影響対象。
   「大敗=DNFに近い結果を無かったことにして分散・最良着順を過小評価する」
   という主題とバグの性質が直接一致する、最も因果的に疑わしいケース。
3. **EXP03**（恒常能力と短期状態）— 主判定Gate2のCI上限+0.000046。
4. **EXP04**（環境不変特徴選別）— 負けた側(M3)に19特徴中9列が実際に選択。
5. **EXP02**（動的対戦能力）— 統計的マージンは決定的だが、固有価値の
   3/4が経験軸(career starts)由来と判明しておりkako5系と土台を共有。

**この段階では実験を再実行していない**（ユーザー指示通り）。

---

## 9. 成果物・停止事項

### 成果物

- 本ファイル: `docs/research/DNF_HISTORY_FEATURE_PARITY_AUDIT_20260922.md`
- `analysis/mcond/p0_dnf_history_parity_audit/DNF_SEMANTIC_SPEC.md`（意味定義spec）
- `analysis/mcond/p0_dnf_history_parity_audit/LIVE_SERVE_DNF_TRACE.md`（C: live serve詳細）
- `analysis/mcond/p0_dnf_history_parity_audit/EXP01_12_IMPACT_CLASSIFICATION.md`（item8詳細）
- `analysis/mcond/p0_dnf_history_parity_audit/out/p0_full_audit_manifest.json`
  （machine-readable impact summary: 年別内訳・top10・invariants・hash）
- `analysis/mcond/p0_dnf_history_parity_audit/out/example_traces.json`（item3実例8件）
- `analysis/mcond/p0_dnf_history_parity_audit/out/model_impact_comparison.json`（item6）
- `analysis/mcond/p0_dnf_history_parity_audit/shadow/corrected_19features.parquet`
  （shadow corrected master、19特徴のみ）+ 上記manifest内にsha256記録
- `models/unified_rank_v6_shadow_corrected.pkl`（shadow retrained model、本番とは別artifact）
- 再現可能スクリプト一式: `build_corrected_universe.py`・`run_full_audit.py`・
  `build_example_traces.py`・`shadow_retrain_and_compare.py`・`check_signed_delta.py`

### 停止事項（本報告完了まで、および完了後もユーザー指示なしには着手しない）

以下は監査完了まで行っていない、また本報告後もユーザーの明示指示なしには着手しない:

- production master上書き
- production model差し替え
- EXP13再開
- EXP14 Stage1
- 2024/2025年追加探索
- ROI評価

関連: [[project_exp13_nonfinish_risk]] [[project_gate0b_feature_dnf_audit_20260922]]
[[project_market_provenance_audit_20260922]] [[project_kekka_ext_data_quirks]]
[[feedback_asof_population_definition]]
