# 市場オッズ Provenance 横断監査 (2026-09-22)

**方式**: 読み取り専用。全ての数値は実ファイル・実コード・実timestampから
算出した（推測・伝聞は用いていない）。再現スクリプト:
`analysis/market_provenance_audit.py`（`venv311\Scripts\python.exe -m
analysis.market_provenance_audit`、`data/`・外部データとも変更しない）。

## 結論を先に述べる

**EXP13 Gate 0A の「2023年 historical_pre_snapshot coverage = 0%」は誤りだった。**

`data/Time _series_odds/TANPUK_*.csv`（2011-2025年、前売りオッズ時系列）という、
EXP13監査時に発見できなかった生ソースが存在する。`analysis/mcond/market.py`が
これを読み、`analysis/mcond/v6base.py`経由で`base.parquet`→
`exp05_design.parquet`という共有artifactを構築しており、**EXP01・EXP04・
EXP05・EXP06・EXP07(Stage2A)・EXP08・EXP09の7実験がこの同一のパイプラインを
既に使用していた**。この「pre」スナップショットは実測で**発走のおよそ26-30分前
（中央値28分）**であり、確定オッズではない。**止馬（DNF）についても
finisherと同水準のcoverageがある**（2023年: finisher 91.0% / DNF 87.6%）。

EXP05の「約35分前」という記述は、`market.py`自身が出力する統計
（「確定オッズ(final)からの経過分」を基準にした値）をそのまま引用したもので、
**「発走時刻」ではなく「確定オッズの記録時刻」を基準にした別の定義**だった
（確定オッズは発走の6-9分後に記録されるため、35分-7分≈28分でEXP07の実測値と
整合する）。EXP07は2026-09-20夜に既にこの問題を発見し、`historical_pre_
snapshot`という統一名称と正しい実測値（26-30分、中央値28分）を確立していた。
本監査はこの用語・定義をそのまま踏襲する。

---

## 1. 横断provenance表

| 実験 | market artifact | raw source file | source date range | snapshot selection code | odds type | timestamp column | minutes-to-post (実測) | final odds使用有無 | market probability計算法 | train/dev/test coverage | artifact hash(先頭16桁) | leak-safe判定 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EXP01 | `base.parquet`(`mkt_p3_pre`,`mkt_p3_am9`,`rank_mkt_pre`) | TANPUK_*.csv | 2013-2025 | market.py "pre"(区分1同日最新) | 単勝 | 月日時分(MMDDHHMM) | 中央値28分(本監査で再確認) | 否(final列は別途保持のみ、特徴に不使用) | de-vig比例正規化→PL top3 | train2016-2021/sel2022/test2023-2025 | market.parquet c158f52a | **Yes**(pre snap使用、確定オッズ不使用) |
| EXP04 | `base.parquet`(`mkt_p3_pre`) | TANPUK_*.csv | 2013-2025 | 同上 | 単勝 | 同上 | 同上 | 否 | 同上 | 同上 | 同上 | **Yes** |
| EXP05 | `exp05_design.parquet`(`mkt_p3_pre`,`mkt_pi_pre`) | TANPUK_*.csv | 2013-2025 | 同上 | 単勝 | 同上 | **「約35分前」と記述したが、これは確定オッズ基準の値。発走基準では中央値28分**(本監査で訂正) | 否(`tan_final_odds`列は保持のみ、特徴に不使用) | 同上 | train2016-2021/sel2022/test2023-2025 | exp05_design.parquet 44204028 | **Yes**(記述の誤りは表記レベル、実際の特徴入力は確定オッズを使っていない) |
| EXP06 | `exp05_design.parquet`(`mkt_p3_pre`,`rank_mkt_pre`) | TANPUK_*.csv(継承) | 2013-2025 | 同上(継承) | 単勝 | 同上 | 中央値28分 | 否 | 同上(継承) | 同上 | 同上 | **Yes** |
| EXP07 | `exp05_design.parquet`(Stage2A)、**独自にTANPUK/master_v2を直接突合(DATA_AUDIT.md§5.4)** | TANPUK_*.csv + UMAREN_*.csv | 2024-2025(直接突合分、n=6,909) | 区分1の同日最終スナップショットとmaster_v2発走時刻を直接突合(EXP07独自実装) | 単勝・複勝・**馬連**(UMAREN同一分布を確認) | 同上 | **26-30分、中央値28分、SD=1.6分(EXP07自身が実測・本監査で追試一致)** | 否(§5.3で確定オッズ不使用を明記・検証済み) | de-vig比例正規化 | Stage2A: test2024-2025 | market.parquet/UMAREN | **Yes**(最も厳密、`historical_pre_snapshot`という用語自体の発祥) |
| EXP09 | `exp05_design.parquet`(`v6_score`等、eligible_races.py) | TANPUK_*.csv(継承) | 2013-2025 | 同上(継承) | 単勝 | 同上 | 中央値28分 | 否 | 同上(継承) | calibration2023/eval2024-2025 | 同上 | **Yes** |
| EXP10 | 外部kekkaファイルの**単勝オッズ列(発走前最終)**、B1市場特徴 | `E:\競馬過去走データ\raw_data\kekka_2010_2025_fix_raceid_v2__keyed.csv` | 2010-2025 | なし(結果ファイルの単一列をそのまま使用、複数timestampの選択ロジックなし) | 単勝 | **無し(タイムスタンプ列自体が存在しない)** | **不明(「発走前最終」とだけ記述、実測未検証)** | **要注意: 記述は「発走前最終」だがTANPUKのhistorical_pre_snapshotとは別ソース。実際には確定オッズと同等の可能性がある(未検証、本監査の対象外だったEXP10当時の設計)** | 単純1/odds | Stage1.5: 2023development | kekka_ext_csv 83992452 | **要再検証**(EXP10は既にGate FAILで完全終了済み、結論への影響は限定的だが記録として残す) |
| EXP12 | `o1_opponent_quality_check.py`が結果ファイルの**単勝オッズ列を直接使用**（Stage0診断のみ、stage1_cv.py本体は市場データ不使用） | `E:\競馬過去走データ\raw_data\kekka_2010_2025_fix_raceid_v2__keyed.csv` | 2023限定 | なし | 単勝 | 無し | 不明 | **Yes(ただしStage0の一診断スクリプトのみ、O0-O3本体のGate評価には市場確率を主要統制変数として使っていない)** | 単純1/odds | Stage0診断のみ | kekka_ext_csv | 診断用途のみのため主結論への影響なし |
| EXP13 | (Gate0A時点)なし、`data/odds/`・`data/forward_prices/`等のみ確認し**TANPUK系列を発見できず「0%」と誤結論** | (本来はTANPUK_*.csv) | (誤: 存在しないと結論／正: 2011-2025) | (未実施) | 単勝 | (未確認) | **誤: 「0%」／正: 2023年finisherで91.0%・DNFで87.6%が26-30分前スナップショットを持つ**(本監査で訂正) | 否(確定オッズは使わない方針自体は正しかった) | (未実施) | (未実施) | market.parquet/TANPUK | **本監査により訂正、§6で詳述** |
| EXP14(予定基盤) | 未構築。DATA_AUDIT.md§9はEXP13 Gate0Aの「0%」を前提に「市場確率込みpooled modelは2023年で構築不能」と記述していた | (予定: TANPUK_*.csv経由) | — | 未設計 | — | — | — | 未定 | 未定 | 未着手 | — | **EXP13の訂正を受け、EXP14のDATA_AUDIT.md§9も訂正が必要（本監査後に反映）** |

---

## 2. 既存市場artifactの実体確認（ファイル名だけで判断せず実データを確認）

### `data/_joint/mkt_horses.parquet`
`analysis/build_joint_substrate.py`が生成。入力は`TANPUK_*.csv`+`UMAREN_*.csv`
（**market.pyとは独立したコードパスで同じ生ファイルを読む**）。「9時
(09:00最寄り)」の区分1レコードを専用抽出——market.pyの`am9`（09:30以前で
最新）と類似の意図だが、実装は別（`am9`はmarket.py用語、mkt_horses側は
「9時」とのみ記述され厳密に同一ロジックかは未検証）。用途はBenter型の
市場内部整合性検定（T-10/T-15 blend関連、`fit_t10_blend.py`等が使用）で
あり、EXP01-13のGate評価には**直接使われていない**（別系統の生存資産）。

### `exp05_design.parquet`
`analysis/mcond/exp05_market_residual_dev/build_features.py`が生成。
先頭行・末尾行を実際に読み込み確認: `train`列(2016-2021)・`sel`列(2022年)・
`year`列(2013-2025の範囲でフィルタ済み、ただし`load_base()`が
`n_field>=5`・`mkt_p3_pre.notna()`等でフィルタするため実際の行はより絞られる)。
`f_v6`・`f_mkt`（logit変換済み）・`calibrated_v6_probability`等の列を持つ。
**`mkt_p3_pre`は`base.parquet`から継承、確定オッズ由来の列(`tan_final_odds`)は
`v6base.py`側に存在するが`exp05_design.parquet`の特徴列には含まれない**
（`build_features.py`のコード上、`tan_final_odds`をmergeしていないことを
確認済み）。

### `TANPUK_*.csv` / `UMAREN_*.csv`
3ファイルずつ（2011-2015/2016-2020/2021-2025）、cp932エンコード、60列
（TANPUK）/158列（UMAREN）。先頭行を実際に読み込み、区分1が1レースあたり
3回（前日23時台・当日9時前後・発走30分弱前）、区分4が1回、という
`market.py`の docstring 記述を実データで確認した（§3参照）。

### `data/odds/`
2026-04-18以降のみ、47ファイル（本番OD CSV、EXP13 Gate0A時点の確認結果
そのまま、変更なし）。**historical_pre_snapshotとは無関係の別artifact**
（用語の混同を避けるため明記）。

### EXP05のmarket builder
`analysis/mcond/market.py`（`TANPUK_*.csv`→`market.parquet`）+
`analysis/mcond/v6base.py`（`market.parquet`→`base.parquet`、v6スコアと結合）。
2段階構成であることを確認した。

### EXP07のhistorical_pre_snapshot builder
専用の独立ビルダースクリプトは存在せず、**`DATA_AUDIT.md`§5.4に埋め込まれた
検証コード（read-onlyな監査、TANPUKとmaster_v2の直接突合）として実施**
されていた。この監査自体がn=6,909レースの実測に基づく一次資料であり、
本監査の§4の数値と独立に整合することを確認した。

### EXP09の市場入力生成コード
専用コードはなく、`exp05_design.parquet`を`eligible_races.py`経由でそのまま
読む（§1の通り）。

### OOF scoreとmarketを結合した中間artifact
`base.parquet`がこれに相当する（`v6base.py`が生成、`v6_score`・`v6_pwin`・
`v6_p3`とmarket.parquetの`mkt_pi_pre`等を`rid16`+`ban`で結合）。

---

## 3. raw-to-derived 追跡（2023・2024・2025 各3レース以上、実データ）

3年から実際のDNF（止）馬を含むレースを選び、生オッズ記録→取得timestamp→
発走予定時刻→スナップショット選択→derived odds→overround除去→市場確率→
artifact最終行、まで一行ずつ追跡した。**確定オッズ・結果ファイル・払戻
ファイルは判断特徴として使っていないことを併せて確認した**（`tan_final_odds`
はartifactに保持されるが`mkt_p3_pre`等の入力には使われない、§2参照）。

### 2023年（`race_id=2023122806050911`, `umaban=9`, DNF馬）
| 段階 | 値 |
|---|---|
| 1. raw odds record | TANPUK: `区分=1, 月日時分=12281514, 9単=30.1` |
| 2. raw取得timestamp | 2023-12-28 15:14 |
| 3. scheduled post time | master_v2実測: 2023-12-28 15:40 |
| 4. snapshot選択 | 同日区分1のうち最新(区分4より前) → 15:14採用 |
| 5. derived odds | 30.1倍 |
| 6. overround除去 | `market.py`のレース内比例正規化ロジック(実装確認済み、本レースでの数値までは未算出) |
| 7. market probability | `implied_raw=1/30.1=0.0332`→レース内正規化後の`pi` |
| 8. artifact最終行 | `base.parquet`/`exp05_design.parquet`の`rid16=2023122806050911, ban=9`行（この馬は`止`のためEXP01-09のいずれも学習・評価母集団の対象外——`master_v2`が着順NaNとしてdropしているため——だが**市場データ自体はTANPUKレベルで取得できていた**） |
| minutes_to_post再計算 | 15:40 - 15:14 = **26分前** |

### 2024年（`race_id=2024122808070906`, `umaban=9`, DNF馬）
| 段階 | 値 |
|---|---|
| raw odds record | 区分=1, 月日時分=12281212, 9単=5.0 |
| raw取得timestamp | 2024-12-28 12:12 |
| scheduled post time | master_v2実測: 2024-12-28 12:40 |
| snapshot選択 | 同日区分1最新 → 12:12採用 |
| derived odds | 5.0倍 |
| minutes_to_post再計算 | 12:40 - 12:12 = **28分前** |
| 確定オッズ(区分4) | 12:47(7.2倍) — 発走7分後、**特徴には不使用** |

### 2025年（`race_id=2025122809050805`, `umaban=1`, DNF馬）
| 段階 | 値 |
|---|---|
| raw odds record | 区分=1, 月日時分=12281109, 1単=24.3 |
| raw取得timestamp | 2025-12-28 11:09 |
| scheduled post time | master_v2実測: 2025-12-28 11:35 |
| snapshot選択 | 同日区分1最新 → 11:09採用 |
| derived odds | 24.3倍 |
| minutes_to_post再計算 | 11:35 - 11:09 = **26分前** |
| 確定オッズ(区分4) | 11:44(47.2倍) — 発走9分後、**特徴には不使用** |

**3年とも一貫して26-28分前**（EXP07実測の中央値28分・範囲26-30分と整合）。
**3レースとも対象馬はDNF（止）馬であり、市場データが正常に取得できていた
ことを直接確認した**——EXP13 Gate0Aの「DNF馬は市場データが無い」という
暗黙の前提が誤りだったことの直接証拠。

追加で2025年の正常完走馬3頭（`2025122806050812`/`811`/`810`の各race、
複数馬）も同様に追跡し、確定オッズ(区分4)が発走の6-9分後に記録される
（＝発走前ではなく発走直後の値）ことを確認した——これが`market.py`の
「確定からの分」ベースの統計とEXP07の「発走時刻からの分」ベースの統計の
差（約35分 vs 約28分）を生む原因である。

---

## 4. coverage再計算（年別、started/DNF別、平地のみ）

`analysis/market_provenance_audit.py`で全件再計算（読み取り専用、
race_id列で結合——[[project_kekka_ext_data_quirks]]が警告する
race_id16の別スキーマ問題を回避）。

| 年 | unique races | horse rows(started) | 26-30分前(historical_pre_snapshot) | その他の発走前 | snapshot無し | market probability作成可能 |
|---|---|---|---|---|---|---|
| 2023 | 3,347 | 46,252 | 42,105 (91.0%) | 2,198 (4.8%) | 1,949 (4.2%) | 44,303 (95.8%) |
| 2024 | 3,345 | 45,804 | 42,064 (91.8%) | 2,275 (5.0%) | 1,465 (3.2%) | 44,339 (96.8%) |
| 2025 | 3,320 | 46,522 | 41,865 (90.0%) | 2,085 (4.5%) | 2,572 (5.5%) | 43,950 (94.5%) |

**started/DNF別（2023-2025合算）**:

| 年 | 母集団 | n | 26-30分前coverage | coverage率 |
|---|---|---|---|---|
| 2023 | 正常完走馬 | 46,107 | 41,978 | 91.0% |
| 2023 | 止(DNF) | 145 | 127 | **87.6%** |
| 2024 | 正常完走馬 | 45,616 | 41,893 | 91.8% |
| 2024 | 止(DNF) | 188 | 171 | **91.0%** |
| 2025 | 正常完走馬 | 46,354 | 41,710 | 90.0% |
| 2025 | 止(DNF) | 168 | 155 | **92.3%** |

**DNF馬のcoverage率は正常完走馬と同水準（±4pt以内）**。`final_odds_only`
（確定オッズしか存在しない件数）・`timestamp不明`は、上記「snapshot無し」
バケツに含まれる（TANPUK自体に該当馬のレコードが無いか、同日区分1が
1件も無いケース）。この内訳の細分（final-onlyとtimestamp不明の分離）は
本監査では未実施——件数が小さく（年間1,465-2,572件、4-6%）、EXP13再開時に
Stage1で個別分類することを推奨する。

---

## 5. 用語の統一

| 用語 | 定義 | データソース | 発走時刻との関係 |
|---|---|---|---|
| `historical_pre_snapshot` | TANPUK/UMAREN由来、同日区分1の最新レコード（確定より前） | `data/Time _series_odds/TANPUK_*.csv`・`UMAREN_*.csv` | **発走の26-30分前(中央値28分)**。2026-09-20夜のユーザー指示でこの名称に統一済み(EXP07)、本監査で追認 |
| `forward_t35` | EXP05-Fの前向き実収集、発走31-38分前を狙って収集 | `reports/exp05fs_odds/`・`data/forward_prices/*_exp05fs_t35_*` | 発走31-38分前(収集目標)。**2026-09-19以降のみ存在、historical_pre_snapshotとは別物・別期間** |
| `t20` | サイトプレビュー向け前向き実収集 | `reports/site_odds/`・`data/forward_prices/*_t20_*` | 発走20分前(目標)。2026-09-11以降のみ |
| `t10` | 本番T-10自動馬券ラインの前向き実収集 | `data/forward_prices/*_t10_*` | 発走10分前(目標)。2026-08-29以降のみ |
| `close` | 発走+60秒の前向き実収集 | `data/forward_prices/*_close_*` | 発走後。2026-08-29以降のみ |
| `final_odds` | TANPUK/UMARENの区分4(確定)。締切後の公式確定値 | `data/Time _series_odds/TANPUK_*.csv`区分=4 | **発走の6-9分後**(本監査で実測)。判断特徴には不使用(EXP01-09で確認済み) |
| `odds_in_result_file` | 結果ファイル(外部kekka等)内の単勝オッズ列 | `E:\競馬過去走データ\raw_data\kekka_*.csv`の`単勝オッズ`、`data/kekka_*.csv` | タイムスタンプ列を持たない。確定オッズと同等かそれ以上に発走から離れた時点の可能性があり**判断時点として使うべきではない**(EXP10のB1特徴がこれに該当、要注意) |

`pre`という列名だけでは安全性を判定していない——`market.py`の`pre`は
`historical_pre_snapshot`の定義通り実測26-30分前だが、これは**列名の
慣習ではなく実際のtimestamp突合によって確認した**（§3・§4）。もし
将来別のコードが`pre`という名前の列を別の定義（例: 確定オッズ直前の
任意の値）で使っていた場合、名前だけで安全とみなさないこと。

---

## 6. 分岐: 2023年 historical snapshot は存在する

**訂正**: EXP13 Gate 0Aの「2023年のhistorical_pre_snapshot coverage 0%」は
誤りであり、正しくは**91.0%（正常完走馬）/87.6%（止馬）**である。

**EXP13は自動再開しない**（ユーザー指示通り）。以下、要求された再監査を行う。

### 止馬にも同じsnapshotが結合できるか
**できる**。§3・§4で3年分・複数レースの実測により確認済み。TANPUK/UMARENは
`race_id`(=rid16)+`umaban`をキーに持つ独立した時系列アーカイブであり、
`master_v2`が非完走馬をdropする処理（`build_dataset.py:321`）とは**完全に
独立したデータソース**——`master_v2`から市場データを引いているわけではない
ため、止馬がmaster_v2に存在しないことは市場データの結合可否に一切影響しない。

### 完走馬との差
coverage率の差は最大4.2pt（2023年、91.0% vs 87.6%）で、統計的に大きな
乖離とは言えない規模。年によって符号も一定しない（2024・2025年はDNF側が
むしろ高い）。**系統的な選択バイアスの明確な証拠はない**（ただし正式な
統計検定は本監査の範囲外、Stage1で必要なら実施）。

### 時刻安全性
TANPUK/UMARENの区分1レコードは、その日その時刻に実際に取得された生の
前売りオッズ時系列であり、**事後に生成されたものではない**（区分1のレコードが
複数回・時系列順に記録されている構造自体が、事後の後知恵で作れるものでは
ないことを裏付ける）。判断時点として「発走26-30分前」を使う限り、EXP01-09が
既に確認済みの時点安全性（§5.3「判断時点より後のオッズ混入なし」等）が
そのままEXP13にも適用できる。**ただし正確な発走時刻との突合精度（±分単位）は
`master_v2`の「発走時刻」列がどの時点で確定した値か（当日変更を含むか）の
確認をStage1で追加すること**（本監査では確認していない、新たなオープン
項目として記録する）。

### 結論（EXP13への影響、Gate0A再判定）
Gate 0Aは**PASSに訂正する**。ただし自動的にEXP13を再開せず、以下を
ユーザーに報告し判断を仰ぐ:
- 訂正されたGate0Aの下で、EXP13のGate0B（numeric parity未実施）・Gate0D
  （境界線上の検出力）は未変更のまま残っている——市場確率が使えることが
  判明しても、他の制約（v6スコア再構築のnumeric parity、陽性数145件の
  検出力）は解消していない。
- EXP13を正式に再開するかどうかは、この2点を含めた総合判断としてユーザーに
  委ねる。

### 2023年 historical snapshot が存在しない場合の分岐
**該当しない**（存在することが確認されたため、このセクションは適用外）。
ただしEXP10のB1市場特徴（§1参照、外部kekkaの単勝オッズを直接使用、
TANPUKのhistorical_pre_snapshotとは別ソース）については、**EXP10自体は
既にGate FAILで完全終了しているため結果の再実行は行わないが**、EXP10の
市場特徴の正体が「発走前最終」と記述されていたものの実際にはTANPUKの
`historical_pre_snapshot`ではなく確定オッズに近い可能性がある点を、
記録として残す（EXP10の結論スコープ自体は変更しない——EXP10は既に
「B3を上回らなかった」という不合格の結論であり、市場特徴の定義がより
不確実だったとしても結論の方向は変わらない）。

---

## 7. 成果物

- 本ファイル: `docs/research/MARKET_DATA_PROVENANCE_AUDIT_20260921.md`
- 機械可読manifest: `docs/research/market_provenance_manifests/*.json`
  （`tanpuk_historical_pre_snapshot.json`・`tanpuk_final_odds.json`・
  `market_parquet.json`・`base_parquet.json`・`exp05_design_parquet.json`・
  `mkt_horses_parquet.json`・`kekka_ext_odds_in_result_file.json`、
  各々 source/date_min/date_max/snapshot_rule/timestamp_provenance/
  minutes_to_post_{median,min,max}/final_odds/safe_for_decision_time/sha256）
- 再現可能スクリプト: `analysis/market_provenance_audit.py`（読み取り専用）

### 監査中に発見した副次的なバグ（本監査自身のスクリプト）
`analysis/market_provenance_audit.py`初版で、`pd.to_datetime(dict(...))`の
返り値（既定RangeIndex）を、groupby+tail(1)由来の非連番indexを持つ
DataFrameへ列代入した際に**pandasのindex整列により大半の行がNaN化する
バグ**を作り込み、初回実行で発見・修正した（`.to_numpy()`で明示的に
indexを外して代入する形に修正、コード内コメントに経緯を残した）。
これ自体が「timestamp・indexの取り扱いを軽視すると容易に誤った0%や
異常値が出る」ことの実例であり、EXP13 Gate0Aの誤りとも同じ教訓（実データを
直接確認せずに構造的な結論を急がない）に帰着する。

---

## まとめ

1. EXP13 Gate 0Aは誤り。2023年のhistorical_pre_snapshotは存在し、
   finisher 91.0%・DNF 87.6%のcoverageを持つ。
2. EXP05の「約35分前」は確定オッズ基準の記述で、発走時刻基準では
   EXP07が既に確立した「26-30分前、中央値28分」が正しい。
3. EXP01・04・05・06・07(Stage2A)・08・09は共通のexp05_design.parquet
   パイプライン（TANPUK由来、確定オッズ不使用）を使っており、これらの
   結論への時点安全性上の懸念は本監査で確認されなかった。
4. EXP10・EXP12(Stage0診断のみ)は結果ファイルの確定オッズに近いソースを
   使っており、TANPUKのhistorical_pre_snapshotとは異なる。ただし両実験
   とも既に終了済みで結論への影響は限定的。
5. EXP13は自動再開しない。Gate0Aは訂正するが、Gate0B・Gate0Dの制約は
   未解消のまま残る。

本監査完了により、EXP14 Stage1・EXP13再開・新しいmarket-control実験・
ROI評価・価格形成モデルのいずれも、依然としてユーザーの明示的な指示なしには
開始しない。

関連: [[project_exp13_nonfinish_risk]] [[project_exp14_regime_moe]]
[[project_research_stopline_20260921]] [[project_kekka_ext_data_quirks]]
[[reference_odds_data_benter]]
