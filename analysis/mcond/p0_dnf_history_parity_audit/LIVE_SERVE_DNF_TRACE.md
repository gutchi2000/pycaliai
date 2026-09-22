# 週次本番サーブ側 DNF-population トレース（2026-09-22）

**状態**: READ-ONLY調査。コード・データは一切変更していない。学習側
(`build_dataset.py`/`parse_kako5.build_from_master`/`build_master_v2.compute_history_features`)
の内部ロジックは `GATE0B_FEATURE_AUDIT.md`・`DNF_SEMANTIC_SPEC.md` で既監査済みのため再監査していない。
本ファイルは **週次本番サーブ経路**（`predict_weekly.py`/`export_weekly_marks.py`/
`serve_history_feats.py`/`build_horse_history.py`/`parse_kako5.build_from_kako5`）だけを対象とする。

対象は本番 v6 経路（`export_weekly_marks.py`）。旧アンサンブル経路
（`predict_weekly.py` 単体実行、Streamlit `app.py`）にのみ存在する仕組みは
その旨明記する。

---

## Q1. 週次サーブは結果判明前に今週の全出走馬を採点しているか（サニティチェック）

**Yes、確認済み。**

- `export_weekly_marks.py:334-339` が今週の出走表CSVをそのままパースする:
  ```python
  df = parse_csv(csv_path)
  df = ensure_date_column(df)
  logger.info(f"パース結果: {len(df):,} 馬 / {df[COL_RID].nunique():,} レース")
  ```
  `parse_csv` は `predict_weekly.py:435` 定義（`export_weekly_marks.py` はこれを import して共用）。
- `parse_csv` 内で行われる唯一の行フィルタは **障害レース除外**（`predict_weekly.py:706-713`）:
  ```python
  _is_hurdle = df["距離"].astype(str).str.contains("障", na=False)
  if "芝・ダ" in df.columns:
      _is_hurdle = _is_hurdle | df["芝・ダ"].astype(str).str.contains("障害", na=False)
  df = df[~_is_hurdle].copy()
  ```
  これは開催種別によるフィルタであり、個々の馬の将来の着順（DNFになるかどうか）とは無関係。
  今週の出走表には「今週結果」列自体が存在しない（未来の情報なので構造的にありえない）ため、
  「あとでDNFになる馬を暗黙に除外する」ようなロジックは**存在しようがない**し、実際存在しない。
- レース単位の中止（台風等）についてのガードは別途 `export_weekly_marks.py:628-646` にあるが、
  これは「オッズ取得できないレース丸ごとをbundleから除外しない」という**逆方向**の安全策
  （venue全滅の誤検知防止）であり、個馬DNFの話ではない。

**結論**: 今週の全出走馬が無条件でスコアリング対象になる。これは自明であり、以降の
Q2-Q4が問う「その馬の**過去**の履歴特徴をどう計算するか」だけが実質的な論点。

---

## Q2 / Q3. `serve_history_feats.py` の履歴データソースと DNF 扱い

### 2.0 全体構造：3つの独立した経路が19特徴を分担している

週次サーブは19特徴を**単一の仕組み**で埋めていない。読解の結果、3つの完全に別の
コードパス・データソースが混在していることが判明した（これは事前の想定と異なる、
今回の調査で新たに確認した事実）。

| 特徴グループ | 件数 | 計算する関数 | データソース |
|---|---|---|---|
| `course_n_prev/win_rate/top3_rate`, `jockey_n_prev/win_rate/top3_rate` | 6 | `serve_history_feats.compute_row_feats()` | `data/_horse_history.parquet`（`build_horse_history.py`生成） |
| `hist_same_cond_*`(3) / `hist_same_place_best_pos`(1) | 4 | 同上 | 同上（※19特徴の対象外だが同じ関数内、比較のため記載） |
| `kako5_avg_pos`〜`kako5_same_cond_best_pos` | 13 | `parse_kako5.build_from_kako5()`（`predict_weekly.py:648-668`から呼ばれる、**`build_from_master()`とは別の関数**） | `data/kako5/{date}.csv`（TARGET週次出力、馬ごとの直近5走が横持ちで埋め込まれた別ファイル） |

`serve_history_feats.py`のdocstring(1-26行)は「10特徴+コード」を謳っているが、これは
`course_*`(3)+`jockey_*`(3)+`hist_same_*`(4)=10の意味であり、**kako5_\* 13特徴は
`serve_history_feats.py`の管轄外**。`export_weekly_marks.py`はkako5_\*列を自前で埋めておらず、
`parse_csv()`が内部で`build_from_kako5()`を呼んで既に埋めた状態のdfを受け取るだけ
（`export_weekly_marks.py:336`の時点でkako5_\*列は存在している）。この分業を見誤ると
「serve_history_feats.py だけ読めばkako5系も分かる」と誤解する。

`data/serve_feature_baseline.json`の実測カバレッジもこれを裏付ける
（`analysis/measure_serve_coverage.py`が生成）: `kako5_race_count=1.00`、
`course_n_prev=1.00`、`jockey_n_prev=0.988` — いずれも高カバレッジ＝「値は入っている」
ことは確認できるが、**値の正しさ（DNF扱い）は別問題**というのが本監査の主題。

---

### 2.1 `course_*` / `jockey_n_prev系`（6特徴）: `_horse_history.parquet` の二重構造

`serve_history_feats.py:75-131`の`_HistoryIndex`が読む`data/_horse_history.parquet`は
`build_horse_history.py`が生成する。この生成ロジックが **時代によって全く別のデータソース**
を継ぎ足している。

**(a) 2013-2025分（大半の馬の履歴の大部分を占める）**: `build_horse_history.py:132-165`
`load_master_history()`が読むのは

```python
MASTER_CSV = BASE / "data" / "master_v2_20130105-20251228.csv"   # build_horse_history.py:43
...
m = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", usecols=cols, low_memory=False)  # L137
```

**`data/master_v2_20130105-20251228.csv`は学習パイプラインの最終成果物そのもの**
（`build_dataset.py`→`parse_kako5.build_from_master`→`build_master_v2.py`の出力、
626,774行）。`GATE0B_FEATURE_AUDIT.md`が実測済みの通り、この行数は
`build_dataset.py:321`の`dropna(subset=["着順"])`によって631,965→626,774に
削られた**後**の値であり、`止`/`外`/`消`の行はこのファイルには**そもそも1行も存在しない**
（学習側のバグと同じ場所で同じ理由により、既に消えている）。

つまり `build_horse_history.py` は「独自にDNF行を除去している」のではなく、
**学習側が既に除去し終えた成果物をそのまま履歴ソースとして再利用している**。
2013-2025年の期間について、`course_n_prev`/`jockey_n_prev`のカウント元となる母集団は
学習側と**完全に同一のファイル**であり、当然ながら**学習側と全く同じ理由・同じ規模で
永続的な過小カウントが起きる**（ある馬が2020年に止を1回経験していれば、
2021年以降にserveが計算するその馬のcourse_n_prev/jockey_n_prevは、2026年に至るまで
恒久的に1少ないまま。GATE0B実測: 該当行1.03%・raw score差 mean=0.022/max=0.326）。

**(b) 2026分（当年の補完）**: `build_horse_history.py:212-264`
`load_2026_history()`が読むのは `data/kekka/{date}.csv`（TARGET週次確定結果）×
`data/weekly/{date}.csv`（TARGET週次出走表）のinner join。ここでは
**学習側のdropnaを経由しない、独立した2026限定の生データ**を使っている:

```python
k = k.assign(
    rid16=k["レースID(新)"].astype(str).str.strip().str[:16],
    馬番=pd.to_numeric(k["馬番"], errors="coerce"),
    pos=pd.to_numeric(k["確定着順"], errors="coerce"),
)
k["pos"] = k["pos"].where(k["pos"] > 0)      # build_horse_history.py:238-240
...
merged = k[["rid16", "馬番", "馬名", "pos"]].merge(w, on=["rid16", "馬番"], how="inner")
```

ここには**dropnaが一切ない**。`確定着順`が数値化できない（NaNになる）行も、
`pos>0`でない行（0を含む）も、**行自体は削除されず`pos=NaN`のまま残る**。
実データで確認した具体例（`data/kekka/20260905.csv`、rid16=`2026090501020508`）:

```
馬番  馬名(garbled表示)   確定着順
...
2     (該当馬)             0        ← 全13頭中、馬番1-13が連続して揃っている
```

同じ`rid16`で`data/weekly`側の馬番1-13が欠けなく対応しており、この「0」行は
**除外されず、全頭の中の1頭として最後までレコードに残る**ことを確認した。
`merged`はこの行をそのまま含んだまま`_horse_history.parquet`に追加される
（`pos=NaN`、`place`/`surface`/`dist`は実値）——これは`DNF_SEMANTIC_SPEC.md`§3.2が
求める「止の走は着順=None、TD/距離/場所は実値」という設計と**結果的に一致する**。

`serve_history_feats.py`の`course_n_prev`/`jockey_n_prev`計算（`compute_row_feats()`
L256-308）は「行の存在」でカウントする（`sel.sum()`、pos値は無関係）ため、
**2026年内に発生したDNF (`止`) はこの行を通じて正しく経験1回として数えられる**。
これは学習側（dropna後の626,774行を使うためDNF行が丸ごと消える）とは**逆に正しい**挙動。

**しかしQ3で詳述する通り、この「0」コードは`止`と`外`/`消`を区別できない**ため、
2026年内に発生した**発走前除外・取消**（本来は分母に含めてはいけない）も同じ「0」行として
`_horse_history.parquet`に混入し、`course_n_prev`/`jockey_n_prev`を**過大カウント**する
リスクがある（学習側とは逆方向の、serve固有の新しい誤差）。この事象の実発生頻度・規模は
本監査では未計測（静的コード読解の範囲を超えるため、値の水準としては「構造的に起こりうる」
までの確認に留める）。

### 2.2 Q3: DNFコードの区別能力 — kekka系ソースは`止`/`外`/`消`を1つに潰している

学習側が参照する外部生データ（`add_20130105-20251228.csv`、`kekka_2010_2025_fix_raceid_v2__keyed.csv`）
は`着順`列に**`止`/`外`/`消`という別々の漢字文字列**を持ち、両ソース間で完全一致することが
`LABEL_CODEBOOK.md`で実測確認済み（止=2,946/2,946、外=1,212/1,212、消=1,007/1,007）。

これに対し、**本リポジトリが週次で受け取る`data/kekka/{date}.csv`（TARGET週次確定結果CSV）
は`確定着順`列にこの区別を一切保持していない**。実データ実測（`data/kekka/20260905.csv`、
455行36レース）:

```python
df['確定着順'].value_counts()
# '1'..'17' の通常着順に加えて '0' が3件のみ
```

**`止`・`外`・`消`のいずれであっても、この週次kekkaファイルでは単一のコード`"0"`に
潰されている**（漢字コードは一切出現しない）。したがって
`build_horse_history.py`が`pd.to_numeric(..., errors="coerce")`→`.where(pos>0)`で
`pos=NaN`に変換する処理より**手前の、データソース自体の時点で**`止`と`外`/`消`の
区別情報は失われている。これは「serve側の実装ミス」ではなく「TARGET週次結果CSVの
エクスポート仕様がそもそも区別を出力しない」という、**データソースレベルの制約**である。

結果として：
- **学習側**: `止`/`外`/`消`を区別できる生データを持つが、`build_dataset.py`が
  意図せず全部dropしてしまう（バグ、Vol.III/GATE0Bで既知）。
- **serve側（2026年分）**: 区別できる生データを最初から持たない（TARGET週次出力の
  仕様上の制約）。`止`を正しく含めることに**成功している**のは、たまたま
  「区別せず全部を1つのpos=NaN行として残す」という単純な実装だから
  （区別しようがないので逆に取りこぼさない）——ただし同じ理由で`外`/`消`も
  一緒に含めてしまう副作用があり、これは「正しい設計判断の結果」ではなく
  **「区別する手段がないため生じた偶然の副産物」**である点に注意。

なお、`predict_weekly.py:343-402`の`_load_kako5_warnings()`が読む
**別のTARGET出力ファイル**（`data/kako5/{date}.csv`、馬ごとの直近5走が横持ちで
埋め込まれた週次ファイル）は事情が異なり、この列には**漢字の`止`/`外`/`消`が
そのまま出現する**ことをコードが直接証明している:

```python
STOP_CODES = {"止", "外", "消"}
CHAKU_IDXS = {1: 18, 2: 30, 3: 42, 4: 54, 5: 66}
...
val = parts[idx].strip()
if val in STOP_CODES:
    warns.append(f"{n}走前{CODE_LABEL[val]}")
```

つまり**TARGETの出力ファイルによって粒度が異なる**: 週次確定結果CSV
(`data/kekka/*.csv`)は`0`に潰すが、週次過去5走CSV(`data/kako5/*.csv`)は
漢字コードのまま出力する。同じ「非完走」という概念が、ファイルによって
表現の解像度が違う——これがQ3の核心的な発見である。

### 2.3 `kako5_*`（13特徴）: `build_from_kako5()`は`止`も`外`/`消`も無差別に「スロットごと消す」

`parse_kako5.py:343-432`の`build_from_kako5()`が`data/kako5/{date}.csv`
（上記の、漢字コードを保持する方のファイル）から直近5走を抽出する。着目すべきは
L394-406:

```python
past_races = []
for i, (place_i, pos_i, ninki_i, agari_i) in enumerate(race_offsets):
    pos = _safe_int(row[pos_i])
    if pos is None or pos == 0:
        continue                      # ← このスロットを past_races に追加しない
    past_races.append({
        "着順": pos, ...
    })
```

`_safe_int("止")`は`int(float("止"))`が`ValueError`になり`None`を返す
(`parse_kako5.py:205-209`)。つまり**`止`/`外`/`消`のいずれであっても`pos is None`
となり、そのスロットは`past_races`に一切追加されず、まるごと存在しなかったことになる**。

これは学習側（`build_from_master()`、`GATE0B_FEATURE_AUDIT.md` 1-E節）の欠陥とは
**似て非なる、別種の欠陥**である:

- **学習側**: 入力（`master_20130105-20251228.csv`）に`止`の行が**物理的に存在しない**ため、
  `idxs[max(0, seq_i-5):seq_i]`という「直近5"行"」の"行"がそもそも「dropna後に生き残った
  実際の完走レコード」しかない。5行分は必ず埋まる（対象馬の完走レコードが5件以上あれば）。
  結果、**本来の直近5走より過去に遡って**5走ぶんの完走レコードを集めてしまう
  （時間軸が過去方向にズレる。GATE0Bで確定的差分を実測済み）。
- **serve側**: TARGETの週次kako5 CSVは**固定5列**（Race1〜Race5、直近5走の枠が
  最初から確保されている横持ち形式）であり、この5列の中に`止`/`外`/`消`があれば、
  `build_from_kako5()`はそのスロットを**単純に読み飛ばす**（他の古い実走で埋め直す
  手段がない——ファイル自体が5列しか持っていないため、遡って6走目のデータを
  取得できない）。結果、**`kako5_race_count`（=`len(past_races)`）が本来の値より
  少なくなる**（最大5走が、直近5枠内に非完走が多いほど0に近づく）。同時に
  `kako5_same_td_ratio`等の分母`n`も一緒に縮小し、`kako5_pos_trend`の回帰点数も減る。

**この2つは方向もメカニズムも異なる欠陥である**:
学習側は「見えない過去のDNFを迂回して、もっと昔の完走走を代わりに使ってしまう
（気づかれない時間軸のズレ）」。serve側は「直近5枠のうち非完走の枠を単に捨てて、
代わりを探さない（データが薄くなる、n<5になる）」。

また`DNF_SEMANTIC_SPEC.md`§3.1が定める「止はスロットとして数える・外/消は数えない」
という区別も、serve側の`build_from_kako5()`は行っていない
（`pos is None or pos == 0`の一律`continue`——`止`と`外`/`消`を分けるロジックが
存在しない）。したがって:
- `止`（本来スロットとして数えるべき）→ serveは**誤って除外**（過小方向）
- `外`/`消`（本来スロットとして数えないべき）→ serveは**結果的に正しく除外**
  （ただし意図した設計ではなく、`止`と道連れにしているだけ）

### 2.4 `hist_same_cond_*` / `hist_same_place_best_pos`（19特徴の対象外だが比較用）

`serve_history_feats.py:229-254`は`~np.isnan(past["pos"])`という**値ベースの
フィルタ**（母集団に`止`行が存在するか否かに関わらず、`pos=NaN`の行は最初から
集計に寄与しない）。これは学習側の`hist_same_cond_*`が"confirmed_immune"と
判定された理由と全く同じメカニズムであり、`_horse_history.parquet`の
2013-2025部分に`止`行が物理的に存在しない（Q2.1で確認）ことも、2026部分に
`止`/`外`/`消`が`pos=NaN`の行として存在すること（Q2.1で確認）も、**どちらも
計算結果に影響しない**（存在してもNaNなので寄与ゼロ、存在しなくても最初から
寄与するはずだった値がゼロなので同じ）。学習・serve双方で免疫、という
唯一「対称に無罪」のグループ。

---

## Q4. `jockey_stats.csv`/`trainer_stats.csv`の call order

`build_dataset.py`の`build_master()`内の実際の呼び出し順序（`build_dataset.py:300-326`）:

```python
315:    # ---- 行数チェック ----
316:    assert len(master) == 631_965, ...
317:    ...
319:    # ---- 除外・中止・失格を除去 ----
320:    before = len(master)
321:    master = master.dropna(subset=["着順"])          # ← ここでDNF行を除去
322:    logger.info(f"除外・中止除去: {before - len(master):,}件 → {len(master):,}行")
323:
324:    # ---- 騎手・調教師スタッツスナップショット保存（週次予測用） ----
325:    logger.info("エンティティスタッツを保存中...")
326:    save_entity_stats(master)                          # ← dropna後のmasterを渡している
```

**`save_entity_stats()`は`dropna(subset=["着順"])`の"後"に呼ばれている**
（`add_rolling_stats()`がjockey_fuku30/90を計算するのはさらに前段、dropnaより前 —
`GATE0B_FEATURE_AUDIT.md` 1-B節で既検証・確定無罪— だが、`save_entity_stats()`自体の
呼び出しはdropnaの**後**）。

`save_entity_stats()`の中身（`build_dataset.py:182-200`）:

```python
latest = (
    master.sort_values(["日付", "発走時刻"])
    .groupby(code_col)[available]
    .last()                          # ← 各騎手/調教師コードの「最後の行」の値を採用
    .reset_index()
    .dropna(subset=available, how="all")
)
```

つまり`jockey_stats.csv`/`trainer_stats.csv`は**「その騎手/調教師コードが
（dropna後の`master`の中で）最後に登場した行のjockey_fuku30/90値」のスナップショット**
である。`jockey_fuku30`という**値そのもの**はdropna前に計算済み（正しい・免疫）だが、
**「どの行が"最新"としてスナップショットに採用されるか」の選定はdropna後の
母集団に対して行われる**。

したがって、ある騎手のTARGET輸出範囲内での**文字通り直近の騎乗**がたまたま`止`
（中止）だった場合、その行は`master`の`dropna`で消えているため、`save_entity_stats()`の
`.groupby().last()`は**1つ前の（実際にはより過去の）騎乗行**をその騎手の
「最新」として採用してしまう。結果として:

- **数値自体は不正確ではない**（採用された行のjockey_fuku30/90はその時点でのローリング値
  として正しく計算されている——jockey_fuku30/90自体はdropna前計算のため無罪）。
- しかし**「最新である」という主張が不正確**（本当はもう1回、直近に騎乗履歴があるのに
  それを見ていない=1件分古い情報を「最新」として週次予測に配る、stale snapshot）。

これはcourse_n_prev系の「恒久的な過小カウント」とは**異なる種類の症状**（カウントの
誤りではなく、鮮度の誤り）であり、影響も限定的（その騎手/調教師の「直近の1件がDNFだった
週」のみ、かつ「その騎手/調教師にとって"次に新しい"行との差」程度の誤差）。

**重要な適用範囲の限定**: `jockey_stats.csv`/`trainer_stats.csv`は
`grep`で確認した通り**旧アンサンブル経路(`predict_weekly.py`単体実行、Streamlit`app.py`)
専用**であり、`export_weekly_marks.py`（v6本番経路）は一切参照していない
（`export_weekly_marks.py`はjockey_fuku30/90を`serve_history_feats.py`の
`rolling_rate()`——`_horse_history.parquet`から都度計算し直す独立実装——で埋める、
Q2.1で扱った経路とは全く別）。`predict_weekly.py:513-541`で実際の適用箇所を確認:

```python
for fname, code_col, stat_cols in [
    ("jockey_stats.csv",  "騎手コード",  ["jockey_fuku30", "jockey_fuku90"]),
    ("trainer_stats.csv", "調教師コード", ["trainer_fuku30", "trainer_fuku90"]),
]:
    ...
    df = df.merge(stats[[code_col] + stat_cols], on=code_col, how="left")
```

**結論**: `save_entity_stats()`はjockey_fuku30/90の「無罪（dropna前計算）」を
値レベルでは引き継ぐが、「スナップショットとして選ぶ行」の選定はdropna後の
母集団に対して行われるため、**厳密には完全な免疫ではなく、別種の軽微な鮮度劣化バグを
新たに持つ**。ただし影響範囲は旧アンサンブル経路（Streamlit/`predict_weekly.py`単体）
に限定され、本番v6経路（`export_weekly_marks.py`）には影響しない。

---

## 3. 19特徴 個別verdict表

凡例: **SAME**=学習と同じ欠陥（同じメカニズム・同じ方向） / **DIFFERENT**=serve固有の
別の欠陥（メカニズム・方向とも異なる） / **IMMUNE**=無罪（対象外参考) /
**AMBIGUOUS**=データ制約上コードだけでは断定不能な残存リスクあり

| # | 特徴 | 計算箇所 | データソース | Verdict | 理由 |
|---|---|---|---|---|---|
| 1 | course_n_prev | `serve_history_feats.compute_row_feats()` L256-279 | `_horse_history.parquet`（2013-25=master_v2） | **SAME**（2013-25分）+ **DIFFERENT**（2026分、AMBIGUOUS方向） | 2013-25分は学習と同一ファイル(master_v2)を再利用しているため同一の永続的過小カウント。2026分はkekka「0」行が残るため`止`は正しく計上されるが`外`/`消`と区別できず過大計上リスク(未計測) |
| 2 | course_win_rate | 同上 | 同上 | 同上 | 分子は`pos==1`のNaN安全なので影響なし、分母(course_n_prev)の誤りをそのまま継承 |
| 3 | course_top3_rate | 同上 | 同上 | 同上 | 同上 |
| 4 | jockey_n_prev | 同上 L281-308 | 同上 | 同上 | 馬×騎手コードペア単位。母集団の欠陥はcourse系と同一メカニズム |
| 5 | jockey_win_rate | 同上 | 同上 | 同上 | 同上 |
| 6 | jockey_top3_rate | 同上 | 同上 | 同上 | 同上 |
| 7 | kako5_avg_pos | `parse_kako5.build_from_kako5()`→`_compute_features()` | `data/kako5/{date}.csv`（TARGET週次、漢字コード保持） | **DIFFERENT** | `positions`は`着順 is not None`の値のみ集計。`止`のスロットが`build_from_kako5`のL397で丸ごと`continue`されるため`positions`に一切現れない点は学習と類似の結果だが、**入力データ・window構築方式が別物**なので偶然の一致であり同一メカニズムではない |
| 8 | kako5_std_pos | 同上 | 同上 | **DIFFERENT** | 同上。有効件数が減るほど標準偏差の推定が不安定化（学習側は"古い代替走で埋まる"ため件数は減らない、serve側は件数そのものが減る） |
| 9 | kako5_best_pos | 同上 | 同上 | **DIFFERENT** | 同上 |
| 10 | kako5_avg_agari3f | 同上 | 同上 | **DIFFERENT** | `止`行は元々上り3F自体が記録されないため寄与ゼロの点は学習と同じだが、**スロット自体が消える**ためn自体が学習と異なる形で縮む |
| 11 | kako5_best_agari3f | 同上 | 同上 | **DIFFERENT** | 同上 |
| 12 | kako5_same_td_ratio | 同上 | 同上 | **DIFFERENT** | 分母`n=len(past_races)`が`止`/`外`/`消`スロット消失でserve側は縮小。学習側は縮小せず時間軸がズレる。方向・メカニズムとも別 |
| 13 | kako5_same_dist_ratio | 同上 | 同上 | **DIFFERENT** | 同上 |
| 14 | kako5_same_place_ratio | 同上 | 同上 | **DIFFERENT** | 同上 |
| 15 | kako5_pos_trend | 同上 | 同上 | **DIFFERENT** | 回帰点数が学習側は5点維持(ズレた5走)、serve側は非完走スロット分だけ点数が減る(最悪0-1点で回帰不能=NaN) |
| 16 | kako5_race_count | 同上 | 同上 | **DIFFERENT** | 学習側は「直近5"行"以内に何回出走したか」がほぼ常に5(dropna後の行を数えるため)。serve側は`止`/`外`/`消`を積極的に除外するため**むしろ本来より少なく**出る。両者とも「真の直近5走以内の出走数」とは異なるが誤差の出方が逆 |
| 17 | kako5_expected_good_count | 同上 | 同上 | **DIFFERENT** | 好走判定はpositions由来。スロット消失でカウント対象母数がserve側で縮む点が学習と異なる |
| 18 | kako5_hidden_good_count | 同上 | 同上 | **DIFFERENT** | 同上 |
| 19 | kako5_same_cond_best_pos | 同上 | 同上 | **DIFFERENT** | 同上 |

参考（19特徴の対象外、比較のため）:

| 特徴 | Verdict | 理由 |
|---|---|---|
| hist_same_cond_best_pos/top3_rate/count, hist_same_place_best_pos | **IMMUNE**（学習・serve双方） | 値ベースフィルタ(`isnan`)のため母集団に`止`行が存在してもしなくても寄与ゼロで結果不変。Q2.4参照 |
| jockey_fuku30/90, trainer_fuku30/90, horse_fuku10/30（v6本番経路） | **概ねIMMUNE、ただし新規に発見した軽微な例外あり** | `serve_history_feats.rolling_rate()`は行レベルcounting(flag=0扱い)で学習のC1設計に意図的に合わせているが、2013-25分のデータソースが`_horse_history.parquet`経由でmaster_v2(dropna後)である点は course系と同じ構造。ただしDNF行がwindow内に「flag=0として残る/消える」の違いは平均値への影響が小さい(DNFは元々pos>3同然の寄与)ため実害は軽微と推定(未計測)。19特徴の対象外につき詳細検証はしていない |
| jockey_stats.csv/trainer_stats.csv (`jockey_fuku30/90`, `trainer_fuku30/90`、旧アンサンブル経路限定) | **DIFFERENT（鮮度バグ、値バグではない）** | Q4参照。dropna後の`master`から`.groupby().last()`でスナップショット行を選定するため、対象コードの真に最新の騎乗がDNFだと1件古い値を「最新」として配信する。v6本番(`export_weekly_marks.py`)は不使用 |

---

## 4. 総括：学習側とserve側は「同じ間違った定義」なのか「別の間違い」なのか

**結論: 両方が混在しており、単純な「同じバグ」とも「完全に別のバグ」とも言えない。
特徴グループごとに答えが異なる。**

- **course_n_prev系6特徴**は、2013-2025年分の履歴について**学習側と文字通り同一の
  汚染された母集団ファイル(`master_v2_20130105-20251228.csv`)を再利用している**ため、
  「serveが独自に同じロジックを実装してしまった」のではなく「serveが学習側の欠陥を
  そのまま継承している」——**協調した単一の修正（`build_horse_history.py`の
  historical部分を、`master_v2`ではなくdropna前の生母集団から再構築する）で
  両方同時に直る性質の不具合**である。ただし2026年分については学習側の欠陥とは
  無関係な、serve固有の新しい不具合（`止`/`外`/`消`を区別できず後者を誤って
  計上しうる）が別途存在し、これは学習側の修正では直らない、serve側だけの
  追加対応が必要な問題である。
- **kako5_\*13特徴**は、学習側とserve側で**入力データも計算関数もwindow構築方式も
  全く別物**（`build_from_master()`と`build_from_kako5()`は名前が似ているだけの
  別実装、片や dropna後master の位置ベース自己結合、片やTARGET週次横持ちCSVの
  固定5スロット）であり、**症状が似て見える（どちらも"直近5走"の意味がズレる）
  だけで、原因もズレ方の方向も別**。学習側は「見えないDNFの向こうまで遡って
  古い完走走で埋める」、serve側は「非完走スロットを埋め直さず薄くする」。
  これは**片方だけを直しても他方は直らない、独立した2つの不具合**であり、
  「学習が正しくserveが間違っている」でも逆でもなく、**両方とも
  `DNF_SEMANTIC_SPEC.md`の定義（止はスロットとして数える・外消は数えない）から
  外れている**が、外れ方が違う。修正するなら学習側は`parse_kako5.build_from_master()`
  の入力をdropna前に差し替える対応、serve側は`build_from_kako5()`のスロット除外
  ロジックを`止`/`外`/`消`で分岐させる対応と、**それぞれ別の修正が必要**。

したがって「本番修正シナリオ」としては、**同一バグ両側修正（course系の2013-2025分）
と非対称修正（course系の2026分・kako5系全体・jockey_stats鮮度）が同時に存在する
複合パターン**であり、どちらか一方のシナリオだけでは正確な記述にならない。
優先順位をつけるなら: (1) course系2013-2025分はGATE0Bの学習側修正と同時に
`build_horse_history.py`の historical ソースも直せば1回の作業で両方解消できる
「最も投資対効果が高い」対象、(2) kako5系はserve側だけの追加実装
（`build_from_kako5()`に`止`/`外`/`消`分岐を入れる）が別途必要、(3) course系2026分の
`外`/`消`誤計上とjockey_stats鮮度劣化は実発生規模が未計測のため、まず影響の定量化
（実際に該当する行数・馬数がどれだけあるか)を先にやるべき低優先度・低確度の項目。

関連: `analysis/mcond/exp13_nonfinish_risk_dev/GATE0B_FEATURE_AUDIT.md`,
`analysis/mcond/p0_dnf_history_parity_audit/DNF_SEMANTIC_SPEC.md`,
`analysis/mcond/exp13_nonfinish_risk_dev/LABEL_CODEBOOK.md`
