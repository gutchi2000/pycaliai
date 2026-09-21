# EXP12 Stage 1 — 未コミットscratchコードの先行監査

**作成日**: 2026-09-21　**対象**: `analysis/_tmp_oppstrength_gate.py`、
`analysis/_tmp_elo_strict_gate.py`（いずれもgit未追跡、コードは変更していない、
全文読了済み）。

## 共通事項

- **作成日時**: リポジトリのファイルmtimeは両方とも2026-06-25。コード内に
  日付コメントはない。
- **参照データ**: 両方とも`data/master_v2_20130105-20251228.csv`
  （`be.MASTER_CSV`経由、本プロジェクトの標準マスター）。
- **出力先・ログ**: **両方とも`print()`のみで、`to_csv`/`to_json`/
  `to_parquet`等のファイル書き出しは一切ない**（grep確認）。`reports/`・
  `analysis/`・`logs/`のどこにも対応する保存物がないという既存の監査結果
  （Explore agent報告）と整合する。
- **実行済みだった痕跡**: **見つからなかった**。ログファイル・レポートJSON・
  git履歴（未追跡のため commit なし）・メモリファイルのいずれにも実行結果への
  言及がない。実行されたかどうかは不明のまま。

## ★重要な発見: 両スクリプトとも train=2023、test=2024-2025 を使う設計

```python
tr = sub[sub["year"]==2023].copy(); teh = sub[sub["year"].isin([2024,2025])].copy()
```
（`_tmp_oppstrength_gate.py:57`、`_tmp_elo_strict_gate.py:76`、両方同一）。

**これは2024・2025年をtest期間として開封する設計であり、本セッションで
確立した「2024・2025年の性能・ROIは開封しない」という規律に反する**。
ただし前述の通り実行痕跡・保存された結果は一切見つかっていないため、
**実際に2024・2025年の数値が生成・閲覧されたかどうかは確認できない**
（コードの設計が開封する構造だった、という事実のみ確定できる）。本監査では
このコードを実行しておらず、いかなる数値も参照・使用していない。EXP12
Stage1では、このスクリプトの設計をそのまま流用しない（後述の通りas-of
機構に本質的な違いもあるため、そのままでは使えない）。

## `opp_best_beaten`の正確な定義

**`_tmp_oppstrength_gate.py:29-31, 48-50`**:
```python
df["_ph9"] = pd.to_numeric(df.get("前走補9", df.get("prev_hosei9")), errors="coerce")
df["field_str"] = df.groupby(COL_RID)["_ph9"].transform("mean")  # そのレース出走馬全員の「前走」補正タイム平均
...
df["_placed_fs"] = np.where(df[COL_JYUN]<=3, df["field_str"], np.nan)  # 自分が複勝圏内だったレースのみ
df["opp_best_beaten"] = g["_placed_fs"].transform(lambda s: s.shift(1).expanding().max())
```
= 「(対象馬自身が)複勝圏内で終えた過去レースのうち、そのレースの
**フィールド平均**補正タイム指数が最大だったものの値」。**個々の対戦相手を
識別してはいない**——「誰に勝ったか」ではなく「自分が好走したレースの
場の平均強度」という粗い代理。

**`_tmp_elo_strict_gate.py:64, 67-68`（ELO版）**:
```python
df["field_elo"] = df.groupby(COL_RID)["pre_elo"].transform("mean")  # そのレース出走馬全員のpre_elo平均
...
df["_placed_fe"] = np.where(df[COL_JYUN]<=3, df["field_elo"], np.nan)
df["opp_best_beaten_fe"] = g["_placed_fe"].transform(lambda s: s.shift(1).expanding().max())
```
同型、強度の代理指標がprev_hosei9(補正タイム)からpre_elo(独自の逐次ELO)に
変わっただけ。

## 対戦相手の能力をいつの時点で計算しているか

**両方とも「遭遇した時点でのフィールド平均強度」を使い、その後更新しない**。
`field_str`/`field_elo`は、対象馬と対戦相手が**同じレースを走った、その
瞬間**のフィールド平均値として一度だけ計算され、`opp_best_beaten(_fe)`は
その値を`expanding().max()`で保持するだけ——**対戦相手側のその後の成績で
再評価されることはない**。

**これはEXP12（本セッション）の核心仮説と本質的に異なる**。ユーザー指定の
O3の核心は「過去に対戦した時点では評価が低かった相手が、その後、対象日
までに強いと判明した」という情報を遡及的に対象馬へ還元することだが、
scratchコードの`opp_best_beaten`系は遭遇時点のスナップショットのみで、
対戦相手の「その後の判明」を一切反映しない。

## 対象レースより後の相手成績を使っていないか

**対象馬自身の予測行に対しては安全**（`shift(1)`により対象レースを含む
それ以降の情報は使われない）。ただし上記の通り、そもそも対戦相手の
「その後の成績」を使う設計になっていないため、この点でのリークは
構造的に発生しない（同時に、EXP12が意図する情報も含まれていない）。

## 全期間集計を過去行へ付与していないか

**付与していない**。`field_str`/`field_elo`はレース単位の`groupby(...)
.transform("mean")`であり全期間集計ではない。`opp_prev`系・
`opp_best_beaten`系は全て`.shift(1)`＋`.rolling()`/`.expanding()`で、
EXP11で確立したas-of規律に沿っている（対象行より後のレースは含まれない）。

## 市場確率を統制しているか

**していない**。両スクリプトともLightGBMの入力特徴に`v6_score`＋
（格13特徴）＋（相手強さ特徴）のみを使い、市場確率(単勝オッズ等)は
一切含まれていない。

## 出走回数・休養日数を統制しているか

**していない**。`kako5_race_count`・`間隔`等は両スクリプトの特徴リストに
含まれていない。

## O1/O2/O3のどれに相当するか

| スクリプト | 相当する区分 | 理由 |
|---|---|---|
| `_tmp_oppstrength_gate.py`のOPP特徴 | **O1に近い**（field-strength平均のrolling/max、`opp_best_beaten`のみ「複勝圏内」条件付き） | 個体識別なし、フィールド平均の集約のみ |
| `_tmp_elo_strict_gate.py`のELO特徴 | **O1とO2の混成**（強度指標はELO=O2的だが、集約方法はfield平均=O1的） | 独自の逐次ELO実装（既存`build_elo_feats.py`とは別実装）を使うが、対戦相手を個体識別せずフィールド平均に潰している |

**いずれもO3（個々の対戦相手を血統登録番号で追跡し、対象日までに判明した
相手の現在能力を使う）には該当しない**。O3固有の「対戦相手のその後の
判明」という遡及的更新メカニズムが両方に欠けている。

## 等価性の結論

**2本のscratchスクリプトはEXP12のO3と等価ではない**。両方とも
「フィールド平均強度」をas-ofに集約するO1/O2寄りのアプローチであり、
個体識別・対戦相手の事後的な評価更新というO3の核心的差分を持たない。
したがって、これらのコードをO3の代替として採用することはできず、
新規実装が必要である（下記§2以降）。ただし、以下の設計要素は参考に
流用できる: (a) `shift(1)`＋`expanding()`によるas-of集約パターン、
(b) 「複勝圏内で終えたレースに限定」という条件付き集約の発想、
(c) 格13特徴(GF)との組み合わせでLightGBMスタックする評価設計
（ただし本Stage1では市場確率・出走回数・休養日数を含めたfull-control、
かつtrain=2023/test=2023(meeting-day forward chaining)へ変更する）。

**train=2023/test=2024-2025という設計は採用しない**——本Stage1では
Stage0以降確立した「2023年developmentのみでmeeting-day forward-chaining、
2024・2025年は開封しない」という規律に従う。

---

関連: [[project_exp12_opponent_network]]
