# JV-Link 隔離可能性調査 と P1-forward-only 設計（2026-09-22）

**状態**: READ-ONLY 調査。**JV-Link の API は一切呼んでいない**
（`JVInit`/`JVOpen`/`JVSetup`/`JVGets`/`JVRead` いずれも未実行）。
レジストリ・ファイルシステム・リポジトリ内コード・既存利用箇所の**読み取りのみ**。
production 変更・scheduler 変更・モデル変更・再学習・EXP14 Stage 1・
ROI 評価も行っていない。

**ユーザー判断（受理済み）**: 現在の TARGET 共有データストアに対する
JV-Link 期間一括セットアップは**実行しない**。

---

## 1. 結論のスコープ（確定表現）

以下を本件の確定スコープとする。

1. **v6 は障害競走を実学習していた**
   （train split 14,716 行 / 1,267 レース、確定判定 = `トラックコード(JV)` 51-59）
2. **weekly live prediction/betting は障害を対象外にしている**
   （`data/weekly` に障害は含まれず、bundle・印・買い目にも 0 件）
3. **kako5 13 特徴は TARGET kako5 CSV 経由で障害歴を既に保持しており、
   本問題の影響外**（過去走スロットに `TD='S'` が実在、実測 74 件）
4. **現時点で実影響が確定したのは 3 特徴のみ**
   - `horse_fuku30`（73 行）
   - `horse_fuku10`（45 行）
   - `hist_same_place_best_pos`（15 行）
5. **`course_n_prev` / `course_win_rate` / `course_top3_rate` は実測 0 件**
6. **騎手・調教師 rolling（`jockey_n_prev`/`jockey_win_rate`/`jockey_top3_rate`/
   `jockey_fuku30`/`jockey_fuku90`/`trainer_fuku30`/`trainer_fuku90`）は
   コード欠落のため未測定**
7. **0.236% は全体影響率ではなく、決定可能な馬側特徴についての下限**

### 1.1 P1 の呼称

P1 を「全履歴特徴の修正」と呼ばない。正しい呼称は:

> **weekly から欠落する障害履歴による一部 rolling 特徴の補完**

---

## 2. JV-Link 隔離可能性（read-only 調査、API 未呼び出し）

| 確認項目 | 結果 | 根拠 |
|---|---|---|
| ローカルストアの実パス | **`C:\ProgramData\JRA-VAN\Data Lab`**（`cache` / `data` / `event` / `pictures`、3,157 ファイル / 512.9 MB） | レジストリ値 `savepath`、および実ディレクトリの read-only 列挙 |
| 保存先は user 単位か machine 単位か | **machine 単位**。設定は `HKEY_LOCAL_MACHINE\SOFTWARE\WOW6432Node\JRA-VAN Data Lab.\uid_pass` にあり、実体は全ユーザー共有の `C:\ProgramData` 配下 | `HKCU` 配下に JRA/JV/turf 関連キーは**存在しない**ことを確認済み |
| どこで決まるか | レジストリ `HKLM\...\JRA-VAN Data Lab.\uid_pass\savepath`（`saveflag=1`）。GUI は `C:\Program Files (x86)\JRA-VAN\Data Lab\JV-Link設定.exe` | 同上。`installpath` も同キーにある |
| 任意の別ディレクトリへ変更可能か | **レジストリ値としては変更可能に見えるが、変更すると同一マシンの全 JV-Link 利用者（TARGET を含む）が新パスを見るため、隔離にならない**。`JV-Link設定.exe` GUI で変更する想定。**per-process / per-session の上書き手段は確認できなかった（不明）** | 値が HKLM 単一であることから。JV-Link API に保存先を引数で渡す口は、リポジトリ内の既存利用箇所には存在しない |
| 別 Windows user で独立するか | **独立しない**。設定が HKLM、実体が `C:\ProgramData` のため、別ユーザーでログインしても**同じストアを共有する** | HKCU にキーが無いことと ProgramData の性質から |
| TARGET と JV-Link COM が同じストアを共有する根拠 | 設定が HKLM 単一・保存先が ProgramData 単一であり、COM `JVDTLab.JVLink` は全プロセス共通にこのレジストリを読む。TARGET も同じ COM を使う（本プロジェクトの JV-Link 利用キーは TARGET 側設定を共有している旨が `jvlink_probe.py` に明記） | `jvlink_probe.py` docstring「SDK＋利用キー(TARGET設定)が生きてる」 |
| 同時アクセス時の lock / 破損リスク | **不明**。公式資料が手元に無く、実呼び出しで試さない方針のため未検証。ただし `event/` 配下に稼働中の実ファイルがあり（当日 21:52 更新）、T-10/T-20 の常駐タスクが同じ COM を使うため、**一括 setup と常駐タスクの同時実行は避けるべき**と判断する | ディレクトリの更新時刻。リスクの定量化はしていない |
| setup が上書き・更新・追加のどれか | **観測上は「期間単位ファイルの追加」に見える**（`data/` は `RAVM{YYYYMM}99{作成日時}.jvd` の月次ファイル群、2024-06〜2026-05 の 24 か月ぶん）。ただし**同一月を再取得した際に上書きされるのか別ファイルが増えるのかは未確認（不明）** | ファイル名規則と一覧。再取得を試していないため断定しない |
| backup/restore が公式に可能か | **不明**。公式手順を示す資料が手元に無い（`C:\Program Files (x86)\JRA-VAN\Data Lab` 配下に PDF/CHM/HTML/TXT のドキュメントは存在しない）。ディレクトリを丸ごとコピーする非公式手段は物理的には可能だが、公式サポートの有無は確認できていない | ドキュメント探索の結果 |
| setup 対象期間を 1 日へ限定できるか | **できない**。`JVOpen(dataspec, fromtime, option)` に**終了時刻の引数が無い**ため、取得範囲は常に **[fromtime, 現在]** になる。1 日に限定したい場合は `fromtime` を当日にするしかなく、過去日を指定すればそこから現在までが対象になる | リポジトリ内の全 `JVOpen` 利用箇所（`jvlink_trio_odds.fetch_stock_o5`、`analysis/mcond/exp05_forward_shadow/jvlink_race_day_probe.py`）のシグネチャ |
| option 値ごとの動作 | `option=1`（通常データ）・`option=2` は**過去〜当日の確定データで動作実績あり**、未来日は両方とも `rc=-1`（`jvlink_race_day_probe.py` に 2026-09-19 の実測記録）。`option=4` は本調査で 20260307 を指定したところ **9 分以上ブロック**した（前回セッション、今回は未実行）。**option=3 は未検証（不明）** | 既存コードの実測記録 + 前回セッションの観測 |

### 2.1 補足: 取得はダウンロードを伴う

既存の `jvlink_trio_odds.fetch_stock_o5()` は
`JVOpen("RACE", from_ts, 1)` の戻り値 `(rc, readcount, dlcount)` を受け、
**`JVStatus()` が `dlcount` に達するまでポーリングしてからデータを読む**
実装になっている。つまり蓄積系 `JVOpen` は**サーバからのダウンロードを伴い、
その成果物がローカルストアへ書かれる**（`data/` の月次 `.jvd` 群がその実体）。

`fromtime` を 6 か月前にすれば 6 か月ぶんが対象になる。これが前回
ブロックした理由と整合する。

### 2.2 秘匿情報について

レジストリ `uid_pass` には利用キー（`servicekey` / `ukey`）が平文で
格納されている。**本報告書には値を記載しない**（キー名の存在のみ記録）。

---

## 3. 代替取得経路の再確認

| 経路 | 可否 | 根拠 |
|---|---|---|
| **TARGET GUI から過去の障害 race card を 1 日単位で shadow export** | **未確認（要ユーザー操作）**。TARGET は GUI アプリであり、こちらからは操作していない。ただし `data/bunseki`（出走馬分析、122 列・血統登録番号/騎手コード/調教師コード/クラス完備）が**実際に 20260905/0906 の障害レースを含んでいた**ことは実測済みで、**TARGET の出走馬分析エクスポートは障害を除外していない**。過去日を GUI から出せるかは TARGET の仕様次第で、**ユーザーにしか確認できない** | `out/alt_source_coverage_matrix.json`、`missing_race_class_verification.json` |
| **現 JV-Link ローカルストアに既存の障害データを read-only 取得** | **不可（サンクションされた手段としては）**。`data/` には **2024-06〜2026-05** の月次 RA/SE ファイルが実在し、欠落期間のうち **2026-03/04/05 分は物理的に存在する**。しかしファイルは「10 バイトヘッダ + zlib」の内側が**さらに非公開のエンコード**であり（展開後に `RA` リテラルも CRLF レコード区切りも存在せず、制御バイトを含む）、公式のアクセス経路は COM API のみ。非公開形式のリバースエンジニアリングは JRA-VAN ガイドライン遵守方針（[[project_jravan_guideline_compliance]]）に反するため**採らない**。解析は**これ以上行っていない** | ストアの read-only 列挙 + 先頭バイトの確認のみ |
| **`data/bunseki` 等を将来日だけ継続保存** | **可能**。既存の取り込み経路（`place_weekly.py` / `parse_bunseki.py`）がすでにこの形式を扱える。現状 8 日ぶんしか無いのは**保存運用が最近始まったため**であり、継続保存は運用手順の追加だけで足りる。**追加のデータ取得経路を必要としない最有力の候補** | `data/bunseki/` の実ファイルと [[project_target_ichiran_file_discovery]] |
| **開催当日の race card から障害出走馬・騎手・調教師・血統登録番号を履歴専用に保存** | **可能性高**。当日リアルタイム系は `JVRTOpen(spec, race_key)` の**レース単位**呼び出しであり、`jvlink_changes.py` が既に `0B12`（出走馬名表）を含む `0B11〜0B16` を毎開催日に総当たりしている。**期間一括 setup を一切伴わない**。ただし「0B12 の返却内容に血統登録番号・騎手コード・調教師コードが含まれるか」は**本調査では未確認**（API を呼んでいないため） | `jvlink_changes.py:37-38`、`jvlink_odds.py` の JVRTOpen 実装 |
| **レース終了後に結果コードだけ追記** | **可能**。`data/kekka/{date}.csv` に障害レースの着順が既に存在する（欠落 598 行中 564 行が finished）。race_id + 馬番 で card 側と結合すれば足りる | `out/missing_set_manifest.json` |
| **JRA-VAN/TARGET の既存自動出力設定で障害を別ファイルへ出せるか** | **未確認（要ユーザー操作）**。TARGET の出力設定 GUI はこちらから触っていない | — |

**production bundle へ混ぜる案は採用しない**（ユーザー指示どおり）。

---

## 4. P1-forward-only 設計（**未実装**、設計と取得可能性の確認のみ）

### 4.1 位置づけ

**weekly から欠落する障害履歴による一部 rolling 特徴の補完**を、
**将来の障害レースだけ**を対象に行う。過去 backfill は含まない。

### 4.2 レコード契約（案）

障害レースは専用ストア（例 `data/history_only/jump/{date}.jsonl`、
production の `data/weekly` とは**別ディレクトリ**）へ append-only で蓄積する。
各行は以下を必須フィールドとして持つ。

```
{
  "schema_version": "jump-history-only/1",
  "history_only":              true,
  "prediction_eligible":       false,
  "bet_eligible":              false,
  "task_registration_eligible":false,

  "ped_id":        <血統登録番号>,        # 主キー
  "race_id":       "<16桁>",
  "race_date":     <YYYYMMDD>,
  "post_time":     "<HH:MM>",             # 発走時刻
  "jump_class":    "<障害区分>",
  "umaban":        <馬番>,
  "jockey_code":   <騎手コード>,
  "trainer_code":  <調教師コード>,

  "finish_pos":            <着順 or null>,
  "result_available_at":   "<ISO8601 or null>",  # 結果が利用可能になった時刻
  "row_hash":              "<sha256>"
}
```

### 4.3 hard invariants

1. `history_only=true` / `prediction_eligible=false` / `bet_eligible=false` /
   `task_registration_eligible=false` を**全行に必須**とする
2. **T-35 / T-20 / T-10 / Vote の対象外**。スケジューラのレース列挙が
   このストアを参照しない
3. **印・買い目・bundle の通常 race list へ入れない**
4. **血統登録番号を主キー**とする（馬名 join は使わない）
5. `finish_pos` が確定するまで**履歴集計へ入れない**
6. `result_available_at` を記録し、**同日の後続レースへ使う場合も
   `result_available_at` を超えた後だけ**参照を許す
7. **append-only**（既存行を書き換えない。既存パスがあれば連番で別ファイル
   — 既に `jvlink_trio_odds.write_append_only()` が同じ方針で実装済み）
8. `schema_version` と `row_hash` を全行に付与
9. **通常平地レースへの混入を hard test で拒否**

### 4.4 hard test（案、未実装）

| テスト | 内容 | 失敗時 |
|---|---|---|
| T1 混入拒否 | bundle / cowork_input / TACT 公開物の race list に
`history_only=true` の race_id が 1 件でも現れたら FAIL | 生成を中止 |
| T2 スケジューラ非登録 | T-35/T-20/T-10/Vote のタスク登録対象に
jump race_id が現れないこと | 登録を中止 |
| T3 時点安全 | 履歴集計に入る行は全て
`result_available_at <= 参照時刻` かつ `race_date < 対象レース日`
（同日利用時は時刻比較） | 集計を中止 |
| T4 append-only | 既存行の `row_hash` が変化していないこと | 書込を中止 |
| T5 主キー | `ped_id` が null の行を拒否 | 行を捨てる |
| T6 schema | `schema_version` 不一致を拒否 | 読込を中止 |

### 4.5 取得可能性の現状

| 必要項目 | 取得元候補 | 状態 |
|---|---|---|
| race_id / 開催日 / 発走時刻 / 障害区分 | `data/bunseki`（継続保存）または `JVRTOpen("0B12")` | bunseki は**実証済み**、0B12 は未確認 |
| 血統登録番号 / 騎手 / 調教師 | `data/bunseki`（122 列に全て存在、実測） | **実証済み** |
| 着順（finish） | `data/kekka/{date}.csv` | **実証済み**（障害レースの着順は既に入っている） |
| 結果利用可能時刻 | 取り込み時のタイムスタンプ | 実装で付与可能 |

→ **`data/bunseki` の継続保存 + 既存 `data/kekka` の結合だけで、
forward-only の必須項目は全て揃う。JV-Link の一括 setup は不要。**

---

## 5. 過去 backfill の判断分岐

| A の条件 | 充足 |
|---|---|
| 別 Windows user または別ローカルストア | ❌ **不可**。設定は HKLM、実体は `C:\ProgramData` で machine 単位（§2） |
| TARGET 本番と共有しない | ❌ **不可**。同一 COM・同一レジストリ・同一 ProgramData を共有 |
| 1 日限定 setup 可能 | ❌ **不可**。`JVOpen` に終了時刻の引数が無く、範囲は [fromtime, 現在] |
| shadow 出力先 | ⭕ 出力先だけは分離できる（しかしストアへの書込は分離できない） |
| rollback 不要な完全隔離 | ❌ **不可** |

**判定: B（隔離できない）。**

→ **過去 backfill は中止する。共有ストアでは実行しない。**
**P1-forward-only 案だけを候補として残す。**
ユーザーへの実行案の提示も行わない（A の条件を満たしていないため）。

---

## 6. 騎手・調教師影響の未測定範囲

**馬名・騎手名の曖昧 join による補完は行わない**（ユーザー指示）。
以下は**測定できないことの記述**であり、推測値ではない。
**§1 の 0.236% にこれらを加算しない。**

| 項目 | 値 |
|---|---:|
| 未測定の障害レース数 | **47**（欠落 50 レースのうち、`data/bunseki` で騎手/調教師コードを復元できた 3 レースを除く） |
| 未測定の騎乗・出走行数 | **582**（欠落 598 行のうち復元できた 16 行を除く） |
| 未測定期間 | **2026-03-07 〜 2026-09-06** |

**影響し得る後続 serve 期間と理論的最大波及**:

- `jockey_fuku30` / `trainer_fuku30`: 当該主体の**直近 30 走**窓 →
  欠落 1 件がその騎手・調教師の**後続 30 行**に入りうる
- `jockey_fuku90` / `trainer_fuku90`: 同じく**直近 90 走**窓 →
  **後続 90 行**に入りうる
- `jockey_n_prev` / `jockey_win_rate` / `jockey_top3_rate`: 馬×騎手ペアの
  **全キャリア累積**（窓なし）→ 当該ペアの**以後すべての走**に入りうる

理論的最大波及件数（騎手側・90 窓）: 582 × 90 = **52,380 行分のスロット**。
これは 2026 serve 母集団 33,827 行を上回るため、**上限としては
「serve 全行」と等価であり、情報量を持たない**。
下限は 0（欠落騎乗をした騎手・調教師が 2026 の平地 serve に一度も
現れない場合）。

**結論: 実測不能。** 騎手コード・調教師コードが欠落行に存在せず、
復元可能なソースも 44/46 日で存在しないため
（`ALT_SOURCE_RECOVERY_AUDIT_20260922.md` §5）、
**現データでは区間 [0, serve 全行] より狭められない。**

**測定可能になる時期**: forward-only 収集を開始した日 D 以降に発生する
障害レースについては、その時点から騎手・調教師コードが揃うため測定可能になる。
`*_fuku90` の窓が完全に新データで満たされるには、該当主体が 90 走を
積むだけの期間が必要。**D より前（2026-03-07〜09-06）の 47 レース /
582 行は、過去 backfill をしない限り恒久的に測定不能**である。

---

## 7. 現時点の設計比較

| | **P1（第一候補）** | **P2** | **P3** |
|---|---|---|---|
| 内容 | 障害を予測・購入対象へ戻さず、**履歴専用に将来収集**する | flat-only vNext | 現状維持 |
| 状態 | **方向性維持。実装しない** | **別研究。今回は開始しない** | P1 の安全な取得元が確保できなければ採用 |
| 前提 | **共有 JV-Link ストアへの一括 setup を前提にしない**（§5 で B 判定） | — | — |
| 取得元 | `data/bunseki` 継続保存 + `data/kekka` 結合（§4.5、**追加の取得経路不要**） | 不要 | 不要 |
| 過去分 | **backfill しない**（§5）。2026-03-07〜09-06 の欠落は恒久的に残る | 該当なし | 該当なし |
| 是正できる範囲 | 収集開始日以降の `horse_fuku10/30`・`hist_same_place_best_pos`、および騎手・調教師 rolling | 非対称を別方向から解消 | なし |
| 最大リスク | **除外レースを誤って購入**（§4.3-4.4 の hard gate で封じる必要あり） | 再学習コスト | 非対称の固定 |

**共有 JV-Link ストアへの一括 setup は P1 の前提にしない。**

---

## 8. 未確認事項（ユーザーにしか確認できないもの）

1. TARGET GUI から過去日の障害 race card を 1 日単位で shadow export できるか
2. TARGET/JRA-VAN の既存自動出力設定で障害を別ファイルへ出せるか
3. `JVRTOpen("0B12", race_key)` の返却に血統登録番号・騎手コード・
   調教師コードが含まれるか（API 未呼び出しのため未確認）
4. JV-Link ローカルストアの backup/restore 公式手順の有無
5. 同一月の再取得時に `.jvd` が上書きされるか追加されるか
6. `JVOpen` の `option=3` の動作

---

## 9. 停止事項

本調査では: **JV-Link API 呼び出し（`JVInit`/`JVOpen`/`JVSetup`/`JVGets`/
`JVRead` を含む一切）**・JV-Link setup 実行・production 変更・
scheduler 変更・モデル変更・再学習・EXP14 Stage 1・ROI 評価を
いずれも行っていない。非公開ファイル形式の解析も行っていない。

関連: `docs/research/JUMP_RACE_CONTRADICTION_AND_ASOF_IMPACT_20260922.md`,
`docs/research/ALT_SOURCE_RECOVERY_AUDIT_20260922.md`
