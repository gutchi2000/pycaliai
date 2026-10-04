# 2026-10-03 Dry 開催日の診断（観測計画 v2.1.1 Stage 0、読み取りのみ）

**結論**: 監査は CONTRACT_NOT_MET（exit 4）を返したが、その理由 3 系統（t10/close_late の欠損、被覆不一致、final_stock の欠損）はどれも、Dry 側 collector（T−2 / TRIO / FINAL_RT）が実際に取りこぼした結果ではない。

- t10/close_late の欠損と被覆不一致は、**監査の範囲と期待値の式の問題**であることを実データで確認した。
- final_stock の欠損は、蓄積系オッズがまだ配信されていない時点で監査したための問題である可能性が高い。ただし確定はしていない。
- Dry の T−2・TRIO・FINAL_RT の取得 276 件は、すべて rc 0・録あり・race_key 一致・長さ一致・parser 一致だった。全組の発売もそろっていた（登録頭数基準）。
- 確認した方法: 保存済みの監査出力と report JSON を読んだ。加えて scratch で `audit()` / `concurrency()` を関数として呼び、dry と本番を合わせて再計算した（ledger 追記なし・repo への書込なし）。

---

## 1. 監査の結果表（20261003、`--dry`）

**判定**: decision = **CONTRACT_NOT_MET**、exit_code = **4**。並走判定（concurrency_verdict）は CONCURRENCY_OK（reasons なし）。予定 23R（bundle 由来。05040104 は bundle 外）。records は 69、journal_events は 277。

**契約未達の理由**
1. 必須 stage/spec の欠損数が閾値 1 を超えた: t10/0B31〜0B34、close_late/0B31〜0B34、final_stock_candidate/0B31〜0B35（いずれも 23 > 1）
2. 被覆不一致率が 1% を超えた（12 セル）: final_rt_candidate/0B31/wakuren、t2_candidate/0B31/{fukusho, tansho, wakuren}、t2_candidate/0B32/umaren、0B33/wide、0B34/umatan、0B35/trio、trio_t10/0B31/{fukusho, tansho, wakuren}、trio_t10/0B35/trio

### 欠損の原因の内訳（stage × spec × cause）

| stage | spec | cause | 件数 |
|---|---|---|---|
| t10 | 0B31 / 0B32 / 0B33 / 0B34 | task_not_fired | 23 ×4 = 92 |
| close_late | 0B31 / 0B32 / 0B33 / 0B34 | task_not_fired | 23 ×4 = 92 |
| final_stock_candidate | 0B31 / 0B32 / 0B33 / 0B34 / 0B35 | stock_race_absent | 23 ×5 = 115 |
| t2_candidate / trio_t10 / final_rt_candidate | 全 spec | — | 0 |

**レビュアー指定の 4 区分**

| 区分 | 件数 | 補足 |
|---|---|---|
| task not fired | 184 | 誤帰属。§2 参照。範囲を正しく取ると 0 |
| fetch rc | 0 | |
| race_key mismatch | 0 | |
| other | 115 | 内訳は stock_race_absent 115 |

「other」に入る細かい原因は no_records、record_not_stored、spec_capture_absent、record_without_raw_v1、stock_session_failed、stock_race_absent、unattributed の 7 種。このうち実際に出たのは stock_race_absent だけで、unattributed は 0。

### 並走の成功率（by_kind、監査の報告どおり）

| kind | events | solo_n | over_n | rc≠0 | solo_rate | over_rate | 重なり相手 |
|---|---|---|---|---|---|---|---|
| final_rt_candidate | 115 | 5 | 110 | 0 | 1.0 | 1.0 | final_stock_candidate(STOCK) 110 |
| final_stock_candidate(STOCK) | 1 | 0 | 0 | 0 | – | – | final_rt_candidate 1 |
| t2_candidate | 115 | 115 | 0 | 0 | **0.0** | – | なし |
| trio_t10 | 46 | 46 | 0 | 0 | **0.0** | – | なし |
| 合計 | 277 | solo 5/166 = 3.0% | overlapped 110/110 = 100% | | | | |

- **T−2 の取得が、別レースの本番 T−10 取得と重なった件数は 0**。dry と本番の journal を合わせて確認した。T−2 から本番 t10 までの最小間隔は 418.5 秒（p50 421 秒）。
- 発走間隔が 15 分以上あるため、構造上 T−2 と次レースの T−10 は約 7 分離れる。T−2 の 115 件は、どのプロセスの取得とも ±30 秒以内に重ならなかった。最も近いのは bodyweight collector で 118.8 秒。

### stage × spec ごとの成否（dry と本番を合わせて再計算、予定 23R）

| stage | spec | 保存先 | 取得/23 | rc≠0 | 0録 | key/len/parser NG | 区分 | 被覆NG（現行式） | 被覆NG（登録頭数式） | 枠連NG |
|---|---|---|---|---|---|---|---|---|---|---|
| t2_candidate | 0B31〜0B35 | dry | 23/23 ×5 | 0 | 0 | 0/0/0 | 1 | 100% | 0 | 1 |
| trio_t10 | 0B31, 0B35 | dry | 23/23 ×2 | 0 | 0 | 0/0/0 | 1 | 100% | 0 | 1 |
| final_rt_candidate | 0B31〜0B35 | dry | 23/23 ×5 | 0 | 0 | 0/0/0 | 4 | 0 | 0 | 1 |
| final_stock_candidate | 0B31〜0B35 | dry | 0/23 | STOCK セッション rc 0 | O1〜O5 が 0 件 | – | – | – | – | – |
| t10（本番） | 0B31〜0B34 | 本番 | 23/23 ×4 | 0 | 0 | 0/0/0 | 1 | 100% | 0 | 1 |
| close_late（本番） | 0B31〜0B34 | 本番 | 23/23 ×4 | 0 | 0 | 0/0/0 | 1 | 100% | 0 | 1 |
| t20（記述用・本番） | 0B31〜0B34 | 本番 | 22/23 | 1（05040105 0B31 rc_open −413 で以降中断） | 1 | 0 | 1 | 100% | 0 | 1 |
| exp05fs_t35（記述用・本番） | 0B34 | 本番 | 22/23 | 1（05040105 rc_open −413） | 1 | 0 | 1 | 100% | 0 | – |

- 時刻（T−2）: 発走前 p50 118.8 秒に取得、発表から取得までの遅延 p50 61 秒。5 spec 合計の所要は p50 0.31 秒 / p95 1.39 秒 / max 1.86 秒（60 秒の閾値に対して十分小さい）。
- raw と parser の 5 検査は 276 件（本番を合わせると 639 件）すべて fail 0。00:00 型の破損も 0。

### 他プロセスとの重なり（dry と本番の journal を合わせて再計算、全 4,362 件）

| kind（件数） | 重なり相手（±30 秒） | 現行定義の成功率 | 修正定義の成功率 |
|---|---|---|---|
| t10（92） | trio_t10 92、bodyweight 92、exp05fs_t35 4 | 0/92 | 92/92 |
| trio_t10（46） | t10 46、bodyweight 46、exp05fs_t35 2 | 0/46 | 46/46 |
| close_late（92） | jvlink_changes 92 | 0/92 | 92/92 |
| t20（89） | bodyweight 89、exp05fs_t35 57 | 0/89 | 88/89 |
| exp05fs_t35（92） | bodyweight 92、t20 60、t10 4、trio 4 | 0/92 | 91/92 |
| final_rt_candidate（115） | STOCK 110 | 110/110 | 110/110 |
| t2_candidate（115） | なし | solo 0/115 | solo 115/115 |
| 合計 | | overlapped 110/521 = 21.1% | overlapped 519/521 = **99.6%**、solo 120/120 |

- 修正定義は「rc 0・最新録の race_key/長さ一致・枠連以外の全 block で発売中の組数 = 登録頭数からの期待値」。
- trio_t10 と本番 t10 は同じレースを同じ秒に取得しており、本当の同時取得になっている。46/46 と 92/92 が成功した。
- 本番の journal にだけ出てくる測定対象外のプロセスと、その rc の分布（rc_init, rc_open）:
  - jvlink_changes（3,312 件）: (0,0) 1,693、(0,−114) 1,104、(0,−1) 515
  - analysis/bodyweight_forward/collector（407 件）: (0,0) 258、(0,−1) 149
  - exp05fs_calendar 1 件
- 現行定義のまま範囲を正しく取ると、判定は QUEUE_SERIALIZATION_REQUIRED（110/521）に変わる。これは被覆の式（§4）による見かけ上の失敗である。

---

## 2. t10 / close_late が 23/23 欠損している原因 → **[監査範囲/期待値の問題]（確認済み）**

- `--dry` を付けると、監査は forward store と journal を両方とも `*_dry` だけに切り替える（`analysis/obs_stage0_audit.py:476-478`）。
- 一方、必須 stage には本番ライン（`jvlink_odds` / t10_runner）しか書かない t10 と close_late が入っている（`:81`）。
- 本番の保存先には 10/03 の分が全部ある:
  - `data/forward_prices/20261003/` に t10 23、close_late 23、t20 23、decision 23、exp05fs_t35 23 件（すべて schema v2、jv_captures 付き）
  - `data/jvlink_fetch_journal/20261003/` に jvlink_odds の取得 365 件（t10 92、close_late 92、t20 89、exp05fs 92、すべて dry=false）
- dry と本番を合わせて再計算すると、t10/0B31〜0B34 と close_late/0B31〜0B34 はすべて 23/23 で欠損 0 になる。
- 本番 T−10 は 10/03 に動いていた:
  - `logs/t10_20261003.log` に、t10.ps1 -Schedule が PyCaLiAI_T10R_* を 23 本登録したことと、各 -Once の `[close] jvlink_odds attempt=1/3 (exit 0) ok=True` が残っている。Traceback は 0。
  - PyCaLiAI_T10 の LastRunTime は 10/04 9:00（結果 0x0）に上書き済み。10/03 の T10R_* タスクは期限切れで削除済みのため、10/03 の実行は上記のログ・録・journal から確認した。
- 「task_not_fired」という原因名は誤帰属である。タスクは起動していて、監査が別の保存先を見ていただけ。
- 直し方（1 行）: `--dry` のとき、本番 stage（t10/close_late）は本番の保存先と journal から読む（dry と本番の和で監査する）。

## 3. final_stock_candidate が 115 件欠損している原因 → **[未確定]**（期待値・タイミングの問題である可能性が高い）

- **FINAL は 10/03 18:30 に実行済み**。タスク本体は期限切れで削除済みだが、次の 3 点から確認できる。
  - transcript（`logs/obs_20261003_dry.log`）: 18:30:00〜18:30:35
  - final_rt（pid 36508）: 18:30:01.076〜18:30:30.789 に 115 件
  - STOCK（pid 50356）: `data/jvlink_fetch_journal_dry/20261003/20261003183031962_50356_jvlink_obs_RACE_STOCK.json`。内容は rc_init 0、rc_open 0、`n_records_returned 0`、error なし、所要 3.5 秒。
- つまりセッションは成功した（JVOpen("RACE","20261003000000",1) は rc 0 なので何かしらのデータはある）。ただし予定 23R の O1〜O5 は 0 件だった。
- 読んだ総数（`n_records_read` / `download_count`）は journal に保存されていない（`jv_journal.write_event` が書く項目に含まれない）。Python の標準出力もログに残っていない（`obs_schedule.ps1` の `Invoke-Py32` が `& py` の出力を戻り値として飲み込むため。transcript 48 本すべてで Python 出力 0 行）。このため「他の種別の録は読めたが O1〜O5 だけ無かった」のか「ほぼ何も無かった」のかは区別できない。
- **蓄積系 O1〜O5 は月曜に作成されるという強い傍証がある**。過去に蓄積系から取った O5 の 576/576 件（`reports/trio_portfolio_shadow_v2/raw_stock/`、8/18 取得）は次のとおりだった。
  - データ作成日は全件、開催週末の翌月曜（例: 07/04 土 → 07/06 月、08/15〜16 → 08/17）
  - 区分 `'5'`（確定・月曜版）、発表時分 `00000000`
  - レースごとに 1 版だけで、当日版は無い
  - 推定: 当日 18:30 の時点で、蓄積系にはその日の O1〜O5 がまだ無い。
- **REFETCH はまだ実行されていない**。PyCaLiAI_OBS_REFETCH_20261003 は LastRun 1999（未実行）、次回 10/04 20:00。今回の監査（10/04 11:20）は REFETCH より前で、しかも月曜の配信より前に final_stock を必須として要求している。
- 実欠陥になりうる点（推定）: REFETCH は日曜 20:00 なので、月曜配信より前にあたる。月曜 20:00 の REFETCH_20261004 は `--final-stock 20261004` を `--also-previous` なしで呼ぶ。このため対象レースは 10/04 分だけに絞られ（`jvlink_obs.py:207-216`、`obs_schedule.ps1:72-73`）、**今のスケジュールでは 10/03 の final_stock を取得する機会が無い**。月曜 20:00 に配信が間に合うかも未確認。
- 直し方（1 行）: 蓄積系の取得を月曜配信の後（火曜など）に、対象日を明示して（両日分を）行う。あわせて STOCK の読込総数と種別ごとの件数を journal に残す。監査は REFETCH 後に評価する。

## 4. 被覆不一致の原因 → **[監査範囲/期待値の問題]（確認済み）**

| stage/spec/bet | 不一致 | 主な原因 |
|---|---|---|
| t2_candidate/0B31/tansho、fukusho | 23/23 (100%) | 出走頭数の欄 = 0 |
| t2_candidate/0B32/umaren、0B33/wide、0B34/umatan、0B35/trio | 各 23/23 (100%) | 出走頭数の欄 = 0（C(0,k) = 0） |
| trio_t10/0B31/tansho、fukusho、trio_t10/0B35/trio | 各 23/23 (100%) | 同上 |
| t2_candidate、trio_t10、final_rt_candidate の 0B31/wakuren | 各 1/23 (4.35%) | 08040109 は 8 頭立てで枠連が発売されない |

**(a) 単複・馬連・ワイド・馬単・三連複の 100% 不一致**
- 発売中の組数の期待値を、録の**出走頭数**の欄から計算している（`jv_records.py:155-164`）。
- 発走前の中間オッズ（区分 1）では、この欄が全録で 0 だった。本番の t10/close_late/t20/exp05fs も含め、区分 1 の 990 録すべてが n_running = 0。
- 一方、実際の発売数は登録頭数どおりだった。例: 16 頭立てで単勝 16、馬連 120、三連複 560。
- 確定後（区分 4）の final_rt では出走頭数 = 登録頭数になり、23/23 一致した。
- このため、発走前の取得は `complete = False`、`cap.ok = False` になる（`jv_records.py:234`、`:260`）。並走の成功率（`obs_stage0_audit.py:178-179`）でも失敗として数えられ、solo_rate が 0.0 になる原因もこれである。

**(b) 枠連の 1/23**
- 京都 9R（2026100308040109）は 8 頭立てで、`hatsubai_wakuren = '0'`（枠連の発売なし）。36 slot すべて空白だった。
- `wakuren_slot_count(8)` は 28 を期待するが、8 頭以下では枠連が発売されないことを考慮していない（`jv_records.py:65-71`）。
- 全 stage で同じ 1 race が不一致になる。

**取消・除外の影響は無い（確認済み）**: 23R すべてで確定時の出走頭数 = 登録頭数。全 capture で未発売や 0 の組も 0。`reports/live_changes/20261003.json` にも馬体重（WH）しか無い。したがって取消の扱いは 10/03 では試されていない。

**直し方（1 行）**: 発走前は「登録頭数 − 取消による未発売 filler」を期待値にし、枠連は発売フラグ 0（8 頭以下）なら 0 とする。並走の成功判定は被覆の式から切り離す。

## 5. 実欠陥と監査側の問題の区別

| 事項 | 判定 | 根拠 |
|---|---|---|
| t10/close_late の 23/23 欠損（184 件） | 監査範囲/期待値の問題 | 確認済み（§2） |
| 被覆不一致 12 セル（本番 stage も同じ） | 監査範囲/期待値の問題 | 確認済み（§4） |
| dry 監査の CONCURRENCY_OK | 監査範囲の問題 | 確認済み（下記） |
| final_stock 115 件 | 未確定 | 推定（§3） |
| Dry collector（T−2 / TRIO / FINAL_RT）の取りこぼし | **実欠陥なし** | 確認済み |
| 記録系の不足 | 収集の実欠陥（軽微・記録系） | 確認済み（下記） |
| 本番 05040105 の −413 2 件 | 未確定 | 下記 |

- **dry 監査の CONCURRENCY_OK は実質の裏付けが無い**。重なり 110 件はすべて final_rt → STOCK の**逐次実行**だった。final_rt の終了 18:30:30.789 の 1.2 秒後に STOCK が開始しており、±30 秒の近さで数えたにすぎない。dry の範囲では同時取得が 1 件も観測されていない。T−2 の並走安全性は 10/03 のデータでは検定されていない（重なり 0）。
- **Dry collector に実欠陥は無い**。T−2 115、TRIO 46、FINAL_RT 115 の取得がすべて rc 0・1 録・race_key/長さ/parser 一致で、全組の発売もそろっていた（登録頭数基準）。`obs_dry_check` も ok=true（存在だけを見る検査）。
- **記録系の不足（軽微）**:
  - (i) STOCK の読込総数が journal に残らない
  - (ii) `Invoke-Py32` が Python の標準出力を戻り値として飲み込むため、ログに残らない。さらに `-Final` の `$c1 -ne 0` が配列との比較になり、終了コードの意味が崩れている可能性がある（コードを読んでの推定）
- **本番 05040105 の −413 2 件**: t20 の 0B31（12:05:01、以降の spec は中断）と exp05fs_t35 の 0B34（11:50:02）。どちらも bodyweight collector の取得と約 1 秒以内に重なっていたが、重なった取得全体では 519/521 が成功している。並走が原因かは判別できない。Stage 0 の必須対象ではなく、記述用のみ。
- **Dry の分離漏れ（軽微）**: Dry の TRIO も、git 管理下の `reports/trio_portfolio_shadow_v2/manifest.jsonl` に 10/03 分 69 行を追記している（raw は `_dry/` に分かれている）。

**主なファイル**
- `E:\PyCaLiAI\analysis\obs_stage0_audit.py`
- `E:\PyCaLiAI\jv_records.py`
- `E:\PyCaLiAI\jvlink_obs.py`
- `E:\PyCaLiAI\obs_schedule.ps1`
- `E:\PyCaLiAI\reports\obs_stage0_20261003_dry.json`
- 再計算に使った scratch: `...\scratchpad\dry_diag\union_audit.py`、`conc2.py`、`cov.py`、`stagetable.py`、`union_audit_20261003.json`
