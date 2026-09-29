# 観測計画 v2.1 Phase 2 実装報告（2026-09-29）

**宛先**: 戦略企画部 / Fable（レビュアー判断部）
**正本**: `docs/research/OBSERVATION_PLAN_20260928.md` v2.1（sha256 `a23ef19352b7…185d8c`、版固定 commit `bf952d3f`）
**branch / worktree**: `obs-collector-v21-phase2` / `C:/Users/gutch/Documents/Codex/PyCaLiAI-obs`（起点 `bf952d3f`、未 push）

## 状態

- 収集ラインの差分を隔離 branch に実装し、テストした。**本番 master への merge・タスク登録・収集開始はしていない**。
- collector は shadow 収集専用。買い目・投票設定（`masters_vote`）・`compute_bets`・`validate_cowork_bets`・`t10_runner`・`t10.ps1`・`production_policy` は変更していない。
- 最初の 2 開催日は outcome・払戻・ROI・性能を開かない。Stage 0 監査が見るのは欠損率・時刻・全組被覆・raw/parser 一致・並走成功率だけ。500R 到達前は性能・ROI・帯選択を実行させない guard を置いた。

## 1. 指示との対応

| 指示 | 実装 |
|---|---|
| 既存 forward_prices へ T−2 候補断面 | `jvlink_obs.py --stage t2_candidate`：0B31〜0B35 を同一プロセスで連続取得し、`data/forward_prices` へ schema v2 で追記。`obs_schedule.ps1 -Schedule` が発走 2 分前のレース毎タスク（WakeToRun）を登録。「T−2 決定」とは呼ばない |
| 現行 close を close_late へ改名 | `forward_prices.canonical_stage()`：`close` を `close_late` として保存し、`stage_requested=close` を残す。v1 録は改名せず読込時に別名解決。`forward_price_eval.py` とタイミング canary は別名に対応済み |
| stream 別の final 候補を raw のまま保存 | `final_rt_candidate`（RT 系、当日 18:30 に全レースを 1 本ずつ順番に取得）と `final_stock_candidate`（蓄積系 `JVOpen("RACE")` の O1〜O5、当日 18:30＋翌日 20:00 に再取得して不変性を確認）。`final` の認定はしない |
| 全 stage・全券種で raw・区分・announced_at・取得時刻・発売フラグ・票数計・枠連ブロック・parse 済み全組を append-only 保存 | `jv_records.py`（COM 非依存）の capture を `jv_captures` として保存。本番 stage（t10/t20/close_late/exp05fs_t35）は `jvlink_odds.py` の 0B31〜0B34、T−2/final は 0B31〜0B35、三連複 shadow は 0B35＋0B31 |
| 三連複 0B35 collector を同じ運転系へ登録 | `obs_schedule.ps1` が発走 10 分前の `PyCaLiAI_OBS_TRIO_<rid>`（WakeToRun、本番 T10R と同じレース毎タスク方式）を登録。§5.1 の登録前 Dry 開催日用に `-Dry`（`_dry/{date}` と `forward_prices_dry` に分離）と `analysis/obs_dry_check.py` を用意 |
| T−2・他レースの T−10・trio shadow の重なり時間帯の成功率を記録 | `jv_journal.py`：JV-Link の全取得（本番 `jvlink_odds`・`jvlink_obs`・三連複）を 1 取得 1 ファイルで記録（開始・終了 ms、rc、録数、process、stage）。`analysis/obs_stage0_audit.py` が capture と突き合わせ、別プロセス同士が ±30 秒以内に重なった取得の成功率を単独取得と比べる |
| 区分コードを推測で final へ固定しない | 区分・発売フラグは値を保存するだけで解釈しない。テストで確認 |
| 最初の 2 開催日は欠損率・時刻・全組被覆・raw/parser 一致・並走成功率だけを監査 | `python -m analysis.obs_stage0_audit --dates <d1> <d2>`。audit hook で結果系ファイルの open を監視し、結果・払戻を読む関数は import しない（AST テスト） |
| 並走成功率 95% 未満、または `00:00` 型破損 1 件で直列化を先に | 監査の判定が `QUEUE_SERIALIZATION_REQUIRED`（exit 3）になる。キュー直列化の実装は、この判定が出た時点で行う（計画 §3 (6) の順序） |
| 500R 前に性能・ROI・帯選択を実行しない | `analysis/obs_guard.py` の `assert_performance_allowed()`。label-free に数えた有効 race が 500 未満なら PermissionError |

## 2. 本番経路を変えていないことの根拠

- **`jvlink_odds.py`（T−10 の価格取得）**
  - JV-Link の呼び方・失敗時の空返し・例外の伝播は変えていない。取得時刻と rc を添える `fetch_records_timed` を足し、`fetch_records` の返り値と引数は元のまま。
  - `reports/live_odds/{race}.json`（compute_bets の入力）は `bf952d3f` 版と同一。t10 と close の 2 stage で、合成録を使ったテストで確認した。終了コードも同一。
  - raw の構造化や raw 付き保存が失敗しても、従来の保存（market のみ、`capture_error` 付き）の成否だけで終了コードを決める。テスト済み。
  - 追加の処理時間は、実録 16 頭・4 spec で中央値 7.0 ms、最大 11.1 ms（32-bit）。JV-Link 取得は約 2.5 秒。
- **shadow collector は latest view に書かない**。`jvlink_obs.py` の出力は forward store とジャーナルだけ。`obs_schedule.ps1` の本体には masters_vote・compute_bets・validate・live_odds・通知・git・sync-hf の参照が無い（テストで確認）。
- **蓄積系の読み捨て**：`JVOpen("RACE")` から残すのは対象レースの O1〜O5 だけ。HR（払戻）・SE（着順）などは種別 2 文字とレースキーだけ見て捨てる（`stock_route`、テスト済み）。蓄積系の取得は共有ストアへのダウンロードを伴うため、レース時間帯のタスクと重ならない 18:30 / 20:00 にだけ置いた。

## 3. 検証

- **テスト**：`tests/test_obs_phase2.py` の新規 44 本と、既存の forward_prices / eval / canary の 37 本がすべて通過。
- **全 suite**：259 passed / 17 failed / 1 skipped。17 failed は `test_jump_race_p0_gate.py` で、`bf952d3f` の clean worktree でも同じ 17 本が落ちる（215 passed）。gitignore 対象の出馬データが worktree に無いため race_eligibility が fail-closed になる既存事象で、回帰は 0。
- **32-bit Python 3.12.10**：import と実録の構造化を確認した。
- **live smoke（Dry、読取のみ）**：終了済みレース `2026092706040911` を `jvlink_obs --stage t2_candidate --dry` で 1 回取得し、worktree の `forward_prices_dry` に書いた。
  - 5 spec すべて rc 0、長さ一致、全組被覆、既存パーサと一致。
  - 馬連 O2 の録長 2040 を実録で初めて確認した（それまではレイアウトからの期待値）。
  - 5 spec 連続取得の所要は約 2.9 秒。区分 `4`、発売フラグ `7`（どちらも値の保存のみ）。
- **実録照合**：`reports/live_odds/raw`（O1/O3/O4）と三連複 `_smoke`（O1/O5）の実録 7 本で、長さ・race_key・全組・既存パーサ一致を確認した。枠連 slot 数の仮定式（登録 16 頭で 36、14 頭で 34）も実録と一致した。

## 4. 稼働手順（いずれも承認が必要）

1. このブランチをレビューし、master へ pathspec 限定で merge する。本番の T−10 経路（`jvlink_odds.py`）に触れるため、Fable の確認を要する。
2. §5.1 の登録前 Dry 開催日：`.\obs_register_tasks.ps1 -Dry -Apply`（毎週土日 09:05 の `PyCaLiAI_OBS_Schedule` を登録）。
   Dry 日の夜に `python -m analysis.obs_dry_check --date <日付>` で、全レースの 0B35・同時点 0B31・manifest の sha256/発表時刻・forward の trio_t10/t2_candidate を確認する。
3. 通過後に `.\obs_register_tasks.ps1 -Apply` で本番記録へ切り替える。最初の 2 開催日の夜に `python -m analysis.obs_stage0_audit --dates <d1> <d2>` を実行する。
4. 各 collector が実際に登録された時点で `docs/research/COLLECTION_WAIT_REGISTRY.md` に行を追加する（計画 §12）。

Dry は現在、T−2 と三連複の両方にかかる。§5.1 の Dry 要件は三連複だけなので、T−2 を初日から本番記録にするかは判断を仰ぐ。

## 5. 未解決・注意点

1. **並走の未計測経路**：`jvlink_changes.py`（`changes.ps1` の当日変更取得）の JV-Link 取得はジャーナル化していない。
2. **本番に未追跡の読み手**：`analysis/evaluate_wide_residual_forward.py` は本番 tree で未追跡のため branch に含まれない。`stage == "close"` で比較しているので、merge 後の `close_late` 録を拾えない。`canonical_stage()` への対応が要る。
3. **旧三連複ランチャ**：`t10_trio_shadow.ps1`（watch 方式、手動起動）はレース毎タスクで置き換わるが、ファイルは変更していない。
4. **枠連の期待 slot 数**：登録頭数からの仮定式で、実録 3 本と一致した。Stage 0 では契約の合否から外し、別枠で不一致率を報告する。
5. **`.gitignore` の既存 NUL**：HEAD 時点から UTF-16 の `path/to/pred.csv` 行（NUL 18 バイト）が混入しており、git が binary 扱いする。今回の追記 3 行（ジャーナルと Dry store）は有効だが、既存行は触っていない。
6. **本番 tree の別件**（Phase 1 関連、今回は触らない）：
   - `data/masters_vote.json` の作業ツリーに、実行部が Phase 1 指示の到着前に入れた `enabled=false` の未 commit 変更がある。HEAD は `enabled=true` で、復旧報告の記録と食い違う。扱いは判断を仰ぐ。
   - local master は origin に対して 29 ahead / 1 behind。

## 6. Fable 実装レビュー必須 4 点の反映（2026-09-29）

`09143ba6` は変更していない。後続 commit で次のとおり対応した。いずれも未 merge・未 push で、タスク登録・`-Apply`・実データ収集もしていない。

| # | commit | 内容 |
|---|---|---|
| 1 | `dec0281d` | 本番で未追跡だった `analysis/evaluate_wide_residual_forward.py` と、その依存先 `canonical_settlement.py`、それぞれのテストを byte 一致のまま追跡対象にした（sha256 は commit message に記載）。依存先を追跡しないと branch 上で import も検証もできないため |
| 1 | `4fe9776e` | `load_price_lineage` の stage 比較を `canonical_stage()` 経由にし、`close_late` と v1 の `close` の両方を締切録として数える。テスト 4 本を追加（v1 close / close 要求→close_late / close_late 直書き / 他 stage は非該当） |
| 2 | `a1964c7a` | 観測用 3 行を一時的に外し、`bf952d3f` と byte 一致の `.gitignore` に戻す |
| 2 | `c3147123` | UTF-16/NUL のゴミ行（`path/to/pred.csv`、NUL 18 byte）を空行 1 行に置き換える。`git check-ignore -v -n --no-index -z` を 82,282 path で比較し、ignored 71,475 → 71,475、変化 0 |
| 2 | `db150e04` | 観測用 3 行を text diff として再追加。変化は想定した 3 path だけ（ignored 71,475 → 71,478） |
| 3 | `fd8c2a08` | Stage 0 監査の重なり相手を journal 内の全 JV-Link 取得へ広げ、`CONTRACT_NOT_MET` を独立判定にした。詳細は下記 |
| 4 | `b9516e8c` | 500R guard を静的検査で強制し、件数を上書きする引数を削除した。詳細は下記 |

**必須 3 の中身**
- **重なり相手**：journal 内の全取得。本番の t10 / t20 / close_late / vote / exp05fs_t35、T−2、三連複、final RT、蓄積系 STOCK、`jvlink_changes`、EXP05-F calendar、その他すべてを含む。
- **判定の前提**：スケジュールが重ならないことは仮定しない。実際の開始・終了時刻で ±30 秒の重なりを判定し、同一レースでも別プロセスなら重なりとして数える。
- **識別名**：
  - `event_kind()` がすべての取得に識別名を付ける。
  - `jvlink_changes.py` は `set_context(process="jvlink_changes")` で名乗る。
  - EXP05-F calendar は JV-Link セッション 1 回を `process=exp05fs_calendar` として記録する（fail-open、戻り値は不変）。
  - 名前の付いていない取得は、実行スクリプトの repo 相対パス（例：`analysis/bodyweight_forward/collector`）で記録し、`unknown` を残さない。
- **判定と終了コード**：
  - `CONTRACT_NOT_MET`（exit 4）：必須 stage × spec の欠損率 > 1%、被覆不一致率 > 1%（枠連を含む）、raw/parser 不一致、識別できない process のいずれか。
  - 並走側の判定は常に別途計算して報告する：`QUEUE_SERIALIZATION_REQUIRED` 3 / `INSUFFICIENT_OVERLAP_OBSERVED` 5 / `CONCURRENCY_OK` 0。

**必須 4 の中身**
- **静的検査** `analysis/obs_guard.py` の `find_unguarded_modules()`：
  - 対象は追跡 .py 全件の AST。
  - 違反になるのは、観測 stage を参照し、かつ次のいずれかに当たるのに `assert_performance_allowed()` を呼ばない module。
    - 結果・払戻・着順系の import（名前 import・動的 import を含む）
    - path や列名の文字列
  - guard に `root=` を渡して store を差し替えるのも違反。
  - 検査コマンド：`python -m analysis.obs_guard --check`（違反があれば exit 1）。
- **件数上書きの除去**：`assert_performance_allowed(stream, root)` から件数上書き引数を削除した。
- **現状**：追跡 .py 701 本で違反 0。観測 stage を参照する 7 本は、いずれも結果系を扱わない。

**テスト**
- `tests/test_obs_phase2.py` 81 本、stage alias 4 本、評価器・settlement 23 本、既存の forward_prices 系 37 本がすべて pass。
- 全 suite は 323 passed / 17 failed / 1 skipped。17 failed は `bf952d3f` と同じ `test_jump_race_p0_gate.py` の既存事象。

## 7. merge 時の注意と未解決事項

1. **merge 前の未追跡ファイル衝突**：本番 tree には、今回追跡対象にした 4 本が未追跡のまま残っている。そのままでは merge が「untracked working tree files would be overwritten」で止まる。merge 前に本番側 4 本の sha256 を `dec0281d` の記録と照合し、一致すれば退避してから merge する。不一致なら、差分を先に取り込むかを判断する。
2. **本番が未追跡ファイルに依存している（範囲外の所見）**：`compute_bets.py` は live モードで `wide_residual_shadow.py` を import するが、このファイルは本番で未追跡（`bf952d3f` に無い）。バージョン管理外の本番依存で、9/27 と同じく失えば復旧できない。
3. **ジャーナル化していない JV-Link 利用**（→ §9 で訂正。`jvlink_results.py` は `jvlink_odds.fetch_records` 経由でジャーナルに載る）：`jvlink_results.py`・`jvlink_probe.py`・`jvlink_race_day_probe.py`・`jvlink_shadow_probe.py`（いずれも手動実行で、scheduler 経路には無い）。レース時間帯に手動で実行すると、重なり相手として見えない。
4. **契約の閾値**：欠損率・被覆不一致率 > 1% は 1 開催日 23〜36 レースでは 1 件で超える。Stage 0 では 1 件の欠損でも `CONTRACT_NOT_MET` になる（意図どおりの厳しさ）。
5. **枠連の期待 slot 数**：登録頭数からの仮定式で、実録 4 件（smoke を含む）と一致した。契約の被覆判定に含めたので、式が誤っていれば `CONTRACT_NOT_MET` として現れる。
6. **キュー直列化**：未実装。計画 §3 (6) どおり、判定が出た時点で実装する。
7. **Dry と登録**：Dry は T−2 と三連複の両方にかかる（`obs_register_tasks.ps1 -Dry`）。登録・`-Apply` は承認後。

## 8. Fable 指定の追加修正（merge 前、2026-09-29）

| commit | 内容 |
|---|---|
| `b43fbd8e` | 観測計画 v2.1.1。§12 に「診断用 JV-Link probe 3 本は開催時間中（Dry 日・Stage 0 開催日を含む）に手動実行しない。実行は非開催時間に限る」を追記した。`.sha256` を `f711b36d…` に更新し、v2.1 の hash は版表に残した |
| `05cf3831` | Stage 0 の欠損判定を改めた（下記） |

**欠損判定（`05cf3831`）**
- **閾値**：欠損数が `max(1% × 予定レース数, 1 race)`（2 開催日合計）を超えたら `CONTRACT_NOT_MET`。被覆不一致の閾値 1% は変えていない。
- **原因の帰属**：欠損 race ごとに原因を付ける。帰属できない欠損が 1 件でもあれば `CONTRACT_NOT_MET`。原因は次の 9 種。
  - `task_not_fired`
  - `fetch_rc`
  - `race_key_mismatch`
  - `no_records`
  - `record_not_stored`
  - `spec_capture_absent`
  - `record_without_raw_v1`
  - `stock_session_failed`
  - `stock_race_absent`
- **`CONTRACT_NOT_MET` のとき**：
  - 収集は止めない（`collection_continues=True`）。
  - 決定コード・理由・欠損 race・原因を report と標準出力に出す。
  - 判定を `data/obs_stage0_ledger.jsonl` に追記する（追記専用。Dry は `dry=True` で記録）。
- **guard との連動**：`analysis.obs_guard.assert_performance_allowed()` は次の 2 段で判定する。
  - ledger の最新の本番判定が `CONCURRENCY_OK` でなければ拒否する。つまり修正後の開催日から Stage 0 の 2 日を数え直し、通過するまで性能・ROI・帯選択へは進めない。
  - 500R は、その通過窓の初日以降の race だけで数える。
- **静的検査**：guard に `root=` / `ledger=` / 位置引数を渡して差し替えることも違反とする。
- **テスト**：`tests/test_obs_phase2.py` 105 本が pass。全 suite は 347 passed / 17 failed（jump gate の既知事象）。

## 9. 訂正：§7-3「ジャーナル化していない JV-Link 利用」（2026-09-29）

**誤り**：§7-3 で `jvlink_results.py` を「JV-Link を直接呼び、取得ジャーナルに載らない」利用に含めた。これは誤りである。コード変更は無い（記述の訂正のみ）。

**正しい経路**：`jvlink_results.py` は COM を直接呼ばない。
- `jvlink_odds` から `fetch_records` を import して呼ぶ（31 行・64 行・86 行、spec `0B30`）。
- `fetch_records` → `fetch_records_timed` → `_fetch_records_raw`（JVInit/JVRTOpen/JVRead）の順で呼ばれる。`fetch_records_timed` の `finally` で `jv_journal.write_event` が取得ごとに 1 件ジャーナルへ書く。
- `set_context` は呼ばないので、process 名は argv からの補完になる（`process="jvlink_results"`、`process_source="argv"`）。Stage 0 監査では識別済み process として扱われ、未識別にはならない。
- import に失敗したとき（単体テストなど）は、空を返す代替 `fetch_records` になり、COM も呼ばない。

**直接 COM でジャーナルに載らない JV-Link 利用（本番 tree の追跡 .py を再走査した結果）**：
- Fable の指示は「直接 COM は `jvlink_probe.py` だけ」だった。コードを確認すると、残り 2 本の probe も COM を直接呼んでいる。いずれもジャーナルに書かない。

| ファイル | 呼び方 | 種別 |
|---|---|---|
| `jvlink_probe.py` | `Dispatch` → `JVInit` → `JVRTOpen` | 速報系（観測 collector と同じ系統） |
| `analysis/mcond/exp05_forward_shadow/jvlink_race_day_probe.py` | `Dispatch` → `JVInit` → `JVOpen("RACE", …, 2)` → `JVRead` | 蓄積系 |
| `analysis/mcond/p0_dnf_history_parity_audit/jvlink_shadow_probe.py` | `Dispatch` → `JVInit` → `JVOpen(spec, …, 4)` → `JVRead` | 蓄積系 |

- 速報系 `JVRTOpen` を直接呼ぶのは `jvlink_probe.py` だけである。「直接 COM は `jvlink_probe.py` だけ」が速報系に限った意味なら、コードと一致する。
- 3 本とも観測計画 v2.1.1 §12 の probe 3 本と同じもので、開催時間中（Dry 日・Stage 0 開催日を含む）には手動で実行しない。実行は非開催時間に限る。§12 の本文はこのとおりなので、計画書は変更しない。
- ジャーナルに載る JV-Link 利用：`jvlink_odds.py`、`jvlink_trio_odds.py`、`jvlink_obs.py`、`jvlink_results.py`（`jvlink_odds` 経由）、`analysis/mcond/exp05_forward_shadow/jvlink_race_calendar.py`。
- `.claude/worktrees/` 配下の未追跡コピー（`jvlink_odds.py` / `jvlink_probe.py` の古い版）も直接 COM を含むが、本番の scheduler 経路ではない。
