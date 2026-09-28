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
