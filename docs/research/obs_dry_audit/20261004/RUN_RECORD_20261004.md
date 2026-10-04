# 10/04 Dry 監査 実行記録（凍結コード・取得の再実行なし）

- 監査コード: master `78a389fc`（観測関連は凍結中で未変更）。既存の記録を読み取っただけで、取得はやり直していない。
- 実行: `python -m analysis.obs_dry_check --date 20261004` と `python -m analysis.obs_stage0_audit --dates 20261004 --dry`
- 10/03 の証跡は別フォルダ（`../20261003/`）にある。

## 1. 実行の不備: 監査が 2 回走った

- 1 回だけ実行する予定だった。ところが時間制限で「停止」と通知された 1 本目の runner の process が実は生きており、後から起動した 2 本目と両方が FINAL の後に実行された。
  1 本目は 14:18 に起動し 16:18 に「停止」と通知された。実際には生きていて、18:32:55 まで出力を書いた。2 本目は 18:06 に起動した。
- そのため `obs_stage0_audit --dry` は 18:32:22 と 18:32:25 の 2 回走った。`obs_dry_check` も 2 回走ったが、こちらは読むだけで何も書かない。
- ledger `data/obs_stage0_ledger.jsonl` の推移:

| 時点 | 行数 | sha256 |
|---|---|---|
| 10/03 監査後（10/04 11:20） | 1 | `e46528179d1e9a57093bc26255de0db00f8b7ee9a039a234db1a715d1d3b2e99` |
| 1 回目の 10/04 監査後 | 2 | `6593c6416fddf4d47a4e042c5b4d1ab3e48589915133ebc8659ca101276612e9` |
| 2 回目の 10/04 監査後（現在） | 3 | `e0b9e83fe4d647c6b5f4c909e52315848aa69cf428289aca00bcef1e70ceeedf` |

- 1 行目（10/03）は `../20261003/ledger_line_written_by_20261003_audit.jsonl` とバイト単位で一致する。既存行は変わっていない。
- 2 行目と 3 行目は 10/04 の行で、違いは `generated_at`（18:32:22 / 18:32:25）だけ。decision、exit、理由、欠損の帰属はすべて同じ。
  - 2 行の写し: `ledger_lines_written_by_20261004_audit.jsonl`（sha256 `b47ba1ec…`）
- report `reports/obs_stage0_20261004_dry.json` は 2 回書かれ、残っているのは 2 回目の版（sha256 `ba03daa1…`）。写しは同じフォルダにある。10/03 の report は前後で不変（`d424aab7…`）。
- 補足:
  - `run_meta.txt` は 2 本の runner の書込みが混ざっている。冒頭の「before」は 2 本目の視点（1 回目の監査の後）。
  - `ledger_before.jsonl` は 2 行、`ledger_appended_lines.jsonl` は 2 本目が追記した 1 行（3 行目）。
  - 1 本目の視点の値は上の表で補う。
- 削除や巻き戻しはしていない。重複した ledger 行は、二重実行の証跡としてそのまま残す。
- 再発防止: 再起動の前に process が残っていないか確かめる。runner には lock を付ける。

## 2. FINAL が途中で強制終了した（収集側の事象）

- FINAL（`obs_schedule.ps1 -Final -Date 20261004 -Dry`）は 18:30:01 に開始した。
  - FINAL_RT を 13/23 レース（65 取得、すべて rc 0）取り終えた 18:30:16.7 より後に、process が止まった。
  - transcript に終了の行が無い。PowerShell の engine 停止イベント（403）も無い。
  - タスクの結果は `0xC000013A`（STATUS_CONTROL_C_EXIT）。console に Ctrl+C か close が届いたときの終了で、ウィンドウを閉じた場合もこれになる。
- そのため FINAL_RT の残り 10 レースと、蓄積系 STOCK（O1〜O5）のセッションは実行されていない。
- 10/03 の FINAL は 18:30:00〜18:30:35 に正常に終わっている。
- 同じ終了コードが当日もう 1 件ある: 本番 T-10 の `PyCaLiAI_T10R_2026100405040208`（東京 8R、13:50）。
  - PowerShell の engine 開始イベントすら無い。live_odds も T-10・close_late の取得も無いので、このレースは本番 T-10 を処理していない。
  - 同じ 13:50:01 に Dry の TRIO（同じレース）が起動し、1 秒で正常に終わっている。ただし TRIO と T-10 の同時起動は他の 22 レースでも起きていて、そちらは T-10 も正常だった。
- 原因は未確定。Task Scheduler の操作ログは無効で、記録が無い。ユーザーが該当時刻にウィンドウを閉じたかを確認する必要がある。

## 3. 正式判定（凍結コード、`--dry`）

- decision **CONTRACT_NOT_MET**、exit **4**。並走の判定は `INSUFFICIENT_OVERLAP_OBSERVED`。
- 予定 23R。記録 59、journal 226 件。
- 理由:
  1. 必須 stage・spec の欠損が max(1%, 1R) を超えた。
     - t10・close_late: 各 23/23 欠損。
     - final_rt_candidate: 各 10/23 欠損。
     - final_stock_candidate: 各 23/23 欠損。
  2. 被覆の不一致が 1% を超えた: t2_candidate・trio_t10 の全ブロックと、final_rt の枠連。
- 欠損の帰属（全件が task_not_fired で 349 件。fetch rc・race_key 不一致・その他はいずれも 0）:

| stage | 欠損 | 実際の理由 |
|---|---|---|
| t10 / close_late（4 spec × 2） | 各 23 | dry の監査が本番の保存先を見ていない（F1）。本番の保存先には 22/23 あり、東京 8R だけ無い（§2） |
| final_rt_candidate（5 spec） | 各 10 | FINAL の強制終了（§2） |
| final_stock_candidate（5 spec） | 各 23 | STOCK セッションが無い（§2）。月曜配信の問題（F4）は今回は検証できていない |

- by_kind（`--dry` の journal のみ、成功率は凍結コードの定義）:

| kind | events | 単独 | 重なり |
|---|---|---|---|
| t2_candidate | 115 | 115（成功率 0.0、F2・F3） | 0 |
| trio_t10 | 46 | 46（成功率 0.0） | 0 |
| final_rt_candidate | 65 | 65（成功率 1.0） | 0 |

- 生データの検査は 226 件全て合格: 再構造化一致・parser 一致・race_key・録長・raw sha256。00:00 型破損 0。

## 4. 補助集計（`supplement_20261004.json`、正式判定ではない）

- 本番と dry の保存先を合わせた監査: CONTRACT_NOT_MET。
  - 欠損は 173 件（全て task_not_fired）: t10・close_late が各 1（東京 8R）、final_rt が各 10、final_stock が各 23。
  - 並走の判定は凍結コードの成功率の定義だと `QUEUE_SERIALIZATION_REQUIRED` になる（被覆の式が誤っているため、F2・F3）。
- 並走の成功を被覆の式から切り離した定義（F3 案）で数え直した値:
  - 単独 170/180（94.4%）、重なり 392/406（96.6%）。
  - 重なりの失敗 14 件は、本番 t20・t10 の取得と体重 collector・exp05fs との重なり。内容は登録頭数との発売数の不一致で、F2 の取消の扱いで解消するかは B3 で検証する。
- 重複・重なりの件数:
  - T−2 と、別レースの本番 T−10 の ±30 秒以内の重なり: **0 件**。T−2 と本番 t10 の最小間隔は 419.7 秒。
  - TRIO と本番 T−10 の重なり: 同じレースで 176 組、別のレースで 0 組。TRIO は設計どおり同じ T−10 時刻に起動している。
  - 価格取得の rc 異常: 0 件。
  - 体重 collector の `rc_open=-1` 153 件と、jvlink_changes の `-1`/`-114` は Dry 以外の既存 process の値。価格取得ではないので上の rc 異常 0 件には含めていない。内容は調べていない。
- manifest の行の重複: 同じ行 0、(日付, kind, spec, race) の重複 0。

## 5. 三連複 manifest に Dry が追記した行（`trio_manifest/`）

- 保存先は `reports/trio_portfolio_shadow_v2/manifest.jsonl`（git 追跡下）。
  - 最後のコミットは `5557fa69`（2026-08-18）、HEAD の blob は `c69334b6`。
  - ファイルは text mode で書かれ、改行は CRLF。
- 追記前: 582 行、233,859 バイト、sha256 `dc49e902bd110f2367e9211f2abbe515afed23447f7764aa0e135dce1cb372d1`。
- 既存行は変わっていない（HEAD の blob が現在のファイルの完全な先頭部分）。

| 日付 | 行 | 内容 | 取得時刻 | この日付の行を書き終えた時点の manifest sha256 |
|---|---|---|---|---|
| 10/03 | 583–651（69 行） | 23R × raw 0B35 / raw 0B31 / snapshot | 09:40:03〜16:20:00 | `f9d68ef1ca5c590870e40833d749ab172702d45ff5858398b433441169d6c7e1` |
| 10/04 | 652–720（69 行） | 23R × 同上 | 09:40:01〜16:20:01 | `4779e980de63ad49174ff217d1a19a07406da929670f3c28c128821f3bcd8ed4`（現在の値） |

- 参照先は全て `_dry` の下にある。raw 92 件のファイル sha256 は全て記録と一致した（照合は CRLF を LF に戻してから）。
- snapshot の `ok=false` は 46/46。期待組数の式による（F2）。
- **訂正**: 以前に報告した「10/03 追記後の sha256 `55f20ad7…`」は誤り。正しくは `f9d68ef1…`。
  - 原因は証跡スクリプトが改行を 1 バイトと数えていたこと（CRLF なので 2 バイト）。
  - 修正後は、最後の日付の値と現在のファイル全体の hash が一致することを検査している。
  - ファイル全体の hash（14:05 の `fc62c736…` を含む）は、この誤りの影響を受けていない。
