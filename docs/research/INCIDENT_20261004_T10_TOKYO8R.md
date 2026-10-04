# 運用事故: 2026-10-04 東京 8R の本番 T-10 が起動直後に停止（a95 本線では「欠測」）

- 記録: 実行部 2026-10-04 ／ 分類の判断: Fable（2026-10-04 確定）
- 対象: `PyCaLiAI_T10R_2026100405040208`（東京 8R 芝 1800 1勝、発走 14:00、T-10 処理 13:50）
- 状態: 原因は未確定で調査中。コードは変更していない。

## 1. 分類（a95 本線 CW-11）

| 項目 | 扱い |
|---|---|
| a95 本線（CW-11、P2-A95-EQUAL-LIVE-2026） | **欠測**（T-10 が実行されなかった）。戦略による「見送り」には分類しない。有効 race・的中率の分母に入れない |
| 紙上台帳（CW-10、shadow_a95） | 影響なし。前夜 22:50 に朝オッズで凍結済み（`data/shadow_ledger/a95_equal_v1/decisions/20261004.json` の 7 件目、tickets 複勝・馬連 各 5,000 円） |
| `reports/cowork_output/20261004_bets.json` | このレースは `bets: []`。T-10 が書いたものではなく、Phase B の時点のまま。`skip_reason` も無い。集計でこの空配列を「見送り」と読まないこと |

- 500R の集計を実装するときは、このファイルの race を欠測として除く。T-10 の決定記録（forward_prices の decision / t10 録）が無い race は、見送りではなく欠測として扱う。
- 収集待ち台帳 CW-11 への注記は、配信再現性修正（A）の merge の後に追加する。branch が同じ台帳を変更しているため、先に master を触ると衝突する。

## 2. 事実

- タスク: 13:50 に起動し、`LastTaskResult = 0xC000013A`（STATUS_CONTROL_C_EXIT）。
  - action は `powershell.exe -NoProfile -ExecutionPolicy Bypass -File E:\PyCaLiAI\t10.ps1 -Once 2026100405040208 -Date 20261004 -LeadMin 10 -MaxAgeMin 20`。
- 起動していない根拠:
  - Windows PowerShell ログの 13:45〜13:58 に、この t10.ps1 の engine 開始イベント（400）が無い。PowerShell が engine を起こす前に process が終わっている。
  - `logs/t10_20261004.log` にこのレースの処理行が無い。登録行の 2 回だけで、他のレースは 6 回ある。
  - `reports/live_odds/2026100405040208.json` が無い。
  - 本番 journal にこのレースの t10 / close_late の取得が無い。forward_prices にも t10 / close_late の録が無い（他の 22R にはある）。
- 同じ時刻の他の起動:
  - Dry の三連複 collector（同じレースの T-10）が 13:50:01.555 に起動し、13:50:02.218 に正常に終わった（engine 停止イベント 403 あり）。取得は 0B35・0B31 とも rc 0。
  - 体重 collector の取得が 13:50:02.78。
  - 三連複と T-10 の同時起動は他の 22R でも起きていて、そちらの T-10 は正常だった。
- 同じ日の同じ終了コード: `PyCaLiAI_OBS_FINAL_20261004`（18:30:01 開始、18:30:16.7 より後に停止）。
- Application・System のイベントログには、両時刻とも記録が無い。Task Scheduler の操作ログは無効で、記録が無い。
- この端末の既定の console の設定は「Windows に任せる」（`HKCU\Console\%%Startup` の Delegation が全 0）。Windows Terminal に委譲されている可能性がある。

## 3. 原因の候補（未確定）

1. タスクが出した console ウィンドウを、人が閉じた。閉じると CTRL_CLOSE が届き、終了コードは 0xC000013A になる。
   - ユーザーへ確認する: 13:50 頃と 18:30 頃にウィンドウを閉じたか。
2. 既定の terminal への委譲（conhost → Windows Terminal）が、同じ秒の複数起動で失敗した。未検証。
   - 18:30 の FINAL は同時起動が無いので、これだけでは説明できない。
3. その他: JV-Link や Windows による終了。根拠となる記録は無い。

## 4. 次の手

- 観測タスクには F7 で対策する: 窓なしで起動し、開始・終了を追記専用で記録し、強制終了を検出する。平日の試験タスクで確かめてから登録する。
- 本番 T-10・T-20 への適用は別 commit・後続の作業で行う（Fable の指示）。それまで、この事故の型は本番 T-10 で再発しうる。
- 再発したら、このファイルに追記する（削除はしない）。
