# EXP05-F タスクスケジューラ設計 (spec §10、登録は未実施)

`t20_site.ps1`/`t10.ps1`と全く同じ「レース毎タスク」方式を採用する
(`t35_shadow.ps1`実装済み)。理由: 既に本番で実績のある設計をそのまま踏襲することで
新規に検証すべき箇所を最小化する。

## 起動方式

- 土日9:00に常設タスク("PyCaLiAI_EXP05FS_T35", 未登録)が`.\t35_shadow.ps1 -Schedule`を起動
- `-Schedule`は`reports\cowork_input\{date}_bundle.json`の生成を15:00まで待ってから
  (=Phase Aの週次出走表確定を待つのと同じロジック、t20_site.ps1をそのまま踏襲)
  `feature_snapshot.py --date`を1回実行し、その日の全レースについて
  「発走-35分」にWindowsタスクを1つずつ登録する(`New-ScheduledTaskTrigger -Once`,
  `-WakeToRun`, 2時間の実行有効期限)
- 各レースタスクは発走時刻を過ぎていたら何もせず終了 (t20_site_bets.pyと同じガード)

## 祝日開催・月曜開催・代替開催への対応

`load_post_times`は`data/weekly/{date}.csv`から発走時刻を読むだけで曜日を仮定しない
(t10_runner.pyの既存実装をそのまま利用)。よって**祝日(月曜等)開催でも`-Schedule`を
手動実行すれば動く**(t10.ps1の既存運用と同じ、CLAUDE.mdに「祝日開催のみ手動: .\t10.ps1」と
明記されている)。常設タスクを土日9:00固定にする場合、祝日開催週は現状のt10/t20同様に
**手動起動が必要**になる(自動では拾えない)。これはt10/t20の既存運用と同じ制約であり
EXP05-F固有の問題ではない。

当日にJRA開催が無い日は`data/weekly/{date}.csv`が存在しないため`-Schedule`は
`[ERROR] 15:00までにbundle未生成`で終了する(t20_site.ps1と同じ)。

## 多重起動・スリープ

- `-MultipleInstances IgnoreNew`: 同一レースタスクの多重起動を防止 (t20_site.ps1と同じ設定)
- `-WakeToRun` + `-StartWhenAvailable`: PCスリープ時も起床して実行 (同上)
- タスクは実行後6時間で自動削除 (`-DeleteExpiredTaskAfter`)

## 実行環境の前提 (要文書化、spec §10)

| 項目 | 値 |
|---|---|
| 実行ユーザー | タスク登録時のユーザー(通常ログインユーザー)。t10/t20と同じ前提 |
| ログオン状態 | ログオフ中でも`-WakeToRun`で起動可能 (要: タスクのRun設定で「ログオンしていなくても実行する」) |
| 作業ディレクトリ | `E:\PyCaLiAI` (`t35_shadow.ps1`冒頭で`Set-Location`) |
| PowerShell実行ポリシー | `-ExecutionPolicy Bypass`を都度指定 (t20_site.ps1と同じ、システム全体のポリシーは変更しない) |
| 開始時刻 | 発走-35分 (許容ウィンドウ31-38分に収まることを狙う、実際の起床遅延は§8で検証) |
| 最大実行時間 | 30分 (`-ExecutionTimeLimit`) |
| 多重起動時の動作 | 無視 (`IgnoreNew`) |
| 失敗時の再試行 | 無し (次のレースタスクは独立、当該レースはそのまま`ok=false`で記録) |
| PCスリープ時の扱い | `-WakeToRun`で起床 |
| 標準出力・エラーログ | `logs\t35_shadow_{date}.log` (Start-Transcript)、EXP05-F固有の失敗は
  `logs\exp05fs_errors.log`にも追記 (`market_snapshot.py`) |
| タスク削除方法 | `Get-ScheduledTask -TaskName 'PyCaLiAI_EXP05FS_T35R_*' | Unregister-ScheduledTask` |

## 登録コマンド・解除コマンド (登録前Gate通過後にユーザー提示、今回は未実行)

登録:
```powershell
.\t35_shadow.ps1 -Schedule
```
常設の自動起動タスク("PyCaLiAI_EXP05FS_T35"、土日9:00起動)を作る場合は、
既存の`PyCaLiAI_T20_Site`タスクを`schtasks /query`で確認した上で同じ形式のトリガーを
別名で作成する(このセッションでは未実施、ユーザー確認後に行う)。

解除:
```powershell
Get-ScheduledTask -TaskName 'PyCaLiAI_EXP05FS_T35R_*' | Unregister-ScheduledTask -Confirm:$false
Get-ScheduledTask -TaskName 'PyCaLiAI_EXP05FS_T35' | Unregister-ScheduledTask -Confirm:$false
```

## 本番への影響範囲

`t10_runner.py`/`t20_site_bets.py`/`masters_vote.py`/`compute_bets.py`/
`weekly_nicegui.ps1`のいずれも未変更。共有変更は`jvlink_odds.py`/`forward_prices.py`への
stage追加のみ(`MIXED_COMMIT_NOTE.md`参照)。EXP05-Fのタスクが失敗・多重起動・スリープ未起床
しても、本番T-10/T-20/大会投票ラインのタスクはそれぞれ独立したタスク名・独立した
プロセスなので影響しない。
