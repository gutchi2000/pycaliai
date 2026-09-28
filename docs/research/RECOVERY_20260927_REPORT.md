# 9/27 autostash 障害 復旧報告（2026-09-29、観測計画 v2.1 §9）

## 結論

本番ファイルは **2026-09-27 12:10 の commit `aec789bb`（EXP21 Stage 0 commit に autostash 再適用分が同乗）で既に HEAD に入っていた**。本日 9/29 に blob hash で照合し、HEAD = 退避 ref = working tree を全件確認した。残っていた作業は rebase 残骸の除去と CLAUDE.md の 3-way 反映で、本日完了。9/27 午後の T-10 は復旧後の状態で稼働していた（`logs/t10_20260927.log` 16:31 終了、`policy成果物が存在しない` の再発なし）。

## 照合表（git blob hash 先頭 10 桁）

base = 事故直前 HEAD `1ed4c7a1`、REC = `refs/recovery/autostash-20260927`（= `1dfd7a72`）、HEAD = 現 master、WT = working tree の `git hash-object`。

| ファイル | base | REC | HEAD | WT | 判定 |
|---|---|---|---|---|---|
| forward_price_integration.py | 94a7ce7360 | 1242cf50ab | 1242cf50ab | 1242cf50ab | 復旧済 |
| production_policy.py | 4baf87f9ab | f367e26c06 | f367e26c06 | f367e26c06 | 復旧済 |
| changes.ps1 | 6505809574 | 422f3724b6 | 422f3724b6 | 422f3724b6 | 復旧済（BOM 付き版） |
| data/production_policy.json | f181015664 | 04ab9f9278 | 04ab9f9278 | 04ab9f9278 | 復旧済（real_money.enabled=false） |
| data/serve_feature_baseline.json | e6d7742582 | afd9cf61f9 | afd9cf61f9 | afd9cf61f9 | 復旧済 |
| data/serve_code_maps.json | 7e964d6939 | b32d068896 | b32d068896 | b32d068896 | 復旧済 |
| data/pl_payout_curve_v6.pkl | 58e4592e0e | d1ec056489 | d1ec056489 | d1ec056489 | 復旧済（C1 版） |
| data/masters_vote.json | 2ef60a65d5 | a796e90598 | a796e90598 | a796e90598 | 復旧済（enabled=true のままだが、`t10.ps1` が `-WithVote` 無しでは投票タスクを登録せず「大会終了につき投票停止」で止まるため不活性） |
| models/pl_calibrators_v6_serve.pkl | 652ddf1451 | d11d196a62 | d11d196a62 | d11d196a62 | 復旧済。LFS oid `90318635…`、18,609 B = C1 migration v2 版 |
| docs/SPEC/VOL1〜VOL4, ALL_IN_ONE.md | — | 一致 | 一致 | 一致 | 復旧済 |

`git diff HEAD refs/recovery/autostash-20260927` に残る差分は、AGENTS.md / CLAUDE.md（HEAD 側が新しい「収集待ち台帳」節を持ち、REC 側が「2026-09-10 更新」と T-20 節を持つ）、`analysis/mcond/exp19_*`（別セッション編集中、触らない）、`data/level_norms.json`（同上）、`site/data/*` 45 件（HEAD 側が新しい再生成物、復元しない）のみで、本番経路のファイルは含まれない。

## 本日 9/29 の作業

| 手順 | 結果 |
|---|---|
| `.git/rebase-merge/` の除去（中身は `autostash` = 1dfd7a72 のみ、head-name / onto 無し = 進行中の rebase ではない） | 除去。`refs/recovery/autostash-20260927` は保持（`git rev-parse` で 1dfd7a72 を確認） |
| CLAUDE.md 3-way 反映 | REC 側の「新発見（2026-09-10 更新）」段落と「サイト公開用 T-20 買い目プレビュー」節を HEAD 版（収集待ち台帳あり）へ追加。AGENTS.md は HEAD 版を維持 |
| `production_policy.policy_stamp()` の読み込み | 成功。policy_id `topdown-serve34-p667-202…`、real_money_enabled=False（stopped 2026-08-27） |
| `t10.ps1 20260927 -Dry` の再実行 | **実行していない**。`forward_prices` は append-only で、過去日付の手動実行は時刻窓外の録を本番 store に残す（timing canary で事後隔離は可能だが、汚染そのものを避けた）。代替証拠は上記 policy_stamp と 9/27 午後の実稼働ログ |
| `git stash list` | `stash@{0}`, `stash@{1}`（9/19・9/20 の autostash）は削除せず保持。用済みなら利用者判断で drop |

## 未処理（本報告の範囲外、要判断）

- `master` は `origin/master` と **28 ahead / 1 behind** で分岐している。次の `weekly_nicegui.ps1` の `git pull --rebase --autostash` は rebase-merge 除去後は動くが、分岐の解消（pull --rebase）は実行していない。**本番ファイルを含む未コミット変更が無い状態で行うこと**（今回の事故の再発条件そのもの）。
- `data/level_norms.json` と `analysis/mcond/exp19_*` の未コミット変更は別セッションのもの。

## 教訓（既存メモリと同じ）

本番に効くファイル（コード・policy json・baseline・pkl）は変更したら即コミット。weekly ps1 は毎回 autostash するため、未コミット運用は失敗 1 回で全損する。
