# B2: 10/03・10/04 の Dry を修正後の監査で再判定（取得の再実行なし）

- 実行: 2026-10-04 21:19。master `52fca47c`（obs-dry-fix-b の B1 `396a6f2d` を merge）。監査は `stage0-audit-2`、構造化は `jvrec-2`。
- コマンド: `python -m analysis.obs_stage0_audit --dates <date> --dry --scope night --rejudge-of "<元の判定>"`。各日 1 回だけ実行した（runner に lock あり）。
- 範囲は night。蓄積系（final_stock）は当夜の契約から外した。蓄積系は 10/05 夜（配信の実測）と 10/06 20:00 に別に評価する。
- 元の不合格の記録はそのまま残っている。
  - `reports/obs_stage0_20261003_dry.json`: sha256 `d424aab7…` で前後とも同じ。
  - `reports/obs_stage0_20261004_dry.json`: sha256 `ba03daa1…` で前後とも同じ。
  - ledger の元の 3 行も変わっていない（追記だけ。prefix 一致は True）。

## ledger

| 時点 | 行数 | sha256 |
|---|---|---|
| 再判定の前 | 3 | `e0b9e83fe4d647c6b5f4c909e52315848aa69cf428289aca00bcef1e70ceeedf` |
| 再判定の後 | 5 | `4308791ca0811761a3ab91f423053e9bb0842ca22a9a1be4f17d70b9a3a635a7` |

- 追記した 2 行は `ledger_appended_lines.jsonl` にある。どちらも `dry: true`・`scope: night`・`rejudge_of` 付き。
- 性能 guard（`analysis/obs_guard.latest_stage0`）は dry でない行しか見ない。この再判定で解禁されるものは無い。

## 結果

| | 10/03 | 10/04 |
|---|---|---|
| decision / exit | **CONCURRENCY_OK / 0** | **CONTRACT_NOT_MET / 4**（不合格のまま保存） |
| 理由 | なし | final_rt_candidate 5 spec で各 10/23R が欠損（FINAL の強制終了。収集側の運用事故） |
| 欠損の帰属 | 0 件 | task_not_fired 58 件 = final_rt 10R × 5 spec ＋ 東京 8R の本番 t10・close_late 4 spec × 2（1R なので閾値以内） |
| 読込み元 | 観測 stage は dry、t10・close_late・t20・exp05fs_t35 は本番 | 同左 |
| 被覆の不一致（範囲内） | 0 | 0 |
| raw/parser | 全て 0 件（639 録） | 全て 0 件（586 録） |
| 旧版（jvrec-1）の録 | 639 録は比べていない（今の式で構造化し直すと 639 録とも差が出る。版の違いによる差なので不一致には数えない） | 586 録、同じ扱い |
| 並走（主判定: 取得区間の交差） | 単独 542/542、重なり **97/99（98.0%）** | 単独 478/478、重なり **108/108（100%）** |
| 並走（記述: ±30 秒） | 単独 120/120、重なり 519/521 | 単独 180/180、重なり 406/406 |
| 重なりの失敗 | 2 件。10/03 東京 5R の本番 exp05fs_t35 0B34 と t20 0B31（`rc_open=-413`、録なし）。JV-Link 側の取得失敗 | なし |
| 00:00 型破損 | 0 | 0 |
| タスク記録（F7） | 記録なし（F7 の前。契約は 10/10 から） | 同左 |

- 補足: 10/03 の重なり 99 件の相手は、本番の t10・t20・exp05fs_t35 どうし、体重 collector、三連複 T−10 と本番 T−10 の同時起動。
  T−2 と final_rt に重なりは無かった。

## ファイル

- `run_meta.txt`: runner の記録（hash・終了コード）
- `rejudge_2026100{3,4}_stdout.txt`: 監査の標準出力
- `obs_stage0_2026100{3,4}_dry_night_rejudge.json`: report の写し
- `ledger_before.jsonl` / `ledger_after.jsonl` / `ledger_appended_lines.jsonl`
