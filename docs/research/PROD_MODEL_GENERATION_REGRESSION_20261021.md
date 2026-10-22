# 本番モデルの世代退行（C1 → C0）検出報告 — 2026-10-21

> 発見経路: 新実験候補の墓場照合 workflow（2026-10-13〜21）の審査員が「本番 pkl が C0 世代ではないか」と指摘。本報告はその指摘を**実ファイル・git・T-10 stamp で再検証**した結果のみを記す。復元は未実施（本番変更＝ユーザー判断）。

## 1. 結論（Fact）

- `models/unified_rank_v6.pkl` の現物は **C0（P0-5 切替前）** sha256 `0a040c4e…`、mtime 2026-09-20 10:42。
- P0-5 C1 切替（2026-09-11、VOL3 §5.1x）で配置されたはずの C1 モデル `db2ffe65…` は、**どのブランチにもコミットされていない**（`git log -S'db2ffe65' -- models/unified_rank_v6.pkl` = 0 件）。存在するのは `stash@{0}`（9/20 10:18 autostash）・`stash@{1}`（9/19 23:13 autostash）・`candidates/c1_migration_v2_final/models/`・ローカル LFS オブジェクトのみ。
- 2026-10-03 起算の a95 本線 cohort は **全日 C0 で走っている**。`reports/cowork_output/{20261004,20261011,20261012}_bets.json` の `stamp.artifact_sha256.rank_model` が全て `0a040c4e…`。
- 一方、研究用 2024-25 スコア `data/_policy/v6_scores.parquet`（9/17 19:59 生成）と `data/_research/mcond/oof_scores_c1master.parquet`（9/18）は **C1 期間（9/11〜9/20）に生成**されており C1 由来。**a95 の設計表（C1 スコア）と本番（C0 スコア）の確率源が別世代**。

## 2. 切替計画 10 ファイルの現状突合

`candidates/c1_migration_v2_final/SWITCH_PLAN_FINAL.json` の期待 sha256 と現物（2026-10-21 計測）:

| ファイル | C1 期待 | 現物 | 判定 |
|---|---|---|---|
| `data/master_v2_20130105-20251228.csv` | b8032d1b | b8032d1b（9/11） | C1 |
| `data/_horse_history.parquet` | 713ee046 | 27c9e3ec（10/19 再生成） | 週次再生成物。世代判定不能 |
| `data/serve_code_maps.json` | df309b08 | 4e2fad03（10/19 再生成） | 同上 |
| `data/serve_feature_baseline.json` | de6c11a2 | 7e3bbc32（= HEAD、9/27 復旧版） | C1 計画版でも C0 backup 版（d285f955）でもない第 3 版 |
| **`models/unified_rank_v6.pkl`** | **db2ffe65** | **0a040c4e（9/20 10:42）** | **C0 に退行** |
| **`models/pl_calibrators_v6.pkl`** | **a08da051** | **c3defc24（9/20 10:43）** | **C0 に退行**（週次 bundle の `CAL_PKL` が読む） |
| `models/pl_calibrators_v6_serve.pkl` | 90318635 | 90318635（9/27 復旧） | C1 |
| `data/pl_payout_curve_v6.pkl` | 52b0edf6 | 52b0edf6（9/27 復旧） | C1 |

つまり本番は **C1 マスター／C1 配信較正器 × C0 モデル／C0 基本較正器** の混成世代。

## 3. 機構（なぜ起きたか）

1. 9/11 の切替コミット `b2e2bdd4b6` は `build_dataset.py` と `serve_history_feats.py` の **2 ファイルのみ**。pkl 群（LFS）・json は作業ツリーに配置されただけで `git add` されていない。
2. 9/19 23:13・9/20 10:18 の週次 `git pull --rebase --autostash` が未コミット変更を stash し、再適用に失敗して **stash を残したまま作業ツリーが HEAD（C0 ポインタ）に戻った**。LFS smudge が 9/20 10:42〜43 に C0 実体を書き戻した（mtime と一致）。
3. 9/27 の autostash 事故復旧（`RECOVERY_20260927_REPORT.md`）は 28 ファイルを戻したが、対象は 9/27 の stash（`1dfd7a72`）であり、**9/19・9/20 の stash@{0,1} に入っていた `unified_rank_v6.pkl` と `pl_calibrators_v6.pkl` は対象外**だった。memory にも「stash@{0,1} は保持中」とだけ記録されている。
4. 検知されなかった理由: `policy_stamp()` は artifact sha を bets.json に刻むが**期待値と照合しない**。9/22 の監査（`DNF_INCLUSIVE_BASELINE_RECONCILED`、`KAKO5_CONTRACT_SHADOW_REPLAY`）は `0a040c4e` を「モデル hash 不変 PASS」と記録し、10/04 の `REPLAY_SERVE_2026` は「モデル v6（4/30 から不変）」と前提を書いた。いずれも C0 を基準に計測している。

## 4. 影響

- **a95 本線 cohort（CW-11）・紙上台帳（CW-10）**: 10/03 以降の全決定が C0 スコア。設計表は C1 スコア。500R 判定の帰属が崩れる。serve-repro-sp1 反映時の「案 B 再起算」と同時に処理するのが合理的（どちらにせよ再起算）。
- **週次 bundle（9/26 以降）**: C0 モデル × C0 基本較正器で生成。サイト表示の印・p_win も C0。
- **研究**: 9/20 以降に「本番モデル」を読んだ分析（replay_serve_2026、serve-repro 検証、EXP21 以降の 2026 serve 比較）は C0 基準。9/11〜9/20 に生成された研究表（v6_scores / oof_c1master）は C1 基準。両者を混ぜた比較は世代差（P0-5 リーク修正分、学習行の 3.4%）を含む。
- 予測精度への実害は小さい見込み（VOL3: C1 採用は精度改善の実証ではなく正しさのため）。問題は **来歴の一貫性**。

## 5. 復元手順（未実施・ユーザー判断待ち）

復元元は `candidates/c1_migration_v2_final/models/`（sha 検証済み）。stash からでも同一内容。

```powershell
Copy-Item candidates\c1_migration_v2_final\models\unified_rank_v6.pkl models\unified_rank_v6.pkl
Copy-Item candidates\c1_migration_v2_final\models\pl_calibrators_v6.pkl models\pl_calibrators_v6.pkl
Get-FileHash models\unified_rank_v6.pkl, models\pl_calibrators_v6.pkl -Algorithm SHA256
# 期待: db2ffe65ee27fb90… / a08da05124c489c0…
git add models\unified_rank_v6.pkl models\pl_calibrators_v6.pkl
git commit -m "prod: C1 モデル/基本較正器を再配置 (9/20 autostash で C0 に退行していた)"
```

復元後に必要なこと:
1. `stash@{0}` `stash@{1}` の残り差分（docs/SPEC・level_norms・t10_runner 等 29 ファイル）が HEAD に既に入っているか確認してから stash を drop（確認なしで drop しない）。
2. a95 の起算日を再設定（serve-repro-sp1 の案 B と同梱）。`docs/hypothesis_registry.md` と `COLLECTION_WAIT_REGISTRY.md` に世代注記。
3. `production_policy.py` に **artifact sha の期待値照合**（policy json に `artifact_sha256_expected` を持たせ、不一致なら fail-closed）を追加する。stamp だけでは退行を検知できなかった。
4. `REPLAY_SERVE_2026_20261004.md` の「4/30 から不変」を訂正。

## 6. 判断を求める点

- C1 を復元するか（推奨: する。C1 は正しい学習データの帰結であり、復元しないなら VOL3 の「P0-5 本番反映済み ✅」が偽）。
- 復元のタイミング（推奨: serve-repro-sp1 マージと同じ日。a95 再起算を 1 回で済ませる）。

## 7. 解消（2026-10-22）

- ユーザーが `scripts/restore_c1_model.sh` を実行し、C1 を再配置した（commit `bcb2e49a`）。続けてガード（`prod/artifact-pin-guard`、`b1e3c465`）を merge した（`454f38a2`）。テストは 9 件すべて通過。
- 確認: `models/unified_rank_v6.pkl` = `db2ffe65…`、`models/pl_calibrators_v6.pkl` = `a08da051…`。`policy_stamp()` が pin 照合 OK、`export_weekly_marks.py` が「artifact pins OK」を出して bundle を生成。
- 再発防止: `data/artifact_pins.json` と期待値が違えば T-10 と週次 bundle が fail-closed で止まる。モデルを意図して差し替えるときは、pins を同じ commit で更新する。
- a95 の再起算（10/24 予定）は `docs/hypothesis_registry.md` と `COLLECTION_WAIT_REGISTRY.md` に記録した。

### C0 と C1 の差（10 月 7 開催日・185R、同じ入力で両モデルの bundle を作り直して比較、記述のみ）

| 指標 | C0 | C1 |
|---|---|---|
| ◎ 3 着内率 | 61.6% | 61.1% |
| ◎ 勝率 | 26.5% | 25.4% |
| 勝ち馬の logloss | 2.267 | 2.222 |

- ◎ が入れ替わるのは 25%（185R 中 46R）。入れ替わった 46R で 3 着内に来たのは C0 が 24、C1 が 23。
- 差の 95% 区間は ◎3 着内 [−5.9, +4.3]pt、logloss [−0.151, +0.028] で、どちらも 0 をまたぐ。精度の差は検出できない（P0-5 は正しさのための修正で、精度改善を主張していない点と整合）。
