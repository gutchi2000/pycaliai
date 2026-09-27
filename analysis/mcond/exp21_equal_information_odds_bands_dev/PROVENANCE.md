# EXP21 provenance

## Stage 0 commit `aec789bb` の混入（記録のみ。過去 commit は修正しない）

- `aec789bb`（2026-09-27 12:10、EXP21 Stage 0）は、EXP21 の 18 ファイルに加えて、EXP21 と無関係な既存作業 **28 ファイル**を含んでいる。
- 原因: 元の作業ツリーが中断した rebase 状態（`.git/rebase-merge/` に `autostash` だけが残存）で、autostash の再適用により既存作業が
  index にステージされていた。EXP21 の commit が pathspec を付けずに index 全体を commit したため巻き込んだ。
- 28 ファイルは正当な既存作業で、`origin/master` に push 済み。内容は autostash（`1dfd7a72`）と同一で、失われたものは無い。
  revert・reset・rebase・履歴書き換えはしない。

混入した 28 ファイル:

```
category_normalize.py, changes.ps1, data/_inbox/README.txt, data/history_only/jump/intake_ledger.jsonl,
data/history_only/jump/manifest.json, data/hosei/H_20260426.csv, data/kekka/wide_kekka.csv, data/masters_vote.json,
data/pl_payout_curve_v6.pkl, data/production_policy.json, data/serve_code_maps.json, data/serve_feature_baseline.json,
docs/SPEC/ALL_IN_ONE.md, docs/SPEC/VOL1_SYSTEM.md, docs/SPEC/VOL2_BETTING_OPS.md,
docs/SPEC/VOL3_VALIDATION_AND_OPEN_PROBLEMS.md, docs/SPEC/VOL4_CODE_REFERENCE.md, docs/cowork_prompt.md,
docs/forward_price_protocol.md, docs/hypothesis_registry.md, forward_price_integration.py,
models/pl_calibrators_v6_serve.pkl, place_weekly.py, predict_weekly.py, production_policy.py,
reports/cowork_input/20260921_bundle.json, scripts/build_note_article.py, t20_site_bets.py
```

## v0.3 と Stage 1 の作業環境

- 隔離 worktree `E:/PyCaLiAI_exp21_wt`、branch `exp21-v03-stage1`、起点 `origin/master` = `30704408`。
- 元の作業ツリー（`E:/PyCaLiAI`）の rebase / autostash / staged / unstaged 状態は変更していない。
- 勝者表補正の 5 ファイル（`winner_claim.py`、`out/winner_claim_uncertainty.json`、`WINNER_CLAIM_UNCERTAINTY.md`、
  `STAGE0_DATA_AUDIT.md`、`spec.json` の `stage0_results.winner_claim`）は、元の作業ツリーの未 commit 差分を確認したうえで worktree へ複写した。
- 大容量の生データ（git 管理外）は `EXP21_DATA_ROOT=E:/PyCaLiAI/data` で読み取り専用に参照した。中間生成物は worktree 内の
  `data/_research/mcond/exp21/`（git 管理外）へ書いた。
- worktree の git 操作は `-c safe.directory=E:/PyCaLiAI_exp21_wt` を毎回付けて行った（global 設定は変更していない）。
- Stage 0 の G0 監査と帯生成を worktree で再実行し、`g0_audit.json`・`label_free_band_coverage.json` が Stage 0 commit の出力と
  経過時間以外で完全一致することを確認した（ticket 配列に race ID・馬番と terminal 較正 null を追加保存するためだけの再実行）。
- v0.3 凍結 commit と Stage 1 結果 commit は、EXP21 ディレクトリのファイルだけを含むことを機械確認してから作る。
