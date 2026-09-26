# Opus宛て — EXP19 v0.2-frozen Stage 0実装依頼

EXP19「当日馬体重状態 × JRA公式馬場物理値」のStage 0だけを実装してください。Fableはv0.2-frozenを最終承認済みです。

必読:

1. `analysis/mcond/exp19_bodyweight_track_condition_dev/SPEC.md`
2. `analysis/mcond/exp19_bodyweight_track_condition_dev/spec.json`
3. `analysis/mcond/exp19_bodyweight_track_condition_dev/PRIOR_ART_AND_DATA_AUDIT.md`
4. `analysis/mcond/exp15_race_as_set_dev/ERRATUM_CURRENT_BODYWEIGHT_20260926.md`
5. `analysis/mcond/exp15_race_as_set_dev/out/feature_contract.json`末尾ERRATUM
6. `analysis/bodyweight_forward/README.md`

Stage 0で実施すること:

- 歴史torch専用whitelist loaderを作り、禁止列の具体名/prefix assertとloader sha256を実装。
- `R0-clean-nobw` 110列のrolling OOF基盤を構築する。ただし結果評価はまだ開封せず、再現性・未来行違反・manifestを確認。
- forward WHと歴史/当日保存値のparity、status、最初の完全snapshot時刻を監査。4開催日/400行に未達なら「収集中」と報告し、値を推測しない。
- 馬場値は対象日前日までのexpanding as-of標準化。race内定数softmax不変をテスト。
- W/WP特徴、DNF主母集団、full-starter感度母集団、A1/A2/B1/B2の構造を実装し、合成invariantを作る。
- EXP18 v0.5方式でlabel-free powerと実務floorを算出。数値を結果開封前にspecへcommitする。
- 実行時間・最大RSSを測る。

停止条件:

- 2019〜2023の実着順を使ったA1/A2性能、2024/2025、ROI、候補生成、賭金、production変更は行わない。
- forward parity 4日/400行が未達でも、取得済み件数と残数を正直に報告し、floorを緩めない。
- `斤量体重比`をN1へ戻さない。`bw_change_pct`をW/WPへ戻さない。
- TM/DM等の独自指数を追加しない。
- 仕様変更が必要なら結果を開けず停止し、版番号を上げる提案だけを出す。

完了時はStage 0成果物、全Gateの進行可否、未達条件、テスト数、commit hashを報告してください。Stage 1は開始しないでください。
