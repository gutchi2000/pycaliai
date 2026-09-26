# REVIEW COMPLETE

Fableは2026-09-26に、WP #5の修正後は再レビュー不要でv0.2を凍結可能と判定した。本ファイルは履歴として保持する。Stage 0の依頼先は`OPUS_STAGE0_REQUEST.md`。

# Fable宛て — EXP19 v0.1仕様レビュー依頼

EXP19 v0.2の差分再レビューをお願いします。初回レビューの必須6件・推奨5件を反映しました。実装・学習・結果評価はまだ行っていません。

読むファイル:

1. `analysis/mcond/exp19_bodyweight_track_condition_dev/SPEC.md`
2. `analysis/mcond/exp19_bodyweight_track_condition_dev/spec.json`
3. `analysis/mcond/exp19_bodyweight_track_condition_dev/PRIOR_ART_AND_DATA_AUDIT.md`
4. `analysis/mcond/exp19_bodyweight_track_condition_dev/README.md`
5. `analysis/mcond/exp15_race_as_set_dev/ERRATUM_CURRENT_BODYWEIGHT_20260926.md`
6. `analysis/mcond/exp15_race_as_set_dev/out/feature_contract.json` の末尾ERRATUM

特に次を厳しく確認してください。

1. `R0-clean-nobw`（110列）を主nullとする修正で、Wとの当日馬体重重複が除去できているか。
2. A1/A2をpreで独立検定しHolm補正、対応経路だけB1/B2へ進める階層が妥当か。
3. 歴史TARGET馬体重とforward WHのparity floor（値・status一致99.5%、race coverage 99%、4日/400行）およびT−28 complete coverage 95%が十分か。
4. 主母集団でDNF含有raceを除外し、full-starter版を感度分析に分ける修正がN1 OOF被覆と市場分母の双方に対して妥当か。
5. offset conditional-logitのnullが市場温度とclean scoreを十分に統制し、馬体重の増分を分離できるか。
6. official physical track値の標準化に未来期間が混ざらず、race内定数だけで改善する偽経路が閉じているか。
7. Stage 0で結果・人気・オッズが歴史torchからambientに漏れる経路が残っていないか。
8. 実務floorの固定方法、meeting-day bootstrap、LOO、placeboが十分か。

レビュー結果は「凍結可」「修正後に凍結可」「停止」のいずれかで、必須修正と推奨修正を分けて返してください。TM/DM等のJRA-VAN独自指数を追加する提案は対象外です。現行productionの変更、2024/2025開封、ROI、候補生成も提案しないでください。

確認できれば「v0.2凍結可」と返してください。Stage 0実装や結果開封はまだ行わないでください。
