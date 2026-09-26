# Fable宛て — EXP19 v0.1仕様レビュー依頼

EXP19「当日馬体重状態 × JRA公式馬場物理値」の仕様レビューをお願いします。実装・学習・結果評価はまだ行っていません。

読むファイル:

1. `analysis/mcond/exp19_bodyweight_track_condition_dev/SPEC.md`
2. `analysis/mcond/exp19_bodyweight_track_condition_dev/spec.json`
3. `analysis/mcond/exp19_bodyweight_track_condition_dev/PRIOR_ART_AND_DATA_AUDIT.md`
4. `analysis/mcond/exp19_bodyweight_track_condition_dev/README.md`

特に次を厳しく確認してください。

1. 馬体重発表後のT−28市場に対するGate A、terminal closeに対するGate B、interactionのGate Cという順序が、継続的に勝てる情報を探す設計として妥当か。
2. `W_BODY`の10特徴と`WP`の6 interactionが多すぎないか。既存の馬場単独FAILを別表現で救済していないか。
3. 歴史TARGET馬体重とforward WHのparity floor（一致99.5%、race coverage 99%、2日/200行）およびT−28 complete coverage 95%が十分か。
4. DNFをstarterとして残し、取消だけを除く正式母集団が市場分母と一致するか。
5. offset conditional-logitのnullが市場温度とclean scoreを十分に統制し、馬体重の増分を分離できるか。
6. official physical track値の標準化に未来期間が混ざらず、race内定数だけで改善する偽経路が閉じているか。
7. Stage 0で結果・人気・オッズが歴史torchからambientに漏れる経路が残っていないか。
8. 実務floorの固定方法、meeting-day bootstrap、LOO、placeboが十分か。

レビュー結果は「凍結可」「修正後に凍結可」「停止」のいずれかで、必須修正と推奨修正を分けて返してください。TM/DM等のJRA-VAN独自指数を追加する提案は対象外です。現行productionの変更、2024/2025開封、ROI、候補生成も提案しないでください。
