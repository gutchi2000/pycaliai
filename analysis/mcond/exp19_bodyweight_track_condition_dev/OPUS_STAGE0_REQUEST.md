# 宛先: 実装担当 Opus — EXP19 v0.3 Stage 0追補

`spec.json`の`v03_override`を実装してください。Stage 1、2019〜2023のoutcome評価、2024/2025、ROI、production変更は開始しないでください。

1. WPを`wp1=bw_robust_z5*cushion_z`（芝のみ）と`wp3=bw_robust_z5*moist_gp_z`（芝ダ共通係数）の2列へ置換。W10列は維持。絶対値、gradient、extreme項は説明用へ降格。
2. 同じseed・fit窓・bootstrap・補間規則でA2/B2のlabel-free powerを一度だけ再計算。必要power<0.80ならWPを永久に閉じ、再縮約・項交換・閾値変更を禁止。通ればB2 floorを結果開封前に固定。
3. A1/A2はSIGNAL/FAILのみ。Aに経済floorを使わず、MDEは報告のみ。B1 floorは0.006474307618072295。旧B2 floorは使わない。判定pure functionと境界テストを更新。
4. `measurement_age_minutes`を歴史model/powerから外してforward監査だけに残す。`複上1〜4`・`複人気1〜4`禁止assertをspecと一致させる。
5. forward Gateを分離。歴史Stage 1はT−28完全性95%以上・4開催日（現在1日、残り3日）のみ待つ。TARGET対WHの値/status parity 99.5%、4日/400 paired rowsはforward/serve条件で、歴史評価を止めない。
6. WPが通った場合だけStage 1前にC_TRACK_ONLYとnested controlを実装。閉じた場合は両者とA2/B2を未検証・閉鎖と記録。
7. v0.2実測値を消さずsupersededとして残し、更新したspec/report/output、テスト数、commit hash、Stage 1未開始を報告。
