# Opus宛て — EXP21 Stage 0 実装依頼

Fableレビューを反映した SPEC.md / spec.json v0.2-frozen を実装してください。

## 今回の範囲

G0データ契約とlabel-free power/可用性監査までです。ROI帯表、着順・払戻を使ったG1/G2、
2024/2025、production、モデル、買い目、資金配分には進まないでください。

## 順序

1. 8券種それぞれのsource path、sha256、期間、記録区分、価格単位、払戻単位をmanifest化。
2. 単勝・複勝・馬連について、2013〜2018 / 2019〜2023 のD0 terminalと
   historical_pre_snapshot（約T−28）の全ticket被覆を測る。結果列はjoinしない。
3. 2023 91列の時点同定:
   - 単勝をTANPUK区分4と照合。
   - 馬連18slotをUMAREN o_umfinおよびpreと同raceで照合。
   - 一致率99%以上の側だけを採用。どちらでもなければ使用禁止。
4. 枠連・ワイド・馬単・三連複・三連単の列配置を、公式払戻の的中ticketから逆引きする
   検証器を作る。構造推測だけで採用しない。
5. ticket key、一意性、対称/順序、self=0、頭数別ticket数、取消・同着・返還を検査。
6. 複勝・ワイドは sqrt(Lo*Hi) を帯割当キーとして使える被覆を測る。ROI計算はしない。
7. equal expected-hit-mass 10帯、equal ticket-count 10帯、fixed帯について、
   outcomeを使わず境界・ticket数・race数・日数・expected-hit massを生成。
8. 暦日clusterで必要な insufficient 閾値と、G1の形状差を検出するMDE/powerを計算し、
   seed・bootstrap設定とともに結果開封前にcommit。
9. 優勝者資料の公表値だけを使い、三連単帯+10ptの概算不確実性と多重比較上の位置づけを
   別文書にする。組合せ数を独立nに使わない。

## hard stop

- G0 FAIL券種をproxyで救済しない。
- 他5券種をG1/G2へ進めない。
- T−28をT−10と呼ばない。
- terminal帯を実行可能条件と呼ばない。
- 最良帯を見て境界・期間・券種を変更しない。
- 2024/2025を読まない。
- 本番ファイルを変更しない。

完了時は STAGE0_DATA_AUDIT.md、out/data_manifest.json、
out/label_free_band_coverage.json、out/power_audit.json、合成invariant test、
整合検査を作り、EXP21のファイルだけをcommitして停止してください。
