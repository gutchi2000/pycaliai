# PyCaLiAI 収集待ち台帳

**最終更新**: 2026-09-27 10:40 JST  
**目的**: 研究・検証・forward運用のうち、「データが増えるまで結論を出せないもの」を一か所で管理する。

## 更新規則

1. 新しい研究対象にforward収集、標本数、開催日数、手動exportなどの待ち条件が生じた時点で、この台帳へ追加する。
2. `ACTIVE_SHORT`（数開催日）、`ACTIVE_LONG`（長期標本）、`DEPLOYMENT_ONLY`（歴史研究は止めない）、`ACCUMULATING`（用途未確定の継続保存）、`READY`、`CLOSED`を区別する。
3. 各行に、必要量、現在量、残量、次の判断、正本となる成果物を記録する。推測値で埋めない。
4. 完了・閉鎖した行は削除せず、状態と完了日を更新して履歴として残す。
5. 通常のデータ増加だけを理由に自動研究や定期報告を開始しない。失敗、Gate到達、新しい異常、明示依頼など意味のある変化があったときだけ更新・通知する。

## 現在の収集待ち

| ID | 状態 | 対象 | 必要条件 | 現在 | 残り・次の判断 | 正本 |
|---|---|---|---|---|---|---|
| CW-01 | `ACTIVE_SHORT` | **EXP19 当日馬体重 W経路** | T−28までの完全率95%以上を4開催日 | 9/26は16/16で完了。9/27はcollector稼働中で未確定 | 今日が合格すれば2/4日、残り2開催日。4日到達後にA1→B1のStage 1を開始するかユーザー判断 | `analysis/mcond/exp19_bodyweight_track_condition_dev/out/forward_parity.json` |
| CW-02 | `ACTIVE_SHORT` | **障害履歴 forward-only collector** | 自動収集2開催日のcoverage・ID・collision・append-only Gate | 9/26 raw card保存、タスク成功。9/27は22:30処理待ち | 今夜成功後に2開催日Gateを監査。通過しても特徴接続は自動で行わず、別途判断 | `data/history_only/jump/manifest.json`, `logs/jump_history_YYYYMMDD.log` |
| CW-03 | `ACTIVE_LONG` | **EXP05-F T−35前向き市場観測** | valid primary 6,600 race | 76/6,600（1.15%）。market 86、complete 83。2026-09-27 10:40時点 | 残り6,524。到達まで性能・ROI・的中率を開かない。設計上は約99開催週、約1.9年規模 | `analysis/mcond/exp05_forward_shadow/observation_report.py --cumulative` |
| CW-04 | `DEPLOYMENT_ONLY` | **馬体重 WH対TARGET値/status parity** | 4開催日・400 paired horse rows、値/status一致99.5%以上 | paired 0 | 対象日のTARGET torch形式exportが必要。forward/serve接続だけを止め、EXP19歴史Stage 1は止めない | EXP19 `forward_parity.json` |
| CW-05 | `ACCUMULATING` | **T−10/T−20/close価格・timing canary** | 現在は固定の到達Gateなし | append-onlyで継続中。2026-08-29〜09-27の保存日あり | 将来の価格形成仮説、timing異常、新しい収集障害が生じたときに用途を事前登録する | `data/forward_prices/`, `reports/exp05fs_odds/` |
| CW-06 | `ACCUMULATING` | **JRA公式馬場物理値** | 現在は固定の到達Gateなし | `PyCaLiAI_Baba`稼働、9/27 10:00成功 | EXP19 WPは検出力不足で閉鎖済み。新仮説が生じるまで保存のみ | `data/baba_today.json`, `data/baba_feats.parquet` |

## 収集待ちではあるが、現在の主研究を止めないもの

- `strategy_weights`再設計の旧トリガー「2026実績 n≥200」はStreamlit旧系統のバックログで、現在のCowork/v6主系統の研究Gateではない。別途再開を決めた場合にCW行へ昇格する。
- EXP19のWP（馬体重×馬場）は縮約後power 0.4675で永久閉鎖済み。収集が増えても同じ番号では再開しない。
- EXP13、EXP15、EXP16A、EXP17、EXP18は終了済み。通常のデータ増加だけでは再開しない。

## 次に起きる判断点

1. **2026-09-27夜**: CW-02 障害履歴の2開催日Gate監査。
2. **2026-09-27開催終了後**: CW-01 馬体重の2日目を確定。合格なら残り2開催日。
3. **以後の開催ごと**: CW-01の4日到達を確認。到達した時点でEXP19 Stage 1開始可否をユーザーへ提示。
4. **CW-03**: 6,600到達前は中間性能評価を行わず、収集失敗・schema/hash異常だけを扱う。
