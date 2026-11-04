# 収集待ち 現況 (自動生成 2026-11-04 22:17)

`python -m analysis.collection_wait_gate` が Phase C で生成。台帳本体は `COLLECTION_WAIT_REGISTRY.md`（人が書く）。
通知は Gate 到達・減少・消失・停滞のときだけ。

| ID | 対象 | 現在 | 必要 | 進捗 | 最終データ日 | 正本 |
|---|---|---|---|---|---|---|
| CW-01 | EXP19 当日馬体重 W経路 (T−28 完全率) | 1 開催日 | 4 | 25.0% | 20260926 | `analysis\mcond\exp19_bodyweight_track_condition_dev\out\forward_parity.json` |
| CW-02 | 障害履歴 forward-only collector | 20 開催日 | — | (Gate なし) | 20261101 | `data\history_only\jump\manifest.json` |
| CW-03 | EXP05-F T−35 前向き市場観測 (valid primary) | 391 race | 6600 | 5.9% | 20261101 | `analysis.mcond.exp05_forward_shadow.observation_report --cumulative` |
| CW-09 | 三連複 0B35 500R (本番記録のみ、Dry は数えない) | 0 race | 500 | 0.0% | — | `data/forward_prices/*/*trio_t10*` |
| CW-10 | shadow a95_equal_v1 (紙上台帳・全レース) | 113 race | 500 | 22.6% | 20261101 | `data/shadow_ledger/a95_equal_v1/settlements` |
| CW-11 | 本線 a95_equal_v1 前向き cohort (参戦ガード通過・決済済) | 70 race | 500 | 14.0% | 20261101 | `reports/cowork_output/*_bets.json (a95_all.guard_passed) × data/kekka` |

## 今回の通知

- CW-01 停滞: 最終 20260926 だが開催は 20261101 まで進んでいる (EXP19 当日馬体重 W経路 (T−28 完全率))
