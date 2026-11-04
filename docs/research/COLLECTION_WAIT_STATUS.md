# 収集待ち 現況 (自動生成 2026-11-04 23:32)

`python -m analysis.collection_wait_gate` が Phase C で生成。台帳本体は `COLLECTION_WAIT_REGISTRY.md`（人が書く）。
通知は Gate 到達・減少・消失・停滞のときだけ。

| ID | 対象 | 現在 | 必要 | 進捗 | 最終データ日 | 正本 |
|---|---|---|---|---|---|---|
| CW-01 | EXP19 当日馬体重 W経路 (T−28 完全率) | 1 開催日 | 4 | 25.0% | 20260926 | `analysis\mcond\exp19_bodyweight_track_condition_dev\out\forward_parity.json` |
| CW-02 | 障害履歴 forward-only collector | 20 開催日 | — | (Gate なし) | 20261101 | `data\history_only\jump\manifest.json` |
| CW-03 | EXP05-F T−35 前向き市場観測 (valid primary) | 391 race | 6600 | 5.9% | 20261101 | `analysis.mcond.exp05_forward_shadow.observation_report --cumulative` |
| CW-09 | 三連複 500R (§5.3 初回棄却専用) | —  | 500 | Stage 0 未通過 (CW-07/08 Dry 継続)。カウントは obs_guard 管轄、ここでは数えない | — | `analysis/obs_guard.py` |
| CW-10 | shadow a95_equal_v1 (紙上台帳・全レース) | 0 race | 500 | 0.0% | — | `data/shadow_ledger/a95_equal_v1/settlements` |
| CW-11 | 本線 a95_equal_v1 前向き cohort (参戦ガード通過・決済済) | 0 race | 500 | 0.0% | — | `reports/cowork_output/*_bets.json (a95_all.guard_passed) × data/kekka` |

## 今回の通知

- CW-01 停滞: 最終 20260926 だが開催は 20261101 まで進んでいる (EXP19 当日馬体重 W経路 (T−28 完全率))
- CW-10 現在量が減少 113 → 0 (shadow a95_equal_v1 (紙上台帳・全レース))
- CW-11 現在量が減少 70 → 0 (本線 a95_equal_v1 前向き cohort (参戦ガード通過・決済済))
