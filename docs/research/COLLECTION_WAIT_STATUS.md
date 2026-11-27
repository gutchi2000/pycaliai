# 収集待ち 現況 (自動生成 2026-11-27 22:25)

`python -m analysis.collection_wait_gate` が Phase C で生成。台帳本体は `COLLECTION_WAIT_REGISTRY.md`（人が書く）。
通知は Gate 到達・減少・消失・停滞のときだけ。

| ID | 対象 | 現在 | 必要 | 進捗 | 最終データ日 | 正本 |
|---|---|---|---|---|---|---|
| CW-01 | EXP19 当日馬体重 W経路 (T−28 完全率) | 1 開催日 | 4 | 25.0% | 20260926 | `analysis\mcond\exp19_bodyweight_track_condition_dev\out\forward_parity.json` |
| CW-02 | 障害履歴 forward-only collector | 24 開催日 | — | (Gate なし) | 20261115 | `data\history_only\jump\manifest.json` |
| CW-03 | EXP05-F T−35 前向き市場観測 (valid primary) | 598 race | 6600 | 9.1% | 20261123 | `analysis.mcond.exp05_forward_shadow.observation_report --cumulative` |
| CW-09 | 三連複 500R (§5.3 初回棄却専用) | —  | 500 | Stage 0 未通過 (CW-07/08 Dry 継続)。カウントは obs_guard 管轄、ここでは数えない | — | `analysis/obs_guard.py` |
| CW-10 | shadow a95_sniper_v2 (紙上台帳・全レース) | — | — | **正本なし** | — | E:\PyCaLiAI\data\shadow_ledger\a95_sniper_v2\settlements |
| CW-11 | 本線 a95 前向き cohort (参戦ガード通過・決済済) | 11 race | 500 | 2.2% | 20261123 | `reports/cowork_output/*_bets.json (a95_all.guard_passed) × data/kekka` |
| CW-13 | 複勝票数/単勝票数 ゲートの前向き確認 (T-10) | 24 race | 100 | 24.0% | 20261123 | `data/forward_prices (schema v2 votes_total) × data/kekka` |
| CW-14 | 朝 9 時 → 発走前の動き 前向き確認 (Gate A 広い条件) | 40 点 | 300 | 13.3% | 20261123 | `data/forward_prices (am9 × t20) × data/kekka` |
| CW-15 | v7 (JRDB 過去走特徴) 対 v6 の紙上比較 | 138 race | 1000 | 13.8% | 20261123 | `reports/{cowork_input,shadow_v7,shadow_v6} × data/kekka` |
| CW-21 | v8 (JRDB 前日 KYI・外厩) 対 v7 の前向き比較 | 69 race | 1000 | 6.9% | 20261123 | `reports/{cowork_input,shadow_v8,shadow_v7} × data/kekka` |
| CW-27 | ボード一致3 の ◎単勝（帯なし）前向き確認 | 17 race | 150 | 11.3% | 20261123 | `reports/cowork_output/*_bets.json (board_combo, T-10) × data/kekka` |

## 今回の通知

- CW-01 停滞: 最終 20260926 だが開催は 20261128 まで進んでいる (EXP19 当日馬体重 W経路 (T−28 完全率))
- CW-10 正本が無い/読めない: E:\PyCaLiAI\data\shadow_ledger\a95_sniper_v2\settlements
