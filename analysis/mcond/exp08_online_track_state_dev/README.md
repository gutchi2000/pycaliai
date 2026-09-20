# EXP08 — 当日馬場状態オンライン推定AI

## 絶対条件

EXP01〜EXP07・v6・EXP05・EXP05-F・compute_bets.pyは変更しない。本番の印・買い目へ
接続しない。実結果を見て状態変数・半減期・Gateを変更しない。対象レースより後の
結果を使わない。2024〜2025年を完全未使用期間と呼ばない。ROIを主目的にしない。
他セッションの未コミット変更(`analysis/day_state_counting*.py`,
`analysis/day_waku_z_forensics.py`等)を巻き込まない(読み取り専用で参照)。

新規作業は`analysis/mcond/exp08_online_track_state_dev/`へ限定する。

## 現在の進捗(2026-09-20夜時点)

| 段階 | 状態 | 成果物 |
|---|---|---|
| Stage 0(先行研究・既存実装監査) | **完了** | `PRIOR_ART_AUDIT.md` — 内外(枠)次元は`analysis/day_state_counting_stage2.py`(別セッション、未コミット、v6残差化・permutation placebo済み)で既に**却下**(real<placebo)。脚質次元も同スクリプトのStage1で却下(raw ROI版2件も死亡)。時計・上がり・ペース次元は先行研究なし。**スコープ縮小を推奨、ユーザー判断待ち** |
| データ可用性監査(§5) | **未着手** | Stage 0の結論(スコープ確定)を待って着手 |
| Gate 0 | **未着手** | |

## 次の一手

`PRIOR_ART_AUDIT.md`§6の推奨(内外を除外、脚質を探索的に格下げ、時計/上がり/ペースを
主軸にする、permutation placeboをGate基準へ追加する)についてユーザーへ確認済み
(回答待ち)。回答後、確定したスコープでStage 1データ可用性監査へ進む。
