# EXP08 — 当日馬場状態オンライン推定AI

## 絶対条件

EXP01〜EXP07・v6・EXP05・EXP05-F・compute_bets.pyは変更しない。本番の印・買い目へ
接続しない。実結果を見て状態変数・半減期・Gateを変更しない。対象レースより後の
結果を使わない。2024〜2025年を完全未使用期間と呼ばない。ROIを主目的にしない。
他セッションの未コミット変更(`analysis/day_state_counting*.py`,
`analysis/day_waku_z_forensics.py`等)を巻き込まない(読み取り専用で参照)。

新規作業は`analysis/mcond/exp08_online_track_state_dev/`へ限定する。

## スコープ(2026-09-20夜、ユーザー承認済み・確定)

`PRIOR_ART_AUDIT.md`§6の推奨をユーザーが承認。以下へ確定し、以降は変更しない:

- **主仮説**: 時計(走破タイム残差)・上がり性能・ペースの当日オンライン状態のみ
- **内外(枠)・脚質(前残り/差し)**: 採用候補から除外、negative control専用
  (再救済しない。時計・上がり・ペースいずれにも追加情報がなければEXP08を終了する)
- permutation placebo testをGateに事前登録(§14で追加)
- 2023年developmentのみで状態定義・パラメータ・閾値を固定
- RAW/EWMA比較必須

## 現在の進捗(2026-09-20夜時点)

| 段階 | 状態 | 成果物 |
|---|---|---|
| Stage 0(先行研究・既存実装監査) | **完了** | `PRIOR_ART_AUDIT.md` — 内外(枠)は`day_state_counting_stage2.py`(別セッション、未コミット、参照のみ)でpermutation placebo失格により却下済み、脚質も却下済み。時計・上がり・ペースのみ先行研究なし |
| Stage 1データ可用性監査(§5) | **完了** | `DATA_AUDIT.md` — 走破タイム/上がり3F/PCI・RPCI等は外部`kekka_2010_2025_fix_raceid_v2__keyed.csv`で全年安定して取得可能(missing率0-3%台)。**結果利用可能時刻を2023実測で固定**(発走時刻+15分、TANPUK区分4タイムスタンプ中央値7分・99%点13分に安全マージン)。**時計の期待走破タイムモデルは時点安全な既存実装が無く新規構築が必要**(Stage2)。上がり/ペースはレース内相対値化で対応可能な見込み |
| Gate 0 | **PASS** | `DATA_AUDIT.md`末尾のチェックリスト参照 |
| Stage 2(観測信号構築、期待値モデル) | **未着手** | 次の作業 |

## 次の一手

`build_observations.py`(先行レースごとのspeed_signal/pace_signal構築、時計の
期待値モデル実装)へ進む。
