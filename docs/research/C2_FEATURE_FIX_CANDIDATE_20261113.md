# C2 候補: v6 の学習特徴の不具合を直して再学習（2026-11-13）

本番は **v6 (C1) のまま**。C2 は候補で、採否は 2026-11-14 以降の前向き開催日で決める（`docs/hypothesis_registry.md` EVAL-RULE-20261113）。

## 何を直したか（監査 `audit/correctness_20261113/AUDIT_REPORT.md` §C）

| 特徴 | v6 の学習値 | C2 |
|---|---|---|
| jockey/trainer_fuku30/90 | 同じ日の先に終わったレースの結果まで含む | 「その開催週末の初日より前」の騎乗だけ（配信で実際に作れる値と同じ。日曜の bundle は土曜夜に作られ、履歴は前週末までしか無い） |
| trnH/W_days_ago | YYYYMMDD の整数差（月をまたぐと 70 前後になる） | 実日数 |
| kako5_avg/best_agari3f, kako5_hidden_good_count | 各過去走の「前走上り3F」= 1 走ずれ | 各過去走自身の上がり |
| 馬主(最新/仮想) | 未来の値で上書き | 特徴から外す |
| 斤量体重比 | 当日の馬体重 | 外す |
| 前走レースID(新)/(新/馬番無) | 日付の代理。配信では常に欠損 | 外す |
| 母馬 | 数値化で全行欠損 | 外す |
| （c2b のみ）配信で恒常的に死んでいる 18 特徴 | 学習では生きている | 外す |

他の列とハイパーパラメータ・sample weight alpha・515 ラウンドは v6 (C1) と同じ。data/master は上書きしていない（差分だけ作り直した `candidates/c2_feature_fix/data/master_c2.parquet`）。

- 作成: `analysis/c2_feature_fix/build_c2_candidate.py`（4 分）
- 比較: `analysis/c2_feature_fix/eval_c2.py`
- 出力: `E:/PyCaLiAI/candidates/c2_feature_fix/`（models/unified_rank_c2{a,b}_seed{42,1}.pkl, manifest.json, eval_c2.json）

## 結果（較正前の PL、同じレース、開催日クラスタ 95% CI）

比べる相手は `v6_served`。v6 に、配信で実際に入る騎手・調教師成績（開催週末の初日時点）を入れ、配信で死んでいる特徴をマスクした条件。

| test 2024-25（6,908R） | ◎勝率 | ◎3着内 | 勝ち馬の −log p_win |
|---|---|---|---|
| v6（学習時と同じ特徴） | 29.75% | 61.20% | 2.0799 |
| v6_served | 29.55% | 60.94% | 2.0804 |
| c2a（2 seed 平均、115 特徴） | 29.70% | 61.39% | 2.0701 |
| c2b（2 seed 平均、97 特徴） | 29.65% | 61.28% | 2.0635 |

| 差（対 v6_served） | ◎3着内 | ◎勝率 | −log p_win（小さいほど良い） |
|---|---|---|---|
| v6 学習時特徴 | +0.26pt [−0.12, +0.63] | +0.20pt [−0.12, +0.51] | −0.0005 [−0.0016, +0.0005] |
| c2a | +0.44pt [−0.27, +1.17] | +0.16pt [−0.53, +0.85] | −0.0103 [−0.0148, −0.0057] |
| c2b | +0.34pt [−0.41, +1.10] | +0.11pt [−0.60, +0.82] | **−0.0169 [−0.0219, −0.0119]** |

valid 2023（3,456R）: c2b の ◎3着内 +1.23pt [+0.19, +2.26]、−log p_win −0.0158 [−0.0218, −0.0090]。

### 読み方
- **騎手・調教師の同日リークは v6 の精度をほとんど水増ししていない**（+0.26pt、CI は 0 を跨ぐ）。監査で「本番 54% とオフライン 61% の差の主因とみられる」とした仮説は外れ。差の原因は別にある（未特定）。
- C2 は印の当たり方で v6 と区別できない。確率の質（勝ち馬に付けた確率）ははっきり良い。とくに配信で死んでいる特徴を外した c2b。
- test 2024-25 はこれまでの採用判断に使われているので、これで採用は決めない。

## 前向きの比較（2026-11-14 から）
- `weekly_nicegui.ps1` が `models/unified_rank_c2b.pkl` を見つけたら、`reports/shadow_c2/` に紙上 bundle を作る（公開・買い目・凍結版には触れない）。
- 集計: `python analysis/shadow_compare.py --shadow-dir reports/shadow_c2 --since 20261114`。本番側は凍結版だけ、影側は発走前に作った bundle だけを数える。
- 判定の目安（事前に決める）: 500R で ◎3着内の差の CI 下限が 0 を超えたら切替を検討する。下回ったら c2 は閉じる。間は記述。

## 切り替えるときにやること（ユーザー判断）
1. `models/unified_rank_c2b.pkl`（seed 42）を本番名で置く。較正器を c2b のスコアで作り直す（`build_pl_calibrators_serve.py` の MODEL_PKL を差し替え。valid 2023 で fit）。
2. `data/chaos_quantiles.json` を新しい p_win 分布で作り直す（混戦度しきい値が動く）。`class_prior` を作り直す。
3. `data/artifact_pins.json` と `data/production_policy.json`（model、policy id、起算日）を同じ commit で更新する。a95 の cohort は再起算。
4. 配信の定義は bundle のメタ情報（`trn_days_unit=days`, `kako5_agari_window=own`）で自動的に C2 用に切り替わる（`export_weekly_marks.py`）。
