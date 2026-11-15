# v8 候補: v7 + JRDB 前日データ (KYI) + 外厩 — 2026-11-15

**結論: v8 は v7 より当たる（test 2024-25、3 seed 平均で ◎3着内 +1.47pt・◎1着 +1.70pt、全 seed で 95% 区間が 0 を外れる）。
効きの大半は KYI の前日指数（特に騎手指数）で、外厩の上乗せは +0.3pt と誤差の範囲。締切市場に対しては上乗せなし
（M ゲート none）なので、馬券の回収率は変わらない見込み。価値は「オッズが出る前の予想（朝の印・サイトの印）の精度」。**
2026 年を配信経路で作り直した比較でも v7 より AI1位 3着内 +3.65pt [+2.23, +5.04]（47 開催日 1,507R）。
本番の切替はしていない（`production_policy.json`・`artifact_pins.json` は触っていない）。

- 学習: `train_v8_kyi.py` → `models/unified_rank_v8.pkl`（git 管理外、このマシンだけ）、`reports/train_v8_kyi.json`
- 特徴: `jrdb_kyi_features.py`（接頭辞 `k8_`、69 列）。学習と配信が同じ `make_features` を通る
- 較正器・払戻カーブ・クラス別経験率: `run_v8_pipeline.py` → `models/pl_calibrators_v8.pkl`、`data/pl_payout_curve_v8.pkl`、`data/class_prior_v8.json`（すべて git 管理外）
- 市場ゲート: `lab/experiments/v8_market_gate.py` → `reports/v8_market_gate.json`
- 前段: `KYI_TENKAI_20261115.md`（KYI の展開予想は v7 に上乗せ、市場の外ではない）

## 1. 使ったデータ（JRDB 会員、git 管理外・再配布禁止）

| データ | 中身 | 期間 | 手元にあったか |
|---|---|---|---|
| KYI（前日配布の馬ごとデータ） | IDM・騎手指数・情報指数・総合指数・調教指数・厩舎指数・激走指数・脚質・距離適性・上昇度・ローテーション・厩舎評価・重適正・ブリンカー・展開予想 | 2013〜2026 | 11/15 に取得（135MB） |
| 外厩コメント（TARGET 取り込みデータ） | 外厩の施設名・日付・中N週 | 2016〜2026（出走の 48〜63%） | **新規**（11/15 取得） |
| 外厩 馬印 | 施設の略号 | 2016〜 | 新規。外厩コメントに含まれるので特徴には使わない |
| 展開 馬印 / 番手 馬印 | ゴール内外 / 内外ごとの番手 | 2016〜 | KYI の展開予想から作られたもの（2024 年で 100% / 99.3% 一致） |
| IDM 馬印 | IDM 上位 6 頭の印 | 2016〜 | KYI の IDM から作られたもの |
| 成績 IDM | レース後の IDM | 2016〜 | SED の IDM と同じ（v7 で使用済み） |
| 厩舎ランク / 騎手ランク | 調教師・騎手のランク表 | 直近 5 開催日と現在版のみ | 新規。過去分が配布されていないので学習には使えない → 毎週保存を開始 |

使わなかったもの: KYI の基準オッズ・基準人気・人気指数（市場の代わりになる値。v6/v7 と同じくオッズ系は学習に入れない）。

先読みの確認: 発走前に保存した KYI（11/14・11/15）とアーカイブで、IDM・騎手・情報・総合・調教・厩舎指数は全行一致。
展開予想の順位だけ 2〜4% の行が後から変わる（取消後の再計算と見られる）。

## 2. 学習と結果（test 2024-25、6,909 レース、3 seed、開催日 paired bootstrap）

作り方は v7 と同じ（ラベル・学習窓 〜2022・重み・ハイパーパラメータ・回数・エンコーダは本番 v6 C1 の bundle から）。特徴だけ足した。

| アーム | ◎3着内（3 seed 平均） | ◎1着 |
|---|---|---|
| V7（v7 を同条件で再学習） | 62.47% | 30.81% |
| V8K（+ KYI） | 63.65% | 32.22% |
| V8（+ KYI + 外厩） | **63.94%** | **32.51%** |

| 差 | seed 42 | seed 1 | seed 7 |
|---|---|---|---|
| V8 − V7 ◎3着内 | +1.58 [+0.52, +2.66] | +1.38 [+0.28, +2.49] | +1.45 [+0.38, +2.50] |
| V8 − V7 ◎1着 | +2.23 [+1.27, +3.18] | +1.61 [+0.67, +2.55] | +1.27 [+0.34, +2.26] |
| V8 − V8K ◎3着内（外厩の分） | −0.22 [−1.01, +0.55] | +0.61 [−0.32, +1.51] | +0.48 [−0.40, +1.29] |

gain の内訳（seed 42 の V8）: KYI 70.6%、JRDB 過去走 j1 7.5%、外厩 0.5%。上位は 騎手指数のレース内 z（39.7%）と騎手指数（18.7%）。

## 3. 市場ゲート（gate_common、valid 2023 で重み、test 2024-25 の 6,632 レース）

| z | P（本番 v7 → +z） | M（確定市場+v7 → +z） | A10（10時市場+v7 → +z） |
|---|---|---|---|
| V7（対照） | 0.0000 none | 0.0000 none | 0.0000 none |
| V8K | −0.0667 PASS（◎3着内 62.18 → 64.37%） | −0.0002 none | −0.0134 PASS |
| V8 | −0.0674 PASS（62.18 → 64.73%） | −0.0002 [−0.0009, +0.0006] none | −0.0137 PASS（65.40 → 65.65%） |

年別（2024 / 2025）は V8 の P −0.0674 / −0.0675、A10 −0.0115 / −0.0161 で、2 年とも同じ大きさ。

- v8 は v7 の予想を大きく良くし、朝 10 時の市場と組み合わせた予想もさらに良くする。
- 締切の市場（確定オッズ）は JRDB の前日情報をほぼ全部持っている。馬券の控除を越える材料ではない。

## 4. 2026 年を配信経路で再生した比較

`analysis/replay_serve_2026 run --model v7 / v8 --proxy --recover 1 --ignore-gate --truncate`（同じコード・同じ出走表・同じ設定）→
`analysis/replay_ab_compare`。KYI は 47 日とも 100% 埋まり、外厩は年単位アーカイブから 5 割台。

| 範囲 | R / 日 | AI1位 1着（v7 → v8） | AI1位 3着内 | ◎3着内 |
|---|---|---|---|---|
| 全体 | 1,507 / 47 | 27.7 → 31.3%（+3.65pt [+1.81, +5.43]） | 59.3 → 62.9%（+3.65pt [+2.23, +5.04]） | 59.4 → 63.8%（+4.38pt [+2.74, +5.89]） |
| 品質ゲート合格日 | 301 / 10 | +4.32pt [+0.66, +8.04] | +4.32pt [+2.17, +6.69] | 61.5 → 66.4%（+4.98pt [+1.58, +8.65]） |
| 入力欠落日（参考） | 1,206 / 37 | +3.48pt | +3.48pt | +4.23pt |

- AI1位が v7 と同じ馬のレースは 64%。
- 差は test 2024-25（+1.5pt）より大きい。11/12 の実験（配信済み bundle に KYI 前日指数を混ぜて +8.5pt）と同じく、
  2026 の配信で欠ける特徴を前日指数が補っている分を含む。健全な配信での期待は test の +1.5pt 前後と見ておく。
- 2026 の KYI はアーカイブから取ったもの。発走前保存と一致を確かめたのは 11/14・11/15 の 2 日だけ（CW-12 が継続照合）。
- 47 日は、このワークツリーで出走表・結果・配信済み bundle が揃う開催日（v7 の 11/13 再生は 61 日）。v7 と v8 は同じ集合で比べている。
- 参考: 発走 17 分前の 1 番人気の 3 着内率は 2026 で約 61%（`project_jrdb_data_20261112`）。v8 単独で朝の市場並み〜やや上。

## 5. 配信経路（実装済み、紙上のみ）

- `export_weekly_marks.py`: bundle に `kyi_mode` があるモデルだけ `jrdb_kyi_features.add_serve_feats` を呼ぶ。
  KYI は `data/jrdb/snapshots/Kyi/`（発走前保存）を優先し、無ければアーカイブ。KYI が無い・埋まり 90% 未満の日は bundle を不合格にする。
  外厩コメントが無い日は外厩なしとして続ける（警告のみ）。11/14・11/15 の出走表で KYI 100%、外厩 62〜63%（学習の 2025 年は 60%）。
- 較正・カーブ・基準づくりの各スクリプト（`build_pl_calibrators(_serve)`・`build_payout_curve`・`backtest_pl_ev`・`analysis/add_train_cov_baseline`・
  `analysis/measure_serve_coverage`）も `kyi_mode` を見る。
- 公開物からの除外: `marks_shap._PRIVATE_PREFIXES` と `build_site` の根拠表示の除外に `k8_` を追加（JRDB 規約）。
- `weekly_nicegui.ps1` Phase A: `models/unified_rank_v8.pkl` があるときだけ `reports/shadow_v8/` に紙上 bundle を作る（公開・買い目・本線に不干渉、CW-21）。
- `scripts/jrdb_snapshot.py capture`: KYI に加え、TARGET 取り込みデータ（外厩・外厩コメント・厩舎/騎手ランク ほか）の日付単位ファイルを
  `data/jrdb/snapshots/targetmark/` に保存（別 manifest）。厩舎/騎手ランクは過去分が無いので、ここで貯めたものだけが将来の学習に使える。
- `.gitignore`: `pl_calibrators_v[1-9]`・`pl_calibrators_v[7-9]_serve`・`pl_payout_curve_v[1-9]`・`class_prior_v[7-9]`・`reports/shadow_v8/` を追加。
  **本体の checkout で `models/pl_calibrators_v7_serve.pkl` が追跡外かつ除外外のままだった**（JRDB 由来の成果物を誤って commit しうる）のも、これで塞がる。

## 6. 本番へ切り替えるときに要るもの（未実施）

1. JRDB 規約の確認（ユーザー判断。v7 と同じ論点: 予想の公開は可、データ・指数そのものは不可）。
2. 配信用の較正器（`pl_calibrators_v8_serve.pkl`）を 2026 の再生スコアで fit、混戦度の基準分布の作り直し、`serve_feature_baseline.json` に `k8_*` の train_cov。
3. `artifact_pins.json` と `production_policy.json`（新 policy id、a95/sniper の cohort 再起算）を同じ commit で更新。
4. 金曜夜の KYI 保存が Phase A より前に終わっていること（`PyCaLiAI_JRDB_Snapshot` は金土 19:40・土日 07:30）。
5. 前向き確認 CW-21（本番と紙上 v8 の両方にあるレース 1,000）。

## 7. 再現

```
python scripts/jrdb_fetch.py download --types Kyi --from 2013     # KYI アーカイブ
python -c "import jrdb_kyi_features as k; k.build_kyi(); k.build_gaikyu(); k.load_fac_map()"
python train_v8_kyi.py
python run_v8_pipeline.py
python lab/experiments/v8_market_gate.py
python -m analysis.replay_serve_2026 run --out <dir> --model v8 --proxy --recover 1 --ignore-gate --truncate
```
外厩コメントの年単位アーカイブは `data/jrdb/raw/targetmark/`（会員ページ `www.jrdb.com/member/jrdvsp/gk/gaikyu.html` から 11/15 に取得）。
