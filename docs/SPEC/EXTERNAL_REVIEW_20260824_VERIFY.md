# 外部レビュー(ChatGPT/Codex 2026-08-24)の検証記録 — 2026-08-25

対象: `docs/SPEC/EXTERNAL_REVIEW_20260824.md` が主張する変更一式。
ChatGPT が実際に書き換えたのは **mtime 2026-08-24 22:59〜23:48 の 16 ファイル**。
（`CLAUDE.md` / `build_site.py` / `docs/cowork_prompt.md` / `scripts/build_note_article.py` /
`data/kekka/wide_kekka.csv` / `data/pycali_history.parquet` / `data/level_norms.json` /
`reports/cowork_output/20260816_bets.json` は mtime 08-17/08-23 = ユーザー自身の未コミット分で対象外）

検証はすべて読み取り専用。本番 `data/` `models/` `reports/` `site/` は一切書き換えていない
（bundle 再生成は `export_weekly_marks.py --out-dir` でスクラッチパッドへ退避）。

---

## 結論

**コード変更そのものは概ね本物で、報告書の数値もほぼ再現した。**
ただし報告書が一言も触れていない**重大な運用副作用**が 1 件ある。

> ### ★ 見送りゲートが 4 分の 1 に緩む（未開示・要判断）
> `field_chaos_score` は **calibrator 適用後の p_win の正規化エントロピー**
> (`export_marks_json.py:126-133`)。calibrator を差し替えると chaos 分布が丸ごと動く。
>
> | | 旧(本番実出力) | 新(修正後) |
> |---|---:|---:|
> | chaos 平均 | 0.8909 | 0.8201 (−0.0708) |
> | chaos ≥ 0.92 のレース | 233R | 58R |
> | 見送り判定 | 241R (34.5%) | **72R (10.3%)** |
>
> `CHAOS_SKIP = 0.92` は旧分布の **66.7 パーセンタイル (実測 p66.7 = 0.9201)** に置かれていた。
> 新分布で同じ割合を切る閾値は **0.858**。0.92 のままだと 8.3% しか切らない。
> 20日/699R で **170R が「見送り→買い」に反転**（逆方向は 1R のみ）。
> 新たに買うことになる 170R の ◎複勝率は **48.8%**（全体 53.4% より低い＝相対的に弱いレース）。
>
> 参戦レース数 458R → 627R（**+37%**）。参戦プールの ◎複勝は 53.71% → 53.43% でほぼ横ばい。
> つまり「印の精度は落ちないが、参加が 1.4 倍になる」。
> `project_participation_selectivity_realtool` の実証（選別で ◎複勝の負けを約 5pt 削減）に
> 照らすと、これは ROI に直接効く方向の変更であり、**意図した変更として承認するか、
> 閾値を再導出するかを人間が決める必要がある**。
>
> 併せて `data/chaos_quantiles.json`（mtime 2026-06-12、nicegui/betting_judgment の
> パーセンタイル表示に使用）も陳腐化する。

---

## 1. 実測 A/B — 本番 bundle(旧コード) vs 再生成 bundle(新コード)

20 日分（20260613〜20260816）を新コードで再生成し、当時の本番 bundle と実結果で採点。

```
対象 20日 / 699R
旧(本番実出力)   ◎勝率=22.60%  ◎連対=37.48%  ◎複勝=50.50%
新(修正後)      ◎勝率=26.04%  ◎連対=40.49%  ◎複勝=53.36%

◎複勝 差分 = +2.86pt
paired bootstrap CI95 = [-0.57, +6.30] pt   P(改善) = 0.944
◎が変わったレース 274/699 (39.2%)
  └ その 274R だけ: 旧 42.34% → 新 49.64%  (+7.30pt)
```

**方向は良いが 95% 有意ではない**（CI が 0 をまたぐ）。点推定で採否を決めてはならない、
というプロジェクト規律どおり、これは「悪化していない強い証拠」であって「改善の証明」ではない。

◎ が変わる原因は **calibrator ではなく特徴量修正**。
isotonic は単調なので calibrator を替えても順位＝印は動かない
（`reports/calibrators_v6_serve_eval.json` の `hon_top3_pct` は prod/serve で完全一致 60.19%）。

---

## 2. 主張の正誤表

| 報告書の主張 | 判定 | 実測 |
|---|---|---|
| P0-1 騎手/調教師/馬 rolling を as-of 再計算に置換 | **TRUE** | 下記 §3 |
| 2026-08-16 で jockey_fuku30 98.3% / horse_fuku10 73.2% 等 | **TRUE** | 再現（98% / 73%、nunique 17 も一致） |
| P0-2 着度数は 52/53/55 の3形式 | **TRUE** | 全44ファイル走査で 19/52/53/55 のみ。53列が84% |
| P0-2 52/53 から馬体重は回収できない | **TRUE** | 該当列が存在しない |
| P0-3 マスクは baseline 駆動、numeric21+cat13=34 | **TRUE** | pkl meta が 21/13、dead_gain 14.8805 |
| P0-3 dead gain 28.15% → 14.8805% | **TRUE** | 実 canary で 20週 13.11〜15.66%（8/16=14.24%） |
| P0-3 ECE 改善（単勝 0.0262→0.0156 等） | **TRUE だが比較相手が違う** | 下記 §4 |
| P0-4 `R`→`Ｒ` | **TRUE** | 非欠損100% / 値域1-12 = master と一致 |
| R3 低予算で allocate が予算超過 | **TRUE** | 下記 §5 |
| R6 複勝 cluster CI [58.9, 79.7] | **TRUE** | 完全再現 |
| R6 未開催を損失計上していた | **PARTLY_TRUE** | バグは実在。ただし確定データでは影響ゼロ（新旧 total 完全一致 ROI 71.9%）。効くのは Phase B の中間状態のみ（そこでは複勝 −16.3pt） |
| R7 仕様書の訂正点 6 件 | **未反映** | VOL1/VOL3/VOL4/ALL_IN_ONE は今も「39特徴 / 28.15%」のまま |
| `pytest tests -q` → 86 passed | **TRUE** | 86 passed in 11.44s |
| 「本番 pkl を更新した」 | **TRUE かつ影響大** | `export_weekly_marks.py:216` が `pl_calibrators_v6_serve.pkl` を優先ロード＝本番経路 |

---

## 3. P0-1 — 学習時定義との一致検証（本丸）

master_v2 の 2025-06（3,976 行）を serve 経路 `fill_history_features` に通し、
master が持つ正解値と突合。

| 特徴 | master非欠損 | serve非欠損 | 完全一致 | MAE | 平均差 | 相関 |
|---|---:|---:|---:|---:|---:|---:|
| jockey_fuku30 | 100.0% | 100.0% | 52.2% | 0.0214 | +0.0008 | 0.963 |
| jockey_fuku90 | 100.0% | 100.0% | 46.2% | 0.0084 | +0.0016 | 0.993 |
| trainer_fuku30 | 100.0% | 100.0% | 61.9% | 0.0145 | +0.0009 | 0.971 |
| trainer_fuku90 | 100.0% | 100.0% | 58.5% | 0.0052 | +0.0017 | 0.994 |
| horse_fuku10 | 76.2% | 74.6% | 97.8% | 0.0019 | +0.0019 | 0.998 |
| horse_fuku30 | 59.3% | 58.0% | 94.1% | 0.0017 | +0.0017 | 0.999 |

- **系統バイアスは実質ゼロ**（平均差 +0.0008〜+0.0019）。「NaN より悪い誤った値」にはなっていない。
- `horse_fuku10/30` の serve カバレッジ 74.6%/58.0% は**学習時の上限 76.2%/59.3% にほぼ張り付いている**。
  報告書の 73.2%/61.1% は誇張ではなく、むしろ天井近い。
- 残差スキュー: 学習は `shift(1).rolling()` で**同日の先行レースを窓に含む**が、
  serve の `rolling_rate` は `searchsorted(side="left")` で**同日を全除外**する。
  騎手の1日騎乗数は中央値3・p90 8 なので、窓30 に対し中央値10%（p90 27%）の観測がズレる。
  これが fuku30 の完全一致率 52% / MAE 0.021 の正体。相関 0.963 なので実害は小さいが、
  厳密パリティを求めるなら履歴に発走時刻を持たせる必要がある（現 parquet は日付のみ）。
- `data/_horse_history.parquet` は 08-24 23:02 に**再生成済み**（657,159行、
  `jockey_code`/`trainer_code` 列あり、max_date=20260816、pos NaN 0.03%）。主張と矛盾なし。
- `predict_weekly.parse_csv` の静的 `jockey_stats.csv` merge と中央値 0.200 埋めは残っているが、
  `export_weekly_marks.py:361` の `fill_history_features` が**後から上書き**するので serve 側が勝つ。
  ただし `fill_history_features` は try/except で **fail-open** — 例外時は静かに旧来の
  2025年末スナップショット値に戻る（canary は定数化しない限り気づけない）。

---

## 4. P0-3 — calibrator 差し替えの実効果（報告書より良い）

報告書の ECE 表は **新 serve cal vs prod cal(全特徴 fit)** の比較で、
**実際に本番が使っていた旧 serve cal との比較ではない**。
そこで現在の実 serve マスクで test 2024-25（6,909R）を採点し直した:

| calibrator | ECE 単勝◎ | ECE 複勝◎ | ECE 馬連◎〇 | ◎複勝率 |
|---|---:|---:|---:|---:|
| 旧 serve cal (mask 14) | 0.0356 | 0.0386 | 0.0305 | 60.19% |
| **新 serve cal (mask 34)** | **0.0156** | **0.0273** | **0.0201** | 60.19% |
| prod cal (全特徴 fit) | 0.0262 | 0.0371 | 0.0299 | 60.19% |

**旧 serve cal 比で −56.1% / −29.4% / −34.1%**。報告書は自分の成果を過小に書いていた。
◎精度は 3 者とも完全一致（isotonic は単調＝順位不変）。
`load_current_serve_mask` は baseline JSON が無い/壊れていれば **例外を投げる fail-closed**。

確率そのものの移動量（isotonic 出力の grid 比較）: 平均 |Δ| は単勝 0.058 / 馬連 0.045 / ワイド 0.041、
最大 0.375。p=0.30 付近で 0.292→0.364（+7.2pt）。**これが chaos を動かし §0 の副作用を生む。**

ロールバック: `models/pl_calibrators_v6_serve_20260824_231516.pkl` が旧本番 pkl の実体
（18,552 byte = git LFS の size と一致）。`models/*.pkl` は LFS 管理なので
`git checkout` より**このバックアップを戻す方が確実**。

---

## 5. R3 — compute_bets（本番予算では挙動不変、低予算のみ変化）

`reports/cowork_input/20260816_bundle.json` 全35R を新旧で dry 実行し比較:

- **予算 ¥10,000（本番既定）: 出力は完全一致（差分0行）**。ライブオッズ
  (`reports/live_odds`, ワイド91組) を食わせても完全一致。`_wdo_floor` の変更は
  今回のデータでは点数を1点も動かさなかった（保守側への潜在的な正しさの改善）。
- 変わるのは低予算時のみ:

| 予算 | 旧 総投資 | 旧 予算超過R | 新 総投資 | 新 予算超過R |
|---:|---:|---:|---:|---:|
| ¥1,000 | 14,500 | 2R | 13,000 | 0R |
| ¥450 | ¥500券を発行 | — | 0（見送り） | 0 |

  旧は `budget=450` でも ¥500 の券を出していた＝**予算超過が本当のバグ**。修正は正しい。
- `allocate()` の新しい `ValueError` が本番に到達する経路は**無い**。
  shape エンジンは `cap = max(1, min(8 or 5, budget // MIN_BET))`（`compute_bets.py:773,780`）で
  既に候補数を予算で抑えており、topdown 側には新設の事前 pop ループがある。
  `CB_ENGINE=shape` + 低予算でも例外なしを実測（¥1,000/¥500 とも exit=0）。
- prob-first は壊れていない（EV 閾値の再導入なし）。`ENGINE_VERSION` は `2026-08-24` に更新済み。

---

## 6. R4 — fail-closed

- `validate_cowork_bets.main()` は **`--apply` で矯正した後は 0 を返す**（`:392`）。
  したがって t10_runner の `if rc != 0` は正常な矯正では発火しない＝過剰 fail-closed ではない。
  この契約は新テスト `assert vcb.main() == 0` で固定されている。
- `compute_bets.apply_to_bets_json(date_str, computed, stamp=None)` は t10_runner の
  呼び出し `(date_str, [race], stamp={...})` と一致。
- 馬単を `REJECTED_KINDS` へ移した影響はゼロ。過去 35 本の `*_bets.json` に
  馬単は **0 件**（ワイド128/馬連144/単勝59/複勝53 のみ）。
- `gutchi_brain` の残骸は docs と analysis のコメントのみ。本番実行経路に import は残っていない。
- `weekly_nicegui.ps1`: `cowork_results.json` の `generated_at` 不一致/読取失敗を
  Warn → **Fail（exit 1）** に変更。`project_unwired_roi_audit` の「凍結検知 Warn→Fail(最安)」が配線された。

---

## 7. P0-4 / canary

- `Ｒ` は復活。dtype は object（"1".."12"）だが `backtest_pl_ev.py:224` の
  `apply(pd.to_numeric).fillna(-9999)` で数値化されるので学習時 int64(1-12) と整合。gain 0.68%。
- `CONST_OK_COLS={"馬場状態","天気"}` は**意図どおり機能する**:
  - 20260816: 全行 `良(暫定)`/`晴(暫定)` → 既定 0.00 → allow_constant 1.00（偽陽性を回避）
  - 20260726: 全行**空文字** → allow_constant でも 0.00（本当の欠落は見逃さない）
  - 実際 20260726 の再生成は exit=2 でゲート発火した。これは**正しい発火**で、
    その週の weekly CSV に馬場状態/天気が入っていない（隣の 20260725 には入っている）。
- 新設の absolute canary（gate 35%）: 実測 dead gain は 20週で 13.11〜15.66%。
  **35% はヘッドホームが広すぎて、破滅的な破壊しか捕まえられない**。閾値の根拠はコードにも報告書にも無い。
  さらに `try/except` で全例外を warning に落とすため、gain 集計が失敗するとゲートは静かに無効化される（fail-open）。

---

## 8. 残課題 / 未実施

1. **`CHAOS_SKIP` の再導出**（§0）。等価点は 0.858。`data/chaos_quantiles.json`(2026-06-12) も再生成が要る。
2. 見送り4閾値が **3ファイルに重複**: `validate_cowork_bets.py:56-58` / `compute_bets.py:152` /
   `build_bet_plan.py:32`。閾値を動かすなら共通モジュール化が先。報告書の「未実施」は正しい。
3. **`docs/SPEC/` の数値が未更新**（39特徴 / 28.15% のまま）。台帳が二重管理になっている。
4. **`tests/test_serve_parse.py` は untracked (`??`)**。`git add -u` では取りこぼす。
   `data/serve_feature_baseline.json` も同様（`.gitignore` からは除外済みだが未 add）。
5. 回帰テストの穴: `Ｒ` のリネーム、t10_runner の fail-closed 分岐、
   jockey_fuku30 の学習時定義パリティ — いずれも未テスト。
6. `fill_history_features` の fail-open（例外時に 2025 年末スナップショットへ静かに退行）。
7. `by_type["races"]` は実は券数（レース数ではない）。cluster CI の実標本 `n_ci_units` は JSON に出ない。
   馬単は 122券/26レースで、復元標本の 12.3% が払戻ゼロ → `roi_ci95` 下限が 0.0 に貼り付く退化 CI。

---

## 9. 推奨

| 優先 | 対応 |
|---|---|
| **P0** | Phase A を回す前に `CHAOS_SKIP` を決める。0.92 据え置き（参戦 +37% を受け入れる）か 0.858 へ再導出か。判断材料は ◎精度ではなく **ROI/決済実績**であるべき |
| P1 | `docs/SPEC/` の 28.15%/39特徴 を 14.88%/34特徴 に訂正 |
| P1 | `git add` に `tests/test_serve_parse.py` と `data/serve_feature_baseline.json` を含める |
| P2 | absolute canary の 35% を実測分布から再設定（例: 直近中央値 +5pt）／ try-except の fail-open を fail-closed へ |
| P2 | 見送り4閾値の共通モジュール化 |
| — | コード変更自体は採用可。ロールバックは `models/pl_calibrators_v6_serve_20260824_231516.pkl` を戻す |

## 検証コマンド（再現用、すべて読み取り専用）

- 学習時パリティ: master_v2 の 2025-06 を `serve_history_feats.fill_history_features` に通し正解列と突合
- calibrator A/B: 現行 serve マスクで test 2024-25 をスコアし、旧/新/prod cal で `serve_skew_eval.evaluate`
- bundle A/B: `python export_weekly_marks.py --csv data/weekly/{d}.csv --model v6 --out-dir <scratch>` を20日分
- compute_bets A/B: `git show HEAD:compute_bets.py` を退避して `--dry` を新旧同条件で実行し diff
