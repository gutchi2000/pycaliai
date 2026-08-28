# 外部レビュー実行記録 — 2026-08-24

対象: `VOL1_SYSTEM.md` / `VOL2_BETTING_OPS.md` /
`VOL3_VALIDATION_AND_OPEN_PROBLEMS.md` / `VOL4_CODE_REFERENCE.md`

この文書は、仕様書内の「外部 AI へのレビュー依頼」R1–R8 をコード・実データで
検証し、確定した不具合を修正した記録である。仕様書の記述は参考資料として扱い、
実装・データの実測を優先した。

## 結論

- P0-1 / P0-3 / P0-4 の診断は正しかった。
- P0-2 は「着度数 parser の列数不一致」は正しいが、原因と回収可能列の説明に誤りがあった。
- 修復後の serve-dead gain は 28.15% から **14.8805%** へ低下した。
- 現行実態から作ったマスクで serve calibrator を再 fit し、本番 pkl を更新した。
- topdown、決済集計、T-10、Cowork 手動経路に仕様書未記載の不具合を発見して修正した。
- 全テスト、PL 恒等式、λ補正ペア確率、race confidence の境界を検証した。

## R1 — 欠陥台帳の検証

### P0-4 `R` → `Ｒ`

診断どおり。`predict_weekly.parse_csv()` は半角 `R` を生成し、v6 のモデル特徴は全角
`Ｒ` を要求する。`export_weekly_marks.py` と coverage 計測側の `_SERVE_RENAME` に
`"R": "Ｒ"` を追加した。

### P0-2 着度数 CSV

列数不一致は再現した。ただし実在する馬行は **52 / 53 / 55 列の3形式**だった。

- 53列: 55列形式から `馬体重` / `増減` が無い。
- 52列: 53列形式からさらに `単勝` が無い。
- 55列: `馬体重` / `増減` を含む。

したがって「53列対応で前走馬体重も回復する」という記述は誤り。52/53列には元データが
存在しない。また着度数ファイルの中央平地全成績は career 集計であり、モデル特徴の
`horse_fuku10/30`（直近10/30走）とは定義が違う。parser は3形式を受理するよう直し、
`horse_fuku10/30` は履歴 parquet から学習時と同一定義で再計算するようにした。

### P0-1 騎手・調教師ローリング

全馬同値を再現した。静的 `jockey_stats.csv` / `trainer_stats.csv` は未来リークではないが、
2025年末スナップショットであり serve 日時点の rolling 定義とも一致しない。
`_horse_history.parquet` に騎手・調教師コードを持たせ、レース日より厳密に前の走のみから
30/90走 rolling top3率を計算する実装に置換した。

2026-08-16（478頭 / 35R）の実測:

| 特徴 | notna | nunique |
|---|---:|---:|
| jockey_fuku30 | 98.3% | 17 |
| jockey_fuku90 | 98.3% | 38 |
| trainer_fuku30 | 96.0% | 16 |
| trainer_fuku90 | 96.0% | 35 |
| horse_fuku10 | 73.2% | 30 |
| horse_fuku30 | 61.1% | 87 |

同日結果は除外される。Phase C の `build_horse_history.py` 配線は既に存在することも確認した。

### P0-3 serve calibrator mask

ハードコードされた旧14特徴マスクを廃止し、`data/serve_feature_baseline.json` の
`coverage < 0.40` から毎回マスクを作る実装に変更した。現行マスクは numeric 21 +
categorical 13 = **34特徴**、dead gain は **14.8805%**。baseline JSON は canary と
calibrator の再現に必要なため `.gitignore` から除外した。

再 fit 後の test 2024–2025 ECE:

| 対象 | prod calibrator | serve calibrator | 差 |
|---|---:|---:|---:|
| 単勝 | 0.0262 | 0.0156 | -0.0106 |
| 複勝 | 0.0371 | 0.0273 | -0.0098 |
| 馬連 | 0.0299 | 0.0201 | -0.0097 |

## R2 — as-of / leak 検査

履歴検索は `date < race_date` を強制し、未来走だけでなく同日結果も除外する。
horse は直近10/30走、騎手・調教師は直近30/90走で、最小標本も明示した。
静的スナップショット merge より厳密であり、結果リークは確認されなかった。

## R3 — topdown バグ検査

以下を発見・修正した。

1. `budget < 候補数 × MIN_BET` で `allocate()` が予算超過する。
2. 100円単位でない予算は差分調整が収束しない。
3. NaN / 負値 / 全ゼロ weight の契約が未定義。
4. topdown は低予算時に候補を事前削減していなかった。
5. ワイドのトリガミ床が最悪ケースの下限でなくオッズレンジ中点を使っていた。

予算を100円単位へ切り下げ、実現不能な minimum は例外、topdown 側で低確率候補を
落としてから配分する。1候補、cap境界、低予算を合成テストで固定した。

## R4 — fail-safe / fail-closed

仕様書記載分に加えて次を修正した。

- T-10 で validator が非ゼロでも生成済み買い目を表示していた。対象 race を `bets: []`
  へ上書きして通知する fail-closed に変更。
- bundle に存在しない race_id の買い目が validator を通過していた。検証不能として強制見送り。
- `ALLOWED_KINDS` に残っていた馬単を `REJECTED_KINDS` へ移動。
- `cowork_results.json.generated_at` の不一致・欠落・読込失敗を Warn から Fail へ変更。
- 未開催レースと `match_cowork_bet()` 例外を「払戻0の損失」にしていた。決済対象から除外。
- 会場別・週次集計も確定レースだけへ限定。
- 削除済み `gutchi_brain` を毎レース import する dead code を除去。

## R5 — 確率計算

`pl_probs.py` は N=5/10/16 の全恒等式を通過した。λ補正 `pl_pair_probs()` も
`Σ馬連=1`、`Σワイド=3` を満たした。`race_confidence` の entropy は calibrator 後の
`p_win` を再正規化し `log(field_size)` で割っており、等確率で chaos=1、強い一強で
chaos≈0 になる。数学的な不具合は確認されなかった。

## R6 — CI / 統計手続き

Wilson の式と投資加重 ROI の比率は正しい。ただし旧 bootstrap は同一レース内の複数券を
独立標本として再抽出しており、分散を過小評価する。ROI を **race_id 単位 cluster
bootstrap** に変更し、CI 出力の最低条件も10 ticketsから10 racesへ変更した。

現行データの複勝は cluster CI **[58.9, 79.7]** で、`below_takeout` 判定は維持された。
Wilson hit CI は券単位の記述統計として維持したが、同一レース内相関を考慮しないため、
政策判断では ROI cluster CI を優先する。

## R7 — 仕様書の訂正点

1. tyaku は53列だけでなく52/53/55列が存在する。
2. 52/53列から馬体重は回収できない。
3. career top3率を `horse_fuku10/30` と呼ぶのは定義不一致。
4. P0修復後の serve-dead は39特徴/28.15%ではなく34特徴/14.8805%。
5. ROI CI は ticket bootstrap ではなく race cluster bootstrap が正しい。
6. 基準テスト数は73から本変更後 **86**（最終実行時点）へ増加した。

## R8 — 保守性

実施: baseline-driven mask、baseline の版管理、dead `gutchi_brain` 除去、parser schema の
明示辞書化、運用回帰テスト追加。

未実施（挙動変更または別検証が必要）:

- 見送り4閾値の共通モジュール化（4ファイル重複）。
- topdown 全馬 pair への較正済確率配線。
- 前走詳細15列の as-of 回収。
- calibrator Optuna の時系列CV化。
- topdown前向き300bets判定。到達前のpolicy変更はしない。

## 検証コマンド結果

- `python -m pytest tests -q`: **86 passed**。
- `python pl_probs.py`: 全恒等式通過。
- Python / PowerShell 構文検査: 通過。
- HF push / deploy、master CSV再生成、モデル・データ削除は実施していない。
