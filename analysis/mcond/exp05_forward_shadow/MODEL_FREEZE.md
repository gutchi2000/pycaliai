# EXP05-F モデル凍結 (spec §5-6)

`freeze_model.py` / `horse_identity.py` の実行結果。以後、前向き評価が終わるまで変更しない。

## 凍結した成果物

| ファイル | 内容 | sha256_16 |
|---|---|---|
| `out/frozen_model.joblib` | M1/M3/M4の係数・標準化統計・isotonic較正器・tau | `57ba31bd51bf3bc0` |
| `out/horse_state_2025.json` | 馬名→2025年末時点のdyn_skill状態・career_runs | `8f0294e044c26b7f` |
| `analysis/mcond/exp04_invariant_info_dev/out/feature_typing.json` (参照のみ、EXP04所有) | C1の凍結エンコード規則 | `8c03f4e96df5c86b` |
| `data/serve_feature_baseline.json` (参照のみ) | F-serve定義の根拠 | `8824c85efcd6818c` |

`out/freeze_manifest.json` に学習行数・選択済みC・tau推定値を記録。

## モデル定義 (exp05_market_residual_devから変更なし)

- **M1**: `outcome ~ logit(calibrated_v6_probability) + logit(market_probability)`
- **M3**: M1 + `v6_score + v6_rank + score_gap_to_top + score_gap_to_second + score_percentile_in_race + field_score_dispersion`
- **M4**: `offset(logit(M3の予測)) + F-serve(117列)`、offset係数固定1、L2正則化

正則化Cは exp05_market_residual_dev の train(2016-2021)/sel(2022) 分割で選ばれた値を
**再選択せず**再現して使う: C(M1)=0.1, C(M3)=0.1, C(M4)=0.01 (`out/freeze_manifest.json`)。
最終係数・標準化統計・isotonic較正器は **2016-2025の全477,638行**で一度だけ再fit した
(spec §6: 「結果利用可能な最終日までの過去データで一度だけ学習する」)。

## 既知の限界 (正直に記録する)

1. **tauの推定**: v6生スコア(bundle.jsonのai_score)を勝率/3着内率へ変換する温度パラメータは、
   2026-09-11に本番v6が重み計算バグ修正で再学習されているため、2023年基準(v6base.pyの本来の
   やり方)ではなく**2024年の本番v6スコア**で再推定した(`tau_2026_estimate=0.862`, 参考: 旧2023
   基準は0.865で近い値)。2026-09-11以降のモデルで生成されたスコアの尺度がこれと有意にずれて
   いないかは、前向き観測が貯まった時点で診断すべき(分布監視§10のv6生スコア分布モニタで検知する)。
2. **dyn_skill_mu / horse_skill_minus_field**: `horse_identity.py`が2025年末までの全履歴
   (2013-2025, 626,774走)からWeng-Lin状態を再構築し、**馬名で**2026年の出走馬に紐付ける
   (週次CSVに血統登録番号が無いため)。既知の制約2点:
   - 同名別馬の衝突が478件(67,973頭中0.7%)。直近出走を優先する简易ルールで解消しているが、
     稀に別馬の状態を引き継ぐ可能性がある。
   - **2026年に入ってから既に走ったレースの結果は、この状態に反映されない**(v1の設計上の制約、
     2025年末で凍結)。今週の対象レースが初出走でない2026年デビュー馬・既に何度か走っている
     現役馬について、直近の実際のフォームより古い状態を使うことになる。dyn_skill_mu自体の
     価値がGate2で確認された程度(全候補の中で単独の寄与度は未分解)であることを踏まえ、
     v1では許容する。次のバージョンで2026年分の結果を随時追加更新する仕組みを検討する
     (README.md「次の一手」参照)。
3. **C2 (陣営選択の生特徴) は4/8列のみ算出**: `raw_log_int`(間隔), `raw_dist_chg`(距離変化),
   `raw_venue_chg`(競馬場変化), `raw_surface_chg`(芝ダ変化)は週次CSVの現走・前走要約列から
   直接計算できる。`raw_jockey_same`, `raw_cls_chg`, `raw_jq_delta`, `raw_jt_pair`は
   馬の実際の直前レース行(groupby-shift)や騎手の格の時系列集計が必要で、週次CSV単体では
   出せないため**v1では常にNaN(欠損)**として保存する(F-serveの4/117列)。
4. **feature_snapshot.pyのC1列カバレッジ**: 2026-09-19の実データで検証した結果、117列中109列
   (93%)が週次CSVから直接得られた。欠損列: `前走日付`, `前好走`, `限定`, `ブリンカー`
   (いずれも低頻度・低寄与度の列、DATA_AUDIT.md/feature_typing.json参照)。

## 前向き予測での使い方

1. 週末の出走表が来たら `feature_snapshot.py --date <date>` を実行 (production の週次サイクルと同じカデンス、市場と無関係に事前実行可)。
2. 各レースの発走31-38分前に `market_snapshot.py --once <rid> --date <date>` が起動し、
   オッズ取得→検証→`predict_and_store.store_prediction()`でM1/M3/M4予測を計算・保存する。
3. 結果確定後 `join_results.py --date <date>` で結果テーブルを別途保存する。

いずれも `frozen_model.joblib` の係数を読むだけで、再学習は一切発生しない
(`test_forward_shadow.py::test_frozen_model_deterministic`で決定論性を確認済み)。
