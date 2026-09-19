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

## v2更新 (2026-09-19、登録前Gateのための修正)

`live_history.py`を新設し、C2/C3の生成方式を「2025年末で凍結した状態を馬名で単純に繰り越す」
方式(v1)から、**血統登録番号(2025年末まで)＋馬名解決(2026年の確定済みレースのみ)を
つないだ時点安全なchainに対し、Weng-Lin更新とasof統計を逐次実行する**方式(v2)へ置き換えた。
`FEATURE_PARITY.csv`(spec §2の全117列監査)の結果、**F-forward = F-serveの117列全て**が
何らかの値で生成可能と確認した(完全に0埋めになる列は無い)。`SERVE_PARITY.md`
(spec §6の代替検証、後述)で、この新ロジックが既存研究実装(exp01/exp02)と同じ値を出すことも
確認済み。旧v1のC1「93%(109/117)」という記述は、F-serveから既に除外済みの列(前走日付等)を
誤って母数に含めた集計ミスであり訂正する(それらはF-serveに元々含まれない、
`analysis/mcond/exp05_market_residual_dev/DATA_AUDIT.md`参照)。

新しいモデル凍結artifact自体(`frozen_model.joblib`)は**変更していない**
(M1/M3/M4の係数・標準化統計・isotonic較正器・tauは元のまま)。変わったのは
F-serve/F-forwardの**生成ロジック**だけであり、117列という特徴集合自体は同じなので
再fitは不要。

## 既知の限界 (正直に記録する)

1. **tauの推定**: v6生スコア(bundle.jsonのai_score)を勝率/3着内率へ変換する温度パラメータは、
   2026-09-11に本番v6が重み計算バグ修正で再学習されているため、2023年基準(v6base.pyの本来の
   やり方)ではなく**2024年の本番v6スコア**で再推定した(`tau_2026_estimate=0.862`, 参考: 旧2023
   基準は0.865で近い値)。2026-09-11以降のモデルで生成されたスコアの尺度がこれと有意にずれて
   いないかは、前向き観測が貯まった時点で診断すべき(分布監視§10のv6生スコア分布モニタで検知する)。
2. **dyn_skill_mu / horse_skill_minus_field / raw_career_runs (v2で改善済み、残存する限界のみ記載)**:
   `live_history.py`が血統登録番号(既知馬)または`NEW:<馬名>`(2026新規デビュー馬)をキーとした
   chainに対し、2026年の確定済みレース結果を**逐次反映**するようWeng-Lin更新を行う
   (2025年末で凍結する旧方式を廃止)。2026-09-19時点でdyn_skill状態一致率は77.4%
   (旧方式51.2%から改善)。残る既知の制約:
   - 同名別馬の衝突 (2025年末レジストリで478件/67,973頭=0.7%、`horse_identity.py`)。
   - 2026年デビュー馬は`NEW:<馬名>`という馬名限定キーで管理されるため、**同姓同名の
     2026年デビュー馬が複数いた場合はその2頭の成績が混ざる**(血統登録番号が無いため
     原理的に回避不能)。
3. **C2 (陣営選択の生特徴) は8/8列を算出 (v2で全列対応、旧v1は4/8のみだった)**:
   `raw_log_int`, `raw_dist_chg`, `raw_venue_chg`, `raw_surface_chg`, `raw_cls_chg`,
   `raw_jockey_same`は馬のchain(直前の実際のレース)から、`raw_jt_pair`は騎手×調教師コードの
   時系列頻度から、`raw_jq_delta`は騎手の格の時系列変化から、いずれも時点安全に算出する
   (`SERVE_PARITY.md`で既存実装と同じ値になることを確認済み)。行単位では「前走が無い馬」で
   自然にNaNになる(対象レースにデビュー2戦目以降の馬は約77%、EXP04の学習データでも同様の
   欠損率)。
4. **重大な発見・修正: カテゴリ値の表記ゆれ**。週次CSV(`predict_weekly.parse_csv`)の
   `芝・ダ`列は"ダート"、`芝(内・外)`は"内"/"外"(先頭スペース無し)、`馬場状態`/`天気`には
   "(暫定)"接尾辞が付くことがあるが、`master_v2.csv`(学習時)は"ダ"、" 内"/" 外"、接尾辞無しの
   表記。**この不一致は本番`export_weekly_marks.py`の`apply_encoders()`にも存在し、
   本番v6モデルの`芝・ダ`特徴(LabelEncoder, classes=['__NaN__','ダ','芝'])がダート戦で
   毎週`__NaN__`(未知カテゴリ)に落ちている可能性が高い**(ユーザーへ別途報告済み、
   本番コードはこのセッションでは変更していない)。EXP05-F自身の特徴生成
   (`frozen_encode.normalize_categorical`)ではこの表記ゆれを吸収する正規化を追加済み。
5. **feature_snapshot.pyのC1列カバレッジ (訂正)**: 旧記載の「117列中109列(93%)」は誤り
   (母数の取り違え、上記v2更新参照)。正しくは`FEATURE_PARITY.csv`のとおりF-serve 117列全てが
   何らかの値を持つ。列ごとの行単位欠損率(0〜99.65%)は同ファイル参照
   (`prev_hosei`/`prev_hosei9`が最大、これは2026-05-31以降のTARGET補正タイム供給断が原因で
   既知のCLAUDE.md記載問題であり、EXP05-F固有のバグではない)。

## 前向き予測での使い方

1. 週末の出走表が来たら `feature_snapshot.py --date <date>` を実行 (production の週次サイクルと同じカデンス、市場と無関係に事前実行可)。
2. 各レースの発走31-38分前に `market_snapshot.py --once <rid> --date <date>` が起動し、
   オッズ取得→検証→`predict_and_store.store_prediction()`でM1/M3/M4予測を計算・保存する。
3. 結果確定後 `join_results.py --date <date>` で結果テーブルを別途保存する。

いずれも `frozen_model.joblib` の係数を読むだけで、再学習は一切発生しない
(`test_forward_shadow.py::test_frozen_model_deterministic`で決定論性を確認済み)。
