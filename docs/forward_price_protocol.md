# 前向き価格・本番一致プロトコル

施行日: 2026-08-29  
policy: `data/production_policy.json`

## 目的

予測モデルの追加探索ではなく、実運用で観測した価格、実際に使ったモデル成果物、
見送り条件、買い目、shadow、確定結果を同じ来歴で結ぶ。前向き300 betの途中で
policyが変わった場合は継ぎ足さず、別cohortとして数える。

## 1レースの必須記録

1. T-10にJV-Linkから単勝・複勝・ワイド・馬単を取得する。馬連と三連複は
   有効なactual T-10価格がまだ0件なので、正式な券種比較へ混ぜない。
2. 取得payloadをlatest viewへ書くと同時に、
   `data/forward_prices/YYYYMMDD/*_t10_*.json.gz`へ追記専用で保存する。
3. 同じpayloadから本番topdown、shape shadow、事前登録済みwide residual shadowを同時生成する。
4. apply前にdecision snapshotを保存する。ここにはmodel確率、de-vig単勝市場確率、
   市場残差、pair確率、実判断、各shadow、production/shadow policyとartifact hashを含める。
5. 発走予定時刻+60秒にJV-Linkを再取得し、`stage=close`として保存する。
6. 日曜夜に確定着順・払戻を結合する。結果情報は判断生成には一切使わない。

## 厳格な見送り

次のいずれかで買い目を必ず空にする。欠損・変換不能も見送りであり、古い買い目を
残してはならない。

- chaosが凍結参照分布の`skip_percentile`以上
- `field_size <= 7`
- ◎の単勝オッズがない
- ◎の`p_win < 0.05`
- T-10価格取得、decision保存、買い目計算、validatorのいずれかが失敗

閾値の単一ソースは`data/production_policy.json`、実装は`production_policy.py`。
生のchaos値はモデル配線変更で分布が動くため、運用ルールとして直書きしない。

## cohortの不変条件

- 実買い目とshadowの全race entryに同じ`policy_id`があること。
- policy JSON、chaos参照表、rank model、serve calibrator、serve baselineのSHA-256を刻む。
- decisionの`market_sha256`とT-10 market snapshotが完全一致すること。
- `analysis/prospective_topdown_eval.py`はpolicy欠損・混在で終了コード2。
- policy/artifact/閾値を変えた場合は新しい`policy_id`と開始日を発行し、300 betをリセット。

## 評価

`python -m analysis.forward_price_eval`で次を出す。

- decision / T-10 / closeの取得率とhash完全性
- model、T-10市場、close市場のBrier/log loss（確定結果到達分）
- `p_model - p_market_t10`の残差帯別実勝率
- 選択馬がT-10からcloseにかけて市場で支持された比率
- 見送り率と買いレース率
- `python -m analysis.evaluate_wide_residual_forward`によるArm A/M1/M2の同一発火レース比較

JRAはパリミュチュエル方式なので、T-10表示オッズは固定約定価格ではない。
closeとの差は「最終市場への価格ドリフト」であり、取引所型のCLVや確定購入価格とは呼ばない。

## 禁止事項

- 300 bet到達前にROIを見て閾値、券種、予算、policyを変更すること
- 旧186 betと新policyを合算すること
- close価格や結果をdecision特徴へ逆流させること
- latest viewだけを残し、観測履歴を上書きすること

## schema v2（観測計画 v2.1 Phase 2、2026-09-29）

`docs/research/OBSERVATION_PLAN_20260928.md`（v2.1、sha256 `a23ef193…`）§2 の保存契約。
v1 録は読み取り互換のまま残し、ファイル名も改名しない。

- **stage の改名**: 旧 `close`（発走後約 1 分の RT 録）は確定価格ではないため `close_late` として保存する。
  `jvlink_odds.py --stage close` は受け付け、`stage=close_late`・`stage_requested=close` で書く。
  読み込み側は `forward_prices.canonical_stage()` を通して比較する（v1 の `close` も `close_late` に解決）。
- **追加 stage**: `t2_candidate`（発走 2 分前、0B31〜0B35 を同一プロセスで連続取得。「T−2 決定」とは呼ばない）、
  `trio_t10`（三連複 shadow の T−10）、`final_rt_candidate`（RT 系の確定後候補、当日夜）、
  `final_stock_candidate`（蓄積系 O1〜O5、当日夜と翌日夜）。`final` の認定は §2.3 の 3 条件で別途行い、
  区分番号だけでは認定しない。
- **`jv_captures`**: spec ごとに JV-Link が返した全録を raw（CRLF 含む）のまま保存し、録ごとに
  データ区分・発表月日時分（`announced_at`）・登録/出走頭数・発売フラグ・票数計・枠連ブロック（raw と parse）・
  slot 単位の全組（未発売組は filler 文字付きで `unpriced`）・完全性・race_key 照合・
  既存パーサとの一致を付ける（`jv_records.py`）。取得開始・終了時刻（ms, JST）と rc も capture に残す。
  `market`（本番 latest view と同一 payload）は従来どおり別キーで、`market_sha256` も不変。
- **失敗時の扱い**: raw の構造化や raw 付き保存が失敗しても、`jvlink_odds.py` は従来の保存（market のみ、
  `capture_error` 付き）の成否だけで終了コードを決める。`reports/live_odds/{race}.json` の内容は v1 と同一。
- **取得ジャーナル**: JV-Link の全取得を `data/jvlink_fetch_journal/{date}/` に 1 取得 1 ファイルで残す
  （開始・終了時刻、rc、返却録数、process、stage）。並走成功率の一次記録で、書込失敗は無視する。
- **Dry**: `--dry` の取得は `data/forward_prices_dry/` と `data/jvlink_fetch_journal_dry/` に分け、
  Stage 0 と 500R の集計に混ぜない。
- **蓄積系の読み捨て**: `JVOpen("RACE")` から残すのは対象レースの O1〜O5 だけ。HR（払戻）・SE（着順）など
  他の録は種別 2 文字とレースキーだけ見て捨て、保存も解析もしない。
- **500R guard**: 性能・回収率・帯選択・候補選択の集計は `analysis.obs_guard.assert_performance_allowed()`
  を最初に呼ぶ。label-free に数えた有効 race が 500 未満なら止まる。
