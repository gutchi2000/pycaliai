# 前向き価格・本番一致プロトコル

施行日: 2026-08-29  
policy: `data/production_policy.json`

## 目的

予測モデルの追加探索ではなく、実運用で観測した価格、実際に使ったモデル成果物、
見送り条件、買い目、shadow、確定結果を同じ来歴で結ぶ。前向き300 betの途中で
policyが変わった場合は継ぎ足さず、別cohortとして数える。

## 1レースの必須記録

1. T-10にJV-Linkから単勝・複勝・馬連・ワイドを取得する。
2. 取得payloadをlatest viewへ書くと同時に、
   `data/forward_prices/YYYYMMDD/*_t10_*.json.gz`へ追記専用で保存する。
3. 同じpayloadから本番topdownとshape shadowを同時生成する。
4. apply前にdecision snapshotを保存する。ここにはmodel確率、de-vig単勝市場確率、
   市場残差、pair確率、実判断、shadow、policy/artifact hashを含める。
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

JRAはパリミュチュエル方式なので、T-10表示オッズは固定約定価格ではない。
closeとの差は「最終市場への価格ドリフト」であり、取引所型のCLVや確定購入価格とは呼ばない。

## 禁止事項

- 300 bet到達前にROIを見て閾値、券種、予算、policyを変更すること
- 旧186 betと新policyを合算すること
- close価格や結果をdecision特徴へ逆流させること
- latest viewだけを残し、観測履歴を上書きすること
