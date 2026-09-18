# analysis/mcond — 市場条件付き追加情報検定の基盤

基準文書: `docs/research/NEXT_GEN_RESEARCH_PLAN_20260918.md`
本番 (`compute_bets.py` / `export_weekly_marks.py` / 印 / 資金配分) には一切接続しない研究用コード。

## 何を検定するか
「現行 v6 の予測」と「購入判断時点の市場確率」を**両方与えた後でも**、新しい表現が馬の結果を説明するか。
過去の能力系実験 (ELO/Glicko/EWMA 等) は市場抜きの予測ゲートだけで判定されていたので、その穴を埋める共通の物差し。

## 構成
| ファイル | 役割 |
|---|---|
| `market.py` | 時系列オッズ `data/Time _series_odds/TANPUK_*.csv` から取得時刻つきの単勝市場確率を作る |
| `v6base.py` | 時点安全な v6 スコア (2015-23 OOF / 2024-25 本番C1) と、v6・市場それぞれの PL 勝率・3着内率 |
| `rebuild_oof_c1.py` | v6 OOF を P0-5 修正後 (C1) の master で作り直す |
| `evaluate.py` | ロジスティック回帰による M0..Mk 比較、開催日ブロック bootstrap |
| `serve_gap_diagnosis.py` | offline と本番の ◎勝率差の段階分解 (本番処理は変更しない) |
| `exp01_choice_dev/` | 第1実験: 陣営選択の逸脱 |

## データの事実 (2026-09-18 にコードとデータで確認)
| 項目 | 事実 |
|---|---|
| オッズの取得時点 | TANPUK の `区分`: 1=途中 / 4=確定。1レースあたり区分1がほぼ3回 (前日23時台 / 当日9時前後 / 発走約35分前) |
| 主検定の市場 (`pre`) | 当日の区分1で確定の15分以上前の最後のもの。**確定の31〜40分前** (1%点〜99%点)、44,907レース全て。本番の購入判断 (T-10) より前 |
| 感度分析の市場 (`am9`) | 当日の区分1で 09:30 以前の最後のもの |
| オッズの種類 | 単勝 (`{n}単`)。複勝は `{n}複Lo/Hi` (本実験では決済に確定複勝配当を使い、複勝オッズは使わない) |
| 市場確率 | 1/odds をレース内で比例正規化 (overround 中央値 1.260 を除去)、取消馬を除いて再正規化。3着内率は Plackett-Luce (Harville) |
| 最終オッズ・払戻 | 確定単勝オッズは kekka_v2 の単勝配当列 (勝ち馬以外は括弧内)。**特徴量には使わない**。決済と監査のみ |
| T-10 オッズ | 2026-06 以降の JV-Link スナップショットのみ。過去期間には無いので本検定では使わない |
| v6 2015-23 | `oof_scores_c1master.parquet`: 各年 Y を train≤Y-2 / early stop Y-1 で学習 (本番 C1 の学習重みは含まない近似) |
| v6 2024-25 | 本番 C1 モデル (train≤2022)。2024-25 にとって OOS |
| 旧 OOF | `data/oof_scores_v6params.parquet` は **P0-5 修正前 master 由来** (trainer_fuku30/90 の同一レース内リーク) なので使わない |
| 尺度合わせ | PL 温度: OOF=0.845 (2022 で推定) / 本番=0.865 (2023 で推定)。2023 の同一レースで順位相関 0.964 |
| 馬主 | キャリア中に値が変わる馬が 0.0% = **最新値で上書き** → 使用禁止 |
| 調教師 | 17.4% の馬で変わる (転厩1回 6,313頭) = 当時の値 |
| 2024-25 の既往利用 | reports/*.json と *.py に多数の参照あり → 「再利用済み期間での探索的OOS」 |
| 同日の扱い | 陣営・騎手の統計は対象日より前の日だけで集計 (同日の他レースは使わない) |

## 実行順
```
python -m analysis.mcond.market
python -m analysis.mcond.rebuild_oof_c1
python -m analysis.mcond.v6base
python -m analysis.mcond.exp01_choice_dev.build_features
python -m analysis.mcond.exp01_choice_dev.build_bp
python -m analysis.mcond.exp01_choice_dev.audit
python -m analysis.mcond.exp01_choice_dev.test_time_safety
python -m analysis.mcond.exp01_choice_dev.run
python -m analysis.mcond.serve_gap_diagnosis --regen-prec1
```
中間データは `data/_research/mcond/` (gitignore)。

## 次の実験を足すとき
1. `analysis/mcond/expNN_*/` を作り、`spec.json` に仮説・split・特徴・モデル・合否条件を書いて**評価前にコミット**
2. 特徴は `rid16, ban` で `base.parquet` に結合できる形で出す
3. 比較は M1 (v6+市場) と「生の情報を入れた対照」の両方に対して行う
