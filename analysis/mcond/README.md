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

## 実験インデックス（EXP01〜EXP18、2026-09-26 現在）

各実験の結論は**その実験の範囲に限定**される。一般化の可否は各 `spec.json` / REPORT / memory を参照。

| 実験 | 仮説 | 状態 |
|---|---|---|
| EXP01 `exp01_choice_dev` | 陣営選択の逸脱 | 終了（固有価値なし・経済価値なし） |
| EXP02 | 動的対戦能力 | 終了（市場込みで主判定 PASS だが利益差なし。効果の多くは出走回数・休養日数） |
| EXP03 | 恒常能力/短期状態の分離 | 終了（Gate2 主判定 FAIL） |
| EXP04 | 環境不変情報での選別 | 終了（フィルタが有害） |
| EXP05 | 市場残差（判断時点） | 終了（logloss 改善は頑健に実在、ROI 非有意）／EXP05-F は前向き観測を継続 |
| EXP06 | Jev / direct decision | 終了（3 分類すべて FAIL） |
| EXP07 | ロバストポートフォリオ | 終了（Gate1 FAIL） |
| EXP08 | 同日トラック状態 | 終了（Gate2A/2B FAIL） |
| EXP09 | Conformal abstention | 終了（abstention 固有の価値なし） |
| EXP10 | downside risk | 終了（結論範囲を厳密限定） |
| EXP11 | 階層ベイズ | 終了（母集団選択バイアスで仮説撤回） |
| EXP12 | 対戦相手ネットワーク | 終了（placebo 決定的 FAIL） |
| EXP13 | 出走後の中止・非完走リスク | 終了（Gate0 でデータ不足。性能 FAIL ではない） |
| EXP14 `exp14_*` | レジーム分離 / MoE | Stage 0 完了・Stage 1 未着手 |
| EXP15 `exp15_race_as_set_dev` | Race-as-a-Set（同一レース内 interaction） | 終了（Context Gate FAIL） |
| **EXP16A** `exp16a_close_market_residual_dev` | 締切市場に対する表特徴の残差情報 | **Stage 1 完了。Gate A FAIL / Gate B NOT_APPLICABLE**（[REPORT](exp16a_close_market_residual_dev/REPORT.md)） |
| EXP16 | Ticket Candidate Generation | **現行確率源を使う経路は終了**（EXP16A §11。他券種・共同分布・新規情報源・局所的市場非効率へは一般化しない） |
| EXP17 `exp17_transitive_pl_graph_dev` | 共通対戦馬 2-hop → Hodge 射影 → PL | Stage 0 で終了（E0 FAIL・被覆床 FAIL）。閉じたのは EXP12 の 1-hop 表現と、時点安全な共通対戦 2-hop 表現＋Hodge スカラー射影。curl 成分 / identity pair coupling を直接 race 確率へ運ぶ方式は未実装・未検証（ただし使用辺の 84.7% が対戦 1 回で現データでは新実験の根拠不足） |
| **EXP18** `exp18_cross_pool_market_tomography_dev` | 複数プール市場トモグラフィー（較正 terminal 馬連市場への単勝 T1 offset） | **Stage 1 で終了（Gate M1 FAIL: LOO で 2021・2022 除外時に CI 上限 > 0）**。T2 は複勝往復 Gate FAIL で未実装。pooled Δ −0.0014 は実務床 0.0093 の約 1/6（[REPORT](exp18_cross_pool_market_tomography_dev/REPORT.md)） |

EXP16A の副次結果: 事前固定 8 残差特徴に terminal close 市場に対する再現性のある subfloor signal
（Δ −0.0036、5/5 年・5/5 seed、placebo 超過）が残ったが**実務床 0.005 未満**のため EXP16B として追試しない。

## 次の実験を足すとき
1. `analysis/mcond/expNN_*/` を作り、`spec.json` に仮説・split・特徴・モデル・合否条件を書いて**評価前にコミット**
2. 特徴は `rid16, ban` で `base.parquet` に結合できる形で出す
3. 比較は M1 (v6+市場) と「生の情報を入れた対照」の両方に対して行う
