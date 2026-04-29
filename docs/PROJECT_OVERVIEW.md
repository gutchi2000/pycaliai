# PyCaLiAI プロジェクト全体概要

> 個人運用の競馬予測 AI システム。**LightGBM v5 ランキングモデル** + **Cowork (Anthropic Desktop App)** とのハイブリッド構成。
> JRA 中央競馬の週末開催を対象、予算 1R = ¥10,000 で運用。

---

## 1. プロジェクトの目的とコンセプト

### 1-A. 何を解こうとしているか

**「中央競馬で年間 ROI を 100% に近づける、または超える」** という目的。
具体的には:
- 週末 (土日) の開催で 35〜70 R を対象
- 1R あたり ¥10,000 を投じ、1 週間で ¥350,000〜700,000 を運用
- ROI ≥ 90% (損失最小化) を最低ライン、~~110%~~ ≥ 100% (年間プラス) を目標

### 1-B. 役割分担: PyCaLiAI と Cowork

```
┌─────────────────────┐         ┌─────────────────────┐
│ PyCaLiAI            │         │ Cowork              │
│ (Python + LightGBM) │         │ (Anthropic Desktop) │
├─────────────────────┤         ├─────────────────────┤
│ ・出走表 → 印 (◎〇▲△)│  →JSON  │ ・印 + オッズを読む │
│ ・確率算出 (p_win 等) │  ───→   │ ・期待値計算        │
│ ・オッズ取得・整形    │         │ ・馬券構築 (買い目) │
│ ・JSON 出力          │         │ ・予算配分          │
└─────────────────────┘         │ ・見送り判断        │
                                │ ・JSON で返答       │
                                └──────────┬──────────┘
                                           │
                                  ┌────────▼────────┐
                                  │ Streamlit UI    │
                                  │ ・買い目を貼付  │
                                  │ ・解析・保存    │
                                  │ ・的中集計表示  │
                                  └─────────────────┘
```

**なぜハイブリッドか**:
- ML モデル: 客観的な確率算出は得意。でも「このオッズで攻めるべきか守るべきか」の柔軟判断は弱い。
- LLM (Cowork): 数値判断 + 文脈解釈 + 戦略選択が得意。でも統計モデルほどの精度は出せない。
- → 機械が確率出して LLM が判断、を分業させる。

---

## 2. システム全体像

### 2-A. ディレクトリ構造

```
E:\PyCaLiAI\
├─ data/
│   ├─ weekly/{YYYYMMDD}.csv          ... 週次出走表 (TARGET エクスポート、49/99 列)
│   ├─ odds/OD{YYMMDD}.CSV            ... TARGET 形式オッズ (227 列、単複馬連馬単ワイド三連系全部)
│   ├─ kekka/{YYYYMMDD}.csv           ... 着順 + 配当 (15 列)
│   ├─ hosei/H_*.csv                  ... 補正タイム (任意)
│   ├─ tyaku/{YYYYMMDD}.csv           ... 着度数 (任意)
│   ├─ kako5/{YYYYMMDD}.csv           ... 過去 5 走 (任意)
│   ├─ results.json                   ... LGBM v1 系の集計結果
│   ├─ cowork_results.json            ... Cowork 買い目の集計結果 (NEW)
│   ├─ live_results_2026.csv          ... 2026 年実績累積
│   ├─ pl_payout_curve_v5.pkl         ... v5 Kelly payout curve
│   └─ wide_payouts_2016-2025.parquet ... ワイド配当履歴 (~2025-07 まで)
│
├─ models/
│   ├─ unified_rank_v5.pkl            ... v5 LightGBM LambdaRank (採用)
│   ├─ unified_rank_v4.pkl            ... v4 (バックアップ)
│   ├─ pl_calibrators_v5.pkl          ... isotonic calibrator (単勝/複勝)
│   └─ ...
│
├─ reports/
│   ├─ cowork_input/{YYYYMMDD}_bundle.json   ... Cowork に投げる印 + オッズ JSON (~280 KB / 35R)
│   ├─ cowork_input/{YYYYMMDD}/{race_id}.json ... 個別 race 印 JSON
│   ├─ cowork_output/{YYYYMMDD}_bets.json    ... Cowork からの返答 (ユーザー保存)
│   ├─ cowork_bets/{YYYYMMDD}/{race_id}.json ... 解析後の保存買い目
│   └─ pred_{YYYYMMDD}.csv                   ... LGBM v1 系の自動買い目 (旧)
│
├─ docs/
│   ├─ cowork_prompt.md                ... Cowork 投入プロンプト (戦術カタログ込み)
│   ├─ marks_schema.md                 ... 印 JSON スキーマ仕様
│   ├─ operation_roadmap.md            ... 運用ロードマップ
│   └─ PROJECT_OVERVIEW.md             ... 本ファイル
│
├─ weekly_cowork.ps1                   ... 印 JSON 生成 + git push
├─ weekly_post.ps1                     ... 結果集計 + git push
├─ export_weekly_marks.py              ... CSV → 印 JSON
├─ export_marks_json.py                ... export_race の本体
├─ parse_od_csv.py                     ... TARGET OD CSV パーサ
├─ generate_results.py                 ... 集計 (results.json + cowork_results.json)
└─ app.py                              ... Streamlit UI
```

### 2-B. データフロー

```
[木〜金]  TARGET エクスポート
              ├─→ data/weekly/YYYYMMDD.csv      (出走表 49/99 列)
              ├─→ data/odds/OD{YYMMDD}.CSV     (オッズ 227 列、確定オッズ含む)
              └─→ data/kekka/YYYYMMDD.csv      (レース後、着順 + 配当)
                          ↓
[金夜]    .\weekly_cowork.ps1
              ├─ make_weekly_hosei.py (補正タイム生成、任意)
              ├─ export_weekly_marks.py
              │   ├─ predict_weekly.parse_csv で出走表 DF 化
              │   ├─ v5 model で PL ランキング → ◎〇▲△△ 付与
              │   ├─ p_win / p_plc / p_sho 算出 (PL 閉形式)
              │   ├─ isotonic calibrator 適用
              │   ├─ tansho_idx, fuku_idx, umaren_idx を OD CSV から構築
              │   ├─ ai_vs_market 判定 (under / fair / over / unknown)
              │   ├─ race_confidence 計算 (top1_dom / top2_conc / chaos / market_agree)
              │   └─ JSON 出力 (35 race ぶん bundle + 個別)
              └─ git add + commit + push (自動)
                          ↓
[金夜]    Cowork (Anthropic Desktop App)
              ├─ docs/cowork_prompt.md の本文をコピペ
              ├─ bundle.json を添付
              └─ JSON コードブロックで全 race の買い目 + 理由が返る
                          ↓
[金夜〜土朝]  Cowork 返答を reports/cowork_output/{date}_bets.json として保存
                          ↓
[土朝]    streamlit run app.py
              → メインタブ「🤖 Cowork取込」
              → ファイルから自動読込 (4 形式対応)
              → 🔍 解析プレビュー → 💾 全レース保存 + git push (自動)
                          ↓
              reports/cowork_bets/{date}/*.json (35 race ぶん)
                          ↓
[土朝〜]  「今日の買い目」タブ → 🤖 Cowork プラン
              → race ごとに買い目 + 根拠 + 理由 が表示される
                          ↓
[土・日]  JRA PAT で購入
                          ↓
[日夜]    .\weekly_post.ps1
              ├─ generate_results.py
              │   ├─ pred CSV × kekka 突合 (LGBM v1 系: HAHO/HALO/LALO/CQC/TRIPLE)
              │   └─ aggregate_cowork_bets() ← NEW: cowork_bets/ × kekka 突合
              ├─ update_live_results.py (live_results_2026.csv 更新)
              └─ git add + commit + push (kekka + results + cowork_bets)
                          ↓
              data/cowork_results.json 生成
                          ↓
              Streamlit「📊 的中実績」→「🤖 Cowork」タブで集計確認
```

---

## 3. データソース詳細

### 3-A. TARGET 出走表 CSV (`data/weekly/YYYYMMDD.csv`)

- 対象: 週末開催 全会場 全 race の出走馬
- 形式: cp932, 49 or 99 列
- 主要列: レースID(新/馬番無), 場所, R, 馬番, 馬名, 騎手, 厩舎, 単勝, 馬体重, 騎手成績 etc
- 1 ファイル ~80 KB / 700 馬 / 35 race
- エクスポート: TARGET frontier の「投票用 CSV」機能

### 3-B. TARGET OD オッズ CSV (`data/odds/OD{YYMMDD}.CSV`)

- 対象: 同日全 race の **発走前オッズ全種** (確定値)
- 形式: cp932, **227 列**, 馬番ごと 1 行
- カラム配置 (リバースエンジニアリング済):
  ```
  col 0     : レースID (10 桁)  ※"PP YY KD RR HH" PP=場 YY=年 KD=回日 RR=R HH=馬番
  col 1     : 出走頭数
  col 4     : 馬番
  col 6     : 馬名
  col 7     : 単勝オッズ
  col 8-9   : 複勝オッズ下限・上限
  col 10-27 : 馬連 18 slots (this row vs j=1..18, 対称)  ✓ 検証済
  col 28-38 : 枠連 + 発走時刻
  col 39-90 : ワイド + 馬単 (頭数によってオフセット変動 → 未対応)
  col 91+   : 三連複 / 三連単 (未対応)
  ```
- **馬連オッズは確定値で取れる** → 馬連 EV を Cowork で正確に計算可能
- 単勝・複勝も TARGET スナップショット (発走 5 分前相当)
- 出力例: `OD260426.CSV` = 953 KB

### 3-C. 着順 CSV (`data/kekka/YYYYMMDD.csv`)

- レース後の確定データ
- 15 列: 日付 / 場所 / R / 枠番 / 馬番 / 馬名 / 確定着順 / レースID(新) / 単勝配当 / 複勝配当 / 枠連 / 馬連 / 馬単 / ３連複 / ３連単
- **ワイド配当列が存在しない** → ワイド的中時の払戻が計算できない (既知の制限)

### 3-D. ワイド配当履歴 (`data/wide_payouts_2016-2025.parquet`)

- 過去 10 年分のワイド配当 (~34,548 race)
- 最新 2025-07-20 まで → **2025-08 以降は kekka に頼るしかない**
- バックテスト時のみ使用、リアルタイムには使えない

---

## 4. 機械学習モデル (v5)

### 4-A. v5 モデル概要

- **アルゴリズム**: LightGBM **LambdaRank** (`objective="lambdarank"`)
- **ラベル**: `clip(6 - 着順, 0, 5)` 連続化
- **sample_weight**: `1 + 1.325 * log1p(tansho_pay/100)` (穴馬好走を重視)
- **学習設定**:
  - `truncation_level=5`
  - `seed=42`, `deterministic=True`, `force_col_wise=True`
  - Optuna TPE 30 trials, 5-fold race split
- **真 OOS** (test 2024-2025, R=6,878):
  | 指標 | v5 | v4 | 差分 |
  |------|----|----|------|
  | NDCG@5 | **0.6027** | 0.5790 | +0.024 |
  | ◎ 1着率 | **30.28%** | 27.39% | +2.89pt |
  | ◎ 連対率 | **49.04%** | 45.81% | +3.23pt |
  | ◎ 複勝圏率 | **61.66%** | 58.61% | +3.05pt |
  | 1着∈top-3 | **60.88%** | 58.74% | +2.14pt |
  | ECE 単勝(◎) | 0.0186 | 0.0186 | tie |
  | ECE 複勝(◎) | **0.0173** | 0.0193 | -0.002 |

### 4-B. Plackett-Luce 確率変換

スコア → ランキング確率の閉形式:

```python
# weights w_i = exp(score_i)
w = np.exp(scores - scores.max())

# 単勝確率: w_i / Σw_j
p_win = w / w.sum()

# 連対率 (1着 + 2着): 各 j != i について
# P(j 1着) × P(i 2着 | j 1着) = w_j / total × w_i / (total - w_j)
p_plc[i] = p_win[i] + Σ_{j≠i} (w_j/total) × (w_i/(total - w_j))

# 複勝率 (top-3): 同様に j, k != i で 3 重ループ (vec 化済)
```

### 4-C. キャリブレーション

- `models/pl_calibrators_v5.pkl` に isotonic regressor を保存
- 単勝・複勝それぞれに独立 fit (validation set ベース)
- 推論時: `p_win = calibrators["tansho"].predict(p_win_raw)`

### 4-D. 印付与 (`mark_by_idx`)

PL スコア降順で:
- 1 位 → ◎ (本命)
- 2 位 → 〇 (対抗)
- 3 位 → ▲ (単穴)
- 4 位 → △ (連下)
- 5 位 → △ (押え)

---

## 5. レース信頼度メトリクス

### 5-A. race_confidence の 4 指標

各 race に以下を付与:

#### top1_dominance (0-1)
```
= p_win[◎] - p_win[〇]
```
**◎ がどれだけ独走しているか**。0.10 以上で本命安定。

#### top2_concentration (0-1)
```
= p_win[◎] + p_win[〇]
```
**上位 2 頭で決まる確率**。0.50 以上なら馬連 ◎-〇 が本線。

#### field_chaos_score (0-1)
```
= entropy(p_win) / log(N)
```
**確率分布のエントロピー / max entropy**。
- 小 (0.0-0.7): 上位明確 (固い)
- 中 (0.7-0.85): やや混戦
- 大 (0.85-1.0): 全馬団子 (混戦)
- 0.92+: 極度のカオス → 見送り条件

#### ai_market_agreement (-1〜1)
```
= Spearman(p_win 順位, 単勝オッズ昇順 = 人気順)
```
**AI と市場の見方の整合度**。
- +1: 完全一致 (AI = 人気)
- 0: 無関係
- -1: 真逆 (AI が穴推奨)

### 5-B. ai_vs_market (per horse)

各馬について AI 評価 vs 市場評価のラベル:
```python
# 内部スコア = p_win × (1 / tansho_odds)
# 市場スコア = 1 / tansho_odds
# 比率 = AI / 市場

if ratio > 1.3:    "under"  (AI > 市場、つまり市場が過小評価 = 穴狙い候補)
elif ratio < 0.7:  "over"   (AI < 市場、つまり市場が過大評価 = 危険)
elif tansho is None: "unknown"
else:              "fair"
```

---

## 6. Cowork (Anthropic Desktop App) 連携

### 6-A. 全体フロー

1. PyCaLiAI が `bundle.json` (~280 KB / 35 race) を生成
2. ユーザーが Cowork のチャットに **bundle.json + プロンプト** を投げる
3. Cowork が **JSON コードブロック先頭** で返答 (馬券種・買い目・購入額・理由)
4. ユーザーが返答全文を `reports/cowork_output/{date}_bets.json` として保存
5. Streamlit が自動読込 → 解析 → `cowork_bets/` に保存 → git push

### 6-B. プロンプト構造 (`docs/cowork_prompt.md`)

#### 0. 見送り判定 (4 つの厳格条件のみ)
```
1. field_chaos_score >= 0.92 (極度カオス)
2. field_size <= 7 (頭数不足)
3. tansho_odds(◎) is null (オッズ未取得)
4. p_win(◎) < 0.05 (本命確率低)
```
**これ以外は必ず ¥10,000 を投じる**。「妙味乏しい」等の主観で見送り禁止。

#### 1. レース性質判定 (deterministic)
```python
if ai_vs_market(◎) == "under" AND tansho_odds(◎) >= 5.0:
    → 穴推奨
elif top1_dominance >= 0.10 AND field_chaos < 0.70:
    → 固い
elif top1_dominance < 0.05 AND field_chaos >= 0.85:
    → 混戦
else:
    → 中堅 (デフォルト)
```

#### 2. 性質別 候補馬券
| 性質 | 候補馬券種 |
|------|-----------|
| 固い | 単勝(◎), 複勝(◎), 馬連(◎-〇) |
| 中堅 | 複勝, 馬連, ワイド(◎軸), 単勝も可 |
| 混戦 | 馬連 box (◎〇▲△ から 3-6 点), ワイド ◎流し, 単勝も可 |
| 穴推奨 | 単勝, 複勝, 馬連, ワイド(◎-〇) |

**馬単 / 三連複 / 三連単は対象外** (確定オッズ無いため)

#### 3. EV 計算式
```
単勝 EV = p_win(◎) × tansho_odds(◎)
複勝 EV = p_sho(◎) × (fuku_low + fuku_high) / 2
馬連 EV = umaren_matrix[i-j] × p_combined
  p_combined = p_win[i] × p_win[j]/(1-p_win[i]) + p_win[j] × p_win[i]/(1-p_win[j])
ワイド EV (推定) = (umaren_matrix[i-j] / 3) × (p_sho[i] × p_sho[j])
```

#### 4. 配分戦略 (Cowork が race ごとに自由選択)

**10 戦術カタログ** (例示、これに縛られない):

- **6-A 単勝集中**: ◎ EV 1.5+ 突出時、単勝 ¥7,000 等
- **6-B 馬連本線厚張り**: ◎-〇 EV 1.5+ 時、馬連 ¥6,000 + 複勝 ¥2,000 + ワイド ¥2,000
- **6-C ワイド多点拡散** (穴推奨向け): 軸 ¥3,000 + ワイド 4-5 点 ¥4,000-6,000 + 単勝 ¥1,000
- **6-D 二軸並列ヘッジ**: ◎/〇 力差小、単勝 ◎,〇 + 馬連 + ワイド分散
- **6-E 馬連流し+ワイド補強**: 相手 3 頭、馬連 流し 3 点 + ワイド 同 3 点
- **6-F 上位 box** (混戦): 馬連 box + ワイド box + 複勝
- **6-G 配当買い (◎-△重視)**: ◎-〇 オッズ低時、◎-▲, ◎-△ で組む
- **6-H 複勝集中** (低リスク): 複勝 ¥7,000 + ワイド ¥2,000 + 単勝 ¥1,000
- **6-I 〇軸切替**: ◎ 信頼薄時、〇 を実質本命に
- **6-J ▲ undervalued 拾い**: ▲ 市場過小評価時、▲ を準軸に格上げ

**節 10「自由度の宣言」**: カタログは例示。Cowork は LLM として自由に組み合わせて OK。
ただし配分の原則だけは守る:
- ¥10,000 を必ず使い切る
- 個別馬券 EV >= 0.85
- 1 点 ¥500-7,000、100 円単位
- race 全体期待 ROI >= 100% を狙う

### 6-C. Cowork 返答 JSON スキーマ

```json
[
  {
    "race_id": "2026042606010109",
    "race_label": "阪神芝1600 マイラーズC",
    "race_nature": "中堅",
    "race_reason": "本命◎は前走重賞2着で力上位、相手〇も同条件で安定。中堅構成として馬連を本線、ワイドで保険。(自然な日本語、数式変数名は出さない)",
    "bets": [
      {
        "馬券種": "馬連",
        "買い目": "5-11",
        "購入額": 5000,
        "理由": "◎と〇の力が抜けていて3着以下とは差がある。馬連オッズ8.5倍を本線として最厚で張る。"
      },
      ...
    ]
  },
  ...
]
```

**重要**: 数値・式・変数名 (top1_dom=0.082, EV=1.53 等) を表に出さず、競馬予想士の自然な口調で書く規約。

---

## 7. Streamlit UI (`app.py`)

### 7-A. メインタブ構成

```
🏇 レース予想       ← 各 race 詳細 + 自動買い目 (4 戦略タブ)
🎫 今日の買い目     ← STANDARD / Cowork 切替表示 (理由付き)
🎯 プラン選択       ← (旧) プラン選択 UI
⭐ EV候補          ← EV 高い馬リスト
💰 VALUE複勝       ← 複勝 EV 厚いやつ
🤖 Cowork取込      ← Cowork 返答を一括取込 (NEW)
📊 的中実績         ← 集計表示 (STANDARD / Cowork タブ)
📊 ROIヒートマップ
📋 結果フィードバック
```

### 7-B. レース詳細の det_tabs (race detail)

```
📋 STANDARD 単複馬連                  ← 自動 EV ベース買い目
🤖 Cowork (Anthropic Desktop)         ← Cowork 連携 (印 JSON 表示 + 個別入力)
```

(三連系プラン TRIPLE / HAHO / HALO は **scope 外として hide**、`_SHOW_TRIPLE_PLANS = True` で復活可)

### 7-C. 「🤖 Cowork取込」タブ (NEW)

**自動読込フロー**:
1. `reports/cowork_output/` を全スキャン
2. ファイル名から日付抽出 (`{YYYYMMDD}_bets.json` 等 6 パターン対応)
3. プルダウンで日付選択 (サイドバー日付一致がデフォルト)
4. 文字コード自動判定 (UTF-8 / UTF-8 BOM / cp932 / Shift-JIS)
5. 🔍 解析プレビューで内容確認 (race_reason / 理由 込み)
6. 💾 保存 + git push で `cowork_bets/` 保存 + cowork_output も同時 push

**フォールバック**: 折りたたみの「📋 直接貼り付け」expander あり

### 7-D. 「🎫 今日の買い目」 → 🤖 Cowork プラン

各 race に表示:
- 場所 + R + クラス + 距離 + 発走
- ◎馬名 + score
- 性質タグ (固い/中堅/混戦/穴推奨/見送り、色分け)
- 📝 race_reason (オレンジ枠)
- 馬券種ごとの買い目 + 💭 理由 (グレー差込)
- 馬券種小計 + 全体合計
- 馬場フィルタ (芝/ダート切替)

### 7-E. 「📊 的中実績」 → 🤖 Cowork タブ (NEW)

- 4 メトリクス: 対象R / 総投資 / 総払戻+収支 / ROI+的中率
- 馬券種別テーブル (単勝/複勝/馬連/ワイド)
- 会場別テーブル
- 個別レース expander (race_reason + 収支カラー)
- 個別馬券 expander (✅ hit マーク)

---

## 8. 集計ロジック (`generate_results.py`)

### 8-A. 旧系統 (LGBM v1 系)

5 プラン: HAHO / HALO / LALO / CQC / TRIPLE
- 各 plan の pred CSV × kekka を突合
- 馬券種ごとに hit / 払戻 / ROI を集計
- `data/results.json` に保存

### 8-B. Cowork 系 (NEW)

`aggregate_cowork_bets(kekka_cache)` 関数:
1. `reports/cowork_bets/{date}/*.json` を全スキャン
2. 各 race ごとに kekka を引いて `match_cowork_bet(bet, race_kk)` で照合
3. 馬券種別の hit / 払戻 / ROI 集計
4. race 単位 / 馬券単位 / 会場別 / 週次サマリ
5. `data/cowork_results.json` に保存

#### `match_cowork_bet` の詳細

| 馬券種 | 判定 | 配当ソース | 払戻 |
|--------|------|-----------|------|
| 単勝 | 1 着 = 買目 | `単勝配当` | ✅ |
| 複勝 | 買目 ∈ top 3 | `複勝配当` | ✅ |
| 馬連 | 買目 = top 2 | `馬連` | ✅ |
| 馬単 | 買目 = (1着, 2着) | `馬単` | ✅ |
| **ワイド** | 買目両馬 ∈ top 3 | **kekka に列無し** | ⚠️ ¥0 |
| 三連複 | 買目 = top 3 | `３連複` | ✅ |
| 三連単 | 買目 = (1着, 2着, 3着) | `３連単` | ✅ |

複数点 (`5-11,5-3` など) は均等分割して `payout / N` で計算。

### 8-C. NaN セーフ化

`pred CSV` 等に NaN が混じると `round()` が `ValueError` で死ぬ問題があったので:
```python
def _safe_num(v, default=0.0):
    """NaN/None/変換失敗を default に変換した float"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return default if pd.isna(f) else f

def _safe_round(v, default=0):
    """NaN セーフな round"""
    ...
```
全集計 12+ 箇所で `round(NaN)` → `_safe_round()` に置換済。

---

## 9. 週次運用フロー (操作手順)

### 9-A. 木〜金: 準備

```
1. TARGET から 3 ファイルを取得 (週次出走表 + OD オッズ + 着順 はレース後)
2. data/weekly/{YYYYMMDD}.csv に置く
3. data/odds/OD{YYMMDD}.CSV に置く
```

### 9-B. 金夜: 印 JSON 生成

```powershell
.\weekly_cowork.ps1
# 引数なしで最新の data/weekly/*.csv を自動検知
```

裏で:
1. `make_weekly_hosei.py` (補正タイム生成、任意)
2. `export_weekly_marks.py --csv ... --model v5`
   - OD CSV 自動検知 → 単勝・複勝・馬連 matrix を bundle に埋め込み
   - 99% 以上カバー率 (35 race × 平均 100 組 = ~3,556 馬連オッズ)
3. `git add` `commit` `pull --rebase` `push origin HEAD:master` (自動)

出力:
- `reports/cowork_input/{YYYYMMDD}_bundle.json` (~280 KB)
- `reports/cowork_input/{YYYYMMDD}/{race_id}.json` (個別 35 件)

### 9-C. 金夜: Cowork で買い目作成

```
1. Cowork (Anthropic Desktop App) で新規チャット
2. docs/cowork_prompt.md の「プロンプト本文」をコピペ
3. bundle.json を添付
4. 送信 → JSON コードブロックで返答が来る
5. 全選択コピー → reports/cowork_output/{YYYYMMDD}_bets.json として保存
```

### 9-D. 土朝: Streamlit で取込

```powershell
streamlit run app.py
# ブラウザ http://localhost:8501
```

```
1. メインタブ「🤖 Cowork取込」
2. 自動的に reports/cowork_output/ をスキャン → プルダウン表示
3. 該当日付を選択 (デフォルトでサイドバー日付に揃う)
4. 🔍 解析プレビュー
5. 💾 全レース保存 + git push
   → reports/cowork_bets/{date}/{race_id}.json (35 ファイル)
   → cowork_output/ も同時に push
```

### 9-E. 土朝〜土日: 買い目確認 + 購入

```
1. 「🎫 今日の買い目」→「🤖 Cowork」プラン
2. 各 race の race_reason + 理由を読みながら確認
3. JRA PAT で買い目通り購入
```

### 9-F. 日夜: 結果集計

```
1. data/kekka/{YYYYMMDD}.csv 配置 (TARGET エクスポート)
2. .\weekly_post.ps1
   ├─ generate_results.py (HAHO/HALO/LALO/CQC/TRIPLE + Cowork 集計)
   ├─ update_live_results.py
   └─ git push (kekka + results.json + cowork_results.json + cowork_bets)
3. 「📊 的中実績」→「🤖 Cowork」タブで集計確認
```

---

## 10. 直近の改善履歴 (このセッションでの変更)

### 10-A. v5 モデル統合 + Cowork 連携基盤
- v5 LightGBM (sample_weight + LambdaRank) で v4 から +3% 程度の改善
- bundle.json スキーマ確立 (race_meta + horses + race_confidence)

### 10-B. OD CSV 統合
- TARGET OD オッズファイル (227 列) のリバースエンジニアリング
- 馬連 matrix (cols 10-27) を確定値で取得
- 複勝下限/上限を取得 (旧フローでは null だった)
- Cowork が複勝・馬連の EV を正確に計算可能に

### 10-C. Cowork 一括取込 UI
- 旧: 35 race × 5 馬券 = 700+ 回手入力
- 新: ファイル保存 → Streamlit 自動読込 → 1 クリック保存 → 3 操作で完了
- 文字コード自動判定 (UTF-8 / cp932 / Shift-JIS)
- ファイル名: `{YYYYMMDD}_bets.json` 推奨、6 パターン対応
- 永続化: ブラウザリロードで消えない

### 10-D. 理由付き表示
- prompt 改修: race_reason (race 単位) + 理由 (bet 単位) を必須化
- 自然な日本語必須 (top1_dom=X.XX 等の技術表記禁止)
- Streamlit 全画面で理由を表示 (一括取込プレビュー / 今日の買い目 / 個別レース / 結果集計)

### 10-E. 戦術の自由化
- 旧: 性質別 固定配分 (単勝¥3,000 + 馬連¥5,000 + ...)
- 中: 3 戦略 (集中/分散/拡散) の選択
- **新**: 10 戦術カタログ + 「型を超えた組合せ自由」宣言
- EV 帯と金額の目安は感覚値、Cowork が race ごとに自由判断

### 10-F. 三連系プランを hide
- TRIPLE / HAHO / HALO は scope 外として UI から非表示
- 「🎫 今日の買い目」「🏇 レース予想」「📊 的中実績」3 箇所
- 復活は `_SHOW_TRIPLE_PLANS = True` で 1 行変更

### 10-G. 全会場対応
- `EXCLUDE_PLACES = {"東京", "小倉"}` → `set()` に変更
- Phase 6: 全 10 場 (札幌/函館/福島/新潟/東京/中山/中京/京都/阪神/小倉) 対象

### 10-H. Cowork 集計実装
- `aggregate_cowork_bets()` 関数で `cowork_bets/` × `kekka` 突合
- 7 馬券種対応 (ワイドのみ kekka 配当列無し → ¥0)
- `data/cowork_results.json` に保存
- Streamlit「📊 的中実績」→「🤖 Cowork」タブで表示

### 10-I. NaN セーフ化
- 旧: `float(hon.get(X, 0) or 0)` で NaN がそのまま流れる
- 新: `_safe_num()` / `_safe_round()` ヘルパーで全箇所セーフ化
- `weekly_post.ps1` クラッシュ問題を解消

### 10-J. 自動 git push 全工程化
- `weekly_cowork.ps1`: 印 JSON + cowork_output も自動 push
- Streamlit 「💾 保存 + git push」: cowork_bets/ + cowork_output/ 両方同時 add
- `weekly_post.ps1`: kekka + results + cowork_results + cowork_bets 一括 push
- → Streamlit Cloud (https://pycaliai.streamlit.app) で出先からも閲覧可能

---

## 11. 既知の制限・課題

### 11-A. ワイド配当が kekka に無い
- TARGET 着順 CSV (`data/kekka/`) には ワイド配当列が存在しない
- ワイド的中時の払戻が ¥0 で集計される (ROI 過小評価)
- 過去 wide_payouts parquet は 2025-07 まで → 直近データに使えない
- **暫定**: 的中フラグだけ立つ、payout は別途手動確認

### 11-B. 馬単・三連系のオッズ未対応
- OD CSV で馬単・三連複・三連単のカラム配置が頭数によって変わる
- リバースエンジニアリング途中で「row 1 col 59-60 ≠ row 12 col 39-40 (同 pair)」の不整合発覚 → 提供せず
- Cowork は馬連オッズから推定計算で対応 (誤差大)

### 11-C. Cowork のオッズ更新タイミング
- bundle.json は「TARGET エクスポート時点のスナップショット」
- 発走直前のオッズ変動には未追従
- ユーザー側で発走 30 分前に再エクスポート + `weekly_cowork.ps1` 再実行が必要

### 11-D. strategy_weights.json の限定性
- LGBM v1 系 (HAHO/HALO/LALO/CQC) は 4 場 (中京/中山/新潟/福島) のみ対象
- 東京・小倉・京都・阪神・札幌・函館では HAHO 等が生成されない
- ただし Cowork プランは strategy_weights 不要 (印 + EV ベース)

### 11-E. Cowork 利用にコスト
- Anthropic Desktop App (= 月額有料サブスク)
- 週次 35 race × プロンプト ~5K tokens + bundle.json ~280 KB
- ローカル LLM (Ollama 等) では bundle.json サイズ的に厳しい

### 11-F. リアルタイム性
- Cowork は静的スナップショットでしか判断できない
- スクレイピングや API 経由のリアルタイムオッズ取得は未実装
- 「発走 5 分前まで全 race のオッズ更新 + 再判断」は人間操作で対応

### 11-G. アーカイブ・履歴管理
- pred CSV や cowork_bets が累積していく → 古いデータの剪定なし
- 月次再学習 (`retrain_value_model.py`) はあるが、データクリーニングは別途必要

---

## 12. これから検討したいこと (他 AI に意見を求めたい点)

### 12-A. モデル改善
- v5 が真 OOS で NDCG@5=0.6 → これ以上の伸び代は?
- ロード長/距離別/コース別/季節別の特徴量で seg-specific モデルを試すべきか?
- Calibration の温度調整 (Platt scaling vs isotonic) は適切か?
- BNN や ensemble (LGBM + CatBoost + GBM) で gain あるか?

### 12-B. オッズ統合の高度化
- 直前オッズ変動 (発走 5 分前 vs 5 時間前) を特徴量化できないか?
- 「人気が下がったが AI 評価維持」「人気上昇したが AI 評価下落」のシグナル検知
- 馬連 / ワイド / 馬単オッズの整合性チェック (アービトラージ的判定)

### 12-C. Cowork (LLM) 側の改善
- プロンプトをさらに改善する余地は?
- 戦術カタログをもっと増やすべきか? それとも principles だけにすべきか?
- few-shot examples を bundle.json に同梱するべきか?
- 過去の判断ログをフィードバックして「自己改善」させる仕組み?
- Cowork の回答を別の Cowork (二段判断) に通すアンサンブル?

### 12-D. UI/UX 改善
- 「🎫 今日の買い目」の race_reason / 理由表示は読みやすいか?
- 馬場フィルタ + 場所フィルタ + 性質フィルタの追加?
- 買い目を JRA PAT 形式 (CSV) でエクスポートする機能?
- 投票実績との突合 (PAT 履歴 import → 集計)?

### 12-E. 集計・可視化
- ROI のキャッシュフロー表示 (週次の累積)
- 馬券種別 × 性質別の交差表 (どの組み合わせが効いてるか)
- ベイズ更新ベースの ROI 期待値推定 (サンプル少ない時の補正)
- 機械学習で「Cowork のどの判断が良いか/悪いか」を学習

### 12-F. データ拡張
- 父・母父 (種牡馬) 別の傾向
- 騎手・厩舎の最近 N R での好調度
- 馬場状態 (含水率, クッション値) の特徴量化
- 配合理論 / ニックス理論の自動評価

### 12-G. リスク管理
- ケリー基準 (`f* = (bp - q) / b`) ベースの予算配分
- 連敗時のドローダウン抑制 (動的予算調整)
- 月次の総予算上限設定 (loss cap)

### 12-H. 自動化
- TARGET CSV エクスポートの自動化 (現状は手動)
- 着順 CSV 取得の自動化
- JRA PAT への自動投票 (規約的に微妙だが技術的可能性)
- 週次レポートの自動生成 + メール送信

### 12-I. その他
- ローカル LLM (Llama 3 / Mistral) で Cowork 役を代替できるか
- スマホ App 化 (Streamlit → React Native?)
- 共有用 (家族・友人) のマルチユーザ対応

---

## 13. 技術スタック

| 領域 | 採用 |
|------|------|
| 言語 | Python 3.14 |
| ML フレームワーク | LightGBM, scikit-learn (calibration) |
| データ処理 | pandas, numpy, joblib |
| UI | Streamlit |
| LLM 連携 | Anthropic Cowork (Desktop App), 手動 JSON 経由 |
| バージョン管理 | Git, GitHub (https://github.com/gutchi2000/pycaliai) |
| デプロイ | Streamlit Cloud (https://pycaliai.streamlit.app) |
| 開発支援 | Claude Code (Anthropic) |
| 出走表データ | TARGET frontier (有料、JRA-VAN) |
| オッズデータ | TARGET OD CSV |
| OS | Windows 11 (PowerShell 5.1) |

---

## 14. 連絡 (このドキュメントの目的)

このドキュメントは、開発者が **他の生成 AI に PyCaLiAI の現状を共有して改善案を聞く** ために作成。
具体的に求めているフィードバック:

1. 上記「12. これから検討したいこと」のリストへの優先順位提案
2. 見落としている改善ポイント
3. ML / オッズ統合 / UI / リスク管理 のいずれかに関する具体的アイデア
4. 競馬予測システムとして「これは絶対やるべき」という業界標準の視点
5. ROI を改善するための統計的アプローチの提案

---

**生成日**: 2026-04-30  
**ドキュメント版**: v1.0  
**コミットハッシュ**: 22e3b9e5a (本ファイル追加時点)
