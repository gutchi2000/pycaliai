# PyCaLiAI → Cowork 印 JSON スキーマ

PyCaLiAI モデルが出力する「印 + PL 確率 + メタ情報」の JSON フォーマット定義。
Cowork 側はこの JSON を受け取り、**馬券種・点数・予算配分・見送り判断** を決定する。

役割分担:
- **PyCaLiAI**: 印付け (◎〇▲△△) と確率算出。馬券構築は **しない**
- **Cowork**: JSON を読んで race ごとに馬券を組み立てる。レース見送りも cowork 判断

---

## ルートオブジェクト

```json
{
  "race_id":          "2025122806050811",
  "race_meta":        { ... },
  "horses":           [ { ... }, ... ],
  "race_confidence":  { ... },
  "buy_judgment":     { ... }
}
```

| フィールド | 型 | 説明 |
|----------|-----|------|
| `race_id` | string | 16 桁 race_id (馬番除く) |
| `race_meta` | object | レース基本情報 |
| `horses` | array | 出走馬の配列 (umaban 昇順) |
| `race_confidence` | object | レース全体の信頼度メタ |
| `buy_judgment` | object | 買い方判定 (堅さ×妙味→買い方 + 妙味馬リスト)。`betting_judgment.py` が計算 |

---

## `race_meta`

```json
{
  "date":       "20251228",
  "place":      "中山",
  "course":     "芝2500",
  "field_size": 16,
  "class":      "Ｇ１",
  "race_name":  "有馬記念G1"
}
```

| フィールド | 型 | 説明 |
|----------|-----|------|
| `date` | string | YYYYMMDD |
| `place` | string | 競馬場 (中山, 東京, 京都...) |
| `course` | string | 芝/ダ + 距離 (例: "芝2500", "ダ1800") |
| `field_size` | int | 出走頭数 |
| `class` | string | クラス (G1, G2, 3勝クラス...) |
| `race_name` | string | レース名 |

---

## `horses` (配列)

各要素 = 1 馬。**umaban 昇順** で並ぶ。

```json
{
  "umaban":         4,
  "horse_name":     "ミュージアムマイル",
  "mark":           "◎",
  "ai_rank":        1,
  "ai_score":       1.6022,
  "p_win":          0.2184,
  "p_plc":          0.3349,
  "p_sho":          0.4618,
  "tansho_odds":    3.8,
  "fuku_odds_low":  1.4,
  "fuku_odds_high": 2.0,
  "ai_vs_market":   "fair"
}
```

| フィールド | 型 | 説明 |
|----------|-----|------|
| `umaban` | int | 馬番 |
| `horse_name` | string | 馬名 |
| `mark` | string | 印 (`◎`/`〇`/`▲`/`△`/`""`)。**top-5 のみ印あり** |
| `ai_rank` | int | AI 順位 (1=最上位)。1〜18 |
| `ai_score` | float | LightGBM raw score (大きいほど上位) |
| `p_win` | float \| null | PL 1着確率 (キャリブレーター適用済) |
| `p_plc` | float \| null | PL 連対率 (1-2着) |
| `p_sho` | float \| null | PL 複勝率 (1-3着, キャリブレーター適用済) |
| `tansho_odds` | float \| null | 単勝オッズ (発売時刻直前のスナップショット) |
| `fuku_odds_low` | float \| null | 複勝オッズ下限 |
| `fuku_odds_high` | float \| null | 複勝オッズ上限 |
| `ai_vs_market` | string | `"under"` / `"fair"` / `"over"` / `"unknown"` |
| `why` | array \| 省略 | SHAP による印の根拠 (top-k 特徴寄与)。**印馬のみ**付与 (下記) |

### `why` (印の根拠 / SHAP 寄与)

`export_weekly_marks.py` が `--shap-topk K` (default 6, `0` で無効) のとき、
**印が付いた馬 (◎〇▲△△) のみ** に付与する。`marks_shap.py` が生成。
ai_score を `shap.TreeExplainer` で特徴ごとに分解し、`|contrib|` 降順で top-K。

```json
"why": [
  {"feat": "prev_hosei",   "label": "前走 補正タイム",        "value": 99,    "contrib": 0.285},
  {"feat": "jockey_fuku90","label": "騎手 複勝率(直近90日)", "value": 0.389, "contrib": 0.258},
  {"feat": "前距離",        "label": "前距離",                "value": 1200,  "contrib": -0.199}
]
```

| サブフィールド | 型 | 説明 |
|----------|-----|------|
| `feat` | string | モデルの正準特徴名 (ground truth) |
| `label` | string | 日本語表示ラベル (`FEAT_LABELS`、未登録は `feat` と同じ) |
| `value` | number \| string \| null | その馬の **エンコード前の実値** (欠損は null) |
| `contrib` | float | 符号付き SHAP 寄与。**正=ai_score を押し上げ (印に効く) / 負=押し下げ** |

- ベースライン: `shap.expected_value`(v6 ≈ -1.02)。**全120特徴**の `Σ contrib + base = ai_score` (恒等)。`why` はそのうち寄与の大きい top-K のみ抜粋
- **相関ベースの寄与であり因果ではない**。Cowork は narrative の裏取り用に使い、買い目判断は確率を主軸にする
- 無効化: `python export_weekly_marks.py --csv ... --model v6 --shap-topk 0`

### 印の規則

| 印 | 意味 | ai_rank |
|---|-----|---------|
| `◎` | 本命 | 1 |
| `〇` | 対抗 | 2 |
| `▲` | 単穴 | 3 |
| `△` | 連下 | 4, 5 (2 頭とも `△`) |
| `""` | 印なし | 6 以下 |

### `ai_vs_market` 判定ロジック

`market_p = 1 / tansho_odds` (控除率は無視した implied probability)

| 条件 | ラベル | 解釈 |
|------|--------|------|
| `p_win >= market_p × 1.20` | `under` | 市場が低評価 / AI が高評価 (穴候補) |
| `market_p × 0.80 < p_win < market_p × 1.20` | `fair` | AI と市場が概ね一致 |
| `p_win <= market_p × 0.80` | `over` | 市場が高評価 / AI が低評価 (人気崩壊候補) |
| `tansho_odds` が無い | `unknown` | オッズ未取得 |

### 確率の制約

- `p_win`: race 内で `Σ p_win = 1.0` (PL 厳密)
- `p_sho`: race 内で `Σ p_sho = 3.0` (top-3 が 3 席ある分)
- `p_plc`: race 内で `Σ p_plc = 2.0` (top-2 が 2 席ある分)

---

## `race_confidence`

レース全体に対するメタ指標。Cowork が「このレースに乗るか」を判断する材料。

```json
{
  "top1_dominance":      0.0457,
  "top2_concentration":  0.391,
  "field_chaos_score":   0.8442,
  "ai_market_agreement": 0.6613
}
```

| フィールド | 型 | 範囲 | 説明 |
|----------|-----|------|------|
| `top1_dominance` | float | 0〜1 | `p_win[◎] - p_win[〇]`。**大 = ◎ の独走、小 = 混戦** |
| `top2_concentration` | float | 0〜1 | `p_win[◎] + p_win[〇]`。**大 = 上位 2 頭で決まりそう** |
| `field_chaos_score` | float | 0〜1 | エントロピー / 最大エントロピー。**大 = カオス、小 = 固い** |
| `ai_market_agreement` | float \| null | -1〜1 | AI 順位と市場 (オッズ) 順位の Spearman 相関。**大 = 一致、小 = 乖離** |

### Cowork 側の典型的判断ルール

```
if race_confidence.top1_dominance >= 0.10 \
       and race_confidence.field_chaos_score < 0.70:
    # 固いレース → 単勝◎ + 複勝◎ + 馬連◎-〇
elif race_confidence.field_chaos_score >= 0.85:
    # 混戦 → 馬連◎〇▲△△ ボックス OR 見送り
else:
    # 中間 → 複勝◎ のみ など慎重戦略
```

これは一例。Cowork 側で自由にロジックを組める。

---

## `buy_judgment`

```json
{
  "hardness": "固い",
  "chaos_pct": 0.226,
  "has_value": true,
  "headline": "妙味本線（絞って厚く）",
  "category": "go",
  "detail": "妙味馬を本線に。固いレースなので少点数で厚く。",
  "kenshu_hint": "単勝/複勝/馬連で少点数（絞って厚く）",
  "waku_tag": "妙味枠",
  "value_horses": [
    {"umaban": 3, "horse_name": "...", "p_win": 0.173,
     "ev_tan": 16.78, "ev_fuku": 4.86,
     "tan_value": true, "fuku_value": true, "sides": ["単勝", "複勝"]}
  ]
}
```

| フィールド | 型 | 説明 |
|----------|-----|------|
| `hardness` | string | `固い` / `標準` / `荒れ` (混戦度パーセンタイルで判定) |
| `chaos_pct` | float | 混戦度パーセンタイル (0-1) |
| `has_value` | bool | 妙味馬が 1 頭以上いるか |
| `headline` / `detail` | string | 買い方の結論文 |
| `category` | string | `go`/`caution`/`avoid`/`danger` (NiceGUI 配色用) |
| `kenshu_hint` | string | 券種の方向ヒント (拘束ではない) |
| `waku_tag` | string \| null | `妙味枠` / `参加枠` / null (回顧用タグ) |
| `value_horses` | array | 妙味馬 (買い候補)。EV 降順 |

判定ロジック・閾値は `betting_judgment.py` に集約 (NiceGUI 表示と bundle 埋込で共有)。
妙味馬の定義: 「単勝妙味=割安 or 複勝妙味=割安」かつ「該当側 EV>=1.10」かつ「勝率>=5%」。

---

## バッチ出力形式

`export_marks_json.py --year 2025 --out-dir reports/marks_v5/` で各 race を 1 file ずつ出力:

```
reports/marks_v5/
  2025010505010101.json
  2025010505010102.json
  ...
```

ファイル名 = `{race_id}.json` (16 桁)

---

## 利用例

### 単一 race を表示
```bash
python export_marks_json.py --model v5 --race-id 2025122806050811
```

### 期間指定でバッチ出力
```bash
python export_marks_json.py --model v5 --year-from 2024 --year-to 2025 \
    --out-dir reports/marks_v5/
```

### Cowork 側 (例)
```python
import json, glob

for path in glob.glob("reports/marks_v5/2025*.json"):
    with open(path, encoding="utf-8") as f:
        race = json.load(f)
    if race["race_confidence"]["field_chaos_score"] > 0.85:
        print(f"見送り: {race['race_id']} (chaos)")
        continue
    hon = next(h for h in race["horses"] if h["mark"] == "◎")
    if hon["p_sho"] > 0.55 and hon["tansho_odds"] <= 2.0:
        print(f"複勝◎ 100円: {race['race_id']} ({hon['horse_name']})")
```

---

## バージョニング

- `schema_version`: 現在 v1 (将来追加予定)
- **採用モデル**: v5 (unified_rank_v5.pkl, payout-weighted, alpha=1.325, 2026-04-27 採用)
- 後方互換性: フィールド追加は OK、削除/型変更は major bump

---

## v5 モデル品質 (真OOS 2024-2025, R=6,878)

| 指標 | 値 | v4 比 |
|------|----|------|
| NDCG@5 | 0.6027 | +0.024 |
| ◎ 1着率 | 30.28% | +2.89pt |
| ◎ 連対率 | 49.04% | +3.23pt |
| ◎ 複勝圏率 | 61.66% | +3.05pt |
| 1着∈top-3 | 60.88% | +2.14pt |
| 1着∈top-5 | 78.03% | +2.62pt |
| {1,2}⊂top-5 | 54.32% | +3.33pt |
| ECE 単勝(◎) | 0.0186 | ≈tie |
| ECE 複勝(◎) | 0.0173 | -0.002 (改善) |
| ECE 馬連(◎-〇) | 0.0218 | +0.006 (劣化、絶対値は実用範囲) |

**学習の特徴**:
- `clip(6 - 着順, 0, 5)` の連続ラベル (top-5 まで序列化)
- sample_weight = `1 + alpha * log1p(tansho_pay/100)` (穴馬好走を重視)
- LambdaRank with truncation_level=5
- Optuna 30 trials, 5-fold valid (race split), seed=42
- 厳密性: deterministic=True, force_col_wise=True, feature_pre_filter=False

---

## 既知の制約

1. **オッズの取得タイミング**: 出力時に `tansho_odds` 等は「発走 5 分前スナップショット」が入る。リアルタイム反映には別ジョブが必要
2. **オッズが無いレース**: `tansho_odds: null` になる。`ai_vs_market: "unknown"` として cowork 側で扱う
3. **`p_plc` のキャリブレータ**: 現在は raw PL 値 (calibrator 未適用)。`p_win` / `p_sho` は適用済
4. **`ai_market_agreement`**: 出走馬の少なくとも 3 頭にオッズが揃わないと `null`
5. **ECE の解釈**: `p_win` / `p_sho` は isotonic キャリブレータ適用後の値で、ECE 0.018-0.022 程度。`ai_vs_market` 判定 (1.20× / 0.80× 閾値) は実用十分な精度

---

## pair_probs (2026-06-11 追加)

印5頭の C(5,2)=10 ペアについて、**calibrated PL joint 確率**を race 直下に埋め込む。
compute_bets.py のワイド/馬連/馬単 EV 計算が独立積近似 (+21〜27% 系統過大) を
していた問題の解消用。キーは umaren_matrix と同じ「a-b」(a<b の馬番)。

```json
"pair_probs": {
  "5-9": {
    "wide":   0.09801,
    "umaren": 0.03768,
    "umatan": {"9→5": 0.01883, "5→9": 0.01844}
  }
}
```

- `wide` / `umaren`: 無向ペア確率 (calibrated)
- `umatan`: 方向つき (「i→j」= i が1着・j が2着)
- calibrator は本番ラインでは serve 条件 fit 版 (pl_calibrators_v6_serve.pkl) を使用
- pair_probs の無い旧 bundle に対して compute_bets は従来近似へフォールバックする
