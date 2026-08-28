# PyCaLiAI 完全仕様書 Vol. II — 馬券構築・運用仕様

> 版 1.0 / 2026-08-23 / 実測ベース
> 対象: 意思決定層（L6）と運用（ops）
> 前提知識: [Vol. I](VOL1_SYSTEM.md) §1（パリミュチュエル）と §8（印層）

---

## 目次

- §1 馬券構築の設計原理（なぜこうなっているか）
- §2 `compute_bets.py` 完全仕様
- §3 ガード群（fail-safe / fail-closed の全体像）
- §4 決済ドリフト補正（SettleAI）
- §5 枠プラン（`build_bet_plan.py`）
- §6 当日 T-10 ライン
- §7 Cowork narrative 契約
- §8 週次運用フロー（3 フェーズ）
- §9 決済・集計（`generate_results.py`）
- §10 公開層（静的サイト / TACT / note / X）
- §11 実運用実績（実測値）

---

## §1 馬券構築の設計原理

### 1.1 なぜ「印から馬券を組む」のをやめたか

初期設計は `印スロット → 券種テンプレート` だった（現 `CB_ENGINE=shape`）。
これには 2 つの構造的欠陥があった。

1. **印は lossy な中間表現**。120 特徴 → raw score → PL 確率 → **上位 5 頭のラベル**、
   と情報を落としたところから馬券を組んでいた。6 位以下の馬は確率が高くても買えない。
2. **印がモデルの目的関数に侵入している**（`composite` に `◎top3` 等が入る、Vol. I §5.2）。
   つまり「印の当たりやすさ」に最適化されたモデルの印で馬券を組む二重の縛り。

2026-08-09、**完全トップダウンエンジン**に置換（既定化）。
印・shape・妙味ヒューリスティクスをすべてバイパスし、
**全馬の `p_win` → λ補正 PL → 全ペア確率 → 確率順候補 → p 比例配分 → 適応トリガミ床**
という単一の連続的な流れにした。

### 1.2 prob-first（EV による銘柄選抜の廃止）

**Evidence（`project_ev_selection_harmful_probfirst`、2026-06-15 監査）**:
| 選抜方式 | test ROI | CLV |
|---|---:|---|
| EV（価格ズレ）で馬連/ワイドのペアを選ぶ | 66.6% | −5.8% |
| **calibrated `p_pair` 上位 top-K で選ぶ** | **79.0%** | + |

**EV 選抜は毎回 13pt を捨てる。** これは optimizer's curse（モデルと市場が最も
食い違う銘柄＝モデルが最も間違っている銘柄）の典型。
したがって **EV は「選抜」から降格し、「フロア」と「配分の重み」にのみ使う**。

さらに 2026-08-09、**EV による配分（サイジング）も撤去**した:
- 実測: レース内で最も厚く張った点（rank1）の ROI が最悪（62.4% 直近 / 73.5% 全期間）で rank2-3 を下回る
- 「EV で厚くする = 市場乖離に厚くする = optimizer's curse」
- → `CB_ALLOC=p`（p × boost 比例）を既定に。`flat` / `ev`（旧挙動）も env で選べる
- リプレイ A/B（4/18–8/9, 506R 同条件）: **ev 74.1% < flat 76.4% < p 78.2%**（全 5 ヶ月で ev 比プラス）

**⚠️ 規律（2026-06-11 以降）**: boost 係数を点推定でいじるのは**禁止**。
`cowork_results.json` の `roi_verdict` が `above_takeout` / `below_takeout` になったときのみ変更可。
（単勝 30bets ROI 120.8% → 445bets で 96.3% に回帰した事故の構造対策）

### 1.3 適応トリガミ床（ユーザー発、ML が dominate できなかった唯一の改善）

**トリガミ** = 的中したのに払戻 < 総投資。
ユーザーのルール:「**最安組の払戻が総投資を上回るまで、点数を削る**」。

**検証（`analysis/trigami_floor_gate.py`、v6 / 6,878R OOS）**:
| 版 | 結果 |
|---|---|
| 適応点数版（低 p 点を削る） | トリガミ **−74%** / クリーン勝ち **+18%** / ROI 75→77（控除壁で不変） |
| skip 版（床を割るレースを見送る） | **罠**（参加 80% 減・ROI 69%） |

→ **正解は「見送り」ではなく「点数削り」**。

**機械ポリシー総当たり（6 家系 / OOS / paired-bootstrap 敵対検証）でも
このルールを dominate できなかった** — パレートフロンティア上にある。
唯一の頑健な上乗せは **床 + chalk-cap**（chaos < 0.86 → 12 点）でトリガミ −3pt（ROI は互角）。

実装は `compute_bets.py:525-530`:
```python
amts = allocate([c[2] for c in tds], budget=int(budget))
for _ in range(len(tds) - 1):
    if min(c[4] * a for c, a in zip(tds, amts)) >= sum(amts):
        break                                    # 最安見込払戻 ≥ 総投資 → OK
    tds.pop(min(range(len(tds)), key=lambda i: tds[i][2]))   # 最低 p の点を削る
    amts = allocate([c[2] for c in tds], budget=int(budget))
```

### 1.4 禁止事項（ユーザー定義、コードで強制済み）

| 禁止 | 実装箇所 | 根拠 |
|---|---|---|
| **馬単** | `compute_bets` の候補生成から撤去（2026-06-18） | 実測 ROI 22.1%（n=122）/ 2 of 122 的中 = 構造的回収不能 |
| **三連単** | 生成しない + `validate_cowork_bets.REJECTED_KINDS` | 控除率 27.5% |
| **穴推奨の馬連** | `ODDS_CAP["馬連"] = 50.0` | 高オッズ馬連 = 穴を遮断 |
| **高 EV だけを理由にした馬連** | prob-first 化（EV フロアを課さない） | §1.2 |

### 1.5 3 枠方針（ユーザー定義の資金規律）

| 枠 | 1R 予算 | 1 日の本数 |
|---|---:|---:|
| 勝負 | ¥10,000 固定 | 2–3R |
| 準勝負 | ¥5,000–8,000（信頼度でスケール） | 2–4R |
| 消化 | ¥1,000–3,000 | floor 充足まで |

**floor**: 週 10R ∧ ¥100,000（1 日換算 5R ∧ ¥50,000）。
消化枠は **ROI を下げることを承知の上**での割り切り（コンテンツ／網羅目的）。
実装は `build_bet_plan.py`（§5）。

---

## §2 `compute_bets.py` 完全仕様

**ファイル**: 1,008 行 / `ENGINE_VERSION = "2026-08-09"`
**定数**: `BUDGET=10000, MIN_BET=500, MAX_BET=7000`

### 2.1 入出力

**入力**
1. `bundle.json`（Vol. I §10）: `race_meta`, `race_confidence`, `horses[]`,
   `buy_judgment`, `umaren_matrix`, `pair_probs`
2. `reports/live_odds/{rid16}.json`（T-10、`jvlink_odds.py` 出力）— 指定時は**必須**
3. `data/chaos_quantiles.json` / `data/harville_lambda.json` / `data/t10_blend.json`
4. `reports/bet_plan/{date}.json`（`--plan` 指定時）

**出力**: `reports/cowork_output/{date}_bets.json` へ **in-place merge**
（`race_id` 一致は置換 / 無ければ追加。`.bak` 退避 + `.tmp` アトミック置換）

**★ 書込契約（データ消失防止、`compute_bets.py:855-859`）**
```python
old = races_list[i]
if isinstance(old, dict) and old.get("advisor") and not e.get("advisor"):
    e = {**e, "advisor": old["advisor"]}      # Cowork narrative を温存
```
**別ファイルへの書き出しは禁止**。同一ファイルの read-modify-write でなければ advisor が消える。

### 2.2 CLI

| フラグ | 意味 |
|---|---|
| `--bundle PATH` | 必須 |
| `--dry` / `--apply` | 表示のみ / 書込 |
| `--live-odds-dir DIR` | ライブ必須モード（欠損・ok=false・鮮度 NG は fail-safe 見送り） |
| `--max-age-min N` | ライブオッズ許容鮮度（既定 20 分） |
| `--race rid16[,rid16...]` | 指定レースのみ計算・apply（T-10 のレース単位実行用） |
| `--budget N` | 1 レース予算（Discord 再計算コマンドが使う） |
| `--plan PATH` | 枠プラン。**枠外レースは買わない**（`continue`）。枠対象は `force_floor=True` |
| `--fuku-hit` / `--fuku-hit-thr` | 複勝特化モード（§2.8） |

**環境変数**
| 変数 | 既定 | 意味 |
|---|---|---|
| `CB_ENGINE` | `topdown` | `shape` で旧経路 |
| `CB_ALLOC` | `p` | `flat` / `ev`（旧挙動） |

### 2.3 実行フロー（`compute_race_bets()`）

```
① ライブオッズ読込（--live-odds-dir 指定時）
     欠損 / JSON破損 / ok=false / 鮮度NG → 即 見送り (fail-safe)
     単勝・複勝を実値に差し替え（horses はコピー、bundle を破壊しない）
     ワイド (0B33) / 馬単 (0B34) 実値を lwide / lumatan に格納
② T-10 補正印 (hosei_marks) を算出 ← 見送りレースにも付けるため early return 前
③ 印の抽出（◎〇▲△、全角〇/丸○ 正規化）
④ §0 hard 見送りゲート
⑤ カード値（生値 → パーセンタイル）。分位表破損時は 見送り (fail-safe)
⑥ §0b 参戦規律（クリーン帯）— ★現在は force_floor で無効化されている
⑦ engine 分岐
     topdown → ⑧ へ / shape → ⑨ へ
⑧ topdown: λ補正 PL → 候補 4 種 → p 比例配分 → 適応トリガミ床 → return
⑨ shape:   形決定 → 候補生成 → 相手信頼ゲート → prob-first 選抜 → 配分 → return
```

### 2.4 §0 hard 見送りゲート（`compute_bets.py:440-449`）

| # | 条件 | 定数 |
|---|---|---|
| 1 | `field_chaos_score`の凍結分布percentile ≥ 0.667 | `production_policy.chaos_reference.skip_percentile` |
| 2 | `field_size` ≤ 7 | |
| 3 | ◎ の `tansho_odds` が null | |
| 4 | ◎ の `p_win` < 0.05 | |

この 4 条件は **`validate_cowork_bets.py` / `build_bet_plan.py` / `docs/cowork_prompt.md`
の 4 箇所で同じ値が独立にハードコードされている**（🟡 P2: 定数の単一ソース化が必要）。

さらに fail-safe 見送りが 3 つ:
- ライブオッズ関連（欠損 / 破損 / ok=false / 鮮度 NG）
- `chaos_quantiles.json` 欠如・破損（`pct()` が `None`）
- topdown で有効候補ゼロ（オッズ欠損）

### 2.5 §0b 参戦規律（クリーン帯ゲート）— **配線撤回済み・機構のみ残置**

```python
CLEAN_BAND_MAX = 0.33
if chaos > CLEAN_BAND_MAX:
    if not force_floor:
        return 見送り
    if demote_budget and budget > demote_budget:
        budget = demote_budget            # ← 呼び出し側が demote_budget を渡していない
```

**経緯（Vol. III §3.7 の教訓ケース）**:
1. 2024fit→2025eval の OOS 検証（`analysis/test_race_selection_oos.py`）で
   クリーン帯（エントロピー下位 1/3）のみ ◎複勝 ROI ≈90% / 単勝 85% と控除床を明確に超えた。
   実測 **+5.31pt**。
2. 2026 as-served 再検証（`analysis/reverify_clean_band_2026.py`, 686R）で
   **clean 77.6% < 帯外 87.9% と符号反転**。
3. → **配線中止**。`main()` は `demote_budget` を渡していない（`compute_bets.py:938-942` のコメント参照）。

**現在の実効挙動**: `--plan` 使用時は `force_floor=True` かつ `demote_budget=None` なので
**クリーン帯ゲートは完全に無効**。`--plan` 無しの単発実行時のみ「クリーン帯外は見送り」が効く。
つまり **本番（t10_runner 経由 = 常に --plan）ではこのゲートは死んでいる**。

### 2.6 topdown エンジン（既定、`compute_bets.py:479-541`）

#### 候補生成（4 種、最大 5 点）

| 券種 | 選び方 | オッズ上限 | 確率源 |
|---|---|---|---|
| 複勝 | `p_sho` 最大の 1 頭 | なし | bundle `p_sho`（較正済） |
| ワイド | λ補正 PL の `p_wide` 上位 2 ペア | ≤ 50 | `pl_pair_probs()` |
| 馬連 | `p_umaren` 上位 1 ペア | ≤ 50 | `pl_pair_probs()` |
| 単勝 | `p_win` 最大かつ `tansho_odds ≤ 30` の 1 頭 | ≤ 30 | bundle `p_win`（較正済） |

**オッズの取得**:
- ワイド: ライブ実値 `(lo+hi)/2` → 無ければ `umaren_matrix / 3.0` の推定
- 馬連: `umaren_matrix`（bundle、OD CSV 由来）
- 複勝: `(fuku_odds_low + fuku_odds_high) / 2`、**床は `fuku_odds_low`**（トリガミ判定は最悪ケースで）

#### 配分

```python
amts = allocate([p for each candidate], budget)   # p 比例
```
`allocate()`（`compute_bets.py:260-276`）:
1. `budget × w/Σw` を 100 円単位に丸め、`[MIN_BET, MAX_BET]` にクリップ
2. 合計が budget と合うまで、重み順に ±100 円を反復調整（最大 6,000 回）
3. キャップで埋まらない場合は満額未満で止まる（= **-EV に突っ込まない規律**）

#### 適応トリガミ床

§1.3 の通り。最安見込払戻 ≥ 総投資 になるまで最低 p の点を削る。

#### 出力

```json
{"race_nature":"topdown",
 "race_reason":"topdown全券種確率（混戦0.36/市場+0.72）で 2点。 [T-10オッズ 単複14頭/ワイド91組/馬単182組]",
 "confidence":{"top1_pct":...,"top2_pct":...,"chaos_pct":...,"market":...},
 "bets":[{"馬券種":"複勝","買い目":"9","購入額":6000,"枠タグ":"参加枠","理由":"topdown p=0.412（複勝 1.8倍）"}]}
```
表示順は `KIND_ORDER = 単勝→複勝→ワイド→馬連→馬単→三連複→三連単`、同券種内は金額降順。

#### 実測される挙動（2026-08-15/16、本番初適用 2 日分）

| 日 | レース | topdown 点数 | 総額 | shape シャドー点数 |
|---|---:|---:|---:|---:|
| 2026-08-15 | 22R（+13R は narrative のみ） | **34** | ¥91,000 | 102 |
| 2026-08-16 | 22R | **38** | ¥93,000 | 92 |

→ **平均 1.6–1.7 点/R**（shape は 4.4 点/R）。実態は
「最高確率馬の複勝を厚く + 単勝 + 少数ワイド」。

#### リプレイ検証（**in-sample 注意**）

`4/18–8/9, 実買付 506R 同条件・ペアブートストラップ`:
| 版 | ROI |
|---|---:|
| shape（旧） | 74.1% |
| shape + 構築層 4 修正 | 78.2% |
| **topdown** | **82.8%**（別記 83.6%） |

Δ +8.7pt、**CI95 [−0.5, +12.8]**、P(改善) = 0.961。
**CI 下限が 0 を割っている = 有意ではない**。前向き検証中（§11.3 / Vol. III §8）。

### 2.7 shape エンジン（`CB_ENGINE=shape`、旧経路・シャドー用）

#### 形（shape）判定

| 形 | 条件（すべてパーセンタイル） |
|---|---|
| 本命勝負 | `top1≥0.75 ∧ top2≥0.75 ∧ chaos≤0.50` かつ ◎〇 存在 |
| ◎軸 | `top1 ≥ 0.50` |
| 広め流し | `top1 < 0.25` または `top2 < 0.40` |
| カオス薄 | `chaos > 0.75` |
| 標準 | 上記以外 |

閾値定数: `TH_TOP1_GO/OK = 0.75/0.50`, `TH_TOP2_GO/OK/LOW = 0.75/0.50/0.40`,
`TH_CHAOS_HARD/MID = 0.75/0.50`, `TH_MARKET_ANABA = 0.30`

#### 候補生成と boost

```
本命勝負: ◎単勝(妙味時のみ, boost 1.6) + ◎複勝(1.1) + ◎-〇▲ の馬連&ワイド並行(1.3)
◎軸    : ◎単勝(妙味時のみ,1.6) + ◎-〇▲ ペア(1.1) + ◎複勝(1.0)
広め流し: ◎-〇▲ ペア(1.2) + 〇-◎▲ ペア(1.0) + ◎複勝(1.1)
カオス薄: ◎-〇▲ ペア(1.0) + ◎複勝(1.2)
標準    : ◎単勝(妙味時のみ,1.4) + ◎複勝(1.1) + ◎-〇▲ ペア(1.1)
穴overlay: market<0.30 かつ value_horses あり
           → 妙味馬の単勝(≤30倍, 1.3) と 複勝(1.1)  ※ペアは撤去済み
```

**2026-08-09 の構築層 4 修正**（4–8 月の全ベット解剖に基づく）:
| # | 修正 | 実測根拠 |
|---|---|---|
| ① | 穴 overlay の **vb-◎ ペアを撤去** | 妙味馬絡みワイド ROI 60.9%（n=575 / ¥595k）= 最大出血ブロック。妙味馬絡み馬連 72.2% vs 印純ペア馬連 104.8% |
| ② | **◎単勝は妙味 (under) 時のみ** | 妙味◎単勝 ROI 91.6%（n=163）vs 非妙味◎単勝 **14.1%**（n=31） |
| ③ | 点数 cap 6 → **5** | レース内 6 点目以降 ROI 61.9%（直近 4 週）/ 32.5–20.4%（全期間 rank7-8） |
| ④ | EV サイジング → **p×boost 配分** | §1.2 |

#### ペアの並行生成

`c_pair(i,j)` は **馬連とワイドを並行生成**する。理由:
> 馬連 = ROI 主軸（prob-only 79% > 控除 77.5%）、ワイド = 的中率／床防御。
> 混合 prob-first だと `p_wide > p_umaren` なので**全部ワイドに倒れる**
> → 選抜段で **型別に交互配置**する（`compute_bets.py:758-763`）。

#### 相手信頼ゲート（2026-07-23 配線）

```
p23 = (p2 + p3) / (1 - p1)     ※ p は降順ソートした p_win
if p23 < AITE_WEAK_TH (= 0.252):
    ペア候補を落として ◎単複へ予算集中
```
**発見（n=27,596, 2016-23）**: ◎の的中率は相手軸でほぼ不変（◎単勝 39.7→35.3%）だが、
**組み合わせ馬券の的中率は相手弱で半減**（◎強ワイド r1r2 的中 43.8% → 23.0%、
holdout 2024-25 で 43.4% → 24.9% と再現）。

⚠️ **閾値の分布校正**: 発見期（offline OOF）の下位 1/3 境界は 0.328 だが、
serve の `p_win` は低スケールなので 0.328 だと発火率が 69% に膨らむ。
serve 実分布（`reports/cowork_input` 933R）の 33 パーセンタイル **0.252** に校正済み。
**offline 閾値をそのまま serve に持ち込むと壊れる**（同じ現象が `FUKU_HIT_THR` にもある）。

#### 銘柄選抜（prob-first）

```
p_pair = ev / odds     # ev = odds × p の定義から厳密復元（新カラム不要）
cap = min(5, budget // MIN_BET)

1. アンカー: ◎絡みの単複を最大 2 枠確保（EV ≥ 0.80 のフロア）
2. ペア: 馬連リスト・ワイドリストをそれぞれ p 降順にし、交互に採用
3. 残りの単複
4. ◎必須: ◎絡みが 1 つも無ければ最高 prob の◎絡みを先頭に差し込む
```
⚠️ **型混在の生 prob 比較は禁止**（複勝 `p_sho` > ペア `p_pair` なのでペアが押し出される）。

### 2.8 複勝特化モード（`--fuku-hit`）

◎の `p_win ≥ FUKU_HIT_THR` のレースだけ ◎複勝を flat 購入する的中率重視モード。
オッズ非依存（選択はモデル確率のみ）。

- 設計操作点（offline v6 OOS 2024-25, `analysis/hit_rate_frontier.py`）:
  信頼度上位 ~20% 帯 → **的中 ~80% / 回収 ~92%**（valid2023 も一致）
- **`FUKU_HIT_THR = 0.21`**: offline の絶対値 0.36 では serve で発火率 0.8% にしかならない
  （serve の `p_win` 中央値は 0.13 vs offline ~0.25）。serve 実分布 655R で上位 ~20%
  （週 ~15R）になる値に校正
- ⚠️ 的中/回収 80/92% は **offline 射影**。serve スケール差があるため前向き検証が必須（未実施）

### 2.9 `docs/compute_bets_spec.md` との差分（**陳腐化リスト**）

旧仕様書（2026-06-09）は現行実装と以下が食い違う。**古い方を信じないこと。**

| 項目 | 旧仕様書 | 現行実装 |
|---|---|---|
| 既定エンジン | shape のみ | **topdown**（`CB_ENGINE` 既定） |
| 馬連 | 「**全廃**（ROI 56% 最弱）」 | prob-first で**再有効化**（`ODDS_CAP=50`） |
| 馬単 | 本命勝負で 8 点フォメーション採用 | **全廃** |
| 配分 | EV 帯 → 1 点額（EV サイジング） | **p × boost 比例**（`CB_ALLOC=p`） |
| 点数 cap | 本命勝負 8 / その他 6 | **5** |
| ◎単勝 | 常時（boost 1.4–1.6） | **妙味 (under) 時のみ** |
| 穴 overlay | 単勝/ワイド/複勝を上乗せ | **ワイド（vb-◎ペア）撤去**、単複のみ |
| 参戦規律 | 記載なし | §0b が追加されたが**撤回済み** |
| 決済ドリフト | 記載なし | `SETTLE_DRIFT_*` 追加 |

**🟡 P2**: 仕様書の更新が実装に追随していない。本 Vol. II が正典。

---

## §3 ガード群

### 3.1 全体像

```
[生成前]  export_weekly_marks 品質ゲート + serve canary   → exit 2 で push 停止 (fail-closed)
[生成時]  compute_bets §0 hard / fail-safe / トリガミ床   → 見送り or 点数削り
[生成後]  validate_cowork_bets --apply                    → bets を強制矯正
[運用]    weekly_nicegui -BetsOnly の fail-closed          → ガード実行不能なら停止
[運用]    weekly_post の git add 実測検証                  → 2 回空振りで fail-hard
[運用]    weekly_nicegui -Post の generated_at 照合        → Warn のみ（🟡 Fail にすべき）
```

### 3.2 `validate_cowork_bets.py`（380 行）

**背景**: 見送り判定は**決定論的な数値ルール**なのに、LLM (Cowork) の遵守は保証されない。
2026-05-30 に「本来 15/23 レースが見送り条件に該当するのに、全 23 レースに買い目が付いた」事故が発生。

**2 軸の検査**:

#### (A) 見送り 4 条件（bundle = 真値と突合）
`chaos percentile ≥ 0.667` / `field_size ≤ 7` / `◎ tansho_odds is null` / `◎ p_win < 0.05`
→ 該当レースに買い目があれば **`bets: []` / `race_nature: "見送り"` に強制書換**。
`race_reason` に `[自動見送り: <条件>]` を前置。`advisor` / `grade_scope` は残す。

#### (B) 内容検査
| 項目 | ルール |
|---|---|
| 券種 | `ALLOWED = {単勝,複勝,ワイド,馬連,馬単,三連複}` / `REJECTED = {三連単}` |
| 馬番 | bundle の `horses[].umaban` に実在するか |
| 金額 | 正 / 100 円単位 / ≤ ¥10,000（Cowork 手動経路は compute_bets の 7,000 より緩め） |
| 重複 | 同一 `(券種, 買い目)` の 2 個目以降を除去 |

**終了コード**: `0` = 違反なし or 修正完了 / `2` = 違反あり（dry） / `1` = **実行不能**

**fail-closed**（`weekly_nicegui.ps1`）:
```powershell
python validate_cowork_bets.py --date $Date --apply
if ($LASTEXITCODE -eq 1) {
    if ($Force) { Warn "... -Force のため続行 ..." }
    else { Fail "未検証の bets を push しないため停止します" }
}
```
`exit 1`（bundle/bets 不在・JSON 破損）は **HF 同期ごと停止**。`-Force` で明示バイパス可。

**「買えるのに見送っている」は警告のみ** — 買い目を捏造しない設計。

**🟡 既知の穴**: `ALLOWED_KINDS` に `馬単` が残っている（禁止券種のはず）。
compute_bets は生成しないが、Cowork 手動経路では通ってしまう。

### 3.3 fail-safe の哲学

| 状況 | 挙動 |
|---|---|
| ライブオッズが取れない | **見送り**（推定で買わない） |
| 分位表が壊れている | **見送り**（生値フォールバックはしない） |
| overround（Σ1/odds）が [1.0, 1.5] 外 | `jvlink_odds` が `ok=false` → **見送り** |
| 相手信頼ゲートで候補が全滅 | ゲートを適用しない（単複候補が無い時は fail-safe） |
| SHAP explainer 構築失敗 | `why` なしで bundle 生成継続（fail-open） |
| `serve_history_feats` 例外 | 従来どおり欠損のまま続行（fail-open）→ canary が検知 |
| `gutchi_brain` import 失敗 | 例外を握って `brain=None`（🟠 §6.5 参照） |

---

## §4 決済ドリフト補正（SettleAI）

### 4.1 問題

T-10 のオッズは締切ではない。締切までに資金が流入し、**勝ち馬のオッズは系統的に縮む**（steam）。
単勝 EV の期待払戻は `p_win × E[確定オッズ | 勝ち]` なので、
**T-10 オッズ素の EV は系統的に過大**になる。

### 4.2 実測（`reports/settle_drift.json`、2026-07-31 再 fit）

**単勝** — `reports/live_odds` の 462 勝者（2026-06-07〜07-26、T-10 → 確定）:
```
全体平均倍率 0.9221  CI95 [0.9032, 0.9416]   縮小率 61.5%
帯別:
  1.0-2 倍 : n=60,  ×1.0196  CI[0.9887,1.0538]   ← CI が 1 を跨ぐ → 1.00 に丸め
  2-4  倍 : n=118, ×0.9346  CI[0.9098,0.9625]
  4-8  倍 : n=139, ×0.8515  CI[0.8203,0.8831]
  8-20 倍 : n=106, ×0.9350  CI[0.8887,0.9821]   ← 有意
  20+  倍 : n=39,  ×0.9699  CI[0.8702,1.0803]   ← n<60 → 全体平均 0.922 で平滑
```
**複勝**: n=1,369、×0.8852 CI[0.8715, 0.8998]
**ワイド**: n=1,387、×0.920 CI[0.909, 0.932]

### 4.3 配線（`compute_bets.py:185-214`）

```python
SETTLE_DRIFT_TAN  = [(2.0,1.00),(4.0,0.935),(8.0,0.852),(20.0,0.935),(9e9,0.922)]
SETTLE_DRIFT_FUKU = [(1.5,0.988),(2.5,0.887),(5.0,0.826),(9e9,0.900)]
SETTLE_DRIFT_WIDE = [(3.0,0.881),(7.0,0.886),(15.0,0.929),(9e9,0.960)]
```
**適用範囲**: **EV（選別・配分）にのみ適用**。表示オッズ・買い目オッズは生のまま。

**前提監査（2026-07-30）の指摘の解決**: 「15–30 倍帯の符号が逆」は n=43 のノイズと確定
（再 fit で ×0.988 CI[0.89,1.10]）。

### 4.4 鮮度管理（`weekly_post.ps1` Step 2.7）

```powershell
$needRefit = (-not (Test-Path $driftJson)) -or ((Get-Date) - (Get-Item $driftJson).LastWriteTime).Days -ge 28
if ($needRefit) { python -m analysis.measure_settle_drift --notify }
python -m analysis.measure_settle_drift --check --notify   # 配線値との乖離を毎週警告
```
どちらも non-fatal（Warning のみ）。

### 4.5 SettleAI の位置づけ

決済層は「SettleAI サブ AI 第 1 号」として 2026-07-31 にプラン化された。
- 配線済み: 単勝 / 複勝 / ワイドのドリフト補正
- 着手順: **帯別再 fit（済）→ exotics 実市場 EV 初検定（`analysis/exotics_ev_market_test.py`、初回のみ）
  → per-horse 予測器（optional）**
- ❌ **禁止**: 「ドリフト方向で銘柄を選別する」— 両方向でエッジゼロが実証済み（毒）

---

## §5 枠プラン（`build_bet_plan.py`）

前日に、レースを信頼度で 3 枠に自動編成し各レース予算を割る。
買い目・金額の最終決定は T-10 の `compute_bets` が枠予算内で行う。

### 5.1 信頼度スコア

```
conf = 0.45 · top2_concentration + 0.30 · top1_dominance + 0.25 · (1 − field_chaos_score)
```

**◎前走圧勝ボーナス**（`MARGIN_BONUS_W = 0.08`）:
```
◎馬が前走 1 着 かつ 着差タイム < 0（＝圧勝）なら
conf += 0.08 × min(|着差|, 1.0)
```
根拠（2026-07-23 実測、n=4万+）: 前走 1 着の着差が大きいほど次走勝率が**単調増**
（0.0s → 12.7% / 0.3s → 16.1% / 1.0s → **31.5%**）。
⚠️ **ROI は市場に織り込み済みで不変**。したがって「賭け金レバー」ではなく
**「枠の信頼度レバー」= 的中率レバーとしてのみ使う**（`project_margin_rule_mining`）。

### 5.2 枠編成

```
見送り判定 (chaos percentile≥0.667 ∨ field≤7 ∨ ◎odds欠損 ∨ ◎p_win<0.05) → tier="見送り", budget=0
残りを conf 降順ソート:
  勝負   = 上位 3R          → ¥10,000 固定
  準勝負 = 次の 4R          → ¥8,000 → ¥5,000 に線形スケール（500 円丸め）
  消化   = floor 充足まで   → ¥1,000–3,000
  残り   = tier="対象外", budget=0
floor: 1 日 10R ∧ ¥100,000（--weekend で 2 倍）
```

出力: `reports/bet_plan/{date}.json`
```json
{"date":"...","floor":{"min_races":10,"min_yen":100000},
 "rules":{"禁止":["馬単","三連単","穴推奨の馬連","高EVだけの馬連"],
          "見送り":["混戦pct≥0.667","頭数≤7","◎odds欠損","◎p_win<0.05"]},
 "tiers":{"勝負":[...],"準勝負":[...],"消化":[...],"見送り":[...]},
 "totals":{"bet_races":N,"bet_yen":Y,"floor_met":true}}
```

### 5.3 実運用上の含意

`compute_bets --plan` は **枠プランに載っていないレースを一切買わない**（`continue`）。
つまり **1 日の参加レース数と総額は前日に確定している**。
T-10 が決めるのは「そのレースで何を何円ずつ買うか」だけ。

**消化枠は ROI を下げる**（承知の上）。実 OOS 決済では枠運用 ≈60% で黒字化していない
（`project_bet_tier_policy`）。

---

## §6 当日 T-10 ライン

### 6.1 全体像

```
土日 09:00 [Windows タスク PyCaLiAI_T10, WakeToRun]
   → t10.ps1 -Schedule
        bundle 完成を待つ（2 分間隔、15:00 デッドライン）
        旧 PyCaLiAI_T10R_* を全削除
        t10_runner.py --list-schedule で発走時刻を取得
        各レースの発走 -10 分に 1 個ずつタスク PyCaLiAI_T10R_{rid16} を登録（WakeToRun）
        changes.ps1 -DumpRaw（当日変更情報をサイトへ反映 + 生録保存）
   → 各レース T-10 [タスク起動、PC がスリープしていても起床]
        t10.ps1 -Once {rid16}
          keep_awake(True)                        ← SetThreadExecutionState
          ① py -3.12-32 jvlink_odds.py --race {rid16}   → reports/live_odds/{rid16}.json
          ② compute_bets.py --race --live-odds-dir --plan --apply
          ③ validate_cowork_bets.py --date --apply
          ④ 買い目をコンソール表示 + Discord 通知 + ビープ
          ⑤ 発走時刻まで Discord 予算返信（「2000円」）を受付 → 再計算
          changes.ps1（取消/騎手変更/馬体重をサイトへ）
          keep_awake(False)
   → 人間が IPAT で投票
```

**設計思想**: 旧方式（1 本ループで一日中回す）は PC 起動が必須で、スリープすると取りこぼした。
**レース毎タスク方式**なら PC がスリープしても各レースで自動起床し、処理してまた眠る。
- ⚠️ **完全シャットダウンは不可**（JV-Link はこの PC のみ）。スリープは OK。**サインアウトは不可**
- ⚠️ **祝日（月）開催はトリガー外** → 手動 `.\t10.ps1 -Schedule`
- 旧方式は `t10.ps1 20260614 -Loop` として残置

### 6.2 JV-Link パーサ仕様（全 4 券種確定、2026-06-12 raw 突合 + 確定配当照合）

**32-bit COM 必須**: `py -3.12-32 jvlink_odds.py`。64-bit では COM load 不可（`-2147221021`）。

```
JVInit(SID)=0 → JVRTOpen(spec, raceKey)=0 → JVRead ループ
SID: data/jvlink_sid.txt があればその 1 行目、無ければ "UNKNOWN"（個人利用扱い）
```

| spec | 券種 | レイアウト |
|---|---|---|
| 0B31 O1 | 単勝 | `pos45` 起点 `stride8` = odds(4) + 人気(2) + 予備(2)、値は /10 |
| 0B31 O1 | 複勝 | `pos269` 起点 **`stride12`** = lo(4) + hi(4) + 人気等(4)、/10 |
| 0B33 O3 | ワイド | `pos40` 起点 `stride17` = 組番(4) + lo(5) + hi(5) + 人気(3)、/10、153 組 + 票数計(11) |
| 0B34 O4 | 馬単 | `pos40` 起点 `stride13` = 組番(4) + odds(6) + 人気(3)、/10、306 組 + 票数計(11) |

⚠️ **複勝の `stride10` は誤り**（5 頭ごとに 1 スロットずれて別馬のオッズを返す致命バグ）。
2026-06-12 に修正済み。
**検証**: 馬単 40.0 倍 = kekka 4,000 円と完全一致 / ワイド 3 組とも実払戻が lo–hi 内 /
複勝は bundle 全頭一致。

**出力**:
```json
{"race_id":"...","fetched":"2026-08-16T14:50:03","ok":true,
 "tansho":{"1":3.4,...},"fukusho":{"1":[1.4,2.0],...},
 "wide":{"1-5":[3.2,4.1],...},"umatan":{"1>5":12.3,...},
 "overround_tan":1.21}
```
**fail-safe**: `overround`（単勝 Σ1/odds）が `[1.0, 1.5]` の外、または録が無ければ `ok=false`。
ワイド／馬単は取れなくても `ok` に影響しない（compute_bets が推定にフォールバック）。

### 6.3 T-10 補正印（オッズブレンド、表示専用）

```
u = log(p_win) + λ · log(π)          π = de-vig 市場単勝確率 = (1/odds) / Σ(1/odds)
λ = 1.5  (data/t10_blend.json, valid=2023 で fit)
u 降順の上位 5 頭に ◎〇▲△△ を付け直す
```

**OOS 実証（test 2024-25、6,858R）**:
| 系 | ◎top3 |
|---|---:|
| v6 単独 | 61.67% |
| 市場（1 番人気） | 64.33% |
| **ブレンド** | **65.08%** |

Δ(blend − v6) CI95 = **[+2.46pt, +4.30pt]**（有意）
Δ(blend − 市場) CI95 = [−0.09pt, +1.56pt]（**有意ではない**）

→ 「◎top3 62% の天井」は **オッズを使わない場合**の話であり、T-10 では破れる。
ただし市場に対しては有意に勝っていない点を誇張しないこと。

**表示専用**: 買い目計算・公開印には影響しない（`compute_bets.hosei_marks()`、
`fmt_hosei()` で `*` = 元印から昇格/降格を表示）。
`fail-soft`: `t10_blend.json` が読めなければ `None`（補正印なし）。

⚠️ **T-15 補正印のサイト公開は 2026-07-31 に停止**（JRA-VAN 投稿ガイドライン
「JV-Link から取得したデータは投稿できません」対応）。posting-support の照会次第で復活。

### 6.4 Discord 連携

| 方向 | 実装 | 設定 |
|---|---|---|
| 送信 | webhook（`notify()`）。起動サマリ / 各レース買い目（見送り含む）/ 全 R 完了 / bundle 未生成警告 | `notify_config.json` の `discord_webhook` または `PYCALIAI_DISCORD_WEBHOOK` |
| 受信 | Bot API ポーリング（`BotPoller`）。「2000円」等を拾って**そのレースを新予算で再計算** | `notify_config.json` の `bot_token` + `channel_id` |

- `User-Agent` 必須（Python-urllib 既定 UA は Cloudflare に 403 で弾かれる）
- 送信失敗は非致命（買い目生成は止まらない）
- メッセージ上限 2,000 字 → 1,900 字を超えたら 2 通に分割
- 予算コマンドの受理範囲: `500 ≤ v ≤ 100,000`

### 6.5 🟠 既知の欠陥 — `gutchi_brain` の dangling import

`t10_runner.py` には `brain_tickets()` / `render_brain()` が残っており、
`import gutchi_brain`（`:347`, `:369`）を実行する。
**`gutchi_brain.py` は 2026-08-09 に退役・削除済み**（実測: ファイル不在）。

**実害**: `process_race()` が `try/except Exception` で包んでいるため、
毎レース `ImportError` を握りつぶし `brain=None` → `render_brain` が空を返す。
機能的には無害だが:
- 毎レース `[brain] gutchi_brain 失敗 (非致命): No module named 'gutchi_brain'` がログに出る
- コード 90 行が完全な dead code
- 「🧠 俺のブレイン」併記機能はドキュメント上生きていることになっている

→ Vol. III §5 P2-1。

---

## §7 Cowork narrative 契約

### 7.1 役割（2026-06-12 全面改訂）

馬券構築は Cowork から**完全に分離**された。Cowork の役割は **narrative 専用**:
- **(A) advisor 論評**: 注目馬の自由日本語評価（各レース 2〜6 頭）
- **(B) Grade Scope**: G1/G2/G3 限定の読み物的詳細分析

### 7.2 絶対禁則（`docs/cowork_prompt.md`）

```
1. bets（買い目・金額）を書かない。各レースの "bets" は必ず空配列 []
2. advisor を対象レース全てに出力する（レースごとに 2〜6 頭）
3. ◎（本命）の馬は必ず advisor に含める
4. race ごと・馬ごとに個別評価（boilerplate / コメント使い回し禁止）
5. 数値変数（p_win / EV / contrib / pos_trend 等）を出力テキストに出さない
6. Grade Scope は G1/G2/G3 全レース必須
```

advisor を省略してよいのは、compute_bets の hard 見送り 4 条件に該当するレースのみ。

### 7.3 運用手順（Phase B）

1. Claude Desktop に `{date}_bundle.json` を添付 + プロンプト本文を貼る
2. レスポンス先頭の JSON を `reports/cowork_output/{date}_bets.json` として保存
3. `.\weekly_nicegui.ps1 -BetsOnly`
4. 当日 T-10 に `compute_bets.py --apply` が **同一ファイルへ bets を in-place merge**

**ファイル形式**: `{"bets":[{race_id, race_label, bets: [], advisor: [...]}], "grade_scope":[...]}`
実測（2026-08-15/16）: 35 レース中 22 レースに compute_bets のスタンプ、
13 レースは `advisor` のみ（`race_nature` が `null`）。

⚠️ **`docs/cowork_prompt.md` の 1 行目に `yaru` という不要な文字列が混入している**（⚪ P3）。

---

## §8 週次運用フロー

すべて `weekly_nicegui.ps1`（423 行）1 本。`weekly_pre.ps1` / `weekly_post.ps1` は内部呼び出し。

### 8.0 Step 0 — intake 自動振り分け（全フェーズ共通、`-SkipIntake` で無効）

TARGET からエクスポートした CSV を `data\_inbox\` に全部放り込むと、
`place_weekly.py` がファイル名（S/K/H-/W-/OD）と中身（15 列 = 結果 / 174 列 = 払戻→実現バイアス）で
`data/weekly/` `kako5/` `kekka/` `training/` `bias/` へ自動振り分けする。

### 8.1 Phase A — 土曜朝

```powershell
.\weekly_nicegui.ps1              # 最新 data/weekly/*.csv を自動検出
.\weekly_nicegui.ps1 20260816     # 日付指定
```

| Step | 処理 | 失敗時 |
|---|---|---|
| 0 | `place_weekly.py`（intake） | Warn 続行 |
| 1 | `make_weekly_hosei.py --csv` → `data/hosei/H_{date}.csv` | Warn 続行 |
| 2 | `predict_weekly.py`（旧 8 モデル） | **既定 SKIP**。`-WithPredict` で opt-in |
| 3 | `export_weekly_marks.py --model v6` → **bundle.json** | **Fail（停止）** |
| 3b | `build_course_stats.py` → `data/course_stats.json` | Warn 続行 |
| 4 | git add / commit / `git pull --rebase --autostash` / push origin master | Warn |
| 5 | `sync-hf.ps1`（旧 NiceGUI Space） | Warn |
| 5b | `sync-hf-umami.ps1`（**本番静的サイト**） | Warn |

**日付自動検出**: `data\weekly\` の 8 桁 basename のみを対象（`test.csv` 等は弾く）。
BetsOnly モードだけ `reports\cowork_output\{8桁}_bets.json` から検出する。

### 8.2 Phase B — 土曜昼（Cowork narrative 保存後）

```powershell
.\weekly_nicegui.ps1 -BetsOnly
```
1. **見送りガード**: `validate_cowork_bets.py --date --apply`
   - `exit 1`（実行不能）→ **fail-closed で停止**（`-Force` でバイパス）
2. git add `reports/cowork_output` + `reports/cowork_bets/{date}` → commit → push
3. `sync-hf.ps1` → `sync-hf-umami.ps1`

### 8.3 Phase C — 日曜夜（結果 CSV 配置後）

```powershell
.\weekly_nicegui.ps1 -Post
```

`weekly_post.ps1`（193 行）の中身:

| Step | 処理 | 失敗時 |
|---|---|---|
| 1 | `generate_results.py` → `data/results.json` + `data/cowork_results.json` | **exit 1** |
| 2 | `update_live_results.py --date` → `data/live_results_2026.csv` | Warning |
| 2.5 | `build_horse_history.py` → `data/_horse_history.parquet` | Warning |
| 2.7 | `analysis.measure_settle_drift`（28 日で再 fit / 毎週 `--check`） | Warning |
| 3 | **`Invoke-GitAddVerified`** — add 後に `git diff --cached` で実測検証、0 件なら 3 秒後リトライ、それでも 0 件かつ変更が残っていれば **fail-hard exit 1** | exit 1 |
| 4 | commit → `git pull --rebase --autostash` → push | exit 1 |
| 追加 | 月初（1–7 日）の日曜 → `retrain_value_model.py` | Warning |
| 追加 | 日曜 → `run_audit.ps1`（週次監査） | — |

`weekly_nicegui.ps1 -Post` 側:
- `weekly_post.ps1` が非 0 なら **Fail して HF 同期を中止**（結果更新漏れのまま緑の Done が出る事故の対策）
- `cowork_results.json` の `generated_at` が当日でなければ **Warn**（集計凍結の検知）
  → 🟡 P2: これは Fail にすべき

### 8.4 デプロイ（HF Spaces / 独自ドメイン）

| スクリプト | 対象 | 内容 |
|---|---|---|
| `sync-hf-umami.ps1`（168 行） | **本番** Docker Space `gutchi15300/pycaliai-umami` | `build_site.py` で `site/data/*.json` を再生成 → push |
| `sync-hf.ps1`（229 行） | 旧 NiceGUI Space `gutchi15300/pycaliai` | master → hf-spaces orphan ブランチ → push |
| Cloudflare Workers | `pycaliai.com` | git push で自動デプロイ。301 統一 / GSC 登録 / SEO 済 |

⚠️ **HF 反映を伴う push はユーザー確認が必要**（CLAUDE.md の自律性ルール）。
⚠️ Cloudflare の SPA fallback により **404 でも 200 が返る**罠がある。

---

## §9 決済・集計（`generate_results.py`、1,170 行）

### 9.1 入力ソース

| ソース | 形式 |
|---|---|
| `reports/cowork_bets/{YYYYMMDD}/{race_id}.json` | 旧形式（per-race） |
| `reports/cowork_output/{YYYYMMDD}_bets.json` | 現行（bundle） |
| `data/kekka/{YYYYMMDD}.csv` | 着順・払戻 |
| `data/kekka/wide_kekka.csv` | ワイド払戻（2026〜） |
| `data/wide_payouts_2016-2025.parquet` | ワイド払戻（履歴） |

### 9.2 決済ロジック

`match_cowork_bet()` が `(hit, payout_per_100, refund_ratio)` を返す。

**返還の扱い（audit 2026-06-11 で修正）**:
```
refund      = amount × refund_ratio        # 取消・除外馬を含む組
eff_amount  = amount − refund              # 実効投資
ret         = amount × payout_per_100/100  # 的中時
```
取消馬を含む組を全額損失計上すると **券種別 ROI が系統的に過小**になるため、
実効投資から除く。

**決済不能（`settled=False`）**: ワイド払戻が未取込のケース。**集計から除外**する
（0 として計上しない）。

### 9.3 信頼区間（`_bet_cis()`）— 規律の中核

```python
# 的中率: Wilson score interval (z=1.96)
# ROI:    bootstrap（bet 単位リサンプル、投資加重、n_boot=2000、seed=42 で決定的）
verdict = "above_takeout"  if roi_ci_lo > 80.0    # 真に控除率超の証拠
        else "below_takeout" if roi_ci_hi < 80.0  # 真に控除率未満の証拠
        else "inconclusive"                        # CI が 80 を跨ぐ
```
`n ≥ 10` の券種にのみ付与。

**★ この `roi_verdict` が本プロジェクトのポリシー変更の唯一の許可条件である。**
> 規律: boost / 全廃などのポリシー変更は `roi_ci95` が控除率（≈80%）を片側に外れたときだけ行う。
> **点推定での判断は禁止。**

背景: 単勝 30 bets で ROI 120.8% を見て boost を上げたが、445 bets で 96.3% に回帰した事故。

### 9.4 出力

```
data/results.json          … 4 プラン形式（HAHO/HALO/LALO/CQC、Streamlit 表示用）
data/cowork_results.json   … 実運用集計（total / by_type / by_place / weekly / races / bets）
```
`cowork_results.json` は **毎回 commit する**（集計凍結対策、2026-05-26 事故）。

---

## §10 公開層

### 10.1 静的サイト（本番）

| 項目 | 値 |
|---|---|
| 本番 URL | `https://pycaliai.com`（Cloudflare Workers） / `https://gutchi15300-pycaliai-umami.hf.space` |
| ソース | `site/`（`index.html` 26KB / `js/app.js` 88KB / `js/baba.js` 8KB / `css/style.css` 80KB） |
| データ生成 | `build_site.py` → `site/data/{date}.json`（51 日分）+ `manifest.json` |
| 技術 | vanilla JS + ECharts（CDN）。フレームワークなし |

**画面構成（`site/js/app.js` 実測）**:
```
mode  : races（予想）/ results（成績）
views : 出走表 / 全頭分析 / コース / 血統
UI    : hero（表紙）/ landing / venueTabs / raceStrip / raceHeader / viewTabs / drawer（馬詳細）
機能  : 馬指数キャリアチャート、実現トラックバイアスカード、メンバーレベル、
        UMAMI グレード表示、当日変更情報オーバーレイ（取消/騎手変更/時刻/馬体重）
```

**配色**: ネイビー × ゴールド（2026-08-11 のリデザインでライムを試したが不採用）。

### 10.2 TACT（公開買い目ライン）

**TACT は独立エンジンではない。`compute_bets` topdown の公開ラッパである**
（2026-08-09 に旧 `gutchi_brain` 決定木から置換。同一 526R リプレイで 72.4% → 82.8%）。

```python
def build_tact(race):                      # build_site.py:547
    from compute_bets import compute_race_bets
    tickets = compute_race_bets(race, budget=10000, force_floor=True).get("bets") or []
    return {"version":"1.0td",
            "bets":[{"type":..., "selection":..., "reason": _TACT_ODDS_RE.sub("", 理由)}]}
```
- **金額は出さない**（買う人が決める）
- **理由文からオッズ表記を正規表現で除去**（JRA-VAN ガイドライン対応）
- `bets=[]` は見送り

**公開線と実買線の差（`analysis/tact_line_eval.py`）**:
公開線は「朝オッズ + ¥10,000 固定」、実買線は「T-10 + 枠予算」なので
**約 3 割のレースで買い目が一致しない**。ROI 差は **−2.15pt CI[−7.9, +3.5] = 有意でない**。
⚪ **残件**: 公開線自体の成績が未集計 / バージョン固定が未実施。

### 10.3 note 有料販売

- 会場バラ ¥100 + 全場パック ¥100 × 会場数（3 会場 = ¥300、2026-08-15 値下げ）
- **買い目 / 枠格付 / 見送り理由はレース確定までサイト非公開**（note 専売）
- 広告モデルは必要 PV が桁違いのため却下

### 10.4 X（旧 Twitter）

型: `日付 + 会場 + R → ◎〇▲印馬 → 概要 / 140 字圧縮 / ハッシュタグ無し`。
結果は的中しなくても報告（着順 + 印、上位 5 頭は公開なので △ まで可）。
**印だけで獲れる払戻のみ的中主張**。

### 10.5 JRA-VAN 投稿ガイドライン準拠（2026-07-31 対応済み）

サイトから撤去したもの:
- 調教タイム生値 / オッズ生値 / 払戻金額 / EV / ライブ馬体重 / T-15 補正印
- ZI・補正タイム（TARGET 外部指数）— 「抜き出し強調・まとめ公開は不可」

出典表記を追加。撤去処理は `scrub_public` に一元化。
**新データをサイトに出す時は必ずこの基準に照らすこと。**

---

## §11 実運用実績（実測、`data/cowork_results.json` 2026-08-17 生成）

### 11.1 累積

| 指標 | 値 |
|---|---:|
| レース数 | **1,178**（うち見送り 607） |
| 決着済 | 1,178 |
| 馬券点数 | 2,389 |
| 実効投資 | **¥3,795,834** |
| 払戻 | ¥2,728,457 |
| 収支 | **−¥1,067,377** |
| **ROI** | **71.9%** |
| 的中率 | 19.2%（458/2,389） |

### 11.2 券種別（CI 付き）

| 券種 | 点数 | 的中率 [CI95] | 投資 | ROI | ROI CI95 | **verdict** |
|---|---:|---|---:|---:|---|---|
| ワイド | 909 | 18.0% [15.7, 20.7] | ¥1,312,534 | 75.6% | [59.7, 93.4] | inconclusive |
| 単勝 | 307 | 14.7% [11.1, 19.1] | ¥513,500 | **84.9%** | [54.9, 118.3] | inconclusive |
| 複勝 | 390 | 48.2% [43.3, 53.2] | ¥704,100 | 69.0% | [58.7, 79.4] | **below_takeout** |
| 馬連 | 661 | 8.9% [7.0, 11.3] | ¥1,093,200 | 71.0% | [42.0, 104.5] | inconclusive |
| **馬単** | 122 | 1.6% [0.5, 5.8] | ¥172,500 | **22.1%** | [0.0, 58.3] | **below_takeout** |

**読み方**:
- **馬単は統計的に確定した負け** → 全廃済み（過去分の負債）
- **複勝も `below_takeout`** — 的中率 48.2% は高いが配当が薄すぎる。
  これは「当たる ≠ 儲かる」の実証であり、topdown が複勝アンカーに寄せている設計への
  **反証候補**として監視すべき（Vol. III §5 P1-3）
- ワイド・単勝・馬連はいずれも CI が 80% を跨ぐ = **点推定でポリシーを動かしてはいけない**

### 11.3 週次推移（直近 12 週、実測）

| 週（終端） | R 数 | 投資 | 払戻 | ROI |
|---|---:|---:|---:|---:|
| 2026-05-31 | 46 | 150,000 | 24,270 | 16.2% |
| 2026-06-07 | 46 | 170,000 | 81,100 | 47.7% |
| 2026-06-14 | 71 | 426,500 | 146,240 | 34.3% |
| 2026-06-21 | 70 | 163,000 | 99,940 | 61.3% |
| 2026-06-28 | 69 | 188,500 | 155,180 | 82.3% |
| 2026-07-05 | 70 | 199,500 | 165,160 | 82.8% |
| 2026-07-12 | 70 | 72,000 | 70,980 | 98.6% |
| 2026-07-19 | 69 | 146,000 | 147,350 | 100.9% |
| 2026-07-26 | 70 | 200,000 | 114,430 | 57.2% |
| 2026-08-02 | 70 | 198,000 | 196,870 | 99.4% |
| 2026-08-09 | 70 | 195,000 | 101,240 | 51.9% |
| **2026-08-16** | 70 | 184,000 | 100,500 | **54.6%**（topdown 初適用週） |

**週次 ROI の分散が極めて大きい**（16% 〜 101%）。
週 70R / ¥200k 規模では **1 週間のデータで何かを判断してはいけない**。
topdown 初週の 54.6% も同様（n が全く足りない）。

### 11.4 前向き検証の進捗（topdown vs shape）

| 項目 | 状態 |
|---|---|
| 開始 | 2026-08-15 |
| 蓄積 | **72 bets / 44 レース**（2 開催日） |
| 判定閾値 | **n_bets(topdown) ≥ 300** |
| 必要な追加開催日 | 約 8 日（≈4 週末） |
| 判定ツール | `analysis/prospective_topdown_eval.py`（レース単位 paired bootstrap 10,000 回、seed=42） |
| シャドー | `reports/engine_shadow/{date}_shadow.json`（compute_bets が `--apply` 時に自動併記） |

**事前固定した判定基準（`docs/hypothesis_registry.md` P1-TOPDOWN-PROSPECTIVE-2026）**:
- PASS: ΔROI > 0 かつ CI95 下限 > −2pt
- FAIL: ΔROI < 0 かつ CI95 上限 < +2pt
- INCONCLUSIVE: それ以外（必要追加 n を明示して継続）

**判定日まで topdown 固定運用**（未来の結果でエンジンを選ばない）。

---

**→ 続き: [Vol. III 検証史・現在の課題・研究計画](VOL3_VALIDATION_AND_OPEN_PROBLEMS.md)**
