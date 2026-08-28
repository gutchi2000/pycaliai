# PyCaLiAI 完全仕様書 Vol. IV — コードリファレンス（関数レベル）

> 版 1.0 / 2026-08-23
> **目的**: リポジトリ本体を持たない外部レビュワー（ChatGPT 等）が、
> 主要モジュールの構造・契約・落とし穴を関数単位で把握できるようにする。
> 行番号は 2026-08-23 時点。挙動の説明は [Vol. I](VOL1_SYSTEM.md) / [Vol. II](VOL2_BETTING_OPS.md) を参照。

---

## 目次

- §1 依存グラフ（本番ライン）
- §2 `pl_probs.py` — 確率エンジン
- §3 `predict_weekly.py` — 入力パーサ（本番の入口）
- §4 `serve_history_feats.py` — as-of 履歴再計算
- §5 `export_marks_json.py` — 1 レース推論
- §6 `export_weekly_marks.py` — bundle 生成オーケストレータ
- §7 `betting_judgment.py` / `umami.py` — 買い方判定
- §8 `compute_bets.py` — 馬券構築エンジン
- §9 `validate_cowork_bets.py` — ガード
- §10 `t10_runner.py` / `jvlink_odds.py` — 当日ライン
- §11 `build_bet_plan.py` — 枠プラン
- §12 `generate_results.py` — 決済・集計
- §13 `build_site.py` — 公開層
- §14 `optuna_v6_marks.py` — 学習
- §15 横断的な実装パターンと落とし穴

---

## §1 依存グラフ（本番ライン）

```
export_weekly_marks.main()
 ├── predict_weekly.parse_csv                 ← 入力パース（§3）★欠陥の発生源
 ├── export_weekly_marks.ensure_date_column
 ├── (inline) _SERVE_RENAME
 ├── serve_history_feats.fill_history_features ← as-of 履歴（§4）
 ├── kako5_summary.build_histories / build_horse_facts
 ├── parse_od_csv.load_od_matrix_odds          ← オッズ
 ├── marks_shap.build_explainer
 └── export_marks_json.export_race             ← 1 レース推論（§5）
      ├── pl_probs.*                           ← PL 厳密（§2）
      ├── backtest_pl_ev.all_fukusho_vec_fast
      ├── marks_shap.race_contribs
      └── betting_judgment.build_judgment       ← 買い方判定（§7）
           └── umami.umami_for_horse

compute_bets.main()                            ← 馬券構築（§8）
 ├── compute_bets.load_live_odds               ← reports/live_odds/*.json
 ├── compute_bets.pl_pair_probs                ← λ補正 PL
 ├── compute_bets.hosei_marks                  ← T-10 補正印
 ├── compute_bets.compute_race_bets            ← topdown / shape
 └── compute_bets.apply_to_bets_json           ← in-place merge

t10_runner.process_race()                      ← 当日オーケストレータ（§10）
 ├── subprocess: py -3.12-32 jvlink_odds.py --race
 ├── subprocess: compute_bets.py --race --apply
 ├── subprocess: validate_cowork_bets.py --apply
 └── t10_runner.show_race_bets → notify()

generate_results.main()                        ← 決済・集計（§12）
build_site.main()                              ← 公開層（§13）
 └── compute_bets.compute_race_bets            ← TACT（公開買い目）
```

**循環参照に近い箇所**: `build_site.py` が `compute_bets.compute_race_bets` を import し、
`compute_bets` は `betting_judgment` → `umami` を（間接的に）参照する。
`t10_runner` も `compute_bets.load_live_odds` / `fmt_hosei` を import する。
**`compute_bets` は「馬券エンジンのライブラリ」としても使われている**点に注意
（`__main__` ガードは正しく置かれている）。

---

## §2 `pl_probs.py`（240 行）— 確率エンジン

依存: numpy のみ。副作用なし。純関数のみ。

| 関数 | シグネチャ | 契約 |
|---|---|---|
| `pl_weights` | `(scores: ndarray) -> ndarray` | `exp(s − max s)`。オーバーフロー回避のため max を引く。**スケール不変ではない**（PL は差のみに依存するので実は不変） |
| `p_tansho` | `(w, i) -> float` | `w_i / Σw` |
| `p_umatan` | `(w, i, j) -> float` | `i==j` なら 0.0 |
| `p_sanrentan` | `(w, i, j, k) -> float` | `len({i,j,k}) < 3` なら 0.0 |
| `p_umaren` | `(w, i, j)` | `p_umatan(i,j) + p_umatan(j,i)` |
| `p_sanrenpuku` | `(w, i, j, k)` | 6 順列の和 |
| `p_place_at` | `(w, i, pos)` | `pos ∈ {1,2,3}` のみ。それ以外は `ValueError` |
| `p_fukusho` | `(w, i)` | `Σ_{pos=1..3} p_place_at`。O(N²) |
| `p_wide` | `(w, i, j)` | `Σ_{k≠i,j} p_sanrenpuku(i,j,k)`。O(N²) |
| `all_tansho` / `all_fukusho` / `all_umaren` / `all_wide` / `all_sanrenpuku` | `(w) -> ndarray or dict[(i,j)→p]` | ベクトル版。key は **index**（馬番ではない） |
| `_test()` | — | `python pl_probs.py` で 9 種の恒等式を assert |

**レビュー観点**:
- `p_place_at(pos=3)` は O(N²) のループ。N≤18 なので実用上問題ないが、
  `all_fukusho` は O(N³) になる。1 レース 18 頭で 5,832 回 — 許容範囲
- ゼロ除算リスク: `total - w[j] - w[k]` が 0 になるのは全重みが 2 頭に集中した極端ケースのみ。
  `exp()` の出力は常に正なので実質起きない
- **`w` の index と馬番の対応は呼び出し側の責任**。`export_marks_json` は
  `g.sort_values(COL_BAN)` 後の位置 index を使い、`bans[i]` で馬番に戻す

---

## §3 `predict_weekly.py`（2,023 行）— 入力パーサ

**本番が使うのは `parse_csv()` のみ**（`export_weekly_marks.py:57` が import）。
残り（`predict_*` / `ensemble_predict` / `get_bets` 等）は旧 8 モデル系統で、Phase A では既定 SKIP。

### 3.1 定数

| 名前 | 長さ | 内容 |
|---|---:|---|
| `RACE_COLS` | 19 | レースヘッダ行。`レースID(新), 日付S, 曜日, 場所, 開催, R, レース名, クラス名, 芝・ダート, 距離, コース区分, コーナー回数, 馬場状態(暫定), 天候(暫定), フルゲート頭数, 発走時刻, 性別限定, 重量種別, 年齢限定` |
| `HORSE_COLS_33` | 33 | 旧形式 |
| `HORSE_COLS_46` | 46 | **現在エクスポートされている形式** |
| `HORSE_COLS_48` | 48 | 46 + `騎手コード, 調教師コード`（末尾 2 列）★Vol. III P0-1 |
| `HORSE_COLS_49` | 49 | 馬体重 3 列入り |
| `HORSE_COLS_99` | 99 | 3 走前まで |
| `TYAKU_HORSE_COLS` | 55 | 着度数 CSV。**実データは 53 列** ★Vol. III P0-2 |

### 3.2 `parse_csv(path: Path) -> pd.DataFrame`（`:391`）

```
1. cp932 / shift_jis / utf-8 の順にデコード試行
   ⚠️ すべて失敗すると text が未定義のまま次のループへ → UnboundLocalError（Vol. III P3-3）
   ⚠️ path が str だと read_bytes() が AttributeError → 同上
2. 行を列数で分類:
     19 → current_race = dict(zip(RACE_COLS, cols))
     33/46/48/49/99 → horse = dict(zip(HORSE_COLS_*, cols)); horse.update(current_race)
   ★ どの分岐にもヒットしない行は無言で捨てられる
3. DataFrame 化 → COLUMN_MAP でリネーム
4. レースID(新/馬番無) = レースID(新)[:16]
5. 障害競走を除外
6. 派生特徴の生成:
     prev_pos_rel  = (前1角 − 1) / (出走頭数 − 1)
     closing_power = (前1角 − 前4角) / (出走頭数 − 1)
7. jockey_stats.csv / trainer_stats.csv の merge   ← :466-485 ★P0-1（常に else 分岐）
8. _load_tyaku() の merge                          ← :489-524 ★P0-2（常に None）
9. 訓練 valid 中央値による定数補完（15 列）        ← :526-560 ★P1-2
10. _load_kako5_warnings / _load_hosei / 調教 JOIN
```

**副作用**: `logger.info` を多数出す（kako5 カバレッジ、調教 JOIN 等）。
実行時間は **約 30 秒**（調教 CSV 530 万行 + WC 75 万行を毎回読むため）。

**キャッシュ**: `_get_cached(path, key)`（`:74`）でファイル単位のメモ化がある。

### 3.3 `_load_tyaku(date_str) -> pd.DataFrame | None`（`:236`）

```
data/tyaku/{date}.csv を cp932 で読み、
  len(cols) == 19 → current_race_id = cols[0][:16]
  len(cols) == 55 → row = dict(zip(TYAKU_HORSE_COLS, cols))    ★実データは 53
rows が空なら None を返す（★無言）
馬ごと複勝率をベイズ平滑化: smoothed = (着内 + 1.43) / (総走 + 5.0)
   prior = 訓練 valid 中央値 0.286、仮想サンプル 5
```

### 3.4 `_load_hosei(date_str)`（`:361`）

`data/hosei/H_{date}.csv` から `レースID(新), 前走補9, 前走補正` を返す。
`export_weekly_marks._SERVE_RENAME` が `prev_hosei` / `prev_hosei9` にリネームする。

---

## §4 `serve_history_feats.py`（327 行）— as-of 履歴再計算

### 4.1 定数

```python
NUM_FEATS = ["hist_same_cond_best_pos","hist_same_cond_top3_rate","hist_same_cond_count",
             "hist_same_place_best_pos","course_n_prev","course_win_rate","course_top3_rate",
             "jockey_n_prev","jockey_win_rate","jockey_top3_rate"]      # 10
CAT_FEATS = ["騎手コード","調教師コード"]                                 # 2
```
⚠️ `jockey_fuku30/90` / `trainer_fuku30/90` / `horse_fuku10/30` は**含まれない**（Vol. III P0-1）。

### 4.2 `class _HistoryIndex`（`:72`）

`data/_horse_history.parquet` を馬名でインデックス。

`resolve(name, sire, birth_year) -> (entry | None, reason)`（`:94`）:
```
候補を馬名で引く
複数候補 → 父名（種牡馬）一致で絞る
なお複数 → 生年（レース年 − 年齢）±1 で絞る
1 件に絞れなければ (None, "ambiguous")      ← 安全側
0 件なら (None, "new")                      ← 新馬等
```

### 4.3 `compute_row_feats(ent, race_date, place, surface, ...)`（`:151`）

**学習側と同一定義であることが契約**（docstring に定義が明記されている）:
```
hist_same_cond_*   : 同 TD(芝/ダ) かつ距離 ±200m の全キャリア着順から best/top3率/回数
                     過去走ゼロ or 着順全 NaN → NaN
hist_same_place_*  : 同場所の全キャリア最高着順
course_*           : course_key = 場所|芝ダ|距離帯(短≤1400/マ≤1700/中≤2200/長)
                     n_prev = 過去同 key 走数（着順 NaN 含む、初出走 = 0）
                     win/top3 rate = n_prev>0 のときのみ（else NaN）
jockey_*           : 馬 × 騎手コードのペアの累積（騎手単独ではない）
```
**as-of**: `race_date` より**厳密に前**の走のみを使う。

### 4.4 `fill_history_features(df, base=BASE, ...)`（`:248`）

戻り値: `{"hit":int, "new":int, "ambiguous":int, "coverage":{feat: float},
"jockey_code":int, "trainer_code":int, "hist_max_date":int}`

**fail-open**: 呼び出し側（`export_weekly_marks.py:358-377`）が `try/except` で包み、
例外時は「従来どおり欠損のまま」続行する。
**鮮度チェック**: `date_str - hist_max_date > 300` なら「parquet が古い」WARNING。

---

## §5 `export_marks_json.py`（469 行）— 1 レース推論

### 5.1 `export_race(rid, g_orig, model, feats, encs, tansho_idx, fuku_idx, calibrators=None, umaren_idx=None, class_prior_map=None, shap_explainer=None, shap_topk=0, shap_marked_only=True) -> dict`（`:208`）

```
 1. rid_s = str(rid) の float 除去
 2. g = g_orig.sort_values(馬番).reset_index(drop=True)
 3. encoders 適用: 未知値は "__NaN__" に落としてから transform
 4. X = g_enc[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
       ★ CAT_COLS に無い文字列列はここで −9999 になる（Vol. I §4.2 C 群）
 5. scores = model.predict(X)
 6. w = PL.pl_weights(scores)
    p_win_vec = PL.all_tansho(w)
    p_plc_vec = 手計算の連対率（p_win + Σ_j P(j 1着) · P(i 2着 | j 1着)）
    p_sho_vec = all_fukusho_vec_fast(w)          ← backtest_pl_ev の高速版
 7. calibrators 適用: tansho → p_win / fukusho → p_sho     ★p_plc は較正なし
 8. order = argsort(-scores)  → ai_rank, mark（top-5 に ◎〇▲△△）
 9. market_rank = tansho オッズの rank（昇順 = 人気順）
10. contribs = marks_shap.race_contribs(...)（失敗時は None、bundle 生成は継続）
11. horses[] を組み立て（horse_record）→ 馬番昇順にソート
12. conf = race_confidence(...)   ※オッズが 3 頭未満なら market_rank を渡さない
13. payload = {race_id, race_meta, horses, race_confidence, buy_judgment}
14. umaren_matrix（OD CSV 由来）を埋め込み
15. pair_probs（印 5 頭の C(5,2)=10 ペア、較正済）を埋め込み
```

### 5.2 `race_confidence(p_win_vec, p_plc_vec, ai_rank_order, market_rank_by_ban=None)`（`:118`）

```python
p_sorted   = sort(p_win)[::-1]
top1_dom   = clip(p_sorted[0] - p_sorted[1], 0, 1)
top2_conc  = clip(p_sorted[0] + p_sorted[1], 0, 1)     # 較正後は >1 になりうるので clamp
p          = p_win_vec[p_win_vec > 1e-9]
p_norm     = p / p.sum()                                # ★較正で Σ≠1 になるので再正規化
chaos      = clip(-Σ p_norm·log(p_norm) / log(len(p_norm)), 0, 1)
market_corr= Spearman(rank(ai_rank_order), rank(market_rank_by_ban))   # 3 頭未満なら None
```

**レビュー観点**: `ai_rank_order` は `ai_rank_by_idx.copy()` が渡されるが、
`race_confidence` 内で `pd.Series(...).rank()` に通しているので、
順位の順位を取る二重処理になっている（結果は同じ。可読性の問題）。

### 5.3 `horse_record(...)`（`:150`）

`ai_vs_market` の判定:
```python
market_p = 1.0 / tansho_odds                  # 控除率を無視した implied
p_win >= market_p * 1.20 → "under"
p_win <= market_p * 0.80 → "over"
else                     → "fair"
odds が無ければ            "unknown"
```

---

## §6 `export_weekly_marks.py`（598 行）— bundle 生成

### 6.1 `feature_coverage(s: pd.Series, allow_constant=False) -> float`（`:134`）

```python
if s.dtype == object:
    valid = s.notna() & (str(s) != "__NaN__") & (str(s) != "")
else:
    valid = s.notna()
cov = valid.mean()
if not allow_constant and cov > 0 and s[valid].nunique() <= 1:
    return 0.0          # ★定数刷り込みを「情報ゼロ」として検出
return cov
```
`CONST_OK_COLS = {"馬場状態", "天気"}` は `allow_constant=True` で呼ばれる
（快晴開催では全レース同値になるのが正常なため）。

### 6.2 `main()`（`:176`）の処理順（Vol. I §9.6 と同じ）

主要な引数:
| フラグ | 既定 | 意味 |
|---|---|---|
| `--csv` | 必須 | `data/weekly/{date}.csv` |
| `--model` | **`v6`** | 手動実行で誤って退役 v5 の bundle を作らないよう本番既定 |
| `--out-dir` | `reports/cowork_input/{date}/` | |
| `--shap-topk` | 6 | 0 で SHAP 無効 |

**較正器の選択**（`:216-218`）:
```python
serve_cal = BASE / f"models/pl_calibrators_{tag}_serve.pkl"
if serve_cal.exists():
    be.CAL_PKL = serve_cal      # ★存在すれば無条件で優先
```

**終了コード**: `0` = 正常 / `1` = CSV・モデル不在 / **`2` = 品質ゲート不合格**
（bundle は書き出すが push しない）。

---

## §7 `betting_judgment.py`（256 行）/ `umami.py`（224 行）

### 7.1 `betting_judgment` の定数

```python
CHAOS_HARD_MAX  = 0.30    # chaos_pct ≤ これ → 固い
CHAOS_ROUGH_MIN = 0.70    # chaos_pct ≥ これ → 荒れ
VALUE_EV_MIN    = 1.10    # 妙味馬の EV 下限
VALUE_PWIN_MIN  = 0.05    # テール除外
VALUE_BAND      = 0.20    # ±20%（export_marks_json の ai_vs_market と同じ）
```

| 関数 | 契約 |
|---|---|
| `chaos_to_pct(raw, key)` | 分位表が無ければ **生値を返す**（★`compute_bets.pct()` は `None` を返す。**挙動が違う**） |
| `classify_hardness(chaos_pct)` | `None` → `"標準"` |
| `vs_market_status(model_p, odds)` | `under`/`over`/`fair`/`unknown` |
| `extract_value_horses(horses)` | 妙味馬。UMAMI ゲート通過が必須。**UMAMI (xROI) 降順**でソート |
| `decide_strategy(hardness, has_value)` | 6 通りの dict |
| `build_judgment(race_confidence, horses)` | bundle 埋込用の集約 |

**⚠️ 不整合（レビュー対象）**: 分位表破損時の挙動が
`betting_judgment.chaos_to_pct`（生値フォールバック）と
`compute_bets.pct`（`None` → 見送り）で異なる。
同じ事故（テーブル破損）で bundle 側は「固い」と誤判定し、買い目側は見送りになる。

### 7.2 `umami` の定数と契約

```python
P_WIN_FLOOR   = 0.04    # 単勝: 勝率これ未満は「来ない馬」
P_SHO_FLOOR   = 0.12    # 複勝
ODDS_HARD_CAP = 50.0    # 単勝オッズこれ超は実測最悪帯 → 常時ゲート
MIN_CELL_N    = 300     # 参照セルの最小標本
EV_EDGES  = [0.8,0.9,1.0,1.1,1.3,1.5,2.0]   # 8 ビン
FAV_EDGES = [3.0,7.0,15.0,50.0]              # 5 帯
GRADE_EDGES = [(0.85,"S"),(0.80,"A"),(0.72,"B")]
```

`umami(kind, p, odds, tansho_odds=None) -> dict`:
```
{xroi, ev, grade, gated, gate_reason, cell_n}
参照: reports/audit_ev_bin_roi.json の by_ev_x_fav[f"{ev_bin}|{fav_bin}"]
セル n < 300 → by_ev_bin[ev_bin] へフォールバック
テーブルが無ければ xroi=None のまま返す（fail-soft）
```

**⚠️ `_tables()` は `functools.lru_cache(maxsize=1)`** — プロセス内で 1 回だけ読む。
`audit_ev_bin_roi.json` を更新したらプロセス再起動が必要。

---

## §8 `compute_bets.py`（1,008 行）— 馬券構築エンジン

### 8.1 モジュール定数

```python
ENGINE_VERSION = "2026-08-09"
BUDGET, MIN_BET, MAX_BET = 10000, 500, 7000
HOSEI_MARK5 = ["◎","〇","▲","△","△"]
KIND_ORDER  = {単勝:0, 複勝:1, ワイド:2, 馬連:3, 馬単:4, 三連複:5, 三連単:6}
TH_TOP1_GO/OK       = 0.75 / 0.50
TH_TOP2_GO/OK/LOW   = 0.75 / 0.50 / 0.40
TH_CHAOS_HARD/MID   = 0.75 / 0.50
TH_MARKET_ANABA     = 0.30
CHAOS_RAW_SKIP      = 0.92
CLEAN_BAND_MAX      = 0.33      # §0b（配線撤回、機構のみ）
DEMOTE_BUDGET       = 2000      # 同上（main から渡していない）
FUKU_HIT_THR        = 0.21      # serve 分布に校正済（offline 値は 0.36）
AITE_WEAK_TH        = 0.252     # serve 分布に校正済（発見期は 0.328）
ODDS_CAP = {"馬連":50.0, "ワイド":50.0, "馬単":200.0}   # shape 経路のみ
ANA_TAN_ODDS_CAP = 30.0
SETTLE_DRIFT_TAN/FUKU/WIDE = [(上限, 倍率), ...]
```

### 8.2 遅延ロード（グローバルキャッシュ）

| 関数 | ロード先 | 失敗時 |
|---|---|---|
| `_blend_lambda()` | `data/t10_blend.json` → `lambda` | `None`（補正印無効、fail-soft）。sentinel は `-1.0` |
| `_harville_lambda()` | `data/harville_lambda.json` | `(0.8405, 0.7542)` の既定値 |
| `_qtab()` | `data/chaos_quantiles.json` → `quantiles` | `{}` → `pct()` が `None` を返す → **見送り** |

**⚠️ すべてモジュールレベルのグローバル変数にキャッシュされる**。
長時間プロセス（NiceGUI 等）で JSON を更新しても反映されない。

### 8.3 主要関数

#### `pl_pair_probs(horses) -> (dict, dict)`（`:114`）
```
入力: horses[] の umaban / p_win（>0 のみ）。3 頭未満なら ({}, {})
p を正規化 → q1 = p^λ1, q2 = p^λ2
全 (a,b) について pab = p_a · q1_b / (S1 − q1_a)      → um[(min,max)] += pab
その中で全 c について pabc = pab · q2_c / (S2 − q2_a − q2_b)
   → (a,b),(a,c),(b,c) の 3 ペアに wd[...] += pabc
戻り値の key は **馬番のタプル (min, max)**（index ではない）
計算量 O(N³)。N=18 で 4,896 回
```
**レビュー観点**: `wd` の合計は理論上 3.0 になるはずだが、λ 補正により
正規化が崩れる（λ=1 のときのみ厳密に 3.0）。**較正器も通していない**（Vol. III P2-4）。

#### `pct(raw, key)`（`:226`）
```python
t = _qtab().get(key)
if not t or len(t) < 2: return None        # ★fail-safe（旧実装は生値返しで事故）
if raw <= t[0]: return 0.0
if raw >= t[-1]: return 1.0
i = bisect_right(t, raw); lo, hi = t[i-1], t[i]
return (i - 1 + (raw-lo)/(hi-lo)) / (len(t)-1)
```

#### `allocate(weights, budget, mn=500, mx=7000)`（`:260`）
```python
amts = [clip(round(budget*w/Σw/100)*100, mn, mx) for w in weights]
for _ in range(6000):
    d = budget - sum(amts)
    if d == 0: break
    step = ±100
    重み降順（d>0）/ 昇順（d<0）に見て、[mn,mx] に収まる最初の要素を ±100
    どれも動かせなければ break                # ★満額にならず終了しうる（意図的）
```
**レビュー観点**:
- `n == 0` は `[]` を返す（ゼロ除算なし）
- 全要素が `mx` に張り付くと `d > 0` のまま `break` → **予算未消化**。
  これは「キャップで埋まらない薄いレースは満額にしない = -EV に突っ込まない規律」（意図的）
- 6,000 回のループ上限は `budget/100 = 100` 回程度で収束するため十分

#### `load_live_odds(live_dir, rid16, max_age_min) -> (data|None, reason)`（`:279`）
```
ファイル無し / JSON 破損 / ok=false / 鮮度 NG → (None, 理由)
fetched が ISO でパースできない、または無い → ファイル mtime で代用
```

#### `hosei_marks(horses) -> list|None`（`:66`）
```
λ = _blend_lambda()。None なら補正印なし
umaban / p_win / tansho_odds がすべて有効な馬のみ対象。5 頭未満なら None
s_inv = Σ(1/odds)     # de-vig 正規化（レース内定数なので順位不変だが明示）
sort key = -(log(p_win) + λ·log((1/odds)/s_inv))
上位 5 頭に ◎〇▲△△、orig_mark に元の印を保持
```

#### `compute_race_bets(race, live_dir=None, max_age_min=20.0, budget=10000, force_floor=False, demote_budget=None, engine=None) -> dict`（`:357`）

Vol. II §2.3 のフロー。**この関数は `build_site.build_tact()` からも呼ばれる**
（`budget=10000, force_floor=True`）。

戻り値のキー: `race_id, race_label, race_nature, race_reason, confidence?, bets[], hosei_marks?`

#### `apply_to_bets_json(date_str, computed, stamp=None) -> Path`（`:823`）
```
既存ファイルを .bak（重複時は .bak2, .bak3...）へ退避
raw = json.load(...)。wrapper 形式 {bets:[...]} とレガシー array の両対応
race_id 一致 → 置換（★advisor は温存）/ 無ければ追加
.tmp に書いて replace()（アトミック置換）
stamp = {model, engine, engine_version, mode, live, stamped_at}
```

### 8.4 shape 経路の内部関数（`compute_race_bets` 内のクロージャ）

| 関数 | 役割 |
|---|---|
| `fld(b, k)` | `by_ban[b][k]` を float で。無ければ `None` |
| `umaren(i, j)` | `umaren_matrix["min-max"]` |
| `pair_p(i, j, kind)` | bundle の `pair_probs` から `wide`/`umaren`/`umatan` |
| `push(kind, sel, bans, odds, ev, boost)` | 候補追加。ODDS_CAP 適用 + **同一 (kind, sel) の dedup**（boost は強い方） |
| `c_tan / c_fuku / c_umaren / c_wide / c_umatan` | 券種別の候補生成。`pair_p` が `None` なら旧近似にフォールバック |
| `c_pair(i, j, boost)` | 馬連 + ワイドを**並行**生成 |

**candidate のタプル構造**: `[kind, sel, bans(tuple), odds, ev, is_hon(bool), boost]`
（`_p(c) = c[4]/c[3]` で `p_pair` を厳密復元する設計）

---

## §9 `validate_cowork_bets.py`（380 行）

| 関数 | 契約 |
|---|---|
| `find_hon(horses)` | `mark == "◎"` の最初の馬。**丸「○」は見ない**（bundle は全角「〇」で出るので現状は OK だが脆い） |
| `skip_reasons(race_meta, race_conf, hon)` | 見送り 4 条件のリスト。空なら買い対象 |
| `_parse_umaban(sel)` | `re.findall(r"\d+", sel)`。`"3-7"→[3,7]` / `"16→1"→[16,1]` |
| `content_issues(bet, valid_umaban)` | 券種・馬番実在・金額（正 / 100 円単位 / ≤10,000）の違反リスト |
| `race_bet_total(race)` | 購入額合計 |
| `load_bundle_index(bundle_path)` | `{race_id: race}` |
| `backup_path(path)` | `.bak` → `.bak2` → ... のユニーク名 |

**定数**:
```python
CHAOS_SKIP = 0.92; FIELD_SIZE_SKIP = 7; PWIN_SKIP = 0.05
ALLOWED_KINDS  = {単勝, 複勝, ワイド, 馬連, 馬単, 三連複}   # ★馬単が残存（Vol. III P2-9）
REJECTED_KINDS = {三連単}
BET_UNIT = 100; MAX_BET_PER = 10000
```

**`--apply` の 2 段処理**:
1. 見送り違反 race → `bets=[]` / `race_nature="見送り"` / `race_reason` に `[自動見送り: ...]` を前置
2. 内容違反 bet を全除去 + 重複の 2 個目以降を除去
最後に `.bak` 退避 → 上書き。

---

## §10 `t10_runner.py`（677 行）/ `jvlink_odds.py`（208 行）

### 10.1 `t10_runner` の主要関数

| 関数 | 役割 |
|---|---|
| `notify(text)` | Discord webhook。**User-Agent 必須**（既定 UA は CF に 403）。1,990 字で切る |
| `class BotPoller` | Discord Bot API で新着を取得。起動時の最新 id を起点にし、**過去メッセージには反応しない** |
| `parse_budget_command(text)` | `「金額2000円」/「2000円」/「¥2000」/「2000」/「3,000円」` → int。`500 ≤ v ≤ 100000` |
| `acquire_lock(force)` / `release_lock()` | `reports/live_odds/.t10_lock`。12h 未満なら多重起動を拒否 |
| `keep_awake(on)` | `SetThreadExecutionState(ES_CONTINUOUS \| ES_SYSTEM_REQUIRED)` |
| `build_schedule(date, races, lead_min)` | `(post_dt, rid16, label)` のリスト + 発走時刻不明の missing |
| `parse_hhmm(s)` | `'15:40'/'1540'/'15:40:00'/'15時40分'` → `(15,40)` |
| `load_post_times(date)` | 週次 CSV から `{rid16: 'HH:MM'}`（列名は「レースID」「発走」を含む列を自動検出） |
| `run_cmd(cmd, timeout=180)` | `subprocess.run(env=PYTHONUTF8=1, capture_output=True)` → `(rc, out)` |
| `brain_tickets(...)` | 🟠 **dead code**（`import gutchi_brain` が必ず失敗、Vol. III P2-1） |
| `render_brain(brain)` | 同上 |
| `show_race_bets(date, rid16, brain=None)` | bets.json を読み戻して表示 + Discord + ビープ |
| `ensure_plan(date)` | `reports/bet_plan/{date}.json` が無ければ `build_bet_plan.py` を実行（失敗しても続行） |
| `process_race(...)` | 1 レース分の 3 ステップ（オッズ → compute_bets → validate） |

**CLI**: `date`（省略時は最新 bundle） / `--lead-min 10` / `--max-age-min 20` / `--dry` /
`--once rid16` / `--poll-until HH:MM` / `--list-schedule` / `--wait-bundle` /
`--wait-deadline 15:00` / `--force-lock` / `--test-notify`

### 10.2 `jvlink_odds.py`

| 関数 | 役割 |
|---|---|
| `fetch_records(race_key, spec, max_rec=200)` | `win32com` で `JVDTLab.JVLink` を Dispatch → `JVInit(SID)` → `JVRTOpen(spec, key)` → `JVRead` ループ。`size==0` で終了、`size<0` はファイル境界でスキップ |
| `parse_o1(rec)` | 単勝 `pos45/stride8` + 複勝 `pos269/stride12` |
| `parse_o3(rec)` | ワイド `pos40/stride17`、153 組 |
| `parse_o4(rec)` | 馬単 `pos40/stride13`、306 組 |
| `fetch_race(race_key)` | 3 spec を集約 → overround チェック → `{ok, tansho, fukusho, wide, umatan, overround_tan}` |
| `dump_raw(race_key)` | パーサ検証用の生録ダンプ |

`SID`: `data/jvlink_sid.txt` の 1 行目、無ければ `"UNKNOWN"`。

---

## §11 `build_bet_plan.py`（240 行）

| 関数 | 役割 |
|---|---|
| `conf_score(rc)` | `0.45·top2_conc + 0.30·top1_dom + 0.25·(1 − chaos)` |
| `load_margins(date)` | 週次 CSV から `{(rid16, ban): (前走確定着順, 前走着差タイム)}`。失敗時は `{}` |
| `margin_bonus(margins, rid, horses)` | ◎が前走 1 着かつ着差 < 0 なら `0.08 × min(|着差|, 1.0)` |
| `load_races(date)` | bundle → 平坦な dict のリスト |
| `miokuri_reason(r)` | 見送り 4 条件（うち 3 つ。◎オッズ null は見ない） |
| `jun_yen(rank, n)` | 準勝負の金額を 8,000 → 5,000 に線形補間、500 円丸め |
| `_row(r)` | 出力用に 19 キーを抜き出す |

**定数**: `SHOBU_MAX=3, SHOBU_YEN=10000, JUN_MAX=4, JUN_YEN_HI/LO=8000/5000,
SHOUKA_YEN_HI/LO=3000/1000, DAY_MIN_RACES=10, DAY_MIN_YEN=100000, MARGIN_BONUS_W=0.08`

**⚠️ 不整合**: `miokuri_reason` は `chaos / field_size / pwin_top` の 3 条件しか見ない
（`compute_bets` の「◎の tansho_odds が null」に相当する条件が無い）。
`pwin_top` も「◎の p_win」ではなく「レース内 p_win 最大値」を使っている。
→ 見送り判定が 3 実装で微妙に違う（Vol. III P2-5）。

---

## §12 `generate_results.py`（1,170 行）

| 関数 | 役割 |
|---|---|
| `parse_date_to_key('2026.2.22')` | `'20260222'` |
| `parse_haitou(v)` | 括弧付き・nan・空文字は 0.0 |
| `_safe_num(v, default=0.0)` | **`float(x or 0)` は NaN を返す**（NaN は truthy）ので専用関数 |
| `_safe_round(v)` | NaN セーフな `int(round(...))` |
| `split_combos(bet_str)` | `'1-2 / 3-4'` → `[{1,2},{3,4}]` |
| `load_kekka_all()` | `data/kekka/*.csv` を全部読んでキャッシュ |
| `get_race_kk(cache, date_key, place, r_num)` | 該当レースの DataFrame |
| `get_top3 / get_top2 / get_winner / get_top3_ordered / get_cancelled` | 着順抽出 |
| `get_payout_*` | 券種別の払戻抽出（tansho / fukusho / rengo / umatan / sanrenpuku / sanrentan / wide） |
| `_parse_wide_combos(combo_str)` | ワイド払戻文字列のパース |
| `load_wide_payouts()` / `get_wide_cache()` | parquet + 週次 CSV の統合 |
| **`_bet_cis(settled, n_boot=2000, seed=42)`** | Wilson CI（的中率）+ bootstrap CI（投資加重 ROI）+ `roi_verdict` |
| `match_cowork_bet(bet, race_kk, ...)` | `(hit, payout_per_100, refund_ratio)` |
| `parse_race_id_16(rid)` | 場所コード → 場所名、R 番号 |
| `_iter_cowork_race_dicts()` | per-race 形式と bundle 形式の両方を走査 |
| `aggregate_cowork_bets(kekka_cache)` | 実運用集計。`total / by_type / by_place / weekly / races / bets` |

**`_bet_cis` の詳細（レビュー重要）**:
```python
cost = (settled["購入額"] - settled["返還"]).to_numpy(float)   # 実効投資
ret  = settled["払戻"].to_numpy(float)
# Wilson（的中率）
z=1.96; p=hits/n; denom=1+z²/n
center=(p+z²/(2n))/denom; half=z·√(p(1-p)/n + z²/(4n²))/denom
# bootstrap（ROI）: bet 単位で n 個を復元抽出 × 2000 回
idx = rng.integers(0, n, size=(n_boot, n))
rois = ret[idx].sum(1) / cost[idx].sum(1) * 100     # cost>0 のみ
lo, hi = percentile(rois, [2.5, 97.5])
verdict = above_takeout if lo>80 else below_takeout if hi<80 else inconclusive
```
**レビュー観点**: bet 単位のリサンプルなので**同一レース内の相関を無視している**
（同じレースの複数点は独立でない）。レース単位ブロックブートストラップの方が
保守的（CI が広がる）。現状の CI は **やや楽観的**である可能性がある（🟡 未修正）。

---

## §13 `build_site.py`（1,251 行）

| 関数 | 役割 |
|---|---|
| `deep_zen(o)` | NFKC 正規化（半角カナ→全角、全角英数→半角）。**serve 経路には無い** |
| `waku_of(umaban, field_size)` | JRA の枠順割当ルールで馬番 → 枠番 |
| `parse_weekly(date_str)` | 週次 CSV → `(race_info, horse_info)`。列インデックス `IDX_46` / `IDX_49` を使う |
| `classify_style(history)` | 過去走の脚質 → 逃げ/先行/差し/追込 |
| `pairs_top(race, top_n=8)` | ペア確率の上位 |
| `parse_wide_kekka()` / `parse_kekka(date, wide)` | 結果・払戻 |
| `_parse_one_cowork_file` / `load_all_cowork` | cowork_output の走査（json / txt / md） |
| `load_all_grade_scope` | Grade Scope |
| `load_course_stats` / `_slim_course` | コース分析タブ |
| `load_pedigree_index` / `ped_course_entry` | 血統タブ |
| `pick_training_file` / `parse_training_file` | 調教 |
| `_combos(selection, n, ordered)` | 買い目文字列 → タプル列 |
| **`build_tact(race)`** | `compute_bets.compute_race_bets` を呼んで公開買い目を作る。**金額は出さず、理由文からオッズ表記を正規表現で除去** |
| `settle_bet(btype, selection, cost, res)` | 1 bet の決済。ワイド払戻未取込は `settled=False` |
| `_level_norms` / `_raw_to_100` / `_level_tier` / `horse_level` | 馬レベル（近走成績ベース。ZI/補正は撤去済み） |
| `compute_member_level(klass, horse_levels)` | メンバーレベル |
| `transform_bundle(path, cowork, wide_data, ...)` | 日別 view-model の本体 |
| `build_results_json()` | 成績ビュー |
| **`_scrub_text(s)` / `scrub_public(day)`** | **JRA-VAN ガイドライン対応の公開前フィルタ**（オッズ生値・払戻・EV 等を除去） |

**`_TACT_ODDS_RE`**:
```python
re.compile(r"[（(](?:複[\d.]+|[^（）()]*?[\d.]+倍)[)）]")
```
旧 brain の `（複X.X）` と topdown の `（券種 X.X倍）` の両対応。

---

## §14 `optuna_v6_marks.py`（463 行）— 学習

| 関数 | 役割 |
|---|---|
| `load_winner_tansho_pay()` | `kekka_*.csv`（11 列固定）から `{rid_s: 勝ち馬単勝配当}` |
| `prep()` | master_v2 読込 → `label = clip(6-着順,0,5)` → train/valid 分割 → LabelEncoder fit（**train のみ**）→ `feats = columns − LEAK_COLS − label` |
| `make_dataset(d, feats, alpha)` | `lgb.Dataset(X, label, group, weight)`。`group` は `COL_RID` でソート後の `groupby` サイズ。`w = 1 + α·log1p(winner_tansho/100)` |
| `ndcg_at_k(label, score, k=5)` | 自前実装 |
| `evaluate_marks_with_ece(vl_scored, p_threshold=0.10)` | 印指標 + `ECE_high_p`（**単一ビン**、標本 ≤100 なら 0.0） |
| `composite_score(metrics)` | Vol. I §5.2 の式 |
| `compute_cv_composite(model, vl_df, X_vl)` | **valid 内レース ID の 5-fold KFold**（時系列ではない） |
| `objective(trial)` | HP 探索 → `lgb.train(early_stopping(100))` → composite |
| `main()` | study → best params で `num_boost_round = best_iter × 1.1` 再学習 → pkl + json 保存 |

**`_CACHE`**: モジュールレベル dict に `tr` / `vl_df` / `X_vl` / `ds_vl` / `feats` を保持
（trial ごとの再読込を避ける）。

---

## §15 横断的な実装パターンと落とし穴

### 15.1 頻出パターン

| パターン | 例 | 意味 |
|---|---|---|
| wrapper / legacy 両対応 | `raw["bets"] if isinstance(raw, dict) and "bets" in raw else raw` | bets.json が dict と array の両形式を取りうる。**5 箇所以上に散在**（共通化候補） |
| rid16 正規化 | `re.sub(r"\D", "", str(x))[:16]` | レース ID の揺れ吸収 |
| UTF-8 強制 | `sys.stdout.reconfigure(encoding="utf-8")` / `io.TextIOWrapper(sys.stdout.buffer, ...)` | Windows cp932 対策。**2 回ラップすると旧ラッパが GC で close されて死ぬ**（`build_pl_calibrators_serve.py` にコメントあり） |
| アトミック書込 | `.tmp` に書いて `.replace()` | bets.json / shadow.json |
| ユニーク退避 | `.bak` → `.bak2` → ... | 既存 .bak を潰さない |
| 遅延 import | `from umami import umami_for_horse`（関数内） | 循環回避 + stdlib 方針の維持 |
| lru_cache | `_quantile_tables` / `_tables` / `load_halo_thresholds` | プロセス内 1 回。**hot-reload できない** |

### 15.2 レビュー時に注意すべき落とし穴

| # | 落とし穴 |
|---|---|
| 1 | **印の文字**: 全角「〇」(U+3007) と丸「○」(U+25CB) が混在。`compute_bets.py:431` のみ正規化している |
| 2 | **列名の全角/半角**: `Ｒ`（master）vs `R`（parse_csv）。**実害あり**（Vol. III P0-4） |
| 3 | **`float(x or 0)` は NaN で NaN を返す**（NaN は truthy）。`generate_results._safe_num` が対策 |
| 4 | **`pd.to_numeric(errors="coerce").fillna(-9999)`** — CAT_COLS に無い文字列列は全部 −9999 |
| 5 | **分位表破損時の挙動が 2 実装で違う**（`compute_bets.pct` は None / `betting_judgment.chaos_to_pct` は生値） |
| 6 | **見送り 4 条件が 4 箇所に独立ハードコード**、かつ `build_bet_plan` は 3 条件しか見ない |
| 7 | **serve の閾値は offline 値をそのまま使えない**（`FUKU_HIT_THR` / `AITE_WEAK_TH` はどちらも serve 分布で再校正済み）。新しい閾値を導入するときは必ず serve 実分布で発火率を確認すること |
| 8 | **λ はモデル世代を跨ぐと壊れる**（v5 期のデータでは符号が逆） |
| 9 | **`import gutchi_brain` は必ず失敗する**（ファイル削除済み） |
| 10 | **`compute_bets` はライブラリとしても使われる**（`build_site.build_tact`）。副作用を足すと公開層が壊れる |
| 11 | **`parse_csv` は約 30 秒かかる**（調教 CSV 600 万行）。ループ内で呼ばないこと |
| 12 | **bootstrap CI がレース内相関を無視**（bet 単位リサンプル）→ CI がやや楽観的 |

### 15.3 テストの現状

```
tests/test_production_line.py   本番ラインの純関数ゴールデンテスト（合成入力）
tests/test_backtest.py          floor_to_unit / get_actual_payout
tests/test_ensemble.py          assign_marks / ensemble_predict
tests/test_kelly.py             kelly_fraction
tests/test_utils.py             ユーティリティ
→ 73 passed / 20.5s
```

**カバレッジの穴**:
- `predict_weekly.parse_csv`（本番の入口）にテストが無い ← Vol. III P0-1/P0-2 を見逃した原因
- `compute_bets.compute_race_bets` の topdown 経路のゴールデンテストが薄い
- `serve_history_feats.compute_row_feats` と学習側定義の一致を検証するテストが無い
  （`analysis/validate_serve_history_feats.py` は存在するが CI に入っていない）

**推奨追加テスト**:
```python
# tests/test_serve_parse.py
def test_parse_csv_no_constant_stamping():
    df = parse_csv(Path("data/weekly/<最新>.csv"))
    for col in ["jockey_fuku90", "trainer_fuku90", "horse_fuku10",
                "前走馬体重", "前PCI", "前走RPCI"]:
        assert df[col].nunique() > 1, f"{col} が定数刷り込みされている"

def test_serve_gain_coverage():
    """serve で失われる gain の合計が閾値を超えないこと"""
    ...  # Vol. III §10.3 のスニペット参照。現状 28.15% → 目標 <10%
```

---

**← [Vol. I](VOL1_SYSTEM.md) / [Vol. II](VOL2_BETTING_OPS.md) / [Vol. III](VOL3_VALIDATION_AND_OPEN_PROBLEMS.md)**
