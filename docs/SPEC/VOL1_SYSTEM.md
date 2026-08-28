# PyCaLiAI 完全仕様書 Vol. I — システム仕様

> 版 1.0 / 2026-08-23 / 実測ベース
> 対象: データ層・特徴量層・モデル層・確率層・印層・serve 層・出力スキーマ
> 馬券構築と運用は Vol. II、検証史と課題は Vol. III

---

## 目次

- §0 この巻の読み方
- §1 ドメイン前提（これを知らないとコードが読めない）
- §2 システム全体アーキテクチャ
- §3 データ層
- §4 特徴量層（120 特徴の完全目録）
- §5 モデル層（unified_rank_v6）
- §6 学習プロトコルと版管理
- §7 確率層（Plackett-Luce / λ補正 / キャリブレーション）
- §8 印層（marks / race_confidence / buy_judgment / UMAMI）
- §9 serve 層（本番推論経路とその欠損構造）
- §10 出力スキーマ（bundle.json）完全定義
- §11 環境・依存・実行コマンド
- §12 ファイル索引

---

## §0 この巻の読み方

PyCaLiAI は「機械学習モデル」ではなく **6 層のパイプライン**である。層ごとに
目的・評価指標・失敗モードが違い、**上の層の改善が下の層の改善を意味しない**。
この非推移性こそが本プロジェクトの中心的教訓であり、Vol. III の全内容の前提になる。

```
[L1] データ層     : TARGET/JRA-VAN 由来 CSV → master_v2 (626,774行 × 132列)
[L2] 特徴量層     : 132列 − LEAK 12列 = 120特徴
[L3] モデル層     : LightGBM LambdaRank → レース内 raw score
[L4] 確率層       : PL 変換 → Isotonic 較正 → p_win / p_plc / p_sho / pair_probs
[L5] 印層         : score 降順 top-5 に ◎〇▲△△ + race_confidence + buy_judgment
[L6] 馬券層       : (Vol. II) topdown エンジン → 買い目 → 人間が IPAT 投票
```

**評価の非推移性（Fact, Vol. III §2 に証拠）**:
- L3/L4 の精度は天井（◎top3 ≈ 62%）に達しており、**L1/L2 への投資は 1400 特徴規模でも 0 リターン**。
- L4 の較正はほぼ完璧（ECE 0.001–0.019）だが、**ROI は控除率の壁（≈80%）を超えない**。
- 従って改善余地は L6（意思決定）と ops にしか残っていない。

---

## §1 ドメイン前提

### 1.1 対象

日本中央競馬会 (JRA) の平地競走。1 レース 5〜18 頭（フルゲート 16 or 18）。
年間約 3,400 レース、10 会場（札幌・函館・福島・新潟・東京・中山・中京・京都・阪神・小倉）。

- **障害競走は学習・推論の両方から除外**（`predict_weekly.parse_csv` が除外、ログ「障害除外済」）。
- **三連単・馬単は本プロジェクトで廃止**（Vol. II §1.4）。

### 1.2 パリミュチュエル方式（最重要の構造的制約）

日本の公営競技は **pari-mutuel（賭け金プール分配）** であり、ブックメーカー方式ではない。
帰結が 3 つあり、これが本プロジェクトの戦略空間をほぼ決めている。

| 帰結 | 内容 | 影響 |
|---|---|---|
| **控除率が固定の床** | 単勝/複勝 20%、馬連/ワイド/馬単 22.5%、三連複 22.5%、三連単 27.5% がプールから引かれる | 全馬に均等に賭けると ROI は必ず 80%（or 77.5%）。**期待値中立が「上限」ではなく「床」** |
| **オッズが確定するのは発走後** | 買った時点のオッズでは払われない。締切時点のプール比で決まる | **CLV (Closing Line Value) が換金不能**。ブックメーカー戦略の主要概念が丸ごと使えない (Vol. III §3.4) |
| **自分の賭け金がオッズを動かす** | 大口はプールを希釈する | 小口個人（1R ¥10,000）では無視可能。ただし合成オッズ裁定は成立しない |

**この 3 点から導かれる本プロジェクトの目標設定（Fact, `project_roi_max_doctrine`）**:
> 目標は「儲ける」ではなく「**最も負けない線**」。ROI 85–90% は無理筋であり、
> 罠をすべて切っても残 ROI ≈ 79.9% ≈ 控除率 = 期待値中立が床。

### 1.3 券種と払戻の定義

| 券種 | 的中条件 | 控除率 | 本プロジェクトでの扱い |
|---|---|---|---|
| 単勝 | 1 着馬 | 20% | 使用（topdown で確率 1 位・オッズ ≤30 のみ） |
| 複勝 | 3 着以内（5 頭立以下は 2 着以内） | 20% | **主力**（アンカー） |
| ワイド | 3 着以内に入る 2 頭の組 | 22.5% | **主力** |
| 馬連 | 1-2 着の組（順不同） | 22.5% | 使用（prob-first・オッズ ≤50） |
| 馬単 | 1-2 着の順序付き組 | 22.5% | **全廃**（実測 ROI 22.1%、Vol. II §11） |
| 三連複 | 1-3 着の組（順不同） | 22.5% | 未配線（shadow 実験のみ） |
| 三連単 | 1-3 着の順序 | 27.5% | **廃止** |
| 枠連 | 1-2 着の枠の組 | 22.5% | 検定済み・不採用 |
| WIN5 | 指定 5 レースの 1 着を全的中 | 27.5% | 検定済み・死亡 |

**複勝オッズは下限/上限のレンジで表示される**（どの馬が来るかで払戻が変わるため）。
本システムは `(low + high) / 2` を代表値に使う（`compute_bets.py`）。

### 1.4 データ提供元と法的制約

| 源 | 内容 | 制約 |
|---|---|---|
| **TARGET frontier** | 出走表 / 結果 / 過去 5 走 / 着度数 / 補正タイム / 調教 の CSV エクスポート | 手動エクスポート。列構成が予告なく変わる（**§3.6 の障害モード**） |
| **JV-Link (JRA-VAN Data Lab)** | リアルタイムオッズ (0B31/0B33/0B34)、当日変更情報 | **32-bit COM のみ**。この PC でしか動かない。SID 登録は現在 `UNKNOWN`（個人利用扱い） |
| `E:\競馬過去走データ\` | 調教マスター (H 520万行 / W 70万行, cp932)、全頭確定単勝オッズ | **不可侵（読み取り専用）**。ここには書かない |

**JRA-VAN 投稿ガイドライン準拠（Fact, 2026-07-31 対応済）**:
公開サイトから **調教タイム生値 / オッズ生値 / 払戻金額 / EV / ライブ馬体重 / T-15 補正印** を撤去済み。
新しいデータをサイトに出す時は必ずこの基準に照らすこと。撤去処理は `build_site.py` の
`scrub_public` 系および `_TACT_ODDS_RE`（買い目理由からオッズ表記を除去）に一元化。

---

## §2 システム全体アーキテクチャ

### 2.1 役割分担（この分離が設計の核）

| レイヤー | 実体 | 責務 | 出力 |
|---|---|---|---|
| **予測** | `export_weekly_marks.py` + `unified_rank_v6.pkl` | 印付け (◎〇▲△△) と確率算出。**馬券は組まない** | `reports/cowork_input/{date}_bundle.json` |
| **馬券構築** | `compute_bets.py` (T-10 自動) | 当日ライブオッズで買い目・金額を決定 | `reports/cowork_output/{date}_bets.json` の `bets[]` |
| **narrative** | Cowork (Claude Desktop) | 論評（advisor / Grade Scope）専用。**買い目は絶対に書かない** | 同ファイルの `advisor` / `grade_scope` |
| **ガード** | `validate_cowork_bets.py` | 見送り条件と内容の**コード強制** | 同ファイルの in-place 矯正 |
| **表示** | `build_site.py` + `site/` (静的サイト) | 表示専用 | `site/data/{date}.json` |
| **執行** | **人間** | IPAT で投票 | — |

**自動投票は存在しない。** compute_bets は買い目提示までで、金銭の自動執行機能を持たない。

### 2.2 データフロー（本番ライン）

```
                  ┌──────────────── 土曜朝 (Phase A) ────────────────┐
data/_inbox/*.csv ─ place_weekly.py ─→ data/weekly/{date}.csv
                                       data/kako5/{date}.csv
                                       data/tyaku/{date}.csv
                                       data/training/{H,W}-*.csv
                                            │
                                            ├─ make_weekly_hosei.py ─→ data/hosei/H_{date}.csv
                                            ↓
                              export_weekly_marks.py --model v6
                                 ├ predict_weekly.parse_csv       (§9.2 欠損構造の発生源)
                                 ├ _SERVE_RENAME                  (§9.3)
                                 ├ serve_history_feats.fill_*     (§9.4)
                                 ├ unified_rank_v6.pkl → raw score
                                 ├ pl_probs (PL 厳密)  → p_win/p_plc/p_sho
                                 ├ pl_calibrators_v6_serve.pkl → 較正
                                 ├ 印 ◎〇▲△△ + race_confidence
                                 ├ betting_judgment.build_judgment (妙味馬 / UMAMI)
                                 ├ marks_shap → why (SHAP top-6)
                                 ├ kako5_summary → history / horse facts
                                 ├ class_prior_v6.json → race_meta.class_prior
                                 └ 品質ゲート + serve canary  (§9.5, exit 2 で push 停止)
                                            ↓
                          reports/cowork_input/{date}_bundle.json
                                            │
                    ┌───────────────────────┴────────────────────────┐
                    ↓                                                ↓
        Cowork (Claude Desktop)                          当日 T-10 (Vol. II §6)
        narrative のみ                                   jvlink_odds.py (32-bit)
                    ↓                                    → reports/live_odds/{rid16}.json
    reports/cowork_output/{date}_bets.json  ←────────────  compute_bets.py --race --apply
        (advisor 部)                                            ↓
                    └──────────────────────→  validate_cowork_bets.py --apply
                                                                 ↓
                          ┌────────────── 日曜夜 (Phase C) ──────────────┐
                          data/kekka/{date}.csv → generate_results.py
                             → data/results.json / data/cowork_results.json
                             → build_horse_history.py → data/_horse_history.parquet
                                            ↓
                              build_site.py → site/data/*.json
                              sync-hf-umami.ps1 → HF Docker Space / pycaliai.com
```

### 2.3 副系統（本番ではないが残存している）

| 系統 | 実体 | 状態 |
|---|---|---|
| 旧 8 モデルアンサンブル | `predict_weekly.py` (2,023 行) | Streamlit 用。**Phase A ではデフォルト SKIP**（`-WithPredict` で opt-in）。ただし `parse_csv` は本番が依存している（§9.2） |
| Streamlit UI | `app.py` | Cloud 用。`strategy_weights.json` を読む |
| NiceGUI | `nicegui_app.py` | 旧本番。`sync-hf.ps1` で併行更新中 |
| rule-based 戦略 | `strategy_weights.json` | Streamlit のみ参照。**構造的循環（test で採用→test で評価）が既知**（Vol. III §6.3） |

⚠️ **重要**: `predict_weekly.py` は「旧系統」と分類されているが、その `parse_csv()` は
`export_weekly_marks.py:57` が import しており、**本番の入力パースそのもの**である。
「旧系統だから触らなくてよい」は誤り。§9.2 の欠陥はすべてここに存在する。

---

## §3 データ層

### 3.1 master の 4 段構成（Fact, 実測）

| 段 | スクリプト | 出力 | 実測サイズ |
|---|---|---|---|
| 1 | `build_dataset.py` | `data/master_20130105-20251228.csv` | 412 MB |
| 2 | `parse_kako5.py --mode master` | `data/master_kako5.csv` | 469 MB |
| 3 | `build_master_v2.py` | `data/master_v2_20130105-20251228.csv` | **516 MB / 626,774 行 × 132 列** |
| 4（推論） | `make_weekly_hosei.py`, `parse_kako5.py --mode weekly`, `parse_training.py` | `data/hosei/H_{date}.csv` 等 | 週次 |

**split 分布（実測）**: `train=485,252` / `valid=47,273` / `test=94,249`（合計 626,774）。

### 3.2 `build_master_v2.py` の 3 ステージ

| Stage | 内容 | 結合方式 | 実測カバレッジ |
|---|---|---|---|
| 1-3 | 補正タイム | `merge(on=["レースID(新)","馬番"])`、行数不変 assert | `prev_hosei` / `prev_hosei9` |
| 1-4 | 調教（坂路 H / ウッド W） | `merge_asof(by=馬名, direction=backward, allow_exact_matches=False)` | 坂路 ≈80% / WC ≈25%（2021〜、2022 以降 67%） |
| 1-5 | コース／騎手履歴 | `groupby.cumsum() − 自行`（as-of） | — |

**Stage 1-5 の定義（`build_master_v2.py:compute_history_features`）**:
- `course_key = 場所 | 芝ダ | 距離帯`。距離帯 = 短(≤1400) / マ(≤1700) / 中(≤2200) / 長(>2200)
- `course_n_prev = groupby([血統登録番号, course_key]).cumcount()` → 初出走 = 0
- `course_win_rate = (cumsum(is_win) − is_win) / n_prev`（n_prev>0 のときのみ、else NaN）
- `jockey_*` は **馬 × 騎手コードのペア** の累積（騎手単独の成績ではない点に注意）

**leak-safe 性（Fact）**: cumsum から自行を引くことで自レースの結果は入らない。
`merge_asof(allow_exact_matches=False)` により同日の調教も除外される。

### 3.3 ターゲット定義

- **学習ラベル**: `label = clip(6 − 着順, 0, 5)`（`optuna_v6_marks.py:113`）。
  1 着 = 5, 2 着 = 4, …, 5 着 = 1, 6 着以下 = 0。LambdaRank のグレード。
- `fukusho_flag = (着順 ≤ 3)` は `build_dataset.py:256` で作られるが **LEAK_COLS で除外**。
  すなわち **v6 は複勝を直接学習していない**（順位学習のみ）。正例率 ≈21.9% は複勝の base rate。
- `roi_target = 複勝配当 / 100` も同様に除外。
- **sample weight**: `w = 1 + α·log1p(勝ち馬単勝配当 / 100)`。α は Optuna 探索。
  v6 採用値 **α = 0.0308**（ほぼ無重み）、v5 は **α = 1.325**（テール較正崩壊で退役）。

### 3.4 キー・エンコーディングの約束

| 項目 | 規約 | 落とし穴 |
|---|---|---|
| レース ID | **`"レースID(新/馬番無)"`**（16 桁） | 旧 `"レースID(新)"` との揺れが CSV により存在。`_rid16()` = `re.sub(r"\D","",x)[:16]` で正規化するのが安全 |
| 馬 ID | `血統登録番号`（master のみ。週次 CSV には無い） | serve では **馬名 JOIN** に退化 → 同名馬の曖昧性が発生（`serve_history_feats` が父名 / 生年 ±1 で解決） |
| master エンコーディング | `utf-8-sig` | |
| TARGET 出力 CSV | `cp932`（shift_jis / utf-8 フォールバック） | |
| 週次 CSV の行形式 | レースヘッダ = **19 列**、馬行 = **33 / 46 / 48 / 49 / 99 列** | 列数で行種別を判定。**列数が変わると無言で全滅**（§3.6）。実測: 2026 の週次は **46 列**（48 列なら騎手/調教師コードが入る → Vol. III P0-1） |

### 3.5 データファイル一覧（役割つき）

```
data/
  master_v2_20130105-20251228.csv   ★本番マスター 516MB / 626,774行 × 132列
  master_20130105-20251228.csv       旧マスター（互換保持、削除候補）
  master_kako5.csv                   過去5走特徴量入り中間物
  kekka_20130105-20251228.csv        払戻マスター（11 列固定: rid_horse..sanrentan）
  kekka_20160105_20251228_v2.csv     払戻 v2（全頭行あり。※当たり行のみ配当が入る罠あり）
  payout_table.parquet               wide / 三連複 / 三連単 payout
  wide_payouts_2016-2025.parquet     ワイド払戻（2016-2025）

  weekly/{YYYYMMDD}.csv              ★週次入力（TARGET 出走表）
  kekka/{YYYYMMDD}.csv               週次結果・払戻
  kekka/wide_kekka.csv               ワイド払戻（2026〜、ユーザー手動配置）
  hosei/H_{YYYYMMDD}.csv             補正タイム週次
  kako5/{YYYYMMDD}.csv               過去5走詳細
  tyaku/{YYYYMMDD}.csv               着度数（馬体重含む）※§9.2 でパース失敗中
  training/{H|W}-*.csv               坂路 / ウッド調教 週次
  odds/OD{YYMMDD}.CSV                TARGET オッズ（単勝・複勝・馬連 matrix）
  _inbox/                            intake。place_weekly.py が自動振り分け

  chaos_quantiles.json               生値→パーセンタイル変換表（3 指標 × 101 点）
  harville_lambda.json               λ補正 PL の指数 (λ1=0.8405, λ2=0.7542)
  t10_blend.json                     T-10 補正印の λ=1.5 + 検証カーブ
  class_prior_v6.json                クラス×印の経験的中率（bundle 埋込用）
  serve_feature_baseline.json        serve canary の基準カバレッジ
  serve_code_maps.json               騎手名→コード (223) / 調教師名→コード (242)
  _horse_history.parquet             serve 履歴特徴の再計算源（build_horse_history.py）
  jockey_stats.csv / trainer_stats.csv  騎手・厩舎ローリング複勝率（※§9.2 で未使用）
  strategy_weights.json              ⚠️旧 rule-based 戦略（Streamlit のみ）
  cowork_results.json                実運用集計（generate_results.py が毎回 commit）
  live_results_2026.csv              2026 シーズン実績
```

### 3.6 既知のデータ破壊モード（実運用で実際に起きた）

| # | 事象 | 検知 | 対処 |
|---|---|---|---|
| D1 | **週次 CSV を Excel で開くと破壊**（レース ID が指数表記化 + 全行にカンマ padding）→ Phase A 全滅 | パース結果 0 レース | 第一選択は TARGET から再エクスポート。修復は `scripts/repair_excel_weekly_csv.py --inplace` |
| D2 | **TARGET の列数変更**で行が無言で捨てられる | `export_weekly_marks.py:513-519` の品質ゲート（bundle race 数が生 CSV レース数の 50% 未満で exit 2） | パーサの列数分岐を更新 |
| D3 | **着度数 CSV が 53 列**（パーサは 55 列を期待） | **検知されていない** — `_load_tyaku` が `None` を返し、静かに定数フォールバック | 🔴 **P0-2（Vol. III §5）** |
| D4 | `sync-hf` の往復でデータ消失 | worktree 常設化 + `pathspec-from-file` + add 実測ガードで 2026-07-29 修正済 | — |
| D5 | `git add` が「staged」表示のまま 0 件ステージ → 集計凍結 | `weekly_post.ps1` の `Invoke-GitAddVerified`（2 回リトライ後 fail-hard） | — |
| D6 | `cowork_results.json` の generated_at 凍結 | `weekly_nicegui.ps1 -Post` が当日日付を照合し Warn | Warn 止まり（Fail にすべき: 🟡 P2） |

---

## §4 特徴量層（120 特徴の完全目録）

### 4.1 特徴選択方式

**除外リスト方式**（`optuna_v6_marks.py:138`）:
```python
feats = [c for c in tr.columns if c not in LEAK_COLS and c != "label"]
```
master_v2 の 132 列 − LEAK_COLS 12 列 = **120 特徴**。
新しい列を master に足すと**自動的に特徴になる**（明示的ホワイトリストがない）。
これは v8 の affinity leak を許した構造でもある（Vol. III §3.6）。

**LEAK_COLS**（`optuna_v6_marks.py:68-73`）:
```
着順, fukusho_flag, roi_target, レースID(新), レースID(新/馬番無),
馬名, レース名, 発走時刻, date_dt, 日付, 血統登録番号, split
```

**CAT_COLS**（28 列、LabelEncoder、train のみで fit、未知値は `"__NaN__"`）:
```
場所, 芝・ダ, コース区分, 芝(内・外), 馬場状態, 天気, クラス名,
種牡馬, 父タイプ名, 母父馬, 母父タイプ名, 毛色, 馬主(最新/仮想), 生産者,
騎手コード, 調教師コード, 年齢限定, 限定, 性別限定, 指定条件, 重量種別, 性別,
ブリンカー, 前走場所, 前芝・ダ, 前走馬場状態, 前走競走種別, 前好走
```

**数値化規則**: `X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999)`
→ **CAT_COLS に入っていない文字列列は問答無用で -9999 になる**。これが `母馬` の死因（§4.3）。

### 4.2 三元表 — 特徴 × gain × serve カバレッジ

以下は **本番モデル `models/unified_rank_v6.pkl` の実測 gain 寄与率** と
**`data/serve_feature_baseline.json` の実測 serve カバレッジ**（2026-07-18〜07-26 の 4 週中央値）
を突き合わせたもの。**この表が本仕様書で最も重要な表である。**

#### 集計（実測 2026-08-23）

| 区分 | 特徴数 | gain 合計 | 意味 |
|---|---:|---:|---|
| **A. 学習で効き、serve でも生きている** | 75 | **71.85%** | 本当に本番で働いている部分 |
| **B. serve coverage < 0.40** | 34（gain > 0 は32） | **14.88%** | 🔴 train/serve skew。§9 と Vol. III §5 |
| **C. gain = 0（学習時点で死んでいる）** | 6 | 0.00% | 🟠 無駄特徴。うち 3 件は master が 100% NaN |

> **本番の unified_rank_v6 は、学習した gain の 14.88% を失った状態で推論している。**

#### A 群の上位（実際に働いている特徴）

| 特徴 | gain | 系統 | serve cov |
|---|---:|---|---|
| `kako5_avg_pos` | 13.88% | 過去 5 走 平均着順 | 0.90 |
| `前走確定着順` | 11.72% | 前走 | alive |
| `prev_hosei` | 7.56% | 補正タイム（前走） | 0.55–0.94（週変動） |
| `hist_same_cond_top3_rate` | 5.06% | 同条件キャリア | serve_history_feats が再計算 |
| `kako5_best_pos` | 2.38% | 過去 5 走 | 0.90 |
| `前走着差タイム` | 1.66% | 前走 | alive |
| `prev_hosei9` | 1.42% | 補正タイム | alive |
| `間隔` | 1.38% | ローテ | alive |
| `種牡馬` | 1.31% | 血統 (cat) | alive |
| `kako5_avg_agari3f` | 1.26% | 過去 5 走 上り | 0.90 |
| `前走上り3F` | 1.15% | 前走 | alive |
| `kako5_pos_trend` | 1.08% | 形の上下 | 0.90 |
| `調教師コード` | 1.07% | cat | serve_code_maps で復元 |
| `母父馬` | 1.06% | 血統 (cat) | alive |
| `年齢` / `kako5_std_pos` | 各 ≈1.05% | | alive |
| `騎手コード` | 0.97% | cat | serve_code_maps で復元 |

**構造的観察（Evidence, `project_v6_pastform_dominance`）**:
上位 2 特徴（`kako5_avg_pos` + `前走確定着順` = **25.6%**）だけで gain の 1/4。
過去走系全体で **gain の約 60.9%**。素朴な過去走ランカー（◎top3 ≈50%）に対し
v6 の上乗せは **+12pt** に過ぎず、両者の相関は 0.74。
→ **v6 の背骨は「過去の着順」であり、これは市場も同じものを見ている。**
これが「◎が市場と被る」「◎飛びが負けの 37.9% を占める」構造の根本原因。

#### B 群 — 旧監査スナップショット（2026-06: 39特徴、現在値は上表）

| 特徴 | gain | serve cov | 死因（§9 参照） |
|---|---:|---:|---|
| `jockey_fuku90` | **6.79%** | 0.00 | 🔴 定数刷り込み 0.200（騎手コード未解決） |
| `trainer_fuku90` | 1.92% | 0.00 | 🔴 定数 0.211 |
| `生産者` | 1.37% | 0.00 | cat 欠落 → `__NaN__` |
| `jockey_fuku30` | 1.32% | 0.00 | 🔴 定数 0.200 |
| `horse_fuku10` | 1.32% | 0.00 | 🔴 定数 0.286（tyaku 53 列問題） |
| `馬主(最新/仮想)` | 1.27% | 0.00 | cat 欠落 |
| `前走馬体重` | 1.16% | 0.00 | 定数 472（訓練 valid 中央値） |
| `斤量体重比` | 1.01% | 0.00 | 当日馬体重不在 → 定数 |
| `前走平均1Fタイム` | 0.91% | 0.00 | 定数 |
| `前PCI` | 0.86% | 0.00 | 定数 49.0 |
| `前走RPCI` | 0.81% | 0.00 | 定数 48.5 |
| `前走出走頭数` | 0.70% | 0.00 | 定数 15 |
| `horse_fuku30` | 0.69% | 0.00 | 🔴 定数 0.312 |
| `Ｒ` | 0.68% | 0.00 | **列名不一致**: parse_csv は半角 `R`、モデルは全角 `Ｒ` を要求（実測確認済） |
| `調教師年齢` | 0.66% | 0.00 | 定数 53 |
| `騎手年齢` | 0.65% | 0.00 | 定数 30 |
| `前走PCI3` | 0.56% | 0.00 | 定数 |
| `trainer_fuku30` | 0.55% | 0.00 | 🔴 定数 0.200 |
| `前走場所` | 0.50% | 0.00 | cat 欠落 |
| `前走馬体重増減` | 0.50% | 0.00 | 定数 0 |
| `休み明け～戦目` | 0.45% | 0.00 | 定数 2 |
| `前走日付` | 0.43% | 0.00 | 欠落 |
| `course_top3_rate` | 0.41% | 0.39 | 部分回収（serve_history_feats） |
| `前走レースID(新)` | 0.36% | 0.00 | 欠落 |
| `父タイプ名` | 0.25% | 0.10 | 部分 |
| `トラックコード(JV)` | 0.24% | 0.00 | 定数 23 |
| `course_win_rate` | 0.23% | 0.39 | 部分 |
| `前走トラックコード(JV)` | 0.23% | 0.00 | 定数 23 |
| `毛色` | 0.22% | 0.00 | cat 欠落 |
| `馬齢斤量差` | 0.21% | 0.00 | 定数 −1 |
| `前走競走種別` | 0.20% | 0.00 | 定数 13 |
| `指定条件` | 0.17% | 0.00 | cat 欠落 |
| `前好走` | 0.13% | 0.00 | cat 欠落 |
| `コース区分` | 0.13% | 0.30 | 部分 |
| `限定` | 0.10% | 0.00 | cat 欠落 |
| `芝(内・外)` | 0.06% | 0.00 | cat 欠落 |
| `前走レースID(新/馬番無)` | 0.05% | 0.00 | 欠落 |
| `性別限定` | 0.04% | 0.00 | cat 欠落 |
| `ブリンカー` | 0.04% | 0.00 | cat 欠落 |

**重要な留保（誠実性のため明記）**:
gain% は「木がその特徴で分割した際の損失減少の総和」であり、**レース内順位への寄与とは別物**。
定数刷り込みされた特徴はレース内で全馬同値になるため、**その特徴自身の判別力はゼロになるが、
他特徴との交互作用経由で葉の割り当ては変わる**。したがって
「gain 28% 喪失 = 精度 28% 低下」ではない。
実測された offline→serve のギャップは **◎複勝圏率 62.08% → 57.53%（−4.55pt）**
（`reports/serve_skew_eval.json`）であり、補正/調教のリネーム修復後は **≈61.0% まで回復**（memo）。
残差 ≈1pt が B 群の未回収分に相当すると推定される（**Hypothesis**、直接測定はされていない）。

→ **B 群の回収施策の期待効果は「1pt 程度」であり、精度の大幅改善ではない。**
ただし **コストが極めて低く、副作用がなく、確実に方向が正しい**唯一の残存レバーである（Vol. III §5）。

#### C 群 — gain = 0 の 6 特徴（学習時点で死んでいる）

| 特徴 | master 側の状態（実測） | 死因 |
|---|---|---|
| `開催` | `notna=1.000`、`nuniq=316`、値は `"1中1"` 等の**文字列** | CAT_COLS に含まれない → `to_numeric` 失敗 → 全行 −9999 |
| `前走走破タイム` | `notna=0.910`、値は `"1.13.6"`（M.SS.T 形式） | 同上。`utils.parse_time_str()` が存在するのに学習経路で適用されていない |
| `母馬` | `notna=1.000`、`nuniq=11,122`（馬名文字列） | 同上。**`PYcALiAI_RESEARCH.md` の UNKNOWN「母馬 疑似デッド」を本書で実証** |
| `kako5_avg_ninki` | **`notna=0.000`（master で 100% NaN）** | `parse_kako5 --mode master` が人気を出力していない |
| `kako5_pos_vs_ninki` | 同上 | 同上 |
| `kako5_upset_good_count` | 同上 | 同上 |

**非対称性の指摘（新規発見）**: `kako5_avg_ninki` / `kako5_pos_vs_ninki` /
`kako5_upset_good_count` は **serve 側では 90% 埋まっている**（実測 `nuniq=109`）。
つまり「学習では 100% 欠損 → 木が一切使わない → 本番では実値が来るが無視される」
という **逆向きの train/serve 非対称** が存在する。害はない（gain=0 なので分岐に使われない）が、
過去 5 走の人気情報（= 市場に対する馬の位置）という **本来価値がありうる信号が
学習パイプラインの欠陥で捨てられている**。

### 4.3 系統別の特徴インベントリ

| 系統 | 列数 | 代表 | 備考 |
|---|---:|---|---|
| 当日レース条件 | ~12 | 場所, 芝・ダ, 距離, 馬場状態, 天気, クラス名, 出走頭数, フルゲート頭数, 枠番, 馬番, 斤量, 年齢 | serve で確実に取れる |
| 前走成績 | ~23 | 前走確定着順, 前走着差タイム, 前1-4角, 前走上り3F, 前走斤量, 前PCI, 前走RPCI, … | serve で大半が定数化 |
| kako5（過去 5 走集約） | 16 | avg/std/best_pos, avg_ninki, pos_vs_ninki, avg_agari3f, same_*_ratio, pos_trend, hidden_good_count | serve 90%。3 列は学習側で死 |
| 全キャリア履歴 | 4 | hist_same_cond_{best_pos,top3_rate,count}, hist_same_place_best_pos | serve_history_feats が as-of 再計算 |
| ローリング複勝率 | 6 | jockey_fuku{30,90}, trainer_fuku{30,90}, horse_fuku{10,30} | 🔴 serve 全滅（定数） |
| 脚質 | 2 | prev_pos_rel, closing_power | 前走コーナー通過から算出 |
| 補正タイム | 2 | prev_hosei, prev_hosei9 | **前走のみ**（今走補正はリークとして除外） |
| 調教 | 16 | trnH_Time1-4/Lap1-4/days_ago, trnW_5F/4F/3F/Lap1-3/days_ago | 坂路 66–93%、WC 47% |
| コース／騎手履歴 | 6 | course_{n_prev,win_rate,top3_rate}, jockey_* | serve 部分回収（39%） |
| 血統・厩舎・馬主 | ~10 | 種牡馬, 父タイプ名, 母馬, 母父馬, 母父タイプ名, 毛色, 生産者, 馬主, 騎手コード, 調教師コード | cat。serve で半分欠落 |
| 条件フラグ | ~8 | 年齢限定, 限定, 性別限定, 指定条件, 重量種別, 性別, ブリンカー | cat |

### 4.4 モデルが**見ていない**情報（Fact）

これは提案時に必ず参照すべきリスト。「まだ入れていない情報」ではなく
**「意図的に、または既に検定した上で入れていない情報」**である。

| 情報 | 理由 |
|---|---|
| 当日オッズ・人気 | 今走単勝オッズは完全リーク（kekka は勝ち馬のみ収録）。**予測特徴化は禁止**（Vol. III §3.4）。bet 時のブレンド（T-10 補正印）は別物で採用済 |
| 当日馬体重 | master_v2 に列自体が無い。週次 CSV にはあるがモデルに渡らない |
| レース内相対特徴 | `race_relative_feats` は v2/v7 系のみ。v6 未使用 |
| 過去走のラップ生値・不利・位置取り詳細 | 検定済み: priced + JOIN 不能 + 先頭 1678 行が当日 leak（`project_lap_csv_evaluated`） |
| セリ取引価格 | 検定済み priced 死（`project_auction_price_priced_dead`） |
| 回り適性（右/左） | 検定済み priced 死（`project_direction_aptitude_priced_dead`） |
| クッション値・含水率 | 検定済み、馬場状態に吸収（`project_baba_cushion_tested`） |
| 血統 embedding / Elo / Glicko | 検定済み、冗長 or 逆効果 |

---

## §5 モデル層（unified_rank_v6）

### 5.1 モデル定義（`models/unified_rank_v6.pkl` 実測）

```
アルゴリズム : LightGBM  objective="lambdarank"
              lambdarank_truncation_level = 5
              metric = ndcg, eval_at = [1,3,5]
特徴数       : 120
グループ     : レース単位（COL_RID でソート後 groupby サイズ）
ラベル       : clip(6 − 着順, 0, 5)
sample weight: 1 + 0.0308 · log1p(勝ち馬単勝配当 / 100)

Optuna 採用ハイパーパラメータ (seed=42, 40 trials, 5-fold):
  learning_rate     = 0.05083
  num_leaves        = 59
  max_depth         = 12
  min_data_in_leaf  = 197
  feature_fraction  = 0.8761
  bagging_fraction  = 0.7032   (bagging_freq = 5)
  lambda_l1         = 0.001108
  lambda_l2         = 7.5379
  best_iteration    = 469  (retrain は best_iter × 1.1)

再現性フラグ : deterministic=True, force_col_wise=True, feature_pre_filter=False, seed=42
pkl の中身   : {model, feature_cols, encoders, cat_cols, seed, master_csv,
                optuna_best_params, optuna_best_composite, n_folds, label_scheme,
                sample_weight_alpha, ece_penalty_weight, description}
```

### 5.2 Optuna の目的関数（v6 の設計の核）

```
composite = 0.30·NDCG@5
          + 0.25·◎top3率
          + 0.20·(実 top3 ⊂ 予測 top5)率
          + 0.15·(勝ち馬 ∈ 予測 top5)率
          + 0.10·◎top2率
          − 0.50·ECE_high_p
```
`ECE_high_p = |mean(p_win[p_win ≥ 0.10]) − mean(actual[p_win ≥ 0.10])|`

**v5 → v6 の差分（`optuna_v6_marks.py` docstring）**:
1. α の探索範囲を `[0, 2.0]` → `[0, 1.5]` に狭めた（v5 の α=1.325 が穴馬スコアを系統的に嵩上げし
   tail の較正を壊していたため）
2. 目的関数に ECE ペナルティを追加

**この目的関数の既知の欠陥（🟡 P2, Vol. III §6.2）**:
- `ECE_high_p` は **単一ビン**での平均差なので、**過信と過小確信が相殺される**。
  実効的なペナルティが働かず、寄与は約 1% にとどまった（採用 α=0.031 は
  ECE ペナルティではなく探索範囲の縮小によるものと交絡している）。
- 修正版（**4 ビン加重 |gap|**）は `lab/train/optuna_v10_marks.py` に存在するが **未採用**。
- 新実験では **10 ビン ECE** を使うこと（Vol. III §1.3）。
- `composite` に `◎top3` 等の「印の当たり率」が入っている＝**印という lossy な中間表現が
  モデルの目的関数に侵入している**（`project_marks_as_lossy_middle_layer`）。

### 5.3 実測性能（`reports/audit_marks_v6.json`）

| 指標 | 真 OOS 2024-2025 (6,878R) | 3 年 2023-2025 (10,327R) |
|---|---:|---:|
| NDCG@5 | 0.6040 | 0.6002 |
| ◎ 1 着率 | 30.28% | 30.01% |
| ◎ 連対率 (top2) | 49.52% | 48.86% |
| **◎ 複勝圏率 (top3)** | **62.08%** | 61.63% |
| 〇 複勝圏率 | 48.43% | 47.97% |
| ▲ 複勝圏率 | 40.23% | 39.89% |
| △1 / △2 複勝圏率 | 31.81% / 26.24% | 31.99% / 26.46% |
| 勝ち馬 ∈ top3 | 61.34% | 60.79% |
| 勝ち馬 ∈ top5 | 78.23% | 77.82% |
| {1,2} ⊂ top5 | 54.75% | 53.72% |
| ECE 単勝(◎) | 0.0187 | 0.0120 |
| ECE 複勝(◎) | 0.0118 | 0.0086 |
| ECE 馬連(◎-〇) | 0.0144 | 0.0105 |

**ランダム基準**: 16 頭立ての複勝圏率 = 3/16 = 18.75%。◎の 62.08% は **3.3 倍**。
**市場基準（Evidence, `data/t10_blend.json`）**: 単勝オッズ 1 番人気の top3 率 = **64.33%**
（同 6,858R）。すなわち **v6 の◎は市場の 1 番人気に −2.3pt 負けている**。
T-10 オッズブレンド後は 65.08% で市場をわずかに上回る（Δ vs 市場 CI95 = [−0.001, +0.016]、
すなわち **有意には勝っていない**）。

---

## §6 学習プロトコルと版管理

### 6.1 時系列分割（唯一の定義箇所: `build_dataset.py:40-41, 273-281`）

| セット | 期間 | 行数（実測） |
|---|---|---:|
| train | 〜 2022-12-31 | 485,252 |
| valid | 2023-01-01 〜 2023-12-31 | 47,273 |
| test | 2024-01-01 〜（実データ 2025-12-28 まで） | 94,249 |

`split` 列として master_v2 に焼き込まれている。**ランダム分割はリポジトリ内に存在しない。**

### 6.2 学習手順

1. train で fit、valid で early stopping（100 ラウンド）
2. Optuna の目的も valid のみ。**valid 内レース ID の 5-fold KFold**
   ⚠️ これは **時系列 CV ではない**（valid 期間内をランダム分割している）。
   valid が 1 年しかないため fold 間の regime 差は小さいが、時系列的厳密性は無い（🟡 P2）。
3. best_params で `num_boost_round = best_iter × 1.1` により再学習
4. Calibrator は **valid=2023 のみ** で fit（`build_pl_calibrators.py`、メタに `fit_split` 記録）

### 6.3 test の汚染状況（🟠 P1, 重要）

**Fact（`docs/audit_20260615_full.md` VAL-02）**: test 2024-25 は版選定のために **7 回以上開封済み**。
v5/v6/v7/v8/v9/v10/v11 の採否がすべて test の audit を見て決められている。
したがって **v6 の test 数値（◎top3 62.08%）は厳密な OOS ではなく、多重比較で選ばれた値**である。

**今後の規律（v10 プロトコル、`lab/train/optuna_v10_marks.py`）**:
- test = 2025 を封印する
- 版間比較は **valid のみ**で行う
- valid の CI 下限が閾値を超えた最終 1 版のみ、test を **1 回だけ**開封する
- 4 ビン ECE、mean − 1.0·std の保守選択を使う

### 6.4 キャリブレーション（`build_pl_calibrators.py`）

7 券種それぞれについて、valid=2023 で `(PL 予測確率, 実的中 0/1)` のペアを収集し
`sklearn.isotonic.IsotonicRegression` を fit する。

| キー | 収集単位 | 件数/R |
|---|---|---|
| `tansho` | 全馬 | N |
| `fukusho` | 全馬 | N |
| `umatan` | 全順序対 | N(N−1) |
| `umaren` | 全無向対 | C(N,2) |
| `wide` | 全無向対 | C(N,2) |
| `sanrentan` | 全順序三つ組 | ※実装参照 |
| `sanrenpuku` | 全無向三つ組 | ※実装参照 |

**適用箇所（`export_marks_json.py:251-255, 349-356`）**:
- `p_win` ← `calibrators["tansho"]`
- `p_sho` ← `calibrators["fukusho"]`
- `pair_probs.{wide,umaren,umatan}` ← 各較正器
- ⚠️ **`p_plc`（連対率）には較正器が適用されていない**（生 PL 値）。
  `docs/marks_schema.md` の「既知の制約 3」と一致。bundle 利用側は要注意。

**較正後の性質**: Isotonic は単調変換なので **レース内順位は不変**。
ただし `Σ p_win = 1.0` は**保証されなくなる**。そのため
`race_confidence` のエントロピー計算では明示的に再正規化している（`export_marks_json.py:127-133`）。

### 6.5 serve 条件較正器（`build_pl_calibrators_serve.py`）

**目的**: 本番のスコア分布は特徴欠損分布であり、フル特徴で fit した較正器とミスマッチする。
実測で **ECE 複勝が 2.3 倍 / 馬連が 1.8 倍**悪化していた（`docs/audit_20260611.md`）。
→ **fit 自体を serve マスク済みスコアで行う**。

**成果（`reports/calibrators_v6_serve_eval.json`）**: serve fit により ECE 複勝 −36% / 馬連 −29%。

**採用ロジック（`export_weekly_marks.py:216-218`）**:
```python
serve_cal = BASE / f"models/pl_calibrators_{tag}_serve.pkl"
if serve_cal.exists():
    be.CAL_PKL = serve_cal      # 存在すれば無条件で優先
```
offline 監査（`audit_marks.py` 等）はフル特徴スコアなので、従来の `{tag}.pkl` を使い続ける。

**🔴 P0-3 — 較正マスクが実態と乖離している（本書での新規発見）**:
`models/pl_calibrators_v6_serve.pkl` のメタを実測すると、マスクされているのは **14 特徴のみ**:
```
serve_mask_numeric = ['Ｒ','前走走破タイム','前走日付','前走レースID(新)','前走レースID(新/馬番無)','母馬']
serve_mask_cat     = ['芝(内・外)','前走場所','前好走','毛色','馬主(最新/仮想)','限定','指定条件','ブリンカー']
n_races = 3456, fit_split = "valid=2023 (serve マスクスコア)"
```
一方、§4.2 の実測では **coverage < 0.40 が34特徴（うちgain > 0は32件、gain 14.88%）**。
すなわち **較正器は「本番の 1/3 しか壊れていない世界」で fit されている**。
`serve_skew_eval.py` のハードコード定数（`SERVE_DEAD_NOW_EXACT` 6 件 + `SERVE_DEAD_NOW_CAT` 8 件）が
2026-06 時点の状態で凍結されており、以降の実測（`data/serve_feature_baseline.json`）と同期していない。
→ 対処は Vol. III §5 の P0-3。

### 6.6 class_prior（クラス別事前確率）

`scripts/audit_marks_by_class.py --model v6` が `data/class_prior_v6.json` を生成し、
`export_weekly_marks.py:250-270` が `race_meta.class_prior` として bundle に埋め込む。
中身は「そのクラスにおける ◎〇▲△△ の経験的中率」であり、Cowork（narrative）と
人間が「印をどこまで信じてよいか」を判断する材料。
例: G2 は◎1着率 20% と弱い / 未勝利は◎複勝 67% と堅い。

### 6.7 版採否台帳（要約 — 詳細は `docs/version_ledger.md`）

| 版 | 差分 | 採否 | 主因 |
|---|---|---|---|
| v5 | payout 重み α=1.325 | 🗄️ 退役 | tail 較正崩壊 |
| **v6** | v5 + ECE ペナルティ、α=0.031 | ✅ **本番** / ❌ **採用ゲート未達** | Vol. III §6.1 |
| v7 | 目的関数にワイド ROI を直接組込 | ❌ | Δ ROI = **−1.50pt**（valid metric 直接最適化の過学習） |
| v8 | course_affinity +34 列 | ❌ | **自レース込み集計の in-sample leak** |
| v9 | truncation_level 5→3 | ❌ | 別系統の失敗実験 |
| v10 | 監査反映（test=2025 封印 / 4 ビン ECE / mean−std 選択 / パースバグ修正） | ❌ | 封印 test で v6 同等以下。**プロトコルとバグ修正は資産** |
| v11 | 格特徴 15 列フル再学習 | ❌ | +0.62pt CI[−0.20,+1.45] 非有意 = 市場が吸収済み |

**v6 の seed 変種** (`unified_rank_v6_s123/s456/s789/s1234.pkl`) は用途記録なし（⚪ P3）。

---

## §7 確率層

### 7.1 Plackett-Luce 厳密計算（`pl_probs.py`）

LightGBM の raw score `s_i` から重み `w_i = exp(s_i − max s)` を作り、PL モデル
```
P(順列 (h1..hk) が上位 k 着) = Π_{m=1..k}  w_{hm} / (Σw − Σ_{j<m} w_{hj})
```
に基づき **近似なしの閉形式**で全券種の joint を計算する。

| 関数 | 内容 | 計算量 |
|---|---|---|
| `p_tansho(w,i)` | `w_i / Σw` | O(1) |
| `p_umatan(w,i,j)` | `w_i/Σw · w_j/(Σw−w_i)` | O(1) |
| `p_umaren(w,i,j)` | `p_umatan(i,j) + p_umatan(j,i)` | O(1) |
| `p_sanrentan(w,i,j,k)` | 3 段の逐次選択 | O(1) |
| `p_sanrenpuku(w,i,j,k)` | 6 順列の和 | O(1) |
| `p_place_at(w,i,pos)` | pos=1/2/3 の厳密和 | O(N²) for pos=3 |
| `p_fukusho(w,i)` | `Σ_{pos=1..3} p_place_at` | O(N²) |
| `p_wide(w,i,j)` | `Σ_{k≠i,j} p_sanrenpuku(i,j,k)` | O(N) |

**自己検証（`python pl_probs.py`）**: 以下の恒等式を assert する。
```
Σ 単勝 = 1     Σ P(着=pos) = 1 (pos=1,2,3)    Σ 複勝 = 3
Σ 馬連 = 1     Σ 三連複 = 1    Σ ワイド = 3    Σ 三連単 = 1    Σ 馬単 = 1
∀i: Σ_j wide(i,j) = 2 · fukusho(i)
```
N=5,10,16 で全通過を確認済み（本書執筆時に再実行）。

### 7.2 λ 補正 PL（Lo–Bacon-Shone / Stern 型）

素の PL は「1 着の強さがそのまま 2 着・3 着の強さになる」と仮定するが、実際は
上位馬の 2 着・3 着確率が PL の予測より低い（縦目バイアス）。
そこで **べき乗補正**を入れる：
```
P(x→y→z) = p_x · (p_y^λ1 / Z1) · (p_z^λ2 / Z2)
Z1 = Σ_{k≠x} p_k^λ1
Z2 = Σ_{k≠x,y} p_k^λ2
```
`data/harville_lambda.json`（実測）:
```json
{"lambda1": 0.8405, "lambda2": 0.7542,
 "fit_window": "20260531-20260711", "fit_races": 349,
 "excluded": "< 20260531 (v5 期の p_win。lambda が逆符号に出る)"}
```
実装は `compute_bets.pl_pair_probs()`（`compute_bets.py:114-145`）。全ペアの
`p_umaren` / `p_wide` を O(N³) で構築する。

⚠️ **λ はモデル世代を跨ぐと壊れる**（Fact: v5 期のデータでは符号が逆に出るため除外されている）。
モデルを更新したら **必ず λ を再 fit** すること（`analysis/fit_harville_lambda.py`）。
fit 標本は **349 レースしかない**（🟡 P2）。

### 7.3 pair_probs（bundle 埋込）

`export_marks_json.py:334-367` が印 5 頭の C(5,2)=10 ペアについて、
**PL 厳密 joint に較正器を通した値**を bundle に埋め込む。
```json
"pair_probs": {
  "5-9": {"wide": 0.09801, "umaren": 0.03768,
          "umatan": {"9→5": 0.01883, "5→9": 0.01844}}
}
```
**背景**: これ以前は `compute_bets` がワイド確率を `p_sho_i × p_sho_j`（独立積）で
近似しており、**系統的に +21〜27% 過大**だった（`docs/audit_20260611.md` 🔴）。

**現状の使われ方（重要）**: 既定エンジン `topdown` は pair_probs を**使わない**。
`pl_pair_probs()`（λ補正 PL、**較正器なし**）を全馬に対して計算し直している。
pair_probs を使うのは旧 `shape` 経路のみ。
- λ 補正版と bundle の厳密較正値の比は **0.99 ± 0.1**（バイアスなし、`compute_bets.py:96-97`）
- しかし **較正が効いていない**ぶん、topdown の確率は理論値寄り（🟡 P2, Vol. III §5 P2-4）

### 7.4 確率の制約（bundle 利用側の契約）

| 量 | 理論的制約 | 較正後の実際 |
|---|---|---|
| `Σ p_win` | 1.0 | **≠1.0**（Isotonic で崩れる） |
| `Σ p_plc` | 2.0 | 2.0（較正なし） |
| `Σ p_sho` | 3.0 | **≠3.0** |

→ 正規化が必要な計算（エントロピー等）は**利用側で再正規化する契約**。

---

## §8 印層

### 8.1 印の規則

| 印 | 意味 | ai_rank |
|---|---|---|
| ◎ | 本命 | 1 |
| 〇 | 対抗 | 2 |
| ▲ | 単穴 | 3 |
| △ | 連下 | 4, 5（2 頭とも △） |
| `""` | 印なし | 6 位以下 |

**割り当ては `raw score` 降順**（較正確率降順ではない）。Isotonic は単調なので同じ順序になる。

⚠️ **文字の揺れ**: 全角「〇」(U+3007) と丸「○」(U+25CB) が混在する。
`compute_bets.py:431` は `"○" if m == "〇" else m` で正規化している。新規コードは要注意。

### 8.2 race_confidence（`export_marks_json.py:118-147`）

| 指標 | 定義 | 解釈 |
|---|---|---|
| `top1_dominance` | `clip(p_win[1位] − p_win[2位], 0, 1)` | 大 = ◎ 独走 |
| `top2_concentration` | `clip(p_win[1位] + p_win[2位], 0, 1)` | 大 = 上位 2 頭決着 |
| `field_chaos_score` | `H(p_norm) / log(N)`（正規化エントロピー） | 大 = カオス |
| `ai_market_agreement` | AI 順位 vs 市場（オッズ）順位の Spearman 相関 | 大 = 市場一致、小 = 乖離 |

`ai_market_agreement` は **3 頭以上にオッズが揃わないと null**。

**生値 → パーセンタイル変換**: 生値の値域は圧縮されている（chaos は [0.80, 1.0] にほぼ収まる）ため、
`data/chaos_quantiles.json`（3 指標 × 101 点の分位表）で **過去分布のパーセンタイル**に変換して使う。
実装は `compute_bets.pct()` と `betting_judgment.chaos_to_pct()` の 2 箇所（同一ロジック）。

🔴 **fail-safe の設計**: `compute_bets.pct()` は分位表が壊れていたら **`None` を返す**。
旧実装は生値をそのまま返しており、「テーブル破損 → 全レース chaos_pct > 0.75 →
全部カオス薄に倒れる」事故になっていた（`docs/audit_20260611.md`）。
現在は `None` を受けた呼び出し側が **見送りに倒す**（`compute_bets.py:455-459`）。

### 8.3 buy_judgment（`betting_judgment.py`）

```
hardness = 固い (chaos_pct ≤ 0.30) / 標準 / 荒れ (chaos_pct ≥ 0.70)
has_value = 妙味馬が 1 頭以上いるか
→ (hardness × has_value) の 6 通りで headline / category / kenshu_hint / waku_tag を決める
```

**妙味馬 (value_horses) の定義**:
1. 単勝または複勝が「割安」= `model_p ≥ (1/odds) × 1.20`
2. 該当側の EV ≥ 1.10
3. `p_win ≥ 0.05`（テール除外）
4. **UMAMI ゲートを通過**（下記）

**ソート順**: 生 EV 降順ではなく **UMAMI (xROI) 降順**。
理由: `audit_ev_bin_roi` で「高 EV ほど実現 ROI が低い」が実証されたため、
生 EV を「美味しさ」として並べるのは罠だった。

### 8.4 UMAMI（`umami.py`）— 実測補正後期待回収率

生 EV = `p × odds` は「モデルが市場と喧嘩している度合い」であり、
喧嘩の大半はモデル側の間違いである（Evidence: EV 2.0+ の実現 ROI = 64%、
単勝 50 倍超帯は ROI 30–61% / CLV −54〜−67%）。

**UMAMI = 実測テーブルで補正した期待回収率 (xROI)**:
`reports/audit_ev_bin_roi.json` の `by_ev_x_fav`（EV ビン × 単勝オッズ帯 → test 2024-25 の実現 ROI）
を参照テーブルにして、「過去に同じ状況だった馬券の実際の回収率」を返す。

```
EV_EDGES  = [0.8, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0]     (8 ビン)
FAV_EDGES = [3.0, 7.0, 15.0, 50.0]                   (5 帯)
MIN_CELL_N = 300  → 未満なら EV ビン単独へフォールバック
グレード   : xROI ≥0.85 → S / ≥0.80 → A / ≥0.72 → B / else C
```

**ゲート（「妙味があっても明らかに来ない馬は出さない」）**:
| 条件 | 判定 |
|---|---|
| `p_win < 0.04`（単勝）/ `p_sho < 0.12`（複勝） | 罠（来る見込み薄） |
| 単勝オッズ > 50 倍 | 罠（実測最悪帯 ROI 30–61%） |

**位置づけ（Evidence, `project_umami_vs_marks`）**: UMAMI 駆動の買いは印 (◎) 駆動を
**有意には上回らない**（2025 OOS）。UMAMI は印の代替ではなく、**罠ゲート + 厚薄レイヤー**である。

### 8.5 SHAP による印の根拠（`marks_shap.py`）

`export_weekly_marks.py --shap-topk K`（既定 6）のとき、**印の付いた馬のみ**に `why` を付ける。
```json
"why": [{"feat":"prev_hosei","label":"前走 補正タイム","value":99,"contrib":0.285}, ...]
```
- ベースライン `shap.expected_value`（v6 ≈ −1.02）に対し、**全 120 特徴の Σcontrib + base = ai_score**（恒等）
- `why` はそのうち |contrib| 上位 K のみ
- **相関ベースの寄与であり因果ではない**。narrative の裏取り専用

---

## §9 serve 層（本番推論経路とその欠損構造）

**この節が本仕様書で最も実務価値が高い。** 学習と本番で「同じ特徴」が作られていない
構造を、発生源から順に説明する。

### 9.1 serve と train の入力の非対称

| | 学習 (train) | 本番 (serve) |
|---|---|---|
| 入力 | `master_v2_*.csv`（132 列、全履歴 JOIN 済） | `data/weekly/{date}.csv`（TARGET 出走表、19/33/46/49/99 列） |
| 馬の同定 | `血統登録番号` | **馬名文字列** |
| 騎手・調教師 | コード | **名前**（コードは `serve_code_maps.json` で逆引き） |
| 過去走 | master に列として存在 | `data/kako5/{date}.csv` + `_horse_history.parquet` から再構築 |
| 調教 | `merge_asof`、日数制限なし | **14 日カットオフ**、同日許容 |
| 補正タイム | master 列 | `data/hosei/H_{date}.csv` |

### 9.2 欠損の第一発生源 — `predict_weekly.parse_csv()`

**`export_weekly_marks.py:57` が import している。「旧系統」ではなく本番の入力パーサである。**

パーサは週次 CSV に無い列を **訓練 valid 中央値で定数補完**する（`predict_weekly.py:526-560`）。
設計意図は「−9999 の外れ値を避ける」だが、結果として **レース内で全馬同値 = 判別力ゼロ**になる。

**実測（`data/weekly/20260816.csv`、478 頭 / 35 レース、本書執筆時に実行）**:

| 特徴 | notna | nunique | 実際の値 | gain |
|---|---:|---:|---|---:|
| `jockey_fuku30` | 1.000 | **1** | 0.200 | 1.32% |
| `jockey_fuku90` | 1.000 | **1** | 0.200 | **6.79%** |
| `trainer_fuku30` | 1.000 | **1** | 0.200 | 0.55% |
| `trainer_fuku90` | 1.000 | **1** | 0.211 | 1.92% |
| `horse_fuku10` | 1.000 | **1** | 0.286 | 1.32% |
| `horse_fuku30` | 1.000 | **1** | 0.312 | 0.69% |
| `前走馬体重` | 1.000 | **1** | 472 | 1.16% |
| `前PCI` | 1.000 | **1** | 49.0 | 0.86% |
| `前走RPCI` | 1.000 | **1** | 48.5 | 0.81% |
| `前走走破タイム` | **0.000** | 0 | — | 0（学習側も死） |
| `Ｒ`（全角） | — | — | **列が存在しない**（parse_csv は半角 `R` を作る） | 0.68% |
| `prev_hosei`（正常例） | 0.546 | 40 | 実値 | 7.56% |
| `trn_hanro_4f`（正常例） | 0.663 | 152 | 実値 | — |
| `kako5_avg_ninki`（逆非対称） | 0.900 | 109 | 実値 | 0（学習側が死） |

#### 🔴 根本原因 1 — 騎手/調教師ローリング複勝率（gain 合計 10.58%）

`predict_weekly.py:466-485`:
```python
for fname, code_col, stat_cols in [
    ("jockey_stats.csv",  "騎手コード",  ["jockey_fuku30", "jockey_fuku90"]),
    ("trainer_stats.csv", "調教師コード", ["trainer_fuku30", "trainer_fuku90"]),
]:
    if stats_path.exists():
        if code_col in df.columns:          # ← ここが常に False
            ... merge ...
        else:
            for col in stat_cols:
                df[col] = _ROLLING_TRAIN_MEDIANS.get(col, 0.200)   # ← 常にこちら
```
週次 TARGET CSV には **`騎手コード` 列が存在しない**（あるのは `騎手` 名）。
よって merge は **一度も実行されていない**。`data/jockey_stats.csv`（223 名分）と
`data/trainer_stats.csv`（242 名分）は存在するのに **使われていない**。

しかも `serve_history_feats.fill_history_features()` は
**この後で** `serve_code_maps.json` から `騎手コード` / `調教師コード` を復元している。
つまり **コードは手に入るのに、その時点では既に定数が刷り込まれた後**である。

→ **修正は「fill_history_features の後で jockey_stats / trainer_stats を再 merge する」だけ。**
（副作用注意: `jockey_stats.csv` は静的スナップショット（2026-07-29 更新）であり、
学習側の `shift(1)` ローリングとは定義が異なる。過去日付に対しては未来情報を含むため
**バックテストに使うと leak**。前向き serve のみで使うこと。→ Vol. III §5 P0-1）

#### 🔴 根本原因 2 — 着度数 CSV の列数ドリフト（gain 合計 2.01%）

`predict_weekly.py:259`:
```python
elif len(cols) == 55 and cols[0] not in ("枠番","") and current_race_id:
```
**実測: `data/tyaku/*.csv` の馬行は全て 53 列**（本書執筆時に 5 ファイルの列数ヒストグラムを取得）:
```
20260816: {19: 70, 53: 513}
20260419: {19: 72, 53: 526}
20260607: {19: 46, 53: 366}
20260802: {19: 70, 53: 489}
20260815: {19: 70, 53: 511}
```
→ `rows` が空 → `_load_tyaku()` が `None` を返す → `horse_fuku10/30` は定数、
**当日馬体重・増減の取り込みも同時に失われている**。
`data/tyaku/` には 44 ファイルが置かれており、**2026 シーズン全期間で機能していない**。
検知機構は存在しない（ログにも出ない）。

#### 🟠 根本原因 3 — 前走詳細ブロックの定数補完（gain 合計 ≈7.5%）

`predict_weekly.py:526-560` が意図的に定数補完している 15 列:
```
前PCI=49.0, 前走RPCI=48.5, 前走PCI3, 前走平均1Fタイム,
馬齢斤量差=−1, トラックコード(JV)=23, 前走トラックコード(JV)=23,
前走競走種別=13, 前走出走頭数=15, 前走馬体重=472, 前走馬体重増減=0,
騎手年齢=30, 調教師年齢=53, 休み明け～戦目=2, 斤量体重比
```
これらは **`data/_horse_history.parquet` から as-of で再計算可能**なものを多く含む
（前走馬体重 / 前走出走頭数 / 前走競走種別 / 前走場所 / 前走日付 など）。
`serve_history_feats` の `NUM_FEATS` を拡張すれば回収できる。

### 9.3 `_SERVE_RENAME`（列名不一致の修復、成功例）

`export_weekly_marks.py:314-334`。`parse_csv` は補正を旧名（`前走補正` / `前走補9`）、
調教を旧名（`trn_hanro_*` / `trn_wc_*`）で作るが、v6 は `build_master_v2.py` のリネーム後の
名前（`prev_hosei*` / `trnH_*` / `trnW_*`）を要求する。列名を合わせないと
「不足列補完」で −9999 に潰れる。

**効果（`serve_skew_eval.py`）**: 補正 **+2.97pt** / 調教 **+0.33pt** の回収。
本番 ◎複勝圏率が 57.53% → ≈61.0% に戻った主因。

**既知の残差**: 推論の調教 JOIN には **14 日カットオフ**があるが、学習は無制限。
14 日超の追い切りだけが serve で欠損する。坂路カバレッジ 93.9%（週により 66%）で実害は小さいとされ、
欠損分布の差は serve 条件較正器が吸収する設計。→ ただし §6.5 の通り較正器のマスクが不整合。

### 9.4 `serve_history_feats.py`（履歴特徴の as-of 再計算、成功例）

`data/_horse_history.parquet` を **馬名で JOIN** し、レース日より厳密に前の走のみで
学習と同一定義の特徴を再計算する。

**埋める特徴（12 件）**:
```
NUM_FEATS = hist_same_cond_best_pos, hist_same_cond_top3_rate, hist_same_cond_count,
            hist_same_place_best_pos, course_n_prev, course_win_rate, course_top3_rate,
            jockey_n_prev, jockey_win_rate, jockey_top3_rate
CAT_FEATS = 騎手コード, 調教師コード
```
**同名馬の曖昧性解消**: 父名（種牡馬）一致 または 生年（レース年 − 年齢）±1 一致。
解消不能なら NaN（安全側）。未知馬（新馬等）は学習と同じく `n_prev=0` / rate は NaN。

**fail-open**: 例外時は従来どおり NaN のまま続行。埋まらない週は canary が検知する。
**鮮度チェック**: parquet の最終日付がレース日より 300 日以上古いと WARNING。

⚠️ `jockey_fuku30/90` `trainer_fuku30/90` `horse_fuku10/30` **は NUM_FEATS に含まれていない**。
これが §9.2 の欠陥が放置されている理由。

### 9.5 品質ゲートと serve canary（`export_weekly_marks.py:490-588`）

bundle は**書き出した上で**、閾値割れなら **exit 2** して `weekly_nicegui.ps1` の
git push / sync-hf を止める（fail-closed）。

**ゲート条件**:
| # | 条件 | 意図 |
|---|---|---|
| G1 | bundle の race 数 = 0 | parse_csv が全行を捨てた |
| G2 | bundle race 数 / 生 CSV レース数 < 0.5 | TARGET 形式変更・列ズレ |
| G3 | 単勝オッズ被覆率 < 50% | 週次 CSV の単勝列欠落 |
| G4 | `p_win` 非 null 率 < 90% | モデル予測の大量失敗 |
| G5 | **serve canary**（下記） | 特徴の無言死 |

**serve canary の判定**:
```
baseline = data/serve_feature_baseline.json の baseline_cov（健全週 4 週の中央値）
監視対象 = baseline_cov ≥ 0.40 の特徴のみ（既知 dead は対象外）  → 実測 79 特徴
発火条件 = 現在カバレッジ < 0.20 かつ baseline の 40% 未満
```

**カバレッジの定義（`feature_coverage()`、監査 2026-07-30 で 2 つの死角を塞いだ）**:
1. カテゴリ列は `"__NaN__"` 文字列で初期化されるため `notna=100%` になり code map 全滅を検知できなかった
   → `"__NaN__"` / 空文字を欠損扱いにする
2. 中央値フォールバックの定数刷り込みが `notna=100%` で健全に見えた
   → **有効値の `nunique() ≤ 1` なら 0.0 を返す**

**`CONST_OK_COLS = {馬場状態, 天気}`**: 快晴開催では全 35R が「良/晴」に潰れるが
これはデータ死ではなく実態。定数=0.0 ルールを当てると canary が偽陽性で push を止めるため除外。

**canary が現状の欠陥を検知しない理由**: `jockey_fuku90` 等は `baseline_cov = 0.0` として
**baseline に「既知 dead」として焼き込まれている**ため、監視対象外（`exp < 0.40` で continue）。
canary は「昨日まで生きていた特徴が今日死んだ」を検知する装置であり、
**「ずっと死んでいる特徴」は設計上見逃す**。→ Vol. III §5 P1-1。

### 9.6 serve 経路の実行順序（正確な順番）

```
1. parse_csv(weekly.csv)                    ← ここで定数刷り込みが起きる (§9.2)
2. ensure_date_column
3. CSV 血統フォールバック map 構築
4. _SERVE_RENAME                             ← 補正・調教の列名を合わせる (§9.3)
5. feats に無い列を NaN / "__NaN__" で補完
6. serve_history_feats.fill_history_features ← 履歴 12 特徴 + コードを as-of 再計算 (§9.4)
7. kako5_summary.build_histories / build_horse_facts
8. horse_pedigree.json ロード
9. オッズ: data/odds/OD{YYMMDD}.CSV → 無ければ weekly CSV の単勝列のみ
10. レース毎に export_race()                 ← 推論・PL・較正・印付け
11. history / sex / age / pedigree を注入
12. bundle 書き出し
13. 品質ゲート + serve canary                ← exit 2 で push 停止 (§9.5)
```

**手順 1 と 6 の順序が §9.2 P0-1 の直接原因**（コードが手に入るのは 6、定数刷り込みは 1）。

### 9.7 オッズ源の優先順位

| 優先 | ソース | 得られるもの |
|---|---|---|
| 1 | `data/odds/OD{YYMMDD}.CSV`（TARGET） | 単勝 + **複勝下限/上限** + **馬連 matrix** |
| 2 | `data/weekly/{date}.csv` の `単勝` 列 | 単勝のみ（複勝・馬連は null） |
| 当日 | `reports/live_odds/{rid16}.json`（JV-Link T-10） | 単勝・複勝・ワイド・馬単の実値（Vol. II §6） |

bundle のオッズは **朝時点のスナップショット**であり、実際の買い目は T-10 のライブ値で再計算される。

---

## §10 出力スキーマ（bundle.json）完全定義

### 10.1 ルート

```json
{
  "date": "20260816",
  "model": "v6",
  "race_count": 35,
  "races": [ { ...race... } ]
}
```
個別ファイル `reports/cowork_input/{date}/{race_id}.json` は `race` オブジェクト単体。

### 10.2 race オブジェクト

| フィールド | 型 | 生成元 | 備考 |
|---|---|---|---|
| `race_id` | string(16) | 馬番なしレース ID | |
| `race_meta` | object | `export_marks_json.race_meta()` | |
| `horses` | array | 馬番昇順 | |
| `race_confidence` | object | §8.2 | |
| `buy_judgment` | object | `betting_judgment.build_judgment()` | §8.3 |
| `umaren_matrix` | object? | OD CSV 由来 | `{"a-b": odds}`、a<b |
| `pair_probs` | object? | 印 5 頭の 10 ペア | §7.3 |

### 10.3 race_meta

```json
{"date":"20260816","place":"札幌","course":"ダ1700","field_size":14,
 "class":"未勝利","race_name":"3歳未勝利",
 "class_prior": { ...クラス別 ◎〇▲△△ 経験的中率... }}
```

### 10.4 horse オブジェクト

| フィールド | 型 | 説明 |
|---|---|---|
| `umaban` | int | 馬番 |
| `horse_name` | string | |
| `mark` | string | `◎`/`〇`/`▲`/`△`/`""` |
| `ai_rank` | int | 1〜18 |
| `ai_score` | float | LightGBM raw score |
| `p_win` | float\|null | PL 1 着確率（**較正済**） |
| `p_plc` | float\|null | PL 連対率（**較正なし**） |
| `p_sho` | float\|null | PL 複勝率（**較正済**） |
| `tansho_odds` | float\|null | |
| `fuku_odds_low` / `fuku_odds_high` | float\|null | |
| `ai_vs_market` | string | `under` / `fair` / `over` / `unknown` |
| `why` | array? | SHAP top-K（**印馬のみ**） |
| `history` | object? | kako5 由来（`n_runs`, `avg_pos`, `pos_trend`, `runs[]` …） |
| `sex` / `age` | string? / int? | kako5 由来 |
| `pedigree` | object? | `sire` / `sire_type` / `broodmare_sire` / `broodmare_sire_type` |

**`ai_vs_market` 判定**: `market_p = 1/tansho_odds`（控除率無視）
- `p_win ≥ market_p × 1.20` → `under`（AI が高評価 = 妙味候補）
- `p_win ≤ market_p × 0.80` → `over`（AI が低評価 = 過剰人気）
- else `fair`

### 10.5 buy_judgment

```json
{"hardness":"固い","chaos_pct":0.226,"has_value":true,
 "headline":"妙味本線（絞って厚く）","category":"go",
 "detail":"...","kenshu_hint":"...","waku_tag":"妙味枠",
 "value_horses":[{"umaban":3,"horse_name":"...","p_win":0.173,
                  "ev_tan":16.78,"ev_fuku":4.86,
                  "umami_tan":0.83,"umami_fuku":0.79,"umami_grade":"A",
                  "tan_value":true,"fuku_value":true,"sides":["単勝","複勝"]}]}
```
`category` は `go` / `caution` / `avoid` / `danger`（表示配色用）。

### 10.6 出力側（`{date}_bets.json`）

```json
{"bets":[
  {"race_id":"...","race_label":"札幌ダ1700 3歳未勝利",
   "race_nature":"topdown",              // または 見送り / 本命勝負 / ◎軸 / 広め流し / カオス薄 / 標準 / 複勝特化
   "race_reason":"...",
   "confidence":{"top1_pct":0.65,"top2_pct":0.46,"chaos_pct":0.36,"market":0.72},
   "bets":[{"馬券種":"複勝","買い目":"9","購入額":3000,"枠タグ":"参加枠","理由":"topdown p=0.412（複勝 1.8倍）"}],
   "hosei_marks":[{"mark":"◎","umaban":9,"horse_name":"...","orig_mark":"〇"}],
   "advisor":[ ...Cowork narrative... ],
   "stamp":{"model":"v6","engine":"compute_bets","engine_version":"2026-08-09",
            "mode":"default","live":true,"stamped_at":"2026-08-16T14:50:03"}}],
 "grade_scope":[ ... ]}
```
レガシー形式（ルートが配列）も読める（`raw["bets"] if isinstance(raw, dict) and "bets" in raw else raw` パターンが各所に散在）。

---

## §11 環境・依存・実行コマンド

### 11.1 環境

```
OS        : Windows 11 Pro 10.0.22631
Python    : 3.11 (venv311\)  ※JV-Link 用に 32-bit Python 3.12 (py -3.12-32) が別途必要
作業ディレクトリ : E:\PyCaLiAI
GPU       : CUDA 12.8（unified_rank_v6 は LightGBM のみで torch 不要）
```

**依存**（`requirements.txt`）:
`streamlit, pandas, numpy, scikit-learn, lightgbm, catboost, shap, joblib,
matplotlib, japanize-matplotlib, tqdm, optuna, pytest`
（`requirements-lock.txt` / `requirements-nicegui.txt` も存在）

### 11.2 テスト

```bash
./venv311/Scripts/python.exe -m pytest tests/ -q
```
**実測: 73 passed / 20.5 s**（本書執筆時に実行）。

| ファイル | 対象 |
|---|---|
| `tests/test_production_line.py` | **本番ライン（v6 stack → compute_bets → validate → generate_results）の純関数ゴールデンテスト**。データ非依存（合成入力） |
| `tests/test_backtest.py` | `floor_to_unit`, `get_actual_payout` |
| `tests/test_ensemble.py` | `assign_marks`, `ensemble_predict` |
| `tests/test_kelly.py` | `kelly_fraction` |
| `tests/test_utils.py` | ユーティリティ |

⚠️ **カバレッジの穴（🟡 P2）**: `predict_weekly.parse_csv` に対するテストが無い。
§9.2 の 2 つの P0（定数刷り込み / 53 列問題）は、**「実 CSV を 1 本パースして
nunique > 1 を assert する」テストがあれば即座に検出できた**。

### 11.3 主要コマンド

```bash
# --- 週次運用（Vol. II §8 に詳細）---
.\weekly_nicegui.ps1                 # Phase A 土曜朝
.\weekly_nicegui.ps1 -BetsOnly       # Phase B（Cowork narrative 保存後）
.\weekly_nicegui.ps1 -Post           # Phase C 日曜夜

# --- 当日 T-10 ---
.\t10.ps1                            # レース毎タスク登録（通常は 9:00 自動）
.\t10.ps1 -Once <rid16>              # 1 レース即時
.\t10.ps1 20260614 -Loop -Dry        # 旧ループ方式・計算のみ

# --- モデル再構築 ---
python run_v6_pipeline.py            # calibrator + curve + audit
python optuna_v6_marks.py --n-trials 40
python build_pl_calibrators_serve.py
python scripts/audit_marks_by_class.py --model v6

# --- 監査・診断 ---
python audit_marks.py --model v6
python -m analysis.measure_serve_coverage      # serve baseline 再生成
python -m analysis.measure_settle_drift --check
python -m analysis.fit_harville_lambda
python -m analysis.fit_t10_blend

# --- 単発 ---
python pl_probs.py                   # PL 恒等式の自己検証
python export_weekly_marks.py --csv data/weekly/20260816.csv --model v6
PYTHONUTF8=1 python compute_bets.py --bundle reports/cowork_input/20260816_bundle.json --dry
```

⚠️ **Windows コンソールは cp932**。日本語を含む出力は `PYTHONUTF8=1` または
`PYTHONIOENCODING=utf-8` を付けないと文字化けする。多くのスクリプトは冒頭で
`sys.stdout.reconfigure(encoding="utf-8")` を実行している。

### 11.4 `lab/` の扱い

実験・研究スクリプト 97 本が `lab/<theme>/` に集約されている（2026-07-01）。
**再実行は必ず root から `python -m lab.<theme>.<name>`**
（cwd=root が sys.path に乗り `import utils` 等が解決する。直叩きは不可）。

テーマ: `experiments/ betting_lab/ bet_type_lab/ physics_gates/ backtest/ train/
audits/ features_dead/ pipelines_old/ sims/`

---

## §12 ファイル索引

### 12.1 本番ライン（最重要）

| 目的 | ファイル | 行数 |
|---|---|---:|
| bundle 生成（serve 本体） | `export_weekly_marks.py` | 598 |
| 1 レース分の推論・印・確率 | `export_marks_json.py` | 469 |
| **入力パーサ（欠損の発生源）** | `predict_weekly.py` の `parse_csv` | 2,023 |
| PL 厳密計算 | `pl_probs.py` | 240 |
| 買い方判定・妙味馬 | `betting_judgment.py` | 256 |
| UMAMI (xROI) | `umami.py` | 224 |
| serve 履歴再計算 | `serve_history_feats.py` | 327 |
| SHAP | `marks_shap.py` | 195 |
| kako5 履歴要約 | `kako5_summary.py` | 251 |
| **馬券構築** | `compute_bets.py` | 1,008 |
| 見送りガード | `validate_cowork_bets.py` | 380 |
| T-10 オーケストレータ | `t10_runner.py` | 677 |
| JV-Link オッズ (32bit) | `jvlink_odds.py` | 208 |
| 枠プラン | `build_bet_plan.py` | 240 |
| 決済・集計 | `generate_results.py` | 1,170 |
| 静的サイト生成 | `build_site.py` | 1,251 |

### 12.2 学習・較正

| 目的 | ファイル |
|---|---|
| 分割定義 | `build_dataset.py:40-41, 273-281` |
| master v2 生成 | `build_master_v2.py` |
| v6 学習 | `optuna_v6_marks.py`（LEAK_COLS:68 / ラベル:113 / 目的:261-270） |
| 較正器 | `build_pl_calibrators.py` / `build_pl_calibrators_serve.py` |
| 期待払戻カーブ | `build_payout_curve.py` |
| パイプライン一括 | `run_v6_pipeline.py` / `run_v5_pipeline.py` |
| 印監査 | `audit_marks.py` / `scripts/audit_v6_vs_v5.py` |

### 12.3 設定ファイル（本番挙動を決める JSON）

| ファイル | 決めるもの | 再生成 |
|---|---|---|
| `data/chaos_quantiles.json` | 生値→パーセンタイル | `build_chaos_quantiles.py` |
| `data/harville_lambda.json` | λ補正 PL の指数 | `analysis/fit_harville_lambda.py` |
| `data/t10_blend.json` | T-10 補正印の λ | `analysis/fit_t10_blend.py` |
| `data/serve_feature_baseline.json` | canary の基準 | `analysis/measure_serve_coverage.py` |
| `data/serve_code_maps.json` | 騎手/調教師 名→コード（223 / 242 エントリ） | （生成元は要確認 / UNKNOWN） |
| `data/class_prior_v6.json` | クラス別印信頼度 | `scripts/audit_marks_by_class.py` |
| `reports/audit_ev_bin_roi.json` | UMAMI 参照テーブル | `audit_ev_bin_roi.py` |
| `reports/settle_drift.json` | 決済ドリフト係数 | `analysis/measure_settle_drift.py` |
| `data/strategy_weights.json` | ⚠️旧 rule-based（Streamlit のみ） | `build_strategy_walkforward.py` |

---

**→ 続き: [Vol. II 馬券構築・運用仕様](VOL2_BETTING_OPS.md) / [Vol. III 検証史と課題](VOL3_VALIDATION_AND_OPEN_PROBLEMS.md)**
