# PyCaLiAI 完全仕様書（全 4 巻 結合版）
> 個別ファイルは docs/SPEC/ 配下。本ファイルは外部 AI への一括投入用の結合版。
> 版 1.0 / 2026-08-23

---

# PyCaLiAI 完全仕様書 (Complete Specification) — 索引

> **版**: 1.0 / **作成**: 2026-08-23 / **対象コミット**: `fbc442d44d` + working tree
> **想定読者**: 本リポジトリを初めて読む外部 AI（ChatGPT 等）およびエンジニア。
> **目的**: コードを渡されただけでは絶対に分からない「なぜそうなっているか」「何が既に死んだか」
> 「今どこが壊れているか」までを含めて、レビュー可能な単一の真実にする。

---

## 本仕様書の構成（4 巻）

| 巻 | ファイル | 内容 | 主な読者 |
|---|---|---|---|
| **Vol. I** | [`VOL1_SYSTEM.md`](VOL1_SYSTEM.md) | システム仕様: ドメイン前提 / アーキテクチャ / データ層 / 特徴量層 / モデル層 / 確率層 / 印層 / serve 層 / bundle スキーマ | コードを読む前に必ず |
| **Vol. II** | [`VOL2_BETTING_OPS.md`](VOL2_BETTING_OPS.md) | 馬券構築・運用仕様: 設計原理 / compute_bets 完全仕様 / ガード群 / 決済ドリフト / T-10 当日ライン / 週次フロー / 決済集計 / 公開層 | 馬券・運用コードを触る前に |
| **Vol. III** | [`VOL3_VALIDATION_AND_OPEN_PROBLEMS.md`](VOL3_VALIDATION_AND_OPEN_PROBLEMS.md) | 検証史・現在の課題・研究計画: 認識論的規律 / 死亡ルート台帳 / **欠陥台帳 (P0-P3)** / ガバナンス / 研究アジェンダ / レビュー依頼 | 改善提案をする前に必ず |
| **Vol. IV** | [`VOL4_CODE_REFERENCE.md`](VOL4_CODE_REFERENCE.md) | コードリファレンス（関数レベル）: 依存グラフ / 主要 14 モジュールの関数契約 / 横断的パターンと落とし穴 / テスト現状 | コードをレビューするとき |

**外部 AI へ**: 提案を書く前に **Vol. III の §3（死亡ルート台帳）と §9（提案してはいけないこと）** を必ず読むこと。
本プロジェクトは約 1 年・数百本の実験を経ており、汎用 AI が最初に思いつく改善案（特徴量追加 /
モデル変更 / EV 閾値 / Transformer / アンサンブル）は **すべて検定済みで死亡している**。
それを知らずに書かれた提案は無価値であるだけでなく、既に閉じた探索空間へ再投資させる害がある。

---

## 既存ドキュメントとの関係

| 既存 | 位置づけ | 本仕様書との関係 |
|---|---|---|
| `CLAUDE.md` / `AGENTS.md` | エージェント向け作業指示 + 引き継ぎ | 運用手順は Vol. II が正典。CLAUDE.md は要約 |
| `PYcALiAI_RESEARCH.md` | 研究開発憲章 (2026-08-09) | Vol. III の前身。本書は憲章の UNKNOWN を実測で解決し、欠陥台帳を追加 |
| `docs/STATUS_AND_HISTORY.md` | 死亡ルート一次資料 | Vol. III §3 が構造化して再掲 |
| `docs/version_ledger.md` | 版採否台帳 | Vol. I §6.7 / Vol. III §6 が引用 |
| `docs/hypothesis_registry.md` | 仮説事前登録簿 | Vol. III §8 が現況を追記 |
| `docs/marks_schema.md` | bundle スキーマ | Vol. I §10 が実装と突合して更新 |
| `docs/compute_bets_spec.md` | 馬券構築仕様 (2026-06-09) | **陳腐化**。Vol. II §2 が現行実装の正典（差分は Vol. II §2.9 に明記） |

---

## 本仕様書における記法

| 記号 | 意味 |
|---|---|
| **Fact** | コード・データを実測して確認した事実。行番号・数値付き |
| **Evidence** | 実験レポート（`reports/*.json` 等）に根拠がある主張 |
| **Hypothesis** | 検証待ちの仮説 |
| **Speculation** | 推測。根拠なし。設計判断の材料にしてはならない |
| **UNKNOWN** | 確認できなかった項目。推測で埋めない |
| 🔴 **P0** | 本番の出力を毀損している欠陥。即対応 |
| 🟠 **P1** | 定量的損失が実測されている欠陥 |
| 🟡 **P2** | ガバナンス／保守性の問題 |
| ⚪ **P3** | 整理・記録の問題 |

**本書に書かれた数値はすべて 2026-08-23 に実測したもの**であり、引用元のコマンド・ファイルを併記する。
再現できない数値は書かない。過去の会話やメモに由来する数値は「(memo)」と明示する。


---

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


---

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


---

# PyCaLiAI 完全仕様書 Vol. III — 検証史・現在の課題・研究計画

> 版 1.0 / 2026-08-23 / 実測ベース
> **外部 AI（ChatGPT 等）がこのリポジトリに提案を書く前に、この巻を最後まで読むこと。**
> 前提: [Vol. I](VOL1_SYSTEM.md)（システム）/ [Vol. II](VOL2_BETTING_OPS.md)（馬券・運用）

---

## 目次

- §1 認識論的規律（この章を飛ばすと提案が無価値になる）
- §2 予測層の天井（62%）とその証拠
- §3 死亡ルート完全台帳（再走禁止）
- §4 生存・採用済みレバー
- §5 **欠陥台帳（現在の課題）** ← 本巻の中核
- §6 ガバナンス問題
- §7 検証の非対称性（「死亡」の一部は検出不能死）
- §8 前向き検証中の仮説
- §9 研究アジェンダと優先順位
- §10 外部 AI へのレビュー依頼

---

## §1 認識論的規律

### 1.1 主張の 4 階層

本プロジェクトでは、すべての主張を次のいずれかに分類し、**混ぜない**。

| 階層 | 定義 | 例 |
|---|---|---|
| **Fact** | コード／データを実測して確認したもの。行番号・数値付き | 「`unified_rank_v6.pkl` は 120 特徴、α=0.0308」 |
| **Evidence** | 実験レポートに根拠がある。CI・n・期間付き | 「EV 選抜は prob-first に対し test ROI −13pt」 |
| **Hypothesis** | 反証条件が定義されているが未検証 | 「topdown の replay 改善は前向きでも再現する」 |
| **Speculation** | 根拠のない推測 | 「競馬では調教が重要だから調教特徴を増やせば効く」 |

**Speculation を設計判断の材料にしてはならない。** 一般論（「競馬では○○が重要」）を
「だから PyCaLiAI に効く」に飛躍させるのが最も多い失敗パターンである。

### 1.2 説明の上手さを採用根拠にしない

もっともらしい機序の説明は、その機序が実在する証拠ではない。
本プロジェクトで**機序は正しかったのに配線価値がゼロだった**例が多数ある:
- 「多頭数では市場の認知負荷が上がり中位馬が過小評価される」→ **複勝率の上昇とオッズの低下が完全に相殺**（H1 棄却）
- 「夏は牝馬が強い」→ **本当だが完全に priced**（夏牝単勝 ROI 67%）
- 「前走圧勝馬は次走勝率が高い」→ **本当（単調 12.7%→31.5%）だが ROI は不変**
- 「トラックバイアスは実在する」→ **実在するが約 9 割 priced**（内-外差 7pt → 0.48pt）

### 1.3 統計プロトコル（新実験の必須要件）

| 項目 | 要件 |
|---|---|
| 期間 | 複数期間（valid / test / 年別）で方向一致を確認。単一期間の点推定は不可 |
| CI | **paired bootstrap** を基本。レース単位リサンプル、seed 固定 |
| ECE | **10 ビン**（v6 の単一ビン `ECE_high_p` は過信と過小確信が相殺する欠陥版） |
| test | **2025 を封印**。valid の CI 下限が閾値を超えた最終 1 版のみ 1 回開封 |
| 変更 | **ONE CHANGE AT A TIME**（明示的な ablation / interaction のみ例外） |
| 実装 | 既存コードの大規模書換禁止。新規スクリプト（`analysis/` or `lab/`）+ config 切替 + 固定 seed |
| 記録 | 悪い結果も必ず残す。**死亡の記録こそ本プロジェクト最大の資産** |

### 1.4 特徴提案の必須フォーマット

新しい特徴／モデル／ポリシーを提案する場合、以下の 8 項目をすべて埋めること。
埋まらない項目があるなら、その提案はまだ提案の体をなしていない。

```
Hypothesis            : 何が真だと主張するか
Information           : どの情報源が、既存 120 特徴に無い何を持っているか
Mechanism             : なぜそれが着順に効くのか（因果の向き）
Why not already priced: なぜ市場（オッズ）がそれを織り込んでいないと言えるのか  ★最重要
Leakage risk          : as-of / OOF になっているか。生成順序をコードで確認したか
Expected effect       : 効果量の事前予測（pt 単位）と、それが MDE を超えるか
Failure mode          : 外れたときどう外れるか
Falsification criterion: 事前に固定した棄却条件
```

**「Why not already priced」が本プロジェクトで最も多くの提案を殺してきた項目**である。
公開情報（着順・人気・調教タイム・血統・セリ価格・回り適性）由来の特徴は
**priced 前提**で扱い、ROI / CLV ゲートを通ったときのみ採用する。

---

## §2 予測層の天井（62%）とその証拠

### 2.1 主張

> **オッズを使わない条件下で、◎（AI 1 位）の複勝圏率は約 62% が上限であり、
> モデル・特徴量の改良では破れない。**

### 2.2 証拠

| # | 証拠 | 出典 |
|---|---|---|
| E1 | v6 実測 ◎top3 = **62.08%**（真 OOS 6,878R） | `reports/audit_marks_v6.json` |
| E2 | 特徴量ブルートフォース **+100 / +300 / +1000（計 1400 特徴）** で採用ゼロ。1000 特徴版は gain 寄与 94.8% を占めるのに **ΔAUC +0.00007** でゲート未達 | `reports/feat_exam_300_result.json`, `lab/train/train_v1000.py` |
| E3 | 特徴プルーニングは**有意悪化**（P95 ◎top3 −0.72pt、CI 全負）→ 弱い特徴も集団で効いており、冗長性の除去では改善しない | `reports/ablation_prune.json` |
| E4 | 格・クラス変動（v7/v11、15 特徴フル再学習）で **+0.62pt CI[−0.20, +1.45] = 非有意** | `reports/audit_marks_v11.json` |
| E5 | 適性系（距離 / 場所 / 血統 / 回り左右）は **−0.3〜−0.68pt** | `reports/ablation_aptitude.json`, `ablation_direction.json` |
| E6 | 物理（Keller / pace）・EVT（極値統計）・統計力学（2 体 / 3 体 / 自由エネルギー）は ΔAUC 微小 or 符号逆 or 直交ゼロ | `lab/physics_gates/` |
| E7 | Transformer / Set Transformer は汎化ゼロ（配置情報は予測層で exploitable でない） | `exp_havoc_m1.py` |
| E8 | **独立に開発された別 AI「Keiba-ai」（58 特徴 / 2010-25 / NN 込 4 blend）も 58–62% 天井** | `project_keiba_ai_confirms_ceiling` |
| E9 | 競合ベンチマーク（2026-07-19 リサーチ）でも 62% は業界の地の値。競合の高い公表値はチェリーピック | `project_competitor_benchmark_2026` |

### 2.3 天井の正体

**Fact**: v6 の gain の **60.9% が過去走系**（`kako5_avg_pos` 13.88% + `前走確定着順` 11.72% だけで 25.6%）。
素朴な「過去走ランカー」（◎top3 ≈50%）に対し v6 の上乗せは **+12pt** に過ぎず、両者の相関は 0.74。

> **v6 の背骨は「過去の着順」であり、それは市場も見ている。**

**帰結**: モデルと市場は同じ情報を見て同じ結論に至る。
- 実測: v6 の◎は **市場 1 番人気（top3 64.33%）に −2.3pt 負けている**
- 負けの局在: **◎飛び（本命が 3 着以内に来ない）が 37.9%**（`diag_upset_decomposition.py`）
- 較正は完璧（ECE 0.001–0.019）→ **「確率が悪い」のではなく「市場も同じ確率を出している」**

### 2.4 天井の唯一の突破口（採用済み）

**T-10 オッズブレンド**（Vol. II §6.3）:
```
u = log(p_win) + 1.5 · log(π_de-vig)
◎top3: 61.67% → 65.08%   Δ CI95 [+2.46, +4.30]（有意）
```
「62% 天井」は **オッズを使わない場合**の話である。
ただし市場単独（64.33%）に対しては **CI[−0.09, +1.56] = 有意に勝っていない**。
現在は**表示専用**で、買い目には干渉していない。

---

## §3 死亡ルート完全台帳（再走禁止）

### 3.0 死因の型分類（重要）

死亡ルートを一律に扱うと、再検定すべきものとすべきでないものが混ざる。
本仕様書では死因を 5 型に分類する。

| 型 | 定義 | 再検定の可否 |
|---|---|---|
| **PRICED** | 現象は実在するが市場が既に織り込み済み。ROI に変換できない | ❌ 原理死。再走禁止 |
| **ORACLE** | 検証時に未来情報（確定オッズ等）を使っており、実運用では入手不能 | ❌ 原理死 |
| **UNCASHABLE** | エッジは実在するが pari-mutuel の仕組みで換金できない（CLV 等） | ❌ 原理死 |
| **MIRAGE** | 発見期の点推定が OOS／別期間で消滅・符号反転した | ⚠️ 再走は原則禁止。ただし機序が別なら別実験として可 |
| **UNDERPOWERED** | 効果があっても検出できない標本しかなかった（MDE ≈1pt） | ✅ **データが倍増した将来時点での再検定は正当** |

### 3.1 予測層・特徴量（すべて本番 v6 土俵で検定、採用ゼロ）

| ルート | 型 | 死因 | 出典 |
|---|---|---|---|
| 特徴量ブルートフォース（1400 特徴） | UNDERPOWERED/PRICED | gain 94.8% でも正味 0、ΔAUC +0.00007 | `lab/train/train_v1000.py` |
| 格・クラス変動（v7/v11） | PRICED | +0.62pt CI 非有意 | `reports/audit_marks_v11.json` |
| 適性系（距離/場所/血統） | PRICED | −0.3〜−0.68pt | `reports/ablation_aptitude.json` |
| **回り適性（右/左）as-of** | PRICED | 場所が回りを確定させ v6 が既に吸収。◎top3 −0.68pt / ΔAUC +0.00015 / CI 上限 0 | `analysis/ablation_direction_asof.py` |
| 特徴プルーニング | — | **有意悪化** −0.72pt CI 全負 | `reports/ablation_prune.json` |
| Elo / Glicko / 血統 embedding / race level | PRICED | 冗長・逆効果 | `lab/features_dead/` |
| **セリ取引価格** | PRICED | 生 coef 0.62 (p<.001) → 市場統制で 0.055 非有意 / 91% 吸収 / 年別符号反転 / ΔAUC 負 / ROI 0.74 < 市場 | `exp_auction_decisive.py` |
| 物理（Keller / pace_fit / draft） | PRICED | pace_fit 符号逆、draft 冗長。生存は `kl_lscap` のみ | `lab/physics_gates/` |
| EVT（極値統計・分散特徴） | PRICED | fukusho ΔAUC +0.0017、直交 part_corr < 0.01、H2 符号反転は realized σ のトレンド汚染 | `evt_eval.py` |
| 統計力学（2 体 / 3 体 / joint / 順序 / 自由 E） | — | 2 体 M1 は umaren ECE −33% で実在。3 体 β3 は物理的に有意だが M1 上乗せ +0.07pt | `lab/physics_gates/gate2_3body.py` |
| 夏専用 / regime 専用モデル | — | 6 粒度すべてで否定。プール学習が最良（夏 test ◎top3 60.2% vs 全プール 62.6%） | `exp_summer_upweight.py` |
| 「夏は牝馬」 | PRICED | 現象は本物（牝/牡比 0.65→0.92）だが完全 priced（夏牝単勝 ROI 67%） | 同上 |
| 馬場（クッション/含水） | PRICED | 馬場状態に吸収 | `build_baba_feats.py` |
| トラックバイアス（当日 within-card） | MIRAGE | 脚質 leak を潰すと ROI 78%（控除壁下）、valid/test 方向不一致 | `analysis/daybias_within_card_test.py` |
| トラックバイアス（クロスデイ 土→日） | PRICED | 前残り持続は本物（Spearman +0.36 / 反転 7.6%）だが日曜期待前馬複勝 ROI 75.9% ≒ baseline 74.9%（+1.0pt = ゼロと不可分） | `analysis/crossday_bias_test.py` |
| 不利代理（`kako5_hidden_good_count` の次走 ROI） | PRICED | test 単勝 +4.7pt 非有意 / 複勝 +0.3pt | `analysis/measure_hidden_good_roi.py` |
| 過去走全ラップ CSV | ORACLE+PRICED | S2 pace-fit ΔAUC −0.0017、馬名/ID 無で JOIN 不能、先頭 1,678 行が当日 leak | `project_lap_csv_evaluated` |
| 前走圧勝ルール | PRICED | 勝率は単調（1.0s → 31.5%）だが ROI 全セル 0.6–0.9。圧勝×人気薄はむしろ最悪 | `exp_rule_mining.py` |

### 3.2 モデルアーキテクチャ

| ルート | 死因 |
|---|---|
| Transformer / Set Transformer | 汎化ゼロ。gain 4.8% 使うのに ΔAUC −0.004 |
| Stacking meta | valid Brier 改善 ≤0.001、Isotonic 出力が常に >0.50 で 100% フォールバック（2026-03-28 廃止） |
| MoE 距離別 expert | `models/expert_*_rejected.pkl` の命名通り不採用 |
| custom profit loss | ログで棄却 |
| Deep Value Net（◎単勝 ROI 100% 狙い DL） | **確信度と ROI が単調逆相関**（乖離大 = モデル誤り）。4 反復 + 規律スイープで dead。◎単勝床 = test 80.6% |
| 「消しモデル × 合成印」 | `−A_z` 自体が kill-AUC 0.7748 = 消しは強いの逆数。残差 AUC 0.48–0.58。3 層全死 |
| v7 (ワイド ROI 直接最適化) | Δ ROI −1.50pt（valid metric 直接最適化の過学習） |
| v8 (course_affinity) | **自レース込み集計の in-sample leak** |
| v9 (trunc=3) / v10 / v11 | 採用ゲート未達（v10 のプロトコルとバグ修正は資産） |

### 3.3 市場・オッズ

| ルート | 型 | 死因 |
|---|---|---|
| Benter blend / shrinkage | — | test ROI 0.709（全 τ）、対市場 ΔR² 負。α≈0.3 のみ REAL_BUT_UNPROFITABLE |
| crosspool 裁定（88–92%） | ORACLE | 確定オッズの二重 oracle。betable でない |
| EV 価格エッジ選抜（gate_q2 等） | MIRAGE | ≤2023 で 125–130% → test で 64–67% に崩壊、CLV 負 |
| EV グリッド網羅（289 セル = 会場×芝ダ×距離帯×券種×EV 閾値） | MIRAGE | 全券種 control で控除床張り付き（CI 上限すら 100% 未達）。実オッズ単勝は EV を上げるほど悪化（optimizer's curse）。相性セル 12 個は多重比較ノイズ（CI 有意 0 / 期待 FP 14.5） |
| 市場内部裁定 / Dr.Z | PRICED | 非効率は実在（fair が市場を予測で上回る、独立再現）が **控除 > 非効率**。Kelly dutch 85.5% CI[83,88] で全破産 |
| オッズ軌跡（−60 分 / 10 分毎） | — | 予測 ΔAUC −0.0016、市場 blend γ 非有意 |
| **CLV** | UNCASHABLE | pari-mutuel は確定オッズ払い。CLV を換金する経路が存在しない |
| **オッズを予測特徴に入れる** | — | AUC↑ は市場価格の写像 = ROI 不変 + serve 不可。**禁止**（T-10 の bet 時ブレンドとは別物） |

### 3.4 馬券・ポリシー

| ルート | 型 | 死因 |
|---|---|---|
| **EV 閾値による銘柄選抜** | MIRAGE | **有害 −13pt**。prob-first 化済み（EV はフロア/配分に降格） |
| 学習型ベッティング（learn_bet v1–v5, NN） | — | 9 時価格を超えず。単複は市場 75%/AI 25% でエッジ薄く控除率超えず |
| ポリシー空間ブルートフォース（6 家系 / OOS / paired-boot） | — | ユーザーのトリガミ床ルールを **dominate 不可**（パレートフロンティア上） |
| 三連複 value AI | ORACLE | 実プールオッズ入手不能。TARGET は組合せオッズを出せず、test.csv は単複馬連ワイド馬単のみ。合成オッズは循環 |
| 三連複の買い方 6 ファミリ総当たり | — | 全部控除壁。買い方改善は +13pt（box 69 → probfirst 82）だが損益分岐未超 |
| 三連複フォメの列並べ | MIRAGE | 妙味順ヒモは**有意に有害**（−7〜−13.5pt CI 全負）。最良は軸 mkt / 2 列 ai / 3 列 mkt の 2-3-6 で ROI 81.6%（未配線） |
| **WIN5** | — | 全史 772 回で ROI 0.837 CI[0.436, 1.338]。≤2017 = 0.518 vs ≥2018 = 1.062 → Phase1 の 1.09 は直近窓の蜃気楼。群衆 R = 0.703 = 控除ピッタリ |
| 枠連 | — | 初検定 ROI 84.8% = 壁内。**券種空間の完全踏破が完了**（全券種で群衆超過 ≈ +7pt 一定） |
| ワイド専用 AI（210 config 総当たり OOS 最適化） | — | 最良 = ◎軸 2 点・堅い回（chaos≤0.79）で test 85.9%（1,872R、valid 一致 = 本物）だが**黒字化 config は 0/210**。控除壁が真因（トリガミは的中の 5% のみ） |
| 穴 × under カット（+10.2pt） | MIRAGE | OOS で否定（under 65.9% vs not 67.1%） |
| 市場一致ゲートによる黒字反転 | MIRAGE | in-sample overfit + era artifact。T10 を救えていない |
| **clean-band 参戦ゲート** | MIRAGE | +5.31pt → 2026 as-served で **符号反転**（clean 77.6% < 帯外 87.9%）→ **配線撤回**。**点推定配線の戒め** |
| 「市場エッジ = 利益」（gutchi 実打 10 例） | MIRAGE | leave-one-out で崩壊（2 レース抜くと ROI 84.3% = 控除壁）。事後ラベルの循環、n=10 で有意ゼロ、選択バイアス |
| ダート（特に短距離）は構造的に ROI が低い | MIRAGE | 2026 実ベット 395R で芝 92.9% vs ダ 55.0%（差 +37pt）→ **歴史大標本 31,093R で ◎単勝 ROI 芝 79.0% vs ダ 79.0% = 小数点まで同一**。2026 の差は芝 202R の右尾による蜃気楼 |
| 「土曜全休して観測に徹する（見）」 | — | 観測に参加は不要。参加だけ削る純損 |

### 3.5 その他

| ルート | 死因 |
|---|---|
| G1 直後ローテの RPCI 過小評価（H4） | 検証条件 n≥200 未達（**PENDING**、閉じていない） |
| field_spread × 市場歪み（H1） | 棄却。独立検証で +0.7%（保留条件未達）。市場は「混戦 = 荒れやすい」を正確に織り込んでいる |
| class_gap × 降級効果（H2） | 棄却寄り。3 期間中 2 期間で逆転、独立検証 n=27 |
| EV≥3.0 除外ルール（H3） | 現行維持（EV4.0+ ROI 91.7% は条件① 達成だが EV3.0-4.0 が 79.3% で条件② 未達） |
| 重 × 16 頭除外ルール | 削除（n=130 単年で作ったルールを n=376 の 3 年観察が否定） |

### 3.6 過去に実際に踏んだリーク（再発防止リスト）

| # | リーク | 検出方法 |
|---|---|---|
| L1 | v8 `course_affinity` の**自レース込み集計** | `build_course_affinity.py:120-123` |
| L2 | crosspool 88–92% = 確定オッズ oracle | 実運用不能に気づいた |
| L3 | ラップ CSV 先頭 1,678 行が当日データ | 行の中身を目視 |
| L4 | realized σ のトレンド汚染（EVT H2 の符号反転） | detrend で消えた |
| L5 | `strategy_weights.json`: test ROI で採用 → 同じ test で評価 | 30 エントリ全部が循環 |
| L6 | 当日バイアスの脚質 leak | leak を潰すと ROI 78% に落ちた |

**原則**: 「結果的に少ししかリークしていない」は許容しない。
**生成順序をコードで確認するまで leak-safe と主張しない。**

### 3.7 教訓ケーススタディ — clean-band ゲート

本プロジェクトで最も高くついた教訓なので単独で記す。

```
2026-06-18  OOS 検証（2024fit → 2025eval, analysis/test_race_selection_oos.py）
            クリーン帯（エントロピー下位 1/3）のみ ◎複勝 ROI ≈90% / 単勝 85% / top3 76%
            mid 80.8% / chaotic 82.5% は控除床近傍
            → 「最も負けない線 = クリーン帯に絞る」として CLEAN_BAND_MAX=0.33 を配線
2026-07-25  二段化（帯外は見送りでなく消化枠降格）を配線。実測 +5.31pt
2026-07-23  2026 as-served 再検証（analysis/reverify_clean_band_2026.py, 686R）
            → clean 77.6% < 帯外 87.9% で符号反転
            → 配線中止。機構はコードに残すが main から demote_budget を渡さない
```

**教訓（プロジェクト共通ルール化）**:
> どの点推定も単独で主軸化しない。**配線より「最新期間での CI 付き再検証」に手間を割く。**

同じ構造の候補が現在も複数ある（§8）。同じ轍を踏まないこと。

---

## §4 生存・採用済みレバー

**わずか 8 件しかない。** これが約 1 年・数百本の実験の全収穫である。

| # | レバー | 効果 | 状態 |
|---|---|---|---|
| S1 | **T-10 オッズブレンド補正印** | ◎top3 61.67% → 65.08%（CI[+2.46, +4.30]） | 表示専用。買い目未配線 |
| S2 | **λ補正 PL（Lo–Bacon-Shone）+ topdown エンジン** | replay 82.8%（in-sample） | 2026-08-09 既定化。前向き監視中 |
| S3 | **prob-first**（EV 選抜の廃止） | +13pt | 配線済 |
| S4 | **適応トリガミ床**（ユーザー発ルール） | トリガミ −74% / クリーン勝ち +18% | topdown に内蔵 |
| S5 | **構築層 4 修正**（vb-◎ペア撤去 / ◎単勝は妙味時のみ / cap6→5 / p×boost 配分） | replay 74.1% → 78.2% | 配線済（in-sample 注意） |
| S6 | **馬単・三連単の全廃** | 実測 ROI 22.1%（馬単）を除去 | 配線済 |
| S7 | **serve skew 修復（`_SERVE_RENAME`）+ canary（fail-closed）** | 補正 +2.97pt / 調教 +0.33pt | 配線済。ただし 現行baselineではcoverage < 0.40が34特徴 |
| S8 | **serve 条件 fit calibrator** | ECE 複勝 −36% / 馬連 −29% | 配線済。ただしマスク不整合（§5 P0-3） |
| S9 | **決済ドリフト補正** | EV の系統的過大を補正 | 単勝/複勝/ワイド配線済 |
| S10 | **見送りガード（`validate_cowork_bets`）** | 全 23R 購入事故の再発防止 | fail-closed |
| S11 | **◎前走圧勝 conf ボーナス** | 的中率レバー（ROI は不変） | `build_bet_plan` に配線済 |

**診断として価値があるもの（レバーではないが重要）**:
- 負けは **◎飛び 37.9%** に局在（組み合わせ層は健全: ◎来時 ROI 131–139%）
- v6 は gain 60.9% が過去走支配
- 較正はほぼ完璧（ECE 0.001–0.019）
→ **問題は予測でも確率でもなく「市場との重なり」**

---

## §5 欠陥台帳（現在の課題）

**この節が本仕様書の中核である。** すべて 2026-08-23 に実測して確認した。

### 5.1 サマリ

| ID | 深刻度 | 課題 | 影響（実測） | 修正コスト |
|---|---|---|---|---|
| **P0-1** | 🔴 | 騎手/調教師ローリング複勝率が全馬同値の定数 | gain **10.58%** が判別力ゼロ | 小 |
| **P0-2** | 🔴 | 着度数 CSV パーサが 55 列を期待、実データは 53 列 | gain **2.01%** + 当日馬体重が全滅。2026 全期間 | 極小 |
| **P0-3** | 🔴 | serve 較正器のマスク（14特徴）が実態（coverage < 0.40は34特徴）と乖離 | 較正が「1/3 しか壊れていない世界」で fit | 中 |
| **P0-4** | 🔴 | `Ｒ`（全角）を parse_csv が半角 `R` で作っている | gain **0.68%** が −9999。**1 行で直る** | 極小 |
| **P1-1** | 🟠 | serve canary が「ずっと死んでいる特徴」を構造的に見逃す | 上記 3 件がすべて無検知 | 小 |
| **P1-2** | 🟠 | 前走詳細 15 列が訓練中央値の定数刷り込み | gain **≈7.5%** | 中 |
| **P1-3** | 🟠 | 複勝が `below_takeout` 確定なのに topdown は複勝アンカーに寄せている | 複勝 ROI 69.0% CI[58.7, 79.4] | 要判断 |
| **P1-4** | 🟠 | topdown が in-sample replay のみで本番化されている | 前向き 72 bets / 必要 300 | 観測のみ |
| **P1-5** | 🟠 | master 側で 3 特徴が 100% NaN（serve では 90% 埋まる逆非対称） | gain 0 だが情報を捨てている | 中 |
| **P2-1** | 🟡 | `t10_runner.py` が削除済み `gutchi_brain` を import | dead code 90 行 + 毎レースのエラーログ | 極小 |
| **P2-2** | 🟡 | v6 が自ら定めた採用ゲートを満たさずに本番化 | ガバナンス矛盾 | 要判断 |
| **P2-3** | 🟡 | test 2024-25 が 7 回以上開封済み | すべての test 数値が多重比較で選ばれた値 | プロトコル |
| **P2-4** | 🟡 | topdown が bundle の較正済 `pair_probs` を使わず未較正 λPL を再計算 | 較正の恩恵を捨てている | 小 |
| **P2-5** | 🟡 | 見送り 4 条件が 4 ファイルに独立ハードコード | 同期漏れリスク | 小 |
| **P2-6** | 🟡 | `docs/compute_bets_spec.md` が実装から大きく乖離 | 誤読リスク | 小 |
| **P2-7** | 🟡 | `predict_weekly.parse_csv` にテストが無い | P0-1/P0-2 を検出できなかった | 小 |
| **P2-8** | 🟡 | 較正器の Optuna CV が valid 内ランダム KFold（時系列でない） | 楽観バイアスの可能性 | 中 |
| **P2-9** | 🟡 | `validate_cowork_bets.ALLOWED_KINDS` に禁止券種「馬単」が残存 | Cowork 手動経路の穴 | 極小 |
| **P2-10** | 🟡 | `cowork_results.json` 凍結検知が Warn 止まり | 集計凍結の再発余地 | 極小 |
| **P3-1** | ⚪ | `models/` 66 ファイル / seed 変種の用途記録なし | 保守性 | 小 |
| **P3-2** | ⚪ | `docs/cowork_prompt.md` の 1 行目に `yaru` が混入 | — | 極小 |
| **P3-3** | ⚪ | `parse_csv` が str を渡されると `UnboundLocalError` | エラーメッセージが不明瞭 | 極小 |
| **P3-4** | ⚪ | TACT 公開線の成績が未集計 / バージョン未固定 | 検証不能 | 小 |

### 5.2 回収可能な gain の合計

| ID | 対象 | gain | 修正コスト |
|---|---|---:|---|
| P0-4 | `Ｒ` の全角/半角リネーム | 0.68% | 1 行 |
| P0-2 | 着度数 CSV 53 列対応（`horse_fuku10/30`） | 2.01% | 小 |
| P0-1 | 騎手/調教師 stats の再 merge | 10.58% | 小〜中 |
| P1-2 | 前走詳細ブロックの as-of 再計算 | ≈7.5%（一部は回収不能の可能性） | 中 |
| — | **合計** | **≈20.8%**（2026-06旧監査の28.15%を基準にした当時の回収可能量。現行baselineは14.88%） | |

**残り ≈7.3%** は cat 特徴（生産者 / 馬主 / 毛色 / 前走場所 / 指定条件 / 限定 / 芝(内・外) /
性別限定 / ブリンカー / 父タイプ名）で、週次 CSV に元データが無いため回収は難しい。

⚠️ **繰り返すが gain% ≠ 精度**。20.8% の gain 回収が ◎top3 を 20% 上げることは絶対にない。
実測された offline→serve の残差ギャップは **約 1pt** であり、期待値はその範囲内である。
**それでも「捨てている情報を届ける」ことは、新しいアイデアを必要としない唯一の確実な仕事**である。

---

### 🔴 P0-1 — 騎手/調教師ローリング複勝率が定数（gain 10.58%）

**実測（`data/weekly/20260816.csv`、478 頭）**
| 特徴 | notna | **nunique** | 値 | gain |
|---|---:|---:|---|---:|
| `jockey_fuku90` | 1.000 | **1** | 0.200 | **6.79%**（モデル第 4 位） |
| `trainer_fuku90` | 1.000 | **1** | 0.211 | 1.92% |
| `jockey_fuku30` | 1.000 | **1** | 0.200 | 1.32% |
| `trainer_fuku30` | 1.000 | **1** | 0.200 | 0.55% |

**根本原因**（`predict_weekly.py:466-485`）:
```python
if code_col in df.columns:     # ← "騎手コード" は週次 CSV に存在しない → 常に False
    ... stats.merge ...
else:
    for col in stat_cols:
        df[col] = _ROLLING_TRAIN_MEDIANS.get(col, 0.200)    # ← 常にこちら
```
週次 TARGET CSV は `騎手`（名前）を持つが `騎手コード` を持たない。
`data/jockey_stats.csv`（**342 行**）と `data/trainer_stats.csv`（**329 行**）は存在するのに
**一度も使われていない**。

**皮肉な点**: `serve_history_feats.fill_history_features()` が **この直後に**
`serve_code_maps.json`（騎手 223 / 調教師 242 エントリ）から `騎手コード` / `調教師コード` を復元している。
**コードは手に入るのに、定数刷り込みはその前に完了している。**

**★ さらに重要な発見 — これは本来「運用の問題」である**

`predict_weekly.py` は週次 CSV の **5 つの列数フォーマット**を処理できる:
```python
len(cols) == 33 → HORSE_COLS_33 (33 列)
len(cols) == 46 → HORSE_COLS_46 (46 列)
len(cols) == 48 → HORSE_COLS_48 (48 列)  ← 末尾 2 列が 騎手コード / 調教師コード
len(cols) == 49 → HORSE_COLS_49 (49 列、馬体重 3 列入り)
len(cols) == 99 → HORSE_COLS_99 (99 列、3 走前まで)
```
**実測: 現在エクスポートされている週次 CSV は 46 列**（`{19: 70, 46: 513}` @ 20260816、
2026-04 以降の 5 ファイルすべて 46 列）。

つまり **`HORSE_COLS_48` は「騎手コード付きでエクスポートすれば merge が動く」設計として
最初から用意されている**。コードは正しく、**TARGET のエクスポート設定が 46 列になっている**
というのが真の原因である。

**修正案（優先順）**

| 案 | 内容 | 長所 | 短所 |
|---|---|---|---|
| **A（推奨）** | TARGET の出走表エクスポートを **48 列形式**（騎手コード・調教師コードを含む）に変更する | **コード変更ゼロ**。設計どおりに動く。leak なし | ユーザーの手動設定変更が必要。過去の 46 列 CSV は救えない |
| B | `export_weekly_marks.py` の `fill_history_features()` 直後に、復元された `騎手コード` / `調教師コード` で `jockey_stats.csv` / `trainer_stats.csv` を再 merge | 過去分にも効く。自動 | `serve_code_maps.json` の名前→コード解決（騎手 223 / 調教師 242 エントリ）に漏れがあると一部欠損 |
| C | A + B の両方（A が効かない週の保険として B） | 最も堅牢 | — |

**まず確認すべきこと**: TARGET で 48 列形式のエクスポートが実際に可能か。
可能なら案 A を試し、1 週分の CSV で `parse_csv` の `jockey_fuku90.nunique() > 1` を確認する。

**⚠️ 注意点（これを守らないと leak になる）**:
0. **案 A / B のどちらでも、`jockey_stats.csv` の as-of 性の問題は残る**（下記 1）。
   48 列形式で得られるのは「騎手コード」であって「その時点の複勝率」ではない。
1. `data/jockey_stats.csv` は **静的スナップショット**（2026-07-29 更新）であり、
   学習側の `shift(1)` ローリングとは定義が違う。**過去日付に対して使うと未来情報が入る。**
   → **前向き serve 専用**。バックテストに使ってはならない
2. 定義差（窓幅・as-of 基準）が残るなら、serve 較正器も再 fit する必要がある（P0-3 と連動）
3. **ONE CHANGE AT A TIME**: 修正後、`analysis/measure_serve_coverage.py` で baseline を
   再生成し、`analysis/diag_serve_full_impact.py` 等で ◎top3 / ECE の変化を測ること

**期待効果（Hypothesis）**: 実測された offline→serve ギャップの残差は約 1pt。
gain 10.58% の回収がそのまま 10% の精度改善になることは**ない**（gain% ≠ 判別力）。
期待は **◎top3 +0.3〜1.0pt 程度**。効果量が MDE（≈1pt）近傍なので、
**CI 付きで測って有意でなければ「配線したが効果は測定限界以下」と正直に記録する**こと。

---

### 🔴 P0-2 — 着度数 CSV パーサの列数ドリフト（gain 2.01% + 当日馬体重）

**実測**: `data/tyaku/*.csv` の馬行は**すべて 53 列**。
```
20260816: {19: 70, 53: 513}    20260419: {19: 72, 53: 526}
20260607: {19: 46, 53: 366}    20260802: {19: 70, 53: 489}    20260815: {19: 70, 53: 511}
```
**パーサの期待値**（`predict_weekly.py:259`）: `elif len(cols) == 55 and ...`
`TYAKU_HORSE_COLS` の長さも 55。

**結果**: `rows` が空 → `_load_tyaku()` が `None` を返す →
- `horse_fuku10` = 0.286（全馬）/ `horse_fuku30` = 0.312（全馬）
- **当日馬体重・増減の取り込みも同時に失われる**

**影響範囲**: `data/tyaku/` に 44 ファイルが配置されており、**2026 シーズン全期間で機能していない**。
ログにも出ず、canary も検知しない（`baseline_cov` に 0.0 として焼き込まれているため）。

**修正案**: 列数を 53/55 の両対応にし、`TYAKU_HORSE_COLS`（現在 55 要素）を実データと突合して
53 列版を確定する。**さらに `_load_tyaku()` が `None` を返したら WARNING をログに出す**こと
（現状は完全に無言で、成功時のみ `着度数CSV読み込み済` が出る = 出ないことに気づけない）。

**P0-1 との共通構造**: どちらも **「TARGET のエクスポート形式が変わったのにパーサ側の
列数分岐が追随していない」**という同一クラスの欠陥である。
週次出走表は 46/48 列の選択（P0-1）、着度数は 53/55 列（P0-2）。
→ **恒久対策**: 列数分岐にヒットしなかった行数を数え、
「行を 1 行も取れなかった / 取れた行が期待の半分未満」なら WARNING を出す共通ヘルパを作る。
（`export_weekly_marks.py` の品質ゲート G2 が出走表側で同じ役割を果たしているが、
`_load_tyaku` にはそれが無い）

**⚠️ 当日馬体重の扱い**: 馬体重はモデル特徴に無い（Vol. I §4.4）。
tyaku を修復すると `馬体重` / `馬体重増減` が df に入るが、**モデル特徴に追加してはならない**
（当日情報の予測特徴化は別途 leak 検証が必要）。回収対象は `horse_fuku10/30` のみ。

---

### 🔴 P0-3 — serve 較正器のマスクが実態と乖離

**実測（`models/pl_calibrators_v6_serve.pkl` のメタ）**:
```
serve_mask_numeric = ['Ｒ','前走走破タイム','前走日付','前走レースID(新)','前走レースID(新/馬番無)','母馬']  (6)
serve_mask_cat     = ['芝(内・外)','前走場所','前好走','毛色','馬主(最新/仮想)','限定','指定条件','ブリンカー']  (8)
                                                                                          計 14
fit_split = "valid=2023 (serve マスクスコア)"   n_races = 3456
```
**実際に serve で死んでいる特徴**（`data/serve_feature_baseline.json` 実測）: **34特徴（coverage < 0.40。gain > 0は32件）/ gain 14.88%**。

マスクの出所は `serve_skew_eval.py:69-76` のハードコード定数
`SERVE_DEAD_NOW_EXACT`（6 件）+ `SERVE_DEAD_NOW_CAT`（8 件）で、
**2026-06 時点の状態で凍結**されており、以降の実測と同期していない。

**帰結**: serve 較正器は「本番の 1/3 しか壊れていない世界」のスコア分布で fit されている。
「ECE 複勝 −36%」という成果は本物だが、**残りのミスマッチは未補正**。

**修正案**:
1. `serve_skew_eval.py` のハードコード定数を廃し、
   **`data/serve_feature_baseline.json` を単一ソースにする**
   （`baseline_cov < 0.40` の特徴を自動的にマスク対象にする）
2. `build_pl_calibrators_serve.py` を再実行して較正器を作り直す
3. `reports/calibrators_v6_serve_eval.json` で ECE の変化を確認
4. ⚠️ **P0-1 / P0-2 を先に直すなら、その後で較正器を作り直すこと**
   （順序を誤ると 2 回作り直すことになる）

**推奨順序**: `P0-1 + P0-2 修正` → `measure_serve_coverage で baseline 再生成` →
`P0-3 で較正器再 fit` → `ECE と ◎top3 を CI 付きで測定` → 記録。

---

### 🔴 P0-4 — `Ｒ`（レース番号）の全角/半角ミスマッチ（gain 0.68%、1 行で直る）

**実測**:
```python
df = parse_csv(Path('data/weekly/20260816.csv'))
'Ｒ' in df.columns   # → False   （モデルが要求する全角）
'R'  in df.columns   # → True    （parse_csv が作る半角）
```
モデルの `feature_cols` には **全角 `Ｒ`** が入っている（master_v2 の列名が全角のため）。
一方 `predict_weekly.RACE_COLS` は **半角 `R`** を使っている。
結果、`export_weekly_marks.py` の「不足列補完」で `Ｒ` は NaN → **−9999 に潰れる**。

**修正**: `export_weekly_marks._SERVE_RENAME` に 1 行足すだけ。
```python
"R": "Ｒ",
```
`_serve_rename` の適用条件は `k in df.columns and v in feats and v not in df.columns` なので、
既存の安全弁がそのまま効く（`Ｒ` が既にあれば何もしない）。

⚠️ **効果は小さい**（gain 0.68% = レース番号）。だが**修正コストが実質ゼロ**であり、
かつ「全角/半角の不一致で特徴が死ぬ」というクラスの欠陥が他にも無いかを
点検するきっかけになる（`deep_zen` / NFKC 正規化が `build_site.py` にはあるが
serve 経路には無い）。

---

### 🟠 P1-1 — serve canary の構造的死角

**現行の判定**（`export_weekly_marks.py:555-563`）:
```python
for col in feats:
    exp = base_cov.get(col)
    if exp is None or exp < 0.40:
        continue                    # ← 既知 dead は監視対象外
    cur = feature_coverage(df[col], ...)
    if cur < 0.20 and cur < exp * 0.40:
        silent_deaths.append(...)
```
canary は「**昨日まで生きていた特徴が今日死んだ**」を検知する装置である。
`baseline_cov = 0.0` として baseline に焼き込まれた特徴は永久に監視対象外になる。

つまり **P0-1 / P0-2 のような「ずっと死んでいる」欠陥は、設計上、絶対に検知されない**。

**修正案**:
1. baseline 生成時（`analysis/measure_serve_coverage.py`）に、
   **「gain > 0 なのに serve cov < 0.40」の特徴を "REGRESSION CANDIDATE" として別枠で列挙**し、
   その総 gain% をレポートに出す
2. bundle 生成時に「serve で死んでいる特徴の gain 合計 %」を **ログに毎回出す**
   （現在の 14.88% を数字として毎週見える状態にする）
3. その値が閾値（例: 35%）を超えたら品質ゲートで止める

**設計思想**: canary は差分検知、この提案は絶対水準の可視化。両方必要である。

---

### 🟠 P1-2 — 前走詳細 15 列の定数刷り込み（gain ≈7.5%）

`predict_weekly.py:526-560` が意図的に訓練 valid 中央値で埋めている:
```
前PCI=49.0, 前走RPCI=48.5, 前走PCI3, 前走平均1Fタイム, 馬齢斤量差=−1,
トラックコード(JV)=23, 前走トラックコード(JV)=23, 前走競走種別=13,
前走出走頭数=15, 前走馬体重=472, 前走馬体重増減=0,
騎手年齢=30, 調教師年齢=53, 休み明け～戦目=2, 斤量体重比
```
このうち **`前走馬体重` / `前走馬体重増減` / `前走出走頭数` / `前走競走種別` /
`前走場所` / `前走日付` / `前走トラックコード(JV)`** は
`data/_horse_history.parquet` から **as-of で厳密に再計算可能**である。

**修正案**: `serve_history_feats.NUM_FEATS` / `CAT_FEATS` を拡張する。
`fill_history_features()` は既に「レース日より厳密に前の走のみ」で as-of 計算しており、
**この関数に追加するのが最も leak-safe**。

**⚠️ 注意**: `前PCI` / `前走RPCI` / `前走PCI3` / `前走平均1Fタイム` は
TARGET 由来の指数であり parquet に無い可能性が高い（要確認 / **UNKNOWN**）。
これらは回収不能かもしれない。

---

### 🟠 P1-3 — 複勝 `below_takeout` と topdown の複勝アンカー設計の衝突

**実測（`data/cowork_results.json`）**:
```
複勝: 390 点 / 的中率 48.2% [43.3, 53.2] / 投資 ¥704,100 / ROI 69.0% / CI95 [58.7, 79.4]
      → roi_verdict = "below_takeout"（真に控除率未満の証拠）
```
`roi_verdict` が `below_takeout` になったのは **馬単（22.1%）と複勝（69.0%）の 2 券種のみ**。
馬単は全廃済み。**複勝は残っている**。

一方 topdown エンジンは **必ず「`p_sho` 最大の 1 頭の複勝」を第 1 候補に入れる**設計で、
実測では平均 1.6–1.7 点/R の大半が複勝アンカーになっている。

**これは Vol. II §1.2 で確立した規律**
> ポリシー変更は `roi_verdict` が片側に外れたときだけ行う

**に照らせば、複勝は既に条件を満たしている。**

**ただし判断は単純ではない（重要）**:
1. この 390 点は **旧 shape エンジンで生成されたもの**であり、topdown の複勝選択とは
   選び方が異なる（shape は「◎の複勝」、topdown は「`p_sho` 最大馬の複勝」）
2. 複勝はトリガミ床の「床」として機能しており、外すと点数削りの基準が変わる
3. 「最も負けない線」を目標とするなら、的中率 48.2% の複勝を外すと分散が跳ね上がる

**推奨アクション（提案ではなく観測）**:
- topdown 生成分だけを切り出した複勝 ROI を別集計する
  （`stamp.engine_version >= "2026-08-09"` でフィルタ可能）
- n が貯まるまで**触らない**。前向き検証（P1-4）と同じ土俵で判断する

---

### 🟠 P1-4 — topdown が in-sample replay のみで本番化されている

**Fact**:
- 根拠のリプレイ（4/18–8/9, 506R, 82.8% vs 74.1%）は **同一期間・同一データの in-sample**
- paired CI95 = **[−0.5pt, +12.8pt]** → **下限が 0 を割っており有意ではない**
- 前向きデータは 2026-08-15/16 の **72 bets / 44R** のみ
- 判定閾値 300 bets まで **約 8 開催日（4 週末）**

**これが現在の運用上の最大リスク**である。clean-band ゲート（§3.7）と同じ構造
（in-sample の点推定を配線 → 前向きで符号反転）を再演する可能性がある。

**遵守事項**:
- 判定日まで **topdown 固定運用**（未来の結果でエンジンを選ばない = 事後選択バイアスの回避）
- 判定は `analysis/prospective_topdown_eval.py` で **1 回だけ**
- FAIL 時の切り分け候補: replay 過学習 / regime 依存 / サンプル不足 / 実装差異

---

### 🟠 P1-5 — master 側の 3 特徴が 100% NaN（逆向きの非対称）

**実測**:
| 特徴 | master notna | serve nunique | gain |
|---|---:|---:|---:|
| `kako5_avg_ninki` | **0.000** | 109 | 0 |
| `kako5_pos_vs_ninki` | **0.000** | — | 0 |
| `kako5_upset_good_count` | **0.000** | — | 0 |

学習では 100% 欠損なので木が一切使わず、本番では実値が来るが無視される。
**害はない**（gain=0 = 分岐に使われない）が、
**「過去 5 走の人気」= 馬が市場からどう見られてきたかという情報が、
パイプラインの欠陥で丸ごと捨てられている**。

`parse_kako5.py --mode master` が人気を出力していないのが原因（要確認）。

**⚠️ ただし配線前に必ず考えること**: 「過去走の人気」は市場情報である。
Vol. III §3.3 の「オッズを予測特徴に入れる」禁止則に抵触するか？
- **抵触しない可能性**: 過去走の人気は as-of で確定しており、今走のオッズではない。
  serve でも取得可能（kako5 CSV にある）
- **抵触する可能性**: 「市場が過去にこの馬をどう評価したか」は結局市場の写像であり、
  ΔAUC は上がるが ROI は上がらない（`project_odds_ou_first_gate_pass` と同型）

→ **実験するなら「Why not already priced」を先に答えること。**
`kako5_pos_vs_ninki`（着順 vs 人気の乖離）は「市場の誤りの履歴」なので、
純粋なオッズ写像より一段情報量がある可能性はある（**Hypothesis**）。

---

### 🟡 P2 群（要約）

| ID | 内容 | 具体 |
|---|---|---|
| **P2-1** | `t10_runner.py:347, 369` が削除済み `gutchi_brain` を import。`try/except` で握りつぶすため無害だが、dead code 90 行 + 毎レースのエラーログ | `brain_tickets()` / `render_brain()` / `show_race_bets(brain=)` を削除 |
| **P2-2** | v6 が採用ゲート（単勝高 EV ROI +0.05 or 複勝 +0.03）を満たさず本番化（実測 単勝 **−0.006** / 複勝 +0.002、レポート自身の結論は「❌ v6 採用見送り」） | §6.1 |
| **P2-3** | test 2024-25 が 7 回以上開封済み。v6 の 62.08% は多重比較で選ばれた値 | §6.3 |
| **P2-4** | topdown が bundle の**較正済** `pair_probs` を使わず、`pl_pair_probs()`（**未較正** λPL）を再計算。λPL と bundle 厳密値の比は 0.99±0.1 でバイアスはないが、**較正の恩恵は捨てている** | 印 5 頭以外のペアも扱うため単純置換はできない。全馬 pair に較正器を適用する形が正しい |
| **P2-5 ✅** | 見送り4条件を `production_policy.py` と `data/production_policy.json` に単一ソース化。各経路は共通関数を使用 | 2026-08-25 解消 |
| **P2-6** | `docs/compute_bets_spec.md`（2026-06-09）が実装と 9 項目乖離（Vol. II §2.9） | 本 Vol. II を正典とし、旧仕様書に DEPRECATED を明記 |
| **P2-7** | `predict_weekly.parse_csv` にテストが無い。**「実 CSV を 1 本パースして主要特徴の nunique > 1 を assert する」テストがあれば P0-1/P0-2 は即検出できた** | `tests/test_serve_parse.py` を追加 |
| **P2-8** | Optuna の CV が **valid 内レース ID のランダム 5-fold**。時系列 CV ではない | valid が 1 年なので regime 差は小さいが、厳密性は無い |
| **P2-9** | `validate_cowork_bets.ALLOWED_KINDS` に禁止券種「馬単」が残存。compute_bets は生成しないが Cowork 手動経路では通る | `REJECTED_KINDS` へ移動 |
| **P2-10** | `weekly_nicegui.ps1 -Post` の `generated_at` 凍結検知が **Warn 止まり** | Fail にする（`weekly_post.ps1` の git add ガードと同レベルにすべき） |

### ⚪ P3 群

| ID | 内容 |
|---|---|
| **P3-1** | `models/` に 66 ファイル。`unified_rank_v6_s123/s456/s789/s1234.pkl` の用途記録なし。日付付き pkl / `expert_*_rejected.pkl` は `models/archive/` へ |
| **P3-2** | `docs/cowork_prompt.md` の 1 行目に `yaru` という不要文字列 |
| **P3-3** | `predict_weekly.parse_csv(path)` に `str` を渡すと `except Exception: continue` で全エンコーディングが失敗し `UnboundLocalError: text` になる。`Path` 強制または明示エラーに |
| **P3-4** | TACT 公開線の成績が未集計 / バージョン未固定（`analysis/tact_line_eval.py` は差分測定のみ） |
| **P3-5** | `reports/serve_skew_eval.json` の `dead_numeric` が 2026-06 時点の内容で凍結（現状と不一致） |
| **P3-6** | 旧 master `data/master_20130105-20251228.csv`（412MB）は v6 が使わないので削除候補 |

---

## §6 ガバナンス問題

### 6.1 v6 の採用ゲート矛盾（🟡 P2-2）

```
採用基準（run_v6_pipeline.py:13-14 / scripts/audit_v6_vs_v5.py:278）:
  「単勝高 EV ROI が v5 比 +0.05 以上、または複勝 +0.03 以上」

実測（reports/audit_v6_vs_v5_20260520.md:45-52）:
  単勝 −0.006 / 複勝 +0.002

レポート自身の結論:
  「## 結論: ❌ v6 採用見送り … 大幅な改善なし。v5 維持。」

にもかかわらず:
  export_weekly_marks.py の default = "v6"
  CLAUDE.md は「v6 本番投入」
  git log は毎週 model=v6
```

**= 定量ゲートの出力と実運用が正面から矛盾している。**

v6 が本番に入った後付け理由は ECE 改善（複勝 −32% / 馬連 −34%）だが、
これは α の低下（1.325 → 0.031）と交絡しており **ECE ペナルティ項の純効果は不明**
（寄与は約 1% と推定されている）。実弾 ROI 上の v6 化リターンは未確認。

**推奨**:
1. 採否ゲートを ROI 単独でなく「**CLV + 高 EV 帯 ROI + 多ビン ECE**」の複合スコアに正式化
2. `run_v*_pipeline` が **exit code で pass/fail を返す**形にする
3. 「基準を満たさないのに採用」を文書で黙認しない（基準改訂 or 差し替えを明示する）

### 6.2 ECE_high_p の欠陥（🟡）

```python
ece_high = abs(p_arr.mean() - a_arr.mean())      # 単一ビンの平均差
```
過信（p > actual）と過小確信（p < actual）が**相殺**する。
修正版（**4 ビン加重 |gap|**、過信側に非対称ペナルティ）は
`lab/train/optuna_v10_marks.py` にあるが未採用。
新実験では **10 ビン ECE** を使うこと。

### 6.3 test の汚染（🟠 P2-3）

test 2024-25 は版選定で **7 回以上開封**されている（v5/v6/v7/v8/v9/v10/v11）。
したがって v6 の test 数値（◎top3 62.08%）は
**「多重比較で最良に見えた値」であり、無バイアスの汎化性能推定ではない**。

**運用ルール（v10 プロトコル）**:
- test = 2025 を封印
- 版間比較は valid のみ
- valid で勝った最終 1 版だけ、test を **1 回だけ**開封
- **test 開封台帳を `docs/version_ledger.md` の拡張として運用する**（未実施）

### 6.4 `strategy_weights.json` の構造的循環（既知・未解消）

```
採用判断: ROI_test >= 80%（test = 2024-2025）
評価:     同じ 2024-2025 データで ROI を測定
→ 30 エントリ全部が「test で良かったから採用 → test で測ると良い」の循環
```
定量的証拠: 函館 2勝 馬連 valid 224.7% vs test 136.9%（乖離 87.8pt）等。

**現状**: Streamlit のみが参照。本番ラインは無関係。
**位置づけが書面化されていない**（廃止予定なのか維持なのか不明 = ⚪ P3）。

---

## §7 検証の非対称性（「死亡」の一部は検出不能死）

### 7.1 問題

本プロジェクトの「採用」基準は厳しく（CI 下限が閾値超え）、「死亡」基準は緩い（有意差なし）。
これは**非対称**であり、次を意味する:

> **効果があっても検出できない標本しかなかった実験が、「死亡」として記録されている。**

前提監査（2026-07-30、`project_premise_audit_20260730`）の推定 **MDE ≈ 1pt**。
つまり **+0.5pt の真の改善は、本プロジェクトの標本規模では原理的に「死亡」と判定される**。

### 7.2 対処（未実施 / 提案）

1. `docs/hypothesis_registry.md` の各死亡ルートに **MDE と実際の n を明記**する
2. 死因を §3.0 の 5 型でラベル化し、
   **PRICED / ORACLE / UNCASHABLE は永久閉鎖、MIRAGE は原則閉鎖、
   UNDERPOWERED のみデータ倍増後の再検定を許可**する
3. 現在の分類（本仕様書 §3 で実施済み）を registry に反映する

### 7.3 具体的に UNDERPOWERED 疑いのあるルート

| ルート | 記録された効果 | 疑い |
|---|---|---|
| 格・クラス変動（v11） | +0.62pt CI[−0.20, +1.45] | 点推定は正だが CI が MDE と同程度。**n を増やせば有意になりうる** |
| 統計力学 3 体項 | +0.07pt | 効果量が小さすぎ、n を増やしても実用にならない可能性が高い |
| EVT win 側 | ΔAUC +0.0029 | fukusho より鋭い。detrend 残差 σ での再検定は正当な理由になりうる |
| joint_m1 umaren | ECE −33% / ROI +0.3〜0.7pt | **ECE の改善は明確**。ROI 効果は MDE 未満だが方向は正 |

---

## §8 前向き検証中の仮説

### 8.1 P1-TOPDOWN-PROSPECTIVE-2026（登録済み・PENDING）

登録日 2026-08-10（結果確認前の事前登録）。詳細は Vol. II §11.4。

| 項目 | 内容 |
|---|---|
| 仮説 | replay 改善（Δ+8.7pt）が未来データでも再現する |
| Treatment | 本番 topdown（`stamp.engine_version >= "2026-08-09"`） |
| Baseline | 同一レース・同一 T-10 オッズの shape シャドー（`reports/engine_shadow/`） |
| 評価 | レース単位 paired bootstrap 10,000 回、seed=42 |
| 判定 | n_bets ≥ 300 で 1 回。PASS: ΔROI>0 ∧ CI下限>−2pt / FAIL: ΔROI<0 ∧ CI上限<+2pt |
| 進捗 | **72 bets / 44R（2026-08-15, 16）** — 判定まで約 8 開催日 |

### 8.2 H4: G1 直後ローテの RPCI 過小評価（PENDING、n≥200 待ち）

「前走 G1 → 今走 G2 以下」で前走 RPCI < 54 の馬の複勝率は、
同条件で前走 RPCI ≥ 56 の馬より高い（市場が過小評価している）。
棄却条件: n≥200 かつ RPCI<54 群の ROI が RPCI≥56 群を +5% 以上上回らない場合。

### 8.3 監視中（未登録・登録すべき）

| 対象 | 監視すべき理由 |
|---|---|
| 構築層 4 修正（S5） | replay 74.1→78.2 は topdown と同じ **in-sample** |
| `FUKU_HIT_THR = 0.21` の serve 校正 | offline 射影（的中 80% / 回収 92%）は前向き未検証 |
| `AITE_WEAK_TH = 0.252` の serve 校正 | 発見期 0.328 → serve 分布で再校正した値。前向き未検証 |
| λ（`harville_lambda.json`） | fit 標本 349R のみ。モデル世代を跨ぐと壊れる |
| 決済ドリフト 20+ 倍帯 | n=39 で全体平均に平滑中。次回再 fit で要確認 |

---

## §9 研究アジェンダと優先順位

### Priority 0 — 欠陥の修復（最も確実にリターンがある）

| # | 内容 | 期待効果 | リスク |
|---|---|---|---|
| 0-0 | **P0-4（`R`→`Ｒ` のリネーム 1 行）** | gain 0.68% 回収 | 実質ゼロ |
| 0-1 | **P0-2（tyaku 53 列）を直す** | gain 2.01% 回収 | 極小。純粋なバグ修正 |
| 0-2 | **P0-1（騎手/調教師 stats 再 merge）を直す** | gain 10.58% 回収 → ◎top3 +0.3〜1.0pt（Hypothesis） | 中（静的スナップショットの as-of 性に注意） |
| 0-3 | **P0-3（serve 較正器のマスク同期）** | ECE 改善 | 中。0-1/0-2 の後に実施 |
| 0-4 | **P1-1（canary の絶対水準可視化）** | 再発防止 | 極小 |
| 0-5 | **P2-7（parse_csv のテスト追加）** | 再発防止 | 極小 |

**この 5 件は「新しいアイデア」を一切必要とせず、既に存在する情報を本番に届けるだけ**である。
本プロジェクトで残っている数少ない確実な仕事。

### Priority 1 — topdown の前向き検証（最重要・最安）

観測するだけ。難易度低、リーク危険なし。§8.1。

### Priority 2 — 検証済み未配線 ROI の回収

| # | 内容 | 状態 |
|---|---|---|
| 2-1 | **joint_m1 umaren の配線**（ECE −33% / ROI +0.3–0.7pt） | 本番 export は素の PL 積のまま。要: 本番土俵での CI 再検証 → export 配線 |
| 2-2 | **UMAMI 正配線**（馬連 cap10 = 84% / ワイド cap3-4 = 82.8%） | topdown 化後の帰属再検証が先 |
| 2-3 | **調教 JOIN の学習/serve 非対称解消**（学習側にも 14 日カットオフを揃える再学習） | 低コスト・低リスク |
| 2-4 | **P2-4（topdown で較正済ペア確率を使う）** | 全馬 pair への較正器適用 |

### Priority 3 — SettleAI（決済層）の続き

exotics 実市場 EV の初検定（`analysis/exotics_ev_market_test.py` は初回実行のみ）を完遂。
per-horse 予測器は optional。**ドリフト方向による銘柄選別は毒・禁止**。

### Priority 4 — UNKNOWN 実験の決着（低コスト）

- `exp_recency_sweep` / `exp_window` / `quantile_exp` / `learn_target_compare` /
  `train_v6_multiseed` の採否を registry に記録して閉じる
- **v10 のパースバグ修正（`前走走破タイム` 等の −9999 量死）だけを v6 に単独移植して ablation**
  （ONE CHANGE）— これは P0 群と同系統の「捨てている情報の回収」
- `母馬` の疑似デッド → **本仕様書で実証済み（Vol. I §4.2 C 群）。registry に記録して閉じる**

### Priority 5 — 統計ガバナンスの整備

- §7.2 の死因ラベル化（PRICED / ORACLE / UNCASHABLE / MIRAGE / UNDERPOWERED）を registry に反映
- test 開封台帳の運用
- v6 採用ゲート矛盾の解消（§6.1）

### 明示的に狙わないこと

| 対象 | 理由 |
|---|---|
| ROI 85–90% | 無理筋。床は ≈80%（控除率） |
| 予測精度の追求への回帰 | §2 の天井。62% は業界の地の値であり、弱いのはプロダクト/配信側 |
| 新しい特徴量ファミリの探索 | 1400 特徴で採用ゼロ |
| モデルアーキテクチャの変更 | 全滅実績 |
| 券種空間の探索 | 完全踏破済み（全券種で群衆超過 ≈ +7pt 一定） |

---

## §10 外部 AI（ChatGPT 等）へのレビュー依頼

### 10.1 このリポジトリで**やってほしいこと**

| # | 依頼 | 見るべき場所 |
|---|---|---|
| R1 | **§5 の欠陥台帳の検証**。特に P0-1 / P0-2 / P0-3 の私の診断が正しいか、実装を読んで反証してほしい | `predict_weekly.py:236-290, 450-560`, `export_weekly_marks.py`, `serve_history_feats.py`, `build_pl_calibrators_serve.py` |
| R2 | **P0-1 の修正案に leak が無いかの検査**。静的スナップショット `jockey_stats.csv` を serve で使うことの as-of 妥当性 | `data/jockey_stats.csv`, `predict_weekly.py:466-485` |
| R3 | **`compute_bets.py` の topdown 経路のバグ検査**。特に トリガミ床のループ、`allocate()` の収束、キャップ境界、候補が 1 点になったときの挙動 | `compute_bets.py:479-541, 260-276` |
| R4 | **fail-safe / fail-closed の漏れ**。「静かに悪い出力を出す」経路が他に無いか | 全体。特に `except Exception: pass` / `fail-open` の箇所 |
| R5 | **確率計算の数学的正しさ**。`pl_probs.py` の閉形式、`pl_pair_probs()` の λ補正、`race_confidence` の正規化 | `pl_probs.py`, `compute_bets.py:114-145`, `export_marks_json.py:118-147` |
| R6 | **統計手続きの妥当性**。`_bet_cis()` の bootstrap（投資加重 ROI の CI）、Wilson 区間、`roi_verdict` の閾値 80% | `generate_results.py:187-226` |
| R7 | **本仕様書の誤り**。実測値・行番号・因果の記述に間違いがあれば指摘してほしい | 本 3 巻 |
| R8 | **保守性・可読性の改善提案**（挙動を変えないもの）。定数の単一ソース化、dead code 除去、型ヒント | 全体 |

### 10.2 **やってほしくないこと**（読まずに書かれると害になる）

| # | 禁止提案 | 理由 |
|---|---|---|
| N1 | 「新しい特徴量を追加しては？」（血統・調教・馬場・展開・ラップ・適性・セリ価格・回り…） | §3.1。1400 特徴を本番 v6 土俵で検定して**採用ゼロ** |
| N2 | 「Transformer / GNN / ニューラルネットを使っては？」 | §3.2。汎化ゼロ |
| N3 | 「アンサンブル / スタッキングを強化しては？」 | §3.2。Brier 改善 ≤0.001、100% フォールバック |
| N4 | 「EV / 期待値の高い馬券に絞っては？」 | §3.4。**有害 −13pt**。289 セル網羅で全滅 |
| N5 | 「オッズを特徴量に入れては？」 | §3.3。AUC↑ は市場価格の写像で ROI 不変 + serve 不可 |
| N6 | 「CLV を KPI にしては？」 | §3.3。pari-mutuel で換金不能 |
| N7 | 「ROI 85–90% を目指しては？」 | §1.2 / §9。床が控除率 ≈80% |
| N8 | 「Kelly 基準で資金管理しては？」 | 検定済み。エッジが控除率未満なので Kelly は 0 を返すか破産する |
| N9 | 「三連単 / WIN5 / 枠連を試しては？」 | §3.4。全部検定済み・死亡 |
| N10 | 「モデルを再学習しては？」（根拠なし） | v7〜v11 が全部ゲート未達。**再学習は情報を増やさない** |
| N11 | 「ハイパーパラメータをもっと探索しては？」 | §9「明示的に狙わないこと」 |
| N12 | 「馬場・天候・当日バイアスを見ては？」 | §3.1。当日・クロスデイの両方で検定済み・死亡 |

**もしこれらを提案したいなら**、「以前の実験と何が違うか」を明示すること。
検出力不足（UNDERPOWERED、§7）による再検定は正当な理由になりうるが、
「思いついたから」は理由にならない。

### 10.3 レビュー時に必ず参照すべきコマンド

```bash
# テストが通ること
./venv311/Scripts/python.exe -m pytest tests/ -q          # 73 passed

# PL の恒等式が満たされること
./venv311/Scripts/python.exe pl_probs.py

# serve の欠損構造を自分で再現する（P0-1 / P0-2 の再現）
export PYTHONIOENCODING=utf-8
./venv311/Scripts/python.exe -c "
from pathlib import Path
from predict_weekly import parse_csv
df = parse_csv(Path('data/weekly/20260816.csv'))
for c in ['jockey_fuku90','trainer_fuku90','horse_fuku10','前走馬体重','前PCI']:
    print(c, 'notna=%.3f nuniq=%d' % (df[c].notna().mean(), df[c].nunique()))
"

# gain × serve coverage の三元表を再現
./venv311/Scripts/python.exe -c "
import joblib, json
b = joblib.load('models/unified_rank_v6.pkl'); f = b['feature_cols']
imp = dict(zip(f, b['model'].feature_importance('gain'))); tot = sum(imp.values())
bc = json.load(open('data/serve_feature_baseline.json', encoding='utf-8'))['baseline_cov']
dead = [(k, imp[k]/tot*100) for k in f if imp[k] > 0 and (bc.get(k) or 0) < 0.40]
print('serve-dead feats:', len(dead), 'gain lost: %.2f%%' % sum(v for _, v in dead))
"

# 実運用実績
./venv311/Scripts/python.exe -c "
import json; d = json.load(open('data/cowork_results.json', encoding='utf-8'))
print(json.dumps(d['total'], ensure_ascii=False))
print(json.dumps(d['by_type'], ensure_ascii=False, indent=1))
"
```

### 10.4 提案フォーマット（必須）

```
【提案 ID】
【分類】 欠陥修正 / 未配線 ROI 回収 / 新規実験 / 保守性
【対象】 ファイル:行番号
【現状】 コードを読んで確認した事実（推測と区別すること）
【問題】 何が壊れているか / 何を取りこぼしているか
【根拠】 実測コマンドとその出力、または既存レポートの引用
【提案】 具体的な変更内容
【Why not already priced】 ※新規実験の場合のみ、必須
【Leakage risk】 as-of / OOF になっているか。生成順序をコードで確認したか
【期待効果】 pt 単位の事前予測。MDE(≈1pt) を超えるか
【反証条件】 事前に固定した棄却条件
【死亡ルートとの衝突】 Vol. III §3 のどの行とも矛盾しないことの確認
```

---

## 付録 A — 本仕様書執筆時に実測したコマンドと結果

| 項目 | 結果 |
|---|---|
| `pytest tests/ -q` | 73 passed / 20.47s |
| master_v2 行数 | 626,774 |
| split 分布 | train 485,252 / valid 47,273 / test 94,249 |
| v6 特徴数 | 120 |
| v6 α | 0.03083978412534253 |
| v6 gain 上位 | kako5_avg_pos 13.88% / 前走確定着順 11.72% / prev_hosei 7.56% / jockey_fuku90 6.79% |
| gain=0 特徴 | 開催, 前走走破タイム, 母馬, kako5_avg_ninki, kako5_pos_vs_ninki, kako5_upset_good_count |
| serve low-coverage 特徴 | 34件（gain>0は32件）/ gain 14.88% |
| serve 較正器のマスク | 14 件（numeric 6 + cat 8） |
| serve canary 監視対象 | 86特徴 / 既知 low-coverage 34 |
| tyaku CSV 列数 | 53（パーサ期待値 55） |
| 2026-08-16 parse_csv | 478 頭 / 35R。jockey_fuku90 / trainer_fuku90 / horse_fuku10 / 前走馬体重 / 前PCI / 前走RPCI がすべて nuniq=1 |
| `Ｒ` 全角/半角 | parse_csv 出力に `'Ｒ'` は不在、`'R'` が存在 |
| jockey_stats.csv / trainer_stats.csv | 342 行 / 329 行（どちらも未使用） |
| cowork_results 累積 | 1,178R / 2,389 bets / ROI 71.9% / −¥1,067,377 |
| topdown 前向き | 72 bets / 44R（2026-08-15, 16） |
| harville λ | λ1=0.8405, λ2=0.7542（fit 349R, 2026-05-31〜07-11） |
| t10_blend λ | 1.5。test 6,858R: v6 61.67% / 市場 64.33% / blend 65.08% |
| settle drift | 単勝 462 勝者 ×0.9221 / 複勝 1,369 ×0.8852 / ワイド 1,387 ×0.920 |

## 付録 B — 用語集

| 用語 | 意味 |
|---|---|
| 印 | ◎（本命）〇（対抗）▲（単穴）△（連下）。AI スコア上位 5 頭に付与 |
| ◎top3 | ◎が実際に 3 着以内に入った率。本プロジェクトの主要精度指標 |
| 控除率 | 売上からプールが引く割合。単複 20% / 馬連系 22.5% / 三連単 27.5% |
| トリガミ | 的中したのに払戻 < 総投資 |
| 妙味 (under) | AI 確率が市場 implied 確率の 1.20 倍以上 |
| UMAMI (xROI) | 実測テーブルで補正した期待回収率。生 EV の代替 |
| serve skew | 学習時と本番時で同じ特徴が作られない現象 |
| canary | serve skew の無言死を検知する仕組み |
| topdown | 印を介さず全馬確率から直接馬券を組むエンジン（現行既定） |
| shape | 印スロット + 形テンプレートの旧エンジン（現在はシャドー） |
| 枠 | 勝負 / 準勝負 / 消化 の資金配分区分 |
| クリーン帯 | 正規化エントロピー下位 1/3 のレース群（ゲートは撤回済） |
| T-10 | 発走 10 分前。ライブオッズ取得と買い目確定のタイミング |
| priced | 現象は実在するが市場が既に織り込んでいる状態 |
| MDE | 最小検出可能効果。本プロジェクトでは ≈1pt |
| roi_verdict | ROI の 95% CI と控除率 80% の位置関係。ポリシー変更の唯一の許可条件 |

---

**← [Vol. I システム仕様](VOL1_SYSTEM.md) / [Vol. II 馬券構築・運用仕様](VOL2_BETTING_OPS.md)**


---

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
CHAOS_SKIP_PERCENTILE = 0.667  # data/production_policy.jsonが単一ソース
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
CHAOS_SKIP_PERCENTILE = 0.667; FIELD_SIZE_SKIP = 7; PWIN_SKIP = 0.05
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
    ...  # Vol. III §10.3 のスニペット参照。現状 14.88% → 目標 <10%
```

---

**← [Vol. I](VOL1_SYSTEM.md) / [Vol. II](VOL2_BETTING_OPS.md) / [Vol. III](VOL3_VALIDATION_AND_OPEN_PROBLEMS.md)**


---
