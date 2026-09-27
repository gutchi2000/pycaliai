# PyCaLiAI 完全仕様書（全 4 巻 結合版）
> 個別ファイルは docs/SPEC/ 配下。本ファイルは外部 AI への一括投入用の結合版。
> 版 1.0 / 2026-08-23（本文初版）、結合版は working tree（未コミット分含む）から 2026-09-07 に再生成

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

- §0 この巻の読み方（§0.1 頻出略語ミニ用語集）
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

### 0.1 頻出略語ミニ用語集（初出前に多用されるもの）

全巻を通じて正式な定義が本文の後方（各章の該当節）に来る用語のうち、
定義前に多用されるものだけ先出しする。詳細な定義は各章を参照。

| 略語/用語 | 意味（一言） | 詳しい定義 |
|---|---|---|
| **ECE** | Expected Calibration Error。予測確率と実際の的中頻度のズレ（0に近いほど較正が良い） | §7.3 |
| **T-10** | 発走 10 分前。当日ライブオッズを取得し買い目を確定するタイミング | Vol. II §6 |
| **gain** | LightGBM の特徴重要度指標（その特徴が分岐に使われた際の損失減少量の合計） | §4.2 |
| **PL** | Plackett-Luce。着順（1着→2着→…の逐次選択）を確率的にモデル化する手法 | §7.1 |
| **UMAMI** | 頭字語ではなく「旨味」由来の造語。実測テーブルで補正した期待回収率（xROI）の呼称 | §8.3 |
| **IPAT** | JRA公式のインターネット/電話投票システム（人間が最終的に馬券を購入する窓口） | §1.4 |
| **chalk** | 競馬俗語で「本命（人気馬）」。chalk-cap は本命側の点数上限のこと | Vol. II §1.3 |
| **0B31/0B33/0B34** | JV-Link（JRA-VAN データ配信）のオッズ種別コード | §9.4 |
| **SHAP** | SHapley Additive exPlanations。特徴が予測に寄与した度合いを分解する説明手法 | §4.2 |

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
| **控除率が固定の床** | 単勝/複勝 20%、枠連/馬連/ワイド 22.5%、馬単/三連複 25%、三連単 27.5%、WIN5 30% がプールから引かれる（JRA 設定払戻率の裏返し） | 全馬に均等に賭けると ROI は必ず控除率ぶん目減りする。**期待値中立が「上限」ではなく「床」** |
| **オッズが確定するのは発走後** | 買った時点のオッズでは払われない。締切時点のプール比で決まる | **CLV (Closing Line Value) が換金不能**。ブックメーカー戦略の主要概念が丸ごと使えない (Vol. III §3.4) |
| **自分の賭け金がオッズを動かす** | 大口はプールを希釈する | 小口個人（1R ¥10,000）では無視可能。ただし合成オッズ裁定は成立しない |

**この 3 点から導かれる本プロジェクトの目標設定（Fact, `project_roi_max_doctrine`）**:
> 目標は「儲ける」ではなく「**最も負けない線**」。ROI 85–90% は無理筋であり、
> 罠をすべて切っても残 ROI ≈ 79.9% ≈ 控除率 = 期待値中立が床。

### 1.3 券種と払戻の定義

| 券種 | 的中条件 | 控除率 | 本プロジェクトでの扱い |
|---|---|---|---|
| 単勝 | 1 着馬 | 20% | 使用（topdown で確率 1 位・オッズ ≤30 のみ） |
| 複勝 | 3 着以内（発売時点の出走予定頭数が 5〜7 頭の場合は 2 着以内） | 20% | **主力**（アンカー） |
| ワイド | 3 着以内に入る 2 頭の組 | 22.5% | **主力** |
| 馬連 | 1-2 着の組（順不同） | 22.5% | 使用（prob-first・オッズ ≤50） |
| 馬単 | 1-2 着の順序付き組 | 25% | **全廃**（実測 ROI 22.1%、Vol. II §11） |
| 三連複 | 1-3 着の組（順不同） | 25% | 未配線（shadow 実験のみ） |
| 三連単 | 1-3 着の順序 | 27.5% | **廃止** |
| 枠連 | 1-2 着の枠の組 | 22.5% | 検定済み・不採用 |
| WIN5 | 指定 5 レースの 1 着を全的中 | 30% | 検定済み・死亡 |

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
CLEAN_BAND_MAX = production_policy["chaos_reference"]["skip_percentile"]
if chaos > CLEAN_BAND_MAX:
    if not force_floor:
        return 見送り
    if demote_budget and budget > demote_budget:
        budget = demote_budget            # ← 呼び出し側が demote_budget を渡していない
```

旧 `0.33` は 2026 as-served で効果が符号反転したため撤回した。現在は hard gate と同じ production policy（`0.667`）を参照し、§0b が別の参戦母集団を作らない。

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
| **topdown** | **82.8%** |

Δ +8.7pt（vs shape旧）、**CI95 [−0.5, +12.8]**、P(改善) = 0.961。
**CI 下限が 0 を割っている = 有意ではない**。前向き検証中（§11.3 / Vol. III §8）。

⚠️ **未reconcile**: `compute_bets.py:99` のコード内コメントは同じ 4/18-8/9・506R リプレイを
「topdown 83.6%（Δ+6.1pt）」と記す。基準（vs どの版か）が本文と揃っておらず、
どちらが最終値か未確定。次回リプレイ再実行時に一本化すること。

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

**✅ 解消済み（2026-09-11 再検証、旧記載は誤り）**: `ALLOWED_KINDS` に `馬単` は残っていない。
現行 `ALLOWED_KINDS = {"単勝","複勝","ワイド","馬連","三連複"}`、`REJECTED_KINDS = {"馬単","三連単"}`
に分離済み（Vol. III P2-9）。

### 3.3 fail-safe の哲学

| 状況 | 挙動 |
|---|---|
| ライブオッズが取れない | **見送り**（推定で買わない） |
| 分位表が壊れている | **見送り**（生値フォールバックはしない） |
| overround（Σ1/odds）が [1.0, 1.5] 外 | `jvlink_odds` が `ok=false` → **見送り** |
| 相手信頼ゲートで候補が全滅 | ゲートを適用しない（単複候補が無い時は fail-safe） |
| SHAP explainer 構築失敗 | `why` なしで bundle 生成継続（fail-open） |
| `serve_history_feats` 例外 | 従来どおり欠損のまま続行（fail-open）→ canary が検知 |
| ~~`gutchi_brain` import 失敗~~ | ✅ 2026-09-11 再検証で解消済み確認（§6.5参照、stale記載） |

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

### 6.5 ✅ 解消済み（2026-09-11 再検証、旧記載「gutchi_brain の dangling import」は誤り）

**2026-09-11 実データ再検証**: `grep -n "gutchi_brain" t10_runner.py` は0件。
本節記載の `import gutchi_brain`（`:347`, `:369`）・`brain_tickets()`/`render_brain()` は
現行 `t10_runner.py` に存在しない（stale診断、いつ解消されたかは未特定だが現状は解消済み）。
以下は当時の記録として残す（実害は既に解消済み）:

`gutchi_brain.py` は 2026-08-09 に退役・削除済み（実測: ファイル不在）。旧記載の実害は
`process_race()` が `try/except Exception` で包み `ImportError` を握りつぶす無害な dead code
（毎レースのログ出力のみ）というもので、現在は当該 import 自体が存在しないため該当しない。

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

## §11 実運用実績（実測、`data/cowork_results.json` 2026-09-02 生成）

### 11.1 累積

| 指標 | 値 |
|---|---:|
| レース数 | **1,248**（うち見送り 634） |
| 決着済 | 1,248 |
| 馬券点数 | 2,438 |
| 実効投資 | **¥3,968,834** |
| 払戻 | ¥2,849,457 |
| 収支 | **−¥1,119,377** |
| **ROI** | **71.8%** |
| 的中率 | 19.8%（482/2,438） |

### 11.2 券種別（CI 付き）

| 券種 | 点数 | 的中率 [CI95] | 投資 | ROI | ROI CI95 | **verdict** |
|---|---:|---|---:|---:|---|---|
| ワイド | 914 | 18.1% [15.7, 20.7] | ¥1,317,134 | 75.7% | [59.5, 94.6] | inconclusive |
| 単勝 | 308 | 14.6% [11.1, 19.0] | ¥514,100 | **84.8%** | [53.7, 122.0] | inconclusive |
| 複勝 | 433 | 48.7% [44.1, 53.4] | ¥871,900 | 69.1% | [59.8, 78.9] | **below_takeout** |
| 馬連 | 661 | 8.9% [7.0, 11.3] | ¥1,093,200 | 71.0% | [43.6, 105.6] | inconclusive |
| **馬単** | 122 | 1.6% [0.5, 5.8] | ¥172,500 | **22.1%** | [0.0, 52.6] | **below_takeout** |

**読み方**:
- **馬単は統計的に確定した負け** → 全廃済み（過去分の負債）
- **複勝も `below_takeout`** — 的中率 48.7% は高いが配当が薄すぎる。
  これは「当たる ≠ 儲かる」の実証であり、topdown が複勝アンカーに寄せている設計への
  **反証候補**として監視すべき（Vol. III §5 P1-3）
- ワイド・単勝・馬連はいずれも CI が 80% を跨ぐ = **点推定でポリシーを動かしてはいけない**
- **馬連・馬単は投資額が前回スナップショット（08-17、n=1,178R）から1円も動いていない**
  （¥1,093,200 / ¥172,500 で完全一致）。topdown エンジン移行後の直近2週で新規に
  1点も買われていないということであり、累積表の馬連・馬単の行は**もはや現行エンジンの
  成績ではなく凍結された過去の負債**として読むべき。逆に複勝は+43レース・+¥167,800と
  直近の新規投資の大半を占めており、topdown の「複勝厚」設計がそのまま出ている。

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
| 2026-08-16 | 70 | 184,000 | 102,400 | 55.7%（topdown 初適用週、決済確定後に再集計） |
| **2026-08-30** | 70 | 173,000 | 119,100 | **68.8%** |

**週次 ROI の分散が極めて大きい**（16% 〜 101%）。
週 70R / ¥200k 規模では **1 週間のデータで何かを判断してはいけない**。
topdown 初週の 55.7% も同様（n が全く足りない）。
**2026-08-23 週は `cowork_results.json` の weekly 配列に該当行なし**（未確認の欠測。
原因未調査 — kekka 未配置か開催なしかは要確認）。

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
| E7 | ~~Transformer / Set Transformer は汎化ゼロ（配置情報は予測層で exploitable でない）~~ **[ERRATUM 2026-09-23]** 旧記述の根拠 `exp_havoc_m1.py` は M1 2体結合特徴を GBM に入れた波乱読み出しで、Transformer ではない（根拠の誤帰属）。正: 旧 RaceTransformer（`train_transformer.py` / `transformer_pl_v2`、同一レース全頭の self-attention・位置符号なし）は**実施済み**で、単独 ◎top3 54.36%（v6 62.03%、test 2024-25）と低かった。ただし旧特徴 48 本・旧 master・複勝 AUC 選択等の交絡がある。同一入力・同容量 no-context 対照による race context 固有効果は**未検証**（EXP15 で検定）。DeepSets・Sinkhorn 系・マルチタスクは未実装 | ~~`exp_havoc_m1.py`~~ `reports/evaluate_transformer_v6_stack.json`, `analysis/mcond/exp15_race_as_set_dev/` |
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

### 3.0.1 遡及訂正ログ（バグで評価が反転した実験）

**背景（2026-09-02）**: 外部 AI（ChatGPT）が本仕様書を読んで「PL→Gamma/Thurstone 分布置換」を
提案した際、当時の本書に残っていた `joint_m1 umaren`（§3.2・§7.3・§9 Priority 2）の記載が
「ECE −33% / ROI +0.3〜0.7pt、方向は正」という**2026-06-27 時点で既に単位バグと判明し撤回済みの
古い数値のまま**だった。正しい照合（100% 基準）では黒字セル 0/36・CI 下限 >100% も 0/36 で
ROI 回収ゼロが確定していたにもかかわらず、本書がそれを反映していなかったため、外部 AI に
「近い先行実験は好意的な結果だった」という誤った印象を与えかねない状態だった。

**教訓**: 死因 5 型（PRICED/ORACLE/UNCASHABLE/MIRAGE/UNDERPOWERED）だけでは、
「一度は生存判定を受けたが後日のバグ修正・再検証で評価が反転した」実験を表現できない。
この種の実験は MIRAGE に分類されるべきだが、**分類の変更点と根拠（訂正日・訂正理由・出典）を
明示しないと、更新前の楽観的な数値だけが独り歩きし、外部レビュアーが再発見のつもりで
死亡ルートを再提案してしまう**（今回がその実例）。

**恒久ルール**: 検証結果を撤回・反転させる場合は、該当箇所を打ち消し線で残しつつ
「◯年◯月◯日に◯◯（バグ／再検証）により訂正、旧値は△△」を併記する。単に数値を書き換えて
古い記述を消さない。過去の楽観値が仕様書のどこにも残っていない状態を避け、
「なぜかつて有望に見えたのか」まで含めて記録する。

**適用例**: `joint_m1 umaren` の項目（§3.2、§7.3、§9 Priority 2-1）を本改訂で修正済み。

**適用例 2（2026-09-23, ERRATUM）**: E7 と §3.2 の「Transformer / Set Transformer は汎化ゼロ」を打ち消し線付きで訂正。これはバグによる反転ではなく**根拠の取り違え**（M1 結合 GBM の数値を Transformer の死因として引用）による過大主張の訂正。出典: EXP15 Stage 0 監査 `analysis/mcond/exp15_race_as_set_dev/RACE_AS_SET_PRIOR_ART_AUDIT.md` §5。

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
| 統計力学（2 体 / 3 体 / joint / 順序 / 自由 E） | MIRAGE | 2 体 M1 は umaren ECE −33% で実在（校正指標としては本物）。だが **ROI +0.3〜0.7pt は 2026-06-27 に単位バグと判明**（0-100 スケール ROI を素の 1.0 と比較していただけ）。100% 基準で正しく照合すると**黒字セル 0/36・CI 下限>100% も 0/36**＝実馬券 OOS で ROI 回収ゼロ確定。3 体 β3 は符号は物理的に正しいが M1 上乗せ +0.07pt は同様に無価値 | `lab/physics_gates/gate2_3body.py`, `analysis/_joint_m1_wedge.py`（2026-06-27 訂正） |
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
| Transformer / Set Transformer | ~~汎化ゼロ。gain 4.8% 使うのに ΔAUC −0.004~~ **[ERRATUM 2026-09-23]** 旧記述の数値は M1 結合 GBM（`exp_havoc_m1.py`）のもので Transformer の結果ではない。旧 RaceTransformer は実施済みで単独成績は低い（◎top3 54.36% vs v6 62.03%）が、旧特徴 48 本・旧 master・複勝 AUC 選択の交絡があり**未決着**。同一入力・同容量 no-context 対照による context 固有効果は未検証（EXP15）。DeepSets・Sinkhorn 系・マルチタスクは未実装 |
| Race-as-a-Set DeepSets（EXP15, 2026-09-23） | **未検出（UNDERPOWERED 寄り）**: 同一入力111列・同容量の no-context 双子対照、2016-21 学習 / 2022 選択 / 2023 fixed-model。ΔLL −0.0073（CI 上限 −0.00004、5/5 seed）だが ΔBrier −0.0009 は CI が 0 をまたぎ、事前登録 Context Gate FAIL → R2 へ進まず終了。「現在の特徴、事前固定したモデル容量、2023 development、および本実験の検出力では、事前基準を超えるrace-context増分を検出できなかった」。NN 基準自体が R0-clean に ΔLL +0.068 劣後。`analysis/mcond/exp15_race_as_set_dev/REPORT.md` |
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
| 枠連 | — | 初検定（単一期間・CI/n 未併記, Hypothesis tier）ROI 84.8% = 壁内。他券種と合わせ**券種空間の踏破は一巡**（全券種で群衆超過 ≈ +7pt 一定）。枠連単独の再検定（別期間・CI）は未実施 |
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
| S2 | **λ補正 PL（Lo–Bacon-Shone）+ topdown エンジン** | replay 82.8%（in-sample）⚠️CI95[−0.5,+12.8]で下限が0未満＝**非有意**（§8.1） | Hypothesis tier。2026-08-09 既定運用だが統計的には未証明、前向き監視中 |
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
**⚠️ 実測日のスナップショットである**: P0-1/P0-2/P0-4 は 2026-08-23 時点で🔴と記載されていたが
2026-09-10 の実データ再検証で解消済みと判明した実例がある。逆に P0-5 は当初「修正済み」と
記載されたが 2026-09-11 の再検証でコード未コミット・データ未反映と判明した（後述）。
**P1-1〜P1-5・P2-6〜P2-8・P3-1・P3-3・P3-4 は 2026-08-23 時点の実測のまま「現状未確認」であり、
「そのまま信頼できる」という意味ではない**。参照・引用する際は必ず現行コード・現行データで
再確認すること。

### 5.1 サマリ

| ID | 深刻度 | 課題 | 影響（実測） | 修正コスト |
|---|---|---|---|---|
| **P0-1** | ✅ | 騎手/調教師ローリング複勝率が全馬同値の定数 | gain **10.58%** が判別力ゼロ（診断当時）。**2026-09-10 実データ再検証で解消確認**（§5.1a） | 小 |
| **P0-2** | ✅ | 着度数 CSV パーサが 55 列を期待、実データは 53 列 | gain **2.01%** + 当日馬体重が全滅。2026 全期間（診断当時）。**2026-09-10 実データ再検証で解消確認**（§5.1a） | 極小 |
| **P0-3** | 🟡 | serve 較正器のマスクがbunseki配線後の実態(19特徴)と乖離(現行34特徴でfit) | 候補較正器・実配信スコアのシャドー評価基盤を作成(§5.1d〜j)。pilot 419R(実配信スコア)では全体的に候補cal優位方向だが、bunseki有無での効果は支持されず複勝はbunseki週で悪化(n=70)。評価計画を確定・ロック済み(現行trial_003): MDE=複勝Brier0.0005、固定N=600R、適格判定は追記専用の予測台帳(発走日時>ロック時刻・記録が発走前・記録後内容変更なし)ベースで正常系往復テスト6項目PASS。**週次自動フックを`weekly_nicegui.ps1`に実装済み(§5.1j)**: Phase A直後にscore、Phase C直後にsettleをfail-openで呼ぶ(本番の終了コード/出力に無関係、タイムアウト/失敗は専用ログのみ、前段失敗時はシャドーに到達しない構造)。成功/失敗/タイムアウト/前段失敗スキップの4項目を隔離環境でPASS確認済み。**現状の正確な言い方は「フック実装・隔離検証済み、本番動作は未確認」**——次回の実運用Phase A/Cで(1)台帳への発走前記録・適格件数増加 (2)結果到着分のみ決済・未到着分保持 (3)シャドー成否と本番終了状態の記録分離、の3点を確認する。judgeは自動実行しない(手動のみ)。**本番pkl未置換** | 小（自動化完了、初回実運用確認待ち） |
| **P0-4** | ✅ | `Ｒ`（全角）を parse_csv が半角 `R` で作っている | gain **0.68%** が −9999（診断当時）。**2026-09-10 実データ再検証で解消確認**（§5.1a） | 極小 |
| **P0-5** | ✅ | 学習データ生成の `trainer_fuku30/90` が同一レース内でリーク（行単位 shift がレース境界を無視） | 626,774 行中 21,539 行（3.4%）に混入。C1（最小修正、window意味=行レベルcountingを保持しつつリークのみ除去）をtrain/serve双方で統一し、**2026-09-11 本番切替完了**（コミット`b2e2bdd4b6`、`master_v2*.csv`・`unified_rank_v6.pkl`・較正器・curve・baselineの計10ファイルを配置、sha256照合済み）。切替前に隔離環境で品質ゲート・買い目生成まで検証（serve canary無言死0、`compute_bets.py --dry`成功）。経緯の詳細（原因調査・列順バグ訂正・パイプライン段順の原因特定・環境不一致事故と修正・学習重み欠落バグ修正）は§5.1k〜x参照。C1採用は精度/収支の改善が実証されたためではなく、リークのある学習データを正すこと自体が目的（収支改善は未実証）。trial_003（旧モデル対象の較正器検証）は切替成功確認後に理由付きで退役（`trial_003_retired.json`）、後続trialは未設計 | 解消済み（§5.1x） |
| **P1-1** | 🟠 | serve canary が「ずっと死んでいる特徴」を構造的に見逃す | 上記 3 件がすべて無検知 | 小 |
| **P1-2** | 🟠 | 前走詳細 15 列が訓練中央値の定数刷り込み | gain **≈7.5%** | 中 |
| **P1-3** | 🟠 | 複勝が `below_takeout` 確定なのに topdown は複勝アンカーに寄せている | 複勝 ROI 69.0% CI[58.7, 79.4] | 要判断 |
| **P1-4** | 🟠 | topdown が in-sample replay のみで本番化されている | 前向き 72 bets / 必要 300 | 観測のみ |
| **P1-5** | 🟠 | master 側で 3 特徴が 100% NaN（serve では 90% 埋まる逆非対称） | gain 0 だが情報を捨てている | 中 |
| **P1-6** | 🟡 | `jockey_fuku30/90`/`trainer_fuku30/90` の as-of 締切定義が学習(日付+時刻、同日先行含む)と本番(日付のみ、同日全除外)で不一致（§5.1b③・§5.1c、2026-09-10定量化） | オフライン再スコアリングで Brier/logloss とも95%CIが0を跨ぎ有意差なし。◎変更率3.35%、hon_top3はserve側+0.232pt[-0.029,+0.484]で有意差の閾値付近。選択肢(a)は原理的に不可能（§5.1b） | 低〜中（次回定例再学習時にserve定義へ統一を検討、緊急性なし） |
| **P1-7** | 🟠 | `馬主(最新/仮想)` が **future overwrite**（2026-09-23 独立監査）。TARGET export 時点の所有者を過去レース行へ遡って適用。master 内で全 67,973 頭が生涯一定（対照の調教師は 11.79% 変化）、export 時点違い（2026-03 vs 2026-06）で同一過去行が 501行/68頭 不一致、TARGET 自身が `馬主(レース時)` を別列で持つ（手元 export は空で直接照合不可） | v6 は gain **1.27%（120列中15位・788分岐）** で学習に使用。ただし serve 被覆 0% で本番予測には届かず、影響は offline 学習/評価と train-serve skew。切り離し ablation（2023、5seed）では NDCG@3 +0.00103 CI[-0.0007,+0.0028]・logloss +0.00063 で**有意差なし** | 中（production 即時変更はしない。time-safe 列は再 export が必要）。詳細 `analysis/mcond/owner_id_time_safety_audit/REPORT.md` |
| **P2-12** | 🟡 | `前走レースID(新)` は time-safe（未来 ID 0 件）だが、①先頭8桁が前走日付と100%一致する日付 proxy、②前走 ID 一致で「前走が同じレースだった馬」を結べる（87.6% のレースに該当）、③serve 被覆 0%（−9999）、④18桁版は float64 精度を超え馬番情報が消えて 16 桁版の重複列（異なり値 54,958 < 55,024） | gain 0.36%/0.05%。vNext clean contract では除外済み | 小（生値を vNext に入れるかは別途判断） |
| **P2-1** | ✅ | ~~`t10_runner.py` が削除済み `gutchi_brain` を import~~ | 2026-09-11 再検証で解消済み確認（stale診断） | 解消済み |
| **P2-2** | 🟡 | v6 が自ら定めた採用ゲートを満たさずに本番化 | ガバナンス矛盾 | 要判断 |
| **P2-3** | 🟡 | test 2024-25 が 7 回以上開封済み | すべての test 数値が多重比較で選ばれた値 | プロトコル |
| **P2-4** | 🟡 | topdown が bundle の較正済 `pair_probs` を使わず未較正 λPL を再計算（2026-09-11 `compute_bets.py:483-484` で再確認、既定エンジン） | 較正の恩恵を捨てている。**訂正**: λPLとbundle厳密値の比が0.99±0.1に収まることは数値近似の一致を示すのみで、実結果に対する較正・買い目への影響（ECE/ROI等）は未検証。「バイアスなし」と結論づけるのは誤りで現状未確認として扱う | 小〜中（要再評価） |
| **P2-5** | 🟡 | 見送り 4 条件が 4 ファイルに独立ハードコード | 同期漏れリスク | 小 |
| **P2-6** | 🟡 | `docs/compute_bets_spec.md` が実装から大きく乖離 | 誤読リスク | 小 |
| **P2-7** | 🟡 | `predict_weekly.parse_csv` にテストが無い | P0-1/P0-2 を検出できなかった | 小 |
| **P2-8** | 🟡 | 較正器の Optuna CV が valid 内ランダム KFold（時系列でない） | 楽観バイアスの可能性 | 中 |
| **P2-9** | ✅ | ~~`validate_cowork_bets.ALLOWED_KINDS` に禁止券種「馬単」が残存~~ | 2026-09-11 再検証で解消済み確認（`REJECTED_KINDS`に分離済み、stale診断） | 解消済み |
| **P2-10** | 🟡 | `cowork_results.json` 凍結検知が Warn 止まり | 集計凍結の再発余地 | 極小 |
| **P2-11** | 🟡 | 除外・中止馬を rolling統計で「複勝を外した(0)」として算入（P0-5とは独立、§5.1n） | 意味論的に不正確、精度への影響は未検証 | 中（要検証） |
| **P3-1** | ⚪ | `models/` 66 ファイル / seed 変種の用途記録なし | 保守性 | 小 |
| **P3-2** | ⚪ | `docs/cowork_prompt.md` の 1 行目に `yaru` が混入 | — | 極小 |
| **P3-3** | ⚪ | `parse_csv` が str を渡されると `UnboundLocalError` | エラーメッセージが不明瞭 | 極小 |
| **P3-4** | ⚪ | TACT 公開線の成績が未集計 / バージョン未固定 | 検証不能 | 小 |

### 5.1a P0-1/P0-2/P0-4 再検証（2026-09-10、実装照合）

本節の「2026-08-23 実測」診断日より前に、以下が **既にコミット済み** だったことを
git log で確認した：

- `b848778b8b`（2026-07-29）「serve skew残差回収: hist/course/jockey履歴特徴+騎手・
  調教師コードをserveで再計算」— `serve_history_feats.py` の `fill_history_features()`
  が `_HistoryIndex.rolling_rate()` で騎手/調教師コード別に as-of（`date < race_date`、
  レース単位で1カウント集約、P0-5と同じレース境界整合）の複勝率を計算しており、
  全馬定数にはならない実装になっている。
- `predict_weekly.py:239-246` の `TYAKU_HORSE_SCHEMAS = {52: …, 53: …, 55: …}` は
  52/53/55 列の3スキーマを実ヘッダー幅で判別する実装で、「55列決め打ち」ではない。
- `export_weekly_marks.py:315` の `_SERVE_RENAME = {"R": "Ｒ", …}` が半角→全角の
  リネームを行っている。

**実行して確認**（`python export_weekly_marks.py --csv data/weekly/20260906.csv`）:
```
[serve history] 解決 483/484 頭 (新馬 1 / 曖昧 0) 騎手コード 484 / 調教師コード 466
  — cov: course_n_prev=100% jockey_n_prev=100% hist_cond=83%
着度数CSV読み込み済: horse_fuku mean=0.306, std=0.126   ← 定数(0.286)ではなく馬ごとに分散
[serve canary absolute] dead gain=6.64% (18特徴, gate=35.00%)
```
`jockey_n_prev` カバレッジ100%、`horse_fuku` に分散あり、canary の無言死 gain 合計が
6.64%（18特徴）と、P0-1 単独の claim（gain 10.58%）を下回っている。これは
「P0-1/P0-2/P0-4 は 2026-08-23 時点で既に修正コードが本番経路に乗っていた」ことと
整合する。**2026-08-23 の診断がどの経路（v6 本番 or 旧 predict_weekly 経路、または
古い baseline キャッシュ）を見ていたかは特定できていない**が、現行コード・現行データ
では 3 件とも再現しない。

⚠️ 上記は **今日 1 回・1 週（20260906）の実行結果**であり、`data/serve_feature_baseline.json`
自体（週次生成・cache）はまだ再生成していない。§5.2 の gain 合計・P0-3 の「34特徴/
gain14.88%」もこの baseline を参照した値なので、**baseline 再生成 → P0-3 の較正器
再 fit の要否判断、の順で次に進めること**。

### 5.1b P0-1 の直接分散証跡 + P0-3 診断（2026-09-10、本番ファイル未変更）

**方針**: 較正器の再 fit（本番モデル隣接の変更）は行わず、①別ファイルへの baseline 再生成、
②現行較正器マスクとの照合、③学習/本番の履歴締切差の定量化、の3点を診断のみ実施した。
使用スクリプトはスクラッチ領域（`diag_serve_baseline.py` / `diag_sameday_cutoff.py`）、
出力は `data/serve_feature_baseline.json` ではなく別ファイルに保存し、**本番の
baseline・calibrator・モデルは一切書き換えていない**。

**① P0-1 の直接分散証跡**（`jockey_n_prev=100%` という coverage だけでなく、4特徴そのものの分散を確認）:

直近4週（20260829/20260830/20260905/20260906）で `jockey_fuku30/90`, `trainer_fuku30/90` を実測：

| 特徴 | week | n | nunique | std | mean |
|---|---|---:|---:|---:|---:|
| jockey_fuku30 | 20260906 | 484 | 19 | 0.145 | 0.233 |
| jockey_fuku90 | 20260906 | 484 | 36 | 0.131 | 0.237 |
| trainer_fuku30 | 20260906 | 466 | 17 | 0.103 | 0.226 |
| trainer_fuku90 | 20260906 | 466 | 35 | 0.082 | 0.224 |

4週とも同様の分散（nunique 16〜40、std 0.08〜0.15）を確認。旧診断が指す「訓練 valid
中央値へのフォールバック定数」（`jockey_fuku30=0.200` 等、`predict_weekly.py:489-499`）は
発生しておらず、実際に馬ごとに異なる値が入っている。**P0-1 は coverage だけでなく分散でも解消確認**。

**② P0-3 診断**: 別ファイルへ baseline 再生成し、現行 `models/pl_calibrators_v6_serve.pkl`
のマスク（実測: numeric 21 + cat 13 = 34件。VOL3 旧記載の「6+8=14件、ハードコード凍結」
は誤り — `serve_skew_eval.py` に `SERVE_DEAD_NOW_EXACT` 等の定数は現存せず、
`load_current_serve_mask()` が `data/serve_feature_baseline.json` を fit 時点で読んで
動的にマスクを作る**単一ソース設計に既になっている**。P0-3 の「修正案1」は実装済み）
と突合：

- 新 baseline（直近4週）の dead（coverage<40% かつ gain>0）は **18特徴 / gain合計6.64%**
  （現行較正器 fit 時の baseline は 34特徴/14.88%）。
- **新baselineでdeadだが現行マスクに含まれない特徴: 0件**（マスクが緩すぎて漏れている
  リスクは無い）。
- **現行マスクにあるが新baselineでは40%以上生きている特徴: 16件**
  （`前走馬体重`0%→50%, `前走出走頭数`0%→50%, `前走場所`0%→41%, `コース区分`0%→56%,
  `course_win_rate`/`course_top3_rate`0%→48%, `騎手年齢`/`調教師年齢`0%→50%,
  `母馬`/`馬主(最新/仮想)`/`生産者`/`毛色`0%→47〜50%, `馬齢斤量差`0%→50%,
  `トラックコード(JV)`0%→50%, `前走馬体重増減`0%→50%, `前走走破タイム`は0%のまま）。

  **原因（訂正、2026-09-10）**: 当初「前走情報がある馬(2走目以降)/ない馬(初出走)」の
  二分と推測したが誤り。実際は**開催日単位の全有無**だった。`data/bunseki/{date}.csv`
  （TARGET「出走馬分析」、`parse_bunseki.py` により 2026-09-05 配線）の有無で
  日ごとに coverage が 0% ⇄ 93〜100% に切り替わる（測定に使った4日＝2週末のうち、
  20260905/20260906（同一週末の土日）だけ bunseki ファイルが存在し coverage 93-100%、
  20260829/20260830（別の週末）は不存在で coverage 0%。median を取ると見かけ上
  47〜56%になっていた）。個別実行で確認済み：
  `前走馬体重`は bunseki 有る日=100%/無い日=0%、`騎手年齢`も同様。
  例外は `コース区分`(53〜62%、bunseki非依存)と`course_win_rate`/`course_top3_rate`
  (44〜50%、bunseki非依存、こちらは serve_history_feats.py の馬自身の「同コース×同距離帯
  での前走経験有無」による本物の per-horse 差の可能性があるが未確認)。
  **bunseki は今後の週も継続して提供される前提なら、この13特徴は今後 coverage
  93-100%で安定するはず**（要継続監視）。

**結論（診断のみ、再fitは実施していない）**: 現行較正器は「16特徴を常時-9999/未知値」
という前提で fit されているが、bunseki 配線後の週では実際に90%超の馬に実値が入っている。
これは「即座に較正が崩れている」ではなく「較正器が本番の実力より悲観的な分布を仮定している」
方向のズレで、bunseki が今後も継続提供されるなら**このズレは一時的でなく恒常化する**
（2週に1度のコイントスではなく、今後は毎週この13特徴が生き続ける）。dead gain の
絶対量（6.64%）は小さいが、**この13〜16特徴は較正器が最も悲観視している集合そのもの**
なので、再fitで ECE がどちらに動くか（悪化側にも改善側にも振れうる）は測定するまで
分からない。**再fitの要否は ECE 実測後に判断すること**（本節はその判断材料の提供まで
で、再fit実施の可否についてはユーザー判断待ち）。

**③ 学習/本番の履歴締切差を定量化**（読み取り専用診断、`master_v2` 実データで検証）:

`build_dataset.add_rolling_stats()` は主体×レース単位に集約後 `(日付, 発走時刻)` 順で
`shift(1)` するため、**同日の先行レースは窓に含まれる**。一方 `serve_history_feats.py`
の `_HistoryIndex.rolling_rate()` は `np.searchsorted(dates, race_date, side="left")`
と日付のみで切るため、**同日のレースは（先行・後続とも）全て除外される**。

実測（`data/master_v2_20130105-20251228.csv` 全期間、主体×レース単位）:

| 特徴 | 同日に先行レースがある行の割合 | train/serve定義で値が実際に変わる行 | 変化量（|差|、変化した行のみ） |
|---|---:|---:|---:|
| jockey_fuku30 | 76.42%（479,011/626,774） | 41.35%（258,120/624,239） | mean 0.045 / median 0.033 / p90 0.067 / max 0.346 |
| trainer_fuku30 | 60.07%（363,568/605,235） | 27.59%（166,383/603,000） | mean 0.036 / median 0.033 / p90 0.067 / max 0.231 |

**解釈**: 「同日に先行レースがある」行は多数（騎手76%・調教師60%、1日に何鞍も乗るため）
だが、30走窓の平均値が実際に変わるのは一部（騎手41%・調教師28%）。変化量は
median 0.033 = ほぼ「30走窓に1走分が出入りする」際の典型的シフト量で、この特徴自身の
週内 std（0.08〜0.15、①参照）の **20〜40%程度**に相当し、無視できる大きさではない。

**これは未解決・未定量化だった実際の train/serve 定義差である**（外部レビューで指摘された
懸念の直接検証）。P0-1 が「定数化」という重度の欠陥でないことは①で確認
できたが、この「日付 vs 日付+時刻」の定義差は残存する**軽度〜中度の別問題**として
P1-6 の ID を新規に切って記録した（§5.1 サマリ参照）。

**選択肢(a)「serveを学習側(同日先行含む)に合わせる」は原理的に不可能**: 週次一括生成
（`weekly_nicegui.ps1` Phase A、土曜朝＝その週末のレースが1つも行われる前）は、生成時点で
その週末のどのレースもまだ結果が出ていない。したがって「同日の先行レース結果」は
そもそも生成時点で存在しない — これは実装の穴ではなく時間的に不可能な要求。
選択肢は実質 (b)学習側をserveの厳格な定義(同日除外)に合わせて再学習、
(c)実害をオフライン再スコアリングで測ってから優先度を決める、の2つに絞られる。
（T-10当日ライブ経路は発走10分前に再計算するため同日の先行結果は理論上取得可能だが、
`t10_runner.py` が同じ `serve_history_feats.py` を使っているなら同じ制約を持つ。未確認。）

### 5.1c P1-6 オフライン再スコアリング（2026-09-10、較正器は再fitせず固定）

**方針**: 較正器 (`models/pl_calibrators_v6_serve.pkl`) を一切変更せず、同一レース・
同一モデル (`unified_rank_v6.pkl`) を固定し、`jockey_fuku30/90`/`trainer_fuku30/90`
の4特徴だけを train定義/serve定義で差し替えて再スコアリングした（他116特徴は
`master_v2` の値のまま）。**両定義とも P0-5 修正後の「主体×レース単位」集計から
自前で再計算**（`master_v2` に残る旧 `jockey_fuku30` 列などは使っていない）。
対象は `master_v2` の test+valid（10,327〜10,365R、141,522頭）。**この期間は
診断専用として扱い、この結果を根拠に test を「開いた」とはカウントしない
（P2-3のtest多重使用ガバナンス問題とは別枠）**。較正器の再fit自体の効果検証は別途必要。

| 指標 | train定義 | serve定義 | 差(serve−train) | 95%CI（race単位 paired bootstrap, B=2000） |
|---|---:|---:|---:|---:|
| 生スコア差（全馬、|diff|>0の行） | — | — | mean +0.00117 / std 0.04224 / 60.52%の行で非ゼロ | — |
| ◎（トップ1）変更率 | — | — | 3.35%（346/10,327R） | — |
| hon_1st_pct（◎が1着） | 29.912% | 30.038% | +0.126pt | — |
| hon_top3_pct（◎が複勝圏） | 61.296% | 61.528% | +0.232pt | [-0.029, +0.484]pt |
| Brier（単勝, ◎） | 0.20161 | 0.20220 | +0.00059 | [-0.00069, +0.00188] |
| Brier（複勝, ◎） | 0.22230 | 0.22202 | -0.00028 | [-0.00091, +0.00036] |
| logloss（単勝, ◎） | 0.59974 | 0.60101 | +0.00127 | [-0.00166, +0.00431] |
| logloss（複勝, ◎） | 0.64290 | 0.64215 | -0.00075 | [-0.00212, +0.00053] |
| 固定ビンECE（単勝, ◎） | 0.01916 | 0.01978 | +0.00062 | （bootstrap未実施） |
| 固定ビンECE（複勝, ◎） | 0.01945 | 0.02101 | +0.00156 | （bootstrap未実施） |

**解釈**:
- 生スコアは60.52%の行で変わるが、平均差はほぼゼロ（+0.00117、std比で無視できる水準）
  — 系統的な一方向バイアスではなく、行ごとに大小両方向へのブレ。
- ◎の変更は3.35%（346R）で発生 — 無視できないが多数派ではない。
- **hon_1st_pct/hon_top3_pctはserve定義の方が高い**（+0.126pt/+0.232pt）。95%CIは
  hon_top3で下限-0.029ptとほぼゼロに接するが、方向は一貫して「同日先行を除外する
  serve側の定義の方が的中率で劣らない、むしろ僅かに良い」。これは選択肢(a)が不可能
  という結論をさらに補強する（train定義に揃えても的中率は改善しない可能性が高い）。
- Brier/loglossは単勝・複勝とも**95%CIが0を跨ぐ**（統計的に有意な差ではない）。
- 固定ビンECEはserve定義の方がやや悪化（単勝+0.00062、複勝+0.00156）だが絶対値は
  小さく（ベース0.019〜0.022に対し3〜8%の相対悪化）、bootstrapは未実施（必要なら追加可）。

**結論**: P1-6の定義差は**実在するが、少なくとも現在の評価では小さく、有意でない
（Brier/loglossのCIはいずれも0を含む）**。的中率はむしろserve定義側が僅かに良い
方向で、選択肢(a)（serveをtrain定義に合わせる）は不可能かつ的中率上のメリットも
無さそうという二重の理由で却下できる。残る選択肢は (b) 次回の定例再学習時に
学習側をserve定義（同日除外）に揃える、または (c) 現状維持して監視を続ける、のいずれか
——**今回の結果は優先度を上げる根拠にはならない**（有意差なし）が、次回再学習の
ついでに(b)を実施するコストは低い。**再学習を伴う判断のため、実施はユーザー判断待ち**。

### 5.1d P0-3 候補較正器の作成とオフライン比較（2026-09-10、本番pkl未置換）

**固定した条件（ユーザー指摘への対応）**:
- **対象環境**: baselineは bunseki 配線後（2026-09-05〜）のデータのみ（20260905/20260906＝
  **同一週末の土日2開催日、週として2つあるわけではない**）で再生成。配線前週
  （20260829/20260830）とは混ぜていない。ただし **同一週末1本分のみ**であり、
  入力状態（bunseki項目の取得安定性等）の週をまたいだ再現性を確認できる期間としては
  短すぎる。信頼区間も取れず、bunseki が今後も継続して同じ形式で提供される前提での
  暫定値であることに注意（§5.1b訂正と同じ限界。次の週末以降のデータで要再確認）。
- **入力の再現**: 2023-2025の過去レースに「今のCSV」を流用してはいない。過去レースは
  `master_v2` のオフライン(フル)特徴のまま、候補マスク（bunseki配線後baselineでcoverage<40%の
  numeric11+cat8=19特徴）を `-9999`/`__NaN__` に潰して再スコアリングする、既存
  `build_pl_calibrators_serve.py`/`serve_skew_eval.py` と同じ「特徴削除による模擬劣化」手法を
  そのまま踏襲した（bunseki自体を過去に遡って復元することは不可能なため、これが現実的な近似）。
- **比較条件**: 同一モデル(`unified_rank_v6.pkl`)・同一評価対象(test=2024-2025, 6,909R)。
  fit(候補較正器 = valid=2023の候補マスクスコアで新規fit、`sklearn.isotonic.IsotonicRegression`、
  `build_pl_calibrators_serve.py`と同一ロジック)と評価(test=2024-2025)は時系列分離。
  **現行較正器・候補較正器のどちらも「同じ候補マスクスコア(test期間)」に適用**し、
  マッピングだけを差し替えて比較（＝bunseki配線後の本番相当スコアに対し、どちらの
  calibratorが実際の的中と合うかを見る）。
- **評価**: 主評価は**全対象馬**（◎に限らずレース内の全馬）のBrier・logloss・固定ビンECE。
  ◎のみの指標は補助。差の95%CIはレース単位クラスタブートストラップ(B=1500、
  レースをresampleしてそのレースの全馬をまとめて採用)。
- 候補較正器は `models/pl_calibrators_v6_serve.pkl` を置換せず別ファイルに保存した
  （診断用、本番は現行のまま）。

**結果（主評価: 全対象馬 n=94,249頭 / 6,909R、test=2024-2025）**:

| 指標 | 現行cal | 候補cal | 差(候補−現行) | 95%CI（race単位クラスタブートストラップ） |
|---|---:|---:|---:|---:|
| Brier（単勝, 全馬） | 0.06083 | 0.06078 | -0.0000523 | **[-0.0001046, -0.0000011]（0を含まず、上限は事実上0に接触）** |
| Brier（複勝, 全馬） | 0.13831 | 0.13826 | -0.00005 | [-0.00014, +0.00004]（0を含む） |
| logloss（単勝, 全馬） | 0.22009 | 0.21955 | -0.00054 | **[-0.00107, -0.00012]（0を含まず）** |
| logloss（複勝, 全馬） | 0.43077 | 0.43029 | -0.00048 | **[-0.00089, -0.00011]（0を含まず）** |
| 固定ビンECE（単勝, 全馬） | 0.00282 | 0.00123 | -0.00158 | （bootstrap未実施） |
| 固定ビンECE（複勝, 全馬） | 0.00555 | 0.00427 | -0.00128 | （bootstrap未実施） |

補助評価（◎のみ、n=6,909R）でも全指標が同方向（候補calの方が良い）で、絶対値の改善幅は
◎に絞るとやや大きい（ECE複勝: 0.02368→0.01749、-0.00619）。

**解釈（訂正、2026-09-10 ユーザー指摘反映）**:
- 4指標中3指標（Brier単勝・logloss単勝・logloss複勝）で95%CIが0を含まないが、
  **これは探索的な結果として扱う**。理由: (1) test=2024-2025は既存の複数の検証
  （PRED-03A等）で繰り返し参照済みの期間で、本結果もその「開封」の一つに数えるべき
  （P2-3のtest多重使用ガバナンス問題）。(2) 4指標を同時に見ており多重比較の補正をして
  いない。(3) Brier単勝のCI上限は -0.0000011（実質ゼロに接触）で、四捨五入すれば
  「0を含む」とも言える境界例。**「有意」を確定的な証拠として扱わず、次の一次データ
  （§5.1e）で再現するかを見るための仮説として扱う**。
- 絶対的な改善幅は小さい（Brier差約-0.00005、logloss差約-0.0005）。
- **19特徴の一律マスクは本番入力の再現ではない**点を明記する。実際の本番劣化は
  (a) マスク対象外の特徴にも部分欠損があり得る、(b) P1-6（§5.1c）で確認した
  jockey/trainer_fuku30/90 の学習/本番締切差が数値化されずに残っている、
  (c) この比較で使った `master_v2` の非マスク特徴（jockey/trainer_fuku30/90含む）
  自体がP0-5修正前の値を引きずっている可能性を排除できていない（今回はP1-6の実験と
  異なりこれらの列を再計算していない）。**したがって本節の結論は「一律マスクによる
  近似条件下では」に限定する**。実際の本番入力での効果は §5.1e（実配信入力の
  シャドー評価）で別途確認する。
- 候補較正器の fit 結果は raw≈cal≈emp（isotonic回帰がほぼ恒等写像）だが、これは
  fit期間(valid=2023)の候補マスクスコアに対する当てはまりの良さであり、
  **評価期間(test)での較正の質を保証するものではない**（恒等写像=もともと較正不要、
  という意味であり、それ自体が「良い較正」の証明にはならない）。

**この段階でやったこと/やっていないこと**:
- やった: 候補較正器の作成・別ファイル保存・現行との時系列分離オフライン比較（一律マスク近似）。
- やっていない: 本番 `models/pl_calibrators_v6_serve.pkl` の置換。`umaren`（馬連）較正器の
  同様の比較（tansho/fukushoのみ実施）。bunsekiデータが同一週末1本のみという小標本の限界の解消
  （今後の週次データで候補baselineを再生成し、傾向が安定するか確認する余地あり）。
  信頼度曲線（ビンごとの予測確率 vs 実測率）の提示。
- **本番置換の根拠は §5.1e の実配信入力シャドー評価で揃える**（この一律マスク近似結果は
  仮説形成の材料であり、置換判断の根拠には使わない）。置換する場合も
  `backup_model()` で退避してから行う（既存 `build_pl_calibrators_serve.py` の慣行）。

### 5.1e P0-3 実配信スコアによるシャドー評価基盤（2026-09-10、評価計画は未確定）

§5.1dの一律19特徴マスク近似の限界を受けて構築した仕組み。**当初版は「現行コードで
過去CSVを再パースするリプレイ」だったが、ユーザー指摘により訂正**: リプレイは
現行コード・現行 `_horse_history.parquet` を使うため、当時実際に配信されたスコアと
一致する保証がない。正しくは `reports/cowork_input/{date}_bundle.json` の
`horses[].ai_score`（本番が実際にその週使った生スコア、`export_marks_json.py:290`
で埋め込み済み）を正として使うべきで、**この12週は全て bundle.json が既に存在した
ため、リプレイは1件も使っていない**（`source=served_bundle` 100%。フォールバックの
`csv_replay` はbundle.jsonが無い週のみ使い、出力に明示される）。

**構成（`analysis/calibrator_shadow_eval.py`、本番pkl・本番出力は無変更）**:
- `score`: bundle.json優先で生スコア取得 → `reports/calibrator_shadow_scores_raw.parquet`
  に保存。週ごとにソース内容のハッシュを保持し、**ハッシュが変わった週だけ再スコアリング**
  する（同一内容の週は毎回スキップ、動作確認済み）。
- `settle`: `data/kekka/{date}.csv` と突合し `reports/calibrator_shadow_scores_settled.parquet`
  に保存。**scoreとsettleを分離**しているため、結果未着のレースがあってもscoreは進められ、
  kekka到着後に`settle`だけ再実行すれば後日決済できる（何度再実行してもよい）。
- `report`: 決済済みデータを集計（`--bunseki-only` で絞り込み可）。
- `power`: 下記の検出力分析。

**初回実行結果（2026-09-10、bundle.json実在の直近12週、実配信スコア100%）**:

| 対象 | n頭 | Rレース数 |
|---|---:|---:|
| 全期間 | 5,570 | 419 |
| bunseki有り週 | 931 | 70（20260905/20260906＝同一週末） |
| bunseki無し週 | 4,639 | 349（20260718〜20260830） |

| 指標（全期間） | 現行cal | 候補cal | 差 |
|---|---:|---:|---:|
| Brier（単勝, 全馬） | 0.066392 | 0.066209 | -0.000183 |
| Brier（複勝, 全馬） | 0.154354 | 0.154184 | -0.000170 |
| logloss（単勝, 全馬） | 0.246535 | 0.245823 | -0.000711 |
| logloss（複勝, 全馬） | 0.475364 | 0.474720 | -0.000645 |

（旧版のリプレイベース数値とはやや異なる — 例: 複勝Brier全期間は旧0.150886→新0.154354。
差は数%オーダーで、**リプレイと実配信スコアが完全には一致しないことの直接証拠**でもある）

**bunseki有無での内訳**: bunseki有り週（n=70R）は単勝が候補cal優位（Brier -0.000399）
だが**複勝は悪化**（Brier +0.000387、logloss +0.000771）。bunseki無し週（n=349R）は
全指標で候補cal優位。**「候補較正器がbunseki配線後の週に特化して効く」という仮説は
この実データでも支持されない**（旧リプレイ版と同じ結論だが、今回は実配信スコアに
基づく分、より信頼できる）。

**この12週=419Rは確認用サンプル（pilot）として分離し、以後の判定には使わない**
（ユーザー指示）。以後の判定用サンプルは、評価計画を固定した時点より後に収集する
新規レースに限定する。

**評価計画（未確定、次節の検出力分析を踏まえてユーザーと合意する）**:
- 評価対象: 計画固定後に配信される、bunseki有り・現行配信条件のレース
  （bunsekiは2026-09-05以降の通常配信条件になっているため、今後の週はほぼ全て該当する
  想定。該当しない週が出た場合は評価対象から除外し、その旨を記録する）。
- 主指標: 全対象馬の**複勝Brier**（現行の複勝中心の運用に対応）。
- 副指標: 単勝Brier、単勝・複勝logloss、固定ビンECE。
- 判定時期: 下記§の検出力分析に基づき、検出したい改善幅(MDE)に応じた必要レース数を
  **事前に**固定し、そのNに達した時点で一度だけ判定する（毎週CIを見て「0を割ったら
  採用」という逐次判定はしない — 多重検定による偽陽性膨張を避けるため）。
- 蓄積方法: 本番出力を変更しない独立した週次ジョブ（`score`→`settle`）として運用。
  毎週は欠測・入力品質（bundle.json/kekka の有無、source=csv_replayが混入していないか）
  だけ確認し、有意性の逐次チェックはしない。

### 5.1f 検出力分析（2026-09-10、ユーザー指摘で重み付け不整合を修正 + ロック機構追加）

**訂正（旧版の問題点）**: 旧版は「レース単位で先に平均してから、その単純平均のSD」を
使っていたが、`report`の主指標（全対象馬プールの複勝Brier、= 馬数で自然加重した
プール平均）とは重みが異なっていた。レース間で出走頭数が違う（今回は34〜35頭/R）ため、
**単純なレース単位平均のSDでは主指標の実際のサンプリング変動を過小/過大評価しうる**。

**修正方法**: `report`と全く同じ統計量（`_pooled_brier_diff`: 全馬をプールしてから
候補/現行それぞれの平均二乗誤差の差を取る）を、レース単位クラスタbootstrap
（レースをresample→そのレースの全馬をまとめて採用、B=2000）で直接推定し、
pilotサイズ N0 でのSEを得た。そこから解析的スケーリング
`SE(N) = SE(N0)·√(N0/N)` → `N = N0·(z_total·SE(N0)/MDE)²` で必要Nを算出する
（z_total = z_α/2 + z_β）。この方法なら**report の主指標と完全に同じ重み付け**になる。

**① 全pilot（419R、bunseki有70R+無349R混在）**:
```
点推定 = -0.000170   race単位クラスタbootstrap SE(N=419) = 0.000197
```

| MDE | 必要N（検出力80%） | 必要N（検出力90%） |
|---:|---:|---:|
| 0.0001 | 12,755 | 17,075 |
| 0.0002 | 3,189 | 4,269 |
| 0.0003 | 1,418 | 1,898 |
| 0.0005 | 511 | 683 |
| 0.0007 | 261 | 349 |
| 0.0010 | 128 | 171 |

**検算**: N(MDE=0.0001)/N(MDE=0.0010) = 99.648（理論値 (0.0010/0.0001)² = 100.000）
— 一致（同一SE0/N0を使う限りNはMDE⁻²に厳密比例するはずで、実際そうなっている。
旧版で「0.0005→627なのに0.0002は3,799」と見えたのも実際には同じ関係を満たしており
（627×(0.0005/0.0002)² ≈ 3,919 ≈ 旧表の3,914〜3,799、MDEの丸め差の範囲内）、
**旧版に計算矛盾があったわけではない**。今回の主目的は矛盾の解消ではなく重み付けの
訂正であり、結果として新しいSE0（0.000197、旧の0.000196とほぼ同じ）が得られた
ため、全pilotでのNの値自体は旧版と大差ない（511 vs 旧627、ズレは主に重み付け訂正分）。

**② bunseki有り週のみ（70R、今後の評価対象に近いがnが小さい）**:
```
点推定 = +0.000387（bunseki週は複勝が悪化方向、§5.1eの通り）
race単位クラスタbootstrap SE(N=70) = 0.000486
```

| MDE | 必要N（検出力80%） | 必要N（検出力90%） |
|---:|---:|---:|
| 0.0001 | 12,978 | 17,374 |
| 0.0003 | 1,442 | 1,931 |
| 0.0005 | 520 | 695 |
| 0.0010 | 130 | 174 |

**③ 対象集団のズレ（ユーザー指摘）への対応**: 419Rの82%（349R）はbunseki無し週で、
今後の評価対象（bunseki有り条件）の分散とは異なる可能性がある。①と②を比較すると、
**必要Nの値はほぼ一致**（MDE=0.0005で511 vs 520、2%差）——これはbunseki有無で
1レースあたりの分散構造が大きく変わっていないことを示唆し、①のSE推定を②の代わりに
使うことへの一定の裏付けにはなる。ただし②はn=70からの推定でSE自体の標本誤差が大きく
（bunseki週が増えるまでSEの再推定値は変わりうる）、**この必要N表はいずれも暫定推定**
として扱う（ユーザー指示通り）。

**運用上の措置（ユーザー指示に対応、`analysis.calibrator_shadow_eval lock` 新設）**:
MDE未決定でも配信スコアの収集（`score`/`settle`）は継続してよい。ただし
「確認試験（confirmatory）」として数える範囲を後から曖昧にしないため、評価計画を
固定する際に **候補較正器・現行較正器・モデル・処理コード自身のハッシュ**を
`reports/calibrator_shadow_trial_config.json` に固定するコマンドを追加した：
```
python -m analysis.calibrator_shadow_eval lock --trial-start-date YYYYMMDD --mde 0.0005
```
ロック後は `report --confirmatory-only` で `trial_start_date` 以降の行だけを
確認試験サンプルとして集計できる。`report`/`power` は実行のたびに現在のモデル/
較正器/コードのハッシュをロック時と比較する。**2026-09-10 さらに訂正**: 当初「変化
していれば警告を出す」実装だったが、ユーザー指摘で**警告だけでは不十分**（気づかずに
無効なデータを混ぜて集計し続けるリスクがある）と判明し、**ハッシュ不一致なら集計・
判定そのものを停止する(fail-closed)** に変更。また「re-lockで既存trialを上書きする」
設計だと過去の試験設定が消えてしまうため、**`lock`は常に新しいtrial_id
（`trial_001`, `trial_002`, …）を発番し、既存trialには一切書き込まない**方式に変更した
（設定を変えたい時は新しいtrial_idを作る）。詳細は次節§5.1gで確定した実際の
trial_001を参照。

**未解決**: 上記SEはいずれもpilotからの一点推定であり、SE自体の標本誤差（特に②）は
考慮していない。目標Nが決まった後、確認試験データが増えるにつれてSEの再推定値が
変わる可能性がある。

### 5.1g 評価計画の確定（2026-09-10確定 → 2026-09-11 運用上の穴を3点修正）

ユーザーが以下を確定し、`analysis.calibrator_shadow_eval lock` で
`reports/calibrator_shadow_trials/trial_001.json` にロック済み:

| 項目 | 確定値 |
|---|---|
| MDE | 複勝Brier 0.0005（§5.1fの必要N=511〜520に対し余裕を持たせた実務値。**検出力80%を保証する値ではなく、0.0005自体がROI上の実用性を証明された値でもない**） |
| 固定標本数 | 新規600R（pilotの419Rとは別。trial_start_date=2026-09-10以降のみ） |
| 対象条件 | `bunseki=True` かつ `source=served_bundle`（csv_replayフォールバック行は確認試験サンプルから除外） |
| 主指標 | 全対象馬をプールした複勝Brier、差は「候補−現行」 |
| 判定 | 600R到達後に**一度だけ**レース単位クラスタbootstrap 95%CIを算出。**上限<0なら主指標の改善を支持**（採用ではない） |
| 本番置換 | 主指標の支持だけでは自動採用しない。副指標（単勝Brier/単複logloss/固定ビンECE）・**未評価のumaren較正器**・買い目（bets）への影響を別途確認してから判断する |

**運用上の安全装置**:
1. **fail-closed**: `report --confirmatory-only` / `judge` は、ロック時に記録した
   モデル・現行較正器・候補較正器・処理コード自身のハッシュと現在の状態を照合し、
   **一致しなければ集計・判定を一切行わず停止する**（警告に留めない）。
2. **trial単位の不変性**: `lock` は常に新しい `trial_id` を発番し、既存trialのファイルを
   書き換えない。設定（MDE/対象条件/コード）を変えたい場合は新しいtrial_idを作る。
3. **一度きりの判定＋正式結果は上書き禁止（2026-09-11修正）**: `judge` は600R到達後に
   最初に到達した600件（時系列順）だけを固定して判定し、`trial_001_result.json`
   に保存する。**このファイルは一度書かれたら二度と上書きしない**。訂正が必要な場合は
   `judge --trial-id trial_001 --correction-reason "..."` を使う——元の結果はそのまま残り、
   `trial_001_result_correction_N.json` に訂正理由付きの別記録として保存される
   （どちらを採るかは人間が判断する）。
4. 週次の `score`/`settle` 実行はMDE/trial確定と無関係に継続してよい
   （データ収集そのものはtrialに依存しない）。

**2026-09-11 追加修正（ユーザー指摘2点、1点目は§5.1hでさらに訂正）**:
- **開始日時とロック時刻の関係（1回目の対応、不十分だった）**: `trial_start_date`
  （日付文字列、YYYYMMDD）だけで対象を絞ると、ロック当日の・ロックより前に生成された
  データを誤って確認試験に含める恐れがあった。当初は `bundle.json` 自体のファイル
  生成時刻（`bundle_mtime`）> ロック時刻、という条件を追加したが、**mtimeは
  ファイル最終更新時刻に過ぎず、ロック後に過去のbundle.jsonを再生成・編集すると
  簡単に条件を満たしてしまう**とユーザーに指摘され不十分と判明。§5.1hで
  追記専用の予測台帳に設計変更した。
- **pilot 419R / 探索用データの分離凍結**: 動作確認のため `score`（週指定なし）を
  実行したところ、意図せず過去72週分（2,321R、うちcsv_replayフォールバック
  13,837行=43%）が `calibrator_shadow_scores_raw/settled.parquet` に混入した。
  ユーザー指示により: (a) `freeze-pilot` コマンドを新設し、最初に報告した12週/419Rを
  `calibrator_shadow_pilot_419_frozen.parquet` に凍結・データハッシュ記録
  （`report --dataset pilot419` で常にこのスナップショットを集計、settled parquetが
  今後どれだけ拡張されても当時の報告を再現できる。**実行して数値が完全一致することを
  確認済み**）。(b) `report --dataset exploratory` で pilot419 の12週を除いた探索用
  データを `source=served_bundle`/`source=csv_replay` に分けて別集計する
  （混ぜない。本番置換の判断根拠にはしない、と明記）。(c) trial_001の標本数・判定条件は
  この拡張データを見て変更していない。

**探索用データの参考値（`report --dataset exploratory`、本番判断には使わない）**:
複勝Brier差(候補−現行): served_bundle週 +0.000219、csv_replay週 +0.000182
——**いずれもpilot(419R)の-0.000170とは符号が逆**（候補の方が悪い方向）。
これは「候補較正器が一貫して優位」という単純な仮説への追加の反証材料であり、
確認試験（trial_001）の必要性をさらに補強する。

### 5.1h 確認試験の適格判定を追記専用台帳ベースに再設計（2026-09-11、mtime方式の欠陥修正）

**ユーザー指摘**: 「mtimeはファイルの最終更新時刻。過去のbundleをロック後に
再生成・編集すると `bundle_mtime > ロック時刻` を満たしてしまう」——その通りで、
§5.1gのmtimeゲートは**改ざん・再生成に対して無防備**だった。また「旧trialを退役
させた場合は削除でなく別IDに進むべき」との指摘も受けた（実際には2度、削除して
同じ`trial_001`を再ロックしてしまっていた——設定を変えるたびに新IDを発番する
という自分自身の設計方針に違反していた）。

**修正した設計**: `reports/calibrator_shadow_ledger.parquet`（追記専用の予測台帳）を
新設した。`score` 実行のたびに、新規の (week, rid, umaban) 行だけを台帳に追加する。
**台帳の既存行（_score・bundle_hash・first_recorded_at・race_post_datetime）は
理由の如何を問わず二度と書き換えない**。唯一許される更新は、後日 `score` を
再実行した際に同じ (week, rid, umaban) の `source_hash`（bundle.jsonの内容ハッシュ）
が記録時と食い違っていた場合に `content_changed_after_record` フラグを
False→Trueにすることだけ（＝記録後に中身が変わったという事実の記録。実際に
擬似的な改ざんを注入して検知・スコア値が変化しないことを確認済み）。

**確認試験の適格条件（3条件すべてを満たす行のみ）**:
1. **`race_post_datetime`（発走日時）> `trial.locked_at`**
   — レース自体がロックより後に実施されたこと。週次CSVのレース見出し行
   （日付＋発走時刻）から軽量に抽出する（`_extract_race_post_times()`、
   `parse_csv`の重いJOINは使わない）。過去レースのbundleを後から編集しても、
   レースの発走日時という不変の事実は変えられないため、これが本質的な防御線になる。
2. **`first_recorded_at`（台帳への初回記録時刻）< `race_post_datetime`**
   — 予測が発走前に保存されたこと（結果を見てからの後出しでないことの証明）。
3. **`content_changed_after_record == False`**
   — 台帳記録後にbundle.jsonの中身が変わっていないこと。

いずれか欠ける（発走日時不明・台帳未記録・記録後変更あり）行は自動的に除外される。
`_confirmatory_eligible(trial)` に一本化し、`report --confirmatory-only` と
`judge` の両方がこれを使う（bundle_mtimeベースの旧ロジックは削除）。

**trialの再発番**: 旧 `trial_001`（mtime方式でロック）は削除せず
`trial_001_retired.json` に退役理由を記録して残し、新しい `trial_002` を
台帳ベースの現行スクリプトでロックした（設定内容はtrial_001と同一）。
**以後、設定やコードを変える場合は必ず新しいtrial_idを発番し、既存trial番号を
再利用・削除しない**（削除して同IDで再ロックする、という今回の誤りは繰り返さない）。

**動作確認**: `score`で台帳に931行（2週分）を記録 → 対象は全て2026-09-06以前
（ロック時刻2026-09-11より前）のため `judge --trial-id trial_002` は適格0件・
「目標未達」を正しく返した。台帳の1行のsource_hashを意図的に書き換えて再度
`score`相当の処理を通したところ、`content_changed_after_record` がTrueになり、
かつ元の `_score` 値（-0.4184）が一切変化しないことを確認した（テスト後に
フラグは元に戻した）。

**2026-09-11時点の状態**: `trial_002` が現行の確認試験。確認試験サンプルは0R
（対象週がまだ生成されていない）。旧 `trial_001` は退役済み。

### 5.1i 用語の中立化・監査ログ・週次ジョブの実在確認（2026-09-11）

**用語の中立化（ユーザー指摘）**: `content_changed_after_record` を「改ざん」と呼んで
いたが不正確——bundle.jsonの内容変更は正当なデータ訂正でも起こりうる。以後
「**記録後の内容変更**」という中立的な呼称に統一した（コード変数名は元々
`content_changed_after_record`のまま、ドキュメント上の説明文言を修正）。
**この判定は結果を見て後から解除しない**——変更後の内容がどちらに転んでも、
一度記録した予測を判定に使う。

**除外件数・理由の永続監査ログ**: `_confirmatory_eligible()` を、各適格条件を
段階的に絞り込みながら件数を記録する形に書き直し、`report --confirmatory-only`/
`judge` を実行するたびに `reports/calibrator_shadow_eligibility_audit.jsonl` へ
1行追記するようにした（台帳総数 → 発走日時記録あり → 発走>ロック時刻 →
記録が発走前 → 記録後未変更 → source一致 → bunseki一致 → kekka決済済み、
の各段階の件数）。標準出力だけでなくファイルに残るため、後から
「どの条件でどれだけ除外されたか」を追跡できる。

**正常系の往復テスト（ユーザー指摘、本番から完全隔離した一時ディレクトリで実施）**:
「ロック後発走・発走前記録・記録後の内容変更なし」のレースについて、以下を確認した
（`BASE`/`LEDGER_PARQUET` をテスト用一時ディレクトリに差し替え、本番ファイルには
一切触れていない）:
1. 未来のレース（発走時刻がロックより後）を`score`相当の処理で台帳に記録 → 3頭とも
   台帳に入る。
2. kekka未到着の時点で確認試験の適格性を見ると、条件は満たすが決済0件（正しく
   「まだ判定できない」状態になる）。
3. kekka到着後に再度確認すると3頭とも適格かつ決済済みになる。
4. 同じ内容で`score`相当の処理を再実行しても台帳は重複せず、`first_recorded_at`も
   変化しない。
5. 確認試験の集計を再実行しても結果は変わらない（冪等）。
6. 記録後に内容が変わったケース（source_hash変化）を模擬すると、正しく除外され、
   かつ記録済みのスコア自体は変化しないことを確認した。
**6項目すべてPASS**。

**週次ジョブが実際に登録されているかの確認（ユーザー指摘、重要な発見）**:
`weekly_nicegui.ps1` / `weekly_pre.ps1` / `weekly_post.ps1` / `t10.ps1` を検索したが、
**`calibrator_shadow_eval` への参照は一切無い**。つまり現時点では
**このスクリプトを自動実行する週次ジョブは存在せず、ここまでの実行はすべて
このセッション内の手動実行のみ**である。これは重大な運用上の欠落で、このまま
では確認試験（trial_003）のサンプルは永遠に0件のままになる
（`score`が週末レース終了後にしか走らなければ、全レースの`first_recorded_at`が
`race_post_datetime`より後になり、適格条件②を満たせない）。

**必要な自動化（提案、まだ実施していない・ユーザー確認が必要）**: Phase A
（土曜朝、bundle生成直後・レース開始前）の直後に `score` を、Phase C（日曜夜、
kekka配置後）の直後に `settle` を実行する必要がある。最も確実な組み込み先は
`weekly_nicegui.ps1` の Phase A/C 完了直後（fail-open・本番の終了コードや出力に
影響しない形）だが、**本番の週次運用スクリプトへの変更になるため、ユーザーの
明示的な確認を得てから着手する**（このメモを書いている時点では未実装）。
代替案として独立したタスクスケジューラ登録も考えられるが、Phase Aの実行自体が
（T-10/T-20と異なり）固定時刻ではなくユーザーの手動トリガーであるため、
確実性では `weekly_nicegui.ps1` への追記の方が高い。

**trialの再発番（今回も発生）**: 上記のコード修正（用語中立化＋監査ログ追加）で
script hashが変わったため、`trial_002` を退役させ（`trial_002_retired.json`）、
`trial_003` を現行スクリプトでロックした（設定内容は同一）。judge実行で適格0件・
監査ログへの記録も正常動作を確認済み。

### 5.1j 週次自動フックの実装（2026-09-11、`weekly_nicegui.ps1` に追加）

§5.1iで発覚した「自動実行する週次ジョブが存在しない」問題に対応した。
`weekly_nicegui.ps1` に `Invoke-ShadowStep` 関数を追加し、以下の2箇所から
fail-open で呼ぶ:

- **Phase A（bundle生成直後）**: `export_weekly_marks.py` が成功し
  `reports\cowork_input\${Date}_bundle.json` の存在を確認した**直後**に
  `Invoke-ShadowStep -PyArgs @("score","--weeks",$Date)` を呼ぶ。
  `--weeks` で対象日を明示し、全履歴自動列挙はしない。
- **Phase C（結果確認直後）**: `weekly_post.ps1` 成功 + `cowork_results.json`
  の `generated_at` が当日であることを確認した**直後**に
  `Invoke-ShadowStep -PyArgs @("settle")` を呼ぶ。`settle` は全週を毎回
  再チェックする設計のため `--weeks` 指定は不要——結果未着分は自然に
  未決済のまま残り、次回実行時に再試行される。

**本番からの分離（fail-open）**: `Invoke-ShadowStep` は
`reports\calibrator_shadow_pipeline.log` に成功/失敗/タイムアウトを記録するだけで、
例外・タイムアウト・非ゼロ終了のいずれであっても呼び出し元の `$LASTEXITCODE` を
復元し、本番フローの後続処理（git push・HF同期の分岐判定）に一切影響させない。
タイムアウトは `Start-Job`/`Wait-Job -Timeout` で実現し、超過時は `Stop-Job` で
確実に打ち切る。

**「前段が失敗すれば古いbundle/kekkaを代用しない」保証の根拠**: `weekly_nicegui.ps1`
の `Fail()` 関数は `exit 1` でスクリプト全体を即終了する。シャドー呼び出しは
いずれも本番側の成功確認（`Test-Path $bundlePath` 等）が通った**後**の行に
置いているため、前段が失敗すれば `Fail()` がスクリプトごと終了させ、
シャドー呼び出しの行そのものに実行が到達しない。追加のガード条件は不要
（構造的に保証される）。

**検証（本番ファイルには一切触れず、隔離環境で実施）**:
1. **成功系**: `judge`（即座に完了するコマンド）を呼び、ログに`OK`が記録され
   `$LASTEXITCODE`が呼び出し前の値のまま変化しないことを確認。
2. **失敗系**: 存在しないtrial IDを渡して意図的にPythonの例外を起こし、
   ログに`FAILED`とスタックトレースが記録され、`$LASTEXITCODE`は不変で
   あることを確認。
3. **タイムアウト系**: 完了に数十秒以上かかる処理を2秒のタイムアウトで呼び、
   `TIMEOUT`がログに記録され、`$LASTEXITCODE`は不変であることを確認。
4. **前段失敗時のスキップ**: `Fail()`が呼ばれる分岐（bundle不在を模した
   ダミー条件）を子プロセスで再現し、終了コード1でプロセスが終わること、
   かつシャドー呼び出しが実行された場合にのみ作られるマーカーファイルが
   **作られていないこと**（＝到達していないこと）を確認。
**4項目すべてPASS**。

**judgeは自動実行しない**: 週次フックは`score`と`settle`のみを呼ぶ。
`judge`（600R到達後の一度きりの判定）は今後も手動実行に限定し、確認試験の
判定タイミングを人間の意思決定から切り離さない。

**構文チェック**: `weekly_nicegui.ps1` 全体を実行せずにパース専用チェック
（`[System.Management.Automation.Language.Parser]::ParseFile`）で構文エラーが
無いことを確認済み（本番の実行系統に影響する変更のため、実際にPhase A/C全体を
走らせるテストはgit push/HF同期を伴うため行っていない——構文チェック＋
関数単体の隔離テストで代替した）。

**現状の正確な言い方（2026-09-11時点）**: 「フック実装・隔離検証済み、
**本番動作は未確認**」。上記の4項目PASSはすべて `weekly_nicegui.ps1` の
実行系統から切り離した隔離環境でのテストであり、実際の週次運用
（Phase A/C の本番実行）でこのフックが動くところはまだ一度も見ていない。
追加の仕様変更は現時点では不要——次にやるべきことは実装ではなく、
初回の実運用での確認のみ。

**初回実運用で確認すべき3点（次回のPhase A/C実行時）**:
1. **Phase A後**: 対象日の予測が発走前に台帳へ入り、`trial_003`の適格件数が
   増えること。`judge --trial-id trial_003`（進捗表示のみ、判定はしない）
   または `report --confirmatory-only --trial-id trial_003` で確認する。
2. **Phase C後**: 結果到着分が決済され、未到着分は保持されること
   （`reports/calibrator_shadow_scores_settled.parquet` に新しい週が
   追加され、kekka未着の週は除外されたまま残っているか）。
3. **記録の分離**: シャドー処理の成否（`reports\calibrator_shadow_pipeline.log`）
   と、本番処理自体の終了状態（`weekly_nicegui.ps1` 自身のコンソール出力・
   終了コード）が、それぞれ独立して記録されており、シャドーの失敗が本番の
   成功表示に影響していないこと。

この3点が確認できるまでは、`trial_003`の条件（MDE・target_n・適格条件）は
固定したまま変更しない。`judge`は今後も手動実行のみ。較正器の本番置換は
引き続き保留。

### 5.2 回収可能な gain の合計

| ID | 対象 | gain | 修正コスト |
|---|---|---:|---|
| P0-4 | `Ｒ` の全角/半角リネーム | 0.68% | 1 行 |
| P0-2 | 着度数 CSV 53 列対応（`horse_fuku10/30`） | 2.01% | 小 |
| P0-1 | 騎手/調教師 stats の再 merge | 10.58% | 小〜中 |
| P1-2 | 前走詳細ブロックの as-of 再計算 | ≈7.5%（一部は回収不能の可能性） | 中 |
| — | **合計** | **≈20.8%**（2026-06旧監査の28.15%を基準にした当時の回収可能量。P0-3記載の14.88%はP0-1/2/4診断時点のbaseline値で、§5.1aの通りP0-1/2/4は既に解消済みのため**この合計自体が要再計算**。2026-09-10のcanaryは6.64%/18特徴） | |

**残り ≈7.3%** は cat 特徴（生産者 / 馬主 / 毛色 / 前走場所 / 指定条件 / 限定 / 芝(内・外) /
性別限定 / ブリンカー / 父タイプ名）で、週次 CSV に元データが無いため回収は難しい。

⚠️ **繰り返すが gain% ≠ 精度**。20.8% の gain 回収が ◎top3 を 20% 上げることは絶対にない。
実測された offline→serve の残差ギャップは **約 1pt** であり、期待値はその範囲内である。
**それでも「捨てている情報を届ける」ことは、新しいアイデアを必要としない唯一の確実な仕事**である。

---

### ✅ P0-1 — 騎手/調教師ローリング複勝率が定数（gain 10.58%、診断当時）— 2026-09-10 解消確認（§5.1a）

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

### ✅ P0-2 — 着度数 CSV パーサの列数ドリフト（gain 2.01% + 当日馬体重、診断当時）— 2026-09-10 解消確認（§5.1a）

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
**実際に serve で死んでいる特徴**（`data/serve_feature_baseline.json` 実測、P0-1/2/4 修正前の値）: **34特徴（coverage < 0.40。gain > 0は32件）/ gain 14.88%**。

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
4. ✅ **P0-1 / P0-2 / P0-4 は §5.1a の通り既に修正コードが本番経路に乗っていることを
   2026-09-10 に確認済み**。上記「34特徴/gain14.88%」はその前の baseline 値なので、
   まず baseline を再生成してから較正器を作り直すこと（P0-1/2/4 修正待ちではない）。

**推奨順序**: `measure_serve_coverage で baseline 再生成`（P0-1/2/4 反映後の実態を測る）→
`P0-3 で較正器再 fit`（新 baseline の coverage < 0.40 を機械的にマスク対象化）→
`ECE と ◎top3 を CI 付きで測定` → 記録。

---

### ✅ P0-4 — `Ｒ`（レース番号）の全角/半角ミスマッチ（gain 0.68%、診断当時）— 2026-09-10 解消確認（§5.1a）

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

### 🔴 P0-5 — 学習データ生成時の同一レース内リーク（`trainer_fuku30/90`、2026-09-07 発見、2026-09-11 コード修正コミット済み・データ未反映）

**発見経緯**: 外部レビュー（ChatGPT/Codex, ASTRA-01）の指摘を実データで再現・検証して確定。

**バグ**（`build_dataset.py:add_rolling_stats`、修正前）:
```python
master = master.sort_values(["日付", "発走時刻", "レースID(新/馬番無)", "馬番"]).reset_index(drop=True)
master[col] = (
    master.groupby(code_col, sort=False)["fukusho_flag"]
    .transform(lambda x: x.shift(1).rolling(window, min_periods=5).mean())
)
```
`shift(1)` は「同じ主体（騎手/調教師）の**直前の行**」を参照するが、ソート順は
日付・発走時刻・レースID・**馬番**なので、同一レースに同じ調教師の馬が複数頭
出走すると、後の馬番の行が**同一レースの先の馬番の行**を「直前の過去走」として拾う。
**レース結果が確定する前に、同じレースの別馬の結果が特徴量に混入する。**

**実測**（`master_v2_20130105-20251228.csv`, 626,774 行）:
| 項目 | 値 |
|---|---:|
| 同一レース×同一調教師の複数頭グループ | 20,961 組 |
| リーク対象行（グループ先頭以外） | 21,539 行（3.4%） |
| うち特徴量が非 NaN（実際に学習で使われる） | 21,494 行 |
| うち混入した同レース結果が「複勝圏内」（flag=1） | 3,745 行 |
| 騎手側の同型リーク | **0 件**（1 レースに同一騎手は 1 頭のみのため構造的に無傷） |

具体例（`レースID=2013011408010501`, 調教師コード=1095）: 馬番8（fukusho_flag=1、複勝）と
馬番10（同レース、同調教師）で、修正前は馬番10の `trainer_fuku30` に馬番8の**当該レースの
結果**が混入し 0.250 になっていた（正しくは両馬とも同じ「レース前」値 0.142857 であるべき）。

**影響**: `trainer_fuku90` は gain 重要度 120 特徴中 **7 位（1.92%）**、`trainer_fuku30` は
43 位（0.55%）。影響行は全体の 3.4% で、1 件あたりの変動幅も `min_periods=5〜window=30`
により有界（最大 ±1/5 程度）なので、◎top3 ≈ 62% 天井のような大きな結論を覆すとは
考えにくいが、**train/valid/test すべてこの master から作られているため、これまでの
Fact 表記の一部（特に trainer_fuku 関連の gain・重要度）は軽微に汚染されている**。

**serve（週次予測）は無傷**: `serve_history_feats.py::rolling_rate` は行 shift ではなく
`np.searchsorted(dates, race_date, side="left")` による日付カットオフ実装で、当該レース
当日のデータを丸ごと除外するため、この形のリークは構造的に発生しない
（`data/serve_feature_baseline.json` で `trainer_fuku30/90` の coverage は 0.965 と高く、
既存の「serve skew」台帳（coverage 不足系）とは別種の、学習側だけの生成ロジック不整合）。

**修正**（2026-09-07、`build_dataset.py:add_rolling_stats`）: 主体×レース単位に一度集約
（同一レースの複数頭は fukusho_flag の平均を「そのレースでの結果」とする）してから
`shift(1).rolling()` する方式に変更。同一レース内の馬は全員が同じ「レース前」値を
共有するようになり、レース境界を跨がない。`master_v2` の実データで再検証し、
5,000 組サンプルでリーク行 0 件・行数 626,774 行のまま変化なしを確認。

**未実施（要判断）**: `data/master*.csv` の再生成・再学習は行っていない
（CLAUDE.md のガード対象操作のため確認必須）。したがって現行の本番モデル・
本番指標（gain 表・ECE・◎top3 等）はまだこの修正を反映していない。

**2026-09-11 再検証（コミット状況・実データ照合）**: 上記「修正済み」表記を🔴に訂正する根拠:
1. `git log --oneline -- build_dataset.py` に P0-5 修正のコミットが存在せず、`git status` は
   `build_dataset.py` を未コミット変更としてのみ報告していた（コミットされたことが一度もない）。
2. 本番 `data/master_v2_20130105-20251228.csv`（mtime 2026-06-09）から具体例行（レースID
   2013011408010501・調教師コード1095・馬番10）を直接読み出すと `trainer_fuku30=0.25`
   （リーク値）のままだった。
3. `models/unified_rank_v6.pkl`（2026-07-29）・`pl_calibrators_v6.pkl`（同）・
   `models/pl_calibrators_v6_serve.pkl`（2026-08-24）は全て修正コード作成日（09-07）より前に
   生成されており、学習ログ `reports/optuna_v6_full.log:10` からも同一 master ファイルを
   参照したことを確認した。**現行本番モデル・較正器は未修正データで学習されたまま**。

**次の一手**:
1. ✅ 修正パッチをコミット済み（`916af4fc6c`、2026-09-11）。
2. `data/master*.csv` の本番再生成は CLAUDE.md ガード対象。§5.1k の候補比較結果を見た上で
   ユーザー確認後に実施するかどうか判断する。
3. 再学習・較正器再fit・本番差し替えも同様にユーザー確認後、`backup_model()` で退避してから。
4. **訂正（2026-09-11）**: 「trial_003とは無関係」としていた記述は不正確。無関係なのは
   隔離環境での診断・候補データ/候補モデル作成までであり、本番モデルを実際に置換すれば
   trial_003 の前提（対象モデルの予測）が変わる。P0-5の必要な修復を600R到達待ちで遅らせる
   ことはしないが、**本番置換を行う場合は旧trialを退役させ、新しい本番モデルの下で新しい
   trialを設計し直す**。

**候補データ・候補モデルによる修復検証（2026-09-11、本番非破壊、詳細は Vol. III §5.1k）**:
リーク除去は網羅的証跡（同一レース×同一調教師グループ20,961件全数で不一致0件）と因果
ミューテーション検定（他馬の結果を反転させても対象レースの特徴量は不変）の両方で確認した。
修正は「同一レース内リークの除去」に加え**集計単位を行レベル→レース単位rollingに変更する
副作用を伴い**、trainer_fuku30/90の値は37.7%/68.0%の行で変化。**訂正（2026-09-11）**:
「直接リーク対象6.8%（42,500行）」はグループ先頭（実際にはリークしていない行）を含む
グループ全行数であり、実際に同レース結果が混入した行数は3.4%（21,539行、グループ先頭を
除く）である。いずれの基準と比べても37.7%/68.0%の変化ははるかに広く、変化の大半はリーク
そのものではなく窓幅の意味変更である点に変わりはない。本番v6と同一のハイパー
パラメータ・sample weight・特徴量・学習期間で before/after を2seed再学習した結果、
testの◎複勝(top3)/◎連対(top2)/ndcg5は両seedとも僅かにマイナス方向（-0.09〜-0.67pt、
改善の確証なし・悪化の確証もなし、n=2のためCI未算出）。候補モデルは
`models/candidates/p0_5/`、生成記録は `analysis/p0_5_verification/p0_5_generation_manifest.json`
に保存（本番ファイルは無変更）。

**死亡ルートとの衝突**: なし。新規特徴追加やモデル変更ではなく、既存特徴の
生成ロジックのバグ修正（P0-1〜P0-4 と同系統の「捨てている/汚している情報の是正」）。

**フェーズ2・初版（🔴 2026-09-11 訂正: 出力は無効、履歴として保持）**:
ユーザー指摘を受け、(1)保存済み候補モデルを再学習せず同一クリーン入力で先に再スコアリング、
(2)修復候補を「window意味を保持する最小修正C1」と「レース単位集約する現行候補C2」に分離、
(3)騎手側の原因未特定の差(P0-5と無関係)を本比較から遮断、を実施したが、初版実装に
**特徴量の列順バグ**があった: `feats` をモデル保存済みの `bundle["feature_cols"]`
（学習時の実際の列順、jockey/trainer_fuku4列が120列中68-71番目）ではなく独自に再構成し、
これら4列が末尾に移動した状態のままLightGBMへ推論させていた（`predict()`は列名でなく列位置
で対応づけるため、実質的にデタラメな特徴量割り当て）。当時報告した「baseがarmよりseed42で
有意優位（raw Brier diff=-0.00051 CI[-0.00060,-0.00043]等）」は無効。

**フェーズ2・訂正版（2026-09-11、探索的評価、詳細はVol. III §5.1m）**: 特徴量順序バグを
修正（各モデル自身の `bundle["feature_cols"]` でreindex、列不足・重複を事前assert）し、
併せてC1実装の「先頭値取得」を `groupby(...).transform("first")`（pandas既定でNaNをスキップ
するため、ブロック先頭行が真にNaNでも後続行の非NaNリーク値を誤って配る恐れがあった）から
位置ベース（NaNを保持したまま配布）に修正。「先頭NaN・後続行は有効値」の実例
（レースID=2013011206010304、調教師コード=1089）で、修正後はブロック全体が正しくNaNのまま
で、他馬15頭の結果を反転させても不変であることを検証済み。

同一の保存済みモデル・同一2seed（42, 1、追加なし）で再スコアリングした結果、**最重要の比較
（C2の同一クリーン入力で揃えた base vs arm）は訂正後、Brier/logloss全指標・両seedでCIが0を
跨ぎ有意差なし**（初版の「base有意優位」は列順バグのアーティファクトだった）。C1 vs C2の
入力表現差はcalibrated loglossのみ両seedで有意だが符号が逆転しており（seed42はC1が僅かに
良い、seed1は僅かに悪い）頑健な方向ではない。**この比較は既存モデル（base/arm）への入力
差し替えであり、「C1で実際に学習したモデル」の評価ではない点に注意**。2seed・訓練1本ずつの
点推定であり本番置換の可否判断には使わない。

**master格納値 vs 素の再計算値が不一致する原因の特定（2026-09-11、Vol. III §5.1n）**:
`jockey_fuku30/90`(5.1%/14.0%)・`trainer_fuku30/90`(5.0%/14.8%)の「master格納値と、
base列からの素の再計算値が一致しない」問題（従来「原因未特定」）を特定・完全解消した。
**根本原因はパイプライン内の実行順序**: `build_dataset.py:main()`は`add_rolling_stats()`を
着順dropna**前**の631,965行（除外・中止等で着順NaNの5,191行を含む）に対して実行し、その
**後**でこの5,191行を落として626,774行の最終masterにする。**訂正（2026-09-11、フラグ扱いの
説明を修正）**: この5,191行の`fukusho_flag`はNaNではなく**`0`**（`NaN <= 3`はnumpyの
比較セマンティクスにより`False`と評価されるため、実測で5,191行全て`fukusho_flag=0`を確認）。
つまり除外・中止馬は「値には寄与せず枠だけ消費する」のではなく「出走していないのに
"複勝圏外(0)"として rolling平均の分子・分母に算入される」。それでも`.rolling(window)`は
行の位置ベースの固定長ウィンドウなので、この5,191行が window の1コマを占有し以降の全窓を
ずらす点は変わらない（P0-5窓幅変更の副作用と同じメカニズム、原因は別＝リークではなく
パイプライン段順）。具体例（騎手コード=1102、2013-01-05の除外レース、fukusho_flag=0）:
正しい計算(631,965行universe込み、除外分を0扱いで算入)ではjockey_fuku30=1/6=0.1667、
誤った計算(626,774行のsurvivors-onlyで再計算、除外分が丸ごと消失)では1/5=0.2000。lgbm+cat+addを
`build_dataset.py`と同一JOIN順で結合し631,965行の完全なpre-dropna masterを再構築して
旧アルゴリズムを適用したところ、`jockey_fuku30/90`・`trainer_fuku30/90`の4列全てmaster_v2
格納値とdiff=0で完全一致し、原因特定を実証した。**除外馬を「0(複勝を外した)」として数える
扱い自体の妥当性は未検証の別課題**として記録し（Vol. III P2-11、P0-5とは独立）、今回の
P0-5修復には混ぜていない。

C1(最小修正)もこの正しい631,965行universeで構築し（before→C1の変化行はtrainer_fuku30/90で
1.1%/1.15%、jockeyはほぼ皆無）、本番v6と同一のハイパーパラメータ・alpha・特徴量セットで
学習。初版（`train_c1_true_universe_candidates.py`）はC1TRUEの特徴量順序をdrop+merge後の
データフレーム列順から自己導出しており、baseとは異なる列順だった（数学的には無効ではないが
単一ソース化のためユーザー指摘を受け是正）。**訂正版**（`c1true_ordered_retrain_and_compare.py`）
は特徴量順序を常にbase保存済みの`feature_cols`から明示的に取得し、既存2seed(42,1、追加なし)
のみで再学習・`models/candidates/p0_5/unified_rank_v6_C1TRUE_seed{42,1}.pkl`を上書き。
§5.1mと同じ厳格な指標（Brier・logloss・固定10ビンECE、レース単位paired bootstrap、
raw/calibrated）で同一C1TRUE入力によりbaseと比較した結果、seed42のraw ECE・seed1の
calibrated loglossの2箇所でCIが0を跨がずC1TRUEが僅かに良い方向を示したが、同じ指標が
両seedで一貫して有意というわけではなく、Brierは両seed・raw/calibratedとも非有意。
「明確に改善している」と言えるほど頑健なシグナルではないが、一貫してマイナス方向という
わけでもない中立的な結果。本番ファイル・trial_003は無変更、新規seedの追加もなし。

**C1本番移行案（2026-09-11、提案のみ・本番未実施、Vol. III §5.1o）**: 本番・trial_003は
本項目により一切変更しない。①現状確認: `build_dataset.py`(学習)はC2実装済み・コミット済み
（`916af4fc6c`）だが`data/master_v2*.csv`は未再生成(P0-5修正前のまま)。`serve_history_feats.py`
(serve)もC2実装済みで本日コミット（`579b6a211d`、本セッション中に発見・実施）だが、実際の
`data/_horse_history.parquet`（mtime 2026-09-07）に`race_id`列が無いため**実際にはfallbackの
行レベル集計で動作中**——過去の「serve/train P0-5整合は検証済み」という記述は、race_id付きの
テスト用スクラッチファイルでの検証であり、本番ファイルそのものでの検証ではなかったことが
判明した。学習・serveどちらも「意図したコード」と「実際に読んでいるデータ」が一致していない
のが現状。②C1移行に必要な変更: `build_dataset.py`と`serve_history_feats.py`双方のC2実装を
C1実装に置き換え（serve側も変えないと学習/serveの集計単位不一致がP0-5と別の形で再発する）、
`data/master_v2*.csv`・`data/_horse_history.parquet`の再生成、`unified_rank_v6.pkl`の再学習、
`pl_calibrators_v6*.pkl`・`pl_payout_curve_v6.pkl`の再fit、`data/serve_feature_baseline.json`
の再生成、の連鎖的変更が必要（P0-5単独修正より影響範囲が大きい）。③同日履歴締切差(P1-6)は
既に定量化済み・選択肢(a)は原理的に不可能・選択肢(b)は緊急性なしと結論済みのため、本移行案
には一切含めない。④バージョン管理: `models/candidates/p0_5/versions/{v1_base_leaky,
v2_c2_racelevel,v3_c1true_minimalfix}/`にsha256・生成スクリプト・検証結果ポインタを記録した
`VERSIONS.json`を新設、今後既存バージョンは上書きしない。⑤本番置換前に必要なパッケージ
（変更一覧・検証結果・復旧手順・trial_003退役と後続trial_004の扱い）を提示済み。検証結果
自体が「本番置換を今すぐ正当化する根拠は無い」ことを示している。**現時点の結論**:
本番ファイル・trial_003は維持し、ユーザー承認まで置換・退役のいずれも実行しない。

**隔離候補一式の生成と動作確認（2026-09-11、Vol. III §5.1p、本番未実施）**: 
`candidates/c1_migration_v1/`に本番から完全隔離した候補一式（候補master・候補
`_horse_history.parquet`(race_id付き)・候補モデル(既存C1TRUE seed42、追加学習なし)・
候補`pl_calibrators_v6.pkl`・候補`pl_payout_curve_v6.pkl`）を生成し、sha256を
`CANDIDATE_MANIFEST.json`に固定。①母集団照合の新発見: `build_horse_history.py`は
production/候補いずれも`master_v2`(dropna後、626,774行)から履歴を作るため、除外・中止馬
5,191行は候補`_horse_history.parquet`にも構造的に含まれない——学習側C1(631,965行
pre-dropna universe、除外馬を`fukusho_flag=0`で算入)とserve側候補は「除外馬の扱い」で
母集団が完全一致しない(P2-11とは別軸のtrain/serve間母集団差、今回は是正せず既知の残差
として記録)。②P1-6(同日締切差)は候補一式でも是正せず、既知の差のまま維持。③週次入力
→予測→bundle生成のスモークテスト(`data/weekly/20260906.csv`、本番が同日実際に処理した
週)を実施し、候補履歴にrace_id列ありでフォールバックが発生していないことを確認、解決率
483/484頭・35レース中35成功(production同日と一致)。④本番bundleとの出力差:
ai_score全484頭で変化(候補は別学習インスタンスのため想定通り)、p_win平均差+0.0004
(ほぼ中立)、mark(印)は484頭中99頭(20.5%)で変化——§5.1mの中立的な結果と整合する規模感。
⑤復旧対象はモデル単体でなくコード・master・履歴・較正器・curve・baselineの一式とし、
世代混在を防ぐ一括コピー+sha256検証の切替/復旧手順を用意。本番ファイル・trial_003は今回も
無変更。trial_004は本番へ配置する一式が確定・実際に置換された後、その実体を対象に新しい
未使用IDでロックする(先行ロックしない)。

**serve_c1_patch.pyバグ修正・serve較正器完成・買い目生成までの確認（2026-09-11、
Vol. III §5.1q）**: 初版`serve_c1_patch.py`はtrain側のbroadcast(同一レース内リーク除去)
ロジックを誤ってserve側(既に確定済みの過去レースのみ扱う、リーク懸念なし)に流用しており、
過去レースで同一調教師の2頭が[的中,不的中]だった場合に両者を1つの値へ統一してしまっていた。
「レース単位集約をしない(各行の実際の値をそのまま保持)」だけの実装に修正し、回帰テスト
(`test_serve_c1_patch_regression.py`)で過去2頭の結果が別々に保持されること・馬走単位の窓・
日付締切の3点をPASS確認。修正前bundle(35レース成功=処理完走の確認のみ、C1の正しさの確認
には使えない)は参考値として保持し、修正後に再生成(ai_score が70.2%の頭数で変化=バグの
実影響を確認)。未生成だった候補`serve_feature_baseline.json`・`pl_calibrators_v6_serve.pkl`
を生成し(test2024-25でECE改善: 単勝-0.0007/複勝-0.0058/馬連-0.0117)、production と同じ
「serve版較正器を優先」ロジックをスモークテストに追加して実際に使われたことを確認
(初版はserve版未生成のため通常版を使っていた)。`compute_bets.py --bundle <候補bundle> --dry`
で買い目生成まで動作確認(候補20買い/¥140,000 vs production21買い/¥147,000、1レースのみ
参戦判定差)。本番ファイル・trial_003は無変更。

**ハッシュ照合の完成・baseline再修正（2026-09-11、Vol. III §5.1r）**: 生成物9ファイルに
加え、生成に使った9スクリプト（C1データ生成〜モデル学習〜serveパッチ〜候補一式生成〜
スモークテスト〜回帰テスト）全てのsha256を`CANDIDATE_MANIFEST.json`の`candidate_scripts`に
記録し、最終コードと生成物を1組で固定(全18項目の再計算照合PASS確認)。また
`build_candidate_serve_baseline.py`がbunseki配線(2026-09-05)前後の週を混ぜていた問題
（既出の「週による入力ファイル有無が部分欠損に見える」罠の再発）を、配線後の
20260905/20260906(同じ週末の2開催日、週としては1つ)のみに絞って再修正。
新旧baselineのserve_dead特徴リストを比較し
完全一致(マスク不変)を確認したため、ユーザー指示どおりserve較正器の再fitは実施していない。
serve較正器のECE改善は近似した過去入力上の結果、買い目dry実行の完走は動作確認に過ぎず、
いずれも本番収支の改善を証明しない。追加のモデル学習・seed探索なし。本番置換・trial_003
退役は最終移行内容の確認まで保留。**2026-09-11ユーザー確認**: 上記2点はユーザー側の実ファイル
確認でも解消済みと確認された(ハッシュ18項目一致・マスク17特徴一致)。対象期間は「2週」でなく
「9月5・6日の同じ週末の2開催日」が正確(表記のみ訂正、スクリプトのハッシュは保持)。

**切替・復旧手順の最終確認（2026-09-11、Vol. III §5.1s、本番未実施）**: 手順を見直し2件の
欠落を発見した。①`build_dataset.py`/`serve_history_feats.py`は、C1ロジックへの書き換えが
まだ本体に反映されていない(候補のC1挙動は別スクリプト/実行時パッチで実現している)——このまま
候補データだけコピーすると、次回の本体再実行でC2/fallbackへ静かに戻り「コードとデータの
不一致」が再発する。対応: 切替前にこの2ファイルをC1ロジックへ書き換えてコミットする手順を
追加。②一括コピー対象に`serve_code_maps.json`が抜けていたため追加。`class_prior_v6.json`は
production既存版の流用で問題なし(緊急性なし)。この改訂は本番置換の是非（精度/収支の改善が
実証されたか）には影響しない——目的はリークのある学習データの是正であり、精度・収支の改善は
未実証。本節時点でも本番ファイル・trial_003は無変更、実際の切替はユーザーの明示的承認後。

**本番切替の実施・品質ゲート検知によるロールバック（2026-09-11、Vol. III §5.1t）**:
ユーザー承認を受け実際に切替を実施したが、**ステップ4（通常の入口`export_weekly_marks.py`
経由の動作確認）で品質ゲートが「父タイプ名」「性別限定」の無言死を検知し、ロールバックまで
実施した**。この2特徴はjockey_fuku/trainer_fuku(C1対象)とは無関係で、実測した真の
カバレッジ(9.7%/0%)は旧本番baseline(10.2%/0.0%)と整合していた——**誤っていたのは候補
`serve_feature_baseline.json`の方**（100%/100%と誤記録）。根本原因は、その生成スクリプトを
**本プロジェクトの正しい環境(venv311、Python3.11/pandas2.3.3)ではなくシステムPython
(3.14/pandas3.0.0)で実行していた**こと。venv311で同一ロジックを再実行すると正しい低い値が
再現され、pandas3.0系での文字列/NaN判定の挙動差が強く示唆される。他の候補生成物への影響は
未確認。ユーザー指示「異常があれば退避した旧一式へ戻す」に従い、コード2ファイル（C2実装へ
差し戻すコミット）・データ4ファイル・モデル/較正器/curve4ファイルを一括ロールバックし全件
sha256一致を確認、誤って上書きされたbundle.jsonもgit checkoutで復元した。**本番は完全に
P0-5修正前の状態へ復帰、trial_003も無変更のまま**。次は候補生成物をvenv311で全て再生成・
再検証してから同じ5段階手順をやり直す方針。

**候補v2の再生成・venv311統一・隔離検証（2026-09-11、Vol. III §5.1u、🔴学習重みが本番と
異なる参考値・訂正は§5.1v）**: `candidates/c1_migration_v2/`（旧v1は無変更保持）に全工程
venv311で再生成。`env_guard.py`新設(想定=Python3.11.9/pandas2.3.3/numpy2.4.2/lightgbm4.6.0/
sklearn1.8.0と不一致なら即停止)、stdout多重ラップ回避のため較正器/curve/baseline/serve較正器
は独立プロセスで実行。既存2seed(42,1)のみ・追加探索なし。baselineに§5.1t事故の再発防止
サニティチェックを追加し正しく生成できた。隔離検証では`--out-dir`を候補ディレクトリ配下に
設定しbundle_pathの隔離を実行前に検証、`export_weekly_marks.main()`を実際に呼び出して
品質ゲートまで通し合格。**ただしこのモデルは学習重みバグを含んでおり、以下§5.1vで訂正。**

**候補v2の学習重み欠落バグ修正（2026-09-11、Vol. III §5.1v）**: ユーザー指摘により発見。
`build_candidate_v2_full_pipeline.py`の`prep_from_df()`が`winner_tansho`列を作成しておらず、
`ovm.make_dataset(alpha=...)`が全学習行weight=1.0(sample_weight_alpha=0.0308が実質無視)で
学習していた。①`optuna_v6_marks.load_winner_tansho_pay()`(data/kekka_20130105-20251228.csv、
存在・列数確認済み、無ければ停止)を用い`prep()`と同一ロジックでwinner_tanshoを付与、
独立再計算との全件一致を検証。修正後は学習行の100%でweight≠1.0(旧バグ再現では0%)を確認。
②旧(重み欠落)版モデル3ファイルは`models/UNWEIGHTED_BUG_reference_only/`へ参考値として保持、
master・履歴はハッシュ照合の上再利用、重み修正版で既存2seedのみ再学習(追加探索なし)。
③モデル依存の較正器・curve・baseline(サニティチェック再PASS)・serve較正器
(test2024-25 ECE単勝Δ-0.0083/複勝Δ-0.0134/馬連Δ-0.0074)を再生成し、隔離検証を再実行:
serve canary無言死0件・dead gain6.70%<35%・戻り値0(合格)、本番7ガードパス無変更確認、
`compute_bets.py --dry`終了コード0(18買い/17見送り/¥126,000、重み修正前の19買い/¥133,000から
僅かに変化)。④§5.1uの旧数値は撤回せず「学習重みが本番と異なる参考値」と明記して保持。
`CANDIDATE_MANIFEST_v2.json`は重み修正後の最終版ハッシュに更新済み。本番ファイル・trial_003は
本節でも無変更。実際の本番切替はユーザーの判断・指示待ち。

**本番切替前の最終確認（2026-09-11、Vol. III §5.1w）**: 重み修正が元の学習処理との
485,252行直接照合で完全一致とユーザー確認済みを受け実施。①重み修正済み一式を
`candidates/c1_migration_v2_final/`へ固有版名で凍結（`FROZEN_MANIFEST.json`にコア10ファイル
sha256・生成手順を記録、以後不変）。②配置予定のC1本体コード(`f0adf494f6`と完全一致検証済み
の`staged_code/`)を、`serve_c1_patch.py`のメソッド差し替えではなく**実体を
`sys.modules['serve_history_feats']`として直接ロード**する方式で検証、補助入力
(class_prior/horse_pedigree/kako5)は本番からコピーして揃えbundleへの反映も確認。結果:
戻り値0(品質ゲート合格)・serve canary無言死0・dead gain6.70%<35%・本番ファイル
コード含む9項目の無変更確認・`compute_bets.py --dry`終了コード0(18買い/¥126,000、実行時
パッチ版と完全一致)。③調査の結果**`build_horse_history.py`は2026-07-29時点で既に
race_id対応済みでロールバック範囲外だったため追加変更不要**と判明、実際に書き換え・コピーが
必要なのは**10ファイル**(コード2+データ4+モデル/較正器/curve4)。補助ファイル・復旧先
(`models/production_backup/20260911_pre_c1/`)を`SWITCH_PLAN_FINAL.json`に確定記録。
本番ファイル・trial_003は無変更のまま。実際の本番切替・trial_003退役はまだ実行していない。

**本番切替を実施（2026-09-11、完了・✅、Vol. III §5.1x）**: ユーザー承認を受け5段階手順で
実施、全ステップ成功・ロールバックなし。①事前確認: 週次非実行、本番11ファイルと退避先が
記録済みハッシュと全件一致を確認。②`SWITCH_PLAN_FINAL.json`記載の10ファイル
(コード2+データ4+モデル/較正器/curve4)を配置しsha256照合、コード2ファイルをコミット
(`b2e2bdd4b6`)。`build_horse_history.py`は対象外(既に対応済み)。③パッチ一切なしで
`export_weekly_marks.py`を実際にCLI起動、終了コード0(品質ゲート合格)・serve canary無言死0・
dead gain6.70%<35%、ガード対象12項目の無変更を確認、`compute_bets.py --dry`終了コード0
(18買い/¥126,000、凍結候補での隔離検証と完全一致)。④ロールバックは発生せず。
⑤trial_003を理由付きで退役(`trial_003_retired.json`新設、`trial_003.json`自体は保持)——
config_hash.modelが切替前モデルを指しており評価対象の前提が崩れたため。後続trialは
自動開始せず未設計。本番はP0-5修正(C1、train/serve集計統一)済みの状態へ切り替わった。
C1採用の根拠はリークのある学習データを正すことであり、精度/収支の改善は未実証のまま。

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
| **P2-1 ✅** | ~~`t10_runner.py:347, 369` が削除済み `gutchi_brain` を import~~ | 2026-09-11 再検証: `grep gutchi_brain t10_runner.py` が0件、既に解消済み（stale診断） |
| **P2-2** | v6 が採用ゲート（単勝高 EV ROI +0.05 or 複勝 +0.03）を満たさず本番化（実測 単勝 **−0.006** / 複勝 +0.002、レポート自身の結論は「❌ v6 採用見送り」） | §6.1 |
| **P2-3** | test 2024-25 が 7 回以上開封済み。v6 の 62.08% は多重比較で選ばれた値 | §6.3 |
| **P2-4** | topdown が bundle の**較正済** `pair_probs` を使わず、`pl_pair_probs()`（**未較正** λPL）を再計算（`compute_bets.py:483-484`、既定エンジン。bundle較正済み`pair_probs`は非既定の`shape`エンジンでのみ使用）。**訂正（2026-09-11）**: 「λPLとbundle厳密値の比0.99±0.1でバイアスなし」は確率同士の比較に過ぎず、実際の的中結果に対する較正の質や買い目への影響を検証したものではない。**実結果に対する較正・買い目への影響は未確認として扱い、今回は変更しない** | 印 5 頭以外のペアも扱うため単純置換はできない。全馬 pair に較正器を適用する形が正しい（対応は保留、要再評価） |
| **P2-5 ✅** | 見送り4条件を `production_policy.py` と `data/production_policy.json` に単一ソース化。各経路は共通関数を使用 | 2026-08-25 解消 |
| **P2-6** | `docs/compute_bets_spec.md`（2026-06-09）が実装と 9 項目乖離（Vol. II §2.9） | 本 Vol. II を正典とし、旧仕様書に DEPRECATED を明記 |
| **P2-7** | `predict_weekly.parse_csv` にテストが無い。**「実 CSV を 1 本パースして主要特徴の nunique > 1 を assert する」テストがあれば P0-1/P0-2 は即検出できた** | `tests/test_serve_parse.py` を追加 |
| **P2-8** | Optuna の CV が **valid 内レース ID のランダム 5-fold**。時系列 CV ではない | valid が 1 年なので regime 差は小さいが、厳密性は無い |
| **P2-9 ✅** | ~~`validate_cowork_bets.ALLOWED_KINDS` に禁止券種「馬単」が残存~~ | 2026-09-11 再検証: 現行 `ALLOWED_KINDS = {"単勝","複勝","ワイド","馬連","三連複"}`、`REJECTED_KINDS = {"馬単","三連単"}` に既に分離済み（stale診断） |
| **P2-10** | `weekly_nicegui.ps1 -Post` の `generated_at` 凍結検知が **Warn 止まり** | Fail にする（`weekly_post.ps1` の git add ガードと同レベルにすべき） |
| **P2-11**（2026-09-11新規、P0-5とは独立） | `build_dataset.add_rolling_stats` の元になる `fukusho_flag=(着順<=3).astype("Int8")` は、着順=NaN（除外・中止・失格等、5,191行）の馬を **NaNではなく`0`（複勝を外した）として** rolling統計に算入する。出走していない馬を「敗北」扱いするのは意味論的に不正確 | 実害未確認・要検証。P0-5の窓幅変更・リーク修正とは独立の別課題として扱う（混ぜない） |

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
| ~~joint_m1 umaren~~ | ~~ECE −33% / ROI +0.3〜0.7pt~~ | **削除済み**: UNDERPOWERED ではなく MIRAGE と判明（§3.2 参照）。ROI +0.3〜0.7pt は単位バグで、正しい照合では黒字セル 0/36 |

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

### 8.4 未決着の先行実験（放置されたが有望かもしれない）

**背景（2026-09-02）**: 外部 AI（ChatGPT）が「v6 の 1 本の LambdaRank score を PL に通す代わりに、
WIN/TOP2/TOP3 を別 target として直接学習する」（PRED-01、着順を直接教師にする設計）を提案。
調査の結果、**死亡ルートでも生存レバーでもない「一度手を付けて結論を出さず放置された実験」**が
見つかった。今後また同種の外部提案が来たときのために記録する。

| 資産 | 内容 | 現状 |
|---|---|---|
| `models/fukusho_binary_v1.pkl` | `fukusho_flag`（3着以内）を直接教師にした LightGBM 二値分類器。PRED-01 の「TOP3 head」相当 | ファイル現存（2026-05-25 作成）。再学習不要 |
| `models/order_model_v1.pkl` | 1着 / 2-3着 / 4着以下 の **3 クラス softmax**（`lab/train/train_order_model.py`）。PRED-01 の WIN/TOP2/TOP3 構想に近い | ファイル現存（2026-07-29 作成）。HALO formation 用に作られたが v6 との直接比較は未実施 |
| `archive/experiments/backtest_fukusho_stack.py` → `reports/backtest_fukusho_stack.json` | **現行 `unified_rank_v6.pkl` + 現行 `master_v2` を使用**。v6 PL 由来複勝確率と `fukusho_binary_v1` を α 配合し test2024-25（n=6909、標準封印 test と同一母集団）で複勝 ROI を比較済み | 結果は下表。正式な PASS/FAIL 判定は記録なし（lab 整理で放置された可能性が高い） |

**結果**（α=0 が純 v6、α=1 が純 binary、単位: 複勝 100 円 3 頭買い）:

| 手法 | ROI | 的中率 |
|---|---:|---:|
| 純 v6（PL 由来） | 82.73% | 49.53% |
| v6+binary 配合（α=0.6〜0.95） | 83.2〜83.4% | 49.2〜49.5% |
| **純 binary（fukusho_binary_v1 のみ）** | **83.71%（最高）** | 49.12%（最低） |
| v6 印固定（◎〇▲ 全部） | 83.15% | **49.76%（最高）** |

方向としては「複勝を直接学習した単体モデルが v6+PL より ROI +0.98pt」で、外部 AI 提案（PRED-01）の
事前予測（+0.3〜1.3pt）と整合する。ただし複勝馬券の ROI/的中率グリッドとしての評価であり、
「argmax が実際に ◎top3=62.08% を超えるか」「10-bin ECE」「race 単位 paired bootstrap」という
現行の検証規律では一度も評価されていない。WIN 単体 head・TOP2 単体 head・確率整合射影
（Σp1=1, Σp2=2, Σp3=3, 単調性）はこの実験にも存在せず、真に未検証。

**次の一手（安価）**: PRED-01 をゼロから 3 head 新設する前に、**再学習不要**でこの 2 資産
（`fukusho_binary_v1.pkl` / `order_model_v1.pkl`）の予測確率から argmax per race の top3 率と
10-bin ECE を現行封印 test で計測し、62.08% と直接比較する。数百行の新規学習より安い。

### 8.4.1 ★決着（2026-09-02、`analysis/gate0_direct_outcome_eval.py`）: FAIL・PRED-01 終了

ChatGPT 自身が事前登録した判定木にそのまま従って判定。**明確な FAIL**（改善なしどころか悪化）。

| モデル | argmax top3 率（test2024-25, n=6,909R） | v6 比（race 単位 paired bootstrap） |
|---|---:|---|
| v6（オフライン基準 62.08% と一致） | **62.05%** | — |
| fukusho_binary_v1 | 60.31% | **Δ−1.74pt CI95[−2.46, −1.01]**（有意に悪化） |
| order_model_v1 | 60.50% | **Δ−1.55pt CI95[−2.52, −0.56]**（有意に悪化） |

**核心（disagreement subset）**: v6 と argmax が割れたレースに限定しても challenger は勝てない。

| | n | challenger 的中率 | v6 的中率 | Δ |
|---|---:|---:|---:|---:|
| binary vs v6 不一致 | 1,204R | 46.68% | 56.64% | v6 が +9.97pt 圧勝 |
| order vs v6 不一致 | 2,185R | 49.84% | 54.74% | v6 が +4.90pt 勝ち |

確率品質（Brier/logloss）も v6 が両challengerに優る。race coherence（Σp_top3、目標3.0）は
v6=3.0（PL由来で数学的に保証）・binary=3.14・order=2.95 とそこそこ近く、破綻の原因は
確率の整合性ではなく**素の順位づけ能力**（LambdaRank のペアワイズ最適化 vs 独立
binary/multiclass 分類の非最適化）にある。

**結論**: PRED-01（判定木の Case C = 両モデルとも v6 以下）で終了。3 head 新設や
`Σp_top3` projection 研究に進む価値なし。死因型は **MIRAGE**（発見期の期待値が実測で消滅）。
再走するなら根本的に異なる特徴セット・目的関数設計が必要で、現行の「同じ入力を
binary/multiclass で再学習するだけ」の変種は再走禁止。詳細: `reports/gate0_direct_outcome_eval.json`。

### 8.4.2 PRED-02（Serve-Aware Masked LambdaRank）: 前提のミスマッチで実装せず（2026-09-02）

外部 AI（ChatGPT）が「本番の特徴欠損パターンを学習時に再現し、欠損に頑健な LambdaRank を
学習する」（domain randomization / missingness augmentation）を提案。先行実験は本当に存在せず、
その点の用心（実装前に prior art を確認せよ）は正しかった。ただし **前提となるメカニズムが
`data/serve_feature_baseline.json` の実測と食い違う**ことが判明し、実装前に却下。

**実測（v6 の 120 特徴、直近週の実効カバレッジ）**:

| カバレッジ | 特徴数 | 性質 |
|---|---:|---|
| 0%（毎週決定論的に死亡） | 30 | 確率的欠損ではない |
| 1〜94%（部分的） | 62 | 一見確率的だが下記参照 |
| 95%以上（実質健全） | 28 | 問題なし |

1. **決定論的死亡 30 特徴のうち 28 は既に「回収可能」（`serve_skew_recoverable`）と分類済み**
   （残り回収価値 ~1–1.4pt、既に scope 済み）。真に構造的に取り戻せないのは
   ブリンカー・性別限定の**2 特徴のみ**。100% 決定論的に死ぬ特徴に対する
   マスク学習は「その列を除いて学習する」ことと数学的に同値であり、
   ブロック単位の複数マスクパターンを用意する設計上のメリットがない。
2. **部分カバレッジ 62 特徴のうち一部は確率的でなく構造起因**（バグとは限らない）。
   例: `前1角`（前走 1 角通過順位, カバレッジ 40.8%）vs `前走確定着順`（同じく前走由来, 90.9%）。
   当初「パースバグの疑い」として調査したが、**2026-09-02 の追跡調査で否定・解消済み**。
   `前1角` カバレッジは距離と完全に相関する（≤1400m: 7.9% / 1400-1800m: 49.5% /
   1800-2200m: 70.6% / 2200m+: 90.1%）＝**短距離戦は物理的に「1 角」という区切りが
   実況記録上ほぼ存在しないため**の構造的欠損であり、パイプラインのバグではない。
   train/valid/test でカバレッジが 38.1%/39.8%/39.6% とほぼ同一＝train 時点から
   一貫した特性で、serve だけの劣化でもない。`前3角`/`前4角`（88.5%/89.1%）は
   `前走確定着順`（89.8%）とほぼ同水準＝距離を問わず記録される後方コーナーは
   健全。**この件は解消済み、再調査不要**（詳細: 本節末尾の追跡結果）。

**結論**: PRED-02 が想定する「irreducible な確率的欠損」はこのプロジェクトにはほぼ存在せず、
実際の欠損は「決定論的（100% 死亡）」か「構造起因の部分欠損（距離帯依存など、上記 8.4.2
本文参照）」に二分される。どちらも masked training ではなく **パイプライン修復 or 現状追認**
で対処すべきで、実装しない。死因型は **MIRAGE**（提案時の前提が実測で崩れた）。

### 8.4.3 PRED-03（Margin-Aware Ranking）: prior-art 確認済み・実装は serve 修復後（2026-09-02）

「LambdaRank のペア損失に、着差（ハナ差 vs 大差）を教師信頼度として重み付けする」提案。
着差を **特徴量として使う話ではなく、学習時のペア重みにのみ使う**点が新規性。

**prior-art 確認**: 該当なし。`sample_weight`/`weight` 系のコード検索では
`optuna_v6_marks.py` の既存重み（`w = 1 + alpha*log1p(勝ち馬単勝オッズ/100)`、
**レース単位**で「人気薄が勝った波乱レースを重視する」市場重み）がヒットしたのみで、
**レース内のペア単位**で着差を信頼度として使う実装は過去に存在しない。真に新規。

**実装上の留意点（着手前に設計で潰す）**: LightGBM 標準の `lambdarank` objective は
ペア重みを離散ラベル（`clip(6-着順,0,5)`）の `label_gain` 差分からしか算出せず、
連続値の着差（秒差）を直接ペア重みに反映する口は無い。実現するには
(a) 独自 objective（`fobj`）を書く、(b) 行単位 `weight`（人単位、ペア単位ではない）で
近似する、のいずれかが必要で、DR-01A/PRED-01/PRED-02 より実装コストが高い。
設計段階でこれを先に決めてから着手すること。

**状態**: prior-art はクリア。ただし優先順位表（§9）どおり **serve 修復完了後**に着手。
今すぐ実装しない。

### 8.4.4 PRED-03A-0（影響量監査）: 対象量は極めて大きい・03A-1 着手（2026-09-02）

実装前に「この変更が何レースに効くのか」を測定。今走の着差・走破タイムは自リポジトリの
master には無いため `E:\競馬過去走データ\raw_data\kekka_1986_2025_enhanced.csv`（1986-2025、
今走の `走破タイム`/`異常コード`/`確定着順` を含む）を日付+場所+R+馬番で join
（match率 99.4%、`着順` 一致率 99.99% でサニティ確認済み）。

**罠**: `着差タイム` は 1 着だけ符号規則が特殊（2 着に勝った差の符号反転）で、単純に
2 頭分を引き算すると 1 着絡みのペアだけ実際の 2 倍になるバグを生む。**教師 margin は
必ず `走破タイム` の差分から計算する**（`着差タイム` は使わない）。

**測定結果（2013-2025、正常完走馬同士の隣接着順ペア、n=575,837 ペア / 44,643 レース）**:

| 指標 | 値 |
|---|---:|
| 1-2 着 exact-tie（走破タイム差 0.0 秒）率 | 24.2% |
| 2-3 着 | 22.3% |
| 3-4 着 | 25.1% |
| 4-5 着 | 26.4% |
| top5 内に exact-tie が 1 組以上あるレース率 | **77.66%**（n=44,643） |
| 芝 vs ダ（1-2 着） | 芝 26.9% / ダ 21.5%（大差なし） |
| 距離帯別（1-2 着） | 20-25% 台でほぼ均一（極端な偏りなし） |

「2% しかないなら見送り」という事前の懸念は完全に外れ、**対象は大多数のレースに存在する
一般的な現象**と判明。03A-1（学習実験）に進む十分な根拠あり。

### 8.4.5 ★決着（2026-09-03、`analysis/pred03a_neartie_collapse_eval.py`）: PRED-03A は FAIL

v6 実機の `optuna_best_params`（HP）・120 特徴量・boosting round 数（515、実モデルの
`model.num_trees()` と同一）を完全固定し、教師ラベルのみ
`clip(6-着順,0,5)` → near-tie collapse 版に差し替えて再学習（train split のみ、
Optuna 不使用）。train 485,252 行中 46,773 行（9.64%）でラベルが変化。

**結果（test2024-25, n=6,909R）**:

| 指標 | **matched retrain baseline**（注1） | near-tie |
|---|---:|---:|
| ◎top3 率 | 61.35% | 61.35% |
| Δ | — | **+0.00pt** CI95[-0.81, +0.77] |

> **注1（重要）**: この 61.35% は「本番 `unified_rank_v6.pkl`」の値（62.05〜62.08%）とは
> **別物**であり、その数値として引用しないこと。本実験は HP・特徴量・boosting round 数を
> pkl から複製したが、sample_weight は uniform 近似（実際の
> `winner_tansho` ベース重みを未実装）にしたなど、**PRED-03A の A/B 比較にのみ有効な
> matched retrain**であり、本番モデルの bit-for-bit 再現ではない。この 0.7pt 差の原因
> （最終 train 範囲・`best_iter×1.1` の丸め方・encoder・sample weight 完全再現・
> Optuna 後の本番リトレイン手順との差異）は **PRED-03A の判定には不要だが、
> 再現性台帳の別課題として記録**する（未着手、優先度低）。

**disagreement subset（pick が割れた 1,368R、test 全体の 19.8%）**: near-tie 51.54% vs
base 51.54%、**Δ=+0.00pt** CI95[-3.95, +3.95]。

一見「集計バグ」を疑うレベルの完全一致だったため、生スコアと pick 馬を直接検証した
（raw score 差分の絶対値合計 15,200 超・pick 馬が実際に異なるレースあり・
disagreement 1,368R は debug 検証の argmax 不一致率 19.8% と一致）。**バグではなく
本物の結果**だが、base 705 / near-tie 705 の完全一致自体は**興味深い副産物であって
本質ではない**。

**本質**: **argmax pick を約 20% のレースで変更しておきながら、的中率は CI [-0.81, +0.77]
の範囲内（点推定 +0.00pt）で、事前登録した採用基準（Δtop3 ≥ +0.5pt）を満たす動きは
見られなかった**。これは「変化が小さすぎて検出できなかった」（PRED-01 型の検出力不足）
とは異なり、**メカニズムは確実に発火したが、採用基準を満たす便益は検出できなかった、
という綺麗な FAIL**。この CI は ±0.5pt 程度の実効果を排除できるほど狭くはない点に注意
（「便益ゼロが確定した」ではなく「採用条件未達」が正確な言い方）。

**判定**: PASS 条件（Δtop3 ≥ +0.5pt かつ disagreement subset でも勝つ）を満たさず、
**PRED-03A は FAIL**。事前登録した判定木に従い PRED-03B（pairwise margin-weighted
custom objective）へは進まない。ただし否定されたのは**「0.1 秒量子化された走破タイム
が同一の馬を同一 relevance grade に collapse する」という具体的な方法**のみであり、
「ペア単位で連続的な confidence を与える custom loss なら効く」という一般仮説そのものは
反証されていない。とはいえ 46,773 行（9.64%）のラベルを変更し 1,368R で◎を変えても
採用基準を満たす便益は確認できなかった以上、custom objective 実装（03B）に工数を割く
事前期待値は大きく下がった。**正確な位置づけは「PRED-03B: 理論的には未決着・実務的には
採用条件未達で打ち切り」**であり、「僅差順位ノイズ仮説が否定された」または「効果ゼロが
確定した」と一般化して書かない。死因型 **MIRAGE**。
詳細: `reports/pred03a_neartie_collapse_eval.json`。

### 8.4.6 2026-09 新理論 4 系統の総括

外部 AI（ChatGPT）発の新規予測設計 4 系統（DR-01A・PRED-01・PRED-02・PRED-03）が
全て決着した。並べると示唆が強い。

| 系統 | 変更したもの | 結果 |
|---|---|---|
| DR-01A | score → 確率分布（Gamma/Thurstone） | P3（校正止まり、ROI 不変の同型パターン 3 例目） |
| PRED-01 | 学習 target（direct binary/order） | **明確に悪化**（argmax top3 有意に低下） |
| PRED-02 | missingness robustness（masked training） | **前提不成立**（本番欠損は決定論的/構造起因） |
| PRED-03A | ranking 教師の confidence（near-tie collapse） | **pick を 20% 変えても採用基準未達**（Δ+0.00pt CI[-0.81,+0.77]） |

ここから言えるのは、**現状のボトルネックは「LambdaRank という学習方法」ではなさそうだ**
ということ。PRED-01 は LambdaRank を外すと悪化し、PRED-03A は LambdaRank 内部の教師を
変えても改善しない＝現行の学習方式そのものは的確に情報を使い切っている。v6 の gain は
上位 2 特徴だけで 25.6%、過去走系全体で 60.9% を占め（§2.3）、モデル数理をさらに
ひねるより **「既に学習した情報を本番に 100% 届ける」方向のリターンが相対的に高い**。
研究優先順位を §9 のとおり serve 修復側に戻す根拠として記録する。

---

## §9 研究アジェンダと優先順位

### Priority 0 — 欠陥の修復（最も確実にリターンがある）

| # | 内容 | 期待効果 | リスク |
|---|---|---|---|
| 0-0 | ✅ **P0-4（`R`→`Ｒ` のリネーム 1 行）** | 2026-09-10 解消確認済み（§5.1a） | 実質ゼロ |
| 0-1 | ✅ **P0-2（tyaku 53 列）を直す** | 2026-09-10 解消確認済み（§5.1a） | 極小。純粋なバグ修正 |
| 0-2 | ✅ **P0-1（騎手/調教師 stats 再 merge）を直す** | 2026-09-10 解消確認済み（§5.1a） | 中（静的スナップショットの as-of 性に注意） |
| 0-3 | **`measure_serve_coverage` で baseline 再生成 → P0-3（serve 較正器のマスク同期）** | ECE 改善。0-0〜0-2 が解消済みなので baseline 再生成が次の実作業 | 中 |
| 0-4 | **P1-1（canary の絶対水準可視化）** | 再発防止 | 極小 |
| 0-5 | **P2-7（parse_csv のテスト追加）** | 再発防止 | 極小 |

**この 5 件は「新しいアイデア」を一切必要とせず、既に存在する情報を本番に届けるだけ**である。
本プロジェクトで残っている数少ない確実な仕事。

### Priority 1 — topdown の前向き検証（最重要・最安）

観測するだけ。難易度低、リーク危険なし。§8.1。

### Priority 2 — 検証済み未配線 ROI の回収

| # | 内容 | 状態 |
|---|---|---|
| ~~2-1~~ | ~~joint_m1 umaren の配線~~ | **中止**（2026-06-27 単位バグ判明・MIRAGE 確定、§3.2/§8.3 参照）。再走禁止 |
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
| 控除率 | 売上からプールが引く割合。単複 20% / 枠連・馬連・ワイド 22.5% / 馬単・三連複 25% / 三連単 27.5% / WIN5 30% |
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
**訂正（2026-09-10）**: 旧記述は誤り。各段で残存馬について `z1 = S1 - q1[a]` / `z2 = S2 - q2[a] - q2[b]` により条件付き確率として正規化しているため、λ の値によらず `um` の合計は 1.0、`wd` の合計は 3.0 になる（有限入力・非ゼロ分母の下で数学的に保証される）。**較正器を通していない**点のみ実在の懸念（Vol. III P2-4）。

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
ALLOWED_KINDS  = {単勝, 複勝, ワイド, 馬連, 三連複}   # 2026-09-11 再検証: 馬単は含まれない（Vol. III P2-9 ✅解消済み）
REJECTED_KINDS = {馬単, 三連単}
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
    ...  # Vol. III §10.3 のスニペット参照。現状 14.88%（2026-06旧監査28.15%から改善済） → 目標 <10%
```

---

**← [Vol. I](VOL1_SYSTEM.md) / [Vol. II](VOL2_BETTING_OPS.md) / [Vol. III](VOL3_VALIDATION_AND_OPEN_PROBLEMS.md)**

