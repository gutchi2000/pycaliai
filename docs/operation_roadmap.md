# PyCaLiAI 運用ロードマップ (2026-04-27 更新版)

v5 モデル GATE PASS + **Cowork (Anthropic Desktop App) 連携** 追加に伴う最新版。
旧 `weekly_pre.ps1` (LGBM v1 系の自動買い目生成) と並行して、
新 `weekly_cowork.ps1` (v5 印 JSON → Cowork → 手入力) を運用する 2 系統構成。

---

## 1. システム全体図

```
[週次CSV  data/weekly/YYYYMMDD.csv]
                │
        ┌───────┴───────┐
        │               │
        ▼               ▼
weekly_pre.ps1      weekly_cowork.ps1   ← NEW
(LGBM v1 自動)      (v5 印 JSON 生成)
        │               │
        │               ▼
        │       reports/cowork_input/{YYYYMMDD}_bundle.json
        │               │
        │               ▼
        │       [Cowork (Anthropic Desktop App)]
        │               │
        │               ▼
        │       [Cowork が買い目を返す]
        │               │
        ▼               ▼
   reports/pred_     streamlit run app.py
   YYYYMMDD.csv     →「🤖 Cowork」タブで手入力 → 保存
        │               │
        │               ▼
        │       reports/cowork_bets/{YYYYMMDD}/{race_id}.json
        │
        ▼
   Streamlit 上で
   従来 4 戦略表示
   (TRIPLE/HAHO/HALO/STANDARD)
```

**両系統を併用**: 自動派 (TRIPLE/HAHO/HALO/STANDARD) と Cowork 提案派 (手入力) を
同じ画面で見比べて検討できる。

---

## 2. 週次ワークフロー (NEW)

### 2-A. 従来フロー (LGBM v1 自動買い目)

そのまま継続:
```powershell
.\weekly_pre.ps1 20260426    # 自動: 印 + 4 戦略の買い目を pred CSV へ
```

### 2-B. Cowork フロー (v5 印 → Cowork → 手入力)  ← NEW

```powershell
# Step 1: v5 印 JSON を生成 + git push
.\weekly_cowork.ps1 20260426 v5

# 出力 (local):
#   reports/cowork_input/20260426_bundle.json   (Cowork 投入用、35 races なら ~216 KB)
#   reports/cowork_input/20260426/{race_id}.json (個別、Streamlit から expander で見る用)
#
# git: 上記ファイル + data/weekly/20260426.csv を origin/master に自動 push
#      → Streamlit Cloud に数秒で反映 (出先からも閲覧可)
```

```text
# Step 2: Cowork (Anthropic Desktop App) を開く
# Step 3: bundle JSON を Cowork のチャットに投げる
#    プロンプト例:
#      「この JSON は PyCaLiAI 競馬予測モデル (v5) の印データです。
#       docs/marks_schema.md の役割分担に従い、馬券構築 (単複馬連馬単ワイド) と
#       予算配分・見送り判断を出してください。
#       1R = 10,000円基準。レース性質 (固い/中堅/混戦/穴推奨/見送り) を判定して。
#       期待値計算には p_win × tansho_odds を使う。」
#
# Step 4: Cowork が race ごとに買い目を提案する
```

```powershell
# Step 5: Streamlit で買い目を入力・保存 + git push
streamlit run app.py
#   1. サイドバーで日付を選択
#   2. 会場・レースを選択
#   3. 「🤖 Cowork (Anthropic Desktop)」タブを開く
#   4. Cowork の提案を見ながらフォームに入力
#       - 馬券種 (単/複/馬連/馬単/ワイド/三連複/三連単)
#       - 買い目 (例: "4" / "4-7" / "4-7,4-9")
#       - 購入額
#   5. ➕追加 → 入力済み一覧に積む → 💾 保存 + git push
#
# 保存先 (local):
#   reports/cowork_bets/20260426/{race_id}.json
#
# git: 保存ボタン押下時に subprocess で
#      git add → commit → pull --rebase → push origin HEAD:master
#      失敗してもファイル保存は維持される (warning 表示のみ)
```

```powershell
# Step 6: レース後に従来通り weekly_post.ps1 で結果集計 + 全 push
.\weekly_post.ps1 20260426
# - data/kekka/20260426.csv, results.json, live_results_2026.csv を git add
# - 当日 reports/cowork_bets/20260426/ がある場合はそれも git add (的中判定対象)
# - git commit + pull --rebase + push origin HEAD:master
```

---

## 3. コマンドリファレンス

| コマンド | 役割 | 系統 |
|---------|------|------|
| `.\weekly_pre.ps1 [DATE]` | LGBM v1 自動買い目 (pred CSV 生成) → push | 従来 |
| `.\weekly_cowork.ps1 [DATE] [MODEL]` | v5 印 JSON 生成 + cowork_input/ を git push | NEW |
| `python export_weekly_marks.py --csv <CSV> --model v5` | 印 JSON 生成 (PowerShell ラッパー無し) | NEW |
| `streamlit run app.py` | UI: 4 戦略表示 + 🤖 Cowork タブで手入力 | 拡張 |
| `.\weekly_post.ps1 [DATE]` | レース結果取得 + 集計 | 従来 |
| `python export_marks_json.py --year 2025 --out-dir reports/marks_v5/` | バッチ出力 (年単位、過去データから) | 既存 |
| `python verify_marks_batch.py --dir reports/marks_v5/` | バッチ JSON 検証 | 既存 |

---

## 4. ファイル配置

```
E:\PyCaLiAI\
├─ data/
│   ├─ weekly/{YYYYMMDD}.csv          ... 週次出走表 (TARGET エクスポート)
│   ├─ tyaku/{YYYYMMDD}.csv           ... 着度数CSV (任意)
│   ├─ kako5/{YYYYMMDD}.csv           ... 過去5走CSV (任意)
│   └─ hosei/H_{YYYYMMDD}.csv         ... 補正タイム
│
├─ models/
│   ├─ unified_rank_v5.pkl            ... v5 LightGBM (採用、payout-weighted, alpha=1.325)
│   ├─ unified_rank_v4.pkl            ... v4 (バックアップ)
│   ├─ pl_calibrators_v5.pkl          ... isotonic キャリブレータ
│   └─ ...                              (既存 LGBM v1 系も保持)
│
├─ data/
│   └─ pl_payout_curve_v5.pkl         ... Kelly payout curve
│
├─ reports/
│   ├─ marks_v5/{race_id}.json        ... バックテスト用 (2024-2025 OOS = 6,908 races)
│   ├─ cowork_input/{YYYYMMDD}/{race_id}.json   ... NEW: Cowork に渡す印 JSON (個別)
│   ├─ cowork_input/{YYYYMMDD}_bundle.json      ... NEW: Cowork に渡す印 JSON (集約)
│   ├─ cowork_bets/{YYYYMMDD}/{race_id}.json    ... NEW: Cowork からの買い目 (手入力)
│   └─ pred_{YYYYMMDD}.csv            ... 従来 LGBM v1 系の自動買い目
│
├─ docs/
│   ├─ marks_schema.md                ... 印 JSON スキーマ仕様
│   ├─ operation_roadmap.md           ... THIS FILE
│   └─ hypothesis_registry.md
│
├─ weekly_pre.ps1                     ... 従来 (LGBM v1 自動)
├─ weekly_cowork.ps1                  ... NEW (v5 印 JSON → Cowork)
├─ weekly_post.ps1                    ... 従来 (結果集計)
├─ export_weekly_marks.py             ... NEW (CSV → 印 JSON)
├─ export_marks_json.py               ... 既存 (年単位バッチ)
├─ verify_marks_batch.py              ... 既存 (バッチ検証)
└─ app.py                             ... Streamlit UI (Cowork タブ追加済)
```

---

## 5. v5 モデル品質 (真OOS 2024-2025, R=6,878)

| 指標 | v5 | v4 | 差分 |
|------|----|----|------|
| NDCG@5 | **0.6027** | 0.5790 | +0.024 |
| ◎ 1着率 | **30.28%** | 27.39% | +2.89pt |
| ◎ 連対率 | **49.04%** | 45.81% | +3.23pt |
| ◎ 複勝圏率 | **61.66%** | 58.61% | +3.05pt |
| 1着∈top-3 | **60.88%** | 58.74% | +2.14pt |
| 1着∈top-5 | **78.03%** | 75.41% | +2.62pt |
| {1,2}⊂top-5 | **54.32%** | 50.99% | +3.33pt |
| ECE 単勝(◎) | 0.0186 | 0.0186 | ≈tie |
| ECE 複勝(◎) | **0.0173** | 0.0193 | -0.002 (改善) |
| ECE 馬連(◎-〇) | 0.0218 | 0.0154 | +0.006 (劣化、絶対値は実用範囲) |

**学習設定** (再現可能):
- `clip(6 - 着順, 0, 5)` 連続ラベル
- `sample_weight = 1 + 1.325 * log1p(tansho_pay/100)` (穴馬好走を重視)
- LambdaRank, truncation_level=5
- Optuna TPE, 30 trials, 5-fold race split
- `seed=42`, `deterministic=True`, `force_col_wise=True`, `feature_pre_filter=False`

---

## 6. JSON スキーマ (Cowork 連携)

詳細: `docs/marks_schema.md`

主要フィールド:
- `race_meta`: date, place, course, field_size, class, race_name
- `horses[]`: umaban, horse_name, mark (◎/〇/▲/△/""), p_win, p_plc, p_sho,
              tansho_odds, fuku_odds_low/high, ai_vs_market (under/fair/over/unknown)
- `race_confidence`: top1_dominance, top2_concentration, field_chaos_score, ai_market_agreement

**バックテスト用 (年単位)** と **週次 (Cowork 入力用)** で同一スキーマ。

---

## 7. トラブルシューティング

### 7-1. `weekly_cowork.ps1` で印 JSON が 0 races になる

```
[ERROR] rid=...: "['Ｒ', '芝(内・外)', ...] not in index"
```

→ `export_weekly_marks.py` 内で v5 feature_cols に対する不足列を NaN/`__NaN__` で補完済み。
   このメッセージが出続ける場合は v5 model の feature_cols が変わった可能性。
   `bundle["feature_cols"]` を確認して `_ROLLING_COLS` / `cat_cols` 区別を見直す。

### 7-2. Streamlit で「印 JSON が未生成です」と表示される

→ `weekly_cowork.ps1 YYYYMMDD` を先に実行すること。
   `reports/cowork_input/YYYYMMDD/` がそのまま空でないかも確認。

### 7-3. Cowork が JSON を読まない / 文字化け

→ JSON 自体は UTF-8。bundle ファイルは `_bundle.json` の方が race 数が多い場合
   コピペ困難。代替: 個別 race の JSON を 1 つずつ投げる。

### 7-4. Streamlit 上で日付や race_id 取得失敗

→ `_rd_date` (例: "2026.4.26") を YYYYMMDD に変換するロジックが
   Cowork タブの先頭にある。CSV ファイル名と日付列が一致しているか確認。

### 7-5. 保存先のパーミッション

→ `reports/cowork_bets/{YYYYMMDD}/` は streamlit プロセスが書き込み可能であること。
   本番運用ではディレクトリ存在チェックを `mkdir(parents=True)` で行う。

### 7-6. Streamlit の保存ボタンで git push が失敗する

→ 保存自体は完了しているので慌てずに。エラーメッセージで原因を確認:
   - `Authentication failed`: GitHub credential helper が未設定。
     `git config --global credential.helper manager-core` 等で設定後、手動で
     `cd E:\PyCaLiAI; git push origin HEAD:master` を実行。
   - `Updates were rejected`: remote に未取得の commit あり。
     `git pull --rebase --autostash origin master` してから再 push。
   - 「差分なし or commit 済」と caption が出る: 既に同内容で push 済 / 別 commit に含まれた可能性。
     `git status` で確認。

### 7-7. weekly_cowork.ps1 が「no changes to commit」で終わる

→ 既に同じ bundle.json を生成 + push 済の場合は正常動作。
   再生成したい場合は `git rm --cached reports/cowork_input/{YYYYMMDD}_bundle.json`
   してから再実行。

---

## 8. 既知の制約

1. **オッズ取得タイミング**: 週次CSVの `単勝` 列 = 当日 5 分前スナップショット相当。
   リアルタイム反映なし (再生成すれば最新化)。
2. **複勝オッズ**: 週次CSVに含まれないため `null`。Cowork 側では `p_sho × 推定` を使う。
3. **`p_plc` キャリブレータ**: 現在 raw PL のみ。`p_win` / `p_sho` は適用済。
4. **`ai_market_agreement`**: 出走馬 3 頭以上にオッズが揃わないと `null`。
5. **戦略外レース** (東京・小倉・新馬・障害): Streamlit でも参考予想として表示。
   Cowork タブはこれらのレースでも入力可能 (PyCaLiAI フィルタとは独立)。

---

## 9. 今後の拡張候補 (未着手)

- [ ] `app.py` から `weekly_cowork.ps1` 相当を内部実行する「印 JSON 再生成」ボタン
- [ ] Cowork 出力テキストの自動 parse (Markdown テーブル / 行形式の自動認識)
- [ ] 複勝オッズの別ソース取得と週次 JSON への混入
- [ ] Cowork 提案と従来 4 戦略の **同一画面比較** (Streamlit に diff ビューを追加)
- [ ] レース後の `cowork_bets` 的中判定 → ROI 集計を `weekly_post.ps1` に組み込み
