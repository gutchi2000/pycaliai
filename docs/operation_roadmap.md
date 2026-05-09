# PyCaLiAI 運用ロードマップ (2026-05-09 更新版・NiceGUI 主軸)

NiceGUI 版 (`nicegui_app.py`) を **主 UI** として HuggingFace Spaces にデプロイする
構成に切り替え。Streamlit 版 (`app.py`) は **手元/ローカル運用の補助** として並行維持。

---

## 1. システム全体図 (NiceGUI 主軸)

```
[週次CSV  data/weekly/YYYYMMDD.csv]    ← TARGET エクスポート
                │
                ▼
        weekly_nicegui.ps1     ← NEW: pre 用一括スクリプト
                │
       ┌────────┼────────────┐
       │        │            │
       ▼        ▼            ▼
  hosei /   bundle.json   git push
  kako5    生成 (v5 印)   origin/master
  生成
       │        │            │
       └────────┴────────────┘
                │
                ▼
      sync-hf.ps1 ──→ HuggingFace Spaces (NiceGUI)
                          https://gutchi15300-pycaliai.hf.space
                │
                │ (任意/オプション)
                ▼
        Streamlit Cloud (app.py)  ← 補助 UI、必要に応じて
```

**運用パターン**
- **メイン**: NiceGUI on HF Spaces (PC / スマホ / 出先からアクセス)
- **サブ**: Streamlit ローカル (`streamlit run app.py`) — Cowork 連携入力やバックテスト確認

---

## 2. データファイル配置

### 毎週 土曜朝に置くもの (TARGET から取得)

| ファイル | 置き場所 | 用途 (NiceGUI で使う？) | 必須/任意 |
|---|---|---|---|
| 出走表CSV | `data/weekly/YYYYMMDD.csv` | ✅ a〜f 軸 (前走系特徴量) | **必須** |
| 着度数CSV | `data/tyaku/YYYYMMDD.csv` | (Streamlit のみ) | 任意 |
| 過去5走CSV | `data/kako5/YYYYMMDD.csv` | 将来拡張 (現状未使用) | 任意 |
| 坂路調教CSV | `data/training/H-YYYYMMDD-YYYYMMDD.csv` | ✅ g 軸 (調教) | **任意 (g 軸が欲しいなら必須)** |
| WC調教CSV | `data/training/W-YYYYMMDD-YYYYMMDD.csv` | ✅ g 軸 (調教) | **任意 (g 軸が欲しいなら必須)** |

### 毎週 日曜夜に置くもの

| ファイル | 置き場所 | 用途 |
|---|---|---|
| レース結果CSV | `data/kekka/YYYYMMDD.csv` | results.json 集計、NiceGUI 表示には未使用 |

---

## 3. 週次コマンド (NiceGUI 主軸)

### 【土曜朝】レース前 — `weekly_nicegui.ps1`

```powershell
.\weekly_nicegui.ps1                 # 当日自動判定
.\weekly_nicegui.ps1 20260502        # 日付指定
.\weekly_nicegui.ps1 20260502 -SkipHF  # HF push せず生成だけ
```

自動実行:
1. `make_weekly_hosei.py` → `data/hosei/H_YYYYMMDD.csv` (前走補正/前走補9)
2. `predict_weekly.py` → `reports/pred_YYYYMMDD.csv` (Streamlit 用、optional)
3. `export_weekly_marks.py --model v5` → `reports/cowork_input/{date}_bundle.json`
4. `git add` 関連ファイル → `git push origin HEAD:master`
5. `sync-hf.ps1` で HuggingFace Spaces (NiceGUI) に push

### 【日曜夜】レース後 — `weekly_post.ps1` + `sync-hf.ps1`

```powershell
.\weekly_post.ps1 20260502     # 既存 (results.json + git push)
.\sync-hf.ps1                  # HF にも反映 (任意)
```

NiceGUI は当面結果を表示しないので post 工程は最小。Streamlit 側で確認。

### Cowork 買い目連携 (オプション)

NiceGUI でも `🎫 Cowork 買い目` タブで cowork_output JSON を表示できる:

```powershell
# 1. bundle.json を Cowork (Anthropic Desktop) に投げる
# 2. Cowork が JSON で買い目を返す
# 3. その JSON を reports/cowork_output/{date}_bets.json として保存
# 4. .\sync-hf.ps1 で HF に反映 (NiceGUI 上で表示される)
```

---

## 4. データフロー図

```
【土曜朝】
TARGET
  ├── 出走表CSV    → data/weekly/YYYYMMDD.csv         必須
  ├── 着度数CSV    → data/tyaku/YYYYMMDD.csv          任意
  ├── 過去5走CSV   → data/kako5/YYYYMMDD.csv          任意
  ├── 坂路調教CSV  → data/training/H-*.csv            任意 (NiceGUI g 軸用)
  └── WC調教CSV    → data/training/W-*.csv            任意 (NiceGUI g 軸用)

.\weekly_nicegui.ps1
  ├─ make_weekly_hosei.py    → data/hosei/H_YYYYMMDD.csv
  ├─ predict_weekly.py       → reports/pred_YYYYMMDD.csv (Streamlit 用)
  ├─ export_weekly_marks.py  → reports/cowork_input/{date}_bundle.json
  ├─ git push origin master  → Streamlit Cloud
  └─ sync-hf.ps1             → HuggingFace Spaces (NiceGUI)

【日曜夜】
TARGET
  └── 成績CSV      → data/kekka/YYYYMMDD.csv          必須

.\weekly_post.ps1
  ├─ generate_results.py     → data/results.json
  ├─ build_pycali_history.py → data/pycali_history.parquet
  └─ git push origin master  → Streamlit Cloud
.\sync-hf.ps1 (任意)         → HuggingFace Spaces (NiceGUI)
```

---

## 5. NiceGUI 機能一覧 (v8.1, 2026-05-09 時点)

### UI 構造 (上から下に水平レイアウト)

| 行 | 内容 | 備考 |
|---|---|---|
| 1 行目 | 📅 開催日 select | dropdown |
| 2 行目 | 場所タブ (東京 / 京都 / 新潟 …) | 選択中ハイライト (青塗り) |
| 3 行目 | レース番号ボタン (1R 〜 12R) | 選択中ハイライト |
| メイン | 左パネル + 右パネル + タブ群 | 全幅、サイドバーなし |

### メインパネル

**左パネル** (印 + AI 評価 + オッズ概況)
- レース名 / クラス / レース性質バッジ (固い/混戦/穴推奨/見送り)
- ◎ ◯ ▲ の馬名・勝率・単勝オッズ
- 🤖 AI 評価コメント (本命の独走度、馬連本線、混戦判定 等)
- 🏁 オッズ概況 (1番人気/最高/平均/上位3頭合計勝率)

**右パネル** (4 chip + 過去成績 + 馬場バイアス)
- ◎独走度 / 上位2頭集中 / 混戦度 / 市場一致 の 4 chip
  状態バッジ (固い/混戦/独走 等) + 「→ 何をすべきか」 1 行アクション
- 📊 過去同コース成績 (master_v2 がある場合のみ、HF にはなし)
- 🎯 枠バイアス (内枠/中枠/外枠の勝率)

### タブ

| タブ | 内容 |
|---|---|
| 📋 出走表 | 馬番/印/馬名/勝率/複勝率/単勝/単勝EV/複勝/vs市場 |
| 🔍 全頭分析 | AI vs 市場 散布図 + **PyCaLi 出走馬評価リスト** (a〜g 7 軸レーダー、Streamlit 互換) |
| 🎫 Cowork 買い目 | reports/cowork_output からその race の買い目を表示 |

### PyCaLi 評価リスト (Streamlit 互換 a〜g 7 軸)

| 軸 | ラベル | データソース | 候補列 |
|---|---|---|---|
| a | 総合力 | bundle.json | p_win |
| b | スピード | weekly + hosei | 前走補正 / 前走Ave-3F / 前走走破タイム |
| c | 末脚 | weekly + hosei | 前走補9 / 前走上り3F |
| d | 前走成績 | weekly | 前走確定着順 |
| e | 市場評価 | weekly | 前走人気 / 前走単勝オッズ |
| f | ペース適性 | weekly | 前走RPCI / 前走PCI3 / 前走Ave-3F |
| g | 調教 | training | trn_hanro_lap1 / trn_hanro_time1 / trn_wc_3f / trn_wc_5f |

**PyCaLi 指数** = `p_sho × 100 + (補正特徴量 b〜f 平均 - 5) × 0.5` (0-100 にクランプ)
表示順位は `ai_rank` (印 順) を採用 → **「1位 = ◎」が常に成立**。
PyCaLi 値が ◎ < ◯ となるレアケースは「補正特徴量では◯が前走実績で勝るが、AI モデルは
今走条件込みで◎」というシグナルとして読む (バグではない)。

---

## 6. デプロイ系コマンド

| コマンド | 役割 |
|---------|------|
| `.\weekly_nicegui.ps1 [DATE]` | 週次データ生成 + GitHub push + HF Spaces push (NEW) |
| `.\weekly_pre.ps1 [DATE]` | 旧: 予測 + GitHub push のみ (Streamlit 単独運用時) |
| `.\weekly_cowork.ps1 [DATE]` | 旧: bundle.json 生成 + GitHub push (HF push なし) |
| `.\weekly_post.ps1 [DATE]` | レース結果取得 + 集計 + GitHub push |
| `.\sync-hf.ps1` | master の最新を hf-spaces orphan branch に同期 + HF push |
| `.\sync-hf.ps1 -DryRun` | 差分確認のみ (commit/push しない) |
| `streamlit run app.py` | Streamlit ローカル起動 (補助 UI) |
| `python nicegui_app.py` | NiceGUI ローカル起動 (port 8080) |

---

## 7. ファイル配置 (NiceGUI 関連)

```
E:\PyCaLiAI\
├─ nicegui_app.py                ... NiceGUI 主アプリ (HF/ローカル共通)
├─ Dockerfile                    ... HF Spaces ビルド用
├─ requirements-nicegui.txt      ... NiceGUI 用最小依存
├─ .dockerignore                 ... HF ビルド除外 (大物 CSV 等)
├─ README.md                     ... HF Spaces YAML フロントマター入り
│
├─ weekly_nicegui.ps1            ... NEW: 週次 pre 一括 (NiceGUI 主)
├─ sync-hf.ps1                   ... NEW: master → hf-spaces orphan + HF push
│
├─ weekly_pre.ps1                ... 旧 (LGBM v1 予測のみ)
├─ weekly_cowork.ps1             ... 旧 (bundle 生成 + GitHub push のみ)
├─ weekly_post.ps1               ... 結果集計 (現役)
│
├─ data/
│   ├─ weekly/{YYYYMMDD}.csv         ... 週次出走表 (HF 必須)
│   ├─ hosei/H_{YYYYMMDD}.csv        ... 補正タイム (HF 必須、~600KB/週)
│   ├─ training/H-*.csv W-*.csv      ... 調教 (HF g 軸用、2026 系のみ ~7MB)
│   ├─ kako5/{YYYYMMDD}.csv          ... 過去5走 (HF 任意)
│   └─ master_v2_*.csv               ... 過去成績 390MB (HF 除外、ローカル限定)
│
├─ reports/
│   ├─ cowork_input/{date}_bundle.json    ... HF 必須
│   ├─ cowork_output/*.json               ... 買い目 (HF 任意)
│   └─ pred_{date}.csv                    ... Streamlit 用
│
└─ docs/
    ├─ operation_roadmap.md      ... THIS FILE
    ├─ marks_schema.md
    └─ ...
```

---

## 8. HuggingFace Spaces 構成

### URL

- Space: https://huggingface.co/spaces/gutchi15300/pycaliAI
- 直接: https://gutchi15300-pycaliai.hf.space

### ブランチ運用

| ブランチ | 用途 | 内容 |
|---------|------|------|
| `master` | GitHub 開発本流 | 全コード + 全データ |
| `hf-spaces` | HF Spaces orphan | NiceGUI 関連だけの最小構成 (大物データ除外) |

`hf-spaces` は orphan branch で master とは履歴独立。同期は `sync-hf.ps1` のみで行う。

### sync-hf.ps1 の動き

1. master ブランチで実行
2. hf-spaces に切替
3. master から指定ファイル群を checkout:
   - `Dockerfile` `README.md` `.dockerignore` `requirements-nicegui.txt` `nicegui_app.py`
   - `data/weekly/*.csv` (全週)
   - `data/hosei/H_2026*.csv` (大物 H_2013-2025 は除外)
   - `data/training/[HW]-2026*.csv` (大物 2015 系は除外)
   - `data/kako5/*.csv`
   - `reports/cowork_input/*.json`
   - `reports/cowork_output/*`
4. `git commit` → `git push hf hf-spaces:main`
5. master に戻る

---

## 9. NiceGUI v5〜v8.1 主要変更履歴

| バージョン | 日付 | 変更点 |
|---|---|---|
| v8.1 | 2026-05-09 | PyCaLi 指数の逆転現象緩和 (p_sho × 100 base、boost ×0.5、ai_rank 順) + sync-hf.ps1 が data subdir を一括同期 |
| v8 | 2026-05-09 | PyCaLi 評価軸を Streamlit と同じ a〜g 7 軸に統一 (weekly + hosei + training を merge) |
| v7 | 2026-05-08 | サイドバー廃止、場所タブ + レース番号ボタン UI |
| v6 | 2026-05-07 | 4 chip 素人向け刷新 (状態バッジ + アクション)、PyCaLi 評価リスト初版 (独自 6 軸) |
| v5 | 2026-05-06 | AI 評価を印の下に移動、過去同コース成績 + 枠バイアス追加 |
| v4 以前 | 2026-05-05 | MVP (Streamlit 互換 banner、HTML テーブル、左右分割) |

---

## 10. 既知の制約

1. **g 軸 (調教) は HF では 2026 系のみ**: 2015-2025 の H-20150401-20260313.csv (364MB) は
   .dockerignore で除外。よって 2025 年以前の馬で初出走の場合は g が「−」表示。
2. **過去成績 / 枠バイアス**: master_v2_*.csv (390MB) は HF 除外。HF では「データ無し」表示
   になり、ローカル (`python nicegui_app.py`) でのみ表示。
3. **オッズ取得タイミング**: 週次CSVの `単勝` 列は当日 5 分前スナップショット相当。
   リアルタイム反映なし。
4. **複勝オッズ**: 週次CSVに含まれないため null。bundle.json に推定値を入れている。
5. **PyCaLi history (スパークライン)**: NiceGUI 版では未実装。Streamlit 版は実装済。
6. **戦略外レース** (新馬・障害): NiceGUI でも参考表示。フィルタは適用しない。

---

## 11. トラブルシューティング (NiceGUI / HF 関連)

### 11-1. HF Spaces で "Stopped" / "Error"

```
NiceGUI ready to go on http://localhost:8080
```

→ HF は port 7860 を期待。Dockerfile の `ENV PORT=7860` が反映されているか確認。
   `git -C E:\PyCaLiAI show hf-spaces:Dockerfile | grep PORT` で確認。

### 11-2. HF push で YAML エラー

```
"colorTo" must be one of [red, yellow, green, blue, indigo, purple, pink, gray]
```

→ `README.md` の YAML フロントマターの `colorTo` を許可値に修正。

### 11-3. HF Spaces で a〜g の値がすべて 5.0 (valid=False)

→ `data/weekly/{date}.csv` か `data/hosei/H_{date}.csv` が hf-spaces ブランチに存在しない。
   `.\sync-hf.ps1` を再実行して同期。

### 11-4. HF Spaces で g 軸だけ「−」

→ `data/training/[HW]-{date 周辺}.csv` が hf-spaces にない。
   `.\sync-hf.ps1` を実行 (master の 2026 系 training ファイルが自動同期される)。

### 11-5. `sync-hf.ps1` が「checkout hf-spaces failed」

→ `models/transformer_optuna_v1.pkl` の壊れた LFS pointer が原因のことが多い。
   スクリプトは `git checkout --force hf-spaces` で対応済。
   それでも失敗するなら手動: `git stash && git checkout hf-spaces`。

### 11-6. `weekly_cowork.ps1` で印 JSON が 0 races

```
[ERROR] rid=...: "['Ｒ', '芝(内・外)', ...] not in index"
```

→ v5 feature_cols に対する不足列。`export_weekly_marks.py` 内で NaN 補完済み。
   それでも続く場合は `bundle["feature_cols"]` を確認。

### 11-7. 逆転現象 (PyCaLi 1位 ≠ ◎)

→ 表示順位は ai_rank なので 1位 ラベルは常に ◎。値が ◎ < ◯ になるのは仕様
   (補正特徴量シグナル)。詳細は §5 末尾参照。

---

## 12. 今後の拡張候補

- [ ] NiceGUI に PyCaLi history (スパークライン) を移植
- [ ] NiceGUI に EV 単勝候補 / VALUE 複勝タブを追加
- [ ] NiceGUI 上で Cowork 買い目を **編集** (現状は表示のみ)
- [ ] HF Spaces で master_v2 の軽量化版 (場所×コース統計だけ) を別ファイルとして配置
  → 過去成績/枠バイアスを HF でも表示できるように
- [ ] `weekly_nicegui.ps1` に post 用モード追加 (`-Post` スイッチで weekly_post + sync-hf を 1 発)
- [ ] HF Spaces 起動時間短縮 (現状 master_v2 ロード ~5-10 秒、HF にはないのでスキップ)
