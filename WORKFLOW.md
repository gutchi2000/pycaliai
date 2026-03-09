# 週次運用チートシート

---

## 前半：レース前（金〜土朝）

### やること
出走表 CSV を `data/weekly/` に置いて push → Streamlit Cloud に自動反映

```bash
# 1. weeklyCSVを所定フォルダに保存
#    例: 3/8開催分 → E:\PyCaLiAI\data\weekly\20260308.csv

# 2. push
cd E:\PyCaLiAI
git add data/weekly/20260308.csv
git commit -m "add weekly csv 20260308"
git push

# → 1〜2分でStreamlit Cloudに反映
```

> **確認**: Streamlit Cloud の「買い目」ページを開き、当該日付のレースが表示されれば OK

---

## 後半：レース後（日夜〜月）

### やること
払戻 CSV を `data/kekka/` に置いて `generate_results.py` → push → 的中実績に反映

```bash
# 1. kekkaCSVを所定フォルダに保存
#    例: 3/8開催分 → E:\PyCaLiAI\data\kekka\20260308.csv

# 2. results.json を再生成
cd E:\PyCaLiAI
python generate_results.py

# 3. まとめてpush
git add data/kekka/20260308.csv data/results.json
git commit -m "add kekka 20260308"
git push

# → 1〜2分で的中実績ページに反映
```

> **確認**: Streamlit Cloud の「的中実績」ページで HAHO/HALO のレース数が増えれば OK

---

## ファイル名規則

| フォルダ | ファイル名 | 中身 |
|---|---|---|
| `data/weekly/` | `YYYYMMDD.csv` | 出走表（JRA出力CSV） |
| `data/kekka/` | `YYYYMMDD.csv` | 払戻結果（JRA出力CSV） |
| `reports/pred_*.csv` | `pred_YYYYMMDD.csv` | 予測結果（自動生成） |

---

## トラブルシューティング

| 症状 | 原因 | 対処 |
|---|---|---|
| 買い目ページにレースが出ない | weekly CSV がない or push 忘れ | `git push` 済みか確認 |
| 的中実績が増えない | results.json が古い | `python generate_results.py` 再実行 → push |
| スコアが全馬 0% | モデルファイル不整合 | `models/` 以下の pkl が揃っているか確認 |
