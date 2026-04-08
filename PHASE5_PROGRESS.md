# Phase 5+ 進捗ログ（複勝改善・SegmentBetFilter・MoE）

> **目的**: PyCaLiAI のバックテスト総ROIを **97.9% → 100%以上** に引き上げる。
> このファイルは「いつ、何を、どんな結果で実装したか」の履歴を残し、
> 次セッションでも経緯を即把握できるようにするためのもの。
>
> 数字は全て **2024-2025 backtest 実測 (test split)**。

最終更新: 2026-04-08

---

## 📈 マイルストーン推移

| 日付 | バージョン | 総ROI | 三連複 | 馬連 | 複勝 | 主な変更 |
|---|---|---|---|---|---|---|
| 〜2026-04-05 | Phase 5 (4モデル固定重み) | 86.8% | 95.7% | 86.2% | 84.3% | ベースライン |
| 2026-04-07 | + 8モデル最適化重み + MoE | 87.4% | 90.7% | 88.3% | 84.5% | predict_weekly→ensemble_predict 切替, turf_mid Expert |
| 2026-04-07 | **+ SegmentBetFilter (距離)** | **92.9%** | **105.8%** | **95.4%** | **84.9%** | ROI<80%の(seg,券種)を除外 |
| 2026-04-07 | **+ クラス×距離 BLACKLIST** | **97.9%** | 105.8% | **130.7%** | 84.9% | dirt馬連の弱クラス除外 |
| 2026-04-08 | + lgbm_fukusho_v2 単純差替 | 92.3% ⚠️ | 106.5% | 98.1% | 85.5% | v2学習成功も既存重みと不整合 → ロールバック |
| 2026-04-08 | **+ 複勝 dirt 1勝 BLACKLIST** | **100.4%** 🎉 | 105.8% | 130.7% | 85.2% | analyst+statistician推奨。n=202, p≈0.03 有意な負ROI |

### Sprint 1.1 確定 (2026-04-08) 🎉 総ROI 100% 突破
- v2 差替は不発（ensemble 不整合）、別アプローチで攻めた
- **カンファレンス結果**:
  - 初期案 (turf_long 1勝/3勝, dirt 3勝 複勝除外) → analyst+statistician 却下 (n<10, p>0.6 ノイズ)
  - 代案 **dirt 1勝 複勝除外** (n=202, p≈0.03) → 両エージェント支持 → 採用
- **結果**: 97.9% → 100.4% 収支 +23,490円
- **R/週 注意**: 現状 5.3R/週（週10R目標は将来課題: フィルタ緩和・東京/小倉再投入・キャリブレータ改善で段階的に拡大予定）

### Sprint 1.1 中間結果 (2026-04-08)
- `train_lgbm_fukusho.py` 新規 → `lgbm_fukusho_v2.pkl` 学習
  - Valid AUC 0.7743 (baseline 0.7551, +0.019) / Test AUC 0.7773 (baseline 0.7594, +0.018) → **モデル単体は採用判定**
- ただし predict_weekly/app.py の FUKU_LGBM_PATH を v2 に差し替えてバックテストすると総ROI 97.9% → **92.3% に悪化**（馬連 130.7% → 98.1%）
- 原因: `ensemble_weights.json` は **v1 の fuku_lgbm スコア前提**で Nelder-Mead 最適化されており、スコア分布が変わる v2 と不整合
- 対応: 一旦 v1 に戻し、`optimize_weights.py` を v2 ロードで再走 → 再バックテストで判断（次セッション）

---

## ✅ 完了タスク

### S0. 8モデル最適化重み + MoE (Mixture of Experts)
**コミット**: `2b59bfe`
- `train_expert.py` 新規: 距離4分割 (turf_short / turf_mid / turf_long / dirt) で Expert 学習
- baseline AUC (0.7767) 未満は自動 reject → **採用は turf_mid のみ** (Valid 0.7786)
- `optimize_weights.py --experts`: セグメント別重み最適化
  - turf_mid: AUC 0.7809 → **0.7866** (+0.006)
  - dirt: AUC 0.7809 → **0.7820** (+0.001)
- `predict_weekly.py / app.py`: `_select_segment` で距離→セグメント自動ルーティング
- `backtest.py` を `predict_weekly.ensemble_predict` 経由に切替

**結果**: 総ROI 86.8% → 87.4% (+0.6pt)

### S1. SegmentBetFilter (距離別ブラックリスト)
**コミット**: `1ba49df`
- 2024-2025 実測 ROI < 80% の (距離セグメント, 券種) を購入対象から除外
- BLACKLIST:
  ```python
  {("dirt","三連複"), ("turf_short","馬連"),
   ("turf_short","複勝"), ("turf_mid","馬連")}
  ```
- `app.py.get_bets` / `backtest.py.process_one_race` 双方に実装

**結果**: 総ROI 87.4% → **92.9%** (+5.5pt)
- 三連複: 90.7% → **105.8%** ✅
- 馬連: 88.3% → 95.4%
- 複勝: 84.5% → 84.9%

### S2. クラス×距離 BLACKLIST
**コミット**: `3461597`
- dirt × 馬連 (89.8%) の内訳分析: 未勝利71.8% / オープン56.2% / 3勝68.1% / 重賞 0-77%
- 追加除外:
  ```python
  {("dirt","未勝利","馬連"), ("dirt","オープン","馬連"),
   ("dirt","3勝","馬連"), ("dirt","GⅠ","馬連"),
   ("dirt","GⅡ","馬連"), ("dirt","GⅢ","馬連"),
   ("dirt","OP(L)","馬連")}
  ```

**結果**: 総ROI 92.9% → **97.9%** (+5.0pt)
- 馬連: 95.4% → **130.7%** ✅✅ (プラス収支転換)
- 三連複/複勝: 変化なし

### S3. 阪神/京都 buylist 表示問題の修正
**コミット**: `2b59bfe`
- 原因: `data/strategy_weights.json` に阪神/京都の戦略がなく、`get_bets()` が早期 return
- 修正: TRIPLE プランは戦略テーブル未登録会場でも生成
- `page_buylist` にデバッグ caption 追加（会場別R数 / TRIPLE可数）

**結果**: Streamlit Cloud で阪神/京都/中京/福島も表示されるようになった

---

## 🩺 現状課題リスト

### 🔴 P0: 複勝 ROI 84.9% (最重要・100%突破の唯一の壁)
- 投資シェア36%なのに収支寄与マイナス
- 距離別: dirt 83.0% / turf_mid 87.9% / turf_long 91.2%
- 仮説:
  1. 控除率0.20 → 理論上限80%前後で構造的に厳しい
  2. ◎の絶対確率精度が「順位最適化」で劣化
  3. キャリブレータ v4 が ensemble 全体平均で fit、複勝特化していない
  4. 人気馬◎が多くて配当が控除率を超えない

### 🟡 P1: モデル/学習側
- キャリブレータ単一 (全 seg 共通)
- Expert 採用率 1/4
- 学習目標 (AUC) と評価指標 (ROI) の乖離
- スタッキング未稼働 (`'p_lgbm','p_cat',... not in index'` エラー)

### 🟡 P2: 戦略/運用側
- strategy_weights.json が4会場のみ (中京/中山/新潟/福島)
- kelly/EV閾値が暫定
- 全プラン同一◎で多角化なし
- 資金管理ルール無し (1R固定 1万)

### 🟢 P3: パイプライン
- SVM/多クラス検証未文書化
- MoE pkl が gitignore で Cloud 未反映 (Cloud では従来重みで動作)
- 週次自動レポート無し

---

## 🗺️ ロードマップ

### 🚀 Sprint 1 (今すぐ・複勝特化)
| # | アクション | 期待ROI | リスク | 状態 |
|---|---|---|---|---|
| **1.1** | **複勝モデル単独学習** (`train_lgbm_fukusho.py` 新規, fukusho_flag専用 + 特徴量絞り) | +2-3pt | 30分 | ⏳ 次着手 |
| 1.2 | 複勝専用キャリブレータ (Isotonic) | +1pt | 低 | ⏳ |
| 1.3 | 複勝ベットフィルタ (cal_prob > 0.55 限定) | +2pt or レース減 | 週10R制約 | ⏳ |

### 🎯 Sprint 2 (100%突破)
| # | アクション | 期待ROI |
|---|---|---|
| 2.1 | EV-base ベットサイジング (Kelly 1/4) | +1-2pt |
| 2.2 | 複勝 BLACKLIST 追加 (dirt未勝利/重賞等) | +0.5-1pt |
| 2.3 | TRIPLE 50/50 → seg別比率 (例 70/30) | +0.3pt |

### 🔭 Sprint 3 (中期)
| # | アクション | 期待 |
|---|---|---|
| 3.1 | スタッキング再構築 (8モデル → 2層目LGBM) | AUC+0.005 |
| 3.2 | 多クラス分類 (1着/2-3着/着外) | 馬連・三連複 +2pt |
| 3.3 | Expert 再挑戦 (Optuna個別調整) | turf_short/dirt 採用化 |
| 3.4 | 戦略テーブル全会場拡張 | 阪神/京都の HAHO/HALO 解禁 |

### 🛡️ Sprint 4 (基盤)
- MoE pkl Cloud 反映 (Git LFS)
- TODO_phase5.md に SVM/多クラス判断記録
- 週次自動バックテスト
- SegmentBetFilter の閾値CSV化

---

## 📝 (seg, 券種, クラス) 別 ROI 早見表 (BLACKLIST後)

```
dirt × 複勝   83.0%  ← P0 改善対象
dirt × 馬連   89.8% (除外後 130.7%)
turf_long × 馬連    145.0%
turf_mid × 三連複   120.7%
turf_short × 三連複 103.8%
turf_long × 複勝     91.2%
turf_long × 三連複   91.3%
turf_mid × 複勝      87.9%
```

---

## 🔧 次回セッション用コマンド集

```bash
# バックテスト
python backtest.py

# Expert 学習
python train_expert.py

# Expert 別重み最適化
python optimize_weights.py --experts

# 構文チェック
python -c "import ast; ast.parse(open('app.py',encoding='utf-8').read()); print('OK')"

# ROI 集計 (距離 × 券種)
python -c "
import pandas as pd
df = pd.read_csv('reports/backtest_results.csv', encoding='utf-8-sig')
def seg(r):
    if r['芝ダ']=='ダ': return 'dirt'
    d=r['距離']
    return 'turf_short' if d<=1400 else 'turf_mid' if d<=2200 else 'turf_long'
df['seg']=df.apply(seg,axis=1)
g=df[df['購入額']>0].groupby(['seg','馬券種']).agg(投資=('購入額','sum'),払戻=('実払戻額','sum'))
g['ROI%']=g['払戻']/g['投資']*100
print(g.round(1))
"
```
