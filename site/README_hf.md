---
title: PyCaLiAI
emoji: 🏇
colorFrom: red
colorTo: yellow
sdk: static
pinned: false
license: mit
short_description: AI 競馬予想 (静的版・予想/成績ビューア)
---

# 🏇 PyCaLiAI — AI 競馬予想（静的版）

LightGBM v6 LambdaRank モデルで JRA 中央競馬の印付け（◎〇▲△）と Plackett-Luce
確率を算出し、Anthropic Cowork で馬券構築する個人運用システムの **表示専用** サイト。
推論済みデータ（`data/*.json`）を読んで可視化するだけの静的サイトです。

## 見られるもの

**予想**
- 📋 出走表（AI印 / 勝率 / 複勝圏 / 単勝EV / 市場乖離 / 近5走スパークライン）
- 🏆 重賞 Grade Scope（G1/G2/G3 の LLM 詳細見解）
- 🔍 全頭分析（AI評価×人気 散布図 / 能力レーダー / 🍣 全頭 UMAMI テーブル）
- 📊 コース（枠別・脚質別・年齢別・性別 複勝率）
- ⏱ 調教（坂路 / ウッド 最終追い切り + 好調教 Best5）
- 🧬 血統（父・母父のコース別複勝率ランク）

**成績**
- 累計 ROI / 的中率 / 券種別収支
- 🎯 的中一覧（クリックで買い目 + 全配当表）

## データ更新

`python build_site.py` で `site/data/*.json` を再生成し、`sync-hf-static.ps1`
で本 Space に push される（週次運用フローに組込み）。

⚠️ AI 予測は参考情報です。馬券の購入判断はご自身の責任で。
