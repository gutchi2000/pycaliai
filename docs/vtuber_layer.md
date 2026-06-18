# PyCaLiAI VTuber 解説レイヤー設計

PyCaLiAI の週次マーク出力を **「ずんだもん × 四国めたん」掛け合い台本**に変換し、
YouTube Shorts（縦9:16・約45秒）として投稿するための薄い追加層。

予測システムには一切手を入れない。**既存の週次成果物を "喋らせる" だけ**。

---

## なぜこれが強いのか（コンセプト）

世の中の競馬予想チャンネルはほぼ「雰囲気・印象」で語る。PyCaLiAI の出力は
**キャリブレーション済み確率**（勝率22%は長期的に本当に約22%）なので、他チャンネルが
言えないことを誠実に言える：

1. **期待値・妙味を提示できる** — `ai_vs_market`（under=妙味/over=罠）、`value_horses[].ev_tan`
2. **信頼度を数字で語れる** — `class_prior.hon_top3_pct`（このクラスの◎は複勝圏◯%）、`race_confidence`
3. **誠実な答え合わせ** — 当たり外れ・ROI を全部公開する文化（既にある）

→ 売りは「絶対当たる」ではなく **「キャリブレーション済みAIが算出した"妙味"を淡々と提示」**。

---

## パイプライン全体

```
[既存] export_weekly_marks.py
        └─→ reports/cowork_input/{date}_bundle.json   (印・確率・妙味・信頼度)
[既存] Cowork (Claude Desktop)
        └─→ reports/cowork_output/{date}_bets.json    (advisor コメント・重賞コラム)
                       │
                       ▼
[新規] generate_vtuber_script.py   ← このPR
        └─→ vtuber_scripts/{date}/{race_id}_daihon.txt   (YMM4貼り付け用)
        └─→ vtuber_scripts/{date}/{race_id}_script.json  (テロップ・meta付き)
                       │
                       ▼  (半自動: ここから先は手作業 or 後日自動化)
   VOICEVOX (ずんだもん/四国めたん) で音声生成
                       │
                       ▼
   ゆっくりMovieMaker4 で 口パク+字幕+BGM、9:16で書き出し
                       │
                       ▼
   YouTube に Shorts として投稿
```

---

## 入力コントラクト（このスクリプトが消費するフィールド）

すべて `docs/marks_schema.md` 準拠。**実フィールドだけを言い換え、数値は捏造しない**。

| 用途 | フィールド | 出所 |
|------|-----------|------|
| 本命提示 | `horses[].mark`(◎〇▲△), `horse_name`, `umaban` | bundle |
| 信頼度（勝率） | `horses[].p_win`（キャリブレーション済） | bundle |
| クラス実績 | `race_meta.class_prior.hon_top3_pct` 等 | bundle |
| 妙味 | `buy_judgment.value_horses[].ev_tan`/`umami_grade`, `horses[].ai_vs_market` | bundle |
| 罠（人気崩壊候補） | `ai_vs_market == "over"` かつ単勝オッズ小 | bundle |
| レース難度 | `buy_judgment.hardness`(固い/標準/荒れ), `race_confidence` | bundle |
| 買い方の方向性 | `buy_judgment.headline` | bundle |
| 予想士口調コメント | `bets[].advisor[].comment`/`grade`/`tag` | bets（任意） |
| 重賞コラム | `grade_scope[].markdown` | bets（任意・概要欄/ロング動画用に同梱） |

> bets.json があれば advisor コメントを本命の "根拠" として優先採用する
> （既に競馬予想士口調で書かれた "ほぼ完成原稿" のため）。

---

## 使い方

```bash
# 候補レース一覧（面白さ順）
python generate_vtuber_script.py --date 20260614 --list

# おすすめ1本を自動生成（重賞 > 妙味 > 鉄板の優先順）
python generate_vtuber_script.py --date 20260614 --mode auto

# レース指定
python generate_vtuber_script.py --date 20260614 --race-id 2026061409030411

# モード: auto / grade(重賞優先) / value(妙味優先) / solid(鉄板) / chaos(波乱)
```

依存は標準ライブラリのみ（json/argparse/unicodedata…）。追加 pip 不要。

---

## 週次パイプラインへの組み込み（推奨フック点）

`weekly_nicegui.ps1` の **Phase B（`-BetsOnly`）**、`validate_cowork_bets.py --apply` が
通過した直後（git/HF push の前）。この瞬間だけ bundle（印・確率・妙味）と
bets（解説コメント・重賞コラム）が同じ日付で揃う。

```powershell
# weekly_nicegui.ps1 Phase B, "見送りガード通過" の直後あたり
python generate_vtuber_script.py --date $Date --mode grade   # 重賞を1本
# 複数本まとめたい場合は --list の race_id をループ
```

**副フック（Phase A）**: `export_weekly_marks.py` 直後でも生成可能。ただし bets が無いので
「AI印だけの事前予想ティザー」用途（解説コメントは付かない）。

---

## ペルソナ設計

既存の `.claude/agents`（analyst=解説役 / critic=懐疑役 / chair=司会）の討論構造を
2人に圧縮する：

- **ずんだもん（本命・妙味を語る予想役）** … analyst + chair。語尾「〜なのだ」。
  キャリブレーション済み勝率を誠実に提示し、妙味馬を推す。
- **四国めたん（懐疑役・締め）** … critic。標準語。「荒れそう」「本当に？」とツッコみ、
  最後に「結果は全部公開」と登録誘導。

VOICEVOX 公式キャラ＝音声・立ち絵とも無料。YMM4 にそのまま読み込める。

---

## 責任あるギャンブル方針（既存規約を継承）

PyCaLiAI は元々 EV-honest。VTuber でも厳守する：

- 「絶対当たる/必ず儲かる」は禁止。EV・妙味は**推定値であって保証ではない**
  （JRA控除率20-25%、すべての馬券はEV -20%から始まる）。
- **見送りも正解**として尊重（`見送りを恐れない`）。
- 断定的な購入指示はしない。「AIの結論は◯◯／買うかは自己責任」に留める。
- 常時テロップ：`※キャリブレーション済みAIによるエンタメ予想｜馬券は自己責任・20歳以上`
- 実績（当たり外れ・ROI）は隠さず公開する。

---

## リスク・要確認事項

- **デスクトップ前提**: 週次 PS1 は `E:\PyCaLiAI` / `venv311` ハードコード。本番デスクトップで実行する。
- **数値の扱い**: このチャンネルの差別化として勝率%は「キャリブレーション済み」と添えて提示してよいが、
  EV/妙味は「推定」と明示する（保証表現は規約違反）。
- **著作権**: JRA中継映像・netkeiba画面は使わない。数字は自作テロップに起こす。
- **VOICEVOX/立ち絵のクレジット**を概要欄に明記。
- **オッズのタイミング**: bundle の `tansho_odds` は発売直前スナップショット。T-10 ライブ値を
  使う場合は `compute_bets.py --live-odds-dir` 経由で更新された値を使う。

## 今後（フル自動化の余地）

- VOICEVOX ENGINE の HTTP API（`/audio_query`→`/synthesis`）で音声 wav を自動生成。
- 立ち絵口パク＋字幕焼き込みを ffmpeg/YMM4 CLI で自動化 → 投稿だけ手動の「ほぼ全自動」へ。
