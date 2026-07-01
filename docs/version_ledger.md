# PyCaLiAI モデル版 採否台帳（version_ledger）

> **位置づけ**: `unified_rank` / `optuna_v*_marks` の版（v5〜v10）について、
> 「目的・差分・採否・根拠 commit/report・ROI/ECE 実測・状態」を 1 表で管理する。
> OBJ-07（版乱立）・OBJ-04（採用ゲート未達のまま本番化）・contradictions の解消を目的とする。
>
> 出典: `reports/optuna_v*_marks.json`、`reports/audit_v*_vs_*.md`、各 `optuna_v*_marks.py` docstring、git log。
> 不明な点は「不明」と明記する。
>
> 最終更新: 2026-06-15（2026-06-15 全領域監査に基づき初版作成）

---

## ⚠️ 最重要: ガバナンス矛盾（OBJ-04）

**現本番は v6。だが v6 は自分で定めた採用ゲートを満たしていない。**

- 採用基準（`run_v6_pipeline.py:13-14` / `scripts/audit_v6_vs_v5.py:278`）:
  「単勝高EV ROI が v5 比 **+0.05 以上**、または複勝 **+0.03 以上**」
- 実測（`reports/audit_v6_vs_v5_20260520.md:45-52`）: 単勝 **-0.006**・複勝 **+0.002**
- レポート自身の結論: 「**## 結論: ❌ v6 採用見送り** … 大幅な改善なし。**v5 維持**。」
- にもかかわらず `export_weekly_marks.py:147` は `default="v6"`、CLAUDE.md は「v6 本番投入」、
  git log は毎週 `model=v6`。

**= 定量ゲートの出力と実運用が正面から矛盾している。** v6 が本番に入った後付け理由は ECE 改善だが、
OBJ-03 で alpha 低下と交絡し純効果不明。実弾 ROI 上の v6 化リターンは未確認（CLAUDE.md:92「v5 と同等 ±0.01」）。

**推奨**: v5/v6/v10 を test=2025 封印で同一パイプライン再走し本番を一度確定。採否ゲートを ROI 単独でなく
「CLV + 高EV帯 ROI + 多ビン ECE」の複合スコアに正式化し、`run_v*_pipeline` が exit code で pass/fail を返す形に。
「基準を満たさないのに採用」を文書で黙認しない（基準改訂 or 差し替え）。

---

## 採否台帳（v5〜v10）

> composite の比較注意:
> - v5/v6/v7/v8/v9 の `best_composite` は **valid=2023** 単一ホールドアウト上の値。
> - v7/v8/v9 は目的関数・特徴量が v6 と異なるため composite の絶対値は **v6 と直接比較不可**。
> - v10 の `best_composite_mean`（0.5701）は **valid=2024** 上の mean−std 保守選択値で、**他版と同一土俵ではない**。
> - したがって採否は composite ではなく **test OOS の audit（ROI/ECE）** で判断する（VAL-02/VAL-03 参照）。

| 版 | 目的 / 主な差分 | 採否 | 根拠（commit / report） | alpha / composite / ECE 実測 | 状態 |
|---|---|---|---|---|---|
| **v5** | 旧本番。LGBM LambdaRank。sample_weight `1+alpha·log1p(payout/100)` で高配当（穴）を重み付け。composite=純ランキング（NDCG@5/◎top3/…） | 🗄️ **退役**（rollback 用に保持） | `reports/optuna_v5_marks.json`、CLAUDE.md「v5 退役」 | alpha=**1.325** / composite=**0.5541** / 高EV: 単勝 0.737・複勝 0.913（`audit_v6_vs_v5_20260520.md:45-46`） | 保持。alpha=1.325 で本命 1.98〜大穴 8.03 と 8 倍差→tail calibration 崩壊（`optuna_v6_marks.py:8-10` の反省点） |
| **v6** | 現本番（建前）。v5 + 目的関数に ECE_high_p penalty（composite − 0.5×ECE）。alpha を Optuna が near-zero へ収束 | ✅ **本番**（`export_weekly_marks.py:147` default=v6）／ ❌ **採用ゲートは未達** | `reports/optuna_v6_marks.json`、`audit_v6_vs_v5_20260520.md:48-52`（❌見送り）、commit `5d7cd9d005`「marginal improvement, not adoption-worthy」 | alpha=**0.031** / composite=**0.5490** / ece_high_p=**0.0112** / 高EV: 単勝 0.731(-0.006)・複勝 0.915(+0.002) | **ガバナンス矛盾（OBJ-04）**。ECE penalty は寄与約 1% で実質空振り（OBJ-03）。ECE_high_p 自体が単一ビン平均バイアスの欠陥版（OBJ-02） |
| **v7** | v6 + ワイド ROI を目的関数に組込（`composite_v6 + W_WIDE_BONUS×(wide_roi_3box−1.0) − W_WIDE_ECE×ECE_wide`）+ race_relative_feats v2（22 列）。Cowork 弱点のワイドを黒字側へ | ❌ **不採用** | `optuna_v7_marks.py:1-24`、`reports/optuna_v7_marks.json`、`audit_v7_vs_v6_20260525.md:65`「**Δ ROI (v7−v6) = -1.50 ppt**」 | alpha=0.143 / composite=0.4820（別目的のため非比較）/ ワイド box3 ROI v7 79.04% vs v6 80.54% | 退役。valid metric（ワイド ROI）を直接最適化して過学習（VAL-02/VAL-03 の典型）。ROI 項を test 封印せず最適化した失敗例 |
| **v8** | v6 + race_relative_feats v2 + course_affinity_feats v1（sire/msire/jockey×場所・horse_dist 統計、計 +34 列）。目的関数は v6 と同一（特徴量のみ追加） | ❌ **不採用**（affinity in-sample leak） | `optuna_v8_marks.py:1-19`、`reports/optuna_v8_marks.json`、memory `project_v8_affinity_insample_leak.md`、`optuna_v10_marks.py:6`「v8=affinity leak」 | alpha=0.008 / composite=**0.4763**（最低、v6 0.549 比で大崩れ） | 退役。**affinity が自レース込み集計の in-sample target leak**（VAL-05: `build_course_affinity.py:120-123`）。⚠️ **v8 docstring は leak を自認せず「採用前提」で書かれている**。leak 修正前提のまま実行可能な状態で残るのは事故の元。**archive 隔離＋as-of/OOF 修正が必要**（本台帳は推奨記載のみ、ファイル移動は未実行） |
| **v9** | v6 を「上位 3 位集中」へ。`lambdarank_truncation_level` 5→3、composite を NDCG@3+上位印重視に再配分（W_NDCG3=0.40/W_HON_TOP3=0.30/W_HON_TOP2=0.20/W_TOP3_SUBSET_TOP3=0.10、winner_in_top5 廃止） | ❌ **不採用**（別系統の失敗実験、`optuna_v10_marks.py:6`） | `optuna_v9_marks.py:1-27`、`reports/optuna_v9_marks.json`、`optuna_v10_marks.py:6`「v9 は別系統の失敗実験」 | alpha=0.006 / composite=0.4829（composite 定義が v9 専用のため非比較） | 退役。relevance 境界を馬券（複勝/連対）に寄せる方向性は正しい（ALG-06）が、版として本番化されず |
| **v10** | 本番 v6 の学習プロセス監査（`docs/audit_20260611.md`）を反映した v6 直系の再学習。① split スライド（train≤2023/valid=2024/**test=2025 封印**）② タイム文字列列の修理（parse_time_str）③ 100%NaN 死亡特徴の検出除外 ④ **ECE_high_p の 4 ビン多ビン化**（過信×過小の相殺解消）⑤ CV を mean−1.0×std の保守選択に | ❌ **不採用**（test2025 で v6 同等以下） | `optuna_v10_marks.py:1-35`、`reports/optuna_v10_marks.json`、commit **`5eb533ae0f`**「docs(v10): 封印 test2025 で v6 同等以下と確定、採用見送り (v6 続投)」 | alpha=0.018 / **best_composite_mean=0.5701（valid=2024、他版と非同一土俵）/ std=0.0140** / best ece_high_p=0.0165（4 ビン版） | 退役。監査指摘に体系的に対応した正しい方向（OBJ-02/VAL-02/VAL-03/VAL-04 の修正を内包）だが、封印 test=2025 で v6 を上回らず。**v10 の改善（4 ビン ECE・test 封印プロトコル・mean−std 選択）は次版へ移植する価値あり** |

---

## 各 optuna_v*_marks.py 冒頭に付けるべき STATUS タグ案（OBJ-07 推奨）

各スクリプトを見ただけで「死んだ実験か本番系譜か」を機械可読に判別できるよう、docstring 冒頭への STATUS 行追加を推奨。
**本台帳は提案のみ。実ファイルへの追記は未実行**（既存コード不変更の方針）。

```
# optuna_v5_marks.py
STATUS: RETIRED (旧本番, rollback用保持; alpha=1.325 で tail calibration 崩壊)

# optuna_v6_marks.py
STATUS: PRODUCTION (default) / ⚠️ADOPTION-GATE NOT MET
        (audit_v6_vs_v5_20260520.md で ❌見送り判定だが export default=v6。OBJ-04 のガバナンス矛盾)

# optuna_v7_marks.py
STATUS: REJECTED (ワイドROI直接最適化で -1.50ppt 過学習, audit_v7_vs_v6_20260525.md:65)

# optuna_v8_marks.py
STATUS: REJECTED + LEAK (affinity in-sample target leak 未修正, VAL-05/optuna_v10_marks.py:6)
        DO NOT RUN as-is; archive へ隔離 + as-of/OOF 化が必須

# optuna_v9_marks.py
STATUS: REJECTED (trunc=3 別系統の失敗実験, optuna_v10_marks.py:6)

# optuna_v10_marks.py
STATUS: REJECTED (封印test2025でv6同等以下, git 5eb533ae0f)
        ただし 4ビンECE / test封印プロトコル / mean-std選択 は次版へ移植価値あり
```

---

## 派生・seed 変種（補足）

- `models/unified_rank_v6_s123.pkl` / `_s456.pkl` / `_s789.pkl` / `_s1234.pkl`: v6 の seed 変種。
  検証段で存在を確認（VAL-02 注記）。本番は seed=42 の `unified_rank_v6.pkl`。これら seed 版の採否・用途は
  本監査では**不明**（importance/分散測定用と推定 [推定]）。整理時は `models/archive/` 隔離候補。
- `models/unified_rank_v5.pkl`（退役・rollback 用）/ `pl_calibrators_v5.pkl`: CLAUDE.md 記載通り保持。

---

## 次版（v11 等）を出す際のチェックリスト（VAL-02/VAL-03/OBJ-04 反映）

1. **test=2025（以降）を封印**。学習・early stop・HP 選択・中間 audit のどこにも使わない（v10 プロトコル）。
2. **版間比較は valid のみ**。test は valid で勝った最終 1 版の事後確認専用（1 回だけ開封）。
3. **ECE は v10 の 4 ビン加重 \|gap\|**（過信側に非対称ペナルティ）を使う。v6 の単一ビン平均バイアスは使わない。
4. **採否は ROI ゲートで機械判定**（`run_v*_pipeline` が exit code）。基準を満たさないなら採用しない、
   または基準自体を明示改訂して文書化する（OBJ-04 の再発防止）。
5. **affinity 等の集計特徴は as-of / OOF 化必須**（VAL-05）。静的 train 集計テーブルの貼付は禁止。
6. **本台帳（version_ledger.md）に 1 行追記**してから本番化する。

---

*出典の実測値は `reports/optuna_v*_marks.json`（alpha/composite）、`reports/audit_v*.md`（ROI/ECE）、*
*git log（採否 commit）から取得。比較不能な composite は本文で明記。不明点は「不明」「[推定]」と表記。*
