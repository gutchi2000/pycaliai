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
