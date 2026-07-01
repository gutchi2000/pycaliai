# PyCaLiAI 全領域ガチ監査レポート（2026-06-15）

> **位置づけ**: 外部の競馬ML/馬券アドバイザー、および将来の自分（引き継ぎ者）向けの正式監査記録。
> 9 専門領域をマルチエージェントで精読 → 重大指摘を**敵対的に再検証**して優先度付けした。
> 各指摘には根拠 `file:line` を併記する。検証段（verification）で verdict が
> `refuted` / `partially_confirmed` / severity 是正されたものは「※検証段で〇〇と是正」を明記し、
> 是正後の正しい内容のみを採用する。`confirmed` は太字で強調する。
>
> 監査スコープ = 17 エージェント / 9 領域:
> 目的関数 / 確率キャリブレーション / メインアルゴリズム / 市場オッズ・Benter・馬券最適化 /
> 時系列・検証・リーク / データ品質 / アーキテクチャ / 天井突破戦略。
>
> 関連: 前回監査 `docs/audit_20260611.md`、版台帳 `docs/version_ledger.md`、`docs/PROJECT_SUMMARY.md`。

---

## エグゼクティブサマリ

本システムは「**印を当てる装置**」としては商用水準にある（leak 規約一貫・PL 閉形式厳密・監査文化成熟）。
しかし「**馬券として控除率 20% を超える出口**」が無いまま、実弾 ROI **68.6%**（n=600、`data/cowork_results.json`）で
出血を制度化している。根本原因は 3 つ。

1. **目的関数・銘柄選択が ROI / 市場と切れている**
   v6 目的関数は EV 皆無の純ランキング。`compute_bets.py:392` の最終ソートは `EV×boost` だが、
   自リポの実証（`reports/gate_q2_umaren_priceedge.json`）が「**EV 選抜は prob-only top-K=79% を 66% へ
   13pt 悪化させ、CLV も負**」と断言済み。**EV 選抜の全廃が最大かつ最安のレバー**。

2. **検証数値が本番を過大評価している**
   serve 時に 10〜12 特徴が `-9999` で死に、offline ◎複勝 62.08% → serve 57.53%（リネーム後 61.02%）/
   ECE 2.3 倍に劣化（`reports/serve_skew_eval.json`）。test=2024-25 を版選択で 7 回以上覗いており、
   しかも v6 は自分の採用ゲート（`scripts/audit_v6_vs_v5.py:278`）を満たさず本番化＝**ガバナンス破綻**。

3. **Benter 単複は実装済み・OOS 全 τ で控除割れと確定済み**
   ここに工数を割くのは死んだ仮説の再実装（`stage1_benter_blend.py`）。
   生きた +EV は「**exotics prob-first 絞り＋参戦ゲート＋見送り増**」のみ。
   crosspool の 88-92% は確定オッズ参照のオラクル上限で **betable ではない**（※検証段で high→medium 是正）。

> 一言でいえば: **綺麗な確率エンジンの上に、利益と切れた目的関数・素通りの LLM 配分・死蔵特徴という
> 濁った蓋が乗っている。**

---

## ROI を動かす順（top_priorities 13 件）

severity の正式値は検証段（verification）で是正された値を採用。`expected_impact` の数値は監査 JSON の実測値。

| rank | 施策 | 領域 | severity | 根拠（実測値・file:line） | 工数 | 期待 |
|---|---|---|---|---|---|---|
| **1** | 馬連/ワイドの銘柄選択を EV 選抜 → prob-only top-K（印近接流し）へ全面転換 | 馬券最適化 / エッジ源泉 | **critical** | `compute_bets.py:392` 最終ソート=`EV×boost`。`gate_q2_umaren_priceedge.json`: EV 選抜 test ROI 0.666 vs prob-only topK 0.79（+13pt）。控除床 80% を超える唯一の実証経路 | S | 馬連 67.5%→79% 帯へ最大 +11pt |
| **2** | 弱点券種（複勝・馬連）をデフォルト見送り化、見送り基準を損失帯に当てる | エッジ源泉 / 馬券最適化 | high | `cowork_results.json`: 全 4 券種 CI inconclusive・点推定 80% 割れ。馬連 67.5%(bet 617.5k)・複勝 67.1%(bet 351.3k) に最大の張りが集中 | S | 総合 68.6%→控除床近傍（黒字化でなく出血最小化） |
| **3** | serve 時に死ぬ 10〜12 特徴を解消し、検証数値=本番値に一致させる | 時系列・リーク / データ品質 | **critical** | `serve_skew_eval.json`: offline ◎複勝 62.08%→serve 57.53%（リネーム後 61.02%）、ECE 複勝 0.0118→0.0272（2.31 倍）。`course_*`/`jockey_*`/`hist_same_*` が serve で履歴 DB 不能→`-9999`。`export_weekly_marks.py:283-292` | L | 本番 ◎複勝 +3〜4pt・ECE 複勝 -50%、複勝/馬連 ROI +2〜4pt 相当 |
| **4** | EV/点数/配分/見送りの数式を `compute_bets`（決定論）へ全面移管、Cowork は narrative 専用化 | アーキテクチャ / 馬券最適化 | high | `validate_cowork_bets.py` は見送り 4 条件＋内容（券種/馬番/100 円/1 点上限）のみ強制。EV 妥当性・点数過多・レース総額/予算上限はノーガード | M | 同一 bundle で買い目が揺れ再現/監査不能の解消。配分ロジックの A/B が可能に |
| **5** | CLV（終値超過）を週次運用ループに常時接続し、的中前にエッジ判定 | 馬券最適化 | high | CLV 計測は `stage1_benter_blend.py:216`・`gate_q2:189-192` に実装済だが本番（`cowork_results`）未接続。gate_q2 で EV 選抜の CLV=-5.8%（逆選択）を既に示す | M | ROI 確定を待たず数週でエッジ判定。rank1-2 の検証も高速化 |
| **6** | PL 温度 τ を valid の馬連/三連的中尤度で MLE フィットし、joint の借り物温度を正す | メインアルゴリズム / 確率キャリブレーション | high | `pl_probs.py:25-28` `exp(s-s.max())` で LambdaRank 順序専用スコアを温度 1 固定流用。1D isotonic は分布の鋭さを直せない。馬連が実弾最弱と整合 | M | 馬連/ワイド ROI +2〜4pt 見込み |
| **7** | v6 採用ゲートを再判定し、CLV+高EV帯ROI+多ビンECE の複合スコアで機械 pass/fail 化 | 目的関数 | high | `audit_v6_vs_v5_20260520.md:45-52` が「❌ v6 採用見送り/v5 維持」（実測 単勝 -0.006・複勝 +0.002、基準 +0.05/+0.03）。`export_weekly_marks.py:147` default=v6 と矛盾 | M | 今後の v7/v10 採否の恣意性を解消、意思決定プロセスの信頼回復 |
| **8** | p_plc（連対率）を未補正で配信している非整合を解消、確率 3 点の単調性を assert | 確率キャリブレーション / メインアルゴリズム | high | `export_marks_json.py:241-255` で p_win/p_sho は calibrator 適用、p_plc は raw PL のまま。`build_pl_calibrators.py:87-88` buckets に renai/plc キー無し。p_win≤p_plc≤p_sho の単調逆転が構造的に起こる | S | LLM が「連対>複勝」の論理破綻データを掴むのを防ぐ。馬連の軸選定を直接改善 |
| **9** | 本番スコアリング数理（PL→isotonic→印→joint）の回帰テストと CI を整備 | アーキテクチャ | high | `export_marks_json.py:237-359` が本番心臓部だが `tests/` に PL 数理を触るテスト 0 件（grep ゼロヒット）。`serve_skew_eval.py` は存在も全パイプライン未配線。`.github` 無し=CI 皆無 | M | 確率が静かに歪んでも日曜夜まで発覚しない構造の解消 |
| **10** | ECE_high_p を多ビン加重 \|gap\|（過信非対称ペナルティ付き）へ置換 | 目的関数 / 確率キャリブレーション | medium | `optuna_v6_marks.py:243` `abs(mean(p)-mean(actual))` は高 p 帯の平均バイアス 1 個。寄与は composite の約 1.0%（0.5×0.0112=0.0056）で実質空振り。v10 が 4 ビン化済（`optuna_v10_marks.py:265-279`） | S | 版選択の信号を浄化、再学習ごとの無効最適化を止める |
| **11** | 母馬・前走走破タイム等の死蔵特徴（-9999 定数）を復活、importance 監査を CI 化 | データサイエンス / 特徴量 | high※ | `optuna_v6_marks.py:150` `fillna(-9999)`。母馬（nunique 10,740）・前走走破タイム（nunique 1,818）・kako5 人気系が numeric_coverage 0.0%。`utils.parse_time_str()` は `utils.py:161` に実在も未呼出 | M | 前走生タイム復活で印精度 +0.1〜0.3pt、母馬で配合系の細い上積み |
| **12** | ワイド配当推定 `o=馬連/3` を実測 payout 中央値へ置換、独立積フォールバックを厳格遮断 | 市場オッズ・馬券最適化 | medium | `compute_bets.py:298-313` c_wide が `o=o_um/3.0`。旧 bundle 独立積フォールバックは +21〜27% 過大（既知🟠、現 bundle は pair_probs 埋込で解消済） | S | ワイド（実弾 ROI 62.4% で最も溶かす券種）の EV 歪み解消 |
| **13** | 真の前向き検証（expanding-window walk-forward）へ移行し、版選択ノイズを排除 | 時系列・検証設計 | high | `optuna_v6_marks.py:273-294` の「5-fold CV」は単一 valid 予測を事後ランダム 5 分割しただけ（fold 再学習なし）。early stop・HP argmax・CV が全部同一単年 valid に依存 | M | 見かけ改善版の誤採用で ROI を毀損する事故源を断つ |

> ※ rank11 の severity は review 段 high。検証段でも high 据え置き（出力は壊れないが機会損失型）。

---

## 領域別所見

severity は検証段で是正された値。`confirmed` 指摘は**太字タイトル**。是正があったものは明記。

### 1. 目的関数の設計と最適化

> **一行所見**: 本番 v6 の目的関数は「印精度ランキングのプロキシ」を高い職人技で組んでいるが、
> (1) ROI/EV と一切結合せず馬券利益と構造的に乖離、(2) 看板の calibration-aware ECE は符号付き
> バイアスで影響ほぼゼロ、(3) しかも自分で定めた採用ゲートを満たさず本番化＝**目的関数より先に
> ガバナンスが壊れている**。

| id | severity | title | evidence | recommendation | effort |
|---|---|---|---|---|---|
| **OBJ-01** | high | 目的関数に EV/payout が一切入っておらずランキング最適と ROI が構造的に乖離 | `optuna_v6_marks.py:261-270` composite は NDCG@5/◎top3 等の純ランキング − 0.5×ECE のみ。本番 alpha=0.031 で payout 加重も実質ゼロ。ROI 直結試行（v7 wide_bonus `:345`、`learn_bet_customloss.py:78-81`）は本番未投入 | 二段構え：一段目は印精度 composite 維持、その上に Benter 型ブレンドを別レイヤーで CLV 目的に最適化。ROI 項は test 封印・walk-forward 後に採否 | L |
| **OBJ-02** | high | ECE_high_p が符号付き単一バイアスで過信と過小が打ち消し合う「看板倒れ」 | `optuna_v6_marks.py:243` はビン化された真の ECE ではなく高 p 帯全体の平均ギャップ 1 個。v10 は 4 ビン加重 \|gap\| で修正済（`optuna_v10_marks.py:266-279`）だが v10 不採用で本番は欠陥版のまま | evaluate を v10 の 4 ビン加重に差し替え。penalty 対象を p_plc/ペア確率へ拡張 | S |
| OBJ-03 | medium | ECE penalty 係数 0.5 が実質無効（composite 寄与 1.9%）、探索主因は alpha 相関 | 40 trials で 0.5×ECE の平均寄与は 1.9%。corr(alpha,ece)=0.844 で ECE 改善は alpha 低下の副産物 | 真の多ビン化＋W_ECE を 2.0〜5.0 へ。alpha を grid 固定して penalty 単独効果を ablation | M |
| **OBJ-04** | **critical** | **本番 v6 は自分で定めた採用ゲートを満たさず本番化（ガバナンス破綻）** | `run_v6_pipeline.py:13-14`/`scripts/audit_v6_vs_v5.py:278` の基準=単勝高EV ROI +0.05 or 複勝 +0.03。実測 -0.006/+0.002 で `audit_v6_vs_v5_20260520.md:48-52` が「❌ v6 採用見送り/v5 維持」。にもかかわらず `export_weekly_marks.py:147` default=v6、CLAUDE.md「v6 本番投入」、git log model=v6 | ROI ゲートで再判定し、満たさないなら基準自体を改訂して文書化。run_v*_pipeline が exit code で pass/fail 判定。v5/v6 を test=2025 封印で再走し本番を確定 | M |
| OBJ-05 | medium | composite 5 指標は実効次元が低く（NDCG@5 と ◎top3 が r=0.76）、重みに根拠なし | 40 trials で corr(ndcg5,hon_top3)=0.764、corr(composite,ndcg5)=0.832。trial 間 spread わずか 0.031 | 指標を直交化（純ランキング=NDCG@5、calibration=多ビン ECE、利益=高EV ROI の 3 軸）。重みは z-score 標準化＋等重み or Pareto | M |
| OBJ-06 | low | alpha（高配当 sample_weight）は機構として残すべきだが本番値 0.031 で完全に死んでいる | alpha=0.031 で本命 1.023〜大穴 1.164 とほぼ均一。v5 の 1.325 では 8 倍差で tail calibration 崩壊。Optuna が corr(alpha,ece)=0.844 を嫌い alpha≈0 へ収束 | 一律 tail 加重を撤去し穴は別レイヤー（市場乖離×中人気帯）へ。探索次元から外す | S |
| OBJ-07 | medium | 版が v5〜v10 まで 6 系統乱立、本番ファイルと運用文書・採否結論が相互矛盾 | optuna_v5〜v10 並存。v7=-1.5ppt 不採用、v8=affinity leak、v9=trunc3、v10=test2025 で v6 同等以下不採用。各 docstring は改善前提だが結論はレポート/git のみ | 各 optuna_v*.py 冒頭に STATUS 行を必須化。不採用版を archive へ隔離。`docs/version_ledger.md` を作成（→本監査で作成済） | M |

**検証段の補足**: OBJ-01/02/04 は全要素を一次資料で裏取り **confirmed**。OBJ-01 の「◎top3 が valid 改善でも」
の部分は不正確（レポートが直接示すのは valid NDCG@5 -0.009）だが、load-bearing な「ランキング目的でも単勝高EV
ROI=-0.006」は `audit_v6_vs_v5_20260520.md:45` に厳密一致し主張は成立。

### 2. 確率キャリブレーション

> **一行所見**: PL 閉形式は厳密で正しいが、その上に被せた「7 馬券独立 Isotonic」が PL 整合性を破壊。
> 本番が実際に使う **serve calibrator は監査対象ですらない** ──「綺麗な土台に濁った蓋」状態。

| id | severity | title | evidence | recommendation | effort |
|---|---|---|---|---|---|
| CAL-01 | ~~high~~→**medium**※ | v6 昇格根拠 ECE に fit 済 valid2023 が混入（監査ラベルの瑕疵） | `audit_marks.py:116` が `include_valid=True` で valid2023 を集計。「3 年」ラベルは fit set（in-sample）と真 OOS の加重平均 | ヘッドライン ECE は真 OOS（2024-2025）のみ使用、「3 年」ラベルを reliability 比較から除外 | S |
| **CAL-02** | high | 本番が使うのは serve calibrator なのに監査はフル特徴版を測っている | `export_weekly_marks.py:183-185` serve 優先。`audit_marks.py:108` はフル版を測定。本番相当値 `calibrators_v6_serve_eval.json:45-56` の test ece_fuku=0.0167/ece_umaren=0.0214 は監査値の約 1.4〜1.5 倍 | audit に `--serve` フラグ追加、本番採否は serve 値で判定。券種別に serve 版 vs フル版を真 OOS で個別選択 | M |
| **CAL-03** | high | 7 馬券種を独立 Isotonic で補正し PL 整合性を破壊、EV が券種間で不整合 | `build_pl_calibrators.py:151-158` で完全独立 fit。`export_marks_json.py:251-255/349-356` で再正規化なし。Σ_i cal(p_win_i)≠1 | 単勝 Σ=1 / 複勝 Σ=3 正規化。本筋は補正済 w' から PL 再計算で全券種を閉形式生成 | L |
| **CAL-04** | medium | p_plc（連対率）は calibrator が無く raw PL のまま Cowork に渡る | `export_marks_json.py:241-247` で raw 算出、calibrator 適用は tansho/fukusho のみ | renai バケット追加 or p_plc を bundle から落とす | S |
| CAL-05 | medium | Optuna 目的の ECE_high_p は真の ECE でなく平均バイアス 1 個（OBJ-02/ALG-03/VAL-07 と同根） | `optuna_v6_marks.py:243`。寄与ほぼ無 | 多ビン加重 ECE へ置換、W_ECE_PENALTY をスケール調整 | M |
| CAL-06 | medium | rolling/beta/2stage refit は全部書いてあるが本番未投入、valid2023 単年 fit のまま | `build_pl_calibrators.py:175` fit_split=valid=2023 固定。rolling/beta の成果物は production import 0 件 | rolling/beta の真 OOS を確認。改善 <2pp なら「検証済・不採用」記録。毎四半期 refit を pipeline に組込 | M |
| CAL-07 | medium | 三連単/三連複の rare-event Isotonic は陽性極少で階段化・端点潰れ、reliability diagram なし | `build_pl_calibrators.py:140-144` 陽性 1/R で isotonic fit、`:156` で y_max=1 クリップ。現状 pair_probs に三連系は不含 | rare-event は Beta calibration かログ空間 parametric へ。reliability diagram 出力を追加 | M |
| CAL-08 | low | PL softmax 温度 w=exp(s) が固定で温度校正なし（ALG-01 と同根） | `pl_probs.py:25-28` 温度 1 固定 | valid2023 で温度 T を 1 次元グリッド探索（→rank6） | S |

**検証段の補足**:
- **CAL-01 は ~~high~~→medium に是正、かつ中核主張は REFUTED**。
  監査ラベルに valid 汚染が混入しているのは事実（瑕疵）だが、CLAUDE.md:91 の「**ECE 複勝 -32% / 馬連 -34%**」は
  算術で再計算した結果 **真 OOS 値にピタリ一致**（複勝 -32.1% / 馬連 -33.9%）。汚染版を使えば -44%/-43% と
  「もっと良く」見えたはずで、「汚染で誇張」という因果はむしろ**逆**。喧伝されている数字は
  **クリーンな OOS 比較であり、誇張ではない**。Cowork が過大な ECE 信頼で張り続けるという懸念も前提が崩れる。
- CAL-02/03/04 は **confirmed**。

### 3. メインアルゴリズム（LambdaRank → Plackett-Luce）

> **一行所見**: 「印を当てる」装置としては堅実だが、確率推定としては土台が借り物 ──
> LambdaRank の順序専用スコアを exp して PL 温度に流用しており、joint（馬連/ワイド/三連系）の温度が無根拠。
> これが控除率を超えられない構造的要因の一つで、PL-MLE 化と二段ブレンドが優先。

| id | severity | title | evidence | recommendation | effort |
|---|---|---|---|---|---|
| **ALG-01** | **critical** | LambdaRank の順序専用スコアを exp() して PL 温度に流用 ── joint 確率の温度が無根拠 | `pl_probs.py:25-28` `exp(s-s.max())`。s は `objective='lambdarank'` の生 margin（`optuna_v6_marks.py:307`）。1D isotonic は温度を直せない | PL 温度 τ を valid の 1-2-3 着順列 PL 尤度で MLE フィット（`w=exp(s/τ)`）。本筋は PL/ListMLE 直接学習（`optuna_transformer_pl.py:8` に実装あり） | M |
| **ALG-02** | high | p_plc だけ calibrator が無く raw PL のまま bundle に載る（CAL-04 と同一） | `export_marks_json.py:241-256`、`build_pl_calibrators.py:87` に renai/plc キー無し | renai バケット追加。p_win≤p_plc≤p_sho の単調性 assert | S |
| **ALG-03** | high | ECE_high_p は真の ECE でなく平均バイアス（OBJ-02 と同根） | `optuna_v6_marks.py:240-245`。寄与誤差レベル | ビン化版へ置換、過信側 max(0,…) 非対称ペナルティ | S |
| **ALG-04** | high | PL の IIA を無補正で馬連/三連系に適用 ── 同型馬・展開相関を無視 | `pl_probs.py:58-60` p_umaren が Luce 公理（IIA）。Henery 補正・順位減衰なし。本リポ自身 `joint_calibration_v6.py`/gate_q2 で部分着手済 | Henery 一般化（2 着以降 w^γ, γ<1）を valid の馬連/三連複 MLE で推定 | M |
| ALG-05 | ~~critical~~→**low**※ | 市場オッズが一段目特徴にも二段ブレンドにも未使用 | `export_marks_json.py:300` market_p はラベル専用 | （※下記是正参照）現状維持が正しい。市場は罠ゲート/参戦ゲートの補助に | S |
| ALG-06 | medium | label の relevance 設計が学習（clip(11-着順)）と本番 Optuna（clip(6-着順)）で不一致、4-5 着に正の relevance | `train_unified_rank.py:76` vs `optuna_v6_marks.py:113` | relevance を馬券境界に（1着=3/2着=2/3着=1/4着以下=0）。死コード train_unified_rank を legacy へ | M |
| ALG-07 | medium | transformer_pl 系・8 モデルアンサンブルは本番経路で完全死蔵、PL 損失の正解実装が遊んでいる | grep: transformer_pl は Streamlit レガシー線のみ。`optuna_transformer_pl.py:8` に ListMLE 実装あり | PL 損失を LGBM custom objective へ移植する種に活用。Streamlit 線を legacy/ 隔離 | M |
| ALG-08 | low | race_confidence の field_chaos を calibrated p_win(Σ≠1) で計算、再正規化が分布を歪める | `export_marks_json.py:307,127-133` | race_confidence に raw PL p_win を渡す | S |

**検証段の補足**:
- ALG-01/02/03/04 は **confirmed**（ALG-01 の `spread mean=3.12` 等の集計値は独立検証未了だが骨子は支持）。
- **ALG-05 は REFUTED、~~critical~~→low に是正**。「Benter 二段はどこにも無い（grep で実装なし）」は事実誤り。
  `stage1_benter_blend.py` が `p∝softmax(α·log f + β·log π)` を実装し、fit≤2023/eval=test、bootstrap CI・CLV・
  EV ゲートまで完備。`stage1_benter_blend.json`: 変種A blend ROI test=**0.7092**（控除床 0.80 を全 τ で下回る）、
  ΔR²_vs_market=**-0.0198**（blend は market-only より悪化）。「市場未使用が天井の正体」は本プロジェクトの OOS で
  明確に否定済み。フロンティアは exotics+参戦ゲート側、という既存結論が正しい。

### 4. 市場オッズ・Benter 二段・馬券最適化

> **一行所見**: 「Benter 未実装」という前提は誤りで、Benter は実装済み・leak-safe 監査済みで「単複は控除率を
> 超えない」と既に結論が出ている。本当の病巣は (a) compute_bets が高EV=過大評価 longshot を買いに行く構造、
> (b) 本番に Kelly もエッジ連動 stake sizing も無い素朴フラット配分、(c) +EV frontier の出口が無いまま控除率
> 近傍 ROI 68% で資金を流し続けていること。

| id | severity | title | evidence | recommendation | effort |
|---|---|---|---|---|---|
| **BENTER-01** | high | 「Benter 未実装」は事実誤認 ── 実装済み・監査済みで「単複は控除率を超えない」が確定済み | `stage1_benter_blend.py:1-299`、`stage1_benter_blend.json`: 変種A α=1.0076(CI[0.98,1.04]), β=0.320, ΔR²=-0.0198, test ROI 0.7092（2024 0.6764/2025 0.7426）。`learn_bet_v1/v4/v5_clv.py`・`audit_market_layer.json`（overall_verdict='no profitable edge は ROBUST'）も同結論 | Benter 単複の**再実装は不要**。唯一の正の知見: shrinkage α≈0.3 で test R²=0.2457>market 0.2383（ΔR²+0.007）。これは calibration には使えるが ROI には乗らない。確率の最終キャリブレーションに低 weight ブレンドとしてのみ検討 | S |
| **EVSEL-02** | **critical** | 高EV選抜が構造的に負ける ── umaren 価格エッジ選抜は prob-only(79%) を 66% へ悪化 | `gate_q2_umaren_priceedge.json`: EV 選抜 raw_am9 τ*=2.0 → test ROI 0.6662(n=60298, CLV -5.8%)。control prob-only topK3/5/7 = 0.793/0.7936/0.7913。oracle（確定odds）ですら 0.59-0.61。`compute_bets.py:392` の最終ソートが `EV×boost` | 選抜・ソートを EV 順 → calibrated p_pair 順へ。EV は -EV を切るフロアとしてのみ使う | M |
| **KELLY-03** | high | 本番に Kelly もエッジ連動 stake sizing も無い ── フラット EV 重み配分のみ。Kelly 一式は legacy 死蔵 | `compute_bets.py:89-113` 階段関数＋weight 比例配分。`:35` で kelly を一切 import せず。kelly/ev_filter/value_model は Streamlit/test 専用 | Kelly 系を legacy/ 隔離し「本番は Kelly 不使用、フラット配分」を明記。+EV セグメント確立後に限りその内部で fractional Kelly。※控除率内では f*≤0 で大半 0 ベットになる留保あり | M |
| WIDE-04 | medium | ワイド odds 推定 `o=馬連/3` が系統バイアス、旧 bundle 独立積フォールバックは +21〜27% 過大のまま | `compute_bets.py:298-313`。現 bundle は pair_probs 全 36R 埋込で確率側は救済済（→確認）だが /3 近似は配当推定で効く | `payout_table.parquet` の実測ワイド配当中央値へ置換。フォールバックは pair_probs 必須に厳格化 | S |
| LLM-05 | medium | EV/点数/配分という決定論問題を LLM(Cowork) に委ねる設計は再現性・監査性を損なう | CLAUDE.md 役割分担。同一 bundle でも LLM サンプリングで買い目が変わりバックテスト不能 | 数式を 100% compute_bets へ移管、LLM は narrative/定性フラグのみ | L |
| VALUEMODEL-06 | medium | train_value_model の時系列分割が全体規約と矛盾、calibrator も同一 2024 内 fit でリーク懸念 | `train_value_model.py:124-161` で 2024 データを 4 分割。宣伝 ROI140-159% は 2024 in-year 楽観値 | value_model 系は legacy 確定。retrain の weekly 自動実行を停止 | S |
| FRONTIER-07 | high | +EV の現実的 frontier が未確立のまま控除率近傍 ROI68% で資金を流し続けている | `cowork_results.json`: total ROI 68.6%(n=600, bet 182万, pnl -57万)。全券種 inconclusive。gate_q2 prob-only umaren topK=79% が控除 77.5% を僅かに上回る唯一の灯 | 3 手: exotics prob-first 絞り / 参戦ゲート / 中人気妙味。各々 ≤2023 tune・test 評価で控除超部分集合の実在を確認、無ければ見送りを増やす | L |
| CLV-08 | medium | CLV は計測コードは存在するが運用ループに未接続 ── 唯一の先行指標が研究の引き出しに死蔵 | `stage1_benter_blend.py:216`/`gate_q2:189-192` に実装、本番 `cowork_results` には CLV 集計なし | weekly_post に CLV 集計を追加（→rank5） | M |
| MARKET-09 | low | 一段目特徴に市場オッズ未使用は「弱み」ではなく検証済みの正しい設計判断 | `audit_market_layer.json`: market-only R²=0.2383>fund-only 0.1936 だが blend は -0.0198 悪化。今走オッズはリーク | 現状維持で正しい。de-vig 市場 marginal を参戦ゲート/罠検知の補助に | S |

**検証段の補足**: BENTER-01/EVSEL-02/KELLY-03/FRONTIER-07 は **confirmed**（EVSEL-02 は high→critical に**昇格**）。
EVSEL-02 の `compute_bets.py:392` ソートキー `-(c[4]*c[6])` は逐語一致で裏取り済み。

### 5. 時系列解析・検証設計・データリーク

> **一行所見**: 時系列の前向き安全性（shift/cumsum/merge_asof）は概ね堅牢だが、検証設計が
> 「単一ホールドアウト＋同一 valid で early stop/HP選択/疑似CV」という三重依存。さらに致命的なのは
> 本番 serve で 10〜12 特徴が再構築不能のまま死亡し offline 62.08%→serve 57.5%（リネーム後 61.0%）に
> 劣化している点で、これは**検証数値が本番 ROI を過大評価する構造的欠陥**。

| id | severity | title | evidence | recommendation | effort |
|---|---|---|---|---|---|
| **VAL-01** | **critical** | 本番 serve で 10〜12 特徴が再構築不能のまま死亡 → offline 62% は本番では達成不能 | `serve_skew_eval.json`: baseline 62.08%/ECE 0.0118 → 全マスク 57.53%/0.0272、回収+リネーム18 で 61.02%/0.0258。`course_*`/`jockey_*`/`hist_same_*` が serve で履歴 DB 不能→-9999。`export_weekly_marks.py:283-292` | (A) Stage1-5 集計を as-of スナップショットとして export 時に再計算。(B) 重いなら死特徴を学習からも除外した v6' を再学習し offline=serve を保証 | L |
| **VAL-02** | high | test=2024-25 を版選択で 7 回以上覗いている（本番 v6 は test 汚染下で選ばれた） | audit_marks v4/v5/v6/v7/v10＋比較レポート 3 本＋seed 変種。`scripts/audit_v6_vs_v5.py:278` の採用判定が test の EV ビン別 ROI を直接見る | v10 の「test=2025 封印・採用判定 1 回のみ」を運用ルール化。版間比較は valid のみ、test は最終 1 版の事後確認専用 | S |
| **VAL-03** | high | 「5-fold CV」は CV ではない：単一モデルの単一 valid 予測を後からランダム 5 分割した分散推定 | `optuna_v6_marks.py:273-294`/`optuna_v10_marks.py:302-325` は fold ごとに再学習していない。early stop も同じ valid（`:327-329`） | expanding-window で各 fold 別モデル学習。early stop と HP 選択のデータを分離 | M |
| **VAL-04** | high | 単一ホールドアウト（valid=単年）で年次非定常に対し版選択が過適合 | train≤2022/valid=2023単年/test≥2024 を実測確認。v10 も valid=2024 単年 | valid を複数年に拡張 or expanding-window で複数 (fit,eval) 年ペアの平均で版選択 | M |
| VAL-05 | medium | course_affinity horse_dist テーブルは train 内 in-sample target leak（v8 系の真因、再利用厳禁） | `build_course_affinity.py:120-123` で train 全期間集計、自分の着順が自分の特徴に混入。n=5 で 2top3 の馬は feature=0.40 だが真 LOO は 0.25。本番 v6 は affinity 不使用で現状実害なし | 将来使うなら必ず as-of 化（cumsum LOO or OOF target encoding）。静的 train 集計テーブルの貼付は破棄 | M |
| VAL-06 | medium | serve calibrator が「欠損 30 特徴版」で fit され offline と二系統に分岐 → どちらの ECE も本番真値ではない | `export_weekly_marks.py:177-185`。audit/serve_skew は offline 用フル特徴版を使用 | VAL-01 で死特徴を解消すれば一本化できる。当面 serve refit を週次必須ステップに | M |
| VAL-07 | medium | ECE_high_p は単一ビンの符号付き平均バイアス（OBJ-02/CAL-05/ALG-03 と同根） | `optuna_v6_marks.py:176-245`。v10 で 4 ビン化済 | v10 の多ビン ECE を移植、過信側に非対称ペナルティ | S |
| VAL-08 | low | winner_tansho sample_weight が kekka（勝ち馬のみ）由来で、勝てなかった馬を一律 100 円扱いにする非対称重み | `optuna_v6_marks.py:96-105,140-145`。レース勝ち馬配当を全馬一律付与＝波乱レース属性であり馬個体妙味でない | payout-aware にしたいなら馬単位の市場乖離で重み付け。alpha≈0 が最適なら sample_weight 機構自体を削除 | S |
| VAL-09 | low | 調教 merge は学習=履歴無制限・serve=14 日カットオフで結合条件が非対称 | `build_master_v2.py:138-145` vs `export_weekly_marks.py:252-255`。坂路 93.9% で実害小 | 学習側 merge にも同じ日数カットオフ。欠損フラグ is_trn_missing を明示特徴化 | S |

**検証段の補足**: VAL-01/02/03/04 は **confirmed**。
- VAL-01 の「28 特徴中 13 特徴」は厳密には **永久死は 10〜12 列**（numeric 10 列＋categorical 2 列）。「ECE 2.2 倍」は
  むしろ控えめで実測 **2.31 倍**（0.0272/0.0118）。severity critical 維持。
- VAL-02 の evidence パス `audit_v6_vs_v5.py:278` は実体 `scripts/audit_v6_vs_v5.py:278`（内容・行番号は正確）。

### 6. データサイエンス / 特徴量 / データ品質

> **一行所見**: リーク対策とエンコーディング前処理は概ね健全だが、本番 v6 に特徴量重要度監査が一切無いため
> 母馬・開催・前走走破タイム等が恒久的に -9999 定数=完全死蔵。fillna(-9999) と LabelEncoder 序数化という
> 古い LightGBM 作法が分割効率を地味に削っており、本質は「**特徴量を使い切れていない機会損失**」。

| id | severity | title | evidence | recommendation | effort |
|---|---|---|---|---|---|
| **DQ-01** | high | 母馬・開催・前走走破タイム・kako5 人気系が恒久的に -9999 の完全死蔵特徴 | `optuna_v6_marks.py:150` `fillna(-9999)`。valid+test 141,522 行で母馬（nunique 10,740）/開催/前走走破タイム（nunique 1,818）/kako5_avg_ninki/pos_vs_ninki が numeric_coverage 0.0%。`utils.parse_time_str()` は `utils.py:161` に実在も未呼出。unified_rank_v6 の importance 監査ファイル不在 | CAT_COLS に母馬・開催を追加。前走走破タイムを parse_time_str で float 秒化。kako5 人気系は feats から除外。lgbm importance を CI 化 | M |
| DQ-02 | medium | 欠損を一律 fillna(-9999) ── LightGBM のネイティブ NaN 分岐を捨てている | `optuna_v6_marks.py:150`。trnW(25%)・hist_same・course_*_rate が連続軸片端に固まる | make_dataset で fillna(-9999) を削除し NaN のまま渡す。serve 側も統一 | M |
| DQ-03 | medium | 高カーディナリティ名義カテゴリ 28 列を LabelEncoder 序数化（native categorical 未使用） | `optuna_v6_marks.py:120-135`。種牡馬 407/母父 812/生産者 1013/馬主 1225。整数の大小に偽の順序 | astype('category') 保持＋`categorical_feature=CAT_COLS`。min_data_per_group/cat_smooth を Optuna に | M |
| DQ-04 | medium | WC 調教 25% 等の高欠損特徴を -9999 で埋める「データ無し=超低値」バイアス | `build_master_v2.py:165`。train(2013-) で trnW ほぼ全欠損→-9999 | NaN 保持＋has_training フラグ分離。train 調教 JOIN にも 14 日カットオフ | L |
| DQ-05 | high | 全頭確定単勝オッズ（外部 E:\競馬過去走データ）が一段目に未統合＝Benter ブレンドの前処理欠落 | master_v2 に今走/前走単勝オッズ列ともに不在。`make_weekly_hosei.py:167` の全馬収録 CSV は補正タイム照合専用 | 全頭 as-of 確定オッズを master_v2 に JOIN（二段ブレンド専用、一段目には入れず leak 回避） | L |
| DQ-06 | low | kako5_pos_trend の符号が docstring（負=上昇）と実装（改善馬で正 slope）で逆 | `parse_kako5.py:261,139-142`。実測 改善馬 slope=+2.3 | `-coeffs[0]` で意図に合わせる or docstring を実装に合わせる。xai/cowork_prompt と整合 | S |
| DQ-07 | low | fuku30/horse_fuku30 はカレンダー 30 日でなく「直近 30 走」のローリング（命名が誤解を招く） | `build_dataset.py:104-110` は行数ベースの窓 | 命名を _last30rides に正す or time-based rolling('30D') へ。CLAUDE.md も修正 | S |
| DQ-08 | low | pedigree_stats の距離バケット中央値計算が境界でズレる（表示専用） | `build_pedigree_data.py:192`。1600m→1700 とズレ | `(距離+100)//200*200` or `round(距離/200)*200`。表示専用なので effort 最小 | S |

**検証段の補足**: DQ-01/DQ-05 は **confirmed**（DQ-01 は母馬と母父馬の区別が正確 ── 母父馬は CAT_COLS 所属で生存、
母馬のみ死蔵。`utils.parse_time_str` は `utils.py:161` 実在も grep 0 件で未呼出を確認）。

### 7. システム / ソフトウェアアーキテクチャ

> **一行所見**: 本番スコアリング数理（PL→isotonic→印）はテスト 0・CI 不在・serve-skew 検知が未配線で
> 「静かに壊れても誰も気づかない」構造。`.git` が 94 pack/11.6GiB に肥大、フラットなスクリプト 100 本 soup・
> 手動 Cowork コピペが運用の単一障害点 ── モデル精度より先にこのプロセス信頼性が馬券 ROI の足を引っ張る。

| id | severity | title | evidence | recommendation | effort |
|---|---|---|---|---|---|
| **ARCH-01** | **critical** | 本番スコアリング数理（PL→isotonic→印→joint）がテスト 0・serve-skew 検知が全パイプライン未配線 | `export_marks_json.py:237-359` が心臓部。`tests/` に PL 数理を触るテスト 0 件（grep ゼロ）。`serve_skew_eval.py` は存在も weekly_nicegui/run_v6/run_v5/t10 で未配線。`.github` 無し | export_race の golden JSON 回帰テスト追加。serve_skew_eval を export 末尾 gate に組込 fail-closed。pytest を Phase A 先頭で必須化 | M |
| **ARCH-02** | high | .git が 94 pack / 11.6GiB に肥大 ── gc.auto=0 で重複 pack 放置 | `git count-objects -vH`: packs=94, size-pack=11.64GiB。398M 級 pack が複数連続＝同一大型 blob の重複格納。data/*.parquet（411MB 級）の多リビジョン蓄積 | git gc --aggressive。git filter-repo で data/*.parquet 履歴除去→fresh clone。.gitignore 追加・DVC/HF Datasets へ逃がす | L |
| **ARCH-03** | high | v6 marks スタックと旧 8 モデルアンサンブルの 2 系統並走＋本番 export が legacy predict_weekly に import 依存 | `export_weekly_marks.py:57` `from predict_weekly import parse_csv`（predict_weekly は実測 2022 行）。models/ に旧 pkl 群・unified_rank v6/v10 両方 tracked | parse_csv を io_weekly.py 等の中立モジュールへ。旧アンサンブル一式を legacy/ 隔離 | M |
| **ARCH-04** | high | Cowork（外部 LLM 手動コピペ）が馬券意思決定の中核 ── ガードは見送り/内容のみで配分は素通り | `validate_cowork_bets.py` は skip_reasons 4 条件＋content_issues（券種/馬番/100円/1点上限）のみ。EV妥当性・点数過多・レース総額/予算上限は enforcement なし | compute_bets を一次決定論ベットラインに昇格、Cowork は narrative/却下権のみ。validate に総額/点数/配分上限を enforce | M |
| ARCH-05 | medium | ルート直下に本番と使い捨て実験が未分離で .py 111 本 ── src/experiments/legacy パッケージ化が皆無 | `git ls-files` root .py=111 本。src/ experiments/ legacy/ いずれも無し | src/pycaliai/ に本番を移し package 化。experiments/ に exp_*/sweep_*/learn_bet_*/audit_* 退避 | L |
| ARCH-06 | medium | 学習環境 requirements.txt がゼロ exact-pin、モデル再学習の再現性が担保されない | requirements.txt の `==` 0 件。sklearn/LightGBM 学習側固定なし | requirements-train.txt で lightgbm/sklearn/optuna を == 固定。各 pkl に sidecar JSON で lineage | M |
| ARCH-07 | low | race ID 列名と master CSV パスが 49〜60 本のスクリプトにハードコード散在、単一 config 不在 | central config 無し。「レースID(新…」が root .py 60 本、master パスが 49 本に出現 | constants.py に COL_RID/COL_BAN/MASTER_GLOB を集約 | M |
| ARCH-08 | medium | sync-hf の orphan-branch + checkout --force 多用が運用上の自己破壊リスク、HF リモートに master/main 二重 | `sync-hf.ps1:115` `git checkout --force hf-spaces`。コメント自身が「--force は未 stage 編集を黙って破壊」と認める | orphan 同期を git worktree 方式へ。hf/master 不要なら削除し main 一本化 | M |

**検証段の補足**: ARCH-01/02/03/04 は **confirmed**（ARCH-03 の predict_weekly は実測 2022 行で「1800 行」はやや過小評価。
2 系統 calibrator 並存（pl_calibrators_v6.pkl と _serve.pkl）も裏取り済み）。

### 8. 予測天井の突破とエッジ源泉（プロ馬券師視点）

> **一行所見**: 「機械的 +EV は全方向で死亡」は正しく実証済みだが、エッジは「予測」でなく
> 「**市場間の歪み**」と「**打たない判断**」に残っている。

| id | severity | title | evidence | recommendation | effort |
|---|---|---|---|---|---|
| EDGE-01 | ~~high~~→**medium**※ | crosspool 整合性裁定の本命/中穴帯 ROI 88-92% を experiment のまま放置 | `crosspool_umaren.json`: 本命帯 馬連<10 ROI 0.9167(n=627)、中穴帯 10-50 ROI 0.8848(n=2725)。`crosspool_umaren.py:209-211` で層別。compute_bets/cowork 未配線 | （※下記是正）Stage2 で前売り 9 時 odds・CLV 計測へ。f_win を PL に差し替え。控除超が再現したら crosspool レーンを追加 | M |
| **EDGE-02** | high | EV ゲートが prob-only top-K より一貫して劣る ──「EV で選ぶ」設計自体が逆効果 | `gate_q2_umaren_priceedge.json`: ≤2023 ROI 125-130% が test で 64-67% へ崩壊、CLV 全閾値マイナス。control prob-only topK=0.79（EV 0.66 を +13pt）。control の CLV は +0.13〜0.15 で正 | 馬連/ワイドの銘柄選択を EV 閾値 → prob-only top-K へ全面転換。EV 閾値ゲートは廃止 or 罠除外専用に格下げ | S |
| **EDGE-03** | high | 複勝・馬連を実弾で買い続けている ── 最新実績で全券種が控除割れ・CI inconclusive | `cowork_results.json`(2026-06-11, 600 bets/479 races): 総合 68.6%。単勝 89.8%(CI 34-150)/ワイド 62.4%/複勝 67.1%/馬連 67.5%、全 inconclusive。馬連 617.5k 円・複勝 351.3k 円が最大の張り | 複勝は原則停止 or ◎単独 p_sho>0.40 のみ。馬連は EDGE-01/02 通過分のみ。cowork_prompt に「複勝・馬連はゲート通過時のみ、デフォルト見送り」を明記 | S |
| EDGE-04 | medium | 「打たない」最適化（参戦ゲート）が見送り率の運用に留まり、レース選別の判別器が存在しない | 見送り 296/600 は LLM 主観＋validate コード強制。市場緩さの meta-model 不在。`crosspool_umaren.json` の層別が市場効率のセグメント依存を示すが未活用 | レース単位の市場緩さ判別器を構築（HHI/頭数/クラス/crosspool 不整合度/race_confidence）。bet_worthiness を race_meta に載せ Cowork へ | M |
| EDGE-05 | medium | 穴で勝てないのに本命サイドの「市場が過小評価する本命」検出に集中できていない（縮小 Benter α が monetize 未着手） | memory: ◎大穴 15+ 倍 勝率 0%。縮小 α≈0.3-0.4 で blend R² 0.245>市場単独 0.238、◎単勝 CLV +0.13〜0.35。だが monetize 未着手 | 縮小 α の p_win_blend を bundle に別列追加。本命帯<7 倍かつ p_win_blend>1/odds×1.2 のみ単勝候補。穴帯 15+ は除外 | M |
| EDGE-06 | medium | PL 厳密 joint がライブに流れず、馬連が単勝確率の近似で再計算（bundle に p_win のみ） | bundle は p_win/p_plc/p_sho と実オッズのみ。Cowork は条件付き近似（`docs/cowork_prompt.md:361-368`）。コードには all_umaren_mat（PL 厳密）あり | export_marks_json で PL 厳密ペア確率行列を bundle に同梱。crosspool/参戦ゲートの土台インフラ | M |
| EDGE-07 | low | distortion_exp の「発見されたが test で消失」を再挑戦せず打ち切るべき | `distortion_exp.json`: verdict=NO_ARBITRAGE。≤2023 発見の 4 分位すべて test/Bonferroni 後で控除超を再現せず。leak 違反 0 で検査 clean | distortion_exp/gate_q2 は「検証済み死亡」として封印し再走しない。資源は crosspool Stage2 へ | S |

**検証段の補足**:
- EDGE-02/03 は **confirmed**（EDGE-02 の数値は逐語一致、control の CLV が正で「prob-only は速い金、EV 選択は遅い金」を裏付け）。
- **EDGE-01 は ~~high~~→medium に是正**。数値・未配線は事実だが、この ROI は **確定オッズ(KB==4)で EV 判定し
  同じ確定オッズで決済するオラクル上限**（`crosspool_umaren.py:65,151,200`）。層別セルも「その買い目自身の確定
  オッズ帯」で区切る＝確定オッズを知らねば帯が決まらない**二重オラクル条件**で **betable ではない**。
  スクリプト自身が docstring で「Stage1=存在テスト、oracle 上限版」「Stage2 で前売り(区分1)運用版へ」と明記。
  gate_q2 が示す通り前売 odds 化すれば 66% 帯へ崩れる公算が高い。**未配線は機会損失でなく正しい判断**。

---

## 文書とコードの矛盾（contradictions 7 件）

1. **v6 採用ゲート未達のまま本番運用**: CLAUDE.md/PROJECT_SUMMARY は「v6 本番投入」を謳うが、採用判定
   `scripts/audit_v6_vs_v5.py:278` の adopt 条件を満たさず、`reports/audit_v6_vs_v5_20260520.md:48-52` 自身が
   「❌ v6 採用見送り/大幅な改善なし。v5 維持」と出力。`export_weekly_marks.py:147` は default='v6'。
   定量ゲートの結論と実運用が正面から矛盾。

2. **v6 本番化根拠の自己矛盾**: CLAUDE.md:92「機械買い ROI は v5 と同等（±0.01）」と「v6 本番化の根拠」が
   自己矛盾。実弾 ROI 上の v6 化リターンは未確認で、根拠は ECE 改善（OBJ-03 で alpha と交絡し純効果不明）に依存。

3. **「Benter 二段は未実装」は事実誤認**: `stage1_benter_blend.py`（`p∝softmax(α logf+β logπ)`）・
   learn_bet_v1/v4/v5_clv・audit_market_layer が実在し、OOS 全 τ で控除割れと既に結論済み
   （`stage1_benter_blend.json`: test ROI 0.7092）。前任の grep が未追跡新規ファイルを取りこぼした。

4. **「5-fold CV, seed=42」は誤導的表現**: `optuna_v6_marks.py:427` の bundle description は実体が単一
   ホールドアウト＋事後ランダム 5 分割の分散推定で、fold 再学習は無く真の CV ではない。

5. **value_model の位置づけ矛盾**: CLAUDE.md は value_model を「HALO/旧経路用」と位置づけるが、
   `retrain_value_model.py` が weekly_nicegui から日曜+月初に自動実行され続け、本番非依存の楽観モデルを毎週再生産。

6. **「前走単勝オッズは使用 OK」だが列が無い**: CLAUDE.md は「前走単勝オッズは使用 OK」と記すが、master_v2 に
   前走単勝オッズ列が存在しない（grep 0 件）。記述上使えるはずの列が実データに無い。

7. **crosspool 評価の誇張**: 前任所見「crosspool 88-92% を放置=最大の機会損失」は誇張。
   `crosspool_umaren.py:65,151,200` は確定オッズ(KB==4) で EV 判定・決済・帯層別する二重オラクル条件で
   betable ではない。Stage2（前売 odds・CLV）未通過。未配線は機会損失でなく正しい判断。

> 加えて DQ-06: `kako5_pos_trend` の符号が docstring（負=調子上昇）と実装（改善馬で正の slope）で逆。
> モデルは同関数なので実害小だが、xai/cowork_prompt の説明と整合せず人間/LLM 介在層を誤らせる。

---

## 実は良い点（what_is_actually_good）

- **PL 閉形式エンジンが本プロジェクト最良の部品**: `pl_probs.py` の Plackett-Luce joint が閉形式で厳密実装され、
  Σ単勝=1/Σ複勝=3/Σワイド=3/Σ三連単=1/整合性 Σ_j wide(i,j)=2·fuku(i) 等を `_test()` で自動検証。
  近似に逃げない確率エンジンは全 upgrade の土台。
- **leak 規約・時系列前向き安全性が一貫して堅牢**: shift(1).rolling / cumsum-self / train-only fit、
  調教 merge_asof（direction=backward, allow_exact_matches=False）で当日除外、LEAK_COLS で着順/fukusho_flag/
  roi_target/単勝オッズを明示除外、LabelEncoder の `__NaN__` 未知値処理。意思決定=9 時前売り/決済=確定/
  閾値=≤2023/評価=test OOS の分離が全ハーネスで衛生的。
- **「+EV は無い」を願望でなく多数の独立 OOS 実験で潰し切っている誠実さ**: gate_q2/distortion_exp/benter/
  audit_market_layer/exp_umami。Bonferroni 片側下限・bootstrap CI・walk-forward 3fold・de-vig overround 1.259
  整合・CLV 符号定義の一貫性まで forensic で、馬券研究としての作法は商用水準。
- **control（価格を見ない prob-only top-K）を必ず併走させ価格情報の寄与を分離**。これにより「EV ゲートはむしろ
  有害（66% vs 79%）」という反直感的だが正しい結論に到達できている。
- **成熟した監査文化と防御の多層化**: cowork_results.json に Wilson CI+bootstrap CI+roi_verdict 常設で過去の
  小標本ノイズ（単勝 120%/馬連 56% 全廃）を自ら統計的に否定。validate_cowork_bets が見送り 4 条件を bundle 真値と
  突合し fail-closed。export 品質ゲート（race0/被覆<50%/p_null）で無言 push 事故を停止。serve_skew_eval.py で本番
  劣化を数値可視化。v10 で監査指摘（split 鮮度・多ビン ECE・mean−std 保守選択・test 封印）に体系的に対応。
- **sample_weight alpha を Optuna が一貫して near-zero（v6=0.031/v8=0.008/v9=0.006）に追い込み**、「穴を払戻で
  重み付けして当てに行く」筋の悪い細工をデータ自身が否定。longshot を無理に持ち上げない判断が裏取りされている。
- **models/*.pkl は .gitattributes で LFS 管理され 13GB git 肥大の主因ではない**（肥大は巨大 parquet/CSV の
  多リビジョン）。モデルだけは正しく LFS に逃がせている。

---

## 検証済み死亡（再走するな）

以下は ≤2023 で見えたエッジが test/Bonferroni 後に消失、または構造的にオラクル条件付きと**敵対的検証で確定**した
死んだ仮説。**再実装・再バックテストに工数を割くな**。

| 死亡仮説 | 根拠 | 結論 |
|---|---|---|
| **Benter 単複ブレンド** | `stage1_benter_blend.json`: 変種A test ROI **0.7092**（2024 0.6764/2025 0.7426）全 τ で控除床 0.80 割れ、ΔR²_vs_market=-0.0198。`benter_singlewin_noedge.json`: 全 odds 帯/value 層で <100%、最高 value 層が最悪 0.678。`audit_market_layer.json`: walk-forward 3fold 全て ΔR²<0 | 単複の市場ブレンドは控除率を超えない。実装済み（`stage1_benter_blend.py`）・OOS 確定済み。再実装不要。唯一の残骸＝shrinkage α≈0.3 の微小増分は calibration 用途のみ |
| **crosspool oracle ROI 88-92%** | `crosspool_umaren.py:65,151,200` が確定オッズ(KB==4) で EV 判定・決済・帯層別する二重オラクル | 現状はオラクル上限で betable ではない。前売 odds 化すれば gate_q2 同様 66% 帯へ崩れる公算大。**Stage2（前売 odds・CLV）を通すまで +EV と言えない** |
| **distortion_exp（券種間整合性の破れ・モデル-市場乖離）** | `distortion_exp.json`: verdict=**NO_ARBITRAGE**。B_fedge 等 4 分位すべて test/Bonferroni 後 CI 下限で控除超を再現せず（B_fedge_q0.95: le2023 0.933→test 0.773, CI 下限 0.60） | 「モデル vs 市場」軸は精度天井に律速され全滅。再走しない |
| **gate_q2 EV 価格エッジ選抜** | `gate_q2_umaren_priceedge.json`: ≤2023 ROI 125-130% が test 64-67% へ崩壊、CLV 全閾値マイナス。oracle（確定 odds）ですら 0.59-0.61 | EV で妙味を拾う発想は test で再現せず逆選択（CLV 負）。prob-only top-K に全面転換すべき |

> 新規の歪み探索を行うなら必ず「**市場 vs 市場**」の軸に限定し、「モデル vs 市場」（=精度天井に律速）は避ける。
> 生きた frontier は (a) exotics prob-first 絞り、(b) 参戦レースゲート、(c) 絞られた中人気の過小評価、の 3 つのみ。

---

*生成: 2026-06-15 / 監査スコープ 9 領域・17 エージェント / 全指摘を敵対的検証済み。*
*数値は監査 JSON および一次資料（reports/*.json, *.md, ソースコード）の実測値。推測は本文中 [推定] と明記。*
