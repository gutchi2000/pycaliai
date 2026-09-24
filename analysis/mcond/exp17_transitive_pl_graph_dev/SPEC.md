# EXP17 仕様書 — 共通対戦馬を介した動的Hodge–Plackett–Luce

**版:** 0.2（Stage 0 完了・再固定案）
**作成日:** 2026-09-24（v0.1-draft）／ 2026-09-25（v0.2）
**状態:** **Stage 0 で終了（事前登録した停止規律 1「EXP02 scalar ability と等価」および 2「被覆が凍結床未満」に該当）。Stage 1 以降は実施しない。**
**本番影響:** なし

---

## 0. Stage 0 の結果と終了判定（v0.2 で追加）

| 監査 | 成果物 | 判定 |
|---|---|---|
| 先行研究との数理的・実測等価性 | `PRIOR_ART_EQUIVALENCE_AUDIT.md`, `out/equivalence_audit.json` | **E0 FAIL**。(A) Hodge 射影の出力は馬ごとスカラーで、race-level 確率 arm は EXP02 と同じモデル族（推定量が異なるだけ）。(B) 2022 label-free snapshot で、未対戦 pair の `d_ij − b·Δμ_ij` の分散は EXP02 能力から各対戦を再抽選した null の 0.840 倍（CI95 [0.825, 0.856]、事前固定 FAIL 条件 上限 < 1.10）。(C) pair 証拠の 47% は射影で捨てられ、残りは馬ごとスカラー |
| 時点安全なグラフ構築のデータ監査 | `GRAPH_DATA_AUDIT.md`, `out/race_population.json`, `out/invariant_tests.json` | ID 欠損 0・不正 0・衝突 0。正式 race set 15,951R（EXP16A と年別完全一致）。合成 invariant 17 項目 PASS |
| 被覆率 | `out/GRAPH_COVERAGE.json` | 暫定床 4 本中 **1 本 FAIL**（80% のレースで全 pair の半数以上 covered: 実測 61.7%）。他 3 本 PASS（未対戦 pair の共通対戦馬あり率 53.7%、新馬以外の所属率 95.1%、主要帯比 0.82） |
| 検出力 | `POWER_AUDIT.md`, `out/power_audit.json` | G1 floor 0.001 nats/pair で detect power 1.000（MDE < 0.0005）。**検出力は停止理由ではない** |
| 計算量 | `COMPUTE_DRY_RUN.json` | 1 設定 414 秒、placebo 400 draw で 16 並列 2.9 時間。実行可能だった |

**FAIL 文（§2.4 の範囲限定、これ以外に一般化しない）:**

> 事前固定した時点安全な 2-hop 共通対戦馬表現は、対象母集団・期間・統制の下で必要なペア固有情報を追加しなかった。

**データ構造上の根本原因:** 使用した (h,c) 辺の 78.8%（2022 では 84.7%）が対戦 1 回で、1 回の対戦から作る log-odds は符号 1 ビット。pair 固有の状態を推定する繰り返し観測がデータに存在しない。パラメータ（α・減衰・条件重み）の凍結値をどう選んでも変わらない。

**救済しないもの:** PageRank、3-hop 以上、embedding、GNN、統計力学 M1 型の identity 結合項、地方・海外履歴、他券種。必要なら新しい実験番号と未使用評価計画で行う。

---

## 1. 目的（v0.1 のまま）

次の仮説を、既に終了したElo・Glicko・EXP02・EXP12の言い換えにせず検証する。

> 直接対戦したことがない2頭でも、双方が過去に対戦した共通対戦馬を介せば、その2頭に固有の相対能力情報を時点安全に推定できる。この2-hop証拠を現在の出走馬全組合せについて保持すれば、馬ごとの単一能力値、R0-clean、terminal close単勝市場を超える情報が得られる。

問いを二つに分離する。

1. **機構の問い（G1）:** 共通対戦馬の証拠は、未対戦2頭のどちらが先着するかを予測するか。
2. **実務情報の問い（G2）:** そのペア情報から作るレース確率は、既存動的能力・R0-cleanを統制した後にもterminal close単勝市場を0.005 nats/race以上改善するか。

機構が通っても市場価値を意味しない。terminal close市場を通らなければ、候補馬券や配分へ進まない。

## 2. 先行研究との境界

### 2.1 実施済みで再実験しないもの

- 静的・逐次Elo（多頭数同時更新を含む）
- Glicko-2のペア分解と不確実性
- EXP02の動的ベイズPlackett–Luce能力と条件別部分プーリング
- EXP12の1-hop対戦相手identity集約（degree・年齢・経験量placeboでFAIL）
- field-strength平均、レースレベル等のスカラー集約
- **統計力学 M1 coupling（PL 分配関数への pair 結合項、脚質ベース。umaren ECE −33%・実馬券回収ゼロ）** — v0.2 で追記。pair 固有項を race-level 確率へ運ぶ唯一の構造で、既に検定済み

### 2.2 等価性の罠（v0.2 で E0 として形式化）

各馬に単一の`theta_h(t)`だけを推定し `P(i > j) = σ(theta_i − theta_j)` とするなら、共通対戦相手による推移比較はEXP02の能力値に既に内包される。

E0 の判定式:
- **E0-A**: 履歴 log-odds 行列 L = grad(θ) + R のとき `d_ij = (θ_i − θ_j) + Σ_c ω (R_ic − R_jc)/Σ_c ω`。Hodge 射影は d の gradient 成分を s に、curl 成分を残差に分ける。**s は常に馬ごとスカラー**。
- **E0-B**: ρ = Var_real(d_ij − b·Δμ_ij) / E_null[Var]、null は EXP02 μ から各対戦を再抽選。FAIL if CI95 上限 < 1.10。
- **E0-C**: curl share（射影で捨てる pair 証拠の割合）を報告。

### 2.3 EXP17でのみ検証しようとした部分

対象ペア`(i,j)`ごとに、共通対戦馬のidentityと履歴から`d_ij(t)`を作り、現在レースのペア行列として保持した後、Hodge射影で整合する馬スコアへ変換する。主実験は2-hopまで。**Stage 0 の E0-A により、この「射影で変換する」設計は保持した pair 固有成分を射影の瞬間に捨てることが確認された。**

### 2.4 FAILの範囲

> 事前固定した時点安全な2-hop共通対戦馬表現は、対象母集団・期間・統制の下で必要なペア固有情報を追加しなかった。

PageRank、3-hop以上、embedding、GNN、地方・海外履歴、新規データ、他券種、すべての関係モデルが失敗したとは書かない。

## 3. 母集団とID

### 3.1 馬ID

- 主キーは`血統登録番号`（10 桁数字の文字列）。馬名joinは禁止。
- 欠損・不正IDは件数を記録して除外し、馬名から補完しない（実測: 欠損 0・不正 0・衝突 0、同名別馬 261 件）。
- `graph_core.validate_hids` が形式違反・同一 ID 複数馬名で fail-closed。

### 3.2 正式race set（2019〜2023 実測 15,951R）

- JRA平地のみ。`トラックコード(JV)` 51〜59を除外（2013〜2023 で障害 1,394R）。
- 公式勝馬が一意（同着 37R 除外）。
- 有効starterが5頭以上（1R 除外）。
- 取消・除外は master に無いので自然に除く。
- DNF を含むレースは正式 race-level Gate から除く（proxy: master 行数 < 出走頭数、656R 除外）。結果条件付きの暫定処理。
- 同着した2頭間のpair labelは作らない。
- 年別: 2019 3,185 / 2020 3,169 / 2021 3,200 / 2022 3,206 / 2023 3,191（EXP16A と一致）。

### 3.3 履歴グラフ

- 2013 年以降の JRA 平地 finisher 行。対象日の開始時点より前（date < day）のみ。
- 同日の全レースは同じday-start snapshot（同日先行レースの結果も使わない）。
- edge方向は公式最終着順だけ。オッズ・払戻・ROIを使わない。
- 履歴レースの DNF 馬は master に無いため辺を持たない（記録のみ）。

## 4. 数理定義と凍結パラメータ

### 4.1 過去の2頭間証拠

`p_hc = (W_hc + α)/(W_hc + L_hc + 2α)`、`ell_hc = logit(clip(p_hc, eps, 1−eps))`。同着は W にも L にも加えない。

**Stage 0 で使用した暫定値（Stage 1 に進む場合は ≤2018 で下記格子から凍結する予定だった）:**

| パラメータ | Stage 0 値 | 探索格子（≤2018 のみ） |
|---|---|---|
| α | 1.0 | {0.5, 1, 2} |
| eps | 0.01 | 固定 |
| 半減期 | なし | {∞, 730 日} |
| 条件一致重み cond_bonus | 0 | {0, 0.5} |
| lookback | なし | {∞, 1095 日} |
| Hodge λ | 1e-6（一意性のみ） | 固定 |
| 射影に直接対戦 pair を含める | しない | {しない, 含める} — 未解決事項として記録 |
| ω_ijc | 調和平均信頼度 n_ic·n_jc/(n_ic+n_jc) × (1 + cond_bonus × 条件一致率) | — |

### 4.2 共通対戦馬を介した間接比較

`d_ij^(c) = ell_ic − ell_jc`、`d_ij = Σ_c ω_ijc d_ij^(c) / Σ_c ω_ijc`。`d_ji = −d_ij`。証拠なし pair は missing（0 に置換しない）。主機構検定は直接対戦歴の無い pair のみ。

### 4.3 現在レース内のHodge射影

`min_s Σ_(i<j) w_ij (s_i − s_j − d_ij)² + λ Σ s_i²`、連結成分ごとに Σ s_i = 0。uncovered 馬は s=0 + `graph_uncovered=true`。`p_graph(i) = softmax(s/τ)`。

**v0.2 注記:** この射影は最小二乗射影であり、d の非 gradient 成分（pair 固有成分）を残差として捨てる。合成テスト（純 gradient 入力 → s=θ、純 curl 入力 → s≡0）で確認。

## 5. Stage 0 — 必須監査（実施済み）

### 5.1 数理・実測等価性 → `PRIOR_ART_EQUIVALENCE_AUDIT.md`（E0 FAIL）

### 5.2 時点安全性の必須テスト → `test_invariants.py`（17 項目 PASS）

1. 未来年削除で bit 一致 2. 対象レース結果改変で不変 3. 同日 snapshot 共有 4. 共通対戦馬の対象日後成績が入らない 5. ID 衝突 fail-closed 6. 不正 ID fail-closed 7. 馬名補完なし 8. antisymmetry（0.0） 9. 馬順置換不変 10. 行順置換不変 11. Hodge 再現性 12. 確率和=1（1.1e-16） 13. 非連結成分と uncovered 馬 14. 主 sample に直接対戦 pair 混入なし（非空） 15. missing を 0 にしない 16. 推移的入力で s=θ 17. curl 入力で s≡0

### 5.3 被覆率監査 → `GRAPH_DATA_AUDIT.md`

| 床（v0.1 で暫定固定、v0.2 で変更せず） | 実測 | 判定 |
|---|---|---|
| 未対戦pairの50%以上に共通対戦証拠 | 0.537 | PASS |
| 80%以上のレースで全pairの50%以上がcovered | **0.617**（未対戦のみで数えても 0.591） | **FAIL** |
| 新馬以外のstarterの90%以上が履歴graphに属する | 0.951（2 歳 0.767、出走 1〜5 は 0.885） | PASS |
| 主要芝ダ・頭数帯の被覆が全体の半分未満へ落ちない | 最小帯比 0.82 | PASS |

床は成績を見た後に下げない。停止規律 2 に該当。

### 5.4 検出力監査 → `POWER_AUDIT.md`

- 推論単位は**開催日（暦日）**（EXP16A の場×日より保守的）。同一レース内pairを独立標本にしない。
- label-free effect injection: G1 floor 0.001 nats/pair で detect power 1.000（Wilson 下限 0.990、400 rep、seed 20260925）、MDE < 0.0005。
- **G1 機構 floor = 0.001 nats/pair**（Stage 0 で固定）。race-level 実務床 = 0.005 nats/race（EXP16A 継承）。
- 等級は EXP16A 同型: 検出（CI95 上限 < 0 ∧ 安定性 ∧ placebo）と floor 到達（CI95 上限 < −floor）を分離。点推定は等級条件に使わない。

## 6. Rolling評価（未実施・設計のみ）

- development 2019〜2023、2024/2025 封印。評価年 Y の graph は対象日より前、ハイパーパラメータ・較正・τ・subset 境界・結合係数は Y より前だけで決める。2022 で決めた値を 2019〜2021 へ遡及適用しない。
- pooled CI は year-stratified 開催日 bootstrap。各年、pooled、leave-one-year-out、seed 別を報告。retrospective crossfit であり未使用 holdout ではない。
- graph 構築と Hodge は決定論的（seed 不要）。seed は M2_R0 の rolling OOF と placebo 抽選のみ（5 seed、median、4/5 方向一致）。

## 7. 比較arm（未実施・設計のみ、v0.1 のまま）

| arm | 定義 |
|---|---|
| `M0_CLOSE` | de-vig terminal close単勝市場 |
| `M1_DYNPL` | EXP02相当のscalar dynamic PL |
| `M2_R0` | rolling OOF R0-clean 111列 |
| `M3_BASE` | close offset + M1 + M2 + graph構造・交絡control（出走数、休養日数、degree、pair/共通対戦馬被覆、component size、missing flag、EXP02 能力平均・不確実性、R0-clean 確率、市場 entropy、頭数） |
| `M4_PATH` | close offset + common-opponent射影score |
| `M5_FULL` | M3 + path score + 事前固定したpath不確実性 |
| `P_PLACEBO` | degree・年齢・経験量を保ってidentityをrewireしたM5 |
| `PRE_FULL` | historical_pre_snapshot offset + M5の非市場項 |

## 8. Placeboと負の対照（Stage 0 で具体化、`POWER_AUDIT.md` §5）

- **P1 degree-preserving identity rewire（主）**: 辺属性を保持し相手 identity のみをセル（年 × 芝ダ × 頭数帯 × 年齢帯 × 出走数帯）内で degree 保存復元抽出。200 draw、seed 20260925+draw、97.5 percentile。
- **P2 direction null**: 各辺の方向を確率 0.5 で反転、endpoint と被覆は不変。200 draw。
- **P3 identity-free control**: d_ij を degree 差・出走数差・休養差・直近性・成分サイズの Y−1 fit 関数で置換。
- real 改善は P1・P2 の 97.5 percentile を共に超え、かつ P3 を超えること。
- EXP12 との差: EXP12 は馬行集約値の CLT 近似置換（mean 系 3 特徴、1,000 draw）。EXP17 P1 は辺リスト上の真の identity 再抽選で、辺年齢・W/L・条件一致も保持し、共通対戦馬構造だけが壊れる（control で吸収）。

## 9. 指標とGate（未実施・設計のみ）

### 9.1 G1 — 機構Gate
対象 = 現在同走・過去に直接対戦なし・共通対戦馬あり pair。主指標 = binary pairwise logloss（race 内平均 → 開催日単位）。
検出: `PATH − DYNPL` の CI95 上限 < 0、5 年中 4 年、5 seed 中 4、LOO 全て CI95 上限 < 0、全 placebo 97.5 percentile 超え。floor 到達: CI95 上限 < −0.001。

### 9.2 G2 — terminal close残差Gate
G1 検出時のみ。`M5_FULL − M3_BASE` の race-level categorical winner logloss。等級 PASS-PRACTICAL（CI95 上限 < −0.005）／PASS-SIGNAL（< 0）／FAIL。安定性・placebo 条件は G1 と同じ。PASS-PRACTICAL だけが別途レビューの実務 Stage を提案できる。

### 9.3 G3 — 情報時点診断
`PRE_FULL` は凍結 run 条件を満たす場合だけ。G2 を救済しない。

### 9.4 副指標
multiclass Brier、calibration intercept/slope、adaptive-bin ECE、top1/top3/NDCG@3、pairwise AUC（記述のみ）、帯別被覆と誤差、直接/間接 pair の分離。副指標で主 Gate を救済しない。

## 10. 停止規律（該当したもの: 1, 2）

1. **EXP02 scalar abilityと代数的・実測上等価 — 該当（E0 FAIL）**
2. **被覆が凍結床未満 — 該当（80% レース床 0.617）**
3. 未来edgeまたは削除不変性違反 — 非該当（17 項目 PASS）
4. 機構floorへのpowerが80%未満 — 非該当（1.000）
5〜9. G1/G2/placebo/交絡/小標本 — 未到達

停止後にlookback、条件セル、母集団を結果に合わせて変えたり、PageRank、長いpath、embedding、GNN、identity 結合項を救済実装したりしない。

## 11. Stage別の許可範囲（Stage 0 で実施したもの）

実施: read-only 先行研究監査、label を見ない被覆・連結性監査、合成テスト、power simulation、計算量・容量 dry-run、2022 label-free 等価性監査（対象レース着順は不使用、履歴対戦と EXP02 pre-race μ のみ）。
未実施: 2019〜2023 の成績評価、2024/2025 成績、ROI・払戻、production 変更、候補馬券生成、資金配分、PageRank、3-hop 以上、embedding、GNN。

## 12. 成果物（Stage 0、すべて本ディレクトリ）

`PRIOR_ART_EQUIVALENCE_AUDIT.md`、`GRAPH_DATA_AUDIT.md`、`out/GRAPH_COVERAGE.json`、`POWER_AUDIT.md`、`out/power_audit.json`、`COMPUTE_DRY_RUN.json`、`test_invariants.py` + `out/invariant_tests.json`、`SPEC.md`（本書 v0.2）、`spec.json`（v0.2）、`README.md`、コード `graph_core.py` / `coverage_audit.py` / `equivalence_audit.py` / `power_audit.py` / `stage0_checks.py`。

## 13. 凍結規則

v0.2 は「Stage 0 完了・終了」の版として commit hash とともに記録する。以後、本実験番号での Gate・閾値・母集団・arm・placebo・停止規律の変更は行わない。再開は新しい実験番号と未使用評価計画でのみ行う。
