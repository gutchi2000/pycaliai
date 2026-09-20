# EXP08 Stage 0 — 先行研究・既存実装監査

**作成日**: 2026-09-20夜　**方法**: リポジトリ全体をキーワード検索(`track bias`, `馬場バイアス`,
`内外`, `前残り`, `差し`, `逃げ`, `脚質有利`, `same day`, `当日傾向`, `track variant`,
`going allowance`, `online update`, `Kalman`, `state space`, `change point`, `race-day`,
`開催日` 等)し、ヒットしたファイルを実際に読んで検証した(メモリの要約だけに頼らず、
一次ソースのコード・レポートを直接確認)。

## 総合判定: 重複部分が大きい。固有仮説は残るが、当初スコープの一部(枠=内外)は
## 既に決着済みの死亡ルートであり、再走に当たる。詳細は§6「推奨」参照。

**【2026-09-20夜追記】** Stage1完了後にユーザー指摘で、結果利用可能時刻の根拠
(TANPUK確定オッズタイムスタンプ)が誤りと判明し、DATA_AUDIT.md §5.2を全面訂正した
(詳細は同ファイル参照)。本ファイル(Stage0の先行研究監査)自体はこの訂正の対象では
なく、内外/脚質の除外判断を含む結論はすべて有効。

## §1 表示専用の既存バイアス機能(モデル特徴量ではない)

| ファイル | 内容 | モデル入力として使用されているか |
|---|---|---|
| `build_baba_bias.py` | クッション値/含水率 × 脚質・枠 の**13年プライア**バイアス表(`data/baba_bias.json`) | ❌ 表示のみ(`explain_race.py`/`jusho_protocol.py`) |
| `build_realized_bias.py` | TARGET結果CSVから会場×面ごとの**実現**前残り率(脚質)・内外(枠)を算出(`data/realized_bias.json`) | ❌ 表示のみ(`build_site.py`→出走表タブ) |
| `fetch_baba_today.py` | クッション帯→13年プライア早見 | ❌ 表示のみ |

`train_unified_rank.py`/`optuna_v6_marks.py`/`predict_weekly.py`/`export_weekly_marks.py`/
`compute_bets.py`のいずれもこれらのJSONを読んでいない(grep で確認済み)。**「同日先行
レース情報がv6+市場の後に予測精度を改善するか」という問いは、これらの既存機能では
一度も検証されていない**(人間向けの参考表示のみ)。ここはEXP08の固有領域として残る。

## §2 raw-ROIベースの同日/翌日バイアス検定(2件、いずれもROI軸、logloss軸ではない)

### `analysis/daybias_within_card_test.py`(同日within-card版)
- 手法: 同日同場同surfaceで対象レースより前に発走した完走済みレースの
  `front_overperf`(前が3着内割合−前が出走割合)を単純累積平均した`asof_bias`を作り、
  「期待前型馬」(各馬の過去走からのserve-safe事前脚質)の複勝率・複勝ROIを
  帯別に比較。
- **v6/市場の予測確率で残差化していない**(生の複勝率・複勝ROIを直接見る)。
- 結論(ファイル内コメント、`crossday_bias_test.py`からの引用): **「死亡, 78.4%」**
  (複勝ROIが控除率を超えない)。
- EXP08との差: EXP08§6は「v6+市場からの期待値残差」で信号を作ることを必須としており、
  このテストはそれをしていない。**raw ROIでの死亡は、v6残差化後のlogloss改善が
  ゼロであることを意味しない**(本プロジェクトの他の多くの例と同型: 判別力はあるが
  市場に織り込まれておりROIにならない、[[project_umami_vs_marks]] [[project_aite_confidence_gate]])。

### `analysis/crossday_bias_test.py`(土→日のクロスデー版)
- 土曜の確定済み全レードから前残り度を集計し、翌日日曜の同場同面レースへの
  持続性(Spearman/Pearson)・符号反転率・複勝ROIを検定。
- 同じくraw ROI軸。`VERDICT`ロジックは`beats_dead(78.4)`を明示的な比較基準として持つ
  (within-card版の「死亡」ラインを再利用)。
- EXP08の対象は**同日within-card**(§1「後続レース」)であり、クロスデー版は隣接だが
  別仮説。参考情報として有用だが直接の重複ではない。

## §3 v6残差化ずみ・rigorous版: `analysis/day_state_counting.py` → `_stage2.py` →
## `day_waku_z_forensics.py`(2026-08-31、別セッションの未コミット作業、未修正で参照のみ)

**重要**: これはEXP08が要求する「v6(p_sho)を期待値とした残差の当日累積」を実際に
実装し、時系列リーク境界(「あるレースのday_*_z特徴は、そのレースより前に確定した
レースのResidualだけから計算」を素朴なforループで1レースずつ処理、全レース先に
集計してからshiftする実装は明示的に禁止)を守った上で、2024年fit→2025年完全holdout
のOOS評価・permutation placebo・cross-day placebo・null feature比較・bootstrap CI・
leave-one-track-out まで行った、**EXP08のGate 1/2A/2Bに相当する検証をほぼ実施済み**の
先行研究。EXP01-08の絶対条件により本セッションはこれらのファイルを一切変更・
コミットしていない(読み取りのみ)。

### Stage 1(4次元をスクリーニング、`day_state_counting.py`)
| 次元 | 判定 | 理由 |
|---|---|---|
| 人気帯 | 却下 | カテゴリ間のstatic base-rate差による擬似相関(pooled qcutの罠) |
| 脚質(前残り/差し) | 却下 | OOSで有意な逐次状態信号なし |
| 市場当日累積(steam) | 却下 | 個別odds moveには情報があるが当日累積状態には予測力なし |
| **枠(内外)** | **Stage2へ進行** | 9-12R限定で単調勾配、10場で同符号(留保: 年別安定性が弱い) |

### Stage 2(枠のみ、`day_state_counting_stage2.py`)
- モデル: `logit(p_new) = logit(p_base) + beta * exposure`、
  `exposure`=馬自身の枠カテゴリ(inner/mid/outer)に対応する当日累積z、
  `p_base`=v6(`reports/marks_v5/*.json`のp_sho、真のOOS期間2024-2025)。
- betaは2024年9-12Rのみでfit、2025年は完全holdoutで再fitなし。
- **2025年OOS主結果(9-12R, n=15,796)**: ΔLogLoss **+0.00050**、ΔBrier +0.00017、
  ECE改善(0.0059→0.0044)、AUC微増。**bootstrap 95%CIは[+0.00013,+0.00088]で0を
  跨がない**(統計的に有意)。LOTO(場除外)でも符号反転なし。quarter別もほぼ安定。
  9条件チェックのうち**8/9クリア**。
- **しかし決定的な反証テスト(item13, permutation placebo)で失格**:
  「real(day_waku_own_z)のΔLogLoss +0.00050」に対し、「同日同カテゴリ内で
  どの馬にどのz値を割り当てるかをシャッフルしたplacebo」のΔLogLossは**+0.00089**
  ―― **placeboの方がrealを上回る**。これは「レースを重ねるほど情報が蓄積する」
  という当日オンライン更新の中核前提と矛盾する。
- **最終判定: 却下**(「競馬版カードカウンティングとして成立したか: NO」)。

### Forensics(day_waku_zの正体調査、`day_waku_z_forensics.py`)
5種の反証テスト(§3成分分解、§6-8線形コントロール、§9同日内permutation 500回、
§10別日permutation 500回、§11 null feature比較)を積み上げた結論:

> **day_waku_zが運ぶ情報は「特定の日・特定の競馬場・特定の馬場状態・特定の枠
> カテゴリ」という組の水準でのみ意味を持つ実現バイアスであり、そのカテゴリに
> 属する馬なら日内のどのレースの馬に紐付けても(シャッフルしても)ほぼ同じ実現
> バイアスを代弁できる ── つまり「レースを重ねるごとに情報が蓄積していく」
> というカードカウンティングの中核主張とは異なる性質の変数だった。**

言い換えると: 信号自体は本物(null feature比較・cross-day placeboで確認)だが、
**その正体は「その日・その馬場は(何らかの理由で)内/外に偏っていた」という
group-levelの実現トラックバイアスであり、時系列順に逐次更新するKalmanフィルタ的な
構造には特有の価値がない**。同日の「後半になるほど情報が増える」という
EXP08§8のオンライン状態モデルの根拠そのものに、この最も近い先行研究は
反証を突きつけている。

## §4 EXP03との数理的重複チェック

`analysis/mcond/exp03_latent_state_dev/latent_state.py`は個々の**馬**の長期能力`a`と
短期状態`s`をPlackett-Luce尤度・Kalman型分散配分で filtering する(「同日: その日の
全レースを前日までの状態で予測し、日の終わりにまとめて更新」)。EXP08が推定する
`z[開催日,競馬場,芝ダ,時刻]`(**馬場**の潜在状態、日内で逐次更新)とは、
(a)状態の帰属主体(馬 vs 馬場)、(b)更新粒度(日1回 vs 日内逐次)の両方が異なる。
**数理的重複は限定的**(Kalman型の分散配分という技法の型は共通するが、対象・粒度が
別)。EXP08の主張通り「別仮説」として扱ってよい。

## §5 未検証の次元: 時計・上がり・ペース

§3のday_state_counting Stage 1は「人気帯・脚質・市場steam・枠」の4次元のみを
対象とし、EXP08が挙げる「時計の速い/遅い馬場」「上がり性能」「ペース」の
同日オンライン推定は**いずれの先行研究にも見つからなかった**。§1の表示専用
`build_baba_bias.py`はクッション値・含水率という**外部センサ値**(TARGET提供の
馬場情報)を使った13年プライアのみで、**当日の先行レース結果から時計バイアスを
逐次推定する試みは皆無**。ここがEXP08で唯一、真に手つかずの領域。

## §6 推奨(Stage 0終了時点の判断)

仕様書§4「既に同等の時点安全な実験が存在する場合は、重複部分を明示し、固有仮説が
残らなければ終了する」に基づき、以下を提案する(結果を見て後付けした基準ではなく、
Stage 1着手前の設計判断として)。

1. **内外(枠)バイアス次元は対象から除外する**。`day_state_counting_stage2.py`が
   v6残差化・厳密な時系列リーク境界・2024fit/2025holdout・bootstrap CI・
   permutation placeboまで揃えた検証で明確に却下しており、同じ問いを別実装で
   再度尋ねることは「同一仮説の再走」に当たる([[project_dr01a_erratum_policy]]の
   「同仮説の複数回目はP3降格」原則に準ずる)。
2. **前残り/差し(脚質)次元は優先度を下げる**。v6残差化版のStage 1では「OOSで
   有意な逐次状態信号なし」と却下済みだが、これはスクリーニング目的の簡易検定
   (枠のような9条件フルチェックは受けていない)。raw ROI版2件も死亡/priced。
   **完全に新規ではないが、枠ほど decisively closed ではない**。EXP08で扱うなら
   探索的位置づけに留め、Gate 2Bまで進める主軸には据えない。
3. **時計・上がり・ペース次元を主軸にする**。先行研究が存在しない、EXP08で
   唯一の真に未検証な領域。
4. **Gate基準に permutation placebo test を追加する**。EXP08の現行spec(§13時点
   安全テスト12項目)には「同日内で観測をシャッフルしても同等以上の改善が
   出ないか」を直接確認する項目がない。day_state_counting_stage2が正にこの
   テストで最有力候補(枠)を沈めており、**これを欠いたままGate 2Bへ進むと、
   day_waku_zと同じ「real < placebo」の落とし穴を見逃すリスクが高い**。
   §13へ「item13: 同日同カテゴリ内で観測をランダム再配置したplacebo state
   のΔloglossが、real stateのΔloglossを上回らない」を追加することを推奨する。
5. 多次元(内外+前残り+時計+ペース)を単一の結合状態ベクトルとして同時推定する
   仕様書§8の設計は、内外を除外し前残りを探索的に格下げすると、実質的に
   「時計・上がり・ペースを中心とした1〜2次元の状態モデル」まで縮小する。
   これならKalmanフィルタの複雑性も相対的に正当化しやすい(次元が少なければ
   RAW/EWMAとの差が出やすい)。

**固有仮説は残る(時計・上がり・ペースの同日推定は未検証)ため、Stage 0では
終了しない。ただしスコープを上記の通り絞ることを強く推奨し、続行するか・
スコープを絞るか・中止するかはユーザー判断を仰ぐ。**

関連メモリ: [[project_baba_cushion_tested]] [[project_crossday_trackbias_dead]]
[[project_realized_bias_and_weekly_intake]] [[project_day_state_counting_stage1]]
[[feedback_realized_vs_prior_drawbias]] [[project_umami_vs_marks]]
