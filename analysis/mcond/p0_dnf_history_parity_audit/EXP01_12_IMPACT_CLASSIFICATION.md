# EXP01-EXP12 影響監査: DNF系19特徴バグ + PL-softmax分母欠落バグ

## ★2026-09-22訂正: 新分類（4区分）— 本節が最新、以下は元の一次調査(証拠として保持)

ユーザー指示により、旧分類(4分類: 影響なし/数値は変わるが結論影響は小さい見込み/
再評価が必要/判定不能)を廃止し、以下の4区分へ再分類する。**実験は再実行していない**、
証拠は下記の元調査（§「監査結果一覧」）と同一。

- **unaffected**: 19特徴・PL-softmaxいずれにも数値的に依存しない、またはGate/主判定
  自体に到達していない。
- **exposed but decisive Gate independent**: 19特徴やPL-softmaxに露出しているが、
  主判定のGateが(a)複数の独立した理由で既に頑健なnullである、(b)効果量が閾値と
  桁違いに乖離している、(c)placebo等の差分設計でreal armとplacebo armの双方が
  同一の統制変数を使うため系統的バイアスが対称に相殺されやすい、のいずれかに該当し、
  露出そのものを理由に自動的な再評価対象とはしない。
- **numerical metrics may shift**: 該当特徴の値がバグの影響を受けるため個別の数値
  (logloss・Brier等)は変化しうるが、主判定のPASS/FAIL自体を覆すほどの近さではない。
- **core conclusion potentially affected**: 主判定のCI境界がほぼゼロに接しており、
  かつバグの影響を受ける特徴が主判定の勝敗を分けた側に直接関与しているため、
  修正後に結論(PASS/FAIL)が反転する可能性を否定できない。

| 実験 | 新分類 | 根拠(訂正点があれば明記) |
|---|---|---|
| EXP01 陣営選択の逸脱 | exposed but decisive Gate independent | M4 vs M2 Δ=-0.000125 [-0.000395,+0.000137]、上位20調教師除外で符号反転する等、複数の独立指標が既に頑健なnullを支持 |
| EXP02 動的対戦能力 | numerical metrics may shift | 主Gate(M3 vs M1、M4 vs M2)のCIはゼロから十分離れ決定的PASSであり主判定は揺るがない。ただし「固有価値の3/4は経験軸(career starts)由来」という探索的な内訳の数値は、`kako5_race_count`等と同じ「masterの行数=経験」という土台を使うため変わりうる |
| EXP03 恒常能力と短期状態の分離 | **core conclusion potentially affected** | Gate2主判定のCI上限が+0.000046とほぼゼロに接した状態でFAIL。土台のM2が経験特徴(career-runs/days-since) |
| EXP04 環境不変特徴選別 | **core conclusion potentially affected** | 「M3(環境選別)がM2に負ける」主比較のCI下限が-0.00006とほぼゼロ。負けた側(M3)に19特徴のうち9列が実際に選択されている |
| EXP05 市場条件付き第2段残差モデル | exposed but decisive Gate independent | Δlogloss=-0.00132 [-0.00184,-0.00078]、CI両端とも0から半信頼幅の2倍以上離れる。レポート自身も探索的評価と明記済み |
| EXP06 Jev意思決定層 | exposed but decisive Gate independent | pooled係数=-0.0226、95%CI[-0.0751,+0.0330]、年度間で符号不一致——既に「ほぼゼロ」という決定的null |
| EXP07 ロバスト・ポートフォリオ最適化 | exposed but decisive Gate independent | 年度間で方向不一致、95%CI[-17.5,+28.5]円、利益集中度87.1%と複数の独立した理由で既に頑健性が否定済み |
| EXP08 当日オンライン馬場状態 | exposed but decisive Gate independent | 実測効果量が要求閾値に対し4〜5桁小さい、placebo不合格。摂動の影響を受ける余地が数値的にない |
| EXP09 Conformal/OOD保証付き見送り | exposed but decisive Gate independent | Gate3 CI=[-0.1121,+0.1717]と広大、既存の非対称性の説明と整合的なnull |
| EXP10 大敗・完走能力未発揮リスク | numerical metrics may shift | **訂正**: EXP10は正常完走馬だけを対象にした研究であり、DNFを目的変数で非イベント扱いした研究ではない。過去DNFがR1の入力特徴(`kako5_std_pos`/`kako5_best_pos`/`kako5_avg_pos`)・B3の`kako5_race_count`から欠落する影響はあるため、一部履歴特徴の数値再現性に注意が必要。ただし8条件中6条件が既に不成立で悪化方向に一貫しており(logloss/Brier/PR-AUC全て悪化)、この一貫したパターンが少数特徴の補正で反転する根拠はない |
| EXP11 階層ベイズ(疎データ) | unaffected | 正式なモデル・Gate判定に到達せず、終了理由は別の確立済みバグ(as-of化していない母集団選択バイアス) |
| EXP12 対戦相手ネットワーク | exposed but decisive Gate independent | **訂正**: decisive placebo FAILを維持(実測改善+0.00037がplacebo平均+0.00045を下回り97.5%ile+0.00057も超えない、coin-flipではない)。統制変数`kako5_race_count`はreal armとplacebo armの双方で同一の(バイアスを含む)`career_band`層別に使われる差分設計のため、系統的バイアスは両腕に対称に乗り相殺されやすい。露出のみを理由に再評価対象へ自動的に含めない |

**新分類の集計**: unaffected=1(EXP11) / exposed but decisive Gate independent=7(EXP01,05,06,07,08,09,12) / numerical metrics may shift=2(EXP02,10) / core conclusion potentially affected=2(EXP03,04)

**実験は再実行していない。** 上記は既存spec.json/README.mdの数値の再解釈のみ。

---

## 元の一次調査（証拠として保持、分類ラベルは上記が最新）


対象の2バグ:
1. **19特徴バグ**: `course_n_prev` `course_win_rate` `course_top3_rate` `jockey_n_prev` `jockey_win_rate` `jockey_top3_rate` `kako5_*`(13列)が、過去にDNF(止)歴のある馬の行で実測不正確(全行の0.5-1.8%、raw v6スコア変化最大0.326)。
2. **PLソフトマックス分母欠落バグ**: `master_v2`から完全に削除されているDNF馬を含めずに、オフライン評価のPLソフトマックス(v6_p3/mkt_p3_pre等)を計算している。DNF馬を正しく分母に戻すと、完走馬のwin/top3確率は平均+0.68pt・最大+16.4pt変化する。

EXP01-04は共通基盤(`analysis/mcond/v6base.py` / `market.py` / `data/_research/mcond/base.parquet`、いずれも`master_v2`から構築)でv6_p3・mkt_p3_preをオフセットとして全モデルに使っており、EXP05-12もこの系譜のデータ・確率(または production `pl_calibrators_v6`)を再利用している。したがって**PLソフトマックス依存(問2)は実質的に12実験全てで「あり」**。差が出るのは(a) 19特徴(問1)を実際にモデル入力・統制変数として使ったか、(b) 主判定の余白(マージン)がこの規模の摂動に対して頑健かどうか、の2点。

## 監査結果一覧

| 実験 | 19特徴を使用したか(具体名) | PL-softmax指標に依存 | 分類 | 根拠(実数値) |
|---|---|---|---|---|
| EXP01 陣営選択の逸脱 | 使用せず(raw_choice/deviation系は独自特徴) | Yes(v6_p3+mkt_p3_pre、Harville、全モデル共通オフセット) | **数値は変わるが結論影響は小さい見込み** | 主要な「否定」結論(M4がM2に固有価値なし)はM4 vs M2 Δ=-0.000125 [-0.000395,+0.000137](2023)で既にCIが0を広く跨ぎ、上位20調教師除外で符号反転(+0.000069)するなど、そもそも「効果なし」という結論自体が複数の弱い指標で重ねて支持されている。経済評価(Gate3)もR1 −1.14pt[−3.3,+1.0]・R2 +0.15pt[−0.4,+0.7]と広いCIで既に有意差なし。0.68pt規模の確率摂動がこの既存の「null」の物語を覆す可能性は低い。 |
| EXP02 動的対戦能力 | 使用せず(T1/T2/ELO/Glickoは独自算出だが、num_updates等の実体はmasterの過去行数＝DNF行を含まない点でkako5_race_countと同じ数え方の欠陥を共有) | Yes | **再評価が必要**(ユーザー指定基準により) | 主判定M3 vs M1(Δ=-0.00074 [-0.00116,-0.00032])・M4 vs M2(Δ=-0.00060 [-0.00085,-0.00034])はいずれも統計的には決定的PASSで、マージン自体はゼロから十分離れている。しかし探索分析で「固有価値の約3/4は出走回数・休養日数という生の事実」と判明しており(REPORT.md 問4)、`raw_career_runs`相当の値はmaster(DNF行削除済み)の行数から数えているため、19特徴のjockey_n_prev/course_n_prevと同じ土台の上に立つ。「career starts」を主軸とする実験であるため、数値自体は大きくは動かなくても、効果の帰属先(能力平均 vs 経験の事実)の内訳がDNF修正後に変わりうる。 |
| EXP03 恒常能力と短期状態の分離 | 使用せず(raw_career_runs/raw_days_since/raw_prev_finはmaster行数・前走行から独自算出、19特徴とは別列だが同じ「masterの行数=経験」という土台) | Yes | **再評価が必要** | 主判定M4 vs M3(短期状態の固有価値)のGate2 CI上限が **+0.000046** で、2023年のCIがほぼゼロに接する形でFAIL(Δ=-0.000167 [-0.000373,+0.000046])。この主判定はM2(=M1+raw_recency_experience、career-runs/days-since等の経験軸)を土台にしており、これらの経験特徴がDNF歴のある馬で過小/過大になっている可能性がある(0.5-1.8%行)。CIの上限がほぼゼロという状態は、本監査が定義する「近い」の典型例。 |
| EXP04 環境不変特徴選別 | **使用(直接)**: 候補145列(C1)に19特徴全て含まれ、環境安定性選別(Gate1)で `course_n_prev` `course_win_rate` `course_top3_rate` `jockey_n_prev` `jockey_win_rate` `kako5_same_td_ratio` `kako5_same_dist_ratio` `kako5_race_count` `kako5_expected_good_count` の9列が最終選択集合(M3の50特徴)へ実際に採用済み。残り10列(`jockey_top3_rate` `kako5_avg_pos` `kako5_std_pos` `kako5_best_pos` `kako5_avg_agari3f` `kako5_best_agari3f` `kako5_same_place_ratio` `kako5_pos_trend` `kako5_hidden_good_count` `kako5_same_cond_best_pos`)はM2/M4/M5の全候補プールには含まれるがM3には非選択。 | Yes | **再評価が必要** | 実験の核心的結論「環境安定性選別は通常学習(M2)や単純top-k(M5)に負ける」を支える主比較M3 vs M2のΔlogloss(2023)は **+0.00033、99%CI [-0.00006, +0.00071]**——下限が-0.00006とほぼゼロに接しており、レポート自身が「CIはほぼ0だが点推定は一貫して正」と明記(REPORT.md #3)。19特徴のうち9列がまさにこのM3(負けた側)に含まれているため、それらの値が0.5-1.8%の行で歪んでいれば、この僅差の符号自体が変わりうる。 |
| EXP05 市場条件付き第2段残差モデル | **使用(直接)**: EXP04のC1候補(145列)をそのまま継承。F-serve(117列)からもDNF系19特徴は除外されていない(除外された28列は前走PCI等、19特徴とは無関係)。よってF-serve/F-fullとも19特徴全てを含む。 | Yes(v6base.py由来のv6_p3・市場、加えて自前isotonic較正) | **数値は変わるが結論影響は小さい見込み** | 主判定M4 vs M3のΔlogloss(2023)=-0.00132 [-0.00184,-0.00078]、2024/2025も同水準で一貫。CIの両端とも0から十分離れており(半信頼幅の2倍以上)、0.68pt規模の確率摂動で符号が反転する可能性は低い。加えてこのレポート自体が「2026年ロック不可のため探索的評価であり最終確証ではない」と明記済みで、結論の格下げは既に織り込まれている。 |
| EXP06 Jev意思決定層 | 使用せず | Yes(m4_top_prob/market_prob等の統制変数、EXP05由来の確率パイプライン) | **数値は変わるが結論影響は小さい見込み** | Gate2 full-control判定: pooled risk_prob係数=-0.0226、95%CI[-0.0751,+0.0330]、2024(-0.0636)と2025(+0.0165)で符号不一致、`sign_consistent_across_years=false`。CIは0の周辺に広く分布し「係数がほぼゼロ」という決定的なnull。0.68pt規模の摂動でここまで広いCIの帰結が変わる可能性は低い。 |
| EXP07 ロバスト・ポートフォリオ最適化 | 使用せず | Yes(較正済みtansho確率＝production pl_calibrators_v6由来、共同着順分布の計算にHarville型手法を使用) | **数値は変わるが結論影響は小さい見込み** | Gate1主判定P6 vs P1: 2024=+15.64円/レース(改善)、2025=-5.88円/レース(悪化)で方向不一致、meeting-day bootstrap 95%CI=[-17.536,+28.516]円(0を大きく跨ぐ)、P(改善>0)=0.6558。加えて利益集中度(P6上位10%が黒字合計の87.1%)から「2024のプラスは少数の大穴依存」と自己診断済み。複数の独立した理由で既に頑健性が否定されており、DNF起因の確率摂動で結論が変わる公算は低い。 |
| EXP08 当日オンライン馬場状態 | 使用せず | Yes(オフセットはv6+市場) | **数値は変わるが結論影響は小さい見込み** | Gate2A/2Bとも要求閾値(絶対改善0.0005)に対し実測効果量が **4〜5桁小さい**(約4×10⁻⁸〜2.6×10⁻⁶)。placebo permutation(1,000回)にも不合格、EWMA対照とも小数第6位まで完全一致。0.68pt規模の確率摂動がこの5桁の差を埋める可能性はない。 |
| EXP09 Conformal/OOD保証付き見送り | 使用せず | Yes(APSはv6のPL全着順確率`pl_probs.all_tansho()`から予測集合を構築) | **数値は変わるが結論影響は小さい見込み** | 決定的関門Gate3: conformal係数=0.0256、95%CI=[-0.1121,+0.1717](0を大きく跨ぐ)。Gate2の優位性(全4参加率×6方式)もmax_prob/entropyに対しては差が-0.01〜-0.09と小さく、OOD/feature_missingに対してのみ-0.10〜-0.68と大きい非対称性があり、Gate3の結論(確信度指標の再表現に過ぎない)と整合的。CIの幅が広大で、結論を左右するほどの摂動ではない。 |
| EXP10 大敗・完走能力未発揮リスク | **使用(直接)**: B3(全モデル共通の統制ベースライン)に`kako5_race_count`(出走回数)、R1(検証対象の主特徴集合5列中3列)に`kako5_std_pos` `kako5_best_pos`・`kako5_avg_pos`。 | Yes(ラベル定義がPL順位分布だが、**対象母集団は完走馬に厳密限定**——後述) | **数値再現性に注意（自動的な再評価対象ではない）** | Stage1.5の判定はR1 vs B3で8条件中6条件不成立、logloss差=+0.00096(悪化)・Brier差=+0.00014(悪化)・95%CI=[-0.00171,+0.00370](0を跨ぐ)。**訂正**: EXP10は正常完走馬だけを対象にした研究であり、DNFを目的変数で非イベント扱いした研究ではない（`catastrophic_downside`はfinisher間の相対的な下方乖離であり、DNFという事象そのものを扱っていない）。過去DNFがR1の入力特徴(`kako5_std_pos`/`kako5_best_pos`/`kako5_avg_pos`)・B3の`kako5_race_count`から欠落する影響はあるため、**一部履歴特徴が影響を受けるため数値再現性に注意**が必要——ただし「catastrophic downsideと機械的に同じバグ」という広い主張はしない。 |
| EXP11 階層ベイズ(疎データ) | 使用(診断のみ): `kako5_race_count`を「career_band」層別変数として使用(low_freq診断の一部)。モデル構築(Stage1項目4-8)は未実施。 | 未実施(v6-市場相関の記述統計のみ、正式なGate/主判定に到達せず) | **影響なし** | 実験を終了させた根拠は「低頻度騎手×調教師ペア」の62%過大予測(v6予測0.059 vs実現0.036)が、全期間集計(as-of化していない)という**別の確立済みバグ**(`feedback_asof_population_definition`)による母集団選択バイアスだったという発見であり、DNF系19特徴の正確性とは無関係に仮説が撤回された。正式なモデル・Gate判定が一度も実行されていないため、動かせる「数値」自体が存在しない。 |
| EXP12 対戦相手ネットワーク | **使用(直接、主軸)**: `confound_controls`(full-control回帰の統制変数)に`career_starts(kako5_race_count)`。placebo検定の層別変数(`career_band`)にも同じ列を使用。 | Yes(v6_probability/entropy等も統制変数、O1診断もv6+市場オフセット上で実施) | **再評価が必要(最優先級)** | 決定的関門(placebo検定)の数値が全実験中で最も僅差: 実測改善=+0.00037、placebo平均=+0.00045(**実測がplacebo平均を下回る**)、placebo 97.5%ile=+0.00057。続行条件②(開催日bootstrap CI上限<0)も**上限=+0.00002**というほぼゼロの僅差でFAIL。confound控除に使う`kako5_race_count`(career starts)自体がDNF歴のある馬で過小算出されうるため、この統制の歪みだけでplacebo平均と実測の大小関係(現在は僅差でplacebo優位)が入れ替わっても不思議ではない規模。 |

## バケット別集計

| 分類 | 件数 | 実験 |
|---|---|---|
| 影響なし | 1 | EXP11 |
| 数値は変わるが結論影響は小さい見込み | 6 | EXP01, EXP05, EXP06, EXP07, EXP08, EXP09 |
| 再評価が必要 | 5 | EXP02, EXP03, EXP04, EXP10, EXP12 |
| 判定不能 | 0 | (該当なし) |

## 優先再評価候補(元のマージンがゼロに近かった順)

1. **EXP12 対戦相手ネットワーク** — **decisive placebo FAILは維持する**（実測改善+0.00037がplacebo平均+0.00045を下回り、97.5パーセンタイル+0.00057も超えていない。これは僅差ではなく決定的な不合格であり「coin-flip」ではない）。統制変数`kako5_race_count`自体がバグの直接対象だが、real armとplacebo armの双方が同一の(バイアスを含む)`career_band`層別変数を使う差分設計のため、系統的バイアスは両腕に対称に乗り相殺されやすい——詳細は本ファイル末尾の再分類(new taxonomy)参照。
2. **EXP10 大敗・完走能力未発揮リスク** — （訂正: 優先再評価候補からは外す）対象母集団は完走馬限定でありDNFを目的変数として扱っていないため、「catastrophic downsideと同じバグ」という広い主張はしない。主特徴3列(`kako5_std_pos`/`kako5_best_pos`/`kako5_avg_pos`)とB3の`kako5_race_count`が過去DNF歴の影響を受けるため、数値再現性の注意点として記録するに留める。
3. **EXP03 恒常能力と短期状態の分離** — 主判定Gate2のCI上限が+0.000046とほぼゼロに接した状態でFAIL。土台のM2が経験特徴(career-runs/days-since)。
4. **EXP04 環境不変特徴選別** — 「環境選別はM2に負ける」を決めた主比較のCI下限が-0.00006とほぼゼロ。負けた側(M3)に19特徴のうち9列が実際に選択されている。
5. **EXP02 動的対戦能力** — 統計的マージン自体は決定的(CIがゼロから十分離れている)だが、固有価値の3/4が「出走回数・休養日数」という経験軸に由来すると探索分析で判明しており、この軸はkako5系と同じ「masterの行数=経験」という土台を共有する。ユーザー指定の基準(経験数を主軸とする実験)により再評価対象に含める。
