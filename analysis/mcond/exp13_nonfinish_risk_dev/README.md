# EXP13 — 出走後の中止・非完走リスク（started_but_did_not_finish）

## ★★★【2026-09-22 最終終了】Gate 0で終了（ユーザー承認・確定）

> **historical_pre_snapshotは存在したが、full-starter再構築で19特徴の非parity
> が判明し、market-complete母集団のselection 2022・development 2023はいずれも
> 事前EPV基準を下回ったため、S0〜S5を実装せず終了した。**

**この結論はモデル性能FAILを意味しない**——S0-S5のモデル実装・学習・比較は
一度も行っていないため、性能についての判定自体が存在しない。終了理由は
データ品質（19特徴非parity）とサンプルサイズ（EPV不足）の2点のみ。

**次のステップはEXP14 Stage1ではない**。19特徴のtrain/serve/offline parity
問題を独立したP0監査として実施する（`analysis/mcond/
p0_dnf_history_parity_audit/`、EXP13・EXP14いずれの継続でもない）。
詳細は`spec.json`の`final_verdict_20260922_gate0_reopen_closure`参照。

## ★★【2026-09-22 再開・評価完了（経緯として保持）】Gate 0Bから再開、Gate 0B/0Dともユーザー基準未達

ユーザー承認によりGate 0Aを訂正PASSとし、同一実験番号のままGate 0Bから
再開・評価した。詳細・全数値は`GATE0B_GATE0D_REOPEN_REPORT.md`参照。

**要旨**: 120特徴のうち**19特徴（course_n_prev/jockey_n_prev系6 +
kako5系13）で実測非一致を確認**（該当行の最大1.8%、raw v6 score差は
確認できた範囲で平均0.022・最大0.326）。原因は完全特定済み（dropna後
母集団での位置/カウントベース集計、jockey/trainer_fukuバグ[[P0-5]]と
対称の未修正issue）。全starter確率再計算では完走馬の確率が平均0.68pt・
最大16.4pt変化。Gate 0Dは市場snapshot必須のcomplete-case制限で
2022年selection EPV=8.92・2023development EPV=9.77となり目安下限(10)を
下回る。**Gate 0B・0Dともユーザー事前基準「通過」とは言えず、S0-S5へは
進んでいない**。モデル学習・2024/2025年開封・ROI評価は未実施。
決定はユーザーに委ねる。

副次的発見（EXP13の範囲外）: 上記19特徴の不一致は本番`master_v2`
（v6学習データそのもの）にも実在する未修正issue。

## ★【2026-09-22 同日訂正・旧記録】Gate 0Aの「市場snapshot 0%」は誤りだった

`docs/research/MARKET_DATA_PROVENANCE_AUDIT_20260921.md`（横断provenance
監査）により、Gate 0Aの結論を訂正した: `data/Time _series_odds/TANPUK_*.csv`
（2011-2025年）由来の`historical_pre_snapshot`（発走26-30分前、中央値28分）
が実在し、2023年flatで正常完走馬91.0%・止(DNF)87.6%のcoverageがある
（`exp05_design.parquet`という既存artifact経由、EXP01/04/05/06/07/08/09が
既に使用中だった）。この訂正を受けて上記の通りGate 0Bから再開した。

---

## 【2026-09-22】最終結論: Gate 0でデータ不足終了（2024・2025年は未開封）

> **EXP13は非完走リスクモデルの性能FAILではない。2013〜2025年の結果ラベルと
> 事前特徴は存在するが、development期間である2023年に判断時点市場snapshotが
> 存在せず、v6＋市場を統制した主仮説を時点安全に検証できないため、Gate 0で
> データ不足終了した。**

Gate 0A（市場オッズ取得時点）で、development期間である2023年に判断時点
market snapshot（historical_pre_snapshot）が正常完走馬・止馬ともに構造的に
0%であることが確定した。確定オッズ・最終人気・結果ファイル内オッズの代用は
行わない。市場統制なしのS0/S2/S3/S4比較は予測可能性の探索にはなっても、
主仮説「v6と市場の後にも非完走リスク情報が残る」を検証できないため、
**S1以降は実装しない**。Gate 0D（2023年145陽性・EPV約11・事前効果量なし）は
単独では即時中止理由ではないが、Gate 0Aの構造的欠損と合わせて継続根拠が
不足すると判断した。**結果を見てS4の特徴を削減したり比較条件を緩めたりする
ことは行っていない**。

**結論の範囲は上記一文に厳密に限定する**。以下は**未検証**（失敗ではない）
として記録し、EXP13への追加・再実行は禁止、実施する場合は必ず
`spec.json`の`reopen_conditions`を全て満たした上でEXP13として再開する:
非完走リスクの予測可能性／過去非完走履歴の価値／履歴変動・大敗・馬体重履歴
の価値／LightGBMによる非完走モデル／binary hazard model／経済的な見送り価値。

詳細は`GATE0_REPORT.md`・`spec.json`参照。

### Gate 0の最終状態（詳細は`GATE0_REPORT.md`・`spec.json`）

- **Gate 0A（市場オッズ時点）**: **FAIL（構造的制約）**。2023年には判断時点の
  市場オッズスナップショット（historical_pre_snapshot）が構造的に存在しない
  （EXP05-F/T-10の前向き価格収集基盤は2026年新設のため）。確定オッズの代用は
  行わない方針。
- **Gate 0B（全starter特徴再構築）**: **`architecture feasible / numeric
  parity unverified`**（既存raw CSVは非完走行を含んだままJOINされており、
  v6の120特徴にオッズ・人気は含まれない等、アーキテクチャ上の実現可能性は
  確認済み。raw score一致の数値照合はGate 0Aで主検証が成立しないため今回は
  実行せず、EXP13再開時の未完了タスクとして記録する）。
- **Gate 0C（結果コード完全表）**: 完了。全36 unique値を`LABEL_CODEBOOK.md`
  に掲載、失格・タイムオーバーは本データソースに存在しないコードと確定。
- **Gate 0D（標本数・検出力）**: 境界線上（単独では中止理由に至らず）。
  2023年flat 145陽性/46,252started。S4想定のEPV(events per variable)は
  約11で目安の下限付近、厳密な検出力保証は効果量の事前情報がないため不可能。

## 再開条件

EXP13を再開できるのは、次を全て満たした場合だけである（`spec.json`
`reopen_conditions`と同一）:
判断時点market snapshotを持つ十分な年数／完走馬・非完走馬の両方を含む
full-starter特徴／v6 raw scoreのnumeric parity／非完走陽性数の事前power
analysis／train/development/OOSを分離できる期間／確定オッズを使わない設計。

## 再利用可能な資産

`LABEL_CODEBOOK.md`（結果コード完全表・主ラベル定義、2つの独立ソースで
完全一致確認済み）／`result_code_audit.py`（結果コード監査+Gate0D集計の
再現可能スクリプト、読み取り専用）／`PRIOR_ART_AUDIT.md`・`DATA_AUDIT.md`
（先行研究・時点安全性監査）／`MINIMAL_FALSIFICATION_PLAN.md`（S0-S6設計案・
Gate構造）。

## 目的

予測対象は「**馬券購入後、競走を開始した馬が正常に完走しないリスク**」。
取消・競走除外・発走除外・出走取消・返還対象（＝馬券購入前に判明する事象）は
主目的変数から明確に分離する。落馬・故障・失格・タイムオーバー・降着を
理由別に区別することは、データに理由コードが存在しないため**行わない**
（`LABEL_CODEBOOK.md`§1参照）。

**主候補ラベル**:
```
started = 1[着順コード ∈ {数値, 止(中止), 丸数字(降着)}]
started_but_did_not_finish = 1[着順コード = 止]   （母集団: started=1のみ）
```

理由コードがないため、**「故障予測」ではなく常に「非完走リスク（non-finish
risk）」と表記する**。

## EXP10との違い

EXP10（大敗・中止・能力未発揮リスク）は**完走馬限定**のcatastrophic
downside検定で、2026-09-21にStage1.5で完全終了した（詳細は
[[project_exp10_downside_risk]]）。EXP13はEXP10が明示的にスコープ外とした
「DNF・中止・取消・除外の別母集団」を対象とする**独立した新規実験**であり、
EXP10の蒸し返しではない。母集団が根本的に異なる（EXP10=完走馬のみ、
EXP13=出走馬全体で完走/非完走を目的変数にする）。詳細は
`PRIOR_ART_AUDIT.md`§3参照。

## 絶対条件

- EXP01-12・v6・compute_bets.pyは変更しない。新規作業は
  `analysis/mcond/exp13_nonfinish_risk_dev/`へ限定する。
- 「負けた馬=陽性」という設計は禁止。開発は2023年データのみ、2024・2025年は
  本Stage0を含め開封しない。
- 「非完走リスク」を「故障予測」等の理由付き予測として喧伝しない（理由コード
  がないため）。
- 障害競走は主解析（平地）に混ぜない。別集計。
- 既存の時点安全性規律（[[feedback_asof_population_definition]]）を遵守する。

## Stage 0 成果物

| ファイル | 内容 |
|---|---|
| `PRIOR_ART_AUDIT.md` | リポジトリ全体の先行研究検索、EXP10との差分、決済層での既存DNF区別 |
| `DATA_AUDIT.md` | v6母集団との接続（非完走馬への特徴再構築・PLソフトマックス分母問題）、時点安全な候補特徴の監査 |
| `LABEL_CODEBOOK.md` | 結果コードの実測監査（2つの独立ソースで完全一致）、主ラベルの確定定義、年別・条件別の陽性数 |
| `MINIMAL_FALSIFICATION_PLAN.md` | S0-S6比較モデル案・評価方法・Gate構造・経済仮説（設計のみ、未実装） |
| `GATE0_REPORT.md` | Stage1 Gate 0A〜0D報告（市場オッズ時点・全starter特徴再構築・結果コード完全表・標本数検出力）、2026-09-22 |
| `result_code_audit.py` | 結果コード監査+Gate 0D集計の再現可能スクリプト（読み取り専用） |

## Stage 0 主要な発見

1. **主ラベルの正確な定義**: `LABEL_CODEBOOK.md`§2参照。`止`（中止）=陽性、
   `外`/`消`（除外/取消、発走前）=母集団から除外、丸数字（降着）=正常完走
   扱い（走破タイム記録あり、非完走ではない——EXP10の未確認事項をここで
   解消した）。
2. **データソース間の完全一致**: 独立した2つの生ソース（`add_20130105-
   20251228.csv`と外部kekkaファイル）で、2013-2025年の`止`/`外`/`消`件数が
   **1件の誤差もなく完全一致**（2,946/1,212/1,007）。
3. **年別・条件別陽性数**: 平地は年間150-260件規模、陽性率0.31-0.38%
   （芝/ダートで大差なし）。**障害は5.26%と平地の14-17倍**——主解析へ
   混ぜない根拠が実測で確認された。新馬はやや高い陽性率（0.42% vs
   0.34%）、年齢による差は小さい。
4. **v6スコア再生成の可否**: 技術的に可能だが、既存パイプライン
   （master_v2）は非完走行を完全に除去しており、新規の特徴再構築パスが
   Stage1で必要（`DATA_AUDIT.md`§2）。
5. **市場確率の可否**: **確定オッズなら100%保持だが、Gate 0A（`GATE0_REPORT.md`）
   で判断時点スナップショット（historical_pre_snapshot）が2023年には
   構造的に存在しないことが確定した。確定オッズの学習入力・統制変数への
   使用は禁止のため、2023年development dataでは市場確率という特徴自体が
   構築不可能**（前回のStage0時点の「可能」という記述は本Gate0Aでより
   精密化・訂正された）。
6. **特徴の時点安全性**: 候補特徴の大半は既存列または既存列からの導出で
   時点安全に構築可能。**今走馬体重のみ構造的に利用不可**（`master_v2`に
   列自体が存在しない、同日情報のため）。
7. **欠損・選択バイアス**: 非完走馬はv6の学習・評価母集団の外側に構造的に
   存在する。**障害レースはv6が一度も学習・スコアリングしたことがない**
   （新規発見、`DATA_AUDIT.md`§3）。
8. **EXP10との差**: `PRIOR_ART_AUDIT.md`§3の表参照。母集団・データソース・
   結論のいずれも異なる独立実験。
9. **最小反証実験**: 主比較候補S4/S6 vs S2。S6（survival/hazard）は
   正確な時刻情報が無いため「survival」と呼ばず、binary modelの拡張として
   扱う（`MINIMAL_FALSIFICATION_PLAN.md`§1.1）。
10. **必要計算量**: 非完走馬向け特徴再構築パスとPL確率再計算パスの新規実装が
    主コスト。規模はEXP10/EXP11程度と見積もる。
11. **中止条件**: Gate3 full-control FAILで再検定しない、陽性率不安定で
    Gate0停止、障害はS1以降の主解析に含めない（`MINIMAL_FALSIFICATION_PLAN.md`
    §6）。

## 次の一手

**2026-09-22、ユーザー承認によりGate 0でデータ不足終了（本README冒頭の
結論参照）**。S0-S5モデルの実装は一切行っていない。EXP13への追加・再実行は
禁止。再開する場合は上記「再開条件」を全て満たすこと。

関連: [[project_exp10_downside_risk]] [[project_exp13_nonfinish_risk]]
[[project_research_stopline_20260921]] [[project_kekka_ext_data_quirks]]
[[feedback_asof_population_definition]] [[project_jravan_guideline_compliance]]
