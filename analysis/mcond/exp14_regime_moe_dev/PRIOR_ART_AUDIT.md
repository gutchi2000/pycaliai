# EXP14 Stage 0 — 先行研究監査（レジーム分離・Mixture of Experts）

**作成日**: 2026-09-22　**状態**: Stage 0 監査のみ。実装・学習・バックテスト・
2024/2025年性能開封・ROI評価は一切行っていない（既存の過去実験結果の
**引用**のみを行い、新規に2024/2025年を開封する操作は行っていない。
下記§1・§2で引用する`exp_surface_split.py`の結果は本監査以前に実施済みの
既存成果物`reports/exp_surface_split.json`であり、本セッションで新規実行は
していない）。

---

## 1. mixture of experts / regime / cluster / segment / specialist / gating
## model / condition-specific model の既存研究（項目1）

リポジトリ全体を`mixture of experts|regime|gating|specialist|condition-
specific`（英語）・`クラスタリング|クラスタ`（日本語）で検索した。

### 直接該当する既存研究（3件、いずれも「実装済み・厳密検証済み」）

| ファイル/資産 | 手法 | 対象regime | 結果 |
|---|---|---|---|
| `lab/train/train_expert.py`（旧8アンサンブル時代、Phase5+） | **手動条件分割**（ハードsplit学習） | 芝短距離/芝中距離/芝長距離/ダート全距離 | `reports/expert_metrics.json`: 4 Expertのうちturf_mid(芝中距離)のみ僅差で採用(+0.0019 AUC)、turf_short(-0.030)・dirt(-0.005)は不採用、turf_longはサンプル不足でスキップ。**4分の3が失敗** |
| `analysis/exp_surface_split.py`（v6世代、交絡排除設計） | **手動条件分割**（M_pool同一レシピ、学習データのみ芝/ダで分離） | 芝 vs ダート | `reports/exp_surface_split.json`: 芝test ◎top3 M_turf 61.05% vs M_pool 61.36%(**プールが僅差で上**、diff=-0.31pt, CI[-1.51,+0.94]有意差なし)。ダtest M_dirt 60.23% vs M_pool 60.94%(**プールが上**、diff=-0.71pt, CI[-1.91,+0.53])。**分割が有意に勝つケースはゼロ** |
| [[project_summer_specific_model_dead]]（v6世代、季節×地域regime、最も厳密） | **手動条件分割 + サンプル重み付け**（両方試行） | 夏（6-9月）×夏局所開催場（札幌/函館/福島/新潟/小倉） | M_sum(夏専用)が全敗: 夏test ◎top3 60.23% vs M_all(全プール)62.55%(Δ-2.32pt)、**非夏のみで学習したM_nonですら夏testでM_sumを上回る**(61.62%>M_sum)。同データ量のM_allsub(61.54%)もM_sumを上回る＝**regime純度の上乗せはゼロ、むしろ有害**。ハイパラ再チューニング(24grid)を最大限favorしても並ぶだけで上回らない。サンプル重み付け(k=2,4)も無効(valid/test方向不一致 or 両方劣化) |

**3件とも独立した実験設計・独立した年度（train_expertは旧システム、
exp_surface_splitとsummer-specificはv6世代）で収束的に同じ結論に至って
いる: 「手動条件分割によるregime純度の追求は、プール学習に対し優位性を
示さない」**。

### 間接的に関連する既存研究

- `analysis/diag_baba_robustness.py`: v6の◎的中率が馬場状態別（良/稍/重/
  不良）でどう変化するかを診断（regime別**専用モデル**ではなく、**単一
  pooled modelの条件別頑健性**を見る診断）。項目4（v6の条件interaction）
  に関連。
- `_trio_bake_participation.py`の"gating"は馬券参加可否のgating（賭けるか
  否か）であり、モデル出力を混合するgating networkとは別概念。混同しない
  よう注意。
- `analysis/null_policy_2026.py`・`analysis/focus_umatan.py`等の「regime」
  は「期間」を指す一般語であり、モデルのregime分離とは無関係。

### 探索したが該当なしと確認した領域（項目1・2・3、真に未踏査）

以下は関連キーワード検索・コード検索で**該当ファイルが1件も見つからず**、
「条件別モデルを構築した実験」自体が存在しないことを確認した:
- **競馬場（venue）単独**を分離軸とした専用モデル実験
- **クラス（class）単独**を分離軸とした専用モデル実験
- **年齢単独**を分離軸とした専用モデル実験
- **新馬・未勝利専用**モデル
- **重馬場専用**モデル（`diag_baba_robustness.py`は診断のみで専用モデルは
  作っていない）
- **教師なしクラスタリング**によるregime定義（`KMeans`等の検索でヒットした
  10ファイルはいずれも別用途——calibratorのshadow評価・馬券組み合わせ探索
  等——で、特徴空間クラスタリングによるregime定義は存在しない）
- **decision tree gate**・**soft gating network**・**mixture of logistic
  experts**・**mixture of LightGBM experts**・**regime-switching
  time-series model**として明示的に実装された既存コード

---

## 2. 芝ダート・距離・競馬場・クラス・年齢・季節別モデルの有無（項目2）

| 条件軸 | 専用モデル実験の有無 | 結果 |
|---|---|---|
| 芝・ダート | **あり**（§1参照、2件） | 分割はプールに勝たない |
| 距離 | **あり**（`train_expert.py`、旧システム） | 4分の3失敗、1件僅差採用のみ |
| 競馬場 | **単独では無し**。季節×venue subsetの組合せでのみ実施（夏局所開催場） | 組合せでは全敗（§1） |
| クラス | **無し** | 未踏査 |
| 年齢 | **無し** | 未踏査 |
| 季節 | **あり**（[[project_summer_specific_model_dead]]、最も厳密） | 全敗（複数の緩和策も含め） |

## 3. 新馬・未勝利・短距離・長距離・重馬場専用モデルの有無（項目3）

| 条件 | 専用モデル実験の有無 | 結果 |
|---|---|---|
| 新馬 | 無し | 未踏査 |
| 未勝利 | 無し | 未踏査 |
| 短距離（芝短距離のみ） | あり（`train_expert.py`のturf_short） | 不採用（-0.030 AUC） |
| 長距離（芝長距離のみ） | あり（`train_expert.py`のturf_long） | サンプル不足でスキップ（学習未実施） |
| 重馬場 | 無し（診断のみ、専用モデルは無し） | 未踏査 |

---

## 5. 条件別に別モデルを作った過去実験（項目5、§1-3の集約）

**収束的な結論**: これまでに実施された3件の独立した手動条件分割実験
（distance-band/surface/season×venue）は、**いずれもプール学習に対する
有意な優位性を示さなかった**。うち2件（distance-band・season）は複数の
バリエーション（ハイパラ再チューニング・サンプル重み付け・同データ量統制）
を尽くしても結論が覆らなかった。この「data量 > regime純度」という原則は
[[project_summer_specific_model_dead]]で明示的に結論化されている。

**EXP14が新規性を持ちうる範囲**: 上記3件は全て**手動のハード分割**（学習
データを条件でフィルタし、regime内で別々に学習）である。ユーザーが列挙する
7方式のうち、**手動条件分割以外（教師なしクラスタリング・decision tree
gate・soft gating network・mixture of logistic/LightGBM experts・
regime-switching time-series model）は1件も試されていない**。ただし
これらのより高度な手法も、根本的に「異質な予測関係が条件間に存在する」
という前提が成立しなければ機能しない。3件の収束的な失敗は、少なくとも
これまで試された条件軸（芝ダ・距離・季節×地域）については**この前提
自体が疑わしい**ことを示す強い事前証拠であり、EXP14はこれを踏まえた
上で「他の条件軸（クラス・年齢・単独venue）」または「より高度なgating
機構」のいずれか、あるいは両方で新規性を主張する必要がある。

---

## まとめ

1. 手動条件分割によるregime専用モデルは、独立した3実験（distance-band・
   surface・season×venue）全てでプール学習に勝てなかった——「実装済み・
   厳密検証済み」の死亡ルートとして記録する。
2. venue単独・class単独・age単独・新馬/未勝利専用・重馬場専用の各regime、
   および手動分割以外の6手法（クラスタリング〜regime-switching）は
   **真に未踏査**。
3. 「data量 > regime純度」という確立済み原則（項目4のtree interaction
   capacity、`DATA_AUDIT.md`参照）とどう整合させてEXP14を設計するかが
   Stage1着手前の核心的な論点になる。

関連: [[project_summer_specific_model_dead]] [[project_exp10_downside_risk]]
[[project_research_stopline_20260921]] [[project_v6_pastform_dominance]]
