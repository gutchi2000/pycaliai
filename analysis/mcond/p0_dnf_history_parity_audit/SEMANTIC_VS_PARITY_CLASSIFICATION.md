# 19特徴: semantic bug と parity bug の分離（A-E分類）

**方法**: `train_serve_same_input_parity.py`で、同一の（意味定義spec準拠の）
入力を training相当ロジックと実際のlive serve関数
（`serve_history_feats.compute_row_feats()` / `parse_kako5.build_from_kako5()`、
いずれも実コードをそのままimportして実行、複製ではない）へ与え、
コードレベルで一致するかを実測した上で分類する。

分類定義:
- **A. train=serveだが両方意味的に誤り**: 同一データソースを参照するため
  train/serve間で値は一致するが、その値自体がDNF_SEMANTIC_SPEC.mdの定義から
  外れている。
- **B. train≠serveで実際のserve skew**: 同一の正しい入力を与えてもコード
  ロジック自体が異なる値を返す。真のserve skew。
- **C. offlineだけ誤り**: featureの値自体は正しいが、オフライン評価
  （PLソフトマックス）が別の理由で誤っている。
- **D. 2026経路だけ別定義**: 2026年に限り、他の年とは異なるデータソース・
  ロジックを使っている。
- **E. 未確認**: コード式は一致する（または一致が確認できていない）が、
  実際に投入されるデータ集合が食い違う可能性があり、実データでの数値一致は
  未検証。

「production bug」という呼称は **B、または現在liveへ実際に影響するA** に
限定する。C・D・Eは現時点でproduction bugと断定しない。

---

## 分類表（19特徴）

| # | 特徴 | training definition | offline replay definition | current live serve definition | desired semantic definition | train/serve parity | semantic correctness | affected years | affected current production path | migration requirement | 分類 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1-6 | course_n_prev, course_win_rate, course_top3_rate, jockey_n_prev, jockey_win_rate, jockey_top3_rate | `build_master_v2.compute_history_features()`: post-dropna(626,774行)母集団でexpanding cumcount。DNFが分母から永続的に欠落 | 該当なし(featureそのもの、PL分母問題とは別軸) | `serve_history_feats.compute_row_feats()`が`data/_horse_history.parquet`参照。2013-2025分は`master_v2`をそのまま再利用 | DNFは分母に含める(経験1回)、外/消は含めない | **PARITY確認済み**(同一入力で全8シナリオ一致、コード式は同一) | **誤り**(両者ともDNF分母欠落) | 2013-2025(train全期間+serve歴史部分) | Yes(v6本番、export_weekly_marks.py経由) | Legacy-compatible: 要修正(実際のserve skewあり)。ただし修正はtrain側データソースの是正が本丸、serve側コードは無罪 | **A** |
| — | (同上、2026年分のみ) | 該当なし(2026はtrain対象外、training masterは2025-12-28まで) | 該当なし | `build_horse_history.py:load_2026_history()`。dropna無し、止はpos=NaN行として残るが外/消も同一"0"コードに潰れ区別不能 | 同上 | 未確認(2026データでのtrain側比較対象が存在しない) | **誤りうる**(外/消の誤混入で過大カウントのリスク、方向がAとは逆) | 2026のみ | Yes(v6本番、2026年の新馬・当年復帰馬の一部) | 実発生規模の定量化が先決、その後の対応要否を判断 | **D** |
| 7 | kako5_race_count | `parse_kako5.build_from_master()`: post-dropna母集団の位置ベース直近5行window。DNFが不可視のため本来より古い走まで遡る | 該当なし | `parse_kako5.build_from_kako5()`(train用`build_from_master()`とは別関数)。DNF/外/消いずれも`_safe_int()`失敗でスロットを無条件skip、代替を探さない | DNFはwindow slotへ含める(結果値は欠損) | **MISMATCH実測確認**(同一入力でtrain相当=3、serve=2等、コードロジック自体が異なる) | **誤り**(両者とも、ただし誤り方が違う) | train:2013-2025全期間 / serve:全期間(kako5 CSV経由) | Yes(v6本番) | 両方に個別の修正が必要(共通コード化ではなく、それぞれ別実装の是正) | **B** |
| 8-10 | kako5_same_td_ratio, kako5_same_dist_ratio, kako5_same_place_ratio | 同上 | 該当なし | 同上 | 分母nにDNFを含める、TD/距離/場所はDNF自身も実値で寄与 | **MISMATCH実測確認**(異種条件混在シナリオでtrain=0.667、serve=1.0) | **誤り**(両者とも) | 同上 | Yes(v6本番) | 同上 | **B** |
| 11-19 | kako5_avg_pos, kako5_std_pos, kako5_best_pos, kako5_avg_agari3f, kako5_best_agari3f, kako5_pos_trend, kako5_expected_good_count, kako5_hidden_good_count, kako5_same_cond_best_pos | 同上(window drift: 誤ったwindow構成でも値計算式自体は正しい) | 該当なし | 同上 | DNF自身の着順・上り3Fは集計に寄与しない(値ベースフィルタ) | **コード式は一致**(同一の有効レース集合が与えられれば同じ値、実測確認)。ただし**実際にtrain/serveへ渡る「直近5走」の実レース集合が食い違う可能性**は未検証(trainはwindow drift方向、serveはTARGET側5列固定フォーマットの打ち切り方向で、原因が異なる) | 誤りうる(train側はwindow drift由来の誤りが既知、serve側は未計測) | train:2013-2025全期間 / serve:未計測 | Yes(v6本番、ただし実害未計測) | 実データでの実際のskew定量化が先決 | **E** |

---

## 「production bug」と呼ぶ範囲（確定）

- **A（course/jockey 6特徴、2013-2025分）**: 実際のlive v6へ影響する、意味的に
  誤った定義。ただしtrain/serve間のコード自体は一致しているため、
  「parityの欠如」ではなく「共有された意味論バグ」。
- **B（kako5_race_count・same_td/dist/place_ratio、計4特徴）**: 実際のlive v6へ
  影響する、真のserve skew（train/serveのコードロジックが異なる）。

Cに該当する特徴は19特徴の中には存在しない（offlineだけが誤っている特徴は
今回の19特徴の枠組みでは確認されなかった。ただし別軸の「PLソフトマックス
分母からDNF馬が完全消失している」問題はB(offline replay)の一般的欠陥として
既に別途記録済み——`docs/research/DNF_HISTORY_FEATURE_PARITY_AUDIT_20260922.md`
§1参照、これは19特徴の値の問題ではなくレース母集団の問題であるため本分類
表の対象外）。

D（course/jockey 2026年分）・E（kako5の残り9特徴）は、**現時点でproduction
bugと断定しない**——実発生規模の定量化が先決事項として残る。

---

## 実測根拠

`out/same_input_parity.json`（`train_serve_same_input_parity.py`実行結果）:
9シナリオ×course/jockey6特徴 + 8シナリオ×kako5系16特徴（13特徴+3参考特徴）
を実測。course/jockey系は全シナリオ完全一致。kako5系は
`kako5_race_count`/`kako5_same_td_ratio`/`kako5_same_dist_ratio`/
`kako5_same_place_ratio`の4特徴のみ、DNFが異種条件で存在する場合に不一致
（train相当=0.667、serve=1.0等）。残り9特徴（位置・上がりタイム系集約）は
DNFが存在するどのシナリオでも一致した。

関連: `DNF_SEMANTIC_SPEC.md`, `GATE0B_FEATURE_AUDIT.md`,
`LIVE_SERVE_DNF_TRACE.md`, `out/v6_contract_manifest.json`
