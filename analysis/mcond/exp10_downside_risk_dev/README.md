# EXP10 — 大敗・中止・能力未発揮リスク

## 目的

通常の着順精度向上ではなく、**v6が高く評価した馬が大きく崩れる下方リスク**を、
既存の勝率・entropy・市場人気の後にも検知できるかの検証。

**主仮説**: v6確率、raw score、entropy、市場確率、人気、頭数、出走回数、
休養日数を統制した後にも、専用の下方リスクモデルが大敗を識別できるか。

**主目的変数**: `catastrophic_downside = 1[observed_rank > q90_predicted_rank]`
（AI◎自身のPlackett-Luce予測分布から見て、実際の着順が上位90%分位の外に
あったか＝「v6自身の予測分布からみても異常な下方乖離」）。

## 絶対条件

EXP01-09・v6・compute_bets.pyは変更しない。新規作業は
`analysis/mcond/exp10_downside_risk_dev/`へ限定する。
「負けた馬=陽性」という設計は禁止（v6の逆モデルになるため）。

## 現在の進捗(2026-09-21時点)

| 段階 | 状態 | 成果物 |
|---|---|---|
| Stage 0(先行研究・データ監査) | **完了** | `PRIOR_ART_AUDIT.md` / `DATA_AUDIT.md` |
| Stage 1(定義確定・2023実件数・エンジン検証) | **完了** | `STAGE1_DESIGN.md` / `spec.json` / 下記コード |
| Stage 1.5(最小反証実験) / Stage 2 | **未着手** | ユーザー承認待ち |

## Stage 1 の要点

- **PL順位分布エンジン**(`pl_rank_distribution.py`): Gumbel-max モンテカルロ
  (本番、K=50,000)+厳密bitmask DP(中規模n検証)+全順列列挙(n<=8 oracle)の
  3層構成。合成oracleテスト25件全通過(`test_pl_rank_distribution.py`)、
  実2023年データ42レースでのMC-DP突合で最大誤差0.00523・q90閾値差は
  最大1ランク(`validate_mc_vs_dp_realdata.py`)。
- **対象馬選択**(`eligible_races.py`): raw v6 score最大の馬、
  結果非依存の決定論的tie-break、2023年3,456レース中tie-break発動0件。
- **主ラベル**(`labels.py`): 2023年development、3,379レース中
  catastrophic_downside陽性554件(16.40%)。q90緩和は不要と判断し確定。
- **既存flop detectorとの等価性**: catastrophic_downsideはtop3_failure
  (既存flop)の厳密な部分集合(100%⊂41.97%)。Stage0の「強い重複」評価を
  「ラベル定義が異なる新規性あり」へ訂正。
- **R1専用特徴の時点安全監査**: 10候補中4つ既存列で対応可、5つ低中リスクで
  構築可能、1つ(過去DNF履歴)は外部結合が必要で優先度を下げる。
- **B0〜B3/R1仕様・Stage1.5最小反証実験・中止条件**を`spec.json`に凍結。

詳細は`STAGE1_DESIGN.md`参照。**2024・2025年の性能・ROIは未開封**。

## コード構成

| ファイル | 役割 |
|---|---|
| `pl_rank_distribution.py` | PL順位分布計算エンジン(MC/厳密DP/全順列列挙) |
| `test_pl_rank_distribution.py` | 合成oracleテスト25件 |
| `validate_mc_vs_dp_realdata.py` | 実2023データでのMC-DP突合検証 |
| `eligible_races.py` | 対象馬(◎)選択・tie-break・母集団構築 |
| `labels.py` | 主ラベル(catastrophic_downside)・副次診断計算 |
| `subsidiary_favorite.py` | 1番人気版副次診断 |
| `spec.json` | 全定義の凍結 |

## 次の一手

ユーザー承認待ち。承認後はStage1.5（2023年のみの5-fold CVによる最小反証実験）
から着手し、そこを通過した場合のみStage2（B0-R1構築+2024/2025 Gate評価）へ進む。
