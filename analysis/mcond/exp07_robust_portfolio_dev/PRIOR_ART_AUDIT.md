# EXP07 Stage 0 — `analysis/robust_ticket_portfolio.py` 由来・健全性監査

**実施日**: 2026-09-20　**方式**: read-only（ファイルは一切編集していない。既存テストの
実行[`pytest`]のみ行った — これは「学習」でも「本番接続」でもなく、既存コードの主張を
検証するための読み取り専用な確認行為として実施した）。

**総合判定: 再利用可能（ただし3点の既知の制約を明記した上で）**

---

## 1. Git管理対象か

**対象外（未コミット）**。`git status --short`で`??`、`git log --all --oneline -- analysis/robust_ticket_portfolio.py`は0件。
同様に`tests/test_robust_ticket_portfolio.py`、および参照元の`docs/plans/`ディレクトリ全体も未コミット。

## 2. 初出時期

ファイルシステムのタイムスタンプ: `tests/test_robust_ticket_portfolio.py`が2026-08-25 22:55、
`analysis/robust_ticket_portfolio.py`本体が2026-08-25 23:04（テストを先に書いてから実装、
9分差）。同日付の`docs/plans/exhaustive_betting_policy_verdict_20260825.md`（後述）と同一の
監査セッション内で作成されたと推定される。

## 3. Git履歴に過去版があるか

**なし**。`git log --all`でこのパスへの言及は一切なく、単一バージョンのみが現存する。

## 4. 他の未コミット変更との関係

`docs/plans/`ディレクトリ全体（`exhaustive_betting_policy_verdict_20260825.md`,
`audit_wide_residual_20260825.md`, `market_conditioned_profit_research_20260917.md`等）が
すべて未コミットで、この最適化モジュールはそれら一連の「券種横断・買い目最適化」監査群の
一部として生まれた。単独の孤立したファイルではなく、当時進行中だった大規模監査
（`exhaustive_betting_policy_verdict_20260825.md`）の付属ツールという位置づけ。

## 5. 作成目的（**重要な自己訂正の記録あり**）

`docs/plans/exhaustive_betting_policy_verdict_20260825.md`本文（88-105行）によれば、当初の
コンセプトは「18頭立て候補券最大6,360種を共通の着順状態払戻ベクトルへ落とし、100円整数stake・
no-bet・確率下限・オッズ下限・最悪期待利益・CVaR・予算制約の下で最悪シナリオ期待対数成長を
最大化する厳密solver」として設計された。

**しかし同文書には同日付（2026-08-25）の「独立監査」による訂正が明記されている**:

> 「この solver は実データでは未実行である。`optimise_portfolio`の呼び出し元は
> `tests/test_robust_ticket_portfolio.py`のみで、`reports/`に実行成果物は無い。さらに既定
> `min_worst_expected_profit_yen = 0.0`により、保守EV>1の券が1枚も無ければ0円は制約から
> 自明に出る。したがって『実弾0円』の根拠は券種別ROI実測表であって、網羅最適化の探索結果では
> ない。solverは将来エッジが認定された際の配分器であり、現時点の結論を支える証拠ではない。」

すなわち、**このモジュールは一度「重要な結論（実弾0円）の根拠」として過大に扱われかけたが、
即日の自己監査でそれが誤りだと訂正され、「エッジが認定された後の配分専用ツール」という
正しい位置づけに修正された経緯がある**。EXP07がこのモジュールを再利用する際も、同じ誤り
（「このsolverが動く＝有効性が証明された」という飛躍）を繰り返さないよう、Gate 0〜3を経ずに
結論を導かないことが重要。

2026-09-17の`market_conditioned_profit_research_20260917.md`（50行）でも「G2を通過した候補
だけを対象に、既存`robust_ticket_portfolio.py`を再利用する」と記載されており、訂正後の
位置づけ（配分器、判断根拠ではない）が踏襲されている。

## 6. 入出力

- **入力**: `Ticket`データクラスの列（`name`, `probability`, `odds_floor`,
  `probability_floor`[任意], `state_payoffs`[任意]）＋`optimise_portfolio()`の予算・
  bankroll・CVaR設定等のキーワード引数。
- **出力**: `PortfolioResult`（`stakes_yen`辞書, `total_stake_yen`, `objective`,
  `worst_expected_profit_yen`, `expected_profit_by_scenario_yen`, `worst_cvar_loss_yen`,
  `state_profits_yen`, `evaluated_portfolios`）。
- ファイルI/Oは一切なし（`open(`, `to_csv`, `json.dump`等の呼び出しをgrepで確認、0件）。
  標準ライブラリ（`dataclasses`, `itertools`, `math`, `typing`）のみに依存し、pandas/numpy
  すら使わない完全自己完結モジュール。

## 7. 使用する確率

呼び出し側が渡す`probability`/`probability_floor`（周辺確率）または`state_probabilities`/
`state_probability_scenarios`（同時状態確率）をそのまま使う。**モジュール自身はv6・PL・
Harville等いかなる本番予測パイプラインも呼び出さない**（import文に本番コードへの参照なし）。
確率の出所（時点安全性）は完全に呼び出し側の責任であり、このモジュール単体では検証不能。

## 8. 使用するオッズ

`odds_floor`（保守的な総払戻倍率、呼び出し側が渡す固定値）のみ。ライブオッズ取得・T-10取得等の
機能は一切なし。

## 9. 実結果参照の有無

**なし**。確定着順・確定payout・結果ラベルへの参照はコード全体を通じてゼロ。純粋な数理最適化
モジュールであり、方向9が警告する「結果を見て調整された」痕跡はコードレベルでは存在しない
（ただし§5の通り、周辺ドキュメントのナラティブでは一度過大な主張がなされ即日訂正されている
＝コード自体ではなく**その役割の説明**が一度誤っていた、という区別が重要）。

## 10. CVaRの定義

`_cvar_loss()`（191-210行）: 損失を`max(0, -profit)`で定義し（非負の損失、符号は健全）、
確率降順ではなく**損失降順**にソートしてから、信頼水準`alpha`に対応する上側`(1-alpha)`の
確率質量分（tail_mass）だけ加重平均する、標準的な**upper-tail CVaR**の実装。数式・符号とも
教科書的定義と一致することを確認した。

## 11. 損失の符号

`profit`（正=利益、負=損失）を基準に、CVaR計算時のみ`max(0, -profit)`で損失（非負）へ変換。
目的関数側では`log1p(profit / bankroll_yen)`という対数成長率を直接最大化するため符号反転は
不要。一貫して健全。

## 12. 最悪ケース集合の定義

**重要な制約**: `state_probability_scenarios`は呼び出し側が**明示的に渡す有限個の確率ベクトルの
リスト**であり、モジュール自身がキャリブレーション誤差やbootstrap区間から自動的に不確実性集合
（例: Wasserstein球、信頼楕円体）を構築するロジックは一切持たない。最悪ケース最適化は
「渡された候補シナリオの中でのmin/max」であり、真の分布的ロバスト最適化（分布の集合全体に
対する最悪ケース）ではなく**有限シナリオ近似**である点をEXP07で明記する必要がある。
シナリオ自体の構築（§7 不確実性集合の設計）はEXP07が別途新規実装しなければならない。

## 13. 最適化ソルバー

**外部ソルバー依存なし**。整数stakeベクトルの**全列挙**（`_unit_vectors`, `_composition_count`）
による厳密探索。小規模な券数・予算では正確だが組合せ爆発するため、`max_portfolios=1_000_000`
（既定）で列挙数上限を事前チェックし、超過時は`ValueError`で拒否する設計（250-260行台）。
`max_states=65_536`は独立ベルヌーイ展開時の状態数上限（2^チケット数のため実質16枚程度が上限）。
scipy.optimize/cvxpy等の凸最適化ライブラリは未使用。

## 14. 制約条件

予算上限（`budget_yen`）、非負整数の100円単位stake、任意の1券当たり上限（`max_units_per_ticket`）、
任意の最悪期待利益下限（`min_worst_expected_profit_yen`）、任意のCVaR上限（`max_cvar_loss_yen`）
を実装済み。

**既知の欠落（EXP07で追加実装が必要）**: `Ticket`データクラスには馬番・出走馬identityを表す
フィールドが存在しない（`name`は不透明な文字列識別子のみ）。したがって**「同一馬への露出上限」
という仕様書7節の制約はこのモジュール単体では表現不可能**であり、EXP07側で（a）`Ticket.name`に
構成馬情報をエンコードする、または（b）このモジュールの外側で事前に露出上限フィルタをかける、
のいずれかの追加実装が必要。

## 15. 100円単位処理

`unit_yen: int = 100`（既定）を厳格に整数チェックし、全stakeを`units * unit_yen`で計算。
`test_budget_remainder_and_unit_grid_are_respected`で250円予算→200円消化（100円単位の端数
切り捨て）を確認済み。健全。

## 16. no-bet解の有無

**あり、かつ優先される**。全ゼロベクトルは`_unit_vectors`の列挙に必ず含まれ、`_is_better()`の
タイブレーク規則（目的関数値・期待利益・CVaRが同点の場合はより小さいstake総額を優先）により、
無意味な賭けよりno-betが選好される設計。予算を必ず使い切る制約は存在しない（仕様書7.3の要件と
整合）。`test_negative_ev_tickets_choose_no_bet`で確認済み。

## 17. 数値不安定性

`_TOL = 1e-12`を全ての浮動小数点比較（タイブレーク、制約判定）で一貫使用。`log1p(profit/bankroll_yen)`
は`bankroll_yen > budget_yen ≥ total_stake`が入力検証で強制されるため、最悪でも
`profit ≥ -total_stake > -bankroll_yen`となり`profit/bankroll_yen > -1`が構造的に保証される
（`log1p`の定義域を外れることがない）。この設計は意図的かつ正しい。

## 18. 例外時の挙動

不正入力（重複チケット名、非正の`unit_yen`、負の`budget_yen`、`bankroll_yen ≤ budget_yen`、
`cvar_alpha`が(0,1)外、負の`max_cvar_loss_yen`/`cvar_penalty`、explicit/marginal混在、
状態ベクトル長不一致、列挙グリッド超過）は全て`ValueError`。列挙が完了しても解が見つからない
場合のみ`RuntimeError`（コード内コメント曰く「no-betは検証済みの非負制約下で常に実行可能な
はずなので、ここに到達するのは内部の列挙バグを示す」＝到達しないことが期待される防御的分岐）。

**EXP07仕様書8節の合成テスト項目13「ソルバー失敗時はno-bet」との齟齬**: このモジュールは
外部ソルバーを呼ばない（全列挙のため数値的失敗の概念自体が薄い）。呼び出し側が想定する
「ソルバー失敗」（タイムアウト・数値発散等）に相当する事象は、実質的に「列挙グリッド超過
（`ValueError`）」または「候補データの不整合（`ValueError`）」のみであり、これらは
**現状RuntimeErrorではなく例外送出で終わり、自動でno-betへフォールバックしない**。
EXP07側のラッパー（`policies.py`）で、これらの`ValueError`/`RuntimeError`をキャッチして
no-betへ変換する層を追加実装する必要がある（Stage 1合成テストの項目13でこれを検証する）。

## 19. 本番ファイルへの副作用

**なし**。ファイルI/O・ネットワークI/O皆無、`compute_bets.py`や`models/`等への参照もimportにも
本文にも一切なし。孤立した純粋関数群。

## 20. 結果を見ながら調整された形跡

コード自体（定数・ロジック）に特定レース・特定日付・特定確率値に対する場当たり的なチューニング
の痕跡は見当たらない（`cvar_alpha=0.95`, `max_portfolios=1_000_000`, `max_states=65_536`は
いずれも一般的な安全域としての既定値であり、特定の実データにfitさせた形跡はない）。
ただし§5に記載の通り、**このモジュールの「役割」に関する周辺ドキュメントの記述**は一度、
実際より重要な役割（結論の根拠）を担っているかのように書かれ、即日の独立監査で訂正されている。
コード自体の健全性とは区別した上で、EXP07ではこの種の役割の誇張を再発させないよう、
「配分器はGate 0〜3を通過するまで結論の根拠にしない」という規律を厳守する。

---

## 総合判定: 再利用可能（3つの制約を明記の上で）

| 項目 | 評価 |
|---|---|
| 数学的健全性 | 高い（CVaR定義・符号・対数成長率のlog1p安全性・100円単位処理いずれも精査して問題なし） |
| テスト網羅性 | 8件が既存、全PASS確認済み（2026-09-20、本監査で再実行） |
| 副作用リスク | ゼロ（純粋関数、外部I/Oなし） |
| **制約1** | 同一馬への露出上限を表現するフィールドがない（EXP07で追加実装要） |
| **制約2** | 不確実性集合(`state_probability_scenarios`)は呼び出し側が構築する必要がある（EXP07で新規に発達させる） |
| **制約3** | ソルバー例外(`ValueError`/`RuntimeError`)を自動でno-betへ変換する層がない（EXP07のラッパーで追加） |
| 由来の健全性 | 良好だが要注意：役割について一度過大な主張がなされ即日訂正された経緯あり。EXP07はGate規律を厳守しこれを再発させない |

**再利用方針**: `optimise_portfolio()`本体はそのまま`analysis/mcond/exp07_robust_portfolio_dev/policies.py`
からimportして利用する（コピー・改変はしない、単一ソースを維持）。制約1〜3への対応は
EXP07側の薄いラッパー層で実装する。
