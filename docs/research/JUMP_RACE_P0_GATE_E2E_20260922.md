# 障害レース除外 P0 gate — 実 production 入力での end-to-end 証明（2026-09-22）

**状態**: **P0 完了**。完了判定 6 項目すべて充足。
モデル変更・特徴定義変更・`_horse_history.parquet` 置換・障害レース予測・
障害馬券購入・corrected-vNext・Optuna・EXP14 Stage 1・ROI 評価は
**いずれも行っていない**。

再現:
`analysis/jump_history_only/p0_end_to_end_replay.py`
`analysis/jump_history_only/layer_evidence_audit.py`
`analysis/jump_history_only/preregistration_gate.py`

---

## 1. default 値を eligibility evidence として禁止

### 1.1 何が危なかったか

`predict_weekly.parse_csv` は週次 CSV に `トラックコード(JV)` 列が無いとき
`_MISSING_FEATURE_MEDIANS` で **23（芝・内）** を埋める。
この 23 を JV コードとして扱うと**障害レースが平地として通る**。

初版の `export_weekly_marks.py` は `g["トラックコード(JV)"].iloc[0]` を
`evaluate_race()` へ渡していた。これは**まさにその欠陥**だった。

### 1.2 対策

`evaluate_race(race_id, *, track_code=None, track_code_source=None)` へ
**provenance を必須化**した。

| provenance | 証拠として採用 |
|---|---|
| `raw_jv` | ✅ |
| `bunseki` | ✅ |
| `weekly_explicit` | ✅ |
| `calendar` | ✅ |
| **`default`** | ❌ **絶対に使わない** |
| `missing` | ❌ |
| source 未指定 | ❌（安全側で不採用） |

hard rule:

- `track_code_source == "default"` は判定根拠に使わない
- **値が 23 でも source が default なら `unknown`**
- 23 へ変換する**前の raw presence** を `raw_presence.track_code_present` に保持
- **`unknown` は prediction / bet / task すべて ineligible（fail-closed）**
- unknown 件数は `logger.error` と標準出力へ記録（bundle warning 相当）
- unknown を通常平地として黙って通さない

さらに `export_weekly_marks.py` は **df の値を一切渡さない**。
`evaluate_race()` が `data/bunseki` の authoritative 値だけを読む。

> モデル入力用の欠損埋め 23 と、eligibility 判定用の JV track code は
> **完全に分離された**。

### 1.3 テスト（`tests/test_jump_race_p0_gate.py`）

| テスト | 結果 |
|---|---|
| `default` 23 は証拠にならず `unknown` になる | PASS |
| `default` source では 3 つの eligible がすべて False | PASS |
| source 未指定の値は採用しない | PASS |
| 23 変換前の raw presence を保持している | PASS |
| authoritative source（4 種）は採用される | PASS |
| `unknown` が 5 層すべてで fail-closed | PASS |

---

## 2. 既知 3 日の実ファイル end-to-end 再生

**保存済みの実 weekly / bunseki / production bundle** を読み、
shadow directory（`analysis/jump_history_only/shadow/bundle_replay/`）へ出力。
**production bundle は一切書き換えていない**（前後の sha256 一致を検証）。

| 項目 | 20260913 | 20260919 | 20260920 |
|---|---|---|---|
| weekly sha256 | `98a9ff…`※ | `…` | `…` |
| bunseki sha256 | 記録済 | 記録済 | 記録済 |
| **全 race 数（weekly）** | **24** | **24** | **24** |
| **障害 race 数** | **1** | **1** | **1** |
| **通常 race 数** | **23** | **23** | **23** |
| **unknown race 数** | **0** | **0** | **0** |
| jump 判定に使った field | `トラックコード(JV)` | 同左 | 同左 |
| jump 判定の source | `bunseki` | `bunseki` | `bunseki` |
| **bundle へ残った通常 race** | **23** | **23** | **23** |
| **bundle から除外した障害 race_id** | `2026091306040401` | `2026091909040504` | `2026092009040601` |
| **誤って除外された通常 race** | **0** | **0** | **0** |
| **task schedule の障害 race** | **0** | **0** | **0** |
| **task schedule の通常 race** | **23** | **23** | **23** |

※ 実ハッシュは `out/p0_end_to_end_replay.json` に全文記録。

`flat_source_distribution` は 3 日とも `{"bunseki": 23}` で、
**`default` は 1 件も証拠に使われていない**。

### 2.1 期待結果の検証 — **30/30 PASS**

各日について:

| 検査 | 20260913 | 20260919 | 20260920 |
|---|---|---|---|
| 既知障害が prediction bundle から除外 | PASS | PASS | PASS |
| task 登録 0 | PASS | PASS | PASS |
| compute_bets 0 点 | PASS | PASS | PASS |
| validate が非空買い目を拒否 | PASS | PASS | PASS |
| submit 直前でも拒否 | PASS | PASS | PASS |
| `masters_vote.submit` で拒否 | PASS | PASS | PASS |
| 同日の全通常レースが残る | PASS | PASS | PASS |
| 通常レースを誤除外していない | PASS | PASS | PASS |
| **production bundle 不変** | PASS | PASS | PASS |
| default を証拠に使っていない | PASS | PASS | PASS |

---

## 3. bunseki が無い場合の実挙動 — morning source の確定

### 3.1 「履歴収集用 bunseki」と「朝の eligibility 用情報源」は別物

混同しない。両方とも `data/bunseki/{date}.csv` という**同じファイル**を見るが、
**必要になる時刻が違う**。

- 朝（Phase A、bundle 生成時）: eligibility 判定に必要
- 夜 22:30（collector）: 履歴収集に必要

### 3.2 実測 — bunseki は**前夜**に置かれている

`data/history_only/jump/intake_ledger.jsonl` の実測（8 日すべて）:

| race_date | exported_at | 当日 09:00 までのリード |
|---|---|---:|
| 20260905 | 2026-09-04 22:26 | **+10.6h** |
| 20260906 | 2026-09-05 22:05 | +10.9h |
| 20260912 | 2026-09-11 22:37 | +10.4h |
| 20260913 | 2026-09-12 23:29 | +9.5h |
| 20260919 | 2026-09-18 21:59 | +11.0h |
| 20260920 | 2026-09-19 22:50 | +10.2h |
| 20260921 | 2026-09-20 23:51 | +9.1h |
| 20260922 | 2026-09-21 23:02 | +10.0h |

**8/8 日で前夜（21:59〜23:29）に export されており、当日朝の予測時点では
必ず存在していた。** リードは +9.1〜+11.0 時間。

→ **「22:30 の手動 bunseki では当日の予測 gate に間に合わない」という懸念は
実データ上は発生していない。** 対象日 D の bunseki は D-1 の夜に置かれ、
D の朝には揃っている。22:30 の collector は D の夜に D の分を処理する。

### 3.3 朝の時点で使える authoritative source

| source | 朝に使えるか | 障害を含むか |
|---|---|---|
| **`data/bunseki/{date}.csv`** | **Yes（前夜に配置）** | **Yes（8/8 日で障害 1 レースを保持）** |
| `data/bias/{date}.csv` | 結果系のため**朝は不可**（cross-check 専用） | Yes |
| `data/tyaku/{date}.csv` | 朝に存在しうる | **障害をほぼ含まない**（3/50）ため**不在が非障害の証明にならない** |
| `data/weekly/{date}.csv` | Yes | **不安定**（8 日中 4 日のみ）。さらに `トラックコード(JV)` 列を持たない |
| 開催カレンダー JSON | Yes | レース種別を持たない |

→ **朝に使える authoritative source は `data/bunseki` のみ。**

### 3.4 どれも無い場合

> ### 運用表現（2026-09-22 訂正）
>
> **誤**: 「bunseki が無い朝は全 race を unknown で止め、その日 bundle が空になる」
>
> **正**: **「bunseki が無い、または予定 race を 100% 覆わない場合、
> bundle を公開せず非 0 終了し、直前の正常成果物を保持する」**
>
> fail-closed とは**空の成果物を公開することではない**。
> 処理を非 0 で終了させ、既存成果物をそのまま残すことである。

`data/bunseki/{date}.csv` が朝に無い、または予定 race を 100% 覆わない場合、
`eligibility_coverage_gate` が **bundle 生成前に**検出し、
**`ELIGIBILITY_EVIDENCE_INCOMPLETE`（exit 3）で終了する**。

このとき:

- **production bundle を作らない／上書きしない**（temp すら作らずに戻る）
- site / cowork 出力を更新しない
- task 登録を行わない・既存 task を削除しない
- `logs/eligibility_gate_error.log` へ
  unknown race_id 一覧 / expected・observed・missing / bunseki path+hash /
  weekly hash を JSON Lines で記録する
- **直前の正常な bundle がそのまま残る**

これは意図した設計である。理由:

- 障害か否かを authoritative に判定できないまま予測・購入するより、
  **その日を止める方が安全**
- 空 bundle を「成功」として公開すると、下流（site/Cowork/task）が
  「今日はレースが無い」と誤認しうる。**非 0 終了 + 既存保持**ならその誤認が起きない
- 復旧は簡単（TARGET から出走馬分析を export して `_inbox` へ置き、Phase A を再実行）
- 実測では 8/8 日で前夜に揃っており、発生確率は低い
- 夜 22:30 の collector が `MISSING_BUNSEKI_EXPORT`(exit 3) で
  `logs/jump_history_error.log` に記録するため、翌朝までに気付ける

**残存リスク**: bunseki 未配置の朝は当日の bundle が更新されない
（前日までの成果物が残る）。これは「障害を誤って買う」より軽いと判断した
上での trade-off である。

---

## 4. 各防御層の evidence 可用性

実測（`layer_evidence_audit.py`、probe: jump=`2026091306040401` /
flat=`2026091306040402` / unknown=`2099123106040401`）:

| layer | available fields | authoritative source | default 使用 | unknown 時動作 |
|---|---|---|---|---|
| **1. bundle creation**<br>`export_weekly_marks` | `race_id` のみ（**df の トラックコード は渡さない**） | `data/bunseki` の `トラックコード(JV)` + `data/bias` の `平・障` | **No** | `prediction_eligible=False` で除外、ERROR ログ + 標準出力警告 |
| **2. task registration**<br>`t10_runner.build_schedule` | `race_id`（bundle の race dict から） | 同上を **race_id から再取得** | **No** | schedule から除外（T-10/T-20 Site/T-35/Vote が全てこの関数を経由） |
| **3. compute_bets** | `race_id` + bundle の eligibility metadata | `verify_metadata()` が race_id から**再計算**し metadata と突合 | **No** | `bets=[]` / `race_nature=見送り` / `eligibility_determination=unknown`。metadata 欠落・schema 不一致・hash 不一致は**そのレースだけ** fail-closed |
| **4. validate_cowork_bets** | `race_id`（bets.json から。**bundle 不在でも可**） | `evaluate_race()` で独立に再取得 | **No** | 非空買い目なら違反計上、`--apply` で `bets=[]` に矯正 |
| **5. final submit**<br>`masters_vote.submit` | payload の `race_id`（+任意で metadata） | `assert_bettable()` → `verify_metadata`/`evaluate_race` で再取得 | **No** | 非空買い目なら **`JumpRaceBettingError`** を送出して停止 |

### 4.1 実測結果

| layer | jump | flat | unknown |
|---|---|---|---|
| 1. bundle creation | **False** | True | **False** |
| 2. task registration | **False** | True | **False** |
| 3. compute_bets | **False** | True | **False** |
| 4. validate | **False** | True | **False** |
| 5. final submit | **REFUSED** | True | **REFUSED** |

**全層で jump / unknown が遮断され flat が通る: PASS。**

上流で消えたことに依存していない —— 各層が `race_id` だけから
evidence を再取得できることを実測で確認した。

---

## 5. eligibility metadata の改ざん・欠落対策

bundle の各 race に以下を保存する（`export_weekly_marks` が付与）:

```json
"eligibility": {
  "eligibility_schema_version": "race-eligibility/1",
  "race_id": "...",
  "computed": {"is_jump", "prediction_eligible", "bet_eligible",
               "task_registration_eligible", "history_only_eligible",
               "determination", "reason"},
  "evidence": {"track_code_value", "track_code_source",
               "flat_jump_value", "flat_jump_source",
               "raw_presence", "in_jump_history_store"},
  "source_hashes": {"bunseki": "<sha256>", "bias": "<sha256>"}
}
```

下流は **boolean を信用しない**。`verify_metadata()` が共通関数で
**再計算**して突き合わせる。

| 事象 | 動作 |
|---|---|
| metadata 欠落 かつ 再計算が ineligible | **fail-closed** |
| `eligibility_schema_version` 不一致 | **fail-closed** |
| `source_hashes` 不一致 | **fail-closed** |
| metadata が「平地」と偽っている | **再計算が優先**（テストで検証済み） |

テスト: `test_metadata_schema_mismatch_is_fail_closed` /
`test_metadata_source_hash_mismatch_is_fail_closed` /
`test_metadata_missing_on_jump_is_fail_closed` /
`test_downstream_does_not_trust_boolean_only` — いずれも PASS。

---

## 6. テスト範囲の明示（訂正）

前回「full suite 294 件」と書いたのは**不正確**だった。訂正する。

### 6.1 実行したコマンドと結果

| scope | コマンド | 結果 |
|---|---|---|
| **A. 本体テスト** | `venv311/Scripts/python.exe -m pytest tests/ -q` | **275 passed** |
| **B. 関連スイート** | `... -m pytest tests/ analysis/jump_history_only/test_jump_history_invariants.py -q` | **305 passed** |
| **C. リポジトリ全体** | `... -m pytest tests/ analysis/jump_history_only/ analysis/mcond/ -q` | **収集エラー 2 件で実行不可** |

- skipped / xfailed / xpassed / deselected: **いずれも 0**（scope A・B とも）
- **scope B を「full suite」と呼ばない**。正しくは「関連スイート（relevant suite）」

### 6.2 リポジトリ全体で収集可能なテスト

`find . -name "test_*.py"`（`venv311` / `.claude/worktrees` 除く）: **80 ファイル**

| ディレクトリ | ファイル数 | 収集 |
|---|---:|---|
| `tests/` | 実行対象 | **275 件 collected & passed** |
| `analysis/jump_history_only/` | 2 | **30 件 collected & passed** |
| `analysis/mcond/exp*/` | 多数 | 385 件 collected、**+2 errors** |
| `analysis/*.py`（`*_test.py` 命名） | 4 | pytest 収集対象外（スクリプト） |

### 6.3 収集エラー 2 件は **P0 と無関係・既存**

`analysis/mcond/` 配下に同名テストファイルが重複している
（`test_evaluate.py` ×2、`test_time_safety.py` ×5）ことによる
pytest の import 衝突。個別ファイル単位では全て正常に収集できる
（1 ファイルずつ回して 0 エラーを確認済み）。

本 P0 作業は `analysis/mcond/exp*/` を一切変更していない。
この 2 件は**本作業以前から存在する構造的な問題**である。

### 6.4 P0 関連テストが含まれていることの確認

| テスト | scope | 件数 |
|---|---|---:|
| `tests/test_jump_race_p0_gate.py` | A・B | **59** |
| `analysis/jump_history_only/test_jump_history_invariants.py` | B | **30** |
| 既存の本番ライン（`test_production_line.py` 等） | A・B | 残り |

> 補足: P0 gate 導入により `tests/` の既存 4 件が一度 FAIL した
> （合成 race_id `2099...` が `unknown` になり fail-closed したため）。
> **gate を緩めるのではなく**、`tests/conftest.py` を追加して
> **`2099` 始まりの合成日付に限り**「authoritative に平地」と宣言することで解決した。
> 実日付には一切影響しない。

---

## 7. タスク状態 —「次回実行」と「最初の実開催検証日」を分ける

| 項目 | 値 |
|---|---|
| タスク名 | `PyCaLiAI_JumpHistory` |
| State | **Ready（登録済み）** |
| Action | `powershell.exe -NoProfile -ExecutionPolicy Bypass -File "E:\PyCaLiAI\analysis\jump_history_only\jump_history_collect.ps1"` |
| WorkingDirectory | `E:\PyCaLiAI` |
| MultipleInstances | `IgnoreNew` |
| ExecutionTimeLimit | `PT30M` |
| **次回実行（スケジュール上）** | **2026-09-23（水）22:30** |
| **その日の想定** | **非開催日**。`is_race_day()` が証跡なしと判定し **exit 0（正常終了・no-op）** |
| **最初の実開催検証日** | **2026-09-26（土）以降の最初の開催日**（通常は土日。JRA 開催日が確定次第 `data/jra_known_race_days_override.json` で補強） |
| 観測 Gate 完了見込み | **実開催 2 日ぶんの収集後**（最短で 2026-09-27(日) の夜） |

### 7.1 Gate 通過までの制約（維持）

- collector のみ稼働
- **feature 接続なし**
- **model 入力変更なし**
- **`_horse_history.parquet` 変更なし**

---

## 8. 完了判定

| # | 条件 | 状態 |
|---|---|---|
| 1 | default 23 が eligibility 判定へ使われない | ✅ **充足**（provenance 必須化 + `export_weekly_marks` から値を渡さない + 回帰テスト 6 件） |
| 2 | 実ファイル 3 日で既知障害を全件拒否 | ✅ **充足**（30/30 PASS、5 層すべてで拒否） |
| 3 | 同日の通常 race を誤拒否しない | ✅ **充足**（3 日とも誤除外 0、通常 23 race が bundle・task に残る） |
| 4 | morning 時点で利用可能な authoritative source が確定 | ✅ **充足**（`data/bunseki`。8/8 日で前夜 +9.1〜11.0h に配置。無ければ全 race を止める） |
| 5 | 5 層すべてで unknown が fail-closed | ✅ **充足**（`layer_evidence_audit` で実測、全層 PASS） |
| 6 | 実行したテスト範囲が明示される | ✅ **充足**（§6。scope A=275 / B=305 / C=収集不可、"full suite" の呼称を訂正） |

**→ P0 完了。**

---

## 9. 実装範囲の確認

**実施した**: 障害の prediction/betting hard gate（5 層）/ history-only
collector / collector task 登録 / ログ・canary・テスト・文書

**行っていない**: モデル変更 / 特徴定義変更 / `_horse_history.parquet` 置換 /
障害レース予測 / 障害馬券購入 / corrected-vNext / Optuna /
EXP14 Stage 1 / ROI 評価

---

## 10. transactional fail（2026-09-22 追加、P0 完全クローズ条件）

### 10.1 bundle 生成前の完全性 Gate

`eligibility_coverage_gate.check_coverage()` を
**`export_weekly_marks.py` の race ループ手前**で呼ぶ。
当日の予定 race 集合（weekly）と eligibility evidence 集合（bunseki）を突き合わせる。

| 必須条件 | 実装 |
|---|---|
| scheduled race coverage = 100% | `expected == observed` かつ `missing == 0` |
| race_id 重複 = 0 | `Counter` で検出 |
| authoritative track code coverage = 100% | `track_code_value is None` を不足として計上 |
| 許可済み source のみ | `{raw_jv, bunseki, weekly_explicit, calendar}` 以外は不可 |
| unknown race = 0 | `determination == "unknown"` を計上 |
| flat/jump disagreement = 0 | `determination == "conflict"` を計上 |

1 件でも満たさなければ **bundle 生成・push・task 登録をすべて中止**する。

### 10.2 unknown 時の動作

- exit code **3（非 0）**
- **`ELIGIBILITY_EVIDENCE_INCOMPLETE`**
- unknown race_id 一覧 / expected・observed・missing race 数
- bunseki file path + sha256 / weekly sha256 / bias sha256
- `logs/eligibility_gate_error.log` へ JSON Lines で追記
- **production bundle を作らない・上書きしない**
- site / cowork 出力を更新しない
- **task 登録を行わない・既存 task を削除しない**
- **空 bundle を「成功」として出力しない**

### 10.3 atomic publish

正常時も次の順番を守る。

1. **temp directory**（`.{date}__tmp` / `.{date}_bundle.json__tmp`）へ全成果物を生成
2. eligibility coverage Gate（※1 は 2 の後に開始する。Gate NG なら temp すら作らない）
3. category / feature canary
4. bundle 内部整合性（race 数・odds 被覆など既存の品質ゲート）
5. 全 Gate PASS
6. **atomic replace**（`os.replace` で bundle を差し替え、個別 JSON を本番へ移動）

途中 FAIL 時は `publish.abort()` で **temp だけを破棄**し、
**既存 production 成果物を保持**する。

### 10.4 実ファイル回帰テスト — **28/28 PASS**

`analysis/jump_history_only/p0_transactional_fail_test.py`
（production を書き換えず、原本は必ず復元）

**A. bunseki 全欠損**（実 weekly あり、対象日 20260913）

| 検査 | 結果 |
|---|---|
| exit 非 0 | **PASS**（exit=3） |
| `ELIGIBILITY_EVIDENCE_INCOMPLETE` を出力 | PASS |
| 全 24 race が evidence 欠落（observed=0） | PASS |
| `track_code_source` が全て `missing` | PASS |
| **bundle 未生成・既存 bundle hash 不変** | **PASS**（sentinel 一致） |
| task 登録 0 | PASS |
| error log に記録 | PASS |
| **A2**: bias も無い朝は（収集済み障害を除く）全 race が `unknown` | PASS（23/24。残り 1 は history-only store が証拠） |
| A2 でも gate は FAIL し公開しない | PASS（exit=3） |

**B. bunseki 部分欠損**（実 bunseki のコピーから `2026091306040405` の行だけ除去）

| 検査 | 結果 |
|---|---|
| exit 非 0 | **PASS**（exit=3） |
| **missing race_id を正確に報告** | **PASS**（`['2026091306040405']`） |
| expected=24 / observed=23 | PASS |
| **23/24 の部分 bundle を公開しない**（hash 不変） | **PASS** |
| task 登録 0 | PASS |

**C. 正常**（実 weekly + 完全 bunseki）

| 検査 | 結果 |
|---|---|
| exit 0 | PASS |
| coverage 100%（expected=observed=24, missing=0） | PASS |
| unknown 0 / 重複 0 / 不許可 source 0 / disagreement 0 | PASS |
| 障害 1 race は bundle へ入らず history-only に保存済み | PASS |
| 通常 23 race のみが flat | PASS |
| `abort()` で既存 bundle 不変・temp 消去 | PASS |
| `commit()` で bundle 差し替え・個別 JSON 移動 | PASS |

production の `reports/cowork_input/20260913_bundle.json` は
A・B とも sha256 `75a5d043…` のまま不変。
`data/bunseki` / `data/bias` の原本も復元済み（`.txfail_bak` 残留なし）。

### 10.5 テスト収集エラーの扱い

`analysis/mcond/` の同名 test import 衝突（`test_evaluate.py` ×2、
`test_time_safety.py` ×5）は **本 P0 とは別件**として記録するに留める。
将来 pytest の import mode か test package 名の整理で解消できるが、
**今回は着手しない**。

---

## 11. P0 完全クローズ

| 条件 | 状態 |
|---|---|
| 1. default 23 が eligibility 判定へ使われない | ✅ |
| 2. 実ファイル 3 日で既知障害を全件拒否 | ✅（30/30） |
| 3. 同日の通常 race を誤拒否しない | ✅（誤除外 0） |
| 4. morning authoritative source が確定 | ✅（`data/bunseki`、前夜 +9.1〜11.0h） |
| 5. 5 層すべてで unknown が fail-closed | ✅ |
| 6. 実行したテスト範囲が明示される | ✅ |
| **7. unknown 時の transactional fail** | ✅（28/28、空成果物を公開せず既存を保持） |

**→ P0 完全クローズ。**

以後は **2026-09-26（土）・09-27（日）の collector 観測のみ**を行い、
**特徴接続は行わない**。
