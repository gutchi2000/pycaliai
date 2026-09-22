# 障害レース除外 P0 hard gate — 実装完了報告（2026-09-22）

**状態**: P0 hard gate を 5 層すべてに実装。`PyCaLiAI_JumpHistory` タスク登録済み。
モデル変更・特徴定義変更・`_horse_history.parquet` 置換・障害レース予測・
障害馬券購入・corrected-vNext・Optuna・EXP14 Stage 1・ROI 評価は**いずれも
行っていない**。

---

## 1. なぜ P0 か

既知 3 件が `bets=[]` だったことは安全性の証明にならない。
**障害レースが production bundle へ入り、v6 で採点され、印まで付いた時点で
安全契約違反**である。実購入が 0 円だったのは Cowork の判断結果であって、
コードレベルの拒否は存在しなかった。

---

## 2. 共通 eligibility 関数

`race_eligibility.py` — **判定はここ 1 か所だけ**。

```python
evaluate_race(race_id, *, track_code=None) -> {
    "race_id", "is_jump", "determination",
    "prediction_eligible", "bet_eligible",
    "task_registration_eligible", "history_only_eligible",
    "reason", "raw_fields",
}
```

障害の場合の戻り値:

| フィールド | 値 |
|---|---|
| `is_jump` | `True` |
| `prediction_eligible` | `False` |
| `bet_eligible` | `False` |
| `task_registration_eligible` | `False` |
| `history_only_eligible` | **`True`**（完全に捨てない） |
| `reason` | `"jump_race_excluded"` |

### 2.1 判定ソースと優先順位

| 順位 | ソース | 位置づけ |
|---|---|---|
| 1 | `トラックコード(JV) ∈ 51..59` | **主判定（authoritative）**。`data/bunseki` 由来、または呼び出し側が明示的に渡した値 |
| 2 | `平・障 == "1"`（`data/bias`） | **cross-check** |
| 3 | `data/history_only/jump/raw_card/{date}.jsonl` | 収集済み障害 race_id による補強 |

- **レース名・`芝・ダ` 列は使わない**（`芝・ダ` は障害を芝/ダへ recode するため
  判定器として機能しない。EXP13 がこれで誤判断した）
- **1 と 2 が不一致なら fail-closed** → `is_jump=True`,
  `reason="jump_detection_conflict"`
- 判定材料が皆無なら `determination="unknown"`,
  `reason="jump_undetermined"`（通すが必ずログへ残す）

`raw_fields` には判定に使った生値（`track_code` / `track_code_source` /
`hira_shogai` / `in_jump_history_store` / `date`）を必ず含める。

---

## 3. Defense in depth — 5 層すべてに独立ゲート

| 層 | 場所 | 実装 |
|---|---|---|
| **1. bundle 作成・race list** | `export_weekly_marks.py` のレースループ | `prediction_eligible=False` なら `continue`。除外件数と race_id を `logger.warning` |
| **2. task 登録** | `t10_runner.build_schedule()` | `task_registration_eligible=False` を schedule から除外。T-10 / T-20 Site / T-35(EXP05-F) / Vote は**すべてこの関数を経由**するため一括で効く |
| **3. compute_bets** | `compute_bets.compute_race_bets()` 冒頭 | 障害なら常に `bets: []`、`race_nature="見送り"`、`is_jump=True`、`excluded_reason` を返す |
| **4. validate_cowork_bets** | `validate_cowork_bets.main()` のレースループ | 障害 × 非空買い目を違反として計上。`--apply` で `bets: []` へ強制。bundle に無くても独立に検査 |
| **5. 購入・送信直前** | `masters_vote.vote_race()`（早期）と `masters_vote.submit()`（実送信直前） | `submit()` は `assert_bettable()` で **`JumpRaceBettingError` を送出**。上流で除外済みでも省略しない |

> 最終購入経路では、障害レースに対する**非空買い目を必ず例外で拒否**する。
> 空の買い目は通す（見送りは正常系）。

---

## 4. 保存上の扱い — 完全に捨てない

- production prediction bundle からは**除外**
- history-only raw card には**保存可能**（`history_only_eligible=True`）
- 通常の印・買い目・AI score 対象には**しない**
- 除外件数と race_id を**必ずログへ残す**（`log_exclusions()` が
  `logger.warning` と標準出力の両方へ出す）

---

## 5. 回帰テスト（`tests/test_jump_race_p0_gate.py`、**48 件すべて PASS**）

### 5.1 fixture（実際に混入していた 3 レース）

`2026091306040401` / `2026091909040504` / `2026092009040601`

| 確認項目 | 結果 |
|---|---|
| jump 判定 = true（`determination="authoritative"`、track_code ∈ 51..59） | PASS |
| prediction 対象外 | PASS |
| task 登録対象外（`build_schedule` から消える） | PASS |
| `compute_bets` が空（`bets=[]`, `is_jump=True`） | PASS |
| 非空買い目を validate へ渡すと拒否（exit != 0） | PASS |
| `--apply` で `bets` が `[]` に矯正される | PASS |
| 購入直前経路でも拒否（`assert_bettable` / `masters_vote.submit` が例外） | PASS |
| history-only 保存は可能 | PASS |
| `raw_fields` に判定根拠が入っている | PASS |

### 5.2 positive control（通常の芝・ダートレース）

`2026091306040402` / `2026091306040403` / `2026091909040505` / `2026092009040602`

| 確認項目 | 結果 |
|---|---|
| `is_jump=False`, `reason="flat_race_ok"` | PASS |
| prediction / bet / task すべて eligible | PASS |
| `compute_bets` が障害として弾かない | PASS |
| 買い目があっても `assert_bettable` を通過 | PASS |
| `masters_vote.submit` が gate を通過する | PASS |

### 5.3 判定ロジックの性質

| 確認項目 | 結果 |
|---|---|
| トラックコードと平・障の不一致 → fail-closed（障害扱い） | PASS |
| 判定材料なし → `unknown` として記録（黙って通さない） | PASS |
| 明示的 `track_code` 引数が最優先 | PASS |
| 空の買い目は拒否しない | PASS |

---

## 6. collector の変更（②の指示を反映）

### 6.1 開催日判定 — 「bunseki が無ければ常に exit 0」を廃止

`is_race_day()` は**曜日では判断しない**。次の証跡を見る:
`data/jra_known_race_days_override.json` の `known_race_days`、
`data/{weekly,kekka,tyaku,kako5,bias}/{date}.csv` の存在。

| 状況 | 終了コード |
|---|---|
| 非開催日 & bunseki なし | **0**（正常） |
| **開催日 & bunseki なし** | **3 `MISSING_BUNSEKI_EXPORT`** |
| 開催日 & bunseki あり & 障害なし（他ソースとも整合） | 0（正常） |
| 障害があるはずなのに raw card 0 件 | **4 `JUMP_RACE_MISSING`** |
| venue ごとの race/horse coverage 異常 | **5 `COVERAGE_ANOMALY`** |
| キー衝突 | **6 `COLLISION`**（全体停止） |

### 6.2 pending settlement — 当日だけで終わらせない

毎回:
1. 当日の bunseki を raw card へ append
2. 過去の未 settled raw card を**すべて列挙**
3. 利用可能になった kekka だけを settled へ append
4. 未 settled 件数・最古日・理由をログ
5. 原本は変更しない

**期限で削除しない。** 現在 6 件が未 settled（最古 20260912、理由は
いずれも「kekka なし」）。

### 6.3 DNF／除外の未分離 — 偽の boolean を入れない

settled schema を **v2** に上げた（v1 は削除せず
`schema_version` をキーに含めた **append-only revision** として残す）:

| フィールド | 値 |
|---|---|
| `completed` | `true` / `false` |
| `noncompletion_kind` | raw code、無ければ `"unknown"`、完走なら `null` |
| `dnf` | **常に `null`** |
| `scratched` | **常に `null`** |
| `dnf_jogai_separable` | **`false`** |
| `usable_for` | `["legacy_v6_completed_only"]` |
| `not_usable_for` | `["corrected_vnext"]` |

card にあり結果に無い行は `started=false`,
`noncompletion_kind="absent_from_kekka"`（取消と断定しない）。

### 6.4 rollback の訂正 — artifact は保持する

**`data/history_only` を削除する手順は rollback から除外した。**
append-only の監査記録を消さない。

正しい rollback:
1. `Disable-ScheduledTask -TaskName PyCaLiAI_JumpHistory`
   （または `Unregister-ScheduledTask`）
2. consumer 接続があれば disable（**現時点で接続は存在しない**）
3. **collected artifact は保持**
4. manifest へ停止時刻と理由を追記:
   `python -m analysis.jump_history_only.jump_history_collector --record-stop "理由"`
   → `manifest.collection_active=false` と `stop_history[]` に追記

---

## 7. 登録前 Gate — **15/15 PASS**

`analysis/jump_history_only/preregistration_gate.py`

| Gate | 結果 |
|---|---|
| PowerShell syntax | PASS |
| 既存 8 日で idempotency（added=0） | PASS |
| collision で全体停止・部分書込なし | PASS |
| 非開催日 no-input → exit 0 | PASS |
| **開催日 no-input → exit 3**（bunseki を一時退避して実測、原本は復元確認済み） | PASS |
| pending settlement 列挙 | PASS |
| 通常 prediction 経路へ**新規**混入 0 | PASS |
| 収集済み障害 race が全て bet 不可 | PASS |
| P0 障害除外テスト（48 件） | PASS |
| **full test suite（294 件）** | PASS |
| task Action 固定 | PASS |
| WorkingDirectory 固定 | PASS |
| Python interpreter 固定 | PASS |
| ログ出力先固定 | PASS |
| error log 出力先固定 | PASS |

---

## 8. タスク登録状態

| 項目 | 値 |
|---|---|
| **タスク名** | `PyCaLiAI_JumpHistory` |
| **State** | `Ready`（登録済み） |
| **Action** | `powershell.exe -NoProfile -ExecutionPolicy Bypass -File "E:\PyCaLiAI\analysis\jump_history_only\jump_history_collect.ps1"` |
| **WorkingDirectory** | `E:\PyCaLiAI` |
| **Trigger** | 毎日 22:30 |
| **MultipleInstances** | **`IgnoreNew`** |
| **ExecutionTimeLimit** | **`PT30M`**（30 分、finite） |
| その他 | `StartWhenAvailable` / `WakeToRun` / バッテリー時も実行 |
| **次回実行** | **2026-09-23 22:30** |
| ログ | `logs/jump_history_{date}.log` |
| error log | `logs/jump_history_error.log` |

実行のたびに collector → **hard invariant + P0 gate テストを自動実行**し、
FAIL なら非 0 で終了する。

---

## 9. bunseki 手動運用（当面維持 + 監視）

TARGET GUI の自動操作は**導入しない**。運用は:

1. 開催日に bunseki を**手動 export**
2. `data/_inbox/` へ配置
3. `place_weekly.py`（`weekly_nicegui.ps1` Step 0 で自動実行）
4. **22:30 collector が存在・完全性を確認**
5. 未出力なら **scheduled collection failure**（exit 3）として
   `logs/jump_history_error.log` に記録

README へ**開催日の必須作業**として明記した。

記録項目（`data/history_only/jump/intake_ledger.jsonl`、append-only）:
`exported_at` / `source_sha256` / `target_date` / `race_count` /
`jump_race_count` / `horse_row_count` / `jump_horse_row_count` /
`operator="manual"` / `placed_at` / `collector_processed_at` / `intake_path`

---

## 10. 文書訂正（ERRATUM）

| 旧結論 | 訂正先 |
|---|---|
| weekly は障害を一貫して除外する | `ALT_SOURCE_RECOVERY_AUDIT_20260922.md` §2 |
| 欠落は 2026-09-06 で停止した | 同上 |
| 障害が production bundle へ入らない | 同上 / `JVLINK_ISOLATION_AND_P1_FORWARD_ONLY_20260922.md` 冒頭 |
| 買い目ゼロなので安全である | 同上 |

正しい結論:
- weekly の障害包含は**日によって不安定**
- **少なくとも 3 障害レースが production bundle で採点・印付けされた**
- **実購入は 0 円**だった
- **コードレベルの購入拒否が存在しなかった**
- **P0 hard gate で再発を防止する**

---

## 11. 実装範囲の確認

**実施した**:
障害の prediction/betting hard gate / history-only collector /
collector task 登録 / ログ・テスト・文書

**行っていない**:
モデル変更 / 特徴定義変更 / `_horse_history.parquet` 置換 /
障害レース予測 / 障害馬券購入 / corrected-vNext / Optuna /
EXP14 Stage 1 / ROI 評価
