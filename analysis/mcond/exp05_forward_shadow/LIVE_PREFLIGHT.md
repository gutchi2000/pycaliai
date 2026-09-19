# EXP05-F 実データ試験 (spec §8)

タスクスケジューラへの登録前に、実際のJRA開催日(2026-09-19、土曜)に手動で1レース実行した。

## 実施内容

```powershell
venv311\Scripts\python.exe -m analysis.mcond.exp05_forward_shadow.market_snapshot \
    --once <rid16> --date 20260919
```

(このコマンドは `t35_shadow.ps1 -Once <rid16>` と同じPythonエントリポイントを叩く。
`-Schedule`自体はWindowsタスクスケジューラへの登録を伴うため、今回は`-Once`による
単発の手動実行で検証する。)

## 確認したいこと

- [ ] JV-Link(32-bit, `py -3.12-32`)からの実オッズ取得が成功する
- [ ] `scheduled_post`との差(分)が記録され、31-38分ウィンドウとの位置関係が分かる
- [ ] `feature_snapshot.py`が事前生成した特徴量とjoinしてM1/M3/M4が計算される
- [ ] `reports/exp05fs_odds/`・`data/_research/mcond/exp05fs_predictions/`に
  本番`reports/live_odds/`・`reports/site_odds/`と混ざらず保存される
- [ ] 本番T-10ライン(`t10_runner.py`)・サイトT-20(`t20_site_bets.py`)に影響が無い

## 結果 (2026-09-19、実開催日に実施)

対象レース: `2026091906040505` (発走12:25、阪神?中山系、12頭立て)。

| 実行 | 時刻 | minutes_to_start | valid_for_primary | 結果 |
|---|---|---|---|---|
| 1回目 | 11:46:58 | 38.0 (境界のすぐ外) | **False** | オッズ取得自体は成功(単勝12頭, overround=1.263)。ウィンドウ判定が正しく除外した |
| 2回目 | 11:47:30 | 37.6 | **True** | 主評価に使える記録として保存、market_probability合計=1.0で正しくde-vig |

## 確認できたこと

- [x] JV-Link(32-bit, `py -3.12-32`)からの実オッズ取得が成功する
- [x] `scheduled_post`との差(分)が正しく記録され、31-38分ウィンドウの内外を正確に判定する
  (1回目は38.0分でわずかに境界外、`valid_for_primary=False`で正しく除外。
  2回目は37.6分で正しく採用。**別時刻へのフォールバックは発生しない**、spec §7遵守を実地で確認)
- [x] `feature_snapshot.py`が事前生成した特徴量とjoinしてM1/M3/M4が計算される
  (M4_probability=0.048, edge_M4=2.15等、妥当な値域)
- [x] `reports/exp05fs_odds/`・`data/_research/mcond/exp05fs_predictions/`に
  本番`reports/live_odds/`・`reports/site_odds/`と混ざらず保存される (両ディレクトリを
  実行前後で確認、汚染なし)
- [x] append-only revision機構が実データでも機能する (1回目=rev1(invalid)、
  2回目=rev2(valid)、rev1は上書きされず両方残存)
- [x] 本番T-10ライン(`t10_runner.py`)・サイトT-20(`t20_site_bets.py`)に影響が無い
  (別プロセス・別ディレクトリ、実行中に他の本番タスクとの競合なし)

## 運用上の知見 (SCHEDULER_PLAN.mdへの追記事項)

LeadMin=35で起動しても、JV-Link呼び出し自体に数秒かかるため実際の観測時刻は
「35分前」よりわずかに遅れる(今回は起動時点で38.0分前だったのに対し、取得完了時点では
ウィンドウの終端ギリギリだった)。ウィンドウ[31,38]の**終端寄り**に着地する傾向があるため、
タスクスケジューラ本登録時はLeadMin=35のままで問題ないが、起床遅延(WakeToRunのオーバーヘッド)
が数分単位である場合はウィンドウを外れるリスクがある。実登録後の最初の数開催は
`valid_for_primary`の分布(何分前に着地しているか)を監視することを推奨する。
