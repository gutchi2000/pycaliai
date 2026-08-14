==============================================================
 PyCaLiAI 全体の流れ (運用まとめ)          最終更新: 2026-08-14
==============================================================

JRA 中央競馬の AI 予想システム。
  予測   = LightGBM v6 (unified_rank_v6.pkl) → 印 (◎〇▲△△) + PL確率
  買い目 = compute_bets.py (トップダウンエンジン, CB_ENGINE=topdown)
  論評   = Cowork (Claude Desktop, narrative 専用。買い目は書かない)
  表示   = 静的サイト https://pycaliai.com (HF Space pycaliai-umami) が本番
  投票   = 人間が IPAT で手動 (自動投票なし)


--------------------------------------------------------------
 0. 週次CSVの投入 (毎週の最初にやること)
--------------------------------------------------------------
TARGET から出した CSV は全部 data/_inbox/ に放り込むだけ。
weekly_nicegui.ps1 が冒頭で place_weekly.py を自動実行して振り分ける。

  ファイル名ルール:
    S20260815.csv     出走表      → data/weekly/
    K20260815.csv     過去5走     → data/kako5/
    T20260815.csv     着度数      → data/tyaku/
    20260815.csv      結果(15列)  → data/kekka/
    bias20260815.csv  土曜結果(174列払戻) → data/bias/ + 実現バイアス生成
    H-* / W-*         調教        → data/training/
    OD*               オッズ      → data/odds/

  詳細: data/_inbox/README.txt


--------------------------------------------------------------
 1. Phase A -- 土曜朝 (出走表エクスポート後)
--------------------------------------------------------------
  .\weekly_nicegui.ps1              # または .\weekly_nicegui.ps1 20260815

  中でやること (デフォルト model=v6):
    1. place_weekly.py           inbox 振り分け
    2. make_weekly_hosei.py      補正タイム生成 → data/hosei/
    3. export_weekly_marks.py    印 + PL確率 + calibration
                                 → reports/cowork_input/{date}_bundle.json ★キー出力
    4. build_course_stats.py     コース分析用統計
    5. git push → sync-hf.ps1 (旧NiceGUI) → sync-hf-umami.ps1 (本番サイト)

  ※ predict_weekly.py (旧8モデル) はデフォルト SKIP。欲しい週だけ -WithPredict


--------------------------------------------------------------
 2. Phase B -- 土曜昼 (Cowork の narrative が返ってきたら)
--------------------------------------------------------------
  1. Claude Desktop に {date}_bundle.json を投入 (プロンプト: docs/cowork_prompt.md)
     → Cowork は advisor 論評 (narrative) のみ返す
  2. 返答を reports/cowork_output/{date}_bets.json として保存
  3. .\weekly_nicegui.ps1 -BetsOnly

  中でやること:
    1. validate_cowork_bets.py --apply   見送りガード強制 (fail-closed)
    2. git push → sync-hf.ps1 → sync-hf-umami.ps1


--------------------------------------------------------------
 3. 当日 (土日) -- T-10 自動買い目ライン
--------------------------------------------------------------
  土日 9:00 にタスクスケジューラ「PyCaLiAI_T10」が t10.ps1 を自動起動。
  手動起動は不要。祝日(月)開催のみ手動:
    .\t10.ps1                    # 本番
    .\t10.ps1 20260614 -Dry      # テスト

  各レース発走10分前に:
    JV-Link オッズ取得 → compute_bets.py (topdown) → validate → 買い目表示
    補正印 (オッズ blend, 表示専用) も出す。投票は人間が IPAT で。

  仕様: docs/compute_bets_spec.md


--------------------------------------------------------------
 4. Phase C -- 日曜夜 (結果エクスポート後)
--------------------------------------------------------------
  結果 CSV を data/_inbox/ (または data/kekka/) に置いてから:
  .\weekly_nicegui.ps1 -Post

  中でやること:
    1. weekly_post.ps1 (失敗したら fail-hard で HF 同期中止)
       - generate_results.py     → data/results.json / cowork_results.json
       - update_live_results.py  → data/live_results_2026.csv
       - git push + reports/cowork_bets/{date}/
    2. sync-hf.ps1 → sync-hf-umami.ps1 (本番サイト反映)
    3. 日曜+月初1-7日なら retrain_value_model.py 自動実行
    4. 日曜なら run_audit.ps1 自動実行 (週次監査)

  Phase C 後は git status で HF に実際に届いたか確認する習慣。


--------------------------------------------------------------
 5. 週次フロー外の手動コマンド
--------------------------------------------------------------
  # note 有料記事ドラフト (会場パック/重賞単体, JRA-VAN準拠スクラブ付き)
  python scripts/build_note_article.py 20260815
    → reports/note/{date}/{会場}.md + _compliance_report.txt

  # Cowork 集計の再生成 (kekka 追加後など)
  python generate_results.py

  # NiceGUI ローカル起動 (旧本番の目視確認)
  python nicegui_app.py          # http://localhost:8080

  # Streamlit 起動 (副系統)
  streamlit run app.py


--------------------------------------------------------------
 6. モデル再構築 (四半期〜半期、普段はやらない)
--------------------------------------------------------------
  python run_v6_pipeline.py      # 本番 v6 系 (calibrator/curve/class_prior)
  python run_v5_pipeline.py      # rollback 用 v5 系
  python audit_marks.py --model v6
  python -m analysis.fit_t10_blend   # T-10 補正印の λ 再fit


--------------------------------------------------------------
 7. 確認が必要な操作 (自律でやらないもの)
--------------------------------------------------------------
  - HuggingFace への反映を伴う push (sync-hf*.ps1 単独実行含む)
  - データ・モデルの削除
  - data/master*.csv の再生成 (数十分〜時間規模)


--------------------------------------------------------------
 詳しくは
--------------------------------------------------------------
  CLAUDE.md                    全体像・引き継ぎ (最重要)
  WORKFLOW.md                  週次フロー詳細
  docs/compute_bets_spec.md    買い目エンジン仕様
  docs/marks_schema.md         bundle.json スキーマ
  lab/README.md                実験スクリプト群 (python -m lab.<theme>.<name>)
