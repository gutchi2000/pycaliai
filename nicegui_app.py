"""
nicegui_app.py
==============
NiceGUI 版 PyCaLiAI (実験 MVP v11)

v10 → v11 変更点:
  1. 📈 Cowork 累計収支 (シーズン P/L) セクション追加 (collapsible):
     - 累計投資 / 累計収支 / 回収率 / 的中率 の 4 chip
     - 日別累計収支ラインチャート (ECharts) + 日別収支バー
     - 馬券種別 (単/複/馬連/馬単/三連複/三連単) 別 P/L テーブル
  2. parse_kekka / compute_bet_pl / load_all_cowork_outcomes 新設:
     - data/kekka/{date}.csv をパース → race 結果 dict
     - cowork_output/{date}_bets.json と JOIN して bet 単位の収支算出
     - ワイドは kekka に payout 無いため -cost 計上 (将来 wide_payouts で対応)

v9.1 → v10 変更点:
  1. 🏃 コース別好走脚質 セクションをコース分析タブに追加:
     - 場所×芝/ダ×距離 ごとに 逃げ/先行/差し/追込 の勝率/連対率/複勝率
     - master_v2 の 前4角通過位置 × 出走頭数 で脚質推定
     - 事前生成 course_stats.json にも組み込み (HF でも動く)
  2. 🏇 展開予想セクションに「適性 A/B/C/D」バッジ追加:
     - 各馬の想定脚質 vs コース好走脚質を照合
     - そのコースで一番勝つ脚質 = A、最下位 = D
  3. 🐴 馬個別モーダル (出走表タブ下部のボタンで開く):
     - 馬名/印/騎手/斤量/性齢/AI 指標 (勝率/連対/複勝/EV/vs市場)
     - 過去 5 走テーブル (kako5 CSV から抽出)
     - 同コース脚質別勝率
  4. load_kako5_horses パーサ追加 (parse_kako5.py の解析ロジックを軽量移植)

v9 → v9.1 変更点:
  1. ⚡「直近 1 週間の好調教 Best 5」セクション追加 (collapsible expansion):
     - 坂路 終い1F (data/training/H-*.csv の Lap1 最小値)
     - WC 3F     (data/training/W-*.csv の 3F  最小値)
     - 各 5 頭、**当週末出走馬のみ** に絞る
     - その馬のレース情報 (場所/R/印/AI 勝率) を併記
  2. コース分析タブの 馬番テーブルを「2x2 グリッド (枠|馬番 / 年齢|性別)」化
  3. コース分析テーブルのフォント拡大 (13->17px)

v8 → v9 変更点:
  1. 出走表に「騎手」列を追加 (weekly CSV の 騎手 列から merge)
  2. 新タブ「📊 コース分析」を 全頭分析 と Cowork 買い目 の間に追加:
     上段: 🏇 展開予想 — 上位 5 頭の前4角通過位置から脚質を推定し、
            ハイ/スロー/平均ペースを判定 + 1 行アクション
     下段: 📊 過去成績 — 場所×芝/ダ×距離 の master_v2 統計を 4 軸で出力
            (枠順 1-8 / 年齢 3歳-8歳~ / 性別 牡牝セ / 馬番 1-18)
            画像と同じ「1着 / 2着 / 3着 / 着外 / 勝率 / 連対率 / 複勝率」表
  3. master_v2 読込列に「年齢」「性別」を追加
  4. compute_course_stats_v2 / render_course_analysis / render_tenkai_yoso 新設

v7 → v8 変更点:
  1. PyCaLi 評価リストの軸を Streamlit と同じ a〜g 7 軸に統一:
     a 総合力 (AI 勝率) / b スピード (前走補正/Ave-3F/走破タイム) /
     c 末脚 (前走補9/上り3F) / d 前走成績 (前走確定着順) /
     e 市場評価 (前走人気/単勝オッズ) / f ペース適性 (RPCI/PCI3/Ave-3F) /
     g 調教 (坂路/WC、HF Spaces には training CSV 未配置のため「−」表示)
  2. 新規 loader 追加:
     - load_weekly_horses(date) — weekly/{date}.csv の多行フォーマットを
       Streamlit と同一パーサで horse-level DataFrame に変換
     - load_hosei_all() — data/hosei/H_*.csv を glob して 前走補正/前走補9 取得
     - get_horse_features(date,race_id,umaban) — 1 馬分の Streamlit 互換特徴量
  3. compute_pyca_features を multi-column candidate + valid/direction
     フラグ対応に拡張 (Streamlit の PYCA_INDICATORS と同じ評価ロジック)

v6 → v7 変更点:
  1. サイドバー (縦型 expansion) を廃止 → 上部の水平タブ + ボタン UI に刷新:
     - 開催日 select → 1 行目
     - 場所タブ (東京/京都/新潟…) → 2 行目 (Streamlit 風 q-tabs スタイル)
     - レース番号ボタン (1R/2R/…/12R) → 3 行目
     - メイン (左右パネル + 出走表/全頭分析/Cowork) → 全幅で大きく表示
  2. 場所/レース選択時に該当ボタンをハイライト (青塗り)、状態は state で管理
  3. 場所切替で 1R を自動選択、日付切替で最初の場所の 1R を自動選択

v5 → v6 変更点:
  1. 4 chip (◎独走度/上位2頭集中/混戦度/市場一致) を素人向けに刷新:
     状態バッジ (固い/混戦/独走 等) と「→ 何をすべきか」の 1 行アクションを表示
  2. 全頭分析タブに Streamlit 版互換の「PyCaLi指数 出走馬評価リスト」追加:
     左 (印 + 馬名 + PyCaLi指数) / 中央 (6 軸レーダー) / 右 (a〜f 内訳バー)
     bundle.json データのみで完結する 6 軸 (総合力/連対力/圏内力/人気度/単勝妙味/複勝妙味)
  3. 旧 5 軸ミニレーダー grid は削除 (情報量重複のため)

v4 → v5 変更点:
  1. AI 評価コメント: 右パネル → 左パネル (印 ◎/〇/▲ の下) に移動
  2. 信頼度メトリクス chip: フォント拡大 (22→32px、14→17px)
  3. 右パネルに「📊 過去同コース成績」「🎯 馬場バイアス」を追加
     (master_v2.csv から計算、起動時に 1 回読み込んでキャッシュ)
"""
from __future__ import annotations

import bisect
import functools
import json
import re
from pathlib import Path

import os

import pandas as pd
from nicegui import ui, run

import betting_judgment as bj  # 買い方判定 (堅さ×妙味→買い方) の共有ロジック

try:
    from assistant import chat as soudan  # 対話相談AI (ローカル Ollama)
except Exception:
    soudan = None  # 読み込めなくても本体は動かす (HF/本番を保護)

BASE = Path(__file__).parent
COWORK_INPUT_DIR  = BASE / "reports" / "cowork_input"
COWORK_BETS_DIR   = BASE / "reports" / "cowork_bets"
COWORK_OUTPUT_DIR = BASE / "reports" / "cowork_output"
WEEKLY_DIR        = BASE / "data" / "weekly"
HOSSEI_DIR        = BASE / "data" / "hosei"
TRAINING_DIR      = BASE / "data" / "training"
MASTER_CSV        = BASE / "data" / "master_v2_20130105-20251228.csv"


# ============================================================
# weekly CSV 列定義 (app.py から最低限ポート、Streamlit 互換)
# ============================================================
RACE_COLS = [
    "レースID(新)","日付S","曜日","場所","開催","R","レース名","クラス名",
    "芝・ダート","距離","コース区分","コーナー回数","馬場状態(暫定)","天候(暫定)",
    "フルゲート頭数","発走時刻","性別限定","重量種別","年齢限定",
]
HORSE_COLS_46 = [
    "枠番","B","馬番","馬名S","性別","年齢","人気_今走","単勝","ZI印","ZI","ZI順",
    "斤量","減M","替","騎手","所属","調教師","父","母父","父タイプ","母父タイプ",
    "前走月","前走日","前走開催","前走間隔","前走レース名","前走TD","前走距離","前走馬場状態",
    "前走B","前走騎手","前走斤量","前走減","前走人気","前走単勝オッズ","前走着順","前走着差",
    "マイニング順位","前走通過1","前走通過2","前走通過3","前走通過4","前走Ave3F",
    "前走上り3F","前走上り3F順位","前走1_2着馬",
]
HORSE_COLS_48 = HORSE_COLS_46 + ["騎手コード","調教師コード"]
HORSE_COLS_49 = [
    "枠番","B","馬番","馬名S","性別","年齢","馬体重","馬体重増減_raw","馬体重増減",
    "人気_今走","単勝","ZI印","ZI","ZI順","斤量","減M","替","騎手","所属","調教師",
    "父","母父","父タイプ","母父タイプ",
    "前走月","前走日","前走開催","前走間隔","前走レース名","前走TD","前走距離","前走馬場状態",
    "前走B","前走騎手","前走斤量","前走減","前走人気","前走単勝オッズ","前走着順","前走着差",
    "マイニング順位","前走通過1","前走通過2","前走通過3","前走通過4","前走Ave3F",
    "前走上り3F","前走上り3F順位","前走1_2着馬",
]
HORSE_COLS_99 = HORSE_COLS_49 + [
    "二走前月","二走前日","二走前開催","二走前間隔","二走前レース名","二走前TD",
    "二走前距離","二走前馬場状態","二走前B","二走前騎手","二走前斤量","二走前減",
    "二走前人気","二走前単勝オッズ","二走前着順","二走前着差","二走前マイニング順位",
    "二走前通過1","二走前通過2","二走前通過3","二走前通過4","二走前Ave3F",
    "二走前上り3F","二走前上り3F順位","二走前1_2着馬",
    "三走前月","三走前日","三走前開催","三走前間隔","三走前レース名","三走前TD",
    "三走前距離","三走前馬場状態","三走前B","三走前騎手","三走前斤量","三走前減",
    "三走前人気","三走前単勝オッズ","三走前着順","三走前着差","三走前マイニング順位",
    "三走前通過1","三走前通過2","三走前通過3","三走前通過4","三走前Ave3F",
    "三走前上り3F","三走前上り3F順位","三走前1_2着馬",
]
COLUMN_MAP = {
    "前走Ave3F":      "前走Ave-3F",
    "前走着順":        "前走確定着順",
    "前走上り3F順位":  "前走上り3F順",
}


# ============================================================
# データロード
# ============================================================
def list_dates() -> list[str]:
    if not WEEKLY_DIR.exists():
        return []
    # 拡張子の大小文字を区別しない (TARGET が .CSV で出力する事があり、
    # Linux の HF Spaces では .csv glob にマッチせず日付が欠落するため)
    return sorted({p.stem for p in WEEKLY_DIR.glob("????????.*")
                    if p.suffix.lower() == ".csv"
                    and p.stem.isdigit() and len(p.stem) == 8}, reverse=True)


def load_bundle(date_str: str) -> dict | None:
    p = COWORK_INPUT_DIR / f"{date_str}_bundle.json"
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_cowork_bets(date_str: str, race_id: str) -> dict | None:
    """Streamlit 経由で保存された個別 race ファイル (旧経路)"""
    p = COWORK_BETS_DIR / date_str / f"{race_id}.json"
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _parse_one_cowork_file(path: Path) -> dict[str, dict]:
    """1 ファイルを読んで race_id → race_data 辞書を返す。"""
    try:
        raw_bytes = path.read_bytes()
        text = ""
        for enc in ["utf-8-sig", "utf-8", "cp932", "shift_jis"]:
            try:
                text = raw_bytes.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            text = raw_bytes.decode("utf-8", errors="replace")
    except Exception:
        return {}

    # JSON コードブロック抽出 (Markdown 内 or 直接)
    m = re.search(r"```(?:json|JSON)?\s*\n([\s\S]+?)\n\s*```", text)
    raw_json = m.group(1) if m else text.strip()

    try:
        data = json.loads(raw_json)
    except Exception:
        return {}

    # 受け入れる top-level 形式:
    #   - list (レガシー): そのまま使う
    #   - {"races": [...]}: 旧 wrapper、races を使う
    #   - {"bets": [...], "grade_scope": [...]}: 新 wrapper (Grade Scope 対応)
    #     → ここでは bets だけ抽出 (grade_scope は _load_all_grade_scope で別途処理)
    if isinstance(data, dict):
        data = data.get("bets") or data.get("races", [data])
    if not isinstance(data, list):
        return {}

    out: dict[str, dict] = {}
    for entry in data:
        if not isinstance(entry, dict):
            continue
        rid_raw = entry.get("race_id") or entry.get("レースID") or entry.get("rid")
        if not rid_raw:
            continue
        rid = str(rid_raw)[:16]
        bets_raw = entry.get("bets", entry.get("買い目", []))
        bets = []
        if isinstance(bets_raw, list):
            for b in bets_raw:
                if not isinstance(b, dict):
                    continue
                bets.append({
                    "馬券種": b.get("馬券種") or b.get("type", ""),
                    "買い目": b.get("買い目") or b.get("selection", ""),
                    "購入額": b.get("購入額") or b.get("amount", 0),
                    "理由":   b.get("理由")   or b.get("reason", ""),
                })
        # advisor (Cowork narrative 論評): 任意フィールド、原形で通す
        advisor_raw = entry.get("advisor", [])
        advisor = []
        if isinstance(advisor_raw, list):
            for a in advisor_raw:
                if not isinstance(a, dict):
                    continue
                advisor.append({
                    "umaban":     a.get("umaban") or a.get("馬番"),
                    "horse_name": str(a.get("horse_name") or a.get("馬名", "")),
                    "grade":      str(a.get("grade") or ""),
                    "tag":        a.get("tag") or None,
                    "comment":    str(a.get("comment") or a.get("コメント", "")),
                })
        out[rid] = {
            "race_id":     rid,
            "race_label":  str(entry.get("race_label", "")),
            "race_nature": str(entry.get("race_nature", "")),
            "race_reason": str(entry.get("race_reason", "")),
            "bets":        bets,
            "advisor":     advisor,
            "source":      f"cowork_output:{path.name}",
        }
    return out


def _cowork_output_cache_key() -> str:
    """全ファイルの mtime ハッシュ (ファイル追加/更新で cache 自動 invalidate)"""
    if not COWORK_OUTPUT_DIR.exists():
        return "no-dir"
    files = sorted(COWORK_OUTPUT_DIR.iterdir())
    return "|".join(f"{p.name}:{p.stat().st_mtime:.0f}" for p in files if p.is_file())


@functools.lru_cache(maxsize=4)
def _load_all_cowork_output(_cache_key: str) -> dict[str, dict]:
    """reports/cowork_output/ 全ファイルをスキャンして race_id 別 dict にマージ。

    複数日まとめた 1 ファイル / 日別ファイル / どちらでも OK。
    同じ race_id が複数ファイルにあれば 最新 mtime のファイルが勝つ。
    """
    if not COWORK_OUTPUT_DIR.exists():
        return {}

    out: dict[str, dict] = {}
    # mtime 古い順で読む → 後勝ち (新しいファイルが上書き)
    files = sorted(
        [p for p in COWORK_OUTPUT_DIR.iterdir() if p.is_file()
         and p.suffix.lower() in (".json", ".txt", ".md")],
        key=lambda x: x.stat().st_mtime,
    )
    for p in files:
        out.update(_parse_one_cowork_file(p))
    return out


def load_cowork_bets_unified(date_str: str, race_id: str) -> dict | None:
    """個別ファイル (cowork_bets/) → 全 _bets.json 横断検索 の順で取得。

    Streamlit で保存された個別ファイルが優先 (人間が編集した可能性あり)、
    無ければ cowork_output/ 内の **全ファイル** をスキャンして race_id で引く。
    日付ファイル名に縛られないので、複数日まとめた 1 ファイルでも OK。
    """
    individual = load_cowork_bets(date_str, race_id)
    if individual:
        return individual
    all_bets = _load_all_cowork_output(_cowork_output_cache_key())
    return all_bets.get(race_id)


# ─────────────────────────────────────────────────────────────────────────────
# Grade Scope (重賞 LLM 詳細見解) ローダー
#
# Cowork が wrapper 形式 ({"bets": [...], "grade_scope": [...]}) で出力する
# G1/G2/G3 詳細見解を cowork_output/ から抽出する。
# scope[].race_id をキーに dict 化して返す。
# ─────────────────────────────────────────────────────────────────────────────

def _parse_one_grade_scope_file(path: Path) -> dict[str, dict]:
    """1 ファイルから grade_scope[] を抽出 → race_id 別 dict。"""
    try:
        raw_bytes = path.read_bytes()
        text = ""
        for enc in ["utf-8-sig", "utf-8", "cp932", "shift_jis"]:
            try:
                text = raw_bytes.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            text = raw_bytes.decode("utf-8", errors="replace")
    except Exception:
        return {}

    m = re.search(r"```(?:json|JSON)?\s*\n([\s\S]+?)\n\s*```", text)
    raw_json = m.group(1) if m else text.strip()

    try:
        data = json.loads(raw_json)
    except Exception:
        return {}

    # wrapper 形式のみ grade_scope を含む。レガシー (top-level list) には無い
    if not isinstance(data, dict):
        return {}
    scope_list = data.get("grade_scope") or []
    if not isinstance(scope_list, list):
        return {}

    out: dict[str, dict] = {}
    for entry in scope_list:
        if not isinstance(entry, dict):
            continue
        rid_raw = entry.get("race_id")
        if not rid_raw:
            continue
        rid = str(rid_raw)[:16]
        out[rid] = {
            "race_id":    rid,
            "race_label": str(entry.get("race_label", "")),
            "class":      str(entry.get("class", "")),
            "markdown":   str(entry.get("markdown", "")),
            "source":     f"cowork_output:{path.name}",
        }
    return out


@functools.lru_cache(maxsize=4)
def _load_all_grade_scope(_cache_key: str) -> dict[str, dict]:
    """cowork_output/ 全ファイルから grade_scope を集約 → race_id 別 dict。
    後勝ち (新しい mtime のファイルが優先)。
    """
    if not COWORK_OUTPUT_DIR.exists():
        return {}
    out: dict[str, dict] = {}
    files = sorted(
        [p for p in COWORK_OUTPUT_DIR.iterdir() if p.is_file()
         and p.suffix.lower() in (".json", ".txt", ".md")],
        key=lambda x: x.stat().st_mtime,
    )
    for p in files:
        out.update(_parse_one_grade_scope_file(p))
    return out


def load_grade_scope_for_date(date_str: str) -> list[dict]:
    """指定日付の重賞詳細見解を取得 (race_id 先頭 8 桁が date_str に一致するもの)。
    出力は race_id 順にソート済みリスト。
    """
    all_scope = _load_all_grade_scope(_cowork_output_cache_key())
    matched = [v for rid, v in all_scope.items() if rid.startswith(date_str)]
    return sorted(matched, key=lambda x: x["race_id"])


# ============================================================
# weekly/{date}.csv パーサ (Streamlit と同じ多行フォーマット対応)
#   - 19列 = race 行 (RACE_COLS)
#   - 46/48/49/99列 = horse 行 (HORSE_COLS_*)
#   bundle.json に無い 前走 系特徴量 (前走Ave-3F, 前走人気, 前走単勝オッズ,
#   前走確定着順, 前走上り3F) をここから抽出して PyCaLi 評価に使う
# ============================================================
@functools.lru_cache(maxsize=8)
def load_weekly_horses(date_str: str) -> pd.DataFrame | None:
    """weekly/{date}.csv を読んで horse-level DataFrame を返す。

    返り値の各行は 1 馬。列に race_id_16, 馬番, 前走Ave-3F, 前走人気,
    前走単勝オッズ, 前走確定着順, 前走上り3F などが含まれる。
    """
    p = WEEKLY_DIR / f"{date_str}.csv"
    if not p.exists():
        p = WEEKLY_DIR / f"{date_str}.CSV"   # 大文字拡張子フォールバック
    if not p.exists():
        return None
    raw = p.read_bytes()
    text = ""
    for enc in ["cp932", "utf-8-sig", "utf-8"]:
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if not text:
        return None

    races: list[dict] = []
    current_race: dict | None = None
    for line in text.splitlines():
        cols = line.split(",")
        if not cols or cols[0] in ("レースID(新)", "枠番", "番", ""):
            continue
        if len(cols) == 19:
            current_race = dict(zip(RACE_COLS, cols))
        elif len(cols) == 46 and current_race:
            h = dict(zip(HORSE_COLS_46, cols)); h.update(current_race)
            races.append(h)
        elif len(cols) == 48 and current_race:
            h = dict(zip(HORSE_COLS_48, cols)); h.update(current_race)
            races.append(h)
        elif len(cols) == 49 and current_race:
            h = dict(zip(HORSE_COLS_49, cols)); h.update(current_race)
            races.append(h)
        elif len(cols) == 99 and current_race:
            h = dict(zip(HORSE_COLS_99, cols)); h.update(current_race)
            races.append(h)
    if not races:
        return None

    df = pd.DataFrame(races).rename(columns=COLUMN_MAP)
    df["race_id_16"] = df["レースID(新)"].astype(str).str[:16]
    df["馬番"] = pd.to_numeric(df["馬番"], errors="coerce")
    for col in ["前走確定着順", "前走人気", "前走単勝オッズ",
                "前走Ave-3F", "前走上り3F", "前走上り3F順"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@functools.lru_cache(maxsize=1)
def load_hosei_all() -> pd.DataFrame | None:
    """data/hosei/H_*.csv を全 glob して 前走補正/前走補9 を返す (Streamlit と同手順)"""
    if not HOSSEI_DIR.exists():
        return None
    files = sorted(HOSSEI_DIR.glob("H_*.csv"))
    if not files:
        return None
    dfs = []
    for path in files:
        for enc in ["cp932", "utf-8-sig", "utf-8"]:
            try:
                d = pd.read_csv(
                    path, encoding=enc,
                    usecols=["レースID(新)", "馬番", "前走補9", "前走補正"],
                )
                dfs.append(d)
                break
            except Exception:
                continue
    if not dfs:
        return None
    res = pd.concat(dfs, ignore_index=True).drop_duplicates()
    res["race_id_16"] = res["レースID(新)"].astype(str).str[:16]
    res["馬番"] = pd.to_numeric(res["馬番"], errors="coerce")
    for col in ["前走補9", "前走補正"]:
        res[col] = pd.to_numeric(res[col], errors="coerce")
    return res[["race_id_16", "馬番", "前走補9", "前走補正"]]


@functools.lru_cache(maxsize=1)
def load_training_all() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """data/training/H-*.csv (坂路) と W-*.csv (WC) を全 glob して結合。

    H 列: 場所,年月日,馬名,Time1(4F全体),Time2,Time3,Time4(ラスト1F),Lap1...
    W 列: 場所,年月日,馬名,5F,4F,3F,Lap1,Lap2,Lap3
    """
    if not TRAINING_DIR.exists():
        return None, None
    h_dfs, w_dfs = [], []
    for fp in sorted(TRAINING_DIR.glob("H-*.csv")):
        for enc in ["cp932", "utf-8-sig", "utf-8"]:
            try:
                d = pd.read_csv(fp, encoding=enc, dtype=str,
                                  on_bad_lines="skip")
                if "年月日" not in d.columns or "馬名" not in d.columns:
                    break
                d["年月日"] = pd.to_numeric(d["年月日"], errors="coerce")
                for c in ["Time1", "Time4", "Lap1"]:
                    if c in d.columns:
                        d[c] = pd.to_numeric(d[c], errors="coerce")
                h_dfs.append(d)
                break
            except Exception:
                continue
    for fp in sorted(TRAINING_DIR.glob("W-*.csv")):
        for enc in ["cp932", "utf-8-sig", "utf-8"]:
            try:
                d = pd.read_csv(fp, encoding=enc, dtype=str,
                                  on_bad_lines="skip")
                if "年月日" not in d.columns or "馬名" not in d.columns:
                    break
                d["年月日"] = pd.to_numeric(d["年月日"], errors="coerce")
                for c in ["5F", "3F"]:
                    if c in d.columns:
                        d[c] = pd.to_numeric(d[c], errors="coerce")
                w_dfs.append(d)
                break
            except Exception:
                continue
    h_all = pd.concat(h_dfs, ignore_index=True) if h_dfs else None
    w_all = pd.concat(w_dfs, ignore_index=True) if w_dfs else None
    return h_all, w_all


def _latest_training(name: str, race_date_int: int) -> dict[str, float | None]:
    """1 頭の最新調教 (race 日以前) を取得。Streamlit の _attach_latest_training と同じ。"""
    out: dict[str, float | None] = {
        "trn_hanro_lap1": None, "trn_hanro_time1": None,
        "trn_wc_3f": None, "trn_wc_5f": None,
    }
    if not name:
        return out
    h_all, w_all = load_training_all()
    if h_all is not None:
        sub = h_all[(h_all["馬名"] == name) &
                     (h_all["年月日"] < race_date_int)]
        if not sub.empty:
            latest = sub.sort_values("年月日").iloc[-1]
            t1 = latest.get("Time1")
            l1 = latest.get("Lap1")
            if pd.notna(t1):
                out["trn_hanro_time1"] = float(t1)
            if pd.notna(l1):
                out["trn_hanro_lap1"] = float(l1)
    if w_all is not None:
        sub = w_all[(w_all["馬名"] == name) &
                     (w_all["年月日"] < race_date_int)]
        if not sub.empty:
            latest = sub.sort_values("年月日").iloc[-1]
            f5 = latest.get("5F")
            f3 = latest.get("3F")
            if pd.notna(f5):
                out["trn_wc_5f"] = float(f5)
            if pd.notna(f3):
                out["trn_wc_3f"] = float(f3)
    return out


def _safe_int(s) -> int | None:
    try:
        v = int(float(s)) if s not in ("", None) else None
        return v
    except (ValueError, TypeError):
        return None


def _safe_float(s) -> float | None:
    try:
        v = float(s) if s not in ("", None) else None
        return v
    except (ValueError, TypeError):
        return None


# ============================================================
# Cowork 累計 P/L 計算用: kekka パース + 払戻判定
# ============================================================
@functools.lru_cache(maxsize=1)
def parse_wide_kekka() -> dict[tuple, dict[tuple[int, int], int]] | None:
    """data/kekka/wide_kekka.csv をパース。

    フォーマット (cp932, no header):
      年,月,日,場所,R,クラス,芝・ダ,距離,頭数,"01-03 \220 (1)/ 04-11 \870 (12)/ ..."

    返り値: {(year, month, day, place, R): {(ban_low, ban_high): payout, ...}}
    """
    p = BASE / "data" / "kekka" / "wide_kekka.csv"
    if not p.exists():
        return None
    import csv as _csv
    pair_pat = re.compile(r"(\d+)\s*[-―]\s*(\d+)\s*[\\¥￥]\s*(\d+)")
    out: dict = {}
    try:
        text = ""
        raw = p.read_bytes()
        for enc in ["cp932", "utf-8-sig", "utf-8"]:
            try:
                text = raw.decode(enc); break
            except UnicodeDecodeError:
                continue
        else:
            text = raw.decode("utf-8", errors="replace")

        import io as _io
        reader = _csv.reader(_io.StringIO(text))
        for row in reader:
            if len(row) < 10:
                continue
            try:
                year  = int(row[0])
                month = int(row[1])
                day   = int(row[2])
                place = str(row[3]).strip()
                r_num = int(row[4])
            except (ValueError, TypeError):
                continue
            wide_str = row[9] or ""
            pairs: dict[tuple[int, int], int] = {}
            for chunk in wide_str.split("/"):
                m = pair_pat.search(chunk)
                if not m:
                    continue
                try:
                    a, b, pay = int(m.group(1)), int(m.group(2)), int(m.group(3))
                except (ValueError, TypeError):
                    continue
                pairs[(min(a, b), max(a, b))] = pay
            if pairs:
                out[(year, month, day, place, r_num)] = pairs
    except Exception as e:
        print(f"[parse_wide_kekka error] {e}")
        return None
    return out


@functools.lru_cache(maxsize=64)
def parse_kekka(date_str: str) -> dict[str, dict] | None:
    """data/kekka/{date}.csv をパースして race_id_16 → 結果 dict を返す。

    kekka 列構成 (15 cols):
      日付, 場所, Ｒ, 枠番, 馬番, 馬名, 確定着順, レースID(新, 18 桁),
      単勝配当, 複勝配当, 枠連, 馬連, 馬単, ３連複, ３連単

    返り値 dict 各 race:
      winner: 1着 馬番
      top3:   [1着, 2着, 3着] (存在する分のみ)
      tansho: 1着の単勝配当 (¥)
      fukusho_by_uma: {馬番: 複勝配当}
      umaren / umatan / wakuren / sanrenpuku / sanrentan: 各払戻金 (¥)
    """
    p = BASE / "data" / "kekka" / f"{date_str}.csv"
    if not p.exists():
        return None
    import csv as _csv
    races: dict[str, dict] = {}

    try:
        with open(p, encoding="cp932", errors="replace") as f:
            reader = _csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) < 15:
                    continue
                rid_18 = str(row[7]).strip()
                if len(rid_18) < 16:
                    continue
                rid_16 = rid_18[:16]
                try:
                    umaban = int(row[4])
                    chakujun = int(row[6])
                except (ValueError, TypeError):
                    continue
                if rid_16 not in races:
                    races[rid_16] = {
                        "date": row[0],
                        "place": row[1],
                        "R": row[2],
                        "_horses_raw": [],
                        "umaren": None, "umatan": None,
                        "wakuren": None,
                        "sanrenpuku": None, "sanrentan": None,
                    }
                # 単勝配当: 1 着のみ実数値、他着は () 付きの参考値
                tansho_raw = str(row[8]).strip()
                tansho = None
                if tansho_raw and not tansho_raw.startswith("("):
                    try: tansho = int(tansho_raw)
                    except ValueError: pass
                fukusho = _safe_int(row[9])
                races[rid_16]["_horses_raw"].append({
                    "umaban": umaban,
                    "chakujun": chakujun,
                    "tansho": tansho,
                    "fukusho": fukusho,
                })
                # Compound payouts: 1 着行のみ実値、2-3 着行は空。
                # None で既存値を上書きしないよう注意。
                for key, idx in [("wakuren", 10), ("umaren", 11),
                                  ("umatan", 12), ("sanrenpuku", 13),
                                  ("sanrentan", 14)]:
                    v = _safe_int(row[idx]) if idx < len(row) else None
                    if v is not None and races[rid_16].get(key) is None:
                        races[rid_16][key] = v
    except Exception as e:
        print(f"[parse_kekka error {date_str}] {e}")
        return None

    # Post-process: extract winner / top3 / payouts
    wide_data = parse_wide_kekka()
    for rid_16, r in races.items():
        horses = sorted(r["_horses_raw"], key=lambda h: h["chakujun"])
        r["winner"] = horses[0]["umaban"] if horses else None
        r["tansho"] = horses[0]["tansho"] if horses else None
        r["top3"] = [h["umaban"] for h in horses if h["chakujun"] <= 3]
        r["fukusho_by_uma"] = {h["umaban"]: h["fukusho"]
                                  for h in horses if h["chakujun"] <= 3}
        # ワイド払戻 (wide_kekka.csv から JOIN)
        r["wide_pays"] = {}
        if wide_data:
            try:
                year  = int(rid_16[:4])
                month = int(rid_16[4:6])
                day   = int(rid_16[6:8])
                r_num = int(r.get("R") or 0)
                key = (year, month, day, r.get("place", ""), r_num)
                wide_pays = wide_data.get(key)
                if wide_pays:
                    r["wide_pays"] = wide_pays
            except (ValueError, TypeError):
                pass
        del r["_horses_raw"]
    return races


def _parse_combos(selection: str, n_parts: int,
                    ordered: bool) -> list[tuple]:
    """買い目文字列 '4-7,4-9' のような表記を combo タプルのリストにする。
    n_parts: 2 (馬連/馬単/ワイド) or 3 (三連複/三連単)
    ordered: True なら順序を保つ (馬単/三連単)、False なら sorted
    """
    out: list[tuple] = []
    for c in str(selection).split(","):
        parts = c.strip().split("-")
        if len(parts) != n_parts:
            continue
        try:
            nums = [int(x) for x in parts]
        except (ValueError, TypeError):
            continue
        out.append(tuple(nums) if ordered else tuple(sorted(nums)))
    return out


def compute_bet_pl(bet: dict, race: dict) -> tuple[float, bool]:
    """1 つの Cowork bet (馬券種/買い目/購入額) と race 結果 dict から
    (利益¥, 的中フラグ) を返す。利益 = (受取 - 支払)。
    複数 combo 指定時は 購入額を均等分配して計算 (JRA 標準慣行)。
    """
    btype = (bet.get("馬券種") or bet.get("type") or "").strip()
    selection = str(bet.get("買い目") or bet.get("selection") or "").strip()
    try:
        cost = float(bet.get("購入額") or bet.get("amount") or 0)
    except (TypeError, ValueError):
        cost = 0.0
    if cost <= 0:
        return (0.0, False)

    if btype == "単勝":
        try:
            uma = int(selection)
        except ValueError:
            return (-cost, False)
        if uma == race.get("winner"):
            pay = race.get("tansho") or 0
            return (cost * pay / 100.0 - cost, True)
        return (-cost, False)

    if btype == "複勝":
        try:
            uma = int(selection)
        except ValueError:
            return (-cost, False)
        if uma in race.get("top3", []):
            pay = race.get("fukusho_by_uma", {}).get(uma) or 0
            return (cost * pay / 100.0 - cost, True)
        return (-cost, False)

    if btype == "馬連":
        combos = _parse_combos(selection, 2, ordered=False)
        if not combos:
            return (-cost, False)
        top3 = race.get("top3", [])
        if len(top3) < 2:
            return (-cost, False)
        winning = tuple(sorted([top3[0], top3[1]]))
        unit = cost / len(combos)
        if winning in combos:
            pay = race.get("umaren") or 0
            return (unit * pay / 100.0 - cost, True)
        return (-cost, False)

    if btype == "馬単":
        combos = _parse_combos(selection, 2, ordered=True)
        if not combos:
            return (-cost, False)
        top3 = race.get("top3", [])
        if len(top3) < 2:
            return (-cost, False)
        winning = (top3[0], top3[1])
        unit = cost / len(combos)
        if winning in combos:
            pay = race.get("umatan") or 0
            return (unit * pay / 100.0 - cost, True)
        return (-cost, False)

    if btype == "ワイド":
        combos = _parse_combos(selection, 2, ordered=False)
        if not combos:
            return (-cost, False)
        wide_pays = race.get("wide_pays") or {}
        if not wide_pays:
            # データ無し → 損失計上 (wide_kekka.csv が未配置の date 等)
            return (-cost, False)
        unit = cost / len(combos)
        total_payout = 0.0
        n_hit = 0
        for combo in combos:
            if combo in wide_pays:
                total_payout += unit * (wide_pays[combo] or 0) / 100.0
                n_hit += 1
        if n_hit > 0:
            return (total_payout - cost, True)
        return (-cost, False)

    if btype == "三連複":
        combos = _parse_combos(selection, 3, ordered=False)
        if not combos:
            return (-cost, False)
        top3 = race.get("top3", [])
        if len(top3) < 3:
            return (-cost, False)
        winning = tuple(sorted(top3[:3]))
        unit = cost / len(combos)
        if winning in combos:
            pay = race.get("sanrenpuku") or 0
            return (unit * pay / 100.0 - cost, True)
        return (-cost, False)

    if btype == "三連単":
        combos = _parse_combos(selection, 3, ordered=True)
        if not combos:
            return (-cost, False)
        top3 = race.get("top3", [])
        if len(top3) < 3:
            return (-cost, False)
        winning = tuple(top3[:3])
        unit = cost / len(combos)
        if winning in combos:
            pay = race.get("sanrentan") or 0
            return (unit * pay / 100.0 - cost, True)
        return (-cost, False)

    # unknown bet type
    return (-cost, False)


def _kekka_files_cache_key() -> str:
    """kekka/ 内の全ファイルの mtime ハッシュ (ファイル追加で auto invalidate)"""
    kdir = BASE / "data" / "kekka"
    if not kdir.exists():
        return "no-dir"
    files = sorted(kdir.glob("*.csv"))
    return "|".join(f"{p.name}:{p.stat().st_mtime:.0f}" for p in files)


@functools.lru_cache(maxsize=4)
def load_all_cowork_outcomes(_cache_key: str = "") -> list[dict]:
    """全 reports/cowork_output/*_bets.json × data/kekka/*.csv を JOIN して
    bet 単位の収支リストを返す。
    各行: {date, race_id, race_label, btype, selection, cost, profit, is_win}
    """
    rows: list[dict] = []
    if not COWORK_OUTPUT_DIR.exists():
        return rows

    for bets_file in sorted(COWORK_OUTPUT_DIR.iterdir()):
        if not bets_file.is_file():
            continue
        if bets_file.suffix.lower() not in (".json", ".txt", ".md"):
            continue
        # date は filename から抽出 (YYYYMMDD_bets.json or 同類)
        stem = bets_file.stem
        date_str = ""
        # まず先頭 8 桁数字を試す
        for i in range(min(8, len(stem)), 7, -1):
            if stem[:i].isdigit() and len(stem[:i]) == 8:
                date_str = stem[:8]
                break
        if not date_str:
            continue
        kekka = parse_kekka(date_str)
        if not kekka:
            continue

        # bets file パース
        try:
            raw_bytes = bets_file.read_bytes()
            text = ""
            for enc in ["utf-8-sig", "utf-8", "cp932", "shift_jis"]:
                try:
                    text = raw_bytes.decode(enc); break
                except UnicodeDecodeError:
                    continue
            else:
                text = raw_bytes.decode("utf-8", errors="replace")
            m = re.search(r"```(?:json|JSON)?\s*\n([\s\S]+?)\n\s*```", text)
            raw_json = m.group(1) if m else text.strip()
            data = json.loads(raw_json)
        except Exception:
            continue

        if isinstance(data, dict):
            # wrapper 形式に両対応:
            #   - {"races": [...]}: 旧
            #   - {"bets": [...], "grade_scope": [...]}: 新 (Grade Scope 対応)
            data = data.get("bets") or data.get("races", [data])
        if not isinstance(data, list):
            continue

        for race_entry in data:
            if not isinstance(race_entry, dict):
                continue
            rid = race_entry.get("race_id") or race_entry.get("レースID") or ""
            rid_16 = str(rid)[:16]
            result = kekka.get(rid_16)
            if not result:
                continue
            bets_raw = race_entry.get("bets") or race_entry.get("買い目", [])
            if not isinstance(bets_raw, list):
                continue
            for bet in bets_raw:
                if not isinstance(bet, dict):
                    continue
                profit, is_win = compute_bet_pl(bet, result)
                rows.append({
                    "date": date_str,
                    "race_id": rid_16,
                    "race_label": race_entry.get("race_label", ""),
                    "btype": (bet.get("馬券種") or bet.get("type") or ""),
                    "selection": str(bet.get("買い目")
                                       or bet.get("selection") or ""),
                    "cost": float(bet.get("購入額")
                                    or bet.get("amount") or 0),
                    "profit": profit,
                    "is_win": is_win,
                })
    return rows


@functools.lru_cache(maxsize=8)
def load_kako5_horses(date_str: str) -> dict | None:
    """kako5/{date}.csv を読んで (race_id_16, umaban) → 過去5走 dict を作る。

    kako5 ファイル構造:
      - 19 列の race header 行 (race ID 先頭)
      - 72 列の horse data 行 (馬番 col 2、馬名S col 7、5 走 × 12 列が後続)

    各 past race block (12 cols): 月, 日, 場所, TD, 距離, 馬場, 着順, 人気,
                                    レース名, 上り3F, 決手, 間隔
    """
    p = BASE / "data" / "kako5" / f"{date_str}.csv"
    if not p.exists():
        return None
    import csv as _csv
    out: dict = {}
    current_rid_16 = None

    try:
        with open(p, encoding="cp932", errors="replace") as f:
            reader = _csv.reader(f)
            for row in reader:
                if len(row) <= 1:
                    continue
                # Race header (19 cols, race ID starts with year digits)
                if (len(row) == 19 and row[0]
                        and len(row[0]) >= 10 and row[0][:4].isdigit()):
                    current_rid_16 = row[0][:16]
                    continue
                # Column header
                if len(row) == 72 and row[0] in ("枠番",):
                    continue
                # Data row
                if (len(row) == 72 and row[0] and row[0].isdigit()
                        and current_rid_16):
                    umaban = _safe_int(row[2])
                    if umaban is None:
                        continue
                    name = str(row[7]).strip() if len(row) > 7 else ""

                    past_races: list[dict] = []
                    # 5 race blocks at offsets 12, 24, 36, 48, 60 (12 cols each)
                    for i in range(5):
                        base = 12 + i * 12
                        if base + 11 >= len(row):
                            break
                        chakujun = _safe_int(row[base + 6])
                        if chakujun is None or chakujun == 0:
                            continue
                        past_races.append({
                            "month":    str(row[base + 0]),
                            "day":      str(row[base + 1]),
                            "place":    str(row[base + 2]),
                            "td":       str(row[base + 3]),
                            "dist":     _safe_int(row[base + 4]),
                            "baba":     str(row[base + 5]),
                            "chakujun": chakujun,
                            "ninki":    _safe_int(row[base + 7]),
                            "race_name": str(row[base + 8]),
                            "agari3f":  _safe_float(row[base + 9]),
                            "kimete":   str(row[base + 10]),
                        })
                    out[(current_rid_16, umaban)] = {
                        "name": name,
                        "past_races": past_races,
                    }
    except Exception as e:
        print(f"[load_kako5_horses error] {e}")
        return None
    return out


@functools.lru_cache(maxsize=8)
def compute_training_top5(date_str: str, days_back: int = 7) -> dict:
    """直近 days_back 日間の調教ベストタイム 上位 5 頭 (坂路 + WC)。

    **当週末 (bundle.json) に出走する馬のみ** を対象とする。
    調教は速くても出走しない馬を出しても役に立たないため。

    坂路 (data/training/H-*.csv):
      - Lap1 (ラスト 1F、終い時計) が小さい順 (速い)
    WC (data/training/W-*.csv):
      - 3F (3F 合計タイム) が小さい順 (速い)

    返り値: {"hanro": [...], "wc": [...]}
    各 item: {"name", "lap1"/"f3", "time1"/"f5", "date", "place", "race"}
    """
    out: dict[str, list[dict]] = {"hanro": [], "wc": []}
    if not date_str or len(date_str) != 8 or not date_str.isdigit():
        return out

    h_all, w_all = load_training_all()
    if h_all is None and w_all is None:
        return out

    try:
        race_date_int = int(date_str)
    except ValueError:
        return out

    from datetime import datetime, timedelta
    race_dt = datetime.strptime(date_str, "%Y%m%d")
    cutoff_dt = race_dt - timedelta(days=days_back)
    cutoff_int = int(cutoff_dt.strftime("%Y%m%d"))

    # 当週末出走馬の名前セットを bundle.json から取得 (Top5 のフィルタ)
    bundle = load_bundle(date_str)
    entering_names: set[str] = set()
    if bundle:
        for race in bundle.get("races", []):
            for h in race.get("horses", []):
                nm = (h.get("horse_name") or "").strip()
                if nm:
                    entering_names.add(nm)

    # 坂路 best 5 (馬ごとに最良 Lap1 をとる、Lap1 = 終い1F 秒、低いほど速い)
    # 注: 0 や NaN 等の不完全行を除外 (Lap1 は典型的に 12-17 秒)
    # H CSV のヘッダ: 年月日, 馬名, Time1, Time2, Time3, Time4, Lap4, Lap3, Lap2, Lap1
    #               場所列は無いので "" を出す
    if h_all is not None and {"Lap1", "馬名", "年月日"}.issubset(h_all.columns):
        recent_h = h_all[(h_all["年月日"] >= cutoff_int) &
                          (h_all["年月日"] < race_date_int)].copy()
        recent_h = recent_h.dropna(subset=["Lap1", "馬名"])
        recent_h = recent_h[recent_h["Lap1"] >= 10.0]   # ノイズ行除外
        # 当週末出走馬に絞る (entering_names が空なら filter スキップ)
        if entering_names:
            recent_h = recent_h[recent_h["馬名"].astype(str).str.strip()
                                  .isin(entering_names)]
        if not recent_h.empty:
            best_idx = recent_h.groupby("馬名")["Lap1"].idxmin()
            best = recent_h.loc[best_idx].sort_values("Lap1").head(5)
            for _, r in best.iterrows():
                out["hanro"].append({
                    "name": str(r["馬名"]).strip(),
                    "lap1": float(r["Lap1"]),
                    "time1": float(r["Time1"]) if "Time1" in r.index and pd.notna(r["Time1"]) else None,
                    "date":  int(r["年月日"]),
                    "place": str(r.get("場所", "")) if "場所" in r.index else "",
                })

    # WC best 5 (3F = 3 ハロン合計秒、低いほど速い、典型 33-40 秒)
    if w_all is not None and {"3F", "馬名", "年月日"}.issubset(w_all.columns):
        recent_w = w_all[(w_all["年月日"] >= cutoff_int) &
                          (w_all["年月日"] < race_date_int)].copy()
        recent_w = recent_w.dropna(subset=["3F", "馬名"])
        recent_w = recent_w[recent_w["3F"] >= 20.0]   # ノイズ除外
        if entering_names:
            recent_w = recent_w[recent_w["馬名"].astype(str).str.strip()
                                  .isin(entering_names)]
        if not recent_w.empty:
            best_idx = recent_w.groupby("馬名")["3F"].idxmin()
            best = recent_w.loc[best_idx].sort_values("3F").head(5)
            for _, r in best.iterrows():
                out["wc"].append({
                    "name": str(r["馬名"]).strip(),
                    "f3":   float(r["3F"]),
                    "f5":   float(r["5F"]) if "5F" in r.index and pd.notna(r["5F"]) else None,
                    "date": int(r["年月日"]),
                    "place": str(r.get("場所", "")) if "場所" in r.index else "",
                })

    # 出走レース照合 (bundle.json から 馬名 で引く)
    bundle = load_bundle(date_str)
    name_to_race: dict[str, dict] = {}
    if bundle:
        for race in bundle.get("races", []):
            rid = race.get("race_id", "")
            meta = race.get("race_meta", {}) or {}
            place = meta.get("place", "")
            r_num = rid[-2:].lstrip("0") if len(rid) >= 16 else "?"
            course = meta.get("course", "")
            for h in race.get("horses", []):
                hname = (h.get("horse_name") or "").strip()
                if hname:
                    name_to_race[hname] = {
                        "place": place,
                        "race_num": r_num,
                        "race_id": rid,
                        "course": course,
                        "mark": h.get("mark") or "",
                        "p_win": h.get("p_win") or 0,
                    }

    for item in out["hanro"] + out["wc"]:
        item["race"] = name_to_race.get(item["name"])

    return out


def get_horse_features(date_str: str, race_id: str,
                        umaban: int) -> dict[str, float | None]:
    """1 頭分の Streamlit 互換特徴量を辞書で返す。

    weekly CSV (前走Ave-3F, 前走人気 等) と hosei (前走補正, 前走補9) を merge。
    存在しない or 該当なし → None
    """
    rid_16 = str(race_id)[:16]
    out: dict[str, float | None] = {}
    keys = [
        "前走補正", "前走Ave-3F", "前走走破タイム",
        "前走補9", "前走上り3F",
        "前走確定着順", "前走人気", "前走単勝オッズ",
        "前走RPCI", "前走PCI3",
        "trn_hanro_lap1", "trn_hanro_time1",
        "trn_wc_3f", "trn_wc_5f",
    ]
    for k in keys:
        out[k] = None

    horse_name = ""
    weekly = load_weekly_horses(date_str)
    if weekly is not None:
        sub = weekly[(weekly["race_id_16"] == rid_16) &
                       (weekly["馬番"] == int(umaban))]
        if not sub.empty:
            row = sub.iloc[0]
            for k in keys:
                if k in row.index:
                    v = row[k]
                    if pd.notna(v):
                        try:
                            out[k] = float(v)
                        except (TypeError, ValueError):
                            pass
            if "馬名S" in row.index:
                horse_name = str(row["馬名S"]).strip()

    hosei = load_hosei_all()
    if hosei is not None:
        sub = hosei[(hosei["race_id_16"] == rid_16) &
                     (hosei["馬番"] == int(umaban))]
        if not sub.empty:
            row = sub.iloc[0]
            for k in ["前走補正", "前走補9"]:
                v = row.get(k)
                if pd.notna(v):
                    out[k] = float(v)

    # 調教 (g 軸): data/training/ が存在すれば 馬名 + race 日付 で検索
    if horse_name and len(date_str) == 8 and date_str.isdigit():
        try:
            race_date_int = int(date_str)
            trn = _latest_training(horse_name, race_date_int)
            for k, v in trn.items():
                if v is not None:
                    out[k] = v
        except Exception:
            pass

    return out


# ============================================================
# Master CSV 読み込み (起動時 1 回、コース統計用)
# ============================================================
@functools.lru_cache(maxsize=1)
def get_master_df() -> pd.DataFrame | None:
    """master_v2 から必要列だけ読み込み、メモリ最適化してキャッシュ。"""
    if not MASTER_CSV.exists():
        return None
    try:
        # 必要列のみ + 型最適化 (コース分析タブで年齢/性別/脚質を使うため拡張)
        usecols = ["日付", "場所", "Ｒ", "枠番", "馬番", "着順",
                    "芝・ダ", "距離", "馬場状態", "年齢", "性別",
                    "前4角", "出走頭数"]
        df = pd.read_csv(
            MASTER_CSV, encoding="utf-8-sig",
            usecols=usecols, dtype=str, low_memory=False, on_bad_lines="skip",
        )
        df["日付"] = pd.to_numeric(df["日付"], errors="coerce")
        df["枠番"] = pd.to_numeric(df["枠番"], errors="coerce")
        df["馬番"] = pd.to_numeric(df["馬番"], errors="coerce")
        df["着順"] = pd.to_numeric(df["着順"], errors="coerce")
        df["距離"] = pd.to_numeric(df["距離"], errors="coerce")
        df["年齢"] = pd.to_numeric(df["年齢"], errors="coerce")
        if "前4角" in df.columns:
            df["前4角"] = pd.to_numeric(df["前4角"], errors="coerce")
        if "出走頭数" in df.columns:
            df["出走頭数"] = pd.to_numeric(df["出走頭数"], errors="coerce")
        df = df.dropna(subset=["着順", "距離"]).copy()
        return df
    except Exception as e:
        print(f"[master 読込失敗] {e}")
        return None


def parse_course_str(course_str: str) -> tuple[str | None, int | None]:
    """'ダート1200' → ('ダ', 1200)、'芝2000' → ('芝', 2000)"""
    if not course_str:
        return None, None
    if "ダート" in course_str or "ダ" in course_str:
        surface = "ダ"
    elif "芝" in course_str:
        surface = "芝"
    else:
        return None, None
    m = re.search(r"\d+", course_str)
    if not m:
        return surface, None
    return surface, int(m.group())


def _stats_breakdown(sub: pd.DataFrame, col: str,
                       values: list, label_fn=None) -> list[dict]:
    """sub DataFrame を col 別に集計し、1着/2着/3着/着外 + 各率を返す。"""
    out: list[dict] = []
    for v in values:
        grp = sub[sub[col] == v]
        total = len(grp)
        if total == 0:
            continue
        wins = int((grp["着順"] == 1).sum())
        top2 = int((grp["着順"] <= 2).sum())
        top3 = int((grp["着順"] <= 3).sum())
        label = label_fn(v) if label_fn else str(v)
        out.append({
            "label": label,
            "n_1": wins,
            "n_2": top2 - wins,
            "n_3": top3 - top2,
            "n_out": total - top3,
            "n_total": total,
            "win_rate":    wins / total * 100,
            "rentai_rate": top2 / total * 100,
            "fuku_rate":   top3 / total * 100,
        })
    return out


COURSE_STATS_JSON = BASE / "data" / "course_stats.json"
CHAOS_Q_JSON = BASE / "data" / "chaos_quantiles.json"


@functools.lru_cache(maxsize=1)
def _load_quantile_tables() -> dict:
    """build_chaos_quantiles.py が生成した分位テーブルをロード。"""
    if not CHAOS_Q_JSON.exists():
        return {}
    try:
        return json.loads(CHAOS_Q_JSON.read_text(encoding="utf-8")).get("quantiles", {})
    except Exception:
        return {}


def _to_pct(raw: float, key: str = "field_chaos_score") -> float:
    """生値を過去分布のパーセンタイル(0-1)に変換。

    正規化エントロピー等は値域が狭く(混戦度は 0.80-1.0)生値では判別力が無いため、
    過去レース分布での相対順位に変換して 0-1 に均等化する。テーブル欠如時は生値返し。
    """
    t = _load_quantile_tables().get(key)
    if not t or len(t) < 2:
        return float(raw)
    if raw <= t[0]:
        return 0.0
    if raw >= t[-1]:
        return 1.0
    i = bisect.bisect_right(t, raw)
    lo, hi = t[i - 1], t[i]
    frac = 0.0 if hi == lo else (raw - lo) / (hi - lo)
    return (i - 1 + frac) / (len(t) - 1)
PEDIGREE_STATS_JSON = BASE / "data" / "pedigree_stats.json"


@functools.lru_cache(maxsize=1)
def load_precomputed_course_stats() -> dict | None:
    """build_course_stats.py が生成した data/course_stats.json をロード。
    master_v2 が無い HF 環境でもコース分析が出るためのフォールバック。
    """
    if not COURSE_STATS_JSON.exists():
        return None
    try:
        with open(COURSE_STATS_JSON, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[course_stats.json 読込失敗] {e}")
        return None


@functools.lru_cache(maxsize=1)
def load_pedigree_stats() -> dict | None:
    """build_pedigree_data.py が生成した data/pedigree_stats.json をロード。"""
    if not PEDIGREE_STATS_JSON.exists():
        return None
    try:
        with open(PEDIGREE_STATS_JSON, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[pedigree_stats.json 読込失敗] {e}")
        return None


def lookup_pedigree_course(place: str, td: str, dist: int) -> dict | None:
    """コース key を生成して pedigree_stats から該当エントリを返す。
    厳密一致が無ければ ±200m / ±400m 帯で探索。"""
    data = load_pedigree_stats()
    if not data:
        return None
    courses = data.get("courses", {}) or {}
    # 距離 bucket は 200m 単位 + 100 中央値 (build 側と整合)
    primary_bucket = (dist // 200) * 200 + 100
    candidates = [primary_bucket]
    # 隣接 bucket も候補に
    candidates += [primary_bucket - 200, primary_bucket + 200,
                    primary_bucket - 400, primary_bucket + 400]
    for b in candidates:
        key = f"{place}_{td}_{b}"
        if key in courses:
            return courses[key]
    return None


@functools.lru_cache(maxsize=128)
def compute_course_stats_v2(place: str, course_str: str) -> dict | None:
    """同コース過去成績の全条件別統計 (枠 / 馬番 / 年齢 / 性別)。

    優先度:
      1. master_v2_*.csv があれば実時間集計 (常に最新)
      2. data/course_stats.json (build_course_stats.py で事前生成) を参照
      3. どちらも無ければ None

    返り値: { "n_races", "n_starts", "waku", "uma", "age", "sex" }
    各 dimension は [{"label","n_1","n_2","n_3","n_out","n_total",
                     "win_rate","rentai_rate","fuku_rate"}, ...]
    """
    if not place or not course_str:
        return None

    # 1. master_v2 ある場合 (ローカル)
    df = get_master_df()
    if df is not None:
        surface, dist = parse_course_str(course_str)
        if not surface or not dist:
            return None
        sub = df[(df["場所"] == place) &
                 (df["芝・ダ"] == surface) &
                 (df["距離"] == dist)]
        if len(sub) < 100:
            return None

        sub = sub.copy()
        n_races = sub["日付"].nunique() if "日付" in sub.columns else len(sub) // 14
        n_starts = len(sub)

        waku = _stats_breakdown(sub, "枠番", list(range(1, 9)),
                                  label_fn=lambda v: str(v))
        uma_values = sorted(sub["馬番"].dropna().unique().astype(int))
        uma_values = [v for v in uma_values if 1 <= v <= 18]
        uma = _stats_breakdown(sub, "馬番", uma_values,
                                 label_fn=lambda v: str(v))

        age = _stats_breakdown(sub, "年齢", [3, 4, 5, 6, 7],
                                 label_fn=lambda v: f"{v}歳")
        grp_8 = sub[sub["年齢"] >= 8]
        total = len(grp_8)
        if total > 0:
            wins = int((grp_8["着順"] == 1).sum())
            top2 = int((grp_8["着順"] <= 2).sum())
            top3 = int((grp_8["着順"] <= 3).sum())
            age.append({
                "label": "8歳~",
                "n_1": wins, "n_2": top2 - wins, "n_3": top3 - top2,
                "n_out": total - top3, "n_total": total,
                "win_rate":    wins / total * 100,
                "rentai_rate": top2 / total * 100,
                "fuku_rate":   top3 / total * 100,
            })

        sex = _stats_breakdown(sub, "性別", ["牡", "牝", "セ"],
                                 label_fn=lambda v: {"牡":"牡馬","牝":"牝馬","セ":"セン馬"}.get(v, v))

        # 脚質別好走 (前4角 + 出走頭数 から各馬の脚質を推定)
        kyaku: list[dict] = []
        if {"前4角", "出走頭数"}.issubset(sub.columns):
            sub2 = sub.copy()
            sub2["_kyaku"] = [
                _classify_kyakushitsu(None, p, int(f) if pd.notna(f) else 16)
                for p, f in zip(sub2["前4角"], sub2["出走頭数"])
            ]
            kyaku = _stats_breakdown(sub2, "_kyaku",
                                       ["逃げ", "先行", "差し", "追込"])

        return {
            "place": place, "course": course_str,
            "n_races": n_races, "n_starts": n_starts,
            "waku": waku, "uma": uma, "age": age, "sex": sex,
            "kyaku": kyaku,
        }

    # 2. master_v2 が無い場合 (HF) → 事前生成 JSON にフォールバック
    cache = load_precomputed_course_stats()
    if cache is None:
        return None
    key = f"{place}|{course_str}"
    entry = cache.get(key)
    if not entry:
        return None
    return {
        "place": place,
        "course": course_str,
        "n_races":  entry.get("n_races", 0),
        "n_starts": entry.get("n_starts", 0),
        "waku":  entry.get("waku",  []),
        "uma":   entry.get("uma",   []),
        "age":   entry.get("age",   []),
        "sex":   entry.get("sex",   []),
        "kyaku": entry.get("kyaku", []),
    }


@functools.lru_cache(maxsize=128)
def compute_course_stats(place: str, course_str: str) -> dict | None:
    """同コース過去成績 + 馬場バイアス"""
    df = get_master_df()
    if df is None or not place or not course_str:
        return None
    surface, dist = parse_course_str(course_str)
    if not surface or not dist:
        return None

    sub = df[(df["場所"] == place) &
             (df["芝・ダ"] == surface) &
             (df["距離"] == dist)]
    if len(sub) < 100:
        return None

    n_races = sub["日付"].nunique() if "日付" in sub.columns else len(sub) // 14
    n_starts = len(sub)
    win_rate = (sub["着順"] == 1).mean() * 100  # 全馬で見たトリビア値だが、人気1のは別計算

    # 枠別 (内 1-2, 中 3-6, 外 7-8 の 3 段)
    def waku_band(w):
        if w <= 2:
            return "内枠"
        if w <= 6:
            return "中枠"
        return "外枠"
    sub = sub.copy()
    sub["_waku_band"] = sub["枠番"].apply(waku_band)
    waku_stats = {}
    for band in ["内枠", "中枠", "外枠"]:
        wb = sub[sub["_waku_band"] == band]
        if len(wb) > 0:
            waku_stats[band] = {
                "starts": len(wb),
                "wins": int((wb["着順"] == 1).sum()),
                "win_rate": (wb["着順"] == 1).mean() * 100,
                "top3_rate": (wb["着順"] <= 3).mean() * 100,
            }

    # 馬場状態 別 race 数
    baba_dist = {}
    if "馬場状態" in sub.columns:
        for baba, grp in sub.groupby("馬場状態"):
            baba_dist[str(baba)] = grp["日付"].nunique() if "日付" in grp.columns else len(grp)

    # 馬番 1 着の平均 (単勝何番が来やすいか)
    first_uma_avg = sub[sub["着順"] == 1]["馬番"].mean()

    return {
        "n_races": n_races,
        "n_starts": n_starts,
        "waku": waku_stats,
        "baba_dist": baba_dist,
        "first_uma_avg": first_uma_avg,
    }


# ============================================================
# 定数
# ============================================================
GRADE_COLORS = {
    "Ｇ１": "#e74c3c", "G1": "#e74c3c",
    "Ｇ２": "#9b59b6", "G2": "#9b59b6",
    "Ｇ３": "#2980b9", "G3": "#2980b9",
}
NATURE_COLORS = {
    "固い": "#a6e3a1", "中堅": "#89b4fa", "混戦": "#cba6f7",
    "穴推奨": "#fab387", "見送り": "#6c7086",
}
MARK_COLORS = {
    "◎": "#e74c3c", "〇": "#3498db", "▲": "#9b59b6", "△": "#f39c12",
}
MARKET_COLORS = {
    # 色弱対応で深め: 割安=緑 / 適正=青 / 割高=赤
    "under": "#37b24d", "fair": "#748ffc",
    "over": "#f03e3e", "unknown": "#788293",
}
# under/over/fair を「市場オッズに対して買い得か割高か」= 株の割安/割高に言い換え
MARKET_LABELS = {
    "under": "割安", "fair": "適正", "over": "割高", "unknown": "-",
}

# Advisor 用 grade/tag 配色 (Cowork 出力の grade/tag フィールド)
ADVISOR_GRADE_COLORS = {
    "SS": "#ffd700",  # gold
    "S":  "#fab387",  # amber
    "A":  "#a6e3a1",  # green
    "B":  "#89b4fa",  # blue
    "C":  "#6c7086",  # gray
}
ADVISOR_TAG_COLORS = {
    "軸":   "#5865f2",  # indigo
    "妙味": "#a6e3a1",  # green
    "穴":   "#cba6f7",  # purple
    "罠":   "#f38ba8",  # red
    "消":   "#45475a",  # dark gray
}


def race_nature(rc: dict) -> str:
    top1 = rc.get("top1_dominance") or 0
    chaos = _to_pct(rc.get("field_chaos_score") or 0)  # パーセンタイル(0-1)
    if chaos >= 0.85:
        return "見送り"
    if top1 >= 0.10 and chaos < 0.30:
        return "固い"
    if top1 < 0.05 and chaos >= 0.70:
        return "混戦"
    return "中堅"


def ai_comment(race: dict) -> str:
    rc = race.get("race_confidence", {}) or {}
    horses = race.get("horses", []) or []
    nature = race_nature(rc)
    top1 = _to_pct(rc.get("top1_dominance") or 0, "top1_dominance")
    top2 = _to_pct(rc.get("top2_concentration") or 0, "top2_concentration")
    chaos = _to_pct(rc.get("field_chaos_score") or 0)
    market = rc.get("ai_market_agreement") or 0
    hon = next((h for h in horses if h.get("mark") == "◎"), None)
    parts = []

    if hon:
        p_win = (hon.get("p_win") or 0) * 100
        if top1 >= 0.75:
            parts.append(f"本命 {hon.get('horse_name','◎')} は他馬と明確な確率差を持つ独走候補 (勝率 {p_win:.1f}%)")
        elif top1 >= 0.50:
            parts.append(f"本命 {hon.get('horse_name','◎')} (勝率 {p_win:.1f}%) は対抗との力差はあるが油断できない")
        else:
            parts.append(f"本命 {hon.get('horse_name','◎')} と対抗の力差は紙一重 (独走度は過去比 下位)")

    if top2 >= 0.75:
        parts.append("上位 2 頭で決まる確率が高く、馬連狙いの好レース")
    elif top2 <= 0.25:
        parts.append("上位馬の確率も分散しており、本命単独の安定には欠ける")

    if chaos >= 0.75:
        parts.append("過去比で上位の荒れレース、box やワイドで広めに / 見送りも検討")
    elif chaos >= 0.50:
        parts.append("やや混戦寄り、box やワイドが有効")
    elif chaos <= 0.25:
        parts.append("過去比で堅い部類、上位馬の優位が明確で堅い決着が予想される")

    if market > 0.7:
        parts.append("AI 予想と市場オッズ順がほぼ一致、配当妙味は控えめ")
    elif market < 0:
        parts.append("AI と市場の見方が大きく食い違い、波乱の可能性大")
    elif market < 0.3:
        parts.append("AI と市場の評価にズレがあり、AI が穴推しの可能性")

    if not parts:
        parts.append("特筆すべき特徴のない standard なレース")
    return f"性質判定: <b style='color:{NATURE_COLORS.get(nature,'#89b4fa')}'>{nature}</b>。" + "。".join(parts) + "。"


# ============================================================
# 左パネル: バナー + AI 評価 + コース概況
# ============================================================
def make_left_panel_html(race: dict) -> str:
    meta = race.get("race_meta", {}) or {}
    horses = race.get("horses", []) or []
    rc = race.get("race_confidence", {}) or {}
    nature = race_nature(rc)
    cls = meta.get("class", "")
    grade_color = GRADE_COLORS.get(cls, "#27ae60")
    place = meta.get("place", "")
    course = meta.get("course", "")
    field_size = meta.get("field_size", 0)
    race_name = meta.get("race_name") or cls or ""
    rid = race.get("race_id", "")
    r_num = rid[-2:].lstrip("0") if len(rid) >= 16 else "?"
    nat_color = NATURE_COLORS.get(nature, "#89b4fa")

    hon = next((h for h in horses if h.get("mark") == "◎"), None)
    tai = next((h for h in horses if h.get("mark") == "〇"), None)
    san = next((h for h in horses if h.get("mark") == "▲"), None)

    def mark_row(mark, color, h):
        if not h:
            return ""
        score = (h.get("p_win") or 0) * 100
        odds = h.get("tansho_odds")
        odds_str = f"{odds:.1f}倍" if odds else "-"
        return f"""
        <div style="display:flex;align-items:center;gap:12px;padding:6px 0">
          <span style="background:{color};color:#fff;font-size:18px;font-weight:bold;
                       width:34px;height:34px;line-height:34px;text-align:center;
                       border-radius:50%;flex-shrink:0">{mark}</span>
          <span style="color:#888;font-size:14px;width:36px">{h.get("umaban","?")}番</span>
          <span style="color:#cdd6f4;font-size:17px;font-weight:600;flex-grow:1">
            {h.get("horse_name","-")}</span>
          <span style="color:#a6e3a1;font-size:15px;font-weight:bold;
                       background:rgba(166,227,161,0.12);padding:3px 12px;
                       border-radius:12px">{score:.1f}%</span>
          <span style="color:#fab387;font-size:14px;background:rgba(250,179,135,0.12);
                       padding:3px 12px;border-radius:10px">単勝 {odds_str}</span>
        </div>
        """

    # コース概況統計 (オッズ系)
    odds_list = [h.get("tansho_odds") for h in horses if h.get("tansho_odds")]
    avg_odds = sum(odds_list) / len(odds_list) if odds_list else 0
    min_odds = min(odds_list) if odds_list else 0
    max_odds = max(odds_list) if odds_list else 0
    top3_p = sum(((horses[i].get("p_win") or 0) for i in range(min(3, len(horses)))))
    top3_p = top3_p * 100

    return f"""
    <div style="
      background:linear-gradient(135deg,#0d1421 0%,#16213e 50%,#1a2845 100%);
      border:1px solid #f39c12;border-radius:14px;
      padding:18px 22px;position:relative;overflow:hidden;
      box-shadow:0 4px 20px rgba(243,156,18,0.15);height:100%">
      <div style="position:absolute;top:0;left:0;right:0;height:4px;
                  background:linear-gradient(90deg,#e74c3c,#f39c12,#e74c3c)"></div>

      <!-- メタ情報 -->
      <div style="display:flex;flex-wrap:wrap;align-items:center;gap:10px;
                  margin-bottom:10px;font-size:15px">
        <span style="background:{grade_color};color:#fff;padding:4px 12px;
                     border-radius:4px;font-size:14px;font-weight:bold">{cls}</span>
        <span style="background:{nat_color};color:#1e1e2e;padding:4px 12px;
                     border-radius:4px;font-size:14px;font-weight:bold">{nature}</span>
        <span style="color:#cdd6f4;font-weight:600">{place} {r_num}R</span>
        <span style="color:#888">|</span>
        <span style="color:#f5c2e7">{course}</span>
        <span style="color:#888">|</span>
        <span style="color:#f5c2e7">{field_size}頭</span>
      </div>

      <!-- レース名 -->
      <h2 style="font-size:30px;font-weight:900;color:#cdd6f4;margin:0 0 12px 0;
                 letter-spacing:1.5px;line-height:1.1">{race_name}</h2>

      <!-- 印 (◎/〇/▲) -->
      <div style="background:rgba(0,0,0,0.25);border-left:3px solid #f39c12;
                  padding:6px 16px;border-radius:6px">
        {mark_row("◎", "#e74c3c", hon)}
        {mark_row("〇", "#3498db", tai)}
        {mark_row("▲", "#9b59b6", san)}
      </div>

      <!-- AI 評価 (印の下に移動) -->
      <div style="background:rgba(137,180,250,0.05);border-left:3px solid #89b4fa;
                  padding:14px 18px;margin-top:14px;border-radius:8px">
        <div style="color:#89b4fa;font-size:14px;font-weight:bold;margin-bottom:8px">
          🤖 AI 評価
        </div>
        <div style="color:#cdd6f4;font-size:15px;line-height:1.7">
          {ai_comment(race)}
        </div>
      </div>

      <!-- コース概況 (オッズ統計) -->
      <div style="background:rgba(245,194,231,0.04);border-left:3px solid #f5c2e7;
                  padding:14px 18px;margin-top:10px;border-radius:8px">
        <div style="color:#f5c2e7;font-size:14px;font-weight:bold;margin-bottom:8px">
          🏁 オッズ概況
        </div>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;
                    color:#cdd6f4;font-size:14px">
          <div><span style="color:#6c7086">1番人気:</span>
               <b style="color:#f5c2e7">{min_odds:.1f}倍</b></div>
          <div><span style="color:#6c7086">最高:</span>
               <b style="color:#f5c2e7">{max_odds:.1f}倍</b></div>
          <div><span style="color:#6c7086">平均:</span>
               <b style="color:#f5c2e7">{avg_odds:.1f}倍</b></div>
          <div style="grid-column:1/-1">
            <span style="color:#6c7086">上位3頭合計勝率:</span>
            <b style="color:#a6e3a1;font-size:15px">{top3_p:.1f}%</b>
          </div>
        </div>
      </div>
    </div>
    """


# ============================================================
# 4 chip 診断 (素人向け: 数値 → 状態ラベル + アクション提案)
# ------------------------------------------------------------
# 色は「値が示す行動」に紐づける (値の絶対水準ではなく、何をすべきか):
#   go      = 緑   : 狙う方向 (本命軸・本線で勝負できる / 妙味あり)
#   caution = 黄   : 標準・注意 (基本路線で OK / 相手を広げる)
#   avoid   = グレー: 見送り寄り (妙味薄・様子見)
#   danger  = 赤   : 荒れ・危険 (信頼度低、広めか見送り)
# go/caution/avoid/danger の 4 カテゴリだが、見せ方は「緑/黄/赤かグレー」の 3 系統。
#
# 総合判定・各カードの閾値は【仮】。実データ(audit_marks / cowork_results)で調整可。
# ============================================================

# --- 行動カテゴリ → 色 ---
_CAT_COLOR = {
    "go":      "#37b24d",  # 緑 (濃いめ・色弱でも判別しやすい高彩度)
    "caution": "#f0a500",  # 黄→琥珀 (濃いめ。淡黄は背景に埋もれるため彩度UP)
    "avoid":   "#788293",  # グレー (濃いめ)
    "danger":  "#f03e3e",  # 赤 (濃いめ・高彩度)
}

# --- 総合判定の閾値 (仮。実データで調整) ---
TH_MARKET_HIGH = 0.8    # 市場一致がこれ以上 → 妙味薄 (見送り寄り)
TH_MARKET_LOW  = 0.3    # 市場一致がこれ未満 (マイナス含む) → 乖離 (妙味)
TH_TOP1_HIGH   = 0.60   # 独走度がこれ以上 → 本命が抜けている
TH_TOP2_LOW    = 0.40   # 上位2頭集中がこれ未満 → 分散 (広く流す)
TH_CHAOS_HIGH  = 0.75   # 混戦度がこれ以上 → 荒れ (信頼度低)


def _overall_verdict(top1: float, top2: float, chaos: float,
                     market: float) -> tuple[str, str, str]:
    """4 指標から 1 行の結論を出す。優先順位:
       市場一致 → 独走度 → 上位2頭集中 → 混戦度
       → (headline, category, detail)。閾値は上の TH_* (仮値)。"""
    # 1. 市場一致が高い → 妙味薄
    if market >= TH_MARKET_HIGH:
        return ("見送り寄り（妙味薄）", "avoid",
                f"AI ≒ 市場 (一致 {market:+.2f})。期待値の上澄みが薄く、無理に買う旨味が小さい。")
    # 2. 市場と乖離 かつ 独走度が高い → 本命軸で妙味勝負
    if market < TH_MARKET_LOW and top1 >= TH_TOP1_HIGH:
        return ("本命軸で妙味勝負", "go",
                f"市場と乖離 (一致 {market:+.2f}) しつつ本命が抜けている (独走 {top1:.2f})。◎軸で攻めたい。")
    # 3. 上位2頭集中が低い → 広く流す / box
    if top2 < TH_TOP2_LOW:
        return ("広く流す／box", "caution",
                f"上位2頭が分散 (集中 {top2:.2f})。相手を絞らず box・流しで面を取る。")
    # 4. 混戦度が高い → 信頼度低、広め or 見送り
    if chaos >= TH_CHAOS_HIGH:
        return ("信頼度低・広め or 見送り", "danger",
                f"混戦度が高い (荒れ {chaos:.2f})。広く取るか、妙味が無ければ見送り。")
    # 該当なし → 標準
    return ("標準（印通りで OK）", "caution",
            "突出した警戒シグナルなし。基本路線 (印通り) で組み立てて OK。")


def _diag_top1(v: float) -> tuple[str, str, str]:
    """◎独走度(0-1: 過去レース中で本命がどれだけ抜けているか) → (badge, category, action)
       高い=本命が抜けている=狙える方向。"""
    if v >= 0.75:
        return ("◎独走", "go", "本命を信頼。単勝・複勝で勝負可")
    if v >= 0.50:
        return ("◎やや優位", "caution", "本命有力だが油断不可。複勝で安全に")
    if v >= 0.25:
        return ("拮抗", "caution", "本命の優位性は薄い。馬連で広めに")
    return ("混戦本命", "avoid", "本命の抜けは下位。馬連・ワイドで広めに")


def _diag_top2(v: float) -> tuple[str, str, str]:
    """上位2頭集中(0-1) → (badge, category, action)
       高い=本線濃厚=狙える方向。低い=分散=相手を広げる(注意)。"""
    if v >= 0.75:
        return ("本線濃厚", "go", "◎-〇 で決まる確率高。馬連本線が有効")
    if v >= 0.50:
        return ("やや本線", "caution", "馬連 ◎-〇 + 流し相手数頭が無難")
    return ("分散", "caution", "上位2頭以外も来る。流しか box で広めに")


def _diag_chaos(v: float) -> tuple[str, str, str]:
    """混戦度(0-1) → (badge, category, action)
       低い=固い=狙える方向。高い=荒れ=危険。"""
    if v <= 0.25:
        return ("固い", "go", "上位馬の優位明確。素直に印通りで")
    if v <= 0.50:
        return ("やや固め", "caution", "標準的な信頼度。基本路線で OK")
    if v <= 0.75:
        return ("混戦", "caution", "力差小さい。box やワイドで広めに")
    return ("カオス", "danger", "上位25%の荒れレース。見送り推奨")


def _diag_market(v: float) -> tuple[str, str, str]:
    """市場一致(-1..+1) → (badge, category, action)
       高い=AIと市場が一致=妙味薄(見送り寄り)。低い/負=乖離=妙味(狙える)。"""
    if v >= TH_MARKET_HIGH:
        return ("市場と一致", "avoid", "AI ≒ 市場。妙味は薄い、見送りも視野")
    if v >= TH_MARKET_LOW:
        return ("やや一致", "caution", "AI と市場ほぼ同方向。素直に買えるが妙味は中")
    if v >= 0:
        return ("ややズレ", "go", "AI が穴推し気味。妙味あり、慎重に")
    return ("逆方向", "go", "AI と市場が逆。波乱・穴の妙味、リスクは高め")


def _gauge_html(frac: float, color: str, center: bool = False) -> str:
    """値のレンジ上の位置を示す横ゲージ。
       frac=0-1 (レンジ内での位置)。center=True で中央(0.5)に基準線を引く
       (市場一致のように 0 が意味を持つ指標用)。"""
    frac = max(0.0, min(1.0, frac))
    pct = frac * 100
    center_tick = (
        '<div style="position:absolute;top:-1px;bottom:-1px;left:50%;width:1px;'
        'background:#585b70"></div>' if center else ""
    )
    return f"""
    <div style="position:relative;height:6px;background:#313244;border-radius:3px;
                margin-top:8px;overflow:visible">
      {center_tick}
      <div style="position:absolute;top:0;left:0;height:100%;width:{pct:.1f}%;
                  background:{color};border-radius:3px"></div>
      <div style="position:absolute;top:-2px;left:calc({pct:.1f}% - 4px);width:8px;
                  height:10px;background:{color};border-radius:2px;
                  box-shadow:0 0 0 1px #11111b"></div>
    </div>
    """


def _chip_html(title: str, value_str: str, diag: tuple[str, str, str],
               gauge_frac: float | None = None, gauge_center: bool = False) -> str:
    label, category, action = diag
    color = _CAT_COLOR.get(category, "#9399b2")
    gauge = (_gauge_html(gauge_frac, color, gauge_center)
             if gauge_frac is not None else "")
    return f"""
    <div style="background:#1e1e2e;border:1px solid #313244;border-left:4px solid {color};
                border-radius:8px;padding:12px 16px">
      <div style="display:flex;justify-content:space-between;align-items:center;
                  margin-bottom:6px">
        <span style="color:#6c7086;font-size:13px">{title}</span>
        <span style="background:{color};color:#1e1e2e;padding:2px 10px;
                     border-radius:10px;font-size:12px;font-weight:bold">{label}</span>
      </div>
      <div style="color:#cdd6f4;font-size:28px;font-weight:bold;line-height:1.0;
                  margin-bottom:2px">{value_str}</div>
      {gauge}
      <div style="color:#a6adc8;font-size:14.4px;line-height:1.4;margin-top:6px">
        → {action}
      </div>
    </div>
    """


def _verdict_html(verdict: tuple[str, str, str]) -> str:
    """4 指標の総合判定バッジ (4 カードの上に置く 1 行の結論)。"""
    headline, category, detail = verdict
    color = _CAT_COLOR.get(category, "#9399b2")
    return f"""
    <div style="background:linear-gradient(90deg,{color}22,#1e1e2e 60%);
                border:1px solid #313244;border-left:5px solid {color};
                border-radius:10px;padding:12px 16px">
      <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap">
        <span style="color:#6c7086;font-size:11px;letter-spacing:1px">総合判定</span>
        <span style="background:{color};color:#11111b;padding:4px 14px;
                     border-radius:14px;font-size:16px;font-weight:bold">{headline}</span>
      </div>
      <div style="color:#bac2de;font-size:12px;line-height:1.5;margin-top:7px">
        {detail}
      </div>
    </div>
    """


def _buy_judgment_html(race: dict) -> str:
    """買い方判定 (堅さ × 妙味馬の有無 → 買い方) バッジ + 妙味馬リスト。
    総合判定 4 指標パネルの「下」に置く。bundle に buy_judgment があれば読み、
    無ければ betting_judgment で計算 (旧 bundle 互換)。"""
    j = race.get("buy_judgment")
    if not j:
        j = bj.build_judgment(race.get("race_confidence"), race.get("horses"))

    color = _CAT_COLOR.get(j.get("category"), "#9399b2")
    headline = j.get("headline", "-")
    detail = j.get("detail", "")
    hint = j.get("kenshu_hint", "")
    waku = j.get("waku_tag")
    hardness = j.get("hardness", "")
    waku_pill = (
        f'<span style="background:#313244;color:#bac2de;padding:2px 8px;'
        f'border-radius:8px;font-size:10px;margin-left:6px">{waku}</span>'
        if waku else ""
    )

    # 妙味馬リスト
    vhs = j.get("value_horses") or []
    if vhs:
        rows = []
        for v in vhs:
            sides = v.get("sides") or []
            side_pills = ""
            if v.get("tan_value"):
                side_pills += (
                    f'<span style="background:{_CAT_COLOR["go"]};color:#11111b;'
                    f'padding:1px 7px;border-radius:8px;font-size:11px;font-weight:bold;'
                    f'margin-right:4px">単勝割安 EV{v.get("ev_tan",0):.2f}</span>'
                )
            if v.get("fuku_value"):
                side_pills += (
                    f'<span style="background:{_CAT_COLOR["go"]};color:#11111b;'
                    f'padding:1px 7px;border-radius:8px;font-size:11px;font-weight:bold;'
                    f'margin-right:4px">複勝割安 EV{v.get("ev_fuku",0):.2f}</span>'
                )
            rows.append(f"""
            <div style="display:flex;align-items:center;gap:8px;padding:5px 0;
                        border-bottom:1px solid #313244">
              <span style="color:#cdd6f4;font-weight:bold;min-width:26px;
                           text-align:center">{v.get("umaban","?")}</span>
              <span style="color:#cdd6f4;font-size:14px;min-width:120px">{v.get("horse_name","-")}</span>
              <span style="flex:1">{side_pills}</span>
              <span style="color:#a6e3a1;font-size:12px">勝率 {(v.get("p_win",0)*100):.1f}%</span>
            </div>
            """)
        horses_block = "".join(rows)
    else:
        horses_block = (
            '<div style="color:#9399b2;font-size:13px;padding:6px 2px">'
            '該当馬なし（妙味馬の条件を満たす馬はいません）</div>'
        )

    return f"""
    <div style="background:linear-gradient(90deg,{color}1c,#1e1e2e 60%);
                border:1px solid #313244;border-left:5px solid {color};
                border-radius:10px;padding:12px 16px">
      <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap">
        <span style="color:#6c7086;font-size:11px;letter-spacing:1px">買い方判定</span>
        <span style="background:{color};color:#11111b;padding:4px 14px;
                     border-radius:14px;font-size:16px;font-weight:bold">{headline}</span>
        <span style="color:#6c7086;font-size:11px">堅さ: <b style="color:#bac2de">{hardness}</b></span>
        {waku_pill}
      </div>
      <div style="color:#bac2de;font-size:12px;line-height:1.5;margin-top:7px">
        {detail}　<span style="color:#6c7086">／ 券種ヒント: {hint}</span>
      </div>
      <div style="margin-top:10px">
        <div style="color:#6c7086;font-size:11px;margin-bottom:4px">妙味馬（買い候補）</div>
        {horses_block}
      </div>
      <div style="color:#585b70;font-size:10px;margin-top:8px;line-height:1.4">
        ※ 妙味馬リストの提示まで。最終取捨（券種/点数/金額）はコース分析タブ＋当日馬場で人間が判断。
      </div>
    </div>
    """


# ============================================================
# 右パネル: メトリクス chip (大きめ) + 過去成績 + 馬場バイアス
# ============================================================
def make_right_panel_html(race: dict) -> str:
    rc = race.get("race_confidence", {}) or {}
    meta = race.get("race_meta", {}) or {}

    top1 = _to_pct(rc.get("top1_dominance") or 0, "top1_dominance")
    top2 = _to_pct(rc.get("top2_concentration") or 0, "top2_concentration")
    chaos = _to_pct(rc.get("field_chaos_score") or 0)  # 生値→パーセンタイル(0-1)
    market = rc.get("ai_market_agreement") or 0

    # 過去成績
    place = meta.get("place", "")
    course_str = meta.get("course", "")
    cstats = compute_course_stats(place, course_str)

    if cstats:
        n_races = cstats["n_races"]
        n_starts = cstats["n_starts"]
        waku = cstats.get("waku", {})

        # 枠別表示
        waku_html = ""
        if waku:
            for band, color in [("内枠", "#a6e3a1"), ("中枠", "#89b4fa"),
                                  ("外枠", "#fab387")]:
                if band in waku:
                    w = waku[band]
                    waku_html += f"""
                    <div style="display:flex;justify-content:space-between;
                                padding:4px 0;font-size:14px">
                      <span style="color:#a6adc8">{band}</span>
                      <span><b style="color:{color}">{w['win_rate']:.1f}%</b>
                            <span style="color:#6c7086;font-size:12px;margin-left:6px">
                              (top3 {w['top3_rate']:.1f}%)</span></span>
                    </div>
                    """

        # 枠バイアス判定
        if waku and len(waku) >= 3:
            best_band = max(waku.keys(), key=lambda k: waku[k]["win_rate"])
            best_rate = waku[best_band]["win_rate"]
            bias_msg = f"<b style='color:#fab387'>{best_band}有利</b> (勝率 {best_rate:.1f}%)"
        else:
            bias_msg = "<span style='color:#888'>データ不足</span>"

        kako_html = f"""
        <div style="background:#1e1e2e;border-left:3px solid #fab387;
                    padding:14px 16px;border-radius:8px">
          <div style="color:#fab387;font-size:14px;font-weight:bold;margin-bottom:8px">
            📊 過去同コース成績
          </div>
          <div style="color:#a6adc8;font-size:13px;margin-bottom:6px">
            {place} {course_str}: {n_races:,}R / {n_starts:,} 出走 (2013-2025)
          </div>
          <div style="color:#cdd6f4;font-size:13px;margin-top:6px">
            <b>1着馬の平均馬番:</b>
            <span style="color:#f5c2e7;font-size:15px;font-weight:bold">
              {cstats.get('first_uma_avg', 0):.1f}番</span>
          </div>
        </div>
        """

        baba_html = f"""
        <div style="background:#1e1e2e;border-left:3px solid #cba6f7;
                    padding:14px 16px;border-radius:8px">
          <div style="color:#cba6f7;font-size:14px;font-weight:bold;margin-bottom:10px">
            🎯 枠バイアス
          </div>
          {waku_html}
          <div style="color:#cdd6f4;font-size:13px;margin-top:10px;
                      padding-top:8px;border-top:1px solid #313244">
            判定: {bias_msg}
          </div>
        </div>
        """
    else:
        kako_html = """
        <div style="background:#1e1e2e;border-left:3px solid #6c7086;
                    padding:14px 16px;border-radius:8px;color:#6c7086;font-size:13px">
          📊 過去成績データ無し<br>
          (master_v2 が無いか、サンプル数 100 未満)
        </div>
        """
        baba_html = ""

    # 総合判定 (4 指標 → 1 行の結論。優先順位: 市場一致→独走度→上位2頭集中→混戦度)
    verdict_html = _verdict_html(_overall_verdict(top1, top2, chaos, market))
    # 買い方判定 (堅さ × 妙味馬 → 買い方 + 妙味馬リスト。4指標パネルの下に置く)
    buy_judgment_html = _buy_judgment_html(race)

    # カードは重要度順 (go/no-go の市場一致を先頭に)。各カードに横ゲージ付き。
    # ゲージの frac はレンジ内の位置: top1/top2/chaos は 0-1、market は -1..+1。
    chip_market = _chip_html("市場一致",     f"{market:+.3f}", _diag_market(market),
                             gauge_frac=(market + 1) / 2, gauge_center=True)
    chip_top1   = _chip_html("◎独走度",     f"{top1:.3f}",    _diag_top1(top1),
                             gauge_frac=top1)
    chip_top2   = _chip_html("上位2頭集中", f"{top2:.3f}",    _diag_top2(top2),
                             gauge_frac=top2)
    chip_chaos  = _chip_html("混戦度",       f"{chaos:.3f}",   _diag_chaos(chaos),
                             gauge_frac=chaos)

    return f"""
    <div style="display:flex;flex-direction:column;gap:10px;height:100%">
      {verdict_html}
      <div style="color:#6c7086;font-size:11px;margin-bottom:-4px;
                  padding:0 4px;letter-spacing:0.5px">
        ※ <b style="color:{_CAT_COLOR['go']}">緑=狙う方向</b> /
        <b style="color:{_CAT_COLOR['caution']}">黄=標準・注意</b> /
        <b style="color:{_CAT_COLOR['danger']}">赤</b>・<b style="color:{_CAT_COLOR['avoid']}">グレー</b>=見送り寄り。横ゲージは値のレンジ上の位置。
      </div>
      <!-- 信頼度メトリクス 4 chip (重要度順: 市場一致→独走度→上位2頭集中→混戦度) -->
      <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:10px">
        {chip_market}
        {chip_top1}
        {chip_top2}
        {chip_chaos}
      </div>

      <!-- 買い方判定 (4指標パネルの下: 堅さ×妙味→買い方 + 妙味馬リスト) -->
      {buy_judgment_html}

      {kako_html}
      {baba_html}
    </div>
    """


# ============================================================
# 出走表 (ピュア HTML テーブル)
# ============================================================
def _vs_market_status(model_p: float | None, odds: float | None) -> str:
    """モデル確率 vs 市場オッズ(implied=1/odds) を ±20% バンドで判定。
    単勝も複勝も同じロジック (複勝は odds=複勝オッズ中央値)。
    → 'under'(割安/妙味) / 'over'(割高) / 'fair'(適正) / 'unknown'。"""
    if not odds or odds <= 0 or not model_p:
        return "unknown"
    market_p = 1.0 / odds
    if model_p >= market_p * 1.20:
        return "under"   # AI > 市場 = 過小評価 = 割安(妙味)
    if model_p <= market_p * 0.80:
        return "over"    # AI < 市場 = 過大評価 = 割高
    return "fair"


def _market_badge_html(status: str) -> str:
    """割安/適正/割高 のピル型バッジ。"""
    color = MARKET_COLORS.get(status, "#788293")
    label = MARKET_LABELS.get(status, "-")
    return (f'<span style="background:{color};color:#11111b;padding:2px 10px;'
            f'border-radius:10px;font-size:12px;font-weight:bold">{label}</span>')


def make_shutsuba_table_html(race: dict, date_str: str | None = None) -> str:
    horses = race.get("horses", [])
    horses_sorted = sorted(horses, key=lambda h: h.get("umaban") or 0)

    # 騎手列用に weekly CSV を取得 (race_id_16 + 馬番 で参照)
    rid_16 = str(race.get("race_id", ""))[:16]
    weekly_df = load_weekly_horses(date_str) if date_str else None
    jockey_lookup: dict[int, str] = {}
    if weekly_df is not None:
        sub = weekly_df[weekly_df["race_id_16"] == rid_16]
        for _, row in sub.iterrows():
            try:
                u = int(row["馬番"])
                jockey_lookup[u] = str(row.get("騎手", "")).strip()
            except (TypeError, ValueError):
                continue

    rows = []
    for h in horses_sorted:
        mark = h.get("mark") or ""
        umaban = h.get("umaban") or "?"
        name = h.get("horse_name") or "-"
        sex = h.get("sex") or ""
        age = h.get("age")
        # 性別年齢を 1 列に詰める: "牡4" / "牝3" / セ表記もあり
        sex_age = f"{sex}{age}" if (sex and age is not None) else ""
        try:
            jockey = jockey_lookup.get(int(umaban), "")
        except (TypeError, ValueError):
            jockey = ""
        p_win = (h.get("p_win") or 0) * 100
        p_sho = (h.get("p_sho") or 0) * 100
        tan = h.get("tansho_odds") or 0
        ev_tan = (h.get("p_win") or 0) * tan
        fuku_low = h.get("fuku_odds_low") or 0
        fuku_high = h.get("fuku_odds_high") or 0
        # 複勝EV = 複勝率(p_sho) × 複勝オッズ。オッズはレンジなので中央値で 1 値化
        # (単勝EV = p_win × 単勝odds と同じ考え方)
        fuku_mid = (fuku_low + fuku_high) / 2
        ev_fuku = (h.get("p_sho") or 0) * fuku_mid
        # 単勝妙味: bundle 既存の ai_vs_market (p_win vs 1/単勝odds)
        market_tan = h.get("ai_vs_market") or "unknown"
        # 複勝妙味: 同じ ±20% ロジックを p_sho vs 1/複勝odds(中央値) で算出
        market_fuku = _vs_market_status(h.get("p_sho"),
                                        fuku_mid if fuku_mid > 0 else None)

        mark_color = MARK_COLORS.get(mark, "#6c7086")
        mark_html = (
            f'<span style="background:{mark_color};color:#fff;font-size:14px;'
            f'font-weight:bold;width:26px;height:26px;line-height:26px;'
            f'text-align:center;border-radius:50%;display:inline-block">{mark}</span>'
        ) if mark else ""

        bg = ""
        if mark == "◎":
            bg = "background:rgba(231,76,60,0.06);"
        elif mark == "〇":
            bg = "background:rgba(52,152,219,0.06);"
        elif mark == "▲":
            bg = "background:rgba(155,89,182,0.06);"

        ev_color = "#f9e2af" if ev_tan >= 1.0 else "#a6adc8"
        ev_fuku_color = "#f9e2af" if ev_fuku >= 1.0 else "#a6adc8"
        ev_fuku_str = f"{ev_fuku:.2f}" if fuku_mid > 0 else "-"

        rows.append(f"""
        <tr style="{bg}border-bottom:1px solid #313244;height:42px">
          <td style="padding:6px 12px;color:#cdd6f4;font-weight:bold;text-align:center">{umaban}</td>
          <td style="padding:6px 8px;text-align:center">{mark_html}</td>
          <td style="padding:6px 12px;color:#cdd6f4;font-size:16px;font-weight:600">{name}</td>
          <td style="padding:6px 8px;color:#cba6f7;font-size:14px;font-weight:600;text-align:center">{sex_age}</td>
          <td style="padding:6px 10px;color:#a6adc8;font-size:13px">{jockey}</td>
          <td style="padding:6px 12px;text-align:right;color:#a6e3a1;font-weight:bold">{p_win:.1f}%</td>
          <td style="padding:6px 12px;text-align:right;color:#89b4fa">{p_sho:.1f}%</td>
          <td style="padding:6px 12px;text-align:right;color:#f5c2e7">{tan:.1f}</td>
          <td style="padding:6px 12px;text-align:right;color:{ev_color};font-weight:bold">{ev_tan:.2f}</td>
          <td style="padding:6px 12px;text-align:center">{_market_badge_html(market_tan)}</td>
          <td style="padding:6px 12px;text-align:right;color:#a6adc8">{fuku_low:.1f}-{fuku_high:.1f}</td>
          <td style="padding:6px 12px;text-align:right;color:{ev_fuku_color};font-weight:bold">{ev_fuku_str}</td>
          <td style="padding:6px 12px;text-align:center">{_market_badge_html(market_fuku)}</td>
        </tr>
        """)

    return f"""
    <table style="width:100%;border-collapse:collapse;background:#0a0a14;
                  border-radius:8px;overflow:hidden">
      <thead>
        <tr style="background:#1e1e2e;border-bottom:2px solid #f39c12">
          <th style="padding:12px;color:#f39c12;font-size:14px;text-align:center">番</th>
          <th style="padding:12px;color:#f39c12;font-size:14px;text-align:center">印</th>
          <th style="padding:12px;color:#f39c12;font-size:14px;text-align:left">馬名</th>
          <th style="padding:12px;color:#f39c12;font-size:14px;text-align:center">性齢</th>
          <th style="padding:12px;color:#f39c12;font-size:14px;text-align:left">騎手</th>
          <th style="padding:12px;color:#f39c12;font-size:14px;text-align:right">勝率</th>
          <th style="padding:12px;color:#f39c12;font-size:14px;text-align:right">複勝率</th>
          <th style="padding:12px;color:#f39c12;font-size:14px;text-align:right">単勝</th>
          <th style="padding:12px;color:#f39c12;font-size:14px;text-align:right">単勝EV</th>
          <th style="padding:12px;color:#f39c12;font-size:14px;text-align:center">単勝妙味</th>
          <th style="padding:12px;color:#f39c12;font-size:14px;text-align:right">複勝</th>
          <th style="padding:12px;color:#f39c12;font-size:14px;text-align:right">複勝EV</th>
          <th style="padding:12px;color:#f39c12;font-size:14px;text-align:center">複勝妙味</th>
        </tr>
      </thead>
      <tbody>
        {"".join(rows)}
      </tbody>
    </table>
    """


# ============================================================
# コース分析タブ (条件別成績 + 展開予想)
# ============================================================
def _stats_table_html(title: str, rows: list[dict],
                        emoji: str = "", min_width: str = "100%") -> str:
    """1 つの条件別成績テーブル (枠順/馬番/年齢/性別 共通)。
    画像のスタイルに合わせて 1着/2着/3着/着外 + 勝率/連対率/複勝率 を出す。
    """
    if not rows:
        return ""
    body = []
    for r in rows:
        body.append(f"""
        <tr style="border-bottom:1px solid #313244;height:36px">
          <td style="padding:10px 14px;color:#cdd6f4;font-weight:bold;font-size:16px">{r['label']}</td>
          <td style="padding:10px 14px;text-align:right;color:#a6e3a1;font-size:16px">{r['n_1']}</td>
          <td style="padding:10px 14px;text-align:right;color:#89b4fa;font-size:16px">{r['n_2']}</td>
          <td style="padding:10px 14px;text-align:right;color:#f9e2af;font-size:16px">{r['n_3']}</td>
          <td style="padding:10px 14px;text-align:right;color:#6c7086;font-size:16px">{r['n_out']}</td>
          <td style="padding:10px 14px;text-align:right;color:#a6e3a1;font-weight:bold;font-size:17px">
            {r['win_rate']:.1f}</td>
          <td style="padding:10px 14px;text-align:right;color:#89b4fa;font-size:17px">
            {r['rentai_rate']:.1f}</td>
          <td style="padding:10px 14px;text-align:right;color:#fab387;font-weight:bold;font-size:17px">
            {r['fuku_rate']:.1f}</td>
        </tr>
        """)
    return f"""
    <div style="margin-bottom:22px;min-width:{min_width}">
      <h3 style="color:#f5e0dc;margin:0 0 12px 0;font-size:20px;font-weight:bold">
        {emoji} {title}</h3>
      <table style="width:100%;border-collapse:collapse;background:#0a0a14;
                    border-radius:8px;overflow:hidden">
        <thead>
          <tr style="background:#1e1e2e;border-bottom:2px solid #f39c12">
            <th style="padding:12px 14px;color:#f39c12;text-align:left;font-size:15px">条件</th>
            <th style="padding:12px 14px;color:#f39c12;text-align:right;font-size:15px">1着</th>
            <th style="padding:12px 14px;color:#f39c12;text-align:right;font-size:15px">2着</th>
            <th style="padding:12px 14px;color:#f39c12;text-align:right;font-size:15px">3着</th>
            <th style="padding:12px 14px;color:#f39c12;text-align:right;font-size:15px">着外</th>
            <th style="padding:12px 14px;color:#f39c12;text-align:right;font-size:15px">勝率</th>
            <th style="padding:12px 14px;color:#f39c12;text-align:right;font-size:15px">連対率</th>
            <th style="padding:12px 14px;color:#f39c12;text-align:right;font-size:15px">複勝率</th>
          </tr>
        </thead>
        <tbody>{"".join(body)}</tbody>
      </table>
    </div>
    """


def _classify_kyakushitsu(pos1: float | None,
                            pos4: float | None,
                            field_size: int = 16) -> str:
    """前1角・前4角の通過位置から脚質を推定。
    field_size に対する相対位置で 逃げ/先行/差し/追込 を判定。
    """
    pos = pos4 if pos4 is not None else pos1
    if pos is None or pd.isna(pos):
        return "不明"
    p = float(pos) / max(field_size, 1)
    if p <= 0.20:
        return "逃げ"
    if p <= 0.45:
        return "先行"
    if p <= 0.70:
        return "差し"
    return "追込"


def make_horse_detail_html(horse: dict, race: dict,
                              date_str: str | None) -> str:
    """馬個別モーダルの中身を HTML 文字列で構築。
    出走表 / 評価リストの「詳細」ボタン押下時に呼ばれる。
    """
    name   = horse.get("horse_name", "?")
    umaban = horse.get("umaban", "?")
    mark   = horse.get("mark") or ""
    p_win  = (horse.get("p_win") or 0) * 100
    p_plc  = (horse.get("p_plc") or 0) * 100
    p_sho  = (horse.get("p_sho") or 0) * 100
    tan    = horse.get("tansho_odds") or 0
    fuku_lo = horse.get("fuku_odds_low") or 0
    fuku_hi = horse.get("fuku_odds_high") or 0
    ev_tan = (horse.get("p_win") or 0) * tan
    market = horse.get("ai_vs_market") or "unknown"
    mk_color = MARK_COLORS.get(mark, "#6c7086")
    mkt_color = MARKET_COLORS.get(market, "#6c7086")

    # weekly CSV から騎手・斤量・性齢
    jockey = sex_age = kin = ""
    if date_str:
        weekly_df = load_weekly_horses(date_str)
        if weekly_df is not None:
            rid_16 = str(race.get("race_id", ""))[:16]
            try:
                sub = weekly_df[(weekly_df["race_id_16"] == rid_16) &
                                  (weekly_df["馬番"] == int(umaban))]
                if not sub.empty:
                    row = sub.iloc[0]
                    jockey  = str(row.get("騎手", "")).strip()
                    kin     = str(row.get("斤量", "")).strip()
                    sex     = str(row.get("性別", "")).strip()
                    age     = str(row.get("年齢", "")).strip()
                    sex_age = f"{sex}{age}"
            except (TypeError, ValueError):
                pass

    # Header
    header_html = f"""
    <div style="background:linear-gradient(135deg,#0d1421 0%,#16213e 100%);
                border:1px solid {mk_color};border-radius:12px;
                padding:16px 20px;margin-bottom:14px">
      <div style="display:flex;align-items:center;gap:14px;margin-bottom:8px">
        <span style="background:{mk_color};color:#fff;width:42px;height:42px;
                     line-height:42px;text-align:center;border-radius:50%;
                     font-size:22px;font-weight:bold">{mark or '−'}</span>
        <span style="color:#a6adc8;font-size:15px">{umaban}番</span>
        <h2 style="margin:0;color:#cdd6f4;font-size:24px;font-weight:bold;flex:1">
          {name}</h2>
      </div>
      <div style="color:#a6adc8;font-size:14px;margin-bottom:10px">
        {sex_age} / 斤量 {kin}kg / 騎手 {jockey}
      </div>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;
                  font-size:14px">
        <div style="background:rgba(0,0,0,0.25);padding:8px 12px;border-radius:6px">
          <div style="color:#6c7086;font-size:12px">勝率 / 連対率 / 複勝率</div>
          <div style="color:#cdd6f4;font-size:15px;font-weight:bold">
            <span style="color:#a6e3a1">{p_win:.1f}%</span> /
            <span style="color:#89b4fa">{p_plc:.1f}%</span> /
            <span style="color:#fab387">{p_sho:.1f}%</span>
          </div>
        </div>
        <div style="background:rgba(0,0,0,0.25);padding:8px 12px;border-radius:6px">
          <div style="color:#6c7086;font-size:12px">単勝オッズ</div>
          <div style="color:#f5c2e7;font-size:18px;font-weight:bold">{tan:.1f}倍</div>
        </div>
        <div style="background:rgba(0,0,0,0.25);padding:8px 12px;border-radius:6px">
          <div style="color:#6c7086;font-size:12px">複勝オッズ</div>
          <div style="color:#cdd6f4;font-size:15px">{fuku_lo:.1f}〜{fuku_hi:.1f}倍</div>
        </div>
        <div style="background:rgba(0,0,0,0.25);padding:8px 12px;border-radius:6px">
          <div style="color:#6c7086;font-size:12px">単勝EV / vs市場</div>
          <div style="color:#cdd6f4;font-size:15px">
            <span style="color:{('#a6e3a1' if ev_tan >= 1.0 else '#f9e2af')};font-weight:bold">
              {ev_tan:.2f}</span> /
            <span style="background:{mkt_color};color:#1e1e2e;padding:1px 8px;
                         border-radius:8px;font-size:12px;font-weight:bold">{market}</span>
          </div>
        </div>
      </div>
    </div>
    """

    # 過去 5 走
    rid_16 = str(race.get("race_id", ""))[:16]
    kako5 = load_kako5_horses(date_str) if date_str else None
    past_races: list[dict] = []
    if kako5:
        try:
            entry = kako5.get((rid_16, int(umaban)))
            if entry:
                past_races = entry.get("past_races", [])
        except (TypeError, ValueError):
            past_races = []

    if past_races:
        rows = []
        for i, pr in enumerate(past_races, 1):
            chk = pr["chakujun"]
            chk_color = ("#a6e3a1" if chk == 1 else
                          "#89b4fa" if chk <= 3 else
                          "#cdd6f4" if chk <= 5 else "#6c7086")
            ninki = pr.get("ninki") or "-"
            agari = pr.get("agari3f")
            agari_s = f"{agari:.1f}" if agari is not None else "-"
            rows.append(f"""
            <tr style="border-bottom:1px solid #313244">
              <td style="padding:8px 10px;color:#6c7086">{i}走前</td>
              <td style="padding:8px 10px;color:#a6adc8">{pr['month']}/{pr['day']}</td>
              <td style="padding:8px 10px;color:#cdd6f4">{pr['place']}</td>
              <td style="padding:8px 10px;color:#cdd6f4">
                {pr['td']}{pr.get('dist') or ''}</td>
              <td style="padding:8px 10px;color:#a6adc8">{pr.get('baba','')}</td>
              <td style="padding:8px 10px;color:#cdd6f4;font-weight:600;
                         max-width:200px;overflow:hidden;text-overflow:ellipsis">
                {pr.get('race_name','')}</td>
              <td style="padding:8px 10px;text-align:right;color:{chk_color};
                         font-weight:bold;font-size:16px">{chk}着</td>
              <td style="padding:8px 10px;text-align:right;color:#fab387">{ninki}人気</td>
              <td style="padding:8px 10px;text-align:right;color:#cba6f7">{agari_s}</td>
              <td style="padding:8px 10px;color:#a6adc8;font-size:13px">
                {pr.get('kimete','')}</td>
            </tr>
            """)
        past_html = f"""
        <div style="margin-bottom:14px">
          <h3 style="color:#f5e0dc;font-size:18px;font-weight:bold;margin:0 0 8px 0">
            📜 過去 5 走
          </h3>
          <table style="width:100%;border-collapse:collapse;background:#0a0a14;
                        border-radius:8px;overflow:hidden">
            <thead>
              <tr style="background:#1e1e2e;border-bottom:2px solid #f39c12">
                <th style="padding:10px;color:#f39c12;text-align:left">時期</th>
                <th style="padding:10px;color:#f39c12;text-align:left">月/日</th>
                <th style="padding:10px;color:#f39c12;text-align:left">場所</th>
                <th style="padding:10px;color:#f39c12;text-align:left">コース</th>
                <th style="padding:10px;color:#f39c12;text-align:left">馬場</th>
                <th style="padding:10px;color:#f39c12;text-align:left">レース名</th>
                <th style="padding:10px;color:#f39c12;text-align:right">着順</th>
                <th style="padding:10px;color:#f39c12;text-align:right">人気</th>
                <th style="padding:10px;color:#f39c12;text-align:right">上り3F</th>
                <th style="padding:10px;color:#f39c12;text-align:left">決手</th>
              </tr>
            </thead>
            <tbody>{"".join(rows)}</tbody>
          </table>
        </div>
        """
    else:
        past_html = """
        <div style="background:#1e1e2e;border-left:3px solid #6c7086;
                    padding:12px 16px;border-radius:6px;color:#6c7086;
                    margin-bottom:14px">
          📜 過去 5 走データなし
          <span style="font-size:12px">(data/kako5/{date}.csv が無いか、馬未掲載)</span>
        </div>
        """

    # 同コース過去成績 (場所×芝/ダ×距離 の全体傾向、馬個別ではない)
    meta = race.get("race_meta", {}) or {}
    course_stats = compute_course_stats_v2(
        meta.get("place", ""), meta.get("course", ""))
    course_html = ""
    if course_stats and course_stats.get("kyaku"):
        ky_html = "".join(
            f'<span style="color:#cdd6f4;margin-right:14px">'
            f'<b>{k["label"]}</b>: 勝率 <b style="color:#a6e3a1">'
            f'{k["win_rate"]:.1f}%</b></span>'
            for k in course_stats["kyaku"]
        )
        course_html = f"""
        <div style="background:#1e1e2e;border-left:3px solid #cba6f7;
                    padding:12px 18px;border-radius:6px">
          <h3 style="color:#cba6f7;font-size:16px;font-weight:bold;margin:0 0 8px 0">
            🏃 {meta.get('place','')} {meta.get('course','')} の脚質別勝率
          </h3>
          <div>{ky_html}</div>
        </div>
        """

    return header_html + past_html + course_html


def render_cowork_pl_chart() -> None:
    """全 cowork_output × kekka からシーズン累計 P/L チャートを描画。
    上段: 4 chip サマリ (累計投資 / 累計収支 / 回収率 / 的中率)
    中段: 累計収支ライン (ECharts)
    下段: 馬券種別収支テーブル
    """
    rows = load_all_cowork_outcomes(_kekka_files_cache_key())
    if not rows:
        ui.html("""
        <div style="background:#1e1e2e;border-left:3px solid #6c7086;
                    padding:14px 18px;border-radius:8px;color:#6c7086;
                    font-size:14px">
          📈 Cowork 累計収支データなし<br>
          <span style="font-size:12px">
            reports/cowork_output/{date}_bets.json と
            data/kekka/{date}.csv の両方が揃った日付がまだありません。
          </span>
        </div>
        """)
        return

    df = pd.DataFrame(rows).sort_values("date")
    daily = df.groupby("date").agg(
        cost=("cost", "sum"),
        profit=("profit", "sum"),
        n_bets=("date", "size"),
        n_wins=("is_win", "sum"),
    ).reset_index()
    daily["cum_cost"] = daily["cost"].cumsum()
    daily["cum_profit"] = daily["profit"].cumsum()

    total_cost = float(daily["cost"].sum())
    total_profit = float(daily["profit"].sum())
    total_bets = int(df.shape[0])
    total_wins = int(df["is_win"].sum())
    hit_rate = total_wins / total_bets * 100.0 if total_bets > 0 else 0.0
    roi = ((total_cost + total_profit) / total_cost * 100.0
           if total_cost > 0 else 100.0)
    profit_color = "#a6e3a1" if total_profit >= 0 else "#f38ba8"

    ui.html(f"""
    <div style="background:linear-gradient(135deg,#0d1421 0%,#16213e 100%);
                border:1px solid {profit_color};border-radius:12px;
                padding:20px 24px;margin-bottom:14px">
      <h2 style="margin:0 0 14px 0;color:{profit_color};font-size:24px;
                 font-weight:bold">📈 Cowork 累計収支</h2>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:14px">
        <div style="background:rgba(0,0,0,0.3);padding:14px 18px;border-radius:8px">
          <div style="color:#6c7086;font-size:13px">累計投資</div>
          <div style="color:#cdd6f4;font-size:26px;font-weight:bold">
            ¥{total_cost:,.0f}</div>
        </div>
        <div style="background:rgba(0,0,0,0.3);padding:14px 18px;border-radius:8px">
          <div style="color:#6c7086;font-size:13px">累計収支</div>
          <div style="color:{profit_color};font-size:26px;font-weight:bold">
            ¥{total_profit:+,.0f}</div>
        </div>
        <div style="background:rgba(0,0,0,0.3);padding:14px 18px;border-radius:8px">
          <div style="color:#6c7086;font-size:13px">回収率 (ROI)</div>
          <div style="color:{profit_color};font-size:26px;font-weight:bold">
            {roi:.1f}%</div>
        </div>
        <div style="background:rgba(0,0,0,0.3);padding:14px 18px;border-radius:8px">
          <div style="color:#6c7086;font-size:13px">的中率</div>
          <div style="color:#cdd6f4;font-size:26px;font-weight:bold">
            {hit_rate:.1f}%
            <span style="font-size:14px;color:#6c7086">
              ({total_wins}/{total_bets})</span>
          </div>
        </div>
      </div>
    </div>
    """)

    # ECharts ライン: 累計収支推移
    dates_disp = [f"{d[4:6]}/{d[6:8]}" for d in daily["date"].tolist()]
    cum_profit = [round(x, 0) for x in daily["cum_profit"].tolist()]
    daily_profit = [round(x, 0) for x in daily["profit"].tolist()]
    line_color = "#a6e3a1" if total_profit >= 0 else "#f38ba8"

    chart_opt = {
        "title": {"text": "累計収支推移 (Cowork)",
                   "textStyle": {"color": "#cdd6f4", "fontSize": 16}},
        "tooltip": {"trigger": "axis",
                     "backgroundColor": "#1e1e2e",
                     "borderColor": "#313244",
                     "textStyle": {"color": "#cdd6f4"}},
        "legend": {"data": ["累計収支", "日別収支"],
                    "textStyle": {"color": "#a6adc8"}},
        "xAxis": {"type": "category", "data": dates_disp,
                   "axisLabel": {"color": "#a6adc8"},
                   "axisLine": {"lineStyle": {"color": "#313244"}}},
        "yAxis": {"type": "value",
                   "axisLabel": {"color": "#a6adc8", "formatter": "¥{value}"},
                   "splitLine": {"lineStyle": {"color": "#313244"}}},
        "series": [
            {"name": "累計収支", "type": "line", "data": cum_profit,
             "smooth": True,
             "lineStyle": {"color": line_color, "width": 3},
             "itemStyle": {"color": line_color},
             "areaStyle": {"opacity": 0.18, "color": line_color},
             "markLine": {
                 "data": [{"yAxis": 0,
                            "lineStyle": {"color": "#6c7086", "type": "dashed"}}],
                 "symbol": "none", "label": {"show": False}}},
            {"name": "日別収支", "type": "bar", "data": daily_profit,
             "itemStyle": {"color": "#89b4fa", "opacity": 0.6}},
        ],
        "grid": {"left": 80, "right": 30, "top": 60, "bottom": 50},
        "backgroundColor": "transparent",
    }
    ui.echart(chart_opt).classes("w-full").style("height: 360px")

    # 馬券種別収支テーブル
    by_b = df.groupby("btype").agg(
        n=("date", "size"),
        wins=("is_win", "sum"),
        cost=("cost", "sum"),
        profit=("profit", "sum"),
    ).reset_index().sort_values("cost", ascending=False)

    body_rows = []
    for _, r in by_b.iterrows():
        c = float(r["cost"])
        p = float(r["profit"])
        n = int(r["n"])
        w = int(r["wins"])
        roi_b = (c + p) / c * 100.0 if c > 0 else 100.0
        hit_b = w / n * 100.0 if n > 0 else 0.0
        pc = "#a6e3a1" if p >= 0 else "#f38ba8"
        body_rows.append(f"""
        <tr style="border-bottom:1px solid #313244">
          <td style="padding:10px 14px;color:#cdd6f4;font-weight:bold;font-size:16px">
            {r['btype']}</td>
          <td style="padding:10px 14px;text-align:right;color:#a6adc8">{n}</td>
          <td style="padding:10px 14px;text-align:right;color:#a6e3a1">{w}</td>
          <td style="padding:10px 14px;text-align:right;color:#cdd6f4">
            {hit_b:.1f}%</td>
          <td style="padding:10px 14px;text-align:right;color:#cdd6f4">
            ¥{c:,.0f}</td>
          <td style="padding:10px 14px;text-align:right;color:{pc};
                     font-weight:bold">¥{p:+,.0f}</td>
          <td style="padding:10px 14px;text-align:right;color:{pc};
                     font-weight:bold">{roi_b:.1f}%</td>
        </tr>
        """)
    ui.html(f"""
    <div style="margin-top:14px">
      <h3 style="color:#f5e0dc;font-size:18px;margin:0 0 8px 0;font-weight:bold">
        📊 馬券種別収支
      </h3>
      <table style="width:100%;border-collapse:collapse;background:#0a0a14;
                    border-radius:8px;overflow:hidden">
        <thead>
          <tr style="background:#1e1e2e;border-bottom:2px solid #f39c12">
            <th style="padding:12px 14px;color:#f39c12;text-align:left;font-size:14px">
              馬券種</th>
            <th style="padding:12px 14px;color:#f39c12;text-align:right;font-size:14px">
              買い目数</th>
            <th style="padding:12px 14px;color:#f39c12;text-align:right;font-size:14px">
              的中</th>
            <th style="padding:12px 14px;color:#f39c12;text-align:right;font-size:14px">
              的中率</th>
            <th style="padding:12px 14px;color:#f39c12;text-align:right;font-size:14px">
              投資</th>
            <th style="padding:12px 14px;color:#f39c12;text-align:right;font-size:14px">
              収支</th>
            <th style="padding:12px 14px;color:#f39c12;text-align:right;font-size:14px">
              回収率</th>
          </tr>
        </thead>
        <tbody>{"".join(body_rows)}</tbody>
      </table>
    </div>
    """)


def render_grade_scope(date_str: str | None) -> None:
    """指定日付の重賞詳細見解 (Grade Scope) を pulldown + markdown で表示。

    Cowork 出力の wrapper 形式 ({"bets":..., "grade_scope":...}) から
    G1/G2/G3 の markdown 見解を抽出。該当週に重賞が無ければ案内表示のみ。
    """
    if not date_str:
        ui.html('<div style="color:#6c7086;padding:12px">日付未選択</div>')
        return

    scopes = load_grade_scope_for_date(date_str)
    if not scopes:
        ui.html(f"""
        <div style="background:#1e1e2e;border-left:3px solid #6c7086;
                    padding:14px 18px;border-radius:8px;color:#6c7086;
                    font-size:13px">
          🏆 {date_str} の重賞詳細見解はまだありません<br>
          <span style="font-size:12px">
            該当週に G1/G2/G3 レースがあり、Cowork が
            <code>{{"grade_scope": [...]}}</code> 形式で出力済みのときに表示されます。
          </span>
        </div>
        """)
        return

    # ヘッダ: 何件あるか
    ui.html(f"""
    <div style="color:#cdd6f4;font-size:13px;margin-bottom:10px">
      {len(scopes)} 件の重賞詳細見解 ({date_str})
    </div>
    """)

    # レース選択 dropdown (race_label を表示)
    options = {s["race_id"]: s.get("race_label") or s["race_id"]
               for s in scopes}
    scope_map = {s["race_id"]: s for s in scopes}

    md_container = ui.element("div").classes("w-full") \
        .style("background:#11111b;border-radius:10px;"
               "padding:16px 22px;margin-top:8px;color:#cdd6f4;"
               "line-height:1.7;font-size:14px")

    def _render(rid: str) -> None:
        md_container.clear()
        s = scope_map.get(rid)
        if not s:
            return
        with md_container:
            # クラスバッジ + ラベル
            cls = s.get("class", "")
            label = s.get("race_label", rid)
            ui.html(f"""
            <div style="display:flex;align-items:center;gap:10px;
                        margin-bottom:14px;padding-bottom:10px;
                        border-bottom:1px solid #313244">
              <span style="background:#f5c2e7;color:#11111b;
                           padding:3px 10px;border-radius:6px;
                           font-weight:bold;font-size:12px">{cls}</span>
              <span style="color:#cdd6f4;font-size:15px;font-weight:bold">{label}</span>
            </div>
            """)
            # markdown 本文
            md = s.get("markdown", "")
            if md:
                ui.markdown(md)
            else:
                ui.html('<div style="color:#6c7086">本文がありません</div>')

    select_el = ui.select(
        options=options,
        value=scopes[0]["race_id"],
        on_change=lambda e: _render(e.value),
    ).classes("w-full").style("max-width:520px")

    # 初期描画
    _render(scopes[0]["race_id"])


def render_training_top5(date_str: str | None) -> None:
    """直近 1 週間の好調教 Top5 (坂路 + WC) と各馬の出走予定を表示。
    weekly_nicegui.ps1 の date を引いて呼ぶ。
    """
    if not date_str:
        return
    data = compute_training_top5(date_str)
    has_hanro = bool(data.get("hanro"))
    has_wc = bool(data.get("wc"))
    if not has_hanro and not has_wc:
        return  # 訓練 CSV 無し or 該当馬無し → 何も出さない

    def _item_html(item: dict, time_label: str, time_key: str,
                     time_unit: str = "秒") -> str:
        name = item.get("name", "?")
        date_str_ = str(item.get("date", ""))
        date_fmt = (f"{date_str_[4:6]}/{date_str_[6:8]}"
                      if len(date_str_) == 8 else "?")
        place = item.get("place") or ""
        if place.lower() == "nan":
            place = ""
        time_val = item.get(time_key)
        time_str = f"{time_val:.1f} {time_unit}" if time_val is not None else "-"
        race = item.get("race")
        if race:
            mark = race.get("mark") or ""
            mark_color = MARK_COLORS.get(mark, "#6c7086")
            mark_html = (f'<span style="background:{mark_color};color:#fff;'
                          f'padding:2px 7px;border-radius:10px;font-size:12px;'
                          f'font-weight:bold;margin-right:6px">{mark}</span>'
                          if mark else "")
            p_win = (race.get("p_win") or 0) * 100
            race_html = (
                f'{mark_html}'
                f'<span style="color:#a6e3a1;font-weight:bold">'
                f'{race["place"]} {race["race_num"]}R</span>'
                f'<span style="color:#a6adc8;font-size:13px;margin-left:6px">'
                f'({race.get("course","")})</span>'
                f'<span style="color:#fab387;font-size:13px;margin-left:6px">'
                f'AI勝率 {p_win:.1f}%</span>'
            )
        else:
            race_html = '<span style="color:#6c7086;font-size:13px">出走予定なし</span>'

        return f"""
        <div style="display:flex;align-items:center;gap:10px;padding:8px 12px;
                    border-bottom:1px solid #313244">
          <span style="color:#cdd6f4;font-size:16px;font-weight:bold;min-width:140px">
            {name}</span>
          <span style="color:#fab387;font-size:16px;font-weight:bold;min-width:110px;
                       text-align:right">{time_label}: {time_str}</span>
          <span style="color:#6c7086;font-size:13px;min-width:90px">
            {place} {date_fmt}</span>
          <span style="font-size:14px;flex:1">{race_html}</span>
        </div>
        """

    hanro_html = "".join(
        _item_html(it, "終い1F", "lap1") for it in data["hanro"]
    ) or '<div style="color:#6c7086;padding:10px">該当データなし</div>'
    wc_html = "".join(
        _item_html(it, "3F", "f3") for it in data["wc"]
    ) or '<div style="color:#6c7086;padding:10px">該当データなし</div>'

    ui.html(f"""
    <div style="background:#1e1e2e;border:1px solid #f39c12;border-radius:12px;
                padding:18px 22px;margin-bottom:14px">
      <h2 style="margin:0 0 12px 0;color:#f39c12;font-size:22px;font-weight:bold">
        ⚡ 直近 1 週間の好調教 Best 5
      </h2>
      <div style="color:#a6adc8;font-size:13px;margin-bottom:10px">
        race 日 ({date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}) の前 7 日間で
        馬ごとに最良タイムを採り、坂路は終い1F、WC は 3F が速い順。
        その馬が当週末に出走するなら場所/レース番号/印/AI 勝率を併記。
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">
        <div>
          <h3 style="color:#a6e3a1;font-size:18px;margin:0 0 6px 0;font-weight:bold">
            🏔️ 坂路 終い1F Best 5
          </h3>
          {hanro_html}
        </div>
        <div>
          <h3 style="color:#89b4fa;font-size:18px;margin:0 0 6px 0;font-weight:bold">
            🌲 WC 3F Best 5
          </h3>
          {wc_html}
        </div>
      </div>
    </div>
    """)


def render_tenkai_yoso(race: dict, date_str: str | None = None) -> None:
    """展開予想 — SPAIA 風の視覚的隊列図 (全頭表示)。

    Layout:
      [逃げ] 1 ナスノソナタ
        ↓ (約 N 馬身)
      [先行] 5 リールフォート ── 9 ハリーバローズ ── ...
        ↓
      [中団] 6 シャドウメテオ ── ...
        ↓
      [後方] 14 サンダーバード ── ...

      想定ペース: H / M / S + 解説
      有利な脚質
    """
    horses = list(race.get("horses", []))
    if not horses:
        return
    meta = race.get("race_meta", {}) or {}
    field_size = meta.get("field_size", len(horses)) or len(horses)

    # 全頭の脚質を分類
    weekly_df = load_weekly_horses(date_str) if date_str else None
    rid_16 = str(race.get("race_id", ""))[:16]

    kyaku_records: list[dict] = []
    for h in horses:
        umaban = h.get("umaban")
        pos1 = pos4 = None
        if weekly_df is not None and umaban is not None:
            sub = weekly_df[(weekly_df["race_id_16"] == rid_16) &
                              (weekly_df["馬番"] == int(umaban))]
            if not sub.empty:
                row = sub.iloc[0]
                for c in ["前1角", "前走通過1"]:
                    if c in row.index and pd.notna(row[c]):
                        try: pos1 = float(row[c])
                        except (TypeError, ValueError): pass
                        break
                for c in ["前4角", "前走通過4"]:
                    if c in row.index and pd.notna(row[c]):
                        try: pos4 = float(row[c])
                        except (TypeError, ValueError): pass
                        break
        kyaku = _classify_kyakushitsu(pos1, pos4, field_size)
        kyaku_records.append({"horse": h, "kyaku": kyaku, "pos4": pos4})

    # 脚質別 group: 逃げ / 先行 / 差し / 追込 / 不明
    groups: dict[str, list[dict]] = {"逃げ": [], "先行": [], "差し": [], "追込": [], "不明": []}
    for r in kyaku_records:
        groups[r["kyaku"]].append(r)
    counts = {k: len(v) for k, v in groups.items()}

    # 各 group 内は前4角通過位置順 (前にいる順) で sort、無ければ馬番順
    for k in groups:
        groups[k].sort(key=lambda r: (
            r["pos4"] if r["pos4"] is not None else 99,
            r["horse"].get("umaban") or 99,
        ))

    # ペース判定
    n_nige = counts["逃げ"]
    n_sen  = counts["先行"]
    n_sashi = counts["差し"] + counts["追込"]
    if n_nige >= 2:
        pace = "Hペース"; pace_color = "#f38ba8"
        pace_msg = "逃げ馬複数 → 前崩れ気配、差し・追込有利"
    elif n_nige == 0 and n_sen >= 3:
        pace = "Sペース"; pace_color = "#89b4fa"
        pace_msg = "明確な逃げ馬不在 + 先行多 → スロー濃厚、上り勝負・先行有利"
    elif n_sashi >= n_sen + n_nige:
        pace = "Mペース (差し決着)"; pace_color = "#cba6f7"
        pace_msg = "差し型多数 → 上り3F が決まる末脚勝負"
    else:
        pace = "Mペース"; pace_color = "#a6e3a1"
        pace_msg = "脚質バランス均等 → 標準的な流れ、ポジション戦"

    # 有利脚質判定
    if pace == "Hペース":
        advantage = "差し・追込"
    elif pace == "Sペース":
        advantage = "逃げ・先行"
    elif "差し決着" in pace:
        advantage = "差し"
    else:
        advantage = "先行〜差し"

    # 各馬の chip HTML 生成
    def _chip(rec: dict) -> str:
        h = rec["horse"]
        mark = h.get("mark") or ""
        name = h.get("horse_name", "?")
        umaban = h.get("umaban", "?")
        pwin = (h.get("p_win") or 0) * 100
        mark_color = MARK_COLORS.get(mark, "#475569") if mark else "#475569"
        # 馬の chip: 馬番 + 名前 + 印
        # 印がある馬は枠を太く強調
        border = f"2px solid {mark_color}" if mark else "1px solid #313244"
        return (
            f'<div style="display:inline-flex;align-items:center;gap:6px;'
            f'background:#181825;border:{border};padding:6px 12px;'
            f'border-radius:8px;margin:3px 4px;min-width:120px">'
            f'<span style="background:{mark_color};color:#fff;width:24px;height:24px;'
            f'line-height:24px;text-align:center;border-radius:50%;font-size:12px;'
            f'font-weight:bold">{mark or umaban}</span>'
            f'<span style="color:#cdd6f4;font-size:13px;font-weight:600;'
            f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'
            f'max-width:100px">{umaban}{(" " + name) if not mark else " " + name}</span>'
            f'<span style="color:#a6adc8;font-size:11px;margin-left:auto">{pwin:.0f}%</span>'
            f'</div>'
        )

    # 隊列図のセクション (逃げ → 先行 → 差し → 追込)
    section_meta = [
        ("逃げ", "#f38ba8", "🏃"),
        ("先行", "#fab387", "→"),
        ("差し", "#a6e3a1", "←"),
        ("追込", "#89b4fa", "←←"),
    ]
    lineup_html_parts = []
    prev_filled = False
    for k, color, icon in section_meta:
        recs = groups.get(k, [])
        if not recs:
            continue
        # セクション間の "↓" arrow
        if prev_filled:
            lineup_html_parts.append(
                f'<div style="text-align:center;color:#6c7086;font-size:14px;'
                f'margin:6px 0;letter-spacing:4px">↓</div>'
            )
        chips = "".join(_chip(r) for r in recs)
        lineup_html_parts.append(f"""
        <div style="display:flex;align-items:flex-start;gap:14px;padding:10px 4px;
                    border-left:3px solid {color};background:rgba(0,0,0,0.25);
                    border-radius:6px;margin:2px 0">
          <div style="min-width:80px;color:{color};font-size:16px;font-weight:bold;
                      padding-top:8px;text-align:center">
            <div style="font-size:20px">{icon}</div>
            <div>{k} ({len(recs)})</div>
          </div>
          <div style="flex-grow:1;display:flex;flex-wrap:wrap;align-items:center">
            {chips}
          </div>
        </div>
        """)
        prev_filled = True

    # 不明セクション (あれば最後に灰色で)
    unknown = groups.get("不明", [])
    unknown_section = ""
    if unknown:
        chips = "".join(_chip(r) for r in unknown)
        unknown_section = f"""
        <div style="margin-top:14px;padding:8px 12px;background:rgba(108,112,134,0.1);
                    border-radius:6px;border:1px dashed #45475a">
          <div style="color:#6c7086;font-size:12px;margin-bottom:6px">
            脚質不明 ({len(unknown)} 頭、過去走情報不足)
          </div>
          <div style="display:flex;flex-wrap:wrap">{chips}</div>
        </div>
        """

    ui.html(f"""
    <div style="background:linear-gradient(135deg,#0d1421 0%,#16213e 100%);
                border:1px solid {pace_color};border-radius:14px;padding:22px 26px;
                margin-bottom:18px">
      <h2 style="margin:0 0 14px 0;color:#cdd6f4;font-size:24px;font-weight:bold">
        🏇 想定隊列
      </h2>
      <div style="display:flex;align-items:center;gap:14px;margin-bottom:14px;
                  flex-wrap:wrap">
        <span style="background:{pace_color};color:#1e1e2e;padding:6px 16px;
                     border-radius:12px;font-weight:bold;font-size:16px">{pace}</span>
        <span style="color:#cdd6f4;font-size:14px">{pace_msg}</span>
      </div>
      <div style="background:rgba(0,0,0,0.18);padding:12px 14px;border-radius:8px">
        {"".join(lineup_html_parts)}
      </div>
      {unknown_section}
      <div style="margin-top:14px;display:flex;justify-content:space-between;
                  align-items:center;color:#a6adc8;font-size:13px;flex-wrap:wrap;gap:10px">
        <div>
          脚質分布: 逃げ {counts['逃げ']} / 先行 {counts['先行']} /
          差し {counts['差し']} / 追込 {counts['追込']}
          {' (不明 ' + str(counts['不明']) + ')' if counts['不明'] > 0 else ''}
        </div>
        <div style="color:#fab387;font-weight:bold;font-size:14px">
          有利脚質: {advantage}
        </div>
      </div>
    </div>
    """)


def render_pedigree_analysis(race: dict) -> None:
    """血統分析タブのメイン描画。

    上段: 父・母父のコース別ランキング (pedigree_stats.json)
    下段: 今回の合致馬リスト (bundle.json 内の各馬の sire を rank と JOIN)
    """
    meta = race.get("race_meta", {}) or {}
    place = meta.get("place", "")
    course_str = meta.get("course", "")
    surface, dist = parse_course_str(course_str)

    if not (place and surface and dist):
        ui.html(f"""
        <div style="background:#1e1e2e;border-left:3px solid #6c7086;
                    padding:14px 16px;border-radius:8px;color:#a6adc8;margin-bottom:14px">
          🧬 コース情報を解析できませんでした ({place} / {course_str})
        </div>
        """)
        return

    course_stats = lookup_pedigree_course(place, surface, dist)
    if not course_stats:
        ui.html(f"""
        <div style="background:#1e1e2e;border-left:3px solid #6c7086;
                    padding:14px 16px;border-radius:8px;color:#a6adc8;margin-bottom:14px">
          🧬 <b style="color:#cdd6f4">{place} {surface}{dist}m</b> の血統データなし<br>
          <span style="color:#6c7086;font-size:12px">
            (data/pedigree_stats.json 未配置、または距離帯がサンプル不足。
             ローカルでは <code>python build_pedigree_data.py</code> で生成)
          </span>
        </div>
        """)
        return

    band = course_stats.get("dist_band", [dist, dist])
    n_total = course_stats.get("n_total_runs", 0)
    base_fuku = course_stats.get("baseline_fuku_rate", 0)
    sire_stats = course_stats.get("sire", []) or []
    bms_stats = course_stats.get("broodmare_sire", []) or []

    # ヘッダ
    ui.html(f"""
    <div style="background:linear-gradient(135deg,#1e1e2e,#181825);
                border:1px solid #f9e2af;border-radius:8px;
                padding:14px 18px;margin-bottom:14px">
      <div style="color:#f9e2af;font-size:18px;font-weight:bold;margin-bottom:6px">
        🧬 血統分析: {place} {surface}{dist}m
      </div>
      <div style="color:#a6adc8;font-size:13px;line-height:1.5">
        過去 10 年集計 / 距離帯 {band[0]}-{band[1]}m /
        サンプル {n_total:,} 走 / 全体複勝率 {base_fuku:.1f}%
      </div>
    </div>
    """)

    rank_color = {
        "SS": "#ffd700", "S": "#fab387", "A": "#a6e3a1",
        "B": "#89b4fa", "C": "#94a3b8", "D": "#f38ba8",
    }

    def _stats_table(title: str, stats: list[dict], emoji: str) -> str:
        if not stats:
            return f"""
            <div style="color:#6c7086;padding:10px 16px">
              {emoji} {title}: サンプル不足 (出走 15 回以上の {title} 無し)
            </div>"""
        body = []
        for s in stats[:15]:  # top 15 表示 (全 30 のうち)
            rk = s.get("rank", "")
            rc = rank_color.get(rk, "#6c7086")
            unfit = s.get("unfit_reason", "")
            body.append(f"""
            <tr style="border-bottom:1px solid #313244">
              <td style="padding:8px 12px;text-align:center;
                         background:{rc};color:#1e1e2e;font-weight:bold;
                         font-size:13px;width:48px">{rk}</td>
              <td style="padding:8px 14px;color:#cdd6f4;font-weight:600;font-size:14px">{s['name']}</td>
              <td style="padding:8px 12px;text-align:right;color:#a6adc8;font-size:13px">{s['n_runs']}</td>
              <td style="padding:8px 12px;text-align:right;color:#a6adc8;font-size:13px">
                {s['n_1']}-{s['n_2']}-{s['n_3']}-{s['n_out']}
              </td>
              <td style="padding:8px 12px;text-align:right;color:#a6e3a1;font-weight:bold">{s['win_rate']:.1f}%</td>
              <td style="padding:8px 12px;text-align:right;color:#fab387;font-weight:bold">{s['fuku_rate']:.1f}%</td>
              <td style="padding:8px 14px;color:#f38ba8;font-size:12px;line-height:1.4">
                {unfit if rk == "D" else ""}
              </td>
            </tr>""")
        return f"""
        <div style="background:#0a0a14;border-radius:8px;padding:12px;margin-bottom:14px">
          <div style="color:#fab387;font-weight:bold;font-size:15px;margin-bottom:10px">
            {emoji} {title}別成績 (出走 15 回以上、上位 15)
          </div>
          <table style="width:100%;border-collapse:collapse">
            <thead>
              <tr style="background:#1e1e2e;border-bottom:2px solid #fab387">
                <th style="padding:8px;color:#fab387;font-size:12px;text-align:center">ランク</th>
                <th style="padding:8px;color:#fab387;font-size:12px;text-align:left">{title}</th>
                <th style="padding:8px;color:#fab387;font-size:12px;text-align:right">出走</th>
                <th style="padding:8px;color:#fab387;font-size:12px;text-align:right">成績</th>
                <th style="padding:8px;color:#fab387;font-size:12px;text-align:right">勝率</th>
                <th style="padding:8px;color:#fab387;font-size:12px;text-align:right">複勝率</th>
                <th style="padding:8px;color:#fab387;font-size:12px;text-align:left">不向き理由</th>
              </tr>
            </thead>
            <tbody>{"".join(body)}</tbody>
          </table>
        </div>"""

    ui.html(_stats_table("父", sire_stats, "★"))
    ui.html(_stats_table("母父", bms_stats, "★"))

    # ── 今回の合致馬 ──
    # 各馬の sire / broodmare_sire を rank と JOIN
    sire_rank_map = {s["name"]: s for s in sire_stats}
    bms_rank_map = {s["name"]: s for s in bms_stats}

    # 馬を rank 強度 (SS=6 .. D=1) で sort
    rank_order = {"SS": 6, "S": 5, "A": 4, "B": 3, "C": 2, "D": 1, "": 0}

    horses = race.get("horses", []) or []
    match_rows = []
    for h in horses:
        umaban = h.get("umaban") or "?"
        name = h.get("horse_name") or "-"
        mark = h.get("mark") or ""
        ped = h.get("pedigree") or {}
        sire = ped.get("sire", "")
        bms = ped.get("broodmare_sire", "")
        s_entry = sire_rank_map.get(sire) if sire else None
        b_entry = bms_rank_map.get(bms) if bms else None
        if not s_entry and not b_entry:
            continue  # 該当無しは表示しない
        # 父・母父の中で高い方を表示ランクに採用
        best_rank = ""
        best_score = 0
        for ent in (s_entry, b_entry):
            if ent and rank_order.get(ent.get("rank", ""), 0) > best_score:
                best_score = rank_order.get(ent.get("rank", ""), 0)
                best_rank = ent.get("rank", "")
        match_rows.append({
            "umaban": umaban, "name": name, "mark": mark,
            "sire": sire, "sire_rank": s_entry.get("rank") if s_entry else "",
            "sire_fuku": s_entry.get("fuku_rate") if s_entry else None,
            "sire_unfit": s_entry.get("unfit_reason") if s_entry else "",
            "bms": bms, "bms_rank": b_entry.get("rank") if b_entry else "",
            "bms_fuku": b_entry.get("fuku_rate") if b_entry else None,
            "bms_unfit": b_entry.get("unfit_reason") if b_entry else "",
            "best_score": best_score, "best_rank": best_rank,
        })

    match_rows.sort(key=lambda r: -r["best_score"])

    if not match_rows:
        ui.html("""
        <div style="background:#1e1e2e;border-left:3px solid #6c7086;
                    padding:12px 16px;border-radius:8px;color:#a6adc8">
          ✨ 今回の合致馬: なし (どの馬の父・母父も上位ランクにヒットしませんでした)
        </div>""")
        return

    body = []
    for m in match_rows:
        rk = m["best_rank"]
        rc = rank_color.get(rk, "#6c7086")
        mark = m["mark"]
        mark_color = MARK_COLORS.get(mark, "#475569") if mark else "transparent"
        mark_html = (
            f'<span style="background:{mark_color};color:#fff;font-weight:bold;'
            f'padding:2px 8px;border-radius:4px;font-size:13px">{mark}</span>'
            if mark else ""
        )
        def _cell(name, rank, fuku, unfit):
            if not name:
                return '<td style="padding:8px 10px;color:#6c7086;font-size:12px">—</td>'
            color = rank_color.get(rank, "#6c7086")
            icon = ""
            if rank in ("SS", "S"):
                icon = "★"
            elif rank == "D":
                icon = "⚠"
            badge = (f'<span style="background:{color};color:#1e1e2e;font-weight:bold;'
                     f'padding:1px 6px;border-radius:3px;font-size:11px;margin-right:6px">{rank}</span>'
                     if rank else "")
            note = ""
            if rank == "D" and unfit:
                note = f'<div style="color:#f38ba8;font-size:11px;margin-top:3px">{unfit}</div>'
            return f"""<td style="padding:8px 10px;color:#cdd6f4;font-size:13px">
              {badge}{name}{icon} {f'<span style="color:#6c7086;font-size:11px">({fuku:.1f}%)</span>' if fuku is not None else ''}
              {note}
            </td>"""
        body.append(f"""
        <tr style="border-bottom:1px solid #313244">
          <td style="padding:10px;text-align:center;background:{rc};color:#1e1e2e;
                     font-weight:bold;width:48px">{rk}</td>
          <td style="padding:8px 10px;text-align:center;width:60px">{mark_html}</td>
          <td style="padding:8px 10px;color:#cdd6f4;font-weight:bold;width:50px">{m['umaban']}</td>
          <td style="padding:8px 14px;color:#cdd6f4;font-weight:600">{m['name']}</td>
          {_cell(m['sire'], m['sire_rank'], m['sire_fuku'], m['sire_unfit'])}
          {_cell(m['bms'], m['bms_rank'], m['bms_fuku'], m['bms_unfit'])}
        </tr>""")
    ui.html(f"""
    <div style="background:#0a0a14;border-radius:8px;padding:12px">
      <div style="color:#a6e3a1;font-weight:bold;font-size:15px;margin-bottom:10px">
        ✨ 今回の合致馬 (このコースで父・母父が上位/下位ランクの馬)
      </div>
      <table style="width:100%;border-collapse:collapse">
        <thead>
          <tr style="background:#1e1e2e;border-bottom:2px solid #a6e3a1">
            <th style="padding:8px;color:#a6e3a1;font-size:12px;text-align:center">最大ランク</th>
            <th style="padding:8px;color:#a6e3a1;font-size:12px;text-align:center">印</th>
            <th style="padding:8px;color:#a6e3a1;font-size:12px;text-align:left">馬番</th>
            <th style="padding:8px;color:#a6e3a1;font-size:12px;text-align:left">馬名</th>
            <th style="padding:8px;color:#a6e3a1;font-size:12px;text-align:left">父</th>
            <th style="padding:8px;color:#a6e3a1;font-size:12px;text-align:left">母父</th>
          </tr>
        </thead>
        <tbody>{"".join(body)}</tbody>
      </table>
    </div>""")


def render_course_analysis(race: dict) -> None:
    """コース分析タブのメイン描画。
    上段: 展開予想 (bundle + weekly コーナー位置から)
    下段: 過去成績テーブル (master_v2、HF では unavailable)
    """
    meta = race.get("race_meta", {}) or {}
    place = meta.get("place", "")
    course_str = meta.get("course", "")

    stats = compute_course_stats_v2(place, course_str)
    if not stats:
        ui.html(f"""
        <div style="background:#1e1e2e;border-left:3px solid #6c7086;
                    padding:14px 16px;border-radius:8px;color:#a6adc8;
                    font-size:14px;margin-bottom:14px">
          📊 <b style="color:#cdd6f4">{place} {course_str}</b> の過去成績データなし<br>
          <span style="color:#6c7086;font-size:12px">
            (master_v2_*.csv が無い、またはサンプル数 100 未満。
            HF Spaces では大物 CSV は除外しているのでローカル限定の表示です。)
          </span>
        </div>
        """)
        return

    # Header
    ui.html(f"""
    <div style="background:linear-gradient(135deg,#0d1421 0%,#16213e 100%);
                border:1px solid #f39c12;border-radius:14px;padding:18px 24px;
                margin-bottom:18px">
      <h2 style="margin:0;color:#cdd6f4;font-size:26px;font-weight:bold">
        📊 {place} {course_str} 過去成績
      </h2>
      <div style="color:#a6adc8;font-size:15px;margin-top:6px">
        対象: {stats['n_races']:,} レース / {stats['n_starts']:,} 出走 (2013-2025)
        — 単位 [%]
      </div>
    </div>
    """)

    # コース別好走脚質 (どの脚質が勝ちやすいコースか)
    kyaku_rows = stats.get("kyaku") or []
    if kyaku_rows:
        # 勝率の高い脚質を強調 (▲ marker)
        max_rate = max(r["win_rate"] for r in kyaku_rows)
        annotated_rows = []
        for r in kyaku_rows:
            label = r["label"]
            if r["win_rate"] >= max_rate - 0.3:
                label = f"{label} ★"
            annotated_rows.append({**r, "label": label})
        ui.html(_stats_table_html(
            "コース別好走脚質 (★ = 最も勝率が高い脚質)",
            annotated_rows, "🏃"))

    # 条件別成績の 2x2 グリッド:
    #   Row 1: 枠順   | 馬番
    #   Row 2: 年齢   | 性別
    # 各 cell は flex-1 (画面幅の半分) なので密度が同じ。
    # 馬番は行数多いので Row 1 が縦に長くなり、年齢/性別は下に流れる。
    with ui.row().classes("w-full no-wrap gap-3 items-start"):
        with ui.column().classes("flex-1"):
            ui.html(_stats_table_html("枠順", stats["waku"], "🎫"))
        with ui.column().classes("flex-1"):
            ui.html(_stats_table_html("馬番", stats["uma"], "🏇"))
    with ui.row().classes("w-full no-wrap gap-3 items-start"):
        with ui.column().classes("flex-1"):
            ui.html(_stats_table_html("年齢", stats["age"], "🎂"))
        with ui.column().classes("flex-1"):
            ui.html(_stats_table_html("性別", stats["sex"], "♂♀"))


# ============================================================
# 全頭分析: 散布図 + レーダーチャート
# ============================================================
def race_scatter_option(horses: list) -> dict:
    data = []
    for h in horses:
        p_win = (h.get("p_win") or 0) * 100
        tan = h.get("tansho_odds") or 0
        market = (1 / tan) * 100 if tan else 0
        mark = h.get("mark") or ""
        color = MARK_COLORS.get(mark, "#6c7086")
        size = 22 if mark == "◎" else 18 if mark == "〇" else 14 if mark == "▲" else 9
        name = h.get("horse_name", "")
        data.append({
            "value": [round(market, 2), round(p_win, 2)],
            "name": name, "symbol": "circle", "symbolSize": size,
            "itemStyle": {"color": color,
                           "borderColor": "#fff" if mark in ["◎", "〇", "▲"] else "transparent",
                           "borderWidth": 1},
            "label": {"show": True,
                       "position": "top" if mark in ["◎", "〇", "▲"] else "right",
                       "color": "#cdd6f4",
                       "fontSize": 12 if mark in ["◎", "〇", "▲"] else 10,
                       "formatter": (mark + " " + name) if mark else name},
        })
    max_val = max((d["value"][0] for d in data), default=50) * 1.1
    max_val = max(max_val, max((d["value"][1] for d in data), default=50) * 1.1)
    return {
        "title": {"text": "AI 評価 vs 市場評価",
                   "textStyle": {"color": "#cdd6f4", "fontSize": 16}, "left": "center"},
        "xAxis": {"name": "市場評価 (1/単勝オッズ × 100)",
                   "type": "value", "min": 0, "max": round(max_val, 1),
                   "splitLine": {"lineStyle": {"color": "#313244"}},
                   "axisLabel": {"color": "#a6adc8"},
                   "nameTextStyle": {"color": "#cdd6f4", "fontSize": 13},
                   "nameLocation": "middle", "nameGap": 35},
        "yAxis": {"name": "AI 予想 (勝率 %)",
                   "type": "value", "min": 0, "max": round(max_val, 1),
                   "splitLine": {"lineStyle": {"color": "#313244"}},
                   "axisLabel": {"color": "#a6adc8"},
                   "nameTextStyle": {"color": "#cdd6f4", "fontSize": 13},
                   "nameLocation": "middle", "nameGap": 50},
        "tooltip": {"trigger": "item", "backgroundColor": "#1e1e2e",
                     "borderColor": "#313244", "textStyle": {"color": "#cdd6f4"}},
        "series": [
            {"type": "scatter", "data": data, "z": 2},
            {"type": "line", "data": [[0, 0], [max_val, max_val]],
             "lineStyle": {"type": "dashed", "color": "#555", "width": 1},
             "symbol": "none", "tooltip": {"show": False}, "z": 1, "silent": True},
        ],
        "grid": {"left": 70, "right": 30, "top": 50, "bottom": 60},
        "backgroundColor": "transparent",
    }


# ============================================================
# PyCaLi 出走馬評価リスト (Streamlit と同じ a〜g 7 軸)
#   各軸は複数の候補列をもち「higher_is_better=False」なら反転後 0-10 化。
#   複数候補は分散がある最初の列を採用 (フォールバック)。
#   - 前走補正系       data/hosei/H_*.csv
#   - 前走Ave-3F 等   data/weekly/{date}.csv (本ファイルは Streamlit と同一パーサ)
#   - 調教 (g)        data/training/H-*.csv, W-*.csv (HF Spaces には未配置のため
#                     g は valid=False となり「−」表示。ローカルでは表示される)
# ============================================================
PYCA_AXES = [
    # (key, label, [(列名, higher_is_better), ...])
    ("a", "総合力",   [("__ai_win__", True)]),
    ("b", "スピード", [("前走補正",      True),
                       ("前走Ave-3F",    False),
                       ("前走走破タイム", False)]),
    ("c", "末脚",     [("前走補9",       True),
                       ("前走上り3F",    False)]),
    ("d", "前走成績", [("前走確定着順",  False)]),
    ("e", "市場評価", [("前走人気",       False),
                       ("前走単勝オッズ", False)]),
    ("f", "ペース適性", [("前走RPCI",     True),
                          ("前走PCI3",     True),
                          ("前走Ave-3F",   False)]),
    ("g", "調教",     [("trn_hanro_lap1",  False),
                       ("trn_hanro_time1", False),
                       ("trn_wc_3f",       False),
                       ("trn_wc_5f",       False)]),
]


def _norm_0_10_list(values: list[float | None]) -> tuple[list[float], bool]:
    """値リストを 0〜10 に min-max 正規化。全同値や全欠損なら全 5.0 + valid=False"""
    nums = [v for v in values if v is not None]
    if not nums:
        return [5.0] * len(values), False
    vmin, vmax = min(nums), max(nums)
    if vmax - vmin < 1e-9:
        return [5.0] * len(values), False
    out = [
        (v - vmin) / (vmax - vmin) * 10.0 if v is not None else 5.0
        for v in values
    ]
    return out, True


def _rank_list(values: list[float]) -> list[int]:
    """高い順に 1, 2, ... ranking (同値は同位 = min ランク)"""
    sorted_unique = sorted(set(values), reverse=True)
    rank_map = {v: i + 1 for i, v in enumerate(sorted_unique)}
    return [rank_map[v] for v in values]


def compute_pyca_features(horses: list[dict],
                            date_str: str | None = None,
                            race_id: str | None = None) -> dict:
    """Streamlit と同じ a〜g を計算。

    bundle.json (p_win 等) + weekly CSV (前走 系) + hosei (前走補正/補9) を
    merge し、各軸を 0-10 正規化 + レース内順位を付与。

    返り値:
      'norm'  : {axis_key -> [v0, v1, ...]}    (0-10)
      'rank'  : {axis_key -> [r0, r1, ...]}    (1=最良)
      'valid' : {axis_key -> bool}             (有効サンプルあり)
      'pyca'  : [s0, s1, ...]                  (PyCaLi 指数 0-100)
      'pyca_rank': [r0, r1, ...]
    """
    n = len(horses)
    if n == 0:
        return {"norm": {}, "rank": {}, "valid": {},
                "pyca": [], "pyca_rank": []}

    # --- 各馬の生値テーブル: feature_dict[馬index][列名] = 値 ---
    p_win_list = [(h.get("p_win") or 0) for h in horses]

    feature_dict: list[dict[str, float | None]] = []
    for h in horses:
        row: dict[str, float | None] = {}
        # bundle.json 由来 (a 軸)
        row["__ai_win__"] = h.get("p_win") or 0
        # weekly + hosei から取得
        if date_str and race_id:
            try:
                feats = get_horse_features(date_str, race_id,
                                              h.get("umaban") or 0)
                row.update(feats)
            except Exception:
                pass
        feature_dict.append(row)

    # --- 各軸を 0-10 正規化 ---
    norm: dict[str, list[float]] = {}
    rank: dict[str, list[int]] = {}
    valid: dict[str, bool] = {}

    for key, _label, candidates in PYCA_AXES:
        used: list[float] | None = None
        used_valid = False
        for col, higher_is_better in candidates:
            vals = [feature_dict[i].get(col) for i in range(n)]
            if all(v is None for v in vals):
                continue
            normed, ok = _norm_0_10_list(vals)
            if ok:
                if not higher_is_better:
                    normed = [10.0 - v for v in normed]
                used, used_valid = normed, True
                break
            if used is None:
                used = normed
        if used is None:
            used = [5.0] * n
        norm[key] = used
        rank[key] = (_rank_list(used) if used_valid else [0] * n)
        valid[key] = used_valid

    # --- PyCaLi 指数 ---
    # base   = p_sho × 100 (3 着以内率、Streamlit の `score` と同等のレンジ 30-50)
    # boost  = 補正特徴量 (b〜f) 平均 - 5 を ×0.5 (元 1.0)。係数を抑えたのは
    #          base のわずかな差を boost が打ち消して印 (ai_rank) と PyCaLi 順位が
    #          逆転するのを防ぐため
    p_sho_list = [(h.get("p_sho") or 0) for h in horses]
    pyca: list[float] = []
    for i in range(n):
        base = p_sho_list[i] * 100.0
        booster_keys = [k for k in ["b", "c", "d", "e", "f"] if valid[k]]
        if booster_keys:
            avg = sum(norm[k][i] for k in booster_keys) / len(booster_keys)
            boost = avg - 5.0       # -5..+5
        else:
            boost = 0.0
        score = max(0.0, min(100.0, base + boost * 0.5))
        pyca.append(score)
    # 表示順位は ai_rank (印 順) を踏襲して印と PyCaLi 指数の順位ズレをなくす。
    # ai_rank 不在時は PyCaLi 値の降順でフォールバック。
    if all(h.get("ai_rank") for h in horses):
        ai_rank_list = [h.get("ai_rank") for h in horses]
        pyca_rank = ai_rank_list
    else:
        pyca_rank = _rank_list(pyca)

    return {
        "norm": norm, "rank": rank, "valid": valid,
        "pyca": pyca, "pyca_rank": pyca_rank,
    }


def _pyca_radar_option(values: list[float], labels: list[str],
                        name: str, color: str) -> dict:
    return {
        "tooltip": {"backgroundColor": "#1e1e2e",
                     "textStyle": {"color": "#cdd6f4"}},
        "title": {"text": name,
                   "textStyle": {"color": "#f5e0dc", "fontSize": 14},
                   "left": "center", "top": 6},
        "radar": {
            "shape": "polygon",
            "indicator": [{"name": lbl, "max": 10} for lbl in labels],
            "splitArea": {"show": True,
                "areaStyle": {"color": ["rgba(255,255,255,0.02)",
                                          "rgba(255,255,255,0.04)"]}},
            "splitLine": {"lineStyle": {"color": "#313244"}},
            "axisLine":  {"lineStyle": {"color": "#313244"}},
            "axisName":  {"color": "#a6adc8", "fontSize": 11},
            "center": ["50%", "58%"], "radius": "62%",
        },
        "series": [{
            "type": "radar",
            "data": [{
                "value": [round(v, 2) for v in values],
                "name": name,
                "lineStyle": {"color": color, "width": 2},
                "areaStyle": {"color": color, "opacity": 0.3},
                "symbol": "circle", "symbolSize": 5,
                "itemStyle": {"color": color},
            }],
        }],
        "backgroundColor": "transparent",
    }


def _eval_left_html(h: dict, pyca: float, prank: int) -> str:
    name = h.get("horse_name", "?")
    uma = h.get("umaban", "?")
    mark = h.get("mark") or ""
    mark_color = MARK_COLORS.get(mark, "#6c7086")
    mark_html = (
        f'<span style="background:{mark_color};color:#fff;font-size:18px;'
        f'font-weight:bold;width:32px;height:32px;line-height:32px;'
        f'text-align:center;border-radius:50%;display:inline-block;'
        f'margin-right:8px">{mark}</span>'
    ) if mark else ''
    odds = h.get("tansho_odds") or 0
    p_sho_pct = (h.get("p_sho") or 0) * 100
    rank_color = "#a6e3a1" if prank == 1 else (
        "#89b4fa" if prank <= 3 else "#cdd6f4")
    return f"""
    <div style="padding:6px 0">
      <div style="color:#888;font-size:14px;margin-bottom:4px">{uma}番</div>
      <div style="color:#cdd6f4;font-size:22px;font-weight:bold;line-height:1.2;
                  margin-bottom:8px">
        {mark_html}{name}
      </div>
      <div style="color:#a6adc8;font-size:13px;margin-bottom:14px">
        単勝 {odds:.1f}倍 / 複勝予測 {p_sho_pct:.1f}%
      </div>
      <div>
        <div style="color:#6c7086;font-size:13px;margin-bottom:2px">PyCaLi指数</div>
        <div style="line-height:1.0">
          <span style="font-size:42px;font-weight:bold;color:{rank_color}">
            {pyca:.1f}</span>
          <span style="font-size:18px;color:#cdd6f4;margin-left:6px">
            ({prank}位)</span>
        </div>
      </div>
    </div>
    """


def _eval_bars_html(feats: dict, idx: int) -> str:
    rows = ['<div style="font-size:13px;color:#6c7086;margin-bottom:6px">'
             '指数内訳 / 値 / レース内順位</div>']
    for key, label, _desc in PYCA_AXES:
        v = feats["norm"][key][idx]
        rk = feats["rank"][key][idx]
        valid = feats["valid"][key]
        bar = max(0, min(100, int(round(v * 10))))
        if not valid:
            color = "#6c7086"
            rank_txt = "−"
            star = ""
        else:
            color = "#a6e3a1" if rk == 1 else (
                "#89b4fa" if rk <= 3 else "#cdd6f4")
            rank_txt = f"{rk}位"
            star = "★" if rk <= 3 else ""
        rows.append(f"""
        <div style="display:flex;align-items:center;gap:8px;margin:5px 0;
                    font-size:14px">
          <div style="width:84px;color:#a6adc8;flex-shrink:0">
            {key}. {label}</div>
          <div style="flex:1;height:9px;background:#313244;border-radius:4px;
                      overflow:hidden;min-width:60px">
            <div style="height:100%;width:{bar}%;background:{color}"></div>
          </div>
          <div style="width:42px;text-align:right;color:{color};
                      font-weight:bold">{v:.1f}</div>
          <div style="width:54px;text-align:right;color:#6c7086">
            {rank_txt}{star}</div>
        </div>
        """)
    return "".join(rows)


def render_pyca_eval_list(race: dict, date_str: str | None = None) -> None:
    """Streamlit 版互換の評価リスト (前提: ui コンテナの中で呼ばれる)。

    a〜g 7 軸 (Streamlit と同一定義):
      a 総合力 / b スピード / c 末脚 / d 前走成績 / e 市場評価 /
      f ペース適性 / g 調教 (HF Spaces には training CSV がないため g は「−」)
    """
    horses = list(race.get("horses", []))
    if not horses:
        return
    race_id = race.get("race_id")
    feats = compute_pyca_features(horses, date_str=date_str, race_id=race_id)
    # 並び順は ai_rank (印 順) を優先、なければ PyCaLi 降順
    if all(h.get("ai_rank") for h in horses):
        order = sorted(range(len(horses)),
                        key=lambda i: horses[i].get("ai_rank", 99))
    else:
        order = sorted(range(len(horses)), key=lambda i: -feats["pyca"][i])
    labels = [lbl for _, lbl, _ in PYCA_AXES]

    ui.label("🔍 出走馬評価リスト (PyCaLi指数)").classes(
        "text-xl font-bold mt-6 mb-1")
    ui.label(
        "並び順は AI モデルの最終評価 (◎〇▲△ の印) 順。"
        "PyCaLi指数 = AI 3 着以内率 (p_sho) を基準に、a〜g の 7 軸 (前走補正 / 前走Ave-3F /"
        "前走確定着順 等) で微調整した総合スコア (0-100)。右側のバーは"
        "レース内 0〜10 正規化値。★は上位 3 位以内。データが無い軸は「−」表示。"
    ).classes("text-sm text-slate-400 mb-3")

    for i in order:
        h = horses[i]
        pyca = feats["pyca"][i]
        prank = feats["pyca_rank"][i]
        mark = h.get("mark") or ""
        color = MARK_COLORS.get(mark, "#89b4fa")
        name = h.get("horse_name", "?")
        radar_vals = [feats["norm"][k][i] for k, _, _ in PYCA_AXES]

        with ui.element("div").classes(
            "p-3 bg-slate-900 rounded-lg border border-slate-700 mb-2"
        ):
            with ui.row().classes("w-full no-wrap items-stretch gap-3"):
                with ui.column().classes("flex-shrink-0").style("width:240px"):
                    ui.html(_eval_left_html(h, pyca, prank))
                with ui.column().classes("flex-shrink-0").style("width:300px"):
                    ui.echart(
                        _pyca_radar_option(radar_vals, labels, name, color)
                    ).classes("w-full").style("height:280px")
                with ui.column().classes("flex-grow"):
                    ui.html(_eval_bars_html(feats, i))


def horse_radar_option(h: dict) -> dict:
    p_win = (h.get("p_win") or 0) * 100
    p_sho = (h.get("p_sho") or 0) * 100
    tan = h.get("tansho_odds") or 99
    fuku_low = h.get("fuku_odds_low") or 0
    fuku_high = h.get("fuku_odds_high") or 0
    fuku_mid = (fuku_low + fuku_high) / 2 if fuku_low and fuku_high else 0
    ev_tan = (h.get("p_win") or 0) * tan
    ev_fuku = (h.get("p_sho") or 0) * fuku_mid
    pop = (1 / tan) * 100 if tan else 0
    mark = h.get("mark") or ""
    color = MARK_COLORS.get(mark, "#89b4fa")

    return {
        "tooltip": {"backgroundColor": "#1e1e2e", "textStyle": {"color": "#cdd6f4"}},
        "radar": {
            "shape": "polygon",
            "indicator": [
                {"name": "勝率", "max": 50},
                {"name": "複勝率", "max": 100},
                {"name": "単EV", "max": 3},
                {"name": "複EV", "max": 2.5},
                {"name": "人気", "max": 50},
            ],
            "splitArea": {"show": True,
                           "areaStyle": {"color": ["rgba(255,255,255,0.02)", "rgba(255,255,255,0.04)"]}},
            "splitLine": {"lineStyle": {"color": "#313244"}},
            "axisLine": {"lineStyle": {"color": "#313244"}},
            "axisName": {"color": "#a6adc8", "fontSize": 12},
            "center": ["50%", "55%"], "radius": "60%",
        },
        "series": [{
            "type": "radar",
            "data": [{
                "value": [round(p_win, 1), round(p_sho, 1),
                          round(ev_tan, 2), round(ev_fuku, 2), round(pop, 1)],
                "name": h.get("horse_name", ""),
                "lineStyle": {"color": color, "width": 2},
                "areaStyle": {"color": color, "opacity": 0.3},
                "symbol": "circle", "symbolSize": 6,
                "itemStyle": {"color": color},
            }],
        }],
        "backgroundColor": "transparent",
    }


# ============================================================
# UI 構築
# ============================================================
def _ollama_up() -> bool:
    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False


def _soudan_status():
    """(利用可能か, 理由テキスト) を返す。"""
    if soudan is None:
        return False, "⚠️ 相談AIモジュール (assistant/) を読み込めませんでした。"
    if "HF_SPACE_ID" in os.environ or "DOCKER_CONTAINER" in os.environ:
        return False, ("💡 相談AIは『ローカル起動』専用です（公開版にはLLMが居ません）。\n"
                       "PCで  python nicegui_app.py  を起動し、Ollama を立ち上げてください。")
    if not _ollama_up():
        return False, ("⚠️ Ollama に接続できません (localhost:11434)。\n"
                       "スタートメニューから Ollama を起動 → 🔄 再確認 を押してください。")
    return True, ""


def build_soudan_tab(box, state):
    """相談AI チャットタブを構築。ローカル + Ollama 稼働時のみ有効。"""
    box.clear()
    with box:
        ok, reason = _soudan_status()
        if not ok:
            with ui.column().classes("w-full p-4 gap-3"):
                ui.label(reason).classes("text-orange-300 whitespace-pre-line text-base")
                ui.button("🔄 再確認", on_click=lambda: build_soudan_tab(box, state)) \
                    .props("outline color=orange")
            return

        ui.label("💬 表示中のレースについて自然文で。例:「エフハリストは?」「9番どう?」"
                 "「5番と12番どっち軸?」「このレース見送るべき?」") \
            .classes("text-slate-300 text-base p-2")
        msg_area = ui.scroll_area().classes("w-full") \
            .style("height:55vh;border:1px solid #334155;border-radius:10px")
        sess = {"messages": None, "bundle": None, "date": None, "idx": None}

        async def send():
            q = (inp.value or "").strip()
            if not q:
                return
            date = state.get("date")
            if not date:
                ui.notify("先に開催日を選んでください", type="warning")
                return
            bundle = state.get("bundle")
            if not bundle:
                try:
                    bundle = soudan.tools.load_bundle(date)
                except Exception as e:
                    ui.notify(f"bundle 読込失敗: {e}", type="negative")
                    return
            # 今表示中レースの index を特定（無ければ 0）
            idx = 0
            r = state.get("race")
            if r:
                for i, x in enumerate(bundle["races"]):
                    if x.get("race_id") == r.get("race_id"):
                        idx = i
                        break
            inp.value = ""
            # 日付 or 対象レースが変わったらセッション作り直し（名簿を入れ直す）
            if (sess["messages"] is None or sess["date"] != date
                    or sess["idx"] != idx):
                try:
                    _, msgs = soudan._init_messages(date, focus_idx=idx)
                except Exception as e:
                    ui.notify(f"bundle 読込失敗: {e}", type="negative")
                    return
                sess.update(messages=msgs, bundle=bundle, date=date, idx=idx)
                rm = bundle["races"][idx]["race_meta"]
                with msg_area:
                    ui.label(f"📍 レース[{idx}] {rm.get('place')} {rm.get('course')} "
                             f"{rm.get('class')} を対象に回答します") \
                        .classes("text-xs text-slate-400")
            with msg_area:
                ui.chat_message(q, name="あなた", sent=True)
                spinner = ui.spinner("dots", size="lg", color="primary")
            sess["messages"].append({"role": "user", "content": q})
            try:
                answer = await run.io_bound(
                    soudan._complete, sess["messages"], sess["bundle"])
            except Exception as e:
                answer = f"[エラー] {e}"
            spinner.delete()
            with msg_area:
                ui.chat_message(answer, name="🏇 PyCaLiAI", sent=False)
            try:
                msg_area.scroll_to(percent=1.0)
            except Exception:
                pass

        with ui.row().classes("w-full no-wrap items-center gap-2 mt-2"):
            inp = ui.input(placeholder="質問を入力 (Enter で送信)…") \
                .classes("flex-grow").props("outlined")
            inp.on("keydown.enter", send)
            ui.button("送信", on_click=send).props("color=primary")


@ui.page("/")
def main_page():
    ui.dark_mode().enable()

    ui.add_head_html("""
    <style>
      body, .nicegui-content { font-size: 18px; }
      h1, h2, h3 { line-height: 1.2; }
      .q-tab__label { font-size: 17px !important; }
      .q-field__native { font-size: 17px !important; }
      .q-expansion-item__container { font-size: 16px; }
    </style>
    """)

    with ui.header(elevated=True).classes("bg-slate-900"):
        ui.label("🏇 PyCaLiAI").classes("text-3xl font-bold text-white")
        ui.space()
        ui.label("NiceGUI 版 (実験 MVP v11)").classes("text-base text-slate-400")

    state = {
        "date": None, "race": None, "bundle": None,
        "by_place": {}, "current_place": None,
    }

    dates = list_dates()
    if not dates:
        with ui.column().classes("w-full p-4"):
            ui.label("⚠️ data/weekly/ に CSV がありません") \
                .classes("text-orange-400")
        return

    # =================================================================
    # サイドバー廃止 → ヘッダー直下に水平レイアウト
    #   1 行目: 開催日 select
    #   2 行目: 場所タブ (東京 / 京都 / 新潟 …)
    #   3 行目: レース番号ボタン (1R / 2R / 3R …)
    #   メイン: 左右パネル + 出走表/全頭分析/Cowork タブ
    # =================================================================
    with ui.column().classes("w-full p-4 gap-3"):
        # ── 1 行目: 開催日 ──
        with ui.row().classes("items-center gap-3"):
            ui.label("📅 開催日").classes("text-lg font-bold text-slate-200")
            date_dd = ui.select(dates, value=dates[0]).classes("w-52")

        # ── 2 行目: 場所タブ ──
        place_tabs_container = ui.element("div").classes("w-full")

        # ── 3 行目: レース番号ボタン ──
        race_buttons_container = ui.element("div").classes("w-full mb-1")

        # ── 4 行目: 直近 1 週間の好調教 Top 5 (折りたたみ) ──
        with ui.expansion("⚡ 直近 1 週間の好調教 Best 5",
                            icon="bolt", value=False) \
                .classes("w-full") \
                .style("background:rgba(243,156,18,0.05);"
                        "border:1px solid #f39c12;border-radius:12px"):
            training_top5_box = ui.element("div").classes("w-full")

        # ── 5 行目: Cowork シーズン累計 P/L (折りたたみ) ──
        with ui.expansion("📈 Cowork 累計収支 (シーズン P/L)",
                            icon="trending_up", value=False) \
                .classes("w-full") \
                .style("background:rgba(166,227,161,0.05);"
                        "border:1px solid #a6e3a1;border-radius:12px"):
            cowork_pl_box = ui.element("div").classes("w-full")

        # ── 6 行目: Grade Scope (重賞 LLM 詳細見解、折りたたみ) ──
        with ui.expansion("🏆 Grade Scope (重賞詳細見解)",
                            icon="emoji_events", value=False) \
                .classes("w-full") \
                .style("background:rgba(245,194,231,0.05);"
                        "border:1px solid #f5c2e7;border-radius:12px"):
            grade_scope_box = ui.element("div").classes("w-full")

        # ── メイン: 左右パネル ──
        with ui.row().classes("w-full no-wrap gap-3"):
            left_box = ui.element("div").classes("flex-grow").style("flex: 3")
            right_box = ui.element("div").classes("flex-grow").style("flex: 2")

        # ── メイン: タブ (出走表 / 全頭分析 / コース分析 / 血統分析 / Cowork) ──
        with ui.tabs().classes("w-full") as tabs:
            tab_shutsuba = ui.tab("📋 出走表")
            tab_bunseki = ui.tab("🔍 全頭分析")
            tab_course = ui.tab("📊 コース分析")
            tab_pedigree = ui.tab("🧬 血統分析")
            tab_bets = ui.tab("🎫 Cowork 買い目")
            tab_soudan = ui.tab("💬 相談AI")

        with ui.tab_panels(tabs, value=tab_shutsuba).classes("w-full"):
            with ui.tab_panel(tab_shutsuba):
                shutsuba_box = ui.element("div").classes("w-full")
            with ui.tab_panel(tab_bunseki):
                bunseki_box = ui.element("div").classes("w-full")
            with ui.tab_panel(tab_course):
                course_box = ui.element("div").classes("w-full")
            with ui.tab_panel(tab_pedigree):
                pedigree_box = ui.element("div").classes("w-full")
            with ui.tab_panel(tab_bets):
                bets_box = ui.element("div").classes("w-full")
            with ui.tab_panel(tab_soudan):
                soudan_box = ui.element("div").classes("w-full")
                build_soudan_tab(soudan_box, state)

    # 馬個別モーダル (出走表タブの「🐴 詳細」ボタンで開く)
    horse_dialog = ui.dialog()

    def open_horse_detail(h: dict, race: dict):
        horse_dialog.clear()
        with horse_dialog, ui.card().classes("max-w-[1100px]") \
                .style("width:90vw;max-height:85vh;overflow:auto"):
            ui.html(make_horse_detail_html(
                h, race, state.get("date")))
            with ui.row().classes("w-full justify-end"):
                ui.button("閉じる", on_click=horse_dialog.close) \
                    .props("color=primary")
        horse_dialog.open()

    def render_race(race: dict):
        state["race"] = race

        left_box.clear()
        with left_box:
            ui.html(make_left_panel_html(race))

        right_box.clear()
        with right_box:
            ui.html(make_right_panel_html(race))

        shutsuba_box.clear()
        with shutsuba_box:
            ui.html(make_shutsuba_table_html(race, date_str=state.get("date")))

            # 🐴 馬個別深掘りボタン (各馬名 → モーダルで過去5走 + コース傾向)
            horses_sorted = sorted(race.get("horses", []),
                                    key=lambda h: h.get("umaban") or 0)
            ui.label("🐴 馬個別深掘り (馬名をクリック)").classes(
                "text-lg font-bold text-slate-200 mt-4 mb-2")
            with ui.row().classes("w-full gap-1 flex-wrap"):
                for h in horses_sorted:
                    name = h.get("horse_name", "?")
                    uma = h.get("umaban", "?")
                    mark = h.get("mark") or ""
                    mark_color = MARK_COLORS.get(mark, "#475569")
                    btn = ui.button(f"{uma}番 {name}",
                                      on_click=lambda h=h: open_horse_detail(h, race)) \
                        .classes("text-white").props("dense no-caps")
                    btn.style(f"background:{mark_color}")

        bunseki_box.clear()
        with bunseki_box:
            horses_sorted = sorted(race.get("horses", []),
                                    key=lambda h: -(h.get("p_win") or 0))
            ui.label("📍 全頭ポジショニング (AI vs 市場)").classes(
                "text-xl font-bold mt-2 mb-1")
            ui.label("対角線より上 = AI 高評価。下 = AI 低評価。" +
                     "点が大きい順 ◎ > 〇 > ▲ > △.").classes(
                "text-sm text-slate-400 mb-3")
            ui.echart(race_scatter_option(horses_sorted)).classes(
                "w-full").style("height: 480px")

            # Streamlit 版互換の評価リスト (a〜g 7 軸、weekly/hosei を merge)
            render_pyca_eval_list(race, date_str=state.get("date"))

        # ── コース分析タブ (展開予想 + 条件別成績) ──
        course_box.clear()
        with course_box:
            render_tenkai_yoso(race, date_str=state.get("date"))
            render_course_analysis(race)

        # ── 血統分析タブ (父・母父ランキング + 合致馬) ──
        pedigree_box.clear()
        with pedigree_box:
            render_pedigree_analysis(race)

        bets_box.clear()
        with bets_box:
            cowork = load_cowork_bets_unified(state["date"], race["race_id"])
            if not cowork:
                # 何が見つかってないかを明示
                all_bets = _load_all_cowork_output(_cowork_output_cache_key())
                n_files = sum(1 for p in COWORK_OUTPUT_DIR.iterdir()
                                if p.is_file()) if COWORK_OUTPUT_DIR.exists() else 0
                msg = (
                    f"このレース ({race['race_id']}) の Cowork 買い目データが見つかりません。\n\n"
                    f"検索結果:\n"
                    f"  reports/cowork_output/ 内 {n_files} ファイル中、"
                    f"race_id 一致 {len(all_bets)} race 検出済 (このレースは含まれず)\n"
                    f"  reports/cowork_bets/{state['date']}/{race['race_id']}.json: 存在せず\n\n"
                    f"対応:\n"
                    f"  • {state['date']}_bets.json を作成して reports/cowork_output/ に配置\n"
                    f"  • または既存ファイルにこのレースの bets を追加\n"
                    f"  • または Streamlit の '🤖 Cowork取込' で個別保存"
                )
                ui.label(msg).classes("text-slate-400 p-4 text-sm whitespace-pre-line")
            else:
                race_nature_str = cowork.get("race_nature", "")
                race_reason = cowork.get("race_reason", "")
                bets = cowork.get("bets", [])
                if race_nature_str:
                    nat_color = NATURE_COLORS.get(race_nature_str, "#89b4fa")
                    ui.html(f"""
                    <div style="display:inline-block;background:{nat_color};color:#1e1e2e;
                                padding:6px 18px;border-radius:4px;font-weight:bold;
                                margin-bottom:10px;font-size:16px">{race_nature_str}</div>
                    """)
                if race_reason:
                    ui.html(f"""
                    <div style="background:#1e1e2e;border-left:3px solid #fab387;
                                padding:12px 16px;margin:8px 0;color:#cdd6f4;font-size:16px;
                                line-height:1.7">
                      <b style="color:#fab387">📝 根拠:</b> {race_reason}
                    </div>
                    """)
                if not bets:
                    ui.label("→ 見送り (購入なし)").classes("text-slate-400 mt-3 text-lg")
                else:
                    # 表示順: 単勝 → 複勝 → 馬連 → ワイド → 馬単 → 三連複 → 三連単
                    # 同じ馬券種内は購入額が大きい順 (本線が上)
                    btype_order = {"単勝": 0, "複勝": 1, "馬連": 2, "ワイド": 3,
                                    "馬単": 4, "三連複": 5, "三連単": 6}
                    bets = sorted(bets, key=lambda b: (
                        btype_order.get(b.get("馬券種", ""), 99),
                        -int(b.get("購入額", 0) or 0),
                    ))
                    rows = []
                    for b in bets:
                        rows.append(f"""
                        <tr style="border-bottom:1px solid #313244">
                          <td style="padding:10px 14px;color:#5865f2;font-weight:bold;
                                     font-size:15px">{b.get("馬券種","")}</td>
                          <td style="padding:10px 14px;color:#cdd6f4;font-size:15px;
                                     font-weight:bold">{b.get("買い目","")}</td>
                          <td style="padding:10px 14px;color:#a6e3a1;text-align:right;
                                     font-size:15px;font-weight:bold">¥{b.get("購入額",0):,}</td>
                          <td style="padding:10px 14px;color:#a6adc8;font-size:13px;
                                     line-height:1.6">{b.get("理由","")}</td>
                        </tr>
                        """)
                    ui.html(f"""
                    <table style="width:100%;border-collapse:collapse;
                                  background:#0a0a14;border-radius:8px;overflow:hidden">
                      <thead>
                        <tr style="background:#1e1e2e;border-bottom:2px solid #fab387">
                          <th style="padding:12px;color:#fab387;text-align:left;width:110px">馬券種</th>
                          <th style="padding:12px;color:#fab387;text-align:left;width:180px">買い目</th>
                          <th style="padding:12px;color:#fab387;text-align:right;width:120px">購入額</th>
                          <th style="padding:12px;color:#fab387;text-align:left">理由</th>
                        </tr>
                      </thead>
                      <tbody>{"".join(rows)}</tbody>
                    </table>
                    """)
                    total = sum(b.get("購入額", 0) for b in bets)
                    ui.label(f"合計予算: ¥{total:,}") \
                      .classes("text-2xl font-bold mt-3 text-amber-300")

                # ── advisor (Cowork narrative 論評) ──
                advisor = cowork.get("advisor", []) or []
                if advisor:
                    ui.label("🧠 Advisor (自由日本語論評)").classes(
                        "text-xl font-bold mt-6 mb-2 text-slate-100")
                    ui.label(
                        "Cowork による各馬の narrative 評価。"
                        "grade = 地力 (SS/S/A/B/C)、tag = 役割 (軸/妙味/罠/穴/消)。"
                    ).classes("text-sm text-slate-400 mb-3")
                    cards = []
                    for adv in advisor:
                        uma = adv.get("umaban", "")
                        name = adv.get("horse_name", "?")
                        grade = adv.get("grade", "")
                        tag = adv.get("tag") or ""
                        comment = adv.get("comment", "")
                        g_color = ADVISOR_GRADE_COLORS.get(grade, "#6c7086")
                        t_color = ADVISOR_TAG_COLORS.get(tag, "#6c7086")
                        # 印 (bundle.json) を参照
                        mark = ""
                        for h in race.get("horses", []):
                            if h.get("umaban") == uma:
                                mark = h.get("mark", "") or ""
                                break
                        mark_color = MARK_COLORS.get(mark, "#475569") if mark else "transparent"
                        mark_badge = (f'<span style="display:inline-block;background:{mark_color};'
                                       f'color:#fff;font-weight:bold;padding:2px 8px;border-radius:4px;'
                                       f'margin-right:8px;font-size:14px">{mark}</span>'
                                       if mark else "")
                        grade_badge = (f'<span style="display:inline-block;background:{g_color};'
                                        f'color:#1e1e2e;font-weight:bold;padding:3px 10px;border-radius:4px;'
                                        f'margin-right:6px;font-size:13px;min-width:34px;text-align:center">'
                                        f'{grade}</span>' if grade else "")
                        tag_badge = (f'<span style="display:inline-block;background:{t_color};'
                                      f'color:#1e1e2e;font-weight:bold;padding:3px 10px;border-radius:4px;'
                                      f'margin-right:6px;font-size:13px">{tag}</span>'
                                      if tag else "")
                        cards.append(f"""
                        <div style="background:#1e1e2e;border-left:4px solid {g_color};
                                    padding:12px 16px;margin:8px 0;border-radius:6px">
                          <div style="display:flex;align-items:center;margin-bottom:8px;flex-wrap:wrap">
                            {mark_badge}
                            {grade_badge}
                            {tag_badge}
                            <span style="color:#cdd6f4;font-weight:bold;font-size:16px;margin-left:4px">
                              {uma}番 {name}
                            </span>
                          </div>
                          <div style="color:#cdd6f4;font-size:15px;line-height:1.7;
                                      padding-left:4px">{comment}</div>
                        </div>
                        """)
                    ui.html("".join(cards))

    # =================================================================
    # 場所タブ + レース番号ボタンの動的描画
    # =================================================================
    def _race_num(r: dict) -> int:
        rid = r.get("race_id", "")
        try:
            return int(rid[-2:])
        except ValueError:
            return 0

    def render_place_tabs():
        """場所タブを描画 (東京/京都/新潟 …)。選択中はハイライト。"""
        place_tabs_container.clear()
        if not state["by_place"]:
            with place_tabs_container:
                ui.label("⚠️ bundle.json がありません") \
                    .classes("text-orange-400 text-sm")
            return

        places = list(state["by_place"].keys())
        current = state.get("current_place") or places[0]
        with place_tabs_container:
            with ui.row().classes("w-full no-wrap gap-1 items-end"):
                for p in places:
                    n_races = len(state["by_place"][p])
                    is_active = (p == current)
                    cls = (
                        "px-6 py-3 cursor-pointer text-center font-bold "
                        "rounded-t-lg flex-grow text-base transition-colors "
                    )
                    cls += ("bg-blue-600 text-white shadow-md"
                            if is_active else
                            "bg-slate-700 text-slate-300 hover:bg-slate-600")
                    tab = ui.element("div").classes(cls)
                    tab.on("click", lambda p=p: select_place(p))
                    with tab:
                        ui.html(
                            f'<span>{p}</span> '
                            f'<span style="font-size:13px;opacity:0.75;'
                            f'margin-left:6px">({n_races}R)</span>'
                        )

    def render_race_buttons():
        """レース番号ボタンを描画 (1R / 2R / …)。選択中はハイライト。"""
        race_buttons_container.clear()
        place = state.get("current_place")
        if not place:
            return
        races = state["by_place"].get(place, [])
        races_sorted = sorted(races, key=_race_num)
        current_rid = (state.get("race") or {}).get("race_id")
        with race_buttons_container:
            with ui.row().classes("w-full gap-1 flex-wrap"):
                for r in races_sorted:
                    rid = r.get("race_id", "")
                    r_num = rid[-2:].lstrip("0") if len(rid) >= 16 else "?"
                    is_active = (rid == current_rid)
                    cls = (
                        "min-w-[60px] px-3 py-2 cursor-pointer text-center "
                        "rounded font-bold text-base transition-colors "
                    )
                    cls += ("bg-blue-600 text-white shadow-md"
                            if is_active else
                            "bg-slate-800 text-slate-200 hover:bg-slate-700 "
                            "border border-slate-700")
                    btn = ui.element("div").classes(cls)
                    btn.on("click", lambda r=r: select_race(r))
                    with btn:
                        ui.label(f"{r_num}R")

    def select_place(p: str):
        """場所タブクリック → 場所切替 → 1R を自動選択"""
        if state.get("current_place") == p:
            return
        state["current_place"] = p
        render_place_tabs()
        # 場所が変わったら最初のレースに自動切替
        races = sorted(state["by_place"].get(p, []), key=_race_num)
        if races:
            select_race(races[0])
        else:
            state["race"] = None
            render_race_buttons()

    def select_race(r: dict):
        """レース番号ボタンクリック → render_race + ハイライト更新"""
        state["race"] = r
        render_race_buttons()
        render_race(r)

    def update_for_date(date: str):
        """開催日 select 変更 → 場所タブ群を再構築 → 最初の場所/レースを自動表示"""
        state["date"] = date
        bundle = load_bundle(date)
        state["bundle"] = bundle

        by_place: dict[str, list] = {}
        if bundle:
            for r in bundle.get("races", []):
                place = r.get("race_meta", {}).get("place", "?")
                by_place.setdefault(place, []).append(r)
        state["by_place"] = by_place
        state["current_place"] = (next(iter(by_place.keys()))
                                    if by_place else None)
        state["race"] = None

        render_place_tabs()
        render_race_buttons()

        # 直近 1 週間の好調教 Top 5 (date が変わった時のみ再計算)
        training_top5_box.clear()
        with training_top5_box:
            render_training_top5(date)

        # Cowork 累計 P/L (date には依存しないが、開催日切替時に都度更新)
        cowork_pl_box.clear()
        with cowork_pl_box:
            render_cowork_pl_chart()

        # Grade Scope (日付別: 重賞 LLM 詳細見解)
        grade_scope_box.clear()
        with grade_scope_box:
            render_grade_scope(date)

        if state["current_place"]:
            races = sorted(by_place.get(state["current_place"], []),
                            key=_race_num)
            if races:
                select_race(races[0])

    date_dd.on_value_change(lambda e: update_for_date(e.value))
    update_for_date(dates[0])


if __name__ in {"__main__", "__mp_main__"}:
    import os

    # 起動時に master を 1 回読み込んでキャッシュ (5-10 秒程度)
    print("master_v2 読み込み中... (5-10 秒)")
    df = get_master_df()
    if df is not None:
        print(f"  完了: {len(df):,} 行")
    else:
        print("  master_v2 が見つかりません (過去成績は表示されません)")

    # HF Spaces / Docker / Cloud 対応:
    # 環境変数 PORT があればそれを使う (HF Spaces は 7860)
    # host=0.0.0.0 でコンテナ外からアクセス可能に
    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "0.0.0.0" if "HF_SPACE_ID" in os.environ
                                           or "DOCKER_CONTAINER" in os.environ
                                           else "127.0.0.1")

    ui.run(
        port=port,
        host=host,
        title="🏇 PyCaLiAI (NiceGUI)",
        favicon="🏇",
        dark=True,
        reload=False,
    )
