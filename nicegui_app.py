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

import functools
import json
import re
from pathlib import Path

import pandas as pd
from nicegui import ui

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
    return sorted([p.stem for p in WEEKLY_DIR.glob("????????.csv")
                    if p.stem.isdigit() and len(p.stem) == 8], reverse=True)


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

    if isinstance(data, dict):
        data = data.get("races", [data])
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
        out[rid] = {
            "race_id":     rid,
            "race_label":  str(entry.get("race_label", "")),
            "race_nature": str(entry.get("race_nature", "")),
            "race_reason": str(entry.get("race_reason", "")),
            "bets":        bets,
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
            data = data.get("races", [data])
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
    "under": "#a6e3a1", "fair": "#89b4fa",
    "over": "#f38ba8", "unknown": "#6c7086",
}


def race_nature(rc: dict) -> str:
    top1 = rc.get("top1_dominance") or 0
    chaos = rc.get("field_chaos_score") or 0
    if chaos >= 0.92:
        return "見送り"
    if top1 >= 0.10 and chaos < 0.70:
        return "固い"
    if top1 < 0.05 and chaos >= 0.85:
        return "混戦"
    return "中堅"


def ai_comment(race: dict) -> str:
    rc = race.get("race_confidence", {}) or {}
    horses = race.get("horses", []) or []
    nature = race_nature(rc)
    top1 = rc.get("top1_dominance") or 0
    top2 = rc.get("top2_concentration") or 0
    chaos = rc.get("field_chaos_score") or 0
    market = rc.get("ai_market_agreement") or 0
    hon = next((h for h in horses if h.get("mark") == "◎"), None)
    parts = []

    if hon:
        p_win = (hon.get("p_win") or 0) * 100
        if top1 >= 0.10:
            parts.append(f"本命 {hon.get('horse_name','◎')} は他馬と明確な確率差を持つ独走候補 (勝率 {p_win:.1f}%)")
        elif top1 >= 0.05:
            parts.append(f"本命 {hon.get('horse_name','◎')} (勝率 {p_win:.1f}%) は対抗との力差はあるが油断できない")
        else:
            parts.append(f"本命 {hon.get('horse_name','◎')} と対抗の力差は紙一重 (top1_dom={top1:.3f})")

    if top2 >= 0.50:
        parts.append("上位 2 頭で決まる確率が高く、馬連狙いの好レース")
    elif top2 < 0.30:
        parts.append("上位馬の確率も分散しており、本命単独の安定には欠ける")

    if chaos >= 0.92:
        parts.append("確率分布が極めてフラットなカオス、見送り推奨")
    elif chaos >= 0.85:
        parts.append("出走馬の力差が小さい混戦模様、box やワイドが有効")
    elif chaos < 0.65:
        parts.append("上位馬の優位が明確、堅い決着が予想される")

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
# ============================================================
def _diag_top1(v: float) -> tuple[str, str, str]:
    """◎独走度 → (badge, color, action)"""
    if v >= 0.10:
        return ("◎独走", "#a6e3a1", "本命を信頼。単勝・複勝で勝負可")
    if v >= 0.05:
        return ("◎やや優位", "#89b4fa", "本命有力だが油断不可。複勝で安全に")
    if v >= 0:
        return ("拮抗", "#f9e2af", "本命の優位性は薄い。馬連で広めに")
    return ("逆転濃厚", "#f38ba8", "◎より上位がいる可能性。買い直し検討")


def _diag_top2(v: float) -> tuple[str, str, str]:
    """上位2頭集中 → (badge, color, action)"""
    if v >= 0.50:
        return ("本線濃厚", "#a6e3a1", "◎-〇 で決まる確率高。馬連本線が有効")
    if v >= 0.35:
        return ("やや本線", "#89b4fa", "馬連 ◎-〇 + 流し相手数頭が無難")
    return ("分散", "#f9e2af", "上位2頭以外も来る。流しか box で広めに")


def _diag_chaos(v: float) -> tuple[str, str, str]:
    """混戦度 → (badge, color, action)"""
    if v <= 0.65:
        return ("固い", "#a6e3a1", "上位馬の優位明確。素直に印通りで")
    if v <= 0.85:
        return ("やや固め", "#89b4fa", "標準的な信頼度。基本路線で OK")
    if v <= 0.92:
        return ("混戦", "#f9e2af", "力差小さい。box やワイドで広めに")
    return ("カオス", "#f38ba8", "確率分布フラット。見送り推奨")


def _diag_market(v: float) -> tuple[str, str, str]:
    """市場一致 → (badge, color, action)"""
    if v >= 0.7:
        return ("市場と一致", "#89b4fa", "AI ≒ 市場。妙味は薄い、見送りも視野")
    if v >= 0.3:
        return ("やや一致", "#a6e3a1", "AI と市場ほぼ同方向。素直に買える")
    if v >= 0:
        return ("ややズレ", "#f9e2af", "AI が穴推し気味。妙味あり、慎重に")
    return ("逆方向", "#f38ba8", "AI と市場が逆。波乱の可能性大、穴狙い")


def _chip_html(title: str, value_str: str, diag: tuple[str, str, str]) -> str:
    label, color, action = diag
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
                  margin-bottom:6px">{value_str}</div>
      <div style="color:#a6adc8;font-size:12px;line-height:1.4">
        → {action}
      </div>
    </div>
    """


# ============================================================
# 右パネル: メトリクス chip (大きめ) + 過去成績 + 馬場バイアス
# ============================================================
def make_right_panel_html(race: dict) -> str:
    rc = race.get("race_confidence", {}) or {}
    meta = race.get("race_meta", {}) or {}

    top1 = rc.get("top1_dominance") or 0
    top2 = rc.get("top2_concentration") or 0
    chaos = rc.get("field_chaos_score") or 0
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

    chip_top1   = _chip_html("◎独走度",     f"{top1:.3f}",   _diag_top1(top1))
    chip_top2   = _chip_html("上位2頭集中", f"{top2:.3f}",   _diag_top2(top2))
    chip_chaos  = _chip_html("混戦度",       f"{chaos:.3f}",  _diag_chaos(chaos))
    chip_market = _chip_html("市場一致",     f"{market:+.3f}",_diag_market(market))

    return f"""
    <div style="display:flex;flex-direction:column;gap:10px;height:100%">
      <div style="color:#6c7086;font-size:11px;margin-bottom:-4px;
                  padding:0 4px;letter-spacing:0.5px">
        ※ 各カードの色付きバッジは「いま何をすべきか」を一言でまとめたもの。
      </div>
      <!-- 信頼度メトリクス 4 chip (素人向け: 状態ラベル + アクション付き) -->
      <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:10px">
        {chip_top1}
        {chip_top2}
        {chip_chaos}
        {chip_market}
      </div>

      {kako_html}
      {baba_html}
    </div>
    """


# ============================================================
# 出走表 (ピュア HTML テーブル)
# ============================================================
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
        market = h.get("ai_vs_market") or "unknown"
        market_color = MARKET_COLORS.get(market, "#6c7086")

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

        rows.append(f"""
        <tr style="{bg}border-bottom:1px solid #313244;height:42px">
          <td style="padding:6px 12px;color:#cdd6f4;font-weight:bold;text-align:center">{umaban}</td>
          <td style="padding:6px 8px;text-align:center">{mark_html}</td>
          <td style="padding:6px 12px;color:#cdd6f4;font-size:16px;font-weight:600">{name}</td>
          <td style="padding:6px 10px;color:#a6adc8;font-size:13px">{jockey}</td>
          <td style="padding:6px 12px;text-align:right;color:#a6e3a1;font-weight:bold">{p_win:.1f}%</td>
          <td style="padding:6px 12px;text-align:right;color:#89b4fa">{p_sho:.1f}%</td>
          <td style="padding:6px 12px;text-align:right;color:#f5c2e7">{tan:.1f}</td>
          <td style="padding:6px 12px;text-align:right;color:{ev_color};font-weight:bold">{ev_tan:.2f}</td>
          <td style="padding:6px 12px;text-align:right;color:#a6adc8">{fuku_low:.1f}-{fuku_high:.1f}</td>
          <td style="padding:6px 12px;text-align:center">
            <span style="background:{market_color};color:#1e1e2e;padding:2px 10px;
                         border-radius:10px;font-size:12px;font-weight:bold">{market}</span>
          </td>
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
          <th style="padding:12px;color:#f39c12;font-size:14px;text-align:left">騎手</th>
          <th style="padding:12px;color:#f39c12;font-size:14px;text-align:right">勝率</th>
          <th style="padding:12px;color:#f39c12;font-size:14px;text-align:right">複勝率</th>
          <th style="padding:12px;color:#f39c12;font-size:14px;text-align:right">単勝</th>
          <th style="padding:12px;color:#f39c12;font-size:14px;text-align:right">単勝EV</th>
          <th style="padding:12px;color:#f39c12;font-size:14px;text-align:right">複勝</th>
          <th style="padding:12px;color:#f39c12;font-size:14px;text-align:center">vs市場</th>
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
    """展開予想 (脚質分布 + pace + 一言)。Cowork の買い目スタイル風の文章で。
    各馬にコース適性 (A/B/C/D) も付与。
    """
    horses = list(race.get("horses", []))
    if not horses:
        return
    meta = race.get("race_meta", {}) or {}
    field_size = meta.get("field_size", 16) or 16
    place = meta.get("place", "")
    course_str = meta.get("course", "")

    # コース別好走脚質をロード (適性スコア計算用)
    course_stats = compute_course_stats_v2(place, course_str)
    kyaku_winrate: dict[str, float] = {}
    if course_stats and course_stats.get("kyaku"):
        for entry in course_stats["kyaku"]:
            kyaku_winrate[entry["label"]] = entry.get("win_rate", 0.0)

    def aptitude_grade(kyaku: str) -> tuple[str, str, str]:
        """コースのその脚質の勝率を相対化して A/B/C/D 評価を返す"""
        if kyaku == "不明" or not kyaku_winrate:
            return ("-", "#6c7086", "")
        rates = sorted(kyaku_winrate.values(), reverse=True)
        rate = kyaku_winrate.get(kyaku, 0.0)
        if not rates or rate == 0:
            return ("-", "#6c7086", "")
        # 4 段階 (rank 0 が最良)
        if rate >= rates[0] - 0.5:
            return ("A", "#a6e3a1", "(好相性)")
        if rate >= rates[min(1, len(rates)-1)] - 0.5:
            return ("B", "#89b4fa", "")
        if rate >= rates[min(2, len(rates)-1)] - 0.5:
            return ("C", "#f9e2af", "")
        return ("D", "#f38ba8", "(苦戦)")

    # 上位 5 頭 (p_win 順) で脚質を集計
    top5 = sorted(horses, key=lambda h: -(h.get("p_win") or 0))[:5]

    kyaku_records: list[dict] = []
    weekly_df = load_weekly_horses(date_str) if date_str else None
    rid_16 = str(race.get("race_id", ""))[:16]

    for h in top5:
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
        grade, grade_color, grade_note = aptitude_grade(kyaku)
        kyaku_records.append({"horse": h, "kyaku": kyaku,
                                "grade": grade, "grade_color": grade_color,
                                "grade_note": grade_note})

    # 集計
    counts: dict[str, int] = {"逃げ":0, "先行":0, "差し":0, "追込":0, "不明":0}
    for r in kyaku_records:
        counts[r["kyaku"]] = counts.get(r["kyaku"], 0) + 1

    # ペース判定 (簡易)
    n_nige = counts.get("逃げ", 0)
    n_sen  = counts.get("先行", 0)
    n_sashi = counts.get("差し", 0) + counts.get("追込", 0)
    if n_nige >= 2:
        pace = "ハイペース"
        pace_color = "#f38ba8"
        pace_msg = "逃げ馬複数 → 前崩れ気配、差し・追込有利"
    elif n_nige == 0 and n_sen >= 3:
        pace = "スローペース"
        pace_color = "#89b4fa"
        pace_msg = "明確な逃げ馬不在 + 先行多 → スロー濃厚、上り勝負・先行有利"
    elif n_sashi >= 3:
        pace = "差し決着"
        pace_color = "#cba6f7"
        pace_msg = "上位に差し型多数 → 上り3F が決まる末脚勝負"
    else:
        pace = "平均ペース"
        pace_color = "#a6e3a1"
        pace_msg = "脚質バランス均等 → 標準的な流れ、ポジション戦"

    # 主要馬の一言コメント (各馬: 印 / 馬番 / 馬名 / 脚質 / 適性 / 勝率)
    horse_lines = []
    for rec in kyaku_records:
        h = rec["horse"]
        k = rec["kyaku"]
        grade = rec.get("grade", "-")
        gcol = rec.get("grade_color", "#6c7086")
        gnote = rec.get("grade_note", "")
        mark = h.get("mark") or "△"
        name = h.get("horse_name", "?")
        umaban = h.get("umaban", "?")
        pwin = (h.get("p_win") or 0) * 100
        mark_color = MARK_COLORS.get(mark, "#6c7086")
        grade_html = (
            f'<span style="background:{gcol};color:#1e1e2e;padding:3px 10px;'
            f'border-radius:10px;font-weight:bold;font-size:14px;'
            f'min-width:28px;text-align:center">適性 {grade}</span>'
            f'<span style="color:{gcol};font-size:12px;margin-left:4px">{gnote}</span>'
            if grade != "-" else
            '<span style="color:#6c7086;font-size:12px">適性 −</span>'
        )
        horse_lines.append(f"""
        <div style="display:flex;align-items:center;gap:10px;padding:8px 0">
          <span style="background:{mark_color};color:#fff;width:32px;height:32px;
                       line-height:32px;text-align:center;border-radius:50%;
                       font-size:16px;font-weight:bold;flex-shrink:0">{mark}</span>
          <span style="color:#6c7086;font-size:15px;width:42px">{umaban}番</span>
          <span style="color:#cdd6f4;font-size:17px;flex-grow:1;font-weight:600">
            {name}</span>
          <span style="color:#fab387;font-size:15px;font-weight:bold;
                       background:rgba(250,179,135,0.15);padding:4px 14px;
                       border-radius:12px">{k}</span>
          {grade_html}
          <span style="color:#a6e3a1;font-size:16px;font-weight:bold;
                       width:72px;text-align:right">{pwin:.1f}%</span>
        </div>
        """)

    ui.html(f"""
    <div style="background:linear-gradient(135deg,#0d1421 0%,#16213e 100%);
                border:1px solid {pace_color};border-radius:14px;padding:22px 26px;
                margin-bottom:18px">
      <h2 style="margin:0 0 10px 0;color:#cdd6f4;font-size:24px;font-weight:bold">
        🏇 展開予想
      </h2>
      <div style="margin-bottom:18px">
        <span style="background:{pace_color};color:#1e1e2e;padding:8px 18px;
                     border-radius:14px;font-weight:bold;font-size:18px">{pace}</span>
        <span style="color:#cdd6f4;font-size:16px;margin-left:14px">{pace_msg}</span>
      </div>
      <div style="background:rgba(0,0,0,0.25);border-left:3px solid {pace_color};
                  padding:12px 18px;border-radius:6px">
        <div style="color:#6c7086;font-size:14px;margin-bottom:8px">
          上位 5 頭の想定脚質 (前4角通過位置ベース)
        </div>
        {"".join(horse_lines)}
      </div>
      <div style="margin-top:14px;color:#a6adc8;font-size:14px;line-height:1.5">
        脚質分布: 逃げ {counts['逃げ']} / 先行 {counts['先行']} /
        差し {counts['差し']} / 追込 {counts['追込']}
        {'(不明 ' + str(counts['不明']) + ')' if counts['不明'] > 0 else ''}
      </div>
    </div>
    """)


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

        # ── メイン: 左右パネル ──
        with ui.row().classes("w-full no-wrap gap-3"):
            left_box = ui.element("div").classes("flex-grow").style("flex: 3")
            right_box = ui.element("div").classes("flex-grow").style("flex: 2")

        # ── メイン: タブ (出走表 / 全頭分析 / コース分析 / Cowork) ──
        with ui.tabs().classes("w-full") as tabs:
            tab_shutsuba = ui.tab("📋 出走表")
            tab_bunseki = ui.tab("🔍 全頭分析")
            tab_course = ui.tab("📊 コース分析")
            tab_bets = ui.tab("🎫 Cowork 買い目")

        with ui.tab_panels(tabs, value=tab_shutsuba).classes("w-full"):
            with ui.tab_panel(tab_shutsuba):
                shutsuba_box = ui.element("div").classes("w-full")
            with ui.tab_panel(tab_bunseki):
                bunseki_box = ui.element("div").classes("w-full")
            with ui.tab_panel(tab_course):
                course_box = ui.element("div").classes("w-full")
            with ui.tab_panel(tab_bets):
                bets_box = ui.element("div").classes("w-full")

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
