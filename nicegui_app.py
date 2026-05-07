"""
nicegui_app.py
==============
NiceGUI 版 PyCaLiAI (実験 MVP v6)

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
MASTER_CSV        = BASE / "data" / "master_v2_20130105-20251228.csv"


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
# Master CSV 読み込み (起動時 1 回、コース統計用)
# ============================================================
@functools.lru_cache(maxsize=1)
def get_master_df() -> pd.DataFrame | None:
    """master_v2 から必要列だけ読み込み、メモリ最適化してキャッシュ。"""
    if not MASTER_CSV.exists():
        return None
    try:
        # 必要列のみ + 型最適化
        usecols = ["日付", "場所", "Ｒ", "枠番", "馬番", "着順",
                    "芝・ダ", "距離", "馬場状態"]
        df = pd.read_csv(
            MASTER_CSV, encoding="utf-8-sig",
            usecols=usecols, dtype=str, low_memory=False, on_bad_lines="skip",
        )
        df["日付"] = pd.to_numeric(df["日付"], errors="coerce")
        df["枠番"] = pd.to_numeric(df["枠番"], errors="coerce")
        df["馬番"] = pd.to_numeric(df["馬番"], errors="coerce")
        df["着順"] = pd.to_numeric(df["着順"], errors="coerce")
        df["距離"] = pd.to_numeric(df["距離"], errors="coerce")
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
def make_shutsuba_table_html(race: dict) -> str:
    horses = race.get("horses", [])
    horses_sorted = sorted(horses, key=lambda h: h.get("umaban") or 0)

    rows = []
    for h in horses_sorted:
        mark = h.get("mark") or ""
        umaban = h.get("umaban") or "?"
        name = h.get("horse_name") or "-"
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
# PyCaLi 出走馬評価リスト (Streamlit 互換、bundle.json データから構築)
# ============================================================
# 注: Streamlit 版は前走補正タイム / 調教 / 過去5走履歴を持つが、HF 版の
#     bundle.json には p_win/p_plc/p_sho/odds しか入っていないため、
#     ここでは「6 軸を bundle.json 内で完結する形」に再設計した。
#     (data/master_v2 や data/training の依存を排除して HF にも載せる)
PYCA_AXES = [
    # (key, label, 説明)
    ("a", "総合力",   "AI が予測する勝率"),
    ("b", "連対力",   "AI が予測する 2 着以内率"),
    ("c", "圏内力",   "AI が予測する 3 着以内率"),
    ("d", "人気度",   "市場の評価 (1/単勝オッズ)"),
    ("e", "単勝妙味", "AI 予測 × 単勝オッズ (≥1.0 が妙味)"),
    ("f", "複勝妙味", "AI 予測 × 複勝オッズ中央値"),
]


def _norm_0_10_list(values: list[float]) -> tuple[list[float], bool]:
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


def compute_pyca_features(horses: list[dict]) -> dict:
    """各馬の指標値を計算してレース内 0〜10 に正規化。

    返り値: {
      'norm':  {key -> [v0, v1, ...]},      # 0-10 正規化値
      'rank':  {key -> [r0, r1, ...]},      # レース内順位 (1=最良)
      'valid': {key -> bool},               # その軸が有効サンプルを持つか
      'pyca':  [s0, s1, ...],               # PyCaLi 指数 (0-100)
      'pyca_rank': [r0, r1, ...],
    }
    """
    p_win = [(h.get("p_win") or 0) for h in horses]
    p_plc = [(h.get("p_plc") or 0) for h in horses]
    p_sho = [(h.get("p_sho") or 0) for h in horses]
    tan   = [(h.get("tansho_odds") or 0) or 99 for h in horses]
    flow  = [(h.get("fuku_odds_low")  or 0) for h in horses]
    fhi   = [(h.get("fuku_odds_high") or 0) for h in horses]
    fmid  = [(a + b) / 2 if (a and b) else 0 for a, b in zip(flow, fhi)]
    pop   = [(1.0 / t) if t > 0 else 0 for t in tan]
    ev_t  = [pw * t for pw, t in zip(p_win, tan)]
    ev_f  = [ps * fm for ps, fm in zip(p_sho, fmid)]

    raw = {
        "a": p_win, "b": p_plc, "c": p_sho,
        "d": pop,   "e": ev_t,  "f": ev_f,
    }
    norm: dict[str, list[float]] = {}
    rank: dict[str, list[int]] = {}
    valid: dict[str, bool] = {}
    for k, vs in raw.items():
        n, ok = _norm_0_10_list(vs)
        norm[k] = n
        rank[k] = _rank_list(vs)
        valid[k] = ok

    # PyCaLi 指数: 勝率 * 100 をベースに、連対率/複勝率/妙味の正規化平均で微調整
    n = len(horses)
    pyca = []
    for i in range(n):
        base = p_win[i] * 100.0
        boost_avg = (norm["b"][i] + norm["c"][i] + norm["e"][i]) / 3.0
        boost = boost_avg - 5.0      # -5..+5
        score = max(0.0, min(100.0, base + boost * 0.6))
        pyca.append(score)
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


def render_pyca_eval_list(race: dict) -> None:
    """Streamlit 版互換の評価リスト (前提: ui コンテナの中で呼ばれる)。"""
    horses = list(race.get("horses", []))
    if not horses:
        return
    feats = compute_pyca_features(horses)
    order = sorted(range(len(horses)), key=lambda i: -feats["pyca"][i])
    labels = [lbl for _, lbl, _ in PYCA_AXES]

    ui.label("🔍 出走馬評価リスト (PyCaLi指数)").classes(
        "text-xl font-bold mt-6 mb-1")
    ui.label(
        "PyCaLi指数 = AI 予測勝率を基準に、連対率/複勝率/妙味の補正を加えた"
        "総合スコア (0-100)。右側 a〜f は指数算出に使われた特徴量"
        "(レース内 0〜10 正規化、★は上位 3 位以内)。"
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
        ui.label("NiceGUI 版 (実験 MVP v6)").classes("text-base text-slate-400")

    state = {"date": None, "race": None, "bundle": None}

    with ui.row().classes("w-full no-wrap gap-4 p-4"):
        # サイドバー
        with ui.column().classes("w-72 gap-2 flex-shrink-0"):
            ui.label("📅 開催日").classes("text-xl font-bold text-slate-200")
            dates = list_dates()
            if not dates:
                ui.label("⚠️ data/weekly/ に CSV がありません").classes("text-orange-400")
                return
            date_dd = ui.select(dates, value=dates[0]).classes("w-full")
            ui.label("🏇 場所").classes("text-xl font-bold text-slate-200 mt-3")
            race_list_box = ui.column().classes("w-full gap-1")

        # メイン
        with ui.column().classes("flex-grow gap-3"):
            with ui.row().classes("w-full no-wrap gap-3"):
                left_box = ui.element("div").classes("flex-grow").style("flex: 3")
                right_box = ui.element("div").classes("flex-grow").style("flex: 2")

            with ui.tabs().classes("w-full") as tabs:
                tab_shutsuba = ui.tab("📋 出走表")
                tab_bunseki = ui.tab("🔍 全頭分析")
                tab_bets = ui.tab("🎫 Cowork 買い目")

            with ui.tab_panels(tabs, value=tab_shutsuba).classes("w-full"):
                with ui.tab_panel(tab_shutsuba):
                    shutsuba_box = ui.element("div").classes("w-full")
                with ui.tab_panel(tab_bunseki):
                    bunseki_box = ui.element("div").classes("w-full")
                with ui.tab_panel(tab_bets):
                    bets_box = ui.element("div").classes("w-full")

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
            ui.html(make_shutsuba_table_html(race))

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

            # Streamlit 版互換の評価リスト
            render_pyca_eval_list(race)

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

    def update_race_list(date: str):
        state["date"] = date
        bundle = load_bundle(date)
        state["bundle"] = bundle
        race_list_box.clear()
        if not bundle:
            with race_list_box:
                ui.label("⚠️ bundle.json がありません").classes("text-orange-400 text-sm")
                ui.label(f".\\weekly_cowork.ps1 {date} v5") \
                  .classes("font-mono text-sm text-slate-400")
            return
        races = bundle.get("races", [])
        by_place: dict[str, list] = {}
        for r in races:
            place = r.get("race_meta", {}).get("place", "?")
            by_place.setdefault(place, []).append(r)
        with race_list_box:
            for i, (place, race_list) in enumerate(by_place.items()):
                with ui.expansion(f"{place} ({len(race_list)}R)", icon="place",
                                   value=(i == 0)).classes("w-full"):
                    for r in race_list:
                        rid = r.get("race_id", "")
                        r_num = rid[-2:].lstrip("0") if len(rid) >= 16 else "?"
                        meta = r.get("race_meta", {})
                        cls = meta.get("class", "") or "-"
                        hon = next((h for h in r.get("horses", [])
                                    if h.get("mark") == "◎"), None)
                        hon_name = hon.get("horse_name", "-") if hon else "-"
                        item = ui.element("div").classes(
                            "w-full px-3 py-2 cursor-pointer rounded "
                            "hover:bg-slate-800 transition-colors"
                        )
                        item.on("click", lambda r=r: render_race(r))
                        with item:
                            ui.html(f"""
                            <div style="display:flex;flex-direction:column;gap:3px">
                              <div style="display:flex;align-items:baseline;gap:8px">
                                <span style="color:#cdd6f4;font-size:16px;font-weight:bold;
                                             min-width:42px">{r_num}R</span>
                                <span style="color:#a6adc8;font-size:14px">{cls}</span>
                              </div>
                              <div style="color:#a6e3a1;font-size:14px;padding-left:50px">
                                ◎ {hon_name}
                              </div>
                            </div>
                            """)

    date_dd.on_value_change(lambda e: update_race_list(e.value))
    update_race_list(dates[0])


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
