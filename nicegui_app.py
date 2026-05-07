"""
nicegui_app.py
==============
NiceGUI 版 PyCaLiAI (実験 MVP v3)

v2 → v3 変更点:
  1. 出走表: row 36px + 全 14-18 頭が画面内に収まるよう調整
  2. 全頭分析: レーダーチャート (個別馬) + 散布図 (全頭の AI vs 市場 2D 配置)
  3. サイドバー: 場所ごと expansion でプルダウン化 (36R 一覧の下スクロール解消)
  4. レース概要: AI 評価コメント自動生成 (race_confidence から自然文)
  5. confidence chip 凡例: 各メトリクスの意味を表示
  6. フォント: 16→18 に再拡大
"""
from __future__ import annotations

import json
from pathlib import Path

from nicegui import ui

BASE = Path(__file__).parent
COWORK_INPUT_DIR = BASE / "reports" / "cowork_input"
COWORK_BETS_DIR  = BASE / "reports" / "cowork_bets"
WEEKLY_DIR       = BASE / "data" / "weekly"


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
    p = COWORK_BETS_DIR / date_str / f"{race_id}.json"
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ============================================================
# UI ヘルパー
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
    """race_confidence から AI 評価コメントを自然文で生成"""
    rc = race.get("race_confidence", {}) or {}
    horses = race.get("horses", []) or []
    nature = race_nature(rc)
    top1 = rc.get("top1_dominance") or 0
    top2 = rc.get("top2_concentration") or 0
    chaos = rc.get("field_chaos_score") or 0
    market = rc.get("ai_market_agreement") or 0

    hon = next((h for h in horses if h.get("mark") == "◎"), None)
    tai = next((h for h in horses if h.get("mark") == "〇"), None)
    parts = []

    # ◎ の信頼度
    if hon:
        p_win = (hon.get("p_win") or 0) * 100
        odds = hon.get("tansho_odds") or 0
        if top1 >= 0.10:
            parts.append(f"本命 {hon.get('horse_name','◎')} は他馬と明確な確率差を持つ独走候補 (勝率 {p_win:.1f}%)")
        elif top1 >= 0.05:
            parts.append(f"本命 {hon.get('horse_name','◎')} (勝率 {p_win:.1f}%) は対抗との力差はあるが油断できない")
        else:
            parts.append(f"本命 {hon.get('horse_name','◎')} と対抗の力差は紙一重 (top1_dom={top1:.3f})")

    # 上位集中度
    if top2 >= 0.50:
        parts.append("上位 2 頭で決まる確率が高く、馬連狙いの好レース")
    elif top2 < 0.30:
        parts.append("上位馬の確率も分散しており、本命単独の安定には欠ける")

    # 混戦度
    if chaos >= 0.92:
        parts.append("確率分布が極めてフラットなカオス、見送り推奨")
    elif chaos >= 0.85:
        parts.append("出走馬の力差が小さい混戦模様、box やワイドが有効")
    elif chaos < 0.65:
        parts.append("上位馬の優位が明確、堅い決着が予想される")

    # 市場一致度
    if market > 0.7:
        parts.append("AI 予想と市場オッズ順がほぼ一致、配当妙味は控えめ")
    elif market < 0:
        parts.append("AI と市場の見方が大きく食い違い、波乱の可能性大")
    elif market < 0.3:
        parts.append("AI と市場の評価にズレがあり、AI が穴推しの可能性")

    if not parts:
        parts.append("特筆すべき特徴のない standard なレース")

    return f"性質判定: <b style='color:{NATURE_COLORS.get(nature,'#89b4fa')}'>{nature}</b>。" + "。".join(parts) + "。"


def make_banner_html(race: dict) -> str:
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
        <div style="display:flex;align-items:center;gap:14px;padding:6px 0">
          <span style="background:{color};color:#fff;font-size:20px;font-weight:bold;
                       width:36px;height:36px;line-height:36px;text-align:center;
                       border-radius:50%;flex-shrink:0">{mark}</span>
          <span style="color:#888;font-size:15px;width:36px">{h.get("umaban","?")}番</span>
          <span style="color:#cdd6f4;font-size:19px;font-weight:600;flex-grow:1">
            {h.get("horse_name","-")}</span>
          <span style="color:#a6e3a1;font-size:17px;font-weight:bold;
                       background:rgba(166,227,161,0.12);padding:4px 14px;
                       border-radius:12px">{score:.1f}%</span>
          <span style="color:#fab387;font-size:16px;background:rgba(250,179,135,0.12);
                       padding:4px 14px;border-radius:10px">単勝 {odds_str}</span>
        </div>
        """

    return f"""
    <div style="
      background:linear-gradient(135deg,#0d1421 0%,#16213e 50%,#1a2845 100%);
      border:1px solid #f39c12;border-radius:14px;
      padding:20px 26px;margin-bottom:12px;position:relative;overflow:hidden;
      box-shadow:0 4px 20px rgba(243,156,18,0.15);width:100%">
      <div style="position:absolute;top:0;left:0;right:0;height:4px;
                  background:linear-gradient(90deg,#e74c3c,#f39c12,#e74c3c)"></div>

      <div style="display:flex;flex-wrap:wrap;align-items:center;gap:12px;
                  margin-bottom:10px;font-size:16px">
        <span style="background:{grade_color};color:#fff;padding:4px 14px;
                     border-radius:4px;font-size:15px;font-weight:bold">{cls}</span>
        <span style="background:{nat_color};color:#1e1e2e;padding:4px 14px;
                     border-radius:4px;font-size:15px;font-weight:bold">{nature}</span>
        <span style="color:#cdd6f4;font-weight:600">{place} {r_num}R</span>
        <span style="color:#888">|</span>
        <span style="color:#f5c2e7">{course}</span>
        <span style="color:#888">|</span>
        <span style="color:#f5c2e7">{field_size}頭</span>
      </div>

      <h2 style="font-size:34px;font-weight:900;color:#cdd6f4;margin:0 0 14px 0;
                 letter-spacing:1.5px;line-height:1.1">{race_name}</h2>

      <div style="background:rgba(0,0,0,0.25);border-left:3px solid #f39c12;
                  padding:8px 18px;border-radius:6px">
        {mark_row("◎", "#e74c3c", hon)}
        {mark_row("〇", "#3498db", tai)}
        {mark_row("▲", "#9b59b6", san)}
      </div>
    </div>
    """


def make_ai_eval_html(race: dict) -> str:
    """AI 評価コメントを表示"""
    return f"""
    <div style="background:#1e1e2e;border-left:3px solid #89b4fa;
                padding:14px 18px;margin-bottom:12px;border-radius:8px">
      <div style="color:#89b4fa;font-size:14px;font-weight:bold;margin-bottom:6px">
        🤖 AI 評価
      </div>
      <div style="color:#cdd6f4;font-size:16px;line-height:1.7">
        {ai_comment(race)}
      </div>
    </div>
    """


def make_confidence_html(rc: dict) -> str:
    top1 = rc.get("top1_dominance") or 0
    top2 = rc.get("top2_concentration") or 0
    chaos = rc.get("field_chaos_score") or 0
    market = rc.get("ai_market_agreement") or 0

    def chip(label, value, hint, color="#cdd6f4"):
        return f"""
        <div style="background:#1e1e2e;border:1px solid #313244;border-radius:8px;
                    padding:10px 14px;flex:1">
          <div style="color:#6c7086;font-size:13px;margin-bottom:4px">{label}</div>
          <div style="color:{color};font-size:22px;font-weight:bold;margin-bottom:6px">
            {value}</div>
          <div style="color:#888;font-size:11px;line-height:1.4">{hint}</div>
        </div>
        """

    return f"""
    <div style="display:flex;gap:10px;margin-bottom:12px">
      {chip("◎独走度", f"{top1:.3f}",
             "0.10+ で独走、0.05- で拮抗。値=◎の確率-〇の確率")}
      {chip("上位2頭集中", f"{top2:.3f}",
             "◎+〇 の合計勝率。0.5+ で馬連本線が有効")}
      {chip("混戦度", f"{chaos:.3f}",
             "確率分布のエントロピー。0.92+で見送り、0.65-で固い")}
      {chip("市場一致", f"{market:+.3f}",
             "AI 順位 vs 単勝オッズ順位。+1=完全一致、-1=完全逆")}
    </div>
    """


# ============================================================
# 全頭分析: 散布図 + レーダーチャート
# ============================================================
def race_scatter_option(horses: list) -> dict:
    """全頭の AI vs 市場 散布図"""
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
            "name": name,
            "symbol": "circle",
            "symbolSize": size,
            "itemStyle": {
                "color": color,
                "borderColor": "#fff" if mark in ["◎", "〇", "▲"] else "transparent",
                "borderWidth": 1,
            },
            "label": {
                "show": True,
                "position": "top" if mark in ["◎", "〇", "▲"] else "right",
                "color": "#cdd6f4",
                "fontSize": 12 if mark in ["◎", "〇", "▲"] else 10,
                "formatter": (mark + " " + name) if mark else name,
            },
        })

    # 等値線 (AI = 市場) 用ダミー
    max_val = max((d["value"][0] for d in data), default=50) * 1.1
    max_val = max(max_val, max((d["value"][1] for d in data), default=50) * 1.1)

    return {
        "title": {
            "text": "AI 評価 vs 市場評価",
            "textStyle": {"color": "#cdd6f4", "fontSize": 16},
            "left": "center",
        },
        "xAxis": {
            "name": "市場評価 (1/単勝オッズ × 100, 大=人気)",
            "type": "value", "min": 0, "max": round(max_val, 1),
            "splitLine": {"lineStyle": {"color": "#313244"}},
            "axisLabel": {"color": "#a6adc8"},
            "nameTextStyle": {"color": "#cdd6f4", "fontSize": 13, "padding": [10, 0, 0, 0]},
            "nameLocation": "middle", "nameGap": 35,
        },
        "yAxis": {
            "name": "AI 予想 (勝率 %)",
            "type": "value", "min": 0, "max": round(max_val, 1),
            "splitLine": {"lineStyle": {"color": "#313244"}},
            "axisLabel": {"color": "#a6adc8"},
            "nameTextStyle": {"color": "#cdd6f4", "fontSize": 13},
            "nameLocation": "middle", "nameGap": 50,
        },
        "tooltip": {
            "trigger": "item",
            "backgroundColor": "#1e1e2e",
            "borderColor": "#313244",
            "textStyle": {"color": "#cdd6f4"},
        },
        "series": [
            {"type": "scatter", "data": data, "z": 2},
            {  # 等値線 (y=x)
                "type": "line",
                "data": [[0, 0], [max_val, max_val]],
                "lineStyle": {"type": "dashed", "color": "#555", "width": 1},
                "symbol": "none",
                "tooltip": {"show": False},
                "z": 1,
                "silent": True,
            },
        ],
        "grid": {"left": 70, "right": 30, "top": 50, "bottom": 60},
        "backgroundColor": "transparent",
    }


def horse_radar_option(h: dict) -> dict:
    """1 馬のレーダーチャート (5 軸)"""
    p_win = (h.get("p_win") or 0) * 100
    p_sho = (h.get("p_sho") or 0) * 100
    tan = h.get("tansho_odds") or 99
    fuku_low = h.get("fuku_odds_low") or 0
    fuku_high = h.get("fuku_odds_high") or 0
    fuku_mid = (fuku_low + fuku_high) / 2 if fuku_low and fuku_high else 0
    ev_tan = (h.get("p_win") or 0) * tan
    ev_fuku = (h.get("p_sho") or 0) * fuku_mid
    pop = (1 / tan) * 100 if tan else 0  # 0-100、市場人気度

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
            "splitArea": {
                "show": True,
                "areaStyle": {"color": ["rgba(255,255,255,0.02)", "rgba(255,255,255,0.04)"]},
            },
            "splitLine": {"lineStyle": {"color": "#313244"}},
            "axisLine": {"lineStyle": {"color": "#313244"}},
            "axisName": {"color": "#a6adc8", "fontSize": 12},
            "center": ["50%", "55%"],
            "radius": "60%",
        },
        "series": [{
            "type": "radar",
            "data": [{
                "value": [
                    round(p_win, 1),
                    round(p_sho, 1),
                    round(ev_tan, 2),
                    round(ev_fuku, 2),
                    round(pop, 1),
                ],
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
      /* AGGrid 全体のフォント拡大 */
      .ag-theme-balham-dark { font-size: 16px !important; }
      .ag-theme-balham-dark .ag-cell {
        line-height: 36px !important;
        padding-left: 14px !important;
        padding-right: 14px !important;
      }
      .ag-header-cell-label {
        font-size: 15px !important; font-weight: bold !important;
      }
      .ag-theme-balham-dark .ag-row { height: 36px !important; }
      .ag-theme-balham-dark { --ag-row-height: 36px !important; }
      /* タブ */
      .q-tab__label { font-size: 17px !important; }
      /* セレクト */
      .q-field__native { font-size: 17px !important; }
      /* expansion */
      .q-expansion-item__container { font-size: 16px; }
    </style>
    """)

    with ui.header(elevated=True).classes("bg-slate-900"):
        ui.label("🏇 PyCaLiAI").classes("text-3xl font-bold text-white")
        ui.space()
        ui.label("NiceGUI 版 (実験 MVP v3)").classes("text-base text-slate-400")

    state = {"date": None, "race": None, "bundle": None}

    with ui.row().classes("w-full no-wrap gap-4 p-4"):
        # ===== サイドバー =====
        with ui.column().classes("w-72 gap-2"):
            ui.label("📅 開催日").classes("text-xl font-bold text-slate-200")
            dates = list_dates()
            if not dates:
                ui.label("⚠️ data/weekly/ に CSV がありません").classes("text-orange-400")
                return
            date_dd = ui.select(dates, value=dates[0]).classes("w-full")

            ui.label("🏇 場所").classes("text-xl font-bold text-slate-200 mt-3")
            race_list_box = ui.column().classes("w-full gap-1")

        # ===== メイン =====
        with ui.column().classes("flex-grow gap-2"):
            banner_box = ui.element("div").classes("w-full")
            ai_eval_box = ui.element("div").classes("w-full")
            confidence_box = ui.element("div").classes("w-full")

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

        banner_box.clear()
        with banner_box:
            ui.html(make_banner_html(race))

        ai_eval_box.clear()
        with ai_eval_box:
            ui.html(make_ai_eval_html(race))

        confidence_box.clear()
        with confidence_box:
            rc = race.get("race_confidence", {}) or {}
            ui.html(make_confidence_html(rc))

        # === 出走表 ===
        shutsuba_box.clear()
        with shutsuba_box:
            horses = race.get("horses", [])
            row_data = []
            for h in horses:
                p_win = (h.get("p_win") or 0) * 100
                p_sho = (h.get("p_sho") or 0) * 100
                tan = h.get("tansho_odds")
                ev_tan = (h.get("p_win") or 0) * tan if tan else None
                row_data.append({
                    "番": h.get("umaban"),
                    "印": h.get("mark") or "",
                    "馬名": h.get("horse_name"),
                    "勝率(%)": round(p_win, 1),
                    "複勝率(%)": round(p_sho, 1),
                    "単勝": tan,
                    "単勝EV": round(ev_tan, 2) if ev_tan else None,
                    "複勝下": h.get("fuku_odds_low"),
                    "複勝上": h.get("fuku_odds_high"),
                    "vs市場": h.get("ai_vs_market"),
                })
            ui.aggrid({
                "columnDefs": [
                    {"field": "番", "width": 70, "sortable": True, "pinned": "left"},
                    {"field": "印", "width": 70, "sortable": True},
                    {"field": "馬名", "minWidth": 200, "flex": 1, "sortable": True},
                    {"field": "勝率(%)", "width": 100, "sortable": True,
                     "cellStyle": {"textAlign": "right"}},
                    {"field": "複勝率(%)", "width": 100, "sortable": True,
                     "cellStyle": {"textAlign": "right"}},
                    {"field": "単勝", "width": 90, "sortable": True,
                     "cellStyle": {"textAlign": "right"}},
                    {"field": "単勝EV", "width": 95, "sortable": True,
                     "cellStyle": {"textAlign": "right",
                                   "fontWeight": "bold", "color": "#f9e2af"}},
                    {"field": "複勝下", "width": 85, "sortable": True,
                     "cellStyle": {"textAlign": "right"}},
                    {"field": "複勝上", "width": 85, "sortable": True,
                     "cellStyle": {"textAlign": "right"}},
                    {"field": "vs市場", "width": 100, "sortable": True},
                ],
                "rowData": row_data,
                "defaultColDef": {"resizable": True},
                "domLayout": "autoHeight",
                "rowHeight": 36,
                "headerHeight": 38,
                "suppressHorizontalScroll": False,
            }).classes("w-full").style("--ag-row-height: 36px")

        # === 全頭分析 ===
        bunseki_box.clear()
        with bunseki_box:
            horses_sorted = sorted(race.get("horses", []),
                                    key=lambda h: -(h.get("p_win") or 0))

            # 散布図 (全頭の AI vs 市場 2D 配置)
            ui.label("📍 全頭ポジショニング (AI vs 市場)").classes("text-xl font-bold mt-2 mb-1")
            ui.label("対角線より上 = AI が市場より高評価 (穴推奨)。" +
                     "下 = AI が市場より低評価 (危険)。点が大きい順 ◎ > 〇 > ▲ > △.").classes(
                "text-sm text-slate-400 mb-3")
            ui.echart(race_scatter_option(horses_sorted)).classes(
                "w-full").style("height: 480px")

            # 各馬レーダーチャート (3 列グリッド)
            ui.label("🎯 出走馬レーダーチャート (勝率順)").classes(
                "text-xl font-bold mt-6 mb-1")
            ui.label("5 軸: 勝率 / 複勝率 / 単勝EV / 複勝EV / 人気度").classes(
                "text-sm text-slate-400 mb-3")
            with ui.grid(columns=3).classes("w-full gap-3"):
                for h in horses_sorted:
                    name = h.get("horse_name", "?")
                    umaban = h.get("umaban", "?")
                    mark = h.get("mark") or ""
                    p_win = (h.get("p_win") or 0) * 100
                    tan = h.get("tansho_odds") or 0
                    mark_color = MARK_COLORS.get(mark, "#6c7086")
                    mark_html = (
                        f'<span style="background:{mark_color};color:#fff;font-size:18px;'
                        f'font-weight:bold;width:32px;height:32px;line-height:32px;'
                        f'text-align:center;border-radius:50%;display:inline-block;'
                        f'margin-right:8px">{mark}</span>'
                    ) if mark else '<span style="display:inline-block;width:40px"></span>'

                    with ui.element("div").classes(
                        "p-3 bg-slate-900 rounded-lg border border-slate-700"
                    ):
                        ui.html(f"""
                        <div style="display:flex;align-items:center;margin-bottom:6px">
                          {mark_html}
                          <span style="color:#888;font-size:14px;margin-right:8px">
                            {umaban}番</span>
                          <span style="color:#cdd6f4;font-size:17px;font-weight:bold;
                                       flex-grow:1">{name}</span>
                          <span style="color:#a6e3a1;font-size:14px;font-weight:bold">
                            {p_win:.1f}%</span>
                        </div>
                        <div style="color:#888;font-size:12px;margin-bottom:4px">
                          単勝 {tan:.1f}倍
                        </div>
                        """)
                        ui.echart(horse_radar_option(h)).classes(
                            "w-full").style("height: 240px")

        # === Cowork 買い目 ===
        bets_box.clear()
        with bets_box:
            cowork = load_cowork_bets(state["date"], race["race_id"])
            if not cowork:
                ui.label("Cowork 買い目はまだ保存されていません。") \
                  .classes("text-slate-400 p-4 text-base")
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
                    ui.label("→ 見送り (購入なし)") \
                      .classes("text-slate-400 mt-3 text-lg")
                else:
                    rows = [{
                        "馬券種": b.get("馬券種"),
                        "買い目": b.get("買い目"),
                        "購入額": b.get("購入額"),
                        "理由": b.get("理由", ""),
                    } for b in bets]
                    ui.aggrid({
                        "columnDefs": [
                            {"field": "馬券種", "width": 110},
                            {"field": "買い目", "width": 180},
                            {"field": "購入額", "width": 110,
                             "cellStyle": {"textAlign": "right"}},
                            {"field": "理由", "flex": 1, "wrapText": True,
                             "autoHeight": True},
                        ],
                        "rowData": rows,
                        "defaultColDef": {"resizable": True},
                        "domLayout": "autoHeight",
                        "rowHeight": 60,
                        "headerHeight": 40,
                    }).classes("w-full")

                    total = sum(b.get("購入額", 0) for b in bets)
                    ui.label(f"合計予算: ¥{total:,}") \
                      .classes("text-2xl font-bold mt-3 text-amber-300")

    # ===== サイドバー: 場所別 expansion =====
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
                with ui.expansion(
                    f"{place} ({len(race_list)}R)",
                    icon="place",
                    value=(i == 0),  # 最初の場所だけデフォルト展開
                ).classes("w-full"):
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
    ui.run(
        port=8080,
        title="🏇 PyCaLiAI (NiceGUI)",
        favicon="🏇",
        dark=True,
        reload=False,
    )
