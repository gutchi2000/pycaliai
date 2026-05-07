"""
nicegui_app.py
==============
NiceGUI 版 PyCaLiAI (実験 MVP v2)

v1 → v2 変更点:
  1. フォントサイズ全体を 1.2x 拡大 (CSS 注入)
  2. サイドバーレース一覧の段差バグ修正 (flat row layout)
  3. バナーを縦圧縮 + 印馬と性質を 1 ブロックに統合
  4. AGGrid を rowHeight=44 + autoHeight でスクロール不要
  5. 「🔍 全頭分析」タブを追加 (各馬のレーダーチャート)

実行: python nicegui_app.py
URL:  http://localhost:8080
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
    "Ｇ１": "#e74c3c", "G1": "#e74c3c", "GI": "#e74c3c",
    "Ｇ２": "#9b59b6", "G2": "#9b59b6", "GII": "#9b59b6",
    "Ｇ３": "#2980b9", "G3": "#2980b9", "GIII": "#2980b9",
}
NATURE_COLORS = {
    "固い": "#a6e3a1", "中堅": "#89b4fa", "混戦": "#cba6f7",
    "穴推奨": "#fab387", "見送り": "#6c7086",
}


def race_nature(rc: dict) -> str:
    """race_confidence から性質を判定"""
    top1 = rc.get("top1_dominance") or 0
    chaos = rc.get("field_chaos_score") or 0
    if chaos >= 0.92:
        return "見送り"
    if top1 >= 0.10 and chaos < 0.70:
        return "固い"
    if top1 < 0.05 and chaos >= 0.85:
        return "混戦"
    return "中堅"


def make_banner_html(race: dict) -> str:
    """SPAIA 風だが密に。情報量増、縦サイズ削減"""
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

    # 印馬 (◎/〇/▲)
    hon = next((h for h in horses if h.get("mark") == "◎"), None)
    tai = next((h for h in horses if h.get("mark") == "〇"), None)
    san = next((h for h in horses if h.get("mark") == "▲"), None)

    def mark_row(mark, color, h):
        if not h:
            return ""
        score = (h.get("p_win") or 0) * 100
        odds = h.get("tansho_odds")
        odds_str = f"{odds:.1f}倍" if odds else "-"
        umaban = h.get("umaban", "?")
        return f"""
        <div style="display:flex;align-items:center;gap:14px;padding:5px 0">
          <span style="background:{color};color:#fff;font-size:18px;font-weight:bold;
                       width:32px;height:32px;line-height:32px;text-align:center;
                       border-radius:50%;flex-shrink:0">{mark}</span>
          <span style="color:#888;font-size:14px;width:32px">{umaban}番</span>
          <span style="color:#cdd6f4;font-size:17px;font-weight:600;flex-grow:1">
            {h.get("horse_name","-")}</span>
          <span style="color:#a6e3a1;font-size:15px;font-weight:bold;
                       background:rgba(166,227,161,0.12);padding:3px 14px;
                       border-radius:12px">{score:.1f}%</span>
          <span style="color:#fab387;font-size:14px;background:rgba(250,179,135,0.12);
                       padding:3px 12px;border-radius:10px">単勝 {odds_str}</span>
        </div>
        """

    return f"""
    <div style="
      background:linear-gradient(135deg,#0d1421 0%,#16213e 50%,#1a2845 100%);
      border:1px solid #f39c12;border-radius:14px;
      padding:18px 22px;margin-bottom:12px;position:relative;overflow:hidden;
      box-shadow:0 4px 20px rgba(243,156,18,0.15);width:100%">
      <div style="position:absolute;top:0;left:0;right:0;height:4px;
                  background:linear-gradient(90deg,#e74c3c,#f39c12,#e74c3c)"></div>

      <!-- 1 行目: グレード + 性質 + 場所/R + コース + 頭数 + クラス -->
      <div style="display:flex;flex-wrap:wrap;align-items:center;gap:10px;
                  margin-bottom:10px;font-size:14px">
        <span style="background:{grade_color};color:#fff;padding:3px 12px;
                     border-radius:4px;font-size:13px;font-weight:bold">{cls}</span>
        <span style="background:{nat_color};color:#1e1e2e;padding:3px 12px;
                     border-radius:4px;font-size:13px;font-weight:bold">{nature}</span>
        <span style="color:#cdd6f4;font-weight:600">{place} {r_num}R</span>
        <span style="color:#888">|</span>
        <span style="color:#f5c2e7">{course}</span>
        <span style="color:#888">|</span>
        <span style="color:#f5c2e7">{field_size}頭</span>
      </div>

      <!-- レース名 -->
      <h2 style="font-size:30px;font-weight:900;color:#cdd6f4;margin:0 0 12px 0;
                 letter-spacing:1.5px;line-height:1.1">{race_name}</h2>

      <!-- 印馬 (左の縦ラインで強調) -->
      <div style="background:rgba(0,0,0,0.25);border-left:3px solid #f39c12;
                  padding:6px 16px;border-radius:6px">
        {mark_row("◎", "#e74c3c", hon)}
        {mark_row("〇", "#3498db", tai)}
        {mark_row("▲", "#9b59b6", san)}
      </div>
    </div>
    """


def make_confidence_html(rc: dict) -> str:
    """race_confidence メトリクス 4 chip"""
    top1 = rc.get("top1_dominance") or 0
    top2 = rc.get("top2_concentration") or 0
    chaos = rc.get("field_chaos_score") or 0
    market = rc.get("ai_market_agreement") or 0

    def chip(label, value, color="#cdd6f4"):
        return f"""
        <div style="background:#1e1e2e;border:1px solid #313244;border-radius:8px;
                    padding:10px 14px;text-align:center;flex:1">
          <div style="color:#6c7086;font-size:12px;margin-bottom:4px">{label}</div>
          <div style="color:{color};font-size:18px;font-weight:bold">{value}</div>
        </div>
        """

    return f"""
    <div style="display:flex;gap:8px;margin-bottom:12px">
      {chip("◎独走度", f"{top1:.3f}")}
      {chip("上位2頭集中", f"{top2:.3f}")}
      {chip("混戦度", f"{chaos:.3f}")}
      {chip("市場一致", f"{market:+.3f}")}
    </div>
    """


def render_horse_analysis_card(h: dict, container):
    """1 馬の評価カード (基本情報 + バー + 簡易レーダー)"""
    name = h.get("horse_name", "?")
    umaban = h.get("umaban", "?")
    mark = h.get("mark") or ""
    p_win = (h.get("p_win") or 0) * 100
    p_sho = (h.get("p_sho") or 0) * 100
    tansho = h.get("tansho_odds") or 0
    fuku_low = h.get("fuku_odds_low") or 0
    fuku_high = h.get("fuku_odds_high") or 0
    fuku_mid = (fuku_low + fuku_high) / 2 if fuku_low and fuku_high else 0
    ev_tan = p_win / 100 * tansho if tansho else 0
    ev_fuku = p_sho / 100 * fuku_mid if fuku_mid else 0
    ai_vs_market = h.get("ai_vs_market") or "unknown"
    market_color = {"under": "#a6e3a1", "fair": "#89b4fa",
                     "over": "#f38ba8", "unknown": "#6c7086"}[
                       ai_vs_market if ai_vs_market in ["under","fair","over","unknown"]
                       else "unknown"
                     ]

    mark_color = {
        "◎": "#e74c3c", "〇": "#3498db", "▲": "#9b59b6", "△": "#f39c12",
    }.get(mark, "#6c7086")
    mark_html = (
        f'<span style="background:{mark_color};color:#fff;font-size:18px;font-weight:bold;'
        f'width:34px;height:34px;line-height:34px;text-align:center;border-radius:50%;'
        f'display:inline-block;margin-right:8px">{mark}</span>'
    ) if mark else ""

    def bar(label, value, max_val, color):
        pct = min(100, (value / max_val) * 100) if max_val > 0 else 0
        return f"""
        <div style="margin-bottom:8px">
          <div style="display:flex;justify-content:space-between;margin-bottom:3px;font-size:13px">
            <span style="color:#a6adc8">{label}</span>
            <span style="color:{color};font-weight:bold">{value:.2f}</span>
          </div>
          <div style="background:#313244;border-radius:4px;height:8px;overflow:hidden">
            <div style="background:{color};height:100%;width:{pct}%;
                        transition:width 0.3s"></div>
          </div>
        </div>
        """

    with container:
        ui.html(f"""
        <div style="background:#1e1e2e;border:1px solid #313244;border-radius:10px;
                    padding:14px 16px;margin-bottom:10px">
          <div style="display:flex;align-items:center;margin-bottom:10px">
            {mark_html}
            <span style="color:#888;font-size:14px;margin-right:10px">{umaban}番</span>
            <span style="color:#cdd6f4;font-size:18px;font-weight:bold;flex-grow:1">{name}</span>
            <span style="background:{market_color};color:#1e1e2e;
                         padding:2px 12px;border-radius:4px;font-size:12px;font-weight:bold">
              {ai_vs_market}</span>
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">
            <div>
              {bar("勝率 (p_win)", p_win, 50, "#a6e3a1")}
              {bar("複勝率 (p_sho)", p_sho, 80, "#89b4fa")}
            </div>
            <div>
              {bar("単勝 EV", ev_tan, 3.0, "#f9e2af")}
              {bar("複勝 EV", ev_fuku, 2.0, "#fab387")}
            </div>
          </div>
          <div style="display:flex;gap:14px;margin-top:8px;padding-top:8px;
                      border-top:1px solid #313244;font-size:12px;color:#888">
            <span>単勝 {tansho:.1f}倍</span>
            <span>複勝 {fuku_low:.1f}-{fuku_high:.1f}倍</span>
          </div>
        </div>
        """)


# ============================================================
# UI 構築
# ============================================================
@ui.page("/")
def main_page():
    ui.dark_mode().enable()

    # === グローバル CSS (フォントサイズ拡大 + テーブル見栄え) ===
    ui.add_head_html("""
    <style>
      body, .nicegui-content { font-size: 16px; }
      .ag-theme-balham-dark { font-size: 15px !important; }
      .ag-theme-balham-dark .ag-cell {
        line-height: 44px !important;
        padding-left: 12px !important;
        padding-right: 12px !important;
      }
      .ag-header-cell-label { font-size: 14px !important; font-weight: bold !important; }
      .ag-theme-balham-dark .ag-row { height: 44px !important; }
      .ag-theme-balham-dark { --ag-row-height: 44px !important; }
      /* タブのフォント */
      .q-tab__label { font-size: 16px !important; }
      /* セレクトボックス */
      .q-field__native { font-size: 16px !important; }
    </style>
    """)

    # ヘッダー
    with ui.header(elevated=True).classes("bg-slate-900"):
        ui.label("🏇 PyCaLiAI").classes("text-3xl font-bold text-white")
        ui.space()
        ui.label("NiceGUI 版 (実験 MVP)").classes("text-base text-slate-400")

    state = {"date": None, "race": None, "bundle": None}

    # 2 カラム
    with ui.row().classes("w-full no-wrap gap-4 p-4"):
        # ===== サイドバー =====
        with ui.column().classes("w-72 gap-2"):
            ui.label("📅 開催日").classes("text-lg font-bold text-slate-200")
            dates = list_dates()
            if not dates:
                ui.label("⚠️ data/weekly/ に CSV がありません").classes("text-orange-400")
                return
            date_dd = ui.select(dates, value=dates[0]).classes("w-full text-base")

            ui.label("🏇 レース").classes("text-lg font-bold text-slate-200 mt-3")
            race_list_box = ui.column().classes("w-full gap-1")

        # ===== メイン =====
        with ui.column().classes("flex-grow gap-2"):
            banner_box = ui.element("div").classes("w-full")
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

    # ===== レース詳細表示 =====
    def render_race(race: dict):
        state["race"] = race

        # バナー
        banner_box.clear()
        with banner_box:
            ui.html(make_banner_html(race))

        # 信頼度メトリクス
        confidence_box.clear()
        with confidence_box:
            rc = race.get("race_confidence", {}) or {}
            ui.html(make_confidence_html(rc))

        # 出走表 (AGGrid)
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
                    {"field": "馬名", "width": 220, "sortable": True},
                    {"field": "勝率(%)", "width": 110, "sortable": True,
                     "cellStyle": {"textAlign": "right"}},
                    {"field": "複勝率(%)", "width": 110, "sortable": True,
                     "cellStyle": {"textAlign": "right"}},
                    {"field": "単勝", "width": 90, "sortable": True,
                     "cellStyle": {"textAlign": "right"}},
                    {"field": "単勝EV", "width": 100, "sortable": True,
                     "cellStyle": {"textAlign": "right",
                                   "fontWeight": "bold", "color": "#f9e2af"}},
                    {"field": "複勝下", "width": 90, "sortable": True,
                     "cellStyle": {"textAlign": "right"}},
                    {"field": "複勝上", "width": 90, "sortable": True,
                     "cellStyle": {"textAlign": "right"}},
                    {"field": "vs市場", "width": 100, "sortable": True},
                ],
                "rowData": row_data,
                "defaultColDef": {"resizable": True},
                "domLayout": "autoHeight",
                "rowHeight": 44,
                "headerHeight": 44,
            }).classes("w-full")

        # === 全頭分析 ===
        bunseki_box.clear()
        with bunseki_box:
            horses_sorted = sorted(race.get("horses", []),
                                    key=lambda h: -(h.get("p_win") or 0))
            ui.label(f"📊 出走馬 {len(horses_sorted)} 頭の評価 (勝率順)") \
              .classes("text-lg font-bold mb-2")
            ui.label("各バーは race 内の最大値で正規化。単勝EV / 複勝EV はオッズと確率の積。") \
              .classes("text-sm text-slate-400 mb-3")
            for h in horses_sorted:
                render_horse_analysis_card(h, bunseki_box)

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
                                padding:5px 16px;border-radius:4px;font-weight:bold;
                                margin-bottom:10px;font-size:14px">{race_nature_str}</div>
                    """)
                if race_reason:
                    ui.html(f"""
                    <div style="background:#1e1e2e;border-left:3px solid #fab387;
                                padding:10px 14px;margin:8px 0;color:#cdd6f4;font-size:14px;
                                line-height:1.6">
                      <b style="color:#fab387">📝 根拠:</b> {race_reason}
                    </div>
                    """)
                if not bets:
                    ui.label("→ 見送り (購入なし)") \
                      .classes("text-slate-400 mt-3 text-base")
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
                        "rowHeight": 60,  # 理由が長いので高め
                        "headerHeight": 44,
                    }).classes("w-full")

                    total = sum(b.get("購入額", 0) for b in bets)
                    ui.label(f"合計予算: ¥{total:,}") \
                      .classes("text-xl font-bold mt-3 text-amber-300")

    # ===== レースリスト更新 (フラットレイアウト) =====
    def update_race_list(date: str):
        state["date"] = date
        bundle = load_bundle(date)
        state["bundle"] = bundle

        race_list_box.clear()

        if not bundle:
            with race_list_box:
                ui.label("⚠️ bundle.json がありません").classes("text-orange-400 text-sm")
                ui.label(f".\\weekly_cowork.ps1 {date} v5") \
                  .classes("font-mono text-xs text-slate-400")
            return

        races = bundle.get("races", [])
        by_place: dict[str, list] = {}
        for r in races:
            place = r.get("race_meta", {}).get("place", "?")
            by_place.setdefault(place, []).append(r)

        with race_list_box:
            for place, race_list in by_place.items():
                ui.label(place).classes("text-base text-slate-300 mt-3 font-bold")
                for r in race_list:
                    rid = r.get("race_id", "")
                    r_num = rid[-2:].lstrip("0") if len(rid) >= 16 else "?"
                    meta = r.get("race_meta", {})
                    cls = meta.get("class", "") or "-"
                    hon = next((h for h in r.get("horses", [])
                                if h.get("mark") == "◎"), None)
                    hon_name = hon.get("horse_name", "-") if hon else "-"

                    # フラット 1 row レイアウト (段差バグ対策)
                    item = ui.element("div").classes(
                        "w-full px-3 py-2 cursor-pointer rounded "
                        "hover:bg-slate-800 transition-colors"
                    )
                    item.on("click", lambda r=r: render_race(r))
                    with item:
                        ui.html(f"""
                        <div style="display:flex;flex-direction:column;gap:2px">
                          <div style="display:flex;align-items:baseline;gap:8px">
                            <span style="color:#cdd6f4;font-size:15px;font-weight:bold;
                                         min-width:38px">{r_num}R</span>
                            <span style="color:#a6adc8;font-size:13px">{cls}</span>
                          </div>
                          <div style="color:#a6e3a1;font-size:13px;padding-left:46px">
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
