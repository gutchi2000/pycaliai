"""
nicegui_app.py
==============
NiceGUI 版 PyCaLiAI (実験的 MVP)

実行:
    python nicegui_app.py
ブラウザ:
    http://localhost:8080

このアプリは以下を表示:
  - 日付選択 (data/weekly/*.csv の中から)
  - レース一覧 (場所別)
  - レース詳細:
    - SPAIA 風バナー (G1/G2/G3 + レース名 + 距離 + 馬場 + 印 ◎/〇/▲)
    - 出走表 (AGGrid、ソート/フィルタ可)
    - レース信頼度メトリクス
    - Cowork 買い目 (保存済みの場合)

データソース: reports/cowork_input/{YYYYMMDD}_bundle.json
              (weekly_cowork.ps1 で生成された印 + 確率 + オッズ)

データが無い日付では「先に .\\weekly_cowork.ps1 [DATE] v5 を実行してください」と表示。
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
    """data/weekly/*.csv の日付一覧 (新しい順)"""
    if not WEEKLY_DIR.exists():
        return []
    dates = []
    for p in WEEKLY_DIR.glob("????????.csv"):
        if p.stem.isdigit() and len(p.stem) == 8:
            dates.append(p.stem)
    return sorted(dates, reverse=True)


def load_bundle(date_str: str) -> dict | None:
    """reports/cowork_input/{date}_bundle.json を読み込む"""
    p = COWORK_INPUT_DIR / f"{date_str}_bundle.json"
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_cowork_bets(date_str: str, race_id: str) -> dict | None:
    """reports/cowork_bets/{date}/{race_id}.json を読み込む"""
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
    "固い": "#a6e3a1",
    "中堅": "#89b4fa",
    "混戦": "#cba6f7",
    "穴推奨": "#fab387",
    "見送り": "#6c7086",
}


def make_banner_html(race: dict) -> str:
    """SPAIA 風バナー HTML 生成"""
    meta = race.get("race_meta", {}) or {}
    horses = race.get("horses", []) or []

    # 印馬抽出
    hon = next((h for h in horses if h.get("mark") == "◎"), None)
    tai = next((h for h in horses if h.get("mark") == "〇"), None)
    san = next((h for h in horses if h.get("mark") == "▲"), None)

    cls = meta.get("class", "")
    grade_color = GRADE_COLORS.get(cls, "#27ae60")
    place = meta.get("place", "")
    course = meta.get("course", "")
    field_size = meta.get("field_size", 0)
    race_name = meta.get("race_name") or cls or ""
    rid = race.get("race_id", "")
    r_num = rid[-2:].lstrip("0") if len(rid) >= 16 else "?"

    def mark_row(mark, badge_color, h):
        if not h:
            return ""
        score = h.get("p_win", 0) * 100 if h.get("p_win") is not None else 0
        odds = h.get("tansho_odds")
        odds_str = f"{odds:.1f}" if odds else "-"
        return f"""
        <div style="display:flex;align-items:center;gap:12px;padding:6px 0">
          <span style="background:{badge_color};color:#fff;font-size:18px;font-weight:bold;
                       width:32px;height:32px;line-height:32px;text-align:center;border-radius:50%;flex-shrink:0">
            {mark}
          </span>
          <span style="color:#cdd6f4;font-size:16px;font-weight:600;flex-grow:1">{h.get("horse_name","-")}</span>
          <span style="color:#a6e3a1;font-size:14px;background:rgba(166,227,161,0.1);
                       padding:2px 12px;border-radius:12px">{score:.1f}%</span>
          <span style="color:#fab387;font-size:13px;background:rgba(250,179,135,0.1);
                       padding:2px 10px;border-radius:10px">単勝 {odds_str}</span>
        </div>
        """

    return f"""
    <div style="
      background:linear-gradient(135deg,#0d1421 0%,#16213e 50%,#1a2845 100%);
      border:1px solid #f39c12;border-radius:14px;
      padding:24px 28px;margin-bottom:16px;position:relative;overflow:hidden;
      box-shadow:0 4px 20px rgba(243,156,18,0.15);width:100%">
      <div style="position:absolute;top:0;left:0;right:0;height:4px;
                  background:linear-gradient(90deg,#e74c3c,#f39c12,#e74c3c)"></div>

      <div style="margin-bottom:16px">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;flex-wrap:wrap">
          <span style="background:{grade_color};color:#fff;padding:3px 12px;
                       border-radius:4px;font-size:12px;font-weight:bold">{cls}</span>
          <span style="color:#a6adc8;font-size:13px">{place} {r_num}R</span>
        </div>
        <h2 style="font-size:32px;font-weight:900;color:#cdd6f4;margin:0;
                   letter-spacing:1.5px;text-shadow:0 2px 6px rgba(0,0,0,0.4)">{race_name}</h2>
      </div>

      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:16px">
        <div style="background:rgba(255,255,255,0.04);border:1px solid #313244;
                    border-radius:8px;padding:8px 12px;text-align:center">
          <div style="color:#6c7086;font-size:11px">コース</div>
          <div style="color:#f5c2e7;font-size:15px;font-weight:bold">{course}</div>
        </div>
        <div style="background:rgba(255,255,255,0.04);border:1px solid #313244;
                    border-radius:8px;padding:8px 12px;text-align:center">
          <div style="color:#6c7086;font-size:11px">頭数</div>
          <div style="color:#f5c2e7;font-size:15px;font-weight:bold">{field_size}頭</div>
        </div>
        <div style="background:rgba(255,255,255,0.04);border:1px solid #313244;
                    border-radius:8px;padding:8px 12px;text-align:center">
          <div style="color:#6c7086;font-size:11px">クラス</div>
          <div style="color:#f5c2e7;font-size:15px;font-weight:bold">{cls or "-"}</div>
        </div>
      </div>

      <div style="background:rgba(0,0,0,0.2);border-left:3px solid #f39c12;
                  padding:8px 16px;border-radius:6px">
        {mark_row("◎", "#e74c3c", hon)}
        {mark_row("〇", "#3498db", tai)}
        {mark_row("▲", "#9b59b6", san)}
      </div>
    </div>
    """


def make_confidence_html(race: dict) -> str:
    """race_confidence メトリクス表示 (4 つのチップ)"""
    rc = race.get("race_confidence", {}) or {}
    top1 = rc.get("top1_dominance", 0) or 0
    top2 = rc.get("top2_concentration", 0) or 0
    chaos = rc.get("field_chaos_score", 0) or 0
    market = rc.get("ai_market_agreement", 0) or 0

    # 性質判定 (簡易、Cowork prompt のロジックと同じ)
    if top1 >= 0.10 and chaos < 0.70:
        nature = "固い"
    elif top1 < 0.05 and chaos >= 0.85:
        nature = "混戦"
    elif chaos >= 0.92:
        nature = "見送り"
    else:
        nature = "中堅"
    nat_color = NATURE_COLORS.get(nature, "#89b4fa")

    return f"""
    <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:8px;
                margin-bottom:16px">
      <div style="background:#1e1e2e;border:1px solid #313244;border-radius:8px;
                  padding:10px;text-align:center">
        <div style="color:#6c7086;font-size:11px">レース性質</div>
        <div style="color:{nat_color};font-size:14px;font-weight:bold">{nature}</div>
      </div>
      <div style="background:#1e1e2e;border:1px solid #313244;border-radius:8px;
                  padding:10px;text-align:center">
        <div style="color:#6c7086;font-size:11px">◎独走度</div>
        <div style="color:#cdd6f4;font-size:14px;font-weight:bold">{top1:.3f}</div>
      </div>
      <div style="background:#1e1e2e;border:1px solid #313244;border-radius:8px;
                  padding:10px;text-align:center">
        <div style="color:#6c7086;font-size:11px">上位2頭集中</div>
        <div style="color:#cdd6f4;font-size:14px;font-weight:bold">{top2:.3f}</div>
      </div>
      <div style="background:#1e1e2e;border:1px solid #313244;border-radius:8px;
                  padding:10px;text-align:center">
        <div style="color:#6c7086;font-size:11px">混戦度</div>
        <div style="color:#cdd6f4;font-size:14px;font-weight:bold">{chaos:.3f}</div>
      </div>
      <div style="background:#1e1e2e;border:1px solid #313244;border-radius:8px;
                  padding:10px;text-align:center">
        <div style="color:#6c7086;font-size:11px">市場一致</div>
        <div style="color:#cdd6f4;font-size:14px;font-weight:bold">{market:+.3f}</div>
      </div>
    </div>
    """


# ============================================================
# UI 構築
# ============================================================
@ui.page("/")
def main_page():
    # ダークモード
    ui.dark_mode().enable()

    # ステート
    state = {"date": None, "race": None, "bundle": None}

    # ヘッダー
    with ui.header(elevated=True).classes("bg-slate-900"):
        ui.label("🏇 PyCaLiAI").classes("text-2xl font-bold text-white")
        ui.space()
        ui.label("NiceGUI 版 (実験 MVP)").classes("text-sm text-slate-400")

    # メインレイアウト
    with ui.row().classes("w-full no-wrap gap-4 p-4"):
        # ===== サイドバー (左) =====
        with ui.column().classes("w-64 gap-2"):
            ui.label("📅 開催日").classes("text-base font-bold text-slate-300")
            dates = list_dates()
            if not dates:
                ui.label("⚠️ data/weekly/ に CSV がありません").classes("text-xs text-orange-400")
                return

            date_dd = ui.select(dates, value=dates[0]).classes("w-full")

            ui.label("🏇 レース").classes("text-base font-bold text-slate-300 mt-4")
            race_list_box = ui.column().classes("w-full gap-1")

        # ===== メイン (右) =====
        with ui.column().classes("flex-grow gap-2"):
            banner_box = ui.element("div").classes("w-full")
            confidence_box = ui.element("div").classes("w-full")

            with ui.tabs().classes("w-full") as tabs:
                tab_shutsuba = ui.tab("📋 出走表")
                tab_bets = ui.tab("🎫 Cowork 買い目")

            with ui.tab_panels(tabs, value=tab_shutsuba).classes("w-full"):
                with ui.tab_panel(tab_shutsuba):
                    shutsuba_box = ui.element("div").classes("w-full")
                with ui.tab_panel(tab_bets):
                    bets_box = ui.element("div").classes("w-full")

    # ===== レース選択時の更新 =====
    def render_race(race: dict):
        state["race"] = race

        # バナー
        banner_box.clear()
        with banner_box:
            ui.html(make_banner_html(race))

        # 信頼度メトリクス
        confidence_box.clear()
        with confidence_box:
            ui.html(make_confidence_html(race))

        # 出走表
        shutsuba_box.clear()
        with shutsuba_box:
            horses = race.get("horses", [])
            row_data = []
            for h in horses:
                p_win = (h.get("p_win") or 0) * 100
                p_sho = (h.get("p_sho") or 0) * 100
                row_data.append({
                    "番": h.get("umaban"),
                    "印": h.get("mark") or "",
                    "馬名": h.get("horse_name"),
                    "p_win(%)": round(p_win, 1),
                    "p_sho(%)": round(p_sho, 1),
                    "単勝": h.get("tansho_odds"),
                    "複勝下": h.get("fuku_odds_low"),
                    "複勝上": h.get("fuku_odds_high"),
                    "vs市場": h.get("ai_vs_market"),
                })
            ui.aggrid({
                "columnDefs": [
                    {"field": "番", "width": 60, "sortable": True, "pinned": "left"},
                    {"field": "印", "width": 60, "sortable": True},
                    {"field": "馬名", "width": 200, "sortable": True},
                    {"field": "p_win(%)", "width": 100, "sortable": True,
                     "cellStyle": {"textAlign": "right"}},
                    {"field": "p_sho(%)", "width": 100, "sortable": True,
                     "cellStyle": {"textAlign": "right"}},
                    {"field": "単勝", "width": 90, "sortable": True,
                     "cellStyle": {"textAlign": "right"}},
                    {"field": "複勝下", "width": 90, "sortable": True,
                     "cellStyle": {"textAlign": "right"}},
                    {"field": "複勝上", "width": 90, "sortable": True,
                     "cellStyle": {"textAlign": "right"}},
                    {"field": "vs市場", "width": 100, "sortable": True},
                ],
                "rowData": row_data,
                "defaultColDef": {"resizable": True},
                "domLayout": "autoHeight",
            }).classes("w-full")

        # Cowork 買い目
        bets_box.clear()
        with bets_box:
            cowork = load_cowork_bets(state["date"], race["race_id"])
            if not cowork:
                ui.label("Cowork 買い目はまだ保存されていません。") \
                  .classes("text-slate-400 p-4")
            else:
                race_nature = cowork.get("race_nature", "")
                race_reason = cowork.get("race_reason", "")
                bets = cowork.get("bets", [])

                if race_nature:
                    nat_color = NATURE_COLORS.get(race_nature, "#89b4fa")
                    ui.html(f"""
                    <div style="display:inline-block;background:{nat_color};color:#1e1e2e;
                                padding:4px 12px;border-radius:4px;font-weight:bold;
                                margin-bottom:8px">{race_nature}</div>
                    """)

                if race_reason:
                    ui.html(f"""
                    <div style="background:#1e1e2e;border-left:3px solid #fab387;
                                padding:8px 12px;margin:6px 0;color:#cdd6f4;font-size:13px">
                      <b style="color:#fab387">📝 根拠:</b> {race_reason}
                    </div>
                    """)

                if not bets:
                    ui.label("→ 見送り (購入なし)").classes("text-slate-400 mt-2")
                else:
                    rows = [{
                        "馬券種": b.get("馬券種"),
                        "買い目": b.get("買い目"),
                        "購入額": b.get("購入額"),
                        "理由": b.get("理由", ""),
                    } for b in bets]
                    ui.aggrid({
                        "columnDefs": [
                            {"field": "馬券種", "width": 100},
                            {"field": "買い目", "width": 150},
                            {"field": "購入額", "width": 100,
                             "cellStyle": {"textAlign": "right"}},
                            {"field": "理由", "width": 500, "wrapText": True,
                             "autoHeight": True},
                        ],
                        "rowData": rows,
                        "defaultColDef": {"resizable": True},
                        "domLayout": "autoHeight",
                    }).classes("w-full")

                    total = sum(b.get("購入額", 0) for b in bets)
                    ui.label(f"合計予算: ¥{total:,}").classes("text-lg font-bold mt-2")

    # ===== レースリスト更新 =====
    def update_race_list(date: str):
        state["date"] = date
        bundle = load_bundle(date)
        state["bundle"] = bundle

        race_list_box.clear()

        if not bundle:
            with race_list_box:
                ui.label("⚠️ bundle.json がありません").classes("text-orange-400 text-xs")
                ui.label(f".\\weekly_cowork.ps1 {date} v5").classes("font-mono text-xs text-slate-400")
            return

        races = bundle.get("races", [])
        # 場所別にグルーピング
        by_place: dict[str, list] = {}
        for r in races:
            place = r.get("race_meta", {}).get("place", "?")
            by_place.setdefault(place, []).append(r)

        with race_list_box:
            for place, race_list in by_place.items():
                ui.label(place).classes("text-sm text-slate-300 mt-2 font-bold")
                with ui.column().classes("gap-1 w-full"):
                    for r in race_list:
                        rid = r.get("race_id", "")
                        r_num = rid[-2:].lstrip("0") if len(rid) >= 16 else "?"
                        meta = r.get("race_meta", {})
                        cls = meta.get("class", "") or "-"

                        # ◎の名前
                        hon = next((h for h in r.get("horses", [])
                                    if h.get("mark") == "◎"), None)
                        hon_name = hon.get("horse_name", "-") if hon else "-"

                        with ui.button(
                            on_click=lambda r=r: render_race(r)
                        ).classes("w-full justify-start").props("flat dense"):
                            with ui.column().classes("gap-0 items-start"):
                                ui.label(f"{r_num}R  {cls}").classes("text-xs text-slate-300")
                                ui.label(f"◎ {hon_name}").classes("text-xs text-green-400")

    # 初期表示
    date_dd.on_value_change(lambda e: update_race_list(e.value))
    update_race_list(dates[0])


# ============================================================
# 起動
# ============================================================
if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        port=8080,
        title="🏇 PyCaLiAI (NiceGUI)",
        favicon="🏇",
        dark=True,
        reload=False,
    )
