# -*- coding: utf-8 -*-
"""
power_rating_site.py — 強さスコアのローカル閲覧用ページを生成する
================================================================================
`python power_rating.py site` から呼ばれる。出力は reports/power_rating/site/
（.gitignore 済み）の index.html + data.js の 2 ファイルだけ。

  * **公開サイトには載らない**。sync-hf-umami.ps1 / wrangler が配るのは site/ 配下だけで、
    このスクリプトは site/ に一切書かない（出力先に site/ 配下を指定するとエラーで止める）。
  * データは data.js に埋め込むので、サーバー無しで index.html をダブルクリックすれば開く。
  * 見た目は公開サイト（site/css/style.css）と同じ配色・書体。

使い方:
  python power_rating.py build
  python power_rating.py site                    # 直近 2 開催日の出走表つき
  python power_rating.py site --date 20261114 --date 20261115
  start reports\\power_rating\\site\\index.html      # Windows
"""
from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path

import power_rating as pr

log = logging.getLogger("power_rating")

SITE_OUT = pr.OUT_DIR / "site"
PUBLIC_SITE = pr.BASE / "site"
HIST_MAX = 40          # 1 頭あたり推移に載せる最大走数（直近から）


def _default_dates() -> list[str]:
    """出走表のある直近 2 開催日（YYYYMMDD）。"""
    days = sorted(f.stem for f in pr.WEEKLY_DIR.glob("[0-9]" * 8 + ".csv"))
    return days[-2:]


def _history(names: set[str], anchors: dict) -> dict[str, dict[str, list[list]]]:
    """馬名 → {芝ダ: [[日付, スコア, クラス, 着順, 頭数], …]}（その芝ダのものさしで換算）。"""
    out: dict[str, dict[str, list[list]]] = {}
    path = pr.OUT_DIR / "history.csv"
    if not path.exists():
        return out
    with path.open(encoding="utf-8-sig", newline="") as fp:
        for r in csv.DictReader(fp):
            sf = r["芝ダ"]
            if r["馬名"] in names and sf in anchors:
                a = anchors[sf]
                out.setdefault(r["馬名"], {}).setdefault(sf, []).append(
                    [r["日付"], round(pr.to_score(float(r["R後"]), a["rating_lo"],
                                                  a["rating_hi"]), 1),
                     r["クラス"], int(r["着順"]), int(r["頭数"])])
    return {k: {sf: v[-HIST_MAX:] for sf, v in d.items()} for k, d in out.items()}


def build(dates: list[str] | None = None, out_dir: Path | None = None) -> int:
    """index.html と data.js を書き出す。戻り値は終了コード。"""
    out_dir = (out_dir or SITE_OUT).resolve()
    if out_dir == PUBLIC_SITE.resolve() or PUBLIC_SITE.resolve() in out_dir.parents:
        log.error("出力先 %s は公開サイト (site/) の中です。非公開ページは site/ の外に出してください",
                  out_dir)
        return 2
    table, meta = pr.read_outputs()
    fm = pr.load_form_model()
    anchors = meta["anchors"]

    cards = []
    in_cards: set[str] = set()
    for d in dates or _default_dates():
        if not (pr.WEEKLY_DIR / f"{d}.csv").exists():
            log.warning("出走表なし: %s（スキップ）", d)
            continue
        iso = f"{d[:4]}-{d[4:6]}-{d[6:]}"
        races = pr.card_races(d, table, fm)
        for race in races:
            for x in race["rows"]:
                in_cards.add(x["name"])
                x.pop("rating", None)
                x.pop("fp", None)
                x["score"] = None if x["score"] is None else round(x["score"], 1)
                x["p3"] = None if x["p3"] is None else round(x["p3"], 3)
        # 基準日がその日以降 = 結果を含んだ値（回顧用）
        cards.append({"date": iso, "stale": meta["asof"] >= iso, "races": races})

    def sv(r: dict, sf: str) -> list | None:
        v = pr.surf_row(r, sf)
        return [round(v["score"], 1), round(v["peak"], 1), v["n"], v["last"],
                int(v["active"])] if v else None

    keep = {nm for nm, r in table.items()
            if nm in in_cards or any(r[f"現役_{sf}"] == "1" for sf in pr.SURFACES)}
    hist = _history(keep, anchors)
    horses = []
    for nm in sorted(keep):
        r = table[nm]
        apt = {k: [round(v[0] / v[1], 3), int(v[1])]
               for k, v in json.loads(r.get("適性") or "{}").items()
               if v[1] and k[:2] in ("s:", "d:")}
        # [馬名, 主戦, 芝[score,peak,走数,最終,現役], ダ[…], 最終クラス, 適性, 芝の推移, ダの推移]
        horses.append([nm, r["主戦"], sv(r, "芝"), sv(r, "ダ"), r["最終クラス"], apt,
                       hist.get(nm, {}).get("芝", []), hist.get(nm, {}).get("ダ", [])])

    data = {
        "meta": {
            "asof": meta["asof"],
            "built_at": datetime.now().isoformat(timespec="seconds"),
            "n_races": meta["n_races"],
            "n_horses": meta["n_horses"],
            "elo_per_point": {sf: anchors[sf]["elo_per_point"] for sf in pr.SURFACES},
            "params": meta["params"],
            "eval_from": meta.get("eval_from"),
            "eval": meta.get("eval") or {},
            "form": meta.get("form") or {},
            "form_q": pr.FORM_Q,
        },
        "horses": horses,
        "cards": cards,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    (out_dir / "data.js").write_text(f"window.PR={payload};\n", encoding="utf-8")
    (out_dir / "index.html").write_text(HTML, encoding="utf-8")
    log.info("ローカルページ: %s（%d 頭 / 出走表 %d 日 / data.js %.0f KB）",
             out_dir / "index.html", len(horses), len(cards),
             (out_dir / "data.js").stat().st_size / 1024)
    print(f"保存: {out_dir / 'index.html'}")
    return 0


HTML = r"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex,nofollow">
<title>強さスコア（芝・ダート別、ローカル専用）— PyCaLiAI</title>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;700&family=Zen+Kaku+Gothic+New:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#070b14;--bg2:#0a101e;--card:#0d1424;--card2:#121c31;--line:#1e2b4a;--line2:#32456e;
  --gold:#f5b942;--gold2:#ffd97a;--tx:#edf1fb;--tx2:#a9b6d3;--tx3:#7385a8;
  --good:#2dd4a8;--warn:#e0876a;
  --s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#9085e9;
  --sans:"Zen Kaku Gothic New","Noto Sans JP",sans-serif;
  --disp:"IBM Plex Mono","Zen Kaku Gothic New","Noto Sans JP",monospace;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--tx);font-family:var(--sans);font-size:15px;line-height:1.6}
.top{position:sticky;top:0;z-index:30;background:rgba(7,11,20,.94);border-bottom:1px solid var(--line);backdrop-filter:blur(6px)}
.top-in{max-width:1240px;margin:0 auto;padding:10px clamp(14px,3vw,30px);display:flex;align-items:center;gap:16px;flex-wrap:wrap}
.brand{font-family:var(--disp);font-weight:600;color:var(--gold);letter-spacing:.04em}
.local{font-size:.75rem;color:var(--tx2);border:1px solid var(--line2);border-radius:4px;padding:1px 8px}
.tabs{display:flex;gap:4px;margin-left:auto;flex-wrap:wrap}
.tabs button{font:inherit;font-weight:600;color:var(--tx2);background:none;border:1px solid transparent;border-radius:4px;padding:5px 14px;cursor:pointer}
.tabs button:hover{color:var(--tx)}
.tabs button[aria-selected=true]{color:var(--gold2);border-color:var(--line2);background:var(--card2)}
.wrap{max-width:1240px;margin:0 auto;padding:18px clamp(14px,3vw,30px) 60px}
.note{color:var(--tx3);font-size:.85rem}
.panel{background:var(--bg2);border:1px solid var(--line);border-radius:8px;padding:14px 18px;margin:0 0 14px}
.panel h2{font-size:.95rem;font-weight:700;color:var(--tx2);letter-spacing:.06em;margin:0 0 10px;display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}
.panel h2 small{font-weight:400;font-size:.78rem;color:var(--tx3);letter-spacing:0}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:0 0 12px}
.chip{font:inherit;font-size:.88rem;color:var(--tx2);background:var(--card);border:1px solid var(--line);border-radius:4px;padding:3px 11px;cursor:pointer}
.chip[aria-pressed=true]{color:var(--gold2);border-color:var(--gold);background:var(--card2)}
select,input[type=text]{font:inherit;color:var(--tx);background:var(--card);border:1px solid var(--line2);border-radius:6px;padding:7px 11px}
input[type=text]{width:100%}
input:focus,select:focus,button:focus-visible{outline:2px solid rgba(245,185,66,.5);outline-offset:1px}
table{width:100%;border-collapse:collapse;font-size:.92rem}
th{font-weight:500;color:var(--tx3);font-size:.78rem;text-align:left;padding:4px 8px;border-bottom:1px solid var(--line2);white-space:nowrap}
td{padding:5px 8px;border-bottom:1px solid var(--line);vertical-align:middle;white-space:nowrap}
td.n,th.n{text-align:right;font-family:var(--disp);font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:none}
.hn{color:var(--tx);background:none;border:none;font:inherit;cursor:pointer;text-align:left;padding:0;white-space:nowrap}
.hn:hover{color:var(--gold2);text-decoration:underline}
.bar{display:inline-block;height:6px;border-radius:0 4px 4px 0;background:var(--gold);vertical-align:middle;margin-left:8px;opacity:.85}
.barcell{white-space:nowrap;min-width:190px}
.lab{white-space:nowrap}
.lab .ic{font-size:.8rem;margin-right:5px}
.lab.up .ic{color:var(--good)}.lab.dn .ic{color:var(--warn)}
.why{color:var(--tx3);font-size:.82rem;margin-left:6px}
.race{margin:0 0 14px}
.race h3{font-size:1rem;font-weight:700;margin:0 0 6px;display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}
.race h3 .rn{font-family:var(--disp);color:var(--gold)}
.race h3 small{font-weight:400;color:var(--tx3);font-size:.82rem}
.sug{position:absolute;left:0;right:0;top:100%;z-index:20;background:var(--card);border:1px solid var(--line2);border-radius:6px;margin-top:4px;max-height:300px;overflow:auto}
.sug[hidden]{display:none}
.sug button{display:flex;justify-content:space-between;gap:10px;width:100%;font:inherit;color:var(--tx);background:none;border:none;padding:7px 12px;cursor:pointer;text-align:left}
.sug button:hover,.sug button.on{background:var(--card2);color:var(--gold2)}
.sug small{color:var(--tx3);font-family:var(--disp)}
.sel{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0 0}
.selc{display:flex;align-items:center;gap:7px;background:var(--card);border:1px solid var(--line2);border-radius:4px;padding:3px 6px 3px 10px;font-size:.9rem}
.sw{width:10px;height:10px;border-radius:2px;flex:none}
.selc button{font:inherit;color:var(--tx3);background:none;border:none;cursor:pointer;padding:0 4px}
.selc button:hover{color:var(--tx)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px;margin:0 0 12px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 14px}
.kpi b{display:block;font-family:var(--disp);font-size:1.5rem;font-weight:600;color:var(--tx)}
.kpi span{color:var(--tx3);font-size:.8rem}
.kpi em{font-style:normal;color:var(--tx2);font-size:.8rem;display:block}
.chartbox{position:relative}
svg{display:block;width:100%;height:auto}
.tip{position:absolute;pointer-events:none;background:var(--card2);border:1px solid var(--line2);border-radius:6px;padding:6px 10px;font-size:.82rem;white-space:nowrap;z-index:5}
.tip[hidden]{display:none}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:.85rem;color:var(--tx2);margin:6px 0 0}
.legend span{display:flex;align-items:center;gap:6px}
details{margin-top:10px}summary{cursor:pointer;color:var(--tx2);font-size:.88rem}
.warnbox{border:1px solid var(--line2);border-left:3px solid var(--warn);border-radius:6px;padding:8px 12px;margin:0 0 12px;color:var(--tx2);font-size:.88rem}
.scroll{overflow-x:auto}
ul.plain{margin:0 0 0 1.2em;color:var(--tx2);font-size:.9rem}
ul.plain li{margin:2px 0}
@media (max-width:640px){.barcell{min-width:120px}}
</style>
</head>
<body>
<header class="top"><div class="top-in">
  <span class="brand">PyCaLiAI 強さスコア</span>
  <span class="local">ローカル専用・非公開</span>
  <span class="note" id="asof"></span>
  <nav class="tabs" role="tablist" id="tabs">
    <button role="tab" data-t="card">出走表</button>
    <button role="tab" data-t="cmp">比較</button>
    <button role="tab" data-t="rank">ランキング</button>
    <button role="tab" data-t="val">検証</button>
  </nav>
</div></header>
<main class="wrap" id="main"></main>
<script src="data.js"></script>
<script>
(function(){
"use strict";
var PR = window.PR, main = document.getElementById("main");
if(!PR){ main.textContent = "data.js が読めません。python power_rating.py site を実行してください。"; return; }
var M = PR.meta, EPP = M.elo_per_point;
var SI = {"芝":2, "ダ":3}, HI = {"芝":6, "ダ":7}, SNAME = {"芝":"芝", "ダ":"ダート"};
var COLORS = ["var(--s1)","var(--s2)","var(--s3)","var(--s4)"], MAXSEL = 4;
// horses: [馬名, 主戦, 芝[score,peak,走数,最終,現役]|null, ダ[…]|null, クラス, 適性, 芝の推移, ダの推移]
var H = {}; PR.horses.forEach(function(h){ H[h[0]] = h; });
var state = { tab:"card", card:0, place:null, sel:[], colorOf:{}, surf:"芝", csurf:"芝", minN:3 };
function sv(h, sf){ return h[SI[sf]]; }

function el(tag, attrs, kids){
  var e = document.createElement(tag);
  if(attrs) Object.keys(attrs).forEach(function(k){
    if(k === "text") e.textContent = attrs[k];
    else if(k.slice(0,2) === "on") e.addEventListener(k.slice(2), attrs[k]);
    else if(attrs[k] !== null && attrs[k] !== undefined) e.setAttribute(k, attrs[k]);
  });
  (kids || []).forEach(function(c){ if(c) e.appendChild(typeof c === "string" ? document.createTextNode(c) : c); });
  return e;
}
function svg(tag, attrs){
  var e = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.keys(attrs || {}).forEach(function(k){ e.setAttribute(k, attrs[k]); });
  return e;
}
function f1(x){ return x === null || x === undefined ? "--" : x.toFixed(1); }
function pct(x){ return x === null || x === undefined ? "--" : Math.round(x * 100) + "%"; }
function conf(n){ return n >= 8 ? "高" : n >= 4 ? "中" : n >= 1 ? "低" : "-"; }
function winProb(a, b, sf){ return 1 / (1 + Math.pow(10, -((a - b) * EPP[sf]) / 400)); }

function addSel(name){
  if(!H[name] || state.sel.indexOf(name) >= 0) return;
  if(!state.sel.length && H[name][1]) state.csurf = H[name][1];   // 1 頭目の主戦に合わせる
  if(state.sel.length >= MAXSEL){ var out = state.sel.shift(); delete state.colorOf[out]; }
  // 色は馬に固定する（他の馬を外しても塗り直さない）
  var used = state.sel.map(function(n){ return state.colorOf[n]; });
  for(var i = 0; i < COLORS.length; i++){ if(used.indexOf(i) < 0){ state.colorOf[name] = i; break; } }
  state.sel.push(name);
}
function rmSel(name){ state.sel = state.sel.filter(function(n){ return n !== name; }); delete state.colorOf[name]; }
function nameBtn(name){
  if(!H[name]) return el("span", {text:name});
  return el("button", {class:"hn", title:"比較に追加", text:name,
    onclick:function(){ addSel(name); state.tab = "cmp"; render(); }});
}
function labelCell(x){
  if(!x.label) return el("td");
  var up = x.label === "好走気配";
  return el("td", {class:"lab " + (up ? "up" : "dn")}, [
    el("span", {class:"ic", text: up ? "▲" : "▼"}), x.label,
    x.reason ? el("span", {class:"why", text:x.reason}) : null]);
}

// ---------------------------------------------------------------- 出走表
function viewCard(){
  var box = el("div");
  if(!PR.cards.length){ box.appendChild(el("p", {class:"note", text:"出走表データがありません（data/weekly を確認）。"})); return box; }
  var c = PR.cards[state.card] || PR.cards[0];
  var places = [];
  c.races.forEach(function(r){ if(places.indexOf(r.place) < 0) places.push(r.place); });
  if(places.indexOf(state.place) < 0) state.place = places[0];
  var row = el("div", {class:"row"});
  PR.cards.forEach(function(cd, i){
    row.appendChild(el("button", {class:"chip", "aria-pressed": String(i === state.card), text:cd.date,
      onclick:function(){ state.card = i; state.place = null; render(); }}));
  });
  row.appendChild(el("span", {class:"note", text:"｜"}));
  places.forEach(function(p){
    row.appendChild(el("button", {class:"chip", "aria-pressed": String(p === state.place), text:p,
      onclick:function(){ state.place = p; render(); }}));
  });
  box.appendChild(row);
  if(c.stale) box.appendChild(el("div", {class:"warnbox",
    text:"この日はスコアの基準日（" + M.asof + "）以前です。当日の結果を含んだ値なので、回顧用として見てください（事前評価には build --until で作り直す）。"}));
  c.races.filter(function(r){ return r.place === state.place; }).forEach(function(r){
    var sec = el("section", {class:"panel race"});
    sec.appendChild(el("h3", null, [
      el("span", {class:"rn", text:r.rno + "R"}),
      r.name || r.klass,
      el("small", {text:(r.name ? r.klass + "・" : "") + (r.surface || "") + r.dist + "m・" + r.rows.length + "頭・" + (SNAME[r.surface] || "") + "スコアで表示"})]));
    var t = el("table"), tb = el("tbody");
    t.appendChild(el("thead", null, [el("tr", null, [
      el("th", {class:"n", text:"馬番"}), el("th", {text:"馬名"}), el("th", {text:(SNAME[r.surface] || "") + "スコア"}),
      el("th", {class:"n", text:"3着内率"}), el("th", {class:"n", text:"走数"}),
      el("th", {text:"信頼"}), el("th", {text:"気配"})])]));
    r.rows.forEach(function(x){
      var sc = el("td", {class:"barcell"});
      sc.appendChild(el("span", {style:"font-family:var(--disp)", text:f1(x.score)}));
      if(x.score !== null) sc.appendChild(el("span", {class:"bar", style:"width:" + Math.max(0, Math.min(100, x.score)) * 1.2 + "px"}));
      tb.appendChild(el("tr", null, [
        el("td", {class:"n", text:String(x.ban)}), el("td", null, [nameBtn(x.name)]), sc,
        el("td", {class:"n", text: x.p3 === null ? "--" : pct(x.p3) + (x.score === null ? "?" : "")}),
        el("td", {class:"n", text: x.score === null && x.n_all ? "0（別" + x.n_all + "）" : String(x.n)}),
        el("td", {text:conf(x.n)}), labelCell(x)]));
    });
    t.appendChild(tb);
    sec.appendChild(el("div", {class:"scroll"}, [t]));
    box.appendChild(sec);
  });
  box.appendChild(el("p", {class:"note",
    text:"芝のレースは芝スコア、ダートのレースはダートスコアで並べています。3着内率はスコア（+ 条件補正）からの計算値で、オッズは見ていません。「?」はその芝ダが初めての馬（レース平均で仮置き）、「0（別N）」はもう一方の芝ダで N 走している馬です。馬名を押すと比較に追加します。"}));
  return box;
}

// ---------------------------------------------------------------- 比較
function lineChart(names, sf){
  var W = 920, Hh = 340, mL = 44, mR = 196, mT = 14, mB = 30;
  var pts = [], series = names.map(function(n){
    var s = (H[n][HI[sf]] || []).map(function(p){ return {t:new Date(p[0]).getTime(), v:p[1], d:p[0], k:p[2], pos:p[3], n:p[4], name:n}; });
    pts = pts.concat(s); return {name:n, pts:s, ci:state.colorOf[n]};
  });
  var wrap = el("div", {class:"chartbox"});
  if(!pts.length){ wrap.appendChild(el("p", {class:"note", text:SNAME[sf] + "の推移データがありません。"})); return wrap; }
  var t0 = Math.min.apply(null, pts.map(function(p){ return p.t; })), t1 = Math.max.apply(null, pts.map(function(p){ return p.t; }));
  if(t1 === t0){ t0 -= 864e5 * 15; t1 += 864e5 * 15; }
  var v0 = Math.floor((Math.min.apply(null, pts.map(function(p){ return p.v; })) - 3) / 5) * 5;
  var v1 = Math.ceil((Math.max.apply(null, pts.map(function(p){ return p.v; })) + 3) / 5) * 5;
  function X(t){ return mL + (t - t0) / (t1 - t0) * (W - mL - mR); }
  function Y(v){ return mT + (v1 - v) / (v1 - v0) * (Hh - mT - mB); }
  var s = svg("svg", {viewBox:"0 0 " + W + " " + Hh, role:"img", "aria-label":SNAME[sf] + "の強さスコアの推移"});
  var step = (v1 - v0) > 40 ? 10 : 5;
  for(var v = v0; v <= v1; v += step){
    s.appendChild(svg("line", {x1:mL, x2:W - mR, y1:Y(v), y2:Y(v), stroke:"#1e2b4a", "stroke-width":1}));
    var tx = svg("text", {x:mL - 8, y:Y(v) + 4, "text-anchor":"end", fill:"#7385a8", "font-size":12});
    tx.textContent = v; s.appendChild(tx);
  }
  var d = new Date(t0); d.setDate(1); d.setMonth(d.getMonth() + 1);
  var months = Math.max(1, Math.round((t1 - t0) / (864e5 * 30))), every = Math.ceil(months / 8);
  for(var i = 0; d.getTime() <= t1; d.setMonth(d.getMonth() + 1), i++){
    if(i % every) continue;
    var lx = svg("text", {x:X(d.getTime()), y:Hh - 8, "text-anchor":"middle", fill:"#7385a8", "font-size":12});
    lx.textContent = d.getFullYear() + "-" + ("0" + (d.getMonth() + 1)).slice(-2); s.appendChild(lx);
  }
  // 終点ラベルの重なりを縦にずらして避ける
  var ends = series.filter(function(se){ return se.pts.length; }).map(function(se){
    var last = se.pts[se.pts.length - 1]; return {se:se, y:Y(last.v), x:X(last.t), v:last.v};
  }).sort(function(a, b){ return a.y - b.y; });
  for(var j = 1; j < ends.length; j++) if(ends[j].y - ends[j - 1].y < 15) ends[j].y = ends[j - 1].y + 15;
  series.forEach(function(se){
    if(!se.pts.length) return;
    var col = "var(--s" + (se.ci + 1) + ")";
    s.appendChild(svg("path", {d: se.pts.map(function(p, k){ return (k ? "L" : "M") + X(p.t).toFixed(1) + " " + Y(p.v).toFixed(1); }).join(""),
      fill:"none", stroke:col, "stroke-width":2, "stroke-linejoin":"round", "stroke-linecap":"round"}));
    se.pts.forEach(function(p){
      s.appendChild(svg("circle", {cx:X(p.t), cy:Y(p.v), r:4, fill:col, stroke:"#0a101e", "stroke-width":2}));
    });
  });
  ends.forEach(function(e){
    s.appendChild(svg("rect", {x:W - mR + 10, y:e.y - 5, width:10, height:10, rx:2, fill:"var(--s" + (e.se.ci + 1) + ")"}));
    var tl = svg("text", {x:W - mR + 26, y:e.y + 4, fill:"#edf1fb", "font-size":12.5});
    tl.textContent = (e.se.name.length > 8 ? e.se.name.slice(0, 8) + "…" : e.se.name) + " " + e.v.toFixed(1);
    s.appendChild(tl);
  });
  var cross = svg("line", {y1:mT, y2:Hh - mB, stroke:"#32456e", "stroke-width":1, visibility:"hidden"});
  s.appendChild(cross);
  var tip = el("div", {class:"tip", hidden:""});
  s.addEventListener("mousemove", function(ev){
    var r = s.getBoundingClientRect(), mx = (ev.clientX - r.left) / r.width * W, my = (ev.clientY - r.top) / r.height * Hh;
    var best = null, bd = 1e9;
    pts.forEach(function(p){ var dx = X(p.t) - mx, dy = Y(p.v) - my, dd = dx * dx + dy * dy; if(dd < bd){ bd = dd; best = p; } });
    if(!best || bd > 60 * 60){ tip.hidden = true; cross.setAttribute("visibility", "hidden"); return; }
    cross.setAttribute("x1", X(best.t)); cross.setAttribute("x2", X(best.t)); cross.setAttribute("visibility", "visible");
    tip.textContent = best.name + "｜" + best.d + " " + (best.k || "") + " " + best.pos + "着/" + best.n + "頭 → " + best.v.toFixed(1);
    tip.hidden = false;
    var px = X(best.t) / W * r.width, py = Y(best.v) / Hh * r.height;
    tip.style.left = Math.min(px + 12, r.width - tip.offsetWidth - 4) + "px";
    tip.style.top = Math.max(py - 34, 0) + "px";
  });
  s.addEventListener("mouseleave", function(){ tip.hidden = true; cross.setAttribute("visibility", "hidden"); });
  wrap.appendChild(s); wrap.appendChild(tip);
  return wrap;
}

function viewCmp(){
  var box = el("div");
  var p = el("section", {class:"panel"});
  p.appendChild(el("h2", null, ["馬を選ぶ", el("small", {text:"最大 " + MAXSEL + " 頭。5 頭目を足すと最初の 1 頭が外れます"})]));
  var sbox = el("div", {style:"position:relative"});
  var inp = el("input", {type:"text", placeholder:"馬名を入力（部分一致）", autocomplete:"off", "aria-label":"馬名検索"});
  var sug = el("div", {class:"sug", hidden:""});
  inp.addEventListener("input", function(){
    var q = inp.value.trim(); sug.textContent = "";
    if(!q){ sug.hidden = true; return; }
    var hit = PR.horses.filter(function(h){ return h[0].indexOf(q) >= 0; }).slice(0, 14);
    hit.forEach(function(h){
      var t = ["芝","ダ"].filter(function(k){ return sv(h, k); }).map(function(k){ return k + " " + f1(sv(h, k)[0]); }).join("／");
      sug.appendChild(el("button", {onclick:function(){ addSel(h[0]); render(); }}, [
        el("span", {text:h[0]}), el("small", {text:t})]));
    });
    if(!hit.length) sug.appendChild(el("button", {disabled:"", text:"該当なし（現役馬と出走表の馬だけ収録）"}));
    sug.hidden = false;
  });
  inp.addEventListener("keydown", function(ev){ if(ev.key === "Enter"){ var b = sug.querySelector("button:not([disabled])"); if(b) b.click(); } });
  sbox.appendChild(inp); sbox.appendChild(sug); p.appendChild(sbox);
  var sel = el("div", {class:"sel"});
  state.sel.forEach(function(n){
    sel.appendChild(el("span", {class:"selc"}, [
      el("span", {class:"sw", style:"background:var(--s" + (state.colorOf[n] + 1) + ")"}), n,
      el("button", {"aria-label":n + " を外す", text:"×", onclick:function(){ rmSel(n); render(); }})]));
  });
  p.appendChild(sel); box.appendChild(p);
  if(!state.sel.length){
    box.appendChild(el("p", {class:"note", text:"出走表やランキングの馬名を押しても追加できます。"}));
    return box;
  }
  var sf = state.csurf, other = sf === "芝" ? "ダ" : "芝";
  var srow = el("div", {class:"row"});
  ["芝","ダ"].forEach(function(k){
    srow.appendChild(el("button", {class:"chip", "aria-pressed": String(k === sf), text:SNAME[k] + "で比べる",
      onclick:function(){ state.csurf = k; render(); }}));
  });
  srow.appendChild(el("span", {class:"note", text:"芝とダートは別のものさしです（数字同士は比べられません）"}));
  box.appendChild(srow);
  var all = state.sel.map(function(n){ return H[n]; });
  var rows = all.filter(function(h){ return sv(h, sf); }).sort(function(a, b){ return sv(b, sf)[0] - sv(a, sf)[0]; });
  var none = all.filter(function(h){ return !sv(h, sf); });
  var t = el("table"), tb = el("tbody");
  t.appendChild(el("thead", null, [el("tr", null, [
    el("th", {text:"馬名"}), el("th", {class:"n", text:SNAME[sf] + "スコア"}), el("th", {class:"n", text:"最高"}),
    el("th", {class:"n", text:"走数"}), el("th", {text:"信頼"}), el("th", {text:"最終出走"}),
    el("th", {class:"n", text:"（" + SNAME[other] + "）"}), el("th", {text:"条件別の上振れ（走数）"})])]));
  var APT = [["s:芝","芝"],["s:ダ","ダ"],["d:短","〜1400"],["d:マ","〜1800"],["d:中","〜2200"],["d:長","2300〜"]];
  rows.concat(none).forEach(function(h){
    var v = sv(h, sf), o = sv(h, other);
    var apt = APT.filter(function(a){ return h[5][a[0]]; }).map(function(a){
      var q = h[5][a[0]]; return a[1] + " " + (q[0] >= 0 ? "+" : "") + q[0].toFixed(2) + "(" + q[1] + ")"; }).join("　");
    tb.appendChild(el("tr", null, [
      el("td", null, [el("span", {class:"sw", style:"display:inline-block;margin-right:8px;background:var(--s" + (state.colorOf[h[0]] + 1) + ")"}), h[0]]),
      el("td", {class:"n", text: v ? f1(v[0]) : "--"}), el("td", {class:"n", text: v ? f1(v[1]) : "--"}),
      el("td", {class:"n", text: v ? String(v[2]) : "0"}), el("td", {text: v ? conf(v[2]) : "-"}),
      el("td", {text: v ? v[3] + (v[4] ? "" : "（1年以上なし）") : SNAME[sf] + "は未出走"}),
      el("td", {class:"n", text: o ? f1(o[0]) : "--"}),
      el("td", {class:"note", text:apt || "記録なし"})]));
  });
  t.appendChild(tb);
  var p2 = el("section", {class:"panel"});
  p2.appendChild(el("h2", null, [SNAME[sf] + "の強さスコア", el("small", {text:SNAME[sf] + "の現役最上位 = 100。基準日 " + M.asof})]));
  p2.appendChild(el("div", {class:"scroll"}, [t]));
  box.appendChild(p2);
  if(rows.length >= 2){
    var p3 = el("section", {class:"panel"});
    p3.appendChild(el("h2", null, ["同じ" + SNAME[sf] + "のレースで先着する確率", el("small", {text:"行の馬が列の馬に先着。条件・斤量は平均的と仮定"})]));
    var mt = el("table"), hr = el("tr", null, [el("th")]);
    rows.forEach(function(h){ hr.appendChild(el("th", {class:"n", text:h[0]})); });
    mt.appendChild(el("thead", null, [hr]));
    var mb = el("tbody");
    rows.forEach(function(a){
      var tr = el("tr", null, [el("td", {text:a[0]})]);
      rows.forEach(function(b){ tr.appendChild(el("td", {class:"n", text: a === b ? "—" : pct(winProb(sv(a, sf)[0], sv(b, sf)[0], sf))})); });
      mb.appendChild(tr);
    });
    mt.appendChild(mb); p3.appendChild(el("div", {class:"scroll"}, [mt]));
    p3.appendChild(el("p", {class:"note", style:"margin-top:8px",
      text:"大差の確率はやや強気に出ます（検証タブの較正を参照）。"}));
    box.appendChild(p3);
  }
  var p4 = el("section", {class:"panel"});
  p4.appendChild(el("h2", null, [SNAME[sf] + "スコアの推移", el("small", {text:SNAME[sf] + "の各走の直後の値（今のものさしで換算）"})]));
  p4.appendChild(lineChart(state.sel, sf));
  var det = el("details"); det.appendChild(el("summary", {text:"数値の表を見る"}));
  var ht = el("table"), hb = el("tbody");
  ht.appendChild(el("thead", null, [el("tr", null, [el("th", {text:"馬名"}), el("th", {text:"日付"}), el("th", {text:"クラス"}),
    el("th", {class:"n", text:"着順"}), el("th", {class:"n", text:"頭数"}), el("th", {class:"n", text:"スコア"})])]));
  state.sel.forEach(function(n){ (H[n][HI[sf]] || []).forEach(function(q){
    hb.appendChild(el("tr", null, [el("td", {text:n}), el("td", {text:q[0]}), el("td", {text:q[2]}),
      el("td", {class:"n", text:String(q[3])}), el("td", {class:"n", text:String(q[4])}), el("td", {class:"n", text:q[1].toFixed(1)})]));
  }); });
  ht.appendChild(hb); det.appendChild(el("div", {class:"scroll"}, [ht])); p4.appendChild(det);
  box.appendChild(p4);
  return box;
}

// ---------------------------------------------------------------- ランキング
function viewRank(){
  var box = el("div"), row = el("div", {class:"row"});
  [["芝","芝"],["ダ","ダート"]].forEach(function(o){
    row.appendChild(el("button", {class:"chip", "aria-pressed": String(state.surf === o[0]), text:o[1],
      onclick:function(){ state.surf = o[0]; render(); }}));
  });
  row.appendChild(el("span", {class:"note", text:"｜最低走数"}));
  [1, 3, 5, 8].forEach(function(n){
    row.appendChild(el("button", {class:"chip", "aria-pressed": String(state.minN === n), text:n + "走〜",
      onclick:function(){ state.minN = n; render(); }}));
  });
  box.appendChild(row);
  var sf = state.surf;
  var list = PR.horses.filter(function(h){ var v = sv(h, sf); return v && v[4] && v[2] >= state.minN; })
    .sort(function(a, b){ return sv(b, sf)[0] - sv(a, sf)[0]; }).slice(0, 100);
  var sec = el("section", {class:"panel"});
  sec.appendChild(el("h2", null, [SNAME[sf] + "の現役上位 100", el("small", {text:SNAME[sf] + "で 1 年以内に出走した馬（登録抹消＝引退した馬は除く）。走数は" + SNAME[sf] + "での走数。基準日 " + M.asof})]));
  var t = el("table"), tb = el("tbody");
  t.appendChild(el("thead", null, [el("tr", null, [el("th", {class:"n", text:"順位"}), el("th", {text:"馬名"}),
    el("th", {text:SNAME[sf] + "スコア"}), el("th", {class:"n", text:"走数"}), el("th", {text:"信頼"}),
    el("th", {text:"最終出走"}), el("th", {text:"クラス"})])]));
  list.forEach(function(h, i){
    var v = sv(h, sf);
    var sc = el("td", {class:"barcell"}, [el("span", {style:"font-family:var(--disp)", text:f1(v[0])}),
      el("span", {class:"bar", style:"width:" + Math.max(0, Math.min(100, v[0])) * 1.2 + "px"})]);
    tb.appendChild(el("tr", null, [el("td", {class:"n", text:String(i + 1)}), el("td", null, [nameBtn(h[0])]), sc,
      el("td", {class:"n", text:String(v[2])}), el("td", {text:conf(v[2])}), el("td", {text:v[3]}), el("td", {text:h[4]})]));
  });
  t.appendChild(tb); sec.appendChild(el("div", {class:"scroll"}, [t])); box.appendChild(sec);
  return box;
}

// ---------------------------------------------------------------- 検証
function viewVal(){
  var box = el("div"), E = M.eval || {}, F = M.form || {};
  var s1 = el("section", {class:"panel"});
  s1.appendChild(el("h2", null, ["スコアの前向き検証", el("small", {text:(M.eval_from || "") + " 以降・発走前の値だけで予測"})]));
  if(E.races){
    var k = el("div", {class:"kpis"});
    [["スコア1位の勝率", pct(E.top_win_rate), "無作為 " + pct(E.random_win_rate) + "｜CI95 " + pct(E.top_win_ci95[0]) + "〜" + pct(E.top_win_ci95[1])],
     ["スコア1位の3着内率", pct(E.top_in3_rate), "無作為 " + pct(E.random_in3_rate) + "｜CI95 " + pct(E.top_in3_ci95[0]) + "〜" + pct(E.top_in3_ci95[1])],
     ["2頭の先着を当てた率", (E.pair_concordance * 100).toFixed(1) + "%", E.pairs.toLocaleString() + " ペア"],
     ["対象レース", E.races.toLocaleString(), "全馬がその芝ダで出走歴のあるレース"]].forEach(function(x){
      k.appendChild(el("div", {class:"kpi"}, [el("span", {text:x[0]}), el("b", {text:x[1]}), el("em", {text:x[2]})]));
    });
    s1.appendChild(k);
    var t = el("table"), tb = el("tbody");
    t.appendChild(el("thead", null, [el("tr", null, [el("th", {class:"n", text:"強い側の予測先着率"}),
      el("th", {class:"n", text:"実際"}), el("th", {class:"n", text:"差"}), el("th", {class:"n", text:"ペア数"})])]));
    (E.calibration || []).forEach(function(b){
      var d = b.actual - b.pred;
      tb.appendChild(el("tr", null, [el("td", {class:"n", text:pct(b.pred)}), el("td", {class:"n", text:pct(b.actual)}),
        el("td", {class:"n", text:(d >= 0 ? "+" : "") + Math.round(d * 100) + "pt"}), el("td", {class:"n", text:b.n.toLocaleString()})]));
    });
    t.appendChild(tb); s1.appendChild(el("div", {class:"scroll"}, [t]));
  } else s1.appendChild(el("p", {class:"note", text:"検証対象レースがありません。"}));
  box.appendChild(s1);
  var s2 = el("section", {class:"panel"});
  s2.appendChild(el("h2", null, ["好走気配 / 凡走注意の前向き検証",
    el("small", {text: F.test_rows ? "検証 " + F.test_rows.toLocaleString() + " 頭・" + F.test_races.toLocaleString() + " レース／学習 " + F.train_rows.toLocaleString() + " 頭" : ""})]));
  if(F.labels){
    var t2 = el("table"), b2 = el("tbody");
    t2.appendChild(el("thead", null, [el("tr", null, [el("th", {text:"ラベル"}), el("th", {class:"n", text:"頭数"}),
      el("th", {class:"n", text:"上振れ平均"}), el("th", {class:"n", text:"レース単位 CI95"}),
      el("th", {class:"n", text:"3着内: 見込み"}), el("th", {class:"n", text:"実際"})])]));
    ["好走気配","無印","凡走注意"].forEach(function(nm){
      var v = F.labels[nm]; if(!v) return;
      b2.appendChild(el("tr", null, [el("td", {text:nm}), el("td", {class:"n", text:v.n.toLocaleString()}),
        el("td", {class:"n", text:(v.mean_resid >= 0 ? "+" : "") + v.mean_resid.toFixed(3)}),
        el("td", {class:"n", text:v.resid_ci95[0].toFixed(3) + " 〜 " + v.resid_ci95[1].toFixed(3)}),
        el("td", {class:"n", text:pct(v.top3_expected)}), el("td", {class:"n", text:pct(v.top3_actual)})]));
    });
    t2.appendChild(b2); s2.appendChild(el("div", {class:"scroll"}, [t2]));
    s2.appendChild(el("p", {class:"note", style:"margin-top:8px",
      text:"使った条件: " + (F.features || []).join(", ") + "｜3着内 logloss " + F.test_top3_logloss_base + " → " + F.test_top3_logloss_adj + "（補正あり）"}));
  } else s2.appendChild(el("p", {class:"note", text:"データ不足でモデルなし。"}));
  box.appendChild(s2);
  var s3 = el("section", {class:"panel"});
  s3.appendChild(el("h2", null, ["読み方と限界"]));
  var ul = el("ul", {class:"plain"});
  ["着順だけから作った能力値です。オッズ・人気・時計は使っていません。馬券用の指標ではなく、買い目やサイト本体には一切つないでいません。",
   "芝とダートは別々のレーティングです（芝の現役最上位 = 100、ダートの現役最上位 = 100）。芝スコアとダートスコアの数字同士は比べられません。",
   "スコアは線形です。芝は 10 点差で先着率 約 " + pct(winProb(10, 0, "芝")) + "、20 点差で 約 " + pct(winProb(20, 0, "芝")) + "。ダートは 10 点差で 約 " + pct(winProb(10, 0, "ダ")) + "、20 点差で 約 " + pct(winProb(20, 0, "ダ")) + "。",
   "その芝ダでの走数が 3 以下だと値が荒れます（信頼「低」）。距離・斤量の違いはスコア本体では区別していません。",
   "好走気配／凡走注意は、予測上振れの上位／下位 " + Math.round((M.form_q || 0.15) * 100) + "%。「スコアの見込みより少し上／下に来やすい」程度の目印です。",
   "集計: " + M.n_races.toLocaleString() + " レース・" + M.n_horses.toLocaleString() + " 頭、基準日 " + M.asof + "、ページ生成 " + M.built_at + "。"]
    .forEach(function(x){ ul.appendChild(el("li", {text:x})); });
  s3.appendChild(ul); box.appendChild(s3);
  return box;
}

function render(){
  document.getElementById("asof").textContent = "基準日 " + M.asof;
  Array.prototype.forEach.call(document.querySelectorAll("#tabs button"), function(b){
    b.setAttribute("aria-selected", String(b.dataset.t === state.tab));
  });
  main.textContent = "";
  main.appendChild({card:viewCard, cmp:viewCmp, rank:viewRank, val:viewVal}[state.tab]());
}
document.getElementById("tabs").addEventListener("click", function(ev){
  var b = ev.target.closest("button"); if(!b) return; state.tab = b.dataset.t; render();
});
render();
})();
</script>
</body>
</html>
"""
