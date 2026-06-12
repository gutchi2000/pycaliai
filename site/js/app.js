/* ============================================================
   PyCaLiAI static site — app.js
   data/manifest.json と data/{date}.json (build_site.py 生成) を
   読み込んで描画する。フレームワーク非依存の vanilla JS。
   ============================================================ */
"use strict";

const $ = (sel) => document.querySelector(sel);

const state = {
  manifest: null,
  day: null,        // 選択中日の view-model
  place: null,
  raceId: null,
  sort: "umaban",   // "umaban" | "ai"
};
const dayCache = new Map();

/* ---------------- utils ---------------- */
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}
const WD = ["日", "月", "火", "水", "木", "金", "土"];
function fmtDate(d8) {
  const y = +d8.slice(0, 4), m = +d8.slice(4, 6), d = +d8.slice(6, 8);
  const wd = WD[new Date(y, m - 1, d).getDay()];
  return `${m}/${d} (${wd})`;
}
function pct(v, digits = 1) {
  return v == null ? "—" : (v * 100).toFixed(digits);
}
function num(v, digits = 1) {
  return v == null ? "—" : (+v).toFixed(digits);
}

/* ---------------- boot ---------------- */
async function boot() {
  $("#raceHeader").innerHTML = `<div class="loading">LOADING…</div>`;
  let mf;
  try {
    mf = await (await fetch("data/manifest.json")).json();
  } catch (e) {
    $("#raceHeader").innerHTML =
      `<div class="err">data/manifest.json を読めませんでした。<br>` +
      `python build_site.py を実行してから、HTTP サーバー経由で開いてください。</div>`;
    return;
  }
  state.manifest = mf;
  const sel = $("#dateSel");
  sel.innerHTML = mf.dates.map((d) =>
    `<option value="${d.date}">${fmtDate(d.date)}</option>`).join("");
  sel.onchange = () => loadDay(sel.value);
  $("#footInfo").textContent =
    `PyCaLiAI ${mf.model} — 静的ビルド ${mf.built_at}`;
  if (mf.dates.length) loadDay(mf.dates[0].date);
}

async function loadDay(date) {
  $("#raceHeader").innerHTML = `<div class="loading">LOADING…</div>`;
  $("#shutsuba").innerHTML = "";
  $("#cowork").innerHTML = "";
  let day = dayCache.get(date);
  if (!day) {
    try {
      day = await (await fetch(`data/${date}.json`)).json();
    } catch (e) {
      $("#raceHeader").innerHTML = `<div class="err">data/${date}.json を読めませんでした。</div>`;
      return;
    }
    dayCache.set(date, day);
  }
  state.day = day;
  state.place = day.places[0] ?? null;
  const first = racesOf(state.place)[0];
  state.raceId = first ? first.race_id : null;
  renderNav();
  renderRace();
}

function racesOf(place) {
  return state.day.races
    .filter((r) => r.place === place)
    .sort((a, b) => (a.rno ?? 99) - (b.rno ?? 99));
}
function currentRace() {
  return state.day.races.find((r) => r.race_id === state.raceId);
}

/* ---------------- judgment / market mappings ---------------- */
function judgeView(j) {
  const cat = (j?.category || "").toLowerCase();
  if (cat === "go") return { cls: "go", label: "GO" };
  if (cat === "caution") return { cls: "caution", label: "慎重" };
  if (cat.includes("skip") || cat.includes("pass") || (j?.headline || "").includes("見送")) {
    return { cls: "skip", label: "見送り" };
  }
  return { cls: "na", label: j?.category ? esc(j.category) : "—" };
}
const JDOT = { go: "#2dd4a8", caution: "#f0a132", skip: "#f2555a", na: "#5c6c8e" };

function vsChip(v) {
  if (v === "under") return `<span class="vschip under">妙味</span>`;
  if (v === "over") return `<span class="vschip over">過剰</span>`;
  return `<span class="vschip fair">中立</span>`;
}
function markCls(m) {
  return { "◎": "m1", "〇": "m2", "○": "m2", "▲": "m3", "△": "m4" }[m] || "m0";
}
function subLine(h) {
  const sexage = `${h.sex || ""}${h.age ?? ""}`;
  const sire = h.pedigree?.sire ? `父${h.pedigree.sire}` : "";
  return [sexage, sire].filter(Boolean).join("・") || "—";
}

/* ---------------- nav ---------------- */
function renderNav() {
  const vt = $("#venueTabs");
  vt.innerHTML = state.day.places.map((p) => {
    const n = racesOf(p).length;
    return `<button class="venue ${p === state.place ? "on" : ""}" data-place="${esc(p)}">
      ${esc(p)}<small>${n}R</small></button>`;
  }).join("");
  vt.querySelectorAll(".venue").forEach((b) => {
    b.onclick = () => {
      state.place = b.dataset.place;
      const first = racesOf(state.place)[0];
      state.raceId = first ? first.race_id : null;
      renderNav();
      renderRace();
    };
  });

  const rs = $("#raceStrip");
  rs.innerHTML = racesOf(state.place).map((r) => {
    const jv = judgeView(r.judgment);
    const cls = (r.klass || "").slice(0, 6);
    return `<button class="rpill ${r.race_id === state.raceId ? "on" : ""}"
        data-rid="${r.race_id}" title="${esc(r.judgment?.headline || "")}">
      <span class="rno">${r.rno}R</span>
      <span class="rcls">${esc(cls)}</span>
      <span class="rdot" style="background:${JDOT[jv.cls]}"></span>
    </button>`;
  }).join("");
  rs.querySelectorAll(".rpill").forEach((b) => {
    b.onclick = () => {
      state.raceId = b.dataset.rid;
      rs.querySelectorAll(".rpill").forEach((x) => x.classList.toggle("on", x === b));
      renderRace();
    };
  });
}

/* ---------------- race header ---------------- */
function donut(label, frac, color) {
  const C = (2 * Math.PI * 20).toFixed(1);
  const v = Math.max(0, Math.min(1, frac ?? 0));
  return `<div class="gauge">
    <svg viewBox="0 0 48 48" width="56" height="56" aria-hidden="true">
      <circle class="g-track" cx="24" cy="24" r="20"></circle>
      <circle class="g-fill" cx="24" cy="24" r="20"
        style="stroke:${color};stroke-dasharray:${C};stroke-dashoffset:${C};--off:${(C * (1 - v)).toFixed(1)}"></circle>
    </svg>
    <div class="g-val num">${Math.round(v * 100)}<small>%</small></div>
    <div class="g-label">${label}</div>
  </div>`;
}

function renderHeader(r) {
  const j = r.judgment || {};
  const jv = judgeView(j);
  const isTurf = (r.course || "").startsWith("芝");
  const cw = r.cowork;
  const conf = r.confidence || {};
  $("#raceHeader").innerHTML = `<div class="card rh">
    <div style="min-width:0">
      <div class="rh-title">
        <span class="rh-place">${esc(r.place)}</span>
        <span class="rh-rno">${r.rno}R</span>
        ${r.race_name ? `<span class="rh-name">${esc(r.race_name)}</span>` : ""}
      </div>
      <div class="rh-sub">
        <span class="tdchip ${isTurf ? "turf" : "dirt"}">${esc(r.course)}</span>
        <span class="mchip">${esc(r.klass)}</span>
        <span class="mchip">${r.field_size}頭</span>
      </div>
      <div class="judge">
        <span class="jbadge ${jv.cls}">${jv.label}</span>
        ${j.headline ? `<span class="jdetail"><b>${esc(j.headline)}</b>　${esc(j.detail || "")}</span>` : ""}
        ${j.hardness ? `<span class="jtag">${esc(j.hardness)}</span>` : ""}
        ${j.waku_tag ? `<span class="jtag">${esc(j.waku_tag)}</span>` : ""}
      </div>
    </div>
    <div class="gauges">
      ${donut("本命優位", conf.top1_dominance, "#f5b942")}
      ${donut("上位集中", conf.top2_concentration, "#5ba0f5")}
      ${donut("混戦度", conf.field_chaos_score, "#f2555a")}
      ${donut("市場一致", conf.ai_market_agreement, "#2dd4a8")}
    </div>
    ${cw?.race_reason ? `<div class="rh-quote"><b>COWORK</b>${esc(cw.race_reason)}</div>` : ""}
  </div>`;
}

/* ---------------- shutsuba table ---------------- */
function sortedHorses(r) {
  const hs = [...r.horses];
  if (state.sort === "ai") {
    hs.sort((a, b) => (a.ai_rank ?? 99) - (b.ai_rank ?? 99));
  } else {
    hs.sort((a, b) => (a.umaban ?? 99) - (b.umaban ?? 99));
  }
  return hs;
}

function renderTable(r) {
  const valueSet = new Set((r.judgment?.value_horses || []).map((v) => v.umaban));
  const maxP = Math.max(...r.horses.map((h) => h.p_win ?? 0), 0.001);
  const hs = sortedHorses(r);

  const rows = hs.map((h, i) => {
    const isHonmei = h.mark === "◎";
    const isValue = valueSet.has(h.umaban) || h.vs_market === "under";
    const w = ((h.p_win ?? 0) / maxP * 100).toFixed(1);
    const ev = h.ev_tan;
    return `<div class="hrow ${isHonmei ? "honmei" : ""} ${!h.mark ? "dim" : ""} ${isValue ? "value" : ""}"
        style="--i:${i}" data-uma="${h.umaban}" role="button" tabindex="0">
      <span class="mark ${markCls(h.mark)}">${h.mark || "・"}</span>
      <span><span class="wk w${h.waku ?? 1}">${h.umaban}</span></span>
      <span style="min-width:0">
        <span class="hname">${esc(h.name)}</span><br>
        <span class="hsub">${esc(subLine(h))}</span>
      </span>
      <span class="pbar col-bar"><span class="rk">#${h.ai_rank ?? "—"}</span><span class="bar"><i style="width:${w}%;--i:${i}"></i></span></span>
      <span class="pwin num ta-r">${pct(h.p_win)}<small>%</small></span>
      <span class="psho num ta-r col-sho">${pct(h.p_sho, 0)}<small>%</small></span>
      <span class="odds num ta-r col-odds">${num(h.odds)}${ev != null && ev >= 1.2 ? `<span class="oddsev">EV ${ev.toFixed(2)}</span>` : ""}</span>
      <span class="ta-r col-vs">${vsChip(h.vs_market)}</span>
    </div>`;
  }).join("");

  $("#shutsuba").innerHTML = `
    <div class="sh-head">
      <span class="sh-title"><b>AI印</b>出走表</span>
      <div class="sortseg">
        <button data-sort="umaban" class="${state.sort === "umaban" ? "on" : ""}">馬番順</button>
        <button data-sort="ai" class="${state.sort === "ai" ? "on" : ""}">AI評価順</button>
      </div>
    </div>
    <div class="card htable">
      <div class="hh">
        <span>印</span><span>馬番</span><span>馬名</span>
        <span class="col-bar">AI評価</span><span class="ta-r">勝率</span>
        <span class="ta-r col-sho">複勝圏</span><span class="ta-r col-odds">単オッズ</span><span class="ta-r col-vs">市場</span>
      </div>
      ${rows}
    </div>`;

  $("#shutsuba").querySelectorAll(".sortseg button").forEach((b) => {
    b.onclick = () => { state.sort = b.dataset.sort; renderTable(currentRace()); };
  });
  $("#shutsuba").querySelectorAll(".hrow").forEach((row) => {
    const open = () => openDrawer(+row.dataset.uma);
    row.onclick = open;
    row.onkeydown = (e) => { if (e.key === "Enter") open(); };
  });
}

/* ---------------- cowork section ---------------- */
const BET_COLOR = {
  "単勝": "#f5b942", "複勝": "#2dd4a8", "ワイド": "#5ba0f5",
  "馬連": "#b78cf2", "馬単": "#e4549a", "三連複": "#f0a132",
};
function tagCls(tag) {
  if (!tag) return "t-etc";
  if (tag.includes("軸")) return "t-jiku";
  if (tag.includes("妙味")) return "t-myomi";
  if (tag.includes("罠")) return "t-wana";
  return "t-etc";
}

function renderCowork(r) {
  const cw = r.cowork;
  if (!cw || (!cw.bets?.length && !cw.advisor?.length)) {
    $("#cowork").innerHTML = `<div class="cw">
      <div class="cw-title"><b>COWORK</b>買い目・見解</div>
      <div class="card cw-empty">このレースの Cowork 出力はありません。</div>
    </div>`;
    return;
  }
  const tickets = (cw.bets || []).map((b, i) => {
    const col = BET_COLOR[b.type] || "#97a4c2";
    const amt = typeof b.amount === "number" ? b.amount.toLocaleString() : esc(b.amount);
    return `<div class="card ticket" style="--bcol:${col};--i:${i}">
      <div class="ticket-type">${esc(b.type)}</div>
      <div class="ticket-sel">${esc(b.selection)}</div>
      <div class="ticket-amt"><b>¥${amt}</b></div>
      ${b.reason ? `<div class="ticket-reason">${esc(b.reason)}</div>` : ""}
    </div>`;
  }).join("");

  const advisors = (cw.advisor || []).map((a, i) => {
    const g = ["A", "B", "C"].includes(a.grade) ? a.grade : "X";
    const horse = r.horses.find((h) => h.umaban === a.umaban);
    return `<div class="card adv" style="--i:${i}">
      <div class="adv-medal g${g}">${esc(a.grade || "—")}</div>
      <div class="adv-body">
        <div class="adv-head">
          ${horse ? `<span class="wk w${horse.waku}">${horse.umaban}</span>` : ""}
          <span class="adv-name">${esc(a.horse_name)}</span>
          ${a.tag ? `<span class="adv-tag ${tagCls(a.tag)}">${esc(a.tag)}</span>` : ""}
        </div>
        <div class="adv-comment">${esc(a.comment)}</div>
      </div>
    </div>`;
  }).join("");

  $("#cowork").innerHTML = `<div class="cw">
    ${tickets ? `<div class="cw-title"><b>COWORK</b>買い目<span class="cw-src">${esc(cw.source || "")}</span></div>
      <div class="bet-grid">${tickets}</div>` : ""}
    ${advisors ? `<div class="cw-title"><b>COWORK</b>全頭見解${tickets ? "" : `<span class="cw-src">${esc(cw.source || "")}</span>`}</div>
      <div class="adv-grid">${advisors}</div>` : ""}
  </div>`;
}

/* ---------------- drawer ---------------- */
function openDrawer(umaban) {
  const r = currentRace();
  const h = r.horses.find((x) => x.umaban === umaban);
  if (!h) return;

  const whys = (h.why || []).slice(0, 6);
  const maxC = Math.max(...whys.map((w) => Math.abs(w.contrib ?? 0)), 0.001);
  const whyHtml = whys.map((w, i) => {
    const neg = (w.contrib ?? 0) < 0;
    const width = (Math.abs(w.contrib ?? 0) / maxC * 100).toFixed(0);
    const val = w.value == null ? "—" : (typeof w.value === "number" ? +(+w.value).toFixed(1) : esc(w.value));
    return `<div class="why ${neg ? "neg" : ""}" style="--i:${i}">
      <span class="wl">${esc(w.label)} <small>${val}</small></span>
      <span class="bar"><i style="width:${width}%;--i:${i};${neg ? "" : "background:linear-gradient(90deg,#c98f1f,#ffd97a)"}"></i></span>
      <span class="wv">${(w.contrib >= 0 ? "+" : "") + num(w.contrib, 2)}</span>
    </div>`;
  }).join("");

  const runs = (h.history?.runs || []).slice(0, 5);
  const runsHtml = runs.map((u) => {
    const p = u.pos;
    const pcls = p === 1 ? "p1" : (p === 2 || p === 3) ? "p23" : "px";
    return `<div class="run ${p <= 3 ? "best" : ""}">
      <div class="pos ${pcls}">${p ?? "—"}</div>
      <div class="rc">${esc(u.place || "")}${esc(u.td || "")}${u.dist ?? ""}</div>
      <div class="rc">${esc(u.track || "")}・${u.ninki ?? "—"}人気</div>
      <div class="rc">上り${num(u.agari3f)}</div>
    </div>`;
  }).join("");

  const ped = h.pedigree || {};
  $("#drawer").innerHTML = `
    <button class="dw-close" id="dwClose" aria-label="閉じる">✕</button>
    <div class="dw-head">
      <span class="wk w${h.waku}">${h.umaban}</span>
      <span class="dw-mark mark ${markCls(h.mark)}">${h.mark || ""}</span>
      <span class="dw-name">${esc(h.name)}</span>
    </div>
    <div class="dw-sub">${esc(h.sex)}${h.age ?? ""} ・ AIランク #${h.ai_rank ?? "—"} ・ スコア ${num(h.ai_score, 3)}</div>
    <div class="dw-ped">父 <b>${esc(ped.sire || "—")}</b> ／ 母父 <b>${esc(ped.broodmare_sire || "—")}</b>
      ${ped.broodmare_sire_type ? `（${esc(ped.broodmare_sire_type)}）` : ""}</div>
    <div class="dw-stats">
      <div class="dw-stat"><div class="v ${h.mark === "◎" ? "gold" : ""}">${pct(h.p_win)}%</div><div class="k">勝率</div></div>
      <div class="dw-stat"><div class="v">${pct(h.p_plc)}%</div><div class="k">連対率</div></div>
      <div class="dw-stat"><div class="v">${pct(h.p_sho)}%</div><div class="k">複勝圏</div></div>
      <div class="dw-stat"><div class="v ${h.ev_tan >= 1.2 ? "teal" : ""}">${num(h.ev_tan, 2)}</div><div class="k">単勝EV</div></div>
    </div>
    <div style="font-size:11.5px;color:var(--tx2)">単勝 <b class="num">${num(h.odds)}</b> 倍 ／
      複勝 <b class="num">${num(h.fuku_low)}〜${num(h.fuku_high)}</b> 倍 ／ ${vsChip(h.vs_market)}</div>
    ${whyHtml ? `<div class="dw-sec">AI の根拠（特徴量寄与）</div>${whyHtml}` : ""}
    ${runsHtml ? `<div class="dw-sec">近 5 走（左が最新）</div><div class="runs">${runsHtml}</div>` : ""}
    <div class="dw-note">勝率・複勝圏は v6 calibrator 補正後の Plackett-Luce 確率。EV = 勝率 × 単勝オッズ。</div>`;

  $("#dwClose").onclick = closeDrawer;
  $("#overlay").classList.add("show");
  $("#drawer").classList.add("show");
  $("#drawer").scrollTop = 0;
}
function closeDrawer() {
  $("#overlay").classList.remove("show");
  $("#drawer").classList.remove("show");
}
$("#overlay").onclick = closeDrawer;
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });

/* ---------------- compose ---------------- */
function renderRace() {
  const r = currentRace();
  if (!r) {
    $("#raceHeader").innerHTML = `<div class="err">レースがありません</div>`;
    $("#shutsuba").innerHTML = "";
    $("#cowork").innerHTML = "";
    return;
  }
  renderHeader(r);
  renderTable(r);
  renderCowork(r);
}

boot();
