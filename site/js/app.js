/* ============================================================
   PyCaLiAI static site — app.js (v2)
   data/manifest.json と data/{date}.json (build_site.py 生成) を
   読み込んで描画する。フレームワーク非依存の vanilla JS。
   ============================================================ */
"use strict";

const $ = (sel) => document.querySelector(sel);

const state = {
  manifest: null,
  day: null,
  place: null,
  raceId: null,
  sort: "uma",
  sortAsc: true,
  view: "shutsuba",
};
const dayCache = new Map();
let charts = [];

/* ---------------- utils ---------------- */
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}
const WD = ["日", "月", "火", "水", "木", "金", "土"];
function fmtDate(d8) {
  const y = +d8.slice(0, 4), m = +d8.slice(4, 6), d = +d8.slice(6, 8);
  return `${m}/${d} (${WD[new Date(y, m - 1, d).getDay()]})`;
}
function pct(v, digits = 1) {
  return v == null ? "—" : (v * 100).toFixed(digits);
}
function num(v, digits = 1) {
  return v == null ? "—" : (+v).toFixed(digits);
}
function yen(v) {
  return v == null ? "—" : "¥" + (+v).toLocaleString();
}

/* ---------------- boot ---------------- */
async function boot() {
  $("#raceHeader").innerHTML = `<div class="loading">LOADING…</div>`;
  let mf;
  try {
    mf = await (await fetch("data/manifest.json?t=" + Date.now())).json();
  } catch (e) {
    $("#raceHeader").innerHTML =
      `<div class="err">data/manifest.json を読めませんでした。<br>` +
      `python build_site.py を実行してから、HTTP サーバー経由で開いてください。</div>`;
    return;
  }
  state.manifest = mf;
  const sel = $("#dateSel");
  sel.innerHTML = mf.dates.map((d) =>
    `<option value="${d.date}">${fmtDate(d.date)}${d.has_results ? " ✓" : ""}</option>`).join("");
  sel.onchange = () => loadDay(sel.value);
  $("#footInfo").textContent =
    `PyCaLiAI ${mf.model} — 静的ビルド ${mf.built_at} ／ ← → キーでレース移動`;
  if (mf.dates.length) loadDay(mf.dates[0].date);
}

async function loadDay(date) {
  $("#raceHeader").innerHTML = `<div class="card skel">
    <div class="skel-bar" style="width:34%"></div>
    <div class="skel-bar" style="width:72%"></div>
    <div class="skel-bar" style="width:55%"></div>
  </div>`;
  $("#viewbody").innerHTML = `<div class="card skel">
    ${`<div class="skel-bar"></div>`.repeat(6)}
  </div>`;
  let day = dayCache.get(date);
  if (!day) {
    const v = encodeURIComponent(state.manifest?.built_at || "0");
    try {
      day = await (await fetch(`data/${date}.json?v=${v}`)).json();
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
function allRacesFlat() {
  return state.day.places.flatMap((p) => racesOf(p));
}
function currentRace() {
  return state.day.races.find((r) => r.race_id === state.raceId);
}

/* ---------------- mappings ---------------- */
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
function wk(h) {
  return `<span class="wk w${h.waku ?? 1}">${h.umaban}</span>`;
}
function posBadge(pos) {
  if (pos == null) return "";
  const cls = pos === 1 ? "r1" : pos === 2 ? "r2" : pos === 3 ? "r3" : "rx";
  return `<span class="resb ${cls}">${pos}</span>`;
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
      renderRaceVT();
    };
  });

  const rs = $("#raceStrip");
  rs.innerHTML = racesOf(state.place).map((r) => {
    const jv = judgeView(r.judgment);
    const hit = r.result && r.horses.some((h) => h.mark === "◎" && r.result.top3.includes(h.umaban));
    return `<button class="rpill ${r.race_id === state.raceId ? "on" : ""}"
        data-rid="${r.race_id}" title="${esc(r.judgment?.headline || "")}">
      <span class="rno">${r.rno}R${hit ? `<i class="rhit">◎</i>` : ""}</span>
      <span class="rcls">${esc((r.klass || "").slice(0, 6))}</span>
      <span class="rdot" style="background:${JDOT[jv.cls]}"></span>
    </button>`;
  }).join("");
  rs.querySelectorAll(".rpill").forEach((b) => {
    b.onclick = () => selectRace(b.dataset.rid);
  });
}

function selectRace(rid) {
  state.raceId = rid;
  const r = currentRace();
  if (r && r.place !== state.place) {
    state.place = r.place;
    renderNav();
  } else {
    document.querySelectorAll(".rpill").forEach((x) =>
      x.classList.toggle("on", x.dataset.rid === rid));
  }
  renderRaceVT();
}

function renderRaceVT() {
  if (document.startViewTransition) {
    document.startViewTransition(() => renderRace());
  } else {
    renderRace();
  }
}

/* ---------------- race header ---------------- */
function donut(label, frac, color) {
  const C = (2 * Math.PI * 20).toFixed(1);
  const v = Math.max(0, Math.min(1, frac ?? 0));
  return `<div class="gauge">
    <svg viewBox="0 0 48 48" width="66" height="66" aria-hidden="true">
      <circle class="g-track" cx="24" cy="24" r="20"></circle>
      <circle class="g-fill" cx="24" cy="24" r="20"
        style="stroke:${color};stroke-dasharray:${C};stroke-dashoffset:${C};--off:${(C * (1 - v)).toFixed(1)}"></circle>
    </svg>
    <div class="g-val num"><span class="cv" data-cv="${Math.round(v * 100)}">0</span><small>%</small></div>
    <div class="g-label">${label}</div>
  </div>`;
}

function runCounters(scope) {
  scope.querySelectorAll(".cv").forEach((el) => {
    const target = +el.dataset.cv || 0;
    const t0 = performance.now(), dur = 800;
    const tick = (t) => {
      const p = Math.min((t - t0) / dur, 1);
      el.textContent = Math.round(target * (1 - Math.pow(1 - p, 3)));
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  });
}

function priorStrip(r) {
  const cp = r.class_prior;
  if (!cp) return "";
  return `<div class="prior">
    <span class="pr-t">クラス実績<small>${esc(r.klass)} n=${(cp.n_samples ?? 0).toLocaleString()}</small></span>
    <span class="pr-i"><b class="m1c">◎</b> 勝${num(cp.hon_1st_pct, 1)}% 連${num(cp.hon_top2_pct, 1)}% 複${num(cp.hon_top3_pct, 1)}%</span>
    <span class="pr-i"><b>〇</b> 複${num(cp.tai_top3_pct, 1)}%</span>
    <span class="pr-i"><b>▲</b> 複${num(cp.san_top3_pct, 1)}%</span>
    <span class="pr-i"><b>△</b> 複${num(cp.del1_top3_pct, 1)}%</span>
  </div>`;
}

function resultBanner(r) {
  const res = r.result;
  if (!res) return "";
  const byPos = {};
  Object.entries(res.order).forEach(([u, p]) => { byPos[p] = +u; });
  const top = [1, 2, 3].map((p) => {
    const u = byPos[p];
    const h = r.horses.find((x) => x.umaban === u);
    if (!h) return "";
    return `<span class="res-h">${posBadge(p)}${wk(h)}
      <span class="res-n">${esc(h.name)}</span>
      ${h.mark ? `<span class="mark ${markCls(h.mark)}" style="font-size:13px">${h.mark}</span>` : ""}</span>`;
  }).join("");
  const pays = res.pays || {};
  const fuku = Object.entries(pays.fuku || {}).map(([u, v]) => `${u}番 ${yen(v)}`).join("・");
  const wide = Object.entries(pays.wide || {}).map(([k, v]) => `${k} ${yen(v)}`).join("・");
  const honmei = r.horses.find((h) => h.mark === "◎");
  const hPos = honmei ? res.order[String(honmei.umaban)] : null;
  const hitChip = hPos === 1 ? `<span class="hitchip">◎ 1着的中</span>`
    : hPos != null && hPos <= 3 ? `<span class="hitchip">◎ 複勝圏 (${hPos}着)</span>`
    : honmei ? `<span class="hitchip miss">◎ ${hPos ?? "—"}着</span>` : "";
  return `<div class="resbar">
    <span class="res-t">RESULT</span>${top}${hitChip}
    <span class="res-pays">
      <i>単勝 ${yen(pays.tan)}</i>${fuku ? `<i>複勝 ${fuku}</i>` : ""}
      <i>馬連 ${yen(pays.umaren)}</i>${wide ? `<i>ワイド ${wide}</i>` : ""}
      <i>馬単 ${yen(pays.umatan)}</i><i>三連複 ${yen(pays.sanrenpuku)}</i>
    </span>
  </div>`;
}

function renderHeader(r) {
  const j = r.judgment || {};
  const jv = judgeView(j);
  const isTurf = (r.course || "").startsWith("芝");
  const conf = r.confidence || {};
  const baba = (r.baba || "").replace("(暫定)", "");
  const weather = (r.weather || "").replace("(暫定)", "");
  $("#raceHeader").innerHTML = `<div class="card rh">
    <div class="rh-main">
      <div class="rh-title">
        <span class="rh-place">${esc(r.place)}</span>
        <span class="rh-rno">${r.rno}R</span>
        ${r.race_name ? `<span class="rh-name">${esc(r.race_name)}</span>` : ""}
        ${r.start_time ? `<span class="rh-time num">${esc(r.start_time)} 発走</span>` : ""}
      </div>
      <div class="rh-sub">
        <span class="tdchip ${isTurf ? "turf" : "dirt"}">${esc(r.course)}</span>
        <span class="mchip">${esc(r.klass)}</span>
        <span class="mchip">${r.field_size}頭</span>
        ${baba ? `<span class="mchip">${esc(weather)} / ${esc(baba)}</span>` : ""}
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
    ${r.cowork?.race_reason ? `<div class="rh-quote"><b>COWORK</b>${esc(r.cowork.race_reason)}</div>` : ""}
    ${priorStrip(r)}
  </div>
  ${resultBanner(r)}`;
  runCounters($("#raceHeader"));
}

/* ---------------- shutsuba table ---------------- */
const SORTS = {
  uma:   { key: (h) => h.umaban ?? 99, dir: 1 },
  ninki: { key: (h) => h.ninki ?? 99, dir: 1 },
  odds:  { key: (h) => h.odds ?? 9999, dir: 1 },
  ai:    { key: (h) => h.ai_score ?? -1, dir: -1 },
  win:   { key: (h) => h.p_win ?? -1, dir: -1 },
  sho:   { key: (h) => h.p_sho ?? -1, dir: -1 },
  ev:    { key: (h) => h.ev_tan ?? -1, dir: -1 },
};

function spark(h) {
  const runs = (h.history?.runs || []).slice(0, 5);
  if (!runs.length) return `<span class="nosp">初出走</span>`;
  const pts = [...runs].reverse();
  const W = 64, H = 26, n = pts.length;
  const x = (i) => n === 1 ? W / 2 : 5 + (W - 10) * i / (n - 1);
  const y = (p) => 4 + (H - 9) * (Math.min(p ?? 16, 16) - 1) / 15;
  const poly = pts.map((u, i) => `${x(i).toFixed(1)},${y(u.pos).toFixed(1)}`).join(" ");
  const dots = pts.map((u, i) => {
    const p = u.pos;
    const c = p === 1 ? "#f5b942" : p <= 3 ? "#2dd4a8" : "#55648a";
    return `<circle cx="${x(i).toFixed(1)}" cy="${y(u.pos).toFixed(1)}" r="${i === n - 1 ? 3 : 2}"
      fill="${c}" style="--d:${(i * 0.07).toFixed(2)}s"/>`;
  }).join("");
  const tip = runs.map((u) => `${u.n_ago}走前:${u.place || ""}${u.td || ""}${u.dist || ""} ${u.pos ?? "?"}着`).join(" ");
  return `<svg class="spark" width="64" height="26" viewBox="0 0 64 26"><title>${esc(tip)}</title>
    <polyline points="${poly}" pathLength="100" fill="none" stroke="#33415e" stroke-width="1.5"/>${dots}</svg>`;
}

function subLine(h) {
  const parts = [];
  const sexage = `${h.sex || ""}${h.age ?? ""}`;
  if (sexage) parts.push(sexage);
  if (h.kinryo) parts.push(num(h.kinryo) + "k");
  if (h.jockey) parts.push(h.jockey + (h.kawari ? "(替)" : ""));
  if (!h.jockey && h.pedigree?.sire) parts.push("父" + h.pedigree.sire);
  return parts.join(" ") || "—";
}

function renderTable(r, flip = false) {
  const prevTops = new Map();
  if (flip) {
    $("#shutsuba").querySelectorAll(".hrow").forEach((el) =>
      prevTops.set(el.dataset.uma, el.getBoundingClientRect().top));
  }
  const valueSet = new Set((r.judgment?.value_horses || []).map((v) => v.umaban));
  const maxP = Math.max(...r.horses.map((h) => h.p_win ?? 0), 0.001);
  const scores = r.horses.map((h) => h.ai_score).filter((v) => v != null);
  const sMin = scores.length ? Math.min(...scores) : 0;
  const sRange = scores.length ? Math.max(Math.max(...scores) - sMin, 1e-9) : 1;
  const hasRes = !!r.result;
  const s = SORTS[state.sort] || SORTS.uma;
  const hs = [...r.horses].sort((a, b) =>
    (s.key(a) - s.key(b)) * s.dir * (state.sortAsc ? 1 : -1));

  const rows = hs.map((h, i) => {
    const isHonmei = h.mark === "◎";
    const isValue = valueSet.has(h.umaban) || h.vs_market === "under";
    const wbar = ((h.p_win ?? 0) / maxP * 100).toFixed(1);
    const idx = h.ai_score == null ? "—"
      : Math.round((h.ai_score - sMin) / sRange * 100);
    const ev = h.ev_tan;
    const resPos = hasRes ? r.result.order[String(h.umaban)] : null;
    const inTop3 = resPos != null && resPos <= 3;
    return `<div class="hrow ${flip ? "still" : ""} ${isHonmei ? "honmei" : ""} ${!h.mark ? "dim" : ""} ${isValue ? "value" : ""} ${inTop3 ? "intop3" : ""}"
        style="--i:${i}" data-uma="${h.umaban}" role="button" tabindex="0">
      ${hasRes ? `<span class="c-res">${posBadge(resPos)}</span>` : ""}
      <span class="mark ${markCls(h.mark)}">${h.mark || "・"}</span>
      <span>${wk(h)}</span>
      <span class="hcell">
        <span class="hname">${esc(h.name)}</span>
        <span class="hsub">${esc(subLine(h))}</span>
      </span>
      <span class="c-ninki num ta-c">${h.ninki ?? "—"}</span>
      <span class="odds num ta-r c-odds">${num(h.odds)}${ev != null && ev >= 1.2 ? `<span class="oddsev">EV ${ev.toFixed(2)}</span>` : ""}</span>
      <span class="pbar c-ai">
        <span class="aidx num ${h.ai_rank === 1 ? "top" : ""}">${idx}</span>
        <span class="bar"><i style="width:${wbar}%;--i:${i}"></i></span>
        <span class="rk">#${h.ai_rank ?? "—"}</span>
      </span>
      <span class="pwin num ta-r">${pct(h.p_win)}<small>%</small></span>
      <span class="psho num ta-r c-sho">${pct(h.p_sho, 0)}<small>%</small></span>
      <span class="c-spark ta-c">${spark(h)}</span>
      <span class="ta-r c-vs">${vsChip(h.vs_market)}</span>
    </div>`;
  }).join("");

  const sortBtn = (key, label) => {
    const on = state.sort === key;
    const arrow = on ? (state.sortAsc ? "▲" : "▼") : "";
    return `<button class="hsort ${on ? "on" : ""}" data-key="${key}">${label}${arrow}</button>`;
  };

  $("#shutsuba").innerHTML = `
    <div class="sh-head">
      <span class="sh-title"><b>AI印</b>出走表</span>
      <span class="sh-note">行クリックで詳細 ／ 列見出しでソート</span>
    </div>
    <div class="card htable ${hasRes ? "hasres" : "nores"}">
      <div class="hh">
        ${hasRes ? `<span class="c-res">着</span>` : ""}
        <span>${sortBtn("uma", "印")}</span><span>馬番</span><span>馬名・騎手</span>
        <span class="ta-c c-ninki">${sortBtn("ninki", "人気")}</span>
        <span class="ta-r c-odds">${sortBtn("odds", "単オッズ")}</span>
        <span class="c-ai">${sortBtn("ai", "AI指数")}</span>
        <span class="ta-r">${sortBtn("win", "勝率")}</span>
        <span class="ta-r c-sho">${sortBtn("sho", "複勝圏")}</span>
        <span class="ta-c c-spark">近5走</span>
        <span class="ta-r c-vs">市場</span>
      </div>
      ${rows}
    </div>`;

  $("#shutsuba").querySelectorAll(".hsort").forEach((b) => {
    b.onclick = (e) => {
      e.stopPropagation();
      const key = b.dataset.key;
      if (state.sort === key) {
        state.sortAsc = !state.sortAsc;
      } else {
        state.sort = key;
        state.sortAsc = true;
      }
      renderTable(currentRace(), true);
    };
  });
  $("#shutsuba").querySelectorAll(".hrow").forEach((row) => {
    const open = () => openDrawer(+row.dataset.uma);
    row.onclick = open;
    row.onkeydown = (e) => { if (e.key === "Enter") open(); };
  });

  if (flip) {
    $("#shutsuba").querySelectorAll(".hrow").forEach((el, i) => {
      const prev = prevTops.get(el.dataset.uma);
      if (prev == null) return;
      const d = prev - el.getBoundingClientRect().top;
      if (Math.abs(d) < 1) return;
      el.animate(
        [{ transform: `translateY(${d}px)` }, { transform: "none" }],
        { duration: 340 + i * 12, easing: "cubic-bezier(.2,.7,.3,1)" });
    });
  }
}

/* ---------------- extras: 展開 + 馬連/ワイド ---------------- */
const LANES = ["逃げ", "先行", "差し", "追込"];

function renderExtras(r) {
  // --- 展開 ---
  const lanes = { "逃げ": [], "先行": [], "差し": [], "追込": [] };
  const unk = [];
  r.horses.forEach((h) => (h.style ? lanes[h.style] : unk).push(h));
  const nFront = lanes["逃げ"].length + lanes["先行"].length;
  const ratio = nFront / Math.max(r.horses.length, 1);
  const pace = lanes["逃げ"].length >= 3 || ratio >= 0.5 ? ["ハイ想定", "#f2555a"]
    : ratio >= 0.3 ? ["ミドル想定", "#f0a132"] : ["スロー想定", "#2dd4a8"];
  const laneHtml = LANES.map((ln) => {
    const chips = lanes[ln]
      .sort((a, b) => a.umaban - b.umaban)
      .map((h) => `<span class="tk ${h.mark === "◎" ? "tk-hon" : h.mark ? "tk-mk" : ""}"
          title="${esc(h.name)} (${esc(h.jockey || "")})">${wk(h)}<i>${esc(h.name)}</i></span>`)
      .join("");
    return `<div class="lane">
      <span class="lane-l">${ln}<small>${lanes[ln].length}</small></span>
      <div class="lane-c">${chips || `<span class="lane-none">—</span>`}</div>
    </div>`;
  }).join("");
  const unkHtml = unk.length
    ? `<div class="lane"><span class="lane-l">不明<small>${unk.length}</small></span>
        <div class="lane-c">${unk.map((h) => `<span class="tk" title="${esc(h.name)}">${wk(h)}<i>${esc(h.name)}</i></span>`).join("")}</div></div>`
    : "";

  // --- ペア ---
  const top2 = r.result
    ? Object.entries(r.result.order).filter(([, p]) => p <= 2).map(([u]) => +u)
    : null;
  const pairRows = (r.pairs || []).map((p, i) => {
    const ha = r.horses.find((h) => h.umaban === p.a);
    const hb = r.horses.find((h) => h.umaban === p.b);
    if (!ha || !hb) return "";
    const win = top2 && top2.includes(p.a) && top2.includes(p.b);
    return `<div class="pair ${win ? "win" : ""}" style="--i:${i}">
      <span class="pair-h">${wk(ha)}${ha.mark ? `<b class="mark ${markCls(ha.mark)}">${ha.mark}</b>` : ""}</span>
      <span class="pair-x">×</span>
      <span class="pair-h">${wk(hb)}${hb.mark ? `<b class="mark ${markCls(hb.mark)}">${hb.mark}</b>` : ""}</span>
      <span class="pair-v num">${pct(p.p_umaren)}<small>%</small></span>
      <span class="pair-v num">${pct(p.p_wide)}<small>%</small></span>
      <span class="pair-v num odds-col">${p.umaren_odds != null ? num(p.umaren_odds) : "—"}<small>倍</small></span>
      ${win ? `<span class="pair-win">的中 ${yen(r.result.pays.umaren)}</span>` : ""}
    </div>`;
  }).join("");

  $("#extras").innerHTML = `<div class="ex-grid">
    <div class="card ex">
      <div class="ex-t">展開予想 <span class="pace" style="color:${pace[1]};border-color:${pace[1]}">${pace[0]}</span>
        <small>近5走の脚質から推定</small></div>
      ${laneHtml}${unkHtml}
    </div>
    <div class="card ex">
      <div class="ex-t">馬連・ワイド 妙味 <small>AI確率 × 実オッズ・上位${(r.pairs || []).length}ペア</small></div>
      <div class="pair hh2">
        <span></span><span></span><span></span>
        <span class="pair-v">馬連率</span><span class="pair-v">ワイド率</span><span class="pair-v odds-col">馬連オッズ</span>
      </div>
      ${pairRows || `<div class="cw-empty">ペアデータなし</div>`}
      ${(r.pairs || []).some((p) => p.umaren_odds != null) ? `<div class="pair-note">
        <b>馬連率/ワイド率</b>＝AIが算出した、その2頭が馬連/ワイドで的中する確率。
        <b>馬連オッズ</b>＝実際に出ている馬連の配当（倍）。<u>率の割にオッズが高いペアが妙味</u>。
      </div>` : ""}
    </div>
  </div>`;
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
      <div class="cw-title"><b>COWORK</b>買い目・AI上位馬の見解</div>
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
    const resPos = r.result && a.umaban != null ? r.result.order[String(a.umaban)] : null;
    return `<div class="card adv" style="--i:${i}">
      <div class="adv-medal g${g}">${esc(a.grade || "—")}</div>
      <div class="adv-body">
        <div class="adv-head">
          ${horse ? wk(horse) : ""}
          <span class="adv-name">${esc(a.horse_name)}</span>
          ${a.tag ? `<span class="adv-tag ${tagCls(a.tag)}">${esc(a.tag)}</span>` : ""}
          ${resPos != null ? posBadge(resPos) : ""}
        </div>
        <div class="adv-comment">${esc(a.comment)}</div>
      </div>
    </div>`;
  }).join("");

  $("#cowork").innerHTML = `<div class="cw">
    ${tickets ? `<div class="cw-title"><b>COWORK</b>買い目</div>
      <div class="bet-grid">${tickets}</div>` : ""}
    ${advisors ? `<div class="cw-title"><b>COWORK</b>AI上位馬の見解</div>
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
  const runRows = runs.map((u) => {
    const p = u.pos;
    const pcls = p === 1 ? "rt1" : (p === 2 || p === 3) ? "rt23" : "";
    return `<tr class="${pcls}">
      <td class="rt-ago">${u.n_ago === 1 ? "前走" : `${u.n_ago}走前`}</td>
      <td class="rt-course">${esc(u.place || "")}${esc(u.td || "")}${u.dist ?? ""}</td>
      <td class="ta-c">${esc(u.track || "—")}</td>
      <td class="ta-c rt-pos">${posBadge(p) || "—"}</td>
      <td class="ta-c num">${u.ninki ?? "—"}<small>人気</small></td>
      <td>${esc(u.style || "—")}</td>
      <td class="ta-c">${esc(u.weight_change || "—")}</td>
      <td class="ta-r num">${num(u.agari3f)}</td>
      <td class="ta-r num">${u.interval_weeks != null ? u.interval_weeks + "<small>週</small>" : "—"}</td>
    </tr>`;
  }).join("");
  const hist = h.history || {};
  const histSummary = runs.length
    ? `<div class="dw-hsum">平均 <b class="num">${num(hist.avg_pos)}</b> 着 ・ 最高
        <b class="num">${hist.best_pos ?? "—"}</b> 着
        ${hist.deogure_count ? ` ・ 出遅れ <b class="num">${hist.deogure_count}</b> 回` : ""}
        ${hist.same_td_ratio != null ? ` ・ 同馬場率 <b class="num">${pct(hist.same_td_ratio, 0)}%</b>` : ""}</div>`
    : "";

  const ped = h.pedigree || {};
  const resPos = r.result ? r.result.order[String(h.umaban)] : null;
  const infoRows = [
    ["騎手", h.jockey ? `${esc(h.jockey)}${h.kawari ? " <i class='kw'>乗替</i>" : ""}` : "—"],
    ["斤量", h.kinryo ? num(h.kinryo) + " kg" : "—"],
    ["厩舎", h.trainer ? `${esc(h.shozoku ? h.shozoku + "・" : "")}${esc(h.trainer)}` : "—"],
    ["馬体重", h.taiju ? `${h.taiju}${h.taiju_diff ? ` (${esc(h.taiju_diff)})` : ""}` : "—"],
    ["脚質", h.style ? esc(h.style) : "—"],
    ["ZI指数", h.zi != null ? `${num(h.zi, 0)} <small>(${h.zi_rank ?? "—"}位)</small>` : "—"],
  ].map(([k, v]) => `<div class="ir"><span>${k}</span><span>${v}</span></div>`).join("");

  $("#drawer").innerHTML = `
    <button class="dw-close" id="dwClose" aria-label="閉じる">✕</button>
    <div class="dw-head">
      ${wk(h)}
      <span class="dw-mark mark ${markCls(h.mark)}">${h.mark || ""}</span>
      <span class="dw-name">${esc(h.name)}</span>
      ${resPos != null ? posBadge(resPos) : ""}
      <span class="dw-sub2">${esc(h.sex)}${h.age ?? ""} ・ ${h.ninki ?? "—"}番人気 ・ AIランク #${h.ai_rank ?? "—"}</span>
    </div>
    <div class="dw-ped">父 <b>${esc(ped.sire || "—")}</b> ／ 母父 <b>${esc(ped.broodmare_sire || "—")}</b>
      ${ped.broodmare_sire_type ? `（${esc(ped.broodmare_sire_type)}）` : ""}</div>
    <div class="dw-top ${whyHtml ? "" : "solo"}">
      ${whyHtml ? `<div class="dw-left">
        <div class="dw-sec" style="margin-top:0">AI の根拠（特徴量寄与）</div>${whyHtml}
      </div>` : ""}
      <div class="dw-right">
        <div class="dw-stats">
          <div class="dw-stat"><div class="v ${h.mark === "◎" ? "gold" : ""}">${pct(h.p_win)}%</div><div class="k">勝率</div></div>
          <div class="dw-stat"><div class="v">${pct(h.p_plc)}%</div><div class="k">連対率</div></div>
          <div class="dw-stat"><div class="v">${pct(h.p_sho)}%</div><div class="k">複勝圏</div></div>
          <div class="dw-stat"><div class="v ${h.ev_tan >= 1.2 ? "teal" : ""}">${num(h.ev_tan, 2)}</div><div class="k">単勝EV</div></div>
        </div>
        <div class="dw-odds">単勝 <b class="num">${num(h.odds)}</b> 倍 ／
          複勝 <b class="num">${num(h.fuku_low)}〜${num(h.fuku_high)}</b> 倍 ／ ${vsChip(h.vs_market)}</div>
        <div class="dw-info">${infoRows}</div>
      </div>
    </div>
    <div class="dw-sec">近 5 走</div>
    ${runRows ? `<div class="rt-wrap"><table class="rt">
      <thead><tr><th></th><th>コース</th><th>馬場</th><th>着</th><th>人気</th><th>クラス</th><th>脚質</th><th>上り3F</th><th>間隔</th></tr></thead>
      <tbody>${runRows}</tbody>
    </table></div>${histSummary}` : `<div class="cw-empty">出走歴なし（初出走）</div>`}
    <div class="dw-note">勝率・複勝圏は v6 calibrator 補正後の Plackett-Luce 確率。EV = 勝率 × 単勝オッズ。ZI は TARGET 指数。</div>`;

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

/* ---------------- keyboard ---------------- */
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") { closeDrawer(); return; }
  if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
  if (!state.day) return;
  const flat = allRacesFlat();
  const idx = flat.findIndex((r) => r.race_id === state.raceId);
  if (idx < 0) return;
  const next = idx + (e.key === "ArrowRight" ? 1 : -1);
  if (next < 0 || next >= flat.length) return;
  selectRace(flat[next].race_id);
});

/* ================= 全頭分析 / コース / 調教 / 血統 ================= */
const VIEWS = [
  { key: "shutsuba", label: "出走表" },
  { key: "bunseki", label: "全頭分析" },
  { key: "course", label: "コース" },
  { key: "training", label: "調教" },
  { key: "pedigree", label: "血統" },
];
const MARK_COLOR = { "◎": "#f5b942", "〇": "#d9e2f2", "○": "#d9e2f2", "▲": "#d08b4c", "△": "#8d9cba", "": "#46587e" };
const RANK_COLOR = { SS: "#f5b942", S: "#2dd4a8", A: "#5ba0f5", B: "#8d9cba" };
const UMAMI_COLOR = { S: "#f5b942", A: "#2dd4a8", B: "#5ba0f5", C: "#8d9cba", "罠": "#f2555a" };
let umamiSort = "xroi";

function disposeCharts() {
  charts.forEach((c) => { try { c.dispose(); } catch (e) { /* noop */ } });
  charts = [];
}
function mkChart(id, option) {
  const el = document.getElementById(id);
  if (!el || typeof echarts === "undefined") return null;
  const c = echarts.init(el, null, { renderer: "canvas" });
  c.setOption(Object.assign({
    backgroundColor: "transparent",
    textStyle: { fontFamily: '"Noto Sans JP", sans-serif', color: "#a9b6d3" },
    animationDuration: 600,
  }, option));
  charts.push(c);
  return c;
}
const GRID = { left: 8, right: 14, top: 28, bottom: 8, containLabel: true };
function axisStyle() {
  return {
    axisLine: { lineStyle: { color: "#32456e" } },
    axisLabel: { color: "#a9b6d3", fontSize: 12 },
    splitLine: { lineStyle: { color: "rgba(50,69,110,.35)" } },
    nameTextStyle: { color: "#7385a8" },
  };
}
function normArr(vals) {
  const ok = vals.filter((v) => v != null && !isNaN(v));
  const lo = Math.min(...ok, 0), hi = Math.max(...ok, 1e-9);
  const span = hi - lo || 1;
  return (v) => (v == null || isNaN(v)) ? 0 : Math.round((v - lo) / span * 100);
}
function rankBadge(rank) {
  if (!rank) return "";
  return `<span class="rkb" style="background:${RANK_COLOR[rank] || "#46587e"}">${rank}</span>`;
}

/* ---------------- 全頭 UMAMI テーブル ---------------- */
function umamiTableHtml(r) {
  const SORTS = {
    xroi: (h) => h.umami?.xroi ?? -1,
    uma: (h) => -(h.umaban ?? 99),
    evt: (h) => h.umami?.ev_tan ?? -1,
    evf: (h) => h.umami?.ev_fuku ?? -1,
  };
  const key = SORTS[umamiSort] ? umamiSort : "xroi";
  const hs = [...r.horses].sort((a, b) => SORTS[key](b) - SORTS[key](a));
  const sb = (k, label) => `<button class="usort ${umamiSort === k ? "on" : ""}" data-uk="${k}">${label}${umamiSort === k ? " ▼" : ""}</button>`;

  const rows = hs.map((h) => {
    const u = h.umami || {};
    const g = u.grade || "—";
    const gated = u.side == null;
    const xroiTxt = u.xroi != null ? `${Math.round(u.xroi * 100)}<small>%</small>` : "—";
    const xroiCls = u.xroi == null ? "" : u.xroi >= 0.8 ? "good" : u.xroi >= 0.72 ? "mid" : "low";
    return `<div class="um-row ${gated ? "gated" : ""} ${h.mark === "◎" ? "honmei" : ""}" data-uma="${h.umaban}">
      <span class="mark ${markCls(h.mark)}">${h.mark || "・"}</span>
      <span>${wk(h)}</span>
      <span class="um-name">${esc(h.name)}</span>
      <span class="num ta-r um-odds">${num(h.odds)}</span>
      <span class="num ta-r um-evt">${u.ev_tan != null ? u.ev_tan.toFixed(2) : "—"}</span>
      <span class="num ta-r um-evf">${u.ev_fuku != null ? u.ev_fuku.toFixed(2) : "—"}</span>
      <span class="num ta-r um-xroi ${xroiCls}">${xroiTxt}</span>
      <span class="ta-c"><span class="um-g" style="background:${UMAMI_COLOR[g] || "#46587e"}">${g}</span></span>
      <span class="um-side">${gated ? "—" : esc(u.side || "")}</span>
      <span class="um-reason">${esc(u.reason || "")}</span>
    </div>`;
  }).join("");

  return `<div class="card um-card">
    <div class="an-t">🍣 全頭 UMAMI テーブル
      <small>UMAMI = 実測補正後の期待回収率(xROI)。0.80=控除率中立、それ以上が勝負どころ。「罠」は妙味でも来る見込み薄</small></div>
    <div class="um-head">
      <span></span><span>${sb("uma", "馬番")}</span><span>馬名</span>
      <span class="ta-r um-odds">単オッズ</span><span class="ta-r um-evt">${sb("evt", "EV単")}</span>
      <span class="ta-r um-evf">${sb("evf", "EV複")}</span><span class="ta-r">${sb("xroi", "UMAMI")}</span>
      <span class="ta-c">格</span><span class="um-side">推奨</span><span class="um-reason">理由</span>
    </div>
    ${rows}
  </div>`;
}

/* ---------------- 全頭分析 ---------------- */
function renderBunseki(r, vb) {
  vb.innerHTML = `<div class="card anc">
      <div class="an-t">AI評価 × 市場人気
        <small>右上＝人気薄なのにAI高評価（妙味） / 左下＝人気だがAI低評価（過剰）・点の大きさ＝勝率</small></div>
      <div id="ch-scatter" class="chart"></div>
    </div>
    <div class="card anc">
      <div class="an-t">能力レーダー（上位6頭・1頭ずつ）
        <small>能力 / 勝率 / 複勝安定 / 妙味 / 瞬発(上がり) / 実績 をレース内で正規化。<b style="color:#7385a8">グレー＝出走全体の平均</b></small></div>
      <div class="radar-grid" id="radar-grid"></div>
    </div>
  <div id="um-wrap">${umamiTableHtml(r)}</div>`;

  const wireUmami = () => {
    $("#um-wrap").querySelectorAll(".usort").forEach((b) => {
      b.onclick = (e) => { e.stopPropagation(); umamiSort = b.dataset.uk; $("#um-wrap").innerHTML = umamiTableHtml(r); wireUmami(); };
    });
    $("#um-wrap").querySelectorAll(".um-row").forEach((row) => {
      row.onclick = () => openDrawer(+row.dataset.uma);
    });
  };
  wireUmami();

  const hs = r.horses;
  const N = hs.length;
  // --- scatter: x=人気(1→左), y=AI複勝圏率% ---
  const pts = hs.map((h) => ({
    value: [h.ninki ?? N, +( (h.p_sho ?? 0) * 100).toFixed(1)],
    itemStyle: { color: MARK_COLOR[h.mark] ?? MARK_COLOR[""], borderColor: "#0b1322", borderWidth: 1 },
    symbolSize: 10 + (h.p_win ?? 0) * 70,
    label: { show: true, formatter: String(h.umaban), color: "#0b1322", fontSize: 10, fontFamily: "Oswald" },
    h,
  }));
  const avgSho = hs.reduce((s, h) => s + (h.p_sho ?? 0), 0) / Math.max(N, 1) * 100;
  mkChart("ch-scatter", {
    grid: GRID,
    tooltip: {
      backgroundColor: "rgba(13,20,36,.95)", borderColor: "#32456e",
      textStyle: { color: "#edf1fb", fontSize: 12 },
      formatter: (p) => { const h = p.data.h; return `<b>${h.umaban} ${esc(h.name)}</b> ${h.mark || ""}<br>`
        + `${h.ninki ?? "—"}番人気 / 単${num(h.odds)}倍<br>AI複勝圏 ${pct(h.p_sho)}% / 勝率 ${pct(h.p_win)}%<br>単EV ${num(h.ev_tan, 2)}`; },
    },
    xAxis: Object.assign({ name: "人気→", min: 0.5, max: N + 0.5, interval: 1, inverse: false }, axisStyle()),
    yAxis: Object.assign({ name: "AI複勝圏率 %", min: 0 }, axisStyle()),
    series: [{
      type: "scatter", data: pts,
      markLine: {
        silent: true, symbol: "none", lineStyle: { color: "#3a4d75", type: "dashed" },
        label: { show: false },
        data: [{ yAxis: +avgSho.toFixed(1) }, { xAxis: (N + 1) / 2 }],
      },
    }],
  });

  // --- radar: 上位6頭を「1頭ずつ」小分割 (重ねない)。グレー=全体平均 ---
  const top = [...hs].sort((a, b) => (a.ai_rank ?? 99) - (b.ai_rank ?? 99)).slice(0, 6);
  const agari = (h) => {
    const rs = (h.history?.runs || []).map((u) => u.agari3f).filter((v) => v);
    return rs.length ? Math.min(...rs) : null; // 速い(小)ほど良 → 後で反転
  };
  const jisseki = (h) => {
    const rs = h.history?.runs || [];
    return rs.length ? rs.filter((u) => (u.pos ?? 9) <= 3).length / rs.length : 0;
  };
  const nAbility = normArr(hs.map((h) => h.ai_score));
  const nWin = normArr(hs.map((h) => h.p_win));
  const nSho = normArr(hs.map((h) => h.p_sho));
  const nEv = normArr(hs.map((h) => Math.min(h.ev_tan ?? 0, 3)));
  const nAg = normArr(hs.map((h) => { const a = agari(h); return a == null ? null : -a; }));
  const nJis = normArr(hs.map(jisseki));
  const indVal = (h) => [nAbility(h.ai_score), nWin(h.p_win), nSho(h.p_sho),
    nEv(Math.min(h.ev_tan ?? 0, 3)), nAg(agari(h) == null ? null : -agari(h)), nJis(jisseki(h))];
  // 全体平均 (グレー参照ポリゴン)
  const avgVal = [0, 1, 2, 3, 4, 5].map((k) =>
    Math.round(hs.reduce((s, h) => s + indVal(h)[k], 0) / Math.max(hs.length, 1)));
  const AXES = ["能力", "勝率", "複勝安定", "妙味", "瞬発", "実績"];

  const grid = $("#radar-grid");
  grid.innerHTML = top.map((h, i) =>
    `<div class="rd-cell"><div id="rd-${i}" class="rd-chart"></div>
      <div class="rd-cap"><span class="mark ${markCls(h.mark)}">${h.mark || ""}</span>${wk(h)}
      <span class="rd-nm">${esc(h.name)}</span></div></div>`).join("");
  top.forEach((h, i) => {
    const col = MARK_COLOR[h.mark] ?? "#5ba0f5";
    mkChart(`rd-${i}`, {
      tooltip: { backgroundColor: "rgba(13,20,36,.95)", borderColor: "#32456e", textStyle: { color: "#edf1fb", fontSize: 11 } },
      radar: {
        indicator: AXES.map((n) => ({ name: n, max: 100 })),
        radius: "66%", center: ["50%", "54%"],
        axisName: { color: "#8597ba", fontSize: 10 },
        splitNumber: 3,
        splitLine: { lineStyle: { color: "rgba(50,69,110,.45)" } },
        splitArea: { areaStyle: { color: ["rgba(255,255,255,.015)", "rgba(255,255,255,.035)"] } },
        axisLine: { lineStyle: { color: "rgba(50,69,110,.45)" } },
      },
      series: [{
        type: "radar", symbolSize: 3,
        data: [
          { value: avgVal, name: "全体平均", lineStyle: { color: "#46587e", width: 1, type: "dashed" }, itemStyle: { color: "#46587e" }, areaStyle: { color: "rgba(70,88,126,.12)" } },
          { value: indVal(h), name: `${h.umaban} ${h.name}`, lineStyle: { color: col, width: 2 }, itemStyle: { color: col }, areaStyle: { color: col, opacity: 0.18 } },
        ],
      }],
    });
  });
}

/* ---------------- コース ---------------- */
function renderCourse(r, vb) {
  const key = `${r.place}|${r.course}`;
  const cs = state.day.courses ? state.day.courses[key] : null;
  if (!cs) {
    vb.innerHTML = `<div class="card cw-empty">このコース（${esc(key)}）の集計データがありません。</div>`;
    return;
  }
  const marked = r.horses.filter((h) => h.mark);
  const honmei = r.horses.find((h) => h.mark === "◎");
  const wakuOf = (h) => String(h.waku ?? "");
  const tags = (pred) => marked.filter(pred).map((h) =>
    `<span class="cc-tag" style="color:${MARK_COLOR[h.mark]}">${h.mark}${h.umaban}</span>`).join("");

  vb.innerHTML = `<div class="cc-head card">
    <div class="cc-h-t">${esc(r.place)} ${esc(r.course)} <small>過去 ${cs.n_races?.toLocaleString()}レース / ${cs.n_starts?.toLocaleString()}頭 の傾向</small></div>
    ${honmei ? `<div class="cc-h-s">◎${honmei.umaban} ${esc(honmei.name)} … <b>${wakuOf(honmei)}枠</b> / 脚質 <b>${esc(honmei.style || "—")}</b></div>` : ""}
  </div>
  <div class="cc-grid">
    <div class="card anc"><div class="an-t">枠順別 複勝率 <small>印馬の枠を強調</small></div><div id="ch-waku" class="chart sm"></div></div>
    <div class="card anc"><div class="an-t">脚質別 複勝率 <small>${["逃げ", "先行", "差し", "追込"].map((k) => `${k}${tags((h) => h.style === k) ? "•" : ""}`).join(" ")}</small></div><div id="ch-kyaku" class="chart sm"></div></div>
    <div class="card anc"><div class="an-t">年齢別 複勝率</div><div id="ch-age" class="chart sm"></div></div>
    <div class="card anc"><div class="an-t">性別 複勝率</div><div id="ch-sex" class="chart sm"></div></div>
  </div>`;

  const baseFuku = cs.n_starts ? null : null;
  const markedWaku = new Set(marked.map((h) => String(h.waku)));
  const markedStyle = {};
  marked.forEach((h) => { markedStyle[h.style] = (markedStyle[h.style] || 0) + 1; });

  const barChart = (id, rows, opt = {}) => {
    const cats = rows.map((x) => x.label);
    const vals = rows.map((x) => x.fuku);
    const hl = opt.highlight || (() => false);
    mkChart(id, {
      grid: { left: 6, right: 12, top: 16, bottom: 6, containLabel: true },
      tooltip: {
        backgroundColor: "rgba(13,20,36,.95)", borderColor: "#32456e", textStyle: { color: "#edf1fb", fontSize: 12 },
        formatter: (p) => { const x = rows[p[0].dataIndex]; return `${x.label}<br>複勝率 ${x.fuku}% / 勝率 ${x.win}%<br>n=${(x.n || 0).toLocaleString()}`; },
      },
      xAxis: Object.assign({ type: "category", data: cats }, axisStyle()),
      yAxis: Object.assign({ type: "value", name: "複勝率%", min: 0 }, axisStyle()),
      series: [{
        type: "bar", data: vals.map((v, i) => ({
          value: v,
          itemStyle: { color: hl(rows[i].label) ? "#f5b942" : "#3f6fb8", borderRadius: [3, 3, 0, 0] },
        })),
        barWidth: opt.barWidth || "58%",
        label: { show: true, position: "top", color: "#a9b6d3", fontSize: 10, formatter: (p) => p.value + "%" },
      }],
    });
  };
  barChart("ch-waku", cs.waku, { highlight: (l) => markedWaku.has(l) });
  barChart("ch-kyaku", cs.kyaku, { highlight: (l) => (markedStyle[l] || 0) > 0, barWidth: "46%" });
  barChart("ch-age", cs.age, { barWidth: "46%" });
  barChart("ch-sex", cs.sex, { barWidth: "40%" });
}

/* ---------------- 調教 ---------------- */
function lapMini(laps) {
  if (!laps || !laps.length) return "";
  const lo = Math.min(...laps), hi = Math.max(...laps);
  const span = hi - lo || 1;
  const n = laps.length;
  return `<span class="lapmini" title="200mごとのラップ（右＝終い・高い＝速い）">${laps.map((l, i) => {
    const h = 6 + (1 - (l - lo) / span) * 16; // 速い(小)=高い
    return `<i class="${i === n - 1 ? "last" : ""}" style="height:${h.toFixed(0)}px" title="${l}秒"></i>`;
  }).join("")}</span>`;
}
function renderTraining(r, vb) {
  const top5 = state.day.training_top5 || [];
  const top5Html = top5.length ? `<div class="card tr-top">
    <div class="an-t">⚡ 今週の好調教 Best5 <small>坂路 終い200m が速い順（開催全体）</small></div>
    <div class="tr-top-row">${top5.map((t, i) => `<div class="tr-top-c">
      <span class="tr-rk">${i + 1}</span>
      <div><div class="tr-top-n">${esc(t.name)}</div>
        <div class="tr-top-m">${esc(t.place)}${t.rno}R・${t.umaban}番</div></div>
      <span class="tr-top-v num">${num(t.lap1)}<small>終い</small></span>
    </div>`).join("")}</div>
  </div>` : "";

  const rows = r.horses.map((h) => {
    const t = h.training;
    const hanro = t && t.hanro;
    const wc = t && t.wc;
    if (!hanro && !wc) {
      return `<div class="tr-row none">
        <span class="mark ${markCls(h.mark)}">${h.mark || "・"}</span>${wk(h)}
        <span class="tr-name">${esc(h.name)}</span>
        <span class="tr-na">追い切りデータなし</span></div>`;
    }
    const hanroHtml = hanro ? `<span class="tr-set"><b>坂路</b>
      <span class="tr-kv">4F <em class="num">${num(hanro.t4f)}</em></span>
      <span class="tr-kv">終い <em class="num">${num(hanro.lap1)}</em></span>
      ${lapMini(hanro.laps)}<span class="tr-d">${esc((hanro.date || "").slice(4))}</span></span>` : "";
    const wcHtml = wc ? `<span class="tr-set"><b>W</b>
      <span class="tr-kv">5F <em class="num">${num(wc.f5)}</em></span>
      <span class="tr-kv">終い <em class="num">${num(wc.lap1)}</em></span>
      ${lapMini(wc.laps)}<span class="tr-d">${esc((wc.date || "").slice(4))}</span></span>` : "";
    return `<div class="tr-row">
      <span class="mark ${markCls(h.mark)}">${h.mark || "・"}</span>${wk(h)}
      <span class="tr-name">${esc(h.name)}</span>
      <span class="tr-sets">${hanroHtml}${wcHtml}</span></div>`;
  }).join("");

  const nCov = r.horses.filter((h) => h.training).length;
  vb.innerHTML = `${top5Html}
    <div class="card tr-list">
      <div class="an-t">出走馬の最終追い切り <small>${nCov}/${r.horses.length}頭にデータ・坂路は美浦/栗東のみ</small></div>
      <div class="tr-legend">読み方：<b>4F/5F</b>＝追い切り全体のタイム（短いほど速い）／<b>終い</b>＝ラスト200mのタイム／
        <span class="lapmini lg"><i style="height:9px"></i><i style="height:13px"></i><i style="height:17px"></i><i class="last" style="height:21px"></i></span>＝200mごとのラップ（右が終い・棒が高いほど速い）</div>
      ${rows}
    </div>`;
}

/* ---------------- 血統 ---------------- */
function renderPedigree(r, vb) {
  const withStats = r.horses.filter((h) => h.ped_stats && (h.ped_stats.sire || h.ped_stats.bms));
  const baseline = withStats.length ? withStats[0].ped_stats.baseline : null;

  // bar: 各馬の父 fuku% (このコース) vs baseline
  const sireRows = r.horses
    .filter((h) => h.ped_stats && h.ped_stats.sire)
    .map((h) => ({ h, fuku: h.ped_stats.sire.fuku, rank: h.ped_stats.sire.rank }))
    .sort((a, b) => b.fuku - a.fuku);

  const chartHtml = sireRows.length
    ? `<div class="card anc"><div class="an-t">父 × このコース 複勝率 <small>${baseline != null ? `全体平均 ${baseline}%` : ""}</small></div><div id="ch-ped" class="chart sm"></div></div>`
    : "";

  const cards = r.horses.map((h) => {
    const ped = h.pedigree || {};
    const ps = h.ped_stats;
    const sire = ps && ps.sire;
    const bms = ps && ps.bms;
    return `<div class="card ped-c">
      <div class="ped-head">${wk(h)}<span class="mark ${markCls(h.mark)}">${h.mark || ""}</span>
        <span class="ped-name">${esc(h.name)}</span></div>
      <div class="ped-line"><span class="ped-l">父</span>
        <span class="ped-sire">${esc(ped.sire || "—")}</span>
        ${sire ? `${rankBadge(sire.rank)}<span class="ped-f">複勝 <b>${sire.fuku}%</b> <small>n=${sire.n}</small></span>`
          : `<span class="ped-na">データ少</span>`}</div>
      <div class="ped-line"><span class="ped-l">母父</span>
        <span class="ped-sire">${esc(ped.broodmare_sire || "—")}</span>
        ${bms ? `${rankBadge(bms.rank)}<span class="ped-f">複勝 <b>${bms.fuku}%</b> <small>n=${bms.n}</small></span>`
          : `<span class="ped-na">データ少</span>`}</div>
    </div>`;
  }).join("");

  vb.innerHTML = `${chartHtml}
    <div class="ped-note">★ このコース（${esc(r.place)} ${esc(r.course)}）における種牡馬・母父の過去複勝率。
      ランク SS&gt;S&gt;A&gt;B は全体平均との比。出走数15以上の血統のみ集計対象。</div>
    <div class="ped-grid">${cards}</div>`;

  if (sireRows.length) {
    mkChart("ch-ped", {
      grid: { left: 6, right: 14, top: 16, bottom: 6, containLabel: true },
      tooltip: {
        backgroundColor: "rgba(13,20,36,.95)", borderColor: "#32456e", textStyle: { color: "#edf1fb", fontSize: 12 },
        formatter: (p) => { const x = sireRows[p[0].dataIndex]; return `${x.h.umaban} ${esc(x.h.name)}<br>父 ${esc(x.h.pedigree.sire)}<br>複勝率 ${x.fuku}% (${x.rank})`; },
      },
      xAxis: Object.assign({ type: "category", data: sireRows.map((x) => x.h.umaban) }, axisStyle()),
      yAxis: Object.assign({ type: "value", name: "複勝率%", min: 0 }, axisStyle()),
      series: [{
        type: "bar", barWidth: "56%",
        data: sireRows.map((x) => ({ value: x.fuku, itemStyle: { color: RANK_COLOR[x.rank] || "#3f6fb8", borderRadius: [3, 3, 0, 0] } })),
        label: { show: true, position: "top", color: "#a9b6d3", fontSize: 10, formatter: (p) => p.value + "%" },
        markLine: baseline != null ? {
          silent: true, symbol: "none", lineStyle: { color: "#f2555a", type: "dashed" },
          label: { color: "#f3989b", fontSize: 10, formatter: "平均" },
          data: [{ yAxis: baseline }],
        } : undefined,
      }],
    });
  }
}

/* ---------------- view machinery ---------------- */
function renderViewTabs() {
  const vt = $("#viewTabs");
  vt.innerHTML = VIEWS.map((v) =>
    `<button class="vt ${v.key === state.view ? "on" : ""}" data-view="${v.key}">${v.label}</button>`).join("");
  vt.querySelectorAll(".vt").forEach((b) => {
    b.onclick = () => {
      if (state.view === b.dataset.view) return;
      state.view = b.dataset.view;
      renderViewTabs();
      renderView();
    };
  });
}

function renderView() {
  disposeCharts();
  const r = currentRace();
  const vb = $("#viewbody");
  if (!r) { vb.innerHTML = `<div class="err">レースがありません</div>`; return; }
  if (state.view === "shutsuba") {
    vb.innerHTML = `<section id="shutsuba"></section><section id="extras"></section><section id="cowork"></section>`;
    renderTable(r); renderExtras(r); renderCowork(r);
  } else if (state.view === "bunseki") {
    renderBunseki(r, vb);
  } else if (state.view === "course") {
    renderCourse(r, vb);
  } else if (state.view === "training") {
    renderTraining(r, vb);
  } else if (state.view === "pedigree") {
    renderPedigree(r, vb);
  }
}

window.addEventListener("resize", () => { charts.forEach((c) => { try { c.resize(); } catch (e) { /* noop */ } }); });

/* ---------------- compose ---------------- */
function renderRace() {
  const r = currentRace();
  if (!r) {
    $("#raceHeader").innerHTML = `<div class="err">レースがありません</div>`;
    $("#viewbody").innerHTML = "";
    return;
  }
  renderHeader(r);
  renderViewTabs();
  renderView();
}

boot();
