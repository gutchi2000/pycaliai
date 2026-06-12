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
  $("#shutsuba").innerHTML = `<div class="card skel">
    ${`<div class="skel-bar"></div>`.repeat(6)}
  </div>`;
  ["#extras", "#cowork"].forEach((s) => { $(s).innerHTML = ""; });
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
    <svg viewBox="0 0 48 48" width="56" height="56" aria-hidden="true">
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
    <div style="min-width:0">
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
      <span style="min-width:0">
        <span class="hname">${esc(h.name)}</span><br>
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
      <span class="pair-v num">${p.fair != null ? num(p.fair) : "—"}<small>倍</small></span>
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
      <div class="ex-t">馬連・ワイド AI確率 <small>上位${(r.pairs || []).length}ペア</small></div>
      <div class="pair hh2">
        <span></span><span></span><span></span>
        <span class="pair-v">馬連率</span><span class="pair-v">ワイド率</span><span class="pair-v">適正</span>
      </div>
      ${pairRows || `<div class="cw-empty">ペアデータなし</div>`}
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
    </div>
    <div class="dw-sub">${esc(h.sex)}${h.age ?? ""} ・ ${h.ninki ?? "—"}番人気 ・ AIランク #${h.ai_rank ?? "—"}</div>
    <div class="dw-ped">父 <b>${esc(ped.sire || "—")}</b> ／ 母父 <b>${esc(ped.broodmare_sire || "—")}</b>
      ${ped.broodmare_sire_type ? `（${esc(ped.broodmare_sire_type)}）` : ""}</div>
    <div class="dw-stats">
      <div class="dw-stat"><div class="v ${h.mark === "◎" ? "gold" : ""}">${pct(h.p_win)}%</div><div class="k">勝率</div></div>
      <div class="dw-stat"><div class="v">${pct(h.p_plc)}%</div><div class="k">連対率</div></div>
      <div class="dw-stat"><div class="v">${pct(h.p_sho)}%</div><div class="k">複勝圏</div></div>
      <div class="dw-stat"><div class="v ${h.ev_tan >= 1.2 ? "teal" : ""}">${num(h.ev_tan, 2)}</div><div class="k">単勝EV</div></div>
    </div>
    <div style="font-size:11.5px;color:var(--tx2);margin-bottom:4px">単勝 <b class="num">${num(h.odds)}</b> 倍 ／
      複勝 <b class="num">${num(h.fuku_low)}〜${num(h.fuku_high)}</b> 倍 ／ ${vsChip(h.vs_market)}</div>
    <div class="dw-info">${infoRows}</div>
    ${whyHtml ? `<div class="dw-sec">AI の根拠（特徴量寄与）</div>${whyHtml}` : ""}
    ${runsHtml ? `<div class="dw-sec">近 5 走（左が最新）</div><div class="runs">${runsHtml}</div>` : ""}
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

/* ---------------- compose ---------------- */
function renderRace() {
  const r = currentRace();
  if (!r) {
    $("#raceHeader").innerHTML = `<div class="err">レースがありません</div>`;
    ["#shutsuba", "#extras", "#cowork"].forEach((s) => { $(s).innerHTML = ""; });
    return;
  }
  renderHeader(r);
  renderTable(r);
  renderExtras(r);
  renderCowork(r);
}

boot();
