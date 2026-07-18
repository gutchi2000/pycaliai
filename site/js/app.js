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
  mode: "races",
};
const dayCache = new Map();
let charts = [];
let resultsData = null;
let realizedBias = null;   // data/realized_bias.json (直近開催の実現トラックバイアス)

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
  try {
    realizedBias = await (await fetch("data/realized_bias.json?t=" + Date.now())).json();
  } catch (e) { realizedBias = null; }   // 非開催週などで無くても描画は続行
  const sel = $("#dateSel");
  sel.innerHTML = mf.dates.map((d) =>
    `<option value="${d.date}">${fmtDate(d.date)}${d.has_results ? " ✓" : ""}</option>`).join("");
  sel.onchange = () => loadDay(sel.value);
  $("#footInfo").textContent =
    `PyCaLiAI ${mf.model} — 静的ビルド ${mf.built_at} ／ ← → キーでレース移動`;
  $("#modeNav").querySelectorAll("button").forEach((b) => {
    b.onclick = () => setMode(b.dataset.mode);
  });
  if (mf.dates.length) loadDay(mf.dates[0].date);
}

/* ---------------- 予想 / 成績 モード ---------------- */
function setMode(mode) {
  state.mode = mode;
  $("#modeNav").querySelectorAll("button").forEach((b) =>
    b.classList.toggle("on", b.dataset.mode === mode));
  const isResults = mode === "results";
  ["#venueTabs", "#raceStrip", "#main"].forEach((s) => { $(s).hidden = isResults; });
  $("#resultsMain").hidden = !isResults;
  $("#dateWrap").style.visibility = isResults ? "hidden" : "";
  // 馬場バイアスは会場スコープの予想ビュー専用。成績ビューでは隠し、戻ったら再評価。
  const bb = $("#babaBias");
  if (bb) bb.hidden = true;
  closeDrawer();
  if (isResults) renderResults();
  else if (window.renderBabaBias) window.renderBabaBias();
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

  // 上部の馬場バイアスを選択中の会場に同期（baba.js が __babaPlace で1場に絞る）
  window.__babaPlace = state.place;
  if (window.renderBabaBias) window.renderBabaBias();
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
        style="stroke:${color};stroke-dasharray:${C};stroke-dashoffset:${(C * (1 - v)).toFixed(1)};--c:${C};--off:${(C * (1 - v)).toFixed(1)}"></circle>
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
    ${memberLevelEl(r)}
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

const KYAKU_CLS = { "逃げ": "k-nige", "先行": "k-senko", "差し": "k-sashi", "追込": "k-oikomi" };
function kyakuChip(h) {
  return h.style ? `<span class="kyaku ${KYAKU_CLS[h.style] || ""}">${h.style}</span>` : "";
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

/* ---------------- レベル (近走成績ベース, 公開データのみ, 非蓄積) ---------------- */
// 各馬レベル: build_site が history.runs(着順/人気=公開事実)から算出した 0-100 + S〜D。
// ELO(蓄積)や ZI/補正タイム(TARGET外部指数)には依存しない。
function levelChip(h) {
  const lv = h.level;
  if (!lv || lv.tier == null) return "";
  return `<span class="lvchip lv-${lv.tier}" title="各馬レベル = 近走の着順・人気(公開データ)から算出した成績スコア ${lv.score}/100。ELO(蓄積)や外部指数は不使用">${lv.tier}<i>${lv.score}</i></span>`;
}
// メンバーレベル: 上位3頭の平均レベルを同クラス分布で位置づけ (build_site が算出)。
// D→S のタイア帯スペクトラム上に「この組の位置」と「クラス平均」を印で示す。
function memberLevelEl(r) {
  const m = r.member_level;
  if (!m) return "";
  if (!m.tier) {
    return `<div class="mlvl lv-none">
      <div class="mlvl-head">
        <span class="mlvl-badge">–</span>
        <span class="mlvl-body"><span class="mlvl-t">メンバーレベル <b class="mlvl-lab">${esc(m.label || "—")}</b></span>
        <span class="mlvl-ctx">出走馬の前走成績が少なく判定不能（新馬など）</span></span>
      </div></div>`;
  }
  const ctx = [];
  if (m.top_level != null) ctx.push(`上位レベル <b>${m.top_level}</b>`);
  if (m.class_avg != null) ctx.push(`${esc(m.class_key || "")}平均 ${m.class_avg}`);
  if (m.pct != null) ctx.push(`上位 ${Math.max(1, 100 - m.pct)}%`);
  const clamp = (v) => Math.max(3, Math.min(97, v ?? 0));
  const zones = ["D", "C", "B", "A", "S"]
    .map((t) => `<span class="mls-zone lv-${t}">${t}</span>`).join("");
  const avg = m.avg_pct != null
    ? `<span class="mls-avg" style="left:${clamp(m.avg_pct)}%"><em>平均</em></span>` : "";
  return `<div class="mlvl lv-${m.tier}" title="出走馬の近走成績(公開データ)から算出。上位3頭の平均レベルを同クラスの分布で位置づけ。ELOや外部指数は不使用。">
    <div class="mlvl-head">
      <span class="mlvl-badge">${m.tier}</span>
      <span class="mlvl-body">
        <span class="mlvl-t">メンバーレベル <b class="mlvl-lab">${esc(m.label)}</b></span>
        <span class="mlvl-ctx">${ctx.join(" ・ ")}</span>
      </span>
    </div>
    <div class="mls" role="img" aria-label="メンバーレベル ${m.tier} / 5段階">
      <div class="mls-track">
        ${zones}
        ${avg}
        <span class="mls-mark" style="left:${clamp(m.pct)}%"><b>${m.tier}</b></span>
      </div>
      <div class="mls-cap"><span>低調</span><span>ハイレベル</span></div>
    </div>
    ${honmeiVerdictEl(r)}
    <div class="mlvl-note">※ 近走成績ベースの目安。買い目の最終判断は 勝率・AI指数 と合わせて。</div>
  </div>`;
}

// 「本命◎ はこの組で格上か」= ◎の近走レベル − 相手の中央値。格上ほど◎が来やすい(2026実測)。
function honmeiVerdictEl(r) {
  const hon = r.horses.find((h) => h.mark === "◎" && h.level);
  const others = r.horses.filter((h) => h.mark !== "◎" && h.level)
    .map((h) => h.level.score).sort((a, b) => a - b);
  if (!hon || others.length < 3) return "";
  const med = others[Math.floor(others.length / 2)];
  const gap = hon.level.score - med;
  const v = gap >= 41 ? { t: "格上", c: "up", hint: "◎の複勝率が高めの型 (2026実測 約58%)" }
    : gap >= 20 ? { t: "やや上", c: "mid", hint: "標準的な型 (2026実測 約51%)" }
    : { t: "ほぼ同格", c: "flat", hint: "混戦・◎も楽ではない型 (2026実測 約40%)" };
  return `<div class="mlvl-verdict cls-${v.c}" title="◎の近走レベルが組の中央値をどれだけ上回るか。格上ほど◎が好走しやすい傾向(2026年 実測)。">
    <span class="mlvl-vlead">本命<b class="m1c">◎</b>${esc(hon.name)} は この組で <b class="vv">${v.t}</b></span>
    <span class="vsub">近走レベル ${hon.level.score} / 組の中央値 ${med} ・ ${v.hint}</span>
  </div>`;
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
        <span class="hsub">${levelChip(h)}${kyakuChip(h)}${esc(subLine(h))}</span>
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
      <span class="sh-note">行クリックで詳細 ／ 列見出しでソート ／ <b class="lvchip lv-S" style="margin:0">Lv</b>=馬レベル(近走成績・S〜D)</span>
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

/* ---------------- extras: 馬連/ワイド 妙味 ---------------- */
// 展開予想は廃止 (コース分析タブの「想定隊列」に集約)。左半分は当面空白。
function renderExtras(r) {
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
    ${realizedCard(r)}
  </div>`;
}

/* 実現トラックバイアス カード (出走表タブ・妙味の隣). 直近開催の結果から会場×面で算出。
   前残り(脚質)=信頼できる定数 / 枠=小標本で反転しやすい弱信号、として見せる。 */
function realizedCard(r) {
  // クロスデイ: 前日(土)の実現を翌日(日)のカードに載せる。土曜カード(=前日なし)や
  // 無関係な日には出さない。show_on_date(=結果の日の翌開催日) が今見てる日と一致する時のみ。
  const showOn = realizedBias?.show_on_date;
  if (!showOn || state.day?.date !== showOn) return "";
  const surf = (r.course || "").startsWith("芝") ? "芝" : "ダ";
  const rb = realizedBias && realizedBias.venues
    ? realizedBias.venues[`${r.place}|${surf}`] : null;
  if (!rb) {
    return `<div class="card ex">
      <div class="ex-t">実現バイアス <small>直近開催の結果から</small></div>
      <div class="cw-empty">この会場×面の実現データがまだありません。</div>
    </div>`;
  }
  const wcls = rb.waku === "内" ? "in" : rb.waku === "外" ? "out" : "flat";
  const fp = Math.round(rb.front_rate * 100);
  const ih = Math.round(rb.inner_half * 100);
  return `<div class="card ex">
    <div class="ex-t">実現バイアス <small>${esc(realizedBias.label)}・${esc(r.place)}${surf}・${rb.n}R</small></div>
    <div class="rb-row">
      <span class="rb-k">前残り</span>
      <span class="rb-bar"><i style="width:${fp}%"></i></span>
      <span class="rb-v num">${fp}<small>%</small></span>
    </div>
    <div class="rb-row">
      <span class="rb-k">枠</span>
      <span class="rb-waku ${wcls}">${esc(rb.waku)}</span>
      <span class="rb-sub">内半 ${ih}% ・ ${esc(rb.baba)}</span>
    </div>
    ${biasFitBlock(r, rb)}
    <div class="pair-note">
      <b>前残り</b>＝道中 前1/3 で勝った割合（脚質バイアス・<u>信頼できる定数</u>）。
      <b>枠</b>＝小標本で日々反転しやすい弱信号、参考程度に。
      <b>合致/逆風</b>＝各印馬の脚質を実現バイアスと突合した<u>読みの目安</u>（買い目には未反映）。
    </div>
  </div>`;
}

/* 実現バイアス × 各印馬の脚質を突合して「バイアス合致=評価↑ / 逆風=評価↓」を言語化。
   前残り(脚質)の robust 軸のみで判定。枠は弱信号なので添える程度。表示専用＝買い目不干渉。 */
const _FRONT_STY = { "逃げ": 1, "先行": 1 };
const _CLOSE_STY = { "差し": 1, "追込": 1 };
const _MARK_RANK = { "◎": 0, "〇": 1, "○": 1, "▲": 2, "△": 3 };
function biasFitBlock(r, rb) {
  const fp = Math.round(rb.front_rate * 100);
  // 前有利(+1) / 差し有利(-1) / フラット(0)。前脚質は元々勝ちやすいので中立帯を広めに。
  let favor = 0;
  if (fp >= 58) favor = 1;
  else if (fp <= 34) favor = -1;
  const dirWord = favor > 0 ? "前残り" : "差し・追込";
  if (favor === 0) {
    return `<div class="rb-fit-flat">前残り${fp}% ＝ ほぼフラット。脚質での評価上下は今回なし。</div>`;
  }
  // AI指数(出走表と同じ: レース内で ai_score を 0-100 正規化)
  const scores = (r.horses || []).map((h) => h.ai_score).filter((v) => v != null);
  const sMin = scores.length ? Math.min(...scores) : 0;
  const sRange = scores.length ? Math.max(Math.max(...scores) - sMin, 1e-9) : 1;
  const marked = (r.horses || [])
    .filter((h) => h.mark && _MARK_RANK[h.mark] != null)
    .sort((a, b) => _MARK_RANK[a.mark] - _MARK_RANK[b.mark]);
  const rows = [];
  for (const h of marked) {
    const sty = h.style;
    if (!sty) continue;
    const isFront = _FRONT_STY[sty], isClose = _CLOSE_STY[sty];
    if (!isFront && !isClose) continue;      // 中間脚質など判定外
    // dir: +1=追い風(バイアス合致) / -1=逆風
    const dir = (favor > 0) ? (isFront ? 1 : -1) : (isClose ? 1 : -1);
    const low = _MARK_RANK[h.mark] >= 2;      // ▲△ = 軽い印
    let msg;
    if (dir > 0) msg = low ? "バイアス合致。実質格上げの狙い目" : "バイアスも後押し、信頼度アップ";
    else msg = low ? "元々軽い上に逆風、消し寄り" : "本命級だが逆風、頭は危険・軽めに";
    const idx = h.ai_score == null ? "—" : Math.round((h.ai_score - sMin) / sRange * 100);
    rows.push(`<div class="rb-fit ${dir > 0 ? "up" : "dn"}">
      <span class="mark ${markCls(h.mark)}">${h.mark}</span>
      <span class="rb-fit-n">${h.umaban} ${esc(h.name)}</span>
      <span class="rb-fit-sty">${esc(sty)}</span>
      <span class="rb-fit-ai">AI<b>${idx}</b>${h.ai_rank ? `<span class="rb-fit-rk">#${h.ai_rank}</span>` : ""}</span>
      <span class="rb-fit-arw">${dir > 0 ? "追い風▲" : "逆風▼"}</span>
      <span class="rb-fit-msg">${msg}</span>
    </div>`);
  }
  if (!rows.length) return "";
  const small = rb.n < 3 ? `<span class="rb-fit-warn">※${rb.n}Rの小標本・目安</span>` : "";
  return `<div class="rb-fit-hd">${dirWord}有利の脚質補正 ${small}</div>${rows.join("")}`;
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

  // 値のある特徴だけを根拠に出す (初出走の過去走など欠損寄与ノイズを除外)
  const whys = (h.why || []).filter((w) => w.value != null).slice(0, 6);
  const isFirstRun = !((h.history?.runs || []).length);
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
    ["馬レベル (近走成績)", h.level
      ? `<b class="lvchip lv-${h.level.tier}" style="margin-right:6px">${h.level.tier}</b>${h.level.score} <small>/100</small>`
      : "—（出走歴なし）"],
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
    <div class="dw-top ${(whyHtml || isFirstRun) ? "" : "solo"}">
      ${(whyHtml || isFirstRun) ? `<div class="dw-left">
        <div class="dw-sec" style="margin-top:0">AI の根拠（特徴量寄与）</div>
        ${isFirstRun ? `<div style="font-size:11px;color:#8a97b5;margin:2px 0 8px;line-height:1.45">初出走：過去走なし。調教・騎手・厩舎・血統中心の評価です。</div>` : ""}
        ${whyHtml || `<div style="font-size:12px;color:#8a97b5">計上できる特徴がありません</div>`}
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
    <div class="dw-note">勝率・複勝圏は v6 calibrator 補正後の Plackett-Luce 確率。EV = 勝率 × 単勝オッズ。馬レベルは近走の着順・人気(公開データ)から算出。</div>`;

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
    nameTextStyle: { color: "#c2cde4", fontSize: 12 },
  };
}
// 枠順別チャート用 JRA 枠色 (黒枠は暗背景でも見えるよう薄枠線を併用)
const WAKU_BAR = { "1": "#f4f6fb", "2": "#14161c", "3": "#d33b3b", "4": "#2667d6", "5": "#f0c93c", "6": "#2c9e57", "7": "#e2702a", "8": "#e4549a" };
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

/* ---------------- 想定隊列 (NiceGUI 移植) ---------------- */
// 各馬の脚質 (h.style) で 逃げ/先行/差し/追込 に分類 → ペース判定 + 有利脚質。
function taikeiHtml(r) {
  const horses = r.horses || [];
  if (!horses.length) return "";
  const ORDER = ["逃げ", "先行", "差し", "追込"];
  const groups = { "逃げ": [], "先行": [], "差し": [], "追込": [], "不明": [] };
  horses.forEach((h) => groups[ORDER.includes(h.style) ? h.style : "不明"].push(h));
  Object.keys(groups).forEach((k) => groups[k].sort((a, b) => (a.umaban ?? 99) - (b.umaban ?? 99)));
  const counts = {}; Object.keys(groups).forEach((k) => counts[k] = groups[k].length);

  // ペース判定 (NiceGUI と同一ロジック)
  const nNige = counts["逃げ"], nSen = counts["先行"], nSashi = counts["差し"] + counts["追込"];
  let pace, paceColor, paceMsg;
  if (nNige >= 2) { pace = "Hペース"; paceColor = "#f38ba8"; paceMsg = "逃げ馬複数 → 前崩れ気配、差し・追込有利"; }
  else if (nNige === 0 && nSen >= 3) { pace = "Sペース"; paceColor = "#89b4fa"; paceMsg = "明確な逃げ馬不在 + 先行多 → スロー濃厚、上り勝負・先行有利"; }
  else if (nSashi >= nSen + nNige) { pace = "Mペース (差し決着)"; paceColor = "#cba6f7"; paceMsg = "差し型多数 → 上り3F が決まる末脚勝負"; }
  else { pace = "Mペース"; paceColor = "#a6e3a1"; paceMsg = "脚質バランス均等 → 標準的な流れ、ポジション戦"; }

  let advantage;
  if (pace === "Hペース") advantage = "差し・追込";
  else if (pace === "Sペース") advantage = "逃げ・先行";
  else if (pace.includes("差し決着")) advantage = "差し";
  else advantage = "先行〜差し";

  // 印馬は枠色 (.w1〜.w8) でバッジ+枠線を着色。無印は navy。
  const WAKU_HEX = { 1: "#f4f6fb", 2: "#3c4254", 3: "#d33b3b", 4: "#2667d6", 5: "#f0c93c", 6: "#2c9e57", 7: "#e2702a", 8: "#e4549a" };
  const chip = (h) => {
    const mark = h.mark || "";
    const pwin = Math.round((h.p_win || 0) * 100);
    const nm = `<span class="tk-nm">${h.umaban} ${esc(h.name)}</span><span class="tk-win">${pwin}%</span>`;
    if (mark) {
      const wkc = h.waku ? `w${h.waku}` : "tk-badge-non";
      const bc = WAKU_HEX[h.waku] || "#46587e";
      return `<div class="tk-chip mk" data-uma="${h.umaban}" style="border-color:${bc}">
        <span class="tk-badge ${wkc}">${mark}</span>${nm}</div>`;
    }
    return `<div class="tk-chip" data-uma="${h.umaban}">
      <span class="tk-badge tk-badge-non">${h.umaban}</span>${nm}</div>`;
  };

  const SECTIONS = [["逃げ", "#f38ba8", "🏃"], ["先行", "#fab387", "→"], ["差し", "#a6e3a1", "←"], ["追込", "#89b4fa", "←←"]];
  let lineup = "", prevFilled = false;
  for (const [k, color, icon] of SECTIONS) {
    const recs = groups[k];
    if (!recs.length) continue;
    if (prevFilled) lineup += `<div class="tk-arrow">↓</div>`;
    lineup += `<div class="tk-row" style="border-left-color:${color}">
      <div class="tk-lab" style="color:${color}"><div class="tk-icon">${icon}</div><div>${k} (${recs.length})</div></div>
      <div class="tk-chips">${recs.map(chip).join("")}</div></div>`;
    prevFilled = true;
  }
  const unk = groups["不明"];
  const unkSection = unk.length ? `<div class="tk-unknown">
      <div class="tk-unknown-h">脚質不明 (${unk.length}頭・過去走情報不足)</div>
      <div class="tk-chips">${unk.map(chip).join("")}</div></div>` : "";

  return `<div class="card tk-card" style="border-color:${paceColor}">
    <div class="tk-title">🏇 想定隊列</div>
    <div class="tk-pace"><span class="tk-pace-chip" style="background:${paceColor}">${pace}</span>
      <span class="tk-pace-msg">${esc(paceMsg)}</span></div>
    <div class="tk-lineup">${lineup}</div>
    ${unkSection}
    <div class="tk-foot">
      <div>脚質分布: 逃げ ${counts["逃げ"]} / 先行 ${counts["先行"]} / 差し ${counts["差し"]} / 追込 ${counts["追込"]}${counts["不明"] ? ` (不明 ${counts["不明"]})` : ""}</div>
      <div class="tk-adv">有利脚質: ${advantage}</div>
    </div></div>`;
}

/* ---------------- 全頭分析 ---------------- */
function renderBunseki(r, vb) {
  vb.innerHTML = `<div class="card anc">
      <div class="an-t">AI評価 × 市場人気
        <small>右上＝人気薄なのにAI高評価（妙味） / 左下＝人気だがAI低評価（過剰）・点の大きさ＝勝率</small></div>
      <div id="ch-scatter" class="chart"></div>
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
    grid: { left: 10, right: 16, top: 30, bottom: 46, containLabel: true },
    tooltip: {
      backgroundColor: "rgba(13,20,36,.95)", borderColor: "#32456e",
      textStyle: { color: "#edf1fb", fontSize: 12 },
      formatter: (p) => { const h = p.data.h; return `<b>${h.umaban} ${esc(h.name)}</b> ${h.mark || ""}<br>`
        + `${h.ninki ?? "—"}番人気 / 単${num(h.odds)}倍<br>AI複勝圏 ${pct(h.p_sho)}% / 勝率 ${pct(h.p_win)}%<br>単EV ${num(h.ev_tan, 2)}`; },
    },
    xAxis: Object.assign({ name: "市場人気（左ほど上位人気）", nameLocation: "middle", nameGap: 30, min: 0.5, max: N + 0.5, interval: 1, inverse: false }, axisStyle()),
    yAxis: Object.assign({ name: "AI複勝圏率（%）", nameLocation: "middle", nameRotate: 90, nameGap: 34, min: 0 }, axisStyle()),
    series: [{
      type: "scatter", data: pts,
      markLine: {
        silent: true, symbol: "none", lineStyle: { color: "#3a4d75", type: "dashed" },
        label: { show: false },
        data: [{ yAxis: +avgSho.toFixed(1) }, { xAxis: (N + 1) / 2 }],
      },
    }],
  });

}

/* ---------------- コース ---------------- */
function renderCourse(r, vb) {
  const taikei = taikeiHtml(r);
  const wireTaikei = () => vb.querySelectorAll(".tk-chip").forEach((c) => { c.onclick = () => openDrawer(+c.dataset.uma); });
  const key = `${r.place}|${r.course}`;
  const cs = state.day.courses ? state.day.courses[key] : null;
  if (!cs) {
    vb.innerHTML = taikei + `<div class="card cw-empty">このコース（${esc(key)}）の集計データがありません。</div>`;
    wireTaikei();
    return;
  }
  const marked = r.horses.filter((h) => h.mark);
  const honmei = r.horses.find((h) => h.mark === "◎");
  const wakuOf = (h) => String(h.waku ?? "");
  const tags = (pred) => marked.filter(pred).map((h) =>
    `<span class="cc-tag" style="color:${MARK_COLOR[h.mark]}">${h.mark}${h.umaban}</span>`).join("");

  vb.innerHTML = taikei + `<div class="cc-head card">
    <div class="cc-h-t">${esc(r.place)} ${esc(r.course)} <small>過去 ${cs.n_races?.toLocaleString()}レース / ${cs.n_starts?.toLocaleString()}頭 の傾向</small></div>
    ${honmei ? `<div class="cc-h-s">◎${honmei.umaban} ${esc(honmei.name)} … <b>${wakuOf(honmei)}枠</b> / 脚質 <b>${esc(honmei.style || "—")}</b></div>` : ""}
  </div>
  <div class="cc-grid">
    <div class="card anc"><div class="an-t">枠順別 複勝率 <small>印馬の枠を強調</small></div><div id="ch-waku" class="chart sm"></div></div>
    <div class="card anc"><div class="an-t">脚質別 複勝率 <small>${["逃げ", "先行", "差し", "追込"].map((k) => `${k}${tags((h) => h.style === k) ? "•" : ""}`).join(" ")}</small></div><div id="ch-kyaku" class="chart sm"></div></div>
    <div class="card anc"><div class="an-t">年齢別 複勝率</div><div id="ch-age" class="chart sm"></div></div>
    <div class="card anc"><div class="an-t">性別 複勝率</div><div id="ch-sex" class="chart sm"></div></div>
  </div>`;
  wireTaikei();

  const baseFuku = cs.n_starts ? null : null;
  const markedWaku = new Set(marked.map((h) => String(h.waku)));
  const markedStyle = {};
  marked.forEach((h) => { markedStyle[h.style] = (markedStyle[h.style] || 0) + 1; });

  const barChart = (id, rows, opt = {}) => {
    const cats = rows.map((x) => x.label);
    const vals = rows.map((x) => x.fuku);
    const hl = opt.highlight || (() => false);
    mkChart(id, {
      grid: { left: 8, right: 12, top: 34, bottom: 6, containLabel: true },
      tooltip: {
        backgroundColor: "rgba(13,20,36,.95)", borderColor: "#32456e", textStyle: { color: "#edf1fb", fontSize: 12 },
        formatter: (p) => { const x = rows[p[0].dataIndex]; return `${x.label}<br>複勝率 ${x.fuku}% / 勝率 ${x.win}%<br>n=${(x.n || 0).toLocaleString()}`; },
      },
      xAxis: Object.assign({ type: "category", data: cats }, axisStyle()),
      yAxis: Object.assign({ type: "value", name: "複勝率 %", nameGap: 10, min: 0 }, axisStyle()),
      series: [{
        type: "bar", data: vals.map((v, i) => {
          const label = rows[i].label;
          const high = hl(label);
          const color = opt.colorByWaku ? (WAKU_BAR[String(label)] || "#3f6fb8") : (high ? "#f5b942" : "#3f6fb8");
          return {
            value: v,
            itemStyle: {
              color, borderRadius: [3, 3, 0, 0],
              borderColor: opt.colorByWaku ? (high ? "#f5b942" : "rgba(233,238,250,.28)") : "transparent",
              borderWidth: opt.colorByWaku ? (high ? 2.5 : 1) : 0,
            },
            label: (opt.colorByWaku && high) ? { color: "#f5b942", fontWeight: 700 } : undefined,
          };
        }),
        barWidth: opt.barWidth || "58%",
        label: { show: true, position: "top", color: "#a9b6d3", fontSize: 10, formatter: (p) => p.value + "%" },
      }],
    });
  };
  barChart("ch-waku", cs.waku, { highlight: (l) => markedWaku.has(l), colorByWaku: true });
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

/* ---------------- 重賞 Grade Scope ---------------- */
function mdToHtml(md) {
  const inline = (t) => esc(t)
    .replace(/\*\*(.+?)\*\*/g, '<b>$1</b>')
    .replace(/`(.+?)`/g, "$1");
  let html = "", inList = false;
  const closeList = () => { if (inList) { html += "</ul>"; inList = false; } };
  for (const raw of String(md || "").split("\n")) {
    const line = raw.trim();
    if (/^#{1,4}\s+/.test(line)) {
      closeList();
      html += `<h3 class="gs-h">${inline(line.replace(/^#{1,4}\s+/, ""))}</h3>`;
    } else if (/^[-・*]\s+/.test(line)) {
      if (!inList) { html += '<ul class="gs-ul">'; inList = true; }
      html += `<li>${inline(line.replace(/^[-・*]\s+/, ""))}</li>`;
    } else if (line === "") {
      closeList();
    } else {
      closeList();
      html += `<p class="gs-p">${inline(line)}</p>`;
    }
  }
  closeList();
  return html;
}

function renderGrade(r, vb) {
  const gs = r.grade_scope;
  if (!gs) {
    vb.innerHTML = `<div class="card cw-empty">この重賞の詳細見解はまだありません。</div>`;
    return;
  }
  vb.innerHTML = `<div class="card gs-card">
    <div class="gs-bar">
      <span class="gs-badge">${esc(gs.klass || "重賞")}</span>
      <span class="gs-title">${esc(gs.race_label || r.race_name || (r.place + r.rno + "R"))}</span>
    </div>
    <div class="gs-body">${mdToHtml(gs.markdown)}</div>
    <div class="gs-foot">🏆 Anthropic Cowork (Claude) による重賞詳細見解</div>
  </div>`;
}

/* ---------------- view machinery ---------------- */
function viewsFor(r) {
  const vs = [...VIEWS];
  if (r && r.grade_scope) vs.splice(1, 0, { key: "grade", label: "🏆 重賞" });
  return vs;
}

function renderViewTabs() {
  const r = currentRace();
  const views = viewsFor(r);
  if (!views.some((v) => v.key === state.view)) state.view = "shutsuba";
  const vt = $("#viewTabs");
  vt.innerHTML = views.map((v) =>
    `<button class="vt ${v.key === state.view ? "on" : ""} ${v.key === "grade" ? "vt-grade" : ""}" data-view="${v.key}">${v.label}</button>`).join("");
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
  if (state.view === "grade" && !r.grade_scope) state.view = "shutsuba";
  if (state.view === "shutsuba") {
    vb.innerHTML = `<section id="shutsuba"></section><section id="extras"></section><section id="cowork"></section>`;
    renderTable(r); renderExtras(r); renderCowork(r);
  } else if (state.view === "grade") {
    renderGrade(r, vb);
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

/* ================= 成績 (Cowork 的中一覧 + 累計収支) ================= */
const BTYPE_COLOR = {
  "単勝": "#f5b942", "複勝": "#2dd4a8", "ワイド": "#5ba0f5",
  "馬連": "#b78cf2", "馬単": "#e4549a", "三連複": "#f0a132", "三連単": "#f2555a",
};
function rsDate(d8) {
  return `${+d8.slice(4, 6)}/${+d8.slice(6, 8)}`;
}

async function renderResults() {
  const rm = $("#resultsMain");
  rm.innerHTML = `<div class="rs-wrap"><div class="loading">LOADING…</div></div>`;
  if (!resultsData) {
    try {
      const v = encodeURIComponent(state.manifest?.built_at || "0");
      resultsData = await (await fetch(`data/results.json?v=${v}`)).json();
    } catch (e) {
      rm.innerHTML = `<div class="rs-wrap"><div class="err">data/results.json を読めませんでした。</div></div>`;
      return;
    }
  }
  const a = resultsData.agg || {};
  const hits = resultsData.hits || [];
  const pCls = (a.total_profit ?? 0) >= 0 ? "pos" : "neg";

  const byType = Object.entries(a.by_type || {})
    .sort((x, y) => (y[1].roi || 0) - (x[1].roi || 0))
    .map(([t, v]) => `<div class="rs-bt">
      <span class="rs-bt-h"><span class="rs-dot" style="background:${BTYPE_COLOR[t] || "#888"}"></span>${esc(t)}</span>
      <span class="rs-bt-roi ${v.roi >= 100 ? "pos" : v.roi >= 80 ? "mid" : "neg"}">${v.roi}%</span>
      <span class="rs-bt-sub">的中 ${v.wins}/${v.n}・収支 ${v.profit >= 0 ? "+" : ""}${(v.profit).toLocaleString()}</span>
    </div>`).join("");

  const cards = hits.map((h) => `<button class="hit-card t-${esc(h.btype)}" data-date="${h.date}" data-rid="${esc(h.race_id)}">
    <div class="hit-stamp">的中</div>
    <div class="hit-info">
      <div class="hit-meta">${rsDate(h.date)} ${esc(h.place)}${h.rno}R</div>
      <div class="hit-name">${esc(h.name)}</div>
      <div class="hit-row"><span class="hit-type" style="background:${BTYPE_COLOR[h.btype] || "#888"}">${esc(h.btype)}</span>
        <span class="hit-pay">${yen(h.payout)}</span></div>
    </div>
  </button>`).join("");

  rm.innerHTML = `<div class="rs-wrap">
    <div class="card rs-sum">
      <div class="rs-sum-grid">
        <div class="rs-stat"><div class="k">累計投資</div><div class="v">${yen(a.total_cost)}</div></div>
        <div class="rs-stat"><div class="k">累計収支</div><div class="v ${pCls}">${(a.total_profit ?? 0) >= 0 ? "+" : ""}${yen(a.total_profit).slice(1)}</div></div>
        <div class="rs-stat"><div class="k">回収率 ROI</div><div class="v ${pCls}">${a.roi}%</div></div>
        <div class="rs-stat"><div class="k">的中率</div><div class="v">${a.hit_rate}%<small> ${a.n_wins}/${a.n_bets}</small></div></div>
      </div>
      <div class="rs-bt-grid">${byType}</div>
      ${a.n_unsettled ? `<div class="rs-note">⚠ ワイド ${a.n_unsettled}件 (¥${(a.unsettled_cost).toLocaleString()}) は払戻データ未取込のため集計外</div>` : ""}
    </div>
    <div class="rs-h">🎯 的中一覧 <small>${hits.length}件・新しい順／配当順・カードで詳細</small></div>
    <div class="hit-grid">${cards || `<div class="cw-empty">的中データがありません。</div>`}</div>
  </div>`;

  rm.querySelectorAll(".hit-card").forEach((c) => {
    c.onclick = () => openResultDetail(c.dataset.date, c.dataset.rid);
  });
}

function horseByUma(r, uma) {
  return r.horses.find((h) => h.umaban === uma);
}

function resultTableHtml(r) {
  const res = r.result;
  if (!res) return "";
  const pays = res.pays || {};
  const top3 = res.top3 || [];
  const nin = (u) => { const h = horseByUma(r, u); return h && h.ninki != null ? `${h.ninki}番人気` : "—"; };
  const wk2 = (u) => { const h = horseByUma(r, u); return h ? h.waku : "?"; };
  const rows = [];
  const add = (label, sel, pay, pop) => {
    if (pay == null) return;
    rows.push(`<tr><td class="rt2-l">${label}</td><td class="rt2-s">${sel}</td>
      <td class="rt2-p num">${yen(pay)}</td><td class="rt2-n">${pop || ""}</td></tr>`);
  };
  if (top3[0] != null) add("単勝", top3[0], pays.tan, nin(top3[0]));
  top3.forEach((u, i) => add(i === 0 ? "複勝" : "", u, (pays.fuku || {})[String(u)], nin(u)));
  if (top3.length >= 2) add("枠連", `${wk2(top3[0])}-${wk2(top3[1])}`, pays.wakuren, "");
  if (top3.length >= 2) add("馬連", `${Math.min(top3[0], top3[1])}-${Math.max(top3[0], top3[1])}`, pays.umaren, "");
  if (top3.length >= 2) add("馬単", `${top3[0]}→${top3[1]}`, pays.umatan, "");
  const wide = pays.wide || {};
  let wfirst = true;
  if (top3.length >= 3) {
    [[0, 1], [0, 2], [1, 2]].forEach(([i, j]) => {
      const a = Math.min(top3[i], top3[j]), b = Math.max(top3[i], top3[j]);
      const p = wide[`${a}-${b}`];
      if (p != null) { add(wfirst ? "ワイド" : "", `${a}-${b}`, p, ""); wfirst = false; }
    });
  }
  if (top3.length >= 3) add("三連複", top3.slice(0, 3).slice().sort((a, b) => a - b).join("-"), pays.sanrenpuku, "");
  if (top3.length >= 3) add("三連単", top3.slice(0, 3).join("→"), pays.sanrentan, "");

  const finish = top3.map((u, i) => {
    const h = horseByUma(r, u);
    return `<span class="rt2-fin">${posBadge(i + 1)}${h ? wk(h) : ""}<b>${esc(h ? h.name : u)}</b>${h && h.mark ? `<span class="mark ${markCls(h.mark)}">${h.mark}</span>` : ""}</span>`;
  }).join("");

  return `<div class="card rs-result">
    <div class="rs-sec">レース結果</div>
    <div class="rt2-finish">${finish}</div>
    <table class="rt2"><tbody>${rows.join("")}</tbody></table>
  </div>`;
}

async function openResultDetail(date, rid) {
  const rm = $("#resultsMain");
  rm.innerHTML = `<div class="rs-wrap"><div class="loading">LOADING…</div></div>`;
  let day = dayCache.get(date);
  if (!day) {
    try {
      const v = encodeURIComponent(state.manifest?.built_at || "0");
      day = await (await fetch(`data/${date}.json?v=${v}`)).json();
      dayCache.set(date, day);
    } catch (e) {
      rm.innerHTML = `<div class="rs-wrap"><div class="err">${date} のデータを読めませんでした。</div></div>`;
      return;
    }
  }
  const r = (day.races || []).find((x) => String(x.race_id) === String(rid));
  if (!r) { rm.innerHTML = `<div class="rs-wrap"><div class="err">レースが見つかりません。</div></div>`; return; }

  const cw = r.cowork || {};
  const bets = cw.bets || [];
  const settled = r.bets_settled || [];
  const isTurf = (r.course || "").startsWith("芝");
  const betCards = bets.map((b, i) => {
    const st = settled[i] || {};
    const col = BTYPE_COLOR[b.type] || "#97a4c2";
    const amt = typeof b.amount === "number" ? b.amount.toLocaleString() : esc(b.amount);
    return `<div class="card ticket ${st.is_win ? "won" : st.settled === false ? "" : "lost"}" style="--bcol:${col}">
      <div class="ticket-type">${esc(b.type)}${st.is_win ? `<span class="won-badge">的中 ${yen(st.payout)}</span>` : ""}</div>
      <div class="ticket-sel">${esc(b.selection)}</div>
      <div class="ticket-amt"><b>¥${amt}</b>${st.settled !== false ? ` <span class="ticket-pl ${st.profit >= 0 ? "pos" : "neg"}">${st.profit >= 0 ? "+" : ""}${Math.round(st.profit || 0).toLocaleString()}</span>` : ""}</div>
      ${b.reason ? `<div class="ticket-reason">${esc(b.reason)}</div>` : ""}
    </div>`;
  }).join("");

  rm.innerHTML = `<div class="rs-wrap">
    <button class="rs-back" id="rsBack">← 的中一覧へ戻る</button>
    <div class="card rh" style="margin-top:10px">
      <div class="rh-main">
        <div class="rh-title"><span class="rh-place">${esc(r.place)}</span><span class="rh-rno">${r.rno}R</span>
          ${r.race_name ? `<span class="rh-name">${esc(r.race_name)}</span>` : ""}
          <span class="rh-time num">${rsDate(date)}</span></div>
        <div class="rh-sub"><span class="tdchip ${isTurf ? "turf" : "dirt"}">${esc(r.course)}</span>
          <span class="mchip">${esc(r.klass)}</span><span class="mchip">${r.field_size}頭</span></div>
      </div>
    </div>
    ${bets.length ? `<div class="rs-sec2">AI予想・買い目</div><div class="bet-grid">${betCards}</div>` : ""}
    ${resultTableHtml(r)}
  </div>`;
  $("#rsBack").onclick = () => renderResults();
  window.scrollTo(0, 0);
}

boot();
