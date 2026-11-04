/* PyCaLiAI 回顧ページ (2026-11-04)
   data/manifest.json の結果あり開催日 → GET /api/kaiko?date= で記事一覧 →
   未生成のレースは POST /api/kaiko/gen をこの端末から順に呼んで埋める（KV に保存されるので 2 人目以降は即表示）。 */
"use strict";
const API = location.hostname.endsWith("pycaliai.com") ? "" : "https://pycaliai.com";
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const WD = ["日", "月", "火", "水", "木", "金", "土"];
const fmtDate = (d8) => `${+d8.slice(4, 6)}/${+d8.slice(6, 8)} (${WD[new Date(+d8.slice(0, 4), +d8.slice(4, 6) - 1, +d8.slice(6, 8)).getDay()]})`;
const MK = { "◎": "m1", "〇": "m2", "○": "m2", "▲": "m3", "△": "m4" };

let dates = [], cur = null, data = null, genRunning = false, stopGen = false;

function artHtml(t) {
  return esc(t).replace(/(^|\n)(教訓[:：][^\n]*)/, (m, p, s) => `${p}<span class="lesson">${s}</span>`);
}
function genHtml(msg) { return `<span class="gen">${esc(msg)}<span class="dots"><i></i><i></i><i></i></span></span>`; }

function renderDay() {
  const el = $("#dayCard");
  const d = data.day;
  const head = `<h2>総括 <span class="d">${fmtDate(cur)} ${data.places.map(esc).join("・")}</span></h2>`;
  if (!d) {
    el.innerHTML = head + (data.missing.includes("day") ? genHtml("総括を生成中…") : `<span class="gen">総括はまだありません</span>`);
    return;
  }
  const s = d.stats || {};
  el.innerHTML = head + `
    <div class="stats">
      <span class="stat">レース <b>${s.n_races ?? data.n_races}</b></span>
      <span class="stat">◎ 勝利 <b>${s.honmei_win ?? "—"}</b></span>
      <span class="stat">◎ 複勝圏 <b>${s.honmei_top3 ?? "—"}</b></span>
      <span class="stat">サイト買い目 的中 <b>${s.site_bets_hit ?? "—"}</b>/${s.site_bets_n ?? "—"}</span>
    </div>
    <div class="art">${artHtml(d.text)}</div>
    ${d.x_post ? `<div class="xbox"><div class="xt">X 投稿下書き（${d.x_post.length} 字）<button class="cpy" id="cpyX">コピー</button></div><pre id="xTxt">${esc(d.x_post)}</pre></div>` : ""}`;
  const b = $("#cpyX");
  if (b) b.onclick = () => { navigator.clipboard.writeText(d.x_post).then(() => { b.textContent = "コピーした"; setTimeout(() => (b.textContent = "コピー"), 1500); }).catch(() => {}); };
}

function raceCard(r) {
  const hit = r.bets_n ? `<span class="hit ${r.bets_hit ? "on" : "miss"}">買い目 ${r.bets_hit}/${r.bets_n}</span>` : "";
  const marks = r.marks.map((m) => `<span><span class="mk ${MK[m.mark] || ""}">${esc(m.mark)}</span>${m.umaban}<span class="pos ${m.pos <= 3 ? `p${m.pos}` : ""}">→${m.pos ?? "?"}着</span></span>`).join("");
  const top3 = r.top3.map((t, i) => `<span><span class="pos p${i + 1}">${i + 1}着</span> ${t.umaban} ${esc(t.name)}${t.mark ? `<span class="mk ${MK[t.mark] || ""}">${esc(t.mark)}</span>` : ""}<span class="k">(AI${t.ai_rank ?? "?"}位)</span></span>`).join("");
  return `<article class="panel rc" id="rc-${r.race_id}">
    <div class="rc-h"><span class="rno">${r.rno}R</span><span class="nm">${esc(r.race_name || r.klass)}</span><span class="cl">${esc(r.course)}${r.race_name ? ` ${esc(r.klass)}` : ""}</span>${hit}</div>
    <div class="line"><span class="k">印</span>${marks || "—"}</div>
    <div class="line"><span class="k">結果</span>${top3}</div>
    <div class="art" id="art-${r.race_id}">${r.text ? artHtml(r.text) : genHtml("回顧を生成中…")}</div>
  </article>`;
}

function renderRaces() {
  const byPlace = {};
  for (const r of data.races) (byPlace[r.place] = byPlace[r.place] || []).push(r);
  $("#races").innerHTML = Object.entries(byPlace).map(([p, rs]) =>
    `<div class="place-h">${esc(p)} <small>${rs.length}R</small></div><div class="rgrid">${rs.map(raceCard).join("")}</div>`).join("");
}

function setStatus(t, err = false) { const el = $("#st"); el.textContent = t; el.classList.toggle("err", err); }

async function genMissing() {
  if (genRunning) return;
  genRunning = true; stopGen = false;
  const todo = [...data.missing.filter((x) => x !== "day"), ...(data.missing.includes("day") ? ["day"] : [])];
  let done = 0;
  for (const rid of todo) {
    if (stopGen) break;
    setStatus(`生成中 ${done + 1}/${todo.length}…`);
    try {
      const res = await fetch(`${API}/api/kaiko/gen`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ date: cur, race_id: rid }) });
      const j = await res.json().catch(() => ({}));
      if (!res.ok) {
        if (j.code === "daily_limit" || j.code === "month_limit" || j.code === "no_key") { setStatus(j.error || "生成枠が終了しました", true); break; }
        if (j.code === "busy") { await new Promise((r) => setTimeout(r, 4000)); continue; }
        const el = $(`#art-${rid}`); if (el) el.innerHTML = `<span class="gen err">${esc(j.error || "生成できませんでした")}</span>`;
        continue;
      }
      if (rid === "day") { data.day = j.item; renderDay(); }
      else {
        const r = data.races.find((x) => x.race_id === rid);
        if (r) { r.text = j.item.text; r.bets_hit = j.item.site_bets_hit; r.bets_n = j.item.site_bets_n; const el = $(`#rc-${rid}`); if (el) el.outerHTML = raceCard(r); }
      }
      done++;
    } catch (e) {
      setStatus("通信に失敗しました", true); break;
    }
  }
  genRunning = false;
  if (!stopGen && done === todo.length) setStatus(todo.length ? `${todo.length} 件を生成しました` : "");
}

async function load(date) {
  cur = date; stopGen = true;
  location.hash = date;
  $("#dayCard").innerHTML = `<span class="gen">読み込み中…</span>`;
  $("#races").innerHTML = "";
  setStatus("");
  try {
    const res = await fetch(`${API}/api/kaiko?date=${date}`);
    data = await res.json();
    if (!res.ok) throw new Error(data.error || res.status);
  } catch (e) {
    $("#dayCard").innerHTML = `<span class="gen err">回顧を読み込めませんでした（${esc(e.message)}）</span>`;
    return;
  }
  renderDay(); renderRaces();
  const nMissing = data.missing.length;
  setStatus(nMissing ? `未生成 ${nMissing} 件` : `${data.n_races} レース`);
  if (nMissing) { genRunning = false; genMissing(); }
}

async function boot() {
  let mf;
  try { mf = await (await fetch("data/manifest.json?t=" + Date.now())).json(); }
  catch (e) { $("#dayCard").innerHTML = `<span class="gen err">manifest を読めませんでした</span>`; return; }
  dates = (mf.dates || []).filter((d) => d.has_results);
  const sel = $("#dateSel");
  sel.innerHTML = dates.map((d) => `<option value="${d.date}">${fmtDate(d.date)} ${d.places.join("・")}</option>`).join("");
  sel.onchange = () => load(sel.value);
  const h = location.hash.replace("#", "");
  const first = dates.find((d) => d.date === h) ? h : (dates[0] && dates[0].date);
  if (!first) { $("#dayCard").innerHTML = `<span class="gen">結果のある開催日がまだありません</span>`; return; }
  sel.value = first;
  load(first);
}
document.addEventListener("DOMContentLoaded", boot);
