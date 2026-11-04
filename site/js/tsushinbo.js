/* PyCaLiAI AI の通信簿 (2026-11-04): data/report_card.json（build_report_card.py）を表にする。LLM 不使用。 */
"use strict";
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtD = (d8) => d8 ? `${+d8.slice(4, 6)}/${+d8.slice(6, 8)}` : "—";
const MIN_N = 30, RANK_N = 40;

function bar(v, lo, hi, max = 100) {
  if (v == null) return `<span class="bar"></span>`;
  const w = Math.min(100, v / max * 100);
  const cl = lo != null ? Math.min(100, lo / max * 100) : null, ch = hi != null ? Math.min(100, hi / max * 100) : null;
  return `<span class="bar"><i style="width:${w}%"></i>${cl != null ? `<u style="left:${cl}%;width:${Math.max(0, ch - cl)}%"></u>` : ""}</span>`;
}
function roiBar(v) {
  if (v == null) return `<span class="bar roi" style="--mark:53"></span>`;
  const max = 150;
  return `<span class="bar roi ${v >= 80 ? "ok" : ""}" style="--mark:${80 / max * 100}"><i style="width:${Math.min(100, v / max * 100)}%"></i></span>`;
}

function kpis(d) {
  const o = d.overall;
  $("#kpis").innerHTML = `
    <div class="kpi"><div class="l">集計レース</div><div class="v num">${o.n.toLocaleString()}</div><div class="s">${fmtD(d.period.from)}〜${fmtD(d.period.to)}・${d.period.n_days} 開催日</div></div>
    <div class="kpi"><div class="l">◎ 勝率</div><div class="v num">${o.win_rate}<small>%</small></div><div class="s">95% CI ${o.win_lo}〜${o.win_hi}%</div></div>
    <div class="kpi"><div class="l">◎ 複勝圏率</div><div class="v num">${o.top3_rate}<small>%</small></div><div class="s">95% CI ${o.top3_lo}〜${o.top3_hi}%</div></div>
    <div class="kpi"><div class="l">◎ が朝 1 番人気</div><div class="v num">${o.fav_rate}<small>%</small></div><div class="s">市場と一致した割合</div></div>
    <div class="kpi ${o.roi != null && o.roi < 80 ? "neg" : ""}"><div class="l">表示買い目 回収率</div><div class="v num">${o.roi ?? "—"}<small>%</small></div><div class="s">${o.bets_n.toLocaleString()} 点・的中 ${o.bets_hit_rate ?? "—"}%</div></div>`;
}

function dimTable(dim) {
  const ranked = dim.rows.filter((r) => r.n >= RANK_N && r.top3_rate != null);
  const best = ranked.length >= 2 ? ranked.reduce((a, b) => (b.top3_rate > a.top3_rate ? b : a)).key : null;
  const worst = ranked.length >= 2 ? ranked.reduce((a, b) => (b.top3_rate < a.top3_rate ? b : a)).key : null;
  const rows = dim.rows.map((r) => `<tr class="${r.n < MIN_N ? "thin" : ""} ${r.key === best ? "best" : r.key === worst ? "worst" : ""}">
    <td class="l"><b>${esc(r.key)}</b></td>
    <td class="num">${r.n.toLocaleString()}</td>
    <td class="num">${r.win_rate ?? "—"}%</td>
    <td>${bar(r.top3_rate, r.top3_lo, r.top3_hi)}<span class="num">${r.top3_rate ?? "—"}%</span></td>
    <td class="num">${r.fav_rate ?? "—"}%</td>
    <td class="num">${r.bets_n ? `${r.bets_hit_rate}%` : "—"}</td>
    <td>${roiBar(r.roi)}<span class="num">${r.roi ?? "—"}${r.roi != null ? "%" : ""}</span></td>
  </tr>`).join("");
  return `<section class="panel dim"><h2>${esc(dim.label)} <small>${dim.rows.length} 区分</small></h2>
    <div class="tw"><table class="rc"><thead><tr>
      <th class="l">条件</th><th>件数</th><th>◎勝率</th><th>◎複勝圏率（帯は 95% CI）</th><th>◎=1番人気</th><th>買い目的中</th><th>買い目回収率</th>
    </tr></thead><tbody>${rows}</tbody></table></div></section>`;
}

async function boot() {
  let d;
  try { d = await (await fetch("data/report_card.json?t=" + Date.now())).json(); }
  catch (e) { $("#dims").innerHTML = `<span class="note err">data/report_card.json を読めませんでした</span>`; return; }
  kpis(d);
  $("#dims").innerHTML = d.dims.map(dimTable).join("");
  $("#note").textContent = `${d.note} 集計 ${d.built_at}。`;
}
document.addEventListener("DOMContentLoaded", boot);
