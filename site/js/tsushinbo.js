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
  // 見出しは本番の予測モデルで発走前の版と確認できた日だけ。切替直後でその版の開催日が無いときは直前の版。
  // 全体は下に併記 (2026-11-13 監査)。
  const cur = d.current && d.current.n ? d.current : null;
  const o = cur || d.overall;
  const all = d.overall;
  const who = cur && cur.era_label
    ? (cur.is_production ? `現行モデル（${cur.era_label}）` : `${cur.era_label}（${cur.production_era_label} に切替前）`)
    : "現行モデル";
  const scope = cur
    ? `${who}・発走前の版と確認済み ${fmtD(cur.from)}〜${fmtD(cur.to)}・${cur.n_days} 開催日`
    : `${fmtD(d.period.from)}〜${fmtD(d.period.to)}・${d.period.n_days} 開催日`;
  const allNote = cur ? `<div class="s">全期間（v5・旧版を含む）: ${all.n.toLocaleString()}R ◎勝率 ${all.win_rate}% ・複勝圏率 ${all.top3_rate}%</div>` : "";
  $("#kpis").innerHTML = `
    <div class="kpi"><div class="l">集計レース</div><div class="v num">${o.n.toLocaleString()}</div><div class="s">${scope}</div>${allNote}</div>
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

/* ---------- AI の調子（週ごと）: ECharts 折れ線 + CI 帯 + 累積 ---------- */
const cssVar = (n, fb) => (getComputedStyle(document.documentElement).getPropertyValue(n) || "").trim() || fb;
const charts = [];

function bandSeries(name, rows, lo, hi, color) {
  // 帯 = 下限を透明で積み、(上限-下限) を半透明で積む（ECharts の stack 方式）
  const base = rows.map((r) => (r[lo] == null ? null : r[lo]));
  const span = rows.map((r) => (r[lo] == null || r[hi] == null ? null : +(r[hi] - r[lo]).toFixed(1)));
  const st = "band-" + name;
  return [
    { name: name + " 下限", type: "line", data: base, stack: st, symbol: "none", lineStyle: { opacity: 0 }, itemStyle: { color }, silent: true, tooltip: { show: false }, z: 1 },
    { name: name + " 95% CI", type: "line", data: span, stack: st, symbol: "none", lineStyle: { opacity: 0 }, itemStyle: { color }, areaStyle: { color, opacity: 0.16 }, silent: true, tooltip: { show: false }, z: 1 },
  ];
}

function trendCharts(d) {
  const w = d.weekly;
  if (!w || !w.rows || w.rows.length < 2 || typeof echarts === "undefined") return;
  const rows = w.rows, cum = w.cumulative;
  const x = rows.map((r) => r.label);
  $("#trend").hidden = false;              // init 前に表示しないと幅 0 で描画される
  const teal = cssVar("--teal", "#2dd4a8"), gold = cssVar("--gold", "#f5b942"), gold2 = cssVar("--gold2", "#ffd97a");
  const tx3 = cssVar("--tx3", "#7385a8"), line = cssVar("--line", "#1e2b4a"), blue = "#5ba0f5", unc = cssVar("--uncertain", "#e0876a");
  const font = cssVar("--disp", "IBM Plex Mono, monospace");
  const axis = {
    xAxis: { type: "category", data: x, boundaryGap: false, axisLine: { lineStyle: { color: line } }, axisTick: { show: false },
      axisLabel: { color: tx3, fontFamily: font, fontSize: 10, interval: Math.max(0, Math.ceil(x.length / 9) - 1), hideOverlap: true } },
    yAxis: { type: "value", axisLabel: { color: tx3, fontFamily: font, fontSize: 10, formatter: "{value}%" }, splitLine: { lineStyle: { color: line } } },
    grid: { left: 42, right: 12, top: 14, bottom: 28 },
    textStyle: { fontFamily: font },
    tooltip: { trigger: "axis", backgroundColor: cssVar("--bg2", "#0a101e"), borderColor: cssVar("--line2", "#32456e"), textStyle: { color: cssVar("--tx", "#eef2ff"), fontSize: 12 },
      formatter: (ps) => {
        const i = ps[0]?.dataIndex; if (i == null) return "";
        const r = rows[i], c = cum[i];
        return `<b>${esc(r.week.slice(0, 4))}/${esc(r.label)} の週</b>　${r.n} レース<br>` + ps.filter((p) => p.value != null).map((p) => `${p.marker} ${esc(p.seriesName)} <b>${p.value}%</b>`).join("<br>") +
          `<br><span style="color:${tx3}">累積 ${c.n} R / ◎勝率 ${c.win_rate}% / 複勝圏 ${c.top3_rate}% / 回収率 ${c.roi ?? "—"}%</span>`;
      } },
  };
  // 左: ◎ 勝率・複勝圏率
  const c1 = echarts.init($("#chart-marks"), null, { renderer: "canvas" });
  c1.setOption({ ...axis, yAxis: { ...axis.yAxis, min: 0, max: 100 }, series: [
    ...bandSeries("◎複勝圏率", rows, "top3_lo", "top3_hi", teal),
    ...bandSeries("◎勝率", rows, "win_lo", "win_hi", gold),
    { name: "◎複勝圏率（週）", type: "line", data: rows.map((r) => r.top3_rate), symbol: "circle", symbolSize: 5, lineStyle: { color: teal, width: 2 }, itemStyle: { color: teal }, z: 3 },
    { name: "◎勝率（週）", type: "line", data: rows.map((r) => r.win_rate), symbol: "circle", symbolSize: 5, lineStyle: { color: gold, width: 2 }, itemStyle: { color: gold }, z: 3 },
    { name: "◎複勝圏率（累積）", type: "line", data: cum.map((r) => r.top3_rate), symbol: "none", lineStyle: { color: teal, width: 1.5, type: "dashed", opacity: 0.9 }, itemStyle: { color: teal }, z: 4 },
    { name: "◎勝率（累積）", type: "line", data: cum.map((r) => r.win_rate), symbol: "none", lineStyle: { color: gold2, width: 1.5, type: "dashed", opacity: 0.9 }, itemStyle: { color: gold2 }, z: 4 },
  ] });
  // 右: 表示買い目の回収率（週=棒、累積=点線、80% 基準線）
  const roiMax = Math.max(150, ...rows.map((r) => r.roi ?? 0)) + 10;
  const c2 = echarts.init($("#chart-roi"), null, { renderer: "canvas" });
  c2.setOption({ ...axis, xAxis: { ...axis.xAxis, boundaryGap: true }, yAxis: { ...axis.yAxis, min: 0, max: Math.ceil(roiMax / 50) * 50 }, series: [
    { name: "回収率（週）", type: "bar", data: rows.map((r) => ({ value: r.roi, itemStyle: { color: r.roi != null && r.roi >= 80 ? gold : blue, opacity: 0.8 } })), barMaxWidth: 18, z: 2,
      markLine: { silent: true, symbol: "none", lineStyle: { color: tx3, type: "dotted" }, label: { show: true, position: "insideEndTop", color: tx3, fontSize: 10, formatter: "80%" }, data: [{ yAxis: 80 }] } },
    { name: "回収率（累積）", type: "line", data: cum.map((r) => r.roi), symbol: "none", lineStyle: { color: unc, width: 2, type: "dashed" }, itemStyle: { color: unc }, z: 4 },
  ] });
  charts.push(c1, c2);
  $("#trend-sub").textContent = `${rows.length} 週・${esc(rows[0].week.slice(0, 4))}/${rows[0].label}〜${rows[rows.length - 1].label}`;
  $("#trend-legend").innerHTML = [
    [teal, "", "◎複勝圏率（週）"], [gold, "", "◎勝率（週）"], [teal, "dash", "複勝圏率（累積）"], [gold2, "dash", "勝率（累積）"],
    [teal, "band", "95% 信頼区間"], [blue, "band", "回収率（週・棒）"], [unc, "dash", "回収率（累積）"],
  ].map(([c, cl, t]) => `<span><i class="${cl}" style="color:${c};${cl === "band" ? `background:${c}` : ""}"></i>${esc(t)}</span>`).join("");
  addEventListener("resize", () => charts.forEach((c) => c.resize()));
}

/* ---------- 見送りの成績（買っていたら？） ---------- */
function mioCol(title, tag, s, skip) {
  const ci = (lo, hi) => (lo == null || hi == null ? "" : `<small>95% CI ${lo}〜${hi}%</small>`);
  return `<div class="mio-col ${skip ? "skip" : ""}">
    <div class="t"><b>${esc(title)}</b><span class="tag">${esc(tag)}</span><span>${s.n_races} レース・${s.n_days} 開催日</span></div>
    <div class="mio-row"><div class="l">レース的中率</div><div><div class="v">${s.hit_rate ?? "—"}<small>%</small>${ci(s.hit_lo, s.hit_hi)}</div>${bar(s.hit_rate, s.hit_lo, s.hit_hi)}</div></div>
    <div class="mio-row"><div class="l">回収率</div><div><div class="v ${s.roi != null && s.roi < 80 ? "low" : ""}">${s.roi ?? "—"}<small>%</small>${ci(s.roi_lo, s.roi_hi)}</div>${roiBar(s.roi)}</div></div>
    <div class="mio-row"><div class="l">収支プラスのR</div><div><div class="v">${s.plus_rate ?? "—"}<small>%</small></div></div></div>
  </div>`;
}

function miokuri(d) {
  const m = d.miokuri;
  if (!m || !m.sanse || !m.miokuri || (m.sanse.n_races + m.miokuri.n_races) === 0) return;
  const v = m.verdict || {};
  const cls = (v.code || "insufficient").split("_")[0];
  $("#mio-verdict").className = "verdict " + cls;
  $("#mio-verdict").textContent = v.text || "";
  $("#mio-grid").innerHTML = mioCol("参戦したレース", "表示した買い目", m.sanse, false) + mioCol("見送ったレース", "買っていたら", m.miokuri, true);
  $("#mio-sub").textContent = `${fmtD(m.period.from)}〜${fmtD(m.period.to)}・${m.period.n_days} 開催日`;
  const skipped = (m.skipped_dates || []).length ? ` 結果未取込で飛ばした開催日: ${m.skipped_dates.map(fmtD).join(", ")}。` : "";
  const all = m.all && m.all.n_races ? ` 参考: 全レースに参戦していた場合は ${m.all.n_races} R・的中 ${m.all.hit_rate}%・回収率 ${m.all.roi}%（CI ${m.all.roi_lo}〜${m.all.roi_hi}%）。` : "";
  $("#mio-note").textContent = `${m.note}${all}${skipped}`;
  $("#mio").hidden = false;
}

async function boot() {
  let d;
  try { d = await (await fetch("data/report_card.json?t=" + Date.now())).json(); }
  catch (e) { $("#dims").innerHTML = `<span class="note err">data/report_card.json を読めませんでした</span>`; return; }
  kpis(d);
  try { trendCharts(d); } catch (e) { console.warn("trend chart", e); }
  try { miokuri(d); } catch (e) { console.warn("miokuri", e); }
  $("#dims").innerHTML = d.dims.map(dimTable).join("");
  $("#note").textContent = `${d.note} 集計 ${d.built_at}。`;
}
document.addEventListener("DOMContentLoaded", boot);
