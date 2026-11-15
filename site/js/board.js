/* デイリーボード (2026-11-16): その日の全レースを 1 枚に。
   データは公開済みの site/data/{date}.json と board_stats.json だけ。オッズの数字は出さない。 */
(function () {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = (v, d = 1) => (v == null ? "—" : (v * 100).toFixed(d));
  const HARD = { "固い": 0, "標準": 1, "荒れ": 2 };
  const WD = "日月火水木金土";
  let MF = null, DAY = null, STATS = null, venue = "all", sortKey = "time";

  function fmtDate(d) {
    const dt = new Date(+d.slice(0, 4), +d.slice(4, 6) - 1, +d.slice(6, 8));
    return `${+d.slice(4, 6)}/${+d.slice(6, 8)}（${WD[dt.getDay()]}）`;
  }
  const wk = (h) => `<span class="wk w${h.waku ?? 1}">${h.umaban}</span>`;

  // 1 レースぶんの要約
  function summarize(r) {
    const hs = (r.horses || []).filter((h) => !h.scratched);
    const by = (u) => hs.find((h) => h.umaban === u);
    const hon = hs.find((h) => h.mark === "◎") || null;
    const sp = hs.filter((h) => h.speed && h.speed.v != null);
    const pw = hs.filter((h) => h.pw != null);
    const er = hs.filter((h) => h.speed && h.speed.er != null);
    const si1 = sp.length ? sp.reduce((a, b) => (b.speed.v > a.speed.v ? b : a)) : null;
    const pw1 = pw.length ? pw.reduce((a, b) => (b.pw > a.pw ? b : a)) : null;
    const er1 = er.length ? er.reduce((a, b) => (b.speed.er < a.speed.er ? b : a)) : null;
    let agree = null;
    if (hon && si1 && pw1 && er1) {
      agree = (si1.umaban === hon.umaban) + (pw1.umaban === hon.umaban) + (er1.umaban === hon.umaban);
    }
    const res = r.result && r.result.order ? r.result : null;
    let win = null, honPos = null;
    if (res) {
      const w = Object.entries(res.order).find(([, p]) => p === 1);
      win = w ? by(+w[0]) || (r.horses || []).find((h) => h.umaban === +w[0]) : null;
      if (hon) honPos = res.order[String(hon.umaban)] ?? null;
    }
    const j = r.judgment || {};
    return { r, hon, si1, pw1, er1, agree, res, win, honPos, hard: j.hardness || null, chaos: j.chaos_pct };
  }

  function topCell(cls, label, h, hon) {
    if (!h) return `<span class="tp ${cls}"><span class="lb">${label}</span><span class="na">—</span></span>`;
    const same = hon && h.umaban === hon.umaban;
    if (same) return `<span class="tp ${cls}" title="◎と同じ馬"><span class="lb">${label}</span><span class="row"><span class="same-chip">＝◎</span></span></span>`;
    return `<span class="tp ${cls}"><span class="lb">${label}</span><span class="row">${wk(h)}<span class="nm">${esc(h.name)}</span></span></span>`;
  }

  function rowHtml(s) {
    const r = s.r;
    const rid = r.race_id;
    const hd = s.hard != null ? HARD[s.hard] : null;
    const chaos = s.chaos == null ? null : Math.max(0, Math.min(1, s.chaos));
    const ag = s.agree == null
      ? `<span class="ag"><span class="na">—</span></span>`
      : `<span class="ag k${s.agree}" title="◎と同じ馬だった物差しの数（速度1位・PW1位・推定1番人気）">${[0, 1, 2].map((i) => `<i class="${i < s.agree ? "on" : ""}"></i>`).join("")}<b>${s.agree}/3</b></span>`;
    let res = `<span class="res"><span class="na">${s.res ? "—" : "結果待ち"}</span></span>`;
    if (s.res && s.win) {
      const hp = s.honPos;
      res = `<span class="res">${wk(s.win)}<span class="nm">${esc(s.win.name)}</span>${s.hon ? `<span class="hp ${hp != null && hp <= 3 ? "ok" : "ng"}">◎ <b>${hp == null ? "—" : hp + "着"}</b></span>` : ""}</span>`;
    }
    const field = r.field_size ?? (r.horses || []).length;
    return `<a class="bd-row ${s.agree === 3 ? "a3" : ""}" href="index.html#race=${esc(rid)}">
      <span class="bd-time">${esc(r.start_time || "")}</span>
      <span class="bd-rno"><small>${esc(r.place)}</small>${r.rno}R</span>
      <span class="bd-cond"><b>${esc(r.race_name || r.klass || "")}</b>${esc(r.klass && r.race_name ? r.klass + " ・ " : "")}${esc(r.course || "")} ・ ${field}頭</span>
      <span class="hd">${hd == null ? `<span class="na">—</span>` : `<span class="hd-tag h${hd}">${esc(s.hard)}</span>`}${chaos == null ? "" : `<span class="hd-bar" title="混戦度（右ほど荒れやすい）"><i style="width:${Math.round(chaos * 100)}%"></i></span>`}</span>
      <span class="hon">${s.hon ? `<span class="mk">◎</span>${wk(s.hon)}<span class="nm">${esc(s.hon.name)}</span><span class="pw">${pct(s.hon.p_win)}%</span>` : `<span class="na">◎なし</span>`}</span>
      ${topCell("si", "速度1位", s.si1, s.hon)}
      ${topCell("pwc", "PW1位", s.pw1, s.hon)}
      ${topCell("er", "推定1番人気", s.er1, s.hon)}
      ${ag}
      ${res}
    </a>`;
  }

  function render() {
    if (!DAY) return;
    let list = DAY.races.map(summarize);
    if (venue !== "all") list = list.filter((s) => s.r.place === venue);
    const tkey = (s) => (s.r.start_time || "99:99") + s.r.place + String(s.r.rno).padStart(2, "0");
    list.sort((a, b) => {
      if (sortKey === "agree") return (b.agree ?? -1) - (a.agree ?? -1) || tkey(a).localeCompare(tkey(b));
      if (sortKey === "hard") return (a.chaos ?? 9) - (b.chaos ?? 9) || tkey(a).localeCompare(tkey(b));
      return tkey(a).localeCompare(tkey(b));
    });
    $("#rows").innerHTML = list.length ? list.map(rowHtml).join("") : `<div class="empty">レースがありません</div>`;

    // KPI
    const n = list.length;
    const cnt = (f) => list.filter(f).length;
    const h = [cnt((s) => s.hard === "固い"), cnt((s) => s.hard === "標準"), cnt((s) => s.hard === "荒れ")];
    const a3 = cnt((s) => s.agree === 3), a2 = cnt((s) => s.agree >= 2);
    const done = list.filter((s) => s.res && s.hon);
    let kp = `<div class="kpi"><div class="l">レース数</div><div class="v">${n}<small>R</small></div>
        <div class="hard3"><span>固い <b style="color:#5fe3bd">${h[0]}</b></span><span>標準 <b>${h[1]}</b></span><span>荒れ <b style="color:#f88f92">${h[2]}</b></span></div></div>
      <div class="kpi"><div class="l">一致 3/3（全部そろった）</div><div class="v" style="color:#ffd84a">${a3}<small>R</small></div><div class="s">2 つ以上そろった ${a2}R</div></div>`;
    if (done.length) {
      const w = done.filter((s) => s.honPos === 1).length, t3 = done.filter((s) => s.honPos != null && s.honPos <= 3).length;
      const d3 = done.filter((s) => s.agree === 3);
      const d3t = d3.filter((s) => s.honPos != null && s.honPos <= 3).length;
      kp += `<div class="kpi"><div class="l">この日の◎</div><div class="v">${t3}<small>/ ${done.length}R 3着内</small></div><div class="s">1 着 ${w}R</div></div>
        <div class="kpi"><div class="l">この日の一致 3/3 の◎</div><div class="v">${d3.length ? d3t : "—"}<small>${d3.length ? "/ " + d3.length + "R 3着内" : ""}</small></div><div class="s">全部そろったレースだけ</div></div>`;
    } else if (STATS) {
      const s3 = STATS.by_agree.find((x) => x.k === 3), s0 = STATS.by_agree.find((x) => x.k === 0);
      kp += `<div class="kpi"><div class="l">一致 3/3 の◎（2026 年）</div><div class="v">${pct(s3.top3, 0)}<small>% 3着内</small></div><div class="s">${s3.n.toLocaleString("ja-JP")}R ・ 1 着 ${pct(s3.win, 0)}%</div></div>
        <div class="kpi"><div class="l">一致 0/3 の◎（2026 年）</div><div class="v" style="color:var(--tx2)">${pct(s0.top3, 0)}<small>% 3着内</small></div><div class="s">${s0.n.toLocaleString("ja-JP")}R ・ 1 着 ${pct(s0.win, 0)}%</div></div>`;
    }
    $("#kpis").innerHTML = kp;
  }

  function renderVenues() {
    const places = [...new Set(DAY.races.map((r) => r.place))];
    if (!places.includes(venue)) venue = "all";
    $("#venuePills").innerHTML = [["all", "全場"], ...places.map((p) => [p, p])]
      .map(([k, l]) => `<button data-v="${esc(k)}" class="${k === venue ? "on" : ""}">${esc(l)}</button>`).join("");
  }

  function renderStats() {
    if (!STATS || !STATS.n_races) return;
    $("#statPanel").hidden = false;
    const f = STATS.from, t = STATS.to;
    $("#statRange").textContent = `${+f.slice(4, 6)}/${+f.slice(6, 8)}〜${+t.slice(4, 6)}/${+t.slice(6, 8)} ・ ${STATS.n_races.toLocaleString("ja-JP")} レース`;
    $("#agreeTbl tbody").innerHTML = STATS.by_agree.map((x) => `<tr>
        <td class="k"><span class="ag k${x.k}">${[0, 1, 2].map((i) => `<i class="${i < x.k ? "on" : ""}"></i>`).join("")}<b>${x.k}/3</b></span></td>
        <td class="n">${x.n.toLocaleString("ja-JP")}</td><td class="n">${pct(x.win)}%</td><td class="n"><b style="color:var(--tx)">${pct(x.top3)}%</b></td>
        <td><div class="gbar"><u style="left:${x.top3_lo * 100}%;width:${(x.top3_hi - x.top3_lo) * 100}%"></u><i style="width:${x.top3 * 100}%;opacity:.85"></i></div></td></tr>`).join("");
    $("#hardTbl tbody").innerHTML = STATS.by_hardness.map((x) => `<tr>
        <td class="k"><span class="hd-tag h${HARD[x.k]}">${esc(x.k)}</span></td>
        <td class="n">${x.n.toLocaleString("ja-JP")}</td><td class="n">${pct(x.win)}%</td><td class="n">${pct(x.top3)}%</td></tr>`).join("");
  }

  async function load(date) {
    $("#rows").innerHTML = `<div class="empty">読み込み中…</div>`;
    try {
      DAY = await fetch(`data/${date}.json`, { cache: "no-cache" }).then((r) => r.json());
    } catch (e) {
      $("#rows").innerHTML = `<div class="empty">この日のデータを読み込めませんでした</div>`;
      return;
    }
    const u = new URL(location.href);
    u.searchParams.set("date", date);
    history.replaceState(null, "", u);
    document.title = `PyCaLiAI — デイリーボード ${fmtDate(date)}`;
    renderVenues();
    render();
  }

  async function init() {
    try {
      MF = await fetch("data/manifest.json", { cache: "no-cache" }).then((r) => r.json());
    } catch (e) {
      $("#rows").innerHTML = `<div class="empty">読み込めませんでした</div>`;
      return;
    }
    STATS = await fetch("data/board_stats.json", { cache: "no-cache" }).then((r) => (r.ok ? r.json() : null)).catch(() => null);
    renderStats();
    const dates = MF.dates.map((d) => d.date);
    const want = new URLSearchParams(location.search).get("date");
    const date = dates.includes(want) ? want : dates[0];
    const sel = $("#dateSel");
    sel.innerHTML = MF.dates.map((d) => `<option value="${d.date}">${fmtDate(d.date)}${d.has_results ? " ✓" : ""}</option>`).join("");
    sel.value = date;
    sel.onchange = () => load(sel.value);
    $("#venuePills").addEventListener("click", (e) => {
      const b = e.target.closest("button"); if (!b) return;
      venue = b.dataset.v; renderVenues(); render();
    });
    $("#sortPills").addEventListener("click", (e) => {
      const b = e.target.closest("button"); if (!b) return;
      sortKey = b.dataset.s;
      document.querySelectorAll("#sortPills button").forEach((x) => x.classList.toggle("on", x === b));
      render();
    });
    load(date);
  }
  init();
})();
