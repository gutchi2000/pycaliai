/* PyCaLiAI — 出世予測 AI 答え合わせページ (site/shusse_check.html)
   データ: data/shusse/check_index.json (年×基準点の指標・混同行列・較正・例) + data/shusse/check_{year}.json (馬ごとの確定行)
   生成元: analysis/shusse/check_export.py → build_site.py が site/data/shusse/ へコピー。表示専用、市場情報なし。 */
(function () {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const KNAME = { 1: "1 走目", 2: "2 走目", 3: "3 走目", 5: "5 走目" };
  const nfkc = (s) => String(s || "").normalize("NFKC").toLowerCase();
  const COL = { k: 0, base: 1, pg12: 2, pd: 3, lvl: 4, pct: 5, ran: 6, alvl: 7, g12: 8, g24: 9 };

  const state = { index: null, years: [], year: null, k: 3, cache: {}, q: "", vd: "all", sort: "pg12", shown: 100, open: new Set() };

  async function getJSON(u) { const r = await fetch(u + "?t=" + Date.now()); if (!r.ok) throw new Error(u); return r.json(); }
  function fmtP(p) { if (p == null || isNaN(p)) return "–"; const v = p * 100; if (v > 0 && v < 0.05) return "<0.1%"; return (v < 10 ? v.toFixed(1) : Math.round(v)) + "%"; }
  function f3(x, nd) { return x == null ? "–" : Number(x).toFixed(nd == null ? 2 : nd); }
  function lvName(i) { const lv = state.index && state.index.levels; return lv && lv[String(i)] ? lv[String(i)] : ["未勝利圏", "1勝クラス", "2勝クラス", "3勝/OP", "重賞掲示板"][i]; }
  function fmtDate(s) { return s ? String(s).replace(/-/g, "/").slice(2) : "–"; }

  /* 判定: narrate.py と同じ規則 */
  function verdict(row) {
    if (row[COL.ran] === 0) return { cls: "drop", label: "離脱" };
    const d = Math.abs(row[COL.lvl] - row[COL.alvl]);
    return d === 0 ? { cls: "hit", label: "的中" } : d === 1 ? { cls: "near", label: "惜しい" } : { cls: "miss", label: "外れ" };
  }
  function rowAt(h, k) { return (h.rows || []).find((r) => r[COL.k] === k) || null; }

  /* ---------------- 指標 ---------------- */
  function renderMetrics() {
    const by = state.index.by_year[String(state.year)];
    const m = by && by.k[String(state.k)];
    if (!m) { $("#kpis").innerHTML = `<div class="kpi"><span class="k">この基準点の確定馬はまだいません</span></div>`; $("#calib").innerHTML = ""; $("#cm").innerHTML = ""; $("#examples").innerHTML = ""; return; }
    const ci = (a) => (a ? `<small>[${f3(a[0])}–${f3(a[1])}]</small>` : "");
    $("#kpis").innerHTML = [
      `<div class="kpi"><span class="k">確定した馬</span><span class="v">${m.n.toLocaleString()}<small>頭</small></span><span class="s">うち 1 年以内に出走 ${m.n_ran.toLocaleString()} 頭、離脱 ${fmtP(m.dropout_rate)}</span></div>`,
      `<div class="kpi"><span class="k">到達クラス 的中率</span><span class="v">${fmtP(m.exact_rate)}</span><span class="s">1 段違いまで含めると ${fmtP(m.within1_rate)}（走った馬）</span></div>`,
      `<div class="kpi"><span class="k">クラス一致度 QWK</span><span class="v">${f3(m.qwk)}${ci(m.qwk_ci)}</span><span class="s">規則ベースの単純予測は ${f3(m.rule_qwk)}。1 が完全一致</span></div>`,
      `<div class="kpi"><span class="k">重賞掲示板 AUC</span><span class="v">${f3(m.g12_auc)}${ci(m.g12_auc_ci)}</span><span class="s">実際に載った馬 ${fmtP(m.graded12_rate)}。0.5 が偶然</span></div>`,
      `<div class="kpi"><span class="k">離脱 AUC</span><span class="v">${f3(m.d_auc)}</span><span class="s">1 年間走らなくなる馬の見分け</span></div>`,
    ].join("");
    // 較正
    $("#calib").innerHTML = (m.calibration || []).map((b) => {
      const pr = Math.round((b.mean_p || 0) * 100), ac = Math.round((b.rate || 0) * 100);
      return `<div class="cb"><span class="rng">${Math.round(b.lo * 100)}–${Math.round(b.hi * 100)}%</span>
        <span class="bars"><i class="pr" style="width:${Math.max(1, pr)}%"></i><i class="ac" style="width:${Math.max(1, ac)}%"></i></span>
        <span class="n">${fmtP(b.rate)} <small>/${b.n}</small></span></div>`;
    }).join("");
    // 混同行列
    const conf = m.confusion, drop = m.dropout_by_pred;
    const mx = Math.max(1, ...conf.flat());
    let html = `<thead><tr><th class="l">予測 ＼ 実際</th>${[0, 1, 2, 3, 4].map((j) => `<th>${esc(lvName(j))}</th>`).join("")}<th>離脱</th><th>計</th></tr></thead><tbody>`;
    for (let i = 0; i < 5; i++) {
      const tot = conf[i].reduce((a, b) => a + b, 0) + drop[i];
      html += `<tr><td class="l">${esc(lvName(i))}</td>${conf[i].map((v, j) => `<td class="${i === j ? "diag" : ""}"><i class="heat" style="opacity:${(0.45 * v / mx).toFixed(2)}"></i><span class="v">${v}</span></td>`).join("")}<td class="drop">${drop[i]}</td><td class="num">${tot}</td></tr>`;
    }
    $("#cm").innerHTML = html + "</tbody>";
    // 例
    const ex = by.examples && by.examples[String(state.k)];
    const card = (e, kind) => {
      let line;
      if (kind === "miss") line = `予測 <b>${fmtP(e.pg12)}</b>・${esc(lvName(e.lvl))} → 実際は ${esc(lvName(e.alvl))}`;
      else if (kind === "dropout") line = `予測 <b>${fmtP(e.pg12)}</b>・離脱確率 ${fmtP(e.pd)} → 1 年間出走なし`;
      else line = `予測 <b>${fmtP(e.pg12)}</b>・${esc(lvName(e.lvl))} → 重賞掲示板に到達`;
      return `<div class="exc" data-id="${e.id}" role="button" tabindex="0"><span class="nm">${esc(e.name)}<small style="color:var(--tx3);font-weight:400"> ${e.trainer ? esc(e.trainer) + "厩舎" : ""}</small></span><span class="line">${line}</span><span class="sum" title="${esc(e.summary || "")}">${esc(e.summary || "")}</span></div>`;
    };
    $("#examples").innerHTML = !ex ? "" : [
      `<div class="ex-h">高い確率を付けたのに、重賞掲示板に届かなかった馬</div><div class="ex-grid">${ex.miss.map((e) => card(e, "miss")).join("")}</div>`,
      `<div class="ex-h">高い確率を付けたのに、1 年間走らなかった馬（故障などの離脱は読めません）</div><div class="ex-grid">${ex.dropout.map((e) => card(e, "dropout")).join("")}</div>`,
      `<div class="ex-h">低い確率だったのに、重賞掲示板に届いた馬（遅咲き。3 走目で未勝利圏にいた馬は拾えません）</div><div class="ex-grid">${ex.surprise.map((e) => card(e, "surprise")).join("")}</div>`,
    ].join("");
    $("#examples").querySelectorAll(".exc").forEach((el) => { el.onclick = () => openHorse(Number(el.dataset.id)); });
  }

  /* ---------------- 年・基準点 ---------------- */
  function renderTabs() {
    const ix = state.index;
    $("#yearTabs").innerHTML = state.years.map((y) => `<button class="venue ${y === state.year ? "on" : ""}" data-year="${y}">${y} 年デビュー<small>${(ix.by_year[String(y)].n_horses || 0).toLocaleString()}頭</small></button>`).join("");
    $("#yearTabs").querySelectorAll("button").forEach((b) => { b.onclick = () => selectYear(Number(b.dataset.year)); });
    $("#kSel").innerHTML = state.index.ks.map((k) => `<button class="${k === state.k ? "on" : ""}" data-k="${k}">${KNAME[k] || k}</button>`).join("");
    $("#kSel").querySelectorAll("button").forEach((b) => { b.onclick = () => { state.k = Number(b.dataset.k); state.shown = 100; state.open.clear(); renderTabs(); renderMetrics(); renderTable(); }; });
  }
  async function selectYear(y) {
    state.year = y; state.shown = 100; state.open.clear();
    try { history.replaceState(null, "", "#" + y); } catch (e) { /* noop */ }
    renderTabs(); renderMetrics();
    if (!state.cache[y]) {
      $("#tbl").innerHTML = `<tbody><tr><td class="empty">読み込み中…</td></tr></tbody>`;
      try { state.cache[y] = await getJSON(`data/shusse/check_${y}.json`); }
      catch (e) { $("#tbl").innerHTML = `<tbody><tr><td class="empty">この年のデータを読み込めませんでした。</td></tr></tbody>`; return; }
    }
    renderTable();
  }

  /* ---------------- 表 ---------------- */
  function filtered() {
    const d = state.cache[state.year];
    if (!d) return [];
    let rows = d.horses.map((h) => ({ h, r: rowAt(h, state.k) })).filter((x) => x.r);
    if (state.vd !== "all") rows = rows.filter((x) => verdict(x.r).cls === state.vd);
    if (state.q) { const q = nfkc(state.q); rows = rows.filter((x) => nfkc(x.h.name).includes(q) || nfkc(x.h.sire).includes(q) || nfkc(x.h.trainer).includes(q)); }
    const gap = (x) => x.r[COL.ran] === 0 ? x.r[COL.lvl] + 1 : Math.abs(x.r[COL.lvl] - x.r[COL.alvl]);
    const cmp = {
      pg12: (a, b) => b.r[COL.pg12] - a.r[COL.pg12],
      alvl: (a, b) => ((b.r[COL.alvl] ?? -1) - (a.r[COL.alvl] ?? -1)) || (b.r[COL.pg12] - a.r[COL.pg12]),
      gap: (a, b) => (gap(b) - gap(a)) || (b.r[COL.pg12] - a.r[COL.pg12]),
      name: (a, b) => a.h.name.localeCompare(b.h.name, "ja"),
    }[state.sort] || ((a, b) => b.r[COL.pg12] - a.r[COL.pg12]);
    rows.sort(cmp);
    return rows;
  }
  function actualChip(r) {
    if (r[COL.ran] === 0) return `<span class="lv lvD">離脱</span>`;
    return `<span class="lv lv${r[COL.alvl]}">${esc(lvName(r[COL.alvl]))}</span>`;
  }
  function rowHTML(x, i) {
    const h = x.h, r = x.r, v = verdict(r);
    const w = Math.max(2, Math.round(Math.min(1, r[COL.pg12]) * 100));
    return `<tr class="hr${state.open.has(h.id) ? " open" : ""}" data-id="${h.id}">
      <td class="num" style="color:var(--tx3);font-size:.8rem">${i + 1}</td>
      <td class="l"><span class="nm">${esc(h.name)}<span class="sex">${esc(h.sex || "")}</span></span><span class="sub">${esc(h.sire || "父不明")}${h.trainer ? " ・ " + esc(h.trainer) + "厩舎" : ""}</span></td>
      <td class="sum c-sum" title="${esc(h.summary)}">${esc(h.summary || "")}</td>
      <td><span class="pbar${r[COL.pg12] < 0.05 ? " dim" : ""}"><span class="bar"><i style="width:${w}%"></i></span><span class="v">${fmtP(r[COL.pg12])}</span></span></td>
      <td><span class="lv lv${r[COL.lvl]}">${esc(lvName(r[COL.lvl]))}</span></td>
      <td>${actualChip(r)}</td>
      <td><span class="vd ${v.cls}">${v.label}</span></td>
      <td class="num c-pct" style="color:var(--tx2);font-size:.8rem">${r[COL.g12] ? "○" : "–"}</td>
    </tr>`;
  }
  function detailHTML(h) {
    const rows = (h.rows || []).map((r) => {
      const v = verdict(r), cur = r[COL.k] === state.k;
      return `<tr><td class="l${cur ? " cur" : ""}">${KNAME[r[COL.k]] || r[COL.k]}</td><td class="l num">${fmtDate(r[COL.base])}</td><td class="num${cur ? " cur" : ""}">${fmtP(r[COL.pg12])}</td><td class="num">${fmtP(r[COL.pd])}</td><td class="l">${esc(lvName(r[COL.lvl]))}</td><td class="l">${r[COL.ran] === 0 ? "離脱" : esc(lvName(r[COL.alvl]))}</td><td class="l"><span class="vd ${v.cls}">${v.label}</span></td><td>${r[COL.g12] ? "○" : "–"}</td><td>${r[COL.g24] == null ? "–" : (r[COL.g24] ? "○" : "×")}</td></tr>`;
    }).join("");
    return `<tr class="det" data-det="${h.id}"><td colspan="8"><div class="det-in"><h4>基準点ごとの予測と結果（${esc(h.name)}、通算 ${h.n} 走）</h4>
      <table class="mini"><thead><tr><th class="l">基準点</th><th class="l">基準日</th><th>P(重賞)</th><th>P(離脱)</th><th class="l">予測クラス</th><th class="l">実際（12 か月）</th><th class="l">判定</th><th>重賞掲示板 12m</th><th>24m</th></tr></thead><tbody>${rows}</tbody></table>
      <p class="hint" style="margin-top:8px">同じ馬でも基準点ごとに 12 か月の窓が違うので、実際の到達クラスが変わることがあります。<a href="shusse.html#${h.dy}:${h.id}" style="color:var(--gold2)">出世予測ページで見る →</a></p></div></td></tr>`;
  }
  function renderTable() {
    const rows = filtered();
    const d = state.cache[state.year];
    const total = d ? d.horses.filter((h) => rowAt(h, state.k)).length : 0;
    $("#count").textContent = rows.length === total ? `${total.toLocaleString()} 頭` : `${rows.length.toLocaleString()} / ${total.toLocaleString()} 頭`;
    const tbl = $("#tbl");
    if (!rows.length) { tbl.innerHTML = `<tbody><tr><td class="empty">該当する馬がいません。</td></tr></tbody>`; $("#moreBtn").hidden = true; return; }
    const head = `<thead><tr><th>#</th><th class="l">馬名</th><th class="l c-sum">基準点までの走</th><th>重賞掲示板<br><small>AI の確率</small></th><th>予測<br><small>到達クラス</small></th><th>実際<br><small>12 か月</small></th><th>判定</th><th class="c-pct">重賞掲示板<br><small>実際</small></th></tr></thead>`;
    const body = rows.slice(0, state.shown).map((x, i) => rowHTML(x, i) + (state.open.has(x.h.id) ? detailHTML(x.h) : "")).join("");
    tbl.innerHTML = head + `<tbody>${body}</tbody>`;
    $("#moreBtn").hidden = rows.length <= state.shown;
    $("#moreBtn").textContent = `さらに表示（残り ${(rows.length - state.shown).toLocaleString()} 頭）`;
    tbl.querySelectorAll("tr.hr").forEach((tr) => { tr.onclick = () => toggle(Number(tr.dataset.id), tr); });
  }
  function toggle(id, tr) {
    const d = state.cache[state.year];
    const h = d.horses.find((x) => x.id === id);
    if (!h) return;
    if (state.open.has(id)) { state.open.delete(id); tr.classList.remove("open"); const det = tr.parentNode.querySelector(`tr.det[data-det="${id}"]`); if (det) det.remove(); }
    else { state.open.add(id); tr.classList.add("open"); tr.insertAdjacentHTML("afterend", detailHTML(h)); }
  }
  function openHorse(id) {
    const d = state.cache[state.year];
    const h = d && d.horses.find((x) => x.id === id);
    if (!h) return;
    $("#q").value = h.name; state.q = h.name; $("#vdSel").value = "all"; state.vd = "all"; state.shown = 100;
    state.open.clear(); state.open.add(id);
    renderTable();
    const tr = $(`#tbl tr.hr[data-id="${id}"]`);
    if (tr) tr.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  /* ---------------- 起動 ---------------- */
  async function boot() {
    try { state.index = await getJSON("data/shusse/check_index.json"); }
    catch (e) { $("#tbl").innerHTML = `<tbody><tr><td class="empty">答え合わせデータがまだありません。</td></tr></tbody>`; return; }
    state.years = (state.index.years || []).slice().sort((a, b) => b - a);
    $("#asofHint").textContent = `${state.index.asof.replace(/-/g, "/")} 時点で確定した ${state.index.n_horses.toLocaleString()} 頭（予測行 ${state.index.n_rows.toLocaleString()}）`;
    const hy = Number((location.hash || "").replace("#", ""));
    const y0 = state.years.includes(hy) ? hy : state.years[state.years.length - 1];   // 既定は最も古い世代 (全基準点が確定)
    $("#q").oninput = () => { state.q = $("#q").value.trim(); state.shown = 100; renderTable(); };
    $("#vdSel").onchange = () => { state.vd = $("#vdSel").value; state.shown = 100; renderTable(); };
    $("#sortSel").onchange = () => { state.sort = $("#sortSel").value; state.shown = 100; renderTable(); };
    $("#moreBtn").onclick = () => { state.shown += 200; renderTable(); };
    addEventListener("hashchange", () => { const y = Number((location.hash || "").replace("#", "")); if (state.years.includes(y) && y !== state.year) selectYear(y); });
    await selectYear(y0);
  }
  boot();
})();
