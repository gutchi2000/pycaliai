/* 騎手・調教師ページ (jt.html?j=騎手名 / jt.html?t=調教師名) — data/jt/{jockeys,trainers}.json (build_jt_pages.py) を描画。
   run 行は配列 [rid16, name_idx, 印, AI順, p_win, 人気, 着順, klass_idx(, 取消等)]。
   日付は rid[:8]、R 番号は rid[14:16]、競馬場は rid[8:10] を places 表で名前に。
   着順: 1 以上 = 確定 / 0 = 結果はあるが着順なし / null = 結果未着。検索は data/jt/_index.json を初回フォーカス時に読む。 */
(function () {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = (v, d = 1) => (v == null ? "—" : (v * 100).toFixed(d));
  const rate = (a, b, d = 0) => (b ? (a / b * 100).toFixed(d) + "%" : "—");
  const WD = ["日", "月", "火", "水", "木", "金", "土"];
  const KIND_LABEL = { j: "騎手", t: "調教師" };
  const KIND_FILE = { j: "jockeys", t: "trainers" };
  const MARKS = new Set(["◎", "〇", "○", "▲", "△"]);
  const PAGE = 120;
  function fmtDate(d8) {
    const y = +d8.slice(0, 4), m = +d8.slice(4, 6), d = +d8.slice(6, 8);
    return `${m}/${d}(${WD[new Date(y, m - 1, d).getDay()]})`;
  }
  const markCls = (m) => ({ "◎": "m1", "〇": "m2", "○": "m2", "▲": "m3", "△": "m4" }[m] || "");
  const posHtml = (p) => p == null ? '<span class="pos px">—</span>' : `<span class="pos ${p <= 3 ? "p" + p : "px"}">${p}</span>`;

  let built = "0";
  let index = null;
  const books = {};          // kind -> 読み込んだ JSON
  let cur = null;            // { kind, name, person, book }
  let filter = "all";
  let shown = PAGE;
  let chart = null;

  function targetFromUrl() {
    const u = new URL(location.href);
    for (const k of ["j", "t"]) {
      const v = u.searchParams.get(k);
      if (v && v.trim()) return { kind: k, name: v.trim() };
    }
    const h = location.hash ? decodeURIComponent(location.hash.slice(1)) : "";
    const m = /^([jt]):(.+)$/.exec(h);
    if (m) return { kind: m[1], name: m[2].trim() };
    return null;
  }

  async function loadIndex() {
    if (index) return index;
    try { index = await (await fetch(`data/jt/_index.json?v=${encodeURIComponent(built)}`)).json(); }
    catch (e) { index = []; }
    return index;
  }

  async function loadBook(kind) {
    if (books[kind]) return books[kind];
    const resp = await fetch(`data/jt/${KIND_FILE[kind]}.json?v=${encodeURIComponent(built)}`);
    const txt = await resp.text();
    if (!resp.ok || !txt.trim().startsWith("{")) throw new Error("not found");
    books[kind] = JSON.parse(txt);
    return books[kind];
  }

  function kpi(l, v, s) { return `<div class="kpi"><div class="l">${l}</div><div class="v">${v}</div><div class="s">${s || ""}</div></div>`; }

  // run 行アクセサ
  const R = {
    rid: (r) => r[0], date: (r) => r[0].slice(0, 8), rno: (r) => +r[0].slice(14, 16),
    place: (r, book) => (book.places || {})[r[0].slice(8, 10)] || "",
    horse: (r, book) => (book.names || [])[r[1]] ?? "",
    mark: (r) => r[2] || "", ai: (r) => r[3], pwin: (r) => r[4], ninki: (r) => r[5],
    pos: (r) => r[6], klass: (r, book) => (book.klass || [])[r[7]] ?? "", scr: (r) => r[8] || "",
  };

  function filteredRuns() {
    const runs = cur.person.runs || [];
    if (filter === "hon") return runs.filter((r) => R.mark(r) === "◎");
    if (filter === "marked") return runs.filter((r) => MARKS.has(R.mark(r)));
    if (filter === "ai1") return runs.filter((r) => R.ai(r) === 1);
    if (filter === "win") return runs.filter((r) => R.pos(r) === 1);
    return runs;
  }

  function renderRows() {
    const book = cur.book;
    const runs = filteredRuns();
    const rows = runs.slice(0, shown).map((r) => {
      const pos = R.pos(r), scr = R.scr(r);
      const posCell = scr ? `<span class="pos px">${esc(scr)}</span>`
        : pos == null ? '<span class="pos px">未</span>'
        : pos === 0 ? '<span class="pos px">—</span>' : posHtml(pos);
      const kaiko = pos != null ? `<a href="kaiko.html#${R.date(r)}" title="回顧を読む">回顧</a>` : "";
      const horse = R.horse(r, book);
      return `<tr>
        <td class="l num"><a href="r/${esc(R.rid(r))}">${fmtDate(R.date(r))}</a></td>
        <td class="l">${esc(R.place(r, book))}${R.rno(r)}R</td>
        <td class="l">${esc(R.klass(r, book))}</td>
        <td class="l"><a href="horse.html?name=${encodeURIComponent(horse)}">${esc(horse)}</a></td>
        <td><span class="mk ${markCls(R.mark(r))}">${esc(R.mark(r) || "・")}</span></td>
        <td class="num">${R.ai(r) ?? "—"}</td>
        <td class="num">${pct(R.pwin(r))}%</td>
        <td class="num">${R.ninki(r) ?? "—"}</td>
        <td>${posCell}</td>
        <td class="l">${kaiko}</td>
      </tr>`;
    }).join("");
    $("#runs-body").innerHTML = rows || `<tr><td colspan="10" class="empty">該当する出走なし</td></tr>`;
    $("#runs-count").textContent = `${runs.length} 走${runs.length > shown ? `（${shown} 走を表示）` : ""}`;
    const more = $("#more");
    if (more) more.hidden = runs.length <= shown;
  }

  function renderChart(st) {
    const el = $("#chart");
    if (!el) return;
    const months = Object.entries(st.months || {});
    if (!months.length) { el.classList.add("empty"); el.textContent = "データなし"; return; }
    if (typeof echarts === "undefined") { el.classList.add("empty"); el.textContent = "グラフを読み込めませんでした"; return; }
    const labels = months.map(([m]) => `${+m.slice(4, 6)}月`);
    const n = months.map(([, a]) => a[0]);
    const top3 = months.map(([, a]) => a[3]);
    const wins = months.map(([, a]) => a[2]);
    if (chart) { chart.dispose(); chart = null; }
    chart = echarts.init(el, null, { renderer: "canvas" });
    chart.setOption({
      backgroundColor: "transparent",
      animationDuration: 300,
      grid: { left: 36, right: 10, top: 30, bottom: 26 },
      legend: { top: 0, right: 0, textStyle: { color: "#a9b6d3", fontSize: 11 }, itemWidth: 10, itemHeight: 10, icon: "roundRect" },
      tooltip: {
        trigger: "axis", axisPointer: { type: "shadow" },
        backgroundColor: "rgba(13,20,36,.95)", borderColor: "#32456e", textStyle: { color: "#edf1fb", fontSize: 12 },
        formatter: (ps) => {
          const i = ps[0].dataIndex;
          const [m, a] = months[i];
          return `<b>${+m.slice(0, 4)}年${+m.slice(4, 6)}月</b><br>出走 ${a[0]}（結果あり ${a[1]}）<br>勝ち ${a[2]}（${rate(a[2], a[1], 1)}）<br>3着内 ${a[3]}（${rate(a[3], a[1], 1)}）`;
        },
      },
      xAxis: { type: "category", data: labels, axisLine: { lineStyle: { color: "#32456e" } }, axisTick: { show: false }, axisLabel: { color: "#7385a8", fontSize: 11 } },
      yAxis: { type: "value", minInterval: 1, splitLine: { lineStyle: { color: "rgba(50,69,110,.35)" } }, axisLabel: { color: "#7385a8", fontSize: 11 } },
      series: [
        { name: "出走", type: "bar", data: n, barGap: "10%", barCategoryGap: "35%", itemStyle: { color: "#3b4d78", borderRadius: [3, 3, 0, 0] } },
        { name: "3着内", type: "bar", data: top3, itemStyle: { color: "#2dd4a8", borderRadius: [3, 3, 0, 0] } },
        { name: "勝ち", type: "bar", data: wins, itemStyle: { color: "#f5b942", borderRadius: [3, 3, 0, 0] } },
      ],
    });
  }

  function renderPerson() {
    const { kind, name, person: p } = cur;
    const st = p.stats || {};
    document.title = `${name} — PyCaLiAI ${KIND_LABEL[kind]}ページ`;
    const attrs = [
      `<span><b>${KIND_LABEL[kind]}</b></span>`,
      `<span>2026 出走 <b>${st.n ?? 0}</b></span>`,
      st.mean_ai_rank != null ? `<span>平均 AI 順位 <b>${st.mean_ai_rank}</b> 位</span>` : "",
      st.mean_ninki != null ? `<span>平均人気 <b>${st.mean_ninki}</b> 番</span>` : "",
      st.mean_pwin != null ? `<span>平均 AI 勝率 <b>${pct(st.mean_pwin)}%</b></span>` : "",
      st.ai1_share != null ? `<span title="結果ありの出走のうち AI が 1 位評価した割合">AI 1 位の割合 <b>${pct(st.ai1_share, 0)}%</b></span>` : "",
    ].join("");
    const other = p.updated ? `<span class="lv">最終${kind === "j" ? "騎乗" : "出走"} <b>${fmtDate(p.updated)}</b></span>` : "";
    const kpis = [
      kpi("勝率", rate(st.wins, st.n_settled, 1), `${st.wins ?? 0} 勝 / ${st.n_settled ?? 0} 走（結果あり）`),
      kpi("3着内率", rate(st.top3, st.n_settled, 1), `3着内 ${st.top3 ?? 0} 回`),
      kpi("AI ◎ のとき", st.hon_n ? `${rate(st.hon_win, st.hon_n, 0)} / ${rate(st.hon_top3, st.hon_n, 0)}` : "—", st.hon_n ? `勝率 / 3着内率（◎ ${st.hon_n} 回）` : "◎ なし"),
      kpi("AI 1 位のとき", st.ai1_n ? `${rate(st.ai1_win, st.ai1_n, 0)}` : "—", st.ai1_n ? `勝率（AI 1 位 ${st.ai1_n} 回・3着内 ${rate(st.ai1_top3, st.ai1_n, 0)}）` : "AI 1 位なし"),
      kpi("印つきのとき", st.marked_n ? rate(st.marked_top3, st.marked_n, 0) : "—", st.marked_n ? `3着内率（◎〇▲△ ${st.marked_n} 回・勝ち ${st.marked_win}）` : "印なし"),
    ].join("");
    const bk = Object.entries(st.by_klass || {});
    const klassRows = bk.map(([k, a]) => `<tr><td class="l">${esc(k || "—")}</td><td class="num">${a[0]}</td><td class="num">${a[1]}</td><td class="num">${a[2]}</td><td class="num">${rate(a[1], a[0], 0)}</td><td class="num">${rate(a[2], a[0], 0)}</td></tr>`).join("");
    $("#jt").innerHTML = `
      <section class="panel hhead">
        <div>
          <div class="nm">${esc(name)}</div>
          <div class="attrs">${attrs}</div>
        </div>
        <div class="right">${other}</div>
      </section>
      <div class="kpis">${kpis}</div>
      <div class="two">
        <section class="panel dim">
          <h2>月別 <small>出走・勝ち・3着内の件数</small></h2>
          <div id="chart" class="chart" role="img" aria-label="月別の出走数・勝ち数・3着内数"></div>
        </section>
        <section class="panel dim">
          <h2>クラス別 <small>結果ありの出走</small></h2>
          <div class="tw"><table class="rc">
            <thead><tr><th class="l">クラス</th><th>走</th><th>勝</th><th>3着内</th><th>勝率</th><th>3着内率</th></tr></thead>
            <tbody>${klassRows || '<tr><td colspan="6" class="empty">結果なし</td></tr>'}</tbody>
          </table></div>
        </section>
      </div>
      <section class="panel dim">
        <h2>2026 年の出走と AI 評価 <small>レース前の印・AI 順位・勝率 と 実際の着順 ・ <span id="runs-count"></span></small></h2>
        <div class="filt" id="filt">
          <button type="button" data-f="all">すべて</button>
          <button type="button" data-f="hon">AI ◎</button>
          <button type="button" data-f="marked">印あり</button>
          <button type="button" data-f="ai1">AI 1 位</button>
          <button type="button" data-f="win">勝ち</button>
        </div>
        <div class="tw"><table class="rc">
          <thead><tr><th class="l">日付</th><th class="l">レース</th><th class="l">クラス</th><th class="l">馬</th><th>印</th><th>AI順</th><th>勝率</th><th>人気</th><th>着順</th><th class="l"></th></tr></thead>
          <tbody id="runs-body"></tbody>
        </table></div>
        <button type="button" class="more" id="more" hidden>さらに ${PAGE} 走を表示</button>
      </section>`;
    for (const b of document.querySelectorAll("#filt button")) {
      b.classList.toggle("on", b.dataset.f === filter);
      b.addEventListener("click", () => {
        filter = b.dataset.f; shown = PAGE;
        for (const x of document.querySelectorAll("#filt button")) x.classList.toggle("on", x === b);
        renderRows();
      });
    }
    $("#more").addEventListener("click", () => { shown += PAGE; renderRows(); });
    renderRows();
    renderChart(st);
  }

  async function showPerson(t) {
    if (!t) return;
    $("#q").value = t.name;
    $("#jt").innerHTML = '<section class="panel"><p>読み込み中…</p></section>';
    try {
      const book = await loadBook(t.kind);
      const person = (book.people || {})[t.name];
      if (!person) throw new Error("not found");
      cur = { kind: t.kind, name: t.name, person, book };
      filter = "all"; shown = PAGE;
      renderPerson();
    } catch (e) {
      $("#jt").innerHTML = `<section class="panel"><p class="err">${KIND_LABEL[t.kind]}「${esc(t.name)}」のページがありません。2026 年に JRA で出走のある騎手・調教師だけ検索できます（表記は出走表と同じ。騎手は姓名／外国人は片仮名、調教師は姓名）。</p></section>`;
    }
  }

  function goto(kind, name) {
    const u = new URL(location.href);
    u.searchParams.delete("j"); u.searchParams.delete("t");
    u.searchParams.set(kind, name); u.hash = "";
    history.pushState(null, "", u.toString());
    showPerson({ kind, name });
    $("#sug").hidden = true;
  }

  function sugRender(q) {
    const box = $("#sug");
    if (!q || !index) { box.hidden = true; return; }
    const nq = q.normalize("NFKC").replace(/\s+/g, "");
    const pre = [], inc = [];
    for (const row of index) {
      const i = row[0].indexOf(nq);
      if (i === 0) pre.push(row); else if (i > 0) inc.push(row);
      if (pre.length >= 20) break;
    }
    const hits = pre.concat(inc).slice(0, 20);
    if (!hits.length) { box.innerHTML = '<a><small>該当なし</small></a>'; box.hidden = false; return; }
    box.innerHTML = hits.map(([nm, n, last, kind]) =>
      `<a href="jt.html?${kind}=${encodeURIComponent(nm)}" data-nm="${esc(nm)}" data-kind="${kind}"><span><span class="kd">${KIND_LABEL[kind]}</span>${esc(nm)}</span><small>${n}走 ・ 最終 ${fmtDate(last)}</small></a>`).join("");
    box.hidden = false;
  }

  async function boot() {
    try { built = (await (await fetch("data/manifest.json?t=" + Date.now())).json()).built_at || "0"; } catch (e) { /* noop */ }
    const q = $("#q");
    q.addEventListener("focus", loadIndex);
    q.addEventListener("input", async () => { await loadIndex(); sugRender(q.value.trim()); });
    q.addEventListener("keydown", async (e) => {
      if (e.key === "Enter") {
        await loadIndex();
        const first = $("#sug a[data-nm]");
        const v = q.value.trim();
        if (first && first.dataset.nm) goto(first.dataset.kind, first.dataset.nm);
        else if (v) {
          // 候補が未表示でも、完全一致があればその人へ
          const hit = (index || []).find((r) => r[0] === v.normalize("NFKC"));
          if (hit) goto(hit[3], hit[0]); else goto(cur ? cur.kind : "j", v);
        }
      } else if (e.key === "Escape") { $("#sug").hidden = true; }
    });
    $("#sug").addEventListener("click", (e) => {
      const a = e.target.closest("a[data-nm]");
      if (!a) return;
      e.preventDefault(); goto(a.dataset.kind, a.dataset.nm);
    });
    document.addEventListener("click", (e) => { if (!e.target.closest(".search")) $("#sug").hidden = true; });
    addEventListener("popstate", () => showPerson(targetFromUrl()));
    addEventListener("resize", () => { if (chart) chart.resize(); });
    const t = targetFromUrl();
    if (t) showPerson(t);
  }
  boot();
})();
