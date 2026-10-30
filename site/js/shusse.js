/* PyCaLiAI — 出世予測 AI ページ (site/shusse.html)
   データ: data/shusse/index.json (メタ・件数・検証) + data/shusse/{debut_year}.json (馬ごとの予測行 + 上位の文章カード)
   生成元: analysis/shusse/site_export.py → build_site.py が site/data/shusse/ へコピー。
   表示専用。市場オッズ・買い目には一切触れない。 */
(function () {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const KNAME = { 1: "1 走目", 2: "2 走目", 3: "3 走目", 5: "5 走目" };
  const nfkc = (s) => String(s || "").normalize("NFKC").toLowerCase();

  const state = {
    index: null, years: [], year: null, cache: {}, q: "", k: "all", sort: "pg12", shown: 100, open: new Set(),
  };

  async function getJSON(u) {
    const r = await fetch(u + "?t=" + Date.now());
    if (!r.ok) throw new Error(u);
    return r.json();
  }

  /* 確率の表示: 10% 未満は小数 1 桁、それ以上は整数 */
  function fmtP(p) {
    if (p == null || isNaN(p)) return "–";
    const v = p * 100;
    if (v > 0 && v < 0.05) return "<0.1%";
    return (v < 10 ? v.toFixed(1) : Math.round(v)) + "%";
  }
  function f3(x, nd) { return x == null ? "–" : Number(x).toFixed(nd == null ? 2 : nd); }
  function topPct(pct) {
    if (pct == null) return "–";
    const t = 100 - pct;
    if (t < 0.1) return "上位 0.1%";
    return "上位 " + (t < 10 ? t.toFixed(1) : Math.round(t)) + "%";
  }
  function lvName(i) {
    const lv = state.index && state.index.levels;
    return lv && lv[String(i)] ? lv[String(i)] : ["未勝利圏", "1勝クラス", "2勝クラス", "3勝/OP", "重賞掲示板"][i];
  }
  function fmtDate(s) { return s ? String(s).replace(/-/g, "/").slice(2) : "–"; }

  /* ---------------- 概要・検証 ---------------- */
  function renderIntro() {
    const ix = state.index;
    const n = ix.n_horses || 0;
    $("#asofHint").textContent = `${ix.asof.replace(/-/g, "/")} 時点の出走までを反映 ／ 判定待ち ${n.toLocaleString()} 頭（${ix.years.map((y) => `${y} 年デビュー ${(ix.n_by_year[y] || 0).toLocaleString()}`).join("・")}）`;
    const kv = [];
    kv.push(`<span>対象 <b>${n.toLocaleString()}</b> 頭</span>`);
    kv.push(`<span>基準点 <b>1・2・3・5 走目</b></span>`);
    kv.push(`<span>更新 <b>毎開催後</b>（最新 ${esc(ix.generated_at || ix.asof)}）</span>`);
    kv.push(`<span>学習データ <b>2013〜2019 年デビュー</b>・検証 2020〜2023</span>`);
    $("#introKv").innerHTML = kv.join("");

    const v = ix.validation || {};
    const ks = ["k1", "k2", "k3", "k5"];
    const rows = ks.filter((k) => v[k]).map((k) => {
      const t = v[k].test2022 || {}, f = v[k].forward2023 || {};
      const fq = f.qwk != null ? `<span class="ai">${f3(f.qwk)}</span>` : "–";
      const ci = f.qwk_ci ? `<small style="color:var(--tx3)"> [${f3(f.qwk_ci[0])}–${f3(f.qwk_ci[1])}]</small>` : "";
      return `<tr><td class="l">${KNAME[k.slice(1)] || k}</td>
        <td class="num">${f.n != null ? f.n.toLocaleString() : "–"}</td>
        <td>${f3(f.rule_qwk)}<span class="arrow">→</span>${fq}${ci}</td>
        <td class="ai">${f3(f.g12_auc)}</td>
        <td>${f3(f.d_auc)}</td>
        <td style="color:var(--tx3)">${f3(t.qwk)} / ${f3(t.g12_auc)}</td></tr>`;
    });
    $("#valTable").innerHTML = `<table class="vtab"><thead><tr><th class="l">基準点</th><th>頭数</th><th>クラス一致度 QWK<br><small>規則 → AI</small></th><th>重賞掲示板<br>AUC</th><th>離脱<br>AUC</th><th>参考: 2022 世代<br><small>QWK / AUC</small></th></tr></thead><tbody>${rows.join("")}</tbody></table>`;
  }

  /* ---------------- 年タブ ---------------- */
  function renderTabs() {
    const ix = state.index;
    $("#yearTabs").innerHTML = state.years.map((y) =>
      `<button class="venue ${y === state.year ? "on" : ""}" data-year="${y}">${y} 年デビュー<small>${(ix.n_by_year[y] || 0).toLocaleString()}頭</small></button>`).join("");
    $("#yearTabs").querySelectorAll("button").forEach((b) => {
      b.onclick = () => selectYear(Number(b.dataset.year));
    });
  }

  async function selectYear(y) {
    state.year = y;
    state.shown = 100;
    state.open.clear();
    try { history.replaceState(null, "", "#" + y); } catch (e) { /* noop */ }
    renderTabs();
    if (!state.cache[y]) {
      $("#tbl").innerHTML = `<tbody><tr><td class="empty">読み込み中…</td></tr></tbody>`;
      try {
        state.cache[y] = await getJSON(`data/shusse/${y}.json`);
      } catch (e) {
        $("#tbl").innerHTML = `<tbody><tr><td class="empty">この年のデータを読み込めませんでした。</td></tr></tbody>`;
        return;
      }
    }
    renderTable();
  }

  /* ---------------- 絞り込み・並び替え ---------------- */
  function filtered() {
    const d = state.cache[state.year];
    if (!d) return [];
    let rows = d.horses.slice();
    if (state.k !== "all") rows = rows.filter((h) => String(h.k) === state.k);
    if (state.q) {
      const q = nfkc(state.q);
      rows = rows.filter((h) => nfkc(h.name).includes(q) || nfkc(h.sire).includes(q) || nfkc(h.trainer).includes(q));
    }
    const s = state.sort;
    const cmp = {
      pg12: (a, b) => (b.pg12 - a.pg12) || (a.pd - b.pd),
      lvl: (a, b) => (b.lvl - a.lvl) || (expLv(b) - expLv(a)) || (b.pg12 - a.pg12),
      pd: (a, b) => (a.pd - b.pd) || (b.pg12 - a.pg12),
      base: (a, b) => (a.base < b.base ? 1 : a.base > b.base ? -1 : 0) || (b.pg12 - a.pg12),
      n: (a, b) => (b.n - a.n) || (b.pg12 - a.pg12),
      name: (a, b) => a.name.localeCompare(b.name, "ja"),
    }[s] || ((a, b) => b.pg12 - a.pg12);
    rows.sort(cmp);
    return rows;
  }
  function expLv(h) { return h.pl ? h.pl.reduce((acc, p, i) => acc + p * i, 0) : 0; }

  /* ---------------- 表 ---------------- */
  function rowHTML(h) {
    const dim = h.pg12 < 0.05 ? " dim" : "";
    const w = Math.max(2, Math.round(Math.min(1, h.pg12) * 100));
    return `<tr class="hr${state.open.has(h.id) ? " open" : ""}" data-id="${h.id}">
      <td class="rank">${h.rank}</td>
      <td class="l"><span class="nm">${esc(h.name)}<span class="sex">${esc(h.sex || "")}</span></span><span class="sub">${esc(h.sire || "父不明")}${h.trainer ? " ・ " + esc(h.trainer) + "厩舎" : ""}</span></td>
      <td class="num c-n">${h.n}<small style="color:var(--tx3)"> 走</small></td>
      <td class="sum c-sum" title="${esc(h.summary)}">${esc(h.summary || "")}</td>
      <td><span class="pbar${dim}"><span class="bar"><i style="width:${w}%"></i></span><span class="v">${fmtP(h.pg12)}</span></span></td>
      <td><span class="lv lv${h.lvl}">${esc(lvName(h.lvl))}</span></td>
      <td class="num">${fmtP(h.pd)}</td>
      <td class="pct c-pct">${topPct(h.pct)}</td>
    </tr>`;
  }

  function detailHTML(h) {
    const d = state.cache[state.year];
    const card = d.cards && d.cards[String(h.id)];
    const pl = h.pl || [0, 0, 0, 0, 0];
    const stack = `<div class="stack">${pl.map((p, i) => `<i class="s${i}" style="width:${(p * 100).toFixed(1)}%" title="${esc(lvName(i))} ${fmtP(p)}"></i>`).join("")}</div>
      <div class="legend">${pl.map((p, i) => `<span><i class="s${i}" style="background:${["var(--line2)", "#4a5c86", "var(--blue)", "var(--teal)", "var(--gold)"][i]}"></i>${esc(lvName(i))}<span class="num">${fmtP(p)}</span></span>`).join("")}</div>`;
    const why = (h.why && h.why.length) ? `<ul class="why">${h.why.map((w) => `<li>${esc(w)}</li>`).join("")}</ul>` : `<p class="hint">根拠の内訳はありません。</p>`;
    const hist = (h.hist && h.hist.length) ? `<table class="mini"><thead><tr><th class="l">基準点</th><th class="l">基準日</th><th>P(重賞)</th><th>P(離脱)</th><th class="l">予測クラス</th></tr></thead><tbody>${h.hist.map((r) => {
      const cur = r[0] === h.k;
      return `<tr><td class="l${cur ? " cur" : ""}">${KNAME[r[0]] || r[0] + " 走目"}</td><td class="l num">${fmtDate(r[1])}</td><td class="num${cur ? " cur" : ""}">${fmtP(r[2])}</td><td class="num">${fmtP(r[3])}</td><td class="l">${esc(lvName(r[4]))}</td></tr>`;
    }).join("")}</tbody></table>` : "";
    let narr = "";
    if (card && card.text) {
      const t = esc(card.text).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/\n/g, "<br>");
      narr = `<div class="det" style="grid-column:1/-1"><h4>AI の見解</h4><p class="narr">${t}</p></div>`;
    }
    let runs = "";
    if (card && card.runs && card.runs.length) {
      runs = `<div class="det"><h4>基準日までの走</h4><table class="mini"><thead><tr><th class="l">日付</th><th class="l">レース</th><th class="l">場所</th><th>着順</th><th>着差</th></tr></thead><tbody>${card.runs.map((r) =>
        `<tr><td class="l num">${fmtDate(r.date)}</td><td class="l">${esc(r.label)}</td><td class="l">${esc(r.place || "")}${r.surface ? " " + esc(r.surface) : ""}${r.dist ? r.dist + "m" : ""}</td><td class="num${r.pos === 1 ? " win" : ""}">${r.pos != null ? r.pos + " 着" : "–"}${r.field ? `<small style="color:var(--tx3)">/${r.field}</small>` : ""}</td><td class="num">${r.margin == null ? "–" : (r.pos === 1 ? "—" : r.margin.toFixed(1) + " 秒")}</td></tr>`).join("")}</tbody></table></div>`;
    }
    return `<tr class="det" data-det="${h.id}"><td colspan="8"><div class="det-in">
      ${narr}
      <div class="det"><h4>走った場合の 1 年後の到達クラス（確率）</h4>${stack}
        <p class="hint" style="margin-top:8px">離脱（1 年間出走なし）の確率 <b style="color:var(--tx)">${fmtP(h.pd)}</b>。重賞掲示板まで ${fmtP(h.pg12)} ＝ (1 − ${fmtP(h.pd)}) × ${fmtP(pl[4])}。2 年以内なら ${fmtP(h.pg24)}。</p></div>
      <div class="det"><h4>根拠（重賞掲示板への寄与が大きい順）</h4>${why}
        <p class="hint" style="margin-top:6px">基準日 ${esc(h.base)}（${KNAME[h.k] || h.k + " 走目"}時点）。同じデビュー年の中で ${esc(topPct(h.pct))}。</p></div>
      ${hist ? `<div class="det"><h4>基準点ごとの推移</h4>${hist}</div>` : ""}
      ${runs}
    </div></td></tr>`;
  }

  function renderTable() {
    const rows = filtered();
    const total = (state.cache[state.year] || { horses: [] }).horses.length;
    $("#count").textContent = rows.length === total ? `${total.toLocaleString()} 頭` : `${rows.length.toLocaleString()} / ${total.toLocaleString()} 頭`;
    const tbl = $("#tbl");
    if (!rows.length) {
      tbl.innerHTML = `<tbody><tr><td class="empty">該当する馬がいません。</td></tr></tbody>`;
      $("#moreBtn").hidden = true;
      return;
    }
    const head = `<thead><tr><th>#</th><th class="l">馬名</th><th class="c-n">出走</th><th class="l c-sum">これまでの走</th><th>重賞掲示板<br><small>1 年以内</small></th><th>到達クラス<br><small>予測</small></th><th>離脱</th><th class="c-pct">同世代</th></tr></thead>`;
    const body = rows.slice(0, state.shown).map((h) => rowHTML(h) + (state.open.has(h.id) ? detailHTML(h) : "")).join("");
    tbl.innerHTML = head + `<tbody>${body}</tbody>`;
    $("#moreBtn").hidden = rows.length <= state.shown;
    $("#moreBtn").textContent = `さらに表示（残り ${(rows.length - state.shown).toLocaleString()} 頭）`;
    tbl.querySelectorAll("tr.hr").forEach((tr) => {
      tr.onclick = () => toggle(Number(tr.dataset.id), tr);
    });
  }

  function toggle(id, tr) {
    const d = state.cache[state.year];
    const h = d.horses.find((x) => x.id === id);
    if (!h) return;
    if (state.open.has(id)) {
      state.open.delete(id);
      tr.classList.remove("open");
      const det = tr.parentNode.querySelector(`tr.det[data-det="${id}"]`);
      if (det) det.remove();
    } else {
      state.open.add(id);
      tr.classList.add("open");
      tr.insertAdjacentHTML("afterend", detailHTML(h));
    }
  }

  /* ---------------- 起動 ---------------- */
  async function boot() {
    try {
      state.index = await getJSON("data/shusse/index.json");
    } catch (e) {
      $("#tbl").innerHTML = `<tbody><tr><td class="empty">出世予測データがまだありません。</td></tr></tbody>`;
      $("#asofHint").textContent = "";
      return;
    }
    state.years = (state.index.years || Object.keys(state.index.n_by_year || {}).map(Number)).slice().sort((a, b) => b - a);
    renderIntro();
    const hy = Number((location.hash || "").replace("#", ""));
    const y0 = state.years.includes(hy) ? hy : state.years[0];
    $("#q").oninput = () => { state.q = $("#q").value.trim(); state.shown = 100; renderTable(); };
    $("#kSel").onchange = () => { state.k = $("#kSel").value; state.shown = 100; renderTable(); };
    $("#sortSel").onchange = () => { state.sort = $("#sortSel").value; state.shown = 100; renderTable(); };
    $("#moreBtn").onclick = () => { state.shown += 200; renderTable(); };
    addEventListener("hashchange", () => {
      const y = Number((location.hash || "").replace("#", ""));
      if (state.years.includes(y) && y !== state.year) selectYear(y);
    });
    await selectYear(y0);
  }
  boot();
})();
