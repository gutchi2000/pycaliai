/* 馬ページ (horse.html?name=馬名) — data/h/{id[:2]}.json (build_horse_pages.py, 256 シャード) を描画。
   id は馬名の FNV-1a 32bit ×2 (Python 側と同一定義)。検索は data/h/_index.json を初回フォーカス時に読む。 */
(function () {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = (v, d = 1) => (v == null ? "—" : (v * 100).toFixed(d));
  const WD = ["日", "月", "火", "水", "木", "金", "土"];
  function fmtDate(d8) {
    const y = +d8.slice(0, 4), m = +d8.slice(4, 6), d = +d8.slice(6, 8);
    return `${m}/${d}(${WD[new Date(y, m - 1, d).getDay()]})`;
  }
  function fnv(s, seed) {
    let h = seed >>> 0;
    for (const b of new TextEncoder().encode(s)) { h = Math.imul(h ^ b, 16777619) >>> 0; }
    return h;
  }
  const horseId = (name) => fnv(name, 0x811c9dc5).toString(16).padStart(8, "0") + fnv(name, 0x01000193).toString(16).padStart(8, "0");
  const markCls = (m) => ({ "◎": "m1", "〇": "m2", "○": "m2", "▲": "m3", "△": "m4" }[m] || "");
  const posHtml = (p) => p == null ? '<span class="pos px">—</span>' : `<span class="pos ${p <= 3 ? "p" + p : "px"}">${p}</span>`;

  let built = "0";
  let index = null;

  function nameFromUrl() {
    const u = new URL(location.href);
    const n = u.searchParams.get("name") || (location.hash ? decodeURIComponent(location.hash.slice(1)) : "");
    return (n || "").trim();
  }

  async function loadIndex() {
    if (index) return index;
    try { index = await (await fetch(`data/h/_index.json?v=${encodeURIComponent(built)}`)).json(); }
    catch (e) { index = []; }
    return index;
  }

  function kpi(l, v, s) { return `<div class="kpi"><div class="l">${l}</div><div class="v">${v}</div><div class="s">${s || ""}</div></div>`; }

  function renderHorse(h) {
    document.title = `${h.name} — PyCaLiAI 馬ページ`;
    const st = h.stats || {};
    const ped = h.pedigree || {};
    const lv = h.level || {};
    const sh = h.shusse;
    const attrs = [
      h.sex ? `<span><b>${esc(h.sex)}${h.age ?? ""}</b></span>` : "",
      h.trainer ? `<span>調教師 <b>${esc(h.trainer)}</b>${h.shozoku ? esc(h.shozoku) : ""}</span>` : "",
      ped.sire ? `<span>父 <b>${esc(ped.sire)}</b></span>` : "",
      ped.broodmare_sire ? `<span>母父 <b>${esc(ped.broodmare_sire)}</b></span>` : "",
      h.style ? `<span>脚質 <b>${esc(h.style)}</b></span>` : "",
    ].join("");
    const right = [
      lv.score != null ? `<span class="lv">馬レベル <b>${esc(lv.score)}</b>${lv.tier ? ` ${esc(lv.tier)}` : ""}</span>` : "",
      sh && sh.id != null ? `<a class="shlink" href="shusse.html#${esc(sh.dy)}:${esc(sh.id)}" title="${esc(sh.summary || "")}">出世予測 ${sh.pct != null ? `同世代上位 ${Math.max(0.1, 100 - sh.pct).toFixed(1)}%` : ""} ↗</a>` : "",
    ].join("");
    const hitRate = st.marked_n ? `${(st.marked_top3 / st.marked_n * 100).toFixed(0)}%` : "—";
    const kpis = [
      kpi("2026 出走", `${st.n ?? 0}`, `結果あり ${st.n_settled ?? 0}`),
      kpi("勝利 / 3着内", `${st.wins ?? 0} / ${st.top3 ?? 0}`, st.n_settled ? `3着内率 ${(st.top3 / st.n_settled * 100).toFixed(0)}%` : ""),
      kpi("AI ◎", `${st.hon_n ?? 0}回`, st.hon_n ? `勝ち ${st.hon_win} ・ 3着内 ${st.hon_top3}` : "◎なし"),
      kpi("印つき時の3着内", hitRate, st.marked_n ? `${st.marked_top3}/${st.marked_n}` : "印なし"),
      kpi("平均 AI 勝率", st.avg_pwin != null ? `${pct(st.avg_pwin)}%` : "—", st.best_ai_rank ? `最高 AI ランク #${st.best_ai_rank}` : ""),
    ].join("");
    const rows = (h.runs || []).map((r) => {
      const hasRes = r.has_result;
      const kaiko = hasRes ? `<a href="kaiko.html#${r.date}" title="回顧を読む">回顧</a>` : "";
      return `<tr>
        <td class="l num"><a href="r/${esc(r.rid)}">${fmtDate(r.date)}</a></td>
        <td class="l">${esc(r.place)}${r.rno}R <small style="color:var(--tx3)">${esc(r.race_name || r.klass)}</small></td>
        <td class="l">${esc(r.course)}</td>
        <td class="l">${esc(r.klass)}</td>
        <td><span class="mk ${markCls(r.mark)}">${esc(r.mark || "・")}</span></td>
        <td class="num">${r.ai_rank ?? "—"}</td>
        <td class="num">${pct(r.p_win)}%</td>
        <td class="num">${pct(r.p_sho, 0)}%</td>
        <td class="num">${r.ninki ?? "—"}</td>
        <td>${r.scratched ? `<span class="pos px">${esc(r.scratched)}</span>` : (hasRes ? posHtml(r.pos) : '<span class="pos px">未</span>')}</td>
        <td class="l">${esc(r.jockey || "")}</td>
        <td class="l">${kaiko}</td>
      </tr>`;
    }).join("");
    const past = (h.past || []).map((p) => `<tr>
        <td class="num">${p.n_ago}走前</td>
        <td class="l">${esc(p.place || "")} ${esc(p.td || "")}${p.dist ?? ""}</td>
        <td class="l">${esc(p.race_name || "")}</td>
        <td class="l">${esc(p.track || "")}</td>
        <td class="num">${p.ninki ?? "—"}</td>
        <td>${posHtml(p.pos)}</td>
        <td class="l">${esc(p.style || "")}</td>
        <td class="num">${p.interval_weeks != null ? p.interval_weeks + "週" : "—"}</td>
      </tr>`).join("");
    $("#horse").innerHTML = `
      <section class="panel hhead">
        <div>
          <div class="nm">${esc(h.name)}</div>
          <div class="attrs">${attrs}</div>
        </div>
        <div class="right">${right}</div>
      </section>
      <div class="kpis">${kpis}</div>
      <section class="panel dim">
        <h2>2026 年の出走と AI 評価 <small>レース前の印・勝率・3着内率 と 実際の着順</small></h2>
        <div class="tw"><table class="rc">
          <thead><tr><th class="l">日付</th><th class="l">レース</th><th class="l">コース</th><th class="l">クラス</th><th>印</th><th>AI順</th><th>勝率</th><th>3着内</th><th>人気</th><th>着順</th><th class="l">騎手</th><th class="l"></th></tr></thead>
          <tbody>${rows || '<tr><td colspan="12" class="empty">出走なし</td></tr>'}</tbody>
        </table></div>
      </section>
      ${past ? `<section class="panel dim">
        <h2>近走 <small>最新出走時点の過去 5 走（出走表の履歴）</small></h2>
        <div class="tw"><table class="rc">
          <thead><tr><th></th><th class="l">場・コース</th><th class="l">レース</th><th class="l">馬場</th><th>人気</th><th>着順</th><th class="l">脚質</th><th>間隔</th></tr></thead>
          <tbody>${past}</tbody>
        </table></div>
      </section>` : ""}
      ${sh && sh.summary ? `<section class="panel dim"><h2>出世予測 AI <small>${esc(sh.dy)} 世代・基準日 ${esc(sh.base || "")}</small></h2>
        <p>${esc(sh.summary)}　<a class="shlink" href="shusse.html#${esc(sh.dy)}:${esc(sh.id)}">出世予測を見る ↗</a></p></section>` : ""}`;
  }

  async function showHorse(name) {
    if (!name) return;
    $("#q").value = name;
    $("#horse").innerHTML = '<section class="panel"><p>読み込み中…</p></section>';
    const id = horseId(name);
    try {
      // 256 シャード (id 先頭 2 桁) の中からこの馬を取り出す
      const resp = await fetch(`data/h/${id.slice(0, 2)}.json?v=${encodeURIComponent(built)}`);
      const txt = await resp.text();
      if (!resp.ok || !txt.trim().startsWith("{")) throw new Error("not found");
      const h = JSON.parse(txt)[id];
      if (!h) throw new Error("not found");
      renderHorse(h);
    } catch (e) {
      $("#horse").innerHTML = `<section class="panel"><p class="err">「${esc(name)}」のページがありません。2026 年に出走した馬だけ検索できます（表記は出走表の馬名と同じ、全角カナ）。</p></section>`;
    }
  }

  function goto(name) {
    const u = new URL(location.href);
    u.searchParams.set("name", name); u.hash = "";
    history.pushState(null, "", u.toString());
    showHorse(name);
    $("#sug").hidden = true;
  }

  function sugRender(q) {
    const box = $("#sug");
    if (!q || !index) { box.hidden = true; return; }
    const nq = q.normalize("NFKC");
    const pre = [], inc = [];
    for (const [nm, n, last] of index) {
      const i = nm.indexOf(nq);
      if (i === 0) pre.push([nm, n, last]); else if (i > 0) inc.push([nm, n, last]);
      if (pre.length >= 20) break;
    }
    const hits = pre.concat(inc).slice(0, 20);
    if (!hits.length) { box.innerHTML = '<a><small>該当なし</small></a>'; box.hidden = false; return; }
    box.innerHTML = hits.map(([nm, n, last]) =>
      `<a href="horse.html?name=${encodeURIComponent(nm)}" data-nm="${esc(nm)}">${esc(nm)}<small>${n}走 ・ 最終 ${fmtDate(last)}</small></a>`).join("");
    box.hidden = false;
  }

  async function boot() {
    try { built = (await (await fetch("data/manifest.json?t=" + Date.now())).json()).built_at || "0"; } catch (e) { /* noop */ }
    const q = $("#q");
    q.addEventListener("focus", loadIndex);
    q.addEventListener("input", async () => { await loadIndex(); sugRender(q.value.trim()); });
    q.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        const first = $("#sug a[data-nm]");
        const v = q.value.trim();
        if (first && first.dataset.nm) goto(first.dataset.nm); else if (v) goto(v);
      } else if (e.key === "Escape") { $("#sug").hidden = true; }
    });
    $("#sug").addEventListener("click", (e) => {
      const a = e.target.closest("a[data-nm]");
      if (!a) return;
      e.preventDefault(); goto(a.dataset.nm);
    });
    document.addEventListener("click", (e) => { if (!e.target.closest(".search")) $("#sug").hidden = true; });
    addEventListener("popstate", () => showHorse(nameFromUrl()));
    const n = nameFromUrl();
    if (n) showHorse(n);
  }
  boot();
})();
