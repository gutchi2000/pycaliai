/* 今日の馬場バイアス パネル（fetch_baba_today.py → data/baba_today.json）
   SPA(app.js)とは独立。非開催日/データ無しなら非表示。 */
(function () {
  function esc(s) { return String(s == null ? "" : s).replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c])); }
  function render(d) {
    var el = document.getElementById("babaBias");
    if (!el) return;
    if (!d || !d.venues || !d.venues.length) { el.hidden = true; return; }
    var cards = d.venues.map(function (v) {
      var s = v.shiba_bias, dd = v.dirt_bias;
      return '<div class="baba-card">'
        + '<div class="bv">' + esc(v.venue) + '</div>'
        + '<div class="bc">クッション ' + esc(v.cushion != null ? v.cushion : "—")
        + ' <b>' + esc(v.cushion_firmness || "") + '</b>'
        + (v.shiba_gp != null ? '　含水 芝' + esc(v.shiba_gp) + '% / ダ' + esc(v.dirt_gp) + '%' : '') + '</div>'
        + (s ? '<div class="bl shiba">' + esc(s.line) + '</div>' : '')
        + (dd ? '<div class="bl dirt">' + esc(dd.line) + '</div>' : '')
        + '</div>';
    }).join("");
    el.innerHTML = '<h3>🏇 今日の馬場バイアス'
      + (d.measured_label ? ' <span class="sub">' + esc(d.measured_label) + ' 測定 / 過去10年実測</span>' : '')
      + '</h3><div class="baba-grid">' + cards + '</div>';
    el.hidden = false;
  }
  fetch("data/baba_today.json?v=" + Date.now())
    .then(function (r) { return r.ok ? r.json() : null; })
    .then(render)
    .catch(function () { });
})();
