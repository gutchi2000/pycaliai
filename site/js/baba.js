/* 画面上部の馬場バイアス パネル（fetch_baba_today.py → data/baba_today.json）
   ★会場スコープ (2026-07-11): 選択中の会場タブ(state.place)の馬場だけ表示する。
     他のUI(出走表/レースヘッダー/実現バイアス等)と同じく「今見てる会場」に揃える。
     app.js が renderNav()/setMode() で window.__babaPlace をセットし window.renderBabaBias()。
   ★見た目 (2026-07-11): カードを 芝(左)|ダート(右) に2分割。各面に「クッション値/含水率」の
     横ゲージ（メンバーレベルバー準拠：低→高が左→右。JRA公式グラフは硬いほど左だが、サイトの
     メンバーレベルバーに揃えて反転＝軟らかい左→硬い右）。芝は含水率も併記（芝バイアスは
     クッション値主体だが含水率データも取得済み）。
   鮮度ゲート (2026-07-04): d.date が閲覧日と一致する時だけ表示（stale バグ対策）。 */
(function () {
  function esc(s) { return String(s == null ? "" : s).replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c])); }
  function todayISO() {
    var d = new Date();
    return d.getFullYear() + "-" + String(d.getMonth() + 1).padStart(2, "0")
      + "-" + String(d.getDate()).padStart(2, "0");
  }
  var babaData = null;   // fetch 結果のキャッシュ（会場切替の度に再fetchしないため）

  // --- ゲージ位置 -------------------------------------------------------------
  function clampPct(p) { return Math.max(3, Math.min(97, p)); }
  // 5ゾーン等幅バー上での値の位置(%)。edges は 6 個の区切り (fetch_baba_today.py の firmness と対応)。
  function pct5(v, edges) {
    if (v == null) return null;
    var c = Math.max(edges[0], Math.min(edges[edges.length - 1], v));
    for (var i = 0; i < 5; i++) {
      if (i === 4 || c < edges[i + 1]) {
        var f = (c - edges[i]) / (edges[i + 1] - edges[i]);
        return ((i + f) / 5) * 100;
      }
    }
    return 50;
  }
  function fx1(v) { return v == null ? "—" : (+v).toFixed(1); }
  function frontW(pp) { return pp == null ? 0 : Math.max(2, Math.min(100, (pp / 40) * 100)); }

  function wakuChip(w) {
    if (w == null) return '<span class="bw flat">—</span>';
    var cls = w >= 1 ? "out" : w <= -1 ? "in" : "flat";
    var txt = w >= 1 ? "外枠有利" : w <= -1 ? "内枠有利" : "枠均等";
    return '<span class="bw ' + cls + '">' + txt + "</span>";
  }
  function frontStat(b) {
    if (!b) return "";
    var pp = b.front_edge;
    return '<div class="bstat"><span class="bk">前残り</span>'
      + '<span class="bbar"><i style="width:' + frontW(pp).toFixed(0) + '%"></i></span>'
      + '<b class="bv">' + (pp >= 0 ? "+" : "") + esc(pp) + "<small>pp</small></b>"
      + '<span class="bsub">逃' + esc(b.nige_win) + " 先" + esc(b.senko_win) + "</span></div>";
  }
  // 有利脚質 = 前残り度(pp, 逃げ・先行の超過勝率)から結論を言語化。JRAは基本前有利で度合いが変わる。
  function favoredStyle(b) {
    if (!b || b.front_edge == null) return null;
    var fe = b.front_edge;
    if (fe >= 27) return { txt: "逃げ・先行 有利", cls: "front", lv: "強" };
    if (fe >= 20) return { txt: "逃げ・先行 有利", cls: "front", lv: "" };
    if (fe >= 13) return { txt: "やや前有利", cls: "front-mild", lv: "" };
    if (fe > 4) return { txt: "ほぼフラット", cls: "flat", lv: "" };
    return { txt: "差し・追込 有利", cls: "back", lv: "" };
  }
  // 脚質 と 枠 を横並び(同じ行)で出す（ユーザー指定 2026-07-11）。
  function chipsRow(b) {
    if (!b) return "";
    var s = favoredStyle(b);
    var kyaku = s
      ? '<span class="kchip ' + s.cls + '">' + s.txt + (s.lv ? "<em>" + s.lv + "</em>" : "") + "</span>"
      : '<span class="kchip flat">—</span>';
    return '<div class="bstat bchips">'
      + '<span class="pair"><span class="bk">脚質</span>' + kyaku + "</span>"
      + '<span class="pair"><span class="bk">枠</span>' + wakuChip(b.waku_out_minus_in) + "</span>"
      + "</div>";
  }

  // 含水率ゲージ (乾←→湿)。芝・ダート共通。0〜15% を左→右にマップ。
  function moistGauge(gp) {
    if (gp == null) return '<div class="cg-empty">含水率 未掲載</div>';
    var left = clampPct((Math.max(0, Math.min(15, gp)) / 15) * 100).toFixed(1);
    return '<div class="cg cg-moist"><div class="cg-track"><span class="cg-fill"></span>'
      + '<span class="cg-mark" style="left:' + left + '%"><b>' + fx1(gp) + "%</b></span>"
      + '</div><div class="cg-cap"><span>乾</span><span class="mid">含水率</span><span>湿</span></div></div>';
  }

  // --- 芝 半分 (クッション値ゲージ 軟←→硬 + 含水率ゲージ 乾←→湿) ----------------
  function shibaHalf(v) {
    var b = v.shiba_bias, cu = v.cushion;
    var gauge;
    if (cu != null) {
      var left = clampPct(pct5(cu, [6, 7, 8, 10, 12, 13])).toFixed(1);
      gauge = '<div class="cg"><div class="cg-track">'
        + '<span class="cg-z z1">軟</span><span class="cg-z z2">やや軟</span>'
        + '<span class="cg-z z3">標準</span><span class="cg-z z4">やや硬</span><span class="cg-z z5">硬</span>'
        + '<span class="cg-mark" style="left:' + left + '%"><b>' + fx1(cu) + "</b></span>"
        + '</div><div class="cg-cap"><span>軟らかい</span><span class="mid">クッション値</span><span>硬い</span></div></div>';
    } else {
      gauge = '<div class="cg-empty">クッション値 未掲載</div>';
    }
    return '<div class="bsurf"><div class="bsurf-h"><span class="bsurf-tag shiba">芝</span>'
      + '<b class="bsurf-band">' + esc(v.cushion_firmness || (b && b.band) || "—") + "</b></div>"
      + gauge + moistGauge(v.shiba_gp)
      + '<div class="bstats">' + chipsRow(b) + frontStat(b) + "</div></div>";
  }

  // --- ダート 半分 (含水率ゲージ, 乾←→湿) ------------------------------------
  function dirtHalf(v) {
    var b = v.dirt_bias;
    return '<div class="bsurf"><div class="bsurf-h"><span class="bsurf-tag dirt">ダート</span>'
      + '<b class="bsurf-band">' + esc((b && b.band) || "—") + "</b></div>"
      + moistGauge(v.dirt_gp)
      + '<div class="bstats">' + chipsRow(b) + frontStat(b) + "</div></div>";
  }

  function venueCard(v) {
    return '<div class="baba-card"><div class="baba-vname">' + esc(v.venue) + "</div>"
      + '<div class="baba-split">' + shibaHalf(v) + dirtHalf(v) + "</div></div>";
  }

  function draw() {
    var el = document.getElementById("babaBias");
    if (!el) return;
    var d = babaData;
    if (!d || !d.venues || !d.venues.length) { el.hidden = true; return; }
    if (d.date !== todayISO()) { el.hidden = true; return; }  // 今日のデータでなければ出さない
    // ★選択中の会場だけに絞る。会場未選択 or その会場の馬場が無ければ出さない。
    var place = window.__babaPlace || null;
    var vs = place ? d.venues.filter(function (v) { return v.venue === place; }) : [];
    if (!vs.length) { el.hidden = true; return; }
    var cards = vs.map(venueCard).join("");
    // 当日測定がまだ掲載されていない早朝は前日値が最新。正直に「前日測定」と示す。
    var stale = (d.is_measured_today === false);
    var head = stale ? "🏇 馬場バイアス（前日測定）" : "🏇 今日の馬場バイアス";
    var note = d.measured_label
      ? ' <span class="sub">' + esc(d.measured_label) + " 測定"
        + (stale ? "・本日値は発表待ち" : "") + " / 過去10年実測</span>"
      : "";
    el.innerHTML = "<h3>" + head + note + '</h3><div class="baba-grid">' + cards + "</div>";
    el.hidden = false;
  }

  // app.js(renderNav / setMode) から会場切替・モード切替の度に呼ばれる。
  window.renderBabaBias = draw;

  // app.js が初回 reveal 前に await できるよう promise を公開 (CLS対策)
  window.__babaReady = fetch("data/baba_today.json?v=" + Date.now())
    .then(function (r) { return r.ok ? r.json() : null; })
    .then(function (d) { babaData = d; draw(); })
    .catch(function () { });
})();
