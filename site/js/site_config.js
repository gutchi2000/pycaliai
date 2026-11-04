/* site_config.js — サイト全体の設定 (2026-11-04)
 * discord_invite: Discord 公開サーバーの招待 URL (例 "https://discord.gg/xxxxxxx")。
 *   空のままならナビに Discord は出ない。入れると全ページのナビ末尾に「Discord」が出る。 */
window.PYCALIAI_CONFIG = {
  discord_invite: "https://discord.gg/8BVbzxysd8"
};
(function () {
  var c = window.PYCALIAI_CONFIG || {};
  var a = document.getElementById("navDiscord");
  if (a && c.discord_invite) {
    a.href = c.discord_invite;
    a.target = "_blank";
    a.rel = "noopener";
    a.hidden = false;
  }
})();
/* ナビ「データ ▾」: 外側クリック / Escape で閉じる (2026-11-05) */
(function () {
  document.addEventListener("click", function (e) {
    document.querySelectorAll("details.nav-more[open]").forEach(function (d) {
      if (!d.contains(e.target)) d.removeAttribute("open");
    });
  });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") document.querySelectorAll("details.nav-more[open]").forEach(function (d) { d.removeAttribute("open"); });
  });
})();
