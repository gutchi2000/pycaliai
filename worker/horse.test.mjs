import test from "node:test";
import assert from "node:assert/strict";
import { horseId, horseDescription } from "./horse.js";

test("horseId は Python build_horse_pages.horse_id と同じ (FNV-1a 32bit ×2, UTF-8)", () => {
  // build_horse_pages.horse_id("ホーフアイゼン") === "1e0935dd2e682663" (site/data/h/1e.json のキー)
  assert.equal(horseId("ホーフアイゼン"), "1e0935dd2e682663");
  assert.match(horseId("テスト"), /^[0-9a-f]{16}$/);
});

test("horseDescription は集計値だけを文章にする", () => {
  const d = horseDescription({ name: "テスト", sex: "牡", age: 3, trainer: "誰か", stats: { n: 5, n_settled: 4, hon_n: 2, hon_win: 1, hon_top3: 2, marked_n: 4, marked_top3: 3, avg_pwin: 0.1234, best_ai_rank: 1 } });
  assert.ok(d.includes("テスト（牡 3歳 誰か厩舎）"));
  assert.ok(d.includes("5 走") && d.includes("AI◎ 2 回で勝ち 1") && d.includes("12.3%"));
  assert.ok(!/オッズ|払戻|EV/.test(d));
});
