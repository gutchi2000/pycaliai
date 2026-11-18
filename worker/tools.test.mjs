// node --test worker/tools.test.mjs  （site/data の最新日 JSON を使う。無ければ skip）
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { compareHorses, courseBias, describeHorse, dispatch, findHorse, pairProb, raceAdvice, rosterLines } from "./tools.js";
import { SYSTEM, TOOLS } from "./prompt.js";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const DATA = path.join(ROOT, "site", "data");
const dates = fs.existsSync(DATA) ? fs.readdirSync(DATA).filter((f) => /^\d{8}\.json$/.test(f)).sort() : [];
const day = dates.length ? JSON.parse(fs.readFileSync(path.join(DATA, dates[dates.length - 1]), "utf8")) : null;
const race = day ? day.races.find((r) => r.horses && r.horses.length >= 8) : null;
const FORBIDDEN = /odds|オッズ|payout|払戻|ev_|期待値|agari|調教/i;

test("prompt: ツールは strict でスキーマが閉じている", () => {
  for (const t of TOOLS) {
    assert.equal(t.strict, true, t.name);
    assert.equal(t.input_schema.additionalProperties, false, t.name);
    assert.ok(Array.isArray(t.input_schema.required), t.name);
  }
  assert.ok(SYSTEM.includes("結論:"));
  assert.ok(SYSTEM.includes("オッズ"));   // 「持っていない」と明記していること
});

test("tools: 出走馬一覧・馬引き・比較・ペア・見送り・枠", { skip: !race && "site/data なし" }, () => {
  const lines = rosterLines(race);
  assert.equal(lines.length, race.horses.length);
  const h1 = race.horses[0];
  assert.equal(findHorse(race, String(h1.umaban)).umaban, h1.umaban);
  assert.equal(findHorse(race, `${h1.umaban}番`).umaban, h1.umaban);
  assert.equal(findHorse(race, h1.name).umaban, h1.umaban);
  assert.equal(findHorse(race, "存在しない馬名X"), null);

  const d = describeHorse(race, h1.umaban);
  assert.equal(d.name, h1.name);
  assert.ok(d.p_plc_pct >= 0 && d.p_plc_pct <= 100);
  assert.ok(Array.isArray(d.past_runs) && d.past_runs.length <= 5);
  assert.ok(d.summary.includes(h1.name));
  assert.ok(!FORBIDDEN.test(JSON.stringify(d)), "公開 JSON に無いはずの数字が出ている");

  const sorted = [...race.horses].sort((a, b) => a.ai_rank - b.ai_rank);
  const c = compareHorses(race, sorted[0].umaban, sorted[1].umaban);
  assert.ok([sorted[0].umaban, sorted[1].umaban].includes(c.pick));
  const pickP = c.pick === sorted[0].umaban ? sorted[0].p_plc : sorted[1].p_plc;
  const otherP = c.pick === sorted[0].umaban ? sorted[1].p_plc : sorted[0].p_plc;
  assert.ok(pickP >= otherP, "軸は複勝率が高い方");
  assert.ok(c.recommendation.includes("複勝率"));

  const pp = pairProb(race, race.pairs[0].a, race.pairs[0].b);
  assert.equal(pp.found, true);
  assert.equal(pp.rank_in_race, 1);
  assert.equal(pairProb(race, 98, 99).found, false);

  const adv = raceAdvice(race);
  assert.ok(["見送り寄り", "妙味次第（厚張りしない）", "勝負可"].includes(adv.verdict));
  assert.equal(adv.top3_by_ai.length, Math.min(3, race.horses.length));
  assert.ok(!FORBIDDEN.test(JSON.stringify(adv)));

  const cb = courseBias(day, race);
  if (!cb.error) {
    assert.ok(Object.keys(cb.bands).length >= 2);
    assert.ok(cb.summary.includes("複勝"));
  }
});

test("dispatch: 未知ツールと見つからない馬はエラー＋名簿", { skip: !race && "site/data なし" }, () => {
  assert.ok(dispatch(day, race, "nope", {}).error);
  const r = dispatch(day, race, "horse_info", { umaban_or_name: "ゼッタイイナイ" });
  assert.ok(r.error && Array.isArray(r.roster));
  assert.equal(dispatch(day, race, "race_advice", {}).race.length > 0, true);
});

// ---- 2026-11-04: 日横断ツールと回顧 facts ----
import { dayRaces } from "./tools.js";
import { dayFacts, kaikoFacts } from "./kaiko.js";

test("day_races: 全レース・堅い順・◎が人気薄の一覧", { skip: !race && "site/data なし" }, () => {
  const d = dayRaces(day, race.race_id);
  assert.equal(d.n_races, day.races.length);
  assert.ok(d.most_solid.length >= 1 && d.most_chaotic.length >= 1);
  assert.ok(d.races.some((r) => r.current === true));
  const chaosSorted = [...d.races].sort((a, b) => a.chaos - b.chaos);
  assert.ok(d.most_solid[0].includes(`${chaosSorted[0].place}${chaosSorted[0].rno}R`));
  assert.ok(!FORBIDDEN.test(JSON.stringify(d)));
});

test("kaiko: facts は結果のあるレースだけ、数字は公開 JSON と一致、禁止語なし", { skip: !day && "site/data なし" }, () => {
  const done = day.races.filter((r) => r.result && r.result.top3);
  const notDone = day.races.find((r) => !r.result || !r.result.top3);
  if (notDone) assert.equal(kaikoFacts(day, notDone), null);
  if (!done.length) return;
  const r = done[0];
  const f = kaikoFacts(day, r);
  assert.equal(f.result_top3.length, r.result.top3.length);
  assert.equal(f.result_top3[0].umaban, r.result.top3[0]);
  assert.equal(f.result_top3[0].result_pos, 1);
  const hon = r.horses.find((h) => h.mark === "◎");
  if (hon) {
    assert.equal(f.marks[0].umaban, hon.umaban);
    assert.equal(f.honmei_result_pos, r.result.order[String(hon.umaban)]);
    assert.equal(f.honmei_in_top3, r.result.order[String(hon.umaban)] <= 3);
  }
  assert.equal(f.site_bets_n, f.site_bets.length);
  assert.ok(f.site_bets_hit <= f.site_bets_n);
  assert.ok(!FORBIDDEN.test(JSON.stringify(f)), "公開 JSON に無いはずの数字が出ている");
  const df = dayFacts(day);
  assert.equal(df.n_races, done.length);
  assert.ok(df.honmei_top3 >= df.honmei_win);
  assert.equal(df.races.length, done.length);
  assert.ok(!FORBIDDEN.test(JSON.stringify(df)));
});

// ---- 2026-11-04: 見どころ facts ----
import { previewFacts } from "./kaiko.js";
test("preview: 結果を使わず全レース、堅い/割れる/荒れる の一覧、禁止語なし", { skip: !day && "site/data なし" }, () => {
  const f = previewFacts(day);
  assert.equal(f.n_races, day.races.length);
  assert.equal(f.races.length, day.races.length);
  assert.ok(f.most_solid.length >= 1 && f.most_chaotic.length >= 1);
  assert.ok(!JSON.stringify(f).includes("result_pos"), "見どころは結果を含めない");
  assert.ok(!FORBIDDEN.test(JSON.stringify(f)));
  const solid = f.races.find((x) => f.most_solid[0].startsWith(`${x.place}${x.rno}R`));
  assert.ok(solid && solid.chaos === Math.min(...f.races.map((x) => x.chaos)));
});

// ---- 2026-11-04: 通信簿ツール ----
import { trackRecord, chaosVerdict } from "./tools.js";
test("ai_track_record: report_card.json をこのレースの条件で引く", { skip: !race && "site/data なし" }, () => {
  const rcPath = path.join(DATA, "report_card.json");
  if (!fs.existsSync(rcPath)) return;
  const rc = JSON.parse(fs.readFileSync(rcPath, "utf8"));
  const t = trackRecord(rc, race);
  assert.ok(t.overall.n_races > 0 && t.this_race_conditions.length >= 3);
  for (const c of t.this_race_conditions) assert.ok(c.n_races > 0 && c.honmei_top3_rate_pct >= 0 && c.honmei_top3_rate_pct <= 100);
  assert.ok(["見送り寄り", "妙味次第", "勝負可"].includes(chaosVerdict(race)));
  assert.ok(trackRecord(null, race).error);
});

// ---- 2026-11-18: 展開予想（race_tenkai / system の【展開予想】行） ----
import { lateWord, paceSummary, raceTenkai, tenkaiLine, tenkaiPositions } from "./tools.js";

const d1115 = fs.existsSync(path.join(DATA, "20261115.json")) ? JSON.parse(fs.readFileSync(path.join(DATA, "20261115.json"), "utf8")) : null;
const eriza = d1115 ? d1115.races.find((r) => (r.race_name || "").includes("エリザベス")) : null;
const tmeta = fs.existsSync(path.join(DATA, "tenkai_meta.json")) ? JSON.parse(fs.readFileSync(path.join(DATA, "tenkai_meta.json"), "utf8")) : null;

test("race_tenkai: エリザベス女王杯 (11/15) は展開タブと同じハイペース寄り", { skip: !eriza && "20261115.json なし" }, () => {
  const t = raceTenkai(eriza, tmeta);
  assert.equal(t.available, true);
  assert.equal(t.pace.label, "ハイ");
  assert.equal(t.pace.sure, "ハイペース寄り");
  assert.equal(t.pace.high_pct, 48);
  assert.equal(t.late_reach, "やや届きやすい");
  assert.deepEqual(t.front_runners.map((x) => x.name), ["セキトバイースト", "レーゼドラマ"]);
  assert.ok(tenkaiLine(eriza, tmeta).startsWith("【展開予想】ペース=ハイ（ハイペース寄り"));
  assert.ok(!FORBIDDEN.test(JSON.stringify(t)));
});

test("race_tenkai: 全レースで予想ラベル=確率最大、番手は 1..n、取消は除いて付け直す", { skip: !d1115 && "20261115.json なし" }, () => {
  for (const r of d1115.races.filter((x) => x.tenkai)) {
    const t = raceTenkai(r, tmeta);
    const v = r.tenkai.pace.map((x) => x ?? 0);
    assert.equal(t.pace.label, PACE_NAMES_T[v.indexOf(Math.max(...v))], r.race_id);
    const live = r.horses.filter((h) => !h.scratched).length;
    assert.deepEqual(t.horses.map((x) => x.early_rank), Array.from({ length: t.horses.length }, (_, i) => i + 1));
    assert.ok(t.horses.length <= live);
    assert.equal(t.front_runners.length, Math.min(2, t.horses.length));
  }
  const r = d1115.races.find((x) => x.tenkai && x.horses.length >= 8);
  const first = raceTenkai(r, tmeta).horses[0].umaban;
  const r2 = { ...r, horses: r.horses.map((h) => (h.umaban === first ? { ...h, scratched: true } : h)) };
  const t2 = raceTenkai(r2, tmeta);
  assert.ok(!t2.horses.some((x) => x.umaban === first), "取消馬は展開予想から外す");
  assert.equal(t2.horses[0].early_rank, 1);
  assert.equal(tenkaiPositions(r2).get(first), undefined);
});
const PACE_NAMES_T = ["スロー", "平均", "ハイ"];

test("race_tenkai: 展開予想の無いレースは available=false で、system の行も「データなし」", { skip: !race && "site/data なし" }, () => {
  const r = { ...race, tenkai: null };
  const t = raceTenkai(r, tmeta);
  assert.equal(t.available, false);
  assert.ok(t.note.includes("展開予想データなし"));
  assert.ok(tenkaiLine(r, tmeta).includes("データなし"));
  assert.equal(describeHorse(r, r.horses[0].umaban).tenkai, null);
  assert.equal(dispatch(day, r, "race_tenkai", {}, { tenkaiMeta: tmeta }).available, false);
});

test("展開予想: 確度の言葉・届くかの区切り・馬の予想位置・ツール定義", { skip: !race && "site/data なし" }, () => {
  assert.equal(paceSummary({ pace: [0.2, 0.32, 0.48] }).sure, "ハイペース寄り");
  assert.equal(paceSummary({ pace: [0.91, 0.08, 0.01] }).sure, "スローペースの公算が大きい");
  assert.equal(paceSummary({ pace: [0.33, 0.34, 0.33] }).sure, "はっきりしない（平均寄り）");
  assert.equal(paceSummary(null), null);
  assert.equal(lateWord(0.02, null), "届きにくい");
  assert.equal(lateWord(0.149, { close: { edges: [0.05391, 0.09364, 0.12887, 0.17277] } }), "やや届きやすい");
  assert.equal(lateWord(null, null), null);
  if (race.tenkai) {
    const d = describeHorse(race, race.horses.find((h) => !h.scratched).umaban);
    assert.ok(d.tenkai && d.tenkai.early_rank >= 1 && ["逃げ", "先行", "差し", "追込"].includes(d.tenkai.tenkai_style));
    assert.equal(dispatch(day, race, "race_tenkai", {}, { tenkaiMeta: tmeta }).available, true);
  }
  assert.ok(TOOLS.some((t) => t.name === "race_tenkai"));
  assert.ok(SYSTEM.includes("race_tenkai") && SYSTEM.includes("展開予想データなし"));
});
