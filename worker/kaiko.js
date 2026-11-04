/* ============================================================
   PyCaLiAI 回顧記事（2026-11-04）— 「負けた理由まで書く AI」
     GET  /api/kaiko?date=YYYYMMDD      その日の回顧一覧（KV にある記事 + 未生成の race_id）
     POST /api/kaiko/gen {date, race_id} 1 レース（race_id="day" で日の総括 + X 下書き）を生成して KV に保存
     scheduled()                          毎週 月曜 03:00 JST に直近 2 開催日の未生成分を埋める（cron）

   設計: 相談AI と同じく数字は公開済み site/data/{date}.json だけ（scrub_public 済み＝オッズ・払戻・
   調教タイム・上り3F は存在しない）。facts を Worker が機械的に組み、LLM は文章にするだけ。
   記事は KV に永続（キー kaiko:v1:{date}:{race_id}）。生成は 1 レース 1 回だけ（コストは月数ドル）。
   ============================================================ */
import Anthropic from "@anthropic-ai/sdk";
import { courseBias, horseByUma, raceLabel } from "./tools.js";
import { cors, json, jstParts, kvJson, kvNum, kvSet, loadDay, loadManifest, microUsd, num, originOk } from "./util.js";

const VER = "v1";
const MARK_TIER = { "◎": 1, "〇": 2, "○": 2, "▲": 3, "△": 4 };
const pct = (v) => (v == null ? null : +(v * 100).toFixed(1));

const SYSTEM_RACE = `あなたは JRA 中央競馬 AI 予想サイト「PyCaLiAI」の回顧記事ライター。
与えられた facts（JSON）だけを根拠に、そのレースの「AI の予想 → 結果 → ズレた理由（当たったなら当たった理由）」を書く。

【厳守】
- 数字（勝率・複勝率・着順・人気順位・枠別複勝率など）は facts にある値だけ。無い数字を作らない。
- facts に無いこと（オッズ、払戻金、道中の不利、馬場の詳細、騎手の判断、調教、馬体重、馬の気持ち）は書かない。推測で補わない。
- 外れたときに言い訳をしない。facts に見える要因だけを挙げる: 人気と AI 評価の乖離（market_view）、
  脚質と course_bias の傾向、枠、◎の近走の不安定さ（avg_pos / last_run）、混戦度（chaos）、勝ち馬の AI 順位と人気。
- 当たったときも持ち上げすぎない。「AI 評価どおり」で済ませ、根拠を 1 つ添える。
- 馬券の購入を促す言葉（買え・勝負・鉄板・おいしい）は使わない。
- 文体は常体。見出し・箇条書き・絵文字なし。5〜8 行、各行は 1 文か 2 文。
- 1 行目は「予想」（◎と評価）、2 行目は「結果」（1〜3 着と◎の着順）、3 行目以降が理由。最後の行は「教訓:」で始める 1 文。`;

const SYSTEM_DAY = `あなたは JRA 中央競馬 AI 予想サイト「PyCaLiAI」の回顧記事ライター。
与えられた facts（JSON）だけを根拠に、その開催日の総括を書く。

【厳守】
- 数字は facts にある値だけ。レース数・◎の勝利数と複勝圏数・買い目の的中数・固有名は facts から。
- 良かった点と悪かった点を同じ重さで書く。外れた日は外れた日として書く。言い訳をしない。
- オッズ・払戻金・収支額は facts に無いので書かない。
- 馬券の購入を促す言葉は使わない。見出し・箇条書き・絵文字なし。常体。

出力形式（この 2 部構成を厳守）:
1 部: 総括本文 6〜10 行。最初の行に日付と開催場、◎の成績（勝利数/複勝圏数/レース数）。次に目立った的中と、いちばん外した（AI 順位や人気が低い馬が勝った）レースを固有名で 1 つずつ。最後の行は「教訓:」で始める 1 文。
2 部: 区切り行「---X---」の後に、X 投稿用の下書きを 140 字以内で 1 本。1 行目は日付と「回顧」、
  ◎の成績と目立ったレース 1 つ、ハッシュタグ無し、末尾に pycaliai.com/kaiko 。`;

/* ---------------- facts ---------------- */
function horseFacts(race, h, order) {
  if (!h) return null;
  const run = ((h.history || {}).runs || [])[0];
  return {
    umaban: h.umaban, name: h.name, mark: h.mark || "無印", ai_rank: h.ai_rank,
    p_win_pct: pct(h.p_win), p_plc_pct: pct(h.p_plc), ninki_morning: h.ninki ?? null,
    market_view: h.vs_market || null, umami: h.umami ? h.umami.grade : null,
    style: h.style || null, waku: h.waku ?? null, sex_age: `${h.sex || ""}${h.age || ""}`,
    avg_pos_last5: (h.history || {}).avg_pos ?? null,
    last_run: run ? `${run.race_name || ""} ${run.pos}着(${run.ninki ? `${run.ninki}人気・` : ""}${run.place || ""}${run.td || ""}${run.dist || ""})` : null,
    result_pos: order[String(h.umaban)] ?? null,
  };
}

/** 1 レース分の facts。結果が無ければ null。 */
export function kaikoFacts(day, race) {
  const res = race.result;
  if (!res || !res.top3 || !res.order) return null;
  const order = res.order;
  const marks = (race.horses || []).filter((h) => h.mark && MARK_TIER[h.mark]).sort((a, b) => MARK_TIER[a.mark] - MARK_TIER[b.mark]);
  const honmei = marks[0] || null;
  const top3 = res.top3.map((u) => horseFacts(race, horseByUma(race, u), order)).filter(Boolean);
  const bets = ((race.tact && race.tact.bets) || (race.cowork && race.cowork.bets) || []);
  const settled = (race.tact && race.tact.bets) ? (race.tact_settled || []) : (race.bets_settled || []);
  const betRows = bets.map((b, i) => ({ type: b.type, selection: b.selection, hit: settled[i] ? !!settled[i].is_win : null }));
  const cb = courseBias(day, race);
  const c = race.confidence || {};
  const j = race.judgment || {};
  const chaos = c.field_chaos_score ?? 0;
  const winner = top3[0] || null;
  return {
    race: raceLabel(race), date: day.date, baba: (race.baba || "").replace("(暫定)", ""), weather: (race.weather || "").replace("(暫定)", ""),
    ai_view: {
      chaos: +chaos.toFixed(2), chaos_verdict: chaos >= 0.85 ? "見送り寄り" : chaos >= 0.6 ? "妙味次第" : "勝負可",
      top1_dominance: c.top1_dominance ?? null, ai_market_agreement: c.ai_market_agreement ?? null,
      site_headline: j.headline || null, site_detail: j.detail || null,
    },
    marks: marks.map((h) => horseFacts(race, h, order)),
    honmei_reasons: honmei ? (honmei.why || []).slice(0, 3).map((w) => `${w.label}${(w.contrib ?? 0) >= 0 ? "(＋)" : "(－)"}`) : [],
    result_top3: top3,
    honmei_result_pos: honmei ? order[String(honmei.umaban)] ?? null : null,
    honmei_in_top3: honmei ? (order[String(honmei.umaban)] ?? 99) <= 3 : null,
    winner_ai_rank: winner ? winner.ai_rank : null, winner_ninki_morning: winner ? winner.ninki_morning : null,
    site_bets: betRows, site_bets_hit: betRows.filter((b) => b.hit).length, site_bets_n: betRows.length,
    course_bias: cb.error ? null : { verdict: cb.verdict, bands: cb.bands, kyaku: cb.kyaku },
  };
}

/** 日の総括用 facts。 */
export function dayFacts(day) {
  const rows = [];
  for (const r of day.races || []) {
    const f = kaikoFacts(day, r);
    if (!f) continue;
    const h = f.marks[0] || null, w = f.result_top3[0] || null;
    rows.push({
      race_id: r.race_id, place: r.place, rno: r.rno, klass: r.klass, race_name: r.race_name || null,
      honmei: h ? { name: h.name, p_win_pct: h.p_win_pct, ninki_morning: h.ninki_morning, result_pos: h.result_pos } : null,
      winner: w ? { name: w.name, ai_rank: w.ai_rank, ninki_morning: w.ninki_morning, p_win_pct: w.p_win_pct } : null,
      chaos: f.ai_view.chaos, site_bets_hit: f.site_bets_hit, site_bets_n: f.site_bets_n,
    });
  }
  const n = rows.length;
  const hw = rows.filter((x) => x.honmei && x.honmei.result_pos === 1).length;
  const ht = rows.filter((x) => x.honmei && x.honmei.result_pos != null && x.honmei.result_pos <= 3).length;
  const upsets = rows.filter((x) => x.winner && (x.winner.ai_rank >= 6 || (x.winner.ninki_morning ?? 0) >= 6))
    .sort((a, b) => (b.winner.ai_rank + (b.winner.ninki_morning || 0)) - (a.winner.ai_rank + (a.winner.ninki_morning || 0))).slice(0, 3);
  const solid = rows.filter((x) => x.honmei && x.honmei.result_pos === 1).sort((a, b) => b.honmei.p_win_pct - a.honmei.p_win_pct).slice(0, 3);
  return {
    date: day.date, places: day.places || [], n_races: n,
    honmei_win: hw, honmei_top3: ht, honmei_win_rate_pct: n ? +(hw / n * 100).toFixed(1) : null, honmei_top3_rate_pct: n ? +(ht / n * 100).toFixed(1) : null,
    site_bets_hit: rows.reduce((s, x) => s + x.site_bets_hit, 0), site_bets_n: rows.reduce((s, x) => s + x.site_bets_n, 0),
    biggest_upsets: upsets, solid_hits: solid, races: rows,
  };
}

/* ---------------- generation ---------------- */
const key = (date, rid) => `kaiko:${VER}:${date}:${rid}`;

async function callClaude(env, system, facts, maxTokens) {
  const model = env.KAIKO_MODEL || "claude-sonnet-5-5";
  const client = new Anthropic({ apiKey: String(env.ANTHROPIC_API_KEY || "").trim(), maxRetries: 1, timeout: 60_000 });
  const res = await client.messages.create({
    model, max_tokens: maxTokens, system, output_config: { effort: "low" },
    messages: [{ role: "user", content: "facts:\n" + JSON.stringify(facts, null, 0) }],
  });
  const text = res.stop_reason === "refusal" ? "" : res.content.filter((b) => b.type === "text").map((b) => b.text).join("").trim();
  return { text, model, usage: { input_tokens: res.usage?.input_tokens || 0, output_tokens: res.usage?.output_tokens || 0 }, stop: res.stop_reason };
}

/** 1 件生成して KV に保存。既にあればそれを返す。戻り値 {item, created} か {error, code}。 */
export async function generateOne(env, baseUrl, date, rid, ctx) {
  const kv = env.SOUDAN_KV;
  const k = key(date, rid);
  const have = await kvJson(kv, k);
  if (have) return { item: have, created: false };
  if (!String(env.ANTHROPIC_API_KEY || "").trim()) return { error: "鍵未設定", code: "no_key" };
  const { ymd, ym, iso } = jstParts();
  const dailyCap = num(env.KAIKO_DAILY_GEN, 150), monthCap = num(env.SOUDAN_MONTH_USD_CAP, 90);
  const [gen, spend] = await Promise.all([kvNum(kv, `g_kaiko:${ymd}`), kvNum(kv, `spend:${ym}`)]);
  if (gen >= dailyCap) return { error: "本日の生成枠は終了しました（後日自動で埋まります）", code: "daily_limit" };
  if (spend / 1e6 >= monthCap) return { error: "今月の生成枠は終了しました", code: "month_limit" };
  const day = await loadDay(env, baseUrl, date);
  if (!day) return { error: "その日のデータがありません", code: "no_day" };
  // 二重生成の抑止（結果整合なので完全ではない。重複しても KV 上書きで害はない）
  const lockKey = `kaiko:lock:${date}:${rid}`;
  if (await kv.get(lockKey)) return { error: "生成中です", code: "busy" };
  await kvSet(kv, lockKey, "1", 60);
  let facts, system, maxTokens;
  if (rid === "day") {
    facts = dayFacts(day);
    if (!facts.n_races) return { error: "結果のあるレースがありません", code: "no_result" };
    system = SYSTEM_DAY; maxTokens = 900;
  } else {
    const race = (day.races || []).find((r) => String(r.race_id) === String(rid));
    if (!race) return { error: "そのレースが見つかりません", code: "no_race" };
    facts = kaikoFacts(day, race);
    if (!facts) return { error: "まだ結果が入っていません", code: "no_result" };
    system = SYSTEM_RACE; maxTokens = 600;
  }
  let out;
  try { out = await callClaude(env, system, facts, maxTokens); }
  catch (e) { return { error: "生成に失敗しました（上流）", code: "upstream", status: e && e.status || null }; }
  if (!out.text) return { error: "生成結果が空でした", code: "empty" };
  let item;
  if (rid === "day") {
    const [body, x] = out.text.split(/\n-{3}X-{3}\n?/);
    item = { date, race_id: "day", text: (body || "").trim(), x_post: (x || "").trim().slice(0, 140), model: out.model, generated_at: iso, usage: out.usage,
      stats: { n_races: facts.n_races, honmei_win: facts.honmei_win, honmei_top3: facts.honmei_top3, site_bets_hit: facts.site_bets_hit, site_bets_n: facts.site_bets_n } };
  } else {
    item = { date, race_id: rid, text: out.text, model: out.model, generated_at: iso, usage: out.usage,
      honmei_in_top3: facts.honmei_in_top3, honmei_result_pos: facts.honmei_result_pos, site_bets_hit: facts.site_bets_hit, site_bets_n: facts.site_bets_n };
  }
  const micro = microUsd(out.model, out.usage.input_tokens, out.usage.output_tokens);
  const writes = Promise.all([
    kvSet(kv, k, JSON.stringify(item)),
    kvSet(kv, `g_kaiko:${ymd}`, gen + 1, 2 * 86400),
    kvSet(kv, `spend:${ym}`, spend + micro, 40 * 86400),
  ]);
  if (ctx) ctx.waitUntil(writes); else await writes;
  return { item, created: true };
}

/** レース一覧 + KV にある記事。 */
export async function listKaiko(env, baseUrl, date) {
  const day = await loadDay(env, baseUrl, date);
  if (!day) return null;
  const kv = env.SOUDAN_KV;
  const races = (day.races || []).filter((r) => r.result && r.result.top3);
  const items = await Promise.all([kvJson(kv, key(date, "day")), ...races.map((r) => kvJson(kv, key(date, r.race_id)))]);
  const dayItem = items[0];
  const out = races.map((r, i) => {
    const order = r.result.order || {};
    const marks = (r.horses || []).filter((h) => h.mark && MARK_TIER[h.mark]).sort((a, b) => MARK_TIER[a.mark] - MARK_TIER[b.mark]);
    const h = marks[0] || null;
    const it = items[i + 1];
    return {
      race_id: r.race_id, place: r.place, rno: r.rno, race_name: r.race_name || "", klass: r.klass, course: r.course, start_time: r.start_time || "",
      honmei: h ? { umaban: h.umaban, name: h.name, p_win_pct: pct(h.p_win), ninki: h.ninki ?? null, pos: order[String(h.umaban)] ?? null } : null,
      marks: marks.map((m) => ({ umaban: m.umaban, mark: m.mark, pos: order[String(m.umaban)] ?? null })),
      top3: r.result.top3.map((u) => { const x = horseByUma(r, u); return { umaban: u, name: x ? x.name : "", mark: x && x.mark ? x.mark : "", ai_rank: x ? x.ai_rank : null }; }),
      bets_hit: it ? it.site_bets_hit : null, bets_n: it ? it.site_bets_n : null,
      text: it ? it.text : null, model: it ? it.model : null, generated_at: it ? it.generated_at : null,
    };
  });
  return {
    date, places: day.places || [], n_races: out.length,
    day: dayItem, races: out,
    missing: [...(dayItem ? [] : ["day"]), ...out.filter((x) => !x.text).map((x) => x.race_id)],
  };
}

/* ---------------- HTTP handlers ---------------- */
export async function handleList(request, env) {
  const origin = request.headers.get("origin") || "";
  const C = originOk(origin) ? cors(origin) : {};
  if (request.method === "OPTIONS") return new Response(null, { status: 204, headers: C });
  const date = new URL(request.url).searchParams.get("date") || "";
  if (!/^\d{8}$/.test(date)) return json({ error: "bad date" }, 400, C);
  const out = await listKaiko(env, request.url, date);
  if (!out) return json({ error: "その日のデータがありません", code: "no_day" }, 404, C);
  return json(out, 200, { ...C, "cache-control": "no-store" });
}

export async function handleGen(request, env, ctx) {
  const origin = request.headers.get("origin") || "";
  const ok = originOk(origin);
  if (request.method === "OPTIONS") return new Response(null, { status: ok ? 204 : 403, headers: ok ? cors(origin) : {} });
  if (request.method !== "POST") return json({ error: "POST only" }, 405);
  if (!ok) return json({ error: "forbidden origin" }, 403);
  const C = cors(origin);
  let body;
  try { body = await request.json(); } catch { return json({ error: "bad json" }, 400, C); }
  const date = String(body.date || ""), rid = String(body.race_id || "");
  if (!/^\d{8}$/.test(date) || !(rid === "day" || /^\d{16}$/.test(rid))) return json({ error: "bad date/race_id" }, 400, C);
  const r = await generateOne(env, request.url, date, rid, ctx);
  if (r.error) return json(r, r.code === "daily_limit" || r.code === "month_limit" ? 429 : r.code === "busy" ? 409 : r.code === "upstream" ? 502 : 404, C);
  return json({ item: r.item, created: r.created }, 200, C);
}

/** cron: 直近 2 開催日（結果あり）の未生成分を順に埋める。 */
export async function warmLatest(env, baseUrl, maxDates = 2) {
  const mf = await loadManifest(env, baseUrl);
  if (!mf) return { error: "no manifest" };
  const dates = (mf.dates || []).filter((d) => d.has_results).slice(0, maxDates).map((d) => d.date);
  const log = [];
  for (const date of dates) {
    const lst = await listKaiko(env, baseUrl, date);
    if (!lst) continue;
    const todo = lst.missing.filter((x) => x !== "day");
    let made = 0, failed = 0;
    for (const rid of todo) {
      const r = await generateOne(env, baseUrl, date, rid, null);
      if (r.error) { failed++; if (r.code === "daily_limit" || r.code === "month_limit" || r.code === "no_key") { log.push(`${date}: stop (${r.code})`); return { log }; } }
      else if (r.created) made++;
    }
    if (lst.missing.includes("day") || made) {
      const r = await generateOne(env, baseUrl, date, "day", null);
      if (r.error) failed++;
    }
    log.push(`${date}: races made=${made} failed=${failed} (missing before=${todo.length})`);
  }
  return { log };
}
