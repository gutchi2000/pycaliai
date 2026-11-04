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
import { chaosPctLabel, chaosVerdict, courseBias, horseByUma, raceLabel } from "./tools.js";
import { cors, json, jstParts, kvJson, kvNum, kvSet, loadDay, loadManifest, microUsd, num, originOk } from "./util.js";

const VER = "v2";   // v1: 2026-11-04 初版（facts のキー名が本文に漏れたため v2 で文体規則を追加して再生成）
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
- facts のキー名（market_view, chaos, top1_dominance, umami, ai_rank, p_win_pct など）やローマ字の値を本文に書かない。
  必ず日本語に言い換える: market_view over＝「人気が AI 評価より先行」/ under＝「AI 評価が人気より高い」/ fair＝「人気と AI 評価が一致」、
  chaos＝「混戦度」、top1_dominance＝「本命優位」、umami 罠＝「人気薄で見かけ上おいしく見えるが AI は推さない馬」、
  ai_rank＝「AI n 位」、p_win_pct＝「勝率 n%」、ninki_morning＝「朝の n 番人気」。
- 文体は常体。見出し・箇条書き・絵文字なし。5〜8 行、各行は 1 文か 2 文。全体で 400 字以内。
- 1 行目は「予想」（◎と評価）、2 行目は「結果」（1〜3 着と◎の着順）、3 行目以降が理由。最後の行は「教訓:」で始める 1 文。`;

const SYSTEM_DAY = `あなたは JRA 中央競馬 AI 予想サイト「PyCaLiAI」の回顧記事ライター。
与えられた facts（JSON）だけを根拠に、その開催日の総括を書く。

【厳守】
- 数字は facts にある値だけ。レース数・◎の勝利数と複勝圏数・買い目の的中数・固有名は facts から。
- 良かった点と悪かった点を同じ重さで書く。外れた日は外れた日として書く。言い訳をしない。
- オッズ・払戻金・収支額は facts に無いので書かない。
- facts のキー名（ai_rank, ninki_morning, chaos など）やローマ字の値を本文に書かない。「AI n 位」「朝の n 番人気」「混戦度」と日本語で書く。
- 馬券の購入を促す言葉は使わない。見出し・箇条書き・絵文字なし。常体。

出力形式（この 2 部構成を厳守）:
1 部: 総括本文 6〜10 行。最初の行に日付と開催場、◎の成績（勝利数/複勝圏数/レース数）。次に目立った的中と、いちばん外した（AI 順位や人気が低い馬が勝った）レースを固有名で 1 つずつ。最後の行は「教訓:」で始める 1 文。
2 部: 区切り行「---X---」の後に、X 投稿用の下書きを 140 字以内で 1 本。1 行目は日付と「回顧」、
  ◎の成績と目立ったレース 1 つ、ハッシュタグ無し、末尾に pycaliai.com/kaiko 。`;

const SYSTEM_PREVIEW = `あなたは JRA 中央競馬 AI 予想サイト「PyCaLiAI」のライター。
与えられた facts（JSON）だけを根拠に、その開催日の「今日の見どころ」（発走前の予告）を書く。

【厳守】
- 数字は facts にある値だけ（勝率・複勝率・混戦度・朝の人気順位・枠別複勝率）。オッズ・払戻・期待値は無いので書かない。
- facts のキー名（chaos, market_view, ai_rank など）やローマ字の値を本文に書かない。「混戦度」「AI 評価が人気より高い」「AI n 位」と日本語で。
- 「買え」「勝負」「鉄板」「おいしい」など購入を促す言葉は使わない。「AI の評価上は」「数字の上では」の言い方。
- 見出し・箇条書き・絵文字なし。常体。全体で 500 字以内。

出力形式（この 2 部構成を厳守）:
1 部: 本文 7〜11 行。1 行目は日付・開催場・レース数と最初の発走時刻。
  次に「AI が堅いと見る 3 鞍」（場所 R、◎馬名、勝率、混戦度）、「AI と人気が割れる鞍」（◎が朝の人気で何番手か、AI 評価が人気より高い理由を根拠ラベルから 1 つ）、
  「見送り寄りの鞍」（混戦度が高い）。重賞があればその◎と根拠を 1 行。最後の行は「注目:」で始める 1 文（その日いちばん見るべき 1 鞍と理由）。
2 部: 区切り行「---X---」の後に X 投稿用の下書きを 140 字以内で 1 本。1 行目は日付と「見どころ」、堅い鞍 1 つと割れる鞍 1 つ、ハッシュタグ無し、末尾に pycaliai.com/kaiko 。`;

/* ---------------- facts ---------------- */
/** 着順。order が 0 でも top3 に居れば同着（build_site の order は同着を 0 にする）→ top3 の位置。
    どちらにも無ければ出走取消・除外・競走中止 → null。 */
function posOf(order, umaban, top3) {
  const p = order[String(umaban)];
  if (p) return p;
  const i = (top3 || []).findIndex((u) => +u === +umaban);
  return i >= 0 ? i + 1 : null;
}

function horseFacts(race, h, order, top3) {
  if (!h) return null;
  const run = ((h.history || {}).runs || [])[0];
  const pos = posOf(order, h.umaban, top3);
  return {
    umaban: h.umaban, name: h.name, mark: h.mark || "無印", ai_rank: h.ai_rank,
    p_win_pct: pct(h.p_win), p_plc_pct: pct(h.p_plc), ninki_morning: h.ninki ?? null,
    market_view: h.vs_market || null, umami: h.umami ? h.umami.grade : null,
    style: h.style || null, waku: h.waku ?? null, sex_age: `${h.sex || ""}${h.age || ""}`,
    avg_pos_last5: (h.history || {}).avg_pos ?? null,
    last_run: run ? `${run.race_name || ""} ${run.pos}着(${run.ninki ? `${run.ninki}人気・` : ""}${run.place || ""}${run.td || ""}${run.dist || ""})` : null,
    result_pos: pos,
    result_note: pos ? (order[String(h.umaban)] ? null : "同着") : "出走取消・除外・競走中止のいずれか（着順なし）",
  };
}

/** 1 レース分の facts。結果が無ければ null。 */
export function kaikoFacts(day, race) {
  const res = race.result;
  if (!res || !res.top3 || !res.order) return null;
  const order = res.order;
  const marks = (race.horses || []).filter((h) => h.mark && MARK_TIER[h.mark]).sort((a, b) => MARK_TIER[a.mark] - MARK_TIER[b.mark]);
  const honmei = marks[0] || null;
  const top3 = res.top3.map((u) => horseFacts(race, horseByUma(race, u), order, res.top3)).filter(Boolean);
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
      chaos: +chaos.toFixed(2), chaos_verdict: chaosVerdict(race), chaos_rank: chaosPctLabel(race),
      top1_dominance: c.top1_dominance ?? null, ai_market_agreement: c.ai_market_agreement ?? null,
      site_headline: j.headline || null, site_detail: j.detail || null,
    },
    marks: marks.map((h) => horseFacts(race, h, order, res.top3)),
    honmei_reasons: honmei ? (honmei.why || []).slice(0, 3).map((w) => `${w.label}${(w.contrib ?? 0) >= 0 ? "(＋)" : "(－)"}`) : [],
    result_top3: top3,
    honmei_result_pos: honmei ? posOf(order, honmei.umaban, res.top3) : null,
    honmei_in_top3: honmei ? (posOf(order, honmei.umaban, res.top3) ?? 99) <= 3 : null,
    honmei_dnf: honmei ? posOf(order, honmei.umaban, res.top3) == null : null,
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

/** 発走前の「今日の見どころ」用 facts（結果は使わない）。 */
export function previewFacts(day) {
  const rows = (day.races || []).map((r) => {
    const c = r.confidence || {};
    const chaos = c.field_chaos_score ?? 0;
    const hon = (r.horses || []).find((h) => h.mark === "◎") || null;
    const maru = (r.horses || []).find((h) => h.mark === "〇" || h.mark === "○") || null;
    const cb = courseBias(day, r);
    const graded = /G[123]/.test(`${r.klass || ""} ${r.race_name || ""}`);
    return {
      place: r.place, rno: r.rno, klass: r.klass, race_name: r.race_name || null, course: r.course, start_time: r.start_time || null, field_size: r.field_size,
      chaos: +chaos.toFixed(2), chaos_pct: (r.judgment || {}).chaos_pct ?? null, verdict: chaosVerdict(r),
      top1_dominance: c.top1_dominance ?? null, ai_market_agreement: c.ai_market_agreement ?? null,
      honmei: hon ? { umaban: hon.umaban, name: hon.name, p_win_pct: pct(hon.p_win), p_plc_pct: pct(hon.p_plc), ninki_morning: hon.ninki ?? null, market_view: hon.vs_market || null, style: hon.style || null, waku: hon.waku ?? null,
        reasons: (hon.why || []).slice(0, 3).map((w) => `${w.label}${(w.contrib ?? 0) >= 0 ? "(＋)" : "(－)"}`) } : null,
      maru: maru ? { umaban: maru.umaban, name: maru.name, p_win_pct: pct(maru.p_win) } : null,
      site_headline: (r.judgment || {}).headline || null,
      course_bias_verdict: cb.error ? null : cb.verdict,
      graded,
    };
  });
  const cv = (x) => (x.chaos_pct != null ? x.chaos_pct : x.chaos);   // 百分位があればそれで並べる
  const byChaos = [...rows].sort((a, b) => cv(a) - cv(b) || ((b.honmei?.p_win_pct ?? 0) - (a.honmei?.p_win_pct ?? 0)));
  const label = (x) => `${x.place}${x.rno}R${x.race_name ? `(${x.race_name})` : ""}`;
  const split = rows.filter((x) => x.honmei && (x.honmei.market_view === "under" || (x.honmei.ninki_morning ?? 1) >= 3))
    .sort((a, b) => (b.honmei.p_win_pct ?? 0) - (a.honmei.p_win_pct ?? 0)).slice(0, 3);
  const starts = rows.map((x) => x.start_time).filter(Boolean).sort();
  return {
    date: day.date, places: day.places || [], n_races: rows.length, first_start: starts[0] || null,
    most_solid: byChaos.slice(0, 3).map(label), most_chaotic: byChaos.slice(-3).reverse().map(label),
    ai_vs_market_split: split.map(label), graded_races: rows.filter((x) => x.graded).map(label),
    races: rows,
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
  } else if (rid === "preview") {
    facts = previewFacts(day);
    if (!facts.n_races) return { error: "出走表がありません", code: "no_race" };
    system = SYSTEM_PREVIEW; maxTokens = 1000;
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
  if (rid === "day" || rid === "preview") {
    const [body, x] = out.text.split(/\n-{3}X-{3}\n?/);
    item = { date, race_id: rid, text: (body || "").trim(), x_post: (x || "").trim().slice(0, 140), model: out.model, generated_at: iso, usage: out.usage,
      stats: rid === "day"
        ? { n_races: facts.n_races, honmei_win: facts.honmei_win, honmei_top3: facts.honmei_top3, site_bets_hit: facts.site_bets_hit, site_bets_n: facts.site_bets_n }
        : { n_races: facts.n_races, n_graded: facts.graded_races.length, first_start: facts.first_start } };
  } else {
    item = { date, race_id: rid, text: out.text, model: out.model, generated_at: iso, usage: out.usage,
      honmei_in_top3: facts.honmei_in_top3, honmei_result_pos: facts.honmei_result_pos, site_bets_hit: facts.site_bets_hit, site_bets_n: facts.site_bets_n };
  }
  const micro = microUsd(out.model, out.usage.input_tokens, out.usage.output_tokens);
  // 記事本体の put は必ず待つ（waitUntil に逃がすと失敗が見えない）。カウンタは後回しでよい。
  const stored = await kvSet(kv, k, JSON.stringify(item));
  const counters = Promise.all([
    kvSet(kv, `g_kaiko:${ymd}`, gen + 1, 2 * 86400),
    kvSet(kv, `spend:${ym}`, spend + micro, 40 * 86400),
  ]);
  if (ctx) ctx.waitUntil(counters); else await counters;
  return { item, created: true, stored };
}

/** レース一覧 + KV にある記事。 */
export async function listKaiko(env, baseUrl, date) {
  const day = await loadDay(env, baseUrl, date);
  if (!day) return null;
  const kv = env.SOUDAN_KV;
  const races = (day.races || []).filter((r) => r.result && r.result.top3);
  const upcoming = races.length === 0;
  const items = await Promise.all([kvJson(kv, key(date, "day")), kvJson(kv, key(date, "preview")), ...races.map((r) => kvJson(kv, key(date, r.race_id)))]);
  const dayItem = items[0], previewItem = items[1];
  const out = races.map((r, i) => {
    const order = r.result.order || {};
    const marks = (r.horses || []).filter((h) => h.mark && MARK_TIER[h.mark]).sort((a, b) => MARK_TIER[a.mark] - MARK_TIER[b.mark]);
    const h = marks[0] || null;
    const it = items[i + 2];
    return {
      race_id: r.race_id, place: r.place, rno: r.rno, race_name: r.race_name || "", klass: r.klass, course: r.course, start_time: r.start_time || "",
      honmei: h ? { umaban: h.umaban, name: h.name, p_win_pct: pct(h.p_win), ninki: h.ninki ?? null, pos: posOf(order, h.umaban, r.result.top3) } : null,
      marks: marks.map((m) => ({ umaban: m.umaban, mark: m.mark, pos: posOf(order, m.umaban, r.result.top3) })),
      top3: r.result.top3.map((u) => { const x = horseByUma(r, u); return { umaban: u, name: x ? x.name : "", mark: x && x.mark ? x.mark : "", ai_rank: x ? x.ai_rank : null }; }),
      bets_hit: it ? it.site_bets_hit : null, bets_n: it ? it.site_bets_n : null,
      text: it ? it.text : null, model: it ? it.model : null, generated_at: it ? it.generated_at : null,
    };
  });
  // missing: 結果がある日は day + 未生成レース。結果が無い日（これから走る日）は preview だけ。
  const missing = upcoming
    ? (previewItem ? [] : ["preview"])
    : [...(dayItem ? [] : ["day"]), ...out.filter((x) => !x.text).map((x) => x.race_id)];
  return {
    date, places: day.places || [], n_races: out.length, n_total: (day.races || []).length, upcoming,
    day: dayItem, preview: previewItem, races: out, missing,
  };
}

/** RSS 2.0: 直近の総括と見どころ（KV にあるものだけ）。 */
export async function handleFeed(request, env) {
  const mf = await loadManifest(env, request.url);
  const kv = env.SOUDAN_KV;
  const dates = (mf && mf.dates ? mf.dates : []).slice(0, 12).map((d) => d.date);
  const got = await Promise.all(dates.flatMap((d) => [kvJson(kv, key(d, "day")), kvJson(kv, key(d, "preview"))]));
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmt = (d8) => `${+d8.slice(4, 6)}/${+d8.slice(6, 8)}`;
  const rfc = (iso) => { const d = new Date((iso || "2026-01-01 00:00").replace(" ", "T") + ":00+09:00"); return Number.isNaN(d.getTime()) ? new Date().toUTCString() : d.toUTCString(); };
  const items = [];
  got.forEach((it, i) => {
    if (!it) return;
    const d = dates[Math.floor(i / 2)];
    const isDay = it.race_id === "day";
    const s = it.stats || {};
    const title = isDay ? `${fmt(d)} 回顧 — ◎${s.honmei_win ?? "?"}勝／複勝圏${s.honmei_top3 ?? "?"}／${s.n_races ?? "?"}R` : `${fmt(d)} 今日の見どころ`;
    items.push({ title, link: `https://pycaliai.com/kaiko#${d}`, guid: `kaiko-${d}-${it.race_id}`, pub: rfc(it.generated_at), desc: it.text });
  });
  items.sort((a, b) => new Date(b.pub) - new Date(a.pub));
  const xml = `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
<title>PyCaLiAI 回顧・見どころ</title>
<link>https://pycaliai.com/kaiko</link>
<description>AI 競馬予想 PyCaLiAI の毎開催の見どころと、全レース回顧（負けた理由まで書く AI）。</description>
<language>ja</language>
${items.map((x) => `<item><title>${esc(x.title)}</title><link>${esc(x.link)}</link><guid isPermaLink="false">${esc(x.guid)}</guid><pubDate>${x.pub}</pubDate><description>${esc(x.desc)}</description></item>`).join("\n")}
</channel></rss>`;
  return new Response(xml, { status: 200, headers: { "content-type": "application/rss+xml; charset=utf-8", "cache-control": "public, max-age=600" } });
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
  if (!/^\d{8}$/.test(date) || !(rid === "day" || rid === "preview" || /^\d{16}$/.test(rid))) return json({ error: "bad date/race_id" }, 400, C);
  const r = await generateOne(env, request.url, date, rid, ctx);
  if (r.error) return json(r, r.code === "daily_limit" || r.code === "month_limit" ? 429 : r.code === "busy" ? 409 : r.code === "upstream" ? 502 : 404, C);
  return json({ item: r.item, created: r.created, stored: r.created ? r.stored : true }, 200, C);
}

/** cron: 結果待ちの開催日は「見どころ」を、直近 2 開催日（結果あり）は未生成の回顧を順に埋める。 */
export async function warmLatest(env, baseUrl, maxDates = 2) {
  const mf = await loadManifest(env, baseUrl);
  if (!mf) return { error: "no manifest" };
  const log = [];
  for (const d of (mf.dates || []).filter((x) => !x.has_results).slice(0, 2)) {
    const r = await generateOne(env, baseUrl, d.date, "preview", null);
    log.push(`${d.date}: preview ${r.error ? `NG(${r.code})` : r.created ? "made" : "exists"}`);
    if (r.error && (r.code === "daily_limit" || r.code === "month_limit" || r.code === "no_key")) return { log };
  }
  const dates = (mf.dates || []).filter((d) => d.has_results).slice(0, maxDates).map((d) => d.date);
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
