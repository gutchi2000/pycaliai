/* ============================================================
   pycaliai.com Cloudflare Worker
     - 静的サイト (site/) は ASSETS binding で配信（従来どおり）
     - POST /api/soudan : 相談AI（Claude API 中継。鍵は secret ANTHROPIC_API_KEY）

   相談AI の守り（2026-11-03 ユーザー承認: Sonnet、全体 1 日 150 問まで、超えたら Haiku）:
     1. Origin 許可リスト（pycaliai.com / HF Space / localhost）以外は 403
     2. IP ごと 1 日 PER_IP_DAY 問（KV カウンタ、IP は日替わり salt で SHA-256 ハッシュ化、生 IP は保存しない）
     3. 全体 1 日 DAILY_SONNET 問までは Sonnet 5.5、以降 DAILY_HARD 問までは Haiku 5.5、超えたら 429
     4. 月の推定支出が MONTH_USD_CAP を超えたら 429（クレジット枯渇前に自分で止める）
     5. 数字の供給元は公開済み site/data/{date}.json だけ（scrub_public 済み＝オッズ・EV が存在しない）
   日付境界は JST。カウンタは KV（結果整合なので数問の誤差は許容）。
   ============================================================ */
import Anthropic from "@anthropic-ai/sdk";
import { SYSTEM, TOOLS } from "./prompt.js";
import { dispatch, findRace, raceLabel, rosterLines } from "./tools.js";

const ALLOWED_ORIGINS = new Set([
  "https://pycaliai.com", "https://www.pycaliai.com",
  "https://gutchi15300-pycaliai-umami.hf.space",
]);
const LOCAL_ORIGIN = /^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/;

// USD per 1M tokens（Claude API 一次料金、2026-10 時点）。推定支出の上限判定にだけ使う。
const PRICE = {
  "claude-sonnet-5-5": { in: 2.0, out: 10.0 },
  "claude-haiku-5-5": { in: 0.10, out: 0.50 },
};
const MAX_TURNS = 10;          // クライアントから受け取る履歴の上限（user+assistant 合計）
const MAX_CHARS = 400;         // 1 発話の上限
const MAX_TOOL_ROUNDS = 5;
const DAY_CACHE_MS = 10 * 60 * 1000;
const dayCache = new Map();    // date -> {at, day}

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    if (url.pathname === "/api/soudan") return soudan(request, env, ctx);
    if (url.pathname === "/api/soudan/health") return health(env);
    if (url.pathname.startsWith("/api/")) return json({ error: "not found" }, 404);
    return env.ASSETS.fetch(request);
  },
};

/* ---------------- helpers ---------------- */
function json(obj, status = 200, extra = {}) {
  return new Response(JSON.stringify(obj), {
    status, headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store", ...extra },
  });
}
function cors(origin) {
  return origin ? { "access-control-allow-origin": origin, "access-control-allow-methods": "POST, OPTIONS",
    "access-control-allow-headers": "content-type", "access-control-max-age": "600", vary: "origin" } : {};
}
function originOk(origin) {
  return !!origin && (ALLOWED_ORIGINS.has(origin) || LOCAL_ORIGIN.test(origin));
}
function jstParts(d = new Date()) {
  const t = new Date(d.getTime() + 9 * 3600 * 1000);
  const y = t.getUTCFullYear(), m = String(t.getUTCMonth() + 1).padStart(2, "0"), day = String(t.getUTCDate()).padStart(2, "0");
  return { ymd: `${y}${m}${day}`, ym: `${y}${m}` };
}
async function sha256Hex(s) {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s));
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");
}
const num = (v, dflt) => (v == null || v === "" || Number.isNaN(+v) ? dflt : +v);

async function kvNum(kv, key) {
  try { const v = await kv.get(key); return v ? +v || 0 : 0; } catch { return 0; }
}
async function kvSet(kv, key, val, ttlSec) {
  try { await kv.put(key, String(val), { expirationTtl: ttlSec }); } catch { /* 結果整合・失敗は許容 */ }
}

async function loadDay(env, request, date) {
  const hit = dayCache.get(date);
  if (hit && Date.now() - hit.at < DAY_CACHE_MS) return hit.day;
  const res = await env.ASSETS.fetch(new Request(new URL(`/data/${date}.json`, request.url), { method: "GET" }));
  if (!res.ok) return null;
  const day = await res.json();
  dayCache.set(date, { at: Date.now(), day });
  return day;
}

/* GET /api/soudan/health: 秘密の値は返さず「設定されているか・形が正しいか」と本日の消費だけ（運用確認用）。 */
async function health(env) {
  const key = String(env.ANTHROPIC_API_KEY || "");
  const { ymd, ym } = jstParts();
  const kv = env.SOUDAN_KV;
  const [g, spend] = kv ? await Promise.all([kvNum(kv, `g:${ymd}`), kvNum(kv, `spend:${ym}`)]) : [0, 0];
  return json({
    ok: key.trim().length > 20, key_set: key.length > 0, key_len: key.length, key_prefix_ok: key.startsWith("sk-ant-"),
    kv: !!kv, today_jst: ymd, questions_today: g, est_spend_month_usd: +(spend / 1e6).toFixed(3),
    model: env.SOUDAN_MODEL || "claude-sonnet-5-5", daily_sonnet: num(env.SOUDAN_DAILY_SONNET, 150), daily_hard: num(env.SOUDAN_DAILY_HARD, 600),
  });
}

/* ---------------- /api/soudan ---------------- */
async function soudan(request, env, ctx) {
  const origin = request.headers.get("origin") || "";
  const ok = originOk(origin);
  if (request.method === "OPTIONS") return new Response(null, { status: ok ? 204 : 403, headers: ok ? cors(origin) : {} });
  if (request.method !== "POST") return json({ error: "POST only" }, 405);
  if (!ok) return json({ error: "forbidden origin" }, 403);
  const C = cors(origin);
  if (!String(env.ANTHROPIC_API_KEY || "").trim()) return json({ error: "相談AIは準備中です（鍵未設定）", code: "no_key" }, 503, C);

  // --- 入力検証 ---
  let body;
  try { body = await request.json(); } catch { return json({ error: "bad json" }, 400, C); }
  const date = String(body.date || "");
  const raceId = String(body.race_id || "");
  const msgs = Array.isArray(body.messages) ? body.messages : [];
  if (!/^\d{8}$/.test(date) || !/^\d{16}$/.test(raceId)) return json({ error: "bad date/race_id" }, 400, C);
  if (!msgs.length || msgs.length > MAX_TURNS) return json({ error: "bad messages" }, 400, C);
  for (const m of msgs) {
    if (!m || (m.role !== "user" && m.role !== "assistant") || typeof m.content !== "string" || !m.content.trim() || m.content.length > MAX_CHARS)
      return json({ error: `各発話は ${MAX_CHARS} 文字以内の文字列`, code: "bad_message" }, 400, C);
  }
  if (msgs[msgs.length - 1].role !== "user") return json({ error: "last message must be user" }, 400, C);

  // --- レート制限（KV） ---
  const kv = env.SOUDAN_KV;
  const { ymd, ym } = jstParts();
  const perIpDay = num(env.SOUDAN_PER_IP_DAY, 30);
  const dailySonnet = num(env.SOUDAN_DAILY_SONNET, 150);
  const dailyHard = num(env.SOUDAN_DAILY_HARD, 600);
  const monthCap = num(env.SOUDAN_MONTH_USD_CAP, 90);
  const ip = request.headers.get("cf-connecting-ip") || "0.0.0.0";
  const ipKey = `ip:${ymd}:${(await sha256Hex(`${ymd}|${ip}`)).slice(0, 24)}`;
  const [ipCount, gCount, spendMicro] = kv
    ? await Promise.all([kvNum(kv, ipKey), kvNum(kv, `g:${ymd}`), kvNum(kv, `spend:${ym}`)])
    : [0, 0, 0];
  if (ipCount >= perIpDay) return json({ error: `本日の相談回数（${perIpDay}問）に達しました。また明日どうぞ。`, code: "ip_limit", remaining: 0 }, 429, C);
  if (gCount >= dailyHard) return json({ error: "本日の相談AIの枠（全員分）は終了しました。明日 0 時（JST）に再開します。", code: "daily_limit", remaining: perIpDay - ipCount }, 429, C);
  if (spendMicro / 1e6 >= monthCap) return json({ error: "今月の相談AIの枠は終了しました。来月 1 日に再開します。", code: "month_limit" }, 429, C);
  const model = gCount < dailySonnet ? (env.SOUDAN_MODEL || "claude-sonnet-5-5") : (env.SOUDAN_FALLBACK_MODEL || "claude-haiku-5-5");

  // --- レース文脈（公開 JSON のみ） ---
  const day = await loadDay(env, request, date);
  if (!day) return json({ error: "その日のデータがありません", code: "no_day" }, 404, C);
  const race = findRace(day, raceId);
  if (!race) return json({ error: "そのレースが見つかりません", code: "no_race" }, 404, C);
  const finished = !!(race.result && race.result.top3);
  const system = [
    { type: "text", text: SYSTEM, cache_control: { type: "ephemeral" } },
    { type: "text", text:
      `【今見ているレース】${raceLabel(race)}（${date.slice(0, 4)}/${+date.slice(4, 6)}/${+date.slice(6, 8)}）` +
      (finished ? "。このレースは終了済み（結果あり）。" : "。発走前または結果未反映。") +
      `\n出走馬一覧（馬番 馬名 [印] AI順位 勝率 複勝率 朝の人気）:\n` + rosterLines(race).join("\n") },
  ];

  // --- Claude（手動ツールループ、最大 MAX_TOOL_ROUNDS 往復） ---
  const client = new Anthropic({ apiKey: env.ANTHROPIC_API_KEY, maxRetries: 1, timeout: 45_000 });
  const conv = msgs.map((m) => ({ role: m.role, content: m.content }));
  let inTok = 0, outTok = 0, answer = "", stop = "", toolsUsed = [];
  try {
    for (let round = 0; round < MAX_TOOL_ROUNDS; round++) {
      const res = await client.messages.create({
        model, max_tokens: 700, system, tools: TOOLS, messages: conv,
        output_config: { effort: "low" },
      });
      inTok += res.usage?.input_tokens || 0; outTok += res.usage?.output_tokens || 0;
      stop = res.stop_reason;
      const uses = res.content.filter((b) => b.type === "tool_use");
      if (stop === "tool_use" && uses.length) {
        conv.push({ role: "assistant", content: res.content });
        conv.push({ role: "user", content: uses.map((u) => {
          toolsUsed.push(u.name);
          return { type: "tool_result", tool_use_id: u.id, content: JSON.stringify(dispatch(day, race, u.name, u.input)) };
        }) });
        continue;
      }
      if (stop === "pause_turn") { conv.push({ role: "assistant", content: res.content }); continue; }
      answer = res.content.filter((b) => b.type === "text").map((b) => b.text).join("").trim();
      break;
    }
  } catch (e) {
    const status = e && e.status;
    const msg = status === 429 ? "相談AIが混み合っています。少し待ってからもう一度どうぞ。"
      : status === 401 || status === 403 ? "相談AIの認証に失敗しました（運営側の設定）。"
      : "相談AIが応答できませんでした。時間をおいて再度お試しください。";
    return json({ error: msg, code: "upstream", status: status || null }, 502, C);
  }
  if (stop === "refusal") answer = "この質問には答えられません。レースの評価についての質問をどうぞ。";
  if (!answer) answer = stop === "max_tokens" ? "（回答が長くなりすぎました。質問を分けてもう一度どうぞ）" : "（うまく答えを作れませんでした。言い方を変えてもう一度どうぞ）";

  // --- カウンタ更新（レスポンス後） ---
  const price = PRICE[model] || PRICE["claude-haiku-5-5"];
  const micro = Math.round((inTok * price.in + outTok * price.out));   // tokens × USD/1M = micro-USD
  if (kv) {
    ctx.waitUntil(Promise.all([
      kvSet(kv, ipKey, ipCount + 1, 2 * 86400),
      kvSet(kv, `g:${ymd}`, gCount + 1, 2 * 86400),
      kvSet(kv, `spend:${ym}`, spendMicro + micro, 40 * 86400),
    ]));
  }
  return json({
    answer, model: model.includes("haiku") ? "haiku" : "sonnet", finished,
    remaining: Math.max(0, perIpDay - ipCount - 1),
    usage: { input_tokens: inTok, output_tokens: outTok, est_usd: +(micro / 1e6).toFixed(4), tools: toolsUsed },
  }, 200, C);
}
