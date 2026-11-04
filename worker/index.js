/* ============================================================
   pycaliai.com Cloudflare Worker
     - 静的サイト (site/) は ASSETS binding で配信（従来どおり）
     - POST /api/soudan            : 相談AI（Claude API 中継。鍵は secret ANTHROPIC_API_KEY）
     - GET  /api/soudan/health     : 鍵の有無・本日の消費（値は返さない）
     - GET  /api/kaiko?date=       : 回顧記事一覧（worker/kaiko.js）
     - POST /api/kaiko/gen         : 回顧記事を 1 件生成（KV に永続）
     - scheduled (cron)            : 月曜 03:00 JST に直近開催の回顧を埋める

   相談AI の守り（2026-11-03 ユーザー承認: Sonnet、全体 1 日 150 問まで、超えたら Haiku）:
     1. Origin 許可リスト（pycaliai.com / HF Space / localhost）以外は 403
     2. IP ごと 1 日 PER_IP_DAY 問（KV カウンタ、IP は日替わり salt で SHA-256 ハッシュ化、生 IP は保存しない）
     3. 全体 1 日 DAILY_SONNET 問までは Sonnet 5.5、以降 DAILY_HARD 問までは Haiku 5.5、超えたら 429
     4. 月の推定支出が MONTH_USD_CAP を超えたら 429（クレジット枯渇前に自分で止める。回顧記事と同じ財布）
     5. 数字の供給元は公開済み site/data/{date}.json だけ（scrub_public 済み＝オッズ・EV が存在しない）
   日付境界は JST。カウンタは KV（結果整合なので数問の誤差は許容）。
   ============================================================ */
import Anthropic from "@anthropic-ai/sdk";
import { SYSTEM, TOOLS } from "./prompt.js";
import { dispatch, findRace, raceLabel, rosterLines } from "./tools.js";
import { handleGen, handleList, warmLatest } from "./kaiko.js";
import { cors, json, jstParts, kvNum, kvSet, loadDay, microUsd, num, originOk, sha256Hex } from "./util.js";

const MAX_TURNS = 10;          // クライアントから受け取る履歴の上限（user+assistant 合計）
const MAX_CHARS = 400;         // 1 発話の上限
const MAX_TOOL_ROUNDS = 5;

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    if (url.pathname === "/api/soudan") return soudan(request, env, ctx);
    if (url.pathname === "/api/soudan/health") return health(env);
    if (url.pathname === "/api/kaiko") return handleList(request, env);
    if (url.pathname === "/api/kaiko/gen") return handleGen(request, env, ctx);
    if (url.pathname.startsWith("/api/")) return json({ error: "not found" }, 404);
    return env.ASSETS.fetch(request);
  },
  async scheduled(event, env, ctx) {
    // cron は wrangler.jsonc の triggers.crons。ASSETS の URL は本番ドメインで解決する。
    ctx.waitUntil(warmLatest(env, "https://pycaliai.com/", 2).then((r) => console.log("kaiko warm", JSON.stringify(r))));
  },
};

/* GET /api/soudan/health: 秘密の値は返さず「設定されているか・形が正しいか」と本日の消費だけ（運用確認用）。 */
async function health(env) {
  const key = String(env.ANTHROPIC_API_KEY || "");
  const { ymd, ym } = jstParts();
  const kv = env.SOUDAN_KV;
  const [g, gk, spend] = kv ? await Promise.all([kvNum(kv, `g:${ymd}`), kvNum(kv, `g_kaiko:${ymd}`), kvNum(kv, `spend:${ym}`)]) : [0, 0, 0];
  return json({
    ok: key.trim().length > 20, key_set: key.length > 0, key_len: key.length, key_prefix_ok: key.startsWith("sk-ant-"),
    kv: !!kv, today_jst: ymd, questions_today: g, kaiko_generated_today: gk, est_spend_month_usd: +(spend / 1e6).toFixed(3),
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
  const day = await loadDay(env, request.url, date);
  if (!day) return json({ error: "その日のデータがありません", code: "no_day" }, 404, C);
  const race = findRace(day, raceId);
  if (!race) return json({ error: "そのレースが見つかりません", code: "no_race" }, 404, C);
  const finished = !!(race.result && race.result.top3);
  const system = [
    { type: "text", text: SYSTEM, cache_control: { type: "ephemeral" } },
    { type: "text", text:
      `【今見ているレース】${raceLabel(race)}（${date.slice(0, 4)}/${+date.slice(4, 6)}/${+date.slice(6, 8)}）` +
      (finished ? "。このレースは終了済み（結果あり）。" : "。発走前または結果未反映。") +
      `\n出走馬一覧（馬番 馬名 [印] AI順位 勝率 複勝率 朝の人気）:\n` + rosterLines(race).join("\n") +
      `\n\n同じ日の他レース（${(day.races || []).length}R: ${(day.places || []).join("・")}）の横断的な質問は day_races を呼ぶ。` },
  ];

  // --- Claude（手動ツールループ、最大 MAX_TOOL_ROUNDS 往復） ---
  // trim: パイプ経由で secret を入れたときの末尾改行を無害化
  const client = new Anthropic({ apiKey: String(env.ANTHROPIC_API_KEY).trim(), maxRetries: 1, timeout: 45_000 });
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
  const micro = microUsd(model, inTok, outTok);
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
