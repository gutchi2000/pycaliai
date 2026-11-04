/* worker/util.js — /api/soudan と /api/kaiko で共用する小物（HTTP・KV・日付・料金） */

export const ALLOWED_ORIGINS = new Set([
  "https://pycaliai.com", "https://www.pycaliai.com",
  "https://gutchi15300-pycaliai-umami.hf.space",
]);
const LOCAL_ORIGIN = /^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/;

// USD per 1M tokens（Claude API 一次料金、2026-10 時点）。推定支出の上限判定にだけ使う。
export const PRICE = {
  "claude-sonnet-5-5": { in: 2.0, out: 10.0 },
  "claude-haiku-5-5": { in: 0.10, out: 0.50 },
};

export function json(obj, status = 200, extra = {}) {
  return new Response(JSON.stringify(obj), {
    status, headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store", ...extra },
  });
}
export function cors(origin) {
  return origin ? { "access-control-allow-origin": origin, "access-control-allow-methods": "GET, POST, OPTIONS",
    "access-control-allow-headers": "content-type", "access-control-max-age": "600", vary: "origin" } : {};
}
export function originOk(origin) {
  return !!origin && (ALLOWED_ORIGINS.has(origin) || LOCAL_ORIGIN.test(origin));
}
export function jstParts(d = new Date()) {
  const t = new Date(d.getTime() + 9 * 3600 * 1000);
  const y = t.getUTCFullYear(), m = String(t.getUTCMonth() + 1).padStart(2, "0"), day = String(t.getUTCDate()).padStart(2, "0");
  return { ymd: `${y}${m}${day}`, ym: `${y}${m}`, iso: t.toISOString().slice(0, 16).replace("T", " ") };
}
export async function sha256Hex(s) {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s));
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");
}
export const num = (v, dflt) => (v == null || v === "" || Number.isNaN(+v) ? dflt : +v);

export async function kvNum(kv, key) {
  try { const v = await kv.get(key); return v ? +v || 0 : 0; } catch { return 0; }
}
export async function kvSet(kv, key, val, ttlSec) {
  try { await kv.put(key, String(val), ttlSec ? { expirationTtl: ttlSec } : {}); return true; }
  catch (e) { console.error("kv put failed", key, String(e && e.message || e)); return false; }   // 結果整合・失敗は許容（ログには残す）
}
export async function kvJson(kv, key) {
  try { const v = await kv.get(key); return v ? JSON.parse(v) : null; } catch { return null; }
}
/** token 数 × USD/1M ＝ micro-USD。月の推定支出 `spend:{ym}` に加算（soudan / kaiko 共通の財布）。 */
export function microUsd(model, inTok, outTok) {
  const p = PRICE[model] || PRICE["claude-haiku-5-5"];
  return Math.round(inTok * p.in + outTok * p.out);
}

const DAY_CACHE_MS = 10 * 60 * 1000;
const dayCache = new Map();    // date -> {at, day}
/** 公開済み site/data/{date}.json（scrub_public 済み）を ASSETS から読む。Worker メモリに 10 分キャッシュ。 */
export async function loadDay(env, baseUrl, date) {
  const hit = dayCache.get(date);
  if (hit && Date.now() - hit.at < DAY_CACHE_MS) return hit.day;
  const res = await env.ASSETS.fetch(new Request(new URL(`/data/${date}.json`, baseUrl), { method: "GET" }));
  if (!res.ok) return null;
  const day = await res.json();
  dayCache.set(date, { at: Date.now(), day });
  return day;
}
let rcCache = null;   // {at, rc}
/** AI の通信簿（site/data/report_card.json）。無ければ null。10 分キャッシュ。 */
export async function loadReportCard(env, baseUrl) {
  if (rcCache && Date.now() - rcCache.at < DAY_CACHE_MS) return rcCache.rc;
  try {
    const res = await env.ASSETS.fetch(new Request(new URL("/data/report_card.json", baseUrl), { method: "GET" }));
    const rc = res.ok ? await res.json() : null;
    rcCache = { at: Date.now(), rc };
    return rc;
  } catch { return null; }
}
export async function loadManifest(env, baseUrl) {
  const res = await env.ASSETS.fetch(new Request(new URL("/data/manifest.json", baseUrl), { method: "GET" }));
  return res.ok ? res.json() : null;
}
