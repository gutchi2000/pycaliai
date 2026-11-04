/* worker/horse.js — 馬ページの OG メタ差し込みと /h/{id} 短縮 URL、相談AI の horse_kaiko ツール (2026-11-04)
 *
 * 馬ページ (site/horse.html?name=馬名) は SPA なので共有すると表紙の OG しか出ない。9,339 頭ぶんの静的 HTML は
 * Workers assets の 20,000 ファイル上限に当たるため、Worker が /horse?name= の応答 (assets の horse.html) に
 * HTMLRewriter でその馬の title / description / og:* / canonical を差し込む。/h/{id16} は馬ページへの 302。
 * データは公開済み site/data/h/{id[:2]}.json (集計値のみ、オッズ等は元から無い)。id は site/js/horse.js と同じ
 * FNV-1a 32bit ×2 (UTF-8 バイト列)。wrangler.jsonc の run_worker_first に /horse, /horse.html, /h/* を登録。 */

function fnv(s, seed) {
  let h = seed >>> 0;
  for (const b of new TextEncoder().encode(s)) h = Math.imul(h ^ b, 16777619) >>> 0;
  return h;
}
export const horseId = (name) => fnv(name, 0x811c9dc5).toString(16).padStart(8, "0") + fnv(name, 0x01000193).toString(16).padStart(8, "0");

const ORIGIN = "https://pycaliai.com";

async function loadShard(env, url, id2) {
  try {
    const r = await env.ASSETS.fetch(new Request(new URL(`/data/h/${id2}.json`, url), { method: "GET" }));
    if (!r.ok) return null;
    return await r.json();
  } catch {
    return null;
  }
}

export async function findHorseRecord(env, url, { name, id }) {
  const hid = id || (name ? horseId(name) : null);
  if (!hid || !/^[0-9a-f]{16}$/.test(hid)) return null;
  const shard = await loadShard(env, url, hid.slice(0, 2));
  return shard && shard[hid] ? shard[hid] : null;
}

const pct = (v) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);

export function horseDescription(h) {
  const st = h.stats || {};
  const who = [h.sex, h.age != null ? `${h.age}歳` : "", h.trainer ? `${h.trainer}厩舎` : ""].filter(Boolean).join(" ");
  const parts = [`2026 年 ${st.n || 0} 走（結果あり ${st.n_settled || 0}）`];
  if (st.hon_n) parts.push(`AI◎ ${st.hon_n} 回で勝ち ${st.hon_win || 0}・3着内 ${st.hon_top3 || 0}`);
  if (st.marked_n) parts.push(`印あり ${st.marked_n} 走で 3着内 ${st.marked_top3 || 0}`);
  if (st.avg_pwin != null) parts.push(`平均 AI 勝率 ${pct(st.avg_pwin)}`);
  if (st.best_ai_rank) parts.push(`最高 AI 順位 ${st.best_ai_rank} 位`);
  return `${h.name}${who ? `（${who}）` : ""}: ${parts.join("、")}。PyCaLiAI の AI 評価と結果の履歴。`;
}

/** GET /horse?name= | /horse.html?name= | /h/{id16} */
export async function horsePage(request, env) {
  const url = new URL(request.url);
  const m = url.pathname.match(/^\/h\/([0-9a-f]{16})$/);
  const name = (url.searchParams.get("name") || "").trim();
  const asset = () => env.ASSETS.fetch(new Request(new URL("/horse.html", url), { method: "GET", headers: request.headers }));
  if (m) {
    const h = await findHorseRecord(env, url, { id: m[1] });
    if (!h) return Response.redirect(new URL("/horse.html", url).toString(), 302);
    return Response.redirect(new URL(`/horse?name=${encodeURIComponent(h.name)}`, url).toString(), 302);
  }
  const base = await asset();
  if (!name || !base.ok || !/text\/html/.test(base.headers.get("content-type") || "")) return base;
  const h = await findHorseRecord(env, url, { name });
  if (!h) return base;
  const title = `${h.name} — AI 評価の履歴 | PyCaLiAI`;
  const desc = horseDescription(h);
  const canon = `${ORIGIN}/horse?name=${encodeURIComponent(h.name)}`;
  const setContent = (v) => ({ element(e) { e.setAttribute("content", v); } });
  return new HTMLRewriter()
    .on("title", { element(e) { e.setInnerContent(title); } })
    .on('meta[name="description"]', setContent(desc))
    .on('meta[property="og:title"]', setContent(`${h.name} — AI 評価の履歴`))
    .on('meta[property="og:description"]', setContent(desc))
    .on('meta[property="og:url"]', setContent(canon))
    .on('link[rel="canonical"]', { element(e) { e.setAttribute("href", canon); } })
    .transform(base);
}

/** 相談AI ツール horse_kaiko: 今見ているレースの馬について、過去の回顧記事 (KV) からその馬に触れた部分を返す。 */
export async function horseKaiko(env, url, race, horse, maxRuns = 3) {
  const h = await findHorseRecord(env, url, { name: horse.name });
  if (!h) return { horse: horse.name, lessons: [], note: "馬ページのデータに無い馬 (2026 年に出走記録なし)" };
  const runs = (h.runs || []).filter((r) => r.has_result && r.rid && r.rid !== race.race_id).slice(0, maxRuns);
  const lessons = [];
  for (const r of runs) {
    let text = null;
    try {
      const raw = await env.SOUDAN_KV.get(`kaiko:v2:${r.date}:${r.rid}`);
      if (raw) text = JSON.parse(raw).text || null;
    } catch { /* KV 不調は無視 */ }
    lessons.push({
      date: r.date, race: `${r.place}${r.rno}R ${r.klass || ""} ${r.course || ""}`.trim(),
      mark: r.mark || null, ai_rank: r.ai_rank, p_win_pct: r.p_win != null ? +(r.p_win * 100).toFixed(1) : null,
      ninki: r.ninki, pos: r.pos, field_size: r.field_size,
      kaiko: text ? excerpt(text, horse.name) : null,
      kaiko_url: `${ORIGIN}/kaiko#${r.date}`,
    });
  }
  return { horse: horse.name, stats_2026: h.stats || null, lessons,
           note: "kaiko は結果が出た後に AI が書いた回顧記事の、この馬に触れた段落 (無ければ null)。数字は公開 JSON の値。" };
}

function excerpt(text, name, limit = 600) {
  const paras = String(text).split(/\n+/).filter((p) => p.trim());
  const hit = paras.filter((p) => p.includes(name));
  const pick = (hit.length ? hit : paras.slice(0, 2)).join("\n");
  return pick.length > limit ? pick.slice(0, limit - 1) + "…" : pick;
}
