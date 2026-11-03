/* ============================================================
   PyCaLiAI 相談AI — 頭脳（決定的ツール群、Cloudflare Worker 版）
   assistant/tools.py の移植。違いは入力が公開済みの site/data/{date}.json
   （build_site.py scrub_public 済み: オッズ・払戻・EV・調教タイム・上り3F が無い）である点。
   → LLM が見られる数字は、サイトに既に表示している値だけ。ここで計算しない数字は存在しない。

   設計原則: LLM には数字を喋らせない。数字は必ずここが返す実値。LLM はそれを日本語にするだけ。
   依存: なし（純粋関数）。worker/tools.test.mjs で検査。
   ============================================================ */

export const CHAOS_PASS = 0.85;   // field_chaos_score これ以上 = 見送り寄り（assistant/tools.py と同値）
export const CHAOS_WARN = 0.60;   // これ以上 = 妙味次第
export const TRAP_NINKI = 10;     // 朝の人気順位これ以下（＝人気薄）で穴の罠警告（オッズの代替。オッズ値は公開していない）
export const TRAP_PWIN = 0.05;    // AI 勝率これ未満も人気薄扱い

const MARK2PRIOR = { "◎": "hon_top3_pct", "〇": "tai_top3_pct", "○": "tai_top3_pct", "▲": "san_top3_pct", "△": "del1_top3_pct" };
const VS_MARKET_JA = {
  over: "市場(人気)の評価がAIより高い＝人気先行・旨みは薄い",
  fair: "AIと市場の評価が近い",
  under: "AIの評価が市場(人気)より高い＝AI的には妙味側",
};
const UMAMI_JA = { S: "妙味S(最上位)", A: "妙味A", B: "妙味B", C: "妙味C(薄い)", "罠": "罠(人気薄で見かけ上おいしく見えるがAIは推さない)" };

export function pct(v, d = 1) { return v == null ? null : +(v * 100).toFixed(d); }

export function findRace(day, raceId) {
  return (day.races || []).find((r) => String(r.race_id) === String(raceId)) || null;
}

export function horseByUma(race, umaban) {
  return (race.horses || []).find((h) => +h.umaban === +umaban) || null;
}

/** 馬番(数字) か 馬名(部分一致、全角半角ゆらぎは NFKC) で 1 頭。見つからなければ null。 */
export function findHorse(race, key) {
  const k = String(key ?? "").normalize("NFKC").trim();
  const digits = k.replace(/[^0-9]/g, "");
  if (digits && /^[0-9]{1,2}$/.test(digits) && !/[^\d番号\s]/.test(k.replace(digits, ""))) {
    const h = horseByUma(race, +digits);
    if (h) return h;
  }
  const name = k.replace(/[\s番号]/g, "");
  if (!name) return null;
  const exact = (race.horses || []).find((h) => String(h.name).normalize("NFKC") === name);
  if (exact) return exact;
  const part = (race.horses || []).filter((h) => String(h.name).normalize("NFKC").includes(name));
  return part.length === 1 ? part[0] : null;
}

export function rosterLines(race) {
  return [...(race.horses || [])].sort((a, b) => a.umaban - b.umaban).map((h) =>
    `${h.umaban}番 ${h.name} [${h.mark || "－"}] AI${h.ai_rank}位 勝率${pct(h.p_win, 1)}% 複勝率${pct(h.p_plc, 1)}%` +
    (h.ninki ? ` 人気${h.ninki}位` : ""));
}

function runsJa(h) {
  return ((h.history || {}).runs || []).slice(0, 5).map((r) => ({
    何走前: r.n_ago,
    レース: r.race_name || null,
    着順: r.pos,
    人気: r.ninki,
    コース: `${r.place || ""}${r.td || ""}${r.dist || ""}${r.track ? `(${r.track})` : ""}`,
    脚質: r.style || null,
    間隔週: r.interval_weeks ?? null,
  }));
}

function formLine(h) {
  const r = ((h.history || {}).runs || [])[0];
  if (!r) return "（過去走データなし＝初出走か不明）";
  return `前走=${r.race_name || "?"} ${r.pos}着(${r.ninki ? `${r.ninki}人気・` : ""}${r.place || ""}${r.td || ""}${r.dist || ""})`;
}

function whyJa(h) {
  // SHAP 上位の「ラベル + 向き」だけ（value は scrub 済みでも数値を LLM に渡さない方針）
  return (h.why || []).slice(0, 4).map((w) => `${w.label}${(w.contrib ?? 0) >= 0 ? "(＋)" : "(－)"}`);
}

function advisorOf(race, umaban) {
  const a = ((race.cowork || {}).advisor || []).find((x) => +x.umaban === +umaban);
  return a ? { grade: a.grade ?? null, tag: a.tag ?? null, comment: a.comment ?? null } : null;
}

/** 1 頭の grounded ファクト。 */
export function describeHorse(race, umaban) {
  const h = horseByUma(race, umaban);
  if (!h) return { error: `${umaban}番は出走していません` };
  const cp = race.class_prior || null;
  const pk = MARK2PRIOR[h.mark];
  const lvl = h.level || {};
  const ps = h.ped_stats || {};
  const trap = (h.umami && h.umami.grade === "罠") || ((h.ninki ?? 0) >= TRAP_NINKI && (h.p_win ?? 0) < TRAP_PWIN);
  const res = race.result && race.result.order ? race.result.order[String(h.umaban)] ?? null : null;
  return {
    umaban: h.umaban, waku: h.waku, name: h.name, mark: h.mark || "無印",
    ai_rank: h.ai_rank, p_win_pct: pct(h.p_win), p_plc_pct: pct(h.p_plc), p_sho_pct: pct(h.p_sho),
    ninki_rank_morning: h.ninki ?? null,
    market_view: h.vs_market ? `${h.vs_market}: ${VS_MARKET_JA[h.vs_market] || ""}` : null,
    umami: h.umami ? `${UMAMI_JA[h.umami.grade] || h.umami.grade}${h.umami.side ? `・${h.umami.side}向き` : ""}` : null,
    longshot_trap: trap,
    horse_level: lvl.tier ? `${lvl.tier}(${lvl.score})` : null,
    sex_age: `${h.sex || ""}${h.age || ""}`, style: h.style || null, jockey: h.jockey || null, trainer: h.trainer || null,
    kinryo: h.kinryo ?? null, blinker: h.blinker || null,
    avg_pos_last5: (h.history || {}).avg_pos ?? null, best_pos_last5: (h.history || {}).best_pos ?? null,
    pos_trend: (h.history || {}).pos_trend ?? null,
    same_dist_ratio: (h.history || {}).same_dist_ratio ?? null,
    past_runs: runsJa(h),
    class_prior_top3_pct: cp && pk ? cp[pk] ?? null : null,
    sire: (h.pedigree || {}).sire || null,
    sire_fuku_pct: ps.sire ? ps.sire.fuku : null, sire_rank: ps.sire ? ps.sire.rank : null, cohort_fuku_pct: ps.baseline ?? null,
    ai_reasons: whyJa(h),
    advisor: advisorOf(race, h.umaban),
    result_pos: res,
    summary:
      `${h.name}（${h.mark || "無印"}・AI${h.ai_rank}位）: AI勝率${pct(h.p_win)}% / 複勝率${pct(h.p_plc)}%` +
      (h.ninki ? ` / 朝の人気${h.ninki}位` : "") +
      (h.vs_market ? ` / 市場比較=${h.vs_market}` : "") +
      (trap ? "。※人気薄＝『穴の罠』でAIは軸に推さない" : "") +
      `。${formLine(h)}` +
      (res ? `。【結果】${res}着` : ""),
  };
}

/** 2 頭比較。軸は『当たる信頼性』＝複勝率で推奨する（妙味を追わない）。 */
export function compareHorses(race, a, b) {
  const A = describeHorse(race, a), B = describeHorse(race, b);
  if (A.error) return A;
  if (B.error) return B;
  const pA = A.p_plc_pct ?? 0, pB = B.p_plc_pct ?? 0;
  const pick = pA >= pB ? A : B, other = pick === A ? B : A;
  const edge = +Math.abs(pA - pB).toFixed(1);
  const chaos = (race.confidence || {}).field_chaos_score ?? 0;
  const notes = [];
  if (other.longshot_trap) notes.push(`${other.umaban}番は人気薄の『穴の罠』側。見かけ上おいしく見えてもAIは軸に推さない（当AIは穴に構造的に弱い）`);
  if (pick.longshot_trap) notes.push(`推奨${pick.umaban}番自身も人気薄＝信頼度は割引いて見る`);
  if (edge < 2.0) notes.push(`複勝率は拮抗（${edge}pt差）。${other.umaban}番の市場比較は${(other.market_view || "").split(":")[0] || "不明"}`);
  if ((pick.market_view || "").startsWith("over")) notes.push(`推奨${pick.umaban}番は市場比較'over'＝人気先行。旨みは薄い`);
  if (chaos >= CHAOS_PASS) notes.push(`混戦度${chaos.toFixed(2)}＝荒れる鞍。軸の信頼度も割引いて見る`);
  const pair = pairProb(race, a, b);
  return {
    race: raceLabel(race), a: A, b: B, pick: pick.umaban, prob_edge_pct: edge, notes,
    pair: pair.found ? pair : null,
    recommendation:
      `軸向きは${pick.umaban}番 ${pick.name}。軸は『当たる信頼性』＝複勝率で選ぶ：${pick.p_plc_pct}%が${other.umaban}番(${other.p_plc_pct}%)を${edge}pt上回る。`,
  };
}

/** 2 頭のペア確率（サイトが表示している AI 算出の馬連率・ワイド率）。 */
export function pairProb(race, a, b) {
  const p = (race.pairs || []).find((x) => (+x.a === +a && +x.b === +b) || (+x.a === +b && +x.b === +a));
  if (!p) {
    return { found: false, a: +a, b: +b, note: `${a}-${b}はAIの上位${(race.pairs || []).length}ペアに入っていない＝ペア確率は上位ペアより低い（具体値は公開していない）` };
  }
  return { found: true, a: +a, b: +b, p_umaren_pct: pct(p.p_umaren), p_wide_pct: pct(p.p_wide),
    rank_in_race: (race.pairs || []).indexOf(p) + 1, n_pairs_listed: (race.pairs || []).length };
}

export function raceLabel(race) {
  return `${race.place} ${race.rno}R ${race.course} ${race.klass}${race.race_name ? ` ${race.race_name}` : ""} ${race.field_size}頭`;
}

/** 鞍全体の判断材料（見送り判断）。 */
export function raceAdvice(race) {
  const c = race.confidence || {};
  const chaos = c.field_chaos_score ?? 0;
  const verdict = chaos >= CHAOS_PASS ? "見送り寄り" : chaos >= CHAOS_WARN ? "妙味次第（厚張りしない）" : "勝負可";
  const top = [...(race.horses || [])].sort((x, y) => (x.ai_rank ?? 99) - (y.ai_rank ?? 99)).slice(0, 3)
    .map((h) => describeHorse(race, h.umaban));
  const j = race.judgment || {};
  const bets = ((race.tact && race.tact.bets) || (race.cowork && race.cowork.bets) || [])
    .map((b) => `${b.type} ${b.selection}`);
  const res = race.result && race.result.top3 ? race.result.top3 : null;
  return {
    race: raceLabel(race), verdict,
    chaos: +chaos.toFixed(2), top1_dominance: c.top1_dominance ?? null, ai_market_agreement: c.ai_market_agreement ?? null,
    site_judgment: j.headline ? { headline: j.headline, detail: j.detail || null, hardness: j.hardness || null, category: j.category || null } : null,
    top3_by_ai: top.map((h) => ({ umaban: h.umaban, name: h.name, mark: h.mark, p_win_pct: h.p_win_pct, p_plc_pct: h.p_plc_pct, market_view: h.market_view })),
    site_bets_shown: bets,
    result_top3: res,
    summary: `【${raceLabel(race)}】判定: ${verdict}（混戦度${chaos.toFixed(2)}、本命優位${c.top1_dominance ?? "?"}）` +
      (j.headline ? `。サイトの判定=${j.headline}` : "") +
      (res ? `。【結果】${res.join("-")}` : ""),
  };
}

/** コースの枠・脚質傾向（サイトのコースタブと同じ集計: 2013-25）。 */
export function courseBias(day, race) {
  const key = `${race.place}|${race.course}`;
  const cs = (day.courses || {})[key];
  if (!cs || !cs.waku) return { error: `${key} のコース集計データがありません` };
  const w = Object.fromEntries(cs.waku.map((x) => [String(x.label), x]));
  const bands = { "内枠(1-2)": [1, 2], "中枠(3-6)": [3, 4, 5, 6], "外枠(7-8)": [7, 8] };
  const out = {};
  for (const [name, frames] of Object.entries(bands)) {
    const rows = frames.map((f) => w[String(f)]).filter(Boolean);
    const n = rows.reduce((s, r) => s + (r.n || 0), 0);
    if (!n) continue;
    const fuku = rows.reduce((s, r) => s + (r.fuku || 0) * (r.n || 0), 0) / n;
    const win = rows.reduce((s, r) => s + (r.win || 0) * (r.n || 0), 0) / n;
    out[name] = { fuku_pct: +fuku.toFixed(1), win_pct: +win.toFixed(1), n };
  }
  const best = Object.keys(out).sort((x, y) => out[y].fuku_pct - out[x].fuku_pct)[0];
  const spread = best ? out[best].fuku_pct - Math.min(...Object.values(out).map((v) => v.fuku_pct)) : 0;
  const marked = (race.horses || []).filter((h) => h.mark).map((h) => `${h.mark}${h.umaban}番=${h.waku}枠/${h.style || "?"}`);
  const kyaku = (cs.kyaku || []).map((k) => ({ 脚質: k.label, fuku_pct: k.fuku, win_pct: k.win, n: k.n }));
  return {
    course: key, n_races: cs.n_races, bands: out,
    verdict: best ? (spread >= 2 ? `${best}が複勝率最高＝やや有利（差${spread.toFixed(1)}pt）` : `枠による差は小さい（最大差${spread.toFixed(1)}pt）`) : "判定不能",
    kyaku, marked_horses: marked,
    summary: `${key}（${cs.n_races}R集計）枠: ` + Object.entries(out).map(([b, v]) => `${b} 複勝${v.fuku_pct}%`).join(" / ") +
      (best ? ` → ${spread >= 2 ? `${best}が有利` : "枠差は小さい"}` : "") +
      (kyaku.length ? `。脚質: ${kyaku.map((k) => `${k.脚質}${k.fuku_pct}%`).join("/")}` : ""),
  };
}

/** LLM のツール呼び出しを橋渡し。戻り値は JSON 化可能な実値。 */
export function dispatch(day, race, name, args) {
  const a = args || {};
  try {
    switch (name) {
      case "horse_info": {
        const h = findHorse(race, a.umaban_or_name);
        if (!h) return { error: `'${a.umaban_or_name}' に該当する馬が見つかりません`, roster: rosterLines(race) };
        return describeHorse(race, h.umaban);
      }
      case "compare_horses": {
        const ha = findHorse(race, a.a), hb = findHorse(race, a.b);
        if (!ha || !hb) return { error: `馬が見つかりません: ${!ha ? a.a : a.b}`, roster: rosterLines(race) };
        return compareHorses(race, ha.umaban, hb.umaban);
      }
      case "pair_prob": {
        const ha = findHorse(race, a.a), hb = findHorse(race, a.b);
        if (!ha || !hb) return { error: `馬が見つかりません: ${!ha ? a.a : a.b}`, roster: rosterLines(race) };
        return pairProb(race, ha.umaban, hb.umaban);
      }
      case "race_advice": return raceAdvice(race);
      case "course_bias": return courseBias(day, race);
      default: return { error: `unknown tool: ${name}` };
    }
  } catch (e) {
    return { error: String(e && e.message || e) };
  }
}
