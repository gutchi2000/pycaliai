"""
PyCaLiAI 対話アシスタント — 頭脳（決定的ツール群）

設計原則（最重要）:
  LLM には数字を喋らせない。数字は必ずこのモジュールが bundle.json /
  監査 JSON から「計算した実値」を返す。LLM はその出力を日本語に翻訳するだけ。
  → これにより「嘘をつかないアシスタント」になり、かつ安いローカル7Bで足りる。

依存: 標準ライブラリ json のみ（torch / lightgbm 不要）。
入力: reports/cowork_input/{date}_bundle.json（export_weekly_marks.py が生成）
"""
from __future__ import annotations
import functools
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BUNDLE_DIR = REPO / "reports" / "cowork_input"


# ---------------------------------------------------------------- ロード
def load_bundle(date_or_path: str) -> dict:
    """'20260531' でも フルパスでも受ける。"""
    p = Path(date_or_path)
    if not p.exists():
        p = BUNDLE_DIR / f"{date_or_path}_bundle.json"
    if not p.exists():
        raise FileNotFoundError(f"bundle が見つからない: {date_or_path}")
    return json.loads(p.read_text(encoding="utf-8"))


def find_race(bundle: dict, ref) -> dict:
    """ref = index(int) / race_id(str) / '東京'などの場所名 で1鞍を引く。"""
    races = bundle["races"]
    if isinstance(ref, int):
        return races[ref]
    for r in races:
        if r.get("race_id") == ref:
            return r
    hits = [r for r in races if ref in (r["race_meta"].get("place", ""))]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise ValueError(f"'{ref}' で {len(hits)} 鞍ヒット。race_id か index で指定して。")
    raise KeyError(f"該当レースなし: {ref}")


def _horse(race: dict, umaban: int) -> dict:
    for h in race["horses"]:
        if h["umaban"] == umaban:
            return h
    raise KeyError(f"{umaban}番が居ない")


# ---------------------------------------------------------------- 計算
def _fuku_mid(h: dict):
    lo, hi = h.get("fuku_odds_low"), h.get("fuku_odds_high")
    return (lo + hi) / 2 if lo and hi else None


def ev_win(h: dict):
    """単勝期待値（1.0=収支トントン、控除前）。"""
    return h["p_win"] * h["tansho_odds"] if h.get("tansho_odds") else None


def ev_plc(h: dict):
    """複勝期待値（複勝オッズ中央値ベース）。"""
    fm = _fuku_mid(h)
    return h["p_plc"] * fm if fm else None


def _ev_label(ev) -> str:
    """EV を『プラス/マイナス』明記の文字列に（LLM の誤読防止）。"""
    if ev is None:
        return "不明"
    if ev >= 1.0:
        return f"{ev:.2f}（プラス=期待値1.0超）"
    return f"{ev:.2f}（1.0未満=期待値マイナス・控除に負ける帯）"


_MARK2PRIOR = {  # 印 → class_prior の複勝率キー
    "◎": "hon_top3_pct", "〇": "tai_top3_pct", "○": "tai_top3_pct",
    "▲": "san_top3_pct", "△": "del1_top3_pct",
}


# 決手の正規値 (新旧 bundle schema 判別用。旧: style=レース名/weight_change=決手)
_STYLE_VALUES = {"逃げ", "先行", "中団", "差し", "ﾏｸﾘ", "マクリ", "後方", "追込"}


def _run_style(r: dict):
    """脚質(決手)を新旧 schema 両対応で取り出す。"""
    for k in ("style", "weight_change"):
        v = str(r.get(k) or "").strip()
        if v in _STYLE_VALUES:
            return v
    return None


def _run_race(r: dict):
    """レース名(クラス)を新旧 schema 両対応で取り出す。"""
    v = r.get("race_name")
    if v:
        return v
    v = str(r.get("style") or "").strip()  # 旧schemaは style にレース名
    return v if v and v not in _STYLE_VALUES else None


def _fmt_runs(hist: dict) -> list:
    """過去走を読みやすい形に（前走どうだった?に答えるため）。"""
    out = []
    for r in (hist.get("runs") or [])[:5]:
        out.append({
            "何走前": r.get("n_ago"), "レース": _run_race(r),
            "着順": r.get("pos"), "人気": r.get("ninki"),
            "コース": f"{r.get('place', '')}{r.get('td', '')}{r.get('dist', '')}",
            "上り3F": r.get("agari3f"), "脚質": _run_style(r),
        })
    return out


def _form_line(hist: dict) -> str:
    runs = hist.get("runs") or []
    if not runs:
        return ""
    r = runs[0]
    return (f"。前走={_run_race(r)} {r.get('pos')}着"
            f"({r.get('ninki')}人気・{r.get('place')}{r.get('td')}{r.get('dist')})")


def describe_horse(race: dict, umaban: int) -> dict:
    """1頭の grounded ファクトを dict で返す。"""
    h = _horse(race, umaban)
    hist = h.get("history", {})
    last = (hist.get("runs") or [{}])[0]
    cp = race["race_meta"].get("class_prior", {})
    prior_key = _MARK2PRIOR.get(h.get("mark"))
    return {
        "umaban": umaban,
        "name": h["horse_name"],
        "mark": h.get("mark") or "無",
        "ai_rank": h.get("ai_rank"),
        "p_win": h.get("p_win"),
        "p_plc": h.get("p_plc"),
        "p_sho": h.get("p_sho"),
        "tansho_odds": h.get("tansho_odds"),
        "ev_win": ev_win(h),
        "ev_plc": ev_plc(h),
        "value_tag": h.get("ai_vs_market"),       # over / fair / under
        "avg_pos": hist.get("avg_pos"),
        "pos_trend": hist.get("pos_trend"),
        "same_dist_ratio": hist.get("same_dist_ratio"),
        "last_pos": last.get("pos"),
        "last_ninki": last.get("ninki"),
        "class_prior_top3_pct": cp.get(prior_key) if prior_key else None,
        "past_runs": _fmt_runs(hist),
        "summary": (
            f"{h['horse_name']}（{h.get('mark') or '無印'}・AI{h.get('ai_rank')}位）: "
            f"複勝率{(h.get('p_plc') or 0)*100:.1f}% / "
            f"単勝EV {_ev_label(ev_win(h))} / 複勝EV {_ev_label(ev_plc(h))} / "
            f"市場評価={h.get('ai_vs_market')} / 単勝{h.get('tansho_odds')}倍"
            + ("。※人気薄＝高EVは穴の罠で軸に不適"
               if (h.get('tansho_odds') or 0) >= 20 else "")
            + _form_line(hist)),
    }


def roster_text(race: dict) -> str:
    """レースの出走馬一覧（馬番・馬名・印・複勝率・単勝オッズ）。LLM への文脈注入用。"""
    m = race["race_meta"]
    lines = [f"{m.get('place')} {m.get('course')} {m.get('class')} ({m.get('field_size')}頭)"]
    for h in sorted(race["horses"], key=lambda x: x["umaban"]):
        mk = h.get("mark") or "－"
        lines.append(f"  {h['umaban']}番 {h['horse_name']} [{mk}] "
                     f"複勝{(h.get('p_plc') or 0)*100:.0f}% 単勝{h.get('tansho_odds','?')}倍")
    return "\n".join(lines)


def find_horse(race: dict, umaban_or_name) -> dict:
    """馬番(数字) か 馬名(部分一致) で1頭を引いて詳細を返す。曖昧時は roster を返す。"""
    key = str(umaban_or_name).strip()
    digits = "".join(ch for ch in key if ch.isdigit())
    if digits:
        try:
            return describe_horse(race, int(digits))
        except KeyError:
            pass
    for h in race["horses"]:
        if key and key in h["horse_name"]:
            return describe_horse(race, h["umaban"])
    return {"error": f"'{key}' に該当する馬が見つからない",
            "roster": [f"{h['umaban']}番 {h['horse_name']}"
                       for h in sorted(race["horses"], key=lambda x: x["umaban"])]}


@functools.lru_cache(maxsize=1)
def _course_stats() -> dict:
    p = REPO / "data" / "course_stats.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def course_bias(race: dict) -> dict:
    """現在レースのコースの枠順バイアス（内/中/外枠の勝率・複勝率＋有利判定, 2013-25集計）。"""
    m = race["race_meta"]
    place, course = m.get("place", ""), m.get("course", "")
    cs = _course_stats()
    keys = [f"{place}|{course}"]
    if course.startswith("ダート"):
        keys.append(f"{place}|ダ{course[3:]}")
    elif course.startswith("芝") or course.startswith("ダ"):
        keys.append(f"{place}|ダート{course[1:]}")
    entry = next((cs[k] for k in keys if k in cs), None)
    if not entry or not entry.get("waku"):
        return {"error": f"{place} {course} のコース統計が見つからない"}
    waku = {str(w.get("label")): w for w in entry["waku"]}
    bands = {"内枠(1-2)": [1, 2], "中枠(3-6)": [3, 4, 5, 6], "外枠(7-8)": [7, 8]}
    out = {}
    for name, frames in bands.items():
        tot = sum(waku[str(f)]["n_total"] for f in frames if str(f) in waku)
        win = sum(waku[str(f)].get("n_1", 0) for f in frames if str(f) in waku)
        top3 = sum(waku[str(f)].get("n_1", 0) + waku[str(f)].get("n_2", 0)
                   + waku[str(f)].get("n_3", 0) for f in frames if str(f) in waku)
        if tot:
            out[name] = {"win_rate": round(win / tot * 100, 1),
                         "fuku_rate": round(top3 / tot * 100, 1)}
    if not out:
        return {"error": f"{place} {course} の枠データ不足"}
    best = max(out, key=lambda b: out[b]["fuku_rate"])
    return {
        "course": f"{place} {course}", "n_races": entry.get("n_races"),
        "bands": out, "verdict": f"{best}が複勝率最高＝有利",
        "summary": (f"{place}{course}（{entry.get('n_races')}R集計）枠バイアス: "
                    + " / ".join(f"{b} 複勝{v['fuku_rate']}%・勝率{v['win_rate']}%"
                                 for b, v in out.items())
                    + f" → {best}が有利"),
    }


LONGSHOT_ODDS = 20.0  # 単勝これ以上＝人気薄。EVは高く出るが『穴の罠』。AIは穴に構造的に弱い。


def compare_horses(race: dict, a: int, b: int, axis: str = "plc") -> dict:
    """2頭を grounded 比較。

    『軸/◎どっち』は的中の"信頼性"＝複勝率で推奨する（EVチェイスしない）。
    高EVが人気薄に出ている時は『穴の罠』として警告する。
    """
    A, B = describe_horse(race, a), describe_horse(race, b)
    rc = race.get("race_confidence", {})

    # --- 推奨は信頼性（複勝率）で決める ---
    pick = A if (A["p_plc"] or 0) >= (B["p_plc"] or 0) else B
    other = B if pick is A else A
    prob_edge = abs((A["p_plc"] or 0) - (B["p_plc"] or 0)) * 100

    notes = []
    # 穴の罠ガード: 高EV側が人気薄なら、それは罠だと明示
    hi_ev = A if (A["ev_plc"] or 0) >= (B["ev_plc"] or 0) else B
    if hi_ev is not pick and (hi_ev["tansho_odds"] or 0) >= LONGSHOT_ODDS:
        notes.append(
            f"{hi_ev['umaban']}番は単勝{hi_ev['tansho_odds']:.0f}倍の人気薄。"
            f"複勝EV{hi_ev['ev_plc']:.2f}は高く見えるが『小確率×大配当』の穴の罠で、"
            f"当AIは穴に構造的に弱い→軸には不適")
    # 拮抗かつ高EV側が極端な人気薄でない時だけ、妙味を補足
    if (prob_edge < 2.0 and (other["ev_plc"] or 0) > (pick["ev_plc"] or 0)
            and (other["tansho_odds"] or 0) < LONGSHOT_ODDS):
        notes.append(f"複勝率は拮抗（{prob_edge:.1f}pt差）。妙味なら{other['umaban']}番"
                     f"（複勝EV{other['ev_plc']:.2f}>{pick['ev_plc']:.2f}）も一考")
    if pick["value_tag"] == "over":
        notes.append(f"推奨{pick['umaban']}番は妙味'over'＝やや過剰人気、旨みは薄い")
    if rc.get("field_chaos_score", 0) >= 0.85:
        notes.append(f"chaos={rc['field_chaos_score']:.2f}＝荒れる鞍。軸の信頼度も割引いて見る")
    if (pick["ev_plc"] or 0) < 1:
        notes.append(f"推奨馬も複勝EV{pick['ev_plc']:.2f}<1.0＝期待値マイナス帯。"
                     f"厚張りせず薄く、見送りも可")

    recommendation = (
        f"{axis}の軸は{pick['umaban']}番 {pick['name']}。"
        f"軸は『当たる信頼性』で選ぶ＝複勝率{pick['p_plc']*100:.1f}%が"
        f"{other['umaban']}番({other['p_plc']*100:.1f}%)を{prob_edge:.1f}pt上回る。")

    return {
        "race": race["race_meta"], "axis": axis, "a": A, "b": B,
        "recommendation": recommendation, "pick": pick["umaban"],
        "prob_edge_pct": round(prob_edge, 1), "notes": notes,
        "summary": _fmt_compare(race, A, B, recommendation, notes),
    }


def race_advice(race: dict) -> dict:
    """鞍全体の play/pass 判断材料。"""
    rc = race.get("race_confidence", {})
    horses = sorted(race["horses"], key=lambda h: h.get("ai_rank", 99))
    top = [describe_horse(race, h["umaban"]) for h in horses[:3]]
    chaos = rc.get("field_chaos_score", 0)
    verdict = "見送り寄り" if chaos >= 0.85 else ("妙味次第" if chaos >= 0.6 else "勝負可")
    return {"race": race["race_meta"], "confidence": rc, "top": top,
            "verdict": verdict, "summary": _fmt_advice(race, top, verdict)}


# ---------------------------------------------------------------- 整形
def _fmt_compare(race, A, B, recommendation, notes) -> str:
    m = race["race_meta"]
    L = [f"【{m['place']} {m['course']} {m['class']}】"]
    for h in (A, B):
        L.append(
            f"  {h['umaban']}番 {h['name']}（{h['mark']}・AI{h['ai_rank']}位）"
            f" P複={h['p_plc']*100:.1f}% 単勝EV={h['ev_win']:.2f} 複勝EV={h['ev_plc']:.2f}"
            f" 妙味={h['value_tag']} 単勝{h['tansho_odds']}倍 近走avg={h['avg_pos']}")
    L.append(f"  → {recommendation}")
    for n in notes:
        L.append(f"    ※ {n}")
    return "\n".join(L)


def _fmt_advice(race, top, verdict) -> str:
    m, rc = race["race_meta"], race["race_confidence"]
    L = [f"【{m['place']} {m['course']} {m['class']}】判定: {verdict}",
         f"  chaos={rc.get('field_chaos_score'):.2f} AI-市場一致={rc.get('ai_market_agreement'):.2f}"]
    for h in top:
        L.append(f"  {h['mark']} {h['umaban']}番 {h['name']} P複={h['p_plc']*100:.1f}% 妙味={h['value_tag']}")
    return "\n".join(L)


# ---------------------------------------------------------------- 自己テスト
if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    date = sys.argv[1] if len(sys.argv) > 1 else "20260531"
    b = load_bundle(date)
    print(f"bundle={date} model={b.get('model')} races={b.get('race_count')}\n")

    # 一番"迷う"鞍（top1_dominance 最小）でデモ
    races = b["races"]
    idx = min(range(len(races)),
              key=lambda i: races[i]["race_confidence"].get("top1_dominance", 1))
    r = races[idx]
    ranked = sorted(r["horses"], key=lambda h: h["ai_rank"])
    a, bb = ranked[0]["umaban"], ranked[1]["umaban"]

    print("● race_advice():")
    print(race_advice(r)["summary"])
    print(f"\n● compare_horses({a},{bb}) — ユーザー『◎どっち迷う』想定:")
    print(compare_horses(r, a, bb)["summary"])
