# -*- coding: utf-8 -*-
"""◎〇▲△△ 印馬の「次走に生かす」レース回顧を1頭ずつ生成し、馬名キーDBに蓄積する.

回顧 = 着順の答え合わせではなく「なぜその結果か」「次走どう狙うか」を残す作業。

結果ソース(優先順):
  1) data/kekka/{date}.csv          ... 全頭の確定着順(高品質。正確な着順で回顧)
  2) reports/web_results/{date}.json ... Webで調べた1-3着(kekka無い日)
  3) reports/web_results/overrides_{date}.json ... 重賞等の手書き回顧(最優先で注入)

出力:
  data/uma_review/{date}.json        ... その日の全印馬レコード(list)
  data/uma_review/horse_notes.json   ... 馬名キーの蓄積DB {馬名:[record,...]}
  reports/review/{weekend}_kaiko.md   ... 読み物回顧

usage:
  python analysis/build_uma_review.py                 # 最新土日を自動検出
  python analysis/build_uma_review.py 20260606 20260607
"""
import sys
import json
import glob
import os

sys.stdout.reconfigure(encoding="utf-8")
BASE = r"E:\PyCaLiAI"
INDIR = rf"{BASE}\reports\cowork_input"
RESDIR = rf"{BASE}\reports\web_results"
KEKKADIR = rf"{BASE}\data\kekka"
DBDIR = rf"{BASE}\data\uma_review"
REVDIR = rf"{BASE}\reports\review"
os.makedirs(DBDIR, exist_ok=True)
os.makedirs(REVDIR, exist_ok=True)

MARK_ORDER = {"◎": 0, "〇": 1, "○": 1, "▲": 2, "△": 3}
NORM = {"○": "〇"}


def latest_weekend():
    files = glob.glob(rf"{INDIR}\*_bundle.json")
    dates = sorted({os.path.basename(f).split("_")[0] for f in files}, reverse=True)
    return list(reversed(dates[:2]))


def race_no(rid):
    return int(str(rid)[-2:])


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_kekka(date):
    """data/kekka/{date}.csv → {(place, R_int): {umaban_int: 確定着順_int}}. 無ければ None."""
    path = rf"{KEKKADIR}\{date}.csv"
    if not os.path.exists(path):
        return None
    import pandas as pd
    try:
        k = pd.read_csv(path, encoding="cp932", header=0)
    except Exception:
        k = pd.read_csv(path, encoding="utf-8", header=0)
    cols = list(k.columns)
    # 想定: 日付,場所,Ｒ,枠番,馬番,馬名,確定着順,レースID(新),...
    cmap = {}
    for _, row in k.iterrows():
        try:
            place = str(row[cols[1]]).strip()
            rno = int(row[cols[2]])
            uma = int(row[cols[4]])
        except Exception:
            continue
        chaku = row[cols[6]]
        try:
            ch = int(chaku)
        except Exception:
            ch = None  # 取消/除外/中止
        cmap.setdefault((place, rno), {})[uma] = ch
    return cmap


def ninki_rank(race):
    rows = [(h["umaban"], h.get("tansho_odds")) for h in race["horses"]
            if h.get("tansho_odds") is not None]
    rows.sort(key=lambda x: x[1])
    return {u: i for i, (u, _o) in enumerate(rows, 1)}


def last_run(h):
    runs = (h.get("history") or {}).get("runs") or []
    return runs[0] if runs else None


def kyakushitsu(h):
    lr = last_run(h) or {}
    # 新schema: style=決手 / 旧schema: weight_change に決手が入っていた
    v = str(lr.get("style") or "").strip()
    if v in {"逃げ", "先行", "中団", "差し", "ﾏｸﾘ", "マクリ", "後方", "追込"}:
        return v
    return lr.get("weight_change")


def auto_kaiko(h, pos, ninki, exact):
    """人気・AI乖離・脚質・着順から『なぜ/次走』を自動生成.
    exact=True(kekka)なら4着以降の正確な着順で精緻化、False(web top3)なら着外一括."""
    o = h.get("tansho_odds")
    vs = h.get("ai_vs_market")
    ks = kyakushitsu(h)
    lr = last_run(h)
    nk = f"{ninki}人気" if ninki else "人気?"
    cheap = o is not None and o < 5
    mid = o is not None and 5 <= o < 8
    longsh = o is not None and o >= 8
    big = o is not None and o >= 15

    # --- なぜ(回顧) ---
    if exact and pos is None:
        why = f"{nk}。出走取消/着順記録なし。"
        jisou = "次走の出否・状態を確認してから評価。"
        return why, jisou

    if pos == 1:
        why = f"{nk}で勝ち切り。地力上位を示した。"
    elif pos == 2:
        why = f"{nk}で2着・連対。勝ち負けの好内容。"
    elif pos == 3:
        why = f"{nk}で3着・複勝圏を確保。展開を大きく外さず堅実。"
    elif exact and pos in (4, 5):
        why = f"{nk}で{pos}着。掲示板＝あと一歩の惜敗で力は出した。"
    elif exact and pos is not None and 6 <= pos <= 9:
        why = f"{nk}で{pos}着、中位どまり。展開か適性が今ひとつ。"
    elif exact and pos is not None and pos >= 10:
        if cheap:
            why = f"{nk}と支持されたが{pos}着大敗。明確に期待を裏切る負け方。"
        elif vs == "under" and o is not None and o >= 10:
            why = f"市場より強気に推した({nk})が{pos}着大敗。高オッズ単穴が飛ぶ構造的弱点の典型。"
        else:
            why = f"{nk}で{pos}着大敗。現級では力不足。"
    else:
        # web(top3のみ) の着外
        if cheap:
            why = f"{nk}と支持されたが着外。人気を裏切る内容で、展開か状態に課題。"
        elif vs == "under" and o is not None and o >= 10:
            why = f"市場より強気に推した({nk})が着外。高オッズ単穴が飛ぶ構造的弱点の典型。"
        else:
            why = f"{nk}で着外。現級の流れでは決め手を欠いた。"

    # --- 次走メモ ---
    in3 = pos in (1, 2, 3)
    kessho = exact and pos in (4, 5)
    if in3 and longsh:
        jisou = "人気薄で好走＝次走も人気はさほど上がらず妙味継続。要マーク。"
    elif in3 and cheap:
        jisou = "人気に応え好走。次走も上位人気なら軸候補(特に複・連)。"
    elif in3:
        jisou = "中人気で結果。次走も妙味を残しつつ軸〜ヒモで。"
    elif kessho and (longsh or mid):
        jisou = "惜敗で人気上がりにくい＝次走妙味継続。展開向けば突き抜け候補、要マーク。"
    elif kessho:
        jisou = "掲示板で力は示した。次走は条件好転なら巻き返し濃厚、軸候補。"
    elif cheap:
        jisou = "次走は人気でも過信禁物。巻き返しは相手弱化・展開緩和を待って。"
    elif vs == "under" and o is not None and o >= 10:
        jisou = "高オッズ単穴は次走も嫌う。狙うなら絞り中人気の妙味に寄せる。"
    elif big:
        jisou = "評価相応に着外。現級では足りず、相手弱化・クラス据え置きまで静観。"
    else:
        jisou = "条件替わり(距離/コース/クラス)を待って次走で見直し。"

    if ks and lr:
        why += f"（前走脚質:{ks} / {lr.get('place','')}{lr.get('dist','')}{lr.get('td','')} {lr.get('pos','')}着）"
    return why, jisou


def auto_kaiko_miss(h, pos, ninki):
    """AIが無印(印なし)なのに着内(複勝圏)に来た馬＝取りこぼし馬の回顧.
    『なぜ本モデルが見落としたか』『次走どう扱うか』を残す."""
    o = h.get("tansho_odds")
    vs = h.get("ai_vs_market")
    ks = kyakushitsu(h)
    lr = last_run(h)
    nk = f"{ninki}人気" if ninki else "人気?"
    r = h["ai_rank"]
    posname = {1: "1着", 2: "2着", 3: "3着"}.get(pos, f"{pos}着")

    if o is not None and o < 5:
        why = f"AI無印(rank{r})ながら{nk}で{posname}。市場上位馬を本モデルが過小評価した取りこぼし。"
        jisou = "実力上位は明白。次走も上位人気なら軽視禁物＝印が追いつくか要観察。"
    elif o is not None and o < 15:
        why = f"AI無印(rank{r})の{nk}が{posname}。中人気の伏兵を拾えず。"
        jisou = "次走も中人気なら警戒。脚質/条件の再現性とAI評価の上昇を確認。"
    else:
        why = f"AI無印(rank{r})の人気薄({nk})が{posname}に激走＝AI・市場とも見落とした紛れ。"
        jisou = "反復性は低め。次走は人気を落としたところの一発に注意。"
    if vs == "under":
        why += " (AIは市場より相対的に高め=underだったが印には届かず)"
    if ks and lr:
        why += f"（前走脚質:{ks} / {lr.get('place','')}{lr.get('dist','')}{lr.get('td','')} {lr.get('pos','')}着）"
    return why, jisou


def build(dates):
    horse_db_path = rf"{DBDIR}\horse_notes.json"
    horse_db = load_json(horse_db_path) or {}
    md = ["# ◎〇▲△△ レース回顧（次走に生かす版）\n"]
    all_records = {}

    for date in dates:
        bundle = load_json(rf"{INDIR}\{date}_bundle.json")
        if not bundle:
            md.append(f"## {date}: bundle なし\n")
            continue
        kekka = load_kekka(date)
        web = load_json(rf"{RESDIR}\{date}.json")
        ov = load_json(rf"{RESDIR}\overrides_{date}.json") or {}
        ov_race = ov.get("race_notes", {})
        ov_horse = ov.get("horse_notes", {})
        web_results = (web or {}).get("results", {})
        if kekka:
            src = "data/kekka(全頭確定着順)"
        elif web:
            src = web.get("source", "web")
        else:
            md.append(f"## {date}: kekka も web_results も無し → スキップ\n")
            continue

        records = []
        races = sorted(bundle["races"],
                       key=lambda r: (r["race_meta"].get("place", ""), race_no(r["race_id"])))
        md.append(f"\n## ── {date} (model {bundle.get('model')}) ──")
        md.append(f"_結果ソース: {src}_\n")

        for r in races:
            m = r["race_meta"]
            place, rno = m.get("place"), race_no(r["race_id"])
            name_by_uma = {int(h["umaban"]): h["horse_name"] for h in r["horses"]}
            exact = False
            finish = None  # {uma:chaku}
            top3 = None    # [(uma,name),...]
            if kekka and (place, rno) in kekka:
                finish = kekka[(place, rno)]
                exact = True
                tlist = sorted([(u, c) for u, c in finish.items() if c in (1, 2, 3)],
                               key=lambda x: x[1])
                top3 = [(u, name_by_uma.get(u, "?")) for u, _c in tlist]
            elif place in web_results and str(rno) in web_results[place]:
                top3 = [(int(u), nm) for u, nm in web_results[place][str(rno)]]

            nk_map = ninki_rank(r)
            marked = sorted([h for h in r["horses"] if h.get("mark")],
                            key=lambda h: MARK_ORDER.get(h["mark"], 9))
            rc = r.get("race_confidence", {})
            chaos = rc.get("field_chaos_score")
            rkey = f"{place}|{rno}"

            head = (f"### {place}{rno}R {m.get('course')} {m.get('class')} "
                    f"({m.get('field_size')}頭)"
                    + (f" / chaos{chaos:.2f}" if chaos is not None else ""))
            md.append(head)
            if top3 is None:
                md.append("  ※結果未取得\n")
                continue

            umk = {int(h["umaban"]): NORM.get(h["mark"], h["mark"]) for h in marked}
            res_line = " → ".join(
                f"{i}着 {u}{umk.get(int(u),'')}{nm}" for i, (u, nm) in enumerate(top3, 1))
            if rkey in ov_race:
                md.append(f"**展開**: {ov_race[rkey]}")
            md.append(f"**結果**: {res_line}\n")
            md.append("| 印 | 馬 | 単/人気/AI | 着 | 回顧 → 次走メモ |")
            md.append("|---|---|---|---|---|")

            def pos_of(u):
                if exact:
                    return finish.get(u)  # 正確な着順 or None(取消)
                for i, (uu, _n) in enumerate(top3, 1):
                    if int(uu) == u:
                        return i
                return None

            for h in marked:
                mk = NORM.get(h["mark"], h["mark"])
                u = int(h["umaban"])
                pos = pos_of(u)
                ninki = nk_map.get(u)
                hkey = f"{place}|{rno}|{u}"
                o = h.get("tansho_odds")

                if hkey in ov_horse:
                    oh = ov_horse[hkey]
                    res_str = oh.get("result_override") or (
                        f"{pos}着" if pos else ("着外" if not exact else "取消"))
                    why = oh.get("kaiko", "")
                    jisou = oh.get("jisou", "")
                else:
                    if exact:
                        res_str = f"{pos}着" if pos else "取消/着順なし"
                    else:
                        res_str = f"{pos}着" if pos else "着外"
                    why, jisou = auto_kaiko(h, pos, ninki, exact)

                why = f"AI{mk} " + why  # 印を回顧文頭に明記(無印馬の「AI無印」と整合)
                ai_s = f"単{o if o is not None else '?'}/{ninki or '?'}人/r{h['ai_rank']}"
                md.append(f"| {mk} | {u} {h['horse_name']} | {ai_s} | {res_str} | {why} → **{jisou}** |")

                rec = {
                    "date": date, "place": place, "R": rno,
                    "course": m.get("course"), "class": m.get("class"),
                    "field_size": m.get("field_size"), "race_id": r["race_id"],
                    "umaban": u, "horse_name": h["horse_name"],
                    "mark": mk, "ai_rank": h["ai_rank"],
                    "p_win": h.get("p_win"), "p_sho": h.get("p_sho"),
                    "tansho_odds": o, "ninki": ninki,
                    "ai_vs_market": h.get("ai_vs_market"),
                    "kyakushitsu_zenso": kyakushitsu(h),
                    "result": res_str, "result_pos": (pos if isinstance(pos, int) else None),
                    "source": ("kekka" if exact else "web_top3"),
                    "kaiko": why, "jisou": jisou,
                }
                records.append(rec)
                horse_db.setdefault(h["horse_name"], [])
                horse_db[h["horse_name"]] = [
                    x for x in horse_db[h["horse_name"]]
                    if not (x.get("date") == date and x.get("race_id") == r["race_id"]
                            and x.get("umaban") == u)]
                horse_db[h["horse_name"]].append(rec)

            # --- AI無印で着内(複勝圏)に来た馬＝取りこぼし馬 ---
            h_by_uma = {int(hh["umaban"]): hh for hh in r["horses"]}
            miss_rows = []
            for i, (u_, nm) in enumerate(top3, 1):
                u_ = int(u_)
                if u_ in umk:
                    continue  # 印あり馬は処理済
                hh = h_by_uma.get(u_)
                if not hh:
                    miss_rows.append(f"| 無 | {u_} {nm} | (bundle該当なし) | {i}着 | 出走表に無く詳細不明。 → **次走要確認。** |")
                    continue
                ninki = nk_map.get(u_)
                hkey = f"{place}|{rno}|{u_}"
                if hkey in ov_horse:  # 重賞等の取りこぼし馬も手書き回顧を上乗せ可
                    oh = ov_horse[hkey]
                    why = f"AI無印(r{hh['ai_rank']}) " + oh.get("kaiko", "")
                    jisou = oh.get("jisou", "")
                else:
                    why, jisou = auto_kaiko_miss(hh, i, ninki)
                o = hh.get("tansho_odds")
                ai_s = f"単{o if o is not None else '?'}/{ninki or '?'}人/r{hh['ai_rank']}"
                miss_rows.append(
                    f"| 無 | {u_} {hh['horse_name']} | {ai_s} | {i}着 | {why} → **{jisou}** |")
                mrec = {
                    "date": date, "place": place, "R": rno,
                    "course": m.get("course"), "class": m.get("class"),
                    "field_size": m.get("field_size"), "race_id": r["race_id"],
                    "umaban": u_, "horse_name": hh["horse_name"],
                    "mark": "無", "ai_miss": True, "ai_rank": hh["ai_rank"],
                    "p_win": hh.get("p_win"), "p_sho": hh.get("p_sho"),
                    "tansho_odds": o, "ninki": ninki,
                    "ai_vs_market": hh.get("ai_vs_market"),
                    "kyakushitsu_zenso": kyakushitsu(hh),
                    "result": f"{i}着", "result_pos": i,
                    "source": ("kekka" if exact else "web_top3"),
                    "kaiko": why, "jisou": jisou,
                }
                records.append(mrec)
                horse_db.setdefault(hh["horse_name"], [])
                horse_db[hh["horse_name"]] = [
                    x for x in horse_db[hh["horse_name"]]
                    if not (x.get("date") == date and x.get("race_id") == r["race_id"]
                            and x.get("umaban") == u_)]
                horse_db[hh["horse_name"]].append(mrec)
            for row in miss_rows:
                md.append(row)
            md.append("")

        all_records[date] = records
        with open(rf"{DBDIR}\{date}.json", "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=1)

    # 馬名DBは日付順に整列して保存
    for nm in horse_db:
        horse_db[nm].sort(key=lambda x: (x.get("date", ""), x.get("race_id", "")))
    with open(horse_db_path, "w", encoding="utf-8") as f:
        json.dump(horse_db, f, ensure_ascii=False, indent=1)

    weekend = f"{dates[0]}_{dates[-1][-4:]}" if len(dates) > 1 else dates[0]
    out_md = rf"{REVDIR}\{weekend}_kaiko.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"印馬レコード: {sum(len(v) for v in all_records.values())}頭")
    print(f"馬名DB: {len(horse_db)}頭分 → {horse_db_path}")
    print(f"回顧md → {out_md}")
    return all_records


if __name__ == "__main__":
    args = sys.argv[1:]
    dates = args if args else latest_weekend()
    build(dates)
