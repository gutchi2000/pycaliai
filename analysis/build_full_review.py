# -*- coding: utf-8 -*-
"""TARGET 全頭詳細CSV から『全レース全頭』の濃い回顧を生成する.

入力(TARGET frontier JV エクスポート, cp932):
  E:\競馬過去走データ\レース回顧_{range}.csv
  列: 日付,場所,Ｒ,レース名,クラス名,枠番,馬番,...,人気,確定着順,単勝オッズ,
      芝・ダ,距離,1角,2角,3角,4角,決め手,脚質,上り3F,上り3F順,着差,走破タイム,...
  → 通過順・脚質・上がり3F(順)・着差から、実際の走りに基づく回顧を全頭ぶん自動生成。

bundle (印・AI評価) と突合して各馬に AI印(◎〇▲△/無印rankN/対象外) を付与。

出力:
  data/uma_review/full/{date}.json           ... 全頭レコード
  data/uma_review/horse_notes_full.json      ... 馬名キー蓄積DB(全頭)
  reports/review/{range}_full_kaiko.md        ... 読み物(着順順)
  reports/review/{range}_full_target_import.csv ... TARGET一括取込(全頭, cp932)

usage:
  python analysis/build_full_review.py                 # data/target_review/ の最新CSVを自動検出
  python analysis/build_full_review.py data\target_review\レース回顧_0418-0531.csv
"""
import sys
import os
import json
import glob
import csv as _csv

sys.stdout.reconfigure(encoding="utf-8")
BASE = r"E:\PyCaLiAI"
INDIR = rf"{BASE}\reports\cowork_input"
DBDIR = rf"{BASE}\data\uma_review"
FULLDIR = rf"{DBDIR}\full"
REVDIR = rf"{BASE}\reports\review"
TARGETDIR = rf"{BASE}\data\target_review"
os.makedirs(FULLDIR, exist_ok=True)
os.makedirs(REVDIR, exist_ok=True)
os.makedirs(TARGETDIR, exist_ok=True)


def latest_target_csv():
    """data/target_review/ 内の最新(mtime)CSVパスを返す。無ければ None."""
    cands = glob.glob(rf"{TARGETDIR}\*.csv")
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)

NORM = {"○": "〇"}
# JRA標準 場コード
PLACE_CODE = {"札幌": "01", "函館": "02", "福島": "03", "新潟": "04", "東京": "05",
              "中山": "06", "中京": "07", "京都": "08", "阪神": "09", "小倉": "10"}


def to_i(x):
    try:
        return int(float(str(x).strip()))
    except Exception:
        return None


def to_f(x):
    try:
        return float(str(x).strip())
    except Exception:
        return None


def odds_lt(o, v):
    f = to_f(o)
    return f is not None and f < v


def odds_ge(o, v):
    f = to_f(o)
    return f is not None and f >= v


def load_bundles(date8_set):
    """(date8,place,R,umaban)->horse, (date8,place,R)->race_id(16桁), (date8,place)->回日."""
    hmap, ridmap, kaiday = {}, {}, {}
    for date8 in date8_set:
        path = rf"{INDIR}\{date8}_bundle.json"
        if not os.path.exists(path):
            continue
        b = json.load(open(path, encoding="utf-8"))
        for r in b["races"]:
            rid = r["race_id"]
            place = r["race_meta"]["place"]
            rno = int(rid[-2:])
            ridmap[(date8, place, rno)] = rid
            kaiday[(date8, place)] = rid[10:14]  # 回(2)+日(2)
            for h in r["horses"]:
                hmap[(date8, place, rno, int(h["umaban"]))] = h
    return hmap, ridmap, kaiday


def ai_prefix(h):
    if h is None:
        return "(AI対象外) "
    mk = NORM.get(h.get("mark"), h.get("mark"))
    if mk:
        return f"AI{mk} "
    return f"AI無印(r{h.get('ai_rank')}) "


def rich_kaiko(row, h):
    """通過順・脚質・上がり・着差から実走ベースの濃い回顧 + 次走メモ."""
    chaku = to_i(row["確定着順"])
    ninki = to_i(row["人気"])
    ks = row["脚質"].strip()
    corners = [c.strip() for c in (row["1角"], row["2角"], row["3角"], row["4角"]) if c.strip()]
    ag = row["上り3F"].strip()
    agr = to_i(row["上り3F順"])
    odds = row["単勝オッズ"]
    td = row["芝・ダ"].strip()
    dist = row["距離"].strip()
    chasa = to_f(row["着差"])
    track = f"{td}{dist}"
    nk = f"{ninki}人気" if ninki else "人気?"

    if not chaku:
        return "出走取消/競走中止。", "次走の出否・状態を確認してから評価。"

    passs = "-".join(corners)
    pos_word = {"逃げ": "ハナ", "先行": "好位先行", "中団": "中団",
                "後方": "後方", "追込": "後方", "ﾏｸﾘ": "まくり"}.get(ks, ks or "")
    posdesc = f"{pos_word}({passs})" if passs else (pos_word or "道中")
    ag_best = agr is not None and agr <= 3
    ag_top = agr == 1
    agtxt = ("上がり最速" + ag) if ag_top else (f"上がり{ag}({agr}位)" if (ag and agr) else (f"上がり{ag}" if ag else ""))
    margin = "" if chasa is None else (f"{abs(chasa):.1f}差" if abs(chasa) > 0 else "ハナ/クビ差")
    front = ks in ("逃げ", "先行")
    closer = ks in ("中団", "後方", "追込", "ﾏｸﾘ")
    cheap = odds_lt(odds, 5)
    longsh = odds_ge(odds, 8)
    big = odds_ge(odds, 15)
    small_m = chasa is not None and abs(chasa) <= 0.3
    in3 = chaku <= 3
    kessho = chaku in (4, 5)

    # --- なぜ(回顧) ---
    if chaku == 1:
        why = f"{nk}で勝利。{posdesc}から{agtxt or '押し切り'}で抜け出した。"
    elif in3:
        if closer and ag_best:
            why = f"{nk}で{chaku}着({margin})。{posdesc}から{agtxt}で{'僅差まで迫った' if small_m else '差し込んだ'}。"
        elif front:
            why = f"{nk}で{chaku}着({margin})。{posdesc}で運び{'粘り込んだ' if small_m else '渋とく好走'}。"
        else:
            why = f"{nk}で{chaku}着({margin})。{posdesc}から{agtxt}でまとめた。"
    elif kessho:
        if closer and ag_best:
            why = f"{nk}で{chaku}着({margin})。{posdesc}から{agtxt}を使うも届かず＝展開待ちの惜敗。"
        else:
            why = f"{nk}で{chaku}着({margin})。{posdesc}で運ぶも詰め切れず掲示板まで。"
    else:  # 着外
        if closer and ag_best:
            why = f"{nk}で{chaku}着も{agtxt}＝脚は使えており展開・位置取りが向かなかった。"
        elif front:
            why = f"{nk}で{chaku}着。{posdesc}から失速({agtxt})＝流れか距離が忙しかった。"
        else:
            why = f"{nk}で{chaku}着。{posdesc}で見せ場を欠いた。"

    # --- 次走メモ ---
    if chaku == 1:
        jisou = (f"人気薄で勝利＝次走も妙味残す。{track}は条件合う。" if longsh
                 else f"地力上位。次走も上位人気で軸級({track}得意)。")
    elif in3:
        if closer and ag_best:
            jisou = f"末脚上位＝直線長め/流れる展開で次走前進。{'人気薄で妙味' if longsh else '軸〜ヒモ'}。"
        elif front:
            jisou = "前々で粘れる脚質＝前残り条件・少頭数・平坦で狙い目。"
        else:
            jisou = "堅実。次走も相手次第でヒモ〜軸。"
    elif kessho:
        if closer and ag_best:
            jisou = "展開向けば突き抜け候補＝人気が上がりにくい次走が狙い目。"
        else:
            jisou = "条件好転(距離/コース/相手)で巻き返し可。"
    else:
        if closer and ag_best:
            jisou = "脚は使えており展開待ち＝次走前付けorペースが流れれば一発。"
        elif front:
            jisou = "距離短縮/ペース緩和で見直し。現条件は忙しい。"
        elif cheap:
            jisou = "人気を裏切る内容＝次走は過信禁物。"
        elif big:
            jisou = "現級では力不足。相手弱化・クラス据え置きまで静観。"
        else:
            jisou = "条件替わりを待って次走で再評価。"
    return why, jisou


def main(csv_path):
    with open(csv_path, encoding="cp932", newline="") as f:
        rows = list(_csv.DictReader(f))
    # date8 集合
    date8_set = sorted({"20" + r["日付"] for r in rows})
    hmap, ridmap, kaiday = load_bundles(date8_set)

    # 重賞等の手書き回顧(Web上乗せ)を date 別に読む
    ov_by_date = {}
    for d8 in date8_set:
        ovp = rf"{BASE}\reports\web_results\overrides_{d8}.json"
        if os.path.exists(ovp):
            ov_by_date[d8] = json.load(open(ovp, encoding="utf-8"))

    # レース単位 group
    races = {}
    for r in rows:
        key = (r["日付"], r["場所"], to_i(r["Ｒ"]))
        races.setdefault(key, []).append(r)

    md = ["# 全レース全頭 レース回顧（TARGET詳細・実走ベース）\n"]
    by_date = {}
    horse_db = json.load(open(rf"{DBDIR}\horse_notes_full.json", encoding="utf-8")) \
        if os.path.exists(rf"{DBDIR}\horse_notes_full.json") else {}
    csv_rows = []
    n_horse = n_race = 0
    cur_date = None

    for (d6, place, rno) in sorted(races, key=lambda x: (x[0], x[1], x[2] or 0)):
        date8 = "20" + d6
        grp = races[(d6, place, rno)]
        meta = grp[0]
        if date8 != cur_date:
            md.append(f"\n## ── {date8} ──\n")
            cur_date = date8
        n_race += 1
        # 新18桁の元になる race_id(16桁): bundleの正規IDを優先、無ければ構築(date8+場+回日+R)
        kd = kaiday.get((date8, place))
        pc = PLACE_CODE.get(place)
        base_rid = ridmap.get((date8, place, rno))
        if not base_rid and kd and pc:
            base_rid = date8 + pc + kd + f"{rno:02d}"

        ov = ov_by_date.get(date8, {})
        ov_race = ov.get("race_notes", {}).get(f"{place}|{rno}")
        ov_horse = ov.get("horse_notes", {})

        grp_sorted = sorted(grp, key=lambda r: (to_i(r["確定着順"]) or 99, to_i(r["馬番"]) or 0))
        md.append(f"### {place}{rno}R {meta['レース名'].strip()} {meta['芝・ダ'].strip()}{meta['距離'].strip()} "
                  f"{meta['クラス名'].strip()} ({meta['出走頭数'].strip()}頭)")
        if ov_race:
            md.append(f"**展開**: {ov_race}")
        md.append("| 着 | 馬番 | 馬名 | AI印 | 人気 | 回顧 → 次走 |")
        md.append("|---|---|---|---|---|---|")

        recs = []
        for r in grp_sorted:
            uma = to_i(r["馬番"])
            if uma is None:
                continue
            h = hmap.get((date8, place, rno, uma))
            pref = ai_prefix(h)
            why, jisou = rich_kaiko(r, h)
            hov = ov_horse.get(f"{place}|{rno}|{uma}")
            if hov:  # 重賞等の手書き回顧で上書き(全頭データ版に文脈を上乗せ)
                why = hov.get("kaiko", why)
                jisou = hov.get("jisou", jisou)
            comment = pref + why
            chaku = to_i(r["確定着順"])
            res_str = f"{chaku}着" if chaku else "取消"
            ninki = to_i(r["人気"])
            md.append(f"| {res_str} | {uma} | {r['馬名'].strip()} | {pref.strip()} | "
                      f"{ninki or '?'} | {why} → **{jisou}** |")

            rid18 = (base_rid + f"{uma:02d}") if base_rid else None
            rec = {
                "date": date8, "place": place, "R": rno,
                "race_name": meta["レース名"].strip(),
                "course": f"{r['芝・ダ'].strip()}{r['距離'].strip()}",
                "class": meta["クラス名"].strip(),
                "field_size": to_i(meta["出走頭数"]),
                "umaban": uma, "horse_name": r["馬名"].strip(),
                "ai_mark": (NORM.get(h.get("mark"), h.get("mark")) if h else None),
                "ai_rank": (h.get("ai_rank") if h else None),
                "ninki": ninki, "result": res_str,
                "result_pos": chaku if chaku else None,
                "kyakushitsu": r["脚質"].strip(),
                "pass": "-".join([c.strip() for c in (r["1角"], r["2角"], r["3角"], r["4角"]) if c.strip()]),
                "agari": r["上り3F"].strip(), "agari_rank": to_i(r["上り3F順"]),
                "chakusa": r["着差"].strip(), "time": r["走破タイム"].strip(),
                "tansho_odds": to_f(r["単勝オッズ"]),
                "race_id18": rid18,
                "kaiko": comment, "jisou": jisou, "source": "target_detail",
            }
            recs.append(rec)
            n_horse += 1
            db = horse_db.setdefault(r["馬名"].strip(), [])
            horse_db[r["馬名"].strip()] = [
                x for x in db if not (x.get("date") == date8 and x.get("place") == place
                                      and x.get("R") == rno and x.get("umaban") == uma)]
            horse_db[r["馬名"].strip()].append(rec)

            if rid18:
                kaisai = f"{place}{rno}R".replace(",", "")
                nm = r["馬名"].strip().replace(",", "、")
                cm = (comment + " → " + jisou).replace(",", "、").replace("\n", " ").replace("〜", "～")
                csv_rows.append(f"{kaisai},{nm},{rid18},{cm}")
        md.append("")
        by_date.setdefault(date8, []).extend(recs)

    for d8, recs in by_date.items():
        with open(rf"{FULLDIR}\{d8}.json", "w", encoding="utf-8") as f:
            json.dump(recs, f, ensure_ascii=False, indent=1)
    for nm in horse_db:
        horse_db[nm].sort(key=lambda x: (x.get("date", ""), x.get("place", ""), x.get("R", 0)))
    with open(rf"{DBDIR}\horse_notes_full.json", "w", encoding="utf-8") as f:
        json.dump(horse_db, f, ensure_ascii=False, indent=1)

    rng = f"{date8_set[0]}_{date8_set[-1][-4:]}"
    md_path = rf"{REVDIR}\{rng}_full_kaiko.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    csv_path_out = rf"{REVDIR}\{rng}_full_target_import.csv"
    with open(csv_path_out, "w", encoding="cp932", errors="replace", newline="") as f:
        f.write("\n".join(csv_rows) + "\n")

    print(f"全頭レコード {n_horse} / レース {n_race} / 馬名DB {len(horse_db)}頭")
    print(f"md  → {md_path}")
    print(f"CSV → {csv_path_out} ({len(csv_rows)}行, TARGET一括取込用)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = latest_target_csv()
        if not path:
            sys.exit(f"CSVが見つかりません。TARGETの「レース回顧」CSVを {TARGETDIR} に置いてください。")
        print(f"自動検出: {path}")
    main(path)
