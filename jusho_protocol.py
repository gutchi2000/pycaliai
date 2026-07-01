# -*- coding: utf-8 -*-
"""
jusho_protocol.py — 重賞プロトコル（重賞＋オープン特別/L の深掘り分析カード）
================================================================================
explain_race.py の分析カードを土台に、重賞向けの厚みを足した「標準手順」。
対象は クラス名 ∈ {Ｇ１,Ｇ２,Ｇ３,ＪＧ*,重賞,OP(L),ｵｰﾌﾟﾝ}（=重賞＋OP特別/L）。

explain_race の既存カード（印/勝率/複勝圏/予想・実オッズ/EV/ELO/Glicko/脚質/位置/馬場/展開）に、
重賞プロトコルが上乗せする層（全て既存データ流用・leak-safe）:
  ① 調教   : 坂路4F時計・ラスト1F・追切何日前（master trnH_*）＋ WC5F（trnW_5F）＋ 厩舎の仕上げ
  ② 血統   : 種牡馬／母父／父タイプ ＋ 当コース好調血統との一致（course_trend 好調血統_父）
  ③ ﾄﾗｯｸﾊﾞｲｱｽ: 当 場×芝ダ×距離区分×OP重賞 の 脚質ランク・好調枠番（course_trend）
  ④ 重賞文脈: グレード・レースレベル(ELO)・出走の厚み（field strength）

出力: reports/jusho/{rid}.json ＋ 標準出力カード
実行:
  PYTHONUTF8=1 ./venv311/Scripts/python.exe jusho_protocol.py              # cache内の最新重賞(デモ)
  PYTHONUTF8=1 ./venv311/Scripts/python.exe jusho_protocol.py 2025122806050811  # rid 指定（有馬記念）
  PYTHONUTF8=1 ./venv311/Scripts/python.exe jusho_protocol.py 20251228      # 日付の全重賞＋OP/L
"""
from __future__ import annotations
import io, json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import explain_race as EX  # 土台カードを再利用（import時に stdout を utf-8 化するので二重ラップしない）

BASE = Path(__file__).parent
MASTER_CSV = BASE / "data/master_v2_20130105-20251228.csv"
COURSE_TREND = BASE / "data/course_trend.json"
OUTDIR = BASE / "reports/jusho"
COL_RID = "レースID(新/馬番無)"; COL_BAN = "馬番"

JUSHO_CLASSES = {"Ｇ１", "Ｇ２", "Ｇ３", "ＪＧ１", "ＪＧ２", "ＪＧ３", "重賞", "OP(L)", "(L)", "ｵｰﾌﾟﾝ", "オープン", "OP"}
GRADE_RANK = {"Ｇ１": "G1", "Ｇ２": "G2", "Ｇ３": "G3", "ＪＧ１": "J.G1", "ＪＧ２": "J.G2",
              "ＪＧ３": "J.G3", "重賞": "重賞", "OP(L)": "L", "(L)": "L", "ｵｰﾌﾟﾝ": "OP", "オープン": "OP", "OP": "OP"}
_EXTRA_COLS = [COL_RID, COL_BAN, "馬名", "クラス名", "距離", "枠番", "種牡馬", "父タイプ名", "母父馬",
               "trnH_Time1", "trnH_Lap1", "trnH_days_ago", "trnW_5F", "調教師"]


def smile_from_dist(d):
    try:
        d = int(d)
        return "S" if d <= 1300 else "M" if d <= 1899 else "I" if d <= 2100 else "L" if d <= 2700 else "E"
    except Exception:
        return "M"


def season_from_month(mm):
    return {12: "冬", 1: "冬", 2: "冬", 3: "春", 4: "春", 5: "春",
            6: "夏", 7: "夏", 8: "夏", 9: "秋", 10: "秋", 11: "秋"}.get(int(mm), "通年")


def is_jusho_class(cls):
    return str(cls).strip() in JUSHO_CLASSES


def load_extra(date=None):
    """重賞層に要る master 追加列を読む（trn/血統/枠/距離）。"""
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False,
                     usecols=lambda c: c in _EXTRA_COLS)
    df[COL_RID] = df[COL_RID].astype(str)
    if date:
        df = df[df[COL_RID].str.startswith(str(date))].copy()
    df["ban"] = pd.to_numeric(df[COL_BAN], errors="coerce").fillna(0).astype(int)
    return df


def course_bias(venue, surf, dist, mm):
    """場×芝ダ×距離区分の脚質・枠バイアス＋好調血統。脚質バイアスはコース物性なので
    重賞セルが薄ければ同コース最大母数セルで代用(全クラス傾向)。血統は重賞セル優先。"""
    try:
        ct = json.loads(COURSE_TREND.read_text(encoding="utf-8"))
        sm = smile_from_dist(dist)
        base = ct.get(venue, {}).get(surf, {}).get(sm, {})
        season = season_from_month(mm)
        # 脚質バイアスは重賞挙動に近い上位クラス優先で採用(OP重賞>3勝>…)。各クラス内は当季優先。
        CLASS_PRI = ["OP/重賞", "3勝", "2勝", "1勝", "未勝利", "新馬"]
        best = None
        for cls in CLASS_PRI:
            seasons = base.get(cls, {})
            ok = {sk: c for sk, c in seasons.items()
                  if isinstance(c, dict) and c.get("脚質ランク") and c.get("n", 0) >= 25}
            if not ok:
                continue
            sk = season if season in ok else max(ok, key=lambda k: ok[k]["n"])
            best = (ok[sk]["n"], cls, sk, ok[sk]); break
        # 好調血統は OP/重賞 を優先
        op_blood = None
        for sk, cell in base.get("OP/重賞", {}).items():
            if isinstance(cell, dict) and cell.get("好調血統_父") and sk in (season, "通年"):
                op_blood = {x["種牡馬"]: x["勝率"] for x in cell["好調血統_父"]}
        if best is None:
            return None
        n, cls, sk, cell = best
        blood = op_blood or {x["種牡馬"]: x["勝率"] for x in cell.get("好調血統_父", [])}
        return {
            "n": n, "smile": sm, "season": sk, "src_class": cls,
            "from_jusho": (cls == "OP/重賞"),
            "脚質ランク": cell.get("脚質ランク", [])[:4],
            "好調枠番": cell.get("好調枠番", [])[:3],
            "好調血統_父": blood, "blood_from_jusho": op_blood is not None,
        }
    except Exception:
        pass
    return None


def training_layer(card, infomap):
    """各馬に 調教(坂路/WC)＋血統 を付与。坂路4F時計で field 内ランクも。"""
    rows = card["horses"]
    t4 = []
    for h in rows:
        ex = infomap.get(h["ban"], {})
        da = EX._f(ex.get("trnH_days_ago"))
        valid = da is not None and 0 <= da <= 30      # 14日以内追切が正。30超は古い記録=無効
        h["trn_days_ago"] = int(da) if valid else None
        # 無効(古い/欠損)な追切は坂路時計も出さない（この一戦の仕上げでない）
        h["trn_hanro_4f"] = EX._f(ex.get("trnH_Time1")) if valid else None
        h["trn_hanro_last1f"] = EX._f(ex.get("trnH_Lap1")) if valid else None
        h["trn_wc_5f"] = EX._f(ex.get("trnW_5F"))
        h["sire"] = ex.get("種牡馬"); h["broodmare_sire"] = ex.get("母父馬"); h["sire_type"] = ex.get("父タイプ名")
        if h["trn_hanro_4f"] is not None:
            t4.append((h["ban"], h["trn_hanro_4f"]))
    # 坂路4F 速い順ランク（小さいほど速い）
    rank = {b: i + 1 for i, (b, _) in enumerate(sorted(t4, key=lambda x: x[1]))}
    for h in rows:
        h["trn_hanro_rank"] = rank.get(h["ban"])
        h["trn_sharp"] = bool(h["trn_hanro_rank"] and h["trn_hanro_rank"] <= max(2, len(t4) // 5))


def build_jusho(rid, g, shared, info_df, extra_df):
    """explain_race の土台カード ＋ 重賞層。"""
    card = EX.build(str(rid), g, shared, info_df)
    info = extra_df[extra_df[COL_RID] == str(rid)]
    infomap = {int(r["ban"]): r.to_dict() for _, r in info.iterrows()}
    meta = info.iloc[0] if not info.empty else None

    training_layer(card, infomap)

    r = card["race"]
    cls_raw = (meta["クラス名"] if meta is not None else r.get("class")) or ""
    r["grade"] = GRADE_RANK.get(str(cls_raw).strip(), str(cls_raw).strip())
    r["is_jusho"] = is_jusho_class(cls_raw)
    cb = course_bias(r["venue"], r["surface"], r["distance"], str(rid)[4:6])
    r["track_bias"] = cb

    # 当コース好調血統に一致する出走馬
    if cb:
        hot = cb["好調血統_父"]
        for h in card["horses"]:
            sw = hot.get(h.get("sire"))
            h["sire_hot_pct"] = sw  # この種牡馬が当コースOP/重賞で示す勝率(なければNone)

    # 重賞メモ: 調教抜群＋好調血統の妙味馬（印上位以外で）
    notes = []
    if cb and cb["脚質ランク"]:
        top_k = cb["脚質ランク"][0]
        src = "OP重賞実績" if cb["from_jusho"] else f"全クラス傾向({cb['src_class']})"
        notes.append(f"当コース({r['venue']}{r['surface']}{cb['smile']})の{src}は"
                     f"【{top_k['脚質']}】勝率{top_k['勝率']}%が最上位。好調枠={'・'.join(cb['好調枠番'])}。")
    # 馬券構築の地図: 馬場バイアス × ◎の脚質/枠 を噛み合わせる
    bb = r.get("baba_bias")
    if bb and card["horses"]:
        hon = card["horses"][0]; hk = hon.get("kyaku")
        front_strong = bb["front_edge"] >= 24
        if hk in ("逃げ", "先行"):
            fit = "前残り傾向に脚質が噛み合う＝軸の信頼度↑" if front_strong else "前で立ち回れる脚質で減点なし"
        elif hk in ("差し", "追込"):
            fit = "前残り傾向の馬場で脚質は展開待ち＝相手に先行馬を厚めに" if front_strong else "差しが届く余地はある馬場"
        else:
            fit = None
        wk = bb["waku_out_minus_in"]
        waku = ("／外枠有利、外の先行も相手候補" if wk >= 1.0 else ("／内枠の立ち回り重視" if wk <= -1.0 else ""))
        if fit:
            notes.insert(0, f"馬券構築: 馬場は[{bb['band']}/前残り度{bb['front_edge']:+.1f}pp]。"
                            f"◎{hon['name']}({hk})は{fit}{waku}。")
    sharp_hot = [h for h in card["horses"] if h.get("trn_sharp") and h.get("sire_hot_pct")]
    for h in sharp_hot[:2]:
        notes.append(f"{h['mark']}{h['name']}: 坂路4F {h['trn_hanro_4f']}秒(field {h['trn_hanro_rank']}位)＋"
                     f"父{h['sire']}は当コース好調(勝率{h['sire_hot_pct']}%)＝仕上がり・適性の二重◎。")
    card["jusho_notes"] = notes
    return card


def render_jusho(card):
    r = card["race"]; L = []
    bar = "━" * 80
    L.append(bar)
    L.append(f" 🏆 [{r.get('grade','')}] {r['name']}  {r['venue']}{r['surface']}{r['distance']}m  {r['field_size']}頭")
    L.append(f"    レースレベル(ELO) {r['race_level_elo']}  ｜  {r['baba']}")
    if r.get("baba_bias"):
        L.append(f"    🏇馬場傾向: {r['baba_bias']['line']}")
    L.append(f"    展開: {r['pace']}")
    cb = r.get("track_bias")
    if cb and cb["脚質ランク"]:
        ks = " / ".join(f"{x['脚質']}{x['勝率']}%" for x in cb["脚質ランク"])
        src = "OP重賞" if cb["from_jusho"] else f"{cb['src_class']}基準・全クラス傾向"
        L.append(f"    ﾄﾗｯｸﾊﾞｲｱｽ({src} {cb['n']}R/{cb['season']}): 脚質 {ks}  好調枠 {'・'.join(cb['好調枠番'])}")
    L.append(bar)
    L.append(f"{'印':2s}{'番':>2s} {'馬名':13s}{'勝率':>6s}{'複勝':>6s}{'実ｵｯｽﾞ':>7s}{'EV':>5s}{'RTG':>5s}{'脚質':4s}"
             f"{'坂路4F':>7s}{'1F':>5s}{'追日':>4s}{'父(当ｺｰｽ)':>12s}")
    for h in card["horses"]:
        nm = (str(h["name"])[:12] + "…") if len(str(h["name"])) > 13 else str(h["name"])
        ro = f"{h['real_odds']}" if h["real_odds"] else "-"
        ev = f"{h['ev']}" if h["ev"] else "-"
        h4 = f"{h['trn_hanro_4f']}" if h.get("trn_hanro_4f") else "-"
        h4 = (h4 + ("◎" if h.get("trn_sharp") else "")) if h.get("trn_hanro_4f") else "-"
        l1 = f"{h['trn_hanro_last1f']}" if h.get("trn_hanro_last1f") else "-"
        da = f"{h['trn_days_ago']}" if h.get("trn_days_ago") is not None else "-"
        sire = str(h.get("sire") or "")[:8]
        if h.get("sire_hot_pct"):
            sire = f"{sire}({h['sire_hot_pct']}%)"
        L.append(f"{h['mark']:2s}{h['ban']:>2d} {nm:13s}{h['p_win_pct']:>5.1f}%{h['p_top3_pct']:>5.1f}%"
                 f"{ro:>7s}{ev:>5s}{str(h['elo'] or '-'):>5s}{h['kyaku']:4s}{h4:>7s}{l1:>5s}{da:>4s}  {sire:>10s}")
    if card.get("comments_top3"):
        L.append("\n【印上位の解説】")
        for c in card["comments_top3"]:
            L.append("  ・" + c)
    if card.get("jusho_notes"):
        L.append("\n【重賞プロトコル注目点】")
        for nt in card["jusho_notes"]:
            L.append("  ・" + nt)
    if card.get("ana_pick"):
        L.append("\n【特選穴馬】\n  ・" + card["ana_pick"])
    if card.get("result_for_demo"):
        rr = card["result_for_demo"]
        L.append(f"\n（参考・確定）1着{rr['win_ban']} 2着{rr['plc_ban']} 3着{rr['sho_ban']} / 単勝{rr['tansho_pay']}円")
    return "\n".join(L)


def jusho_rids_for_date(sc, extra_df, date):
    """指定日の 重賞＋OP/L の rid 一覧（cache×master class）。"""
    rids = sc[sc[COL_RID].str.startswith(str(date))][COL_RID].unique()
    cls = extra_df.drop_duplicates(COL_RID).set_index(COL_RID)["クラス名"].to_dict()
    return [r for r in rids if is_jusho_class(cls.get(str(r)))]


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    sc = pd.read_parquet(EX.SCORE_CACHE); sc[COL_RID] = sc[COL_RID].astype(str)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    if arg and len(arg) == 8:                      # 日付モード
        extra_df = load_extra(arg)
        rids = jusho_rids_for_date(sc, extra_df, arg)
        if not rids:
            print(f"{arg} に重賞＋OP/L は無い"); return
        sub = sc[sc[COL_RID].isin(set(rids))].copy()
        sub["rid16"] = sub[COL_RID].str.replace(r"\D", "", regex=True).str[:16]
        sub["ban"] = pd.to_numeric(sub[COL_BAN], errors="coerce").fillna(0).astype(int)
        sub = EX.join_feats(sub)
        shared = EX.load_shared(); info_df = EX.load_master_subset(arg)
        for rid, gg in sub.groupby(COL_RID, sort=False):
            card = build_jusho(str(rid), gg, shared, info_df, extra_df)
            (OUTDIR / f"{rid}.json").write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
            print(render_jusho(card)); print()
        return

    # 単一 rid モード（無指定なら cache 内の最新重賞）
    if not arg:
        extra_all = load_extra()
        cls = extra_all.drop_duplicates(COL_RID).set_index(COL_RID)["クラス名"].to_dict()
        cand = sorted([r for r in sc[COL_RID].unique() if is_jusho_class(cls.get(str(r)))])
        arg = cand[-1] if cand else None
        if arg is None:
            print("cacheに重賞が無い"); return
    rid, g = EX.load_race(arg); g = EX.join_feats(g)
    date = str(rid)[:8]
    shared = EX.load_shared(); info_df = EX.load_master_subset(date); extra_df = load_extra(date)
    card = build_jusho(rid, g, shared, info_df, extra_df)
    (OUTDIR / f"{rid}.json").write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    print(render_jusho(card))
    print(f"\n[saved] {OUTDIR / (str(rid) + '.json')}")


if __name__ == "__main__":
    main()
