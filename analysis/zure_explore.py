"""ズレ警告の探索的分析（2026-11-18、判定の再集計ではない）。

検証期間（2026-08-01〜11-15）の D3'（モデル 6 位以下 かつ PW 単独 1 位）88 頭と対照群について、
人気・オッズ、順位帯、レース種別・ローテ、PW のマージン、モデル側の事情を記述する。
判定（PASS/FAIL）には一切使わない。結果は docs/research/ZURE_EXPLORATORY_20261118.md。

  python -m analysis.zure_explore  → reports/zure_validation/exploratory.json
"""
from __future__ import annotations

import csv
import io
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import zure_validation as zv
import build_site

ROOT = zv.ROOT
CLASS_LEVEL = {"新馬": 0, "未勝利": 1, "1勝": 2, "2勝": 3, "3勝": 4, "オープン": 5, "OP(L)": 5,
               "G3": 6, "G2": 7, "G1": 8}


def kind(k: str) -> str:
    if k == "新馬":
        return "新馬"
    if k == "未勝利":
        return "未勝利"
    if k in ("1勝", "2勝", "3勝"):
        return "条件(1-3勝)"
    if k in ("オープン", "OP(L)"):
        return "OP・L"
    if k in ("G1", "G2", "G3"):
        return "重賞"
    return "その他"


def final_odds(date: str) -> dict[tuple[str, int], float]:
    """kekka の単勝欄: 1 着は配当 (円/100 円)、それ以外は "(xx.x)" の確定オッズ。"""
    p = build_site.KEKKA_DIR / f"{date}.csv"
    out = {}
    if not p.exists():
        return out
    rd = csv.reader(io.StringIO(build_site._decode(p.read_bytes())))
    next(rd, None)
    for row in rd:
        if len(row) < 9:
            continue
        rid, uma, t = row[7].strip()[:16], build_site._int(row[4]), row[8].strip()
        if uma is None or not t:
            continue
        try:
            v = float(t.strip("()")) if t.startswith("(") else float(t) / 100.0
        except ValueError:
            continue
        out[(rid, uma)] = v
    return out


def extras() -> pd.DataFrame:
    """公開 JSON から人気・近走・PW の値などを 1 頭 1 行で。"""
    rows = []
    for f in zv.day_files():
        d = json.loads(f.read_text(encoding="utf-8"))
        date = d["date"]
        if date < "20260801":
            continue
        fo = final_odds(date)
        dt = pd.Timestamp(date)
        for r in d["races"]:
            rid = str(r["race_id"])[:16]
            course = r.get("course") or ""
            surf, dist = course[:1], build_site._int("".join(ch for ch in course if ch.isdigit()))
            for h in r.get("horses") or []:
                u = int(h["umaban"])
                hist = h.get("history") or {}
                runs = (h.get("speed") or {}).get("runs") or []
                hr = (hist.get("runs") or [])
                last = runs[0] if runs else None
                lay = (dt - pd.Timestamp(last[0])).days if last else None
                prev_k = last[2] if last else None
                prev_surf = (hr[0].get("td") if hr else None)
                prev_dist = (hr[0].get("dist") if hr else None)
                rows.append({
                    "race_id": rid, "umaban": u, "kind": kind(r.get("klass") or ""),
                    "field": r.get("field_size"), "ninki": h.get("ninki"), "fodds": fo.get((rid, u)),
                    "p_win": h.get("p_win"), "pw": h.get("pw"), "n_runs": hist.get("n_runs"),
                    "avg_pos": hist.get("avg_pos"), "best_pos": hist.get("best_pos"),
                    "last_pos": hr[0].get("pos") if hr else None,
                    "layoff_days": lay,
                    "cls_change": (CLASS_LEVEL.get(r.get("klass")) - CLASS_LEVEL[prev_k])
                    if (prev_k in CLASS_LEVEL and r.get("klass") in CLASS_LEVEL) else None,
                    "surf_change": (prev_surf is not None and surf in ("芝", "ダ") and prev_surf != surf),
                    "dist_change": (dist - prev_dist) if (dist and prev_dist) else None,
                })
    return pd.DataFrame(rows)


def wilson(k, n):
    return [None if x is None else round(x * 100, 1) for x in zv.wilson(int(k), int(n))]


def rate_block(g: pd.DataFrame) -> dict:
    return {"n": len(g), "place%": wilson(g["place"].sum(), len(g)), "win%": wilson(g["win"].sum(), len(g))}


def main():
    zv.build_site.KEKKA_DIR = zv._MAIN_KEKKA if zv._MAIN_KEKKA.is_dir() else zv.build_site.KEKKA_DIR
    df, info = zv._phase_frame("validation")
    ex = extras()
    df = df.merge(ex, on=["race_id", "umaban"], how="left")
    # 最終人気 = 確定オッズのレース内順位
    df["fninki"] = df.groupby("race_id")["fodds"].rank(method="min")
    low = df[df["ai_rank"] >= 6]
    d3 = low[low["D3"]]
    ctrl = low[low["control"]]
    out = {"info": info, "n_d3": len(d3), "n_ctrl": len(ctrl)}

    # ---- B4 人気・オッズ
    def dist(s, bins, labels):
        c = pd.cut(s, bins=bins, labels=labels, right=True)
        return (c.value_counts(normalize=True).reindex(labels).fillna(0) * 100).round(1).to_dict()
    nb, nl = [0, 3, 5, 8, 12, 99], ["1-3人気", "4-5人気", "6-8人気", "9-12人気", "13人気-"]
    ob, ol = [0, 5, 10, 20, 50, 9999], ["-5倍", "5-10倍", "10-20倍", "20-50倍", "50倍-"]
    # 対照は順位帯の構成を D3' に合わせて重み付け
    w = d3["band"].value_counts(normalize=True)
    cw = ctrl["band"].map(w / ctrl["band"].value_counts(normalize=True)).fillna(0)
    def wdist(s, wt, bins, labels):
        c = pd.cut(s, bins=bins, labels=labels, right=True)
        t = pd.Series(wt.values, index=c.values).groupby(level=0, observed=False).sum()
        return (t.reindex(labels).fillna(0) / t.sum() * 100).round(1).to_dict()
    out["B4"] = {
        "朝人気_D3": dist(d3["ninki"], nb, nl), "朝人気_対照(帯構成合わせ)": wdist(ctrl["ninki"], cw, nb, nl),
        "確定オッズ_D3": dist(d3["fodds"], ob, ol), "確定オッズ_対照(帯構成合わせ)": wdist(ctrl["fodds"], cw, ob, ol),
        "朝人気の中央値": {"D3": float(d3["ninki"].median()), "対照": float(ctrl["ninki"].median())},
        "確定オッズの中央値": {"D3": float(d3["fodds"].median()), "対照": float(ctrl["fodds"].median())},
        "最終人気の中央値": {"D3": float(d3["fninki"].median()), "対照": float(ctrl["fninki"].median())},
    }
    # 人気で揃えた期待値: 同じ検証期間の全馬 (全順位) の「確定人気ごとの複勝率」から D3' の期待複勝率
    allh = df[df["fninki"].notna()]
    by_fn = allh.groupby(allh["fninki"].clip(upper=14))["place"].mean()
    exp_fn = d3["fninki"].clip(upper=14).map(by_fn)
    by_mn = df[df["ninki"].notna()].groupby(df["ninki"].clip(upper=14))["place"].mean()
    exp_mn = d3["ninki"].clip(upper=14).map(by_mn)
    # 確定オッズ帯（細かく）で揃えた期待値
    ob2 = [0, 3, 5, 7, 10, 15, 20, 30, 50, 100, 9999]
    allh_b = pd.cut(allh["fodds"], ob2)
    by_ob = allh.groupby(allh_b, observed=False)["place"].mean()
    exp_ob = pd.cut(d3["fodds"], ob2).map(by_ob).astype(float)
    # 対照も同じ物差しで
    ctrl_exp_fn = ctrl["fninki"].clip(upper=14).map(by_fn)
    out["B4"]["人気で揃えた比較"] = {
        "D3'_実際の複勝率": round(d3["place"].mean() * 100, 1),
        "D3'_確定人気から見込む複勝率": round(exp_fn.mean() * 100, 1),
        "D3'_朝人気から見込む複勝率": round(exp_mn.mean() * 100, 1),
        "D3'_確定オッズ帯から見込む複勝率": round(exp_ob.mean() * 100, 1),
        "対照_実際": round(ctrl["place"].mean() * 100, 1),
        "対照_確定人気から見込む": round(ctrl_exp_fn.mean() * 100, 1),
        "注": "見込み = 検証期間の全出走馬の、確定人気（または朝人気・確定オッズ帯）ごとの複勝率の平均を当てはめた値",
    }
    # 人気帯ごとの D3' と対照
    tab = []
    for lab, lo, hi in (("1-5人気", 1, 5), ("6-8人気", 6, 8), ("9人気-", 9, 99)):
        a = d3[(d3["fninki"] >= lo) & (d3["fninki"] <= hi)]
        c = ctrl[(ctrl["fninki"] >= lo) & (ctrl["fninki"] <= hi)]
        tab.append({"確定人気": lab, "D3'": rate_block(a), "対照": rate_block(c)})
    out["B4"]["確定人気帯ごと"] = tab

    # ---- B5 順位帯
    out["B5"] = {}
    for lab, lo, hi in (("6-8位", 6, 8), ("9位以下", 9, 99), ("9-12位", 9, 12), ("13位以下", 13, 99)):
        a = d3[(d3["ai_rank"] >= lo) & (d3["ai_rank"] <= hi)]
        c = ctrl[(ctrl["ai_rank"] >= lo) & (ctrl["ai_rank"] <= hi)]
        rate = c.groupby("band")["place"].mean()
        mix = a["band"].value_counts(normalize=True)
        cp = float((mix * rate.reindex(mix.index)).sum()) if len(a) else None
        blk = rate_block(a)
        blk.update({"対照の複勝率%": None if cp is None else round(cp * 100, 1),
                    "lift": None if not cp else round(a["place"].mean() / cp, 2)})
        out["B5"][lab] = blk

    # ---- B6 偏り
    def share(g, col, cats=None):
        v = g[col].value_counts(normalize=True, dropna=False) * 100
        return {str(k): round(x, 1) for k, x in v.items()}
    def cut_share(g, col, bins, labels):
        c = pd.cut(g[col], bins=bins, labels=labels)
        v = c.value_counts(normalize=True, dropna=False) * 100
        return {str(k): round(x, 1) for k, x in v.reindex(list(labels) + [np.nan]).fillna(0).items()}
    lb, ll = [-1, 27, 69, 181, 9999], ["中3週以内", "中4-9週", "10-25週(休み明け)", "26週超"]
    out["B6"] = {}
    for name, g in (("D3'", d3), ("対照", ctrl)):
        out["B6"][name] = {
            "レース種別": share(g, "kind"),
            "間隔": cut_share(g, "layoff_days", lb, ll),
            "クラス変更": {"昇級": round((g["cls_change"] > 0).mean() * 100, 1),
                       "同級": round((g["cls_change"] == 0).mean() * 100, 1),
                       "降級": round((g["cls_change"] < 0).mean() * 100, 1),
                       "不明": round(g["cls_change"].isna().mean() * 100, 1)},
            "芝ダ替わり%": round(g["surf_change"].fillna(False).astype(bool).mean() * 100, 1),
            "距離変更": {"短縮200m以上": round((g["dist_change"] <= -200).mean() * 100, 1),
                     "±100m以内": round((g["dist_change"].abs() <= 100).mean() * 100, 1),
                     "延長200m以上": round((g["dist_change"] >= 200).mean() * 100, 1)},
            "キャリア中央値": float(g["n_runs"].median()),
            "近走平均着順の中央値": float(g["avg_pos"].median()),
            "前走着順の中央値": float(g["last_pos"].median()),
            "AI勝率の中央値%": round(float(g["p_win"].median()) * 100, 2),
        }
    # D3' の中での区分ごとの複勝率（参考）
    sub = {}
    for name, mask in (("新馬", d3["kind"] == "新馬"), ("未勝利", d3["kind"] == "未勝利"),
                       ("条件(1-3勝)", d3["kind"] == "条件(1-3勝)"),
                       ("OP・L・重賞", d3["kind"].isin(["OP・L", "重賞"])),
                       ("休み明け(10週以上)", d3["layoff_days"] >= 70), ("間隔9週以内", d3["layoff_days"] < 70),
                       ("昇級", d3["cls_change"] > 0), ("同級・降級", d3["cls_change"] <= 0),
                       ("芝ダ替わり", d3["surf_change"].fillna(False).astype(bool)),
                       ("距離変更200m以上", d3["dist_change"].abs() >= 200)):
        sub[name] = rate_block(d3[mask])
    out["B6"]["D3'内の区分別"] = sub

    # ---- B7 マージン（PW 1 位 − 2 位）
    pw = df[df["pw"].notna()]
    top2 = pw.sort_values("pw", ascending=False).groupby("race_id")["pw"].apply(lambda s: s.iloc[0] - s.iloc[1] if len(s) > 1 else np.nan)
    d3m = d3.assign(margin=d3["race_id"].map(top2))
    out["B7"] = {"マージン中央値": round(float(d3m["margin"].median()), 2)}
    for lab, lo, hi in (("<2点", 0, 2), ("2-5点", 2, 5), ("5-10点", 5, 10), ("10点以上", 10, 999)):
        g = d3m[(d3m["margin"] > lo) & (d3m["margin"] <= hi)] if lo else d3m[d3m["margin"] <= hi]
        out["B7"][lab] = rate_block(g)
    q = d3m["margin"].quantile([1 / 3, 2 / 3]).tolist()
    for lab, g in (("下位1/3", d3m[d3m["margin"] <= q[0]]),
                   ("中位1/3", d3m[(d3m["margin"] > q[0]) & (d3m["margin"] <= q[1])]),
                   ("上位1/3", d3m[d3m["margin"] > q[1]])):
        out["B7"][f"三分位_{lab}"] = rate_block(g) | {"範囲": [round(float(g["margin"].min()), 2), round(float(g["margin"].max()), 2)]}

    # ---- C8（観察できる範囲）: PW 単独 1 位の馬を「AI も 5 位以内」と「AI 6 位以下 = D3'」で比べる
    pw1 = df[df["race_id"].map(df[df["pw_rank"] == 1].groupby("race_id").size()).eq(1) & (df["pw_rank"] == 1)]
    agree = pw1[pw1["ai_rank"] <= 5]
    out["C8"] = {}
    for name, g in (("PW単独1位×AI5位以内", agree), ("D3'(PW単独1位×AI6位以下)", d3)):
        out["C8"][name] = {
            "n": len(g), "複勝率%": round(g["place"].mean() * 100, 1),
            "キャリア中央値": float(g["n_runs"].median()), "近走平均着順の中央値": float(g["avg_pos"].median()),
            "前走着順の中央値": float(g["last_pos"].median()), "最高着順の中央値": float(g["best_pos"].median()),
            "休み明け(10週以上)%": round((g["layoff_days"] >= 70).mean() * 100, 1),
            "昇級%": round((g["cls_change"] > 0).mean() * 100, 1),
            "降級%": round((g["cls_change"] < 0).mean() * 100, 1),
            "芝ダ替わり%": round(g["surf_change"].fillna(False).astype(bool).mean() * 100, 1),
            "前走6着以下%": round((g["last_pos"] >= 6).mean() * 100, 1),
            "確定オッズ中央値": float(g["fodds"].median()),
        }
    out["C8"]["PW単独1位の総数"] = len(pw1)
    p = ROOT / "reports" / "zure_validation" / "exploratory.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
