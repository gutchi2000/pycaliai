# -*- coding: utf-8 -*-
"""GO 判定 × デイリーボード一致数 × 難しさ の買い目ルール — 探索・再構成 (2026-11-20)。

★ 探索・再構成。前向き記録と混ぜない。ルールは上流で事前に固定 (変更・追加しない)。
  - ボードの 3 ゲージ (速度 / PW / 推定人気) は発走前に公開された値が無い (初掲載 2026-11-15 22:33)。
    ここでは公開サイト JSON (site/data/{date}.json) の後から作り直した値を使う。
  - GO 判定・難しさ・印・◎の単勝オッズは凍結版 bundle (reports/cowork_input_frozen/) の値。
  - 払戻は data/kekka/{date}.csv と wide_kekka.csv の確定払戻だけ。オッズからの換算はしない。
  - 決済は build_site.settle_bet / parse_kekka をそのまま使う (取消=返還、着順0で最終オッズ有り=決済不能、
    同着は JRA ルールで的中組を列挙)。

python -m analysis.board_go_recon_backtest [--kekka-dir E:/PyCaLiAI/data/kekka] [--out reports/board_go_recon]
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import build_site as B  # noqa: E402
import betting_judgment as bj  # noqa: E402

FROZEN = BASE / "reports" / "cowork_input_frozen"
SITE = BASE / "site" / "data"
DATE_FROM = "20260613"   # UMAMI ゲート入りの GO 判定が bundle に載った最初の開催日 (06-12 導入)
DATE_TO = "20261115"
TYPES = ("ワイド", "馬連", "三連複")
PARTNER = ("○", "▲", "△1", "△2")
NBOOT = 4000


def races_of(b):
    return b["races"] if isinstance(b["races"], list) else list(b["races"].values())


def tickets(hon, part):
    """part = {"○": u, "▲": u, "△1": u, "△2": u} → {券種: [(selection, 相手ラベル)]}"""
    out = {"ワイド": [], "馬連": [], "三連複": []}
    for lb in PARTNER:
        out["ワイド"].append((f"{hon}-{part[lb]}", lb))
        out["馬連"].append((f"{hon}-{part[lb]}", lb))
    for a, b in combinations(PARTNER, 2):
        out["三連複"].append((f"{hon}-{part[a]}-{part[b]}", a + b))
    return out


def gauges(site_race):
    hs = site_race.get("horses", [])   # site JSON は scratched を持たない (= 公開時と同じく全頭)
    sp = [h for h in hs if (h.get("speed") or {}).get("v") is not None]
    pw = [h for h in hs if h.get("pw") is not None]
    er = [h for h in hs if (h.get("speed") or {}).get("er") is not None]
    if not (sp and pw and er):
        return None
    # board.js と同じ: 先頭から見て厳密に大きい (小さい) ときだけ入れ替える
    si1 = sp[0]
    for h in sp[1:]:
        if h["speed"]["v"] > si1["speed"]["v"]:
            si1 = h
    pw1 = pw[0]
    for h in pw[1:]:
        if h["pw"] > pw1["pw"]:
            pw1 = h
    er1 = er[0]
    for h in er[1:]:
        if h["speed"]["er"] < er1["speed"]["er"]:
            er1 = h
    return {"si": si1["umaban"], "pw": pw1["umaban"], "er": er1["umaban"]}


def collect(kekka_dir: Path):
    B.KEKKA_DIR = kekka_dir
    wide = B.parse_wide_kekka()
    rows, excl, repro = [], {}, {"n": 0, "has_value_mismatch": 0, "hardness_mismatch": 0,
                                 "category_mismatch": 0, "site_hon_mismatch": 0}

    def ex(reason, k=1):
        excl[reason] = excl.get(reason, 0) + k

    kdates = sorted(p.stem for p in kekka_dir.glob("2026????.csv"))
    for d in kdates:
        if not (DATE_FROM <= d <= DATE_TO):
            continue
        fz = FROZEN / f"{d}_bundle.json"
        sp = SITE / f"{d}.json"
        res_all = B.parse_kekka(d, wide)
        if not fz.exists():
            ex(f"凍結 bundle なし（{d}）", len(res_all)); continue
        if not sp.exists():
            ex(f"サイト JSON なし（{d}）", len(res_all)); continue
        bun = json.loads(fz.read_text(encoding="utf-8"))
        model = bun.get("model")
        site = {r["race_id"][:16]: r for r in json.loads(sp.read_text(encoding="utf-8"))["races"]}
        brs = {str(r["race_id"])[:16]: r for r in races_of(bun)}
        for rid in sorted(set(res_all) - set(brs)):
            ex("結果はあるが bundle に無い（障害・新馬などの対象外レース）")
        for rid, r in brs.items():
            res = res_all.get(rid)
            if not res:
                ex("結果なし"); continue
            j = r.get("buy_judgment") or {}
            if not j.get("category") or j.get("chaos_pct") is None:
                ex("GO 判定なし"); continue
            hs = r.get("horses", [])
            mk = {}
            for h in hs:
                m = h.get("mark") or ""
                mk.setdefault(m, []).append(int(h["umaban"]))
            if len(mk.get("◎", [])) != 1 or len(mk.get("〇", [])) != 1 or len(mk.get("▲", [])) != 1 \
                    or len(mk.get("△", [])) != 2:
                ex("印 ◎〇▲△△ が揃わない"); continue
            hon = mk["◎"][0]
            part = {"○": mk["〇"][0], "▲": mk["▲"][0], "△1": mk["△"][0], "△2": mk["△"][1]}
            sr = site.get(rid)
            if not sr:
                ex("サイト JSON にレースなし"); continue
            g = gauges(sr)
            if g is None:
                ex("ボードのゲージが欠ける（速度・PW・推定人気のどれか）"); continue
            shon = next((h["umaban"] for h in sr["horses"] if h.get("mark") == "◎"), None)
            # GO 判定の再現チェック (保存値を正とし、食い違いは数えるだけ)
            repro["n"] += 1
            if shon != hon:
                repro["site_hon_mismatch"] += 1
            if bj.classify_hardness(j["chaos_pct"]) != j.get("hardness"):
                repro["hardness_mismatch"] += 1
            hv = len(bj.extract_value_horses(hs)) > 0
            if hv != bool(j.get("has_value")):
                repro["has_value_mismatch"] += 1
            if bj.decide_strategy(j.get("hardness"), bool(j.get("has_value")))["category"] != j["category"]:
                repro["category_mismatch"] += 1
            # 決済 (1 点 100 円、点ごと)
            tk = tickets(hon, part)
            settled = {}
            bad = False
            for t, lst in tk.items():
                pts = []
                for sel, lb in lst:
                    s = B.settle_bet(t, sel, 100.0, res)
                    if not s["settled"]:
                        bad = s.get("reason") or "unsettled"; break
                    pts.append((lb, s["received"], s["is_win"]))
                if bad:
                    break
                settled[t] = pts
            for t, sel in (("単勝", str(hon)), ("複勝", str(hon))):
                if bad:
                    break
                s = B.settle_bet(t, sel, 100.0, res)
                if not s["settled"]:
                    bad = s.get("reason") or "unsettled"; break
                settled[t] = [("◎", s["received"], s["is_win"])]
            if bad:
                ex(f"決済不能（{bad}: 着順0で最終オッズのある馬を含む／払戻欠損）"); continue
            vh = [int(v["umaban"]) for v in (j.get("value_horses") or [])]
            if hon in vh:
                vloc = "◎本人を含む"
            elif set(vh) & set(part.values()):
                vloc = "相手印を含む"
            elif vh:
                vloc = "印外のみ"
            else:
                vloc = "妙味馬なし"
            ho = (h for h in hs if int(h["umaban"]) == hon)
            hh = next(ho)
            pos = (res.get("order") or {}).get(str(hon))
            rows.append(dict(
                date=d, rid=rid, model=model, go=j["category"] == "go", hard=j["hardness"],
                chaos=float(j["chaos_pct"]),
                agree=int(g["si"] == hon) + int(g["pw"] == hon) + int(g["er"] == hon),
                gset=("人気" if g["er"] == hon else "") + ("PW" if g["pw"] == hon else "")
                     + ("速度" if g["si"] == hon else ""),
                hon=hon, part=part, hon_top3=bool(pos and 1 <= pos <= 3),
                hon_odds=hh.get("tansho_odds"), vloc=vloc, settled=settled,
            ))
    return rows, excl, repro


def summ(sub, t, boot_seed=0):
    n = len(sub)
    if n == 0:
        return dict(n=0)
    stake = np.array([100.0 * len(r["settled"][t]) for r in sub])
    ret = np.array([sum(p[1] for p in r["settled"][t]) for r in sub])
    hitr = np.array([any(p[2] for p in r["settled"][t]) for r in sub])
    hitp = int(sum(sum(1 for p in r["settled"][t] if p[2]) for r in sub))
    roi = ret.sum() / stake.sum()
    rng = np.random.default_rng(boot_seed)
    idx = rng.integers(0, n, (NBOOT, n))
    bs = ret[idx].sum(1) / stake[idx].sum(1)
    imax = int(np.argmax(ret))
    roi_x = (ret.sum() - ret[imax]) / (stake.sum() - stake[imax]) if n > 1 else float("nan")
    od = [r["hon_odds"] for r in sub if r["hon_odds"]]
    out = dict(n=n, stake=int(stake.sum()), ret=round(float(ret.sum()), 1), roi=round(100 * roi, 1),
               ci=[round(100 * np.percentile(bs, 2.5), 1), round(100 * np.percentile(bs, 97.5), 1)],
               race_hit=round(100 * hitr.mean(), 1), hit_pts=hitp,
               hon_top3=round(100 * np.mean([r["hon_top3"] for r in sub]), 1),
               hon_odds_med=round(float(np.median(od)), 1) if od else None,
               chaos_med=round(float(np.median([r["chaos"] for r in sub])), 3),
               roi_drop_max=round(100 * roi_x, 1), max_race=sub[imax]["rid"], max_ret=round(float(ret[imax]), 1))
    out["candidate"] = bool(n >= 50 and out["roi"] > 100 and out["ci"][0] > 100 and out["roi_drop_max"] > 100)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kekka-dir", default=str(BASE / "data" / "kekka"))
    ap.add_argument("--out", default=str(BASE / "reports" / "board_go_recon"))
    a = ap.parse_args()
    rows, excl, repro = collect(Path(a.kekka_dir))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    G = [r for r in rows if r["go"]]
    HARD3 = ("固い", "標準", "荒れ")
    rep = {"label": "探索・再構成（前向き記録とは別）", "n_races": len(rows),
           "dates": sorted({r["date"] for r in rows}), "models": {}, "excluded": excl, "go_repro": repro}
    for r in rows:
        rep["models"][r["model"]] = rep["models"].get(r["model"], 0) + 1
    # クロス表
    cross = {}
    for go in (True, False):
        for k in range(4):
            for hd in HARD3:
                cross[f"{'GO' if go else '非GO'}|{k}|{hd}"] = sum(
                    1 for r in rows if r["go"] == go and r["agree"] == k and r["hard"] == hd)
    rep["cross"] = cross

    def cellset(name, sub):
        return {"name": name, "n": len(sub), "by_type": {t: summ(sub, t) for t in TYPES + ("単勝", "複勝")}}

    main_cells = []
    for k in (3, 2, 1):
        for hd in ("固い", "標準"):
            main_cells.append(cellset(f"GO∧一致{k}∧{hd}", [r for r in G if r["agree"] == k and r["hard"] == hd]))
    for k in (3, 2, 1):
        main_cells.append(cellset(f"GO∧一致{k}（難しさ不問）", [r for r in G if r["agree"] == k]))
    rep["main"] = main_cells
    dec = [cellset("GOのみ", G)]
    for hd in ("固い", "標準"):
        dec.append(cellset(f"GO∧一致0∧{hd}", [r for r in G if r["agree"] == 0 and r["hard"] == hd]))
    dec.append(cellset("GO∧一致0（難しさ不問）", [r for r in G if r["agree"] == 0]))
    for k in (3, 2, 1, 0):
        dec.append(cellset(f"一致{k}（GO不問）", [r for r in rows if r["agree"] == k]))
    for hd in ("固い", "標準"):
        dec.append(cellset(f"GO∧{hd}", [r for r in G if r["hard"] == hd]))
    rep["decomp"] = dec
    rep["agree2p"] = [cellset(f"GO∧一致2以上∧{hd}", [r for r in G if r["agree"] >= 2 and r["hard"] == hd])
                      for hd in ("固い", "標準")]
    rep["gauge7"] = [cellset(f"GO∧{gs}", [r for r in G if r["gset"] == gs])
                     for gs in ("人気", "PW", "速度", "人気PW", "人気速度", "PW速度", "人気PW速度")]
    # 相手印の寄与 (メインの各セル)
    contrib = []
    for c, (k, hd) in zip(main_cells, [(k, hd) for k in (3, 2, 1) for hd in ("固い", "標準")]
                          + [(k, None) for k in (3, 2, 1)]):
        sub = [r for r in G if r["agree"] == k and (hd is None or r["hard"] == hd)]
        e = {"name": c["name"]}
        for t in TYPES:
            agg = {}
            for r in sub:
                for lb, rc, win in r["settled"][t]:
                    if win:
                        a = agg.setdefault(lb, [0, 0.0]); a[0] += 1; a[1] += rc
            e[t] = {lb: {"hits": v[0], "ret": round(v[1], 1)} for lb, v in agg.items()}
        contrib.append(e)
    rep["partner_contrib"] = contrib
    rep["value_loc"] = [cellset(f"GO∧妙味馬:{v}", [r for r in G if r["vloc"] == v])
                        for v in ("◎本人を含む", "相手印を含む", "印外のみ", "妙味馬なし")]
    allc = main_cells + rep["agree2p"]
    rep["candidates"] = [(c["name"], t) for c in allc for t in TYPES if c["by_type"][t].get("candidate")]
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    # 手計算照合用に 1 レース分の明細
    if rows:
        r0 = next((r for r in G if any(p[2] for p in r["settled"]["三連複"])), G[0])
        (out / "handcheck.json").write_text(json.dumps(r0, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("n_races", "models", "excluded", "go_repro", "candidates")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
