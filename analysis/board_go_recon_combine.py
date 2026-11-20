"""board_go_recon: 回収率 90% 以上のセルを組み合わせたら全体でどうなるか（探索・再構成）。

組み合わせ方: レースがどれか 1 つの選ばれたセルに入れば、そのセルの券種を買う（1 点 100 円、点数は元のまま）。
同じレース・同じ券種は 1 回だけ（重複買いしない）。別の券種は両方買う。
CI はレース単位の bootstrap（そのレースで買った全券種の払戻・投資を合算して 1 単位）。

1. 全期間で選んで全期間で測る（＝選んだ期間でそのまま測る。上振れが入るのを承知の参考値）
   a) §11 の n≥50・判定 A の 3 セル
   b) §11 の帯セルで回収率≥90% のもの全部（n 不問）
   c) §10 の 10 本（帯なし）
2. 前後半で分ける（正直な確認）: 母集団は全集合（メイン・分解・一致2以上・ゲージ組み合わせ・妙味馬の位置）
   × 5 券種 × (帯なし + 6 帯)。前半で「回収率≥90%」のセルを選び、後半で組み合わせて測る。逆向きも。
実行: python -m analysis.board_go_recon_combine --kekka-dir E:/PyCaLiAI/data/kekka
"""
import argparse
import json
from pathlib import Path

import numpy as np

from analysis.board_go_recon_backtest import BASE, collect, summ

TYPES5 = ("ワイド", "馬連", "三連複", "単勝", "複勝")


def all_sets():
    S = []
    for k in (3, 2, 1):
        for hd in ("固い", "標準"):
            S.append((f"GO∧一致{k}∧{hd}", lambda r, k=k, hd=hd: r["go"] and r["agree"] == k and r["hard"] == hd))
        S.append((f"GO∧一致{k}（難しさ不問）", lambda r, k=k: r["go"] and r["agree"] == k))
    S.append(("GOのみ", lambda r: r["go"]))
    for hd in ("固い", "標準"):
        S.append((f"GO∧一致0∧{hd}", lambda r, hd=hd: r["go"] and r["agree"] == 0 and r["hard"] == hd))
        S.append((f"GO∧{hd}", lambda r, hd=hd: r["go"] and r["hard"] == hd))
        S.append((f"GO∧一致2以上∧{hd}", lambda r, hd=hd: r["go"] and r["agree"] >= 2 and r["hard"] == hd))
    S.append(("GO∧一致0（難しさ不問）", lambda r: r["go"] and r["agree"] == 0))
    for k in (3, 2, 1, 0):
        S.append((f"一致{k}（GO不問）", lambda r, k=k: r["agree"] == k))
    for gs in ("人気", "PW", "速度", "人気PW", "人気速度", "PW速度"):
        S.append((f"GO∧{gs}だけ一致", lambda r, gs=gs: r["go"] and r["gset"] == gs))
    for v in ("◎本人を含む", "相手印を含む", "印外のみ"):
        S.append((f"GO∧妙味馬:{v}", lambda r, v=v: r["go"] and r["vloc"] == v))
    return S


def band_of(o, bands):
    for lo, hi, lb in bands:
        if o is not None and o >= lo and (hi is None or o < hi):
            return lb
    return None


def cell_rows(rows, f, band, bands):
    return [r for r in rows if f(r) and (band is None or band_of(r["hon_odds"], bands) == band)]


def combine(rows, cells, bands, seed=0):
    """cells: list of (set_name, f, type, band)。レース×券種で重複除去して合算。"""
    buy = {}  # rid -> set(types)
    for _, f, t, band in cells:
        for r in cell_rows(rows, f, band, bands):
            buy.setdefault(r["rid"], set()).add(t)
    byrid = {r["rid"]: r for r in rows}
    st, rt, by_t = [], [], {}
    for rid, ts in buy.items():
        r = byrid[rid]
        s = sum(100.0 * len(r["settled"][t]) for t in ts)
        x = sum(sum(p[1] for p in r["settled"][t]) for t in ts)
        st.append(s); rt.append(x)
        for t in ts:
            a = by_t.setdefault(t, [0, 0.0, 0.0])
            a[0] += 1; a[1] += 100.0 * len(r["settled"][t]); a[2] += sum(p[1] for p in r["settled"][t])
    if not st:
        return dict(n_races=0)
    st, rt = np.array(st), np.array(rt)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(st), (4000, len(st)))
    bs = rt[idx].sum(1) / st[idx].sum(1)
    im = int(np.argmax(rt))
    return dict(n_cells=len(cells), n_races=len(st), stake=int(st.sum()), ret=round(float(rt.sum()), 1),
                roi=round(100 * rt.sum() / st.sum(), 1),
                ci=[round(100 * float(np.percentile(bs, 2.5)), 1), round(100 * float(np.percentile(bs, 97.5)), 1)],
                roi_drop_max=round(100 * (rt.sum() - rt[im]) / (st.sum() - st[im]), 1),
                by_type={t: dict(n=v[0], roi=round(100 * v[2] / v[1], 1)) for t, v in sorted(by_t.items())})


def select(rows, universe, bands, min_n, need_drop):
    out = []
    for name, f, t, band in universe:
        sub = cell_rows(rows, f, band, bands)
        if len(sub) < max(min_n, 2):
            continue
        s = summ(sub, t)
        if s["roi"] >= 90 and (not need_drop or s["roi_drop_max"] >= 90):
            out.append((name, f, t, band))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kekka-dir", default=str(BASE / "data" / "kekka"))
    ap.add_argument("--out", default=str(BASE / "reports" / "board_go_recon" / "combine.json"))
    a = ap.parse_args()
    bands = json.loads((BASE / "data" / "shadow_policies" / "a95_equal_v1.json").read_text(encoding="utf-8"))["bands"]
    BL = [b[2] for b in bands]
    rows, _, _ = collect(Path(a.kekka_dir))
    S = dict(all_sets())
    rep = {"label": "探索・再構成（セルの組み合わせ）", "n_rows": len(rows), "full": {}, "split": {}}

    # 1. 全期間（参考値）
    ob = json.loads((BASE / "reports" / "board_go_recon" / "oddsband.json").read_text(encoding="utf-8"))
    name_map = {"GO∧PWだけ一致": "GO∧PWだけ一致", "GO∧一致3": "GO∧一致3（難しさ不問）"}
    def fn(nm):
        return S[name_map.get(nm, nm)]
    c_a, c_b, c_c = [], [], []
    for e in ob["sets"]:
        f = fn(e["name"])
        c_c.append((e["name"], f, e["type"], None))
        for lb, s in e["by_band"].items():
            if s["roi"] >= 90:
                c_b.append((e["name"], f, e["type"], lb))
                if s["n"] >= 50 and s["roi_drop_max"] >= 90:
                    c_a.append((e["name"], f, e["type"], lb))
    rep["full"]["a_n50_A_3cells"] = combine(rows, c_a, bands)
    rep["full"]["b_band_ge90_any_n"] = combine(rows, c_b, bands)
    rep["full"]["c_10sets_noband"] = combine(rows, c_c, bands)
    rep["full"]["all_rows_every_type_ref"] = {t: summ(rows, t)["roi"] for t in TYPES5}

    # 2. 前後半
    dates = sorted({r["date"] for r in rows})
    cum, half = 0, None
    cnt = {d: sum(1 for r in rows if r["date"] == d) for d in dates}
    for d in dates:
        cum += cnt[d]
        if cum >= len(rows) / 2:
            half = d; break
    A = [r for r in rows if r["date"] <= half]
    Bh = [r for r in rows if r["date"] > half]
    universe = [(nm, f, t, b) for nm, f in S.items() for t in TYPES5 for b in [None] + BL]
    rep["split"]["half_boundary"] = half
    rep["split"]["n_first"], rep["split"]["n_second"] = len(A), len(Bh)
    rep["split"]["universe_cells"] = len(universe)
    for tag, (sel_rows, ev_rows) in {"前半で選び後半で測る": (A, Bh), "後半で選び前半で測る": (Bh, A)}.items():
        for rule, (mn, nd) in {"回収率≥90（n 不問）": (2, False), "回収率≥90 ∧ 最大除外≥90 ∧ n≥25": (25, True)}.items():
            cells = select(sel_rows, universe, bands, mn, nd)
            rep["split"][f"{tag}|{rule}"] = {"selected_on_pick_half": combine(sel_rows, cells, bands),
                                             "eval_half": combine(ev_rows, cells, bands)}
    # 比べる物差し: 同じ評価半分の全レース ◎複勝・◎単勝
    rep["split"]["ref_first"] = {t: summ(A, t)["roi"] for t in ("単勝", "複勝")}
    rep["split"]["ref_second"] = {t: summ(Bh, t)["roi"] for t in ("単勝", "複勝")}
    Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
