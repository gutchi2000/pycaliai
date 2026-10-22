# -*- coding: utf-8 -*-
"""L-4 — label-free inputs: EXP16A official race set (2016-2023) + pre market + v6 OOF z + calendar layers.

Never reads winner_pos, 着順, payouts or any result column.
Output: analysis/calendar_bias/out/inputs.npz, out/inputs_report.json
python -m analysis.calendar_bias.build_inputs
"""
from __future__ import annotations

import json
import zipfile

import numpy as np
import pandas as pd

from . import common as C

NPZ_ALLOWED = ["rid16", "year", "ban", "offsets", "pi_pre"]


def read_npz():
    out = {}
    with zipfile.ZipFile(C.NPZ) as zf:
        for k in NPZ_ALLOWED:
            with zf.open(k + ".npy") as f:
                out[k] = np.lib.format.read_array(f, allow_pickle=False)
    return out


def master_races():
    parts = []
    for ch in pd.read_csv(C.MASTER, encoding="utf-8-sig", dtype=str, chunksize=300_000,
                          usecols=["日付", "レースID(新/馬番無)", "クラス名", "トラックコード(JV)"]):
        d = pd.to_numeric(ch["日付"], errors="coerce")
        ch = ch[(d >= 20160101) & (d <= 20231231)]
        parts.append(ch.drop_duplicates("レースID(新/馬番無)"))
    m = pd.concat(parts).drop_duplicates("レースID(新/馬番無)")
    m["rid16"] = m["レースID(新/馬番無)"].str.replace(r"\.0$", "", regex=True).str[:16]
    assert m["rid16"].str.len().eq(16).all()
    return m


def main():
    rep = {}
    N = read_npz()
    assert N["year"].max() <= C.YEAR_MAX
    off, rid, year = N["offsets"], N["rid16"].astype(str), N["year"].astype(int)
    seg = C.Seg(off)
    rows = pd.DataFrame({"rid16": seg.spread(rid), "ban": N["ban"].astype(int), "pi_npz": N["pi_pre"]})

    mk = pd.read_parquet(C.MARKET, columns=["rid16", "snap", "ban", "odds"])
    mk = mk[(mk.snap == "pre") & mk.rid16.isin(set(rid))].drop_duplicates(["rid16", "ban"])
    rows = rows.merge(mk[["rid16", "ban", "odds"]], on=["rid16", "ban"], how="left")
    oof = pd.read_parquet(C.OOF)
    oof = oof[oof.rid.isin(set(rid))].drop_duplicates(["rid", "ban"]).rename(columns={"rid": "rid16"})
    rows = rows.merge(oof[["rid16", "ban", "score"]], on=["rid16", "ban"], how="left")
    assert len(rows) == off[-1]

    bad_pre = ~(rows["odds"] > 1.0)
    bad_oof = rows["score"].isna()
    race_bad_pre = np.add.reduceat(bad_pre.to_numpy().astype(int), seg.start) > 0
    race_bad_oof = np.add.reduceat(bad_oof.to_numpy().astype(int), seg.start) > 0
    keep = ~(race_bad_pre | race_bad_oof)
    rep["exclusions"] = {"races_in_npz": int(seg.R), "missing_pre_odds": int(race_bad_pre.sum()),
                         "missing_oof": int(race_bad_oof.sum()), "excluded_any": int((~keep).sum()),
                         "kept": int(keep.sum())}
    races = np.flatnonzero(keep)
    rr, sg = C.sub_seg(seg, races)
    r = rows.iloc[rr].reset_index(drop=True)
    inv = 1.0 / r["odds"].to_numpy()
    pi = inv / sg.spread(sg.sum(inv))
    rep["pi_pre_vs_npz_max_abs_diff"] = float(np.nanmax(np.abs(pi - r["pi_npz"].to_numpy())))
    sc = r["score"].to_numpy(float)
    mu = sg.sum(sc) / sg.cnt
    c = sc - sg.spread(mu)
    sd = np.sqrt(sg.sum(c * c) / sg.cnt)
    zz = np.where(sg.spread(sd) > 1e-12, c / np.maximum(sg.spread(sd), 1e-12), 0.0)

    rid_k, year_k = rid[races], year[races]
    s = pd.Series(rid_k)
    day = s.str[:8].astype(np.int64).to_numpy()
    venue = s.str[8:10].astype(int).to_numpy()
    kai = s.str[10:12].astype(int).to_numpy()
    nichi = s.str[12:14].astype(int).to_numpy()
    R = s.str[14:16].astype(int).to_numpy()
    dv = day * 100 + venue

    m = master_races()
    ms = m["rid16"]
    mdf = pd.DataFrame({"y": ms.str[:4].astype(int), "venue": ms.str[8:10].astype(int),
                        "kai": ms.str[10:12].astype(int), "nichi": ms.str[12:14].astype(int),
                        "dv": ms.str[:8].astype(np.int64) * 100 + ms.str[8:10].astype(int),
                        "R": ms.str[14:16].astype(int), "cls": m["クラス名"].to_numpy(),
                        "track": pd.to_numeric(m["トラックコード(JV)"], errors="coerce").to_numpy(),
                        "rid16": ms.to_numpy()})
    lastn = mdf.groupby(["y", "venue", "kai"])["nichi"].max()
    key = pd.MultiIndex.from_arrays([year_k, venue, kai])
    kai_last_n = lastn.reindex(key).to_numpy()
    assert not np.isnan(kai_last_n).any()
    g1 = mdf[(mdf.cls == "Ｇ１") & ~mdf.track.between(51, 59)]
    g1_dv = set(g1.dv.tolist())
    g1_rids = set(g1.rid16.tolist())

    wd = pd.to_datetime(pd.Series(day).astype(str), format="%Y%m%d").dt.weekday.to_numpy()
    layers = {
        "L4a_R12": R == 12,
        "L4b_WEEKDAY": wd < 5,
        "L4c_KAI_LAST": nichi == kai_last_n,
        "L4d_G1DAY_OTHER": np.isin(dv, list(g1_dv)) & ~np.isin(rid_k, list(g1_rids)),
    }

    # descriptive label audit (no outcomes)
    maxR = mdf.groupby("dv")["R"].max()
    rep["meeting_days_maxR_not_12"] = int((maxR != 12).sum())
    rep["g1_flat_races_by_year"] = g1.groupby("y").size().to_dict()
    rep["g1_rows_track_codes"] = g1.track.value_counts().to_dict()
    rep["g1_meeting_days"] = len(g1_dv)
    rep["g1_same_dv_multiple"] = int((g1.groupby("dv").size() > 1).sum())
    md = pd.Series(np.unique(day))
    wdd = pd.to_datetime(md.astype(str), format="%Y%m%d").dt.weekday
    rep["calendar_days_by_weekday"] = wdd.value_counts().sort_index().rename(
        index=dict(enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]))).to_dict()
    rep["weekday_race_days"] = sorted(int(x) for x in md[wdd < 5])
    rep["kai_last_nichi_distribution"] = pd.Series(lastn.to_numpy()).value_counts().sort_index().to_dict()
    cnt = {}
    for h, msk in layers.items():
        byy = {int(y): int((msk & (year_k == y)).sum()) for y in range(C.YEAR_MIN, C.YEAR_MAX + 1)}
        cnt[h] = {"races_2016_2023": int(msk.sum()), "races_eval_2017_2023": int((msk & (year_k >= 2017)).sum()),
                  "by_year": byy, "calendar_days": int(len(np.unique(day[msk]))),
                  "horse_rows": int(sg.cnt[msk].sum())}
    cnt["ALL"] = {"races_2016_2023": int(len(races)), "races_eval_2017_2023": int((year_k >= 2017).sum()),
                  "calendar_days": int(len(np.unique(day)))}
    rep["layer_counts"] = cnt
    rep["layer_overlap_races"] = {f"{a}&{b}": int((layers[a] & layers[b]).sum())
                                  for i, a in enumerate(C.LAYERS) for b in C.LAYERS[i + 1:]}

    np.savez_compressed(C.INPUTS, offsets=sg.off, rid16=rid_k.astype("U16"), year=year_k, day=day, dv=dv, R=R,
                        logpi=np.log(pi), z=zz, **{f"I_{h}": v for h, v in layers.items()},
                        npz_race_index=races)
    rep["prereg_sha256"] = C.sha256(C.PREREG)
    rep["outcomes_opened"] = False
    (C.OUT / "inputs_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str),
                                              encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "weekday_race_days"}, ensure_ascii=False, indent=1,
                     default=str))


if __name__ == "__main__":
    main()
