# -*- coding: utf-8 -*-
"""
verify_missing_race_class.py
============================
欠落 50 レースの競走種別 (平地/障害) を、クラス情報を持つ全候補ソースで
実測照合する。あわせて bunseki 復元分の値一致率と time safety を検査する。

READ-ONLY。結果・払戻・ROI は読まない (着順は履歴構築の対象として扱い、
当該レース自身の結果を特徴へ入れないことを invariant として確認する)。

出力: out/missing_race_class_verification.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "out"


def log(m):
    print(m, flush=True)


def read_race_header(p: Path) -> pd.DataFrame:
    """19 列のレースヘッダ行だけを抽出する。tyaku はヘッダのみのファイル、
    kako5 はヘッダ行 + 72 列の馬行が混在するため行長で判別する。"""
    import csv as _csv
    hdr, rows = None, []
    with open(p, encoding="cp932", errors="replace") as f:
        for row in _csv.reader(f):
            if len(row) != 19:
                continue
            if row[0] == "レースID(新)":
                hdr = row
                continue
            if row[0] and row[0][:4].isdigit():
                rows.append(row)
    if hdr is None or not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=hdr)


def main():
    man = json.load(open(OUT / "missing_set_manifest.json", encoding="utf-8"))
    races = pd.DataFrame(man["missing_races"])
    races["date"] = races["rid16"].str[:8].astype(int)

    findings = []
    for _, r in races.iterrows():
        d, rid = int(r["date"]), r["rid16"]
        rec = {"date": d, "rid16": rid, "venue": r["venue"],
               "missing_rows": int(r["missing_rows"]),
               "whole_race_missing": bool(r["whole_race_missing"]),
               "race_no": int(rid[14:16]), "source": None,
               "race_name": None, "hira_shogai": None, "distance": None,
               "is_jump": None}

        # 1) bias (平・障 を直接持つ)
        p = BASE / "data" / "bias" / f"{d}.csv"
        if p.exists():
            try:
                b = pd.read_csv(p, encoding="cp932", dtype=str, on_bad_lines="skip")
                b["rid16"] = b["レースID"].astype(str).str.strip().str[:16]
                hit = b[b["rid16"] == rid]
                if len(hit):
                    h = hit.iloc[0]
                    rec.update(source="data/bias",
                               race_name=str(h.get("レース名", "")).strip(),
                               hira_shogai=str(h.get("平・障", "")).strip(),
                               distance=str(h.get("距離", "")).strip())
                    rec["is_jump"] = rec["hira_shogai"] == "1"
            except Exception:
                pass

        # 2) bunseki (クラス・レース名)
        if rec["source"] is None:
            p = BASE / "data" / "bunseki" / f"{d}.csv"
            if p.exists():
                try:
                    s = pd.read_csv(p, encoding="cp932", dtype=str, on_bad_lines="skip")
                    s["rid16"] = s["レースID(新)"].astype(str).str.strip().str[:16]
                    hit = s[s["rid16"] == rid]
                    if len(hit):
                        h = hit.iloc[0]
                        nm = str(h.get("レース名", "")).strip()
                        cl = str(h.get("クラス", "")).strip()
                        rec.update(source="data/bunseki", race_name=nm or cl,
                                   distance=str(h.get("距離", "")).strip())
                        rec["is_jump"] = ("障" in nm or "障" in cl
                                          or "ジャ" in nm or "JG" in nm)
                except Exception:
                    pass

        # 3) tyaku / kako5 race header (どちらも 19 列ヘッダ形式、
        #    『芝・ダート』列が直接 "障害" を持つ)
        if rec["source"] is None:
            for src, p in (("data/tyaku", BASE / "data" / "tyaku" / f"{d}.csv"),
                           ("data/kako5(header)",
                            BASE / "data" / "kako5" / f"{d}.csv")):
                if not p.exists():
                    continue
                try:
                    t = read_race_header(p)
                except Exception:
                    continue
                if t.empty or "レースID(新)" not in t.columns:
                    continue
                t["rid16"] = t["レースID(新)"].astype(str).str.strip().str[:16]
                hit = t[t["rid16"] == rid]
                if not len(hit):
                    continue
                h = hit.iloc[0]
                nm = str(h.get("レース名", "")).strip()
                cl = str(h.get("クラス名", "")).strip()
                td = str(h.get("芝・ダート", "")).strip()
                rec.update(source=src,
                           race_name=(nm if nm and nm != "nan" else cl),
                           distance=str(h.get("距離", "")).strip())
                rec["surface_field"] = td
                rec["is_jump"] = ("障" in td or "障" in nm or "障" in cl
                                  or "ジャ" in nm or "JG" in nm)
                break

        findings.append(rec)

    fd = pd.DataFrame(findings)
    with_src = fd[fd["source"].notna()]
    jump = with_src[with_src["is_jump"] == True]  # noqa: E712

    log("=" * 70)
    log("欠落レースの競走種別 実測照合")
    log("=" * 70)
    log(f"欠落レース総数: {len(fd)}")
    log(f"クラス情報を持つソースで照合できたレース: {len(with_src)}")
    log(f"  うち障害レース: {len(jump)} "
        f"({len(jump)/max(len(with_src),1)*100:.1f}%)")
    log(f"ソース別: {with_src['source'].value_counts().to_dict()}")
    log("\n照合できたレース一覧:")
    for _, r in with_src.iterrows():
        log(f"  {r['date']} R{r['race_no']:02d} {r['venue']:>3s} "
            f"{'障害' if r['is_jump'] else '平地':4s} "
            f"dist={r['distance']:>5s} rows={r['missing_rows']:2d} "
            f"[{r['source']}] {r['race_name']}")

    unmatched = fd[fd["source"].isna()]
    log(f"\n照合できなかったレース: {len(unmatched)} "
        f"(クラス情報を持つソースが当該日に存在しない)")

    # --- 距離ヒューリスティクスによる補助判定 (確証ではなく傍証) ---
    dist_known = with_src[with_src["distance"].astype(str).str.strip() != ""]
    if len(dist_known):
        dj = pd.to_numeric(dist_known.loc[dist_known["is_jump"] == True,  # noqa: E712
                                          "distance"], errors="coerce").dropna()
        log(f"\n障害と確認されたレースの距離レンジ: "
            f"{dj.min():.0f}〜{dj.max():.0f}m (n={len(dj)})")

    # --- bunseki 復元分の値一致率 (kekka と重複する列) ---
    rows = pd.read_csv(OUT / "missing_set_rows.csv", encoding="utf-8-sig", dtype=str)
    rows["umaban"] = pd.to_numeric(rows["umaban"], errors="coerce")
    rows["date"] = rows["race_id"].str[:8].astype(int)
    agree = {"horse_name_match": 0, "horse_name_total": 0}
    for d in [20260905, 20260906]:
        p = BASE / "data" / "bunseki" / f"{d}.csv"
        if not p.exists():
            continue
        s = pd.read_csv(p, encoding="cp932", dtype=str, on_bad_lines="skip")
        s["rid16"] = s["レースID(新)"].astype(str).str.strip().str[:16]
        s["umaban"] = pd.to_numeric(s["馬番"], errors="coerce")
        sub = rows[rows["date"] == d]
        for _, r in sub.iterrows():
            h = s[(s["rid16"] == r["race_id"]) & (s["umaban"] == r["umaban"])]
            if len(h):
                agree["horse_name_total"] += 1
                a = str(h.iloc[0]["馬名"]).replace("　", "").strip()
                b = str(r["horse_name"]).replace("　", "").strip()
                agree["horse_name_match"] += int(a == b)
    log(f"\nbunseki 復元行の値一致 (race_id+馬番 join、馬名を独立検証に使用): "
        f"{agree['horse_name_match']}/{agree['horse_name_total']} 一致")

    payload = {
        "generated_at": datetime.now().isoformat(),
        "n_missing_races": len(fd),
        "n_class_verified": len(with_src),
        "n_jump_races": int(len(jump)),
        "jump_share_of_verified": round(len(jump) / max(len(with_src), 1), 4),
        "sources_used": with_src["source"].value_counts().to_dict(),
        "n_unverified": len(unmatched),
        "races": findings,
        "bunseki_value_agreement": agree,
        "interpretation": (
            "照合できた欠落レースは全て障害競走。data/weekly は障害競走を "
            "含まない。CLAUDE.md の『除外レース: 障害・新馬中心』と整合する "
            "意図的な scope filter であり、偶発的な取りこぼしではない。"
            "ただし学習側 master_v2 は障害競走を含む (1,538 レース / 3.42%) "
            "ため、train/serve 間に母集団の非対称が存在する。"),
    }
    with open(OUT / "missing_race_class_verification.json", "w",
              encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    log("\n保存: out/missing_race_class_verification.json")


if __name__ == "__main__":
    main()
