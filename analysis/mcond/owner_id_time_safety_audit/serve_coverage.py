# -*- coding: utf-8 -*-
"""
serve_coverage.py — A: owner serve coverage を「v6 へ渡る直前の 120 列行列」で実測
                    C: no-label shadow 差分 (S0/S1/S2)
=================================================================================
production には一切書き込まない (bundle も site も触らない)。結果・払戻・ROI は使わない。
serve 経路の再現 (export_weekly_marks.main と同じ順序):
  parse_csv (内部で apply_bunseki: 馬名 join) → ensure_date_column → _SERVE_RENAME
  → category_normalize → 不足列補完 (cat="__NaN__" / num=NaN) → serve_history_feats
  → export_race と同じ encoder 適用 → to_numeric().fillna(-9999) → model.predict
出力: out/serve_coverage.json, out/shadow_s0s1s2.json, out/serve_coverage_table.md
実行: python -m analysis.mcond.owner_id_time_safety_audit.serve_coverage
"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "out"
sys.path.insert(0, str(BASE))
logging.getLogger().setLevel(logging.ERROR)

from predict_weekly import parse_csv  # noqa: E402  (内部で apply_bunseki)
from category_normalize import NORMALIZERS, normalize_categorical, fixed_count  # noqa: E402
from parse_bunseki import load_bunseki, COL_MAP  # noqa: E402
import pl_probs as PL  # noqa: E402

OWN = "馬主(最新/仮想)"
COL_RID = "レースID(新)"
COL_BAN = "馬番"
# export_weekly_marks.main() 内のローカル定数をそのまま写したもの (本体は変更しない)
SERVE_RENAME = {
    "R": "Ｒ", "前走補正": "prev_hosei", "前走補9": "prev_hosei9",
    "trn_hanro_4f": "trnH_Time1", "trn_hanro_3f": "trnH_Time2", "trn_hanro_2f": "trnH_Time3",
    "trn_hanro_1f": "trnH_Time4", "trn_hanro_lap1": "trnH_Lap1", "trn_hanro_lap2": "trnH_Lap2",
    "trn_hanro_lap3": "trnH_Lap3", "trn_hanro_lap4": "trnH_Lap4", "trn_hanro_days": "trnH_days_ago",
    "trn_wc_5f": "trnW_5F", "trn_wc_4f": "trnW_4F", "trn_wc_3f": "trnW_3F",
    "trn_wc_lap1": "trnW_Lap1", "trn_wc_lap2": "trnW_Lap2", "trn_wc_lap3": "trnW_Lap3",
    "trn_wc_days": "trnW_days_ago",
}


def load_v6():
    b = joblib.load(BASE / "models" / "unified_rank_v6.pkl")
    return b["model"], list(b["feature_cols"]), b["encoders"]


def build_serve_df(date_str: str, feats, encs):
    """export_weekly_marks と同じ順序で serve 入力 df を作る (書き込みなし)"""
    df = parse_csv(BASE / "data" / "weekly" / f"{date_str}.csv")
    raw_own_present = OWN in df.columns
    raw_own_nonnull = float(df[OWN].notna().mean()) if raw_own_present else 0.0
    if "日付" not in df.columns:
        df["日付"] = int(date_str)
    ren = {k: v for k, v in SERVE_RENAME.items() if k in df.columns and v in feats and v not in df.columns}
    if ren:
        df = df.rename(columns=ren)
    if fixed_count(df, NORMALIZERS):
        df = normalize_categorical(df, NORMALIZERS)
    cat_cols = set(encs.keys())
    missing = [c for c in feats if c not in df.columns]
    for c in missing:
        df[c] = "__NaN__" if c in cat_cols else np.nan
    try:
        from serve_history_feats import fill_history_features
        fill_history_features(df)
    except Exception as e:  # fail-open (本番と同じ)
        print(f"  [warn] history feats: {e}")
    return df, {"raw_owner_column_present": raw_own_present,
                "raw_owner_nonnull_rate": raw_own_nonnull,
                "owner_in_missing_fill": OWN in missing}


def encode_and_X(df: pd.DataFrame, feats, encs):
    """export_race と同じ encoder 適用 → X"""
    g = df.copy()
    for c, le in encs.items():
        if c not in g.columns:
            continue
        v = g[c].astype(str).fillna("__NaN__")
        v = v.where(v.isin(set(le.classes_)), "__NaN__")
        g[c] = le.transform(v)
    X = g[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999)
    return g, X


def marks_and_probs(model, X: np.ndarray, rid: np.ndarray):
    """レースごとに score → PL 勝率・順位・印 (export_race と同じ手順)"""
    s = model.predict(X)
    p = np.zeros(len(s))
    rank = np.zeros(len(s), dtype=int)
    for r in np.unique(rid):
        m = rid == r
        w = PL.pl_weights(s[m])
        p[m] = PL.all_tansho(w)
        order = np.argsort(-s[m])
        rr = np.zeros(m.sum(), dtype=int)
        for k, idx in enumerate(order):
            rr[idx] = k + 1
        rank[m] = rr
    return s, p, rank


def name_join_diagnostics(df: pd.DataFrame, date_str: str) -> dict:
    """馬名 join の健全性 (同名異馬 / 表記ゆれ / 切詰め / 改名 / 失敗 / 誤join)"""
    path = BASE / "data" / "bunseki" / f"{date_str}.csv"
    if not path.exists():
        return {"bunseki_present": False}
    b = load_bunseki(date_str, BASE)
    raw = pd.read_csv(path, encoding="cp932", dtype=str, low_memory=False)
    key = df["馬名"].astype(str).str.strip()
    bname = b["馬名"].astype(str)
    dup = bname[bname.duplicated(keep=False)]
    matched = key.isin(set(bname))
    # 表記ゆれ候補: 前後空白・全角空白・記号 ($ など) を落とすと一致する行
    def clean(s):
        return s.str.replace(r"[\s　]", "", regex=True).str.lstrip("$＄")
    bclean = set(clean(bname))
    extra_by_clean = (~matched) & clean(key).isin(bclean)
    # 切詰め候補: weekly 名が bunseki 名の接頭辞
    bl = list(bclean)
    trunc = 0
    if (~matched).any():
        for k in clean(key[~matched]).tolist():
            if any(x.startswith(k) and len(x) > len(k) for x in bl):
                trunc += 1
    # 改名/同名異馬 (血統登録番号がある bunseki 側でのみ検出可能)
    same_name_diff_id = 0
    if "血統登録番号" in raw.columns:
        gg = raw.groupby(raw["馬名"].astype(str))["血統登録番号"].nunique()
        same_name_diff_id = int((gg > 1).sum())
    return {
        "bunseki_present": True, "bunseki_rows": int(len(b)),
        "weekly_rows": int(len(df)),
        "name_matched": int(matched.sum()), "name_join_failed": int((~matched).sum()),
        "name_join_failed_rate": float((~matched).mean()),
        "bunseki_duplicate_names": int(dup.nunique()),
        "recoverable_by_normalizing_notation": int(extra_by_clean.sum()),
        "truncation_candidates": int(trunc),
        "same_name_different_pedigree_id_in_bunseki": same_name_diff_id,
        "weekly_has_pedigree_id": bool("血統登録番号" in df.columns),
        "join_key_used_by_production": "馬名 (str.strip) — 血統登録番号は weekly 側に無く使用不可",
    }


def main():
    model, feats, encs = load_v6()
    own_classes = list(encs[OWN].classes_)
    nan_code = int(np.where(np.array(own_classes) == "__NaN__")[0][0])
    dates = sorted(p.stem for p in (BASE / "data" / "weekly").glob("2026*.csv"))
    cov_rows, shadow_rows = [], []
    for d in dates:
        try:
            df, raw = build_serve_df(d, feats, encs)
        except Exception as e:
            cov_rows.append({"date": d, "error": str(e)[:120]})
            print(f"[{d}] parse 失敗: {e}", flush=True)
            continue
        rid = df[COL_RID].astype(str).to_numpy()
        venue = df["場所"].astype(str).to_numpy() if "場所" in df.columns else np.array(["?"] * len(df))
        # --- 段階1: encoder 直前 (文字列)
        pre = df[OWN].astype(str).fillna("__NaN__") if OWN in df.columns else pd.Series(["__NaN__"] * len(df))
        pre_nonnull = float((df[OWN].notna() & ~pre.isin(["nan", "__NaN__", ""])).mean()) if OWN in df.columns else 0.0
        known = float(pre.isin(set(own_classes)).mean())
        eff_known = float((pre.isin(set(own_classes)) & (pre != "__NaN__") & (pre != "nan")).mean())
        # --- 段階2: encoder 直後 / 段階3: predict 直前
        g, X = encode_and_X(df, feats, encs)
        codes = g[OWN].to_numpy()
        Xown = X[OWN].to_numpy()
        nj = name_join_diagnostics(df, d)
        cov = {
            "date": d, "n_horses": int(len(df)), "n_races": int(pd.unique(rid).size),
            "bunseki_present": nj.get("bunseki_present", False),
            "raw_owner_column_present_after_parse": raw["raw_owner_column_present"],
            "owner_filled_by_missing_fill(__NaN__)": raw["owner_in_missing_fill"],
            "pre_encoder_nonnull_rate": pre_nonnull,
            "encoder_known_rate_incl___NaN__": known,
            "effective_known_rate_excl___NaN__": eff_known,
            "__NaN__rate": float((pre == "__NaN__").mean()),
            "post_encoder_nan_code_rate": float((codes == nan_code).mean()),
            "pre_predict_default_-9999_rate": float((Xown == -9999).mean()),
            "pre_predict_nan_code_rate": float((Xown == nan_code).mean()),
            "distinct_owner_codes": int(pd.unique(codes).size),
            "by_venue_nonnull": {v: float((pre[venue == v] != "__NaN__").mean()) for v in np.unique(venue)},
            "name_join": nj,
        }
        cov_rows.append(cov)

        # --- C: S0/S1/S2 ------------------------------------------------------
        X0 = X.to_numpy(dtype=float)
        g1 = g.copy(); g1[OWN] = nan_code
        X1 = g1[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).to_numpy(dtype=float)
        # S2: bunseki から復元できる owner を現行 encoder へ通す (無い週は他週の bunseki も使う=counterfactual)
        own2 = pre.copy()
        if OWN in df.columns:
            need = pre.isin(["__NaN__", "nan", ""])
            if need.any():
                allb = []
                for p in sorted((BASE / "data" / "bunseki").glob("2026*.csv")):
                    try:
                        bb = load_bunseki(p.stem, BASE)
                        if "馬主" in bb.columns or OWN in bb.columns:
                            allb.append(bb)
                    except Exception:
                        pass
                if allb:
                    ab = pd.concat(allb, ignore_index=True).drop_duplicates("馬名")
                    col = OWN if OWN in ab.columns else "馬主"
                    mp = ab.set_index("馬名")[col]
                    rec = df["馬名"].astype(str).str.strip().map(mp)
                    own2 = own2.where(~need, rec.fillna("__NaN__"))
        v2 = own2.astype(str).where(own2.astype(str).isin(set(own_classes)), "__NaN__")
        g2 = g.copy(); g2[OWN] = encs[OWN].transform(v2)
        X2 = g2[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).to_numpy(dtype=float)
        s0, p0, r0 = marks_and_probs(model, X0, rid)
        s1, p1, r1 = marks_and_probs(model, X1, rid)
        s2, p2, r2 = marks_and_probs(model, X2, rid)

        def cmp(sa, pa, ra, sb, pb, rb, tag):
            races = pd.unique(rid)
            rank_ch = mark_ch = top3_ch = 0
            for r in races:
                m = rid == r
                if not np.array_equal(ra[m], rb[m]):
                    rank_ch += 1
                if (ra[m] == 1).argmax() != (rb[m] == 1).argmax():
                    mark_ch += 1
                if set(np.where(m)[0][ra[m] <= 3]) != set(np.where(m)[0][rb[m] <= 3]):
                    top3_ch += 1
            return {"tag": tag, "n_races": int(len(races)),
                    "identical_scores": bool(np.array_equal(sa, sb)),
                    "mean_abs_score_diff": float(np.abs(sa - sb).mean()),
                    "max_abs_score_diff": float(np.abs(sa - sb).max()),
                    "mean_abs_p_win_diff": float(np.abs(pa - pb).mean()),
                    "rank_changed_races": rank_ch, "hon_changed_races": mark_ch,
                    "top3_set_changed_races": top3_ch}
        owner_known = (pre != "__NaN__").to_numpy()
        row = {"date": d, "bunseki_present": cov["bunseki_present"],
               "owner_known_rate": float(owner_known.mean()),
               "S0_vs_S1": cmp(s0, p0, r0, s1, p1, r1, "S0-S1"),
               "S0_vs_S2": cmp(s0, p0, r0, s2, p2, r2, "S0-S2"),
               "S1_vs_S2": cmp(s1, p1, r1, s2, p2, r2, "S1-S2")}
        if owner_known.any() and (~owner_known).any():
            row["S0_vs_S1_by_owner_known"] = {
                "known_mean_abs_score_diff": float(np.abs(s0 - s1)[owner_known].mean()),
                "unknown_mean_abs_score_diff": float(np.abs(s0 - s1)[~owner_known].mean())}
        # bundle 照合 (再現の忠実さ)
        bp = BASE / "reports" / "cowork_input" / f"{d}_bundle.json"
        if bp.exists():
            bj = json.loads(bp.read_text(encoding="utf-8"))
            ref = {}
            for rc in bj["races"]:
                for h in rc["horses"]:
                    ref[(str(rc["race_id"]), int(h["umaban"]))] = h["ai_score"]
            mine = {(str(a).split(".")[0], int(b)): float(c)
                    for a, b, c in zip(df[COL_RID], df[COL_BAN], s0)}
            keys = [k for k in ref if k in mine]
            if keys:
                diff = np.array([abs(ref[k] - mine[k]) for k in keys])
                row["bundle_parity"] = {"matched_horses": len(keys), "of_bundle": len(ref),
                                        "mean_abs_diff": float(diff.mean()),
                                        "max_abs_diff": float(diff.max()),
                                        "within_1e-3": float((diff < 1e-3).mean())}
        shadow_rows.append(row)
        print(f"[{d}] owner既知 {cov['effective_known_rate_excl___NaN__']*100:5.1f}%  "
              f"bunseki={cov['bunseki_present']}  S0=S1:{row['S0_vs_S1']['identical_scores']}  "
              f"◎変更(S0-S2) {row['S0_vs_S2']['hon_changed_races']}/{row['S0_vs_S2']['n_races']}"
              + (f"  bundle一致 {row['bundle_parity']['within_1e-3']*100:.0f}%" if "bundle_parity" in row else ""),
              flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "serve_coverage.json").write_text(json.dumps(
        {"owner_encoder_classes": len(own_classes), "nan_code": nan_code, "dates": cov_rows},
        ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (OUT / "shadow_s0s1s2.json").write_text(json.dumps(shadow_rows, ensure_ascii=False, indent=1,
                                                       default=float), encoding="utf-8")
    ok = [c for c in cov_rows if "error" not in c]
    L = ["# A: owner serve coverage (v6 入力行列で実測) / C: S0-S1-S2 shadow", "",
         f"対象 {len(ok)} 日 / パース失敗 {len(cov_rows)-len(ok)} 日", "",
         "| 日付 | 頭数 | bunseki | encoder直前 非欠損 | 実効既知(__NaN__除く) | __NaN__率 | 既知率(__NaN__含む) | -9999率 | 馬名join失敗 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for c in ok:
        nj = c["name_join"]
        L.append(f"| {c['date']} | {c['n_horses']} | {'有' if c['bunseki_present'] else '無'} | "
                 f"{c['pre_encoder_nonnull_rate']*100:.1f}% | {c['effective_known_rate_excl___NaN__']*100:.1f}% | "
                 f"{c['__NaN__rate']*100:.1f}% | {c['encoder_known_rate_incl___NaN__']*100:.1f}% | "
                 f"{c['pre_predict_default_-9999_rate']*100:.1f}% | "
                 f"{nj.get('name_join_failed', '-')}/{nj.get('weekly_rows', '-')} |")
    L += ["", "## C: shadow 差分", "",
          "| 日付 | owner既知 | S0=S1 | S0-S2 平均|Δscore| | ◎変更R | top3変更R | rank変更R | bundle一致 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in shadow_rows:
        bpar = r.get("bundle_parity")
        L.append(f"| {r['date']} | {r['owner_known_rate']*100:.1f}% | {r['S0_vs_S1']['identical_scores']} | "
                 f"{r['S0_vs_S2']['mean_abs_score_diff']:.5f} | {r['S0_vs_S2']['hon_changed_races']} | "
                 f"{r['S0_vs_S2']['top3_set_changed_races']} | {r['S0_vs_S2']['rank_changed_races']} | "
                 f"{(str(round(bpar['within_1e-3']*100)) + '%') if bpar else '-'} |")
    (OUT / "serve_coverage_table.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\n[saved] out/serve_coverage.json / out/shadow_s0s1s2.json / out/serve_coverage_table.md")


if __name__ == "__main__":
    main()
