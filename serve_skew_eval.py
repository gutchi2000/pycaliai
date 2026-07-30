# -*- coding: utf-8 -*-
"""
serve_skew_eval.py
==================
本番推論で「死んでいる」約30特徴をマスクしたとき、v6 の印精度 / ECE が
どれだけ劣化するかを測る (再学習なし)。docs/audit_20260611.md の
train/serve skew (critical) を定量化するための実験スクリプト。

baseline : master_v2 のフル特徴で predict (= offline 監査で見ている条件)
masked   : 本番 (export_weekly_marks → parse_csv) で値が作れない列を
           本番と同じ値に潰して predict
             - 数値列   → NaN → fillna(-9999)   (export_marks_json と同一)
             - 騎手/調教師コード → "__NaN__"    (apply_encoders で未知扱い)

両者で印付け (raw スコア降順 top5 = ◎〇▲△△) して指標を比較する。
calibrator は v6 をそのまま両方に適用 (masked で ECE が崩れるかを見る)。

使い方: PYTHONUTF8=1 python serve_skew_eval.py
出力   : reports/serve_skew_eval.json + 標準出力に baseline vs masked の差分表
"""
from __future__ import annotations
import io
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import pl_probs as PL
import backtest_pl_ev as be
from backtest_pl_ev import (
    apply_encoders, load_payouts, all_fukusho_vec_fast, all_umaren_mat,
    COL_RID, COL_JYUN, COL_BAN,
)
from audit_marks import expected_calibration_error

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

BASE = Path(__file__).parent
be.MODEL_PKL = BASE / "models/unified_rank_v6.pkl"
be.CAL_PKL = BASE / "models/pl_calibrators_v6.pkl"

# 本番 parse_csv で作られない (= -9999/'__NaN__' 固定になる) 特徴
# ※ 監査時点 (2026-06-11 朝) の全30列。定量化の再現用に保持。
DEAD_PREFIX = ("trnH_", "trnW_", "hist_same_")
DEAD_EXACT_NUM = {
    "course_n_prev", "course_win_rate", "course_top3_rate",
    "jockey_n_prev", "jockey_win_rate", "jockey_top3_rate",
    "prev_hosei", "prev_hosei9",
}
DEAD_CAT = {"騎手コード", "調教師コード"}

# リネーム修正後 (補正 2026-06-11 午前 / 調教 同日午後) も本番で作れない列。
# build_pl_calibrators_serve.py はこちらを使う (serve 特徴を増やしたら必ず更新+再fit)。
# 2026-07-29: hist_same_*/course_*/jockey_*/騎手・調教師コード の 12 列は
# serve_history_feats.py (data/_horse_history.parquet + 馬名JOIN) で回収済み
# (test パリティ 98.6% 行で 100% 一致・残 1.4% は同名曖昧の安全側 NaN 降格 /
#  ◎複勝 +0.84pt, reports/validate_serve_history_feats.json)。
# 2026-07-30 監査 (B/C 独立到達): マスクを空にしたのは誤り。週次CSVから今も
# 作れない当日条件系 14 列 (data/serve_feature_baseline.json の serve_dead) が
# 残っており、空マスク fit は pl_calibrators_v6.pkl と同一物を生むだけだった
# (serve 分布 ◎複勝 ECE 0.0143 vs serve-fit 期待 ~0.011)。dead14 を定義して再fit。
SERVE_DEAD_NOW_PREFIX = ()
SERVE_DEAD_NOW_EXACT: set[str] = {
    "前走レースID(新)", "前走レースID(新/馬番無)", "前走日付",
    "前走走破タイム", "母馬", "Ｒ",
}
SERVE_DEAD_NOW_CAT: set[str] = {
    "ブリンカー", "前好走", "前走場所", "指定条件",
    "毛色", "芝(内・外)", "限定", "馬主(最新/仮想)",
}


def evaluate(te: pd.DataFrame, payouts: dict, cals: dict | None) -> dict:
    """te(_score 付き) を印付け→指標集計。期間別(2023 / 2024-2025)に返す。"""
    rows = []
    for rid, g in te.groupby(COL_RID, sort=False):
        if len(g) < 5:
            continue
        rid_s = str(int(rid)) if isinstance(rid, (int, np.integer)) else str(rid)
        if rid_s not in payouts:
            continue
        po = payouts[rid_s]
        g = g.sort_values(COL_BAN).reset_index(drop=True)
        ban = g[COL_BAN].astype(int).values
        b2i = {int(b): i for i, b in enumerate(ban)}
        wi, pi_, si = b2i.get(po["win"]), b2i.get(po["plc"]), b2i.get(po["sho"])
        if wi is None or pi_ is None or si is None:
            continue
        scores = g["_score"].values
        w = PL.pl_weights(scores)
        order = np.argsort(-scores)
        hon, tai, san = int(order[0]), int(order[1]), int(order[2])
        del1, del2 = int(order[3]), int(order[4])
        top5 = {hon, tai, san, del1, del2}
        actual_top3 = {wi, pi_, si}

        p_tansho_vec = PL.all_tansho(w)
        p_fukusho_vec = all_fukusho_vec_fast(w)
        Pmat = all_umaren_mat(w)
        p_umaren_hono = float(Pmat[hon, tai])
        if cals is not None:
            p_tansho_vec = np.clip(cals["tansho"].predict(p_tansho_vec), 0.0, 1.0)
            p_fukusho_vec = np.clip(cals["fukusho"].predict(p_fukusho_vec), 0.0, 1.0)
            p_umaren_hono = float(np.clip(cals["umaren"].predict([p_umaren_hono])[0], 0.0, 1.0))

        rows.append({
            "year": int(g["year"].iloc[0]),
            "hon_1st": int(hon == wi),
            "hon_top2": int(hon in {wi, pi_}),
            "hon_top3": int(hon in actual_top3),
            "tai_top3": int(tai in actual_top3),
            "san_top3": int(san in actual_top3),
            "winner_in_top5": int(wi in top5),
            "top3_subset_top5": int(actual_top3.issubset(top5)),
            "p_tansho_hon": float(p_tansho_vec[hon]),
            "p_fukusho_hon": float(p_fukusho_vec[hon]),
            "p_umaren_hono": p_umaren_hono,
            "hit_umaren_hono": int({hon, tai} == {wi, pi_}),
        })

    df = pd.DataFrame(rows)
    res = {}
    for label, ys in [("valid_2023", {2023}), ("test_2024_2025", {2024, 2025})]:
        sub = df[df["year"].isin(ys)]
        if len(sub) == 0:
            continue
        ece_tan, _ = expected_calibration_error(sub["p_tansho_hon"], sub["hon_1st"])
        ece_fuku, _ = expected_calibration_error(sub["p_fukusho_hon"], sub["hon_top3"])
        ece_uma, _ = expected_calibration_error(sub["p_umaren_hono"], sub["hit_umaren_hono"])
        res[label] = {
            "races": int(len(sub)),
            "hon_1st_pct": float(sub["hon_1st"].mean() * 100),
            "hon_top2_pct": float(sub["hon_top2"].mean() * 100),
            "hon_top3_pct": float(sub["hon_top3"].mean() * 100),
            "tai_top3_pct": float(sub["tai_top3"].mean() * 100),
            "san_top3_pct": float(sub["san_top3"].mean() * 100),
            "winner_in_top5_pct": float(sub["winner_in_top5"].mean() * 100),
            "top3_subset_top5_pct": float(sub["top3_subset_top5"].mean() * 100),
            "ece_tansho_hon": ece_tan,
            "ece_fuku_hon": ece_fuku,
            "ece_umaren_hono": ece_uma,
        }
    return res


def main():
    bundle = joblib.load(be.MODEL_PKL)
    model, feats, encs = bundle["model"], bundle["feature_cols"], bundle["encoders"]
    dead_numeric = [c for c in feats if c.startswith(DEAD_PREFIX) or c in DEAD_EXACT_NUM]
    dead_cat = [c for c in feats if c in DEAD_CAT]
    print(f"[feats] total={len(feats)}  dead_numeric={len(dead_numeric)}  dead_cat={len(dead_cat)}  "
          f"= {len(dead_numeric)+len(dead_cat)} masked")
    print(f"[dead_numeric] {dead_numeric}")
    print(f"[dead_cat]     {dead_cat}")

    cal = joblib.load(be.CAL_PKL) if be.CAL_PKL.exists() else None
    cals = cal.get("calibrators") if cal else None
    print(f"[calibrator] {'v6 適用' if cals else 'なし(raw)'}")

    print(f"[load] {be.MASTER_CSV.name} ...")
    df = pd.read_csv(be.MASTER_CSV, encoding="utf-8-sig", low_memory=False)
    df = df.dropna(subset=[COL_JYUN, COL_RID, "split"]).copy()
    df = df[df["split"].isin(["test", "valid"])].copy()
    df["year"] = df[COL_RID].astype(str).str[:4].astype(int)
    print(f"[data] test+valid rows={len(df):,}  races={df[COL_RID].nunique():,}")
    payouts = load_payouts()

    def scores_for(mask_num, mask_cat) -> pd.DataFrame:
        """mask_num の数値列を -9999、mask_cat のカテゴリ列を未知にして predict。"""
        d = df.copy()
        for c in mask_cat:
            if c in d.columns:
                d[c] = "__NaN__"
        d = apply_encoders(d, encs)
        for c in mask_num:
            if c in d.columns:
                d[c] = np.nan
        X = d[feats].apply(pd.to_numeric, errors="coerce").fillna(-9999).values
        out = df[[COL_RID, COL_BAN, COL_JYUN, "year"]].copy()
        out["_score"] = model.predict(X)
        return out

    # ---- 死んでいる特徴を群に分類 ----
    GROUPS = {
        "調教(trnH/trnW)": ([c for c in dead_numeric if c.startswith(("trnH_", "trnW_"))], []),
        "補正(prev_hosei)": ([c for c in dead_numeric if c.startswith("prev_hosei")], []),
        "course成績": ([c for c in dead_numeric if c.startswith("course_")], []),
        "jockey成績": ([c for c in dead_numeric if c.startswith("jockey_")], []),
        "hist_same": ([c for c in dead_numeric if c.startswith("hist_same_")], []),
        "騎手/調教師コード": ([], dead_cat),
    }
    rename18_num = [c for c in dead_numeric if c.startswith(("trnH_", "trnW_", "prev_hosei"))]

    # ---- シナリオ: baseline / 本番(全マスク) / 各群を「本番状態から回収」 ----
    scenarios = {
        "baseline(全特徴)": ([], []),
        "本番(全30マスク)": (dead_numeric, dead_cat),
    }
    for gname, (gnum, gcat) in GROUPS.items():
        rec_num = [c for c in dead_numeric if c not in gnum]
        rec_cat = [c for c in dead_cat if c not in gcat]
        scenarios[f"回収+{gname}"] = (rec_num, rec_cat)
    scenarios["回収+リネーム18(調教+補正)"] = (
        [c for c in dead_numeric if c not in rename18_num], dead_cat)
    # 現状 (2026-06-15): リネーム18(補正/調教)は serve 復活済み。実際に本番で死ぬのは
    # SERVE_DEAD_NOW の12個 (hist_same×4 + course×3 + jockey×3 + 騎手/調教師コード×2)。
    # この行が「いま本番が出荷している」◎複勝圏率/ECE の直接値 (回収+リネーム18 と一致する想定)。
    serve_dead_num = [c for c in dead_numeric
                      if c.startswith(SERVE_DEAD_NOW_PREFIX) or c in SERVE_DEAD_NOW_EXACT]
    serve_dead_cat = [c for c in dead_cat if c in SERVE_DEAD_NOW_CAT]
    scenarios["現状(serve12死)"] = (serve_dead_num, serve_dead_cat)

    results = {}
    for name, (mn, mc) in scenarios.items():
        print(f"[predict] {name} (num{len(mn)}/cat{len(mc)} mask) ...")
        results[name] = evaluate(scores_for(mn, mc), payouts, cals)

    out = {"model": "v6", "n_masked": len(dead_numeric) + len(dead_cat),
           "dead_numeric": dead_numeric, "dead_cat": dead_cat,
           "scenarios": results}
    out_path = BASE / "reports" / "serve_skew_eval.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # ===== test 2024-25 の群別回収表 =====
    base = results["baseline(全特徴)"]["test_2024_2025"]
    full = results["本番(全30マスク)"]["test_2024_2025"]
    print(f"\n{'='*72}")
    print(f"=== test 2024-2025 (R={base['races']:,})  ◎複勝圏率 と ECE複勝(◎) ===")
    print(f"{'シナリオ':<26}{'◎複勝圏':>9}{'vs本番':>9}{'ECE複勝':>10}{'vs本番':>9}")
    print(f"  {'offline上限(全特徴)':<24}{base['hon_top3_pct']:>8.2f}%{'':>9}{base['ece_fuku_hon']:>10.4f}")
    print(f"  {'本番(全30マスク)':<24}{full['hon_top3_pct']:>8.2f}%{'(基準)':>9}{full['ece_fuku_hon']:>10.4f}{'(基準)':>9}")
    print("  " + "-"*68)
    order = [f"回収+{g}" for g in GROUPS] + ["回収+リネーム18(調教+補正)"]
    for name in order:
        r = results[name]["test_2024_2025"]
        d_top3 = r["hon_top3_pct"] - full["hon_top3_pct"]
        d_ece = r["ece_fuku_hon"] - full["ece_fuku_hon"]
        print(f"  {name:<24}{r['hon_top3_pct']:>8.2f}%{d_top3:>+8.2f}pt{r['ece_fuku_hon']:>10.4f}{d_ece:>+9.4f}")
    print(f"\n[saved] {out_path}")
    print("注: 「回収+X」= 本番(全30マスク)から X 群だけ正しく作れた場合。vs本番 が +pt なら X を直すと本番がそれだけ改善する。")


if __name__ == "__main__":
    main()
