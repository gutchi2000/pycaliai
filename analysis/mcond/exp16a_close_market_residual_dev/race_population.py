# -*- coding: utf-8 -*-
"""
race_population.py — EXP16A Stage 0: 正式 race set の確定と母集団監査 + 2022 の基準値再計算
============================================================================================
学習しない。2023 の指標は計算しない (2022 selection のみ)。ROI も候補生成もしない。

やること:
  1. 馬の分類: finisher (master) / DNF (締切プールに居たが着順なし) / 取消 (締切前) / 発走除外 (締切後)
     - 異常コードの意味は「確定オッズを持つ割合」から実証的に判定する (仮定しない)
  2. 正式 race set: 勝馬が一意・出走頭数>=5・DNF なし・締切後取消なし・pre/close 両方で de-vig 可能
  3. 感度分析: finisher 再正規化方式 と 全 starter 正規化方式 の Q0 logloss 差
  4. 2022 のみ: terminal_close_market / historical_pre_snapshot の基準値、pre→close gap、MDE、subset 閾値
出力: out/race_population.json, STAGE0_DRY_RUN.json (更新)
実行: python -m analysis.mcond.exp16a_close_market_residual_dev.race_population
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .provenance import (BASE, HERE, OUT, PERIODS, MIN_GAP_PRE, load_master, load_tanpuk,
                         odds_matrix, devig, sha256, TANPUK_DIR)

EXT_KEKKA = Path(r"E:\競馬過去走データ\raw_data\kekka_1986_2025_enhanced.csv")
Z = 1.959964 + 0.841621  # 両側5% / 検出力80%


def load_ext_codes() -> pd.DataFrame:
    """外部 kekka の 異常コード (2016-2023)。join key = 日付+場所+R+馬番"""
    use = ["年", "月", "日", "場所", "レース番号", "馬番", "確定着順", "異常コード"]
    parts = []
    for ch in pd.read_csv(EXT_KEKKA, encoding="cp932", dtype=str, usecols=use, chunksize=300_000):
        y = pd.to_numeric(ch["年"], errors="coerce")
        parts.append(ch[(y >= 16) & (y <= 23)])
    e = pd.concat(parts, ignore_index=True)
    e["date"] = (2000 + pd.to_numeric(e["年"], errors="coerce")) * 10000 + \
                pd.to_numeric(e["月"], errors="coerce") * 100 + pd.to_numeric(e["日"], errors="coerce")
    e["R"] = pd.to_numeric(e["レース番号"], errors="coerce")
    e["ban"] = pd.to_numeric(e["馬番"], errors="coerce")
    e["code"] = pd.to_numeric(e["異常コード"], errors="coerce").fillna(0).astype(int)
    e["jyun"] = pd.to_numeric(e["確定着順"], errors="coerce")
    return e.dropna(subset=["date", "R", "ban"])[["date", "場所", "R", "ban", "code", "jyun"]]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = load_master()
    m["R"] = pd.to_numeric(m["レースID(新/馬番無)"].astype(str).str[14:16], errors="coerce")
    # 検出力監査用の time-safe な表信号 (前走確定着順) を別読みして結合する
    sig_parts = []
    for ch in pd.read_csv(BASE / "data" / "master_v2_20130105-20251228.csv", encoding="utf-8-sig",
                          dtype=str, usecols=["日付", "レースID(新/馬番無)", "馬番", "前走確定着順"],
                          chunksize=200_000):
        dd = pd.to_numeric(ch["日付"], errors="coerce")
        sig_parts.append(ch[(dd >= 20160101) & (dd <= 20231231)])
    sg = pd.concat(sig_parts, ignore_index=True)
    sg["rid16"] = sg["レースID(新/馬番無)"].astype(str).str.replace(r"\.0$", "", regex=True).str[:16]
    sg["ban"] = pd.to_numeric(sg["馬番"], errors="coerce")
    m = m.merge(sg[["rid16", "ban", "前走確定着順"]], on=["rid16", "ban"], how="left")
    t = load_tanpuk(set(m["rid16"]))
    ext = load_ext_codes()
    print(f"master {len(m):,} / tanpuk {len(t):,} / ext {len(ext):,}", flush=True)

    meta = (m.groupby("rid16")
              .agg(date=("date", "first"), year=("year", "first"), period=("period", "first"),
                   post_min=("post_min", "first"), venue=("場所", "first"), surface=("芝・ダ", "first"),
                   cls=("クラス名", "first"), R=("R", "first"), n_fin=("ban", "size")))
    fin_set = m.groupby("rid16")["ban"].apply(set)
    winner = m[m["win"] == 1].groupby("rid16")["ban"].apply(list)
    ext_idx = ext.set_index(["date", "場所", "R", "ban"])["code"]

    rows = []
    for rid, g in t.groupby("rid16", sort=False):
        if rid not in meta.index:
            continue
        md = meta.loc[rid]
        mmdd = f"{(md.date % 10000):04d}"
        fin_rows = g[g["kubun"] == 4]
        pre_c = g[(g["kubun"] == 1) & (g["mmdd"] == mmdd)]
        if not len(fin_rows) or not len(pre_c):
            continue
        f0 = fin_rows.iloc[-1]
        gap = f0.snap_min - pre_c["snap_min"]
        ok = pre_c[gap >= MIN_GAP_PRE]
        if not len(ok):
            continue
        p0 = ok.iloc[-1]
        od_fin, od_pre = odds_matrix(f0), odds_matrix(p0)
        S_fin, S_pre, F = set(od_fin), set(od_pre), fin_set[rid]
        dnf_or_late = sorted(S_fin - F)          # 締切プールに居たが着順なし
        pre_only = sorted(S_pre - S_fin)          # 締切前に消えた = 取消
        fin_not_market = sorted(F - S_fin)        # 着順はあるが確定オッズなし (異常)
        codes = {}
        for b in dnf_or_late + pre_only:
            k = (int(md.date), md.venue, int(md.R) if pd.notna(md.R) else -1, int(b))
            codes[b] = int(ext_idx.get(k, -9))
        rows.append({
            "rid16": rid, "date": int(md.date), "year": int(md.year), "period": md.period,
            "venue": md.venue, "surface": md.surface, "cls": md.cls, "n_fin": int(md.n_fin),
            "n_market_final": len(S_fin), "n_market_pre": len(S_pre),
            "dnf_or_late_scratch": dnf_or_late, "pre_only_scratch": pre_only,
            "fin_not_in_market": fin_not_market, "codes": codes,
            "winner": winner.get(rid, [None])[0] if rid in winner.index else None,
            "dead_heat": len(winner.get(rid, [])) > 1 if rid in winner.index else False,
            "od_fin": od_fin, "od_pre": od_pre, "starters": sorted(F),
        })
    R = pd.DataFrame(rows)
    print(f"races joined {len(R):,}", flush=True)

    # ---- 異常コードの意味を実証的に判定 (確定オッズを持つ割合)
    code_stat = {}
    for _, r in R.iterrows():
        for b, c in r["codes"].items():
            d = code_stat.setdefault(c, {"n": 0, "has_final_odds": 0})
            d["n"] += 1
            d["has_final_odds"] += int(b in r["od_fin"])
    for c, d in code_stat.items():
        d["share_with_final_odds"] = d["has_final_odds"] / d["n"] if d["n"] else None
    # 締切プールに残っていた (=出走した/できる状態) 側を DNF 系、消えていた側を取消系とみなす
    dnf_codes = sorted(c for c, d in code_stat.items() if d["n"] >= 20 and d["share_with_final_odds"] >= 0.9)

    # ---- 正式 race set
    def classify(r):
        if r["dead_heat"] or r["winner"] is None:
            return "excl_no_unique_winner"
        if r["n_fin"] < 5:
            return "excl_small_field"
        if r["fin_not_in_market"]:
            return "excl_finisher_without_market"
        if r["dnf_or_late_scratch"]:
            return "excl_dnf_or_late_scratch"
        return "official"
    R["cls_set"] = R.apply(classify, axis=1)
    per_year = {}
    for y, g in R.groupby("year"):
        per_year[int(y)] = {
            "races": int(len(g)),
            "official": int((g["cls_set"] == "official").sum()),
            **{k: int((g["cls_set"] == k).sum()) for k in
               ["excl_no_unique_winner", "excl_small_field", "excl_finisher_without_market",
                "excl_dnf_or_late_scratch"]},
            "horses_dnf_or_late": int(g["dnf_or_late_scratch"].map(len).sum()),
            "horses_pre_only_scratch": int(g["pre_only_scratch"].map(len).sum()),
            "races_with_pre_only_scratch": int((g["pre_only_scratch"].map(len) > 0).sum()),
        }

    # ---- 2022 基準値 (正式 race set) と感度分析
    def ll_top1(sub, key, mode):
        lls, hit = [], []
        for _, r in sub.iterrows():
            od = r[key]
            starters = r["starters"] if mode == "starters" else [b for b in r["starters"] if b in od]
            pi, ov, miss = devig(od, starters)
            w = r["winner"]
            if not pi or w not in pi:
                continue
            lls.append(-np.log(max(pi[w], 1e-12)))
            hit.append(int(max(pi, key=pi.get) == w))
        return np.array(lls), float(np.mean(hit)) if hit else None

    off = R[(R["cls_set"] == "official") & (R["period"] == "selection")]
    ll_close, top1_close = ll_top1(off, "od_fin", "starters")
    ll_pre, top1_pre = ll_top1(off, "od_pre", "starters")
    # 感度: 旧来方式 (finisher だけで再正規化) — 正式 set では starters==finishers なので一致するはず
    ll_close_f, _ = ll_top1(off, "od_fin", "finishers")
    # DNF ありレースも含めた場合 (参考)
    dnf_sub = R[(R["period"] == "selection") & (R["cls_set"] == "excl_dnf_or_late_scratch")]
    ll_close_dnf_starters, _ = ll_top1(dnf_sub, "od_fin", "starters")
    ll_close_dnf_fin, _ = ll_top1(dnf_sub, "od_fin", "finishers")

    # ---- 検出力監査 (一次近似)
    # π を time-safe な表信号 z (race 内 z 化) で僅かに傾けると q ∝ π exp(δz) となり、
    # per-race の改善は一次で Δ_r ≈ δ·u_r,  u_r = z_winner - Σ_i π_i z_i。
    # 平均も SE も δ に比例するので「何 nats を検出できるか」は δ ではなく **u の形** で決まる。
    # そこで u の平均 c・meeting-day cluster SE を測り、0.005 nats 相当に換算した MDE を出す。
    prevpos = pd.to_numeric(m["前走確定着順"], errors="coerce")
    mm = m.assign(v=prevpos)
    sig = {}
    for rid, g in mm[mm["rid16"].isin(set(off["rid16"]))].groupby("rid16"):
        v = g["v"].to_numpy(dtype=float)
        mu = np.nanmean(v)
        v = np.where(np.isnan(v), mu if np.isfinite(mu) else 0.0, v)
        z = (v - v.mean()) / (v.std() + 1e-9)
        sig[rid] = dict(zip(g["ban"].to_numpy(), -z))     # 前走着順が良い(小さい)ほど +
    u_list, u_days = [], []
    for _, r in off.iterrows():
        pi, _, _ = devig(r["od_fin"], r["starters"])
        w, s = r["winner"], sig.get(r["rid16"], {})
        if not pi or w not in pi:
            continue
        zz = {b2: s.get(b2, 0.0) for b2 in pi}
        u_list.append(zz[w] - sum(pi[b2] * zz[b2] for b2 in pi))
        u_days.append(r["rid16"][:10])
    u = np.array(u_list); u_day = np.array(u_days)
    c_mean = float(u.mean()); u_sd = float(u.std(ddof=1))

    day = off["rid16"].str[:10].to_numpy()
    d = ll_pre - ll_close
    rng = np.random.default_rng(20260924)
    days = np.unique(day)
    pos = {x: np.where(day == x)[0] for x in days}
    bs = np.array([d[np.concatenate([pos[x] for x in rng.choice(days, len(days))])].mean()
                   for _ in range(3000)])
    se = float(bs.std(ddof=1))
    mde = float(Z * se)
    ud = np.unique(u_day)
    upos = {x: np.where(u_day == x)[0] for x in ud}
    bs_u = np.array([u[np.concatenate([upos[x] for x in rng.choice(ud, len(ud))])].mean()
                     for _ in range(3000)])
    se_u = float(bs_u.std(ddof=1))
    ratio = float(Z * se_u / abs(c_mean)) if c_mean else None      # 1 以下なら 0.005 を検出できる
    mde_tilt = float(0.005 * ratio) if ratio is not None else None

    # ---- subset 閾値 (2022 正式 set のみ)
    fav, ent, votes = [], [], []
    tv = t[(t["kubun"] == 1)].set_index(["rid16", "snap_min"])["votes_tan"].to_dict()
    for _, r in off.iterrows():
        pi, _, _ = devig(r["od_pre"], r["starters"])
        if pi:
            fav.append(1.0 / max(pi.values()))
            ent.append(float(-sum(p * np.log(p) for p in pi.values() if p > 0)))
    fav, ent = np.array(fav), np.array(ent)

    pop = {
        "scope": "2016-2023 のみ",
        "anomaly_code_identification": {
            "method": "異常コード別に『確定オッズを持つ割合』を実測。締切プールに残っていた = 出走可能だった側を DNF 系とみなす",
            "by_code": code_stat,
            "codes_treated_as_ran_or_in_pool": dnf_codes,
            "code_-9": "外部 kekka に該当行が無い (join 失敗)",
        },
        "official_race_set_rule": [
            "勝馬が一意 (同着除外)", "出走頭数 >= 5", "着順のある馬が全員 確定オッズを持つ",
            "確定プールに居て着順が無い馬 (DNF / 締切後除外) が 0 頭",
            "pre と確定の両方で de-vig 可能",
        ],
        "per_year": per_year,
        "totals": {k: int(sum(v[k] for v in per_year.values()))
                   for k in ["races", "official", "excl_no_unique_winner", "excl_small_field",
                             "excl_finisher_without_market", "excl_dnf_or_late_scratch",
                             "horses_dnf_or_late", "horses_pre_only_scratch",
                             "races_with_pre_only_scratch"]},
        "post_close_refund_audit": {
            "定義": "確定プールに居たのに着順が無い馬 = DNF または 締切後の発走除外。後者は返還が発生する",
            "件数": "per_year.horses_dnf_or_late (異常コード内訳は anomaly_code_identification 参照)",
            "pre→close の間に消えた馬 (= 締切前取消、pre 側にだけ居る)": "per_year.horses_pre_only_scratch",
        },
        "sensitivity_2022": {
            "official_set_races": int(len(ll_close)),
            "Q0_ll_starter_normalization": float(ll_close.mean()),
            "Q0_ll_finisher_renormalization": float(ll_close_f.mean()),
            "diff": float(ll_close.mean() - ll_close_f.mean()),
            "note": "正式 set では starters == finishers なので定義上一致する。差は DNF ありレースでのみ生じる",
            "dnf_races_only": {
                "n_races": int(len(ll_close_dnf_starters)),
                "Q0_ll_starter_normalization": float(ll_close_dnf_starters.mean()) if len(ll_close_dnf_starters) else None,
                "Q0_ll_finisher_renormalization": float(ll_close_dnf_fin.mean()) if len(ll_close_dnf_fin) else None,
                "diff": float(ll_close_dnf_starters.mean() - ll_close_dnf_fin.mean()) if len(ll_close_dnf_starters) else None,
            },
        },
    }
    (OUT / "race_population.json").write_text(json.dumps(pop, ensure_ascii=False, indent=1, default=float),
                                              encoding="utf-8")

    dry = {
        "purpose": "Stage 0 の実行可能性確認。学習なし・2023 は評価しない・ROI なし",
        "official_race_set": {k: {"races": per_year[k2]["official"] for k2 in per_year
                                  if PERIODS[k][0] // 10000 <= k2 <= PERIODS[k][1] // 10000}
                              for k in PERIODS},
        "official_counts_by_period": {
            k: int(((R["cls_set"] == "official") & (R["period"] == k)).sum()) for k in PERIODS},
        "reference_values_2022_official_set": {
            "n_races": int(len(ll_close)),
            "terminal_close_market": {"race_categorical_logloss": float(ll_close.mean()),
                                      "favorite_top1_rate": top1_close},
            "historical_pre_snapshot": {"race_categorical_logloss": float(ll_pre.mean()),
                                        "favorite_top1_rate": top1_pre},
            "pre_minus_close_gap_nats": float(ll_pre.mean() - ll_close.mean()),
            "R0_clean_table_only": "Stage 1 で rolling OOF (train<=2020 / ES 2021 / predict 2022) を作ってから同一 race set で再計算する。EXP15 の 2022 スコアは ES に 2022 を使っているため OOS ではなく、ここでは使わない",
        },
        "power_audit_2022": {
            "primary": {
                "method": "一次近似。π_close を time-safe な表信号 z (前走確定着順の race 内 z) で δ だけ傾けたとき "
                          "per-race 改善は Δ_r ≈ δ·u_r, u_r = z_winner − Σ π_i z_i。平均も SE も δ に比例するので、"
                          "検出可否は u の形 (平均/cluster SE 比) で決まる",
                "u_mean_c": c_mean, "u_sd": u_sd, "u_cluster_se": se_u,
                "detect_ratio_z*SE/|c|": ratio,
                "MDE_at_0.005_shape_nats": mde_tilt,
                "practical_floor_nats_per_race": 0.005,
                "MDE_le_floor": bool(mde_tilt is not None and mde_tilt <= 0.005),
                "interpretation": "detect_ratio <= 1 なら、この信号と同じ per-race 分散形をもつ 0.005 nats の効果を "
                                  "80% 検出力で検出できる。逆に > 1 なら検出力不足",
                "caveat": "この参照信号 (前走着順) 自体は close 市場に対して改善方向ではない。ここで使っているのは "
                          "効果量ではなく per-race 分散の形だけ",
            },
            "secondary_reference": {
                "method": "per-race (LL(pre) - LL(close))。差が大きい対なので SE も大きく、Gate A の代理としては過大",
                "se_cluster": se, "MDE_80pct_power": mde,
            },
            "decision_rule": "primary の MDE <= 0.005 なら 2023 評価へ進行可。> 0.005 なら 2023 を開封せず停止または設計変更",
        },
        "subset_thresholds_fixed_on_2022_official_set": {
            "field_size_bins": [[5, 8], [9, 12], [13, 15], [16, 18]],
            "surface": ["芝", "ダ"],
            "venue": sorted(off["venue"].dropna().unique().tolist()),
            "class_groups": "新馬/未勝利/1勝/2勝/3勝/OP以上",
            "favorite_pre_odds_tertiles": [float(np.quantile(fav, 1/3)), float(np.quantile(fav, 2/3))],
            "market_entropy_pre_tertiles": [float(np.quantile(ent, 1/3)), float(np.quantile(ent, 2/3))],
            "n_races_used": int(len(fav)),
            "model_market_disagreement": "Q1 rolling OOF が必要なため Stage 1 で 2022 のみを使って固定する",
        },
        "files_sha256": {p.name: sha256(p)[:16] for p in sorted(TANPUK_DIR.glob("TANPUK_*.csv"))},
        "notes": ["2023 development の指標は Stage 0 では一切計算していない",
                  "2022 の R0-clean 基準値は rolling OOF 未作成のため空欄 (Stage 1 で埋める)"],
    }
    (HERE / "STAGE0_DRY_RUN.json").write_text(json.dumps(dry, ensure_ascii=False, indent=1, default=float),
                                              encoding="utf-8")
    print(json.dumps({k: v for k, v in pop["totals"].items()}, ensure_ascii=False))
    print(json.dumps(dry["reference_values_2022_official_set"], ensure_ascii=False)[:400])
    print(json.dumps(dry["power_audit_2022"], ensure_ascii=False))
    print("code_stat:", json.dumps(code_stat, ensure_ascii=False))


if __name__ == "__main__":
    main()
