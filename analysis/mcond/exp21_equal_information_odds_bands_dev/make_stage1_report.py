# -*- coding: utf-8 -*-
"""make_stage1_report.py — out/stage1_results.json から STAGE1_REPORT.md の表を生成する (数値の転記ミスを避ける)"""
from __future__ import annotations

import json

from .loaders import HERE, OUT

NAMES = {"tansho": "単勝", "fukusho": "複勝", "umaren": "馬連"}


def band_rows(rows):
    out = ["| 帯 | odds 範囲 | ticket | race | 的中 | ROI [CI95] | 基準 | 基準との差 [CI95] | 不足 |",
           "|---|---|---:|---:|---:|---|---:|---|---|"]
    for r in rows:
        if not r.get("tickets"):
            out.append(f"| {r['band'] + 1} | — | 0 | — | — | — | — | — | 空 |")
            continue
        out.append(f"| {r['band'] + 1} | {r['odds_min']:.1f}–{r['odds_max']:.1f} | {r['tickets']:,} | {r['races_days'][0]:,} | "
                   f"{r['hits']:,} | {r['roi']:.3f} [{r['roi_ci95'][0]:.3f}, {r['roi_ci95'][1]:.3f}] | {r['base']:.3f} | "
                   f"{r['uplift']:+.3f} [{r['uplift_ci95'][0]:+.3f}, {r['uplift_ci95'][1]:+.3f}] | "
                   f"{'insufficient' if r.get('insufficient_label_free') else '—'} |")
    return "\n".join(out)


def main():
    r = json.loads((OUT / "stage1_results.json").read_text(encoding="utf-8"))
    parts = []
    for t, v in r["types"].items():
        L = v["layers"]
        g1, g2 = v["G1"], v["G2"]
        parts.append(f"## {NAMES[t]}（{t}）\n")
        g1s = (f"Spearman {g1['spearman']:.3f}" + (f"、符号一致 {g1['sign_agreement']}/10" if "sign_agreement" in g1 else
                                                  f"、符号一致（副）{g1['sign_agreement_secondary']}/10") +
               f"、20 帯感度の Spearman {g1['sensitivity_20_bins_spearman']:.3f}")
        parts.append(f"- **G1 = {g1['grade']}**（{g1s}）")
        if g2["grade"] in ("PASS", "FAIL"):
            parts.append(f"- **G2 = {g2['grade']}**（選択帯 {[b + 1 for b in g2['selected_bands']]}、discovery D0 uplift "
                         f"{g2['uplift_discovery_D0']:+.4f} → evaluation D1 {g2['uplift_evaluation_D1']:+.4f}、残存率 "
                         f"{g2['retention']:.2f}、CI95 下限 {g2['uplift_evaluation_D1_ci95_lower']:+.4f}、evaluation D1 の ROI "
                         f"{g2['evaluation_D1_roi_S']:.3f}（1.0 超: {g2['evaluation_D1_roi_S_above_1']}）、ticket "
                         f"{g2['tickets_S_evaluation_D1']:,}）")
        else:
            parts.append(f"- **G2 = {g2['grade']}**（{g2.get('reason', '')}）")
        for lk in ("discovery|D0", "evaluation|D0", "discovery|D1", "evaluation|D1"):
            x = L[lk]
            parts.append(f"\n### {lk}（ticket {x['tickets']:,} / race {x['races']:,} / 暦日 {x['days']}、全 ticket プール ROI "
                         f"{x['pool_roi']:.3f} [{x['pool_roi_ci95'][0]:.3f}, {x['pool_roi_ci95'][1]:.3f}]、同着・不整合で除外 "
                         f"{x['excluded_races']['dead_heat_or_irregular']} race）\n")
            parts.append(band_rows(x["tables"]["primary_mass10"]))
        d = [f"{x['roi_D1_minus_D0']:+.3f}" for x in v["evaluation_D1_minus_D0"]]
        parts.append(f"\nevaluation 期の D1 − D0（帯 1〜10 の ROI 差）: {', '.join(d)}")
        tr = v["evaluation_transition_D1band_to_D0band"]
        diag = sum(tr[i][i] for i in range(10)) / sum(sum(row) for row in tr)
        parts.append(f"D1 帯と D0 帯が一致する ticket の比率（遷移行列の対角）: {diag:.3f}\n")
    (HERE / "STAGE1_TABLES.md").write_text("# EXP21 Stage 1 — primary 10 帯の全表（自動生成）\n\n"
                                          "基準: 単勝・馬連 = 帯の較正 null（terminal 価格の 1/overround の ticket 平均）、複勝 = 同期間・同価格層の全 ticket プール ROI。\n"
                                          "副解析（等 ticket-count 帯・固定帯・20 帯）と集中除外は `out/stage1_results.json`。\n\n"
                                          + "\n".join(parts) + "\n", encoding="utf-8")
    print("ok")


if __name__ == "__main__":
    main()
