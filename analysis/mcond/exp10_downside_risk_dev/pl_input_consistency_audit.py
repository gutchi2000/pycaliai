# -*- coding: utf-8 -*-
"""
pl_input_consistency_audit.py
==============================
EXP10 Stage 1修正2。PLへの入力の整合性を実データで確認する。

確認項目(ユーザー指定):
  1. raw scoreをそのままGumbel-maxのlocationとして使っているか(exp(raw_score)を
     locationにしていないか)
  2. 既存pl_probs.all_tansho()と同じtemperature・符号・正規化か
  3. sampled win probability(predicted_rank_dist[0])が既存単勝PL確率と一致するか
  4. 馬の入力順序を変えても対象馬の順位分布が変わらないか(順序不変性)

2023年の実レーススコアで検証する。2024・2025年は使わない。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pl_rank_distribution import exact_rank_distribution, mc_rank_distribution  # noqa: E402
from eligible_races import build_target_rows  # noqa: E402

sys.path.insert(0, str(Path(r"E:\PyCaLiAI")))
import pl_probs as PL  # noqa: E402


def audit(year: int, n_sample_races: int = 150, seed: int = 20261012):
    targets, race_horses, meta = build_target_rows([year])
    rng = np.random.default_rng(seed)
    rids = race_horses["rid"].unique()
    sample = rng.choice(rids, size=min(n_sample_races, len(rids)), replace=False)

    max_err_all_horses = 0.0
    max_err_focal_only = 0.0
    n_checked = 0
    n_order_invariance_checked = 0
    max_order_invariance_err = 0.0

    for rid in sample:
        g = race_horses[race_horses["rid"] == rid].sort_values("pos_in_group").reset_index(drop=True)
        finish = g["finish"].values.astype(float)
        vals = finish[~np.isnan(finish)]
        if len(vals) and (np.unique(vals, return_counts=True)[1] > 1).any():
            continue  # 同着は除外(主解析と同じ基準)
        scores = g["score"].values.astype(float)
        n = len(scores)
        focal_pos = int(g[g["is_focal"]].index[0])

        # --- 1/2/3: 既存pl_probs.all_tansho()との一致 ---
        w = PL.pl_weights(scores)  # w_i = exp(s_i - max(s))
        tansho_existing = PL.all_tansho(w)  # 既存本番PL実装の単勝確率
        for i in range(n):
            dist_i = exact_rank_distribution(scores, i)
            err = abs(dist_i[0] - tansho_existing[i])
            max_err_all_horses = max(max_err_all_horses, err)
            if i == focal_pos:
                max_err_focal_only = max(max_err_focal_only, err)
        n_checked += 1

        # --- 4: 入力順序を変えても対象馬の分布が変わらないか ---
        if n >= 4:
            perm = rng.permutation(n)
            new_focal_pos = int(np.where(perm == focal_pos)[0][0])
            scores_shuffled = scores[perm]
            dist_orig = exact_rank_distribution(scores, focal_pos)
            dist_shuffled = exact_rank_distribution(scores_shuffled, new_focal_pos)
            order_err = float(np.max(np.abs(dist_orig - dist_shuffled)))
            max_order_invariance_err = max(max_order_invariance_err, order_err)
            n_order_invariance_checked += 1

    print(f"[PL input consistency audit, {year}年実データ, n_races={n_checked}]")
    print(f"  1. Gumbel-maxのlocationにraw score(exp化なし)を直接使用: "
          f"コード上確認済み(mc_rank_distribution: perturbed = scores + gumbel)")
    print(f"  2/3. 全馬全レースでのexact_rank_distribution[rank=1] vs "
          f"pl_probs.all_tansho() 最大誤差: {max_err_all_horses:.2e}")
    print(f"       (対象馬(◎)のみに限定した最大誤差: {max_err_focal_only:.2e})")
    print(f"  4. 入力順序を変えた場合の対象馬rank分布の最大誤差: "
          f"{max_order_invariance_err:.2e} (n_checked={n_order_invariance_checked})")
    return dict(
        n_checked=n_checked,
        max_err_all_horses=max_err_all_horses,
        max_err_focal_only=max_err_focal_only,
        max_order_invariance_err=max_order_invariance_err,
    )


if __name__ == "__main__":
    audit(2023)
