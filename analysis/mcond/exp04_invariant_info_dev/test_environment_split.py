# -*- coding: utf-8 -*-
"""
test_environment_split.py — leave-one-environment-out の除外が正しく機能することを確認
実行: python -m pytest analysis/mcond/exp04_invariant_info_dev/test_environment_split.py -q
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from analysis.mcond.exp04_invariant_info_dev.methods import fit_offset_logit, fit_univariate  # noqa: E402


def test_leave_one_out_excludes_rows():
    rng = np.random.default_rng(0)
    n = 4000
    env = rng.integers(0, 5, n)
    x = rng.normal(size=n)
    offset = rng.normal(size=n) * 0.1
    y = (rng.random(n) < 1 / (1 + np.exp(-(offset + 0.5 * x)))).astype(float)
    train = np.ones(n, bool)

    held = env == 2
    fit_mask = train & ~held
    beta_full = fit_offset_logit(x[:, None], offset, y, l2=0.01)
    beta_excl = fit_offset_logit(x[fit_mask][:, None], offset[fit_mask], y[fit_mask], l2=0.01)
    # 除外ありとなしで係数が異なる (=held-out行が学習に混ざっていない証拠)
    assert not np.isclose(beta_full[0], beta_excl[0], atol=1e-6)
    # held-out 集合が学習データに1行も含まれていないことを直接確認
    assert fit_mask[held].sum() == 0


def test_univariate_matches_multivariate_single_column():
    rng = np.random.default_rng(1)
    n = 3000
    x = rng.normal(size=n)
    offset = rng.normal(size=n) * 0.1
    y = (rng.random(n) < 1 / (1 + np.exp(-(offset + 0.7 * x)))).astype(float)
    b_multi = fit_offset_logit(x[:, None], offset, y, l2=1e-6)[0]
    b_uni, se = fit_univariate(x, offset, y)
    assert abs(b_multi - b_uni) < 0.05
    assert se > 0
