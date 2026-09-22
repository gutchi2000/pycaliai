# -*- coding: utf-8 -*-
"""
nnlib.py — EXP15 の NN (R1 DeepSets / R1-noctx) と race テンソル・学習ループ
============================================================================
R1 と R1-noctx の違いは race context 経路の有無だけ (spec.models)。
- φ: 馬ごとのエンコーダ (共通)
- R1: c_i = [自分を除く平均, 自分を除く最大, log n]、ρ([h_i, c_i, h_i - m_-i])
- R1-noctx: ρ_noctx(h_i)、幅 H をパラメータ数が R1 の ±5% になる最小値に設定
BatchNorm 等バッチ統計を使う層は使わない (T2)。
"""
from __future__ import annotations

import math
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn

from . import common as C

D = 128
HMAX = 18
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def emb_dim(k: int) -> int:
    return int(min(16, math.ceil(1.6 * k ** 0.56)))


class Phi(nn.Module):
    def __init__(self, cards, n_num, W, p):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(k, emb_dim(k)) for k in cards])
        din = sum(emb_dim(k) for k in cards) + n_num
        self.net = nn.Sequential(nn.Linear(din, W), nn.LayerNorm(W), nn.GELU(), nn.Dropout(p),
                                 nn.Linear(W, D), nn.LayerNorm(D), nn.GELU())

    def forward(self, cat, num):
        e = [emb(cat[..., j]) for j, emb in enumerate(self.embs)]
        return self.net(torch.cat(e + [num], dim=-1))


def rival_context(hc, cmask, excl):
    """hc [B,H',D] 文脈集合の表現, cmask [B,H'] 有効, excl [B,H] 除外する文脈 index (-1=なし)
    → m [B,H,D] 自分(=excl)を除く平均, M [B,H,D] 自分を除く最大"""
    B, Hp, Dd = hc.shape
    cm = cmask.unsqueeze(-1).to(hc.dtype)
    S = (hc * cm).sum(1)                                   # [B,D]
    cnt = cmask.sum(1).to(hc.dtype)                        # [B]
    ex_ok = (excl >= 0)
    ex_idx = excl.clamp(min=0)
    h_ex = torch.gather(hc, 1, ex_idx.unsqueeze(-1).expand(-1, -1, Dd))      # [B,H,D]
    ex_valid = ex_ok & torch.gather(cmask, 1, ex_idx)                         # [B,H]
    exf = ex_valid.unsqueeze(-1).to(hc.dtype)
    denom = (cnt.unsqueeze(1) - ex_valid.to(hc.dtype)).clamp(min=1.0).unsqueeze(-1)
    m = (S.unsqueeze(1) - h_ex * exf) / denom
    neg = torch.finfo(hc.dtype).min
    hm = hc.masked_fill(~cmask.unsqueeze(-1), neg)
    k = min(2, Hp)
    v, ix = torch.topk(hm, k, dim=1)                      # [B,k,D]
    v1, i1 = v[:, 0], ix[:, 0]
    v2 = v[:, 1] if k > 1 else torch.full_like(v1, neg)
    is_top = (i1.unsqueeze(1) == excl.unsqueeze(-1)) & ex_valid.unsqueeze(-1)  # [B,H,D]
    M = torch.where(is_top, v2.unsqueeze(1).expand(-1, excl.shape[1], -1),
                    v1.unsqueeze(1).expand(-1, excl.shape[1], -1))
    M = torch.where(M <= neg / 2, torch.zeros_like(M), M)
    return m, M


class R1(nn.Module):
    has_context = True

    def __init__(self, cards, n_num, W, p):
        super().__init__()
        self.phi = Phi(cards, n_num, W, p)
        self.rho = nn.Sequential(nn.Linear(4 * D + 1, 256), nn.GELU(), nn.Dropout(p),
                                 nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, cat, num, mask, ctx=None):
        h = self.phi(cat, num)
        if ctx is None:
            hc, cmask = h, mask
            excl = torch.arange(h.shape[1], device=h.device).unsqueeze(0).expand(h.shape[0], -1)
        else:
            hc = self.phi(ctx["cat"], ctx["num"])
            cmask, excl = ctx["mask"], ctx["excl"]
        m, M = rival_context(hc, cmask, excl)
        logn = torch.log(mask.sum(1, keepdim=True).to(h.dtype)).unsqueeze(1).expand(-1, h.shape[1], 1)
        z = torch.cat([h, m, M, logn, h - m], dim=-1)
        return self.rho(z).squeeze(-1)


class R1NoCtx(nn.Module):
    has_context = False

    def __init__(self, cards, n_num, W, p, H):
        super().__init__()
        self.phi = Phi(cards, n_num, W, p)
        self.rho = nn.Sequential(nn.Linear(D, H), nn.GELU(), nn.Dropout(p),
                                 nn.Linear(H, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, cat, num, mask, ctx=None):
        return self.rho(self.phi(cat, num)).squeeze(-1)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def noctx_width(cards, n_num, W, p, factor=1.0, tol=0.05) -> int:
    target = factor * n_params(R1(cards, n_num, W, p))
    rho_fixed_r1 = n_params(R1(cards, n_num, W, p)) - n_params(Phi(cards, n_num, W, p))
    base = n_params(Phi(cards, n_num, W, p))
    # ρ_noctx(H) = (D+1)H + (H+1)64 + 65
    for H in range(8, 20000):
        tot = base + (D + 1) * H + (H + 1) * 64 + 65
        if factor == 1.0 and abs(tot - target) / target <= tol:
            return H
        if tot >= target:
            return H
    raise RuntimeError(rho_fixed_r1)


def build_model(kind, cards, n_num, W, p):
    if kind == "R1":
        return R1(cards, n_num, W, p)
    if kind == "R1-noctx":
        return R1NoCtx(cards, n_num, W, p, noctx_width(cards, n_num, W, p, 1.0))
    if kind == "R1-noctx-2x":
        return R1NoCtx(cards, n_num, W, p, noctx_width(cards, n_num, W, p, 2.0))
    raise ValueError(kind)


# ------------------------------------------------------------------ テンソル
class RaceTensors:
    """期間ごとの padded テンソル (GPU 常駐)。rows[r, j] = df の行番号 (-1=padding)。"""

    def __init__(self, enc: C.Encoded, df, period: str, min_field: int = 2):
        races = [i for i in C.race_index(df, (df["period"] == period).to_numpy())
                 if len(i) >= min_field]
        assert max(len(i) for i in races) <= HMAX, "T5: n>18"
        R = len(races)
        rows = np.full((R, HMAX), -1, dtype=np.int64)
        for r, idx in enumerate(races):
            rows[r, :len(idx)] = idx
        self.races, self.rows = races, rows
        valid = rows >= 0
        safe = np.where(valid, rows, 0)
        self.mask = torch.tensor(valid, device=DEV)
        self.cat = torch.tensor(enc.cat_codes[safe], device=DEV)
        self.num = torch.tensor(enc.num_nn[safe], device=DEV)
        self.num[~self.mask] = 0.0
        self.cat[~self.mask] = 0
        rel = df["rel"].to_numpy()[safe].astype(np.float32)
        self.rel = torch.tensor(np.where(valid, rel, 0.0), device=DEV)
        self.win = df["win"].to_numpy()[safe] * valid

    def __len__(self):
        return len(self.races)


@torch.no_grad()
def predict(model, T: RaceTensors, bs: int = 1024, ctx_fn=None) -> np.ndarray:
    model.eval()
    out = np.full(T.rows.shape, np.nan, dtype=np.float64)
    for a in range(0, len(T), bs):
        sl = slice(a, a + bs)
        ctx = ctx_fn(sl) if ctx_fn is not None else None
        s = model(T.cat[sl], T.num[sl], T.mask[sl], ctx)
        out[sl] = s.double().cpu().numpy()
    out[T.rows < 0] = np.nan
    return out


def tau_ll(S: np.ndarray, T: RaceTensors) -> tuple[float, float]:
    """padded スコアで τ を最尤 (単独勝ちレース) → (tau, logloss)"""
    return C.fit_tau_padded(S, T.win, T.rows >= 0)


def train(model, Ttr: RaceTensors, Tsel: RaceTensors, seed: int, lr: float,
          max_epochs: int = 30, patience: int = 4, bs: int = 256,
          ctx_tr=None, ctx_sel=None, log=None):
    """pointwise MSE on rel。ES = 2022 τ補正 win logloss。restore best。"""
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    model.to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(seed)
    best, best_state, best_ep, bad = np.inf, None, -1, 0
    hist = []
    torch.cuda.reset_peak_memory_stats() if DEV.type == "cuda" else None
    t_epochs = []
    for ep in range(max_epochs):
        model.train()
        t0 = time.time()
        perm = torch.randperm(len(Ttr), generator=gen).to(DEV)
        for a in range(0, len(Ttr), bs):
            b = perm[a:a + bs]
            ctx = ctx_tr(b) if ctx_tr is not None else None
            s = model(Ttr.cat[b], Ttr.num[b], Ttr.mask[b], ctx)
            m = Ttr.mask[b]
            loss = ((s - Ttr.rel[b]) ** 2)[m].mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if DEV.type == "cuda":
            torch.cuda.synchronize()
        t_epochs.append(time.time() - t0)
        S = predict(model, Tsel, ctx_fn=ctx_sel)
        tau, ll = tau_ll(S, Tsel)
        hist.append({"epoch": ep + 1, "sel_ll": ll, "tau": tau, "train_loss": float(loss)})
        if log:
            log(f"    ep{ep + 1:02d} sel_ll={ll:.5f} tau={tau:.3f} ({t_epochs[-1]:.1f}s)")
        if ll < best - 1e-6:
            best, best_ep, bad = ll, ep + 1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    peak = torch.cuda.max_memory_allocated() / 2 ** 20 if DEV.type == "cuda" else float("nan")
    return {"best_sel_ll": best, "es_epoch": best_ep, "hist": hist,
            "epoch_seconds": float(np.mean(t_epochs)), "peak_gpu_MB": float(peak)}


def set_determinism():
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def flops_per_race(model, cards, n_num, n=14) -> float:
    """概算 (乗算加算=2 FLOPs、Linear のみ + 文脈集約)"""
    tot = 0
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            tot += 2 * mod.in_features * mod.out_features * n
    if getattr(model, "has_context", False):
        tot += 2 * n * n * D  # 平均・最大の集約 (自分除外)
    return float(tot)
