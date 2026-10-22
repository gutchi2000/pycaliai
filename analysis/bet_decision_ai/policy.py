# -*- coding: utf-8 -*-
"""
policy.py — 馬券判断 AI（仕様 §4）
==================================
NN 版（主）: 候補券ごとの MLP スコア u_i と、レースごとの「見送り」ロジット a を並べた softmax で
            賭け金の割合 s_i = softmax([a, u_1..u_S])_{1..S} を出す（Σs ≤ 1、残り = 見送り）。
            a = MLP(券の隠れ表現のマスク付き平均) + b0。b0 の初期値を大きくして「見送り」を既定にする。
            目的: mean log(1 + κ·Σ s_i (R_i − 1)) − l2·mean Σ s_i²（κ = 0.05）。期待値の最大化は使わない。
GBM 版（比較）: 券ごとの対数効用 y = log(1 + κ(R − 1))（その券に予算全額を賭けたときの値）を LightGBM で回帰し、
            予測が閾値 thr を超える券に予測値比例で予算全額を配分（無ければ見送り）。
B4 市場版: 同じ NN を、入力の p_model を q_market に差し替えた特徴（data.ticket_features(mode='market')）で学習。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from analysis.bet_decision_ai.data import KAPPA


class PolicyNet(nn.Module):
    """stake_mode:
      'softmax_abstain' : s = softmax([a, u_1..u_S])[1:]（見送りトークン付き softmax）
      'sigmoid_ticket'  : s_i = σ(u_i − c) / max(1, Σ_j σ(u_j − c))（券ごとの独立ゲート、合計が 1 を超えたら按分）。
                          c = abstain_bias（初期は全券ほぼ 0 = 見送り既定）。見送りロジット a は使わない。
      'relu_st'         : s_i = relu(u_i) / max(1, Σ_j relu(u_j))。u の初期値は −0.01（全券 0 = 見送り既定）。
                          逆伝播だけ負側にも傾き leak を通す（straight-through）。s=0 の境界でも目的の方向微分
                          κ(R−1) が届くので、σ/softmax のように「全部見送り」で勾配が消えて探索が止まることが無い。"""

    def __init__(self, n_feat: int, width: int, depth: int, abstain_bias: float = 6.0, dropout: float = 0.0,
                 stake_mode: str = "softmax_abstain"):
        super().__init__()
        assert stake_mode in ("softmax_abstain", "sigmoid_ticket", "relu_st")
        self.stake_mode = stake_mode
        layers, d = [], n_feat
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.GELU()]
            if dropout:
                layers.append(nn.Dropout(dropout))
            d = width
        self.enc = nn.Sequential(*layers)
        self.score = nn.Linear(width, 1)
        self.abst = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
        self.b0 = nn.Parameter(torch.tensor(float(abstain_bias)))
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)
        self.score_offset = -0.01 if stake_mode == "relu_st" else 0.0

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.enc(X)                                     # [R,S,W]
        u = self.score(h).squeeze(-1)                       # [R,S]
        if self.stake_mode == "relu_st":
            leak = 0.1
            z = u + self.score_offset
            g = (torch.relu(z) + leak * (z - z.detach())) * mask.float()
            tot = torch.relu(z).mul(mask.float()).sum(1, keepdim=True).detach().clamp(min=1.0)
            return g / tot
        if self.stake_mode == "sigmoid_ticket":
            g = torch.sigmoid(u - self.b0) * mask.float()
            return g / g.sum(1, keepdim=True).clamp(min=1.0)
        u = u.masked_fill(~mask, float("-inf"))
        mf = mask.unsqueeze(-1).float()
        pooled = (h * mf).sum(1) / mf.sum(1).clamp(min=1.0)
        a = self.abst(pooled).squeeze(-1) + self.b0         # [R]
        logits = torch.cat([a.unsqueeze(1), u], dim=1)
        return torch.softmax(logits, dim=1)[:, 1:]          # [R,S], Σ ≤ 1


def log_growth(s: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    return torch.log1p(KAPPA * (s * (R - 1.0)).sum(1))


def predict_nn(model: PolicyNet, X: np.ndarray, mask: np.ndarray, device: str, bs: int = 4096) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs]).to(device)
            mb = torch.from_numpy(mask[i:i + bs]).to(device)
            out.append(model(xb, mb).float().cpu().numpy())
    return np.concatenate(out)


def train_nn(cfg: dict, Xtr, Mtr, Rtr, Xsel, Msel, Rsel, seed: int, device: str, log=print) -> tuple[PolicyNet, dict]:
    """学習期間で学習し、選択期間の平均対数成長（連続の割合）で早期打ち切り。最良エポックの重みを返す。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = PolicyNet(Xtr.shape[-1], cfg["width"], cfg["depth"], cfg.get("abstain_bias", 6.0),
                      cfg.get("dropout", 0.0), cfg.get("stake_mode", "softmax_abstain")).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("wd", 0.0))
    Xt = torch.from_numpy(Xtr).to(device)
    Mt = torch.from_numpy(Mtr).to(device)
    Rt = torch.from_numpy(Rtr).to(device)
    Rs = torch.from_numpy(Rsel).to(device)
    n = len(Xt)
    bs = cfg.get("batch", 512)
    g = torch.Generator(device="cpu").manual_seed(seed)
    best, best_ep, best_state, hist, bad = -np.inf, -1, None, [], 0
    for ep in range(cfg["max_epochs"]):
        model.train()
        perm = torch.randperm(n, generator=g)
        tl = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs].to(device)
            s = model(Xt[idx], Mt[idx])
            obj = log_growth(s, Rt[idx]).mean()
            pen = cfg["l2"] * (s ** 2).sum(1).mean()
            loss = -obj + pen
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += float(obj) * len(idx)
        ssel = torch.from_numpy(predict_nn(model, Xsel, Msel, device)).to(device)
        gsel = float(log_growth(ssel, Rs).mean())
        frac = float(ssel.sum(1).mean())
        hist.append({"epoch": ep, "train_lg": tl / n, "sel_lg": gsel, "sel_mean_stake_frac": frac})
        if gsel > best + 1e-9:
            best, best_ep, bad = gsel, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_ep, "best_sel_lg": best, "history": hist}


# ------------------------------------------------------------------ GBM 版
def gbm_target(ret: np.ndarray) -> np.ndarray:
    return np.log1p(KAPPA * (ret - 1.0))


def gbm_stakes(pred: np.ndarray, mask: np.ndarray, thr: float) -> np.ndarray:
    """pred[R,S]（券ごとの予測対数効用）→ 割合。thr 超の券に予測値比例で全額。無ければ見送り。"""
    p = np.where(mask & (pred > thr), pred, 0.0)
    tot = p.sum(1, keepdims=True)
    return np.where(tot > 0, p / np.maximum(tot, 1e-12), 0.0)
