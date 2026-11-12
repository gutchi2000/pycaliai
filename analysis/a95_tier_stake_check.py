# -*- coding: utf-8 -*-
"""
a95_tier_stake_check.py — 「見送り強化 + 勝負レースは厚く・それ以外は薄く」が a95 で効くかの検算。

問い:
  Q1 見送りを強めると（混戦度 percentile 0.667 → 0.50 → 0.333）参戦側の回収率は上がるか。
  Q2 参戦レースを自信度で 3 段に分けたとき、段ごとの回収率は単調か（勝負 > 通常 > 薄）。
  Q3 段ごとに賭け金を変える（例 20,000 / 10,000 / 3,000 円）と、均等 10,000 円より上がるか。

回収率は 1 レースごとの回収率 r_i の賭け金加重平均なので、Q3 が正になるのは Q2 が単調なときだけ。
先に Q2 を見る。

自信度の軸（どちらも発走前に確定している量。結果・払戻は使わない）:
  chaos : 較正 p_win の正規化エントロピー（本番 field_chaos_score と同じ定義、低いほど勝負）
  p1    : AI1位の p_win（高いほど勝負）

注意:
  - 凍結表 a95_equal_v1 は同じ 2023-25 OOS で選んだ表なので、回収率の「水準」は in-sample で甘い。
    見るのは水準でなく「段の間の差」と、その年別の符号の一致。
  - 段の境界は評価年を含まない年で決める版（--oos-cut）も出す。全期間で切った版は参考。
  - 読み取りのみ。方策・台帳・本番設定には触れない。

実行（root から）:
  python -m analysis.a95_tier_stake_check                    # 実データ data/_policy/bet_substrate.pkl
  python -m analysis.a95_tier_stake_check --mock             # モック（自信度と回収率が無関係な世界）
  python -m analysis.a95_tier_stake_check --mock --mock-edge 0.15   # モック（勝負ほど高回収な世界）
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import shadow_a95 as m  # noqa: E402  (読み取りのみ: load_policy / band_of / equal_stakes / sellable_kinds)

logger = logging.getLogger("a95_tier")
SUBSTRATE = BASE / "data" / "_policy" / "bet_substrate.pkl"
COLS = ["tan", "fuku", "umaren", "umatan", "wide", "wakuren", "sanpuku", "sanrentan"]
CAP, UNIT = 10000, 100


def _pl(p: np.ndarray, i: int, j: int, k: int | None = None) -> float:
    """Plackett-Luce で i→j（→k）の順に入る確率。"""
    v = p[i] * p[j] / (1 - p[i])
    return v if k is None else v * p[k] / (1 - p[i] - p[j])


def mock_substrate(n_days: int, edge: float, seed: int) -> list[dict]:
    """検算用のモック substrate を作る。

    全券種の払戻を「厳密な的中確率の逆数 × k」で作るので、edge=0 ならどの券も期待回収率 80%
    （自信度と無関係な帰無の世界）。edge>0 なら低混戦レースほど k が大きい。

    Args:
        n_days: 開催日数（1 日 24 レース）。
        edge: 自信度と回収率の結びつき。
        seed: 乱数種。

    Returns:
        bet_substrate.pkl と同じキーを持つレース dict のリスト。
    """
    rng = np.random.default_rng(seed)
    days = pd.date_range("2023-01-07", periods=n_days, freq="3D").strftime("%Y-%m-%d")
    out: list[dict] = []
    for d in days:
        for _ in range(24):
            n = int(rng.integers(8, 15))
            s = rng.normal(0, rng.uniform(0.4, 1.6), n)
            p = np.exp(s) / np.exp(s).sum()
            h = float(-(p * np.log(p)).sum() / np.log(n))
            k = 0.80 * (1.0 + edge * (0.85 - h) / 0.15)
            ban = np.arange(1, n + 1)
            waku = np.array([m.waku_of(int(b), n) for b in ban])
            a1, a2, a3 = (int(x) for x in np.argsort(-p)[:3])  # 買い目になる AI 上位 3 頭
            idx = range(n)
            # 着順（真の確率 p の Plackett-Luce）
            rest, order = list(idx), []
            for _ in range(3):
                q = p[rest] / p[rest].sum()
                order.append(rest.pop(int(rng.choice(len(rest), p=q))))
            i, j, l = order
            # 買い目側の厳密な的中確率
            p_ex, p_q = _pl(p, a1, a2), _pl(p, a1, a2) + _pl(p, a2, a1)
            p_t3 = _pl(p, a1, a2, a3)
            p_box = sum(_pl(p, x, y, z) for x, y, z in
                        ((a1, a2, a3), (a1, a3, a2), (a2, a1, a3), (a2, a3, a1), (a3, a1, a2), (a3, a2, a1)))
            p_top3 = p[a1] + sum(_pl(p, x, a1) for x in idx if x != a1) + sum(
                _pl(p, x, y, a1) for x in idx for y in idx if len({x, y, a1}) == 3)
            p_wide = sum(_pl(p, x, y, z) for x in idx for y in idx for z in idx
                         if len({x, y, z}) == 3 and {a1, a2} <= {x, y, z})
            wa, wb = int(waku[a1]), int(waku[a2])
            p_waku = sum(_pl(p, x, y) for x in idx for y in idx
                         if x != y and {int(waku[x]), int(waku[y])} == {wa, wb})
            out.append(dict(
                date=d, p=p, ban=ban, waku=waku, odds9=k / p * rng.lognormal(0, 0.1, n),
                first=int(ban[i]), second=int(ban[j]), third=int(ban[l]),
                waku_first=int(waku[i]), waku_second=int(waku[j]),
                tan_pay=100 * k / p[a1], fuku_pay={int(ban[x]): 100 * k / p_top3 for x in order},
                umaren=100 * k / p_q, umatan=100 * k / p_ex, wakuren=100 * k / max(p_waku, 1e-9),
                wide=[(int(ban[x]), int(ban[y]), 100 * k / p_wide) for x, y in ((i, j), (i, l), (j, l))],
                sanpuku=100 * k / p_box, sanrentan=100 * k / p_t3,
            ))
    return out


def build_rows(sub: list[dict], pol: dict) -> pd.DataFrame:
    """substrate から 1 レース 1 行の表を作る（a95 を均等 1 万円で買ったときの払戻つき）。

    行の作り方は analysis/a95_chaos_gate_check.py と同じ。枠連は 9 頭以上のみ（sellable_kinds）。
    """
    rows = []
    for r in sub:
        p, ban, o = np.asarray(r["p"], float), r["ban"], np.asarray(r["odds9"], float)
        n = len(p)
        if n < 5 or not np.isfinite(o).any():
            continue
        order = np.argsort(-p)
        i1, i2, i3 = order[:3]
        o1 = float(o[i1]) if np.isfinite(o[i1]) and o[i1] > 0 else np.nan
        if not np.isfinite(o1):
            continue
        pp = p / p.sum()
        h = float(-(pp[pp > 0] * np.log(pp[pp > 0])).sum() / np.log(n))
        a1, a2, a3 = int(ban[i1]), int(ban[i2]), int(ban[i3])
        f, s, t = r["first"], r["second"], r["third"]
        w1, w2 = int(r["waku"][i1]), int(r["waku"][i2])
        wide = {frozenset((x, y)): v for x, y, v in (r.get("wide") or ())}
        per = dict(
            tan=(r["tan_pay"] if a1 == f else 0.0), fuku=r["fuku_pay"].get(a1, 0.0),
            umaren=(r["umaren"] if {a1, a2} == {f, s} else 0.0),
            umatan=(r["umatan"] if (a1, a2) == (f, s) else 0.0),
            wide=float(wide.get(frozenset((a1, a2)), 0.0)),
            wakuren=(r["wakuren"] if {w1, w2} == {int(r["waku_first"]), int(r["waku_second"])} else 0.0),
            sanpuku=(r["sanpuku"] if {a1, a2, a3} == {f, s, t} else 0.0),
            sanrentan=(r["sanrentan"] if (a1, a2, a3) == (f, s, t) else 0.0),
        )
        per = {k: (float(v) if np.isfinite(v) else 0.0) for k, v in per.items()}
        band = m.band_of(o1, pol)
        kinds = m.sellable_kinds(list(pol["plan"].get(band) or []), n)
        if not kinds:
            continue
        st = m.equal_stakes(kinds, CAP, UNIT)
        ret = sum(per[k] / 100.0 * v for k, v in st.items())
        rows.append(dict(date=r["date"], year=str(r["date"])[:4], n=n, H=h, p1=float(p[i1]),
                         band=band, rr=ret / sum(st.values()), hit=ret > 0))
    return pd.DataFrame(rows)


def boot(df: pd.DataFrame, w: np.ndarray, w0: np.ndarray | None, reps: int, seed: int) -> tuple[float, float, float]:
    """開催日ブロック bootstrap。w0 が無ければ回収率、あれば回収率の差（w − w0）を返す。

    Returns:
        (下側 2.5%, 上側 97.5%, 正である割合。回収率なら 100 超の割合)
    """
    uniq, inv = np.unique(df["date"].values, return_inverse=True)
    rr = df["rr"].values
    num, den = np.bincount(inv, w * rr), np.bincount(inv, w)
    if w0 is not None:
        num0, den0 = np.bincount(inv, w0 * rr), np.bincount(inv, w0)
    rng = np.random.default_rng(seed)
    out = np.empty(reps)
    for k in range(reps):
        s = rng.integers(0, len(uniq), len(uniq))
        v = 100.0 * num[s].sum() / max(den[s].sum(), 1e-9)
        if w0 is None:
            out[k] = v
        else:
            out[k] = v - 100.0 * num0[s].sum() / max(den0[s].sum(), 1e-9)
    ref = 100.0 if w0 is None else 0.0
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), float(np.mean(out > ref))


def roi(df: pd.DataFrame, w: np.ndarray) -> float:
    """賭け金 w での回収率（%）。"""
    return 100.0 * float((w * df["rr"].values).sum()) / max(float(w.sum()), 1e-9)


def line(name: str, df: pd.DataFrame, w: np.ndarray, reps: int, seed: int) -> None:
    """1 行ぶんの回収率を出す。"""
    on = w > 0
    if on.sum() == 0:
        logger.info("  %-34s (買い目なし)", name)
        return
    lo, hi, p = boot(df, w, None, reps, seed)
    ys = " / ".join(f"{y}:{roi(g, w[(df.year == y).values]):.0f}" for y, g in df.groupby("year"))
    logger.info("  %-34s R=%5d ROI %6.1f%% [%5.1f,%5.1f] P>100=%.2f R的中 %4.1f%% | %s",
                name, int(on.sum()), roi(df, w), lo, hi, p, 100 * df["hit"].values[on].mean(), ys)


def tier_of(score: np.ndarray, cuts: tuple[float, float]) -> np.ndarray:
    """自信度スコア（高いほど勝負）を 0=勝負 / 1=通常 / 2=薄 に分ける。"""
    lo, hi = cuts
    return np.where(score >= hi, 0, np.where(score >= lo, 1, 2))


def run(df: pd.DataFrame, stakes: tuple[int, int, int], reps: int, seed: int) -> None:
    """Q1〜Q3 を順に出力する。"""
    hq = df.H.rank(pct=True).values
    base_ok = ((df.n > 7) & (df.p1 >= 0.05)).values
    logger.info("母集団 %dR（a95 の買い目が立つレース）, %d 開催日", len(df), df.date.nunique())

    logger.info("\n■ Q1 見送りの強さ（混戦度 percentile の上限）× 均等 1 万円")
    for q in (1.0, 0.667, 0.50, 0.333):
        keep = base_ok & (hq < q) if q < 1 else np.ones(len(df), bool)
        line(f"混戦度 pct < {q:.3f}" if q < 1 else "ガードなし（全レース）", df, keep.astype(float), reps, seed)
    keep = base_ok & (hq < 0.667)
    kd = df[keep].reset_index(drop=True)
    w_keep, w_tight = np.ones(len(kd)), (hq[keep] < 0.50).astype(float)
    lo, hi, p = boot(kd, w_tight, w_keep, reps, seed)
    logger.info("  差: 0.50 − 0.667 = %+.1fpt [%+.1f,%+.1f] P(>0)=%.2f",
                roi(kd, w_tight) - roi(kd, w_keep), lo, hi, p)

    for axis, score in (("chaos（低混戦ほど勝負）", -kd.H.values), ("p1（AI1位の勝率が高いほど勝負）", kd.p1.values)):
        logger.info("\n■ 自信度の軸 = %s  ※現行ガード通過 %dR の中で 3 等分", axis, len(kd))
        t = tier_of(score, tuple(np.quantile(score, [1 / 3, 2 / 3])))
        logger.info(" Q2 段ごとの回収率（均等 1 万円、全期間で 3 等分）")
        for i, nm in enumerate(("勝負", "通常", "薄")):
            line(nm, kd, (t == i).astype(float), reps, seed)
        w = np.asarray(stakes, float)[t]
        lo, hi, p = boot(kd, w, np.ones(len(kd)), reps, seed)
        logger.info(" Q3 傾斜 %s 円 vs 均等: %.1f%% vs %.1f%% 差 %+.1fpt [%+.1f,%+.1f] P(>0)=%.2f",
                    "/".join(f"{s:,}" for s in stakes), roi(kd, w), roi(kd, np.ones(len(kd))),
                    roi(kd, w) - roi(kd, np.ones(len(kd))), lo, hi, p)
        logger.info(" Q3' 境界を他の年で決めて評価年に当てる（年別の符号を見る）")
        for y in sorted(kd.year.unique()):
            tr, ev = (kd.year != y).values, (kd.year == y).values
            if tr.sum() < 100 or ev.sum() < 100:
                continue
            te = tier_of(score[ev], tuple(np.quantile(score[tr], [1 / 3, 2 / 3])))
            ke = kd[ev].reset_index(drop=True)
            we = np.asarray(stakes, float)[te]
            lo, hi, p = boot(ke, we, np.ones(len(ke)), reps, seed)
            per = " ".join(f"{nm}{roi(ke, (te == i).astype(float)):.0f}" for i, nm in enumerate(("勝負", "通常", "薄")))
            logger.info("   %s: 差 %+.1fpt [%+.1f,%+.1f] P(>0)=%.2f | 段別 %s", y,
                        roi(ke, we) - roi(ke, np.ones(len(ke))), lo, hi, p, per)


def main() -> int:
    """CLI。"""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true", help="モック substrate で動かす")
    ap.add_argument("--mock-edge", type=float, default=0.0, help="モックでの自信度と回収率の結びつき")
    ap.add_argument("--mock-days", type=int, default=300)
    ap.add_argument("--stakes", default="20000,10000,3000", help="勝負,通常,薄 の 1R 金額")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.debug else logging.INFO, format="%(message)s")
    try:
        stakes = tuple(int(x) for x in a.stakes.split(","))
        if len(stakes) != 3 or min(stakes) < 0:
            raise ValueError
    except ValueError:
        logger.error("--stakes は 0 以上の整数を 3 つ（勝負,通常,薄）: %s", a.stakes)
        return 2
    pol = m.load_policy()
    if a.mock:
        logger.info("[mock] edge=%.2f days=%d（実データではない）", a.mock_edge, a.mock_days)
        sub = mock_substrate(a.mock_days, a.mock_edge, a.seed)
    else:
        if not SUBSTRATE.exists():
            logger.error("substrate が無い: %s（ローカル専用の生成物。--mock で動作確認は可能）", SUBSTRATE)
            return 1
        import joblib
        sub = joblib.load(SUBSTRATE)
    df = build_rows(sub, pol)
    if df.empty:
        logger.error("有効なレースが 0 件")
        return 1
    run(df, stakes, a.reps, a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
