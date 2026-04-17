"""
ev_gate.py
馬券種別 EV 計算とゲート判定。

Stage 1 の中核モジュール。

依存:
  - data/payout_table.parquet  (gen_payout_table.py で生成)

提供 API:
  - lookup_payout(馬券種, 人気パターン, 場所, 芝ダ, dist_b, field_b)
  - compute_ev_tansho(p_win, pop_hon, race_meta)
  - compute_ev_fuku(p_fuku, pop, race_meta)
  - compute_ev_umaren(p_hit, pop_a, pop_b, race_meta)
  - compute_ev_santan(p_hit, pop_1, pop_2, pop_3, race_meta)
  - compute_ev_sanren(p_hit, pop_a, pop_b, pop_c, race_meta)
  - pass_ev_gate(馬券種, ev_value)
"""
from __future__ import annotations
import functools
from pathlib import Path
import numpy as np
import pandas as pd

BASE        = Path(__file__).parent
PAYOUT_PARQUET = BASE / "data/payout_table.parquet"

# =====================================================
# 閾値（utils.py からも import 可だが循環回避のためここに重複定義）
# =====================================================
MIN_EV_TANSHO = 1.05
MIN_EV_FUKU   = 1.05
MIN_EV_UMAREN = 1.10
MIN_EV_UMATAN = 1.15
MIN_EV_SANREN = 1.05
MIN_EV_SANTAN = 0.95   # 三連単はレース平均で 0.95 を下限、Stage 1 EV ゲート用


# =====================================================
# テーブルロード
# =====================================================
@functools.lru_cache(maxsize=1)
def _load_table() -> pd.DataFrame:
    if not PAYOUT_PARQUET.exists():
        raise FileNotFoundError(
            f"{PAYOUT_PARQUET} が無い。先に `python gen_payout_table.py` を実行。"
        )
    return pd.read_parquet(PAYOUT_PARQUET)


def _pop_bucket(p) -> str:
    if pd.isna(p): return "?"
    p = int(p)
    if p == 1: return "1"
    if p == 2: return "2"
    if p == 3: return "3"
    if p <= 6: return "4-6"
    return "7+"


def _dist_bucket(d) -> str:
    if pd.isna(d): return "?"
    d = int(d)
    if d <= 1400: return "短"
    if d <= 1700: return "マイル"
    if d <= 2200: return "中"
    return "長"


def _field_bucket(n: int) -> str:
    if n <= 10: return "少"
    if n <= 14: return "中"
    return "多"


# =====================================================
# 中核: payout 検索（条件を緩めながらフォールバック）
# =====================================================
def lookup_payout(
    bet_type: str, pop_pattern: str,
    place: str | None = None, td: str | None = None,
    dist_b: str | None = None, field_b: str | None = None,
    metric: str = "median",
) -> float | None:
    """payout テーブルから条件にマッチする中央値を返す。
    詳細条件で hit しなければ、条件を順次緩める:
      4軸 → 3軸 (field 落とす) → 2軸 (dist 落とす) → 1軸 (場所落とす) → グローバル
    """
    df = _load_table()
    df = df[(df["馬券種"] == bet_type) & (df["人気パターン"] == pop_pattern)]
    if df.empty:
        return None

    # 段階的フォールバック
    constraints = [
        ("場所", place), ("芝ダ", td), ("dist_bucket", dist_b), ("field_bucket", field_b),
    ]
    for drop_n in range(0, len(constraints) + 1):
        active = constraints[: len(constraints) - drop_n]
        m = pd.Series(True, index=df.index)
        for col, val in active:
            if val is None:
                continue
            m &= (df[col] == val)
        sub = df[m]
        if not sub.empty:
            # sample_n 加重平均で中央値を取る
            w = sub["sample_n"]
            if w.sum() > 0:
                return float((sub[metric] * w).sum() / w.sum())
            return float(sub[metric].median())
    return None


# =====================================================
# レース文脈の作成 helper
# =====================================================
def make_race_meta(place: str, td: str, distance: int, field_size: int) -> dict:
    return {
        "place": place,
        "td":    td,
        "dist_b":  _dist_bucket(distance),
        "field_b": _field_bucket(field_size),
    }


# =====================================================
# EV 計算 (馬券種別)
# =====================================================
def compute_ev_tansho(p_win: float, pop_hon: int, meta: dict) -> tuple[float, float]:
    """単勝 EV = p_win × payout_estimate / 100. payout は 100 円賭け基準。
    Returns: (ev, payout_estimate)
    """
    pop_b = _pop_bucket(pop_hon)
    pay = lookup_payout("単勝", pop_b, meta["place"], meta["td"], meta["dist_b"], meta["field_b"])
    if pay is None:
        return 0.0, 0.0
    ev = p_win * (pay / 100.0)
    return ev, pay


def compute_ev_fuku(p_fuku: float, pop: int, meta: dict) -> tuple[float, float]:
    """複勝 EV = p_fuku × payout_estimate / 100."""
    pop_b = _pop_bucket(pop)
    pay = lookup_payout("複勝", pop_b, meta["place"], meta["td"], meta["dist_b"], meta["field_b"])
    if pay is None:
        return 0.0, 0.0
    ev = p_fuku * (pay / 100.0)
    return ev, pay


def compute_ev_umaren(p_hit: float, pop_a: int, pop_b: int, meta: dict) -> tuple[float, float]:
    """馬連 EV. pop_a, pop_b は順序無関係 (sort してパターン化)."""
    a, b = sorted([_pop_bucket(pop_a), _pop_bucket(pop_b)])
    pat = f"{a}-{b}"
    pay = lookup_payout("馬連", pat, meta["place"], meta["td"], meta["dist_b"], meta["field_b"])
    if pay is None:
        return 0.0, 0.0
    ev = p_hit * (pay / 100.0)
    return ev, pay


def compute_ev_umatan(p_hit: float, pop_1: int, pop_2: int, meta: dict) -> tuple[float, float]:
    """馬単 EV. pop_1=1着, pop_2=2着 順序付き."""
    pat = f"{_pop_bucket(pop_1)}-{_pop_bucket(pop_2)}"
    pay = lookup_payout("馬単", pat, meta["place"], meta["td"], meta["dist_b"], meta["field_b"])
    if pay is None:
        return 0.0, 0.0
    ev = p_hit * (pay / 100.0)
    return ev, pay


def compute_ev_sanren(p_hit: float, pop_1: int, pop_2: int, pop_3: int, meta: dict) -> tuple[float, float]:
    """三連複 EV. 順序無関係."""
    s = sorted([_pop_bucket(pop_1), _pop_bucket(pop_2), _pop_bucket(pop_3)])
    pat = "-".join(s)
    pay = lookup_payout("三連複", pat, meta["place"], meta["td"], meta["dist_b"], meta["field_b"])
    if pay is None:
        return 0.0, 0.0
    ev = p_hit * (pay / 100.0)
    return ev, pay


def compute_ev_santan(p_hit: float, pop_1: int, pop_2: int, pop_3: int, meta: dict) -> tuple[float, float]:
    """三連単 EV. 順序付き."""
    pat = f"{_pop_bucket(pop_1)}-{_pop_bucket(pop_2)}-{_pop_bucket(pop_3)}"
    pay = lookup_payout("三連単", pat, meta["place"], meta["td"], meta["dist_b"], meta["field_b"])
    if pay is None:
        return 0.0, 0.0
    ev = p_hit * (pay / 100.0)
    return ev, pay


# =====================================================
# ゲート判定
# =====================================================
_THRESHOLDS = {
    "単勝":   MIN_EV_TANSHO,
    "複勝":   MIN_EV_FUKU,
    "馬連":   MIN_EV_UMAREN,
    "馬単":   MIN_EV_UMATAN,
    "三連複": MIN_EV_SANREN,
    "三連単": MIN_EV_SANTAN,
}


def pass_ev_gate(bet_type: str, ev_value: float, threshold_override: float | None = None) -> bool:
    th = threshold_override if threshold_override is not None else _THRESHOLDS.get(bet_type, 1.0)
    return ev_value >= th


# =====================================================
# 自己テスト
# =====================================================
if __name__ == "__main__":
    import io, sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    print("=== ev_gate.py 動作確認 ===\n")
    meta = make_race_meta(place="中山", td="芝", distance=1600, field_size=16)
    print(f"レース: {meta}\n")

    # 単勝: ◎(1人気) p_win=0.35
    ev, pay = compute_ev_tansho(0.35, 1, meta)
    print(f"単勝 ◎(1人気) p=0.35 → 推定配当={pay:.0f} EV={ev:.3f} gate={'✓' if pass_ev_gate('単勝', ev) else '✗'}")

    # 複勝: ◎(1人気) p_fuku=0.65
    ev, pay = compute_ev_fuku(0.65, 1, meta)
    print(f"複勝 ◎(1人気) p=0.65 → 推定配当={pay:.0f} EV={ev:.3f} gate={'✓' if pass_ev_gate('複勝', ev) else '✗'}")

    # 馬連: ◎(1)-◯(3) p_hit=0.18
    ev, pay = compute_ev_umaren(0.18, 1, 3, meta)
    print(f"馬連 1-3 p=0.18 → 推定配当={pay:.0f} EV={ev:.3f} gate={'✓' if pass_ev_gate('馬連', ev) else '✗'}")

    # 三連単: 1-2-3 p_hit=0.04
    ev, pay = compute_ev_santan(0.04, 1, 2, 3, meta)
    print(f"三連単 1-2-3 p=0.04 → 推定配当={pay:.0f} EV={ev:.3f} gate={'✓' if pass_ev_gate('三連単', ev) else '✗'}")

    # 三連単穴目: 1-3-7 p_hit=0.005
    ev, pay = compute_ev_santan(0.005, 1, 3, 7, meta)
    print(f"三連単 1-3-7 p=0.005 → 推定配当={pay:.0f} EV={ev:.3f} gate={'✓' if pass_ev_gate('三連単', ev) else '✗'}")
