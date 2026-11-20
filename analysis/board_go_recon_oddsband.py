"""board_go_recon_backtest の追加: 判定 A を満たした/90% を超えた 10 集合を ◎の単勝オッズ帯で分ける。

探索・再構成（前向き記録とは別）。結果を見た後の分割。
- オッズ = 凍結版 bundle の ◎ tansho_odds（export 時点＝発走前。確定オッズではない）。
- 帯 = data/shadow_policies/a95_equal_v1.json の bands（2026-09-30 凍結、今回の結果を見る前の区切り）。
  <2倍 は 0.1 刻みなので ≤1.9 倍＝sniper の参戦域と同じ。
実行: python -m analysis.board_go_recon_oddsband --kekka-dir E:/PyCaLiAI/data/kekka
"""
import argparse
import json
from pathlib import Path

from analysis.board_go_recon_backtest import BASE, collect, summ

SETS = [  # (名前, 条件, 券種)
    ("GO∧一致2∧固い", lambda r: r["go"] and r["agree"] == 2 and r["hard"] == "固い", ("馬連", "三連複", "ワイド")),
    ("GO∧一致3∧標準", lambda r: r["go"] and r["agree"] == 3 and r["hard"] == "標準", ("馬連", "複勝")),
    ("GO∧一致2以上∧固い", lambda r: r["go"] and r["agree"] >= 2 and r["hard"] == "固い", ("馬連",)),
    ("一致3（GO不問）", lambda r: r["agree"] == 3, ("単勝",)),
    ("GO∧一致3", lambda r: r["go"] and r["agree"] == 3, ("単勝",)),
    ("GO∧PWだけ一致", lambda r: r["go"] and r["gset"] == "PW", ("ワイド",)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kekka-dir", default=str(BASE / "data" / "kekka"))
    ap.add_argument("--out", default=str(BASE / "reports" / "board_go_recon" / "oddsband.json"))
    a = ap.parse_args()
    bands = json.loads((BASE / "data" / "shadow_policies" / "a95_equal_v1.json").read_text(encoding="utf-8"))["bands"]
    rows, _, _ = collect(Path(a.kekka_dir))
    no_odds = sum(1 for r in rows if not r["hon_odds"])

    def band(o):
        for lo, hi, lb in bands:
            if o is not None and o >= lo and (hi is None or o < hi):
                return lb
        return "オッズなし"

    out = {"label": "探索・再構成（結果を見た後の分割）", "bands": [b[2] for b in bands], "no_odds": no_odds, "sets": []}
    for name, f, types in SETS:
        sub = [r for r in rows if f(r)]
        for t in types:
            e = {"name": name, "type": t, "all": summ(sub, t), "by_band": {}}
            for lb in [b[2] for b in bands] + ["オッズなし"]:
                bs = [r for r in sub if band(r["hon_odds"]) == lb]
                if bs:
                    e["by_band"][lb] = summ(bs, t)
            out["sets"].append(e)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for e in out["sets"]:
        print(f"\n{e['name']} {e['type']}  全体 n={e['all']['n']} {e['all']['roi']}")
        for lb, s in e["by_band"].items():
            print(f"  {lb:6s} n={s['n']:4d} roi={s['roi']:6.1f} ci={s['ci']} drop={s['roi_drop_max']} hit={s['race_hit']}")
    print("odds missing:", no_odds)


if __name__ == "__main__":
    main()
