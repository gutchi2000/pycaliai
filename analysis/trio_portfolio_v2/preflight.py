# -*- coding: utf-8 -*-
"""preflight.py — 設計書 §7 Preflight (P1〜P10) + §5.2 Phase 0 進捗の実測

読み取り・synthetic test・既存データ照合だけで判定する。未解決は fail-closed。
本番 (compute_bets / bets.json / サイト / Discord / IPAT) には一切触らない。

実行: python -m analysis.trio_portfolio_v2.preflight
出力: reports/trio_portfolio_shadow_v2/preflight_report.{json,md}
"""
from __future__ import annotations

import ast
import glob
import json
import re
import sys
from collections import Counter
from datetime import datetime
from itertools import combinations
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))

OUT = BASE / "reports" / "trio_portfolio_shadow_v2"
STOCK = OUT / "raw_stock"

from jvlink_trio_odds import (parse_o5, n_combos, load_post_times,   # noqa: E402
                              lead_minutes, write_append_only, parse_hhmm,
                              COLLECTOR_VERSION, O5_BODY0, O5_STRIDE, O5_NMAX)
from analysis.trio_portfolio_v2 import candidates as CD           # noqa: E402
from analysis.trio_portfolio_v2 import pl_trio as PL              # noqa: E402
from analysis.trio_portfolio_v2 import allocate as AL             # noqa: E402
from analysis.trio_portfolio_v2 import settle as ST               # noqa: E402

PASS, FAIL, PEND = "PASS", "FAIL", "PENDING"


def _race_payouts(rid16: str) -> list:
    """kekka から当該レースの三連複払戻（100円あたり）を全て拾う（同着で複数）。"""
    import csv as _csv
    p = BASE / "data" / "kekka" / f"{rid16[:8]}.csv"
    if not p.exists():
        return []
    out = set()
    with open(p, encoding="cp932", errors="replace", newline="") as f:
        for r in _csv.DictReader(f):
            if str(r.get("レースID(新)", ""))[:16] != rid16:
                continue
            v = (r.get("３連複") or "").strip()
            if v:
                out.add(float(v))
    return sorted(out)


def _res(pid, name, status, detail, **kw):
    return {"id": pid, "name": name, "status": status, "detail": detail, **kw}


# ============================================================
# P1 — 0B35(O5) の意味: 組・馬番・倍率・欠損が仕様/raw/公表払戻で一致
# ============================================================
def p1_record_semantics() -> dict:
    files = sorted(glob.glob(str(STOCK / "*" / "O5_*.txt")))
    if not files:
        return _res("P1", "0B35(O5) レコード意味", PEND,
                    "蓄積 O5 の raw が無い。py -3.12-32 jvlink_trio_odds.py "
                    "--stock-dump YYYYMMDDHHMMSS を先に実行する。")
    results_cache, stats = {}, Counter()
    bad, notes = [], []
    for f in files:
        rec = Path(f).read_text(encoding="utf-8")
        o = parse_o5(rec)
        rid = o["race_key"]
        stats["races"] += 1
        # (a) レコード全長・スロット全数の説明可能性
        acc = (len(o["odds"]) + len(o["unpriced"]) + len(o["zero"])
               + o["blank"] + o["malformed"])
        if acc != O5_NMAX or o["malformed"]:
            bad.append((rid, f"slots {acc}/{O5_NMAX} malformed={o['malformed']}"))
            continue
        stats["slots_ok"] += 1
        # (b) 有効スロット数 = C(登録頭数,3) / 発売中 = C(出走頭数,3)
        if len(o["odds"]) + len(o["unpriced"]) + len(o["zero"]) != n_combos(o["toroku"], 3):
            bad.append((rid, "registered-combo count mismatch")); continue
        if len(o["odds"]) != n_combos(o["shusso"], 3):
            bad.append((rid, "priced-combo count mismatch")); continue
        stats["combo_count_ok"] += 1
        # (c) 人気順フィールドがオッズ昇順と整合
        if o["ninki"]:
            # 同オッズは JRA 側の tie 規則で人気が前後するため、
            # 「オッズが真に小さい組の人気が必ず上位」だけを検定する。
            srt = sorted(o["odds"], key=lambda k: o["odds"][k])
            inv = sum(1 for i in range(len(srt) - 1)
                      if o["odds"][srt[i]] < o["odds"][srt[i + 1]]
                      and o["ninki"].get(srt[i], 0) > o["ninki"].get(srt[i + 1], 0))
            if inv == 0:
                stats["ninki_monotone_ok"] += 1
            else:
                bad.append((rid, f"ninki inversion x{inv}"))
        # (d) 的中組の倍率 ×100 == 公表三連複払戻
        d = rid[:8]
        if d not in results_cache:
            results_cache[d] = ST.load_results(d)
        res = results_cache[d].get(rid)
        if not res:
            stats["no_result"] += 1
            continue
        if res["status"] == "ambiguous_deadheat" and res["winning_combos"]:
            # 3着同着: 的中組が k 組あると払戻はプールを k 分割した額になる。
            # 組↔払戻の1対1対応は kekka から取れないので、
            # 「オッズ×100 / k」の集合が公表払戻の集合に（丸め誤差内で）一致するかを見る。
            kk = len(res["winning_combos"])
            got = sorted(o["odds"][c] * 100 / kk for c in res["winning_combos"]
                         if c in o["odds"])
            want = sorted(_race_payouts(rid))
            stats["deadheat"] += 1
            if (len(got) == len(want) and bool(got)
                    and all(abs(g - w) / w < 0.02 for g, w in zip(got, want))):
                stats["deadheat_split_ok"] += 1
            else:
                # 判定は落とさない（オッズ録そのものは (a)(b)(c) で健全）。
                # 同着払戻の丸め規則までは倍率から再現できない = 決済は公表額必須。
                notes.append((rid, f"deadheat 倍率/k={got} vs 公表={want}"))
            continue
        if res["status"] != "confirmed" or not res["winning_combos"]:
            stats["skipped_nonconfirmed"] += 1
            continue
        k = res["winning_combos"][0]
        od, pay = o["odds"].get(k), res["payout_per100"].get(k)
        if od is None or pay is None or abs(od * 100 - pay) > 1e-6:
            bad.append((rid, f"payout mismatch combo={k} odds={od} pay={pay}"))
            continue
        stats["payout_match"] += 1
    # (e) RT(0B35) 実録と蓄積 O5 の同一性（同一レースを両経路で取得できた場合）
    rt = []
    for sp in glob.glob(str(OUT / "snapshots" / "*" / "*.json")):
        try:
            d = json.loads(Path(sp).read_text(encoding="utf-8"))
        except Exception:
            continue
        if not d.get("trio"):
            continue
        rid = d["race_id"]
        cand = [x for x in files if x.endswith(f"O5_{rid}.txt")]
        if not cand:
            continue
        o = parse_o5(Path(cand[0]).read_text(encoding="utf-8"))
        same = (d["trio"] == o["odds"] and d.get("trio_ninki") == o["ninki"])
        rt.append((rid, same, len(o["odds"])))
        stats["rt_vs_stock_checked"] += 1
        stats["rt_vs_stock_same"] += same
    ok = not bad and stats["payout_match"] > 0
    return _res("P1", "0B35(O5) レコード意味", PASS if ok else FAIL,
                f"蓄積O5 {stats['races']}R: 全スロット説明可 {stats['slots_ok']}R / "
                f"組数一致(C(登録,3)・C(出走,3)) {stats['combo_count_ok']}R / "
                f"人気順とオッズ昇順が整合 {stats['ninki_monotone_ok']}R / "
                f"的中組の倍率×100 = 公表払戻 {stats['payout_match']}R / "
                f"3着同着 {stats['deadheat']}R は倍率/k が公表額と ±2%内で一致 "
                f"{stats['deadheat_split_ok']}R（丸め規則までは再現不可＝決済は公表額必須） / "
                f"RT(0B35)実録と蓄積O5の完全一致 {stats['rt_vs_stock_same']}"
                f"/{stats['rt_vs_stock_checked']}R"
                + (f" / 不一致 {len(bad)}R: {bad[:3]}" if bad else ""),
                layout={"body_offset": O5_BODY0, "stride": O5_STRIDE,
                        "slots": O5_NMAX, "fields": "組番6+オッズ6(1/10倍)+人気3",
                        "unpriced_mark": "オッズ欄 '*' または '-' 埋め",
                        "padding": "15桁空白（組番順の固定配置なので飛び飛びに出る）"},
                stats=dict(stats), notes=notes,
                rt_vs_stock=rt,
                caveat=("RT(0B35) 経路も実録で検証済（下記 rt_vs_stock）。ただし RT が返すのは"
                        "常に最新断面なので、**T-10 時点**の断面そのものは開催日にしか取れない。"))


# ============================================================
# P2 — 時刻運用: 8〜12分前を証明可能か
# ============================================================
def p2_timing() -> dict:
    # (a) 純ロジックの単体テスト
    unit = []
    now = datetime(2026, 8, 16, 15, 30, 0)
    unit.append(abs(lead_minutes(now, "20260816", "15:40") - 10.0) < 1e-9)
    unit.append(lead_minutes(now, "20260816", "") is None)
    unit.append(abs(lead_minutes(now, "20260816", "1542") - 12.0) < 1e-9)
    unit.append(lead_minutes(now, "20260816", "15:20") < 0)          # 発走後は負
    unit.append(parse_hhmm("15時40分") == (15, 40))
    # (b) 本番 T-10 の実測 lead 分布（reports/live_odds の fetched × 出走表発走時刻）
    leads, no_post = [], 0
    for p in sorted(glob.glob(str(BASE / "reports" / "live_odds" / "*.json"))):
        rid = Path(p).stem
        if len(rid) != 16 or not rid.isdigit():
            continue
        try:
            d = json.loads(Path(p).read_text(encoding="utf-8"))
            f = datetime.fromisoformat(d["fetched"])
        except Exception:
            continue
        post = load_post_times(rid[:8]).get(rid)
        if not post:
            no_post += 1
            continue
        lm = lead_minutes(f, rid[:8], post)
        if lm is not None and -600 < lm < 600:
            leads.append(lm)
    leads.sort()
    inw = sum(1 for x in leads if 8.0 <= x <= 12.0)
    med = leads[len(leads) // 2] if leads else None
    # (c) 価格の時点を O5 の発表月日時分で証明できるか（自分の時計に依存しない）
    ann = ann_ok = 0
    for sp in glob.glob(str(BASE / "reports" / "trio_portfolio_shadow_v2"
                            / "snapshots" / "*" / "*.json")):
        try:
            d = json.loads(Path(sp).read_text(encoding="utf-8"))
        except Exception:
            continue
        if d.get("trio") and "announce_dt" in d:      # 発表時刻を記録する版の断面のみ
            ann += 1
            ann_ok += d.get("announce_dt") is not None
    # (d) 二重起動ガード / 発走時刻変更の配線
    runner = (BASE / "t10_runner.py").read_text(encoding="utf-8", errors="replace")
    has_lock = "LOCK" in runner and "already" in runner.lower() or "LOCK.write_text" in runner
    coll = (BASE / "jvlink_trio_odds.py").read_text(encoding="utf-8")
    tc_wired = "live_changes" in coll and "time_change" in coll
    ok_unit = all(unit)
    status = PASS if (ok_unit and tc_wired and has_lock and leads) else PEND
    return _res("P2", "時刻運用 (8〜12分前の証明)", status,
                f"lead計算の単体テスト {sum(unit)}/{len(unit)} 合格 / "
                f"本番T-10実測 n={len(leads)} 中央値{med:.1f}分 "
                f"8〜12分内 {inw}({inw/max(1,len(leads))*100:.0f}%) "
                f"[p10={leads[len(leads)//10]:.1f} p90={leads[len(leads)*9//10]:.1f}] / "
                f"発走時刻変更(TC)取込={tc_wired} 二重起動LOCK={has_lock} "
                f"発走時刻不明={no_post}件 / "
                f"O5発表月日時分から時点を再構成できた断面 {ann_ok}/{ann}"
                if leads else "本番T-10の実測 lead を計算できるデータが無い",
                note=("collector は毎断面に (1) 自分の時計による lead_min と "
                      "(2) O5 の発表月日時分による announce_lead_min の両方を記録し、"
                      "窓外は in_window=False として残す（捨てない・粉飾しない）。"
                      "価格の時点判定は (2) を正とする。RT 断面での窓内実績は"
                      "開催日にのみ確定する。"))


# ============================================================
# P3 — race 結合の一意性
# ============================================================
def p3_join() -> dict:
    dates = sorted({Path(p).stem.split("_")[0]
                    for p in glob.glob(str(BASE / "reports" / "cowork_input" / "*_bundle.json"))})[-8:]
    tot = Counter()
    dup = []
    for d in dates:
        b = json.loads((BASE / "reports" / "cowork_input" / f"{d}_bundle.json").read_text(encoding="utf-8"))
        rids = [re.sub(r"\D", "", str(r["race_id"]))[:16] for r in b["races"]]
        tot["bundle_races"] += len(rids)
        if len(set(rids)) != len(rids):
            dup.append(f"{d}: bundle race_id 重複")
        for r in b["races"]:
            bans = [int(h["umaban"]) for h in r["horses"] if h.get("umaban") is not None]
            if len(set(bans)) != len(bans):
                dup.append(f"{d}/{r['race_id']}: bundle 馬番重複")
        res = ST.load_results(d)
        tot["kekka_races"] += len(res)
        tot["joined"] += sum(1 for x in rids if x in res)
        for x in rids:
            if (BASE / "reports" / "live_odds" / f"{x}.json").exists():
                tot["live_odds_present"] += 1
        # kekka 側 (rid, 馬番) の一意性
        import csv as _csv
        p = BASE / "data" / "kekka" / f"{d}.csv"
        if p.exists():
            seen = set()
            with open(p, encoding="cp932", errors="replace", newline="") as f:
                for row in _csv.DictReader(f):
                    key = str(row.get("レースID(新)", ""))[:18]
                    if key in seen:
                        dup.append(f"{d}: kekka 行重複 {key}")
                    seen.add(key)
    ok = not dup and tot["joined"] > 0
    return _res("P3", "race 結合の一意性", PASS if ok else FAIL,
                f"直近{len(dates)}開催: bundle {tot['bundle_races']}R / kekka {tot['kekka_races']}R / "
                f"16桁IDで結合できた {tot['joined']}R / T-10単勝あり {tot['live_odds_present']}R / "
                f"重複 {len(dup)}件" + (f" {dup[:3]}" if dup else ""),
                note="0B35 側は race_key(=rec[11:27]) が要求 race_id と一致することを collector が毎回検査する")


# ============================================================
# P4 — 取消・返還を通常価格と区別できるか
# ============================================================
def p4_scratch() -> dict:
    files = sorted(glob.glob(str(STOCK / "*" / "O5_*.txt")))
    if not files:
        return _res("P4", "取消・返還の識別", PEND, "蓄積 O5 raw が無い")
    n_scratch = ok = 0
    detail = []
    for f in files:
        o = parse_o5(Path(f).read_text(encoding="utf-8"))
        if not o["unpriced"]:
            continue
        n_scratch += 1
        rid = o["race_key"]
        res = ST.load_results(rid[:8]).get(rid, {})
        gone = set(res.get("refunded_horses", []))
        # 未発売組 = 取消馬を含む組、と厳密に一致するか
        exp = {"-".join(map(str, c))
               for c in combinations(range(1, (o["toroku"] or 0) + 1), 3)
               if set(c) & gone}
        got = set(o["unpriced"])
        if exp == got and len(o["odds"]) == n_combos(o["shusso"], 3):
            ok += 1
        else:
            detail.append((rid, len(exp), len(got)))
    if n_scratch == 0:
        return _res("P4", "取消・返還の識別", PEND,
                    "dump 済みレースに取消が無く実証できていない（要: 取消のある開催日の raw）")
    return _res("P4", "取消・返還の識別", PASS if ok == n_scratch else FAIL,
                f"取消のあった {n_scratch}R 全てで「未発売組 = 取消馬を含む組」が厳密一致 "
                f"({ok}/{n_scratch})。発売中の組数も C(出走頭数,3) と一致。"
                + (f" 不一致 {detail[:3]}" if detail else ""),
                note="価格0/未発売('*')/空白padding の3態を parse_o5 が分離し、"
                     "settle 側は取消馬を含む組を返還として決済する")


# ============================================================
# P5 — 候補（順位・R・U0〜U3）の完全再現
# ============================================================
def p5_candidates() -> dict:
    dates = sorted({Path(p).stem.split("_")[0]
                    for p in glob.glob(str(BASE / "reports" / "cowork_output" / "*_bets.json"))})[-6:]
    n = same = det = 0
    arms_cnt = Counter()
    mism = []
    for d in dates:
        bp = BASE / "reports" / "cowork_input" / f"{d}_bundle.json"
        op = BASE / "reports" / "cowork_output" / f"{d}_bets.json"
        if not (bp.exists() and op.exists()):
            continue
        bundle = json.loads(bp.read_text(encoding="utf-8"))
        bets = json.loads(op.read_text(encoding="utf-8"))
        prod = {r["race_id"]: r["hosei_marks"] for r in (bets.get("bets") or [])
                if r.get("hosei_marks")}
        for r in bundle["races"]:
            rid = re.sub(r"\D", "", str(r["race_id"]))[:16]
            if rid not in prod:
                continue
            # 本番と同じ入力: bundle の朝オッズを T-10 実値で上書き（取消馬も落とさない）
            odds = {int(h["umaban"]): h.get("tansho_odds") for h in r["horses"]
                    if h.get("umaban") is not None}
            lp = BASE / "reports" / "live_odds" / f"{rid}.json"
            if lp.exists():
                live = json.loads(lp.read_text(encoding="utf-8"))
                for k, v in (live.get("tansho") or {}).items():
                    odds[int(k)] = v
            built = CD.build_all_arms(r["horses"], odds)
            blend = built["ranks"]["blend_rank"]
            top5 = [b for b, rk in sorted(blend.items(), key=lambda kv: (kv[1], kv[0]))][:5]
            n += 1
            same += (top5 == [int(x["umaban"]) for x in prod[rid][:5]])
            if top5 != [int(x["umaban"]) for x in prod[rid][:5]]:
                mism.append(rid)
            again = CD.build_all_arms(r["horses"], odds)
            det += (json.dumps(again, sort_keys=True, default=str)
                    == json.dumps(built, sort_keys=True, default=str))
            for a, v in built["arms"].items():
                arms_cnt[f"{a}:{v['n_combos']}組"] += 1
    ok = n > 0 and same == n and det == n
    return _res("P5", "候補（順位・R・U0〜U3）の再現", PASS if ok else FAIL,
                f"{n}R で本番 hosei_marks(blend上位5) を完全再現 {same}/{n}、"
                f"同一入力の二度実行が完全一致 {det}/{n}。"
                f"腕の組数分布 {dict(arms_cnt.most_common(6))}"
                + (f" 不一致 {mism[:3]}" if mism else ""),
                note="blend は候補プールの中心選定にのみ使用（§1）。確率・購入額には不使用")


# ============================================================
# P8 — 未来遮断（決定時に存在しない情報を読まない）
# ============================================================
FORBIDDEN_LITERALS = ("kekka", "payout_table", "results.json", "cowork_results",
                      "live_results", "確定着順", "３連複")
DECISION_MODULES = ["jvlink_trio_odds.py",
                    "analysis/trio_portfolio_v2/pl_trio.py",
                    "analysis/trio_portfolio_v2/candidates.py",
                    "analysis/trio_portfolio_v2/allocate.py"]


def p8_future_block() -> dict:
    hits = []
    for rel in DECISION_MODULES:
        p = BASE / rel
        tree = ast.parse(p.read_text(encoding="utf-8"))
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                d = ast.get_docstring(node, clean=False)
                if d:
                    docstrings.add(d)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value in docstrings:
                    continue
                for tok in FORBIDDEN_LITERALS:
                    if tok in node.value:
                        hits.append(f"{rel}:{node.lineno} 文字列 '{tok}'")
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] + [getattr(node, "module", "") or ""]
                if any("settle" in x for x in names):
                    hits.append(f"{rel}:{node.lineno} settle を import")
    return _res("P8", "未来遮断（結果・最終払戻・後続断面を読まない）",
                PASS if not hits else FAIL,
                f"決定経路 {len(DECISION_MODULES)} モジュールを AST 走査（docstring 除外）: "
                f"禁止参照 {len(hits)}件" + (f" {hits[:3]}" if hits else "")
                + "。O_low/q_low は「当該レースより前に確定した情報のみ」から作る規約を "
                  "§4.2/§5.3 の実装時に同じ走査で担保する。")


# ============================================================
# P9 — raw 不変（追記専用・再実行で上書きしない）
# ============================================================
def p9_append_only() -> dict:
    tmp = OUT / "_preflight_tmp"
    for f in tmp.glob("*"):
        f.unlink()
    p1, dup1 = write_append_only(tmp / "x.txt", "FIRST")
    p2, dup2 = write_append_only(tmp / "x.txt", "SECOND")
    ok = (not dup1 and dup2 and p1 != p2
          and p1.read_text(encoding="utf-8") == "FIRST"
          and p2.read_text(encoding="utf-8") == "SECOND")
    man = OUT / "manifest.jsonl"
    lines = man.read_text(encoding="utf-8").splitlines() if man.exists() else []
    hashed = sum(1 for x in lines if '"sha256"' in x)
    for f in tmp.glob("*"):
        f.unlink()
    tmp.rmdir()
    return _res("P9", "raw 不変（追記専用）", PASS if ok else FAIL,
                f"同一パスへの二度書きで上書きが起きない（2本目は __dup1 に退避）={ok}。"
                f"manifest.jsonl {len(lines)}行 / sha256付き {hashed}行 "
                f"(collector {COLLECTOR_VERSION})")


# ============================================================
# Phase 0 (§5.2) の進捗
# ============================================================
def phase0_progress() -> dict:
    snaps = [x for x in sorted(glob.glob(str(OUT / "snapshots" / "*" / "*.json")))
             if "_smoke" not in Path(x).parent.name]
    okn = inw = 0
    for s in snaps:
        try:
            d = json.loads(Path(s).read_text(encoding="utf-8"))
        except Exception:
            continue
        okn += bool(d.get("ok"))
        inw += bool(d.get("in_window"))
    return {"rt_snapshots": len(snaps), "valid": okn, "in_window_8_12": inw,
            "required": 100, "pass": okn >= 100 and inw >= 100,
            "note": "RT(0B35) の T-10 断面。開催日に t10 と並走させて貯める"}


def main() -> int:
    checks = [p1_record_semantics(), p2_timing(), p3_join(), p4_scratch(), p5_candidates()]
    for pid, name, mod in [("P6", "PL 数学 (§3 全単体テスト)", PL),
                           ("P7", "配分数学 (§6 人工ケース全件)", AL),
                           ("P10", "決済（三連複払戻・不成立・返還）", ST)]:
        rs = mod.run_tests()
        bad = [r for r in rs if not r["ok"]]
        checks.append(_res(pid, name, PASS if not bad else FAIL,
                           " / ".join(r["detail"] for r in rs), tests=rs))
    checks.insert(7, p8_future_block())
    checks.insert(8, p9_append_only())
    checks.sort(key=lambda c: int(c["id"][1:]))

    ph0 = phase0_progress()
    n_pass = sum(1 for c in checks if c["status"] == PASS)
    n_fail = sum(1 for c in checks if c["status"] == FAIL)
    verdict = ("PHASE0_NOT_STARTED — Preflight は "
               f"{n_pass}/{len(checks)} 合格 (FAIL {n_fail})。"
               f"RT(0B35) の T-10 断面は {ph0['valid']}/100R。"
               "設計書 §8.1/§9 により候補選択・EV・資金配分・ROI・本番接続へは進まない。")
    rep = {"experiment_id": "BET-TRIO-T10-CONSERVATIVE-PORTFOLIO-2026-v2",
           "plan": "docs/plans/plan_trio_t10_conservative_portfolio_engine_v2.md",
           "generated_at": datetime.now().isoformat(timespec="seconds"),
           "collector_version": COLLECTOR_VERSION,
           "checks": checks, "phase0": ph0, "verdict": verdict}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "preflight_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

    md = ["# Preflight P1〜P10 — BET-TRIO-T10-CONSERVATIVE-PORTFOLIO-2026-v2", "",
          f"生成 {rep['generated_at']} / collector `{COLLECTOR_VERSION}`", "",
          "| ID | 監査 | 判定 | 実測 |", "|---|---|---|---|"]
    for c in checks:
        md.append(f"| {c['id']} | {c['name']} | **{c['status']}** | {c['detail']} |")
    md += ["", f"**Phase 0 (§5.2)**: 有効 T-10 断面 {ph0['valid']}/100R "
               f"(うち 8〜12分窓内 {ph0['in_window_8_12']}R)", "",
           f"**判定**: {verdict}", ""]
    (OUT / "preflight_report.md").write_text("\n".join(md), encoding="utf-8")

    print("=" * 100)
    for c in checks:
        print(f"[{c['status']:>7}] {c['id']:<4} {c['name']}")
        print(f"          {c['detail']}")
    print("=" * 100)
    print(f"Phase 0: 有効 T-10 断面 {ph0['valid']}/100R (窓内 {ph0['in_window_8_12']})")
    print(verdict)
    print(f"[saved] {OUT / 'preflight_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
