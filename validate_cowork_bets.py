#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
validate_cowork_bets.py — Cowork 買い目の「見送り条件」コード強制ガード

背景
----
馬券構築は Cowork (Claude Desktop) が docs/cowork_prompt.md に従って行うが、
見送り判定は **決定論的な数値ルール** (絶対禁則 2 の 4 条件) であるにもかかわらず、
LLM の遵守は毎回保証されない。実際 2026-05-30 では本来 15/23 レースが
見送り条件に該当するのに、Cowork が全 23 レースに買い目を付けてしまった
(= 全レース購入)。

このスクリプトは bundle.json (PyCaLiAI 出力 = 真値) と突き合わせ、
見送り条件に該当する race に買い目が付いていたら **強制的に bets: [] /
race_nature: 見送り に書き換える** ことで、LLM の逸脱に関わらず
「全レース買う」事故を防ぐ。

見送り 4 条件 (docs/cowork_prompt.md 絶対禁則 2 と一致)
----------------------------------------------------------
  1. race_confidence.field_chaos_score >= 0.92  (極度カオス)
  2. race_meta.field_size <= 7                  (頭数不足)
  3. ◎ の tansho_odds is null                   (オッズ未取得)
  4. ◎ の p_win < 0.05                          (本命の確率が極端に低い)

使い方
------
  # ドライラン (違反を表示するだけ、ファイルは変更しない)
  python validate_cowork_bets.py --date 20260530

  # 強制適用 (違反 race の bets を [] に。元ファイルは .bak に退避)
  python validate_cowork_bets.py --date 20260530 --apply

  # 日付省略時は reports/cowork_output/ の最新 *_bets.json を自動検出
  python validate_cowork_bets.py --apply

終了コード
----------
  0 : 違反なし、または --apply で修正完了
  2 : 違反あり (ドライラン時のみ。--apply すれば 0)
  1 : エラー (ファイル無し等)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from production_policy import hard_skip_reasons

BASE = Path(__file__).resolve().parent
INPUT_DIR = BASE / "reports" / "cowork_input"
OUTPUT_DIR = BASE / "reports" / "cowork_output"


def _stdout_utf8() -> None:
    """Windows コンソールの cp932 で文字化けしないよう UTF-8 に。"""
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass


def find_hon(horses: list[dict]) -> dict | None:
    """◎ (mark=='◎') の馬を返す。無ければ None。"""
    for h in horses:
        if h.get("mark") == "◎":
            return h
    return None


def skip_reasons(race_meta: dict, race_conf: dict, hon: dict | None) -> list[str]:
    """この race が満たす見送り条件のリストを返す (空なら買い対象)。"""
    return hard_skip_reasons(race_meta, race_conf, hon)


# --- content バリデーション (馬番実在・券種・金額。LLM/人為ミスの購入指示化を防ぐ) ---
ALLOWED_KINDS = {"単勝", "複勝", "ワイド", "馬連", "三連複"}
REJECTED_KINDS = {"馬単", "三連単"}  # 方向系券種は本番運用で廃止済み
BET_UNIT = 100
MAX_BET_PER = 10000             # 1 点あたり上限 (Cowork 手動なので compute_bets の 7000 より緩め)


def _parse_umaban(sel) -> list[int]:
    """買い目文字列から馬番を抽出 ('3-7'→[3,7] / '16→1'→[16,1] / '7'→[7])。"""
    return [int(x) for x in re.findall(r"\d+", str(sel))]


def content_issues(bet: dict, valid_umaban: set[int]) -> list[str]:
    """1 bet の内容違反理由リスト (空なら OK)。券種・馬番実在・金額を検査。"""
    issues: list[str] = []
    kind = str(bet.get("馬券種", "")).strip()
    sel = bet.get("買い目", "")
    amt = bet.get("購入額", 0)

    if kind in REJECTED_KINDS:
        issues.append(f"廃止券種({kind})")
    elif kind not in ALLOWED_KINDS:
        issues.append(f"未知券種('{kind}')")

    bans = _parse_umaban(sel)
    if not bans:
        issues.append(f"買い目から馬番抽出不可('{sel}')")
    else:
        bad = [b for b in bans if b not in valid_umaban]
        if bad:
            issues.append(f"存在しない馬番{bad}")

    try:
        a = int(amt)
        if a <= 0:
            issues.append(f"金額が非正({amt})")
        elif a % BET_UNIT != 0:
            issues.append(f"100円単位でない({amt})")
        elif a > MAX_BET_PER:
            issues.append(f"1点上限超({amt}>{MAX_BET_PER})")
    except (TypeError, ValueError):
        issues.append(f"金額が数値でない({amt})")

    return issues


def race_bet_total(race: dict) -> int:
    """race 内の購入額合計。"""
    total = 0
    for b in race.get("bets", []) or []:
        try:
            total += int(b.get("購入額", 0) or 0)
        except (TypeError, ValueError):
            pass
    return total


def load_bundle_index(bundle_path: Path) -> dict[str, dict]:
    """bundle.json を race_id -> race の dict に。"""
    data = json.loads(bundle_path.read_text(encoding="utf-8"))
    races = data["races"] if isinstance(data, dict) and "races" in data else data
    idx: dict[str, dict] = {}
    for r in races:
        rid = str(r.get("race_id"))
        idx[rid] = r
    return idx


def backup_path(path: Path) -> Path:
    """既存 .bak を潰さないようユニークなバックアップ先を返す。"""
    bak = path.with_suffix(path.suffix + ".bak")
    n = 2
    while bak.exists():
        bak = path.with_suffix(path.suffix + f".bak{n}")
        n += 1
    return bak


def main() -> int:
    _stdout_utf8()
    ap = argparse.ArgumentParser(description="Cowork 買い目の見送り条件コード強制ガード")
    ap.add_argument("--date", help="YYYYMMDD。省略時は cowork_output の最新 *_bets.json")
    ap.add_argument("--bets", help="bets.json パス (上書き)")
    ap.add_argument("--bundle", help="bundle.json パス (上書き)")
    ap.add_argument("--apply", action="store_true",
                    help="違反 race の bets を [] に強制書き換え (元は .bak に退避)")
    args = ap.parse_args()

    # --- bets.json 決定 ---
    if args.bets:
        bets_path = Path(args.bets)
        m = re.search(r"(\d{8})", bets_path.name)
        date = args.date or (m.group(1) if m else None)
    else:
        date = args.date
        if not date:
            cands = sorted(OUTPUT_DIR.glob("*_bets.json"))
            if not cands:
                print("[ERROR] reports/cowork_output に *_bets.json が無い", file=sys.stderr)
                return 1
            bets_path = cands[-1]
            m = re.search(r"(\d{8})", bets_path.name)
            date = m.group(1) if m else None
            print(f"[auto] 最新 bets を検出: {bets_path.name}")
        else:
            bets_path = OUTPUT_DIR / f"{date}_bets.json"

    if not bets_path.exists():
        print(f"[ERROR] bets ファイルが無い: {bets_path}", file=sys.stderr)
        return 1

    bundle_path = Path(args.bundle) if args.bundle else (INPUT_DIR / f"{date}_bundle.json")
    if not bundle_path.exists():
        print(f"[ERROR] bundle ファイルが無い: {bundle_path}", file=sys.stderr)
        return 1

    bundle = load_bundle_index(bundle_path)
    raw = json.loads(bets_path.read_text(encoding="utf-8"))
    # wrapper 形式 {bets:[...], grade_scope:[...]} とレガシー array の両対応
    bet_races = raw["bets"] if isinstance(raw, dict) and "bets" in raw else raw

    violations: list[dict] = []   # 本来見送りなのに買っていた
    correct_skips = 0             # 正しく見送り済み
    under_skips: list[dict] = []  # 買えるのに見送っていた (警告のみ)
    bought_ok = 0                 # 正しく買い
    missing_bundle: list[str] = []  # bundle に無い race_id
    unverified_races: list[dict] = []  # bundle 不在なのに bets がある race (fail-closed)
    content_violations: list[dict] = []  # 馬番不在/不正券種/金額異常/重複 (見送り条件とは別軸)

    for race in bet_races:
        rid = str(race.get("race_id"))
        brace = bundle.get(rid)
        if brace is None:
            missing_bundle.append(rid)
            if race.get("bets"):
                unverified_races.append(race)
            continue
        hon = find_hon(brace.get("horses", []))
        reasons = skip_reasons(brace.get("race_meta", {}),
                               brace.get("race_confidence", {}), hon)
        has_bets = bool(race.get("bets"))

        # --- content 検査: 買い目の中身 (馬番実在・券種・金額・重複) ---
        if has_bets:
            valid_umaban = {int(h["umaban"]) for h in brace.get("horses", [])
                            if str(h.get("umaban", "")).strip().isdigit()}
            seen_keys: set = set()
            for bet in race.get("bets", []):
                iss = content_issues(bet, valid_umaban)
                key = (str(bet.get("馬券種", "")), str(bet.get("買い目", "")))
                if key in seen_keys:
                    iss = iss + ["重複買い目"]
                seen_keys.add(key)
                if iss:
                    content_violations.append({
                        "race_id": rid, "label": race.get("race_label", ""),
                        "bet": bet, "issues": iss,
                    })

        if reasons and has_bets:
            violations.append({
                "race_id": rid,
                "label": race.get("race_label", ""),
                "reasons": reasons,
                "amount": race_bet_total(race),
                "n_bets": len(race.get("bets", [])),
                "_race": race,
            })
        elif reasons and not has_bets:
            correct_skips += 1
        elif not reasons and not has_bets:
            under_skips.append({"race_id": rid, "label": race.get("race_label", "")})
        else:
            bought_ok += 1

    # ---------- レポート ----------
    print("=" * 64)
    print(f"  見送り条件ガード : {bets_path.name}")
    print(f"  bundle           : {bundle_path.name}")
    print("=" * 64)
    print(f"  race 総数         : {len(bet_races)}")
    print(f"  正しく買い        : {bought_ok}")
    print(f"  正しく見送り      : {correct_skips}")
    print(f"  ★違反(買い→要見送り): {len(violations)}")
    print(f"  警告(見送り→買える): {len(under_skips)}")
    if missing_bundle:
        print(f"  bundle 不在 race  : {len(missing_bundle)} {missing_bundle}")
    if unverified_races:
        print(f"  ★未検証買い race  : {len(unverified_races)} (bundle 不在のため強制見送り対象)")
    print("-" * 64)

    if violations:
        removed_total = sum(v["amount"] for v in violations)
        print(f"  ★ 見送りすべき {len(violations)} race に買い目 "
              f"(計 ¥{removed_total:,}) が付いています:")
        for v in violations:
            print(f"    - {v['race_id']} {v['label']}")
            print(f"        {v['n_bets']}点 / ¥{v['amount']:,}  ← {' , '.join(v['reasons'])}")
    else:
        print("  ★ 見送り違反なし。全 race が見送り条件と整合しています。")

    if under_skips:
        print("-" * 64)
        print(f"  (参考) 買える条件なのに見送っている race {len(under_skips)} 件:")
        for u in under_skips:
            print(f"    - {u['race_id']} {u['label']}")
        print("    ※ これらは強制買いはしません (買い目を捏造しない)。Cowork 再実行で対応。")
    if content_violations:
        print("-" * 64)
        print(f"  ★ 買い目の内容違反 {len(content_violations)} 件 (馬番不在/不正券種/金額/重複):")
        for cv in content_violations:
            b = cv["bet"]
            print(f"    - {cv['race_id']} {cv['label']}: "
                  f"{b.get('馬券種','?')} {b.get('買い目','?')} ¥{b.get('購入額','?')} "
                  f"← {' , '.join(cv['issues'])}")
    print("=" * 64)

    if not violations and not content_violations and not unverified_races:
        return 0

    if not args.apply:
        todo = []
        if violations:
            todo.append(f"見送り違反 {len(violations)} race の bets を []")
        if content_violations:
            todo.append("内容違反/重複 bet を除去")
        if unverified_races:
            todo.append(f"bundle 不在 {len(unverified_races)} race の bets を []")
        print(f"\n--apply を付けると {' / '.join(todo)} に書き換えます。")
        return 2

    # ---------- 強制適用 ----------
    # (1) 見送り違反 race → bets: []
    for v in violations:
        race = v["_race"]
        cond = " / ".join(v["reasons"])
        orig_reason = race.get("race_reason", "")
        race["bets"] = []
        race["race_nature"] = "見送り"
        race["race_reason"] = (
            f"[自動見送り: {cond}] " + orig_reason
        ).strip()
        # advisor / grade_scope はそのまま残す (narrative は有用)

    # (1b) bundle に無い race は馬番・オッズ・見送り条件を検証不能なので強制見送り。
    for race in unverified_races:
        orig_reason = race.get("race_reason", "")
        race["bets"] = []
        race["race_nature"] = "見送り"
        race["race_reason"] = (
            "[自動見送り: bundle に race_id 不在で検証不能] " + orig_reason
        ).strip()

    # (2) content 違反/重複 bet を除去 (見送り矯正で [] になった race は自然に対象外)
    #     内容違反(馬番不在/不正券種/金額) → 全除去 / 重複 → 1 個目を残し 2 個目以降を除去
    n_removed = 0
    bad_content: dict[str, set] = {}
    for cv in content_violations:
        real = [i for i in cv["issues"] if i != "重複買い目"]
        if real:
            b = cv["bet"]
            bad_content.setdefault(cv["race_id"], set()).add(
                (str(b.get("馬券種", "")), str(b.get("買い目", ""))))
    for race in bet_races:
        rid = str(race.get("race_id"))
        bets = race.get("bets") or []
        if not bets:
            continue
        keep, seen = [], set()
        for b in bets:
            k = (str(b.get("馬券種", "")), str(b.get("買い目", "")))
            if rid in bad_content and k in bad_content[rid]:   # 実質的内容違反 → 除去
                n_removed += 1
                continue
            if k in seen:                                       # 重複の 2 個目以降 → 除去
                n_removed += 1
                continue
            seen.add(k)
            keep.append(b)
        if len(keep) != len(bets):
            race["bets"] = keep

    bak = backup_path(bets_path)
    bak.write_text(bets_path.read_text(encoding="utf-8"), encoding="utf-8")
    bets_path.write_text(
        json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[APPLIED] 見送り違反 {len(violations)} + 未検証 {len(unverified_races)} race を [] に / "
          f"内容違反・重複 {n_removed} bet を除去しました。")
    print(f"  元ファイル退避 : {bak.name}")
    print(f"  修正後保存     : {bets_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
