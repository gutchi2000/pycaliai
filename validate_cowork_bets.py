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

BASE = Path(__file__).resolve().parent
INPUT_DIR = BASE / "reports" / "cowork_input"
OUTPUT_DIR = BASE / "reports" / "cowork_output"

# --- 見送り閾値 (docs/cowork_prompt.md 絶対禁則 2 と一致させること) ---
CHAOS_SKIP = 0.92      # field_chaos_score >= これ → 見送り
FIELD_SIZE_SKIP = 7    # field_size <= これ → 見送り
PWIN_SKIP = 0.05       # ◎ p_win < これ → 見送り


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
    reasons: list[str] = []

    chaos = race_conf.get("field_chaos_score")
    if chaos is not None and chaos >= CHAOS_SKIP:
        reasons.append(f"chaos {chaos:.3f} >= {CHAOS_SKIP}")

    fs = race_meta.get("field_size")
    if fs is not None and fs <= FIELD_SIZE_SKIP:
        reasons.append(f"field_size {fs} <= {FIELD_SIZE_SKIP}")

    if hon is None:
        reasons.append("◎ なし (tansho 不明)")
    else:
        if hon.get("tansho_odds") is None:
            reasons.append("◎ tansho_odds is null")
        pw = hon.get("p_win")
        if pw is not None and pw < PWIN_SKIP:
            reasons.append(f"◎ p_win {pw:.3f} < {PWIN_SKIP}")

    return reasons


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

    for race in bet_races:
        rid = str(race.get("race_id"))
        brace = bundle.get(rid)
        if brace is None:
            missing_bundle.append(rid)
            continue
        hon = find_hon(brace.get("horses", []))
        reasons = skip_reasons(brace.get("race_meta", {}),
                               brace.get("race_confidence", {}), hon)
        has_bets = bool(race.get("bets"))

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
    print("=" * 64)

    if not violations:
        return 0

    if not args.apply:
        print(f"\n--apply を付けると違反 {len(violations)} race の bets を "
              f"強制的に [] (見送り) に書き換えます。")
        return 2

    # ---------- 強制適用 ----------
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

    bak = backup_path(bets_path)
    bak.write_text(bets_path.read_text(encoding="utf-8"), encoding="utf-8")
    bets_path.write_text(
        json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[APPLIED] {len(violations)} race を見送りに修正しました。")
    print(f"  元ファイル退避 : {bak.name}")
    print(f"  修正後保存     : {bets_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
