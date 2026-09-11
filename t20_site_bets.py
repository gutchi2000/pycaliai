# -*- coding: utf-8 -*-
"""t20_site_bets.py — サイト公開用 T-20 買い目プレビュー

大会仕様 (masters_vote.aite_switch_tickets) と同じ判定ロジックで、発走 T-20 の
JV-Link オッズから当該レースの買い目を計算し、
reports/masters_vote_site/{date}/{rid16}.json へ書く。build_site.py がこれを読み、
実投票 (masters_vote, T-4) がまだ無いレースだけ TACT のフォールバック表示として使う
(実投票が入れば、または大会側が「見送り/投票失敗」で最終決定した後は、そちらへ
自動的に差し替わる／速報は消える。詳細は build_site.py の load_all_site_preview)。

やらないこと (実運用の他ラインに一切干渉しない):
  - masters_vote.py の submit()/save_ledger() は呼ばない → 大会 API に投票しない
  - reports/live_odds/ (本番 T-10) や reports/vote_odds/ (大会 T-4) には触れない
    → オッズ取得は reports/site_odds/ という専用ディレクトリを使う
  - forward_prices の stage は "t20" 専用 (t10/vote/close とは別 cohort)
  - Discord 通知はしない (2026-09-10 ユーザー指示)

公開される買い目に生オッズ値は含めない (selection/type/reason のみ)。

レース単位ファイル (2026-09-11): 同時刻帯に複数会場のレースが T-20 に達すると
複数プロセスが並行して起動する。1 ファイルへの読み込み→更新→保存だと後勝ちで
互いの結果を消してしまうため、レースごとに独立したファイルへ書く (共有可変状態が
無いので排他制御なしで安全)。

オッズ検証 (2026-09-11): jvlink_odds.py の終了コードだけでは「古い/不完全なオッズを
そのまま速報にしてしまう」事故を防げない (bundle の各馬 horses[].tansho_odds は
朝の静的値であり、aite_switch_tickets は live 側で穴が空いた馬をそのまま静的値で
埋めてしまう)。validate_market() で ok / race_id 一致 / 鮮度 / 現存馬全頭の単勝
オッズ充足を検査し、どれか一つでも欠けたら判定そのものを行わずに取得失敗として扱う。

実行:
  venv311\\Scripts\\python.exe t20_site_bets.py --once <rid16> --date 20260910
  venv311\\Scripts\\python.exe t20_site_bets.py --date 20260910 --list-schedule --lead-min 20
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from t10_runner import (_rid16, build_schedule, latest_bundle_date,
                        load_post_times, parse_hhmm, run_cmd)

BASE = Path(__file__).parent
PY32 = ["py", "-3.12-32"]          # JV-Link は 32-bit COM
JST = timezone(timedelta(hours=9))  # 競馬は常に JST。発走時刻ガードはこれで tz-aware に統一する
SITE_ODDS_DIR = BASE / "reports" / "site_odds"
OUT_DIR = BASE / "reports" / "masters_vote_site"
MARKET_MAX_AGE_MIN = 10.0          # T-20 で取ったオッズが「古い」と判定するしきい値

sys.stdout.reconfigure(encoding="utf-8")

KIND_LABEL = {"wide": "ワイド", "umaren": "馬連"}


def _entry_path(date_str: str, rid: str) -> Path:
    return OUT_DIR / date_str / f"{rid}.json"


def _save_entry(date_str: str, rid: str, entry: dict) -> None:
    """レース単位ファイルへ atomic 書込み (このレースの書込みは他レースと独立)。"""
    d = OUT_DIR / date_str
    d.mkdir(parents=True, exist_ok=True)
    path = _entry_path(date_str, rid)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def validate_market(market: dict, race: dict, rid: str,
                    max_age_min: float = MARKET_MAX_AGE_MIN) -> tuple[bool, str]:
    """このレースの判定に使ってよい market か検査する。

    どれか一つでも NG なら (False, 理由) — 呼び出し側は aite_switch_tickets を
    一切呼ばず「取得失敗」として扱うこと (bundle の静的 tansho_odds が黙って
    使われる経路に絶対に入らせないため)。
    """
    if not isinstance(market, dict):
        return False, "オッズJSON不正 (dictでない)"
    if not market.get("ok"):
        return False, f"オッズ ok=false ({market.get('reason', '')})"
    got_rid = _rid16(market.get("race_id", ""))
    if got_rid != rid:
        return False, f"race_id不一致 (取得={got_rid!r} 期待={rid!r})"

    fetched = market.get("fetched")
    if not fetched:
        return False, "fetched(観測時刻)欠損"
    try:
        ts = datetime.fromisoformat(str(fetched))
    except ValueError:
        return False, f"fetched形式不正 ({fetched!r})"
    age_min = (datetime.now() - ts).total_seconds() / 60.0
    if age_min < -2.0:
        return False, f"fetchedが未来時刻 ({age_min:.1f}分、時計ズレ疑い)"
    if age_min > max_age_min:
        return False, f"オッズ鮮度NG ({age_min:.1f}分前 > {max_age_min:.0f}分)"

    over = market.get("overround_tan")
    if not isinstance(over, (int, float)) or not (1.0 <= over <= 1.5):
        return False, f"overround異常 ({over!r})"

    active = {
        int(h["umaban"]) for h in (race.get("horses") or [])
        if isinstance(h.get("ai_score"), (int, float)) and h.get("umaban") is not None
    }
    if len(active) < 6:
        return False, f"現存馬不足 (<6, 現在{len(active)})"

    tan_raw = market.get("tansho") or {}
    missing, bad = [], []
    for ban in sorted(active):
        # JSON のキーは必ず str だが、呼び出し側が dict を直接組み立てるテストコード
        # 等では int キーもあり得るので両方見る。
        v = tan_raw.get(str(ban), tan_raw.get(ban))
        if v is None:
            missing.append(ban)
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            bad.append(ban)
            continue
        # null は上で拾う。ここでは NaN/Infinity/0/負値を弾く — 単勝オッズは
        # 常に正の有限値のはずで、これらは JV-Link 側の異常値かパース事故の
        # どちらかであり、bundle の静的値へ黙ってフォールバックさせないためにも
        # 「欠損」と同じ扱い (取得失敗) にする。
        if not math.isfinite(fv) or fv <= 0:
            bad.append(ban)
    if missing:
        return False, f"単勝オッズ欠損 (現存馬のうち{len(missing)}頭未取得: {missing})"
    if bad:
        return False, f"単勝オッズ異常値 (現存馬のうち{len(bad)}頭がnull/NaN/Inf/0以下: {bad})"

    return True, ""


def _public_reason(why: str) -> str:
    """aite_switch_tickets の why 文字列からオッズ生値を落とし、券種だけ残す。

    生文字列の例: "aite_switch:ワイド(相手9.3倍)" / "aite_switch:両方(◎1.8倍/相手3.2倍)"
    → 数値は公開しない (JRA-VAN 投稿ガイドライン)。券種の切替根拠だけ残す。
    """
    w = str(why or "")
    if w.startswith("aite_switch:両方"):
        return "大会仕様(T-20速報)：オッズ帯からワイド+馬連本命の両取り判定"
    if w.startswith("aite_switch:ワイド"):
        return "大会仕様(T-20速報)：オッズ帯からワイド上位2点判定"
    if w.startswith("aite_switch:馬連"):
        return "大会仕様(T-20速報)：オッズ帯から馬連本命1点判定"
    return "大会仕様(T-20速報)"


def tickets_to_bets(tickets: list[dict]) -> list[dict]:
    """masters_vote.aite_switch_tickets() の生 ticket → 公開用 {type, selection, reason}。

    odds_t10_low/high・hon_odds・aite_odds・odds・p_model 等のオッズ/確率フィールドは
    すべて落とす (公開しない)。"""
    out = []
    for t in tickets:
        kind = t.get("kind", "wide")
        out.append({
            "type": KIND_LABEL.get(kind, kind),
            "selection": str(t["selection"]),
            "reason": "大会仕様(T-20速報)",
        })
    return out


def publish_to_site(date_str: str) -> None:
    """sync-hf-umami.ps1 で自動的にサイトへ反映する (build_site.py もその内部で実行)。

    2026-09-10 ユーザー指示: 「勝手にサイトにのっけていってほしい」→ 都度確認しない。
    2026-09-11: 以前はここで build_site.py を直接呼んでから sync-hf-umami.ps1 を
    呼んでいたが、sync-hf-umami.ps1 の step 1 が同じ build_site.py を実行するため
    二重実行になっていた上、二重実行分は sync-hf-umami.ps1 内の排他ロックの外側で
    走るため、同時に複数レースが publish を呼ぶと site/data/*.json への書込みが
    競合し得た。sync-hf-umami.ps1 だけを呼ぶ (中の排他ロックが全呼び出し元を直列化する)。
    失敗は非致命 (次のレース処理・次回実行で再試行される)。"""
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
             "-File", str(BASE / "sync-hf-umami.ps1"), "-Date", date_str],
            cwd=str(BASE), timeout=900, capture_output=True, text=True,
            encoding="utf-8", errors="replace")
        tail = "\n".join((r.stdout or "").splitlines()[-20:])
        print(f"  [publish] sync-hf-umami.ps1 (exit {r.returncode})\n{tail}")
        if r.returncode != 0:
            print("       " + (r.stderr or "").strip()[-500:])
    except Exception as exc:
        print(f"  [publish] sync-hf-umami.ps1 失敗 (非致命): {exc}")


def process_race(date_str: str, rid: str, label: str, dry: bool,
                 scheduled_post: datetime | None = None) -> None:
    import masters_vote as mv

    # 発走時刻ガードは tz-aware (JST) で統一する。呼び出し側が naive datetime を
    # 渡してきても (this module's own main() は tz-aware で渡すが、テスト等の
    # 呼び出しは naive のことがある) ここで JST 扱いに正規化し、以降の
    # datetime.now(JST) との比較で naive/aware 混在の TypeError を起こさせない。
    if scheduled_post is not None and scheduled_post.tzinfo is None:
        scheduled_post = scheduled_post.replace(tzinfo=JST)

    now = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{now}] ▶ T-20 サイト買い目: {label} ({rid})")

    # Note: every failure branch below saves an (empty-bets) entry but does NOT call
    # publish_to_site(). An empty-bets entry is invisible on the site regardless
    # (load_all_site_preview in build_site.py skips entries with no bets), so a
    # publish here would be a full site rebuild + HF/Cloudflare push for zero visible
    # change -- wasted work and needless contention for sync-hf-umami.ps1's lock.
    entry = {"race_id": rid, "label": label, "bets": [], "why": "",
             "scheduled_post": scheduled_post.isoformat() if scheduled_post else None,
             "generated_at": datetime.now().isoformat(timespec="seconds")}

    # 発走時刻が分からない、または既に発走を過ぎているレースの速報は作らない
    # (2026-09-11)。「発走後に起動し、その場で取れた新鮮なオッズ」もここで止める
    # -- 速報はあくまで発走前の推奨情報であって、鮮度だけの問題ではない
    # (スケジュールタスクの起床遅延でこの関数自体が発走後に呼ばれるケースも
    # 同じ分岐で拾われる)。
    if scheduled_post is None:
        entry["why"] = "発走時刻不明のため速報生成不可"
        print(f"  [0/2] {entry['why']}")
        if not dry:
            _save_entry(date_str, rid, entry)
        return
    if datetime.now(JST) >= scheduled_post:
        entry["why"] = f"発走時刻超過のため速報生成不可 (予定発走 {scheduled_post:%H:%M})"
        print(f"  [0/2] {entry['why']}")
        if not dry:
            _save_entry(date_str, rid, entry)
        return

    try:
        race = mv.load_bundle_race(date_str, rid)
    except Exception as exc:
        entry["why"] = f"bundle読込失敗: {exc}"
        print(f"  [0/2] {entry['why']}")
        if not dry:
            _save_entry(date_str, rid, entry)
        return

    SITE_ODDS_DIR.mkdir(parents=True, exist_ok=True)
    jv_cmd = [*PY32, "jvlink_odds.py", "--race", rid, "--stage", "t20",
              "--out-dir", str(SITE_ODDS_DIR)]
    if scheduled_post:
        jv_cmd += ["--scheduled-post", scheduled_post.isoformat()]
    rc, out = run_cmd(jv_cmd)
    line = next((l for l in out.splitlines() if "[jvlink_odds]" in l), out.strip()[-200:])
    print(f"  [1/2] jvlink_odds (exit {rc}) {line}")

    if rc != 0:
        entry["why"] = "T-20価格取得失敗 (jvlink_odds exit != 0)"
        if not dry:
            _save_entry(date_str, rid, entry)
        return

    try:
        market = json.loads((SITE_ODDS_DIR / f"{rid}.json").read_text(encoding="utf-8"))
    except Exception as exc:
        entry["why"] = f"オッズJSON読込失敗: {exc}"
        print(f"  [1.5/2] {entry['why']}")
        if not dry:
            _save_entry(date_str, rid, entry)
        return

    ok, why = validate_market(market, race, rid)
    if not ok:
        entry["why"] = f"オッズ検証NG: {why}"
        print(f"  [1.5/2] {entry['why']}")
        if not dry:
            _save_entry(date_str, rid, entry)
        return

    try:
        tickets, why = mv.aite_switch_tickets(race, market)
    except Exception as exc:
        print(f"  [2/2] 買い目計算失敗: {exc}")
        tickets, why = [], "計算失敗"

    # 発走時刻の再チェック: ここまでの JV-Link 取得+判定計算にも実時間がかかるため、
    # プロセス開始時点では発走前でも、判定が終わった今は発走を過ぎているかもしれない。
    if datetime.now(JST) >= scheduled_post:
        entry["bets"] = []
        entry["why"] = f"判定中に発走時刻超過 (予定発走 {scheduled_post:%H:%M})"
        print(f"  [2/2] {entry['why']}")
        if not dry:
            _save_entry(date_str, rid, entry)
        return

    entry["bets"] = tickets_to_bets(tickets)
    entry["why"] = _public_reason(why) if tickets else why
    print(f"  [2/2] {len(entry['bets'])}点  ({why})")

    if dry:
        return
    _save_entry(date_str, rid, entry)
    # publish_to_site already swallows subprocess-level failures internally, but this
    # extra guard makes it structurally impossible for a publish problem (of any kind)
    # to fail this race's Scheduled Task -- the ticket entry above is already durably
    # saved to disk by this point, which is the part that must not be lost.
    try:
        publish_to_site(date_str)
    except Exception as exc:
        print(f"  [publish] 予期しない例外 (非致命、レース処理自体は完了): {exc}")


def main() -> int:
    ap = argparse.ArgumentParser(description="サイト公開用 T-20 買い目プレビュー")
    ap.add_argument("date", nargs="?", default=None, help="YYYYMMDD (省略時は最新 bundle)")
    ap.add_argument("--lead-min", type=float, default=20.0)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--once", default=None, help="rid16 を指定して 1 レースだけ処理")
    ap.add_argument("--list-schedule", action="store_true",
                    help="rid<TAB>発走HH:MM<TAB>label を出力して終了 (t20_site.ps1 用)")
    args = ap.parse_args()

    date_str = args.date or latest_bundle_date()
    if not date_str:
        print("[ERROR] bundle が見つからない (reports/cowork_input/)")
        return 1
    bundle = BASE / "reports" / "cowork_input" / f"{date_str}_bundle.json"
    if not bundle.exists():
        print(f"[ERROR] {bundle} が無い (Phase A を先に実行)")
        return 1

    d = json.loads(bundle.read_text(encoding="utf-8"))
    races = d.get("races", [])

    if args.list_schedule:
        sched, _ = build_schedule(date_str, races, args.lead_min)
        for pt, rid, label in sched:
            print(f"{rid}\t{pt:%H:%M}\t{label}")
        return 0

    if args.once:
        rid = _rid16(args.once)
        rm = next((r.get("race_meta", {}) for r in races
                   if _rid16(r.get("race_id", "")) == rid), {})
        label = f"{rm.get('place','')}{rm.get('R','') or ''} {rm.get('course','')}".strip() or rid
        hm = parse_hhmm(load_post_times(date_str).get(rid, ""))
        scheduled_post = (
            datetime.strptime(date_str, "%Y%m%d").replace(hour=hm[0], minute=hm[1], tzinfo=JST)
            if hm else None)
        process_race(date_str, rid, label, args.dry, scheduled_post=scheduled_post)
        return 0

    print("[ERROR] --once <rid16> か --list-schedule を指定してください")
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[停止] Ctrl+C")
        raise SystemExit(130)
