# -*- coding: utf-8 -*-
"""
compute_bets.py — 馬券構築（confidence 駆動・カード連動）。docs/compute_bets_spec.md 実装。
========================================================================
方針（2026-06-09 改訂）: EV を主決定から「配分の重み（従）」に降格。
NiceGUI の 4 カード（市場一致 / ◎独走度 / 上位2頭集中 / 混戦度）と **同じパーセンタイル＋しきい値**
で「買いの形」を決め、その形で **おいしい馬（value_horses）+ ◎〇▲△** の馬券を提示する。
人間はカードを見ながら最終 見送り/狙い を判断する前提。

形（shape）:
  本命勝負  : ◎独走(>=0.75) & 本線濃厚(top2>=0.75) & 固い(chaos<=0.50) → 馬単8点(◎〇→◎〇▲△△)+単複◎
  ◎軸      : ◎やや優位(top1>=0.50)                                   → 単勝◎+馬連◎軸流し+ワイド+複勝
  広め流し  : 拮抗(top1<0.25) or 分散(top2<0.40)                       → 馬連box+ワイドbox+複勝
  カオス薄  : カオス(chaos_pct>0.75, raw<0.92)                         → ワイド◎軸 薄く+複勝
  標準     : 上記以外                                                 → 単勝◎+馬連◎-〇+ワイド◎軸+複勝
  穴overlay : 市場乖離(market<0.30) かつ value_horses 有 → その馬の単勝/ワイドを上乗せ

見送り(hard, §0): raw chaos>=0.92 / field<=7 / ◎オッズ無 / p_win(◎)<0.05。

実行: PYTHONUTF8=1 python compute_bets.py --bundle reports/cowork_input/20260607_bundle.json --dry

実装状況 (2026-06-12):
  ✅ pair_probs: ワイド/馬連/馬単の確率は bundle 埋込の calibrated PL joint を優先。
     旧 bundle のみ独立積近似 (+21〜27% 過大) にフォールバック。
  ✅ T-10 ライブオッズ: --live-odds-dir reports/live_odds でライブ必須モード。
     単勝/複勝を 0B31 実値に差替。欠損/ok=false/鮮度NG (--max-age-min) は fail-safe 見送り。
  ✅ 書込契約: --apply で cowork_output/{date}_bets.json へ in-place merge (.bak退避+アトミック置換)。
     書込後は validate_cowork_bets.py --apply を必ず通す。デフォルト(--dry)は表示のみ。
  ✅ ワイド/馬単のライブ実値 (0B33/0B34): raw 突合 + 確定配当照合でパーサ確定 (2026-06-12)。
     ライブ実値を優先、無ければ umaren_matrix 推定にフォールバック。
  ✅ --race rid16: 指定レースのみ計算・apply (t10_runner.py が T-10 にレース単位で使う)。
  ✅ t10_runner.py / t10.ps1: 当日オーケストレータ (発走時刻→T-10 自動発火)。
"""
from __future__ import annotations
import argparse, bisect, io, json, math, sys
from pathlib import Path

BASE = Path(__file__).parent
BUDGET, MIN_BET, MAX_BET = 10000, 500, 7000
CHAOS_Q = BASE / "data" / "chaos_quantiles.json"
T10_BLEND = BASE / "data" / "t10_blend.json"

# T-10 補正印 (2026-07-03): u = log(p_win) + λ·log(π) 降順の上位5頭に印を付け直す。
#   p_win=bundle較正PL単勝率 / π=de-vig市場単勝確率 / λ=analysis/fit_t10_blend.py で
#   valid2023 fit。OOS(test2024-25, 6,858R): ◎top3 61.7%→65.1% (Δ+3.41pt CI[+2.5,+4.3])、
#   市場単独64.3%も上回る。表示専用 — 買い目計算・公開印には影響しない。
HOSEI_MARK5 = ["◎", "〇", "▲", "△", "△"]

_BLEND_LAM = -1.0  # 未ロード sentinel


def _blend_lambda():
    """t10_blend.json から λ をロード (無ければ None = 補正印無効, fail-soft)。"""
    global _BLEND_LAM
    if _BLEND_LAM == -1.0:
        try:
            _BLEND_LAM = float(json.loads(
                T10_BLEND.read_text(encoding="utf-8"))["lambda"])
        except Exception:
            _BLEND_LAM = None
    return _BLEND_LAM


def hosei_marks(horses: list[dict]) -> list[dict] | None:
    """blend 順上位5頭の補正印。p_win/オッズ欠損馬は対象外、5頭未満なら None。"""
    lam = _blend_lambda()
    if lam is None:
        return None
    rows = []
    for h in horses:
        b, p, o = (_num(h.get("umaban")), _num(h.get("p_win")),
                   _num(h.get("tansho_odds")))
        if b is None or p is None or o is None or p <= 0 or o <= 0:
            continue
        rows.append((int(b), float(p), 1.0 / float(o),
                     h.get("horse_name") or "", h.get("mark") or ""))
    if len(rows) < 5:
        return None
    s_inv = sum(r[2] for r in rows)  # de-vig 正規化 (レース内定数、順位不変だが明示)
    rows.sort(key=lambda r: -(math.log(r[1]) + lam * math.log(r[2] / s_inv)))
    return [{"mark": HOSEI_MARK5[i], "umaban": r[0], "horse_name": r[3],
             "orig_mark": r[4]} for i, r in enumerate(rows[:5])]

# 出力 bets[] の券種表示順 (ユーザー指定 2026-06-12。金額配分には影響しない)
KIND_ORDER = {"単勝": 0, "複勝": 1, "ワイド": 2, "馬連": 3, "馬単": 4,
              "三連複": 5, "三連単": 6}

# カード閾値（nicegui_app と一致）
TH_TOP1_GO, TH_TOP1_OK = 0.75, 0.50      # ◎独走 / ◎やや優位
TH_TOP2_GO, TH_TOP2_OK, TH_TOP2_LOW = 0.75, 0.50, 0.40  # 本線濃厚 / やや本線 / 分散
TH_CHAOS_HARD, TH_CHAOS_MID = 0.75, 0.50  # カオス / 混戦（パーセンタイル）
TH_MARKET_ANABA = 0.30                    # 市場乖離→妙味
CHAOS_RAW_SKIP = 0.92                     # §0 hard 見送り（生値）
# §0b 参戦規律フェイルセーフ (2026-06-18): chaos_pct(=field_chaos_score=正規化エントロピーの
# 過去分布百分位) がこの閾値を超えるレースは見送る。OOS検証 (2024fit→2025eval,
# analysis/test_race_selection_oos.py): クリーン帯=エントロピー下位1/3(chaos_pct<=0.33)のみ
# ◎複勝ROI~90%/単勝85%/top3 76% と控除床(80%)を明確に超える。mid(0.33-0.67)80.8% /
# chaotic(0.67-)82.5% は床近傍=参戦しても負けを増やすだけ。「最も負けない線」=クリーン帯に絞る。
# 0.33=クリーン帯のみ(=2/3を見送り) / 0.50=+mid / 1.0=ゲート無効(従来挙動)。
CLEAN_BAND_MAX = 0.33

# 参戦規律の二段化 (2026-07-25 配線 [[unwired_roi_audit]] #1): 枠プラン(force_floor)でも
# クリーン帯外は全額参戦せず消化枠水準へ予算降格。参戦自体は維持(プロ条件 10R∧¥100k)しつつ
# 控除床近傍帯への露出を削る (clean-band ゲート実測 ◎複勝+5.31pt の回収)。
DEMOTE_BUDGET = 2000

# 複勝特化(的中率)モード (--fuku-hit): ◎の p_win(bundle値) が閾値以上のレースだけ ◎複勝を flat 購入。
# 設計操作点(offline v6 OOS 2024-25, analysis/hit_rate_frontier.py): 信頼度上位~20%帯 →
#   的中~80% / 回収~92% (valid2023も一致)。控除床(回収100%)は越えない＝「最も負けない高的中」。
# ★閾値は serve(bundle) の p_win 分布に合わせて設定: 本番bundle の p_win は offline より低スケール
#   (中央値 0.13 vs offline ~0.25)。offline絶対値(0.36)では発火0.8%なので、serve実分布で
#   上位~20%(週~15R)になる 0.21 を既定とする (reports/cowork_input/*_bundle 655R で実測)。
# ※的中/回収 80/92% は offline 射影。serve スケール差があるため、実 serve 結果での前向き検証が必須。
FUKU_HIT_THR = 0.21

# 決済ドリフト補正 (2026-07-23 実測: analysis/measure_settle_drift.py,
# reports/live_odds 392勝者 T-10→確定): 勝者オッズは T-10比 平均 -8.9% [CI -10.9,-7.0]
# 縮む(steam)。単勝EVの期待払戻 = p_win × E[確定|勝ち] なので T-10 オッズ素のEVは系統的
# 過大計上。バンド別に補正 (1倍台は保つため 1.0 に丸め、n<60 帯は全体平均 0.91 で平滑)。
# 表示・買い目オッズは生のまま、EV(選別・配分)にのみ適用。t10ログ蓄積で係数を再実測すること。
SETTLE_DRIFT_TAN = [(2.0, 1.00), (4.0, 0.934), (8.0, 0.86), (20.0, 0.916), (9e9, 0.91)]
# 複勝 (n=1162, ×0.885 CI[0.870,0.901]) / ワイド (n=1177, ×0.917 CI[0.904,0.929])
SETTLE_DRIFT_FUKU = [(1.5, 1.00), (2.5, 0.893), (5.0, 0.84), (9e9, 0.894)]
SETTLE_DRIFT_WIDE = [(3.0, 0.883), (7.0, 0.888), (15.0, 0.925), (9e9, 0.948)]

# 相手信頼ゲート閾値: p23=(p2+p3)/(1-p1) の下位1/3境界。
# ★発見期(offline OOF)境界は 0.328 だが、serve(bundle) の p_win は低スケール
#   (FUKU_HIT_THR と同じ現象) で 0.328 だと発火69%に膨らむ。serve 実分布
#   (reports/cowork_input 933R) の 33pct=0.252 に校正 (発火~1/3=設計意図)。
AITE_WEAK_TH = 0.252


def _settle(o, table):
    for hi, m in table:
        if o < hi:
            return o * m
    return o


def settle_tan(o):
    """T-10 単勝オッズ → 決済(勝者条件付き確定)期待オッズ。"""
    return _settle(o, SETTLE_DRIFT_TAN)


def settle_fuku(o):
    return _settle(o, SETTLE_DRIFT_FUKU)


def settle_wide(o):
    return _settle(o, SETTLE_DRIFT_WIDE)

_QT = None
def _qtab():
    global _QT
    if _QT is None:
        try:
            _QT = json.loads(CHAOS_Q.read_text(encoding="utf-8")).get("quantiles", {})
        except Exception:
            _QT = {}
    return _QT

def pct(raw, key):
    """生値→過去分布パーセンタイル(0-1)。テーブル欠如時は None (fail-safe)。

    旧実装は生値をそのまま返したが、chaos 生値域は [0.80,1.0] に圧縮されている
    ため「テーブル破損 → 全レース chaos_pct>0.75 → カオス薄に倒れる」事故になる
    (audit 2026-06-11)。None を返して呼び出し側で見送りに倒す。
    """
    t = _qtab().get(key)
    if not t or len(t) < 2:
        return None
    raw = float(raw or 0)
    if raw <= t[0]: return 0.0
    if raw >= t[-1]: return 1.0
    i = bisect.bisect_right(t, raw)
    lo, hi = t[i - 1], t[i]
    frac = 0.0 if hi == lo else (raw - lo) / (hi - lo)
    return (i - 1 + frac) / (len(t) - 1)


def _num(x):
    try:
        v = float(x); return v if v == v else None
    except Exception:
        return None


def amount_for_ev(ev):
    if ev >= 1.50: return 5500
    if ev >= 1.20: return 3750
    if ev >= 1.00: return 2250
    if ev >= 0.85: return 1500
    return 900


def allocate(weights, budget=BUDGET, mn=MIN_BET, mx=MAX_BET):
    n = len(weights)
    if n == 0: return []
    s = sum(weights) or 1.0
    amts = [min(mx, max(mn, int(round(budget * w / s / 100)) * 100)) for w in weights]
    for _ in range(6000):
        d = budget - sum(amts)
        if d == 0: break
        step = 100 if d > 0 else -100
        order = sorted(range(n), key=lambda i: (-weights[i] if d > 0 else weights[i]))
        for i in order:
            na = amts[i] + step
            if mn <= na <= mx:
                amts[i] = na; break
        else:
            break
    return amts


def load_live_odds(live_dir: Path, rid16: str, max_age_min: float):
    """reports/live_odds/{rid16}.json (jvlink_odds.py 出力) を読む。
    返値 (data, "") か (None, 見送り理由)。fail-safe: 欠損/破損/ok=false/鮮度NG は None。
    """
    from datetime import datetime
    p = Path(live_dir) / f"{rid16}.json"
    if not p.exists():
        return None, "ライブオッズ未取得"
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None, "ライブオッズJSON破損"
    if not d.get("ok"):
        return None, f"ライブオッズ ok=false ({d.get('reason', '')})"
    ts = None
    if d.get("fetched"):
        try:
            ts = datetime.fromisoformat(str(d["fetched"]))
        except ValueError:
            ts = None
    if ts is None:  # 旧形式 (fetched なし) はファイル mtime で代用
        ts = datetime.fromtimestamp(p.stat().st_mtime)
    age_min = (datetime.now() - ts).total_seconds() / 60.0
    if age_min > max_age_min:
        return None, f"ライブオッズ鮮度NG ({age_min:.0f}分前 > {max_age_min:.0f}分)"
    return d, ""


def compute_fuku_hit(race: dict, thr: float = FUKU_HIT_THR, stake: int = BUDGET,
                     live_dir: Path | None = None, max_age_min: float = 20.0) -> dict:
    """複勝特化(的中率)モード: ◎の p_win(calibrated)>=thr のレースだけ ◎複勝を flat 購入。
    選択はモデル確率のみ依存(odds 非依存)。出力契約は compute_race_bets と同形。
    操作点(v6 OOS, analysis/hit_rate_frontier.py): thr=0.36 → 週~14R/的中80%/回収92%。
    """
    rm = race.get("race_meta", {})
    horses = race.get("horses", [])
    field = _num(rm.get("field_size")) or len(horses)
    rid = str(race.get("race_id") or rm.get("race_id") or "")
    label = f"{rm.get('place','')}{rm.get('course','')} {rm.get('race_name','')}".strip()

    # T-10 ライブ複勝オッズ (任意・表示用。選択は確率ベースなので odds 非依存)
    if live_dir is not None:
        live, why = load_live_odds(live_dir, rid[:16], max_age_min)
        if live is None:
            return {"race_id": rid, "race_label": label, "race_nature": "見送り",
                    "race_reason": f"{why} のため見送り (fail-safe)。", "bets": []}
        live_fuku = {int(k): v for k, v in (live.get("fukusho") or {}).items()}
        horses = [dict(h) for h in horses]
        for h in horses:
            b = _num(h.get("umaban"))
            if b is not None and int(b) in live_fuku:
                h["fuku_odds_low"], h["fuku_odds_high"] = live_fuku[int(b)]

    hon = next((h for h in horses
                if h.get("mark") == "◎" and _num(h.get("umaban")) is not None), None)
    if hon is None:
        return {"race_id": rid, "race_label": label, "race_nature": "見送り",
                "race_reason": "◎不在のため複勝特化対象外。", "bets": []}
    if field < 5:
        return {"race_id": rid, "race_label": label, "race_nature": "見送り",
                "race_reason": "少頭数(<5)で複勝対象外。", "bets": []}
    pwin = _num(hon.get("p_win"))
    if pwin is None or pwin < thr:
        pw_s = f"{pwin:.3f}" if pwin is not None else "NA"
        return {"race_id": rid, "race_label": label, "race_nature": "見送り",
                "race_reason": f"◎p_win {pw_s}<閾値{thr:.2f}（複勝特化の的中ゲート未達）で見送り。",
                "bets": []}

    ban = int(hon.get("umaban"))
    lo, hi = _num(hon.get("fuku_odds_low")), _num(hon.get("fuku_odds_high"))
    odds_note = f"（複勝{lo:.1f}-{hi:.1f}倍）" if lo and hi else ""
    reason = (f"◎p_win {pwin:.3f}>=閾値{thr:.2f} → 高的中複勝{odds_note}。"
              f"複勝特化(的中率重視)選別。※閾値は serve 実測で要校正(offline射影 的中~80%/回収~92%)。")
    bet = {"馬券種": "複勝", "買い目": str(ban), "購入額": int(stake), "理由": reason}
    return {"race_id": rid, "race_label": label, "race_nature": "複勝特化",
            "race_reason": reason, "bets": [bet]}


def compute_race_bets(race: dict, live_dir: Path | None = None,
                      max_age_min: float = 20.0, budget: int = BUDGET,
                      force_floor: bool = False,
                      demote_budget: int | None = None) -> dict:
    # force_floor=True: §0b 参戦規律(クリーン帯ゲート)の「見送り」を緩める。プロ条件
    #   (週10R∧¥100k)を満たすため枠プランで使用。§0 hard(chaos>=0.92/少頭数/◎薄/odds欠)は維持。
    # demote_budget: force_floor 時の二段目。クリーン帯外レースは見送らない代わりに
    #   予算をこの額まで強制降格 (参戦規律ゲートの実効化, 2026-07-25)。
    rm = race.get("race_meta", {})
    rc = race.get("race_confidence", {})
    bj = race.get("buy_judgment", {})
    horses = race.get("horses", [])
    field = _num(rm.get("field_size")) or len(horses)
    rid = str(race.get("race_id") or rm.get("race_id") or "")
    label = f"{rm.get('place','')}{rm.get('course','')} {rm.get('race_name','')}".strip()

    # ---- T-10 ライブオッズ (spec §入力2): 指定時は必須 = 読めなければ fail-safe 見送り ----
    live_note = ""
    lwide: dict = {}    # {(i,j): (lo,hi)} ライブワイド実値
    lumatan: dict = {}  # {(i,j): odds}   ライブ馬単実値 (i=1着)
    if live_dir is not None:
        live, why = load_live_odds(live_dir, rid[:16], max_age_min)
        if live is None:
            return {"race_id": rid, "race_label": label, "race_nature": "見送り",
                    "race_reason": f"{why} のため見送り (fail-safe)。", "bets": []}
        # 単勝/複勝を T-10 実値に差し替え (horses は bundle 入力を破壊しないようコピー)
        live_tan = {int(k): v for k, v in (live.get("tansho") or {}).items()}
        live_fuku = {int(k): v for k, v in (live.get("fukusho") or {}).items()}
        horses = [dict(h) for h in horses]
        n_tan = 0
        for h in horses:
            b = _num(h.get("umaban"))
            if b is None:
                continue
            b = int(b)
            if b in live_tan:
                h["tansho_odds"] = live_tan[b]
                n_tan += 1
            if b in live_fuku:
                h["fuku_odds_low"], h["fuku_odds_high"] = live_fuku[b]
        # ワイド/馬単のライブ実値 (0B33/0B34, raw 突合で確定 2026-06-12)。
        # 無ければ従来どおり umaren_matrix 推定にフォールバック。
        for k, v in (live.get("wide") or {}).items():
            try:
                i, j = (int(x) for x in k.split("-"))
                lwide[(min(i, j), max(i, j))] = (float(v[0]), float(v[1]))
            except (ValueError, TypeError, IndexError):
                continue
        for k, v in (live.get("umatan") or {}).items():
            try:
                i, j = (int(x) for x in k.split(">"))
                lumatan[(i, j)] = float(v)
            except (ValueError, TypeError):
                continue
        live_note = (f" [T-10オッズ 単複{n_tan}頭"
                     + (f"/ワイド{len(lwide)}組" if lwide else "")
                     + (f"/馬単{len(lumatan)}組" if lumatan else "") + "]")

    by_ban = {int(h["umaban"]): h for h in horses if _num(h.get("umaban")) is not None}

    # T-10 補正印 (表示専用)。live 差し替え後の tansho_odds で計算。
    # 見送りレースにも付けるため early return より前に算出する。
    hosei = hosei_marks(horses)

    def fld(b, k):
        return _num(by_ban.get(b, {}).get(k)) if b is not None else None

    marks = {}
    for h in horses:
        m = h.get("mark", "")
        if m in ("◎", "〇", "○", "▲", "△"):
            marks.setdefault("○" if m == "〇" else m, []).append(int(h["umaban"]))
    hon = (marks.get("◎") or [None])[0]
    tai = (marks.get("○") or [None])[0]
    san = (marks.get("▲") or [None])[0]
    osae = marks.get("△", [])

    chaos_raw = _num(rc.get("field_chaos_score")) or 0.0
    pwin_hon, tan_hon = fld(hon, "p_win"), fld(hon, "tansho_odds")

    # ---- §0 hard 見送り ----
    skip = None
    if chaos_raw >= CHAOS_RAW_SKIP: skip = f"極度カオス(chaos {chaos_raw:.3f}>=0.92)"
    elif field <= 7: skip = "少頭数(<=7)"
    elif hon is None or tan_hon is None: skip = "本命オッズ未取得"
    elif pwin_hon is not None and pwin_hon < 0.05: skip = "本命勝率<0.05"
    if skip:
        return {"race_id": rid, "race_label": label, "race_nature": "見送り",
                "race_reason": f"{skip} のため見送り。", "bets": [],
                **({"hosei_marks": hosei} if hosei else {})}

    # ---- カード値（パーセンタイル + market 生値）----
    top1 = pct(rc.get("top1_dominance"), "top1_dominance")
    top2 = pct(rc.get("top2_concentration"), "top2_concentration")
    chaos = pct(rc.get("field_chaos_score"), "field_chaos_score")
    if top1 is None or top2 is None or chaos is None:
        # 分位テーブル欠如/破損 → shape 判定不能。fail-safe 見送り
        return {"race_id": rid, "race_label": label, "race_nature": "見送り",
                "race_reason": "chaos_quantiles.json 欠如/破損で指標変換不能のため見送り (fail-safe)。",
                "bets": [], **({"hosei_marks": hosei} if hosei else {})}

    # ---- §0b 参戦規律(二段): クリーン帯(低エントロピー)外は 見送り or 消化枠降格 ----
    #   force_floor 無し: 従来どおり見送り (最も負けない線)。
    #   force_floor 有り(枠プラン): 参戦は維持しつつ demote_budget へ予算降格 (2026-07-25 配線)。
    demote_note = ""
    if chaos > CLEAN_BAND_MAX:
        if not force_floor:
            return {"race_id": rid, "race_label": label, "race_nature": "見送り",
                    "race_reason": f"クリーン帯外(混戦度 pct{chaos:.2f}>{CLEAN_BAND_MAX:.2f}・参戦規律)で見送り。"
                                   "◎の信頼が薄い帯=OOSで控除床近傍につき不参戦。", "bets": [],
                    **({"hosei_marks": hosei} if hosei else {})}
        if demote_budget and int(budget) > int(demote_budget):
            budget = int(demote_budget)
            demote_note = (f"クリーン帯外(混戦度pct{chaos:.2f}>{CLEAN_BAND_MAX:.2f})"
                           f"→消化枠¥{budget:,}に降格。")
    market = _num(rc.get("ai_market_agreement")) or 0.0
    anaba = market < TH_MARKET_ANABA
    value_bans = [int(v["umaban"]) for v in bj.get("value_horses", []) if _num(v.get("umaban")) is not None]

    # ---- 形（shape）決定 ----
    if hon and tai and top1 >= TH_TOP1_GO and top2 >= TH_TOP2_GO and chaos <= TH_CHAOS_MID:
        shape = "本命勝負"
    elif hon and top1 >= TH_TOP1_OK:
        shape = "◎軸"
    elif top1 < 0.25 or top2 < TH_TOP2_LOW:
        shape = "広め流し"
    elif chaos > TH_CHAOS_HARD:
        shape = "カオス薄"
    else:
        shape = "標準"

    # ---- EV ヘルパ → 候補 (kind, sel, ev, weight_boost) ----
    um = race.get("umaren_matrix", {})
    def umaren(i, j):
        a, b = sorted((i, j)); return _num(um.get(f"{a}-{b}"))
    # bundle 埋込の正確な PL joint 確率 (calibrated, export_marks_json が印馬10ペアに付与)。
    # あればワイド/馬連/馬単の確率に優先使用。無い旧 bundle は従来近似にフォールバック
    # (ワイド独立積は +21〜27% 系統過大 → docs/audit_20260611.md 🔴)。
    pp = race.get("pair_probs", {}) or {}
    def pair_p(i, j, kind):
        d = pp.get(f"{min(i, j)}-{max(i, j)}")
        if not d:
            return None
        if kind == "umatan":
            return _num((d.get("umatan") or {}).get(f"{i}→{j}"))
        return _num(d.get(kind))
    cands = []  # [kind, sel, bans, odds, ev, ish, boost]
    # オッズ上限（モデルのテール過大評価が EV 経由で大穴を生むのを遮断）
    # 馬連 100→50 (2026-06-19 ユーザー禁止「穴推奨の馬連」: 高オッズ馬連=穴を遮断)
    ODDS_CAP = {"馬連": 50.0, "ワイド": 50.0, "馬単": 200.0}
    def push(kind, sel, bans, odds, ev, boost=1.0):
        if not odds or ev is None: return
        cap = ODDS_CAP.get(kind)
        if cap and odds > cap: return
        # 同一 (kind, sel) の重複統合: 穴 overlay (value_horses) と shape 側が
        # 同じ買い目 (例 vb==◎ の単勝/複勝) を生成し得る。dedup しないと同一馬券に
        # 2 行・別金額が出て露出が最大 2×MAX_BET になり、点数上限 cap も浪費する。
        # boost は強い方を採用。
        for c in cands:
            if c[0] == kind and c[1] == sel:
                if boost > c[6]:
                    c[6] = boost
                return
        cands.append([kind, sel, tuple(bans), odds, ev, (hon in bans), boost])
    def c_tan(b, boost=1.0):
        o, pw = fld(b, "tansho_odds"), fld(b, "p_win")
        # EV は決済ドリフト補正後オッズで計算 (表示オッズ o は生のまま)
        if o and pw: push("単勝", str(b), (b,), o, pw * settle_tan(o), boost)
    def c_fuku(b, boost=1.0):
        lo, hi, ps = fld(b, "fuku_odds_low"), fld(b, "fuku_odds_high"), fld(b, "p_sho")
        # EV は決済ドリフト補正後 (表示は生 mid)
        if lo and hi and ps: push("複勝", str(b), (b,), (lo + hi) / 2, ps * settle_fuku((lo + hi) / 2), boost)
    def c_umaren(i, j, boost=1.0):
        o = umaren(i, j)
        if not o:
            return
        p = pair_p(i, j, "umaren")
        if p is None:  # 旧 bundle 互換: Harville 風近似
            pi, pj = fld(i, "p_win"), fld(j, "p_win")
            if not (pi and pj and 0 < pi < 1 and 0 < pj < 1):
                return
            p = pi * pj / (1 - pi) + pj * pi / (1 - pj)
        push("馬連", f"{min(i,j)}-{max(i,j)}", (i, j), o, o * p, boost)
    def c_wide(i, j, boost=1.0):
        lw = lwide.get((min(i, j), max(i, j)))
        if lw:                       # T-10 ライブ実値 (lo+hi)/2
            o = (lw[0] + lw[1]) / 2.0
        else:                        # 推定: ワイド ≈ 馬連/3
            o_um = umaren(i, j)
            if not o_um:
                return
            o = o_um / 3.0
        p = pair_p(i, j, "wide")
        if p is None:  # 旧 bundle 互換: 独立積近似 (+21〜27% 系統過大)
            si, sj = fld(i, "p_sho"), fld(j, "p_sho")
            if not (si and sj):
                return
            p = si * sj
        # EV は決済ドリフト補正後 (表示は生オッズ)
        push("ワイド", f"{min(i,j)}-{max(i,j)}", (i, j), o, settle_wide(o) * p, boost)
    def c_umatan(i, j, boost=1.0):  # i→j (i 1着, j 2着)
        pi, pj = fld(i, "p_win"), fld(j, "p_win")
        ut = lumatan.get((i, j))     # T-10 ライブ実値
        if ut is None:               # 推定: 馬単odds ≈ 馬連 × (pi+pj)/pi
            o = umaren(i, j)
            if not (o and pi and pj and 0 < pi < 1):
                return
            ut = o * (pi + pj) / pi
        p = pair_p(i, j, "umatan")
        if p is None:  # 旧 bundle 互換
            if not (pi and pj and 0 < pi < 1):
                return
            p = pi * pj / (1 - pi)
        push("馬単", f"{i}→{j}", (i, j), ut, ut * p, boost)
    def c_pair(i, j, boost=1.0):
        """ペア馬券: 馬連とワイドを【並行】生成 (2026-06-18 ユーザー指示「馬連と並行して」)。
        馬連=ROI主軸(gate_q2 prob-only 79%>控除77.5%)、ワイド=的中率/床防御で並走。
        馬連オッズ欠損ペアは c_umaren が内部 return → ワイドのみ(従来フォールバックも包含)。
        選抜(後段)で馬連/ワイドを型別に交互確保しないと prob-first が全ワイドに倒れる点に注意。"""
        c_umaren(i, j, boost)   # 馬連（umaren_matrix 欠損なら内部で return）
        c_wide(i, j, boost)     # ワイド 並行

    rel = [x for x in (tai, san, *osae) if x]            # 相手(〇▲△)
    # ① ペアの相手は ◎〇▲ にcap (相手4頭=10点でワイド床割れ76.7%。wide-playbook 2026-06-18)。
    rel_pair = [x for x in (tai, san) if x]              # ペア用相手(〇▲のみ)
    box4 = [b for b in (hon, tai, san, *osae) if b][:4]  # 上位4

    # ---- 形ごとに候補生成（おいしい馬 boost）----
    # ★馬連を prob-first で再有効化 (2026-06-15 監査, c_pair)。旧「全廃」の根拠 ROI56%
    #   は EV選抜時代の馬連の数字。gate_q2_umaren_priceedge は「馬連を p_umaren 上位
    #   top-K で流す」と test ROI 79%(>控除77.5%)/CLV+ を示し、EV選抜(66.6%/CLV-)を
    #   13pt 上回る。よって馬連 prob-first は規律(roi_verdict)の趣旨に合致。ペア生成は
    #   c_pair(馬連優先・欠損時ワイド)、選抜は prob-first(:後段)。
    # ⚠ 規律 (2026-06-11): boost の点推定での弄りは禁止。cowork_results の roi_verdict
    #   が above/below_takeout になった時のみ boost 変更可 (単勝120%→96%回帰の再発防止)。
    if shape == "本命勝負" and hon and tai:
        # 本命勝負(◎独走/本線濃厚/固い): 馬単8点フォメを撤廃し馬連+ワイド並行へ全面置換
        # (2026-06-18 ユーザー指示。馬単は2/122=構造的回収不能 [[loss_forensics_842]]、
        #  馬連UMAMIが馬単に+6pt。固い本命race=◎軸の単複厚+◎-〇▲ペアに集約)。
        c_tan(hon, 1.6); c_fuku(hon, 1.1)                 # 単勝◎厚 + 複勝◎
        for r in rel_pair:
            c_pair(hon, r, 1.3)                           # ◎-〇▲ 馬連+ワイド並行(厚boost)
    elif shape == "◎軸" and hon:
        c_tan(hon, 1.6)                                   # 単勝◎（最重視・実績120%）
        for r in rel_pair:
            c_pair(hon, r, 1.1)                            # ◎-〇▲の馬連+ワイド並行(馬単撤廃)
        c_fuku(hon, 1.0)
    elif shape == "広め流し":                              # ◎弱 → ペア流し(馬連+ワイド並行)
        for r in rel_pair:
            if hon: c_pair(hon, r, 1.2)
        if tai:
            for r in [x for x in (hon, san) if x and x != tai]:
                c_pair(tai, r, 1.0)
        if hon: c_fuku(hon, 1.1)
    elif shape == "カオス薄":
        if hon:
            for r in rel_pair: c_pair(hon, r, 1.0)
            c_fuku(hon, 1.2)
    else:  # 標準
        if hon:
            c_tan(hon, 1.4); c_fuku(hon, 1.1)
            for r in rel_pair:
                c_pair(hon, r, 1.1)

    # ---- 穴 overlay: おいしい馬（value_horses）を上乗せ ----
    # 単勝は 30 倍キャップ: audit_ev_bin_roi で単勝 50+ オッズ帯は ROI 30-61% /
    # CLV -54〜-67% の最悪セグメント (◎大穴帯勝率0%の構造弱点 [[longshot-weakness]])。
    ANA_TAN_ODDS_CAP = 30.0
    if anaba:
        for vb in value_bans:
            if (fld(vb, "tansho_odds") or 999) <= ANA_TAN_ODDS_CAP:
                c_tan(vb, 1.3)
            if hon and vb != hon:
                c_pair(vb, hon, 1.2)
            c_fuku(vb, 1.1)

    # ---- 銘柄選抜: prob-first（gate_q2 実証, 2026-06-15 監査）----
    # gate_q2_umaren_priceedge: 馬連/ワイドのペアを EV(価格ズレ)で選ぶと test ROI 66.6%
    # /CLV-5.8%、calibrated p_pair 上位 top-K で選ぶと 79%/CLV+。EV選抜は毎回約13pt捨てる。
    # → ペアは「ペア内で p_pair 降順 top-K」。EVフロアは課さない（低オッズ本命ペアを
    #   弾かないため。大穴は ODDS_CAP/umami 罠ゲートで別途排除済み）。
    #   p_pair = ev/odds = c[4]/c[3]（ev=odds×p の定義から厳密復元、新カラム不要）。
    # ★型混在の生prob比較は禁止（複勝 p_sho > ペア p_pair でペアが押し出される）。
    #   単複の◎系はアンカーとして別枠確保し、◎必須を切り詰め後も保証する。
    # ※配分(amount_for_ev×boost)は EV-band サイジングのまま=follow-up（選抜が主効果）。
    PAIR = {"馬連", "ワイド"}
    # 馬単撤廃(2026-06-18)で本命勝負も pair(馬連+ワイド)+単複中心 → 馬単専用の特別選抜は不要。
    # 全 shape を prob-first(型別交互)選抜に統一する(EV選抜は prob-first 比 -13pt)。
    is_honmei = False
    _p = lambda c: (c[4] / c[3]) if c[3] else 0.0   # p_pair / p_single（大=良）

    # ---- 相手信頼ゲート (2026-07-23 配線): ◎の的中率は相手軸でほぼ不変だが、
    #   組み合わせ馬券の的中率は相手弱で半減する (発見 2016-23 n=27,596:
    #   ◎強ワイドr1r2 的中 43.8%→23.0% / holdout 2024-25 で 43.4%→24.9% 再現、
    #   ◎単勝は 39.7→35.3% と不変)。相手弱 = p23=(p2+p3)/(1-p1) が発見期下位1/3
    #   境界 (0.328) 未満。該当時はペア/複数頭候補を落とし ◎単複へ予算集中
    #   (ROIは全セル同等=これは的中率・トリガミのレバー)。
    aite_note = ""
    _pws = sorted((_num(h.get("p_win")) or 0.0 for h in horses), reverse=True)
    if len(_pws) >= 3 and 0 < _pws[0] < 1.0:
        _p23 = (_pws[1] + _pws[2]) / max(1e-9, 1.0 - _pws[0])
        if _p23 < AITE_WEAK_TH:
            _nb = len(cands)
            _singles_only = [c for c in cands if len(c[2]) == 1]
            if _singles_only:  # 単複候補が無い時はゲートを適用しない (fail-safe)
                cands = _singles_only
                aite_note = f"相手弱(p23 {_p23:.2f}<{AITE_WEAK_TH}) ペア{_nb - len(cands)}点抑制→◎単複集中。"

    if is_honmei:
        # 本命勝負: 馬単8点 formation 全採用 + 単複◎(EVフロア)、従来 EV×boost 順を維持。
        floor = 0.70
        chosen = [c for c in cands if c[4] >= floor or c[0] == "馬単"]
        if hon and not any(c[5] for c in chosen):
            hc = [c for c in cands if c[5]]
            if hc: chosen.append(max(hc, key=lambda c: c[4] * c[6]))
        if not chosen and cands:
            chosen = [max(cands, key=lambda c: c[4] * c[6])]
        cap = max(1, min(8, int(budget) // MIN_BET))
        chosen = sorted(chosen, key=lambda c: -(c[4] * c[6]))[:cap]
    else:
        # 非本命勝負: ◎単複アンカー → prob-first ペア → 残り単複 の優先順で cap まで。
        floor = 0.80
        cap = max(1, min(6, int(budget) // MIN_BET))
        # 馬連と並行(2026-06-18): 型別に prob-first 確保し交互配置(混合 prob-first だと
        # ワイド p_pair > 馬連 p_pair で全ワイドに倒れ「馬連並行」が消えるため)。
        _pu = sorted([c for c in cands if c[0] == "馬連"], key=lambda c: -_p(c))
        _pw = sorted([c for c in cands if c[0] == "ワイド"], key=lambda c: -_p(c))
        pairs = []
        for _k in range(max(len(_pu), len(_pw))):
            if _k < len(_pu): pairs.append(_pu[_k])
            if _k < len(_pw): pairs.append(_pw[_k])
        singles = sorted([c for c in cands if c[0] not in PAIR and c[4] >= floor],
                         key=lambda c: (0 if c[5] else 1, -_p(c)))
        anchors = [c for c in singles if c[5]][:2]      # ◎絡み単複を最大2枠確保
        anchor_ids = {id(c) for c in anchors}
        chosen = list(anchors[:cap])
        for grp in (pairs, [c for c in singles if id(c) not in anchor_ids]):
            for c in grp:
                if len(chosen) >= cap: break
                chosen.append(c)
        # ◎必須: ◎絡みが1つも無ければ最高prob◎絡みを先頭に差し込む（切り詰めで消えない）
        if hon and not any(c[5] for c in chosen):
            hc = sorted([c for c in cands if c[5]], key=lambda c: -_p(c))
            if hc: chosen = ([hc[0]] + [c for c in chosen if id(c) != id(hc[0])])[:cap]
        if not chosen and cands:
            chosen = [max(cands, key=_p)]

    # ---- 配分（EV×boost 重み、キャップ厳守）----
    base = [amount_for_ev(c[4]) * c[6] for c in chosen]
    amts = allocate(base, budget=int(budget))
    waku = bj.get("waku_tag") or "参加枠"
    bets = []
    for c, amt in zip(chosen, amts):
        kind, sel, bans, odds, ev, ish, boost = c
        role = "おいしい" if (set(bans) & set(value_bans)) else ("◎絡み" if ish else "相手")
        bets.append({"馬券種": kind, "買い目": sel, "購入額": int(amt), "枠タグ": waku,
                     "理由": f"{role}（{kind} {odds:.1f}倍）"})
    # 表示順: 単勝→複勝→ワイド→馬連→馬単→三連複→三連単、同券種内は金額降順
    bets.sort(key=lambda b: (KIND_ORDER.get(b["馬券種"], 9), -b["購入額"]))

    hon_name = by_ban.get(hon, {}).get("horse_name", "本命")
    rr = (f"◎{hon_name}。{shape}（独走{top1:.2f}/集中{top2:.2f}/混戦{chaos:.2f}/"
          f"市場{market:+.2f}）で {len(bets)}点。{demote_note}{aite_note}{live_note}").rstrip()
    return {"race_id": rid, "race_label": label, "race_nature": shape, "race_reason": rr,
            "confidence": {"top1_pct": round(top1, 3), "top2_pct": round(top2, 3),
                           "chaos_pct": round(chaos, 3), "market": round(market, 3)},
            "bets": bets, **({"hosei_marks": hosei} if hosei else {})}


def fmt_hosei(hm: list[dict]) -> str:
    """補正印の1行表示。元印から昇格/降格した馬に * を付ける。"""
    parts = []
    for x in hm:
        chg = "*" if (x.get("orig_mark") or "") != x["mark"] else ""
        parts.append(f"{x['mark']}{x['umaban']}{x['horse_name']}{chg}")
    return "補正印(オッズblend): " + " ".join(parts)


def apply_to_bets_json(date_str: str, computed: list[dict]) -> Path:
    """spec §書込契約: reports/cowork_output/{date}_bets.json へ in-place merge。
    race_id 一致は置換・無ければ追加。書込前に既存を .bak へ退避 (validate と同方式)。
    """
    out_dir = BASE / "reports" / "cowork_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{date_str}_bets.json"
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        races_list = raw["bets"] if isinstance(raw, dict) and "bets" in raw else raw
        # ユニーク .bak 退避
        bak = path.with_suffix(path.suffix + ".bak")
        n = 2
        while bak.exists():
            bak = path.with_suffix(path.suffix + f".bak{n}")
            n += 1
        bak.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[apply] 既存 bets を退避: {bak.name}")
    else:
        raw = {"bets": []}
        races_list = raw["bets"]
    by_rid = {str(r.get("race_id")): i for i, r in enumerate(races_list)}
    n_rep = n_add = 0
    for e in computed:
        i = by_rid.get(e["race_id"])
        if i is not None:
            # 書込契約: Cowork の narrative (advisor) は消さずに引き継ぐ
            old = races_list[i]
            if isinstance(old, dict) and old.get("advisor") and not e.get("advisor"):
                e = {**e, "advisor": old["advisor"]}
            races_list[i] = e
            n_rep += 1
        else:
            races_list.append(e)
            n_add += 1
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)  # アトミック置換
    print(f"[apply] {path.name}: 置換{n_rep} / 追加{n_add}")
    return path


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--dry", action="store_true",
                    help="表示のみ (デフォルト挙動と同じ。--apply の対義として明示用)")
    ap.add_argument("--apply", action="store_true",
                    help="reports/cowork_output/{date}_bets.json へ書込 (in-place merge, .bak退避)")
    ap.add_argument("--live-odds-dir", default=None,
                    help="T-10 ライブオッズ dir (例 reports/live_odds)。指定時は"
                         "ライブ必須モード: 欠損/ok=false/鮮度NG のレースは fail-safe 見送り")
    ap.add_argument("--max-age-min", type=float, default=20.0,
                    help="ライブオッズの許容鮮度 (分, default 20)")
    ap.add_argument("--race", default=None,
                    help="rid16 (カンマ区切り可)。指定レースのみ計算・apply する"
                         " (t10_runner がレース毎 T-10 に使う。他レースの上書き防止)")
    ap.add_argument("--budget", type=int, default=BUDGET,
                    help=f"1レース予算 (default {BUDGET}。Discord 再計算コマンドが使う)")
    ap.add_argument("--plan", default=None,
                    help="build_bet_plan.py の出力 (reports/bet_plan/{date}.json)。"
                         "指定時は各レース予算を枠プランから取得し、枠外(見送り/対象外)は買わない")
    ap.add_argument("--fuku-hit", action="store_true",
                    help="複勝特化(的中率)モード: ◎の p_win>=閾値のレースだけ ◎複勝を flat 購入"
                         " (週~15R/的中80%/回収92%操作点)。--budget が1点の購入額")
    ap.add_argument("--fuku-hit-thr", type=float, default=FUKU_HIT_THR,
                    help=f"複勝特化の p_win(◎) 閾値 (default {FUKU_HIT_THR}=週~14R)。下げるほど本数増")
    args = ap.parse_args()
    d = json.load(open(args.bundle, encoding="utf-8"))
    races = d.get("races", d if isinstance(d, list) else [])

    # 枠プラン: {rid16: budget}。plan 指定時のみ。非指定なら従来の単一 --budget。
    import re as _re2
    plan_budget = {}
    if args.plan:
        pj = json.load(open(args.plan, encoding="utf-8"))
        for _tier, _rows in (pj.get("tiers") or {}).items():
            for _row in _rows:
                if _row.get("budget"):
                    plan_budget[_re2.sub(r"\D", "", str(_row.get("rid")))[:16]] = int(_row["budget"])
        print(f"[plan] {args.plan} 枠対象 {len(plan_budget)}R")
    if args.race:
        import re as _re
        want = {_re.sub(r"\D", "", x)[:16] for x in args.race.split(",") if x.strip()}
        races = [r for r in races
                 if _re.sub(r"\D", "", str(r.get("race_id", "")))[:16] in want]
        if not races:
            print(f"[ERROR] --race {args.race} に一致するレースが bundle に無い", file=sys.stderr)
            return 1
    live_dir = Path(args.live_odds_dir) if args.live_odds_dir else None
    if live_dir is not None:
        print(f"[live] T-10 モード: {live_dir} (鮮度 {args.max_age_min:.0f}分)")
    out = []
    for r in races:
        if args.fuku_hit:
            out.append(compute_fuku_hit(r, thr=args.fuku_hit_thr, stake=args.budget,
                                        live_dir=live_dir, max_age_min=args.max_age_min))
        elif args.plan:
            rid16 = _re2.sub(r"\D", "", str(r.get("race_id", "")))[:16]
            rb = plan_budget.get(rid16)
            if not rb:
                continue  # 枠外(見送り/対象外) → 買わない
            # プロ条件(週10R∧¥100k)充足: 枠対象は必ず買う(§0 hard安全ガードは維持)。
            # ただしクリーン帯外は DEMOTE_BUDGET へ予算降格 (参戦規律の二段化, 2026-07-25)
            out.append(compute_race_bets(r, live_dir=live_dir, max_age_min=args.max_age_min,
                                         budget=rb, force_floor=True,
                                         demote_budget=DEMOTE_BUDGET))
        else:
            out.append(compute_race_bets(r, live_dir=live_dir, max_age_min=args.max_age_min,
                                         budget=args.budget))
    n_bet = sum(1 for e in out if e["bets"]); tot = sum(b["購入額"] for e in out for b in e["bets"])
    shapes = {}
    for e in out: shapes[e["race_nature"]] = shapes.get(e["race_nature"], 0) + 1
    print(f"[compute_bets] races={len(out)} 買い={n_bet} 見送り={len(out)-n_bet} 総額¥{tot:,}  形:{shapes}")
    print("=" * 92)
    for e in out:
        if e["bets"]:
            s = sum(b['購入額'] for b in e['bets'])
            print(f"\n● {e['race_label']} [{e['race_nature']}] {len(e['bets'])}点 ¥{s:,}  {e['race_reason'].split('。',1)[-1]}")
            for b in e["bets"]:
                print(f"   {b['馬券種']:3s} {b['買い目']:8s} ¥{b['購入額']:>5,}  {b['理由']}")
        else:
            print(f"\n— {e['race_label']} [見送り] {e['race_reason']}")
        if e.get("hosei_marks"):
            print("   " + fmt_hosei(e["hosei_marks"]))

    if args.apply:
        m = None
        import re as _re
        mm = _re.search(r"(\d{8})", Path(args.bundle).name)
        if not mm:
            print("[ERROR] bundle 名から日付 (YYYYMMDD) を特定できず --apply 中止", file=sys.stderr)
            return 1
        apply_to_bets_json(mm.group(1), out)
        print("※ 書込後は validate_cowork_bets.py --apply で見送り/内容ガードを必ず通すこと")
    else:
        print("\n(dry: 書込なし。--apply で reports/cowork_output/{date}_bets.json へ反映)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
