#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_vtuber_script.py
=========================
PyCaLiAI の週次マーク JSON を「ずんだもん × 四国めたん」掛け合い台本に変換する。

入力 (どちらも UTF-8):
  - reports/cowork_input/{date}_bundle.json   … モデルの印・確率・妙味 (必須)
  - reports/cowork_output/{date}_bets.json    … Cowork の解説コメント・重賞コラム (任意・あれば優先採用)

出力:
  - vtuber_scripts/{date}/{race_id}_daihon.txt   … 「話者：セリフ」形式 (ゆっくりMovieMaker4 貼り付け用)
  - vtuber_scripts/{date}/{race_id}_script.json  … {meta, lines:[{speaker,text,telop}]} (自動化用 / テロップ付き)

用途: YouTube Shorts (縦9:16・約45秒)。VOICEVOX(ずんだもん/四国めたん) + YMM4 に流し込む前提。

------------------------------------------------------------------------------
設計思想 — PyCaLiAI 既存の規約をそのまま継承する:
  * 確率は「キャリブレーション済み」。だから勝率% を誠実に提示してよい (= このチャンネル最大の差別化)。
    ただし「絶対当たる/必ず儲かる」は禁止。EV・妙味は推定値であって保証ではない。見送りも正解。
  * セリフはモデルの実フィールドだけを言い換える。数値は捏造しない。
  * Cowork の advisor[].comment / grade_scope[].markdown があれば、それを優先して喋らせる
    (既に競馬予想士口調で書かれた "ほぼ完成原稿" なので)。
------------------------------------------------------------------------------

使い方:
  python generate_vtuber_script.py --date 20260614 --list          # 候補レース一覧 (面白さ順)
  python generate_vtuber_script.py --date 20260614 --mode auto     # おすすめ1本を生成
  python generate_vtuber_script.py --date 20260614 --race-id 2026061409030411
  python generate_vtuber_script.py --bundle path/to/_bundle.json --bets path/to/_bets.json --mode grade
"""

import argparse
import glob
import json
import os
import sys
import unicodedata

# --- 印の定数 (bundle 内の実グリフ) ---
MK_HON = "◎"   # 本命 (ai_rank 1)
MK_TAI = "〇"   # 対抗 (ai_rank 2)  ※U+3007
MK_TAN = "▲"   # 単穴 (ai_rank 3)
MK_REN = "△"   # 連下 (ai_rank 4,5)

ZUNDA = "ずんだもん"   # 予想役 (語尾「〜なのだ」)。本命・妙味を語る
METAN = "四国めたん"   # 聞き役/懐疑役 (標準語)。ツッコミと締め

DISCLAIMER = "※キャリブレーション済みAIによるエンタメ予想｜馬券は自己責任・20歳以上"


# ============================================================================
# 読み込み
# ============================================================================
def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resolve_paths(args):
    """--date / --bundle / --bets から実パスを決める。"""
    bundle = args.bundle
    bets = args.bets
    rdir = args.reports_dir
    if not bundle and args.date:
        bundle = os.path.join(rdir, "cowork_input", f"{args.date}_bundle.json")
    if not bets and args.date:
        cand = os.path.join(rdir, "cowork_output", f"{args.date}_bets.json")
        bets = cand if os.path.exists(cand) else None
    return bundle, bets


def extract_races(bundle):
    """bundle の形 ({races:[...]} でも race_id->race の dict でも) を吸収して races のリストにする。"""
    if isinstance(bundle, dict) and "races" in bundle:
        return bundle["races"]
    if isinstance(bundle, list):
        return bundle
    if isinstance(bundle, dict):
        # race_id -> race の辞書形
        vals = list(bundle.values())
        if vals and isinstance(vals[0], dict) and "race_meta" in vals[0]:
            return vals
    raise ValueError("bundle の構造を認識できませんでした (races 配列が見つかりません)")


# ============================================================================
# 表示・言い換えヘルパ
# ============================================================================
def nfkc(s):
    """全角クラス表記 (Ｇ１/ＪＧ１) を半角 (G1/JG1) に正規化。"""
    return unicodedata.normalize("NFKC", s or "")


def grade_of(meta):
    """race_meta.class からグレード文字列を取り出す (G1/G2/G3/OP/.. or '')。"""
    cls = nfkc(meta.get("class", ""))
    for g in ("JG1", "JG2", "JG3", "G1", "G2", "G3"):
        if g in cls:
            return g
    if "OP" in cls or "(L)" in cls or "L" in cls and "ｵｰﾌﾟﾝ" in cls:
        return "OP"
    return ""


def race_title(meta):
    """『阪神 芝2200 G1 宝塚記念』『東京 ダート1400 未勝利』のような見出し。"""
    place = meta.get("place", "")
    course = meta.get("course", "")
    name = (meta.get("race_name") or "").strip()
    cls = nfkc(meta.get("class", "")).strip()
    tail = name if name else cls
    return " ".join(x for x in (place, course, tail) if x)


def win_pct(p):
    return round(p * 100) if isinstance(p, (int, float)) else None


def find_marked(horses):
    """印ごとの馬を返す。△ は2頭なので list。"""
    d = {MK_HON: None, MK_TAI: None, MK_TAN: None, MK_REN: []}
    for h in horses:
        m = h.get("mark", "")
        if m == MK_HON:
            d[MK_HON] = h
        elif m == MK_TAI:
            d[MK_TAI] = h
        elif m == MK_TAN:
            d[MK_TAN] = h
        elif m == MK_REN:
            d[MK_REN].append(h)
    return d


def confidence_phrase(rc, bj):
    """(短ラベル, 喋り用フレーズ) を返す。hardness を主、race_confidence を従に。"""
    hardness = (bj or {}).get("hardness", "")
    dom = (rc or {}).get("top1_dominance") or 0.0
    chaos = (rc or {}).get("field_chaos_score") or 0.0
    if hardness == "固い":
        return "堅", "堅く収まりやすい一戦"
    if hardness == "荒れ":
        return "波乱", "波乱含みで、ヒモは広げたい一戦"
    # 標準 or 未設定 → race_confidence で補足
    if dom >= 0.12 and chaos < 0.7:
        return "中心", "本命中心で考えやすい一戦"
    if chaos >= 0.85:
        return "混戦", "混戦模様で点数を絞りにくい一戦"
    return "標準", "標準的な難易度の一戦"


def ev_str(x):
    return f"{x:.2f}" if isinstance(x, (int, float)) else "—"


def pick_danger(horses):
    """『人気のわりにAI評価が低い』= 罠候補。ai_vs_market=='over' のうち最も人気 (単勝オッズ最小)。"""
    overs = [h for h in horses
             if h.get("ai_vs_market") == "over"
             and isinstance(h.get("tansho_odds"), (int, float))]
    if not overs:
        return None
    cand = min(overs, key=lambda h: h["tansho_odds"])
    # 人気どころ (単勝20倍未満) でないと "罠" として弱いので絞る
    return cand if cand["tansho_odds"] < 20 else None


# ============================================================================
# 台本生成
# ============================================================================
def build_script(race, bet=None, grade_scope=None):
    """1レース分の台本 (lines) と meta を返す。"""
    meta = race.get("race_meta", {})
    horses = race.get("horses", [])
    rc = race.get("race_confidence", {})
    bj = race.get("buy_judgment", {})
    marked = find_marked(horses)
    hon = marked[MK_HON]
    tai = marked[MK_TAI]

    advisor_by_umaban = {}
    if bet:
        for a in bet.get("advisor", []) or []:
            advisor_by_umaban[a.get("umaban")] = a

    title = race_title(meta)
    grade = grade_of(meta)
    conf_label, conf_phrase = confidence_phrase(rc, bj)
    lines = []

    def add(speaker, text, telop=""):
        lines.append({"speaker": speaker, "text": text, "telop": telop})

    # ---- 1. 掴み (本命提示) ----
    if hon:
        add(ZUNDA,
            f"今日は{title}！PyCaLiAIの本命は…{MK_HON}{hon['umaban']}番{hon['horse_name']}なのだ！",
            telop=f"{title}｜本命 {MK_HON}{hon['horse_name']}")
    else:
        add(ZUNDA, f"今日は{title}を解説するのだ！", telop=title)

    # ---- 2. 合いの手 (難度に応じて) ----
    if conf_label in ("堅", "中心"):
        add(METAN, "お、今日は自信ありそうね。根拠は？")
    elif conf_label in ("波乱", "混戦"):
        add(METAN, "ふーん…でも今日はちょっと荒れそうな匂いもするけど？")
    else:
        add(METAN, "なるほど。どのくらい信頼できるの？")

    # ---- 3. 本命の信頼度 (キャリブレーション済み勝率 + クラス実績) ----
    if hon:
        wp = win_pct(hon.get("p_win"))
        cp = meta.get("class_prior") or {}
        top3 = cp.get("hon_top3_pct")
        cls_disp = nfkc(meta.get("class", ""))
        if wp is not None and isinstance(top3, (int, float)):
            txt = (f"{hon['horse_name']}の勝率はキャリブレーション済みでおよそ{wp}％、"
                   f"このクラスの本命は約{round(top3)}％で馬券圏内に来てるのだ。")
            tp = f"{MK_HON}勝率≈{wp}%｜{cls_disp} {MK_HON}複勝圏 {round(top3)}%"
        elif wp is not None:
            txt = (f"{hon['horse_name']}の勝率はキャリブレーション済みでおよそ{wp}％。"
                   f"額面どおりに信頼していい数字なのだ。")
            tp = f"{MK_HON}勝率≈{wp}%（キャリブレーション済）"
        elif isinstance(top3, (int, float)):
            txt = f"このクラスの本命は約{round(top3)}％で馬券圏内に来てるのだ。"
            tp = f"{cls_disp} {MK_HON}複勝圏 {round(top3)}%"
        else:
            txt = tp = None
        if txt:
            add(ZUNDA, txt, telop=tp)

        # Cowork の本命コメントがあれば、予想士口調の "根拠" として喋らせる
        adv = advisor_by_umaban.get(hon.get("umaban"))
        if adv and adv.get("comment"):
            add(ZUNDA, adv["comment"].strip(),
                telop=(f"地力{adv.get('grade','')}" + (f"／{adv.get('tag')}" if adv.get("tag") else "")).strip("／"))

    # ---- 4. ツッコミ ----
    if conf_label in ("波乱", "混戦"):
        add(METAN, "やっぱり相手は広く見たいわね。妙味のある馬はいる？")
    else:
        add(METAN, "堅そうね。じゃあ妙味のある馬はいるの？")

    # ---- 5. 妙味馬 (このチャンネルの核) ----
    vhs = (bj.get("value_horses") or [])
    hon_umaban = hon.get("umaban") if hon else None
    if bj.get("has_value") and vhs:
        non_hon = [v for v in vhs if v.get("umaban") != hon_umaban]
        if non_hon:  # 本命とは別の妙味馬 (穴っぽい妙味) を優先紹介
            v = non_hon[0]
            ev, ug = v.get("ev_tan"), v.get("umami_grade", "")
            add(ZUNDA,
                f"妙味は{v['umaban']}番{v['horse_name']}！市場より高く評価してて、"
                f"単勝の期待値はおよそ{ev_str(ev)}倍"
                + (f"、妙味グレードは{ug}" if ug else "") + "なのだ。",
                telop=f"妙味 {v['horse_name']}｜単勝EV≈{ev_str(ev)}" + (f"／妙味{ug}" if ug else ""))
            add(METAN, "市場が見落としてるってこと？それは美味しいわね。")
        else:  # 妙味馬が本命だけ
            v = vhs[0]
            ev, ug = v.get("ev_tan"), v.get("umami_grade", "")
            add(ZUNDA,
                f"しかも本命自身が妙味なのだ。単勝の期待値はおよそ{ev_str(ev)}倍"
                + (f"、妙味グレード{ug}" if ug else "") + "で、人気でも割安なのだ。",
                telop=f"本命が妙味｜単勝EV≈{ev_str(ev)}" + (f"／妙味{ug}" if ug else ""))
            add(METAN, "本命が割安なら、迷わず狙えるわね。")
    else:
        add(ZUNDA, "今日は明確な妙味馬はなし。無理せず上位印で組み立てるのだ。",
            telop="妙味馬なし＝無理は禁物")

    # ---- 6. 罠 (人気のわりにAI低評価) ----
    danger = pick_danger(horses)
    # Cowork が 罠 タグを付けていればそちらを優先
    trap_adv = next((a for a in advisor_by_umaban.values() if a.get("tag") == "罠"), None)
    if trap_adv:
        add(ZUNDA,
            f"逆に注意は{trap_adv['umaban']}番{trap_adv['horse_name']}。"
            f"人気だけどAIは評価控えめ＝罠の可能性なのだ。",
            telop=f"⚠罠候補 {trap_adv['horse_name']}（人気＞AI評価）")
    elif danger:
        add(ZUNDA,
            f"逆に注意は{danger['umaban']}番{danger['horse_name']}。"
            f"単勝{danger['tansho_odds']}倍と人気だけど、AIの評価は市場より低いのだ。",
            telop=f"⚠罠候補 {danger['horse_name']}（人気＞AI評価）")

    # ---- 7. まとめ (買い方の方向性だけ。断定的な指示はしない) ----
    headline = bj.get("headline") or ""
    hardness = bj.get("hardness") or conf_label
    summary = f"AIの判定は『{headline}』。{conf_phrase}" if headline else f"今日は{conf_phrase}"
    add(ZUNDA, summary + "。買うかどうかは自己責任、見送りも立派な選択なのだ。",
        telop=f"AI判定：{hardness}" + (f"／{headline}" if headline else ""))

    # ---- 8. 締め (記録公開・登録誘導) ----
    add(METAN, "当たりも外れも結果は全部公開。気になる人はチャンネル登録お願いね！")

    script_meta = {
        "race_id": race.get("race_id"),
        "title": title,
        "grade": grade,
        "place": meta.get("place"),
        "course": meta.get("course"),
        "class": nfkc(meta.get("class", "")),
        "field_size": meta.get("field_size"),
        "honmei": (f"{hon['umaban']}番{hon['horse_name']}" if hon else None),
        "hardness": bj.get("hardness"),
        "has_value": bj.get("has_value"),
        "disclaimer": DISCLAIMER,
        "has_grade_column": bool(grade_scope),
    }
    # 重賞でコラムがあれば、概要欄/ロング動画用に丸ごと添付
    if grade_scope and grade_scope.get("markdown"):
        script_meta["grade_column_markdown"] = grade_scope["markdown"]

    return script_meta, lines


# ============================================================================
# レース選定 (面白さスコア)
# ============================================================================
def interest_score(race):
    meta = race.get("race_meta", {})
    rc = race.get("race_confidence", {})
    bj = race.get("buy_judgment", {})
    s = 0.0
    if (meta.get("race_name") or "").strip():
        s += 40
    g = grade_of(meta)
    s += {"G1": 60, "JG1": 55, "G2": 40, "JG2": 38, "G3": 30, "JG3": 28, "OP": 15}.get(g, 0)
    if bj.get("has_value"):
        s += 30
    for v in (bj.get("value_horses") or []):
        ug = v.get("umami_grade", "")
        if ug in ("SS", "S"):
            s += 20
            break
        if ug == "A":
            s += 10
            break
    s += 15 * float(rc.get("top1_dominance") or 0.0)
    return s


def pick_race(races, mode):
    if mode == "grade":
        graded = [r for r in races if grade_of(r.get("race_meta", {})) or (r.get("race_meta", {}).get("race_name"))]
        pool = graded or races
        return max(pool, key=interest_score)
    if mode == "value":
        valued = [r for r in races if r.get("buy_judgment", {}).get("has_value")]
        pool = valued or races

        def vmax(r):
            evs = [v.get("ev_tan") or 0 for v in (r.get("buy_judgment", {}).get("value_horses") or [])]
            return max(evs) if evs else 0
        return max(pool, key=vmax)
    if mode == "solid":
        return max(races, key=lambda r: (r.get("race_confidence", {}).get("top1_dominance") or 0)
                   - (r.get("race_confidence", {}).get("field_chaos_score") or 0))
    if mode == "chaos":
        return max(races, key=lambda r: r.get("race_confidence", {}).get("field_chaos_score") or 0)
    # auto
    return max(races, key=interest_score)


def print_candidates(races, n=15):
    ranked = sorted(races, key=interest_score, reverse=True)
    print(f"\n候補レース (面白さ順 / 全{len(races)}R 中 上位{min(n,len(races))}):\n")
    print(f"{'race_id':<18} {'タイトル':<26} {'堅さ':<5} {'妙味':<4} {'最大EV':<6} 本命(勝率)")
    print("-" * 92)
    for r in ranked[:n]:
        meta = r.get("race_meta", {})
        bj = r.get("buy_judgment", {})
        marked = find_marked(r.get("horses", []))
        hon = marked[MK_HON]
        evs = [v.get("ev_tan") or 0 for v in (bj.get("value_horses") or [])]
        max_ev = max(evs) if evs else 0
        wp = win_pct(hon.get("p_win")) if hon else None
        hon_s = (f"{hon['horse_name']}({wp}%)" if hon else "-")
        print(f"{str(r.get('race_id','')):<18} {race_title(meta)[:26]:<26} "
              f"{(bj.get('hardness') or '-'):<5} {('有' if bj.get('has_value') else '無'):<4} "
              f"{max_ev:>5.2f}  {hon_s}")
    print()


# ============================================================================
# 出力
# ============================================================================
def write_outputs(out_dir, date, script_meta, lines):
    rid = script_meta.get("race_id") or "race"
    day_dir = os.path.join(out_dir, date or "unknown")
    os.makedirs(day_dir, exist_ok=True)

    txt_path = os.path.join(day_dir, f"{rid}_daihon.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"# {script_meta['title']}  ({rid})\n")
        f.write(f"# {DISCLAIMER}\n\n")
        for ln in lines:
            f.write(f"{ln['speaker']}：{ln['text']}\n")
            if ln["telop"]:
                f.write(f"    [テロップ] {ln['telop']}\n")
        f.write(f"\n# 常時表示テロップ: {DISCLAIMER}\n")

    json_path = os.path.join(day_dir, f"{rid}_script.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"meta": script_meta, "lines": lines}, f, ensure_ascii=False, indent=2)

    return txt_path, json_path


# ============================================================================
# main
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description="PyCaLiAI マークJSON → ずんだもん×四国めたん 台本 (YouTube Shorts用)")
    ap.add_argument("--date", help="YYYYMMDD (reports/ から自動でパス解決)")
    ap.add_argument("--bundle", help="bundle.json を直接指定")
    ap.add_argument("--bets", help="bets.json を直接指定 (任意)")
    ap.add_argument("--reports-dir", default="reports", help="reports ディレクトリ (default: reports)")
    ap.add_argument("--race-id", help="生成するレースの race_id (16桁)")
    ap.add_argument("--mode", default="auto", choices=["auto", "grade", "value", "solid", "chaos"],
                    help="race-id 未指定時の自動選定モード")
    ap.add_argument("--out", default="vtuber_scripts", help="出力ディレクトリ")
    ap.add_argument("--list", action="store_true", help="候補レース一覧を表示して終了")
    args = ap.parse_args()

    bundle_path, bets_path = resolve_paths(args)
    if not bundle_path or not os.path.exists(bundle_path):
        sys.exit(f"[ERROR] bundle が見つかりません: {bundle_path}\n  --date か --bundle を指定してください。")

    bundle = load_json(bundle_path)
    races = extract_races(bundle)
    date = (bundle.get("date") if isinstance(bundle, dict) else None) or (args.date or "")

    bets_by_id, grade_by_id = {}, {}
    if bets_path and os.path.exists(bets_path):
        bets = load_json(bets_path)
        for b in bets.get("bets", []) or []:
            bets_by_id[b.get("race_id")] = b
        for g in bets.get("grade_scope", []) or []:
            grade_by_id[g.get("race_id")] = g

    if args.list:
        print_candidates(races)
        return

    # レース選定
    if args.race_id:
        race = next((r for r in races if str(r.get("race_id")) == str(args.race_id)), None)
        if race is None:
            sys.exit(f"[ERROR] race_id {args.race_id} が bundle にありません。--list で確認を。")
    else:
        race = pick_race(races, args.mode)

    rid = race.get("race_id")
    script_meta, lines = build_script(race, bets_by_id.get(rid), grade_by_id.get(rid))
    txt_path, json_path = write_outputs(args.out, date, script_meta, lines)

    # 標準出力にも台本を表示
    print(f"\n=== {script_meta['title']}  ({rid}) ===")
    print(f"{DISCLAIMER}\n")
    for ln in lines:
        print(f"{ln['speaker']}：{ln['text']}")
        if ln["telop"]:
            print(f"    └[テロップ] {ln['telop']}")
    print(f"\n→ {txt_path}")
    print(f"→ {json_path}")
    if script_meta.get("has_grade_column"):
        print("  (重賞コラム markdown を script.json に同梱しました＝概要欄/ロング動画用)")


if __name__ == "__main__":
    main()
