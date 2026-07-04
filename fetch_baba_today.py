# -*- coding: utf-8 -*-
"""
fetch_baba_today.py — JRA 馬場情報(クッション値/含水率)を当日取得し『今日の馬場バイアス』を生成。
================================================================================
土日 8:30 起動想定（クッション値は8:20測定なので8:00だと取りこぼす）。
データ源（ブラウザ不要・直GET）:
  https://www.jra.go.jp/keiba/baba/_data_cushion.html  (場別 クッション値, 最新が先頭)
  https://www.jra.go.jp/keiba/baba/_data_moist.html    (場別 含水率 芝GP/芝4C/ダGP/ダ4C)
  https://www.jra.go.jp/keiba/baba/index.html          (kaisai_tab で 当日3場の順=rcA/rcB/rcC)
当日値 → data/baba_bias.json(10年実測) で『前残り度・枠バイアス』に変換。
出力: data/baba_today.json （build_site.py がサイトパネルに載せる）
実行: PYTHONUTF8=1 ./venv311/Scripts/python.exe fetch_baba_today.py
"""
from __future__ import annotations
import datetime as dt
import json, re, sys, time
from pathlib import Path
import requests

BASE = Path(__file__).parent
BIAS_TABLE = BASE / "data/baba_bias.json"
OUT = BASE / "data/baba_today.json"
JRA = "https://www.jra.go.jp/keiba/baba"
H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Referer": JRA + "/index.html"}
RC = ["rcA", "rcB", "rcC", "rcD"]
DATE_TIME = r"(\d+月\d+日（[^）]+）\s*\d+時\d+分)"


def get(url, tries=3, wait=20):
    """JRA サイトの一時不調に備え 3 回リトライ (確実に取る)。"""
    for k in range(tries):
        try:
            r = requests.get(url, headers=H, timeout=20)
            r.raise_for_status()
            return r.content.decode("shift_jis", "replace")
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(wait)


def measured_iso(label, today):
    """「7月3日（金曜）10時00分」→ ISO 日付。年は today 基準 (年跨ぎ補正付き)。"""
    if not label:
        return None
    m = re.search(r"(\d+)月(\d+)日", label)
    if not m:
        return None
    mo_, d_ = int(m.group(1)), int(m.group(2))
    y = today.year
    # 1月に12月測定を見た場合は前年
    if today.month == 1 and mo_ == 12:
        y -= 1
    try:
        return dt.date(y, mo_, d_).isoformat()
    except ValueError:
        return None


def firmness(cu):
    if cu is None: return None
    return ("硬め" if cu >= 12 else "やや硬め" if cu >= 10 else "標準"
            if cu >= 8 else "やや軟らかめ" if cu >= 7 else "軟らかめ")


def active_venues():
    """kaisai_tab の順で当日開催場 = rcA, rcB, rcC, ..."""
    html = get(JRA + "/index.html")
    i = html.find("kaisai_tab")
    seg = html[i:i + 800] if i >= 0 else html
    seg = seg.split("data-current-course")[0]  # タブ部分だけ
    return re.findall(r"(札幌|函館|福島|新潟|東京|中山|中京|京都|阪神|小倉)競馬場", seg)


def _block(html, rc):
    i = html.find(f'id="{rc}"')
    if i < 0: return ""
    j = min([html.find(f'id="{r}"', i + 1) for r in RC if html.find(f'id="{r}"', i + 1) > 0] or [len(html)])
    return re.sub(r"<[^>]+>", " ", html[i:j])


def parse_cushion():
    html = get(JRA + "/_data_cushion.html"); out = {}
    for rc in RC:
        b = _block(html, rc)
        m = re.search(DATE_TIME + r"\s*(\d+\.\d)", b)
        if m: out[rc] = {"time": re.sub(r"\s+", "", m.group(1)), "cushion": float(m.group(2))}
    return out


def parse_moist():
    html = get(JRA + "/_data_moist.html"); out = {}
    for rc in RC:
        b = _block(html, rc)
        m = re.search(DATE_TIME + r"\s*(\d+\.\d)\s+(\d+\.\d)\s+(\d+\.\d)\s+(\d+\.\d)", b)
        if m:
            out[rc] = {"time": re.sub(r"\s+", "", m.group(1)),
                       "shiba_gp": float(m.group(2)), "shiba_4c": float(m.group(3)),
                       "dirt_gp": float(m.group(4)), "dirt_4c": float(m.group(5))}
    return out


def pick_band(bands, v):
    for b in bands:
        lo = b["lo"] if b["lo"] is not None else -1e9
        hi = b["hi"] if b["hi"] is not None else 1e9
        if lo <= v < hi: return b
    return None


def bias(tbl, val, surf, label):
    band = pick_band(tbl["bands"], val) if tbl else None
    if not band: return None
    wk = band["waku_out_minus_in"]
    wk_txt = "外枠有利" if wk >= 1.0 else ("内枠やや有利" if wk <= -1.0 else "枠ほぼ均等")
    line = (f"{surf}[{band['label']}/{label}] 前残り度{band['front_edge']:+.1f}pp"
            f"（逃{band['nige_win']}%・先{band['senko_win']}%）・{wk_txt}(外-内{wk:+.1f}pp)")
    return {"band": band["label"], "front_edge": band["front_edge"], "nige_win": band["nige_win"],
            "senko_win": band["senko_win"], "waku_out_minus_in": wk, "line": line}


def main():
    try:
        BT = json.loads(BIAS_TABLE.read_text(encoding="utf-8"))
    except Exception:
        sys.exit("data/baba_bias.json が無い（build_baba_bias.py を先に）")
    today = dt.date.today().isoformat()
    try:
        venues = active_venues(); cu = parse_cushion(); mo = parse_moist()
    except Exception as e:
        OUT.write_text(json.dumps({"date": today, "venues": [], "error": str(e)},
                                  ensure_ascii=False), encoding="utf-8")
        sys.exit(f"取得失敗(非開催日?): {e}")
    if not venues:
        OUT.write_text(json.dumps({"date": today, "venues": []}, ensure_ascii=False),
                       encoding="utf-8")
        print("当日開催なし"); return

    rows = []
    for idx, vname in enumerate(venues):
        rc = RC[idx]
        c = cu.get(rc, {}); m = mo.get(rc, {})
        cval = c.get("cushion"); sgp = m.get("shiba_gp"); dgp = m.get("dirt_gp")
        shiba_bias = bias(BT.get("芝_cushion"), cval, "芝", f"ｸｯｼｮﾝ{cval}") if cval is not None else \
                     (bias(BT.get("芝_moisture"), sgp, "芝", f"含水{sgp}%") if sgp is not None else None)
        dirt_bias = bias(BT.get("ダ_moisture"), dgp, "ダ", f"含水{dgp}%") if dgp is not None else None
        rows.append({
            "venue": vname, "rc": rc,
            "cushion": cval, "cushion_firmness": firmness(cval), "cushion_time": c.get("time"),
            "shiba_gp": sgp, "shiba_4c": m.get("shiba_4c"), "dirt_gp": dgp, "dirt_4c": m.get("dirt_4c"),
            "moist_time": m.get("time"),
            "shiba_bias": shiba_bias, "dirt_bias": dirt_bias,
        })
    meas = next((r["cushion_time"] or r["moist_time"] for r in rows if r.get("cushion_time") or r.get("moist_time")), None)
    m_iso = measured_iso(meas, dt.date.today())
    is_today = (m_iso == today)
    # date: 取得実行日 (ISO)。site/js/baba.js が「閲覧日と一致する時だけ今日のとして表示」
    # の鮮度ゲートに使う (stale 表示バグ対策 2026-07-04)。
    # measured_date/is_measured_today: JRA は当日測定を race 当日 8:20 頃に掲載するため、
    # 早朝取得だと前日値が最新。前日値の時は baba.js が「前日測定」と正直表示し、
    # baba_daily.ps1 の午前ポーリングが当日値の掲載を待って取り直す (2026-07-04)。
    payload = {"date": today,
               "fetched_at": dt.datetime.now().isoformat(timespec="seconds"),
               "measured_label": meas, "measured_date": m_iso,
               "is_measured_today": is_today, "venues": rows}
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[saved] {OUT}  ({len(rows)}場)  measured={m_iso} today={is_today}")
    for r in rows:
        print(f"\n■ {r['venue']}（クッション{r['cushion']}={r['cushion_firmness']} / 含水芝{r['shiba_gp']}% ダ{r['dirt_gp']}%）")
        if r["shiba_bias"]: print("  ", r["shiba_bias"]["line"])
        if r["dirt_bias"]:  print("  ", r["dirt_bias"]["line"])


if __name__ == "__main__":
    main()
