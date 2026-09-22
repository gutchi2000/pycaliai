# -*- coding: utf-8 -*-
"""
jvlink_shadow_probe.py — 障害レースカードの過去日再取得 feasibility 確認
=====================================================================
production を一切変更せず、**shadow ディレクトリへのみ**書き出して
JV-Link 蓄積系 (JVOpen "RACE") から過去の障害レースカードを再取得できるかを
確認する。

★必ず 32-bit Python: py -3.12-32 analysis/mcond/p0_dnf_history_parity_audit/jvlink_shadow_probe.py

検査項目: race_id / 血統登録番号 / 芝ダ・障害区分 / 距離 / 騎手 / 調教師 /
          血統 / finish status / 取得時点 / 後知恵情報の有無

出力: analysis/mcond/p0_dnf_history_parity_audit/shadow/jvlink_probe_{date}.json
      (production の data/ 配下には一切書かない)
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHADOW = HERE / "shadow"
SHADOW.mkdir(exist_ok=True)

# 欠落初期 / 欠落期間中央 / 2026-09-06 付近
TARGET_DATES = ["20260307", "20260627", "20260906"]
# 各日の欠落障害レース (missing_set_manifest.json より)
TARGET_RIDS = {
    "20260307": "2026030706020304",
    "20260627": "2026062703020101",
    "20260906": "2026090606040201",
}

# True にすると JVOpen の rc/readcount/dlcount だけ見て JVRead をしない
# (大量ダウンロードを避けるための feasibility 確認モード)
SKIP_READ = "--read" not in sys.argv

RC = {0: "OK", -1: "初期化/汎用エラー", -100: "param不正", -111: "registry読込err",
      -114: "sid不正", -201: "JVInit未実行", -203: "既にopen",
      -211: "ServiceKey未設定", -301: "認証エラー(契約/キー)",
      -302: "利用キー期限切れ", -503: "ファイル無", -504: "該当データ無"}


def log(m):
    print(m, flush=True)


def probe():
    import win32com.client as w

    out = {"generated_at": datetime.now().isoformat(),
           "python": sys.version, "dates": {}}

    jv = w.Dispatch("JVDTLab.JVLink")
    rc = jv.JVInit("PyCaLiAI/1.0")
    out["JVInit"] = {"rc": rc, "meaning": RC.get(rc, "?")}
    log(f"JVInit -> {rc} ({RC.get(rc,'?')})")
    if rc != 0:
        out["verdict"] = "JVInit 失敗、取得不可"
        return out

    for d in TARGET_DATES:
        rec = {"target_rid": TARGET_RIDS[d], "specs": {}}
        # 蓄積系 RACE: 0B12=出走馬名表(カード), RACE=レース詳細+馬毎
        for spec in ["RACE", "0B12"]:
            fromtime = f"{d}000000"
            try:
                # option=4 = セットアップ(ダイアログ非表示)。
                # option=1 は進捗ダイアログでブロックしうるため使わない。
                log(f"  {d} {spec}: JVOpen 呼び出し中 ...")
                r = jv.JVOpen(spec, fromtime, 4)
                rcode = r[0] if isinstance(r, (tuple, list)) else r
                info = {"JVOpen_rc": rcode, "meaning": RC.get(rcode, "?")}
                if isinstance(r, (tuple, list)) and len(r) >= 3:
                    info["readcount"] = r[1]
                    info["dlcount"] = r[2]
                log(f"    JVOpen rc={rcode} readcount={info.get('readcount')} "
                    f"dlcount={info.get('dlcount')}")
                if rcode == 0 and not SKIP_READ:
                    ra, se, other, samples = 0, 0, 0, []
                    for _ in range(40000):
                        try:
                            rr = jv.JVRead()
                        except Exception as e:
                            info["read_exc"] = str(e)[:150]
                            break
                        code = rr[0] if isinstance(rr, (tuple, list)) else rr
                        buf = rr[1] if isinstance(rr, (tuple, list)) and len(rr) > 1 else ""
                        if code == 0:
                            break
                        if code < 0:
                            info["read_rc"] = code
                            break
                        if not isinstance(buf, str) or len(buf) < 3:
                            continue
                        head = buf[:2]
                        if head == "RA":
                            ra += 1
                            if len(samples) < 3:
                                samples.append(buf[:160])
                        elif head == "SE":
                            se += 1
                            if len([s for s in samples if s.startswith("SE")]) < 2:
                                samples.append(buf[:200])
                        else:
                            other += 1
                    info.update(RA=ra, SE=se, other=other, samples=samples)
                try:
                    jv.JVClose()
                except Exception:
                    pass
                rec["specs"][spec] = info
                log(f"  {d} {spec}: {info.get('JVOpen_rc')} "
                    f"RA={info.get('RA')} SE={info.get('SE')}")
            except Exception as e:
                rec["specs"][spec] = {"exception": str(e)[:200]}
                log(f"  {d} {spec}: EXC {str(e)[:120]}")
        out["dates"][d] = rec

    try:
        jv.JVClose()
    except Exception:
        pass
    return out


def main():
    if sys.maxsize > 2**32:
        log("!! 64-bit Python で実行されています。JV-Link は 32-bit COM のため")
        log("   py -3.12-32 で実行してください。")
        json.dump({"error": "must run under 32-bit python",
                   "python": sys.version},
                  open(SHADOW / "jvlink_probe_ERROR.json", "w",
                       encoding="utf-8"), ensure_ascii=False, indent=2)
        return
    try:
        out = probe()
    except Exception as e:
        out = {"error": str(e)[:400], "generated_at": datetime.now().isoformat()}
        log(f"EXC: {e}")
    p = SHADOW / "jvlink_probe_result.json"
    json.dump(out, open(p, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=str)
    log(f"\n保存: {p}")


if __name__ == "__main__":
    main()
