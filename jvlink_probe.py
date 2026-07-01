# -*- coding: utf-8 -*-
"""JV-Link 疎通＋認証プローブ。JVInit が 0 なら SDK＋利用キー(TARGET設定)が生きてる。"""
import win32com.client as w
RC = {0:"OK", -1:"初期化/汎用エラー", -100:"param不正", -111:"registry読込err",
      -114:"sid不正", -201:"JVInit未実行", -203:"既にopen", -211:"ServiceKey未設定",
      -301:"認証エラー(契約/キー)", -302:"利用キー期限切れ", -504:"該当データ無"}
def m(rc): return RC.get(rc, "?")
jv = w.Dispatch("JVDTLab.JVLink")
for SID in ["UNKNOWN", "PyCaLiAI/1.0"]:
    rc = jv.JVInit(SID)
    print(f"JVInit('{SID}') -> {rc}  ({m(rc)})")
    if rc == 0:
        for ds in ["0B31", "0B30"]:
            try:
                rc2 = jv.JVRTOpen(ds, "0000000000")
                print(f"  JVRTOpen('{ds}','0000000000') -> {rc2}  ({m(rc2)})")
            except Exception as e:
                print(f"  JVRTOpen('{ds}') exc: {str(e)[:120]}")
        try: jv.JVClose()
        except Exception: pass
        break
print("done")
