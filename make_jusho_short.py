# -*- coding: utf-8 -*-
"""
make_jusho_short.py — 重賞予想を VOICEVOX で喋らせる YouTube Shorts 用素材を生成する。

方針（再配信リスク回避のため固定）:
  - 喋るのは「印 + 理由」だけ。オッズ・払戻・調教生タイム等の JRA-VAN 由来数値は一切喋らない/出さない。
  - 画面下の免責テロップ文言を sidecar に出力（動画工程で焼き込む）。
  - 素材は reports/cowork_output/{date}_bets.json の grade_scope(重賞 markdown) と bets(印) のみ。

出力（reports/shorts/{date}/{race_id}.*）:
  - script.txt   : 人間確認用の台本（ナレーション + 字幕 + 免責）
  - narration.wav: VOICEVOX 合成音声（エンジン起動時のみ。未起動ならスキップして警告）
  - meta.json    : 動画工程(ffmpeg)が読む構造化データ（行ごとの字幕・印・免責）

使い方:
  python make_jusho_short.py                 # 最新 bets.json の重賞すべて
  python make_jusho_short.py 20260614        # 日付指定
  python make_jusho_short.py --speaker 3     # VOICEVOX 話者(既定 3=ずんだもん ノーマル)
  python make_jusho_short.py --no-tts        # 台本/meta だけ（音声を作らない）
"""
import argparse
import glob
import json
import os
import re
import sys
import urllib.parse
import urllib.request

try:  # Windows コンソール(cp932)でも日本語/絵文字を print できるように
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

VOICEVOX = "http://127.0.0.1:50021"
OUT_ROOT = "reports/shorts"

# advisor.tag → 表示印（喋りでは「本命/対抗…」と読む）
TAG2MARK = {"本命": "◎", "対抗": "〇", "単穴": "▲", "穴": "★", "連下": "△", "△": "△"}
MARK2YOMI = {"◎": "本命", "〇": "対抗", "▲": "単穴", "★": "穴", "△": "連下"}

DISCLAIMER = "馬券の購入は自己責任で。本動画はAIによる予想であり的中・利益を保証するものではありません。"


def load_bets(date: str | None):
    files = sorted(glob.glob("reports/cowork_output/*_bets.json"), key=os.path.getmtime)
    if not files:
        sys.exit("reports/cowork_output/*_bets.json が無い")
    if date:
        hit = [f for f in files if date in os.path.basename(f)]
        if not hit:
            sys.exit(f"{date} の bets.json が無い")
        path = hit[-1]
    else:
        path = files[-1]
    d = json.load(open(path, encoding="utf-8"))
    date = re.search(r"(\d{8})", os.path.basename(path)).group(1)
    return d, date, path


def section(md: str, header_emoji_kw: str) -> str:
    """## 見出し ... の本文を1セクション抜き出す。"""
    blocks = re.split(r"\n##\s+", "\n" + md)
    for b in blocks:
        head = b.splitlines()[0] if b.splitlines() else ""
        if header_emoji_kw in head:
            return "\n".join(b.splitlines()[1:]).strip()
    return ""


def clean_for_speech(text: str) -> str:
    """markdown 記号・印記号を喋れる日本語に正規化。"""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)      # **強調** 除去
    text = re.sub(r"[-•*]\s+", "", text)               # 箇条書き記号
    text = text.replace("——", "、").replace("—", "、")
    for m, yomi in MARK2YOMI.items():
        text = text.replace(m, yomi + "、")
    text = re.sub(r"[#>`]", "", text)
    text = re.sub(r"\s+", "", text)                    # 余分な空白
    return text.strip()


def first_sentences(text: str, n: int) -> str:
    parts = re.split(r"(?<=[。！？])", text)
    parts = [p for p in parts if p.strip()]
    return "".join(parts[:n]).strip()


def parse_marks_from_md(md: str) -> list[dict]:
    """grade_scope の markdown から 印・馬名・理由を抽出（重賞はこれを正とする）。"""
    marks = []
    hon = section(md, "本命")
    m = re.search(r"\*\*◎\s*(\S+?)\*\*\s*[—–\-]*\s*(.+)", hon, re.S)
    if m:
        marks.append({"mark": "◎", "horse_name": m.group(1).strip(),
                      "reason": m.group(2).strip()})
    ana = section(md, "穴")
    # 穴候補の冒頭に太字で挙がる馬を最大2頭（否定文脈の断然人気は末尾なので前2件で拾える）
    seen = {marks[0]["horse_name"]} if marks else set()
    for nm in re.findall(r"\*\*(.+?)\*\*", ana):
        nm = re.sub(r"[◎〇◯▲△★\s]", "", nm).strip()
        if nm and nm not in seen:
            marks.append({"mark": "★", "horse_name": nm, "reason": ana})
            seen.add(nm)
        if len(marks) >= 3:
            break
    return marks


def build_script(race_label: str, marks: list[dict], umaban: dict, md: str):
    """ナレーション行(喋る) と 字幕行(画面) を返す。marks は grade_scope 由来。"""
    honmei = next((m for m in marks if m["mark"] == "◎"), marks[0] if marks else None)
    others = [m for m in marks if m is not honmei][:2]
    tenkai_md = section(md, "展開")

    def num(name):  # advisor から馬番を補完（無ければ空）
        n = umaban.get(name)
        return f"{n}番、" if n else ""

    narration, captions = [], []

    narration.append(f"今週の重賞、{race_label}。AIの本命予想を発表します。")
    captions.append(f"{race_label} AI予想")

    if honmei:
        reason = clean_for_speech(first_sentences(honmei["reason"], 2))
        narration.append(f"本命は、{num(honmei['horse_name'])}{honmei['horse_name']}。{reason}")
        n = umaban.get(honmei["horse_name"])
        captions.append(f"◎{n if n else ''} {honmei['horse_name']}")

    for m in others:
        narration.append(f"穴で狙うのは、{num(m['horse_name'])}{m['horse_name']}。")
        n = umaban.get(m["horse_name"])
        captions.append(f"★{n if n else ''} {m['horse_name']}")

    if tenkai_md:
        narration.append(clean_for_speech(first_sentences(tenkai_md, 1)))
        captions.append("展開予想")

    narration.append("印の詳細は概要欄をチェック。結果は日曜の夜に、当たりも外れも全部公開します。")
    captions.append("結果は日曜夜に全公開")

    return honmei, narration, captions


def marks_for_race(bets_list: list, race_id: str) -> list[dict]:
    r = next((b for b in bets_list if b.get("race_id") == race_id), None)
    if not r:
        return []
    out = []
    for a in r.get("advisor", []):
        mark = TAG2MARK.get(a.get("tag", ""), "")
        if not mark:
            continue
        out.append({"mark": mark, "umaban": a.get("umaban"),
                    "horse_name": a.get("horse_name", ""), "comment": a.get("comment", "")})
    # ◎〇▲★△ の順に
    order = {"◎": 0, "〇": 1, "▲": 2, "★": 3, "△": 4}
    out.sort(key=lambda m: order.get(m["mark"], 9))
    return out


# ---------- VOICEVOX ----------
def vv_alive() -> bool:
    try:
        urllib.request.urlopen(f"{VOICEVOX}/version", timeout=3).read()
        return True
    except Exception:
        return False


def vv_synth_lines(lines: list[str], speaker: int, out_wav: str):
    """各行を合成して連結（行間に無音を挟む）。stdlib only / wav 連結。"""
    import wave
    import struct
    segs = []
    params = None
    for line in lines:
        q = urllib.parse.urlencode({"text": line, "speaker": speaker})
        req = urllib.request.Request(f"{VOICEVOX}/audio_query?{q}", method="POST")
        query = json.loads(urllib.request.urlopen(req, timeout=30).read())
        query["speedScale"] = 1.05
        body = json.dumps(query).encode()
        req2 = urllib.request.Request(f"{VOICEVOX}/synthesis?speaker={speaker}",
                                      data=body, method="POST",
                                      headers={"Content-Type": "application/json"})
        wav = urllib.request.urlopen(req2, timeout=60).read()
        segs.append(wav)
    # wav バイト列を結合（同一フォーマット前提でフレームのみ連結）
    frames = b""
    for i, w in enumerate(segs):
        import io
        wf = wave.open(io.BytesIO(w), "rb")
        if params is None:
            params = wf.getparams()
            silence = b"\x00" * (wf.getframerate() * wf.getsampwidth() * wf.getnchannels() // 3)  # 0.33s
        frames += wf.readframes(wf.getnframes())
        if i != len(segs) - 1:
            frames += silence
        wf.close()
    out = wave.open(out_wav, "wb")
    out.setparams(params)
    out.writeframes(frames)
    out.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date", nargs="?", default=None)
    ap.add_argument("--speaker", type=int, default=3)
    ap.add_argument("--no-tts", action="store_true")
    args = ap.parse_args()

    data, date, path = load_bets(args.date)
    gs = data.get("grade_scope") or []
    if not gs:
        sys.exit(f"{date}: grade_scope（重賞）が無い。重賞開催週の bets.json を指定して。")
    bets_list = data.get("bets") or []

    outdir = os.path.join(OUT_ROOT, date)
    os.makedirs(outdir, exist_ok=True)
    tts_ok = (not args.no_tts) and vv_alive()
    if not args.no_tts and not tts_ok:
        print("⚠ VOICEVOX 未起動 (127.0.0.1:50021)。台本/meta のみ生成。"
              "VOICEVOX アプリを起動してから再実行すると narration.wav も出る。")

    print(f"src: {path}  重賞 {len(gs)} 件 → {outdir}")
    for g in gs:
        rid = g["race_id"]
        label = g.get("race_label", rid)
        md = g.get("markdown", "")
        marks = parse_marks_from_md(md)
        # 馬番は同レースの advisor から名前一致で補完（あれば）
        umaban = {a.get("horse_name", ""): a.get("umaban")
                  for a in next((b.get("advisor", []) for b in bets_list
                                 if b.get("race_id") == rid), [])}
        honmei, narration, captions = build_script(label, marks, umaban, md)

        base = os.path.join(outdir, rid)
        # 台本
        with open(base + "_script.txt", "w", encoding="utf-8") as f:
            f.write(f"# {label}\n\n")
            for i, (n, c) in enumerate(zip(narration, captions), 1):
                f.write(f"[{i}] 字幕: {c}\n    🔊 {n}\n\n")
            f.write(f"画面下テロップ(常時): {DISCLAIMER}\n")
        # meta
        meta = {"race_id": rid, "race_label": label, "date": date,
                "lines": [{"caption": c, "narration": n} for n, c in zip(narration, captions)],
                "disclaimer": DISCLAIMER, "speaker": args.speaker}
        json.dump(meta, open(base + "_meta.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        # 音声
        if tts_ok:
            try:
                vv_synth_lines(narration, args.speaker, base + "_narration.wav")
                print(f"  ✓ {label}: script + meta + wav")
            except Exception as e:
                print(f"  ! {label}: TTS 失敗 {e}（script/meta は出力済み）")
        else:
            print(f"  ✓ {label}: script + meta（wav はスキップ）")

    print("done.")


if __name__ == "__main__":
    main()
