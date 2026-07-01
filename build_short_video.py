# -*- coding: utf-8 -*-
"""
build_short_video.py — make_jusho_short.py の meta.json から縦型(1080x1920)の
YouTube Shorts 動画を組む。VOICEVOX を行ごとに再合成して字幕と完全同期させる。

依存: Pillow, numpy, imageio-ffmpeg(同梱ffmpeg)。VOICEVOX 起動が必要。
立ち絵: assets/vtuber/base.png があれば白背景を抜いて合成。無ければプレースホルダ。

使い方:
  python build_short_video.py 20260614            # その日の重賞ぜんぶ
  python build_short_video.py reports/shorts/20260614/2026061409030411_meta.json
"""
import glob
import io
import json
import os
import subprocess
import sys
import urllib.parse
import urllib.request
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

VOICEVOX = "http://127.0.0.1:50021"
W, H = 1080, 1920
FONT = "C:/Windows/Fonts/YuGothB.ttc"          # Yu Gothic Bold
CHAR_PNG = "assets/vtuber/base.png"
BG_VIDEO = "assets/vtuber/pikari_idle_v1.mp4"

# ブランド配色
GOLD = (245, 185, 66)       # #f5b942
GOLD_LT = (255, 217, 122)   # #ffd97a
GOLD_DK = (45, 32, 2)       # #2d2002
BG = (19, 16, 8)            # #131008
PANEL = (28, 23, 12)        # 字幕パネル地
MINT = (45, 212, 168)       # #2dd4a8
WHITE = (245, 245, 245)
BAR = (10, 8, 5)

FF = None  # ffmpeg path (lazy)


def ff_exe():
    global FF
    if FF is None:
        import imageio_ffmpeg
        FF = imageio_ffmpeg.get_ffmpeg_exe()
    return FF


def font(sz):
    return ImageFont.truetype(FONT, sz, index=0)


# ---------------- VOICEVOX ----------------
def synth(text, speaker):
    q = urllib.parse.urlencode({"text": text, "speaker": speaker})
    req = urllib.request.Request(f"{VOICEVOX}/audio_query?{q}", method="POST")
    query = json.loads(urllib.request.urlopen(req, timeout=30).read())
    query["speedScale"] = 1.05
    body = json.dumps(query).encode()
    req2 = urllib.request.Request(f"{VOICEVOX}/synthesis?speaker={speaker}",
                                  data=body, method="POST",
                                  headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req2, timeout=60).read()


def build_audio(lines, speaker, out_wav, gap=0.35, tail=0.6):
    """行ごとに合成→連結。各行の表示秒数(=音声+gap)を返す。"""
    params = None
    frames = b""
    durs = []
    for i, ln in enumerate(lines):
        wav = synth(ln["narration"], speaker)
        wf = wave.open(io.BytesIO(wav), "rb")
        if params is None:
            params = wf.getparams()
            fr, sw, ch = wf.getframerate(), wf.getsampwidth(), wf.getnchannels()
            sil = b"\x00" * int(fr * sw * ch * gap)
            tail_b = b"\x00" * int(fr * sw * ch * tail)
        n = wf.getnframes()
        frames += wf.readframes(n)
        seg = n / wf.getframerate()
        last = (i == len(lines) - 1)
        frames += (tail_b if last else sil)
        durs.append(seg + (tail if last else gap))
        wf.close()
    out = wave.open(out_wav, "wb")
    out.setparams(params)
    out.writeframes(frames)
    out.close()
    return durs


# ---------------- 描画 ----------------
def wrap(draw, text, fnt, max_w):
    """日本語(空白なし)を幅で折返し。"""
    lines, cur = [], ""
    for ch in text:
        if ch == "\n":
            lines.append(cur); cur = ""; continue
        if draw.textlength(cur + ch, font=fnt) <= max_w:
            cur += ch
        else:
            lines.append(cur); cur = ch
    if cur:
        lines.append(cur)
    return lines


def draw_multiline(draw, xy, text, fnt, fill, max_w, lh, center_x=None):
    x, y = xy
    for ln in wrap(draw, text, fnt, max_w):
        if center_x is not None:
            w = draw.textlength(ln, font=fnt)
            draw.text((center_x - w / 2, y), ln, font=fnt, fill=fill)
        else:
            draw.text((x, y), ln, font=fnt, fill=fill)
        y += lh
    return y


def keyed_character(box_w, box_h):
    """白背景を抜いた立ち絵を box に収まる RGBA で返す。無ければ None。"""
    if not os.path.exists(CHAR_PNG):
        return None
    im = Image.open(CHAR_PNG).convert("RGBA")
    rgb = im.convert("RGB")
    seed = (255, 0, 255)
    for c in [(0, 0), (im.width - 1, 0), (0, im.height - 1), (im.width - 1, im.height - 1)]:
        ImageDraw.floodfill(rgb, c, seed, thresh=42)
    arr = np.array(rgb)
    mask = np.all(arr == seed, axis=-1)
    out = np.array(im)
    out[mask, 3] = 0
    im = Image.fromarray(out, "RGBA")
    # contain
    r = min(box_w / im.width, box_h / im.height)
    return im.resize((int(im.width * r), int(im.height * r)), Image.LANCZOS)


def race_short(label):
    """'阪神芝2200 G1 宝塚記念' → ('G1','宝塚記念')"""
    toks = label.split()
    grade = next((t for t in toks if t.upper().startswith("G") or "Ｇ" in t), "")
    name = toks[-1] if toks else label
    return grade, name


def render_slide(line, label, char_img, disclaimer):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img, "RGBA")

    # 立ち絵 (先に置いて字幕を前面に)
    if char_img is not None:
        x = W - char_img.width - 30
        y = H - char_img.height - 130
        img.paste(char_img, (x, y), char_img)
    else:
        bx, by, bw, bh = 545, 1015, 505, 760
        d.rounded_rectangle([bx, by, bx + bw, by + bh], radius=24,
                            fill=(13, 11, 6), outline=GOLD, width=3)
        f = font(74)
        t = "ぴかり"
        d.text((bx + bw / 2 - d.textlength(t, font=f) / 2, by + bh / 2 - 90),
               t, font=f, fill=GOLD)
        d.text((bx + bw / 2 - d.textlength("立ち絵 (準備中)", font=font(34)) / 2,
                by + bh / 2 + 10), "立ち絵 (準備中)", font=font(34), fill=GOLD_LT)
        # 簡易ニューラルネット
        import math
        cx, cy = bx + bw / 2, by + bh / 2 + 150
        pts = [(cx + 90 * math.cos(a), cy + 55 * math.sin(a))
               for a in [0, 1.04, 2.09, 3.14, 4.18, 5.23]]
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d.line([pts[i], pts[j]], fill=(120, 95, 20), width=1)
        for p in pts:
            d.ellipse([p[0] - 7, p[1] - 7, p[0] + 7, p[1] + 7], fill=GOLD)

    # ヘッダー
    d.text((60, 64), "PyCaLiAI", font=font(70), fill=GOLD)
    d.text((62, 152), "競馬AI予想  ぴかり", font=font(36), fill=GOLD_LT)

    # レースピル
    grade, name = race_short(label)
    pill = f"{grade} {name}".strip()
    pf = font(64)
    pw = d.textlength(pill, font=pf) + 64
    px = W / 2 - pw / 2
    d.rounded_rectangle([px, 258, px + pw, 258 + 96], radius=48, fill=GOLD)
    d.text((W / 2 - d.textlength(pill, font=pf) / 2, 274), pill, font=pf, fill=GOLD_DK)

    # 字幕パネル
    px0, py0, px1, py1 = 48, 520, W - 48, 1000
    d.rounded_rectangle([px0, py0, px1, py1], radius=22,
                        fill=PANEL + (235,), outline=GOLD, width=3)
    # 見出し(印)
    hf = font(62)
    hl = line["caption"]
    draw_multiline(d, (0, py0 + 34), hl, hf, GOLD, px1 - px0 - 80, 74, center_x=W / 2)
    # 本文(理由) — 見出し下から
    d.line([px0 + 40, py0 + 120, px1 - 40, py0 + 120], fill=(120, 95, 20, 180), width=2)
    draw_multiline(d, (px0 + 44, py0 + 142), line["narration"], font(44),
                   WHITE, px1 - px0 - 88, 60)

    # 免責バー
    d.rectangle([0, H - 110, W, H], fill=BAR)
    d.rectangle([0, H - 110, W, H - 107], fill=GOLD)
    df = font(27)
    dl = wrap(d, disclaimer, df, W - 80)
    yy = H - 110 + (110 - len(dl) * 34) / 2
    for ln in dl:
        d.text((W / 2 - d.textlength(ln, font=df) / 2, yy), ln, font=df, fill=WHITE)
        yy += 34
    return img


def build_one(meta_path):
    meta = json.load(open(meta_path, encoding="utf-8"))
    lines = meta["lines"]
    label = meta["race_label"]
    speaker = meta.get("speaker", 3)
    base = meta_path.replace("_meta.json", "")
    outdir = os.path.dirname(meta_path)

    print(f"  音声合成(VOICEVOX, {len(lines)}行)…")
    audio = base + "_voice.wav"
    durs = build_audio(lines, speaker, audio)

    char = keyed_character(505, 760)
    print(f"  立ち絵: {'assets/vtuber/base.png 合成' if char else 'プレースホルダ'}")

    print("  スライド描画…")
    slides = []
    for i, ln in enumerate(lines):
        p = os.path.join(outdir, f"_slide_{i}.png")
        render_slide(ln, label, char, meta["disclaimer"]).save(p)
        slides.append((p, durs[i]))

    # concat リスト
    listf = os.path.join(outdir, "_concat.txt")
    with open(listf, "w", encoding="utf-8") as f:
        for p, dur in slides:
            f.write(f"file '{os.path.abspath(p)}'\n")
            f.write(f"duration {dur:.3f}\n")
        f.write(f"file '{os.path.abspath(slides[-1][0])}'\n")  # 最終フレーム保持

    out_mp4 = base + "_short.mp4"
    print("  ffmpeg エンコード…")
    cmd = [ff_exe(), "-y", "-f", "concat", "-safe", "0", "-i", listf,
           "-i", audio,
           "-vf", "fps=30,format=yuv420p", "-c:v", "libx264", "-preset", "medium",
           "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
           out_mp4]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-1500:]); sys.exit("ffmpeg 失敗")
    for p, _ in slides:
        os.remove(p)
    os.remove(listf)
    dur = sum(durs)
    print(f"  ✓ {os.path.basename(out_mp4)}  ({dur:.1f}秒)")
    return out_mp4


# ────────────────────────────────────────────────────────────
# 動画背景モード（Higgsfield 生成アニメ立ち絵を背景に使う）
# ────────────────────────────────────────────────────────────

def render_static_hud(label, disclaimer, W=1080, H=1920):
    """常時表示 HUD（ヘッダー・レースピル・字幕パネル枠・免責バー）を透過 PNG で返す。"""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img, "RGBA")

    # ヘッダー
    d.text((60, 64), "PyCaLiAI", font=font(70), fill=GOLD)
    d.text((62, 152), "競馬AI予想  ぴかり", font=font(36), fill=GOLD_LT)

    # レースピル
    grade, name = race_short(label)
    pill = f"{grade} {name}".strip()
    pf = font(64)
    pw = d.textlength(pill, font=pf) + 64
    px = W / 2 - pw / 2
    d.rounded_rectangle([px, 258, px + pw, 258 + 96], radius=48, fill=GOLD)
    d.text((W / 2 - d.textlength(pill, font=pf) / 2, 274), pill, font=pf, fill=GOLD_DK)

    # 字幕パネル（テキストなし枠のみ）
    px0, py0, px1, py1 = 48, 520, W - 48, 1000
    d.rounded_rectangle([px0, py0, px1, py1], radius=22,
                        fill=(*PANEL, 220), outline=GOLD, width=3)

    # 免責バー
    d.rectangle([0, H - 110, W, H], fill=(*BAR, 240))
    d.rectangle([0, H - 110, W, H - 107], fill=GOLD)
    df = font(27)
    dl = wrap(d, disclaimer, df, W - 80)
    yy = H - 110 + (110 - len(dl) * 34) / 2
    for ln in dl:
        d.text((W / 2 - d.textlength(ln, font=df) / 2, yy), ln, font=df, fill=WHITE)
        yy += 34

    return img


def render_line_caption(line, W=1080, H=1920):
    """字幕パネル内のテキストのみ透過 PNG で返す（タイミング付きオーバーレイ用）。"""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img, "RGBA")
    px0, py0, px1, py1 = 48, 520, W - 48, 1000

    # 見出し（金）
    draw_multiline(d, (0, py0 + 34), line["caption"], font(62), GOLD,
                   px1 - px0 - 80, 74, center_x=W / 2)
    # セパレータ
    d.line([px0 + 40, py0 + 120, px1 - 40, py0 + 120], fill=(120, 95, 20, 180), width=2)
    # 本文（白）
    draw_multiline(d, (px0 + 44, py0 + 142), line["narration"], font(44),
                   WHITE, px1 - px0 - 88, 60)

    return img


def build_video_bg_one(meta_path, bg_video=BG_VIDEO):
    """アニメ立ち絵動画を背景に使って縦型ショートを生成する。"""
    meta = json.load(open(meta_path, encoding="utf-8"))
    lines = meta["lines"]
    label = meta["race_label"]
    speaker = meta.get("speaker", 3)
    base = meta_path.replace("_meta.json", "")
    outdir = os.path.dirname(meta_path)

    if not os.path.exists(bg_video):
        print(f"  ! 背景動画 {bg_video} が無い。通常モードにフォールバック。")
        return build_one(meta_path)

    print(f"  音声合成(VOICEVOX, {len(lines)}行)…")
    audio = base + "_voice.wav"
    durs = build_audio(lines, speaker, audio)
    total_dur = sum(durs)

    # 静的 HUD と 行ごと字幕 PNG を生成
    print("  オーバーレイ PNG 生成…")
    hud_path = os.path.join(outdir, "_hud.png")
    render_static_hud(label, meta["disclaimer"]).save(hud_path)

    cap_paths = []
    for i, ln in enumerate(lines):
        p = os.path.join(outdir, f"_cap_{i}.png")
        render_line_caption(ln).save(p)
        cap_paths.append(p)

    # ffmpeg filter_complex 構築
    # input 0: bg_video（ループ）, 1: audio, 2: HUD PNG, 3..: caption PNG
    filter_chains = ["[0:v]scale=1080:1920,setsar=1[bg]",
                     "[bg][2:v]overlay=0:0:format=auto[vhud]"]
    prev = "vhud"
    t = 0.0
    for i, (ln, dur) in enumerate(zip(lines, durs)):
        t_end = t + dur
        nxt = f"vc{i}"
        filter_chains.append(
            f"[{prev}][{3 + i}:v]overlay=0:0:format=auto"
            f":enable='between(t,{t:.3f},{t_end:.3f})'[{nxt}]"
        )
        prev = nxt
        t = t_end

    ff = ff_exe()
    cmd = [ff, "-y",
           "-stream_loop", "-1", "-t", str(total_dur + 1.0), "-i", bg_video,
           "-i", audio,
           "-i", hud_path]
    for cp in cap_paths:
        cmd += ["-i", cp]
    cmd += [
        "-filter_complex", ";".join(filter_chains),
        "-map", f"[{prev}]",
        "-map", "1:a",
        "-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-t", str(total_dur),
        "-movflags", "+faststart",
        base + "_short.mp4",
    ]

    print(f"  ffmpeg 合成 ({total_dur:.1f}秒)…")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-2000:])
        sys.exit("ffmpeg 失敗")

    # 一時ファイル削除
    for p in [hud_path] + cap_paths:
        try:
            os.remove(p)
        except OSError:
            pass

    out = base + "_short.mp4"
    print(f"  ✓ {os.path.basename(out)}  ({total_dur:.1f}秒)")
    return out


def main():
    import argparse as _ap
    p = _ap.ArgumentParser()
    p.add_argument("arg", nargs="?", default=None, help="日付 or *_meta.json パス")
    p.add_argument("--video-bg", action="store_true",
                   help=f"ぴかりアニメ動画({BG_VIDEO})を背景に使う")
    args = p.parse_args()

    arg = args.arg
    if arg and arg.endswith("_meta.json"):
        metas = [arg]
    else:
        date = arg or ""
        metas = sorted(glob.glob(f"reports/shorts/{date}*/*_meta.json")) if date \
            else sorted(glob.glob("reports/shorts/*/*_meta.json"))
    if not metas:
        sys.exit("meta.json が無い。先に make_jusho_short.py を実行。")
    # VOICEVOX 確認
    try:
        urllib.request.urlopen(f"{VOICEVOX}/version", timeout=3).read()
    except Exception:
        sys.exit("VOICEVOX 未起動。アプリを起動してから再実行。")
    builder = build_video_bg_one if args.video_bg else build_one
    for m in metas:
        print(f"● {m}")
        builder(m)
    print("done.")


if __name__ == "__main__":
    main()
