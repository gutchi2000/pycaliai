# -*- coding: utf-8 -*-
"""
jv_records.py — JV-Link オッズ録 O1〜O5 の label-free 構造化（COM 非依存・標準ライブラリのみ）
==========================================================================================
観測計画 v2.1 §2.2 の保存列を 1 録ごとに作る（32-bit / 64-bit のどちらからも import 可）。

- 価格の時点は録内の発表月日時分 [27:35] で証明し、自分の時計では証明しない。
- データ区分 [2:3] は値を保存するだけで意味を解釈しない（§2.3: stream 別に実録から同定する。
  区分番号から final を推定しない）。
- 発売フラグも値をそのまま保存する（値の意味は Stage 0 で実録から同定する）。
- 全組を slot 単位で読み、未発売組は理由（filler 文字）付きで残す。未使用 slot は含めない。

録レイアウト（CRLF を除いた長さ。O1〜O5 すべて実録で確認。O2 は 2026-09-29 の RT 実録 1 件で 2040 を確認）:
  共通 [0:2] 種別 / [2:3] データ区分 / [3:11] 作成年月日 / [11:27] レースキー16桁 /
       [27:35] 発表月日時分 / [35:37] 登録頭数 / [37:39] 出走頭数
  O1 単複枠 960  : [39] 発売F単 [40] 発売F複 [41] 発売F枠 [42] 複勝着払キー
                   単勝 28×(馬番2+odds4+人気2) @43 / 複勝 28×(馬番2+lo4+hi4+人気2) @267 /
                   枠連 36×(組番2+odds5+人気2) @603 / 票数計 単・複・枠 各11桁 @927
  O2 馬連   2040 : [39] 発売F / 153×(組番4+odds6+人気3) @40 / 票数計11 @2029
  O3 ワイド 2652 : [39] 発売F / 153×(組番4+lo5+hi5+人気3) @40 / 票数計11 @2641
  O4 馬単   4029 : [39] 発売F / 306×(組番4+odds6+人気3) @40 / 票数計11 @4018
  O5 三連複 12291: [39] 発売F / 816×(組番6+odds6+人気3) @40 / 票数計11 @12280
"""
from __future__ import annotations

import hashlib
from datetime import datetime

SPEC_KIND = {"0B31": "O1", "0B32": "O2", "0B33": "O3", "0B34": "O4", "0B35": "O5"}
RECORD_LEN = {"O1": 960, "O2": 2040, "O3": 2652, "O4": 4029, "O5": 12291}
LEN_SOURCE = {"O1": "observed", "O2": "observed", "O3": "observed", "O4": "observed", "O5": "observed"}
HEAD = {"kind": (0, 2), "kubun": (2, 3), "made": (3, 11), "race_key": (11, 27),
        "announce": (27, 35), "toroku": (35, 37), "shusso": (37, 39)}
O1_FLAGS = {"hatsubai_tansho": 39, "hatsubai_fukusho": 40, "hatsubai_wakuren": 41,
            "fukusho_chakubarai_key": 42}
O1_WAKU_BLOCK = (603, 927)
# (block 名, 開始, stride, slot 数, 組番桁, 値の型) 値の型: 'odds4'|'range44'|'odds5'|'odds6'|'range55'
BLOCKS = {
    "O1": [("tansho", 43, 8, 28, 2, "odds4"), ("fukusho", 267, 12, 28, 2, "range44"),
           ("wakuren", 603, 9, 36, 2, "odds5")],
    "O2": [("umaren", 40, 13, 153, 4, "odds6")],
    "O3": [("wide", 40, 17, 153, 4, "range55")],
    "O4": [("umatan", 40, 13, 306, 4, "odds6")],
    "O5": [("trio", 40, 15, 816, 6, "odds6")],
}
VOTES = {"O1": [("tansho", 927), ("fukusho", 938), ("wakuren", 949)],
         "O2": [("umaren", 2029)], "O3": [("wide", 2641)], "O4": [("umatan", 4018)],
         "O5": [("trio", 12280)]}


def _digits(s: str):
    s = (s or "").strip()
    return int(s) if s.isdigit() else None


def n_comb(n: int, k: int) -> int:
    if n is None or n < k:
        return 0
    r = 1
    for i in range(k):
        r = r * (n - i) // (i + 1)
    return r


def wakuren_slot_count(toroku: int | None) -> int | None:
    """登録頭数から枠連の非空白 slot 数の期待値（JRA 枠番割当: 8 頭以下は 1 枠 1 頭、
    9 頭以上は 8 枠で同枠が max(0, 頭数-8) 個（最大 8））。Stage 0 で実録と照合する仮定値。"""
    if toroku is None:
        return None
    frames = min(toroku, 8)
    return n_comb(frames, 2) + min(8, max(0, toroku - 8))


def _key(block: str, kumi: str) -> str | None:
    if not kumi.isdigit():
        return None
    if block in ("tansho", "fukusho"):
        v = int(kumi)
        return str(v) if v > 0 else None
    if block == "wakuren":
        a, b = int(kumi[0]), int(kumi[1])
        return f"{a}-{b}" if 0 < a <= b else None
    if block == "trio":
        a, b, c = int(kumi[:2]), int(kumi[2:4]), int(kumi[4:6])
        return f"{a}-{b}-{c}" if 0 < a < b < c else None
    a, b = int(kumi[:2]), int(kumi[2:4])
    if not (a > 0 and b > 0 and a != b):
        return None
    if block == "umatan":
        return f"{a}>{b}"
    return f"{a}-{b}" if a < b else None


def parse_block(rec: str, block: str, start: int, stride: int, n: int, klen: int, vtype: str) -> dict:
    """slot 単位の 4 態分類: priced（数値 > 0）/ zero（数値 0 = 発売停止）/
    unpriced（数字でない filler = 未発売・取消。filler 文字を残す）/ blank（未使用 slot）/ malformed。"""
    priced, unpriced, zero = {}, {}, []
    blank = malformed = 0
    for k in range(n):
        s = start + k * stride
        slot = rec[s:s + stride]
        if len(slot) < stride:
            malformed += 1
            continue
        if not slot.strip():
            blank += 1
            continue
        key = _key(block, slot[:klen])
        if key is None:
            malformed += 1
            continue
        body = slot[klen:]
        if vtype in ("odds4", "odds5", "odds6"):
            w = int(vtype[-1])
            raw = body[:w]
            if not raw.strip().isdigit():
                unpriced[key] = raw.strip() or "SPACE"
                continue
            v = int(raw)
            if v <= 0:
                zero.append(key)
            else:
                priced[key] = round(v / 10.0, 1)
        else:
            w = 4 if vtype == "range44" else 5
            lo_s, hi_s = body[:w], body[w:2 * w]
            if not (lo_s.strip().isdigit() and hi_s.strip().isdigit()):
                unpriced[key] = (lo_s.strip() or "SPACE")
                continue
            lo, hi = int(lo_s), int(hi_s)
            if lo <= 0 or hi <= 0:
                zero.append(key)
            elif lo > hi:
                malformed += 1
            else:
                priced[key] = [round(lo / 10.0, 1), round(hi / 10.0, 1)]
    return {"priced": priced, "unpriced": unpriced, "zero": zero, "blank": blank,
            "malformed": malformed, "nonblank": len(priced) + len(unpriced) + len(zero)}


def announced_at(announce: str, race_key: str) -> str | None:
    """'MMDDHHMM' → ISO（年はレースキー先頭4桁）。'00000000' や不正値は None。"""
    a = (announce or "").strip()
    if len(a) != 8 or not a.isdigit() or a == "0" * 8 or len(race_key or "") < 4:
        return None
    try:
        return datetime(int(race_key[:4]), int(a[:2]), int(a[2:4]), int(a[4:6]),
                        int(a[6:8])).isoformat(timespec="minutes") + "+09:00"
    except ValueError:
        return None


def _expected(kind: str, block: str, toroku, shusso):
    """(発売中の組数の期待値, 非空白 slot 数の期待値)。"""
    if block in ("tansho", "fukusho"):
        return shusso, toroku
    if block == "wakuren":
        return None, wakuren_slot_count(toroku)
    if block in ("umaren", "wide"):
        return n_comb(shusso or 0, 2), n_comb(toroku or 0, 2)
    if block == "umatan":
        return (shusso or 0) * max(0, (shusso or 0) - 1), (toroku or 0) * max(0, (toroku or 0) - 1)
    if block == "trio":
        return n_comb(shusso or 0, 3), n_comb(toroku or 0, 3)
    return None, None


def validated_parse(kind: str, rec: str) -> dict:
    """本番で実録突合済みの既存パーサ（jvlink_odds / jvlink_trio_odds）の出力。raw/parser 一致検査用。"""
    if kind == "O1":
        from jvlink_odds import parse_o1
        o = parse_o1(rec)
        return {"tansho": {str(k): v for k, v in o["tansho"].items()},
                "fukusho": {str(k): v for k, v in o["fukusho"].items()}}
    if kind == "O2":
        from jvlink_odds import parse_o2
        return {"umaren": parse_o2(rec)}
    if kind == "O3":
        from jvlink_odds import parse_o3
        return {"wide": parse_o3(rec)}
    if kind == "O4":
        from jvlink_odds import parse_o4
        return {"umatan": parse_o4(rec)}
    if kind == "O5":
        from jvlink_trio_odds import parse_o5
        return {"trio": parse_o5(rec)["odds"]}
    return {}


def structure_record(rec: str, spec: str, race_id: str, stream: str = "rt") -> dict:
    """1 録 → 保存列。raw はそのまま（CRLF 含む）残し、位置計算は CRLF を除いた本体で行う。"""
    body = rec.rstrip("\r\n")
    h = {k: body[a:b] for k, (a, b) in HEAD.items()}
    kind = SPEC_KIND.get(spec, h["kind"])
    toroku, shusso = _digits(h["toroku"]), _digits(h["shusso"])
    out = {"raw": rec, "raw_sha256": hashlib.sha256(rec.encode("utf-8", "replace")).hexdigest(),
           "raw_len": len(body), "stream": stream, "kind": h["kind"], "kubun": h["kubun"],
           "made": h["made"], "race_key": h["race_key"], "race_key_ok": h["race_key"] == race_id,
           "announce_raw": h["announce"], "announced_at": announced_at(h["announce"], h["race_key"]),
           "n_registered": toroku, "n_running": shusso, "anomalies": []}
    an = out["anomalies"]
    if h["kind"] != kind:
        an.append(f"kind {h['kind']!r} != {kind!r}")
    exp_len = RECORD_LEN.get(kind)
    out["length_ok"] = len(body) == exp_len
    if not out["length_ok"]:
        an.append(f"length {len(body)} != {exp_len} ({LEN_SOURCE.get(kind)})")
    if not out["race_key_ok"]:
        an.append(f"race_key {h['race_key']!r} != {race_id!r}")
    a = h["announce"].strip()
    if stream == "rt" and (a == "0" * 8 or (len(a) == 8 and a[4:] == "0000")):
        an.append("announce_0000")           # §3(6) 00:00 型破損の候補（RT で発表時分 00:00）
    if kind == "O1":
        out["hatsubai_flag"] = {k: body[i:i + 1] for k, i in O1_FLAGS.items()}
        out["wakuren_block_raw"] = body[O1_WAKU_BLOCK[0]:O1_WAKU_BLOCK[1]]
    else:
        out["hatsubai_flag"] = {"hatsubai": body[39:40]}
    out["votes_total"] = {b: _digits(body[s:s + 11]) for b, s in VOTES.get(kind, [])}
    parsed, unpriced, counts = {}, {}, {}
    complete = bool(out["race_key_ok"] and out["length_ok"])
    for block, start, stride, n, klen, vtype in BLOCKS.get(kind, []):
        r = parse_block(body, block, start, stride, n, klen, vtype)
        parsed[block] = r["priced"]
        unpriced[block] = r["unpriced"]
        exp_p, exp_s = _expected(kind, block, toroku, shusso)
        c = {"priced": len(r["priced"]), "unpriced": len(r["unpriced"]), "zero": len(r["zero"]),
             "blank": r["blank"], "malformed": r["malformed"], "nonblank": r["nonblank"],
             "expected_priced": exp_p, "expected_nonblank": exp_s}
        c["priced_match"] = None if exp_p is None else len(r["priced"]) == exp_p
        c["nonblank_match"] = None if exp_s is None else r["nonblank"] == exp_s
        if block == "wakuren":
            c["expected_rule"] = "wakuren_slot_count(登録頭数) — Stage 0 で実録照合する仮定"
        else:
            complete = complete and bool(c["priced_match"]) and r["malformed"] == 0
        if r["zero"]:
            c["zero_keys"] = r["zero"]
        counts[block] = c
    out["odds_parsed"], out["unpriced"], out["counts"] = parsed, unpriced, counts
    out["complete"] = complete
    try:
        ref = validated_parse(kind, rec)
        mism = [b for b, v in ref.items() if parsed.get(b) != v]
        out["validated_parser_match"] = not mism
        if mism:
            an.append(f"validated parser mismatch: {mism}")
    except Exception as exc:                    # 既存パーサが import できない環境でも録は保存する
        out["validated_parser_match"] = None
        an.append(f"validated parser unavailable: {type(exc).__name__}")
    return out


def capture(spec: str, race_id: str, records: list[str], meta: dict, stream: str = "rt") -> dict:
    """1 spec 取得（JVRTOpen 1 回）→ 保存単位。records は JV-Link が返した全録（最新は末尾）。"""
    kind = SPEC_KIND.get(spec)
    mine = [r for r in records if r[:2] == kind]
    cap = {"spec": spec, "kind": kind, "stream": stream, **meta,
           "n_records_returned": len(records), "n_records_kind": len(mine),
           "records": [structure_record(r, spec, race_id, stream) for r in mine]}
    last = cap["records"][-1] if cap["records"] else None
    cap["ok"] = bool(last and last["race_key_ok"] and last["complete"])
    cap["announced_at"] = last["announced_at"] if last else None
    return cap
