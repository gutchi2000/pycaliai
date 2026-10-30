# -*- coding: utf-8 -*-
"""
analysis/shusse/narrate.py — 出世予測 AI v2: テンプレートによる日本語プロフィール文
=====================================================================================
1 頭の予測行（`pred_test_k{k}.parquet` の 1 行、または predict_live の同形行）と、その馬の走の明細
（`starts_v2.parquet` の該当馬行）から、3〜6 文の読み物カード（dict）と本文（str）を作る。LLM は使わない。

    card = make_card(row, starts_of_horse, labels, k=3, cohort=pred["p_graded12"])
    text = narrate(card)          # card["text"] にも同じ文が入る

実行（root から）:
    ./venv311/Scripts/python.exe -m analysis.shusse.narrate --demo
    → analysis/shusse/out_v2/cards/cards_test2022.{md,json}

自分で決めたこと（仕様に無い部分の判断）
---------------------------------------
1. **調教師名**は `data/serve_code_maps.json` の `trainer`（{調教師名: コード}、現役 242 名、utf-8）を逆引きして付ける
   （`make_card(..., trainer_names=...)` で差し替え可、省略時は同ファイルを遅延ロード）。コードは starts の `調教師コード`（基準日時点の
   最新走）→ 行の `trainer_debut` の順に探し、引けた馬だけ「尾関知人厩舎」と冠に入れる。引けない（引退調教師など）馬・父（`種牡馬`）欠損は省く。
2. **着差の符号。** v2 の `margin` は「勝ち馬との差（秒、≥0）」で、勝ち馬は常に 0.0（2 着との差は持っていない）。
   よって勝った走は着差を書かず「1 着」のみ、負けた走は「（0.6 秒差）」と書く。仕様例の「−0.2 秒」（v1 の 2 着との差）は再現しない。
3. **クラス名**は `race_level` から 新馬/未勝利/1勝/2勝/3勝/OP(L)/G3/G2/G1。レース名は OP 以上なら必ず、
   1〜3 勝クラスでも固有名（「山吹賞･1勝*」→「山吹賞(1勝)」）が付いていれば添える。`class_norm`=「重賞」（旧表記）は G3 扱い。
4. **同世代分位**は呼び出し側が渡す `cohort`（同じ k・同じデビュー年の p_graded12 の Series）に対する上位 % で、
   上位 25% 以内のときだけ書く（下位は p がほぼ 0 に張り付いて順位が意味を持たない）。渡されなければ省く。
5. **根拠に使うヘッド**: P(重賞掲示板 12m) ≥ 20% か予測クラス ≥ 3勝/OP なら `top3_A_L4`、それ以外で P(離脱) ≥ 50% なら
   `top3_D`、残りは `top3_A_L0`（符号を反転して「未勝利圏から遠ざける/近づける」理由として読む）。
6. **向きの表現**は特徴の語幹ごとの辞書（`_DIRECTION`）。「小さいほど良い」（着順・着差・上り3F・上り順位）と
   「大きいほど良い」（勝利数・3着内・クラス・騎手率・TE 実績）だけ向きを付け、距離・頭数・日齢・馬体重・間隔・月・年齢・
   性・父系タイプ・騎乗数・騎手年齢・調教師年齢・推移系は向きを決めずラベルのみ（「〜が判断材料」）。
   寄与の符号は「そのヘッドの正方向が良い結果か」（A_L4/B は良い、D/A_L0 は悪い）で解釈する。
7. **規則ベースライン**は `rule_level`（k 走目のクラスと勝敗から「勝てば 1 段上げ」）と `pred_level_given_ran` の差で
   「同じ／n 段上／n 段下」。
8. **結果と判定**（ラベルがある馬のみ）: 離脱（ran_12m=0）は P(離脱) ≥ 50% なら「的中」、それ以外「外れ（離脱は読めず）」。
   走った馬は |予測クラス − 実績クラス| が 0 → 的中、1 → 惜しい、2 以上 → 外れ。実績の代表レースは基準日後 365 日以内で
   到達レベルが最高（同点なら着順が良い）の 1 走。
9. **確率の言い回し**: ≥70% は「かなり高い」、≤5% は「ほぼ見ていない」、間は平叙。整数 %、0<p<0.5% は「1% 未満」。
10. **「重賞級」**: P(L3|走る)+P(L4|走る)（重賞で着外以上まで含む）を、L3 が 10% 以上のときだけ併記（§5 の提案に従う）。
11. **レース名の切れ**: TARGET 出力のレース名は全角 4 文字で切れている（「ホープフ」「阪神ジュ」）。若馬が出る主要レースだけ
    `RACE_ALIAS` で補完し、それ以外は切れたまま出す（誤補完を避ける）。全半角は NFKC で揃える。
12. **最有力クラスと予測クラスの不一致**: 「居場所」は argmax、規則との比較と的中判定は期待レベルを閾値で丸めた
    `pred_level_given_ran`（QWK 最適化、§4.1）。両者が違う馬には「確率を均した予測クラスは〜」と 1 文足して矛盾に見えないようにする。
13. predict_live の `top3_*` は特徴名でなく日本語ラベルで来るので、`labels` の逆引きで特徴名に戻してから向きを付ける。
    行に `trainer`（名前）が直接あればコード逆引きより優先せず、逆引きが外れたときの補完に使う。
14. live（未確定）行は `y_level` 等が無い／NaN として扱い、結果文を省く。`k` は引数が無ければ基準日以前の走数から推定。
"""
from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "out_v2"
CARDS_DIR = OUT / "cards"
CODE_MAPS = Path(__file__).resolve().parents[2] / "data" / "serve_code_maps.json"
_TRAINER_NAMES: dict | None = None

COL_HORSE = "血統登録番号"
LEVEL_NAME = {0: "未勝利圏", 1: "1勝クラス", 2: "2勝クラス", 3: "3勝/OP", 4: "重賞掲示板"}
LEVEL_SHORT = {0: "未勝利圏", 1: "1勝", 2: "2勝", 3: "3勝/OP", 4: "重賞"}
RACE_CLASS = {0: "新馬", 1: "未勝利", 2: "1勝", 3: "2勝", 4: "3勝", 5: "OP(L)", 6: "G3", 7: "G2", 8: "G1"}
TARGET_OF_RACE = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 3, 6: 4, 7: 4, 8: 4}
SEX_JA = {"牡": "牡", "牝": "牝", "セ": "セン"}
GENERIC_RACE = {"新馬", "未勝利", "未勝利・牝", "500万", "1000万", "1600万", "1勝", "2勝", "3勝", "1勝クラス", "2勝クラス", "3勝クラス",
                "ｵｰﾌﾞﾟﾝ", "オープン", "OP", ""}
# TARGET の レース名 は全角 4 文字で切れる。若馬が出る主要レースだけ、切れ方が一意なものを補完する（それ以外は切れたまま出す）。
RACE_ALIAS = {
    "ホープフ": "ホープフルS", "東京スポ": "東京スポーツ杯2歳S", "アルテミ": "アルテミスS", "阪神ジュ": "阪神JF", "スプリン": "スプリングS",
    "チューリ": "チューリップ賞", "共同通信": "共同通信杯", "デイリー": "デイリー杯2歳S", "ファルコ": "ファルコンS", "フェアリ": "フェアリーS",
    "シンザン": "シンザン記念", "フラワー": "フラワーC", "アーリン": "アーリントンC", "ニュージ": "ニュージーランドT", "フローラ": "フローラS",
    "京都新聞": "京都新聞杯", "サウジア": "サウジアラビアRC", "京王杯2": "京王杯2歳S", "函館2歳": "函館2歳S", "札幌2歳": "札幌2歳S",
    "新潟2歳": "新潟2歳S", "小倉2歳": "小倉2歳S", "福島2歳": "福島2歳S", "ＮＨＫマ": "NHKマイルC", "NHKマ": "NHKマイルC", "エリザベ": "エリザベス女王杯",
    "ヴィクト": "ヴィクトリアマイル", "マイルチ": "マイルCS", "チャンピ": "チャンピオンズC", "フェブラ": "フェブラリーS", "高松宮記": "高松宮記念",
    "ＪＣ": "ジャパンC", "JC": "ジャパンC", "神戸新聞": "神戸新聞杯", "セントウ": "セントウルS", "ユニコー": "ユニコーンS", "レパード": "レパードS",
    "ラジオNI": "ラジオNIKKEI賞", "ラジオ日": "ラジオ日本賞", "東京新聞": "東京新聞杯", "キーンラ": "キーンランドC", "フィリー": "フィリーズレビュー",
    "メルボル": "メルボルンT", "フューチ": "朝日杯FS", "ファンタ": "ファンタジーS", "函館スプ": "函館スプリントS", "京王杯ス": "京王杯SC",
    "阪神カッ": "阪神C", "スワンS": "スワンS", "セントラ": "セントライト記念", "ローズS": "ローズS", "紫苑S": "紫苑S", "すみれS": "すみれS", "アイビス": "アイビスSD", "オーシャ": "オーシャンS", "葵S": "葵S", "紫苑S": "紫苑S",
}
HI, LO = 0.70, 0.05

# 語幹 → (良い側の言い回し, 悪い側の言い回し)。{p} は「1走目の」などの接頭、無ければ空。
_DIRECTION: dict[str, tuple[str, str]] = {
    "pos": ("{p}着順が良い", "{p}着順が悪い"),
    "relpos": ("{p}相対着順が良い", "{p}相対着順が悪い"),
    "margin": ("{p}勝ち馬との着差が小さい", "{p}勝ち馬との着差が大きい"),
    "agari": ("{p}上り3Fが速い", "{p}上り3Fが遅い"),
    "agari_rank": ("{p}上り3F順位が良い", "{p}上り3F順位が悪い"),
    "win": ("{p}勝利", "{p}敗戦"),
    "lvl": ("{p}出走クラスが高い", "{p}出走クラスが低い"),
    "jky_top3": ("{p}騎手の3着内率が高い", "{p}騎手の3着内率が低い"),
    "jky_pair_top3": ("{p}騎手とのコンビ成績が良い", "{p}騎手とのコンビ成績が乏しい"),
    "n_wins": ("勝利数が多い", "勝利数が少ない"),
    "n_top3": ("3着内回数が多い", "3着内回数が少ない"),
    "best_pos": ("最高着順が良い", "最高着順が低い"),
    "mean_relpos": ("平均相対着順が良い", "平均相対着順が悪い"),
    "best_margin": ("最良の着差が小さい", "最良の着差でも勝ち馬から離されている"),
    "mean_margin": ("平均着差が小さい", "平均着差が大きい"),
    "best_agari_rank": ("最良の上り3F順位が良い", "最良の上り3F順位が低い"),
    "mean_agari_rank": ("平均上り3F順位が良い", "平均上り3F順位が低い"),
    "max_lvl": ("出走した最高クラスが高い", "出走した最高クラスが低い"),
    "te_lvl": ("{e}の過去馬の到達レベルが高い", "{e}の過去馬の到達レベルが低い"),
    "te_l4": ("{e}の過去馬の重賞掲示板率が高い", "{e}の過去馬の重賞掲示板率が低い"),
    "te_l3p": ("{e}の過去馬のOP到達率が高い", "{e}の過去馬のOP到達率が低い"),
    "te_ran": ("{e}の過去馬の継続出走率が高い", "{e}の過去馬の継続出走率が低い"),
}
_ENT_JA = {"trn": "調教師", "sire": "種牡馬", "brd": "生産者"}
_HEAD_GOOD = {"top3_A_L4": True, "top3_B": True, "top3_D": False, "top3_A_L0": False}


# --------------------------------------------------------------------------- small helpers
def load_trainer_names(path: Path = CODE_MAPS) -> dict[int, str]:
    """{調教師名: コード} を逆引きして {コード(int): 名前}。ファイルが無ければ空 dict（名前は省かれるだけで落ちない）。"""
    global _TRAINER_NAMES
    if _TRAINER_NAMES is None or path != CODE_MAPS:
        names: dict[int, str] = {}
        try:
            m = json.loads(Path(path).read_text(encoding="utf-8")).get("trainer", {})
            for nm, code in m.items():
                try:
                    names[int(code)] = str(nm).strip()
                except (TypeError, ValueError):
                    continue
        except (OSError, json.JSONDecodeError, AttributeError):
            names = {}
        if path == CODE_MAPS:
            _TRAINER_NAMES = names
        return names
    return _TRAINER_NAMES


def _isnan(x) -> bool:
    try:
        return x is None or (isinstance(x, float) and math.isnan(x)) or pd.isna(x)
    except (TypeError, ValueError):
        return False


def _f(x, default=None):
    if _isnan(x):
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _i(x, default=None):
    v = _f(x)
    return default if v is None else int(round(v))


def pct(p: float | None) -> str:
    if p is None:
        return "—"
    p = min(max(float(p), 0.0), 1.0)
    if 0 < p < 0.005:
        return "1% 未満"
    return f"{round(p * 100):d}%"


def _parse_top3(v) -> list[tuple[str, float]]:
    if _isnan(v) if not isinstance(v, (list, tuple, np.ndarray)) else False:
        return []
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            return []
    out = []
    for item in list(v) if v is not None else []:
        try:
            f, c = item[0], float(item[1])
        except (TypeError, IndexError, ValueError):
            continue
        if not math.isnan(c):
            out.append((str(f), c))
    return out


def race_display_name(raw, race_level: int) -> str:
    """「山吹賞･1勝*」→「山吹賞」, 「菊花賞G1」→「菊花賞」, 「未勝利*」→ ''（固有名なし）。"""
    if _isnan(raw):
        return ""
    s = str(raw).strip().rstrip("*").strip()
    s = s.split("･")[0].split("・")[0].strip() if ("･" in s or "・" in s) and not s.startswith("未勝利") else s
    s = re.sub(r"(G[１２３123]|\(L\)|Ｌ|L)$", "", s).strip()
    s = unicodedata.normalize("NFKC", s)
    if s in GENERIC_RACE or re.fullmatch(r"\d+万", s or "x"):
        return ""
    return RACE_ALIAS.get(s, s)


def describe_start(st: pd.Series) -> dict:
    """1 走の要約 dict（文と表の両方で使う）。"""
    lvl = _i(st.get("race_level"), 0)
    pos = _i(st.get("pos"), None)
    field = _i(st.get("field"), None)
    margin = _f(st.get("margin"))
    win = (pos == 1) if pos is not None else bool(_i(st.get("win"), 0))
    name = race_display_name(st.get("レース名"), lvl)
    cls = RACE_CLASS.get(lvl, str(lvl))
    label = f"{name}({cls})" if name else cls
    date = st.get("date")
    d = {
        "date": (str(pd.Timestamp(date).date()) if not _isnan(date) else None),
        "place": (None if _isnan(st.get("場所")) else str(st.get("場所"))),
        "class": cls, "race": name or None, "label": label,
        "pos": pos, "field": field, "margin": (None if margin is None else round(margin, 1)), "win": bool(win),
        "surface": (None if _isnan(st.get("芝・ダ")) else str(st.get("芝・ダ"))),
        "dist": _i(st.get("距離")),
    }
    return d


def start_phrase(d: dict) -> str:
    pos = d.get("pos")
    if pos is None:
        return d["label"]
    s = f"{d['label']} {pos} 着"
    if d.get("field"):
        s += f"/{d['field']}頭"
    if not d.get("win") and d.get("margin") is not None and d["margin"] > 0.04:
        s += f"（{d['margin']:.1f} 秒差）"
    return s


def direction_phrase(feature: str, contrib: float, labels: dict[str, str], head_good: bool) -> tuple[str, bool | None]:
    """特徴と寄与符号から向き付きの短句。(句, 向きが決まったか) を返す。"""
    label = labels.get(feature, feature)
    good_side = (contrib > 0) == head_good
    # per-start: stem_j
    head, _, tail = feature.rpartition("_")
    p = ""
    stem = feature
    if tail.isdigit() and head in _DIRECTION:
        stem, p = head, f"{tail}走目の"
    ent = ""
    pfx, _, rest = feature.partition("_")
    if pfx in _ENT_JA and rest in _DIRECTION:
        stem, ent = rest, _ENT_JA[pfx]
    if stem in _DIRECTION:
        g, b = _DIRECTION[stem]
        return (g if good_side else b).format(p=p, e=ent), good_side
    return f"{label}", None


# --------------------------------------------------------------------------- card
def make_card(row: pd.Series | dict, starts_of_horse: pd.DataFrame | None, labels: dict[str, str] | None = None,
              k: int | None = None, cohort: pd.Series | np.ndarray | None = None, why: str | None = None,
              trainer_names: dict | None = None) -> dict:
    """予測行 + その馬の走明細 → カード dict（text 付き）。欠損に強い。trainer_names={コード: 名前}（省略時 serve_code_maps.json）。"""
    r = row if isinstance(row, dict) else row.to_dict()
    labels = labels or {}
    s = starts_of_horse if starts_of_horse is not None else pd.DataFrame()
    if len(s) and "date" in s:
        s = s.sort_values("date")
    base_date = r.get("base_date")
    base_ts = None if _isnan(base_date) else pd.Timestamp(base_date)
    if k is None:
        if "k" in r and not _isnan(r["k"]):
            k = int(r["k"])
        elif base_ts is not None and len(s):
            k = int((s["date"] <= base_ts).sum())
        else:
            k = int(len(s)) or 0

    upto = s[s["date"] <= base_ts] if (base_ts is not None and len(s)) else s
    upto = upto.head(k) if k else upto
    runs = [describe_start(st) for _, st in upto.iterrows()]
    n_wins = sum(1 for d in runs if d["win"])

    name = r.get("馬名")
    if _isnan(name) and len(s) and "馬名" in s:
        name = s["馬名"].iloc[0]
    name = "（馬名不明）" if _isnan(name) else str(name)
    sex = None
    sire = None
    if len(upto):
        if "性別" in upto:
            sex = SEX_JA.get(str(upto["性別"].iloc[-1]), None) if not _isnan(upto["性別"].iloc[-1]) else None
        if "種牡馬" in upto and not _isnan(upto["種牡馬"].iloc[0]):
            sire = str(upto["種牡馬"].iloc[0]).strip() or None
    if sex is None and not _isnan(r.get("sex")):
        sex = SEX_JA.get(str(r["sex"]), str(r["sex"]))
    if sire is None and not _isnan(r.get("sire")):
        sire = str(r["sire"])
    tnames = load_trainer_names() if trainer_names is None else trainer_names
    trainer = None
    tcode = None
    if len(upto) and "調教師コード" in upto and not _isnan(upto["調教師コード"].iloc[-1]):
        tcode = _i(upto["調教師コード"].iloc[-1])
    elif not _isnan(r.get("trainer_debut")):
        tcode = _i(r.get("trainer_debut"))
    if tcode is not None:
        trainer = tnames.get(tcode) or tnames.get(str(tcode))
    if trainer is None and not _isnan(r.get("trainer")) and str(r.get("trainer")).strip():
        trainer = str(r["trainer"]).strip()                  # 行に名前が直接入っている（predict_live の json）

    p_drop = _f(r.get("p_dropout"))
    pl = [_f(r.get(f"p_L{j}_given_ran")) for j in range(5)]
    pl_ok = all(v is not None for v in pl)
    p_g12 = _f(r.get("p_graded12"))
    if p_g12 is None and pl_ok and p_drop is not None:
        p_g12 = (1 - p_drop) * pl[4]
    p_g24 = _f(r.get("p_graded24"))
    pred_level = _i(r.get("pred_level_given_ran"))
    if pred_level is None and pl_ok:
        pred_level = int(np.argmax(pl))
    exp_level = _f(r.get("expected_level_given_ran"))
    rule_level = _i(r.get("rule_level"))
    order = sorted(range(5), key=lambda j: -(pl[j] or 0)) if pl_ok else []

    # cohort percentile（上位 %）
    pct_rank = None
    if cohort is not None and p_g12 is not None:
        c = np.asarray(pd.Series(cohort).dropna(), dtype=float)
        if len(c):
            pct_rank = float((c > p_g12).mean() * 100)

    # 根拠
    if (p_g12 is not None and p_g12 >= 0.2) or (pred_level is not None and pred_level >= 3):
        head = "top3_A_L4"
    elif p_drop is not None and p_drop >= 0.5:
        head = "top3_D"
    else:
        head = "top3_A_L0"
    top3 = _parse_top3(r.get(head))
    if not top3:
        for alt in ("top3_A_L4", "top3_A_L0", "top3_D", "top3_B"):
            top3 = _parse_top3(r.get(alt))
            if top3:
                head = alt
                break
    reasons = []
    inv_labels = {v: k for k, v in labels.items()}          # predict_live の top3 は日本語ラベルで来るので特徴名に戻す
    for f, c in top3[:3]:
        f = f if f in labels else inv_labels.get(f, f)
        phrase, directed = direction_phrase(f, c, labels, _HEAD_GOOD.get(head, True))
        reasons.append({"feature": f, "label": labels.get(f, f), "contrib": round(c, 3), "phrase": phrase, "directed": directed})

    # 結果（ラベルがあるとき）
    ran = r.get("ran_12m")
    y_level = _f(r.get("y_level"))
    outcome = None
    if not _isnan(ran):
        ran_i = int(ran)
        after = pd.DataFrame()
        if base_ts is not None and len(s):
            after = s[(s["date"] > base_ts) & (s["date"] <= base_ts + pd.Timedelta(days=365))].copy()
        best = None
        if len(after):
            after["lvl_v2"] = [4 if (_i(a.race_level, 0) >= 6 and _i(a.pos, 99) <= 5) else (3 if _i(a.race_level, 0) >= 6 else TARGET_OF_RACE.get(_i(a.race_level, 0), 0))
                               for a in after.itertuples()]
            b = after.sort_values(["lvl_v2", "pos"], ascending=[False, True]).iloc[0]
            best = describe_start(b)
        if ran_i == 0:
            verdict = "的中" if (p_drop is not None and p_drop >= 0.5) else "外れ（離脱は読めず）"
            outcome = {"ran_12m": 0, "actual_level": None, "actual_level_name": "離脱（12 か月出走なし）", "best_race": None,
                       "n_starts_12m": 0, "verdict": verdict}
        else:
            yl = None if y_level is None else int(y_level)
            if yl is None and len(after):
                yl = int(after["lvl_v2"].max())
            verdict = None
            if yl is not None and pred_level is not None:
                diff = abs(pred_level - yl)
                verdict = "的中" if diff == 0 else ("惜しい" if diff == 1 else "外れ")
            outcome = {"ran_12m": 1, "actual_level": yl, "actual_level_name": (LEVEL_NAME.get(yl) if yl is not None else None),
                       "best_race": best, "n_starts_12m": _i(r.get("n_starts_12m"), len(after)), "verdict": verdict,
                       "graded12": (None if _isnan(r.get("y_graded12")) else bool(int(r["y_graded12"]))),
                       "graded24": (None if _isnan(r.get("y_graded24")) or not bool(r.get("h24_complete", True)) else bool(int(r["y_graded24"])))}

    card = {
        "馬名": name, "horse_id": (None if _isnan(r.get(COL_HORSE)) else int(r[COL_HORSE])), "k": int(k or 0),
        "why": why, "debut_year": _i(r.get("debut_year")), "base_date": (None if base_ts is None else str(base_ts.date())),
        "sex": sex, "sire": sire, "trainer": trainer, "trainer_code": tcode,
        "runs": runs, "n_wins": n_wins,
        "p_dropout": p_drop, "p_level_given_ran": (dict(zip([LEVEL_NAME[j] for j in range(5)], [round(v, 4) for v in pl])) if pl_ok else None),
        "p_levels": pl if pl_ok else None,
        "p_graded12": p_g12, "p_graded24": p_g24, "p_graded_class": ((pl[3] + pl[4]) if pl_ok else None),
        "pred_level": pred_level, "pred_level_name": (LEVEL_NAME.get(pred_level) if pred_level is not None else None),
        "expected_level": exp_level, "rule_level": rule_level, "rule_level_name": (LEVEL_NAME.get(rule_level) if rule_level is not None else None),
        "level_order": order, "cohort_top_pct": pct_rank, "reason_head": head, "reasons": reasons, "outcome": outcome,
    }
    card["text"] = narrate(card)
    return card


# --------------------------------------------------------------------------- narrate
def _ceil_pct(x: float) -> int:
    return max(1, int(math.ceil(x)))


def narrate(card: dict) -> str:
    """カード dict → 3〜6 文の敬体の読み物。make_card の出力、または同じキーを持つ dict を受ける。"""
    c = card
    name = c.get("馬名") or "この馬"
    k = int(c.get("k") or len(c.get("runs") or []))
    # 1. 冠 + 走の要約
    head_bits = [b for b in (c.get("sex"), (f"父{c['sire']}" if c.get("sire") else None), (f"{c['trainer']}厩舎" if c.get("trainer") else None)) if b]
    head = f"**{name}**" + (f"（{'、'.join(head_bits)}）" if head_bits else "") + "。"
    runs = c.get("runs") or []
    if runs:
        seq = " → ".join(start_phrase(d) for d in runs)
        n_wins = c.get("n_wins", sum(1 for d in runs if d.get("win")))
        win_txt = f"{len(runs)} 走で {n_wins} 勝" if n_wins else f"{len(runs)} 走して未勝利"
        if len(runs) == 1:
            win_txt = "デビュー戦を勝ちました" if n_wins else "デビュー戦は勝てませんでした"
            s1 = f"{head}{seq} と、{win_txt}。"
        else:
            s1 = f"{head}{seq} と {win_txt}です。"
    else:
        s1 = f"{head}（{k} 走目時点、走の明細なし）"

    # 2. 確率
    pg = c.get("p_graded12")
    pd_ = c.get("p_dropout")
    bits = []
    if pg is not None:
        if pg >= HI:
            lead = f"AI は 1 年以内に重賞で掲示板（5 着以内）に載る確率を **{pct(pg)}** とかなり高く見ています"
        elif pg <= LO:
            lead = f"AI が見る 1 年以内の重賞掲示板（5 着以内）の確率は **{pct(pg)}** と、ほぼ見ていません"
        else:
            lead = f"AI は 1 年以内に重賞で掲示板（5 着以内）に載る確率を **{pct(pg)}** と見ます"
        top = c.get("cohort_top_pct")
        if top is not None and top <= 25:
            lead += f"（同世代の上位 {_ceil_pct(top)}%）"
        bits.append(lead + "。")
    if pd_ is not None:
        if pd_ >= HI:
            bits.append(f"一方、離脱（1 年間出走なし）の確率は **{pct(pd_)}** とかなり高く、走り続けられるかを疑っています。")
        elif pd_ >= 0.3:
            bits.append(f"離脱（1 年間出走なし）の確率は {pct(pd_)} と低くありません。")
        else:
            bits.append(f"離脱（1 年間出走なし）の確率は {pct(pd_)}。")
    s2 = "".join(bits)

    # 3. 居場所
    s3 = ""
    pl = c.get("p_levels")
    order = c.get("level_order") or []
    if pl and order:
        first, second = order[0], order[1] if len(order) > 1 else None
        s3 = f"走り続けた場合の 1 年後の居場所は「{LEVEL_NAME[first]}」が最有力（確率 {pct(pl[first])}）"
        if second is not None and pl[second] >= 0.1:
            s3 += f"、次いで「{LEVEL_NAME[second]}」（{pct(pl[second])}）"
        s3 += "。"
        pl_cut = c.get("pred_level")
        if pl_cut is not None and pl_cut != first:
            s3 += f"確率を均した予測クラス（規則との比較に使う値）は「{LEVEL_NAME[pl_cut]}」です。"
        pgc = c.get("p_graded_class")
        if pgc is not None and pl[3] >= 0.10 and pl[4] >= 0.10 and first in (3, 4):
            s3 += f"重賞で着外以上まで含めた「重賞級」なら {pct(pgc)} と読みます。"

    # 4. 根拠 + 規則
    s4 = ""
    reasons = c.get("reasons") or []
    if reasons:
        phr = "、".join(r["phrase"] for r in reasons)
        head_kind = c.get("reason_head")
        if head_kind == "top3_D":
            s4 = f"離脱を疑う材料は、{phr}。"
        elif head_kind == "top3_A_L0":
            s4 = f"判断材料は、{phr}。"
        else:
            s4 = f"根拠は、{phr}。"
    rl, pl_ = c.get("rule_level"), c.get("pred_level")
    if rl is not None and pl_ is not None:
        rn = LEVEL_NAME.get(rl, str(rl))
        diff = pl_ - rl
        if diff == 0:
            s4 += f"規則（勝てば 1 段上げ）でも「{rn}」で、AI の見立てと同じです。"
        elif diff > 0:
            s4 += f"規則（勝てば 1 段上げ）なら「{rn}」止まりで、AI はそれより {diff} 段上を見ています。"
        else:
            s4 += f"規則（勝てば 1 段上げ）なら「{rn}」ですが、AI はそれより {-diff} 段下と慎重です。"

    # 5. 結果
    s5 = ""
    oc = c.get("outcome")
    if oc:
        if oc.get("ran_12m") == 0:
            s5 = f"（結果: 12 か月以内の出走なし＝離脱。{oc.get('verdict') or ''}）"
        else:
            br = oc.get("best_race")
            lvl_name = oc.get("actual_level_name") or "不明"
            if br:
                reach = f"{start_phrase(br)}で「{lvl_name}」"
            else:
                reach = f"「{lvl_name}」"
            tail = ""
            if oc.get("graded24") is True and oc.get("actual_level") != 4:
                tail = "、2 年以内には重賞掲示板に到達"
            s5 = f"（結果: {reach}{tail}。{oc.get('verdict') or ''}）".replace("。）", "）") if not oc.get("verdict") else f"（結果: {reach}{tail}。{oc['verdict']}）"
    parts = [s1, s2, s3, s4, s5]
    return "\n".join(p for p in parts if p)


# --------------------------------------------------------------------------- demo
NAMED_SPEC = ["ミッキーカプチーノ", "ガストリック", "ピピオラ", "ドゥレッツァ", "リバティアイランド", "ドゥラエレーデ",
              "ソールオリエンス", "タスティエーラ"]
NAMED_V1 = ["ベラジオオペラ", "ソールオリエンス", "ペリファーニア", "ハーパー", "タスティエーラ", "ドゥラエレーデ", "ドゥレッツァ",
            "リバティアイランド", "スキルヴィング", "ピピオラ", "アヴェッリーノ", "ミッキーカプチーノ", "ガストリック", "ドゥーラ"]
UPDATE_NAMES = ["ドゥレッツァ", "リバティアイランド", "ドゥラエレーデ"]


def load_labels(out: Path = OUT) -> dict[str, str]:
    p = out / "feature_labels.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def cards_for(pred: pd.DataFrame, starts: pd.DataFrame, labels: dict, k: int, why: str, cohort_all: pd.DataFrame | None = None) -> list[dict]:
    cohort_all = pred if cohort_all is None else cohort_all
    out = []
    for _, r in pred.iterrows():
        sh = starts[starts[COL_HORSE] == r[COL_HORSE]]
        coh = cohort_all.loc[cohort_all["debut_year"] == r["debut_year"], "p_graded12"] if "debut_year" in cohort_all else cohort_all["p_graded12"]
        out.append(make_card(r, sh, labels, k=k, cohort=coh, why=why))
    return out


def render_md(sections: list[tuple[str, list[dict]]], title: str) -> str:
    lines = [f"# {title}", ""]
    for sec_title, cards in sections:
        lines += [f"## {sec_title}", ""]
        for c in cards:
            lines.append(f"### {c['馬名']}（k={c['k']}、基準日 {c['base_date']}）" + (f" — {c['why']}" if c.get("why") else ""))
            lines.append("")
            lines.append(c["text"])
            lines.append("")
    return "\n".join(lines)


def demo(out: Path = OUT) -> Path:
    labels = load_labels(out)
    starts = pd.read_parquet(out / "starts_v2.parquet")
    preds = {k: pd.read_parquet(out / f"pred_test_k{k}.parquet") for k in (1, 3)}
    t3 = preds[3][preds[3]["split"] == "test"].copy()
    t1 = preds[1][preds[1]["split"] == "test"].copy()
    sections: list[tuple[str, list[dict]]] = []

    named = list(dict.fromkeys(NAMED_SPEC + NAMED_V1))
    nm = t3[t3["馬名"].isin(named)].set_index("馬名").reindex([n for n in named if n in set(t3["馬名"])]).reset_index()
    sections.append(("名指し馬（§5.1 + v1 カード）— k=3", cards_for(nm, starts, labels, 3, "名指し馬", t3)))
    used = set(nm[COL_HORSE])

    top = t3[~t3[COL_HORSE].isin(used)].nlargest(10, "p_graded12")
    sections.append(("P(重賞掲示板 12m) 上位 10 — k=3", cards_for(top, starts, labels, 3, "P(重賞) 上位", t3)))
    used |= set(top[COL_HORSE])

    unwon = t1[(t1["rule_level"] == 0) & (t1["p_graded12"] >= 0.2)].nlargest(10, "p_graded12")
    sections.append(("規則は未勝利圏なのに AI は重賞 20% 以上 — k=1（デビュー戦直後）", cards_for(unwon, starts, labels, 1, "規則 L0 / AI 重賞 20%+", t1)))

    rest = t3[~t3[COL_HORSE].isin(used)]
    miss = pd.concat([
        rest[rest["ran_12m"] == 1].nlargest(2, "p_dropout").assign(_why="離脱確率が最大なのに走り続けた"),
        rest[rest["y_level"] == 4].nsmallest(2, "p_graded12").assign(_why="P(重賞) 最小で重賞掲示板に到達"),
        rest[(rest["ran_12m"] == 0)].nsmallest(1, "p_dropout").assign(_why="離脱確率が最小なのに離脱"),
    ])
    miss_cards = []
    for _, r in miss.iterrows():
        sh = starts[starts[COL_HORSE] == r[COL_HORSE]]
        miss_cards.append(make_card(r, sh, labels, k=3, cohort=t3["p_graded12"], why=r["_why"]))
    sections.append(("外れの例 — k=3", miss_cards))

    upd = []
    for n in UPDATE_NAMES:
        for k, t in ((1, t1), (3, t3)):
            rr = t[t["馬名"] == n]
            if len(rr):
                r = rr.iloc[0]
                upd.append(make_card(r, starts[starts[COL_HORSE] == r[COL_HORSE]], labels, k=k, cohort=t["p_graded12"], why=f"更新例 {k} 走目時点"))
    sections.append(("「更新」例: 同じ馬の k=1 → k=3", upd))

    CARDS_DIR.mkdir(parents=True, exist_ok=True)
    md = render_md(sections, "出世予測 AI v2 — 読み物カード（test 世代 = 2022 デビュー）")
    (CARDS_DIR / "cards_test2022.md").write_text(md, encoding="utf-8")
    payload = [{"section": s, "cards": cs} for s, cs in sections]
    (CARDS_DIR / "cards_test2022.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=_json_default), encoding="utf-8")
    n = sum(len(cs) for _, cs in sections)
    print(f"wrote {n} cards -> {CARDS_DIR / 'cards_test2022.md'}")
    return CARDS_DIR / "cards_test2022.md"


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    return str(o)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="test 2022 の名指し馬ほかのカードを out_v2/cards/ に出す")
    a = ap.parse_args()
    if a.demo:
        demo()
    else:
        ap.print_help()
