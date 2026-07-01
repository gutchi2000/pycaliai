# -*- coding: utf-8 -*-
"""
explain_race.py — レース説明(Explainability)カード生成
================================================================================
モデルの内部数値を人間が読める分析カードに変換する。勝てる/勝てないとは別レイヤーの
「説明・可視化」。数値は既存資産を流用(新規学習なし):

  各馬:
    印(◎〇▲△)  勝利確率  予想オッズ(=1/p)  基準オッズ(+10%妙味線)  実オッズ  期待値
    対戦レーティング(ELO horse / vs field)  Glicko(μ±RD=確信度)
    位置取り予測(1角/4角)  脚質
  レース:
    レースレベル(ELO level + クラス)  馬場解説(クッション/含水)  展開予測(ペース)
  コメント:
    印上位3頭の解説  特選穴馬(中穴妙味, 大穴弱点を考慮)

データ源(全て既存/leak-safe):
  data/_ev_grid_scores.parquet (v6 score) / pl_calibrators_v6 /
  elo_feats / glicko_feats / feats_serious(脚質S2) / baba_feats / data/odds / kekka
出力: reports/explain/{rid}.json + 標準出力カード
実行: PYTHONUTF8=1 ./venv311/Scripts/python.exe explain_race.py [rid]
"""
from __future__ import annotations
import io, json, sys, warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import glob
import backtest_pl_ev as BT
import pl_probs as PL


def load_odds():
    """全頭確定オッズ {rid16:{ban:(tan,flo,fhi)}}."""
    out = {}
    for f in sorted(glob.glob(str(Path(__file__).parent / "data/odds/odds_*.csv"))):
        df = pd.read_csv(f, encoding="cp932", low_memory=False)
        rid = df["レースID(新)"].astype(str).str[:16]
        ban = pd.to_numeric(df["馬番"], errors="coerce")
        tan = pd.to_numeric(df["単勝オッズ"], errors="coerce")
        flo = pd.to_numeric(df["複勝オッズ下限"], errors="coerce")
        fhi = pd.to_numeric(df["複勝オッズ上限"], errors="coerce")
        for r, b, t, lo, hi in zip(rid.values, ban.values, tan.values, flo.values, fhi.values):
            if b != b:
                continue
            out.setdefault(r, {})[int(b)] = (float(t) if t == t else np.nan,
                                             float(lo) if lo == lo else np.nan,
                                             float(hi) if hi == hi else np.nan)
    return out

BASE = Path(__file__).parent
SCORE_CACHE = BASE / "data/_ev_grid_scores.parquet"
MASTER_CSV = BASE / "data/master_v2_20130105-20251228.csv"
V6_CAL = BASE / "models/pl_calibrators_v6.pkl"
OUTDIR = BASE / "reports/explain"
COL_RID = "レースID(新/馬番無)"; COL_BAN = "馬番"; COL_JYUN = "着順"

MARKS = {0: "◎", 1: "〇", 2: "▲", 3: "△", 4: "△"}


def log(m): print(m, flush=True)
def _f(x):
    try:
        v = float(x); return v if v == v else None
    except Exception:
        return None


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"
LLM_SYSTEM = (
    "あなたはJRA中央競馬の予想解説者。与えられた数値・事実だけを使う。数字・人気・馬名の創作は厳禁。"
    "◎は最上位評価馬。◎〇▲を『人気薄』『穴』と書くな(与えられた実オッズが低いなら人気馬扱い)。"
    "簡潔な日本語(各馬1〜2文)。勝率と対戦RTG(平均比)と脚質に触れる。EVは1.0が損益分岐(下回れば妙味なし)。"
    "特選穴馬は『特選穴馬データ』で与えた馬についてのみ書き、別の馬を挙げるな。煽らず淡々と。出力は解説本文のみ、英単語禁止。")


def ollama_narrative(facts, timeout=90):
    """ローカルOllamaで自然文解説を生成。到達不可/失敗時は None(テンプレにフォールバック)。"""
    import os
    if os.environ.get("EXPLAIN_NO_LLM"):
        return None
    payload = {"model": os.environ.get("PYCALI_MODEL", OLLAMA_MODEL),
               "system": LLM_SYSTEM, "prompt": facts, "stream": False,
               "options": {"num_ctx": 4096, "temperature": 0.4}}
    try:
        import requests
        r = requests.post(os.environ.get("OLLAMA_URL", OLLAMA_URL), json=payload, timeout=timeout)
        return (r.json().get("response") or "").strip() or None
    except Exception:
        try:
            import urllib.request
            req = urllib.request.Request(os.environ.get("OLLAMA_URL", OLLAMA_URL),
                                         data=json.dumps(payload).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return (json.loads(resp.read().decode("utf-8")).get("response") or "").strip() or None
        except Exception:
            return None


def kyakushitsu(frac):
    if frac is None or frac != frac: return "不明"
    if frac <= 0.20: return "逃げ"
    if frac <= 0.42: return "先行"
    if frac <= 0.66: return "差し"
    return "追込"


def load_race(rid):
    sc = pd.read_parquet(SCORE_CACHE)
    sc[COL_RID] = sc[COL_RID].astype(str)
    g = sc[sc[COL_RID] == str(rid)].copy()
    if g.empty:
        # 既定: 2025東京芝の多頭数レース
        sc["rid16"] = sc[COL_RID].str.replace(r"\D", "", regex=True).str[:16]
        te = sc[(sc["year"] == 2025) & (sc["_venue"] == "東京") & (sc["_surf"] == "芝")]
        cnt = te.groupby(COL_RID).size(); big = cnt[cnt >= 16]
        rid = big.index[len(big) // 2]; g = sc[sc[COL_RID] == str(rid)].copy()
    g["rid16"] = g[COL_RID].str.replace(r"\D", "", regex=True).str[:16]
    g["ban"] = pd.to_numeric(g[COL_BAN], errors="coerce").fillna(0).astype(int)
    return str(rid), g


def join_feats(g):
    for nm, cols in [("elo_feats", ["elo_T2M1_horse", "elo_T2M1_vs", "elo_T2M1_level"]),
                     ("glicko_feats", ["g2_mu", "g2_rd"]),
                     ("feats_serious", ["s2_style_early", "s2_style_late", "s2_is_front",
                                         "s2_race_pace_press"])]:
        d = pd.read_parquet(BASE / f"data/{nm}.parquet")
        d["rid16"] = d["rid16"].astype(str); d["ban"] = pd.to_numeric(d["ban"], errors="coerce").fillna(0).astype(int)
        g = g.merge(d[["rid16", "ban"] + cols], on=["rid16", "ban"], how="left")
    return g


_MASTER_COLS = ["レースID(新/馬番無)", "馬番", "馬名", "レース名", "場所", "芝・ダ", "距離",
                "馬場状態", "クラス名", "騎手", "日付"]


def load_master_subset(date=None):
    """master を usecols で読み(必要なら日付で絞り), ban 付き df を返す(共有用に1回だけ呼ぶ)。"""
    df = pd.read_csv(MASTER_CSV, encoding="utf-8-sig", low_memory=False,
                     usecols=lambda c: c in _MASTER_COLS)
    df[COL_RID] = df[COL_RID].astype(str)
    if date:
        df = df[df[COL_RID].str.startswith(str(date))].copy()
    df["ban"] = pd.to_numeric(df[COL_BAN], errors="coerce").fillna(0).astype(int)
    return df


def master_info(rid, info_df=None):
    df = info_df if info_df is not None else load_master_subset()
    return df[df[COL_RID] == str(rid)].copy()


def load_shared():
    """カード生成で使い回す重いデータを1回ロード。"""
    return {"cal": joblib.load(V6_CAL)["calibrators"],
            "oddsmap": load_odds(), "payouts": BT.load_payouts()}


def baba_comment(venue, datestr, surf):
    try:
        b = pd.read_parquet(BASE / "data/baba_feats.parquet")
        b["日付"] = b["日付"].astype(str)
        row = b[(b["場所"] == venue) & (b["日付"].str.replace(r"\D", "", regex=True).str[:8] == datestr)]
        if not row.empty:
            cush = _f(row.iloc[0].get("cushion"))
            gp = _f(row.iloc[0].get("shiba_gp" if surf == "芝" else "dirt_gp"))
            parts = []
            if cush is not None:
                tone = "軽い(時計速い)" if cush >= 9.5 else ("標準" if cush >= 8 else "重い(時計かかる)")
                parts.append(f"クッション値{cush:.1f}={tone}")
            if gp is not None:
                parts.append(f"含水率{gp:.1f}%")
            if parts:
                return "、".join(parts)
    except Exception:
        pass
    return None


_BABA_BIAS = None


def load_baba_bias():
    global _BABA_BIAS
    if _BABA_BIAS is None:
        try:
            _BABA_BIAS = json.loads((BASE / "data/baba_bias.json").read_text(encoding="utf-8"))
        except Exception:
            _BABA_BIAS = {}
    return _BABA_BIAS


def _pick_band(bands, v):
    for b in bands:
        lo = b["lo"] if b["lo"] is not None else -1e9
        hi = b["hi"] if b["hi"] is not None else 1e9
        if lo <= v < hi:
            return b
    return None


def baba_bias_line(venue, datestr, surf):
    """当日のクッション/含水率→『前残り度・枠バイアス』の記述読み出し（馬券構築の地図）。"""
    bb = load_baba_bias()
    if not bb:
        return None
    try:
        b = pd.read_parquet(BASE / "data/baba_feats.parquet")
        b["日付"] = b["日付"].astype(str)
        row = b[(b["場所"] == venue) & (b["日付"].str.replace(r"\D", "", regex=True).str[:8] == datestr)]
        if row.empty:
            return None
        r = row.iloc[0]
        cu, sg, dg = _f(r.get("cushion")), _f(r.get("shiba_gp")), _f(r.get("dirt_gp"))
        if surf == "芝":
            if cu is not None:
                tbl, val, lab = "芝_cushion", cu, f"クッション{cu:.1f}"
            elif sg is not None:
                tbl, val, lab = "芝_moisture", sg, f"含水{sg:.1f}%"
            else:
                return None
        else:
            if dg is None:
                return None
            tbl, val, lab = "ダ_moisture", dg, f"含水{dg:.1f}%"
        spec = bb.get(tbl)
        band = _pick_band(spec["bands"], val) if spec else None
        if not band:
            return None
        fe = band["front_edge"]; wk = band["waku_out_minus_in"]
        wk_txt = "外枠有利" if wk >= 1.0 else ("内枠やや有利" if wk <= -1.0 else "枠ほぼ均等")
        line = (f"{surf}・{band['label']}({lab}) → 前残り度{fe:+.1f}pp"
                f"（逃{band['nige_win']}%・先{band['senko_win']}%／差{band['sashi_win']}%）・{wk_txt}(外-内{wk:+.1f}pp)")
        return {"line": line, "band": band["label"], "metric": lab, "front_edge": fe,
                "waku_out_minus_in": wk, "nige_win": band["nige_win"], "senko_win": band["senko_win"]}
    except Exception:
        return None


def build(rid, g, shared=None, info_df=None):
    shared = shared or load_shared()
    cal = shared["cal"]
    g = g.sort_values("ban").reset_index(drop=True)
    w = PL.pl_weights(g["_score"].values)
    p_win = cal["tansho"].predict(w / w.sum())
    p_top3 = cal["fukusho"].predict(BT.all_fukusho_vec_fast(w))
    n = len(g)
    oddsmap = shared["oddsmap"].get(str(rid)[:16], {})
    payouts = shared["payouts"].get(str(rid)[:16])
    info = master_info(rid, info_df)
    name_by_ban = dict(zip(info["ban"], info["馬名"])) if not info.empty else {}
    meta = info.iloc[0] if not info.empty else None

    # 印付けは生スコア順(校正isotonicの段差でタイになり馬番依存になるのを回避)。表示は校正p_win。
    order = np.argsort(-g["_score"].values)
    rank_of = {int(idx): r for r, idx in enumerate(order)}

    horses = []
    for i in range(n):
        ban = int(g["ban"].iloc[i])
        pw = float(p_win[i]); p3 = float(p_top3[i])
        fair = round(1.0 / pw, 1) if pw > 0 else None
        kijun = round(1.1 / pw, 1) if pw > 0 else None     # +10%妙味線
        od = oddsmap.get(ban, (np.nan, np.nan, np.nan))
        real = _f(od[0])
        ev = round(real * pw, 2) if real else None
        early = _f(g["s2_style_early"].iloc[i]); late = _f(g["s2_style_late"].iloc[i])
        pos1 = int(round(early * (n - 1) + 1)) if early is not None else None
        pos4 = int(round(late * (n - 1) + 1)) if late is not None else None
        horses.append({
            "ban": ban, "name": name_by_ban.get(ban, f"{ban}番"),
            "mark": MARKS.get(rank_of[i], ""),
            "p_win_pct": round(pw * 100, 1), "p_top3_pct": round(p3 * 100, 1),
            "yoso_odds": fair, "kijun_odds": kijun, "real_odds": real, "ev": ev,
            "buy": bool(real and kijun and real >= kijun),
            "elo": (round(_f(g["elo_T2M1_horse"].iloc[i])) if _f(g["elo_T2M1_horse"].iloc[i]) else None),
            "elo_vs_field": (round(_f(g["elo_T2M1_vs"].iloc[i]), 1) if _f(g["elo_T2M1_vs"].iloc[i]) is not None else None),
            "glicko_mu": (round(_f(g["g2_mu"].iloc[i])) if _f(g["g2_mu"].iloc[i]) else None),
            "glicko_rd": (round(_f(g["g2_rd"].iloc[i])) if _f(g["g2_rd"].iloc[i]) else None),
            "pos_1c": pos1, "pos_4c": pos4, "kyaku": kyakushitsu(early),
            "_score_rank": rank_of[i],
        })
    horses_by_rank = sorted(horses, key=lambda h: h["_score_rank"])

    # レースレベル/馬場/展開
    elo_level = _f(g["elo_T2M1_level"].iloc[0])
    n_front = int(np.nansum(pd.to_numeric(g["s2_is_front"], errors="coerce").values))
    if n_front >= 4: pace = "ハイペース想定(逃げ先行多数)→差し・追込有利"
    elif n_front <= 1: pace = "スローペース想定(逃げ先行手薄)→前残り有利"
    else: pace = "平均ペース想定"
    venue = meta["場所"] if meta is not None else g["_venue"].iloc[0]
    surf = meta["芝・ダ"] if meta is not None else g["_surf"].iloc[0]
    baba = baba_comment(venue, str(rid)[:8], surf) or (f"馬場:{meta['馬場状態']}" if meta is not None else None)
    baba_bias = baba_bias_line(venue, str(rid)[:8], surf)

    race = {
        "rid": str(rid),
        "name": (meta["レース名"] if meta is not None else ""),
        "venue": venue, "surface": surf,
        "distance": (int(_f(meta["距離"])) if meta is not None and _f(meta["距離"]) else None),
        "class": (meta["クラス名"] if meta is not None else None),
        "field_size": n,
        "race_level_elo": round(elo_level) if elo_level else None,
        "baba": baba, "baba_bias": baba_bias, "pace": pace,
        "n_front_runners": n_front,
    }

    # コメント: 印上位3頭
    comments = []
    for h in horses_by_rank[:3]:
        vs = h["elo_vs_field"]
        strg = (f"出走馬中で力上位(平均比+{vs})" if vs and vs > 0 else
                (f"平均並み({vs})" if vs is not None else ""))
        conf = ""
        if h["glicko_rd"] is not None:
            conf = "（実績豊富で評価安定）" if h["glicko_rd"] < 80 else "（キャリア浅/休み明けで評価に幅）"
        pos = f"{h['pos_1c']}番手→4角{h['pos_4c']}番手" if h["pos_1c"] else ""
        comments.append(
            f"{h['mark']}{h['name']}: 対戦RTG {h['elo']}{('('+strg+')') if strg else ''}{conf}。"
            f"脚質={h['kyaku']}{('、'+pos) if pos else ''}。"
            f"AI勝率{h['p_win_pct']}%・複勝圏{h['p_top3_pct']}%（予想オッズ{h['yoso_odds']}倍）。")

    # 特選穴馬: 中穴帯(実オッズ8〜50倍)で AIが市場より高評価=期待値最大の馬(=AI注目の妙味)
    ana = None
    pool = [h for h in horses if h["real_odds"] and 8.0 <= h["real_odds"] <= 50.0 and h["ev"]]
    if pool:
        ana_h = max(pool, key=lambda h: h["ev"])
        npop = sum(1 for h in horses if h["real_odds"] and h["real_odds"] < ana_h["real_odds"]) + 1
        tail = race["pace"].split("→")[-1] if "→" in race["pace"] else "展開次第"
        ana = (f"{ana_h['name']}({ana_h['real_odds']}倍/{npop}番人気): "
               f"AI勝率{ana_h['p_win_pct']}%・対戦RTG{ana_h['elo']}(平均比{ana_h['elo_vs_field']:+})は人気以上の評価。"
               f"脚質{ana_h['kyaku']}・期待値{ana_h['ev']}倍。{tail}向き。"
               f"※AIは大穴に構造的に弱く、この種の高EV妙味は外れ易い(optimizer's curse)ため少額・抑え推奨。")

    # 結果(デモ表示用; 確定後のみ)
    result = None
    if payouts:
        result = {"win_ban": payouts["win"], "plc_ban": payouts["plc"], "sho_ban": payouts["sho"],
                  "tansho_pay": payouts["tansho"]}

    # LLM 自然文ナラティブ(数値はカードから供給、創作禁止)。失敗時 None→テンプレ表示。
    facts_lines = [f"レース: {race['name']} {race['venue']}{race['surface']}{race['distance']}m "
                   f"{race['class']} {race['field_size']}頭。馬場: {race['baba']}。展開: {race['pace']}。"]
    for h in horses_by_rank[:3]:
        vs = f"{h['elo_vs_field']:+}" if h["elo_vs_field"] is not None else "?"
        facts_lines.append(
            f"{h['mark']}{h['name']}: 勝率{h['p_win_pct']}% 複勝圏{h['p_top3_pct']}% "
            f"対戦RTG{h['elo']}(平均比{vs}) Glicko{h['glicko_mu']}±{h['glicko_rd']} "
            f"脚質{h['kyaku']} 予想{h['yoso_odds']}倍 実{h['real_odds']}倍 EV{h['ev']}。")
    if ana:
        facts_lines.append("特選穴馬データ: " + ana.split("※")[0])
    facts = "\n".join(facts_lines) + ("\n\n上記の事実だけで、◎〇▲の短評と特選穴馬の一言を書いて。"
                                       "穴は的中率低い前提で煽らずに。")
    narrative = ollama_narrative(facts)

    return {"race": race, "horses": horses_by_rank, "comments_top3": comments,
            "ana_pick": ana, "llm_narrative": narrative, "result_for_demo": result}


def render(card):
    r = card["race"]; L = []
    L.append("=" * 76)
    L.append(f" {r['name']}  ({r['venue']}{r['surface']}{r['distance']}m / {r['class']} / {r['field_size']}頭)")
    L.append(f" レースレベル(ELO) {r['race_level_elo']}  ｜  {r['baba']}")
    if r.get("baba_bias"):
        L.append(f" 馬場傾向: {r['baba_bias']['line']}")
    L.append(f" 展開: {r['pace']} (逃げ先行 {r['n_front_runners']}頭)")
    L.append("=" * 76)
    L.append(f"{'印':2s}{'馬番':>3s} {'馬名':14s} {'勝率':>6s} {'複勝':>6s} {'予想':>6s} {'基準':>6s} {'実':>6s} {'EV':>5s} {'RTG':>5s}{'vs':>6s} {'脚質':4s}{'位置':>8s}")
    for h in card["horses"]:
        nm = (h["name"][:13] + "…") if len(str(h["name"])) > 14 else h["name"]
        ro = f"{h['real_odds']}" if h["real_odds"] else "-"
        ev = f"{h['ev']}" if h["ev"] else "-"
        vs = f"{h['elo_vs_field']:+.1f}" if h["elo_vs_field"] is not None else "-"
        pos = f"{h['pos_1c']}→{h['pos_4c']}" if h["pos_1c"] else "-"
        buy = "◀買" if h["buy"] else ""
        L.append(f"{h['mark']:2s}{h['ban']:>3d} {nm:14s} {h['p_win_pct']:>5.1f}% {h['p_top3_pct']:>5.1f}% "
                 f"{str(h['yoso_odds']):>6s} {str(h['kijun_odds']):>6s} {ro:>6s} {ev:>5s} "
                 f"{str(h['elo']):>5s}{vs:>6s} {h['kyaku']:4s}{pos:>8s} {buy}")
    if card.get("llm_narrative"):
        L.append("\n【AI解説(Ollama)】")
        for ln in card["llm_narrative"].splitlines():
            if ln.strip():
                L.append("  " + ln.strip())
        L.append("\n  ── 構造化テンプレ(根拠数値) ──")
    else:
        L.append("\n【印上位3頭の解説】(テンプレ; Ollama未起動)")
    for c in card["comments_top3"]:
        L.append("  ・" + c)
    if card["ana_pick"]:
        L.append("\n【特選穴馬】")
        L.append("  ・" + card["ana_pick"])
    if card["result_for_demo"]:
        rr = card["result_for_demo"]
        L.append(f"\n（参考・確定結果）1着{rr['win_ban']}番 2着{rr['plc_ban']}番 3着{rr['sho_ban']}番 / 単勝{rr['tansho_pay']}円")
    return "\n".join(L)


def cards_for_date(date):
    """指定日(YYYYMMDD)の全レースのカードを返す(共有データは1回ロード)。"""
    sc = pd.read_parquet(SCORE_CACHE); sc[COL_RID] = sc[COL_RID].astype(str)
    sc = sc[sc[COL_RID].str.startswith(str(date))].copy()
    if sc.empty:
        return []
    sc["rid16"] = sc[COL_RID].str.replace(r"\D", "", regex=True).str[:16]
    sc["ban"] = pd.to_numeric(sc[COL_BAN], errors="coerce").fillna(0).astype(int)
    sc = join_feats(sc)
    shared = load_shared()
    info_df = load_master_subset(date)
    cards = []
    for rid, gg in sc.groupby(COL_RID, sort=False):
        try:
            cards.append(build(str(rid), gg, shared, info_df))
        except Exception as e:
            log(f"  [skip {rid}] {e}")
    return cards


def main():
    rid_arg = sys.argv[1] if len(sys.argv) > 1 else None
    rid, g = load_race(rid_arg)
    g = join_feats(g)
    card = build(rid, g)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / f"{rid}.json").write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    print(render(card))
    log(f"\n[saved] {OUTDIR / (rid + '.json')}")


if __name__ == "__main__":
    main()
