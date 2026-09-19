# -*- coding: utf-8 -*-
"""
live_history.py — 馬IDが無い2026年データを、実際に完了したレースの結果だけで時点安全に
追跡するための共通基盤 (spec EXP05-F §2, §6, §4)
=====================================================================================
週次CSV/kekka CSVには血統登録番号が無く馬名のみ。そこで:
  - 2025年末までの履歴(hid=血統登録番号, master_v2)は一切書き換えない
  - 2026年に入ってから実際に確定した結果 (data/kekka/*.csv に存在するレース) だけを
    「対象馬の2025年末時点hid (名前一致で解決) または新馬用の合成キー NEW:<馬名>」で
    追跡し、chain (historical + 2026確定分) を1本につなげる
  - target race自身の結果は絶対に使わない (as_of_date より後のkekkaは除外)

これにより Weng-Lin (dyn_skill) の更新も、陣営選択(C2)の「実際の前走」を使った差分計算も、
2026シーズン中に確定したレース結果を正しく反映できる (2025年末で凍結する旧方式を置き換える)。

2026-09-19 v3更新: 馬名だけの一致 (v2) は、同じ馬名でも履歴なので実際は本番の
serve_history_feats._HistoryIndex (data/_horse_history.parquet ベース、種牡馬・生年での
曖昧回避つき) を通した方が大幅に解決率が上がることが判明した
(analysis/mcond/exp05_forward_shadow/dyn_skill_resolution_audit.py の実測: 履歴を持つ馬の
単純名前一致での解決率は65.0%、うち34%が実際は解決可能なのに単純一致に失敗していた)。
そのため identity 解決は _HistoryIndex.resolve() (種牡馬+生年での曖昧回避つき) を使う方式に
刷新した。既知の限界: 種牡馬・生年ともに不明な場合や、種牡馬・生年ともに一致する同名馬が
複数いる場合は "ambiguous" として解決を諦める (安全側、誤った馬の状態を引き継がない)。
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE))
from grade_feats import class_name_to_ord  # noqa: E402
from predict_weekly import (RACE_COLS, HORSE_COLS_33, HORSE_COLS_46, HORSE_COLS_48,  # noqa: E402
                            HORSE_COLS_49, HORSE_COLS_99, COLUMN_MAP)
from serve_history_feats import _load as _load_history_index, _clean_name  # noqa: E402


def ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """export_weekly_marks.ensure_date_column と同一定義 (複製、重いモジュールの
    importを避けるため)。race_meta が「日付」を参照するため YYYYMMDD 文字列列を作る。"""
    if "日付" in df.columns and df["日付"].astype(str).str.len().min() == 8:
        return df
    if "日付S" in df.columns:
        def _to_yyyymmdd(s):
            try:
                parts = [int(x) for x in str(s).split(".")]
                return "{:04d}{:02d}{:02d}".format(*parts)
            except Exception:
                return ""
        df = df.copy()
        df["日付"] = df["日付S"].apply(_to_yyyymmdd)
    return df

MASTER = BASE / "data/master_v2_20130105-20251228.csv"
WEEKLY_DIR = BASE / "data/weekly"
KEKKA_DIR = BASE / "data/kekka"

_identity_cache = None


def parse_csv_light(path: Path) -> pd.DataFrame:
    """predict_weekly.parse_csv の列レイアウト定義 (RACE_COLS/HORSE_COLS_*/COLUMN_MAP、
    import して再利用しここでは複製しない) だけを使い、kako5/坂路/WC/hosei/着度数といった
    重い外部結合を一切せずに軽量パースする。chain構築 (馬名・場所・芝ダ・距離・クラス・
    騎手・調教師・馬番・レースID) だけが目的で、これらの結合は不要なため
    (predict_weekly.parse_csv をそのまま72週分回すと1週10秒超×72で非現実的に遅い)。"""
    for enc in ("cp932", "shift_jis", "utf-8"):
        try:
            text = path.read_bytes().decode(enc)
            break
        except Exception:
            continue
    else:
        return pd.DataFrame()

    layouts = {33: HORSE_COLS_33, 46: HORSE_COLS_46, 48: HORSE_COLS_48,
              49: HORSE_COLS_49, 99: HORSE_COLS_99}
    races: list[dict] = []
    current_race: dict | None = None
    for line in text.splitlines():
        cols = line.split(",")
        if cols[0] in ("レースID(新)", "枠番", "番", ""):
            continue
        if len(cols) == 19:
            current_race = dict(zip(RACE_COLS, cols))
        elif len(cols) in layouts and current_race:
            horse = dict(zip(layouts[len(cols)], cols))
            horse.update(current_race)
            races.append(horse)
    if not races:
        return pd.DataFrame()
    df = pd.DataFrame(races).rename(columns=COLUMN_MAP)
    df["レースID(新/馬番無)"] = df["レースID(新)"].astype(str).str[:16]
    return df


def name_to_hid_2025() -> dict[str, str]:
    """馬名 -> 2025年末までの最新出走行のhid (単純な馬名一致、v2の実装)。
    dyn_skill_resolution_audit.py が「より頑健な解決との比較基準(naive)」として使う以外では
    現在使われていない (resolve_idents を使うこと)。"""
    global _identity_cache
    if _identity_cache is not None:
        return _identity_cache
    m = pd.read_csv(MASTER, encoding="utf-8-sig", low_memory=False,
                    usecols=["日付", "血統登録番号", "馬名"])
    m["date"] = pd.to_datetime(m["日付"].astype(str), format="%Y%m%d", errors="coerce")
    m = m.dropna(subset=["血統登録番号", "馬名", "date"])
    last = m.sort_values("date").groupby("馬名")["血統登録番号"].last()
    _identity_cache = {k: str(v) for k, v in last.astype(str).to_dict().items()}
    return _identity_cache


_hist_index_cache = None


def resolve_idents(df: pd.DataFrame, race_year: int) -> tuple[pd.Series, pd.Series]:
    """serve_history_feats._HistoryIndex (data/_horse_history.parquet、種牡馬+生年での
    曖昧回避つき) を使ってidentを解決する。df は 馬名・種牡馬・年齢 列を持つこと。
    戻り値: (idents, statuses)。ident は 解決成功時=str(ped_id)、正当な初出走・
    解決失敗(ambiguous)時=NEW:<馬名> とする(どちらも「今シーズンの実績を独立に積み上げる」
    という結果自体は変わらないが、statusesで区別できる: "hit"/"new"/"ambiguous")。"""
    global _hist_index_cache
    if _hist_index_cache is None:
        _hist_index_cache = _load_history_index(BASE)
    idx, _maps, _meta = _hist_index_cache

    names = df.get("馬名", pd.Series([""] * len(df), index=df.index)).map(_clean_name)
    sires = df.get("種牡馬", pd.Series([""] * len(df), index=df.index)).map(_clean_name)
    ages = pd.to_numeric(df.get("年齢"), errors="coerce")
    birth_years = (race_year - ages).where(ages.notna())

    idents = pd.Series(index=df.index, dtype=object)
    statuses = pd.Series(index=df.index, dtype=object)
    for i in df.index:
        name = names.loc[i]
        sire = sires.loc[i]
        by = birth_years.loc[i]
        by_arg = int(by) if pd.notna(by) else None
        ent, status = idx.resolve(name, sire, by_arg)
        statuses.loc[i] = status
        idents.loc[i] = str(ent["ped_id"]) if ent is not None else f"NEW:{name}"
    return idents, statuses


def _kekka_finish_map(date_str: str) -> dict[tuple[str, int], float]:
    path = KEKKA_DIR / f"{date_str}.csv"
    if not path.exists():
        return {}
    for enc in ("cp932", "utf-8-sig", "utf-8"):
        try:
            k = pd.read_csv(path, encoding=enc, low_memory=False)
            break
        except Exception:
            continue
    else:
        return {}
    rid_col = next((c for c in k.columns if "レースID" in c), None)
    ban_col = "馬番" if "馬番" in k.columns else None
    fin_col = next((c for c in k.columns if "確定着順" in c), None) or \
        ("着順" if "着順" in k.columns else None)
    if not (rid_col and ban_col and fin_col):
        return {}
    k["_rid16"] = k[rid_col].astype(str).str.replace(r"\D", "", regex=True).str[:16]
    k["_ban"] = pd.to_numeric(k[ban_col], errors="coerce")
    k["_fin"] = pd.to_numeric(k[fin_col], errors="coerce")
    k = k.dropna(subset=["_ban", "_fin"])
    return {(r["_rid16"], int(r["_ban"])): r["_fin"] for _, r in k.iterrows()}


def available_2026_result_dates(as_of_date: str | None = None) -> list[str]:
    dates = sorted(p.stem for p in KEKKA_DIR.glob("2026*.csv"))
    if as_of_date is not None:
        dates = [d for d in dates if d <= as_of_date]
    return dates


def build_2026_log(as_of_date: str | None = None) -> pd.DataFrame:
    """確定済み(kekka存在)の2026年レースだけを対象に、chain構築用の行を作る。
    1行=1頭。ident(=解決hid/ped_idまたはNEW:name)・date・rid16・ban・venue・surface・dist・
    cls_ord・jockey_code・trainer_code・fin・name を持つ。identは resolve_idents
    (_HistoryIndex、種牡馬+生年で曖昧回避) で解決する。"""
    dates = available_2026_result_dates(as_of_date)
    rows = []
    for d in dates:
        wpath = WEEKLY_DIR / f"{d}.csv"
        if not wpath.exists():
            continue
        try:
            df = parse_csv_light(wpath)
        except Exception:
            continue
        if df.empty:
            continue
        finmap = _kekka_finish_map(d)
        if not finmap:
            continue
        df = ensure_date_column(df)
        try:
            from serve_history_feats import fill_history_features
            fill_history_features(df)
        except Exception:
            pass
        if "芝・ダ" in df.columns:
            df["芝・ダ"] = df["芝・ダ"].astype(str).replace({"ダート": "ダ"})
        race_year = int(d[:4])
        idents, statuses = resolve_idents(df, race_year)
        rid_col = "レースID(新/馬番無)" if "レースID(新/馬番無)" in df.columns else "レースID(新)"
        rid16 = df[rid_col].astype(str).str[:16]
        ban = pd.to_numeric(df["馬番"], errors="coerce")
        for i in df.index:
            r16, b = rid16.loc[i], ban.loc[i]
            if pd.isna(b):
                continue
            b = int(b)
            fin = finmap.get((r16, b))
            if fin is None or fin < 1:
                continue  # 出走なし/取消/中止 (結果未確定含む)
            name = str(df.get("馬名", pd.Series(dtype=str)).loc[i]) if "馬名" in df.columns else ""
            ident = idents.loc[i]
            rows.append({
                "ident": ident, "name": name, "date": pd.Timestamp(d),
                "rid16": r16, "ban": b,
                "venue": df.get("場所", pd.Series(dtype=str)).get(i),
                "surface": df.get("芝・ダ", pd.Series(dtype=str)).get(i),
                "dist": pd.to_numeric(pd.Series([df.get("距離", pd.Series(dtype=float)).get(i)]),
                                      errors="coerce").iloc[0],
                "cls_ord": class_name_to_ord(df.get("クラス名", pd.Series(dtype=str)).get(i)),
                "jockey": str(df.get("騎手コード", pd.Series(dtype=str)).get(i, "")),
                "trainer": str(df.get("調教師コード", pd.Series(dtype=str)).get(i, "")),
                "fin": float(fin),
            })
    return pd.DataFrame(rows)


def build_historical_chain() -> pd.DataFrame:
    """master_v2 (2013-2025, hid直接) をchain共通スキーマへ変換。"""
    df = pd.read_csv(MASTER, encoding="utf-8-sig", low_memory=False,
                     usecols=["日付", "レースID(新)", "血統登録番号", "馬番", "着順",
                              "場所", "芝・ダ", "距離", "クラス名", "騎手コード", "調教師コード"])
    df["date"] = pd.to_datetime(df["日付"].astype(str), format="%Y%m%d", errors="coerce")
    df["rid16"] = df["レースID(新)"].astype(str).str[:16]
    df["ban"] = pd.to_numeric(df["馬番"], errors="coerce")
    df["fin"] = pd.to_numeric(df["着順"], errors="coerce")
    df = df.dropna(subset=["ban", "fin", "血統登録番号", "date"])
    df = df[df["fin"] >= 1]
    df["ban"] = df["ban"].astype(int)
    df["ident"] = df["血統登録番号"].astype(str)
    df["cls_ord"] = df["クラス名"].map(class_name_to_ord)
    df["jockey"] = df["騎手コード"].astype(str)
    df["trainer"] = df["調教師コード"].astype(str)
    return df[["ident", "date", "rid16", "ban", "場所", "芝・ダ", "距離", "cls_ord",
              "jockey", "trainer", "fin"]].rename(columns={"場所": "venue", "芝・ダ": "surface",
                                                            "距離": "dist"})


def build_combined_chain(as_of_date: str | None = None) -> pd.DataFrame:
    hist = build_historical_chain()
    live = build_2026_log(as_of_date)
    cols = ["ident", "date", "rid16", "ban", "venue", "surface", "dist", "cls_ord", "jockey", "trainer", "fin"]
    chain = pd.concat([hist[cols], live[cols] if len(live) else pd.DataFrame(columns=cols)],
                      ignore_index=True)
    return chain.sort_values(["ident", "date", "rid16"]).reset_index(drop=True)


# --------------------------------------------------------------------------- C2 (陣営選択の生特徴)
def compute_c2_from_chain(target: pd.DataFrame, chain: pd.DataFrame,
                          k_jockey: float = 50.0) -> pd.DataFrame:
    """target: ident/date/venue/surface/dist/cls_ord/jockey/trainer列を持つ今週のカード。
    chainの各identについて「実際の直前レース」を取り、exp01_choice_dev.build_featuresと
    同じ定義 (K_JOCKEY=50 の平滑化含む) で8列全てのraw_*を計算する
    (時点安全: chainはtarget日より前の確定結果のみ)。"""
    last = (chain.sort_values(["ident", "date"]).groupby("ident", sort=False)
           .agg(prev_date=("date", "last"), prev_venue=("venue", "last"),
                prev_surface=("surface", "last"), prev_dist=("dist", "last"),
                prev_cls_ord=("cls_ord", "last"), prev_jockey=("jockey", "last")))
    t = target.merge(last, left_on="ident", right_index=True, how="left")
    int_days = (t["date"] - t["prev_date"]).dt.days
    out = pd.DataFrame(index=target.index)
    out["raw_log_int"] = np.log(int_days.clip(lower=1))
    out["raw_dist_chg"] = t["dist"] - t["prev_dist"]
    out["raw_venue_chg"] = (t["venue"].astype(str) != t["prev_venue"].astype(str)).astype(float)
    out["raw_venue_chg"] = out["raw_venue_chg"].where(t["prev_venue"].notna())
    out["raw_surface_chg"] = (t["surface"].astype(str) != t["prev_surface"].astype(str)).astype(float)
    out["raw_surface_chg"] = out["raw_surface_chg"].where(t["prev_surface"].notna())
    d = t["cls_ord"] - t["prev_cls_ord"]
    out["raw_cls_chg"] = np.sign(d)
    out["raw_jockey_same"] = (t["jockey"].astype(str) == t["prev_jockey"].astype(str)).astype(float)
    out["raw_jockey_same"] = out["raw_jockey_same"].where(t["prev_jockey"].notna())

    # ---- 騎手の格の変化 (raw_jq_delta) と 騎手×調教師の組合せ頻度 (raw_jt_pair) ----
    c = chain.dropna(subset=["jockey"]).copy()
    c = c[c["jockey"].astype(str).str.len() > 0]
    c["one"] = 1.0
    c["top3"] = (c["fin"] <= 3).astype(float)

    def asof_by(frame, key, cols):
        daily = frame.groupby([key, "date"])[cols].sum().sort_index()
        prior = daily.groupby(level=0).cumsum() - daily
        return prior.add_prefix("prior_")

    jq = asof_by(c, "jockey", ["top3", "one"]).reset_index()
    gdaily = c.groupby("date")[["top3", "one"]].sum().sort_index()
    gprior = (gdaily.cumsum() - gdaily).add_prefix("gprior_").reset_index()
    jq = jq.merge(gprior, on="date")
    jq["jq"] = (jq["prior_top3"] + k_jockey * jq["gprior_top3"] / jq["gprior_one"].clip(lower=1)) / \
              (jq["prior_one"] + k_jockey)
    jqd = jq.set_index(["jockey", "date"])["jq"]
    gmean = (gprior.set_index("date")["gprior_top3"] / gprior.set_index("date")["gprior_one"].clip(lower=1))
    global_mean_latest = float(gmean.iloc[-1]) if len(gmean) else np.nan

    # jq_cur: 「今日時点」の騎手の格。chainは既にtarget日より前の確定結果だけなので、
    # 全chainを使ったcareer累計(=場代最新日の累計値, 同日除外の"prior"ではなく累計込みの
    # 値)がそのまま「今日時点で既知」の値になる (exp01のjq_prevと同じ平滑化式、
    # 対象日が持つ(jockey,date)ペアがchainに無いのでdate一致reindexは使えない)。
    total = c.groupby("jockey")[["top3", "one"]].sum()
    jq_total = (total["top3"] + k_jockey * global_mean_latest) / (total["one"] + k_jockey)
    jq_cur = t["jockey"].astype(str).map(jq_total).to_numpy()

    # exp01_choice_dev.build_features と同じ定義: 前走騎手の格を「対象レース当日時点」で
    # 引く (v["jq_prev"]=jqd.reindex([prev_jockey, date])、無ければgmean(当日)で埋める)。
    # chainは確定済み結果のみで対象日(今日)自体のレコードを持たないため、この参照は
    # 定義上ほぼ必ずgmean_at_date側にフォールバックする(元のバッチ実装でも、前走騎手が
    # 偶然同日に別レースへ騎乗していない限り同じフォールバックが起きる = 元定義どおりの挙動)。
    jq_prev = jqd.reindex(pd.MultiIndex.from_arrays([t["prev_jockey"].astype(str), t["date"]])).to_numpy()
    jq_prev = np.where(pd.isna(jq_prev), global_mean_latest, jq_prev)
    out["raw_jq_delta"] = jq_cur - jq_prev
    out["raw_jq_delta"] = np.where(t["prev_date"].isna(), np.nan, out["raw_jq_delta"])

    trainer = t.get("trainer", pd.Series("", index=t.index)).astype(str)
    # raw_jt_pair も raw_jq_delta の jq_cur と同じ理由 (chainはtarget日を含まない) で
    # 日付一致reindexではなくchain全体の累計(=target日時点で既知の値)を使う。
    pair_col = c["jockey"].astype(str) + "|" + c["trainer"].astype(str)
    pair_total = c.assign(pair=pair_col).groupby("pair")["one"].sum()
    trainer_total = c.groupby("trainer")["one"].sum()
    pair_t = t["jockey"].astype(str) + "|" + trainer
    pn = pair_t.map(pair_total).fillna(0).to_numpy()
    tn = trainer.map(trainer_total).fillna(0).to_numpy()
    out["raw_jt_pair"] = (pn + 1.0) / (tn + 10.0)
    return out


# --------------------------------------------------------------------------- C3 (dyn_skill, 時点安全な逐次更新)
def compute_dyn_skill_live(chain: pd.DataFrame, hp_beta: float = 2.0833,
                           hp_tau2_per_day: float = 0.005) -> dict:
    """chain (identキー, 日付昇順) に対しWeng-Lin更新を逐次実行し、chainの最終日までの
    状態 {ident: {mu, var, n, last_date}} を返す (2026年の確定済みレースも反映される、
    2025年末固定の旧方式を置き換える)。"""
    sys.path.insert(0, str(BASE))
    from analysis.mcond.exp02_dynamic_skill_dev.dyn_skill import State, wl_update, Hyper
    hp = Hyper(beta=hp_beta, tau2_per_day=hp_tau2_per_day)
    st = State(hp)
    for day, dd in chain.groupby("date", sort=True):
        pending = []
        for rid, g in dd.groupby("rid16", sort=False):
            idents = g["ident"].to_numpy()
            pri = [st.prior(h, day) for h in idents]
            mu = np.array([p[0] for p in pri])
            var = np.array([p[1] for p in pri])
            fin = g["fin"].to_numpy(float)
            pending.append((idents, mu, var, fin))
        for idents, mu, var, fin in pending:
            nm, nv, _ = wl_update(mu, var, fin, hp.beta)
            for i, h in enumerate(idents):
                st.chg[h] = float(nm[i] - mu[i])
                st.g_mu[h], st.g_var[h] = float(nm[i]), float(nv[i])
                st.last[h] = day
                st.n[h] = st.n.get(h, 0) + 1
    return {"g_mu": dict(st.g_mu), "g_var": dict(st.g_var), "n": dict(st.n),
           "last": {k: v.strftime("%Y-%m-%d") for k, v in st.last.items()}}
