# -*- coding: utf-8 -*-
"""
marks_shap.py
=============
v6 印モデル (LightGBM LambdaRank) の SHAP 寄与を bundle.json 埋込用に整形する。

役割:
  - export_marks_json.export_race() から呼ばれ、印が付いた馬 (◎〇▲△△) に
    「なぜ上位スコアか」を特徴寄与トップ-k として付与する (horse["why"])。
  - Cowork / NiceGUI は feat (正準列名) + label (日本語ヒント) + value (元値) +
    contrib (符号付き寄与) を見て、印の根拠を言語化できる。

設計上の不変条件:
  - `feat` は常にモデルの正準特徴名 (ground truth)。`label` は読みやすさのための
    ヒントに過ぎず、ラベルが多少ニュアンスを外しても feat で曖昧さは解消できる。
  - `value` は **エンコード前の元値** (g_rows から復元)。LabelEncoder の整数や
    -9999 fill ではなく、人が読める生の値を出す。
  - SHAP が使えない環境 (shap 未導入 / explainer 構築失敗) でも、呼び出し側で
    例外を握り潰せば bundle 生成自体は従来どおり成功する。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 英語名・略号特徴の日本語ラベル (自明な日本語列はフォールバックで feat 名をそのまま使う)
FEAT_LABELS: dict[str, str] = {
    # --- 騎手・厩舎・馬 近走複勝率 (rolling) ---
    "jockey_fuku30": "騎手 複勝率(直近30日)",
    "jockey_fuku90": "騎手 複勝率(直近90日)",
    "trainer_fuku30": "厩舎 複勝率(直近30日)",
    "trainer_fuku90": "厩舎 複勝率(直近90日)",
    "horse_fuku10": "当馬 複勝率(直近10走)",
    "horse_fuku30": "当馬 複勝率(直近30走)",
    # --- レース内相対 ---
    "prev_pos_rel": "前走 相対通過順(レース内)",
    "closing_power": "終い脚力(上がり相対)",
    # --- 過去5走サマリ ---
    "kako5_avg_pos": "過去5走 平均通過順",
    "kako5_std_pos": "過去5走 通過順ばらつき",
    "kako5_best_pos": "過去5走 最高通過順",
    "kako5_avg_ninki": "過去5走 平均人気",
    "kako5_pos_vs_ninki": "過去5走 通過順vs人気",
    "kako5_avg_agari3f": "過去5走 平均上り3F",
    "kako5_best_agari3f": "過去5走 最速上り3F",
    "kako5_same_td_ratio": "過去5走 同芝ダ比率",
    "kako5_same_dist_ratio": "過去5走 同距離比率",
    "kako5_same_place_ratio": "過去5走 同競馬場比率",
    "kako5_pos_trend": "過去5走 通過順トレンド",
    "kako5_race_count": "過去5走 該当走数",
    "kako5_expected_good_count": "過去5走 人気どおり好走数",
    "kako5_upset_good_count": "過去5走 人気薄好走数",
    "kako5_hidden_good_count": "過去5走 隠れ好走数",
    "kako5_same_cond_best_pos": "過去5走 同条件最高通過順",
    # --- 同条件 履歴 (as-of, 自レース除外) ---
    "hist_same_cond_best_pos": "同条件 履歴最高通過順",
    "hist_same_cond_top3_rate": "同条件 複勝圏率(履歴)",
    "hist_same_cond_count": "同条件 出走数(履歴)",
    "hist_same_place_best_pos": "同競馬場 履歴最高通過順",
    # --- 補正タイム ---
    "prev_hosei": "前走 補正タイム",
    "prev_hosei9": "前走 補正タイム(9F基準)",
    # --- 坂路追い切り (Time1=4F合計, Time4=ラスト1F) ---
    "trnH_Time1": "坂路 4F合計タイム",
    "trnH_Time2": "坂路 ラスト3Fタイム",
    "trnH_Time3": "坂路 ラスト2Fタイム",
    "trnH_Time4": "坂路 ラスト1Fタイム",
    "trnH_Lap1": "坂路 最終1Fラップ(200m)",
    "trnH_Lap2": "坂路 2F目ラップ",
    "trnH_Lap3": "坂路 3F目ラップ",
    "trnH_Lap4": "坂路 4F目ラップ",
    "trnH_days_ago": "坂路 追い切り経過日数",
    # --- ウッドチップ追い切り ---
    "trnW_5F": "ウッド 5Fタイム",
    "trnW_4F": "ウッド 4Fタイム",
    "trnW_3F": "ウッド 3Fタイム",
    "trnW_Lap1": "ウッド 最終1Fラップ",
    "trnW_Lap2": "ウッド 2F目ラップ",
    "trnW_Lap3": "ウッド 3F目ラップ",
    "trnW_days_ago": "ウッド 追い切り経過日数",
    # --- 同コース帯 (場所|芝ダ|距離帯) 成績 ---
    "course_n_prev": "当馬 同コース帯 出走数",
    "course_win_rate": "当馬 同コース帯 勝率",
    "course_top3_rate": "当馬 同コース帯 複勝圏率",
    "jockey_n_prev": "騎手 同コース帯 出走数",
    "jockey_win_rate": "騎手 同コース帯 勝率",
    "jockey_top3_rate": "騎手 同コース帯 複勝圏率",
    # --- 斤量・間隔まわり ---
    "馬齢斤量差": "馬齢基準との斤量差",
    "斤量体重比": "斤量/馬体重比",
    "休み明け～戦目": "休み明け~戦目",
    "間隔": "レース間隔(週)",
}


def build_explainer(model):
    """LightGBM Booster から TreeExplainer を構築 (1 回だけ呼び、レース間で使い回す)。"""
    import shap
    return shap.TreeExplainer(model)


def label_for(feat: str) -> str:
    """特徴の日本語ラベル。未登録なら正準名をそのまま返す。"""
    return FEAT_LABELS.get(feat, feat)


def _fmt_value(v):
    """エンコード前の元値を人が読める形に整形。NaN / "__NaN__" / 欠損 → None。"""
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        return s if s and s != "__NaN__" else None
    try:
        f = float(v)
    except (TypeError, ValueError):
        s = str(v).strip()
        return s if s and s != "__NaN__" else None
    if not np.isfinite(f):
        return None
    # 整数で表せるなら int に (通過順・出走数・経過日数などをきれいに)
    if abs(f) < 1e12 and float(int(f)) == f:
        return int(f)
    return round(f, 3)


def _to_2d_shap(sv) -> np.ndarray:
    """shap_values の返り値を (n, n_feats) の 2D に正規化。

    LightGBM の単一出力 (ranking/regression) なら通常 2D。多出力で list / 3D が
    返る稀なケースに備えて先頭出力 (or 合算) に畳む。
    """
    arr = np.asarray(sv)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        # (n_outputs, n, n_feats) or (n, n_feats, n_outputs)
        if arr.shape[0] == 1:
            return arr[0]
        if arr.shape[-1] == 1:
            return arr[..., 0]
        # 想定外: 出力方向を合算して 2D に
        return arr.sum(axis=0) if arr.shape[0] < arr.shape[-1] else arr.sum(axis=-1)
    raise ValueError(f"unexpected shap_values ndim={arr.ndim}")


def race_contribs(explainer, X, feats, g_rows: pd.DataFrame, topk: int = 6):
    """1 レース分の SHAP 寄与を馬ごとに整形して返す。

    Args:
        explainer: build_explainer() で作った TreeExplainer (使い回し可)。
        X:        モデル入力と同一のエンコード済み行列 (n, n_feats), np.ndarray。
        feats:    特徴名リスト (X の列順と一致)。
        g_rows:   X と **行順が一致** した元 DataFrame (値復元用)。
        topk:     馬ごとに残す寄与の数 (|contrib| 降順)。

    Returns:
        list (len n)。各要素は top-k の dict リスト:
            [{"feat", "label", "value", "contrib"}, ...]
        contrib > 0 はスコアを押し上げる方向 (上位=印に効く)。
    """
    if topk <= 0:
        return [[] for _ in range(len(g_rows))]
    sv = _to_2d_shap(explainer.shap_values(X))
    n, m = sv.shape
    feats = list(feats)
    out: list[list[dict]] = []
    for i in range(n):
        row = sv[i]
        order = np.argsort(-np.abs(row))[:topk]
        items = []
        grow = g_rows.iloc[i]
        for j in order:
            f = feats[j]
            raw = grow.get(f) if f in g_rows.columns else None
            try:
                val = _fmt_value(raw)
            except Exception:
                val = None
            items.append({
                "feat": f,
                "label": label_for(f),
                "value": val,
                "contrib": round(float(row[j]), 3),
            })
        out.append(items)
    return out
