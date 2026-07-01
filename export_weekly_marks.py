"""
export_weekly_marks.py
======================
週末出走表 CSV (data/weekly/YYYYMMDD.csv) から
Cowork 入力用の印 JSON を出力する。

役割:
  - PyCaLiAI 側: 印付け (◎〇▲△△) + 確率 + race_confidence を JSON 化
  - Cowork 側 (Anthropic Desktop App): JSON を読んで馬券構築

使い方:
    # 単週分を一括出力 (race ごとに 1 ファイル + 全 race を 1 ファイルにまとめた bundle)
    python export_weekly_marks.py --csv data/weekly/20260426.csv

    # モデル指定
    python export_weekly_marks.py --csv data/weekly/20260426.csv --model v5

    # 出力先指定
    python export_weekly_marks.py --csv data/weekly/20260426.csv \
        --out-dir reports/cowork_input/20260426/

出力:
    reports/cowork_input/{YYYYMMDD}/{race_id}.json   ... 1 race / 1 file
    reports/cowork_input/{YYYYMMDD}_bundle.json       ... 全 race を 1 file に集約

互換性: バックテスト用 reports/marks_v5/ と同一の JSON スキーマ
        (docs/marks_schema.md を参照)
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE = Path(__file__).parent

# 既存ロジックを再利用 (parse_csv で週次CSVを DF 化, export_race で JSON 構築)
from predict_weekly import parse_csv
import backtest_pl_ev as be
from backtest_pl_ev import COL_RID, COL_BAN
from export_marks_json import export_race


def build_tansho_idx_from_weekly(df: pd.DataFrame) -> dict:
    """週次CSV の "単勝" 列から tansho_idx[(rid_s, ban)] = float を構築。"""
    tansho_idx: dict = {}
    if "単勝" not in df.columns:
        logger.warning("週次CSV に '単勝' 列がないため tansho_odds は null になります。")
        return tansho_idx
    for _, r in df.iterrows():
        try:
            rid_s = str(r[COL_RID])[:16]
            ban = int(r[COL_BAN])
            v_raw = r["単勝"]
            if v_raw is None or pd.isna(v_raw):
                continue
            v = float(v_raw)
            if v <= 0:
                continue
            tansho_idx[(rid_s, ban)] = v
        except Exception:
            continue
    logger.info(f"単勝オッズ取得 (weekly CSV): {len(tansho_idx):,} 馬")
    return tansho_idx


def build_odds_from_od_csv(date_str: str):
    """data/odds/OD{YYMMDD}.CSV があれば読んで (tansho_idx, fuku_idx, umaren_idx) を返す。

    OD CSV (TARGET 形式) は単勝・複勝下限/上限・馬連 matrix を含むため、
    weekly CSV のオッズより精度が高い (複勝・馬連は weekly に存在しない)。

    Returns:
        (tansho_idx, fuku_idx, umaren_idx) tuple, or None if file missing / parse failed.
        - tansho_idx[(rid_s, ban)] = float
        - fuku_idx[(rid_s, ban)]   = (low, high)
        - umaren_idx[(rid_s, i, j)] = float  (i<j 対称)
    """
    yy = date_str[2:]  # 20260426 → 260426
    candidates = [
        BASE / "data" / "odds" / f"OD{yy}.CSV",
        BASE / "data" / "odds" / f"OD{yy}.csv",
    ]
    od_path = next((p for p in candidates if p.exists()), None)
    if od_path is None:
        logger.info(f"OD CSV 未配置: data/odds/OD{yy}.CSV (weekly CSV のオッズを使用)")
        return None

    try:
        from parse_od_csv import load_od_matrix_odds
        m = load_od_matrix_odds(od_path, date=date_str)
    except Exception as e:
        logger.warning(f"OD CSV 読み込み失敗: {od_path.name}: {e}")
        return None

    tansho_idx = m["tansho"]
    fuku_idx   = m["fukusho"]
    umaren_idx = m["umaren"]
    logger.info(
        f"OD CSV 取得: {od_path.name} → "
        f"単勝 {len(tansho_idx):,} 馬 / "
        f"複勝 {len(fuku_idx):,} 馬 / "
        f"馬連 {len(umaren_idx):,} 組"
    )
    return tansho_idx, fuku_idx, umaren_idx


def ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """race_meta が "日付" を参照するため YYYYMMDD 文字列列を必ず作る。"""
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True,
                    help="週次CSV (例: data/weekly/20260426.csv)")
    ap.add_argument("--model", default="v6",
                    help="モデル tag (default: v6 = 本番)。手動実行で誤って退役 v5 の "
                         "bundle を生成しないよう本番モデルを既定にする")
    ap.add_argument("--out-dir", default=None,
                    help="出力ディレクトリ "
                         "(default: reports/cowork_input/{YYYYMMDD}/)")
    ap.add_argument("--shap-topk", type=int, default=6,
                    help="印馬 (◎〇▲△△) に付与する SHAP 寄与の数 "
                         "(0 で無効化, default 6)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error(f"CSV not found: {csv_path}")
        return 1

    date_str = csv_path.stem  # YYYYMMDD 期待
    if not (date_str.isdigit() and len(date_str) == 8):
        logger.warning(f"ファイル名が YYYYMMDD でない: {date_str} "
                       "(date 推定不可、出力先指定推奨)")

    out_dir = Path(args.out_dir) if args.out_dir else (
        BASE / "reports" / "cowork_input" / date_str
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------ モデル & calibrator ロード ------
    tag = args.model
    be.MODEL_PKL = BASE / f"models/unified_rank_{tag}.pkl"
    be.CAL_PKL   = BASE / f"models/pl_calibrators_{tag}.pkl"
    # serve 条件 fit の calibrator があれば優先する (serve skew 対策)。
    # 本番のスコアは特徴欠損分布なので、フル特徴で fit した {tag}.pkl だと
    # ECE が複勝2.3倍/馬連1.8倍悪化する (reports/calibrators_v6_serve_eval.json:
    # serve fit で 複勝 -36% / 馬連 -29%)。offline 監査 (audit_marks 等) は
    # フル特徴スコアなので従来の {tag}.pkl を使い続けること。
    # 注意: serve で作れる特徴が変わったら build_pl_calibrators_serve.py で再 fit。
    serve_cal = BASE / f"models/pl_calibrators_{tag}_serve.pkl"
    if serve_cal.exists():
        be.CAL_PKL = serve_cal
    if not be.MODEL_PKL.exists():
        logger.error(f"モデル未存在: {be.MODEL_PKL}")
        return 1

    bundle = joblib.load(be.MODEL_PKL)
    model = bundle["model"]
    feats = bundle["feature_cols"]
    encs  = bundle["encoders"]
    logger.info(f"モデル: {be.MODEL_PKL.name} ({len(feats)} feats)")

    calibrators = None
    if be.CAL_PKL.exists():
        cal_bundle = joblib.load(be.CAL_PKL)
        calibrators = cal_bundle.get("calibrators", cal_bundle)
        logger.info(f"calibrator: {be.CAL_PKL.name}")
    else:
        logger.warning(f"calibrator 未存在: {be.CAL_PKL} → raw PL 確率で出力")

    # ------ SHAP explainer (印の根拠を bundle に埋込) ------
    # レース毎に作り直すと重いので 1 回だけ構築して使い回す。
    shap_explainer = None
    shap_topk = args.shap_topk
    if shap_topk > 0:
        try:
            from marks_shap import build_explainer
            shap_explainer = build_explainer(model)
            logger.info(f"SHAP explainer 構築 → 印馬に top-{shap_topk} 寄与 (why) を付与")
        except Exception as e:
            logger.warning(f"SHAP 無効化 (explainer 構築失敗: {e}) → why 埋込なし")
            shap_explainer = None

    # ------ class prior (経験 ◎〇▲△△ 信頼度) ------
    class_prior_path = BASE / "data" / f"class_prior_{tag}.json"
    class_prior_map: dict | None = None
    if class_prior_path.exists():
        try:
            with open(class_prior_path, encoding="utf-8") as f:
                prior_data = json.load(f)
            class_prior_map = prior_data.get("classes", {}) or None
            if class_prior_map:
                logger.info(
                    f"class_prior: {class_prior_path.name} "
                    f"({len(class_prior_map)} クラス、generated {prior_data.get('generated_at', '?')[:10]})"
                )
        except Exception as e:
            logger.warning(f"class_prior 読み込み失敗: {e}")
    else:
        logger.info(
            f"class_prior 未配置: {class_prior_path.name} "
            "→ bundle に prior 埋め込みなし "
            "(scripts/audit_marks_by_class.py を先に実行)"
        )

    # ------ CSV パース ------
    logger.info(f"weekly CSV パース: {csv_path}")
    df = parse_csv(csv_path)
    df = ensure_date_column(df)
    logger.info(f"パース結果: {len(df):,} 馬 / "
                f"{df[COL_RID].nunique():,} レース")

    # ------ serve skew 対策: 補正タイム + 調教の列名リネーム ------
    # parse_csv は補正を旧名 (前走補正/前走補9)、調教を旧名 (trn_hanro_*/trn_wc_*)
    # で作るが、v6 モデルは build_master_v2.py のリネーム後の名前
    # (prev_hosei*/trnH_*/trnW_*) を要求する。元データ・元列は同一
    # (補正=hosei CSV、調教=H/W CSV の Time1..Lap4 そのまま) なので定義は一致。
    # 列名を合わせないと下の「不足列補完」で -9999 に潰れて死ぬ。
    # serve_skew_eval.py: 補正 +2.97pt / 調教 +0.33pt 回収 (docs/audit_20260611.md)。
    # 既知の残差: 推論の調教 JOIN には 14 日カットオフがある (学習は無制限) ため
    # 14 日超の調教だけ欠損になる。坂路カバレッジ 93.9% で実害は小さく、
    # 欠損分布の差は serve 条件 fit calibrator が吸収する。
    _SERVE_RENAME = {
        "前走補正": "prev_hosei", "前走補9": "prev_hosei9",
        # 坂路 (H CSV: Time1=4F合計, Time2=3F, Time3=2F, Time4=1F)
        "trn_hanro_4f": "trnH_Time1", "trn_hanro_3f": "trnH_Time2",
        "trn_hanro_2f": "trnH_Time3", "trn_hanro_1f": "trnH_Time4",
        "trn_hanro_lap1": "trnH_Lap1", "trn_hanro_lap2": "trnH_Lap2",
        "trn_hanro_lap3": "trnH_Lap3", "trn_hanro_lap4": "trnH_Lap4",
        "trn_hanro_days": "trnH_days_ago",
        # WC (W CSV: 5F/4F/3F/Lap1-3)
        "trn_wc_5f": "trnW_5F", "trn_wc_4f": "trnW_4F", "trn_wc_3f": "trnW_3F",
        "trn_wc_lap1": "trnW_Lap1", "trn_wc_lap2": "trnW_Lap2",
        "trn_wc_lap3": "trnW_Lap3", "trn_wc_days": "trnW_days_ago",
    }
    _serve_rename = {k: v for k, v in _SERVE_RENAME.items()
                     if k in df.columns and v in feats and v not in df.columns}
    if _serve_rename:
        df = df.rename(columns=_serve_rename)
        revived = {v: df[v].notna().mean() * 100 for v in _serve_rename.values()}
        logger.info(f"[serve skew fix] {len(revived)} 列復活: " +
                    ", ".join(f"{k}={v:.0f}%" for k, v in sorted(revived.items())[:6]) +
                    (" ..." if len(revived) > 6 else ""))

    # ------ feats に含まれるが週次CSVにない列を NaN/空で補完 ------
    # 例: 騎手コード, hist_same_cond_*, trnH_*, trnW_*, course_*, jockey_*
    # export_race() 内で encoder 適用 + pd.to_numeric → NaN → -9999 fillna されるので
    # 形だけ揃える。
    cat_cols = set(encs.keys())
    missing = [c for c in feats if c not in df.columns]
    if missing:
        logger.warning(f"週次CSVに不足の {len(missing)} 列を補完: "
                       f"{missing[:10]}{'...' if len(missing) > 10 else ''}")
        for c in missing:
            if c in cat_cols:
                # カテゴリ列 → "__NaN__" 文字列 (encoder 側で未知値扱いに)
                df[c] = "__NaN__"
            else:
                df[c] = np.nan

    # ------ kako5 history + horse facts (advisor + 出走表 表示用) ------
    from kako5_summary import build_histories, build_horse_facts
    kako5_path = BASE / "data" / "kako5" / f"{date_str}.csv"
    histories = build_histories(kako5_path) if kako5_path.exists() else {}
    horse_facts = build_horse_facts(kako5_path) if kako5_path.exists() else {}
    if not histories:
        logger.warning(
            f"kako5 履歴未取得: {kako5_path.name} "
            "→ bundle に history 埋め込みなし (advisor 用材料減)"
        )
    if horse_facts:
        logger.info(f"horse facts (sex/age): {len(horse_facts)} 頭分")

    # ------ 血統 lookup (build_pedigree_data.py が生成、馬名 → sire/...) ------
    horse_pedigree_path = BASE / "data" / "horse_pedigree.json"
    pedigree_map: dict[str, dict] = {}
    if horse_pedigree_path.exists():
        try:
            with open(horse_pedigree_path, encoding="utf-8") as f:
                pedigree_data = json.load(f)
            pedigree_map = pedigree_data.get("horses", {})
            logger.info(
                f"horse_pedigree: {horse_pedigree_path.name} "
                f"({len(pedigree_map):,} 頭, generated {pedigree_data.get('generated_at','?')[:10]})"
            )
        except Exception as e:
            logger.warning(f"horse_pedigree 読み込み失敗: {e}")
    else:
        logger.info(
            "horse_pedigree 未配置: 血統分析タブが「該当馬不明」表示になる "
            "(python build_pedigree_data.py を先に実行)"
        )

    # オッズ: data/odds/OD{YYMMDD}.CSV (TARGET 形式) を優先、無ければ weekly CSV から
    od_result = build_odds_from_od_csv(date_str)
    if od_result is not None:
        tansho_idx, fuku_idx, umaren_idx = od_result
        logger.info("オッズソース: data/odds/OD CSV (単勝・複勝・馬連 matrix)")
    else:
        tansho_idx = build_tansho_idx_from_weekly(df)
        fuku_idx = {}  # 週次CSV に複勝オッズなし
        umaren_idx = {}  # 週次CSV に馬連オッズなし
        logger.info("オッズソース: weekly CSV (単勝のみ、複勝・馬連 は null)")

    # ------ race ごとに JSON 出力 ------
    n_done = 0
    n_skip = 0
    saved_paths: list[Path] = []
    for rid, g in df.groupby(COL_RID, sort=False):
        if len(g) < 5:
            n_skip += 1
            continue
        try:
            payload = export_race(rid, g, model, feats, encs,
                                   tansho_idx, fuku_idx, calibrators,
                                   umaren_idx=umaren_idx,
                                   class_prior_map=class_prior_map,
                                   shap_explainer=shap_explainer,
                                   shap_topk=shap_topk)
        except Exception as e:
            logger.error(f"  rid={rid}: {e}")
            n_skip += 1
            continue
        rid_s = payload["race_id"]
        # kako5 history を horse ごとに注入 (Cowork advisor 用)
        # + horse_facts (sex/age) も同タイミングで注入 (出走表 表示用)
        # + 血統 (sire/sire_type/broodmare_sire/broodmare_sire_type) 注入 (血統分析用)
        for h in payload.get("horses", []):
            key = (rid_s, int(h.get("umaban", 0)))
            if histories:
                hist = histories.get(key)
                if hist:
                    h["history"] = hist
            if horse_facts:
                fact = horse_facts.get(key)
                if fact:
                    if fact.get("sex"):
                        h["sex"] = fact["sex"]
                    if fact.get("age") is not None:
                        h["age"] = fact["age"]
            if pedigree_map:
                name = (h.get("horse_name") or "").strip()
                if name and name in pedigree_map:
                    h["pedigree"] = pedigree_map[name]
        out_path = out_dir / f"{rid_s}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        saved_paths.append(out_path)
        n_done += 1

    logger.info(f"[done] {n_done:,} JSON ({n_skip} skip) → {out_dir}")

    # ------ bundle (1 file に全 race) ------
    bundle_path = out_dir.parent / f"{date_str}_bundle.json"
    races = []
    for p in sorted(saved_paths):
        with open(p, encoding="utf-8") as f:
            races.append(json.load(f))
    with open(bundle_path, "w", encoding="utf-8") as f:
        json.dump({"date": date_str, "model": tag,
                   "race_count": len(races), "races": races},
                  f, indent=2, ensure_ascii=False)
    logger.info(f"[bundle] {bundle_path}  ({len(races):,} races, "
                f"{bundle_path.stat().st_size/1024:.1f} KB)")

    # ------ 品質ゲート (audit 2026-06-11: 空・激減バンドルの無言 push 防止) ------
    # bundle は書き出した上で (デバッグ用に成果物は残す)、閾値割れなら非0 exit して
    # weekly_nicegui.ps1 側の git push / sync-hf を止める。
    gate_errors: list[str] = []
    # 分母はパース後 df でなく「生CSVのレースヘッダ行数 (19列行)」。
    # parse_csv が行を無言で捨てるケース (TARGET 形式変更) を検出するため。
    n_raw_races = 0
    try:
        for enc in ("cp932", "shift_jis", "utf-8"):
            try:
                raw_text = csv_path.read_bytes().decode(enc)
                break
            except Exception:
                continue
        else:
            raw_text = ""
        for line in raw_text.splitlines():
            cols = line.split(",")
            if len(cols) == 19 and cols[0] not in ("レースID(新)", ""):
                n_raw_races += 1
    except Exception:
        pass
    n_csv_races = max(n_raw_races, int(df[COL_RID].nunique()))
    if len(races) == 0:
        gate_errors.append("bundle の race 数が 0 (parse_csv が全行を捨てた可能性)")
    elif n_csv_races > 0 and len(races) / n_csv_races < 0.5:
        gate_errors.append(
            f"bundle race 数が生CSVレース数の半分未満 ({len(races)}/{n_csv_races}) "
            f"(TARGET 出力形式変更や列ズレで馬・レースが無言で捨てられた可能性。"
            f"障害/新馬の除外は正常でも比率を下げるが 0.5 未満は異常)")
    if races:
        all_horses = [h for r in races for h in r.get("horses", [])]
        if all_horses:
            odds_cov = sum(1 for h in all_horses
                           if h.get("tansho_odds") is not None) / len(all_horses)
            pwin_cov = sum(1 for h in all_horses
                           if h.get("p_win") is not None) / len(all_horses)
            if odds_cov < 0.5:
                gate_errors.append(
                    f"単勝オッズ被覆率 {odds_cov*100:.0f}% < 50% "
                    f"(週次CSVの単勝列欠落? EV判断が全滅する)")
            if pwin_cov < 0.9:
                gate_errors.append(
                    f"p_win 非null率 {pwin_cov*100:.0f}% < 90% (モデル予測の大量失敗)")

    # serve skew ゲート (audit 2026-06-15): 補正/調教 (_SERVE_RENAME) が serve 経路で
    # 死ぬと offline ◎複勝 62.08% → serve 57.53% (-4.55pt)・ECE複勝2.3倍に無言劣化する。
    # リネーム破綻/TARGET列ズレ/入力欠落を canary 列の非null率で検知し push を止める。
    # 正常カバレッジは週で振れる (prev_hosei 65〜94% / trnH 80〜93%)。canary の役目は
    # 「リネーム破綻=列が NaN に潰れて 0%」の検知なので、floor は低く (0.20) して破綻
    # (≈0%) のみ捕え、カバレッジの低い正常週 (>40%) は弾かない (偽陽性回避)。
    # serve skew カナリア (audit 2026-06-17 全特徴化): 旧版は手書き4特徴のみ監視で、
    # 「正常時 alive な特徴が serve で無言死」を取りこぼしていた (実測で26死/うち14は未追跡)。
    # data/serve_feature_baseline.json (健全週の特徴別カバレッジ中央値) と毎回比較し、
    # 正常 alive(baseline>=40%) の特徴が壊滅(現在<20% かつ baseline の4割未満)に落ちたら止める。
    # 既知 dead(baseline<40%: course_*/jockey_*/hist_same_*/前走*等 serve未実装) は対象外。
    # 再診断/baseline更新: analysis/measure_serve_coverage.py
    baseline_path = BASE / "data" / "serve_feature_baseline.json"
    if baseline_path.exists():
        try:
            base_cov = (json.load(open(baseline_path, encoding="utf-8")) or {}).get("baseline_cov", {})
        except Exception as e:
            base_cov = {}
            logger.warning(f"[serve canary] baseline 読込失敗 ({e}) → 監視スキップ")
        monitored, silent_deaths = 0, []
        for col in feats:
            exp = base_cov.get(col)
            if exp is None or exp < 0.40:
                continue   # 既知 dead / 監視対象外
            monitored += 1
            cur = float(df[col].notna().mean()) if col in df.columns else 0.0
            if cur < 0.20 and cur < exp * 0.40:
                silent_deaths.append(f"{col}({exp*100:.0f}%→{cur*100:.0f}%)")
        logger.info(f"[serve canary] 監視 {monitored} 特徴 / 無言死 {len(silent_deaths)}")
        if silent_deaths:
            gate_errors.append(
                f"serve特徴が無言死 ({len(silent_deaths)}): {', '.join(silent_deaths[:12])}"
                f"{' ...' if len(silent_deaths) > 12 else ''} "
                f"(parse_csv/_SERVE_RENAME/TARGET列ズレ。offline→serve 劣化。"
                f"analysis/measure_serve_coverage.py で再診断)")
    else:
        logger.warning("[serve canary] baseline 未生成 → 旧4特徴チェックにフォールバック "
                       "(analysis/measure_serve_coverage.py を実行して baseline を作ること)")
        for col, floor_cov in {"prev_hosei": 0.20, "trnH_Time1": 0.20}.items():
            if col in feats:
                cov = float(df[col].notna().mean()) if col in df.columns else 0.0
                if cov < floor_cov:
                    gate_errors.append(
                        f"serve特徴 {col} 非null率 {cov*100:.0f}% < {floor_cov*100:.0f}% "
                        f"(補正/調教リネーム破綻 or 列ズレ)")

    print()
    if gate_errors:
        for ge in gate_errors:
            logger.error(f"[品質ゲート] {ge}")
        print("=== ⚠ 品質ゲート不合格: bundle は生成済みだが push しないこと ===")
        print(f"  集約: {bundle_path}")
        return 2

    print(f"=== Cowork 入力 JSON 出力完了 ===")
    print(f"  個別: {out_dir}/")
    print(f"  集約: {bundle_path}")
    print(f"  → このファイルを Cowork (Anthropic Desktop App) に投げて買い目を受け取る")
    return 0


if __name__ == "__main__":
    sys.exit(main())
