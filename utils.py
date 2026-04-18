"""
utils.py
PyCaLiAI - 共通ユーティリティ
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd

# halo_thresholds.json のデフォルト値（JSON 未存在時フォールバック）
_HALO_DEFAULTS = {
    "gap_12_hi":   10.0,
    "gap_12_lo":    5.0,
    "gap_top4_lo": 15.0,
    "pw_min":       0.50,
    "pw_ratio":     2.0,
}


@lru_cache(maxsize=1)
def load_halo_thresholds() -> dict:
    """data/halo_thresholds.json から最適化済み HALO 閾値を読み込む。

    ファイルが存在しない場合や読み込み失敗時はデフォルト値を返す。
    lru_cache によりプロセス内で 1 回だけ読み込む（hot-reload 不要）。

    Returns:
        dict with keys: gap_12_hi, gap_12_lo, gap_top4_lo, pw_min, pw_ratio
    """
    json_path = Path(__file__).parent / "data" / "halo_thresholds.json"
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        params = data.get("best_params", {})
        # 必須キーが揃っているか確認
        merged = dict(_HALO_DEFAULTS)
        merged.update({k: float(v) for k, v in params.items() if k in _HALO_DEFAULTS})
        return merged
    except FileNotFoundError:
        return dict(_HALO_DEFAULTS)
    except Exception as e:
        import warnings
        warnings.warn(f"[load_halo_thresholds] 読み込み失敗、デフォルト使用: {e}")
        return dict(_HALO_DEFAULTS)


# =====================================================
# HALO 条件別 playbook (Stage 2-05)
#   grid_search_playbook.py で生成される
#   data/halo_formation_playbook.json を読む。
#   セル key: "{芝ダ}_{field_bucket}"  例: "ダ_多"
#   field_bucket: 少<=10 / 中 11-14 / 多 15+
# =====================================================

def _field_bucket_for_playbook(n: int) -> str:
    if n <= 10: return "少"
    if n <= 14: return "中"
    return "多"


@lru_cache(maxsize=1)
def load_halo_playbook() -> dict:
    """data/halo_formation_playbook.json を読み込み、cells dict を返す。

    Returns:
        {"ダ_多": {"top_n": 3, "ev_gate": 0.0, ...}, ...}
        ファイル不存在 or 読み込み失敗時は空 dict（= 常に baseline 動作）。
    """
    json_path = Path(__file__).parent / "data" / "halo_formation_playbook.json"
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("cells", {})
    except FileNotFoundError:
        return {}
    except Exception as e:
        import warnings
        warnings.warn(f"[load_halo_playbook] 読み込み失敗、baseline 動作: {e}")
        return {}


def lookup_halo_policy(shiba_da: str, field_size: int) -> dict:
    """レース条件から HALO playbook のポリシーを引く。

    Args:
        shiba_da: "芝" または "ダ"
        field_size: 出走頭数

    Returns:
        {"top_n": int, "ev_gate": float, "no_bet": bool, "note": str}
        top_n=0 or no_bet=True → このレースは HALO を買わない。
        該当セル無しは baseline (top_n=3, ev_gate=0) を返す。
    """
    playbook = load_halo_playbook()
    key = f"{shiba_da}_{_field_bucket_for_playbook(field_size)}"
    policy = playbook.get(key)
    if policy is None:
        # フォールバック: baseline
        return {"top_n": 3, "ev_gate": 0.0, "no_bet": False,
                "note": "no playbook cell, using baseline"}
    top_n   = int(policy.get("top_n", 3))
    ev_gate = float(policy.get("ev_gate", 0.0))
    no_bet  = (top_n <= 0) or (ev_gate >= 99.0)
    return {"top_n": top_n, "ev_gate": ev_gate, "no_bet": no_bet,
            "note": policy.get("note", ""), "cell": key}


# =====================================================
# 馬券種別 EV ゲート閾値（Stage 1）
#   ev_gate.py からも参照可。
#   pass_ev_gate(bet_type, ev) 判定で使用。
# =====================================================
MIN_EV_TANSHO = 1.05   # 単勝
MIN_EV_FUKU   = 1.05   # 複勝
MIN_EV_UMAREN = 1.10   # 馬連（的中率低めなので高め）
MIN_EV_UMATAN = 1.15   # 馬単
MIN_EV_SANREN = 1.05   # 三連複
MIN_EV_SANTAN = 0.95   # 三連単（控除率 27.5% を考慮し緩め）

# 旧仕様: 単勝オッズ ≥ 2.0 で複勝買い → DEPRECATED, EV ゲートに置換
# Stage 1 以降は MIN_EV_FUKU を使う。残置は backward-compat のみ。
MIN_TANSHO_FOR_FUKU = 2.0   # deprecated, use MIN_EV_FUKU instead


def backup_model(path: Path) -> None:
    """既存モデルファイルを上書き前にタイムスタンプ付きでバックアップする。

    例: lgbm_optuna_v1.pkl → lgbm_optuna_v1_20250101_120000.pkl
    ファイルが存在しない場合は何もしない。
    """
    if not path.exists():
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_stem(f"{path.stem}_{ts}")
    shutil.move(str(path), str(backup_path))
    print(f"[backup_model] {path.name} → {backup_path.name}")


def add_meta(df: pd.DataFrame) -> pd.DataFrame:
    """日付・曜日・土日フラグ・同日同会場R数・週末10Rフラグを付与する。"""
    df = df.copy()
    df["date"] = pd.to_datetime(df["日付"].astype(str), format="%Y%m%d")
    df["曜日"]  = df["date"].dt.dayofweek
    df["土日"]  = df["曜日"].isin([5, 6])
    rc = (
        df.groupby(["日付", "場所"])["race_id"]
        .nunique()
        .reset_index()
        .rename(columns={"race_id": "R数"})
    )
    df = df.merge(rc, on=["日付", "場所"], how="left")
    df["週末10R"] = df["土日"] & (df["R数"] >= 10)
    return df


def parse_time_str(series: pd.Series) -> pd.Series:
    """走破タイム文字列を秒数（float）に変換する。

    形式: "M.SS.T"（分.秒.1/10秒）→ float秒
    例: "1.34.5" → 94.5秒
    """
    def _convert(val: str) -> float | None:
        try:
            parts = str(val).strip().split(".")
            if len(parts) == 3:
                return int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 10
            return float(val)
        except (ValueError, TypeError, AttributeError):
            return None
    return series.apply(_convert)
