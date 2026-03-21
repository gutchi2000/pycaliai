"""
make_weekly_hosei.py
====================
週次CSVの各馬の「前走レース」を kekka CSV で特定し、
hosei CSV から前走補正タイムを引いて、今週レース用の
補正タイムファイル（data/hosei/H_YYYYMMDD.csv）を生成する。

仕組み:
  [weekly CSV] 馬名 + 前走月日
      ↓  kekka CSV (前走日付) で馬名 → レースID(新)18桁 を取得
      ↓  hosei CSV でその18桁ID → 前走補9/前走補正 を取得
      ↓  今週のレースID(18桁) に紐付けて出力

使い方:
  python make_weekly_hosei.py                         # 最新 weekly CSV を自動選択
  python make_weekly_hosei.py --csv data/weekly/20260321.csv
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).parent
WEEKLY_DIR = BASE_DIR / "data" / "weekly"
KEKKA_DIR  = BASE_DIR / "data" / "kekka"
HOSEI_DIR  = BASE_DIR / "data" / "hosei"

# =========================================================
# weekly CSV パース定数（predict_weekly.py と同一）
# =========================================================
RACE_COLS = [
    "レースID(新)","日付S","曜日","場所","開催","R","レース名","クラス名",
    "芝・ダート","距離","コース区分","コーナー回数","馬場状態(暫定)","天候(暫定)",
    "フルゲート頭数","発走時刻","性別限定","重量種別","年齢限定",
]
HORSE_COLS_33 = [
    "枠番","B","馬番","馬名S","性別","年齢","人気_今走","単勝","ZI印","ZI","ZI順",
    "斤量","減M","替","騎手","所属","調教師","父","母父","父タイプ","母父タイプ",
    "前走月","前走日","前走場所","前走TD","前走距離","前走馬場状態","前走着順",
    "前走人気","前走レース名","前走上り3F","前走決手","前走間隔",
]
HORSE_COLS_46 = [
    "枠番","B","馬番","馬名S","性別","年齢","人気_今走","単勝","ZI印","ZI","ZI順",
    "斤量","減M","替","騎手","所属","調教師","父","母父","父タイプ","母父タイプ",
    "前走月","前走日","前走開催","前走間隔","前走レース名","前走TD","前走距離","前走馬場状態",
    "前走B","前走騎手","前走斤量","前走減","前走人気","前走単勝オッズ","前走着順","前走着差",
    "マイニング順位","前走通過1","前走通過2","前走通過3","前走通過4","前走Ave3F",
    "前走上り3F","前走上り3F順位","前走1_2着馬",
]
HORSE_COLS_48 = HORSE_COLS_46 + ["騎手コード", "調教師コード"]
HORSE_COLS_49 = [
    "枠番","B","馬番","馬名S","性別","年齢","馬体重","馬体重増減_raw","馬体重増減",
    "人気_今走","単勝","ZI印","ZI","ZI順","斤量","減M","替","騎手","所属","調教師",
    "父","母父","父タイプ","母父タイプ",
    "前走月","前走日","前走開催","前走間隔","前走レース名","前走TD","前走距離","前走馬場状態",
    "前走B","前走騎手","前走斤量","前走減","前走人気","前走単勝オッズ","前走着順","前走着差",
    "マイニング順位","前走通過1","前走通過2","前走通過3","前走通過4","前走Ave3F",
    "前走上り3F","前走上り3F順位","前走1_2着馬",
]
HORSE_COLS_99 = HORSE_COLS_49 + [
    "二走前月","二走前日","二走前開催","二走前間隔","二走前レース名","二走前TD",
    "二走前距離","二走前馬場状態","二走前B","二走前騎手","二走前斤量","二走前減",
    "二走前人気","二走前単勝オッズ","二走前着順","二走前着差","二走前マイニング順位",
    "二走前通過1","二走前通過2","二走前通過3","二走前通過4","二走前Ave3F",
    "二走前上り3F","二走前上り3F順位","二走前1_2着馬",
    "三走前月","三走前日","三走前開催","三走前間隔","三走前レース名","三走前TD",
    "三走前距離","三走前馬場状態","三走前B","三走前騎手","三走前斤量","三走前減",
    "三走前人気","三走前単勝オッズ","三走前着順","三走前着差","三走前マイニング順位",
    "三走前通過1","三走前通過2","三走前通過3","三走前通過4","三走前Ave3F",
    "三走前上り3F","三走前上り3F順位","三走前1_2着馬",
]


# =========================================================
# weekly CSV パース
# =========================================================
def parse_weekly_csv(path: Path) -> pd.DataFrame:
    for enc in ["cp932", "shift_jis", "utf-8"]:
        try:
            text = path.read_bytes().decode(enc); break
        except Exception:
            continue

    races: list[dict] = []
    current_race: dict | None = None
    for line in text.splitlines():
        cols = line.split(",")
        if cols[0] in ("レースID(新)", "枠番", "番", "B", ""):
            continue
        if len(cols) == 19:
            current_race = dict(zip(RACE_COLS, cols))
        elif current_race:
            mapping = {33: HORSE_COLS_33, 46: HORSE_COLS_46,
                       48: HORSE_COLS_48, 49: HORSE_COLS_49, 99: HORSE_COLS_99}
            if len(cols) in mapping:
                h = dict(zip(mapping[len(cols)], cols))
                h.update(current_race)
                races.append(h)

    df = pd.DataFrame(races)
    if df.empty:
        return df
    df = df.rename(columns={"馬名S": "馬名"})
    for col in ["前走月", "前走日", "馬番"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# =========================================================
# hosei 辞書（18桁ID → 補正タイム）
# =========================================================
def load_hosei_lookup() -> dict[str, tuple]:
    lookup: dict[str, tuple] = {}
    for f in sorted(HOSEI_DIR.glob("H_*.csv")):
        for enc in ["cp932", "utf-8-sig", "utf-8"]:
            try:
                df = pd.read_csv(f, encoding=enc,
                                 usecols=["レースID(新)", "前走補9", "前走補正"],
                                 dtype={"レースID(新)": str})
                for _, row in df.iterrows():
                    rid = str(row["レースID(新)"]).strip().zfill(18)
                    h9 = float(row["前走補9"])  if pd.notna(row["前走補9"])  else None
                    hc = float(row["前走補正"]) if pd.notna(row["前走補正"]) else None
                    lookup[rid] = (h9, hc)
                break
            except Exception:
                continue
    log.info(f"hosei lookup: {len(lookup):,} エントリ読み込み")
    return lookup


# =========================================================
# kekka キャッシュ
# =========================================================
_kekka_cache: dict[str, pd.DataFrame] = {}

def load_kekka(date_key: str) -> pd.DataFrame | None:
    if date_key in _kekka_cache:
        return _kekka_cache[date_key]
    p = KEKKA_DIR / f"{date_key}.csv"
    if not p.exists():
        return None
    for enc in ["cp932", "utf-8"]:
        try:
            df = pd.read_csv(p, encoding=enc)
            _kekka_cache[date_key] = df
            return df
        except Exception:
            continue
    return None


# =========================================================
# 前走日付キー生成
# =========================================================
def prev_date_key(race_date_s: str, prev_month: int, prev_day: int) -> str | None:
    """'2026.3.21', 3, 15 → '20260315'  /  '2026.1.4', 12, 28 → '20251228'"""
    try:
        parts = race_date_s.replace("-", ".").split(".")
        race_year  = int(parts[0])
        race_month = int(parts[1])
    except Exception:
        return None
    year = race_year if prev_month <= race_month else race_year - 1
    return f"{year}{prev_month:02d}{prev_day:02d}"


# =========================================================
# メイン
# =========================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="", help="weekly CSV パス")
    args = parser.parse_args()

    # weekly CSV を決定
    if args.csv:
        csv_path = Path(args.csv)
    else:
        files = sorted(WEEKLY_DIR.glob("????????.csv"), reverse=True)
        if not files:
            log.error("data/weekly/ に CSV がありません。")
            return
        csv_path = files[0]
        log.info(f"自動選択: {csv_path.name}")

    date_key = csv_path.stem                        # "20260321"
    out_path = HOSEI_DIR / f"H_{date_key}.csv"

    # ── Step 1: weekly CSV パース ──────────────────────────
    log.info(f"weekly CSV パース: {csv_path.name}")
    df = parse_weekly_csv(csv_path)
    if df.empty:
        log.error("パース結果が空です。")
        return
    log.info(f"  {len(df)} 頭 / {df['レースID(新)'].nunique()} レース")

    # ── Step 2: hosei 辞書構築 ────────────────────────────
    hosei_lookup = load_hosei_lookup()

    # ── Step 3: 各馬の前走を kekka で特定 → hosei で補正タイム取得 ──
    rows: list[dict] = []
    cnt_hit = cnt_no_prev = cnt_no_kekka = cnt_no_hosei = 0

    for _, horse in df.iterrows():
        horse_name  = str(horse.get("馬名", "")).strip()
        current_ban = horse.get("馬番")
        race_id_16  = str(horse.get("レースID(新)", "")).strip()[:16]
        date_s      = str(horse.get("日付S", ""))

        # 前走情報がない（初出走など）
        prev_m = horse.get("前走月")
        prev_d = horse.get("前走日")
        if pd.isna(prev_m) or pd.isna(prev_d) or int(prev_m) == 0 or int(prev_d) == 0:
            cnt_no_prev += 1
            continue

        pdk = prev_date_key(date_s, int(prev_m), int(prev_d))
        if not pdk:
            cnt_no_prev += 1
            continue

        # kekka CSV から前走の 18桁ID を取得
        kk = load_kekka(pdk)
        if kk is None:
            cnt_no_kekka += 1
            continue

        kk_horse = kk[kk["馬名"].astype(str).str.strip() == horse_name]
        if kk_horse.empty:
            cnt_no_kekka += 1
            continue

        prev_18 = str(kk_horse.iloc[0]["レースID(新)"]).strip().zfill(18)

        # hosei から補正タイムを取得
        entry = hosei_lookup.get(prev_18)
        if entry is None:
            cnt_no_hosei += 1
            continue

        h9, hc = entry
        if h9 is None and hc is None:
            cnt_no_hosei += 1
            continue

        # 今週のレースID(18桁) = race_id_16 + 馬番2桁ゼロパッド
        current_18 = race_id_16 + str(int(current_ban)).zfill(2)

        rows.append({
            "レースID(新)": current_18,
            "馬番":         int(current_ban),
            "前走補9":      h9,
            "前走補正":     hc,
        })
        cnt_hit += 1

    total_prev = cnt_hit + cnt_no_kekka + cnt_no_hosei
    coverage = cnt_hit / total_prev * 100 if total_prev > 0 else 0
    log.info(
        f"結果: 成功={cnt_hit}  前走なし={cnt_no_prev}  "
        f"kekka未照合={cnt_no_kekka}  hosei未照合={cnt_no_hosei}  "
        f"カバレッジ={coverage:.1f}%"
    )

    if not rows:
        log.warning("取得できたデータが0件。ファイルを生成しません。")
        log.warning("  kekka CSV が data/kekka/ に揃っているか確認してください。")
        return

    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    log.info(f"保存: {out_path}  ({len(rows)} 件)")


if __name__ == "__main__":
    main()
