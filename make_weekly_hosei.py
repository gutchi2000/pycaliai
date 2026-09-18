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
def _is_weekly_output(f: Path) -> bool:
    """このスクリプトが吐いた週次ファイル (H_YYYYMMDD.csv) か。"""
    stem = f.stem[2:]
    return len(stem) == 8 and stem.isdigit()


def load_direct_prev_lookup() -> dict[str, tuple]:
    """18桁レースID(今走) → (前走補9, 前走補正)。

    学習側 build_master_v2.py:68-70 と同一定義。TARGET から出した hosei
    エクスポート (H_開始-終了.csv) が今走キーで 前走補正 を持っているので、
    前走を辿らずにそのまま使える = off-by-one が構造的に起こらない経路。
    週次生成ファイルは自分の出力なので除外する。
    """
    lookup: dict[str, tuple] = {}
    n_files = 0
    for f in sorted(HOSEI_DIR.glob("H_*.csv")):
        if _is_weekly_output(f):
            continue
        for enc in ["cp932", "utf-8-sig", "utf-8"]:
            try:
                head = pd.read_csv(f, encoding=enc, nrows=0)
                if "前走補正" not in head.columns or "前走補9" not in head.columns:
                    break
                df = pd.read_csv(f, encoding=enc,
                                 usecols=["レースID(新)", "前走補9", "前走補正"],
                                 dtype={"レースID(新)": str})
                rid = df["レースID(新)"].astype(str).str.strip().str.zfill(18)
                h9 = pd.to_numeric(df["前走補9"], errors="coerce")
                hc = pd.to_numeric(df["前走補正"], errors="coerce")
                for r, a, b in zip(rid, h9, hc):
                    if pd.isna(a) and pd.isna(b):
                        continue
                    lookup[r] = (None if pd.isna(a) else float(a),
                                 None if pd.isna(b) else float(b))
                n_files += 1
                break
            except Exception:
                continue
    log.info(f"直接引き lookup: {len(lookup):,} エントリ / {n_files} ファイル (今走キー×前走補正)")
    return lookup


def load_hosei_lookup() -> dict[str, tuple]:
    """18桁レースID → (そのレース自身の 補9, 補正)。

    ★2026-09-17 修正 (off-by-one)。
    呼び出し側は「前走の 18桁ID」でこの辞書を引き、得た値を今走の prev_hosei として
    書き出す。したがってこの辞書の値は **そのレース自身の補正タイム**(列名「補正」/「補9」)
    でなければならない。従来は「前走補正」/「前走補9」を読んでいたため、serve の
    prev_hosei は実際には **前々走** の補正タイムになっていた。

    実証 (analysis/verify_hosei_offbyone.py, n=464,424):
      H[今走].前走補正 == H[前走].補正        → 100.00%  (学習側 prev_hosei の定義)
      H[今走].前走補正 == H[前走].前走補正     →   5.85%  (従来 serve が入れていた値)
      ズレていた行 94.15% / 平均絶対差 7.95
    影響 (analysis/hosei_bug_impact.py): ◎の勝率 29.58% → 27.04%、top3 60.75% → 58.30%。
    prev_hosei は v6 の gain 第2位 (7.47%)。誤った値を入れるのは
    「特徴を捨てる」(27.87%) より悪い。

    注意: 週次生成の H_{date}.csv には「補正」列が無い (前走補正しか持たない) ため、
    値の供給源になれない。ここでは 補正 列を持つ hosei マスターだけを読む。
    2026 以降に前走があった馬を引くには TARGET から hosei マスターの再エクスポートが要る。
    """
    lookup: dict[str, tuple] = {}
    n_files = 0
    for f in sorted(HOSEI_DIR.glob("H_*.csv")):
        if _is_weekly_output(f):
            continue
        for enc in ["cp932", "utf-8-sig", "utf-8"]:
            try:
                head = pd.read_csv(f, encoding=enc, nrows=0)
                if "補正" not in head.columns or "補9" not in head.columns:
                    break  # 週次生成ファイル: 自身の補正を持たないので使えない
                df = pd.read_csv(f, encoding=enc,
                                 usecols=["レースID(新)", "補9", "補正"],
                                 dtype={"レースID(新)": str})
                rid = df["レースID(新)"].astype(str).str.strip().str.zfill(18)
                h9 = pd.to_numeric(df["補9"], errors="coerce")
                hc = pd.to_numeric(df["補正"], errors="coerce")
                for r, a, b in zip(rid, h9, hc):
                    lookup[r] = (None if pd.isna(a) else float(a),
                                 None if pd.isna(b) else float(b))
                n_files += 1
                break
            except Exception:
                continue
    log.info(f"hosei lookup: {len(lookup):,} エントリ / {n_files} ファイル読み込み "
             f"(列=補正・補9)")
    if n_files == 0:
        log.error("hosei マスター (補正列を持つ H_*.csv) が見つからない。"
                  "prev_hosei は全馬欠損になる。")
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
# kekka マスターフォールバック (全馬収録 v2)
# =========================================================
# 日別 kekka CSV (data/kekka/) は週次運用分 (2026〜) しか無く、前走が 2025 年の
# 馬は 18桁ID を引けず「kekka未照合」で補正が欠損していた (20260607 で 90/313頭)。
# E:\競馬過去走データ\kekka_..._v2.csv は全馬収録 (馬名/日付/レースID(新)18桁)
# なので、日別 CSV に無い日付はこれで引く。
KEKKA_MASTER_V2 = Path(r"E:\競馬過去走データ\kekka_20130105-20251228_v2.csv")
_master_lookup: dict[tuple[str, str], str] | None = None


def load_kekka_master_lookup() -> dict[tuple[str, str], str]:
    """(馬名, 日付8桁) → レースID(新)18桁 の辞書。初回のみロード。"""
    global _master_lookup
    if _master_lookup is not None:
        return _master_lookup
    _master_lookup = {}
    if not KEKKA_MASTER_V2.exists():
        log.warning(f"kekka マスター v2 なし: {KEKKA_MASTER_V2} (フォールバック無効)")
        return _master_lookup
    df = pd.read_csv(KEKKA_MASTER_V2, encoding="cp932",
                     usecols=["日付", "馬名", "レースID(新)"],
                     dtype={"日付": str, "レースID(新)": str})
    # 日付は YYMMDD 6桁 (例 251228) → 20YYMMDD
    dates = "20" + df["日付"].str.strip().str.zfill(6)
    names = df["馬名"].astype(str).str.strip()
    rids = df["レースID(新)"].astype(str).str.strip().str.zfill(18)
    _master_lookup = dict(zip(zip(names, dates), rids))
    log.info(f"kekka マスター v2 lookup: {len(_master_lookup):,} エントリ")
    return _master_lookup


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
    # 2a) 直接引き: 「今走の18桁ID → 前走補正」を持つ hosei エクスポートがあれば
    #     そのまま使う。これは学習側 (build_master_v2: H[今走].前走補正) と
    #     完全に同じ定義なので、前走を辿る処理そのものが不要になり off-by-one も起き得ない。
    direct = load_direct_prev_lookup()
    # 2b) 連鎖引き: 直接引きに無い馬だけ、前走18桁ID → そのレース自身の「補正」で補う。
    hosei_lookup = load_hosei_lookup()

    # ── Step 3: 各馬の前走を kekka で特定 → hosei で補正タイム取得 ──
    rows: list[dict] = []
    cnt_hit = cnt_no_prev = cnt_no_kekka = cnt_no_hosei = 0

    cnt_direct = 0

    for _, horse in df.iterrows():
        horse_name  = str(horse.get("馬名", "")).strip()
        current_ban = horse.get("馬番")
        race_id_16  = str(horse.get("レースID(新)", "")).strip()[:16]
        date_s      = str(horse.get("日付S", ""))

        # ── 直接引き (学習と同一定義)。当たればこの馬は前走を辿る必要がない ──
        if pd.notna(current_ban):
            cur18 = race_id_16 + str(int(current_ban)).zfill(2)
            ent = direct.get(cur18)
            if ent is not None and not (ent[0] is None and ent[1] is None):
                rows.append({"レースID(新)": cur18, "馬番": int(current_ban),
                             "前走補9": ent[0], "前走補正": ent[1]})
                cnt_direct += 1
                cnt_hit += 1
                continue

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

        # kekka CSV から前走の 18桁ID を取得 (日別 → 無ければ全期間マスター v2)
        prev_18 = None
        kk = load_kekka(pdk)
        if kk is not None:
            kk_horse = kk[kk["馬名"].astype(str).str.strip() == horse_name]
            if not kk_horse.empty:
                prev_18 = str(kk_horse.iloc[0]["レースID(新)"]).strip().zfill(18)
        if prev_18 is None:
            prev_18 = load_kekka_master_lookup().get((horse_name, pdk))
        if prev_18 is None:
            cnt_no_kekka += 1
            continue

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
        f"結果: 成功={cnt_hit} (直接引き={cnt_direct} / 前走連鎖={cnt_hit - cnt_direct})  "
        f"前走なし={cnt_no_prev}  "
        f"kekka未照合={cnt_no_kekka}  hosei未照合={cnt_no_hosei}  "
        f"カバレッジ={coverage:.1f}%"
    )
    # 学習時 prev_hosei 充足率は 85.4%。出走全頭に対する実効カバレッジを見ておく
    eff = cnt_hit / len(df) * 100 if len(df) else 0
    log.info(f"  出走全頭に対する実効カバレッジ={eff:.1f}% (学習時 85.4%)")
    if eff < 60:
        log.warning("  ★ 学習時より大幅に低い。hosei マスターの期間が足りていない可能性。"
                    "TARGET から 補正/前走補正 を含む最新エクスポートを取り直すこと。")

    if not rows:
        log.warning("取得できたデータが0件。ファイルを生成しません。")
        log.warning("  kekka CSV が data/kekka/ に揃っているか確認してください。")
        return

    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    log.info(f"保存: {out_path}  ({len(rows)} 件)")


if __name__ == "__main__":
    main()
