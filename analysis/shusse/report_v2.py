# -*- coding: utf-8 -*-
"""
analysis/shusse/report_v2.py — 出世予測 AI v2 の自己完結 HTML レポート
========================================================================
`analysis/shusse/out_v2/` の成果物（metrics_v2.json / v1_comparison.json / ablation_v2.json / calibration_*.csv /
confusion_A_k3.csv / feature_importance_*.csv / pred_test_k{k}.parquet / starts_v2.parquet）と `narrate.py` の文章から、
単一 HTML `out_v2/shusse_report.html` を作る。CSS インライン・JS なし・外部 CDN なし・フォントはシステム任せ。

    ./venv311/Scripts/python.exe -m analysis.shusse.report_v2

節構成: 1 概要 / 2 成績表（k×規則vsAI、混同行列、較正図 SVG）/ 3 v1→v2 / 4 効いた特徴とアブレーション /
5 読み物カード（約 20 枚）/ 6 現役馬（`out_v2/live/shusse_live.json` または `predictions_latest.parquet` があれば）/ 7 限界。

現役馬節の入力は別エージェントの `predict_live.py` が書くもので、本スクリプトは **存在すれば読む・無ければ「未生成」1 行** とする。
実スキーマ（predict_live.py、2026-10-31 時点）を `_normalize_live` で pred_test 形に揃える:
  - predictions_latest.parquet（優先）: 血統登録番号, 馬名, debut_year, 性別, 父, 調教師コード, k, base_date, p_dropout, p_L0..p_L4,
    expected_level, pred_level, p_graded12, p_graded24, rule_level, top3_A_L4/top3_D/top3_B（日本語ラベルの JSON 文字列）, is_latest, confirmed …
    → is_latest かつ未確定の行だけ使う。
  - shusse_live.json: {model, asof, generated_at, n_horses, horses:[{ped_id, name, sex, sire, trainer_code, trainer, k, base_date, p_dropout,
    p_level[5], pred_level, p_graded12, rule_level, reasons:{L4, dropout, graded24}, …}]} → parquet が無いときの代替。
  走の明細は `out_v2/live/starts_latest.parquet` か `starts_ext.parquet`（extend_2026.py の出力）があればそれを、無ければ `starts_v2.parquet`。
"""
from __future__ import annotations

import html
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.shusse.narrate import (
    COL_HORSE, LEVEL_NAME, NAMED_SPEC, NAMED_V1, cards_for, load_labels, make_card, pct, start_phrase,
)

OUT = Path(__file__).resolve().parent / "out_v2"
LIVE = OUT / "live"
REPORT = OUT / "shusse_report.html"
KS = (1, 2, 3, 5)

CSS = """
:root{--ink:#1f2a37;--muted:#5b6775;--line:#d9dee5;--bg:#ffffff;--soft:#f4f6f9;--acc:#1d4f91;--acc2:#b8860b;--ok:#2e7d32;--ng:#b23a3a;--near:#b07a1a}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:"Hiragino Sans","Yu Gothic UI","Meiryo","Noto Sans JP",system-ui,sans-serif;line-height:1.65;font-size:15px}
main{max-width:900px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:26px;margin:0 0 6px;letter-spacing:.01em}
h2{font-size:20px;margin:44px 0 12px;padding-bottom:6px;border-bottom:2px solid var(--acc)}
h3{font-size:16px;margin:26px 0 8px}
h4{font-size:15px;margin:18px 0 6px}
p{margin:8px 0}
.lead{color:var(--muted);margin-bottom:4px}
.meta{color:var(--muted);font-size:13px}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:10px 0 14px}
th,td{border:1px solid var(--line);padding:5px 8px;text-align:right;vertical-align:top}
th{background:var(--soft);font-weight:600;text-align:center}
td:first-child,th:first-child{text-align:left}
td.l{text-align:left}
.ai{color:var(--acc);font-weight:600}
.small{font-size:12.5px;color:var(--muted)}
.card{border:1px solid var(--line);border-radius:8px;padding:14px 16px;margin:14px 0;background:#fff;page-break-inside:avoid}
.card h4{margin:0 0 4px;font-size:16px}
.card .why{display:inline-block;font-size:12px;color:#fff;background:var(--acc);border-radius:4px;padding:1px 7px;margin-left:6px;vertical-align:middle}
.card .txt p{margin:5px 0}
.runs{font-size:13px;margin:8px 0}
.runs td,.runs th{padding:3px 6px}
.bars{display:grid;grid-template-columns:110px 1fr 48px;gap:3px 8px;align-items:center;font-size:12.5px;margin:8px 0}
.bar{height:11px;background:var(--soft);border-radius:3px;overflow:hidden;border:1px solid var(--line)}
.bar i{display:block;height:100%;background:var(--acc)}
.bar.drop i{background:var(--ng)}
.bar.g12 i{background:var(--acc2)}
.verdict{font-weight:700}
.v-ok{color:var(--ok)}.v-ng{color:var(--ng)}.v-near{color:var(--near)}
.twocol{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media (max-width:700px){.twocol{grid-template-columns:1fr}}
figure{margin:10px 0}
figcaption{font-size:12.5px;color:var(--muted)}
.note{background:var(--soft);border-left:4px solid var(--acc2);padding:8px 12px;margin:10px 0;font-size:13.5px}
ol li,ul li{margin:3px 0}
.toc{font-size:13.5px;columns:2;margin:8px 0 0}
.toc a{color:var(--acc);text-decoration:none}
@media print{body{font-size:12px}.card{break-inside:avoid}h2{break-after:avoid}a{color:inherit}}
"""


# --------------------------------------------------------------------------- helpers
def esc(x) -> str:
    return html.escape("" if x is None else str(x), quote=False)


def md_inline(s: str) -> str:
    """narrate の本文（**強調** と改行だけ）を HTML に。"""
    s = esc(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    return "".join(f"<p>{line}</p>" for line in s.split("\n") if line.strip())


def f3(x, nd=3) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    return "—" if np.isnan(v) else f"{v:.{nd}f}"


def ci(x) -> str:
    if not x or len(x) != 2 or any(v is None or (isinstance(v, float) and np.isnan(v)) for v in x):
        return ""
    return f"<span class='small'> [{float(x[0]):.3f}, {float(x[1]):.3f}]</span>"


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def table(headers: list[str], rows: list[list[str]], cls: str = "") -> str:
    h = "".join(f"<th>{c}</th>" for c in headers)
    b = "".join("<tr>" + "".join(f"<td{' class=l' if i == 0 else ''}>{c}</td>" for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f"<table class='{cls}'><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"


# --------------------------------------------------------------------------- sections
def sec_overview(metrics: dict, join: dict | None) -> str:
    n = {k: metrics.get(f"k{k}", {}).get("n", {}) for k in KS}
    rows = [[f"k={k}", f"{n[k].get('train', 0):,}", f"{n[k].get('valid', 0):,}", f"{n[k].get('test', 0):,}", f"{n[k].get('test2023_partial', 0):,}"] for k in KS]
    jr = ""
    if join:
        jr = f"平地明細 {join.get('n_flat_rows', join.get('flat_rows', '—')):,} 行、" if isinstance(join.get("n_flat_rows", join.get("flat_rows")), int) else ""
    return f"""
<h2 id="s1">1. この AI は何をするか</h2>
<p>JRA 平地でデビューした 2〜3 歳馬について、<strong>k 走目（k = 1, 2, 3, 5）を走り終えた時点</strong>の成績だけから、
<strong>1 年後にどのクラスにいるか</strong>（未勝利圏 / 1勝 / 2勝 / 3勝・OP / 重賞で掲示板）、<strong>1 年以内に走らなくなる（離脱）確率</strong>、
<strong>2 年以内に重賞で掲示板（5 着以内）に載る確率</strong>を出します。馬券とは無関係の読み物用 AI です。</p>
<ul>
<li><strong>使っていない情報</strong>: 市場オッズ・人気・馬券の払戻。当日の情報（馬体重は走った結果として使う）。<strong>馬券には転用しません</strong>。</li>
<li><strong>データ範囲</strong>: 2013-01〜2025-12-28 の JRA 平地全走（TARGET 出力の master と外部 kekka を (日付, 場所, R, 血統登録番号) で 100% 結合）。
目的変数は基準日から 365 日 / 730 日の窓が 2025-12-28 以内に閉じる馬だけに付けます。</li>
<li><strong>評価の立て方</strong>: デビュー年でコホート分割。学習 2013〜2019、検証 2020〜2021（ハイパーパラメータと閾値を決める）、
<strong>テスト 2022 デビュー</strong>（判定に使う唯一の数字）。2023 デビューは窓が閉じた馬だけの記述用（選択バイアスあり）。
信頼区間は馬単位ブートストラップ 300 回。</li>
<li><strong>比較相手（規則）</strong>: 「k 走目で勝てば 1 段上のクラス、負ければ今のクラス」という昇級規則。この規則に対して AI が何を足せるかを見ます。</li>
<li><strong>モデル</strong>: LightGBM 3 ヘッド（D: 離脱、A: 走った馬の到達クラス 5 段、B: 重賞掲示板 24 か月）。統合 P(重賞掲示板 12m) = (1 − P(離脱)) × P(重賞掲示板 | 走る)。</li>
</ul>
{table(["基準点", "学習", "検証", "テスト (2022)", "2023 部分"], rows)}
<p class="small">対象頭数（馬 1 行）。k が大きいほど「k 走目まで走った馬」に絞られるので減ります。</p>
"""


def sec_metrics(metrics: dict, conf: pd.DataFrame | None, cal_d: pd.DataFrame | None, cal_g: pd.DataFrame | None) -> str:
    rows = []
    for k in KS:
        m = metrics.get(f"k{k}")
        if not m:
            continue
        A, D, B, G = m["A"], m["D"], m["B"], m["G12"]
        al, ar = A["lgbm_test"], A["rule_test"]
        rows.append([f"k={k}", f"{al['n']:,}",
                     f"{f3(ar['qwk'])}{ci(ar.get('qwk_ci95'))}", f"<span class=ai>{f3(al['qwk_expected_cut'])}</span>{ci(al.get('qwk_expected_cut_ci95'))}",
                     f"{f3(ar['auc_ge_L4'])}{ci(ar.get('auc_ge_L4_ci95'))}", f"<span class=ai>{f3(al['auc_ge_L4'])}</span>{ci(al.get('auc_ge_L4_ci95'))}",
                     f"{f3(D['rule_test']['auc'])}", f"<span class=ai>{f3(D['lgbm_test']['auc'])}</span>{ci(D['lgbm_test'].get('auc_ci95'))}",
                     f"{f3(B['rule_test']['auc'])}", f"<span class=ai>{f3(B['lgbm_test']['auc'])}</span>{ci(B['lgbm_test'].get('auc_ci95'))}",
                     f"{f3(G['rule_test']['auc'])}", f"<span class=ai>{f3(G['lgbm_test']['auc'])}</span>{ci(G['lgbm_test'].get('auc_ci95'))}",
                     f"{f3(D['lgbm_test'].get('ece10'))} / {f3(B['lgbm_test'].get('ece10'))} / {f3(G['lgbm_test'].get('ece10'))}"])
    hdr = ["基準点", "n(走った)", "QWK 規則", "QWK AI", "AUC 重賞 規則", "AUC 重賞 AI", "離脱 AUC 規則", "離脱 AUC AI",
           "重賞24m AUC 規則", "重賞24m AUC AI", "統合12m AUC 規則", "統合12m AUC AI", "ECE10 AI (D/B/G12)"]
    out = [f"""
<h2 id="s2">2. 成績（テスト = 2022 デビュー）</h2>
<p>QWK は到達クラス 5 段の二乗重み付き κ（期待レベルを検証で決めた閾値で丸めた値）。AUC 重賞は「重賞掲示板 (L4) か否か」の判別。
<span class="ai">青字が AI</span>、隣が規則ベースライン。[ ] は 95% 信頼区間（馬単位ブートストラップ）。</p>
{table(hdr, rows)}
<p class="small">規則との差は k が小さいほど大きい（k=1 で QWK +0.43、k=5 で +0.12）。デビュー戦直後の 1 年後予測は規則ではほぼ不可能（QWK 0.17）ですが AI は 0.61。
離脱 AUC の 0.79〜0.86 は「大敗が続いた馬の登録抹消」を当てている分で、好走馬の故障は読めません（§7）。</p>
"""]
    # unwon stratum
    rows2 = []
    for k in KS:
        m = metrics.get(f"k{k}")
        if not m:
            continue
        u = m["A"].get("lgbm_test_unwon")
        ru = m["A"].get("rule_test_unwon")
        if u:
            rows2.append([f"k={k}", f"{u['n']:,}", f"<span class=ai>{f3(u['qwk_expected_cut'])}</span>{ci(u.get('qwk_expected_cut_ci95'))}",
                          f"{f3(ru['auc_ge_L1'])} / <span class=ai>{f3(u['auc_ge_L1'])}</span>", f"{f3(ru.get('auc_ge_L4'))} / <span class=ai>{f3(u.get('auc_ge_L4'))}</span>",
                          f"{u['level_dist'].get('4', 0) * 100:.1f}%"])
    out.append(f"""<h3>未勝利圏にいる馬だけの層別（規則は全頭「未勝利圏」と言うので QWK 0）</h3>
{table(["基準点", "n", "QWK AI", "AUC ≥1勝 規則 / AI", "AUC 重賞 規則 / AI", "層内の重賞掲示板率"], rows2)}
<p class="small">未勝利馬の中から重賞馬を拾う価値があるのは k=1〜2。k≥3 では層内の正例が 8 頭以下で AUC は信頼できません。</p>""")
    # confusion
    if conf is not None:
        names = [LEVEL_NAME[i] for i in range(5)]
        crow = []
        for i in range(5):
            r = conf.iloc[i]
            vals = [int(r[str(j)]) for j in range(5)]
            crow.append([f"実 {names[i]}"] + [f"<strong>{v}</strong>" if j == i else str(v) for j, v in enumerate(vals)] + [str(sum(vals))])
        out.append(f"""<h3>混同行列（k=3、走った 3,407 頭、行 = 実際 / 列 = 予測）</h3>
{table(["", *[f"予 {n}" for n in names], "計"], crow)}
<p class="small">重賞掲示板と予測した 205 頭のうち 69 頭（34%）が掲示板、107 頭は重賞着外以上（3勝/OP 扱い）。実際の重賞掲示板 110 頭の 63% を捕捉し、未勝利圏/1勝に落としたのは 4 頭。</p>""")
    # calibration SVG
    figs = []
    if cal_d is not None:
        figs.append(calib_svg(cal_d, "離脱（D）k=3", "#b23a3a"))
    if cal_g is not None:
        figs.append(calib_svg(cal_g, "統合 P(重賞掲示板 12m)（G12）k=3", "#b8860b"))
    if figs:
        out.append("<h3>較正（予測確率 vs 実測率、10 分位）</h3><div class='twocol'>" + "".join(figs) + "</div>"
                   "<p class='small'>点が対角線に乗っていれば「○%」をそのまま確率として読めます。D の最上位分位は予測 57% / 実測 56%、G12 は 22% / 24%。</p>")
    return "".join(out)


def calib_svg(cal: pd.DataFrame, title: str, color: str) -> str:
    W, H, m = 400, 320, 46
    xs, ys = cal["pred_mean"].astype(float).values, cal["obs_rate"].astype(float).values
    ns = cal["n"].astype(int).values
    vmax = max(0.05, float(np.nanmax(np.concatenate([xs, ys]))) * 1.1)
    vmax = min(1.0, vmax)

    def X(v):
        return m + (W - 2 * m) * v / vmax

    def Y(v):
        return H - m - (H - 2 * m) * v / vmax

    ticks = np.linspace(0, vmax, 6)
    parts = [f"<svg viewBox='0 0 {W} {H}' width='100%' role='img' aria-label='{esc(title)}' style='max-width:{W}px;font-family:inherit'>",
             f"<rect x='0' y='0' width='{W}' height='{H}' fill='#fff'/>",
             f"<line x1='{X(0):.1f}' y1='{Y(0):.1f}' x2='{X(vmax):.1f}' y2='{Y(vmax):.1f}' stroke='#9aa3ad' stroke-dasharray='4 4'/>",
             f"<line x1='{m}' y1='{H - m}' x2='{W - m}' y2='{H - m}' stroke='#333'/>",
             f"<line x1='{m}' y1='{m}' x2='{m}' y2='{H - m}' stroke='#333'/>"]
    for t in ticks:
        parts.append(f"<text x='{X(t):.1f}' y='{H - m + 16}' font-size='11' text-anchor='middle' fill='#333'>{t * 100:.0f}%</text>")
        parts.append(f"<text x='{m - 6}' y='{Y(t) + 4:.1f}' font-size='11' text-anchor='end' fill='#333'>{t * 100:.0f}%</text>")
        parts.append(f"<line x1='{m}' y1='{Y(t):.1f}' x2='{W - m}' y2='{Y(t):.1f}' stroke='#eef1f4'/>")
    pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))
    parts.append(f"<polyline points='{pts}' fill='none' stroke='{color}' stroke-width='1.5' opacity='.7'/>")
    for x, y, n in zip(xs, ys, ns):
        parts.append(f"<circle cx='{X(x):.1f}' cy='{Y(y):.1f}' r='4.5' fill='{color}'><title>予測 {x * 100:.1f}% / 実測 {y * 100:.1f}% (n={n})</title></circle>")
    parts.append(f"<text x='{W / 2:.0f}' y='{H - 8}' font-size='12' text-anchor='middle' fill='#333'>予測確率（分位平均）</text>")
    parts.append(f"<text x='14' y='{H / 2:.0f}' font-size='12' text-anchor='middle' fill='#333' transform='rotate(-90 14 {H / 2:.0f})'>実測率</text>")
    parts.append(f"<text x='{W / 2:.0f}' y='{m - 18}' font-size='13' font-weight='600' text-anchor='middle' fill='#1f2a37'>{esc(title)}</text>")
    parts.append("</svg>")
    return f"<figure>{''.join(parts)}<figcaption>破線 = 完全較正。下位分位は 0〜1% に密集します。</figcaption></figure>"


def sec_v1v2(v1c: dict | None, cards_named: dict[str, dict]) -> str:
    if not v1c:
        return "<h2 id='s3'>3. v1 → v2 で何が変わったか</h2><p>v1_comparison.json が無いため省略。</p>"
    rows = []
    order = ["v1_def_all_horses", "v1_def_model_eval_on_ran", "dropout_separated_v1_L4", "new_def_dropout_separated_board_L4"]
    for key in order:
        r = v1c["rows"].get(key)
        if not r:
            continue
        rule = r.get("rule") or {}
        rows.append([esc(r["desc"]), f"{r['n']:,}", f"<span class=ai>{f3(r['qwk_expected_cut'])}</span>{ci(r.get('qwk_expected_cut_ci95'))}",
                     f"<span class=ai>{f3(r['auc_ge_L4'])}</span>{ci(r.get('auc_ge_L4_ci95'))}",
                     (f"{f3(rule.get('qwk'))} / {f3(rule.get('auc_ge_L4'))}" if rule else "—")])
    rc = v1c.get("L4_reclassification_test", {})
    named = v1c.get("named_horses", [])
    nrows = []
    for h in named:
        act = "出走なし（離脱）" if h.get("actual_ran_12m") == 0 else LEVEL_NAME.get(h.get("actual_y_level_new"), "—")
        nrows.append([esc(h["馬名"]), pct(h.get("v1_model_pL4")), pct(h.get("new_p_dropout")), pct(h.get("new_pL4_given_ran")), pct(h.get("new_pL3_given_ran")), pct(h.get("new_p_graded12")), act])
    picks = [n for n in ("ミッキーカプチーノ", "ピピオラ", "ドゥレッツァ") if n in cards_named]
    cards_html = "".join(card_html(cards_named[n], compact=True) for n in picks)
    return f"""
<h2 id="s3">3. v1 → v2 で何が変わったか</h2>
<ol>
<li><strong>着差を自走の走破タイムから付けた</strong>（v1 は次走の「前走着差」を持ち上げていたため 3 走目の着差が使えなかった）。上り3F・上り3F順位・馬体重・デビュー時日齢も追加。</li>
<li><strong>離脱を別ヘッドに分離</strong>。v1 では「1 年間走らなかった馬」が未勝利圏 (L0) に混ざっていた（test k=3 の v1-L0 {rc.get('v1_L0_count', '—')} 頭のうち {rc.get('v1_L0_that_are_dropout', '—')} 頭 = 23% が離脱）。</li>
<li><strong>L4 を「重賞で 5 着以内（掲示板）」に変更</strong>。v1 の「重賞に出走」だと格上挑戦で惨敗した馬も L4 だった（v1-L4 {rc.get('v1_L4_count', '—')} 頭 → 掲示板 {rc.get('v1_L4_now_L4_board', '—')} 頭、着外 {rc.get('v1_L4_now_L3_unplaced', '—')} 頭は L3 へ）。</li>
<li>基準点を 3 走目だけでなく k = 1, 2, 3, 5 の 4 点に。騎手の率を馬×騎手ペアではなく騎手一般の as-of 3 着内率に。</li>
</ol>
{table(["設定（k=3、同じ特徴・同じパラメータ）", "n", "QWK AI", "AUC 重賞 AI", "規則 QWK / AUC"], rows)}
<p class="small">QWK は v1 定義から動かず（0.720 → 0.726、CI 内）。改善は <strong>AUC 重賞 0.939 → 0.952</strong> と、タスクの意味の明確化（「競走の結果」だけを A に、離脱は別に）。
規則ベースラインも新定義で上がる（0.538 → 0.552）ので、改善の一部はタスクが素直になったことによります。</p>
<h3>名指し馬: v1 の弱点がどうなったか</h3>
{table(["馬名", "v1 P(重賞)", "新 P(離脱)", "新 P(掲示板|走る)", "新 P(重賞着外|走る)", "新 P(掲示板 12m)", "実際"], nrows)}
<p class="small">ミッキーカプチーノ型（好走後に故障離脱）はラベル上は A の誤りから外れたが、<strong>離脱そのものは予測できていない</strong>（P(離脱) 4% は中央値程度）。ピピオラ型（未勝利のまま秋華賞 14 着）は定義変更で解消。</p>
{cards_html}
"""


def sec_features(labels: dict, fi: dict[int, pd.DataFrame], abl: dict | None) -> str:
    cols = []
    for k in (1, 3):
        d = fi.get(k)
        if d is None:
            continue
        top = d.head(10)
        rows = [[esc(labels.get(r.feature, r.get("label_ja", r.feature))), f"{float(r.gain_share) * 100:.1f}%"] for _, r in top.iterrows()]
        cols.append(f"<div><h4>A（到達クラス）k={k}</h4>{table(['特徴', 'gain 比'], rows)}</div>")
    out = [f"""
<h2 id="s4">4. 効いた特徴</h2>
<p>LightGBM の gain 比（学習データから。テストは見ていない）。v1 で取れなかった<strong>自走の着差</strong>が全ヘッドで上位、<strong>デビュー時日齢</strong>（早生まれ・早期デビュー）が新情報として上位 10 に入ります。</p>
<div class="twocol">{''.join(cols)}</div>"""]
    if abl:
        for k in ("k1", "k3"):
            a = abl.get(k)
            if not a:
                continue
            res = a["results"]
            names = {"perf_only": "成績のみ", "perf+jockey": "+騎手", "perf+pedigree_stable": "+血統・厩舎・生産者", "perf+body": "+身体（馬体重・日齢）", "all": "全部"}
            rows = [[names.get(key, key), str(v["n_feats"]), f3(v["A_qwk_test"]), f3(v["A_auc_L4_test"]), f3(v["A_auc_L1_test"]), f3(v["D_auc_test"]), f3(v["B_auc_test"]), f3(v["G12_auc_test"])] for key, v in res.items()]
            out.append(f"<h4>群別アブレーション（{k}、テスト、パラメータ固定）</h4>" + table(["特徴群", "n_feats", "QWK A", "AUC 重賞 A", "AUC ≥1勝 A", "AUC D", "AUC B", "AUC G12"], rows))
        out.append("<p class='small'>k=1 でも血統・厩舎の寄与は小さい（QWK +0.005）。「デビュー直後は血統・厩舎で占う」という読み物上の仮説は数値では支持されず、1 走目の着順・着差・上り3F・日齢がほぼ全部を決めます。差は全て CI 内。</p>")
    return "".join(out)


def _margin_cell(d: dict) -> str:
    if d.get("win") or d.get("margin") is None:
        return "—"
    return f"{float(d['margin']):.1f}"


def card_html(c: dict, compact: bool = False) -> str:
    runs = c.get("runs") or []
    rrows = "".join(
        f"<tr><td class=l>{i}</td><td class=l>{esc(d.get('date') or '')}</td><td class=l>{esc(d.get('place') or '')}</td><td class=l>{esc(d.get('label'))}</td>"
        f"<td class=l>{esc((d.get('surface') or '') + (str(d.get('dist')) if d.get('dist') else ''))}</td><td>{d.get('pos') if d.get('pos') is not None else '—'}</td><td>{d.get('field') or '—'}</td>"
        f"<td>{_margin_cell(d)}</td></tr>"
        for i, d in enumerate(runs, 1))
    runs_tbl = f"<table class='runs'><thead><tr><th>走</th><th>日付</th><th>場所</th><th>レース</th><th>芝ダ距離</th><th>着</th><th>頭</th><th>着差(秒)</th></tr></thead><tbody>{rrows}</tbody></table>" if runs else ""
    bars = []
    pl = c.get("p_levels")
    if pl:
        for j in range(5):
            bars.append(f"<div>{LEVEL_NAME[j]}</div><div class='bar'><i style='width:{pl[j] * 100:.1f}%'></i></div><div>{pct(pl[j])}</div>")
    if c.get("p_dropout") is not None:
        bars.append(f"<div>離脱 12m</div><div class='bar drop'><i style='width:{c['p_dropout'] * 100:.1f}%'></i></div><div>{pct(c['p_dropout'])}</div>")
    if c.get("p_graded12") is not None:
        bars.append(f"<div>重賞掲示板 12m</div><div class='bar g12'><i style='width:{c['p_graded12'] * 100:.1f}%'></i></div><div>{pct(c['p_graded12'])}</div>")
    bars_html = f"<div class='bars'>{''.join(bars)}</div>" if bars else ""
    oc = c.get("outcome") or {}
    v = oc.get("verdict") or ""
    vcls = "v-ok" if v.startswith("的中") else ("v-near" if v.startswith("惜しい") else ("v-ng" if v else ""))
    why = f"<span class='why'>{esc(c['why'])}</span>" if c.get("why") else ""
    meta = f"k={c.get('k')}・基準日 {esc(c.get('base_date') or '—')}" + (f"・{c.get('debut_year')} 年デビュー" if c.get("debut_year") else "")
    verdict = f"<span class='verdict {vcls}'>{esc(v)}</span>" if v else ""
    body = md_inline(c.get("text", ""))
    inner = f"<h4>{esc(c.get('馬名'))}{why}</h4><div class='meta'>{meta} {verdict}</div><div class='txt'>{body}</div>"
    if not compact:
        inner += runs_tbl + bars_html
    else:
        inner += bars_html
    return f"<div class='card'>{inner}</div>"


def pick_cards(preds: dict[int, pd.DataFrame], starts: pd.DataFrame, labels: dict) -> tuple[list[dict], dict[str, dict]]:
    t3 = preds[3][preds[3]["split"] == "test"]
    t1 = preds[1][preds[1]["split"] == "test"]
    named_order = list(dict.fromkeys(NAMED_SPEC + NAMED_V1))
    nm = t3[t3["馬名"].isin(named_order)].set_index("馬名").reindex([n for n in named_order if n in set(t3["馬名"])]).reset_index()
    named_cards = cards_for(nm, starts, labels, 3, "名指し馬", t3)
    named_map = {c["馬名"]: c for c in named_cards}
    used = set(nm[COL_HORSE])
    cards: list[dict] = []
    # 的中した大物 3
    hit = t3[(~t3[COL_HORSE].isin(used)) & (t3["y_level"] == 4)].nlargest(3, "p_graded12")
    cards += cards_for(hit, starts, labels, 3, "的中: P(重賞) 上位で掲示板到達", t3)
    used |= set(hit[COL_HORSE])
    # 名指し 6
    for n in ("リバティアイランド", "ドゥレッツァ", "ドゥラエレーデ", "ミッキーカプチーノ", "ピピオラ", "スキルヴィング"):
        if n in named_map:
            cards.append(named_map[n])
    # 規則と割れた（k=1 未勝利圏 → AI 重賞 20%+）4
    unwon = t1[(t1["rule_level"] == 0) & (t1["p_graded12"] >= 0.2)].nlargest(4, "p_graded12")
    cards += cards_for(unwon, starts, labels, 1, "規則は未勝利圏 / AI は重賞 20%+（k=1）", t1)
    # 外れ: 離脱読めず 1、P(重賞)最小で到達 2、離脱 85% なのに走った 1、期待外れ 1
    rest = t3[~t3[COL_HORSE].isin(used)]
    mis = [
        (rest[rest["ran_12m"] == 0].nsmallest(1, "p_dropout"), "外れ: 離脱確率最小なのに離脱"),
        (rest[rest["y_level"] == 4].nsmallest(2, "p_graded12"), "外れ: P(重賞) 最小で掲示板到達"),
        (rest[rest["ran_12m"] == 1].nlargest(1, "p_dropout"), "外れ寄り: 離脱 85% なのに走り続けた"),
        (rest[rest["y_level"] == 0].nlargest(1, "p_graded12"), "外れ: P(重賞) 高いのに未勝利圏"),
    ]
    for df, why in mis:
        cards += cards_for(df, starts, labels, 3, why, t3)
    # 更新例 1 頭（k=1 → k=3）
    for k, t in ((1, t1), (3, t3)):
        rr = t[t["馬名"] == "ソールオリエンス"]
        if len(rr):
            r = rr.iloc[0]
            cards.append(make_card(r, starts[starts[COL_HORSE] == r[COL_HORSE]], labels, k=k, cohort=t["p_graded12"], why=f"更新例: {k} 走目時点"))
    return cards, named_map


def sec_cards(cards: list[dict]) -> str:
    return f"""
<h2 id="s5">5. 読み物カード（テスト世代 = 2022 デビュー、{len(cards)} 枚）</h2>
<p>基準日時点の情報だけで書いた文章と、その後 12 か月の実際。的中・外れ・AI と規則が割れた例を混ぜています。文章はテンプレート（LLM 不使用）、
確率は整数 %、断定はしません。棒グラフは「走り続けた場合」の 1 年後クラス確率（青）、離脱確率（赤）、統合 P(重賞掲示板 12m)（金）。</p>
{''.join(card_html(c) for c in cards)}
"""


# --------------------------------------------------------------------------- live
def _find_live_rows(obj) -> list[dict]:
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        if isinstance(obj.get("horses"), list):
            return [x for x in obj["horses"] if isinstance(x, dict)]
        rows = []
        for v in obj.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                rows += v
            elif isinstance(v, dict):
                sub = _find_live_rows(v)
                if sub:
                    rows += sub
        return rows
    return []


def _normalize_live(df: pd.DataFrame) -> pd.DataFrame:
    """predict_live の列名を pred_test_k{k} と同じ形に揃える（json / parquet 両対応）。"""
    df = df.copy()
    ren = {"ped_id": COL_HORSE, "name": "馬名", "性別": "sex", "父": "sire", "調教師コード": "trainer_debut", "trainer_code": "trainer_debut",
           "expected_level": "expected_level_given_ran", "pred_level": "pred_level_given_ran"}
    df = df.rename(columns={k: v for k, v in ren.items() if k in df and v not in df})
    for j in range(5):
        if f"p_L{j}" in df and f"p_L{j}_given_ran" not in df:
            df[f"p_L{j}_given_ran"] = df[f"p_L{j}"].astype(float)
    if "p_level" in df and "p_L0_given_ran" not in df:
        for j in range(5):
            df[f"p_L{j}_given_ran"] = df["p_level"].map(lambda v: float(v[j]) if isinstance(v, (list, tuple)) and len(v) > j else np.nan)
    if "reasons" in df and "top3_A_L4" not in df:
        rmap = {"L4": "top3_A_L4", "dropout": "top3_D", "graded24": "top3_B", "L0": "top3_A_L0"}
        for src, dst in rmap.items():
            df[dst] = df["reasons"].map(lambda v: json.dumps(v.get(src)) if isinstance(v, dict) and v.get(src) is not None else None)
    if "p_graded12" not in df and {"p_dropout", "p_L4_given_ran"} <= set(df.columns):
        df["p_graded12"] = (1 - df["p_dropout"].astype(float)) * df["p_L4_given_ran"].astype(float)
    if "base_date" in df:
        df["base_date"] = pd.to_datetime(df["base_date"], errors="coerce")
    # 最新基準点・未確定のみ（parquet は履歴と確定済み行も持つ）
    if "is_latest" in df:
        df = df[df["is_latest"].astype(bool)]
    if "confirmed" in df:
        df = df[~df["confirmed"].astype(bool)]
    return df.reset_index(drop=True)


def load_live() -> tuple[pd.DataFrame | None, dict]:
    meta: dict = {}
    jp, pq = LIVE / "shusse_live.json", LIVE / "predictions_latest.parquet"
    df = None
    if pq.exists():                       # parquet を優先（列が揃っていて is_latest/confirmed がある）
        df = pd.read_parquet(pq)
        meta["source"] = pq.name
    if df is None and jp.exists():
        obj = load_json(jp)
        rows = _find_live_rows(obj)
        if rows:
            df = pd.DataFrame(rows)
            meta["source"] = jp.name
    if jp.exists():
        obj = load_json(jp)
        if isinstance(obj, dict):
            meta.update({k: v for k, v in obj.items() if not isinstance(v, (list, dict)) and k in ("model", "asof", "generated_at", "n_horses")})
    if df is None or not len(df):
        return None, meta
    df = _normalize_live(df)
    return (df if len(df) else None), meta


def sec_live(labels: dict, starts_default: pd.DataFrame) -> tuple[str, bool]:
    df, meta = load_live()
    if df is None:
        return ("<h2 id='s6'>6. 現役馬（2023〜 デビュー）</h2><p class='note'>未生成: <code>out_v2/live/shusse_live.json</code> / "
                "<code>predictions_latest.parquet</code> が無いため、この節は省略しました（<code>predict_live.py</code> 実行後に再生成してください）。</p>"), False
    starts = starts_default
    for sp in (LIVE / "starts_latest.parquet", LIVE / "starts_ext.parquet"):   # extend_2026.py の出力名は starts_ext
        if sp.exists():
            starts = pd.read_parquet(sp)
            break
    if "k" not in df and "base_date" in df and COL_HORSE in df:
        cnt = starts.groupby(COL_HORSE)["date"].apply(list)
        df["k"] = [int(sum(d <= b for d in cnt.get(h, []))) if pd.notna(b) else 0 for h, b in zip(df[COL_HORSE], df["base_date"])]
    if "debut_year" not in df:
        if COL_HORSE in df:
            dy = starts.groupby(COL_HORSE)["date"].min().dt.year
            df["debut_year"] = df[COL_HORSE].map(dy)
        else:
            df["debut_year"] = 0
    out = [f"<h2 id='s6'>6. 現役馬（デビュー年別の上位 20 頭）</h2>",
           f"<p class='small'>入力: {esc(meta.get('source'))}" + "".join(f"、{esc(k)}: {esc(v)}" for k, v in meta.items() if k != "source") + f"、{len(df):,} 頭。"
           "結果は未確定なので的中判定は無し。P(重賞掲示板 12m) 降順。</p>"]
    for y, g in sorted(df.groupby("debut_year"), key=lambda t: t[0]):
        if not y:
            continue
        g = g.sort_values("p_graded12", ascending=False)
        top = g.head(20)
        rows = []
        cards = []
        for i, (_, r) in enumerate(top.iterrows()):
            sh = starts[starts[COL_HORSE] == r[COL_HORSE]] if COL_HORSE in r else pd.DataFrame()
            c = make_card(r, sh, labels, k=(int(r["k"]) if "k" in r and pd.notna(r["k"]) else None), cohort=g["p_graded12"], why=f"{int(y)} 年デビュー 上位 {i + 1}")
            runs = " → ".join(start_phrase(d) for d in c["runs"]) if c["runs"] else "—"
            reason = "、".join(x["phrase"] for x in c["reasons"]) or "—"
            rows.append([esc(c["馬名"]), str(c["k"]), esc(runs), pct(c["p_graded12"]), pct(c["p_dropout"]), esc(c.get("pred_level_name") or "—"), esc(reason)])
            if i < 3:
                cards.append(c)
        out.append(f"<h3>{int(y)} 年デビュー（{len(g):,} 頭）</h3>")
        out.append(table(["馬名", "k", "走の要約", "P(重賞掲示板 12m)", "P(離脱)", "予測クラス", "根拠"], rows))
        out.append("".join(card_html(c) for c in cards))
    return "".join(out), True


def sec_limits() -> str:
    return """
<h2 id="s7">7. 限界と落とし穴</h2>
<ol>
<li><strong>離脱は 2 種類ある。</strong>当てられるのは「大敗が続いた馬の登録抹消」。好走馬の故障離脱（ミッキーカプチーノ 4%、ガストリック 4%）は外形情報からは予測不能。読み物では「走れば ○%」と「走れない確率 ○%」を分けて出すのが誠実。</li>
<li><strong>QWK は v1 から改善していない</strong>（0.720 → 0.726、CI 内）。改善は AUC 重賞（0.939 → 0.952）と、タスクの意味の明確化。</li>
<li><strong>重賞掲示板 (L4) の正例が薄い。</strong>テスト k=3 で 110 頭、未勝利圏層では 8 頭、k=5 の未勝利圏層では 0 頭。層別の AUC 重賞は k≥3 では信頼できない。</li>
<li><strong>父タイプ名は年を追って欠損が増える</strong>（真のデビュー馬 41.5%、2022 行で 60%、2025 行で 86%）。現役馬の推論ではほぼ全て欠損。アブレーションでは血統群なしでも同等以上なので実害は小さい。</li>
<li><strong>騎手 as-of 率は 2013 年に左打ち切り</strong>（2 年窓）。学習の 2013 コホートではやや弱い情報。</li>
<li><strong>2023 デビューの数字は選択バイアス</strong>（k=5 は 2024 年内に 5 走した早熟・多走馬のみ）。記述用。</li>
<li><strong>ハイパーパラメータは 9 点グリッド・seed 1 本。</strong>seed 分散は未測定（QWK の CI 幅 ±0.02）。</li>
<li><strong>外部 kekka の race_id は約 4.4% のレースで壊れている。</strong>(日付, 場所, R, 血統登録番号) で結合して回避した。</li>
<li><strong>文章はテンプレート。</strong>根拠の「向き」は特徴の語幹辞書で付けており、距離・頭数・日齢・馬体重・月・年齢・父系タイプなど向きを決められない特徴はラベルだけを出す。レース名は TARGET 出力で 4 文字に切れており、主要レースだけ補完している。</li>
<li><strong>市場オッズ・人気は不使用。馬券への転用はしない</strong>（予測層天井の既知結論に抵触する）。</li>
</ol>
"""


# --------------------------------------------------------------------------- main
def build() -> Path:
    t0 = time.time()
    metrics = load_json(OUT / "metrics_v2.json") or {}
    v1c = load_json(OUT / "v1_comparison.json")
    abl = load_json(OUT / "ablation_v2.json")
    join = load_json(OUT / "join_report.json")
    labels = load_labels(OUT)
    conf = pd.read_csv(OUT / "confusion_A_k3.csv", encoding="utf-8-sig") if (OUT / "confusion_A_k3.csv").exists() else None
    cal_d = pd.read_csv(OUT / "calibration_D_k3.csv", encoding="utf-8-sig") if (OUT / "calibration_D_k3.csv").exists() else None
    cal_g = pd.read_csv(OUT / "calibration_G12_k3.csv", encoding="utf-8-sig") if (OUT / "calibration_G12_k3.csv").exists() else None
    fi = {k: pd.read_csv(OUT / f"feature_importance_A_k{k}.csv", encoding="utf-8-sig") for k in (1, 3) if (OUT / f"feature_importance_A_k{k}.csv").exists()}
    starts = pd.read_parquet(OUT / "starts_v2.parquet")
    preds = {k: pd.read_parquet(OUT / f"pred_test_k{k}.parquet") for k in (1, 3)}
    cards, named_map = pick_cards(preds, starts, labels)
    live_html, has_live = sec_live(labels, starts)

    gen = time.strftime("%Y-%m-%d %H:%M")
    toc = "".join(f"<a href='#s{i}'>{i}. {t}</a><br>" for i, t in enumerate(
        ["この AI は何をするか", "成績", "v1 → v2", "効いた特徴", "読み物カード", "現役馬", "限界と落とし穴"], 1))
    body = (sec_overview(metrics, join) + sec_metrics(metrics, conf, cal_d, cal_g) + sec_v1v2(v1c, named_map)
            + sec_features(labels, fi, abl) + sec_cards(cards) + live_html + sec_limits())
    doc = f"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>出世予測 AI v2 — レポート</title><style>{CSS}</style></head>
<body><main>
<h1>出世予測 AI v2 — レポート</h1>
<p class="lead">JRA 2〜3 歳馬の「1 年後の居場所」を、k 走目までの成績だけから読む。馬券非転用の読み物用 AI。</p>
<p class="meta">生成 {gen}・テスト = 2022 デビュー・データ 2013〜2025-12-28・現役馬節 {'あり' if has_live else '未生成'}・
コード <code>analysis/shusse/</code>（build_career_v2 / features_v2 / train_v2 / narrate / report_v2）</p>
<nav class="toc">{toc}</nav>
{body}
<p class="small" style="margin-top:40px">文章はテンプレート生成（LLM 不使用）。数値の正本は <code>out_v2/metrics_v2.json</code>。詳細は <code>docs/research/SHUSSE_AI_V2_RESULTS_20261031.md</code>。</p>
</main></body></html>
"""
    REPORT.write_text(doc, encoding="utf-8")
    print(f"wrote {REPORT} ({REPORT.stat().st_size / 1024:.0f} KB, {len(cards)} cards, live={'yes' if has_live else 'no'}) in {time.time() - t0:.1f}s")
    return REPORT


if __name__ == "__main__":
    build()
