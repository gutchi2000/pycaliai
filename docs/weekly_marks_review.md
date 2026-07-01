# 週次 ◎〇▲△△ レース回顧ルーティーン

土日の bundle.json に付いた **印（◎〇▲△△ = top5）の全レース**を回顧する手順。
**回顧 = 着順の答え合わせではない**。1頭ずつ「**なぜその結果か**」「**次走どう狙うか**」を
言語化し、馬名キーDBに貯める＝次にその馬が出たとき「前走こうだった」を即引ける資産にする。
**kekka.csv が無い段階**（＝結果をWebで調べる）でも回せる。

> 最終更新: 2026-06-07（次走資産版に改訂。`build_uma_review.py` が本体）。
> kekka.csv が揃ったら `analysis/review_day.py`（買い目収支版）も併用可。

---

## 🟢 TL;DR — あなた(人間)がやることはこれだけ
1. 「**回顧**」と言う（土日のレース確定後）。→ あとはClaudeが結果取得〜CSV生成まで全部やる。
2. Claudeが渡す **取込CSV** を TARGET の **「ファイルからのコメント一括登録」** で読み込む。
   - Web版CSV: `reports/review/{土}_{日4桁}_target_import.csv`（例 `20260613_0614_target_import.csv`）
3. （任意・数日後）もっと濃くしたい時だけ：TARGETから **「レース回顧」CSVをエクスポート**し、
   **`data/target_review/` に置く**だけ。→ Claudeが `build_full_review.py` で全頭・濃い版CSVを作る。
   これだけが「あなたが用意するファイル」（置き場所が決まったので、そこに入れるだけ）。

> つまり **普段あなたが準備するファイルは無し**。「回顧」と言う→出てきたCSVを取り込む、だけ。
> `export_target_comments.py 20260613 20260614` 等のpythonコマンドは **Claudeが回す**もので、
> あなたが叩く必要はない（20260613 20260614＝その週の土日の日付の例）。

### 裏で動く順番（Claude担当）
①Web結果取得 → `reports/web_results/{date}.json`
②`build_uma_review.py` → 回顧＋馬名DB（`data/uma_review/`）
③`export_target_comments.py {土} {日}` → 取込CSV（`reports/review/..._target_import.csv`）

---

## いつやるか
- 土日のレース確定後（日曜夜〜月曜）。週次 Phase C と同タイミング。
- トリガー語: ユーザーが「**回顧**」と言ったら下記を実行。

## 何を残すか（回顧コメントの型）
印が付いた5頭（◎=1位/〇=2位/▲=3位/△=4,5位）＋**AIが無印なのに着内(複勝圏)に来た馬**に
ついて 1頭ずつ：
- **なぜ**: 人気・AI評価との乖離・前走脚質/適性・(重賞は)展開や位置取り・敗因/勝因
- **次走メモ**: 狙い/消し ＋ 条件（距離・コース・展開・人気妙味）
- **取りこぼし馬(無印で着内)**: `mark="無"`, `ai_miss=true` でDB化。
  「なぜ本モデルが見落としたか＋次走扱い」を残す。特に**単5倍未満の人気馬を無印**にした
  ケースはモデルの弱点ログ（例: 1.1倍1人気を rank7 等）。馬名DBにも入るので無印好走馬も追える。
- 観点の軸: ◎複勝圏率 天井≈62%（[memory] prediction_accuracy_ceiling）、
  穴推し◎(高オッズ`vs:under`)は構造的に弱い（[memory] longshot_weakness）。

---

## 手順（3ステップ）

### 1. 印馬リストの確認（任意）
```bash
venv311\Scripts\python.exe analysis\review_marks_web.py        # 最新土日を自動検出
```

### 2. Webで着順を調べて結果JSONを作る ★ここがClaudeの作業
スポーツナビの「開催日別レース一覧」を WebFetch すると、**1ページで12R分の
1-2-3着（馬番・馬名）**がまとまって取れる（最も安定）。

- URL形式: `https://sports.yahoo.co.jp/keiba/race/list/{YY}{場コード}{回}{日}`
  - 場コード(=JRA標準): 札幌01 / 函館02 / 福島03 / 新潟04 / 東京05 / 中山06 / 中京07 / 京都08 / 阪神09 / 小倉10
  - 例: 26 05 03 02 = 2026年・東京・3回・2日目 → 日曜東京
  - 我々の race_id `2026060705030211` の場合 netkeiba_id = `2026 + rid[8:16]` = `202605030211`、
    sports.yahoo list id = `26 + rid[8:14]`(=場2桁+回2桁+日2桁) = `26050302`
  - 一括で list id を出すには:
    `python -c "import json;b=json.load(open(r'reports/cowork_input/{date}_bundle.json',encoding='utf-8'));s={};[s.setdefault(r['race_meta']['place'],'26'+r['race_id'][8:14]) for r in b['races']];print(s)"`

- 取得した 1-2-3着を `reports/web_results/{date}.json` に保存:
  ```json
  {"date":"YYYYMMDD","source":"...",
   "results":{"東京":{"1":[[馬番,"馬名"],[..],[..]], "2":[...]}, "阪神":{...}}}
  ```

- **信頼性チェック必須**: 取れた「馬番=馬名」が bundle の印馬と一致するか必ず確認する。
  - ⚠️ netkeiba の `result.html` は WebFetch で**ハルシネーションする**ことがある
    （2026-06-07 検証で勝ち馬を捏造）。スポーツナビ list を一次ソースにし、G1等は
    umanity 入線速報 (`umanity.jp/racedata/race_newsdet.php`) でクロスチェック。

### 2b. 重賞など本物の展開が取れたレースは手書き回顧を注入（任意）
Webに回顧記事がある重賞/OP/特別は、展開・位置取り・着差・コメントを調べて
`reports/web_results/overrides_{date}.json` に手書き回顧を入れる（平場は自動生成でOK）。
```json
{"race_notes": {"東京|11":"G1安田記念。前傾消耗戦で…"},
 "horse_notes": {"東京|11|6": {"result_override":"10着",
   "kaiko":"AI最上位も大敗。マイル左回りG1不向き…",
   "jisou":"牝馬限定/1800〜2000m替わりで見直し。人気なら消し。"}}}
```
キー = `place|R`(レース) / `place|R|馬番`(馬)。

### 3. 回顧＋次走メモを生成し、馬名DBに蓄積 ★本体
```bash
venv311\Scripts\python.exe analysis\build_uma_review.py        # 最新土日を自動検出
# 明示: ... build_uma_review.py 20260606 20260607
```
- 平場は「1-3着 + bundleの過去走脚質 + AI乖離 + 人気」から**次走に効く所見を自動生成**。
- 重賞は overrides の手書き回顧を優先注入。
- 出力3点: `data/uma_review/{date}.json`(その日の全印馬) /
  `data/uma_review/horse_notes.json`(**馬名キー蓄積DB**) /
  `reports/review/{weekend}_kaiko.md`(読み物)。

### 3b. TARGET frontier JV へ記録する ★運用の終点
回顧は **TARGET の「結果コメント」欄に馬ごとに記録**していく（次走でその馬を開くと
前走コメントが自動表示＝映像なしで「前走こうだった」が分かる）。

**(A) 一括取込CSV＝手貼りゼロ(推奨)** … TARGET FAQ id=611 の機能を使う。
```bash
venv311\Scripts\python.exe analysis\export_target_comments.py 20260606 20260607
# → reports/review/{weekend}_target_import.csv (cp932, ヘッダ無し)
```
- 形式: `開催名,馬名,レースID(馬番有り),コメント`。レースID(馬番有り)=**新18桁=race_id(16)+馬番(2)**。
  (race_id+馬番 == kekka「レースID(新)」を513/513で検証済)
- 取込: TARGET メインメニュー →**「ファイルからのコメント一括登録」**→ CSVを指定。
  既存コメントとの競合は上書き/スキップ/結合から選ぶ。空コメント・同一コメントは自動無視。
- カンマは「、」、波ダッシュは「～」へサニタイズ済(CSV/cp932安全)。Ver6.21は文字数無制限。

**(B) 馬番順テキスト＝手貼り用(取込を使わない場合)**
```bash
venv311\Scripts\python.exe analysis\review_for_target.py     # → reports/review/{weekend}_target.txt
```
各レースを馬番昇順(=出馬表の行順)に「馬番 印 馬名 [着]: 回顧 → 次走」で並べ、上から順にコピペ。
いずれも無印の着内馬(取りこぼし)を含む。

### 3c. ★全頭・濃い版（TARGET詳細CSVがある場合）= 推奨
TARGET frontier JV から **全頭の詳細成績**をエクスポートすれば、通過順・脚質・上がり3F(順)・
着差から「実際の走り」に基づく濃い回顧を**全レース全頭**ぶん生成できる（平場もWeb不要で濃い）。

エクスポート列(cp932): `日付,場所,Ｒ,レース名,クラス名,枠番,馬番,...,人気,確定着順,単勝オッズ,
芝・ダ,距離,1角,2角,3角,4角,決め手,脚質,上り3F,上り3F順,着差,走破タイム,...`

**置き場所**: エクスポートCSVは **`data/target_review/`** に入れる（`.gitignore`済・README有）。
```bash
venv311\Scripts\python.exe analysis\build_full_review.py        # data/target_review/ の最新CSVを自動検出
# 明示する場合: ...build_full_review.py data\target_review\レース回顧_0418-0531.csv
```
- 出力: `data/uma_review/full/{date}.json`、`data/uma_review/horse_notes_full.json`(全頭DB)、
  `reports/review/{range}_full_kaiko.md`、`reports/review/{range}_full_target_import.csv`(TARGET一括取込)
- 各馬に AI印(◎〇▲△/無印rankN/対象外) を bundle から付与。
- **TARGET取込ID=新18桁=race_id(16)+馬番(2)**。in-bundleはbundleの正規ID、対象外レースは
  `date8+場コード+回日+R+馬番`で構築。kekka「レースID(新)」と全件一致を検証済(527/527)。
- ⚠️ 構築時は **R(レース番号2桁)を必ず入れる**(回日4桁の後)。抜くと16桁になり誤レースに紐づく。

### 4. (任意)スコアカードが欲しいとき
```bash
venv311\Scripts\python.exe analysis\review_marks_join.py     # 印別 勝率/連対率/複勝率
```

---

## 馬名DBの引き方（次走予想で「前走こうだった」を出す）
```python
import json
db = json.load(open(r"E:\PyCaLiAI\data\uma_review\horse_notes.json", encoding="utf-8"))
for rec in db.get("ガイアフォース", []):
    print(rec["date"], rec["course"], rec["result"], "→", rec["jisou"])
```

## 出力ファイル
| パス | 中身 |
|---|---|
| `reports/web_results/{date}.json` | Webで調べた 1-3着（入力、手動/Claude作成） |
| `reports/web_results/overrides_{date}.json` | 重賞等の手書き回顧（任意、入力） |
| `data/uma_review/{date}.json` | その日の全印馬レコード（出力） |
| `data/uma_review/horse_notes.json` | **馬名キー蓄積DB**（毎週追記、出力） |
| `reports/review/{weekend}_kaiko.md` | 回顧レポート（出力） |

## 関連スクリプト
| ファイル | 役割 |
|---|---|
| `analysis/review_marks_web.py` | bundle から印馬を抽出（作業リスト） |
| `analysis/build_uma_review.py` | ★本体: bundle × web_results × overrides → 回顧＋次走メモ＋馬名DB |
| `analysis/review_marks_join.py` | (補助) 印別 勝率/連対率/複勝率スコアカード |
| `analysis/review_day.py` | （kekka版）買い目収支まで含む回顧。kekka.csv 必須 |

## 自動生成ロジックの観点（auto_kaiko）
- 人気薄(単8倍+)で複勝圏 → 「妙味継続・要マーク」
- 人気(単5倍未満)で着外 → 「次走も過信禁物」
- 穴推し(`vs:under`・10倍+)で着外 → 「[longshot弱点]、高オッズ単穴は嫌う」
- 印で1-3着独占レース = 三連複/印BOXの収穫（多い週は荒れ目を取れている）。
