# compute_bets.py 仕様書 — 馬券構築（confidence 駆動・カード連動）

> 2026-06-09 改訂。Cowork(LLM) から bets を分離し、ローカル `compute_bets.py` が **当日 T-10** に
> JV-Link 生オッズで各レースの買い目を生成し、`reports/cowork_output/{date}_bets.json` の同 race_id
> エントリへ `race_nature`(形) / `race_reason` / `confidence` / `bets[]` を **in-place merge** する。
>
> 設計の柱:
> - **EV は主決定でなく「配分の重み（従）」**。買いの「形」は NiceGUI の4カードと **同じパーセンタイル＋しきい値** で決める。
> - **馬連は全廃**（実績 ROI 56% 最弱）。◎独走系は **馬単＋単勝**、◎弱は **ワイド流し**。
> - **穴の罠（モデルのテール過大評価）をオッズ上限で遮断**。
> - 人間は NiceGUI のカードを見ながら最終 見送り/狙い を判断する前提（出力に confidence を同梱）。

---

## 入力
1. `bundle.json` 各 race: race_meta{class,field_size}, race_confidence{top1_dominance,top2_concentration,
   field_chaos_score,ai_market_agreement}, horses[]{umaban,mark,p_win,p_plc,p_sho,tansho_odds,
   fuku_odds_low/high}, buy_judgment{waku_tag,value_horses[]}, （ワイド/馬単推定用に）umaren_matrix。
2. **JV-Link 生オッズ（T-10, jvlink_odds.py）**: 単勝・複勝（0B31）、ワイド（0B33）、馬単（0B34）。
   ※ ライブのワイド/馬単 実値があれば優先、無ければ umaren_matrix から推定（下記）。

## 出力（cowork_output/{date}_bets.json の race_id エントリへ in-place merge）
```json
{ "race_id":"...","race_label":"...","race_nature":"◎軸",
  "race_reason":"◎{馬名}。◎軸（独走0.65/集中0.46/混戦0.36/市場+0.72）で 5点。",
  "confidence":{"top1_pct":0.65,"top2_pct":0.46,"chaos_pct":0.36,"market":0.72},
  "bets":[ {"馬券種":"単勝","買い目":"9","購入額":3000,"枠タグ":"妙味枠","理由":"おいしい（単勝 15.7倍）"}, ...] }
```
- 1 entry = 1 組合せ + 1 金額。買い目: 単複`"9"` / ワイド`"5-9"` / 馬単`"9→11"`。
- 購入額: 100円単位、1点 ¥500〜7,000、race 合計 ¥10,000（キャップで埋まらない薄いレースは ¥7,000 止まりを許容＝-EVに突っ込まない規律）。

> ### ★ 書き込み契約（データ消失防止）
> NiceGUI `_load_all_cowork_output` は race_id 単位で**エントリ丸ごと上書き**。compute_bets は Cowork が
> 保存した `{date}_bets.json` を read-modify-write で **同一ファイル in-place 追記**（別ファイル禁止。
> advisor を消さないため）。Cowork 未出力の race は新規エントリ作成可。

---

## アルゴリズム

### 0. 見送り（hard gate）
`field_chaos_score(生) >= 0.92` / `field_size <= 7` / ◎の tansho_odds 無 / `p_win(◎) < 0.05`
＋ fail-safe（オッズ欠損・鮮度NG・overround [1.0,1.5] 外）→ `bets:[]`、race_nature="見送り"。
（カードの「カオス」はパーセンタイル>0.75 で **見送り推奨表示**だが、自動 hard 見送りは生値0.92。0.75–0.92 は薄く買う＝人間がカード見て判断）

### 1. カード値（nicegui と同一）
`data/chaos_quantiles.json` のクオンタイルで生値→パーセンタイル（`pct()`）:
- `top1` = pct(top1_dominance), `top2` = pct(top2_concentration), `chaos` = pct(field_chaos_score)
- `market` = ai_market_agreement（生値 -1..1）
カード閾値: TOP1 0.75/0.50、TOP2 0.75/0.50/0.40、CHAOS 0.75/0.50、MARKET 妙味<0.30。

### 2. 形（shape）決定
| 形 | 条件 | 買い方 |
|---|---|---|
| **本命勝負** | top1>=0.75 & top2>=0.75 & chaos<=0.50（◎独走＋本線濃厚＋固い） | **馬単8点 formation（◎〇→◎〇▲△△）＋ 単勝◎厚 ＋ 複勝◎** |
| **◎軸** | top1>=0.50（◎やや優位） | **単勝◎（最重視）**。◎単勝≤15倍なら **馬単◎→相手流し**。＋ ワイド◎-相手 数点 ＋ 複勝◎ |
| **広め流し** | top1<0.25（拮抗）or top2<0.40（分散） | **ワイド ◎軸＋〇軸 流し**（馬単/馬連なし）＋ 複勝◎ |
| **カオス薄** | chaos>0.75（生<0.92） | ワイド◎軸 薄く ＋ 複勝◎ |
| **標準** | 上記以外 | 単勝◎ ＋ ワイド◎-相手 ＋ 複勝◎ |
| **穴 overlay** | market<0.30（市場乖離） かつ value_horses 有 | 当該馬の 単勝/ワイド/複勝 を上乗せ（おいしい馬） |

### 3. EV 式（生オッズ。ワイド/馬単はライブ実値優先、無ければ推定）
| 馬券 | EV |
|---|---|
| 単勝 | p_win × odds |
| 複勝 | p_sho × (low+high)/2 |
| ワイド(i,j) | odds_wide × (p_sho_i·p_sho_j)。推定時 odds_wide≈umaren/3.0 |
| 馬単(i→j) | odds_umatan × p_win_i·p_win_j/(1-p_win_i)。推定時 odds_umatan≈umaren×(p_i+p_j)/p_i |

### 4. オッズ上限（穴の罠遮断・必須）
`馬単<=200 / ワイド<=50 / 馬連=不使用`。上限超の組は候補から除外（モデルのテール過大評価が EV 経由で大穴を生むのを防ぐ）。単勝/複勝は◎/value horse なら上限なし（穴推奨を許容）。

### 5. 採否（ソフトEVフロア）
- フロア: 通常 EV>=0.80、**本命勝負の馬単は formation 全採用（フロア免除、上限8点）**。
- **◎必須**: ◎絡みが1つも無ければ最高EVの◎絡みを追加。全候補低EVなら最高EV1点に集中。
- 点数上限: 本命勝負8 / その他6。EV×boost 降順で採用。

### 6. 配分（§5/§7）
EV帯→1点額目安（×boost）で重み付け → ¥10,000 へ正規化（100円, [¥500,¥7,000] キャップ厳守、反復補正）。
boost: 単勝◎ 1.4-1.6（最重視）、馬単formation 1.3、おいしい馬 1.1-1.3 等。

| EV 帯 | 1点額目安 |
|---|---|
| EV>=1.50 | ¥5,500 |
| 1.20-1.50 | ¥3,750 |
| 1.00-1.20 | ¥2,250 |
| 0.85-1.00 | ¥1,500 |
| <0.85 | ¥900（formation等の押さえ） |

### 7. race_reason / confidence
`race_reason` = テンプレ（◎名＋形＋カード値＋点数、数値はカード値のみ可）。
`confidence` = {top1_pct, top2_pct, chaos_pct, market} を同梱 → NiceGUI でカードと突合表示。

---

## 馬券種の方針（実績ベース 2026-05）
単勝120.8%⭐（最重視）/ ワイド97.8%（主力）/ 複勝71.4%（押さえ）/ **馬連56.2%→全廃**。
馬単は ◎独走時の order-specific 表現として採用（◎が実本命≤15倍の時）。

## fail-safe / 運用
- オッズ取得失敗・鮮度NG・overround 異常 → 見送り。
- 投票は人間（IPAT）。compute_bets は買い目提示まで（金の自動執行なし）。
- 出力後 NiceGUI が cowork_output を ui.timer で随時表示（confidence カードと並ぶ）。

## 当日オーケストレータ（t10_runner.py / t10.ps1、2026-06-12）
data/weekly/{date}.csv の発走時刻から全レースをスケジュールし、各レース T-10（`-LeadMin`）に自動で:
1. `py -3.12-32 jvlink_odds.py --race {rid16}` → reports/live_odds/{rid16}.json
2. `compute_bets.py --race {rid16} --live-odds-dir --apply`（当該レースのみ in-place merge）
3. `validate_cowork_bets.py --apply`（見送りガード）
4. 買い目コンソール表示 + ビープ → 人間が IPAT 投票
`--once {rid16}` 単発テスト / `--dry` 書込なし / 発走済みレースはスキップ / HF push なし。

### ルーチン（タスクスケジューラ登録済み、2026-06-12）
**タスク `PyCaLiAI_T10` が毎週土日 9:00 に `t10.ps1 -Routine` を自動起動**（ログオン中のみ・
ウィンドウ表示・二重起動は IgnoreNew + .t10_lock で防止・実行上限 10h・遅延起動可）。
- `-Routine` = 対象日を「今日」に固定 + bundle 未生成なら Phase A 完了を 15:00 まで 2 分間隔で
  待機（→ 人間は土日朝に Phase A を回すだけ。bundle が出来た時点でラインが動き出す）
- 全出力を `logs/t10_{date}.log` に記録
- 非開催日（bundle が出ない日）は 15:00 に自然終了
- ⚠ 祝日（月曜）開催はトリガー外 → 手動で `.\t10.ps1`
- 無効化: `Disable-ScheduledTask -TaskName PyCaLiAI_T10` / 削除: `Unregister-ScheduledTask`

## JV-Link パーサ（全4券種確定、2026-06-12 raw 突合 + 確定配当照合）
| spec | 券種 | レイアウト |
|---|---|---|
| 0B31 O1 | 単勝 | pos45 + stride8 = odds(4)+人気(2)+予備(2)、/10 |
| 0B31 O1 | 複勝 | pos269 + **stride12** = lo(4)+hi(4)+人気等(4)、/10 ※stride10 は誤り（5頭ごとズレ） |
| 0B33 O3 | ワイド | pos40 + stride17 = 組番(4)+lo(5)+hi(5)+人気(3)、/10、153組+票数計(11) |
| 0B34 O4 | 馬単 | pos40 + stride13 = 組番(4)+odds(6)+人気(3)、/10、306組+票数計(11) |
検証: 馬単 40.0倍=kekka 4,000円 完全一致 / ワイド3組とも実払戻が lo-hi 内 / 複勝 bundle 全頭一致。

## クラス別調整（class_prior、任意）
race_meta.class_prior があれば: G2=◎単勝抑制 / G1=◎単勝集中 / L・OP・G3=広めに / 未勝利・1勝=◎厚。
（現 bundle に class_prior 未埋込のレースは class 名から方針のみ適用）
