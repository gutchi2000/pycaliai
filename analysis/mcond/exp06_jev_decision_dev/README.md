# EXP06 — Jev意思決定層・適合性検証

TypeSafe AI の Jev を、新しい着順予測器としてではなく、EXP05/市場/OOD/候補馬券を
統合する「最終判断層」として検証する。検証する仮説は spec.json / このREADME参照。
モデル・特徴・購入ルールへの接続は Gate 1-4 を全て通過するまで一切行わない。

## 絶対条件 (spec.json `absolute_conditions` に固定済み)
- EXP01-05・EXP05-Fを変更しない(このディレクトリ以外を書き換えない)
- 本番の印・買い目・資金配分へ接続しない
- APIキーは `TYPESAFE_API_KEY` 環境変数からのみ読む。コード・ログ・JSON・Gitへ保存しない
- Jevの応答はappend-only保存、同一input_hashへの再課金・再問い合わせを避ける
- 結果を見て質問文・閾値・入力項目を変更しない(spec.jsonは評価開始前に固定・コミット済み)
- 馬名・騎手名・レース名は第1段階で匿名化する
- 着順・払戻・確定オッズなど判断時点より後の情報を入力しない

## 現状 (2026-09-20時点)

**Stage A(API・再現性監査)はブロック中**: 以下の2点が揃わないと着手できない。

1. **`TYPESAFE_API_KEY`が未設定**(この環境の環境変数に無い)。
2. **TypeSafe Jev APIの実際の仕様が不明**: エンドポイントURL、認証ヘッダ形式、
   リクエストボディの形、Choice/Score/Noulレスポンスの実際のJSON表現。これらを
   推測で実装すると「一見動くが実際には壊れている」コードになり、ユーザーの
   実APIコール(課金対象)を無駄にするリスクが高いため、公式ドキュメントか
   サンプルレスポンス(できれば実際に1回叩いた生のレスポンスJSON)の提示を待つ。

`jev_client.py`は上記が判明していなくても固められる部分(キャッシュ・
append-only保存・入力ハッシュ計算・リトライ/タイムアウト骨格・APIキーの扱い)を
先に実装済み。実際のHTTP呼び出し部分(`_call_jev_api_raw`)だけが
`NotImplementedError`で待機している。

## ファイル
| ファイル | 役割 |
|---|---|
| `spec.json` | Q1-Q6の質問文・選択肢、state_fields、比較対象、Gate定義。評価開始前に固定 |
| `jev_client.py` | API呼び出しハーネス(キャッシュ・append-only保存・リトライ骨格)。HTTP部分は未実装 |
| `out/jev_cache/` | input_hash単位のキャッシュ(gitignore対象、再課金防止) |
| `out/jev_responses/{date}.jsonl` | 応答の監査ログ(append-only) |

## Stage
- **Stage A**: API・再現性監査(認証・型・確率和・confidence範囲・モデル名・usage・
  429/5xx処理・同一入力の再現性・言語比較)。英語固定テンプレートを使う。
- **Stage B**: 過去データでの探索的適合性検定(EXP05と同じ時点安全な入力、2023-2025は
  「探索的OOS」と明記)。質問文・統合式はStage B開始前にspec.jsonへ固定済み。
- **Stage C**: 前向きshadow(Stage B通過後のみ)。EXP05-Fが保存したレース状態を読み、
  Jevの判断だけを別ファイルに保存。EXP05-Fの処理が失敗しないようAPI失敗時は
  必ずスキップ可能にする。

## Gate
Gate 0(データ・API健全性)→ Gate 1(Jev確率の情報性)→ Gate 2(既存ゲートへの固有上積み)
→ Gate 3(同一coverageでの改善)→ Gate 4(経済評価、Gate 1-3通過後のみ)。

## 中止条件
spec.json記載の中止条件のいずれかに該当したら終了する。質問文の言い換えや閾値再探索
での救済はしない。
