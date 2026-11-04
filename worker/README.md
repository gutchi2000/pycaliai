# worker/ — pycaliai.com Cloudflare Worker（相談AI）

`wrangler.jsonc` の `main`。`site/` の静的配信（ASSETS）に加えて相談AIの API を持つ。

| ファイル | 役割 |
|---|---|
| `index.js` | `POST /api/soudan`（Claude 中継・レート制限・CORS）、`GET /api/soudan/health`（鍵の有無と本日の消費、値は返さない）。それ以外は `env.ASSETS` へ |
| `tools.js` | `assistant/tools.py` の移植。数字の供給元は **公開済み `site/data/{date}.json` だけ**（`build_site.py scrub_public` 済みなのでオッズ・払戻・EV・調教タイム・上り3F は存在しない） |
| `prompt.js` | system prompt（答える範囲 5 種を固定、持たないデータは断る、購入を促さない、注入耐性）と strict なツール定義 |
| `tools.test.mjs` | `node --test worker/tools.test.mjs`（site/data の最新日で検査） |

## 守り（2026-11-03 ユーザー承認: Sonnet、全体 1 日 150 問まで、超えたら Haiku）
- Origin 許可リスト（pycaliai.com / HF Space / localhost）以外は 403
- IP ごと 1 日 `SOUDAN_PER_IP_DAY`(30) 問。IP は日替わり salt で SHA-256 ハッシュ化し生 IP は保存しない（KV、2 日で失効）
- 全体 1 日 `SOUDAN_DAILY_SONNET`(150) 問までは `claude-sonnet-5-5`、`SOUDAN_DAILY_HARD`(600) 問までは `claude-haiku-5-5`、超えたら 429
- 月の推定支出が `SOUDAN_MONTH_USD_CAP`(90) USD を超えたら 429（クレジット枯渇前に止める）
- 上限値は `wrangler.jsonc` の `vars`。変更後は `npx wrangler deploy`

## 運用
```bash
npm install                                   # 初回のみ（@anthropic-ai/sdk, wrangler）
npx wrangler secret put ANTHROPIC_API_KEY     # 鍵（Max の月次 API クレジットを Console で claim して発行）。git には入れない
npx wrangler deploy                           # sync-hf-umami.ps1 の中でも走る
curl https://pycaliai.com/api/soudan/health   # ok:true なら稼働。key_len 0 なら secret が空
node --test worker/tools.test.mjs
```
`wrangler dev` でローカル実行する場合は `.dev.vars` に `ANTHROPIC_API_KEY=...`（.gitignore 済み）。

## 費用の目安（1 問 約 5K in / 0.3K out）
Sonnet 5.5 ≈ $0.013、Haiku 5.5 ≈ $0.0007。150 問/日を Sonnet で使い切っても月 $60 弱。

## 回顧記事（2026-11-04、`kaiko.js`）
「負けた理由まで書く AI」。結果のある開催日の全レースについて、facts（◎〇▲の評価と着順・1〜3 着の AI 順位と人気・
サイト買い目の的中・混戦度・コース枠/脚質傾向・◎の根拠ラベル）を公開 JSON から機械的に組み、Claude が
「予想→結果→ズレた理由→教訓:」を 5〜8 行で書く。日の総括（◎の勝利/複勝圏数、目立った的中と外し、X 下書き 140 字）も 1 本。
- `GET /api/kaiko?date=YYYYMMDD` 一覧（KV の記事 + 未生成 race_id）、`POST /api/kaiko/gen {date, race_id|"day"}` 1 件生成
- 記事は KV `kaiko:v1:{date}:{race_id}` に永続（1 レース 1 回。再生成したいときは KV のキーを消す）
- 生成経路: Phase C `[1f/3]`（`scripts/kaiko_warm.py`）／cron 月曜 03:00 JST（`triggers.crons`）／`site/kaiko.html` 閲覧時の lazy 生成
- 上限: `KAIKO_DAILY_GEN`(150 件/日) と相談AI と共通の `SOUDAN_MONTH_USD_CAP`。費用は 1 レース ≈ $0.01、1 開催日 ≈ $0.3

## 見どころ・RSS・Discord（2026-11-04）
- `POST /api/kaiko/gen {date, race_id:"preview"}`: 発走前の「今日の見どころ」（結果を使わない `previewFacts()`）。結果待ちの日は一覧 API の `missing` が `["preview"]` になり、ページ閲覧でも生成される。
- `GET /kaiko.xml`（`/api/kaiko/feed`）: 直近 12 開催日の総括と見どころの RSS 2.0（KV にあるものだけ、10 分キャッシュ）。`run_worker_first` に `/kaiko.xml` を入れてあるので assets の SPA フォールバックに食われない。
- cron: `0 18 * * SUN`（月曜 03:00 JST、回顧）/ `0 20 * * FRI,SAT`（土日 05:00 JST、見どころ）。どちらも `warmLatest()`。
- Discord: `scripts/kaiko_warm.py --discord` がこの実行で**新しく生成した**総括・見どころだけを `notify_config.json` の webhook へ投稿（T-10 と同じ設定ファイル。未設定なら skip）。
- 相談AI のツール `ai_track_record`: `site/data/report_card.json`（`build_report_card.py`、AI の通信簿）をそのレースの条件で引く。
