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
