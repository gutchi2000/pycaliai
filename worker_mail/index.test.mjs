// node --test worker_mail/index.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import worker, { decodeMimeWords, buildNotice } from "./index.js";

const b64 = (bytes) => Buffer.from(bytes).toString("base64");

test("decodeMimeWords: UTF-8 B / Q / ISO-2022-JP / 素の ASCII", () => {
  assert.equal(decodeMimeWords(`=?UTF-8?B?${b64(Buffer.from("テスト"))}?=`), "テスト");
  assert.equal(decodeMimeWords("=?utf-8?Q?caf=C3=A9_au_lait?="), "café au lait");
  // ISO-2022-JP の「テスト」= ESC $ B 2546 2539 2548 ESC ( B
  const jis = [0x1b, 0x24, 0x42, 0x25, 0x46, 0x25, 0x39, 0x25, 0x48, 0x1b, 0x28, 0x42];
  assert.equal(decodeMimeWords(`=?ISO-2022-JP?B?${b64(jis)}?=`), "テスト");
  assert.equal(decodeMimeWords("Re: domain name pycal.com"), "Re: domain name pycal.com");
  assert.equal(decodeMimeWords(""), "");
});

test("decodeMimeWords: 隣り合う encoded-word は空白なしでつなぐ、名前+アドレスも読める", () => {
  const a = b64(Buffer.from("予測データ"));
  const b = b64(Buffer.from("の利用について"));
  assert.equal(decodeMimeWords(`=?UTF-8?B?${a}?=\r\n =?UTF-8?B?${b}?=`), "予測データの利用について");
  assert.equal(
    decodeMimeWords(`=?UTF-8?B?${b64(Buffer.from("山田"))}?= <yamada@example.com>`),
    "山田 <yamada@example.com>",
  );
});

test("decodeMimeWords: 読めない charset は生の表記のまま返す", () => {
  const raw = "=?x-unknown-cs?B?AAAA?=";
  assert.equal(decodeMimeWords(raw), raw);
});

test("buildNotice: メンション無効・長さ制限・空件名", () => {
  const n = buildNotice({ from: "@everyone <a@b.c>", to: "contact@pycaliai.com", subject: "x".repeat(400), forwarded: true, now: 0 });
  assert.deepEqual(n.allowed_mentions, { parse: [] });
  assert.ok(n.embeds[0].title.length <= 256);
  assert.equal(n.embeds[0].fields[1].value, "1970-01-01 09:00 JST");
  assert.equal(buildNotice({ from: "", subject: "", forwarded: true }).embeds[0].title, "(件名なし)");
  assert.match(buildNotice({ from: "a", subject: "s", forwarded: false }).content, /転送に失敗/);
});

function fakeMessage({ subject = "テスト", from = "sender@example.com", forwardImpl } = {}) {
  const calls = [];
  return {
    calls,
    from,
    to: "contact@pycaliai.com",
    headers: new Headers({ subject: `=?UTF-8?B?${b64(Buffer.from(subject))}?=`, from }),
    async forward(addr) {
      calls.push(["forward", addr]);
      if (forwardImpl) return forwardImpl(addr);
    },
  };
}

function withFetch(impl, fn) {
  const orig = globalThis.fetch;
  globalThis.fetch = impl;
  return Promise.resolve(fn()).finally(() => {
    globalThis.fetch = orig;
  });
}

const ENV = { FORWARD_TO: "inbox@example.com", DISCORD_WEBHOOK: "https://discord.example/webhook" };

test("email: 先に転送してから Discord に投稿する", async () => {
  const msg = fakeMessage();
  const posted = [];
  await withFetch(
    async (url, init) => {
      msg.calls.push(["fetch", url]);
      posted.push(JSON.parse(init.body));
      return new Response(null, { status: 204 });
    },
    () => worker.email(msg, ENV),
  );
  assert.deepEqual(msg.calls, [
    ["forward", "inbox@example.com"],
    ["fetch", "https://discord.example/webhook"],
  ]);
  assert.equal(posted[0].embeds[0].title, "テスト");
  assert.equal(posted[0].embeds[0].fields[0].value, "sender@example.com");
});

test("email: Discord が落ちても転送済みなら例外を出さない", async () => {
  const msg = fakeMessage();
  await withFetch(
    async () => {
      throw new Error("network down");
    },
    () => worker.email(msg, ENV),
  );
  assert.deepEqual(msg.calls[0], ["forward", "inbox@example.com"]);
});

test("email: 転送に失敗したら Discord に知らせてから例外を投げる", async () => {
  const msg = fakeMessage({
    forwardImpl: () => {
      throw new Error("destination not verified");
    },
  });
  const posted = [];
  await withFetch(
    async (_url, init) => {
      posted.push(JSON.parse(init.body));
      return new Response(null, { status: 204 });
    },
    () => assert.rejects(worker.email(msg, ENV), /destination not verified/),
  );
  assert.match(posted[0].content, /転送に失敗/);
});

test("email: FORWARD_TO 未設定は転送失敗として扱う", async () => {
  const msg = fakeMessage();
  const posted = [];
  await withFetch(
    async (_url, init) => {
      posted.push(JSON.parse(init.body));
      return new Response(null, { status: 204 });
    },
    () => assert.rejects(worker.email(msg, { DISCORD_WEBHOOK: ENV.DISCORD_WEBHOOK }), /FORWARD_TO/),
  );
  assert.equal(msg.calls.filter((c) => c[0] === "forward").length, 0);
  assert.match(posted[0].content, /転送に失敗/);
});
