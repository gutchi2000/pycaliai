// pycaliai-mail — contact@pycaliai.com に届いたメールを Discord（グッチの私用サーバー）に知らせる
// Cloudflare Email Worker（2026-11-05）。
//
// 流れ: Email Routing の contact@ ルール → この Worker →
//   1. 先に転送先 Gmail へ forward（通知が失敗してもメールは失われない）
//   2. Discord に「差出人・件名・受信時刻」を投稿（本文は載せない。読むのは Gmail で）
// 転送に失敗したときも Discord には知らせ、そのうえで例外を投げて送信側に再送を求める。
//
// サイト用 Worker（../worker/、pycaliai）とは別 Worker。デプロイしても site/ や相談AI に触れない。
//   デプロイ: npx wrangler deploy -c worker_mail/wrangler.jsonc
//   秘密:     npx wrangler secret put FORWARD_TO      -c worker_mail/wrangler.jsonc   （転送先 Gmail）
//             npx wrangler secret put DISCORD_WEBHOOK -c worker_mail/wrangler.jsonc   （notify_config.json の discord_webhook）
// テスト: node --test worker_mail/index.test.mjs

const TITLE_MAX = 250; // Discord embed title は 256 字まで
const FIELD_MAX = 1000; // embed field value は 1024 字まで

// RFC 2047 の encoded-word（=?UTF-8?B?...?= / =?ISO-2022-JP?B?...?= / Q 方式）を読める文字に戻す。
// 読めない charset はそのまま返す（通知が落ちるよりは生の表記が出る方がよい）。
export function decodeMimeWords(input) {
  if (!input) return "";
  const unfolded = String(input).replace(/\r?\n[ \t]+/g, " ");
  const joined = unfolded.replace(/(\?=)\s+(=\?)/g, "$1$2"); // 隣り合う encoded-word 間の空白は捨てる
  return joined.replace(/=\?([^?]+)\?([BbQq])\?([^?]*)\?=/g, (whole, charset, enc, text) => {
    try {
      let bytes;
      if (enc.toUpperCase() === "B") {
        const bin = atob(text);
        bytes = Uint8Array.from(bin, (c) => c.charCodeAt(0));
      } else {
        const t = text.replace(/_/g, " ");
        const out = [];
        for (let i = 0; i < t.length; i++) {
          const hex = t.slice(i + 1, i + 3);
          if (t[i] === "=" && /^[0-9A-Fa-f]{2}$/.test(hex)) {
            out.push(parseInt(hex, 16));
            i += 2;
          } else {
            out.push(t.charCodeAt(i));
          }
        }
        bytes = Uint8Array.from(out);
      }
      const cs = charset.split("*")[0].toLowerCase(); // RFC 2231 の言語指定 (utf-8*ja) を落とす
      return new TextDecoder(cs, { fatal: true }).decode(bytes);
    } catch {
      return whole;
    }
  });
}

function clip(s, n) {
  const t = String(s || "").trim();
  return t.length > n ? t.slice(0, n - 1) + "…" : t;
}

function jstNow(now = Date.now()) {
  return new Date(now + 9 * 3600 * 1000).toISOString().slice(0, 16).replace("T", " ") + " JST";
}

export function buildNotice({ from, to, subject, forwarded, now }) {
  const lines = [`📧 **${to || "contact@pycaliai.com"}** にメールが届きました`];
  if (!forwarded) lines.push("⚠️ Gmail への転送に失敗しました（送信元に再送を求めています）");
  return {
    username: "PyCaLiAI お問い合わせ",
    content: lines.join("\n"),
    embeds: [
      {
        title: clip(subject, TITLE_MAX) || "(件名なし)",
        fields: [
          { name: "差出人", value: clip(from, FIELD_MAX) || "(不明)" },
          { name: "受信", value: jstNow(now) },
        ],
        footer: { text: forwarded ? "本文は転送先の Gmail で確認してください" : "Cloudflare の Email Routing と Worker のログを確認してください" },
      },
    ],
    // 件名や差出人に @everyone 等が入っていてもメンションさせない
    allowed_mentions: { parse: [] },
  };
}

async function postDiscord(url, payload) {
  if (!url) {
    console.warn("DISCORD_WEBHOOK が未設定のため通知を省略");
    return;
  }
  const res = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`Discord webhook ${res.status}`);
}

export default {
  async email(message, env) {
    let forwardError = null;
    try {
      if (!env.FORWARD_TO) throw new Error("FORWARD_TO が未設定");
      await message.forward(env.FORWARD_TO);
    } catch (e) {
      forwardError = e;
      console.error("forward failed:", e && e.message);
    }

    const notice = buildNotice({
      from: decodeMimeWords(message.headers.get("from") || message.from),
      to: message.to,
      subject: decodeMimeWords(message.headers.get("subject") || ""),
      forwarded: !forwardError,
    });
    try {
      await postDiscord(env.DISCORD_WEBHOOK, notice);
    } catch (e) {
      console.error("discord notify failed:", e && e.message); // 通知の失敗でメール処理は止めない
    }

    if (forwardError) throw forwardError;
  },
};
