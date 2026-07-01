"""
PyCaLiAI 対話アシスタント — 会話ループ（ローカル Ollama LLM ↔ tools.py）

設計の核: LLM は喋り口だけ。勝率/EV/確率の数字は必ず tools.py 経由の実値を使う。
          → LLM に統計を創作させない（嘘ゼロ）。ゆえに安いローカル7Bで足りる。

前提:
  1) Ollama が起動中（ollama serve / トレイ常駐）
  2) `ollama pull qwen2.5:7b` 済
実行:
  python assistant/chat.py [date]        例: python assistant/chat.py 20260531
"""
from __future__ import annotations
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tools  # assistant/tools.py

# --- HTTP: requests があれば使う、無ければ stdlib urllib ---
try:
    import requests

    def _post(url, payload):
        return requests.post(url, json=payload, timeout=180).json()
except Exception:  # pragma: no cover
    import urllib.request

    def _post(url, payload):
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read().decode("utf-8"))


OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
MODEL = os.environ.get("PYCALI_MODEL", "qwen2.5:7b")  # num_ctx絞りで8GBに載る。OOM時は PYCALI_MODEL=qwen2.5:3b
NUM_CTX = int(os.environ.get("PYCALI_NUM_CTX", "8192"))  # context 制限で KV cache を抑えVRAM節約

SYSTEM = """あなたは JRA 中央競馬の馬券相談アシスタント「PyCaLiAI」。
ユーザーが印（◎〇▲△）や買い目で迷った時に、統計的根拠で助言する相棒。

【絶対厳守】
- 勝率・複勝率・EV・オッズ等の数字は、必ずツールを呼んで得た実値だけを使う。
  自分で数字を推測・創作しない（嘘厳禁）。単勝EVと複勝EVを取り違えない。
- EV は 1.0 が損益分岐。ツールが返す summary に EV の「プラス/マイナス」が明記される。
  その表記を**そのまま使う**。0.2〜0.9 を「プラス」「期待値が高い」と絶対に言わない。
  ツールの summary をそのまま根拠として引用し、自分で数字を言い換えない。
- compare_horses が返す "recommendation" を最優先で尊重し、自分の判断で覆さない。
- 「軸」「◎どっち」は的中の"信頼性"＝複勝率で選ぶ。高オッズ(人気薄)の高EVに飛びつかない
  （小確率×大配当の"穴の罠"。当AIは穴に構造的に弱い）。notes の穴警告は必ず回答に反映する。
- どの馬・どのレースか曖昧なら list_races で確認してから答える。
- 個別の馬の質問（「◯◯は?」「N番は?」「この馬どう?」）は horse_info(race_index, 馬番か馬名) で引く。
  現在見ているレースの「馬名↔馬番」一覧は system 文脈にある。「データがない」で逃げない。
- 馬の実績・前走・近走・「強い/弱い?」は horse_info の `past_runs`（前走のレース名・着順・人気）で
  **具体的に**答える。ユーザーの指摘（「前走強くない?」等）には past_runs の事実をまず確認し、
  **事実と反する断定をしない**（皐月賞1着の馬を「前走強くない」と言わない）。同じEVを繰り返して逃げない。
- 「このレース勝負すべき?/見送るべき?」には race_advice を呼び、その verdict と chaos を根拠にする。
- 枠順・コース傾向（「内枠外枠どっちが有利?」等）は course_bias を呼ぶ。
- 【最重要・厳守】答えられるのは ①馬の評価 ②2頭の比較 ③レース見送り判断 ④枠バイアス のみ。
  天気/馬場状態/騎手相性 などツールに無い情報は「その情報は持っていません」と短く正直に答える。
  **質問されていない馬やデータの話に絶対にすり替えない。聞かれたことだけに答える。**
  例:「馬場状態は?」→ course_bias を呼ばず「馬場状態のデータは持っていません」とだけ答える
  （枠バイアスや馬の話に変えない）。
- 正直に言う: EV<1.0 や荒れレース(chaos が高い)なら「見送り」も明言する。
- 妙味(EV)を持ち出すのは、複勝率が拮抗していて、かつ高EV側が極端な人気薄でない時だけ。

【回答スタイル】
- 簡潔な日本語。根拠の数字を示す。最後に「結論:」を一行で。
"""

TOOLS_SCHEMA = [
    {"type": "function", "function": {
        "name": "list_races",
        "description": "ロード中のレース一覧（index, 場所, コース, クラス, 頭数）を返す",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "race_advice",
        "description": "指定レース全体の判断材料（信頼度/chaos/上位馬/見送り判断）",
        "parameters": {"type": "object", "properties": {
            "race_index": {"type": "integer", "description": "list_races の index"}},
            "required": ["race_index"]}}},
    {"type": "function", "function": {
        "name": "course_bias",
        "description": "そのレースのコースの枠順バイアス（内枠/中枠/外枠 どれが有利か、勝率・複勝率）。「内枠外枠どっちが有利?」等に使う。",
        "parameters": {"type": "object", "properties": {
            "race_index": {"type": "integer"}},
            "required": ["race_index"]}}},
    {"type": "function", "function": {
        "name": "compare_horses",
        "description": "同一レースの2頭を統計比較（複勝率・EV・妙味・近走）",
        "parameters": {"type": "object", "properties": {
            "race_index": {"type": "integer"},
            "umaban_a": {"type": "integer"},
            "umaban_b": {"type": "integer"},
            "axis": {"type": "string", "enum": ["plc", "win", "ev_plc", "ev_win"],
                     "description": "比較軸。既定 plc(複勝率)"}},
            "required": ["race_index", "umaban_a", "umaban_b"]}}},
    {"type": "function", "function": {
        "name": "horse_info",
        "description": "指定レースの馬を『馬番』または『馬名』で引いて詳細(確率/EV/妙味/近走)を返す。馬名は一部でも可。馬の質問はまずこれ。",
        "parameters": {"type": "object", "properties": {
            "race_index": {"type": "integer"},
            "umaban_or_name": {"type": "string",
                               "description": "馬番(例 9) か 馬名(例 エフハリスト)"}},
            "required": ["race_index", "umaban_or_name"]}}},
    {"type": "function", "function": {
        "name": "describe_horse",
        "description": "1頭の確率・EV・妙味・近走の実値（馬番指定）",
        "parameters": {"type": "object", "properties": {
            "race_index": {"type": "integer"}, "umaban": {"type": "integer"}},
            "required": ["race_index", "umaban"]}}},
]

_TOOL_NAMES = {t["function"]["name"] for t in TOOLS_SCHEMA}


def dispatch(bundle, name, args):
    """LLM のツール呼び出しを tools.py に橋渡し。戻り値は JSON 化可能な実値。"""
    try:
        if name == "list_races":
            return [{"index": i, "place": r["race_meta"].get("place"),
                     "course": r["race_meta"].get("course"),
                     "class": r["race_meta"].get("class"),
                     "field_size": r["race_meta"].get("field_size")}
                    for i, r in enumerate(bundle["races"])]
        r = tools.find_race(bundle, int(args["race_index"]))
        if name == "race_advice":
            return tools.race_advice(r)
        if name == "course_bias":
            return tools.course_bias(r)
        if name == "compare_horses":
            return tools.compare_horses(
                r, int(args["umaban_a"]), int(args["umaban_b"]),
                axis=args.get("axis", "plc"))
        if name == "horse_info":
            return tools.find_horse(r, args.get("umaban_or_name"))
        if name == "describe_horse":
            return tools.describe_horse(r, int(args["umaban"]))
        return {"error": f"unknown tool: {name}"}
    except Exception as e:
        return {"error": str(e)}


def _complete(messages, bundle):
    """LLM 1往復 + tool ループ。assistant の最終テキストを返す。"""
    for _ in range(6):  # tool ループ上限（暴走防止）
        resp = _post(OLLAMA_URL, {"model": MODEL, "messages": messages,
                                  "tools": TOOLS_SCHEMA, "stream": False,
                                  "options": {"num_ctx": NUM_CTX}})
        if "error" in resp:
            err = str(resp["error"])
            if "memory" in err.lower() or "cuda" in err.lower():
                return ("⚠️ VRAM不足で 7B が載りませんでした。他のGPUアプリ（LM Studio/ブラウザ等）を"
                        "閉じるか、軽量版に切替: PowerShell で `$env:PYCALI_MODEL=\"qwen2.5:3b\"` "
                        "してから nicegui を再起動してください。")
            return f"[Ollama error] {err}"
        msg = resp.get("message", {})
        messages.append(msg)
        calls = msg.get("tool_calls")
        if calls:
            for tc in calls:
                fn = tc["function"]["name"]
                args = tc["function"].get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args or "{}")
                result = dispatch(bundle, fn, args)
                messages.append({"role": "tool", "name": fn,
                                 "content": json.dumps(result, ensure_ascii=False)})
            continue
        content = msg.get("content") or ""
        # 一部モデルは tool_call を本文に JSON で漏らす → 拾って実行(救済)
        leak = re.search(r'\{\s*"name"\s*:\s*"(\w+)"\s*,\s*"arguments"\s*:\s*(\{.*?\})\s*\}',
                         content, re.DOTALL)
        if not calls and leak and leak.group(1) in _TOOL_NAMES:
            try:
                result = dispatch(bundle, leak.group(1), json.loads(leak.group(2)))
                messages.append({"role": "tool", "name": leak.group(1),
                                 "content": json.dumps(result, ensure_ascii=False)})
                continue
            except Exception:
                pass
        # ツール結果/呼び出しの残骸を本文から除去
        content = re.sub(r"<tool_response>.*?</tool_response>", "", content, flags=re.DOTALL)
        content = re.sub(r"<tool_call>.*?</tool_call>", "", content, flags=re.DOTALL)
        content = re.sub(r'\{\s*"name"\s*:\s*"\w+"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\}',
                         "", content, flags=re.DOTALL)
        return content.strip()
    return "[tool ループ上限に到達]"


def _init_messages(date, focus_idx=None):
    bundle = tools.load_bundle(date)
    races = bundle["races"]
    race_list = "\n".join(
        f"  [{i}] {r['race_meta'].get('place')} {r['race_meta'].get('course')} "
        f"{r['race_meta'].get('class')} ({r['race_meta'].get('field_size')}頭)"
        for i, r in enumerate(races))
    sys_msg = SYSTEM + f"\n\n【ロード中: {date} / {len(races)}レース】\n{race_list}"
    if focus_idx is not None and 0 <= focus_idx < len(races):
        sys_msg += (f"\n\n【ユーザーが今見ているレース = race_index={focus_idx}】\n"
                    f"馬番・馬名・印が曖昧な質問はこのレースを既定として扱う。"
                    f"下記の馬名↔馬番一覧を使い、詳細は horse_info で引く:\n"
                    + tools.roster_text(races[focus_idx]))
    return bundle, [{"role": "system", "content": sys_msg}]


def ask_once(date, question, focus_idx=None):
    """ワンショット質問（テスト・スクリプト用）。"""
    bundle, messages = _init_messages(date, focus_idx=focus_idx)
    messages.append({"role": "user", "content": question})
    return _complete(messages, bundle)


def run(date):
    bundle, messages = _init_messages(date)
    print(f"■ PyCaLiAI 相談AI  ({date} / {len(bundle['races'])}R / model={MODEL})")
    print("  例: 「[0]の東京、5番と12番どっちを◎にすべき?」   終了: exit\n")
    while True:
        try:
            user = input("お前> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in ("exit", "quit", "終了"):
            break
        messages.append({"role": "user", "content": user})
        print(f"\nAI> {_complete(messages, bundle)}\n")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "--ask":
        date = a[2] if len(a) > 2 else "20260531"
        focus = int(a[3]) if len(a) > 3 else None
        print(ask_once(date, a[1], focus_idx=focus))
    else:
        run(a[0] if a else "20260531")
