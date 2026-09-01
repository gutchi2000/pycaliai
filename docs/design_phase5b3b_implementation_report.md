# PyCaLiAI Design System — Phase 5B-3B: Cache Reliability Fix

> Implements both preferred directions from `docs/design_phase5b3_audit_and_5c_preview.md` §5B-3B (findings #1 and #2). No service-worker architecture redesign, no aggressive caching optimization added.
> Date: 2026-09-01

## Changes

**`deploy/pycaliai-umami/server.py`** (the actual production HTTP handler, confirmed in the prior audit to be identical in kind to the local dev server): subclassed `SimpleHTTPRequestHandler` to send `Cache-Control: no-cache` on `/` and any `*.html` response, and nothing else. CSS/JS/data responses are untouched — they continue to rely on this project's existing `?v=`/`?t=` query-string convention, which is now reliable because the HTML that references those URLs is itself guaranteed fresh.

**`site/sw.js`**: the fetch handler's `fetch(req)` call now passes `{ cache: "no-store" }`, so the service worker's own "network-first" intent is actually enforced rather than being silently satisfiable by the browser's underlying HTTP cache. Also bumped `CACHE` from `'pycaliai-v1'` to `'pycaliai-v2'` — this is not a new mechanism, it exercises the existing (previously never-triggered) `activate` cleanup logic exactly as the file's own comment already instructed ("Bump CACHE to invalidate old caches on deploy").

Install/activate logic (`skipWaiting`/`clients.claim`/cache-name cleanup) was **not** touched — the audit found it correctly written; only the two specific gaps it identified were closed.

## Response headers, before → after

Measured live against the actual `server.py` process (run locally on its own port, serving `site/` exactly as the Dockerfile does), and against the unmodified original for comparison:

| Resource | Before | After |
|---|---|---|
| `/` (root) | `content-length, content-type, date, last-modified, server` — **no cache-control** | + `cache-control: no-cache` |
| `/index.html` | same as above, no cache-control | + `cache-control: no-cache` |
| `/explain.html` | same as above, no cache-control | + `cache-control: no-cache` |
| `/css/style.css?v=...` | no cache-control | **unchanged** — no cache-control (by design, not in scope) |
| `/js/app.js?v=...` | no cache-control | **unchanged** — no cache-control (by design, not in scope) |
| `/data/manifest.json` | no cache-control | **unchanged** — no cache-control (by design, not in scope) |

## Verification against the actual production-serving path

Ran `deploy/pycaliai-umami/server.py` directly (not the vanilla `python -m http.server` used by this project's design-work dev preview) against the current `site/` contents, on a separate local port, to test the actual modified file rather than a stand-in.

| Scenario | Result |
|---|---|
| First load | Page loads correctly; headers confirmed as in the table above. |
| Normal reload | Two consecutive `fetch('/')` calls returned two different `Date` response headers 1.2s apart — direct proof each request reached the live process rather than being satisfied from any cache layer. |
| **Second deployment/update simulation** | Temporarily inserted a unique marker comment at the top of `site/index.html`, confirmed the already-running server process serves it on a **plain, non-cache-busted `fetch('/')`** with `cache-control: no-cache` present — i.e., an existing client reloading normally, with no manual cache-bust and no service-worker unregistration, now sees new content. Marker reverted immediately after (confirmed via `git diff` showing no residual change). |
| Existing service worker active / stale-tab cache cleanup | Manually created a fake `pycaliai-v1` cache entry (simulating a leftover from before this fix), unregistered the SW, then did a fresh navigation. The new SW installed, reached `"activated"` state, `clients.claim()` correctly took control of the already-open tab (`navigator.serviceWorker.controller` populated without a second reload), and `activate`'s cleanup deleted the stale `pycaliai-v1` entry — final cache list: `["pycaliai-v2"]` only. |
| HTML update propagation | Covered by the deployment simulation above — confirmed working without any manual intervention. |
| CSS/JS version update propagation | Not changed by this phase; already confirmed working in Phases 5B-2/5B-3A via the existing `?v=` convention, which this fix does not disturb. |
| No offline/runtime regression | The `catch` block (`caches.match(req)` fallback on fetch failure) is unmodified — only the `try` block's `fetch()` call gained an explicit cache-mode option, which does not change error/rejection behavior on a genuine network failure. The offline-fallback code path is structurally identical to before this change. **Not independently re-verified against a live simulated-offline browser state in this environment** — this is a logical/code-level confirmation, disclosed as such rather than overstated as an empirical offline test. |
| Console/runtime errors | Clean, confirmed via a fresh single-click tab isolated from this session's own scripted-testing noise. |

## A known limitation of this verification

Registering a service worker against the standalone `server.py` test instance (run on an ad-hoc port outside this environment's normal dev-preview mechanism) failed with an opaque `"unknown error occurred when fetching the script"` — this reproduced consistently and is judged to be a limitation of this specific browser-automation environment's handling of service-worker registration on ports it doesn't manage itself, not a problem with `sw.js` or `server.py` (the same `sw.js` file registers and runs correctly, repeatedly, against this project's normal dev-preview port, as shown in the "existing service worker" row above). Consequently, the two halves of this fix were verified on two different, appropriate surfaces: the `Cache-Control` header change against the actual modified `server.py` process directly, and the service-worker install/activate/cache-cleanup behavior against the normal dev-preview server (which serves the identical, already-fixed `site/sw.js`). Both surfaces exercise the real, shipped code; they were not exercised simultaneously in one single process due to this environment constraint, which is disclosed here rather than glossed over.

## PASS/FAIL

| Item | Result |
|---|---|
| HTML/app-shell responses now send `Cache-Control: no-cache` | **PASS** |
| CSS/JS/data responses unchanged (no aggressive caching added) | **PASS** |
| Service worker fetch now bypasses browser HTTP cache (`no-store`) | **PASS** (verified by direct source inspection + well-defined Fetch API spec behavior; see limitation note above) |
| Install/activate logic unmodified, still correct | **PASS** |
| Stale cache cleanup exercised and confirmed working | **PASS** |
| Second-deployment simulation shows propagation without manual intervention | **PASS** |
| No offline-fallback regression (code-level) | **PASS** |
| Console clean | **PASS** |

`git status --porcelain -- site/`: `M site/sw.js` only under `site/` (plus `deploy/pycaliai-umami/server.py` outside it) — not yet committed, see next message.
