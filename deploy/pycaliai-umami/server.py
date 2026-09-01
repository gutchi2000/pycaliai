# Static file server for the PyCaLiAI UMAMI site (HF Spaces / Docker SDK).
# Equivalent to `python -m http.server` but registers correct MIME types so the
# PWA manifest is served as application/manifest+json (installability requirement),
# and forces the HTML shell to always revalidate (Phase 5B-3B): SimpleHTTPRequestHandler
# sends no Cache-Control by default, so browsers apply heuristic freshness caching to
# index.html/explain.html themselves — which can silently serve a stale shell (with
# stale ?v=... asset URLs baked in) even to an online client. CSS/JS/data are untouched
# here; they're already cache-safe via this project's existing ?v=/?t= query-string
# convention once the shell that references them is guaranteed fresh.
import os
import mimetypes
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

mimetypes.add_type("application/manifest+json", ".webmanifest")
mimetypes.add_type("image/x-icon", ".ico")


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        p = self.path.split("?", 1)[0]
        if p == "/" or p.endswith(".html"):
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()


port = int(os.environ.get("PORT", "7860"))
print(f"serving {os.getcwd()} on 0.0.0.0:{port}", flush=True)
ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
