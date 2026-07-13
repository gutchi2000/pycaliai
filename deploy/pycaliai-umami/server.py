# Static file server for the PyCaLiAI UMAMI site (HF Spaces / Docker SDK).
# Equivalent to `python -m http.server` but registers correct MIME types so the
# PWA manifest is served as application/manifest+json (installability requirement).
import os
import mimetypes
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

mimetypes.add_type("application/manifest+json", ".webmanifest")
mimetypes.add_type("image/x-icon", ".ico")

port = int(os.environ.get("PORT", "7860"))
print(f"serving {os.getcwd()} on 0.0.0.0:{port}", flush=True)
ThreadingHTTPServer(("0.0.0.0", port), SimpleHTTPRequestHandler).serve_forever()
