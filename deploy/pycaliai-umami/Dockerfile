# Minimal static file server for the PyCaLiAI UMAMI site on HF Spaces (Docker SDK).
# Serves the contents of site/ on port 7860 (HF default) via server.py (http.server
# + correct .webmanifest MIME for PWA installability).
# Docker SDK -> clean URL https://gutchi15300-pycaliai-umami.hf.space (no .static).
FROM python:3.11-slim

WORKDIR /site
COPY *.html /site/
COPY css /site/css
COPY js /site/js
COPY data /site/data
COPY icons /site/icons
COPY manifest.webmanifest /site/
COPY sw.js /site/
COPY apple-touch-icon.png /site/
COPY favicon.ico /site/
COPY server.py /site/

EXPOSE 7860
CMD ["python", "-u", "server.py"]
