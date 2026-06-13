# Minimal static file server for the PyCaLiAI UMAMI site on HF Spaces (Docker SDK).
# Serves the contents of site/ on port 7860 (HF default) via python http.server.
# Docker SDK -> clean URL https://gutchi15300-pycaliai-umami.hf.space (no .static).
FROM python:3.11-slim

WORKDIR /site
COPY index.html /site/
COPY css /site/css
COPY js /site/js
COPY data /site/data

EXPOSE 7860
CMD ["python", "-u", "-m", "http.server", "7860"]
