# -*- coding: utf-8 -*-
"""legacy/ — 旧 8 モデルアンサンブル (Streamlit 系統) の専用コード (2026-11-04, docs/LEGACY_RETIREMENT_PLAN_20261004.md 段階 3)。

本番 (v6 / a95 / 静的サイト) はここを import しない。実行は root から `python -m legacy.<name>`。
中のモジュール同士は旧来どおり `import ev_filter` のように素の名前で参照するので、package の import 時に
このディレクトリ自身を sys.path に足す (root の cwd は -m 実行で既に乗っている)。
"""
import sys as _sys
from pathlib import Path as _Path

_here = str(_Path(__file__).resolve().parent)
if _here not in _sys.path:
    _sys.path.append(_here)      # root (本番) を優先し、legacy は後ろに足す
