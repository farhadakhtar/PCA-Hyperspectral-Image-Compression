"""Shared utilities: logging, formatting, reproducibility."""
from __future__ import annotations

import datetime
import logging
import sys


def get_logger(name: str = "hsic") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def format_bytes(n: int) -> str:
    n = int(n)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n:.2f} GB"


def timestamp_now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def software_versions() -> dict:
    out = {}
    for mod in ("numpy", "scipy", "sklearn", "matplotlib", "pandas", "streamlit", "plotly"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "installed")
        except Exception:
            out[mod] = "not-installed"
    try:
        import platform
        out["python"] = platform.python_version()
    except Exception:
        pass
    return out
