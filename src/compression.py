"""Compression / storage analysis.

Two separate concepts (never conflated):

A. Spectral dimensionality reduction (theoretical):
     B -> K, factor B/K, percent (1-K/B)*100.

B. Actual storage (measured bytes):
     original serialized cube vs compressed package
     (quantized/float Z + basis Wk + mean), saved with np.savez_compressed.

We store Z as float32 by default (scores need not be float64) and Wk/mu as
float32, which is honest: reported sizes are real file sizes on disk.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .utils import get_logger

log = get_logger()


def spectral_reduction(bands: int, k: int) -> dict:
    bands, k = int(bands), int(k)
    return {
        "bands_original": bands,
        "components": k,
        "reduction_str": f"{bands} -> {k}",
        "factor": float(bands / k),
        "percent": float((1 - k / bands) * 100),
    }


def save_package(path: str | Path, Z: np.ndarray, Wk: np.ndarray, mu: np.ndarray,
                  meta: dict) -> Path:
    """Save compressed representation + parameters. Returns file path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path),
                        Z=np.asarray(Z, dtype=np.float32),
                        Wk=np.asarray(Wk, dtype=np.float32),
                        mu=np.asarray(mu, dtype=np.float32),
                        meta=np.array([str(meta)], dtype=object))
    log.info("Saved compressed package: %s (%d bytes)", path, path.stat().st_size)
    return path


def load_package(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    z = np.load(str(path), allow_pickle=True)
    meta = {}
    try:
        meta = eval(str(z["meta"][0])) if "meta" in z else {}
        if not isinstance(meta, dict):
            meta = {"raw": str(meta)}
    except Exception:
        meta = {}
    return (np.asarray(z["Z"], dtype=np.float64),
            np.asarray(z["Wk"], dtype=np.float64),
            np.asarray(z["mu"], dtype=np.float64), meta)


def storage_report(cube: np.ndarray, package_path: str | Path,
                   Z: np.ndarray, Wk: np.ndarray, mu: np.ndarray) -> dict:
    """Compare real on-disk sizes. All byte counts are measured, not theorized."""
    pkg = Path(package_path)
    pkg_bytes = pkg.stat().st_size if pkg.exists() else -1
    # Original as .npy-equivalent raw + header estimate: use actual array bytes.
    orig_bytes = int(np.asarray(cube).nbytes)
    z_bytes = int(np.asarray(Z).nbytes)
    w_bytes = int(np.asarray(Wk).nbytes)
    mu_bytes = int(np.asarray(mu).nbytes)
    total_theoretical = z_bytes + w_bytes + mu_bytes
    return {
        "original_bytes": orig_bytes,
        "z_bytes": z_bytes,
        "basis_bytes": w_bytes,
        "mean_bytes": mu_bytes,
        "theoretical_package_bytes": total_theoretical,
        "actual_package_bytes": int(pkg_bytes),
        "actual_ratio": float(orig_bytes / pkg_bytes) if pkg_bytes > 0 else float("nan"),
        "actual_saving_pct": float((1 - pkg_bytes / orig_bytes) * 100) if pkg_bytes > 0 else float("nan"),
    }
