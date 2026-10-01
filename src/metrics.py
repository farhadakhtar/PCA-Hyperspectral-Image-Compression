"""Quality metrics. Nothing is hard-coded: MAX for PSNR is explicit.

- MSE  = mean((X - Xhat)^2), non-negative.
- MAE  = mean(|X - Xhat|)
- MAXERR = max(|X - Xhat|)
- PSNR = 10*log10(MAX^2 / MSE); inf when MSE == 0.
- Explained variance from eigenvalues.
- SAM (spectral angle mapper, mean over pixels, radians->degrees optional):
  angle between original and reconstructed spectra; 0 = identical shape.
"""
from __future__ import annotations

import numpy as np

from .pca_engine import explained_variance_ratio


def mse(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.mean(d ** 2))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.mean(np.abs(d)))


def max_error(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(d)))


def psnr(a: np.ndarray, b: np.ndarray, max_val: float | None = None) -> float:
    """PSNR in dB. max_val must be supplied or inferred as data peak.

    Returns inf when reconstruction is exact.
    """
    err = mse(a, b)
    if err == 0:
        return float("inf")
    if max_val is None:
        max_val = float(np.max(np.asarray(a, dtype=np.float64)))
    max_val = float(max_val)
    if max_val <= 0:
        raise ValueError(f"MAX must be positive, got {max_val}")
    return float(10.0 * np.log10(max_val ** 2 / err))


def per_band_mse(orig_cube: np.ndarray, recon_cube: np.ndarray) -> np.ndarray:
    o = np.asarray(orig_cube, dtype=np.float64)
    r = np.asarray(recon_cube, dtype=np.float64)
    return np.mean((o - r) ** 2, axis=(0, 1))


def per_band_psnr(orig_cube: np.ndarray, recon_cube: np.ndarray, max_val: float | None = None) -> np.ndarray:
    m = per_band_mse(orig_cube, recon_cube)
    if max_val is None:
        max_val = float(np.max(np.asarray(orig_cube, dtype=np.float64)))
    out = np.full_like(m, np.inf, dtype=np.float64)
    nz = m > 0
    out[nz] = 10.0 * np.log10(max_val ** 2 / m[nz])
    return out


def sam_degrees(orig_cube: np.ndarray, recon_cube: np.ndarray, eps: float = 1e-12) -> float:
    """Mean Spectral Angle Mapper in degrees. Lower = better spectral fidelity."""
    o = np.asarray(orig_cube, dtype=np.float64).reshape(-1, orig_cube.shape[2])
    r = np.asarray(recon_cube, dtype=np.float64).reshape(-1, recon_cube.shape[2])
    no = np.linalg.norm(o, axis=1)
    nr = np.linalg.norm(r, axis=1)
    valid = (no > eps) & (nr > eps)
    if not np.any(valid):
        return 0.0
    cosang = np.sum(o[valid] * r[valid], axis=1) / (no[valid] * nr[valid])
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)).mean())


def variance_report(eigvals: np.ndarray, k: int) -> dict:
    evr = explained_variance_ratio(np.asarray(eigvals, dtype=np.float64))
    cum = np.cumsum(evr)
    k = int(k)
    return {
        "explained_variance_k": float(cum[k - 1]) if k >= 1 else 0.0,
        "individual_k": [float(v) for v in evr[:k]],
        "cumulative": [float(v) for v in cum],
        "full_evr": [float(v) for v in evr],
    }


def full_report(orig_cube: np.ndarray, recon_cube: np.ndarray,
                eigvals: np.ndarray, k: int, max_val: float | None = None) -> dict:
    o = np.asarray(orig_cube, dtype=np.float64)
    r = np.asarray(recon_cube, dtype=np.float64)
    vr = variance_report(eigvals, k)
    return {
        "mse": mse(o, r),
        "mae": mae(o, r),
        "max_error": max_error(o, r),
        "psnr_db": psnr(o, r, max_val),
        "max_val_used": float(max_val) if max_val is not None else float(o.max()),
        "sam_deg": sam_degrees(o, r),
        "explained_variance": vr["explained_variance_k"],
        "k": int(k),
        "bands": int(o.shape[2]),
    }
