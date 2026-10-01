"""Preprocessing: float conversion, invalid-value handling, flatten, mean-centering.

Pipeline (default, per project spec):
    cube (H, W, B) -> X (N, B) float -> per-band mean mu (B,) -> Xc = X - mu

Standardization (zero-mean + unit-variance) is offered as an *option* only,
because it optimizes relative variance instead of MSE and changes the
compression objective. The default is mean-centering.
"""
from __future__ import annotations

import numpy as np

from .utils import get_logger

log = get_logger()


def to_working_float(cube: np.ndarray, dtype: np.dtype = np.float64) -> np.ndarray:
    """Convert cube to float, replacing NaN/Inf safely. No silent scaling."""
    arr = np.asarray(cube, dtype=dtype)
    bad = ~np.isfinite(arr)
    if np.any(bad):
        # Replace NaN/Inf with per-band finite mean; fall back to 0.
        log.warning("Found %d non-finite values; replacing with band means.", int(bad.sum()))
        for b in range(arr.shape[2]):
            band = arr[:, :, b]
            m = band[np.isfinite(band)]
            fill = float(m.mean()) if m.size else 0.0
            band[~np.isfinite(band)] = fill
    return arr


def flatten_cube(cube: np.ndarray) -> np.ndarray:
    """(H, W, B) -> (N, B) with N = H*W, C-order. Inverse is unflatten()."""
    cube = np.asarray(cube)
    if cube.ndim != 3:
        raise ValueError(f"Expected 3D cube, got {cube.shape}")
    h, w, b = cube.shape
    return cube.reshape(h * w, b, order="C")


def unflatten(X: np.ndarray, height: int, width: int) -> np.ndarray:
    """(N, B) -> (H, W, B)."""
    X = np.asarray(X)
    n, b = X.shape
    if n != height * width:
        raise ValueError(f"Cannot reshape ({n},{b}) to ({height},{width},B)")
    return X.reshape(height, width, b, order="C")


def band_means(X: np.ndarray) -> np.ndarray:
    """Per-band mean mu (B,) over pixels."""
    return np.asarray(X, dtype=np.float64).mean(axis=0)


def center_bands(X: np.ndarray, mu: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (Xc, mu) with Xc = X - mu. If mu is None, compute from X."""
    X = np.asarray(X, dtype=np.float64)
    if mu is None:
        mu = X.mean(axis=0)
    else:
        mu = np.asarray(mu, dtype=np.float64)
    if mu.shape != (X.shape[1],):
        raise ValueError(f"mu shape {mu.shape} mismatches bands {X.shape[1]}")
    return X - mu, mu


def standardize_bands(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optional z-scoring. Returns (Xs, mu, sigma). Clearly NOT the default."""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0  # constant band guard
    return (X - mu) / sigma, mu, sigma
