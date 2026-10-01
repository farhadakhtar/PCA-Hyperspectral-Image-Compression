"""Reconstruction: Z (N,K) -> Xhat (N,B) -> cube (H,W,B)."""
from __future__ import annotations

import numpy as np

from .preprocessing import unflatten


def reconstruct_matrix(Z: np.ndarray, Wk: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Xhat = Z Wk^T + mu."""
    Z = np.asarray(Z, dtype=np.float64)
    Wk = np.asarray(Wk, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    if Z.shape[1] != Wk.shape[1]:
        raise ValueError(f"Z cols {Z.shape[1]} != Wk cols {Wk.shape[1]}")
    if Wk.shape[0] != mu.shape[0]:
        raise ValueError(f"Wk rows {Wk.shape[0]} != mu len {mu.shape[0]}")
    return Z @ Wk.T + mu


def reconstruct_cube(Z: np.ndarray, Wk: np.ndarray, mu: np.ndarray,
                     height: int, width: int) -> np.ndarray:
    """Full path back to (H, W, B)."""
    Xhat = reconstruct_matrix(Z, Wk, mu)
    return unflatten(Xhat, height, width)


def eckart_young_mse(eigvals: np.ndarray, k: int) -> float:
    """Predicted per-element MSE for rank-k truncation: sum(discarded lambda)/B.

    (Up to the (N-1)/N convention factor; our measured MSE uses 1/N.)
    Useful as a correctness cross-check.
    """
    ev = np.asarray(eigvals, dtype=np.float64)
    k = int(k)
    if not 0 <= k <= ev.shape[0]:
        raise ValueError(f"k out of range: {k}")
    discarded = ev[k:].sum()
    return float(discarded / ev.shape[0])
