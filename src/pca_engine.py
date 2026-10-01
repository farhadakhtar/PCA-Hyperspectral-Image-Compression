"""PCA engine with two modes sharing one math core.

MODE A — EducationalPCA (manual, transparent):
    Xc -> C = Xc^T Xc / (N-1)  (B x B) -> eigh -> sort desc -> W (B x B), lambda (B,)

MODE B — OptimizedPCA (sklearn, numerically robust SVD-based):
    Same projection math Z = Xc Wk, Xhat = Z Wk^T + mu, but the basis comes
    from sklearn.decomposition.PCA (SVD of Xc, avoids squaring condition number).

Both expose: fit / transform / inverse_transform / explained variance helpers.
State kept together (mu, W, lambda) so forward/inverse can never mismatch.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import get_logger

log = get_logger()


def validate_k(k: int, bands: int) -> int:
    k = int(k)
    if not 1 <= k <= bands:
        raise ValueError(f"K must satisfy 1 <= K <= bands({bands}), got {k}")
    return k


def validate_variance_target(t: float) -> float:
    t = float(t)
    if not 0 < t <= 1:
        raise ValueError(f"Variance target must satisfy 0 < t <= 1, got {t}")
    return t


def k_for_variance(eigvals: np.ndarray, target: float) -> int:
    """Smallest K with cumulative explained variance >= target."""
    target = validate_variance_target(target)
    ev = np.asarray(eigvals, dtype=np.float64)
    total = ev.sum()
    if total <= 0:
        raise ValueError("Eigenvalues sum to 0; cannot compute variance target.")
    cum = np.cumsum(ev) / total
    return int(np.searchsorted(cum, target) + 1)


def explained_variance_ratio(eigvals: np.ndarray) -> np.ndarray:
    ev = np.asarray(eigvals, dtype=np.float64)
    total = ev.sum()
    if total <= 0:
        return np.zeros_like(ev)
    return ev / total


@dataclass
class PCAResult:
    mean_: np.ndarray            # (B,)
    components_: np.ndarray     # (B, B), columns = eigenvectors desc
    eigvals_: np.ndarray        # (B,) desc
    mode: str = "educational"
    n_samples_: int = 0

    @property
    def bands(self) -> int:
        return int(self.eigvals_.shape[0])

    def evr(self) -> np.ndarray:
        return explained_variance_ratio(self.eigvals_)

    def cumulative_evr(self) -> np.ndarray:
        return np.cumsum(self.evr())

    def basis_k(self, k: int) -> np.ndarray:
        return self.components_[:, :validate_k(k, self.bands)]


class EducationalPCA:
    """Manual covariance + eigh implementation (Mode A)."""

    def __init__(self) -> None:
        self.result: PCAResult | None = None

    def fit(self, X: np.ndarray) -> "EducationalPCA":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] < 2:
            raise ValueError(f"Expected (N, B>=2), got {X.shape}")
        n, b = X.shape
        mu = X.mean(axis=0)
        Xc = X - mu
        log.info("Mean centering: X %s -> mu (%d,)", X.shape, b)
        # B x B covariance (never N x N).
        C = (Xc.T @ Xc) / (n - 1)
        # Symmetry guard + eigh (symmetric solver).
        C = 0.5 * (C + C.T)
        vals, vecs = np.linalg.eigh(C)
        vals = np.clip(vals, 0.0, None)  # tiny negatives -> 0
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
        log.info("Eigen decomposition: %d eigenvalues, largest=%.4g", b, float(vals[0]))
        self.result = PCAResult(mean_=mu, components_=vecs, eigvals_=vals,
                                mode="educational", n_samples_=n)
        return self

    def _req(self) -> PCAResult:
        if self.result is None:
            raise RuntimeError("Call fit() before transform().")
        return self.result

    def transform(self, X: np.ndarray, k: int) -> np.ndarray:
        r = self._req()
        Wk = r.basis_k(k)
        X = np.asarray(X, dtype=np.float64)
        return (X - r.mean_) @ Wk  # Z = Xc Wk, (N, K)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        r = self._req()
        Z = np.asarray(Z, dtype=np.float64)
        k = Z.shape[1]
        Wk = r.basis_k(k)
        return Z @ Wk.T + r.mean_  # Xhat = Z Wk^T + mu


class OptimizedPCA:
    """sklearn-backed PCA (Mode B). Mathematically identical projection."""

    def __init__(self, random_state: int = 0) -> None:
        self.random_state = random_state
        self.result: PCAResult | None = None
        self._sk: object = None

    def fit(self, X: np.ndarray) -> "OptimizedPCA":
        from sklearn.decomposition import PCA
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] < 2:
            raise ValueError(f"Expected (N, B>=2), got {X.shape}")
        sk = PCA(n_components=X.shape[1], svd_solver="full", random_state=self.random_state)
        sk.fit(X)
        # sklearn: components_ rows = eigenvectors; eigvals = explained_variance_
        W = np.asarray(sk.components_).T  # -> (B, B) columns
        vals = np.asarray(sk.explained_variance_, dtype=np.float64)
        order = np.argsort(vals)[::-1]
        vals, W = vals[order], W[:, order]
        self._sk = sk
        self.result = PCAResult(mean_=np.asarray(sk.mean_), components_=W,
                                eigvals_=vals, mode="optimized", n_samples_=X.shape[0])
        log.info("Optimized PCA fit: %s, variance[0]=%.4g", X.shape, float(vals[0]))
        return self

    def _req(self) -> PCAResult:
        if self.result is None:
            raise RuntimeError("Call fit() before transform().")
        return self.result

    def transform(self, X: np.ndarray, k: int) -> np.ndarray:
        r = self._req()
        validate_k(k, r.bands)
        return (np.asarray(X, dtype=np.float64) - r.mean_) @ r.basis_k(k)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        r = self._req()
        k = np.asarray(Z).shape[1]
        return np.asarray(Z, dtype=np.float64) @ r.basis_k(k).T + r.mean_


def fit_pca(X: np.ndarray, mode: str = "educational", random_state: int = 0):
    mode = mode.lower()
    if mode in ("educational", "manual"):
        return EducationalPCA().fit(X)
    if mode in ("optimized", "sklearn", "svd"):
        return OptimizedPCA(random_state=random_state).fit(X)
    raise ValueError(f"Unknown PCA mode '{mode}'. Use 'educational' or 'optimized'.")
