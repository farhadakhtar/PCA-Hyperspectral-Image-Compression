"""Acceptance-aligned tests: loader, centering, covariance, eig, shapes, metrics."""
import numpy as np

from src import data_loader as dl
from src import metrics as M
from src import preprocessing as P
from src import reconstruction as R
from src.pca_engine import EducationalPCA, OptimizedPCA


def _toy(n=200, b=10, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, 1)) * np.linspace(5, 1, b)
    return base + rng.normal(scale=0.1, size=(n, b))


def test_loader_returns_3d():
    ds = dl.load_hyperspectral_data("data/raw")
    assert ds.cube.ndim == 3 and ds.cube.shape[2] > 1
    assert ds.bands == ds.cube.shape[2]


def test_centering_zero_mean():
    X = _toy()
    Xc, mu = P.center_bands(X)
    assert np.allclose(Xc.mean(axis=0), 0, atol=1e-10)


def test_covariance_symmetric():
    X = _toy()
    Xc, _ = P.center_bands(X)
    C = (Xc.T @ Xc) / (X.shape[0] - 1)
    assert np.allclose(C, C.T, atol=1e-12)


def test_eigen_sorted_and_orthonormal():
    X = _toy()
    e = EducationalPCA().fit(X)
    ev = e.result.eigvals_
    assert np.all(ev[:-1] >= ev[1:] - 1e-12)
    assert np.allclose(e.result.components_.T @ e.result.components_, np.eye(ev.size), atol=1e-8)


def test_projection_reconstruction_shapes():
    X = _toy()
    e = EducationalPCA().fit(X)
    k = 4
    Z = e.transform(X, k)
    assert Z.shape == (X.shape[0], k)
    Xh = e.inverse_transform(Z)
    assert Xh.shape == X.shape
    Xh2 = R.reconstruct_matrix(Z, e.result.basis_k(k), e.result.mean_)
    assert np.allclose(Xh, Xh2)


def test_full_k_negligible_error():
    X = _toy()
    e = EducationalPCA().fit(X)
    Z = e.transform(X, X.shape[1])
    Xh = e.inverse_transform(Z)
    assert M.mse(X, Xh) < 1e-18


def test_variance_monotone():
    X = _toy()
    e = EducationalPCA().fit(X)
    cum = np.cumsum(e.result.evr())
    assert np.all(np.diff(cum) >= -1e-12) and abs(cum[-1] - 1.0) < 1e-10


def test_mse_nonnegative_and_psnr_inf():
    X = _toy()
    assert M.mse(X, X) == 0.0
    assert M.psnr(X, X) == float("inf")
    assert M.mse(X, X + 1) > 0


def test_optimized_matches_educational_subspace():
    X = _toy()
    a = EducationalPCA().fit(X)
    b = OptimizedPCA().fit(X)
    # Same total variance & same rank-k error (sign-insensitive comparison).
    assert abs(a.result.eigvals_.sum() - b.result.eigvals_.sum()) / a.result.eigvals_.sum() < 1e-6
    for k in (2, 5):
        ea = M.mse(X, a.inverse_transform(a.transform(X, k)))
        eb = M.mse(X, b.inverse_transform(b.transform(X, k)))
        assert abs(ea - eb) / max(ea, 1e-12) < 1e-6
