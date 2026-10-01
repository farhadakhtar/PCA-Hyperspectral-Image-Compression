# Methodology

1. **Load** — `load_hyperspectral_data()` returns cube `(H, W, B)` + separate labels.
2. **Preprocess** — cast to float (no silent scaling), replace non-finite values
   with band means, flatten C-order to `X (N×B)`, `N = H·W`.
3. **Mean-center** — `μ = mean(X)` per band, `Xc = X − μ` (default; standardization
   is opt-in only because it optimizes relative variance, not MSE).
4. **PCA fit** — Educational: `C = XcᵀXc/(N−1)` (`B×B`, never `N×N`), `eigh`,
   sort descending. Optimized: sklearn SVD (same projection math, better
   conditioning since forming `XcᵀXc` squares the condition number).
5. **Select K** — manual, or smallest K with cumulative EVR ≥ target.
6. **Project** — `Z = Xc·Wk`, `(N×K)`: the compressed representation.
7. **Reconstruct** — `X̂ = Z·Wkᵀ + μ`, reshape to `(H, W, B)`.
8. **Evaluate** — EVR, MSE, MAE, MaxErr, PSNR with explicit MAX (= data peak),
   SAM for spectral shape, per-band curves.
9. **Storage audit** — spectral reduction `B→K` reported separately from measured
   `.npz` package bytes (`Z` float32 + `Wk` float32 + `μ` float32, zlib).

Forward and inverse transforms share one `PCAResult` object (`μ`, `W`, `λ`),
so they cannot mismatch.
