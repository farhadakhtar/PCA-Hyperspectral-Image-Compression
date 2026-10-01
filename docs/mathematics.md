# Mathematics

Data matrix `X ∈ R^(N×B)`, `N = H·W` pixels, `B` bands.

- Mean: `μ = (1/N) Σᵢ xᵢ`, per band.
- Centered: `Xc = X − μ`.
- Covariance: `C = (1/(N−1)) XcᵀXc`, `B×B` symmetric.
- Eigendecomposition: `C v = λ v`. `λ` = variance along `v`;
  `v` = principal direction. Solved with `eigh` (symmetric solver);
  tiny negative eigenvalues clipped to 0.
- Sort `λ₁ ≥ λ₂ ≥ … ≥ λB`; columns of `W` ordered accordingly.
  PC1 ↔ largest `λ`.
- Explained variance ratio: `λᵢ / Σλ`. Cumulative sum → variance retained.
- Projection: `Z = Xc·Wk`, `(N×K)`.
- Reconstruction: `X̂ = Z·Wkᵀ + μ`.
- Eckart–Young: rank-K PCA is the optimal linear rank-K approximation;
  per-element error ≈ `Σ_{i>K} λᵢ / B`. Measured MSE must match this.
- Metrics: `MSE = mean((x−x̂)²)`; `PSNR = 10·log₁₀(MAX²/MSE)` with explicit MAX;
  `SAM = arccos(<x,x̂>/(‖x‖‖x̂‖))` per pixel, averaged.
