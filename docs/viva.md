# Viva Q&A — PCA for Hyperspectral Image Compression

1. **What is PCA?** Orthogonal transform converting correlated bands into
   uncorrelated principal components ordered by variance.
2. **Why PCA here?** Neighboring bands are highly correlated (~0.58 mean |corr|
   on Indian Pines); PCA compacts energy into few PCs → fewer numbers to store.
3. **Why hyperspectral data?** Hundreds of narrow bands per pixel (material
   spectra); rich but heavy: 145×145×200 = 4.2M values here.
4. **What is spectral redundancy?** Adjacent bands measure overlapping
   wavelengths → correlated; visible as block structure in band-correlation heatmap.
5. **What is a principal component?** A new axis `v` (eigenvector); its scores
   across pixels form a component image.
6. **What does PC1 represent?** Direction of maximum variance (here ~68.5% alone);
   broad brightness/albedo-like map.
7. **What is an eigenvector?** Non-zero `v` with `Cv = λv`: direction preserved
   (up to scale) by the covariance.
8. **What is an eigenvalue?** `λ`: variance captured along its eigenvector;
   determines ordering and EVR.
9. **Why center the data?** Covariance/variance are defined about the mean;
   without centering the first component chases the mean, not variation.
10. **What is covariance?** Mean pairwise co-deviation of bands; diagonal =
    band variances.
11. **Why use covariance matrix?** Its eigendecomposition gives the
    variance-maximizing orthonormal basis (KLT — optimal linear code under MSE).
12. **What does explained variance mean?** Fraction `λᵢ/Σλ` (or cumulative for K):
    share of total variance retained.
13. **Why does PCA reduce dimensionality?** Keep K<B axes holding most variance;
    each pixel → K scores instead of B bands.
14. **What is the compressed representation?** `Z = Xc·Wk`, `(N×K)` scores.
15. **How is reconstruction performed?** `X̂ = Z·Wkᵀ + μ`, reshape to `(H,W,B)`.
16. **What is MSE?** Mean squared pixel/band error; 0 = exact; ours K=20: 2632.
17. **What is PSNR?** `10·log₁₀(MAX²/MSE)` in dB; higher = smaller error. MAX here
    = data peak 9604 (not 255). Ours K=20: 45.45 dB.
18. **Why does information loss occur?** Discarded PCs' variance
    (`Σ_{i>K}λᵢ`) cannot be recovered from K scores.
19. **What happens if K is increased?** EVR↑, MSE↓, PSNR↑, SAM↓; storage grows.
    Measured: K=10→41.85 dB, K=30→47.98 dB, K=50→52.43 dB.
20. **What happens if K is decreased?** Opposite; K=1 keeps 68.5%, PSNR 31.75 dB.
21. **Why is PCA unsupervised?** No labels used; only data covariance.
22. **Is PCA lossless?** Only at full K without quantization; any truncation or
    float32/quantization makes it lossy.
23. **Is PCA nonlinear?** No — strictly linear (projection onto flat subspace).
24. **Limitations?** Spectral-only (spatial redundancy untouched); basis is
    scene-dependent (must store `W`); MSE-optimal ≠ task-optimal (rare spectra
    can live in discarded PCs); linear only.
25. **Why doesn't B→K equal file-size ratio?** Scores dtype differs (uint16→
    float32), plus `W`, `μ`, headers. Measured: 200→20 is 10× spectrally but
    5.28× on disk (8.02 MB → 1.52 MB).
26. **Why retain PCA parameters?** Without `Wk` and `μ`, scores cannot be
    inverted (`X̂ = Z·Wkᵀ + μ` needs both).
27. **Bands vs components?** Bands = sensor wavelengths (physical); components =
    data-derived mixtures ordered by variance (statistical).
28. **Why might max variance ≠ max task information?** Small anomalies/rare
    materials contribute little variance but may decide classification; they can
    sit in low-λ PCs that compression discards.
