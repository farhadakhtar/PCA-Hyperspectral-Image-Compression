I'll check the current source for the standard hyperspectral datasets, then write the walkthrough.# PCA Hyperspectral Compression: Build Walkthrough

## Verdict

- **Can the README plan be built?** Yes. It is a solid 2–3 week project.
- **Is it a correct compression project as written?** Not yet. It has 4 gaps:
  1. **No bit accounting.** Keeping 20 of 200 PCs is not 10× compression. AVIRIS data is 16-bit integers. The PC scores are float32. So 200×2 bytes ÷ 20×4 bytes gives **5×**, before counting the stored basis. Real compression needs quantization plus entropy coding, measured in bits per pixel per band (**bpppb**).
  2. **No spectral metric.** PSNR ignores spectral shape. Add **SAM (Spectral Angle Mapper)**, the standard hyperspectral fidelity metric.
  3. **Placeholder numbers.** "10 PCs → 85%" is made up. On real scenes, PC1 alone often explains most of the variance. Replace the table with measured results.
  4. **One stated limitation is wrong.** "PCA is expensive for large datasets" doesn't hold here. The covariance matrix is only B×B (200×200). Total cost is O(N·B²), which takes seconds.
- **Structural fix:** merge `reconstruction.py` into `pca.py`. The forward and inverse transforms share state (μ, W), and splitting them invites mismatch bugs.

---

## 1. The Framework: Everything Hangs Off One Fixed Point

**Fixed point: PCA = Karhunen–Loève Transform (KLT) = the optimal linear transform code under MSE.**

Every module follows from this chain:

```
Correlated bands ──► KLT decorrelates ──► energy compacts into few PCs
      │                                            │
      ▼                                            ▼
 Eckart–Young: error = Σ discarded λ      Reverse water-filling: give bits
 (tells you distortion for any k)         to PCs in proportion to log(λ)
      │                                            │
      └──────────────► Rate–Distortion curve ◄─────┘
                    (bpppb vs PSNR/SAM) = your final result
```

- **Energy compaction.** The eigenvalues λ₁ ≥ λ₂ ≥ … are the variances of the PCs. Truncating to k components throws away exactly λ_{k+1…B}.
- **Eckart–Young theorem.** Rank-k PCA is the best linear rank-k approximation. Reconstruction MSE per element equals Σ_{i>k} λᵢ / B, with an (N−1)/N factor.
  - This gives you a **free correctness test**: measured MSE must match predicted MSE.
- **Transform coding.** After decorrelation, you quantize each PC independently.
  - Optimal bit allocation is bᵢ ≈ ½·log₂(λᵢ/D). This is reverse water-filling.
  - PCs with λᵢ < D get 0 bits. **Choosing k is therefore a special case of bit allocation**, not a separate decision.
- **Research tie-in.** PCA followed by JPEG2000 on the PC images is the classic hyperspectral baseline (Du & Fowler, 2007).
  - It combines spectral KLT with spatial wavelets.
  - Plain PCA exploits only **spectral** redundancy. Each PC image still has **spatial** redundancy, which is the natural extension.

---

## 2. File Structure (Minimal, Every File Earns Its Place)

```
pca-hsi-compression/
├── README.md
├── pyproject.toml              # pinned deps; replaces requirements.txt
├── LICENSE
├── .gitignore                  # data/raw/, results/*.npz, .ipynb_checkpoints
│
├── data/
│   ├── README.md               # download links + SHA256 checksums + .mat keys
│   └── raw/                    # gitignored
│
├── src/hsic/
│   ├── __init__.py
│   ├── io.py                   # load .mat → (H,W,B) float64 + metadata (bit depth, dtype)
│   ├── pca.py                  # fit / transform / inverse_transform  (ONE class)
│   ├── quantize.py             # uniform scalar quantizer + bit allocation
│   ├── codec.py                # encode → bytes (header+μ+W+Q(Z)+zlib), decode ← bytes
│   ├── metrics.py              # MSE, PSNR, per-band SNR, SAM, bpppb, CR
│   └── plots.py                # scree, band compare, error maps, R-D curve
│
├── scripts/
│   ├── compress.py             # CLI: --dataset --k --bits → writes .hsic file
│   └── sweep.py                # grid over k × bits → results/tables/rd.csv
│
├── tests/
│   ├── test_pca.py             # vs sklearn, orthonormality, Eckart–Young
│   ├── test_metrics.py         # known-answer tests
│   └── test_codec.py           # encode→decode roundtrip, byte count == reported
│
├── notebooks/
│   └── 01_exploration.ipynb    # visuals ONLY — imports from src/, no logic
│
└── results/
    ├── figures/
    └── tables/
```

**Design rules**

- **Logic lives in `src/`. The notebook only calls it.** Otherwise the results can't be reproduced.
- **`codec.py` produces real bytes.** The compression ratio is `os.path.getsize(original) / len(bytes)`, never a theoretical count.
- **`io.py` records the original dtype and bit depth.** PSNR's MAX value and the compression-ratio denominator both depend on it.

---

## 3. Build Phases (In Order, Each With an Exit Test)

| # | Phase | Build | Exit test (must pass before moving on) |
|---|---|---|---|
| 0 | Setup | venv, `pyproject.toml`, git, pytest | `pytest` runs empty |
| 1 | Load + inspect | `io.py`; print shape, dtype, min/max per band; plot mean spectrum; plot band-correlation heatmap | Heatmap shows the high-correlation blocks, which is your motivation figure |
| 2 | PCA from scratch | `fit`: μ, then Xc, then C = XcᵀXc/(N−1), then `np.linalg.eigh`, then sort descending | Matches `sklearn.PCA` **up to sign**; WᵀW = I; Σλ = trace(C) |
| 3 | Truncate + reconstruct | `transform(k)`, `inverse_transform` | k=B gives reconstruction error ≈ 1e-10; measured MSE equals Eckart–Young prediction |
| 4 | Metrics | MSE, PSNR, SNR per band, SAM | Hand-computed toy cases: identical input gives SAM=0, PSNR=∞ |
| 5 | Honest codec | quantize Z per component; header + μ + W(float32) + zlib(Q(Z)) | decode(encode(x)) error matches the quantization bound; byte count is real |
| 6 | Rate–distortion sweep | k ∈ {1..50}, bits ∈ {4..16} | R-D curve: bpppb vs PSNR and vs SAM |
| 7 | Baselines | (a) keep every n-th band + interpolate, (b) float32 PCA with no quantization, (c) optional: PCA + JPEG2000 on PC images | PCA beats band-subsampling at equal bpppb |
| 8 | Report | README with **measured** tables and figures | A stranger can reproduce everything with 2 commands |

**Core API for `pca.py`** (signatures only; you write the body):

```python
class SpectralPCA:
    def fit(self, X: np.ndarray) -> "SpectralPCA": ...        # X: (N, B) float64
    def transform(self, X, k: int) -> np.ndarray: ...          # → (N, k)
    def inverse_transform(self, Z) -> np.ndarray: ...          # → (N, B)
    # state: mean_ (B,), components_ (B, B) cols = eigvecs, eigvals_ (B,) desc
```

---

## 4. Edge Cases to Plan For

- **uint16 overflow.** Subtracting the mean from uint16 wraps silently. **Cast to float64 before centering.**
- **Reshape order.** Use `(H,W,B) → (H*W, B)` and back with the **same** `order`. A mismatch scrambles the image without raising any error.
- **Eigenvector sign ambiguity.** `v` and `−v` are both valid. Tests against sklearn must compare `abs()` or align signs first.
- **`eig` vs `eigh`.** The covariance matrix is symmetric, so use `eigh`. It is faster, returns real values, and gives an orthonormal basis. Clip tiny negative eigenvalues (−1e-12) to 0.
- **Precision.** Accumulate the covariance in float64. float32 loses precision over 20k+ pixels.
- **PSNR MAX.** Define it explicitly as either the data's peak value or the sensor range (2¹⁶−1). Results differ by several dB, so state which one you used.
- **Water-absorption bands.** Indian Pines ships in two versions. The corrected release drops bands [104–108], [150–163] and 220, reducing 224 bands to 200. Pick one and report it, since the noisy bands inflate error.
- **Basis overhead.** Storing W costs B·k·4 bytes. It is small for 145×145 scenes but still counts toward the ratio.
- **k > B or k = 0.** Guard with assertions.
- **SAM on zero-norm pixels.** Division by zero gives NaN. Add epsilon or mask those pixels.
- **Covariance vs correlation PCA.** Do **not** standardize bands for compression. Standardizing optimizes relative variance instead of MSE.
- **Fit/apply generalization (optional experiment).** Fit W on one scene and apply it to another. This tests whether the basis transfers.

---

## 5. Datasets

- **Indian Pines** (start here: small, fast iteration).
  - Captured by the AVIRIS sensor: 145×145 pixels, 224 bands covering 0.4–2.5 µm.
  - Available as MATLAB files: raw (6.0 MB), corrected (5.7 MB), and ground truth.
- **Pavia University** (a second sensor, to validate generality).
  - ROSIS sensor, 103 bands, 610×340 pixels over an urban area.
- **Salinas** (a larger AVIRIS scene).
  - 512 lines by 217 samples at 3.7 m resolution.
- **Loading:** use `scipy.io.loadmat`.
  - Variable names differ between files, so inspect `.keys()` and document them in `data/README.md`.
  - Drop Rasterio: it is only needed for GeoTIFF or ENVI files.

---

## 6. Skills Required, in Priority Order

**Must know (the project fails without these)**

- **Linear algebra**
  - Eigen-decomposition of symmetric matrices, orthonormal bases, projection.
  - SVD, and why SVD of Xc is numerically better than eig of XcᵀXc (squaring the matrix squares the condition number).
  - Resource: Strang, MIT 18.06, lectures on eigenvalues, SVD, and projections.
- **Statistics**
  - Mean, variance, covariance matrix, and why centering matters.
  - Explained-variance ratio.
- **NumPy fluency**
  - Broadcasting, dtypes, `reshape` order, memory views vs copies, `@` vs `*`.
- **Testing**
  - pytest, and property-style tests using invariants (orthonormality, Eckart–Young) rather than eyeballing plots.

**Should know (separates a toy from a real compression result)**

- **Compression theory**
  - Rate–distortion, scalar quantization, entropy, transform coding, bit allocation.
  - Resource: Sayood, *Introduction to Data Compression*, chapters on quantization and transform coding.
- **Image quality metrics**
  - MSE, PSNR and its pitfalls, SNR, SAM.
- **Hyperspectral domain basics**
  - Radiance vs reflectance, sensor bit depth, noisy and water-absorption bands, what a "spectral signature" is.

**Nice to know (extensions)**

- JPEG2000 / wavelets, for spatial compression of the PC images (the `glymur` library).
- Incremental and randomized PCA for scenes that don't fit in memory.
- Nonlinear alternatives: kernel PCA and autoencoders. These are the frontier comparison, but they need a GPU and a train/test split.

---

## 7. Pros / Cons of PCA for This Problem

**Pros**

- Optimal linear transform under MSE (KLT). It is provably the best rank-k linear code.
- Has a closed form with no training loop, and it is deterministic.
- The error is predictable in advance from the eigenvalues.
- It is cheap: the eigenproblem is only B×B.

**Cons**

- Exploits spectral redundancy only; spatial redundancy is left untouched.
- The basis is scene-dependent, so W must be stored or transmitted.
- MSE-optimal does not mean task-optimal: small anomalies and rare materials can sit in the discarded PCs.
- Linear only, so nonlinear spectral mixing is not captured.
- It is lossy even at full k once you quantize, so this is not a lossless codec.

---

## Sources

- [Hyperspectral Remote Sensing Scenes (EHU)](https://ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)
- [IEEE DataPort: Indian Pines, Pavia University, Botswana, Salinas](https://ieee-dataport.org/documents/hyperspectral-remote-sensing-datasets-indian-pines-pavia-university-botswana-and-salinas)
- [IEEE DataPort: Hyperspectral Remote Sensing Scenes](https://ieee-dataport.org/documents/hyperspectral-remote-sensing-scenes)

Next step, if you want it: I can rewrite the README with these corrections, or put this walkthrough into a doc you can keep and check off.
