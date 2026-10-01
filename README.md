# PCA-Based Hyperspectral Image Compression (Project 77)

**Hyperspectral PCA Lab** — load the Indian Pines cube, run spectral PCA,
inspect principal components, reconstruct the cube, and quantify the
compression-vs-quality trade-off. All numbers are computed from the loaded
dataset; nothing is hard-coded.

## Measured dataset (detected dynamically)

- Source: `data/raw/indianpinearray.npy` (+ `IPgt.npy` labels kept separate)
- Shape: **145 × 145 × 200** (H × W × B, corrected Indian Pines)
- dtype: uint16 (range 955–9604), N = 21,025 px, 4,205,000 values (~8.41 MB array)
- Ground truth: 145×145 uint8, classes 0–16 (0 = background)

## Pipeline

```
cube (H×W×B) → X (N×B) → μ, Xc=X−μ → C=XcᵀXc/(N−1) [B×B]
→ Cv=λv (eigh, sorted) → Z=Xc·Wk (N×K) → X̂=Z·Wkᵀ+μ → (H×W×B)
→ EVR / MSE / PSNR (explicit MAX) / SAM + storage audit
```

Two PCA modes share the math: **educational** (covariance + `eigh`,
transparent) and **optimized** (sklearn SVD, better conditioned).

## Measured results (educational PCA, MAX = data peak 9604)

| K | EVR | MSE | PSNR | MAE | SAM | Spectral |
|---|-----|-----|------|-----|-----|----------|
| 1 | 68.494% | 61628.2 | 31.75 dB | 142.67 | 2.614° | 200× |
| 5 | 95.038% | 9705.9 | 39.78 dB | 47.40 | 1.780° | 40× |
| 10 | 96.917% | 6029.6 | 41.85 dB | 39.17 | 1.381° | 20× |
| 20 | 98.654% | 2632.1 | 45.45 dB | 24.14 | 0.900° | 10× |
| 30 | 99.249% | 1469.2 | 47.98 dB | 18.56 | 0.659° | 6.67× |
| 50 | 99.731% | 526.7 | 52.43 dB | 11.31 | 0.382° | 4× |

Variance targets: 80%→K=2, 90%→K=2, 95%→K=5, 99%→K=25.
PC1 alone ≈ 68.5%, PC1–2 ≈ 92%.

Storage honesty: 200→20 is **10× spectrally**, but the measured `.npz`
package (float32 Z + W + μ, zlib) is 1.52 MB vs 8.02 MB raw = **5.28×** —
scores change dtype and the basis must be stored.

## Installation

```bash
pip install -r requirements.txt
```

## Dataset setup

Already in `data/raw/`. To use another scene, put the `.npy`/`.npz`/`.mat`/`.tif`
cube (and optional `*gt*` labels) in `data/raw/` or pass its path to `--input`.

## Run

GUI:
```bash
streamlit run app/main.py
```

CLI:
```bash
python -m src.pca_compression --input data/raw --components 20 --output results/
python -m src.pca_compression --input data/raw --variance 0.95 --mode optimized --output results/
```

Notebook: `notebooks/PCA_Hyperspectral_Analysis.ipynb` (same `src/` functions).

Tests:
```bash
python -m pytest tests -q
```

## Screenshots

Generate via CLI/GUI; outputs land in `results/figures/`:
`original_falsecolor.png`, `reconstructed_falsecolor.png`, `pc_grid.png`,
`explained_variance.png`, `comparison.png`.

## Project structure

```
src/          data_loader, preprocessing, pca_engine, reconstruction,
              metrics, visualization, compression, utils, pca_compression (CLI)
app/main.py   Streamlit dashboard (9 pages)
notebooks/    GUI-free reproduction
tests/        loader/centering/covariance/eig/shapes/full-K/variance/MSE/PSNR
docs/         methodology, mathematics, viva
data/raw/     indianpinearray.npy + IPgt.npy
results/      figures + metrics JSON + .npz packages (generated)
```

## Limitations / future work

- Spectral-only (no spatial coding; PCA+JPEG2000 on PC images is the classic extension).
- Scene-dependent basis must be transmitted.
- Linear, MSE-optimal — rare spectra in discarded PCs are lost.
- Full-K is lossless only before quantization; float32 packaging is lossy at ~1e-3 level.

## References

- Jolliffe, *Principal Component Analysis*; Gonzalez & Woods, *DIP*;
  Sayood, *Intro. to Data Compression* (quantization/transform coding);
  Du & Fowler (2007) — PCA+JPEG2000 hyperspectral baseline; Strang MIT 18.06.
