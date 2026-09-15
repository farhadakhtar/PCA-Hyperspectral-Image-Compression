# PCA-Based Hyperspectral Image Compression

A Digital Image Processing project that uses **Principal Component Analysis (PCA)** to reduce the spectral dimensionality and compress hyperspectral images while preserving the maximum possible information.

---

## 📌 Overview

Hyperspectral images contain information across hundreds of narrow spectral bands. This makes them extremely useful for applications such as remote sensing, agriculture, mineral exploration, environmental monitoring, and object classification.

However, the large number of spectral bands also creates a major problem:

* High storage requirements
* Large transmission bandwidth
* High computational cost
* Significant redundancy between neighboring spectral bands

This project explores **PCA (Principal Component Analysis)** as a dimensionality-reduction technique for hyperspectral image compression.

The main idea is to transform the original correlated spectral bands into a smaller number of uncorrelated **Principal Components (PCs)** that preserve most of the image's information.

---

## 🎯 Objectives

* Understand the spectral redundancy present in hyperspectral images.
* Apply PCA to hyperspectral image data.
* Reduce the number of spectral dimensions.
* Compress the hyperspectral image using selected principal components.
* Reconstruct the image from the reduced representation.
* Compare the original and reconstructed images.
* Evaluate compression quality using quantitative metrics.

---

## 🛰️ What is a Hyperspectral Image?

A normal RGB image contains three spectral channels:

```text
Red
Green
Blue
```

A hyperspectral image may contain hundreds of spectral bands.

It can therefore be represented as a 3D data cube:

```text
             Spectral Bands
          ┌───────────────────┐
         /│                  /│
        / │                 / │
       └───────────────────┘  │
       │  │                │  │
       │  └────────────────│──┘
       │ /                 │ /
       │/                  │/
       └───────────────────┘

       Width × Height × Bands
```

For example:

```text
Image size = 512 × 512 × 200

Total values = 512 × 512 × 200
             = 52,428,800
```

Storing all these values requires a large amount of memory.

---

## 💡 Why PCA?

Neighboring spectral bands in hyperspectral images are often highly correlated.

For example:

```text
Band 1 ─────┐
Band 2 ─────┤
Band 3 ─────┤── Highly correlated information
Band 4 ─────┤
Band 5 ─────┘
```

PCA finds new axes called **principal components** that capture the maximum variance in the data.

Instead of storing all original bands:

```text
B1 B2 B3 B4 B5 ... B200
```

we can represent the image using fewer components:

```text
PC1 PC2 PC3 ... PC20
```

If the first 20 components preserve most of the variance, the remaining components can be discarded.

---

# 🧠 PCA Concept

PCA transforms correlated variables into a new set of uncorrelated variables called principal components.

The components are ordered according to the amount of variance they explain.

```text
PC1 → Maximum variance
PC2 → Second highest variance
PC3 → Third highest variance
...
PCn → Lowest variance
```

Therefore, the first few principal components generally contain most of the important information.

---

## 🔬 Methodology

The compression process used in this project is:

```text
Hyperspectral Image
        │
        ▼
Convert Image Cube to Data Matrix
        │
        ▼
Mean Centering
        │
        ▼
Calculate Covariance Matrix
        │
        ▼
Eigenvalues & Eigenvectors
        │
        ▼
Sort Principal Components
        │
        ▼
Select First K Components
        │
        ▼
Compressed Representation
        │
        ▼
Reconstruction
        │
        ▼
Compare with Original Image
```

---

# 📐 Mathematical Foundation

Suppose the hyperspectral image contains `N` pixels and `B` spectral bands.

The data matrix can be represented as:

```text
X ∈ R^(N × B)
```

### 1. Mean Centering

Calculate the mean of each spectral band:

```text
μ = mean(X)
```

Then center the data:

```text
Xc = X - μ
```

### 2. Covariance Matrix

The covariance matrix is calculated as:

```text
C = (1 / (N - 1)) Xcᵀ Xc
```

### 3. Eigen Decomposition

Find eigenvalues and eigenvectors:

```text
C v = λ v
```

where:

* `λ` = eigenvalue
* `v` = eigenvector

### 4. Sort Components

Eigenvectors are sorted according to decreasing eigenvalues.

The eigenvector corresponding to the largest eigenvalue represents the first principal component.

### 5. Projection

The centered data is projected onto the selected eigenvectors:

```text
Z = Xc Wk
```

where:

* `Z` = compressed representation
* `Wk` = selected `k` eigenvectors

### 6. Reconstruction

The approximate original data can be reconstructed using:

```text
X̂ = Z Wkᵀ + μ
```

The reconstructed image is an approximation of the original image.

---

# 🗜️ Compression

Suppose the original hyperspectral image contains:

```text
200 spectral bands
```

and PCA retains:

```text
20 principal components
```

Then the spectral dimensionality is reduced from:

```text
200 → 20
```

This represents a **10× reduction in the number of spectral dimensions**, before accounting for any additional storage needed for PCA parameters.

The actual file-level compression ratio depends on the storage format and whether the PCA transformation parameters are also stored.

---

# 📊 Evaluation

The quality of the reconstructed image can be evaluated using:

### Mean Squared Error (MSE)

```text
MSE = (1/N) Σ(x - x̂)²
```

Lower MSE indicates a smaller reconstruction error.

### Peak Signal-to-Noise Ratio (PSNR)

```text
PSNR = 10 log10(MAX² / MSE)
```

Higher PSNR generally indicates better reconstruction quality.

### Explained Variance

Explained variance tells us how much of the original variance is retained by the selected principal components.

For example:

```text
10 PCs  → 85% variance
20 PCs  → 94% variance
30 PCs  → 98% variance
```

This helps determine an appropriate number of components.

---

# 📁 Project Structure

```text
PCA-Hyperspectral-Image-Compression/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── README.md
│
├── src/
│   ├── pca_compression.py
│   ├── reconstruction.py
│   └── metrics.py
│
├── notebooks/
│   └── PCA_Hyperspectral_Compression.ipynb
│
├── results/
│   ├── explained_variance.png
│   ├── original_band.png
│   ├── compressed_pc.png
│   └── reconstructed_band.png
│
└── LICENSE
```

---

# 🛠️ Technologies Used

* Python
* NumPy
* SciPy
* Scikit-learn
* Matplotlib
* Rasterio / Spectral Python
* Jupyter Notebook

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/PCA-Hyperspectral-Image-Compression.git
```

Move into the project directory:

```bash
cd PCA-Hyperspectral-Image-Compression
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

# ▶️ Usage

Run the PCA compression program:

```bash
python src/pca_compression.py
```

The program will:

1. Load the hyperspectral image.
2. Reshape the data into a suitable matrix.
3. Center the spectral data.
4. Perform PCA.
5. Select the required number of components.
6. Generate the compressed representation.
7. Reconstruct the image.
8. Calculate reconstruction metrics.
9. Save visualization results.

---

# 📈 Expected Results

The project aims to demonstrate that a relatively small number of principal components can preserve a large portion of the useful information in a hyperspectral image.

Example:

| Components | Explained Variance | Reconstruction Quality |
| ---------: | -----------------: | ---------------------- |
|          5 |                Low | Poor                   |
|         10 |           Moderate | Acceptable             |
|         20 |               High | Good                   |
|         30 |          Very High | Excellent              |

The exact results depend on the hyperspectral dataset.

---

# 🔍 Applications

PCA-based hyperspectral dimensionality reduction can be useful in:

* Remote sensing
* Satellite imaging
* Agriculture
* Mineral exploration
* Environmental monitoring
* Land-cover classification
* Medical imaging
* Object detection
* Hyperspectral data transmission

---

# ⚖️ Advantages

* Reduces spectral dimensionality.
* Removes highly correlated information.
* Reduces storage requirements.
* Can reduce computational complexity.
* Preserves the directions of maximum variance.
* Simple and widely used dimensionality-reduction technique.

---

# ⚠️ Limitations

* PCA is computationally expensive for very large datasets.
* PCA is a linear transformation.
* Discarding components causes information loss.
* Maximum variance does not always mean maximum task-specific information.
* PCA components can be difficult to interpret physically.
* The transformation parameters must also be retained for accurate reconstruction.

---

# 🔮 Future Improvements

Possible extensions include:

* Comparing PCA with other dimensionality-reduction techniques.
* Adaptive selection of the number of components.
* Combining PCA with lossless compression.
* GPU acceleration for large hyperspectral datasets.
* Comparing PCA with JPEG2000 or other compression methods.
* Using nonlinear dimensionality-reduction techniques.
* Testing the method on multiple hyperspectral datasets.

---

# 👨‍💻 Project Goal

The goal of this project is to demonstrate how **Principal Component Analysis can reduce the dimensionality of hyperspectral images while retaining most of their important information**, making hyperspectral data more efficient to store and process.

---

## 📚 References

* Jolliffe, I. T. — *Principal Component Analysis*
* Gonzalez & Woods — *Digital Image Processing*
* Scikit-learn documentation — PCA
* Hyperspectral image processing literature

---

## ⭐ Conclusion

PCA provides an effective approach for reducing the spectral redundancy of hyperspectral images.

By transforming hundreds of correlated spectral bands into a smaller set of principal components, the amount of data can be significantly reduced while retaining most of the important information.

The trade-off is between:

```text
Number of Components
        ↕
Compression
        ↕
Information Loss
        ↕
Reconstruction Quality
```

This project demonstrates that PCA can serve as an important preprocessing and dimensionality-reduction step in hyperspectral image compression.
