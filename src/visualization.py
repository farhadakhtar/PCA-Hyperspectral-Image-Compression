"""Matplotlib visualizations. Every figure labels exactly what it shows.

- False-color composites are built from 3 user-chosen bands (percentile stretch).
- Single bands shown in grayscale with band number in title.
- PC maps are spatial images of Z columns reshaped to (H, W).
- Error maps: per-pixel RMSE over bands.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _stretch(x: np.ndarray, p1: float = 2, p99: float = 98) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    lo, hi = np.percentile(x, [p1, p99])
    if hi <= lo:
        hi = lo + 1e-12
    return np.clip((x - lo) / (hi - lo), 0, 1)


def false_color(cube: np.ndarray, bands: tuple[int, int, int] | None = None) -> np.ndarray:
    """Return (H, W, 3) float image. Default bands spread across the range."""
    cube = np.asarray(cube, dtype=np.float64)
    b = cube.shape[2]
    if bands is None:
        bands = (int(b * 0.15), int(b * 0.45), int(b * 0.8))
    rgb = np.stack([_stretch(cube[:, :, i]) for i in bands], axis=-1)
    return rgb


def fig_band(cube: np.ndarray, band: int):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cube[:, :, band], cmap="gray")
    ax.set_title(f"Band {band + 1}/{cube.shape[2]} (single band, grayscale)")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def fig_false_color(cube: np.ndarray, bands, title: str):
    rgb = false_color(cube, tuple(bands))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(rgb)
    ax.set_title(f"{title}\nFalse-color R={bands[0]+1} G={bands[1]+1} B={bands[2]+1}")
    ax.axis("off")
    fig.tight_layout()
    return fig


def fig_pc_grid(Z_hw: np.ndarray, evr: np.ndarray, n_show: int = 9):
    """Z_hw: (H, W, K). Titles carry each PC's explained variance."""
    k = Z_hw.shape[2]
    n_show = min(n_show, k)
    cols = 3
    rows = int(np.ceil(n_show / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_2d(axes)
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i < n_show:
            im = ax.imshow(Z_hw[:, :, i], cmap="viridis")
            ax.set_title(f"PC{i+1} spatial map\nEVR={evr[i]*100:.2f}%")
            fig.colorbar(im, ax=ax, fraction=0.046)
        ax.axis("off")
    fig.suptitle("Principal Component spatial maps", fontsize=12)
    fig.tight_layout()
    return fig


def fig_variance(evr: np.ndarray, k_selected: int):
    evr = np.asarray(evr)
    cum = np.cumsum(evr)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.5))
    a1.bar(np.arange(1, len(evr) + 1), evr * 100)
    a1.axvline(k_selected + 0.5 - 1 + 1, color="r", linestyle="--", label=f"K={k_selected}")
    a1.set_xlabel("Principal component")
    a1.set_ylabel("Explained variance (%)")
    a1.set_title("Individual explained variance (scree)")
    a1.legend()
    a2.plot(np.arange(1, len(evr) + 1), cum * 100, marker="o", ms=3)
    a2.axvline(k_selected, color="r", linestyle="--", label=f"K={k_selected}")
    a2.axhline(cum[k_selected - 1] * 100, color="gray", linestyle=":")
    a2.set_xlabel("K components")
    a2.set_ylabel("Cumulative explained variance (%)")
    a2.set_title("Cumulative explained variance")
    a2.set_ylim(0, 100.5)
    a2.legend()
    fig.tight_layout()
    return fig


def pixel_rmse_map(orig_cube: np.ndarray, recon_cube: np.ndarray) -> np.ndarray:
    o = np.asarray(orig_cube, dtype=np.float64)
    r = np.asarray(recon_cube, dtype=np.float64)
    return np.sqrt(np.mean((o - r) ** 2, axis=2))


def fig_comparison(orig_rgb: np.ndarray, recon_rgb: np.ndarray, err: np.ndarray,
                   title_orig: str = "Original — False Color Composite",
                   title_recon: str = "Reconstructed — False Color Composite"):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(orig_rgb)
    axes[0].set_title(title_orig)
    axes[0].axis("off")
    axes[1].imshow(recon_rgb)
    axes[1].set_title(title_recon)
    axes[1].axis("off")
    im = axes[2].imshow(err, cmap="hot")
    axes[2].set_title("Per-pixel RMSE error map")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    fig.tight_layout()
    return fig


def fig_experiment_table(results: list[dict]):
    """Line plots of K vs EVR/MSE/PSNR for the experiment panel."""
    ks = [r["k"] for r in results]
    ev = [r["explained_variance"] * 100 for r in results]
    ms = [r["mse"] for r in results]
    ps = [r["psnr_db"] for r in results]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].plot(ks, ev, marker="o")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Explained variance (%)")
    axes[0].set_title("K vs retained variance")
    axes[1].plot(ks, ms, marker="o")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("MSE")
    axes[1].set_title("K vs MSE (lower = smaller error)")
    axes[2].plot(ks, ps, marker="o")
    axes[2].set_xlabel("K")
    axes[2].set_ylabel("PSNR (dB)")
    axes[2].set_title("K vs PSNR (higher = smaller error)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
