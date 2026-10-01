"""CLI: python -m src.pca_compression --input data/raw --components 20 --output results/

Prints dataset dims, PCA config, selected K, EVR, MSE, PSNR, reduction, paths.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .compression import save_package, spectral_reduction, storage_report
from .data_loader import load_hyperspectral_data
from .metrics import full_report
from .pca_engine import fit_pca, k_for_variance
from .preprocessing import flatten_cube, to_working_float, unflatten
from .reconstruction import reconstruct_cube
from .utils import format_bytes, get_logger, software_versions, timestamp_now
from .visualization import (false_color, fig_band, fig_comparison, fig_false_color,
                            fig_pc_grid, fig_variance, pixel_rmse_map)

log = get_logger()


def run_pipeline(input_path: str, k: int | None, variance: float | None,
                 mode: str, output: str, false_bands: tuple[int, int, int] | None = None) -> dict:
    ds = load_hyperspectral_data(input_path)
    log.info("Dataset shape (H,W,B): %s dtype=%s", ds.cube.shape, ds.dtype_original)
    cube_f = to_working_float(ds.cube, np.float64)
    H, W, B = cube_f.shape
    X = flatten_cube(cube_f)
    log.info("Reshaped cube -> matrix X %s", X.shape)

    engine = fit_pca(X, mode=mode)
    r = engine.result
    assert r is not None
    if variance is not None:
        k = k_for_variance(r.eigvals_, float(variance))
        log.info("Variance target %.2f -> K=%d", float(variance), k)
    if k is None:
        raise ValueError("Provide --components or --variance.")
    if not 1 <= int(k) <= B:
        raise ValueError(f"K must satisfy 1<=K<={B}")
    k = int(k)

    Z = engine.transform(X, k)
    Wk = r.basis_k(k)
    log.info("Projection Z=Xc Wk: %s", Z.shape)
    recon_cube = reconstruct_cube(Z, Wk, r.mean_, H, W)
    log.info("Reconstruction Xhat=Z Wk^T+mu: %s", recon_cube.shape)

    max_val = float(np.max(cube_f))
    rep = full_report(cube_f, recon_cube, r.eigvals_, k, max_val)
    spec = spectral_reduction(B, k)

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    figs = out / "figures"
    figs.mkdir(exist_ok=True)
    pkg_path = out / f"compressed_k{k}.npz"
    save_package(pkg_path, Z, Wk, r.mean_,
                 {"dataset": ds.source_path, "mode": mode, "k": k,
                  "shape": [H, W, B], "time": timestamp_now()})
    stor = storage_report(ds.cube, pkg_path, Z.astype(np.float32),
                          Wk.astype(np.float32), r.mean_.astype(np.float32))

    # Figures
    fb = false_bands or (int(B * 0.15), int(B * 0.45), int(B * 0.8))
    fig_false_color(cube_f, fb, "Original").savefig(figs / "original_falsecolor.png", dpi=150)
    fig_false_color(recon_cube, fb, "Reconstructed").savefig(figs / "reconstructed_falsecolor.png", dpi=150)
    fig_band(cube_f, B // 2).savefig(figs / "original_band.png", dpi=150)
    Z_hw = Z.reshape(H, W, k)
    fig_pc_grid(Z_hw, r.evr()).savefig(figs / "pc_grid.png", dpi=150)
    fig_variance(r.evr(), k).savefig(figs / "explained_variance.png", dpi=150)
    fig_comparison(false_color(cube_f, fb), false_color(recon_cube, fb),
                   pixel_rmse_map(cube_f, recon_cube)).savefig(figs / "comparison.png", dpi=150)

    summary = {
        "dataset": ds.summary(),
        "mode": mode,
        "k": k,
        "explained_variance": rep["explained_variance"],
        "mse": rep["mse"],
        "mae": rep["mae"],
        "max_error": rep["max_error"],
        "psnr_db": rep["psnr_db"],
        "max_val_used": rep["max_val_used"],
        "sam_deg": rep["sam_deg"],
        "spectral_reduction": spec,
        "storage": stor,
        "package": str(pkg_path),
        "figures": str(figs),
        "time": timestamp_now(),
        "versions": software_versions(),
    }
    (out / f"metrics_k{k}.json").write_text(json.dumps(summary, indent=2))
    log.info("Selected K=%d | EVR=%.4f | MSE=%.6g | PSNR=%.2f dB | reduction %s (%.2fx)",
             k, rep["explained_variance"], rep["mse"], rep["psnr_db"],
             spec["reduction_str"], spec["factor"])
    log.info("Original %s | package %s | ratio %.2f",
             format_bytes(stor["original_bytes"]), format_bytes(stor["actual_package_bytes"]),
             stor["actual_ratio"])
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="PCA hyperspectral compression CLI")
    ap.add_argument("--input", required=True, help="Dataset file (.npy/.npz/.mat/.tif) or directory")
    ap.add_argument("--components", type=int, default=None, help="K components")
    ap.add_argument("--variance", type=float, default=None, help="Variance target 0-1")
    ap.add_argument("--mode", default="educational", choices=["educational", "optimized"])
    ap.add_argument("--output", default="results", help="Output directory")
    a = ap.parse_args(argv)
    s = run_pipeline(a.input, a.components, a.variance, a.mode, a.output)
    print(f"Dataset: {s['dataset']['height']}x{s['dataset']['width']}x{s['dataset']['bands']}")
    print(f"Mode: {s['mode']}  K={s['k']}")
    print(f"Explained variance: {s['explained_variance']*100:.4f}%")
    print(f"MSE={s['mse']:.6g}  MAE={s['mae']:.6g}  MAXERR={s['max_error']:.6g}")
    print(f"PSNR={s['psnr_db']:.2f} dB (MAX={s['max_val_used']})  SAM={s['sam_deg']:.4f} deg")
    print(f"Reduction: {s['spectral_reduction']['reduction_str']} "
          f"({s['spectral_reduction']['factor']:.2f}x, {s['spectral_reduction']['percent']:.1f}% fewer dims)")
    print(f"Package: {s['package']}  Figures: {s['figures']}")


if __name__ == "__main__":
    main()
