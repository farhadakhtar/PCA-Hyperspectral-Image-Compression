"""Hyperspectral PCA Lab — Streamlit dashboard.

Run: streamlit run app/main.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.compression import spectral_reduction, storage_report  # noqa: E402
from src.data_loader import band_statistics, load_hyperspectral_data  # noqa: E402
from src.metrics import full_report  # noqa: E402
from src.pca_engine import explained_variance_ratio, fit_pca, k_for_variance  # noqa: E402
from src.preprocessing import flatten_cube, to_working_float  # noqa: E402
from src.reconstruction import reconstruct_cube  # noqa: E402
from src.utils import format_bytes  # noqa: E402
from src.visualization import (false_color, fig_comparison, fig_experiment_table,  # noqa: E402
                               fig_variance, pixel_rmse_map)
from src.image_converter import (SUPPORTED_DTypes, WARNING_TEXT, cast_cube, cube_info,  # noqa: E402
                                 cube_to_npz_bytes, cube_to_npy_bytes, load_standard_image,
                                 rgb_to_cube, synthetic_bands)

st.set_page_config(page_title="Hyperspectral PCA Lab", layout="wide")

st.title("Hyperspectral PCA Lab")
st.subheader("PCA-Based Hyperspectral Image Compression & Reconstruction — Indian Pines")
st.caption("DATA → MATHEMATICS → ALGORITHM → REDUCTION → RECONSTRUCTION → EVALUATION")

PAGES = ["Dashboard", "Dataset", "PCA Analysis", "Components",
         "Reconstruction", "Experiments", "Metrics", "Export",
         "Image → NPY Converter", "About PCA"]
page = st.sidebar.radio("Navigation", PAGES)

# ---------- session state ----------
ss = st.session_state
ss.setdefault("cube", None)
ss.setdefault("ds", None)
ss.setdefault("engine", None)
ss.setdefault("X", None)
ss.setdefault("Z", None)
ss.setdefault("recon", None)
ss.setdefault("k", 20)
ss.setdefault("mode", "educational")
ss.setdefault("fb", None)

DEFAULT_DIR = ROOT / "data" / "raw"


def _load_from(path_str: str):
    ds = load_hyperspectral_data(path_str)
    cube = to_working_float(ds.cube, np.float64)
    ss.ds, ss.cube = ds, cube
    ss.engine, ss.Z, ss.recon = None, None, None
    B = cube.shape[2]
    ss.fb = (int(B * 0.15), int(B * 0.45), int(B * 0.8))


with st.sidebar:
    st.header("Dataset")
    src = st.text_input("Dataset path (file or folder)", value=str(DEFAULT_DIR))
    if st.button("Load Dataset"):
        try:
            with st.spinner("Loading hyperspectral cube…"):
                _load_from(src)
            st.success(f"Loaded {ss.cube.shape}")
        except Exception as e:
            st.error(f"Load failed: {e}")
    up = st.file_uploader("…or upload .npy/.npz/.mat", type=["npy", "npz", "mat"])
    if up is not None and st.button("Load uploaded file"):
        tmp = ROOT / "data" / "raw" / f"_upload_{up.name}"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(up.getvalue())
        try:
            _load_from(str(tmp))
            st.success(f"Loaded upload {ss.cube.shape}")
        except Exception as e:
            st.error(f"Load failed: {e}")

# ================= Image → NPY Converter (works without a loaded cube) =================
if page == "Image → NPY Converter":
    st.header("Image → NPY Converter")
    st.warning(WARNING_TEXT)
    st.caption("True RGB mode stores the 3 measured channels. "
               "Synthetic mode expands RGB into N *approximated* bands for pipeline testing only.")

    img_up = st.file_uploader("Upload image (JPG / JPEG / PNG / BMP / TIFF)",
                              type=["jpg", "jpeg", "png", "bmp", "tiff", "tif"])
    mode = st.radio("Conversion mode",
                    ["True RGB cube (H × W × 3)",
                     "Synthetic hyperspectral (SYNTHETIC / APPROXIMATED)"])
    is_synth = mode.startswith("Synthetic")
    n_bands = 10
    if is_synth:
        n_bands = st.selectbox("Synthetic bands", [10, 20, 50, 100, 200], index=1)
        st.caption("SYNTHETIC/APPROXIMATED hyperspectral data — interpolated from RGB, "
                   "not real spectral measurements.")
    out_dtype = st.selectbox("Output data type", list(SUPPORTED_DTypes), index=0)

    if img_up is not None:
        raw_bytes = img_up.getvalue()
        try:
            rgb, info = load_standard_image(raw_bytes)
        except Exception as e:
            st.error(f"Could not read image: {e}")
            st.stop()
        # A new upload invalidates any previous conversion preview.
        if ss.get("conv_src") != img_up.name:
            ss["conv_cube"] = None
        st.subheader("Before conversion — original image")
        st.image(rgb, caption=f"{img_up.name} (RGB preview)", use_container_width=True)
        st.write(f"Width × Height: **{info['width']} × {info['height']}** · "
                 f"Color channels: **{info['channels']} (R, G, B)** · "
                 f"File size: **{format_bytes(info['file_bytes'])}** "
                 f"(mode {info['original_mode']}, format {info['format'] or 'unknown'})")

        if st.button("Convert"):
            try:
                if is_synth:
                    cube_c = synthetic_bands(rgb, int(n_bands), out_dtype)
                    tag = f"synthetic{int(n_bands)}"
                else:
                    cube_c = rgb_to_cube(rgb, out_dtype)
                    tag = "rgb3"
                ss["conv_cube"] = cube_c
                ss["conv_name"] = f"{Path(img_up.name).stem}_{tag}"
                ss["conv_src"] = img_up.name
            except Exception as e:
                st.error(f"Conversion failed: {e}")
                st.stop()

        if ss.get("conv_cube") is None:
            st.info("Press **Convert** to generate the cube.")

    if (ss.get("conv_cube") is not None and img_up is not None
            and ss.get("conv_src") == img_up.name):
        cube_c = np.ascontiguousarray(ss["conv_cube"])
        ci = cube_info(cube_c, ss.get("conv_name", "converted"))
        st.subheader("After conversion — generated cube")
        st.write(f"Cube dimensions: **{ci['height']} × {ci['width']} × {ci['bands']}** · "
                 f"Bands: **{ci['bands']}** · Data type: **{ci['dtype']}** · "
                 f"Memory: **{format_bytes(ci['memory_bytes'])}**")
        if is_synth:
            st.caption("SYNTHETIC/APPROXIMATED bands — do not treat as real spectra.")
        band_c = st.slider("Preview band", 1, ci["bands"], 1) - 1
        fig_c, ax_c = plt.subplots()
        im_c = ax_c.imshow(np.asarray(cube_c[:, :, band_c]))
        ax_c.set_title(f"Converted cube — band {band_c + 1}/{ci['bands']}"
                       + (" (R/G/B)" if ci["bands"] == 3 else " (synthetic band)"))
        ax_c.axis("off")
        fig_c.colorbar(im_c, fraction=0.046)
        st.pyplot(fig_c)

        npy_bytes = cube_to_npy_bytes(cube_c)
        npz_bytes = cube_to_npz_bytes(cube_c)
        st.download_button("Download .NPY", npy_bytes,
                           file_name=f"{ss.get('conv_name', 'converted')}.npy")
        st.download_button("Download .NPZ", npz_bytes,
                           file_name=f"{ss.get('conv_name', 'converted')}.npz")
        if st.button("Send to PCA pipeline"):
            from src.data_loader import HyperspectralDataset
            ds_c = HyperspectralDataset(cube=np.asarray(cube_c),
                                        name=f"Converted ({ss.get('conv_name', '')})",
                                        source_path="Image → NPY Converter",
                                        dtype_original=str(cube_c.dtype))
            ss.ds, ss.cube = ds_c, cube_c.astype(np.float64)
            ss.engine, ss.Z, ss.recon, ss.X = None, None, None, None
            B_c = cube_c.shape[2]
            ss.fb = (0, min(1, B_c - 1), min(2, B_c - 1))
            ss.k = min(20, B_c)
            st.success(f"Sent {cube_c.shape} to the PCA pipeline. "
                       "Open PCA Analysis and press RUN PCA. (Indian Pines workflow untouched.)")
    elif img_up is None:
        st.info("Upload a JPG/JPEG/PNG/BMP/TIFF image to begin.")
    st.stop()

if ss.cube is None:
    st.info("Load the Indian Pines dataset to begin. Default: `data/raw/` "
            "(expects `indianpinearray.npy` + optional `IPgt.npy`).")
    st.stop()

ds, cube = ss.ds, ss.cube
H, W, B = cube.shape

# ================= Dashboard =================
if page == "Dashboard":
    st.header("Dashboard")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Height", H)
    c2.metric("Width", W)
    c3.metric("Bands (B)", B)
    c4.metric("Pixels (N)", f"{H*W:,}")
    c5.metric("Raw memory", format_bytes(cube.nbytes))
    st.write(f"Source: `{ds.source_path}` · on-disk dtype: `{ds.dtype_original}` · "
             f"total values N×B = **{H*W*B:,}**"
             + (f" · labels: `{ds.labels_source}` {ds.labels.shape}" if ds.labels is not None else ""))
    st.image(false_color(cube, ss.fb), caption=f"Original — False Color Composite (bands {[b+1 for b in ss.fb]})",
             use_container_width=True)
    st.markdown("**Pipeline:** cube (H×W×B) → X (N×B) → μ, Xc=X−μ → C=XcᵀXc/(N−1) → Cv=λv → "
                "Z=XcWₖ → X̂=ZWₖᵀ+μ → MSE/PSNR/EVR.")

# ================= Dataset =================
elif page == "Dataset":
    st.header("Dataset preview")
    band = st.slider("Spectral band", 1, B, B // 2) - 1
    stats = band_statistics(cube, band)
    st.write(f"Band {stats['band']+1}: min={stats['min']:.1f} max={stats['max']:.1f} "
             f"mean={stats['mean']:.1f} std={stats['std']:.1f}")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        ax.imshow(cube[:, :, band], cmap="gray")
        ax.set_title(f"Original — Band {band+1} (grayscale, single band)")
        ax.axis("off")
        st.pyplot(fig)
    with col2:
        st.image(false_color(cube, ss.fb),
                 caption=f"Original — False Color Composite R={ss.fb[0]+1} G={ss.fb[1]+1} B={ss.fb[2]+1}")
    st.subheader("False-color band selection")
    r = st.number_input("R band", 1, B, ss.fb[0] + 1)
    g = st.number_input("G band", 1, B, ss.fb[1] + 1)
    b_ = st.number_input("B band", 1, B, ss.fb[2] + 1)
    ss.fb = (int(r) - 1, int(g) - 1, int(b_) - 1)
    means = cube.reshape(-1, B).mean(axis=0)
    fig2, ax2 = plt.subplots()
    ax2.plot(np.arange(1, B + 1), means)
    ax2.set_xlabel("Band")
    ax2.set_ylabel("Mean radiance")
    ax2.set_title("Mean spectral signature (all pixels)")
    st.pyplot(fig2)

# ================= PCA Analysis =================
elif page == "PCA Analysis":
    st.header("PCA configuration")
    mode = st.radio("PCA Mode", ["educational", "optimized"],
                    help="Educational: covariance+eigh (transparent). Optimized: sklearn SVD (robust). Same math.")
    sel = st.radio("Component selection", ["Manual", "Variance Target"])
    B_ = B
    if sel == "Manual":
        k_in = st.slider("Number of components K", 1, B_, min(20, B_))
        target = None
    else:
        target = st.selectbox("Variance target", [0.80, 0.90, 0.95, 0.99], index=2)
        k_in = None
    if st.button("RUN PCA"):
        with st.spinner("Fitting PCA… (covariance is only B×B)"):
            X = flatten_cube(cube)
            ss.X = X
            eng = fit_pca(X, mode=mode)
            kk = k_for_variance(eng.result.eigvals_, target) if target else int(k_in)
            Z = eng.transform(X, kk)
            recon = reconstruct_cube(Z, eng.result.basis_k(kk), eng.result.mean_, H, W)
            ss.engine, ss.k, ss.mode, ss.Z, ss.recon = eng, kk, mode, Z, recon
        st.success(f"PCA done: B={B_} → K={kk}")
    if ss.engine is not None:
        r = ss.engine.result
        evr = explained_variance_ratio(r.eigvals_)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Original Bands", B)
        c2.metric("Selected PCs", ss.k)
        c3.metric("Variance Retained", f"{evr[:ss.k].sum()*100:.2f}%")
        c4.metric("Reduction", f"{B/ss.k:.1f}×")
        st.pyplot(fig_variance(evr, ss.k))
        with st.expander("Mathematics of this run"):
            st.latex(r"X_c = X - \mu \qquad C = \frac{1}{N-1}X_c^T X_c \qquad Cv=\lambda v")
            st.latex(r"Z = X_c W_k \qquad \hat{X} = Z W_k^T + \mu")
            st.write(f"X: {ss.X.shape} · μ: {r.mean_.shape} · C: ({B}×{B}) · "
                     f"W: {r.components_.shape} · Z: {ss.Z.shape} · X̂: {ss.recon.shape}")

# ================= Components =================
elif page == "Components":
    st.header("Principal component viewer")
    if ss.engine is None:
        st.warning("Run PCA first (PCA Analysis page).")
        st.stop()
    evr = explained_variance_ratio(ss.engine.result.eigvals_)
    n_show = st.slider("PCs to show", 1, min(16, ss.k), min(9, ss.k))
    Zhw = ss.Z.reshape(H, W, ss.k)
    cols = st.columns(3)
    for i in range(n_show):
        with cols[i % 3]:
            fig, ax = plt.subplots()
            ax.imshow(Zhw[:, :, i], cmap="viridis")
            ax.set_title(f"PC{i+1} spatial map · EVR={evr[i]*100:.2f}%")
            ax.axis("off")
            st.pyplot(fig)
    pc = st.selectbox("Enlarge PC", list(range(1, ss.k + 1))) - 1
    fig, ax = plt.subplots()
    im = ax.imshow(Zhw[:, :, pc], cmap="viridis")
    ax.set_title(f"PC{pc+1} enlarged · EVR={evr[pc]*100:.3f}%")
    ax.axis("off")
    fig.colorbar(im, fraction=0.046)
    st.pyplot(fig)

# ================= Reconstruction =================
elif page == "Reconstruction":
    st.header("Reconstruction: Original vs Reconstructed vs Error")
    if ss.engine is None:
        st.warning("Run PCA first.")
        st.stop()
    err = pixel_rmse_map(cube, ss.recon)
    st.pyplot(fig_comparison(false_color(cube, ss.fb), false_color(ss.recon, ss.fb), err))
    band = st.slider("Band for numeric inspection", 1, B, 30) - 1
    o = cube[:, :, band]
    rr = ss.recon[:, :, band]
    st.write(f"Band {band+1}: band-MSE={np.mean((o-rr)**2):.4g} · "
             f"band-MAE={np.mean(np.abs(o-rr)):.4g}")

# ================= Experiments =================
elif page == "Experiments":
    st.header("Experiment: K sweep")
    st.caption("Compares compression vs information loss across K. No universal 'best K' — read the table.")
    ks_txt = st.text_input("K values (comma separated)", "5,10,20,30,50")
    if st.button("Run sweep"):
        ks = sorted({int(x) for x in ks_txt.split(",") if x.strip().isdigit()})
        ks = [k for k in ks if 1 <= k <= B]
        X = flatten_cube(cube)
        eng = fit_pca(X, mode="educational")
        rows = []
        maxv = float(cube.max())
        for k in ks:
            Z = eng.transform(X, k)
            rc = reconstruct_cube(Z, eng.result.basis_k(k), eng.result.mean_, H, W)
            rep = full_report(cube, rc, eng.result.eigvals_, k, maxv)
            rep["reduction"] = B / k
            rows.append(rep)
        ss["sweep"] = rows
    if "sweep" in ss and ss["sweep"]:
        import pandas as pd
        df = pd.DataFrame(ss["sweep"])[["k", "explained_variance", "mse", "mae", "psnr_db", "sam_deg", "reduction"]]
        st.dataframe(df.style.format({"explained_variance": "{:.4f}", "mse": "{:.4g}",
                                              "psnr_db": "{:.2f}", "sam_deg": "{:.3f}",
                                              "reduction": "{:.2f}x"}))
        st.pyplot(fig_experiment_table(ss["sweep"]))

# ================= Metrics =================
elif page == "Metrics":
    st.header("Quality metrics")
    if ss.engine is None:
        st.warning("Run PCA first.")
        st.stop()
    rep = full_report(cube, ss.recon, ss.engine.result.eigvals_, ss.k, float(cube.max()))
    spec = spectral_reduction(B, ss.k)
    c1, c2, c3 = st.columns(3)
    c1.metric("MSE (lower = smaller error)", f"{rep['mse']:.6g}")
    c2.metric("PSNR dB (higher = smaller error)", f"{rep['psnr_db']:.2f}")
    c3.metric("Explained variance", f"{rep['explained_variance']*100:.3f}%")
    c4, c5, c6 = st.columns(3)
    c4.metric("MAE", f"{rep['mae']:.6g}")
    c5.metric("Max error", f"{rep['max_error']:.6g}")
    c6.metric("SAM (deg, lower = better shape)", f"{rep['sam_deg']:.3f}")
    st.info(f"PSNR MAX = data peak ({rep['max_val_used']:.1f}), not assumed 255. "
            f"Spectral reduction {spec['reduction_str']} = {spec['factor']:.2f}× "
            f"({spec['percent']:.1f}% fewer dims). This is dimensionality reduction, "
            "not a file-size claim — see Export for measured bytes.")

# ================= Export =================
elif page == "Export":
    st.header("Export")
    if ss.engine is None:
        st.warning("Run PCA first.")
        st.stop()
    outdir = ROOT / "results"
    outdir.mkdir(exist_ok=True)
    if st.button("Save compressed package (.npz) + metrics (.json)"):
        import json
        from src.compression import save_package
        r = ss.engine.result
        pkg = outdir / f"compressed_k{ss.k}.npz"
        save_package(pkg, ss.Z, r.basis_k(ss.k), r.mean_,
                     {"k": ss.k, "mode": ss.mode, "shape": [H, W, B]})
        rep = full_report(cube, ss.recon, r.eigvals_, ss.k, float(cube.max()))
        (outdir / f"metrics_k{ss.k}.json").write_text(json.dumps(rep, indent=2, default=float))
        st.success(f"Saved {pkg} ({format_bytes(pkg.stat().st_size)})")
        stor = storage_report(ss.ds.cube, pkg, ss.Z.astype(np.float32),
                              r.basis_k(ss.k).astype(np.float32), r.mean_.astype(np.float32))
        st.write(f"Original array bytes: {format_bytes(stor['original_bytes'])} · "
                 f"package: {format_bytes(stor['actual_package_bytes'])} · "
                 f"ratio: {stor['actual_ratio']:.2f}× · saving: {stor['actual_saving_pct']:.1f}%")
    # Downloads
    buf = io.BytesIO()
    np.save(buf, ss.recon)
    st.download_button("Download reconstructed cube (.npy)", buf.getvalue(), "reconstructed.npy")
    buf2 = io.BytesIO()
    np.save(buf2, ss.Z.astype(np.float32))
    st.download_button("Download PCA scores Z (.npy)", buf2.getvalue(), "pc_scores.npy")

# ================= About =================
elif page == "About PCA":
    st.header("Explain PCA — 9 steps")
    steps = [
        ("1. Flatten cube", f"(H,W,B)=({H},{W},{B}) → X (N×B)=({H*W}×{B}). Spectral PCA: features = bands.",
         "reshape, C-order both ways"),
        ("2. Mean-center", "μ=mean(X) per band; Xc=X−μ. Centering is required; no silent scaling.", "Xc = X − μ"),
        ("3. Covariance", f"C=XcᵀXc/(N−1), only {B}×{B} — never N×N.", "C = XcᵀXc/(N−1)"),
        ("4. Eigendecomposition", "Cv=λv with eigh (symmetric). λ=variance, v=direction.", "Cv = λv"),
        ("5. Sort", "λ₁≥λ₂≥…; PC1 = largest λ.", "argsort descending"),
        ("6. Select K", "Manual K or smallest K reaching variance target.", "cumsum(λ)/Σλ ≥ t"),
        ("7. Project", "Z=XcWₖ (N×K): the compressed representation.", "Z = XcWₖ"),
        ("8. Reconstruct", "X̂=ZWₖᵀ+μ, reshape to (H,W,B).", "X̂ = ZWₖᵀ+μ"),
        ("9. Evaluate", "EVR, MSE, PSNR=10log₁₀(MAX²/MSE) with explicit MAX, SAM.", "see Metrics page"),
    ]
    for t, d, e in steps:
        with st.expander(t):
            st.write(d)
            st.code(e)
