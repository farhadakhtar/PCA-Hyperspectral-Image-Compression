"""Tests for the Image -> NPY converter + PCA-loader compatibility."""
import io

import numpy as np
import pytest
from PIL import Image

from src import data_loader as dl
from src import image_converter as ic
from src.pca_engine import EducationalPCA
from src.preprocessing import flatten_cube


def _img_bytes(fmt="PNG", size=(24, 16), mode="RGB", color=(10, 120, 200)):
    im = Image.new(mode, size, color)
    buf = io.BytesIO()
    im.save(buf, format=fmt)
    return buf.getvalue()


def _img_bytes_gray(fmt="PNG"):
    im = Image.new("L", (20, 12), 128)
    buf = io.BytesIO()
    im.save(buf, format=fmt)
    return buf.getvalue()


def _img_bytes_rgba():
    im = Image.new("RGBA", (20, 12), (10, 120, 200, 128))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def test_png_loading_reports_dims():
    rgb, info = ic.load_standard_image(_img_bytes("PNG"))
    assert rgb.shape == (16, 24, 3) and rgb.dtype == np.uint8
    assert info["width"] == 24 and info["height"] == 16 and info["channels"] == 3
    assert info["file_bytes"] > 0


def test_jpeg_loading():
    rgb, info = ic.load_standard_image(_img_bytes("JPEG"))
    assert rgb.shape == (16, 24, 3)
    assert info["format"] == "JPEG"


def test_bmp_loading():
    rgb, info = ic.load_standard_image(_img_bytes("BMP"))
    assert rgb.shape == (16, 24, 3)


def test_tiff_loading():
    rgb, info = ic.load_standard_image(_img_bytes("TIFF"))
    assert rgb.shape == (16, 24, 3)


def test_grayscale_and_alpha_normalized_to_rgb():
    rgb, _ = ic.load_standard_image(_img_bytes_gray())
    assert rgb.shape == (12, 20, 3)
    assert np.array_equal(rgb[:, :, 0], rgb[:, :, 1])
    rgb2, _ = ic.load_standard_image(_img_bytes_rgba())
    assert rgb2.shape == (12, 20, 3)


def test_rejects_unsupported_and_corrupt():
    with pytest.raises(ValueError):
        ic.load_standard_image(b"not an image at all")
    with pytest.raises((ValueError, TypeError)):
        ic.load_standard_image(12345)


def test_rgb_to_npy_shape_and_dtypes():
    rgb, _ = ic.load_standard_image(_img_bytes("PNG"))
    for dt, kind in (("uint8", np.uint8), ("uint16", np.uint16), ("float32", np.float32)):
        cube = ic.rgb_to_cube(rgb, dt)
        assert cube.shape == (16, 24, 3)  # H x W x 3
        assert cube.dtype == kind
    # uint16 scales 0-255 -> 0-65535; float32 stays 0-1.
    assert ic.rgb_to_cube(np.full((2, 2, 3), 255, np.uint8), "uint16").max() == 65535
    assert ic.rgb_to_cube(np.full((2, 2, 3), 255, np.uint8), "float32").max() == pytest.approx(1.0)


def test_synthetic_band_counts_and_shape():
    rgb, _ = ic.load_standard_image(_img_bytes("PNG"))
    for b in (10, 20, 50):
        s = ic.synthetic_bands(rgb, b, "uint8")
        assert s.shape == (16, 24, b)
    with pytest.raises(ValueError):
        ic.synthetic_bands(rgb, 2, "uint8")  # < 3 rejected


def test_synthetic_bands_vary_and_span_input():
    rng = np.random.default_rng(7)
    rgb = rng.integers(0, 256, size=(12, 10, 3), dtype=np.uint8)
    s = ic.synthetic_bands(rgb, 20, "float32").astype(np.float64)
    assert np.unique(np.round(s, decimals=4)).size > 20  # bands actually differ
    assert s.min() >= 0.0 and s.max() <= 1.0


def test_save_load_roundtrip_npy_npz(tmp_path):
    rgb, _ = ic.load_standard_image(_img_bytes("PNG"))
    cube = ic.rgb_to_cube(rgb, "uint8")
    p1 = ic.save_cube(cube, tmp_path / "conv", fmt="npy")
    p2 = ic.save_cube(cube, tmp_path / "conv", fmt="npz")
    assert np.array_equal(np.load(str(p1)), cube)
    z = np.load(str(p2))
    assert np.array_equal(np.asarray(z["cube"]), cube)


def test_loader_compatibility_and_pca(tmp_path):
    """Generated files must flow straight into the existing PCA pipeline."""
    rgb, _ = ic.load_standard_image(_img_bytes("PNG", size=(32, 32)))
    for cube, name in ((ic.rgb_to_cube(rgb, "uint8"), "rgb.npy"),
                       (ic.synthetic_bands(rgb, 20, "uint8"), "synth.npz")):
        p = tmp_path / name
        if name.endswith(".npz"):
            np.savez_compressed(str(p), cube=cube)
        else:
            np.save(str(p), cube)
        ds = dl.load_hyperspectral_data(str(p), dataset_name="Converted")
        assert ds.cube.shape == cube.shape and ds.cube.ndim == 3
        X = flatten_cube(ds.cube.astype(np.float64))
        eng = EducationalPCA().fit(X)
        k = min(5, cube.shape[2])
        Z = eng.transform(X, k)
        assert Z.shape == (X.shape[0], k)
        assert eng.inverse_transform(Z).shape == X.shape
