"""Standard image -> hyperspectral/NPY converter.

Two modes:

1. True RGB cube:  H x W x 3  (R, G, B as measured — no invented data).
2. Synthetic hyperspectral: RGB expanded to B bands by deterministic
   spectral-response mixing. APPROXIMATED — does not recover real spectra.

All outputs are plain (H, W, B) numeric arrays compatible with
``src.data_loader.load_cube_file`` / ``load_hyperspectral_data``.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np

SUPPORTED_INPUT_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
SUPPORTED_DTypes = ("uint8", "uint16", "float32")

WARNING_TEXT = (
    "RGB images contain only 3 measured color channels. "
    "Synthetic hyperspectral conversion does not recover real spectral information."
)


def load_standard_image(data: bytes | BytesIO | str | Path) -> tuple[np.ndarray, dict]:
    """Load JPG/PNG/BMP/TIFF -> RGB uint8 array (H, W, 3) + info dict.

    - Grayscale (L) is replicated to 3 identical channels (documented, no fake color).
    - RGBA / LA / palette is converted to RGB (alpha dropped).
    - 16-bit / other modes are converted to 8-bit RGB via Pillow.
    """
    from PIL import Image

    if isinstance(data, (str, Path)):
        p = Path(data)
        suf = p.suffix.lower()
        if suf not in SUPPORTED_INPUT_SUFFIXES:
            raise ValueError(f"Unsupported image type '{suf}'. Use {sorted(SUPPORTED_INPUT_SUFFIXES)}")
        raw = p.read_bytes()
        name = p.name
    elif isinstance(data, BytesIO):
        raw = data.getvalue()
        name = "upload"
    elif isinstance(data, (bytes, bytearray)):
        raw = bytes(data)
        name = "upload"
    else:
        raise TypeError(f"Expected bytes/BytesIO/path, got {type(data)}")

    try:
        im = Image.open(BytesIO(raw))
        im.load()
    except Exception as e:
        raise ValueError(f"Could not decode image bytes: {e}") from e

    fmt = (im.format or "").upper()
    orig_mode = im.mode
    w, h = im.size

    if im.mode == "RGB":
        rgb = np.asarray(im, dtype=np.uint8)
    elif im.mode in ("RGBA", "LA", "PA"):
        rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    elif im.mode in ("L", "LA"):
        g = np.asarray(im.convert("L"), dtype=np.uint8)
        rgb = np.stack([g, g, g], axis=-1)
    elif im.mode == "P":
        rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    else:
        # I;16, CMYK, YCbCr, ... -> best-effort RGB.
        rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Conversion produced unexpected shape {rgb.shape}")
    # Ensure C-contiguous HxWx3.
    rgb = np.ascontiguousarray(rgb)
    info = {
        "name": name,
        "format": fmt,
        "original_mode": orig_mode,
        "width": int(w),
        "height": int(h),
        "channels": 3,
        "file_bytes": len(raw),
    }
    return rgb, info


def cast_cube(cube01_or_255: np.ndarray, dtype: str) -> np.ndarray:
    """Cast to requested dtype.

    Accepts float 0-1 or 0-255 input; output convention:
    uint8: 0-255, uint16: 0-65535, float32: 0-1.
    """
    if dtype not in SUPPORTED_DTypes:
        raise ValueError(f"dtype must be one of {SUPPORTED_DTypes}, got {dtype}")
    a = np.asarray(cube01_or_255, dtype=np.float64)
    # Detect scale: values > 1.5 mean 0-255 range.
    if a.max() > 1.5:
        a = a / 255.0
    a = np.clip(a, 0.0, 1.0)
    if dtype == "uint8":
        return (np.round(a * 255.0)).astype(np.uint8)
    if dtype == "uint16":
        return (np.round(a * 65535.0)).astype(np.uint16)
    return a.astype(np.float32)


def rgb_to_cube(rgb_uint8: np.ndarray, dtype: str = "uint8") -> np.ndarray:
    """True measured conversion: (H, W, 3) RGB -> (H, W, 3) cube. No invented data."""
    rgb = np.asarray(rgb_uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB (H, W, 3), got {rgb.shape}")
    if rgb.dtype != np.uint8:
        # Assume 0-255-ish numeric; clip and round.
        rgb = np.clip(np.asarray(rgb, dtype=np.float64), 0, 255).astype(np.uint8)
    if dtype == "uint8":
        return np.ascontiguousarray(rgb)
    return cast_cube(rgb.astype(np.float64), dtype)


def synthetic_bands(rgb_uint8: np.ndarray, n_bands: int, dtype: str = "uint8",
                    seed: int = 0) -> np.ndarray:
    """Expand RGB to (H, W, n_bands) SYNTHETIC/APPROXIMATED data.

    Method (deterministic, documented): each synthetic band is a smooth
    convex mixture of normalized R, G, B with sinusoidal weights swept across
    the band index, plus a mild per-band gamma (0.85-1.15) for response-shape
    variety. Bands are therefore correlated mixtures of the same 3 measurements
    — useful for exercising the PCA pipeline, NOT real spectra.
    """
    n_bands = int(n_bands)
    if n_bands < 3:
        raise ValueError(f"Synthetic bands must be >= 3, got {n_bands}")
    if n_bands > 512:
        raise ValueError(f"Synthetic bands capped at 512, got {n_bands}")
    rgb = np.asarray(rgb_uint8, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB (H, W, 3), got {rgb.shape}")
    rgb = np.clip(rgb, 0, 255) / 255.0  # 0-1
    rng = np.random.default_rng(seed)  # reserved for future dither; weights stay analytic
    _ = rng
    h, w, _ = rgb.shape
    t = np.linspace(0.0, 1.0, n_bands)
    # Smooth weights: each channel dominates a third of the sweep.
    wr = 0.15 + 0.85 * np.clip(np.cos((t - 0.15) * 2 * np.pi) * 0.5 + 0.5, 0, 1)
    wg = 0.15 + 0.85 * np.clip(np.cos((t - 0.50) * 2 * np.pi) * 0.5 + 0.5, 0, 1)
    wb = 0.15 + 0.85 * np.clip(np.cos((t - 0.85) * 2 * np.pi) * 0.5 + 0.5, 0, 1)
    s = wr + wg + wb
    wr, wg, wb = wr / s, wg / s, wb / s
    gamma = 0.85 + 0.30 * (0.5 + 0.5 * np.sin(2 * np.pi * t * 2 + 1.0))  # 0.85-1.15
    out = np.empty((h, w, n_bands), dtype=np.float64)
    for j in range(n_bands):
        mix = wr[j] * rgb[:, :, 0] + wg[j] * rgb[:, :, 1] + wb[j] * rgb[:, :, 2]
        out[:, :, j] = np.clip(mix, 0, 1) ** gamma[j]
    return cast_cube(out, dtype)


def cube_to_npy_bytes(cube: np.ndarray) -> bytes:
    buf = BytesIO()
    np.save(buf, np.ascontiguousarray(cube))
    return buf.getvalue()


def cube_to_npz_bytes(cube: np.ndarray) -> bytes:
    buf = BytesIO()
    np.savez_compressed(buf, cube=np.ascontiguousarray(cube))
    return buf.getvalue()


def save_cube(cube: np.ndarray, path: str | Path, fmt: str = "npy") -> Path:
    """Write cube to .npy / .npz (npz uses key 'cube' — loader-compatible)."""
    fmt = fmt.lower().lstrip(".")
    if fmt not in ("npy", "npz"):
        raise ValueError("fmt must be 'npy' or 'npz'")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "npy":
        if p.suffix.lower() != ".npy":
            p = p.with_suffix(".npy")
        np.save(str(p), np.ascontiguousarray(cube))
    else:
        if p.suffix.lower() != ".npz":
            p = p.with_suffix(".npz")
        np.savez_compressed(str(p), cube=np.ascontiguousarray(cube))
    return p


def cube_info(cube: np.ndarray, name: str = "converted") -> dict:
    c = np.asarray(cube)
    return {
        "name": name,
        "shape": tuple(c.shape),
        "height": int(c.shape[0]),
        "width": int(c.shape[1]),
        "bands": int(c.shape[2]),
        "dtype": str(c.dtype),
        "memory_bytes": int(c.nbytes),
    }
