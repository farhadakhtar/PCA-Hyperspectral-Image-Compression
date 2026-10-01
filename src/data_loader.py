"""Dataset loading abstraction for hyperspectral cubes.

Internal representation is always::

    cube.shape == (height, width, bands)

Supported inputs:
- NumPy .npy  (single array, 2D H*W x B or 3D H x W x B)
- NumPy .npz  (first array-like entry that looks hyperspectral)
- MATLAB .mat (scans variables for a 2D/3D numeric array)
- GeoTIFF/TIFF via Pillow or tifffile if available
- ENVI .hdr + binary via `spectral` library if installed (optional)
- A directory: searches for a cube file + optional ground-truth/label file.

Ground-truth / labels / masks are NEVER mixed into the cube used for PCA.
They are returned separately as ``labels`` when detected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .utils import get_logger

log = get_logger()

CUBE_CANDIDATE_SUFFIXES = {".npy", ".npz", ".mat", ".tif", ".tiff", ".hdr"}
GT_NAME_HINTS = ("gt", "groundtruth", "ground_truth", "labels", "mask", "truth")


@dataclass
class HyperspectralDataset:
    cube: np.ndarray              # (H, W, B)
    name: str = "Indian Pines"
    source_path: str = ""
    dtype_original: str = ""
    labels: Optional[np.ndarray] = None       # (H, W) or None
    labels_source: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def height(self) -> int:
        return int(self.cube.shape[0])

    @property
    def width(self) -> int:
        return int(self.cube.shape[1])

    @property
    def bands(self) -> int:
        return int(self.cube.shape[2])

    @property
    def pixels(self) -> int:
        return int(self.height * self.width)

    @property
    def total_values(self) -> int:
        return int(self.pixels * self.bands)

    def summary(self) -> dict:
        return {
            "name": self.name,
            "height": self.height,
            "width": self.width,
            "bands": self.bands,
            "pixels": self.pixels,
            "total_values": self.total_values,
            "dtype_original": self.dtype_original,
            "dtype_loaded": str(self.cube.dtype),
            "raw_bytes": int(self.total_values * np.dtype(self.cube.dtype).itemsize),
            "source": self.source_path,
            "has_labels": self.labels is not None,
            "labels_shape": None if self.labels is None else tuple(self.labels.shape),
        }


def _as_cube(arr: np.ndarray) -> np.ndarray:
    """Coerce a raw array to (H, W, B). Raises ValueError if impossible."""
    arr = np.asarray(arr)
    if arr.ndim == 3:
        h, w, b = arr.shape
        # Guard against band-first (B, H, W): bands are the smallest dim usually
        # for Indian Pines (200 < 145? no, 200 > 145). So use explicit heuristic:
        # if first dim is small (<= 300) and last dim is large (> 300), assume (B,H,W).
        if arr.shape[0] <= 300 and arr.shape[1] > 300 and arr.shape[2] > 300:
            arr = np.moveaxis(arr, 0, -1)
        if arr.shape[2] < 2:
            raise ValueError(f"Cube needs >= 2 bands, got shape {arr.shape}")
        return arr
    if arr.ndim == 2:
        # Could be (N, B) flattened or (H, W) single band (reject).
        n, b = arr.shape
        if b < 2:
            raise ValueError(f"2D array with <2 columns cannot be a cube: {arr.shape}")
        # Try to recover square spatial dims (Indian Pines 145x145).
        s = int(np.sqrt(n))
        if s * s == n:
            return arr.reshape(s, s, b)
        # Otherwise keep flat with W=1? Better: raise and ask user.
        raise ValueError(
            f"2D array shape {arr.shape} is ambiguous: N={n} is not a perfect "
            "square so spatial dims cannot be inferred. Provide a 3D (H,W,B) file."
        )
    raise ValueError(f"Cannot interpret array with ndim={arr.ndim}, shape={arr.shape}")


def _load_npy(path: Path) -> np.ndarray:
    return np.load(str(path), allow_pickle=False)


def _load_npz(path: Path) -> np.ndarray:
    z = np.load(str(path), allow_pickle=False)
    keys = list(z.keys())
    if not keys:
        raise ValueError(f"No arrays in {path}")
    # Prefer the largest numeric array with ndim>=2.
    best = None
    for k in keys:
        a = np.asarray(z[k])
        if a.ndim >= 2 and np.issubdtype(a.dtype, np.number):
            if best is None or a.size > best.size:
                best = a
    if best is None:
        raise ValueError(f"No numeric array found in {path}, keys={keys}")
    return best


def _load_mat(path: Path) -> np.ndarray:
    from scipy.io import loadmat
    d = loadmat(str(path))
    cands = []
    for k, v in d.items():
        if k.startswith("__"):
            continue
        v = np.asarray(v)
        if v.ndim in (2, 3) and np.issubdtype(v.dtype, np.number) and v.size > 1000:
            cands.append((v.size, k, v))
    if not cands:
        raise ValueError(f"No hyperspectral-like variable in {path}. Keys: {list(d.keys())}")
    cands.sort(reverse=True)
    # If the biggest 2D square-ish array looks like labels (few unique ints),
    # prefer the next candidate as cube. Simple guard: handled by caller too.
    return cands[0][2]


def _load_tiff(path: Path) -> np.ndarray:
    try:
        import tifffile as tf
        arr = tf.imread(str(path))
    except ImportError:
        from PIL import Image
        import numpy as _np
        im = Image.open(str(path))
        arr = _np.array(im)
    arr = np.asarray(arr)
    # tifffile often gives (B, H, W) or (H, W, B) or (H, W, S, B)… handle common.
    if arr.ndim == 3:
        # Heuristic: assume smallest dim is bands.
        dims = arr.shape
        i = int(np.argmin(dims))
        if i != 2:
            arr = np.moveaxis(arr, i, -1)
    return arr


def load_cube_file(path: str | Path) -> np.ndarray:
    """Load a single file and coerce to (H, W, B)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {p}")
    suf = p.suffix.lower()
    if suf == ".npy":
        raw = _load_npy(p)
    elif suf == ".npz":
        raw = _load_npz(p)
    elif suf == ".mat":
        raw = _load_mat(p)
    elif suf in (".tif", ".tiff"):
        raw = _load_tiff(p)
    elif suf == ".hdr":
        try:
            import spectral.io.envi as envi
            img = envi.open(str(p))
            raw = np.asarray(img.load())
        except ImportError as e:
            raise ImportError("ENVI .hdr requires the `spectral` package: pip install spectral") from e
    else:
        raise ValueError(f"Unsupported file extension '{suf}' for {p}. Supported: {sorted(CUBE_CANDIDATE_SUFFIXES)}")
    return _as_cube(raw)


def _is_gt_file(path: Path) -> bool:
    n = path.name.lower()
    return any(h in n for h in GT_NAME_HINTS)


def _load_labels_file(path: Path, shape_hw: tuple[int, int] | None = None) -> Optional[np.ndarray]:
    try:
        suf = path.suffix.lower()
        if suf == ".npy":
            a = np.load(str(path))
        elif suf == ".npz":
            a = _load_npz(path)
        elif suf == ".mat":
            a = _load_mat(path)
        else:
            return None
        a = np.asarray(a)
        if a.ndim == 3 and a.shape[2] == 1:
            a = a[:, :, 0]
        if shape_hw is not None and a.shape != tuple(shape_hw):
            # Allow flattened labels.
            if a.size == shape_hw[0] * shape_hw[1]:
                a = a.reshape(shape_hw)
            else:
                log.warning("Label file %s shape %s mismatches cube %s; keeping anyway.", path.name, a.shape, shape_hw)
        return a
    except Exception as e:
        log.warning("Could not load labels from %s: %s", path, e)
        return None


def load_hyperspectral_data(path: str | Path, dataset_name: str = "Indian Pines") -> HyperspectralDataset:
    """Main entry point. Accepts a file or directory. Returns HyperspectralDataset.

    Directory logic:
      1. Collect candidate cube files (ignoring gt/label/mask files).
      2. Prefer names containing 'indian', 'pine', 'array', 'cube', 'data', 'hsi'.
      3. Load the largest candidate as the cube.
      4. Load any gt-like file with matching spatial dims as labels.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Dataset path not found: {p}. "
            "Place indianpinearray.npy next to IPgt.npy in data/raw/ "
            "or pass --input path/to/indianpinearray.npy"
        )
    labels = None
    labels_source = ""
    cube_source = ""
    dtype_original = ""
    metadata: dict[str, Any] = {}

    if p.is_file():
        cube = load_cube_file(p)
        dtype_original = str(np.asarray(cube).dtype)
        cube_source = str(p)
        # Look for sibling gt file.
        for sib in p.parent.iterdir():
            if sib.is_file() and sib != p and _is_gt_file(sib):
                lab = _load_labels_file(sib, (cube.shape[0], cube.shape[1]))
                if lab is not None:
                    labels, labels_source = lab, str(sib)
                    break
    else:
        files = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in CUBE_CANDIDATE_SUFFIXES]
        if not files:
            raise FileNotFoundError(f"No dataset files in {p}. Expected .npy/.npz/.mat/.tif, got: {[f.name for f in p.iterdir()]}")
        non_gt = [f for f in files if not _is_gt_file(f)]
        pool = non_gt or files

        def score(f: Path) -> tuple[int, int]:
            n = f.name.lower()
            keywords = ("indian", "pine", "array", "cube", "data", "hsi", "scene", "image")
            kw = sum(k in n for k in keywords)
            try:
                sz = f.stat().st_size
            except OSError:
                sz = 0
            return (kw, sz)

        pool.sort(key=score, reverse=True)
        cube_file = pool[0]
        cube = load_cube_file(cube_file)
        cube_source = str(cube_file)
        dtype_original = str(np.asarray(cube).dtype)
        metadata["candidates"] = [f.name for f in pool]
        for f in files:
            if _is_gt_file(f):
                lab = _load_labels_file(f, (cube.shape[0], cube.shape[1]))
                if lab is not None:
                    labels, labels_source = lab, str(f)
                    break

    if cube.ndim != 3 or cube.shape[2] < 2:
        raise ValueError(f"Loaded cube has invalid shape {cube.shape}; need (H, W, B>=2)")
    if not np.issubdtype(cube.dtype, np.number):
        raise ValueError(f"Cube dtype {cube.dtype} is not numeric")

    log.info("Loading dataset: %s", cube_source)
    log.info("Dataset shape (H, W, B): %s", tuple(cube.shape))
    ds = HyperspectralDataset(
        cube=np.asarray(cube),
        name=dataset_name,
        source_path=cube_source,
        dtype_original=dtype_original or str(cube.dtype),
        labels=None if labels is None else np.asarray(labels),
        labels_source=labels_source,
        metadata=metadata,
    )
    return ds


def band_statistics(cube: np.ndarray, band: int) -> dict:
    """Min/max/mean/std of one band (0-indexed)."""
    if not (0 <= band < cube.shape[2]):
        raise ValueError(f"band {band} out of range [0, {cube.shape[2]})")
    b = np.asarray(cube[:, :, band], dtype=np.float64)
    return {
        "band": band,
        "min": float(np.min(b)),
        "max": float(np.max(b)),
        "mean": float(np.mean(b)),
        "std": float(np.std(b)),
    }
