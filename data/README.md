# Data

## Supplied dataset (already present)

```
data/raw/
  indianpinearray.npy   # hyperspectral cube, (145, 145, 200), uint16
  IPgt.npy              # ground-truth labels, (145, 145), uint8, 0=background + 16 classes
  archive (2).zip       # original download bundle
```

`indianpinearray.npy` is the **corrected** Indian Pines (AVIRIS 224 bands with
water-absorption bands removed → 200). Ground truth is returned separately by
`load_hyperspectral_data()` and is **never** mixed into the PCA cube.

## Adding another dataset

Point `--input` / the GUI path box at any of:

- `.npy` — 3D `(H, W, B)` or square-flattened 2D `(H*W, B)`
- `.npz` — largest numeric array is used
- `.mat` — largest numeric variable is used (keys logged)
- `.tif` / `.tiff` — smallest axis assumed to be bands
- `.hdr` (ENVI) — requires `pip install spectral`
- a directory — largest non-`gt/label/mask` candidate is the cube,
  any `*gt*/*truth*/*label*/*mask*` sibling is attached as labels.

## Reproducing from Kaggle

Indian Pines downloads vary (raw 224-band vs corrected 200-band). After
downloading, place the cube + optional ground truth in `data/raw/` — the
loader detects dimensions dynamically, so either variant works.
