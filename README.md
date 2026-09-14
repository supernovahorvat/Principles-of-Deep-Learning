# Principles of Deep Learning: EVA-02 Model Inspection and Multi-Dataset Activation Extraction

Inspects the architecture of the EVA-02-Base vision transformer, analyzes two
unrelated image datasets - a behavioral object-recognition dataset and a
subset of the Natural Scenes Dataset (NSD) selected for faces vs. places -
and extracts EVA-02 activations for both. This project only covers
architecture, data, selection, and extraction; scoring/analysis of the
behavioral data against human responses happens in a separate repository
(`Statistical-programming`), not here.

## Notebooks (run in this order)

1. **`01_EVA-02_model_inspection.ipynb`** - loads EVA-02-Base and documents its
   architecture: 12 SwiGLU-gated transformer blocks, rotary position
   embeddings, patch/CLS token counts, and pooling behaviour. Feeds nothing
   downstream programmatically - notebooks 03 and 06 each reload the model
   independently - but is the architectural reference the other notebooks'
   markdown cells point back to.
2. **`02_behavioral_data_inspection.ipynb`** - fetches and SHA1-verifies the
   public `dicarlo.Rajalingham2018.public` behavioral dataset directly from
   S3, and explores its structure (trial data, stimulus set, image format).
3. **`03_behavioral_export_pipeline.ipynb`** - extracts EVA-02's pooled 768-d
   feature vector for all 2,160 behavioral images, selects the 240-image
   held-out test set, and writes `features.parquet`, `images.parquet`,
   `human_trials.parquet`, and `export_info.json` to `data/`.
4. **`04_nsd_data_inspection.ipynb`** - fetches NSD's stimulus metadata (COCO
   id, crop info, `shared1000` flag) from NSD's public S3 bucket and explores
   category composition, laying the groundwork for selecting a face/place
   subset.
5. **`05_nsd_face_place_selection.ipynb`** - selects a balanced set of 900
   face images and 900 place images from NSD's full ~73k-image pool, fully
   automated: an image is labeled **face** if the YuNet detector finds a face
   in it; it's labeled **place** if YuNet finds *no* face *and* a
   Places365-trained ResNet-18 scores its top scene category above 0.5
   confidence (all three model assets download automatically on first run -
   see **Models** below for which does what). Saves the chosen images to
   `stimulus_subset/` and their metadata to
   `data/nsd_face_place_stimulus_subset.csv` - both already committed in this
   repo (see **What's included** below), so re-running this notebook only
   matters if you want to redo or verify the selection yourself.
6. **`06_nsd_export_pipeline.ipynb`** - runs every selected image through
   EVA-02-Base and captures the output of all 12 transformer blocks (not just
   a pooled vector - the full per-patch-token activations), saving two
   `(900, 12, 768, 1024)` HDF5 files to `activations/` (faces and places,
   ~15.8GB each). Not included in this repo - see **Regenerating
   activations** below.

## Setup

```
pip install -r requirements.txt
```

Every image dataset here is fetched automatically over the network on first
run - no manual downloads:
- Notebook 02's behavioral images come from a public DiCarlo-lab S3 bucket
  and are cached in `~/.brainio_manual` afterward.
- Notebooks 04/05's NSD metadata comes from a public S3 bucket
  (`natural-scenes-dataset.s3.amazonaws.com`); the actual images come from
  that bucket (for the `shared1000` subset) or directly from
  `images.cocodataset.org` otherwise.

This only matters if you actually run notebooks 02/04/05 yourself. The 1,800
NSD images notebook 05 selects are **already committed in this repo** as
real image files under `stimulus_subset/` (see **What's included** below) -
not just a list of NSD image IDs - so notebook 06 (and anything else that
just needs the selected images) needs no network access and no re-selection.

An internet connection is required the first time each of notebooks 02, 04,
and 05 runs.

### Models

You only need these if you want to **reselect** the face/place images
yourself - if you're just using the data already in this repo
(`stimulus_subset/`), skip this section entirely. The models:

- **YuNet** - face selection: any detection labels the image a face.
- **Places365 (ResNet-18)** - place selection: scores 365 scene categories
  for whatever YuNet didn't flag as a face; `categories_places365.txt` is
  just that classifier's label list, not a model itself.

All three auto-download into `models/` (gitignored) the first time notebook
05 runs:

| File | Size | Source |
|---|---|---|
| `face_detection_yunet_2023mar.onnx` | 232KB | [OpenCV Zoo](https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx) |
| `resnet18_places365.pth.tar` | 45MB | [places2.csail.mit.edu](http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar) |
| `categories_places365.txt` | 7KB | [CSAILVision/places365](https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt) |

If the auto-download fails (e.g. behind a firewall), download each URL to
the listed filename under `models/` manually.

### Hardware

A CUDA GPU is strongly recommended for notebooks 03 and 06 (activation
extraction). On an RTX 3090, notebook 06 processes 900 images per category
in about a minute; CPU-only will work but be substantially slower.
Notebook 03 checkpoints its extraction (see below), so an interrupted CPU
run can resume rather than restart.

## What's included vs. regenerated locally

**Included directly in this repo** (small enough to commit):
- `data/features.parquet`, `human_trials.parquet`, `images.parquet`,
  `export_info.json` - behavioral dataset outputs from notebook 03
- `data/nsd_face_place_stimulus_subset.csv` - the NSD selection table from
  notebook 05
- `stimulus_subset/` - the 1,800 actual selected NSD images (900 faces + 900
  places, ~560MB)

**Not included, regenerate locally by running the corresponding notebook:**
- `activations/` - the full per-block NSD activations (~31GB total); run
  notebook 06
- `models/` - the three small selection-time assets; auto-downloaded by
  notebook 05
- `checkpoints/` - notebook 03's resumable feature-extraction cache;
  auto-created on first run (the actual deliverable, `features.parquet`, is
  already tracked, so this is only needed if you want to re-run extraction)

## References

- Fang, Y., Sun, Q., Wang, X., Huang, T., Wang, X., & Cao, Y. (2023). EVA-02:
  A visual representation for Neon Genesis. *arXiv*.
  https://doi.org/10.48550/arXiv.2303.11331
- Rajalingham, R., Issa, E. B., Bashivan, P., Kar, K., Schmidt, K., &
  DiCarlo, J. J. (2018). Large-scale, high-resolution comparison of the core
  visual object recognition behavior of humans, monkeys, and
  state-of-the-art deep artificial neural networks. *Journal of
  Neuroscience*, 38(33), 7255–7269.
  https://doi.org/10.1523/JNEUROSCI.0388-18.2018
- Schrimpf, M., Kubilius, J., Hong, H., Majaj, N. J., Rajalingham, R., Issa,
  E. B., Kar, K., Bashivan, P., Prescott-Roy, J., Schmidt, K., Yamins,
  D. L. K., & DiCarlo, J. J. (2018). Brain-Score: Which artificial neural
  network for object recognition is most brain-like? *bioRxiv*.
  https://doi.org/10.1101/407007
- Allen, E. J., St-Yves, G., Wu, Y., Breedlove, J. L., Prince, J. S., Dowdle,
  L. T., Nau, M., Caron, B., Pestilli, F., Charest, I., Hutchinson, J. B.,
  Naselaris, T., & Kay, K. (2022). A massive 7T fMRI dataset to bridge
  cognitive neuroscience and artificial intelligence. *Nature Neuroscience*,
  25, 116–126.
  https://doi.org/10.1038/s41593-021-00962-x
- YuNet face detector. OpenCV Zoo.
  https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
- ResNet-18 Places365 scene classifier. CSAILVision.
  https://github.com/CSAILVision/places365

The code in this repository was co-authored with Claude (Anthropic).
