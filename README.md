# Principles of Deep Learning

Extracts per-transformer-block activations from EVA-02-Base across a new, larger
image dataset (~73,000 images) - unlike the pooled single-vector features used in
the [Statistical-programming](https://github.com/supernovahorvat/Statistical-programming)
project, this looks at every transformer block's output, not just the final one.

## Status: early scaffolding, work in progress

Full-resolution, every-block activations for 73k images is a lot of data and
memory. Before writing the extraction pipeline, the next step is to inspect the
new dataset and work out a sensible subsetting/storage strategy (e.g. a smaller
image subset, or pooling/downsampling per block) that keeps this tractable.

## Notebooks

Carried over from the Statistical-programming project as a starting point -
outputs cleared, not yet re-run or adapted to the new dataset:

1. `01_model_reference.ipynb` - EVA-02-Base loading and inference reference.
2. `02_data_inspection.ipynb` - dataset fetch/inspection template (currently
   still wired to the old objectome dataset - needs pointing at the new one).
3. `03_export_pipeline.ipynb` - feature-extraction/export template (currently
   extracts a single pooled 768-d vector per image - needs extending to
   per-block activations, and to the subsetting decision above).

## Setup

```
pip install -r requirements.txt
```
