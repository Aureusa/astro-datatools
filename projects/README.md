# RaCUN dataset generation project

This project contains the dataset-building pipeline used to prepare the training data for the RaCUN models developed in the following research work: https://github.com/Aureusa/racun

The code here is not the model training code itself; instead, it focuses on producing the curated astronomical dataset used to train those models. The workflow combines radio astronomy source catalogues, cutout generation (via adjacent library `strw_lofar_data_utils`; github: https://github.com/Aureusa/strw_lofar_data_utils), augmentation, component grouping, and COCO-style annotation export.

This project also serves as a practical example of using the `astro-datatools` library for astronomy-focused data preparation and ML pipeline construction.

---

## Overview

The dataset generation process follows these steps:

1. Load source catalogues and component metadata.
2. Select the relevant radio sources and positions.
3. Generate radio cutouts from the LOFAR data products.
4. Apply astronomy-oriented augmentations and rotations.
5. Build candidate source/component proposals.
6. Export the result as a structured dataset for training object-detection or segmentation models.

This is implemented in the `racun` project package under this folder and is designed to work with the supporting utilities and data abstractions from `astro-datatools`.

---

## Repository structure

- `configs/` — YAML configuration files for dataset generation pipelines.
- `pipelines/dataset_pipeline.py` — main pipeline that assembles the training dataset.
- `annotations/` — utilities for annotation enrichment and proposal precomputation.
- `augment/` — astronomy-specific augmentation logic, including LOFAR-to-RGBA conversions.
- `coco/` — COCO dataset structures and builders used for annotation export.
- `scripts/` — helper shell scripts for preparing train/validation/test splits.
- `utils/` — dataset utilities and splitting tools.

---

## Main pipeline

The core pipeline is defined in `pipelines/dataset_pipeline.py` and reads a configuration file such as:

- `configs/dataset_pipeline.yaml`
- `configs/dataset_pipeline_RGZ.yaml`

It performs the following actions:

- loads the source catalogue and component catalogue;
- filters valid sources for the dataset;
- generates cutouts around the selected coordinates;
- rotates and crops cutouts according to configured angles and sizes;
- builds a multi-class or binary dataset using the configured labels;
- saves the resulting dataset to the configured output directory.

The pipeline is highly configurable via YAML, including:

- image cutout size and crop dimensions;
- RMS/threshold settings;
- augmentation rotation angles;
- maximum precomputed islands/proposals;
- class setting and labels;
- parallel worker settings and dataset limits.

---

## Annotation and proposal generation

The `annotations/` package contains the logic for:

- generating proposal candidates from source components;
- grouping components by source and angle;
- creating metadata required for valid training instances;
- preparing annotations in a format suitable for downstream detection training.

This stage is essential because the dataset is not only image data, but structured source/component information linked to the astronomy objects of interest.

---

## Use of astro-datatools

This project demonstrates how `astro-datatools` can be used as a reusable data foundation for astronomy ML workloads.

In particular, it uses the library for:

- working with astronomy data products in a modular way;
- applying astronomical image augmentations;
- keeping the dataset pipeline distinct from the model implementation itself.

In other words, this project is a concrete example of `astro-datatools` in a real research-oriented machine-learning context.

---

## Typical workflow

From the repository root, a typical dataset-generation run looks like this:

```bash
python projects/racun/pipelines/dataset_pipeline.py \
  --config projects/racun/configs/dataset_pipeline.yaml
```

This will generate the dataset under the directory configured in the YAML file.

---

## Related work

This dataset pipeline supports the RaCUN research project:

- https://github.com/Aureusa/racun


---
