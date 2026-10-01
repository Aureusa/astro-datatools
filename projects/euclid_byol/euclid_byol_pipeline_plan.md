# Euclid Q1 → BYOL Pretraining Pipeline — Build Plan

## Context
Building a data pipeline that pulls galaxy image cutouts from the Euclid Q1 archive, preprocesses them consistently, and stores them as a ML-ready uint8 dataset for BYOL self-supervised pretraining. User already has authenticated access to the ESA Euclid Science Archive. Target scale: up to ~26M Q1 detections. Storage budget: ~1 TB.

**Non-negotiable constraint: the preprocessing function (stretch + normalization + quantization) must be frozen and applied identically to every image — pretraining set, fine-tuning set, and any future/test images. Any drift here silently reintroduces distribution shift.**

---

## 1. Data access & source

- [ ] Confirm access credentials for the ESA Euclid Science Archive (EAS): https://eas.esac.esa.int/sas
- [ ] Install/verify `astroquery.esa.euclid` (Python package) for programmatic access — do not rely on the web GUI for bulk work
- [ ] Identify the MER (merged) catalogue tables via ADQL (Q1 has ~26M detections, IRSA/EAS both host copies)
- [ ] Reference data product docs: Euclid Q1 Archive User Guide (IRSA) and EAS SAS User Manual, for MER mosaic / cutout API details
- [ ] Decide primary archive: ESA EAS (`eas.esac.esa.int/sas`) since user has direct access there; note IRSA and AWS Open Data (`registry.opendata.aws/euclid-q1`) as fallback/mirror if EAS rate-limits or is slow for bulk cutouts
- [ ] Note: full Q1 release ≈ 30 TB of raw tile mosaics — do NOT download full tiles; use the cutout service per-object instead

## 2. Catalogue query stage

- [ ] Write ADQL query against the MER final catalogue to pull object IDs + RA/Dec (+ any useful flags: photometry, extent/area, non-spurious/non-stellar flags) for all Q1 detections
- [ ] Paginate/batch the query (26M rows — do not pull in one request; chunk by tile ID or by row ranges)
- [ ] Store the resulting object list (IDs + coordinates + flags) locally as the master index table (e.g. parquet) — this drives everything downstream
- [ ] Optional filtering pass: decide whether to exclude spurious/stellar detections at this stage (recommended, to avoid wasting storage/compute on non-galaxy sources)

## 3. Cutout download stage

- [ ] Use the Euclid SAS cutout service (`sas-cutout/cutout?...POS=CIRCLE,ra,dec,radius`) or the astroquery cutout method to fetch a per-object VIS image cutout
- [ ] Cutout size: **224×224 pixels** (matches Zoobot's training resolution — keep pretraining and any Zoobot-feature comparison on equal footing; do NOT start at 64×64 and rescale later — this changes effective object scale/receptive field and degrades fine morphology like spiral arms/bars)
- [ ] Build a batched/parallelized downloader (async or multiprocessing) against the cutout endpoint — respect archive rate limits, add retry/backoff logic
- [ ] Download once, persist to local disk immediately — do NOT stream cutouts on-the-fly per training batch and discard (network fetch latency will dominate and repeat every epoch; this was explicitly ruled out)
- [ ] Track download progress/failures in a manifest file (object ID → local path → status) so the job is resumable
- [ ] Sanity-check a sample batch of cutouts visually before scaling to full run

## 4. Preprocessing stage (fixed, versioned function — reused everywhere downstream)

- [ ] Implement a single preprocessing function, e.g. `preprocess(raw_cutout) -> uint8_array`, covering:
  1. **Background/noise handling**: estimate local noise level per cutout (or per tile) if not already background-subtracted
  2. **Asinh stretch**: apply an asinh (arcsinh) stretch rather than linear scaling — linear near zero to preserve faint outskirts/background, compresses bright cores so they don't saturate. Use `astropy.visualization` (`AsinhStretch` + `ImageNormalize`) rather than hand-rolling it
  3. **Normalization**: min-max normalize the stretched result into [0, 1]
  4. **Quantization**: rescale to [0, 255] and cast to `uint8`
- [ ] Choose and fix the stretch parameter (`a` in asinh) — either a single global value tuned on a representative sample, or a per-image value derived from local noise; whichever is chosen, document it and never change it silently later
- [ ] Version this function explicitly (e.g. `preprocess_v1.py`) and never edit it in place once data has been generated with it — any change means regenerating the whole dataset, or creating `v2` clearly labeled and never mixing v1/v2 samples in one training run
- [ ] Unit test: verify round-trip visually (stretched uint8 image should still show visible spiral arms/bars on a known galaxy) and check that faint outskirts aren't clipped to 0 or bright cores aren't all clipped to 255
- [ ] Apply this exact function to every cutout as part of the download pipeline (or as an immediate post-download pass) — output stored directly as uint8, not float32 intermediate files kept around at scale

## 5. Storage / dataset format

- [ ] Store final uint8 224×224 single-band arrays in an ML-friendly format (e.g. HDF5, WebDataset shards, or a HuggingFace `datasets`-style parquet+arrow layout — pick one compatible with the eventual BYOL training framework)
- [ ] Storage estimate at this spec: ~1.2 TB for all ~26M detections (uint8, 224×224, single band) — confirms fit within the 1 TB budget if some filtering (e.g. dropping spurious/stellar detections) is applied, or use a subset
- [ ] Keep the master index table (Section 2) linked to storage shard/file locations for reproducibility and later cross-matching
- [ ] Do not keep per-object raw FITS cutouts around after preprocessing unless disk space allows — only if a float32 fallback subset is wanted for tasks needing full dynamic range (e.g. photometry checks), keep that as a small separate sample, not the bulk set

## 6. Cross-match for fine-tuning/eval labels (downstream, not blocking pretraining)

- [ ] Download the Walmsley et al. Euclid Q1 "First Visual Morphology Catalogue" (Zenodo, parquet/CSV; arXiv:2503.15310) — 378,000 labeled galaxies (bars, spiral arms, mergers)
- [ ] Match label catalogue IDs against the master index table (should be direct ID overlap since both derive from Q1 MER)
- [ ] For DECaLS/DESI cross-match (optional additional labels): cross-match on RA/Dec within the known Q1–DESI sky overlap region (~63 deg²) using a small match radius (~1″, consistent with prior Euclid–DESI cross-match work)
- [ ] Keep this labeled subset flagged in the index table so it can be pulled out cleanly for fine-tuning/eval without touching the unlabeled pretraining pool

## 7. Pipeline orchestration

- [ ] Structure as discrete, resumable stages: `query_catalogue.py` → `download_cutouts.py` → `preprocess.py` → `build_dataset_shards.py` → `crossmatch_labels.py`
- [ ] Each stage reads/writes to the manifest so a failed run can resume rather than restart
- [ ] Log counts at each stage (queried, downloaded, failed, preprocessed, final dataset size) to catch silent drop-off
- [ ] Add a config file (resolution, stretch parameter, storage format, output paths) so the whole pipeline is reproducible from one source of truth rather than hardcoded constants scattered across scripts

## Open decisions to confirm before/while building
- [ ] Exact stretch parameter value / estimation method for asinh (global vs per-tile vs per-object)
- [ ] Whether to filter spurious/stellar detections before download (recommended: yes)
- [ ] Final storage format choice (HDF5 vs WebDataset vs Arrow/parquet) — depends on the training framework
- [ ] Primary vs fallback archive endpoint (EAS vs IRSA vs AWS) if download throughput becomes a bottleneck
