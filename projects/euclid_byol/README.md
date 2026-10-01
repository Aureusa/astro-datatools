# Euclid Q1 → BYOL pretraining dataset

Pipeline that turns the Euclid Q1 MER catalogue into a uint8, 224×224, single-band (VIS)
image dataset for BYOL self-supervised pretraining, following
[euclid_byol_pipeline_plan.md](euclid_byol_pipeline_plan.md).

The one rule: **every image, whether used for pretraining, fine-tuning or testing, goes through
the same frozen preprocessing function** ([preprocess_v1.py](preprocess_v1.py)). Its SHA-256
fingerprint is stored in every tile file and shard. The pipeline refuses to mix fingerprints, and
a test pins its exact output bytes.

## Stages

Run from the repository root with the `astro_datatools` environment. Every stage reads the same
config, resumes where it stopped, logs to `<root>/logs/<stage>.log` and writes its counts to
`<root>/stats/<stage>.json`.

```bash
C=projects/euclid_byol/configs/pipeline.yaml   # or configs/sample.yaml (100 objects, separate root)
python -m projects.euclid_byol.pipelines.estimate_storage     --config $C   # disk space of the run; writes nothing
python -m projects.euclid_byol.pipelines.query_catalogue      --config $C
python -m projects.euclid_byol.pipelines.download_cutouts     --config $C [--dry-run] [--part I --num-parts N]
python -m projects.euclid_byol.pipelines.preprocess verify    --config $C
python -m projects.euclid_byol.pipelines.build_dataset_shards --config $C
python -m projects.euclid_byol.pipelines.crossmatch_labels    --config $C
python -m projects.euclid_byol.pipelines.status               --config $C
python -m projects.euclid_byol.pipelines.inspect_cutouts      --config $C   # figures in <root>/inspect/
```

| Stage | What it does | Output under `paths.root` |
|---|---|---|
| `estimate_storage` | Counts the objects the configured selection would download, tile by tile, applying the same selection code as `query_catalogue`. Reports the disk space for images, raw subset and metadata, peak use, and extra size cuts. Writes nothing. | nothing (optionally `--json FILE`) |
| `query_catalogue` | Maps the 352 Q1 MER tiles to their VIS mosaics and catalogue files. Pulls `catalogue.mer_catalogue` one tile per ADQL query and checks each tile's row count. Marks `selected` objects using `catalogue.selection`. | `catalogue/tiles.parquet`, `catalogue/parts/tile_<t>.parquet`, `catalogue/master_index.parquet` |
| `download_cutouts` | Fetches VIS cutouts from the SAS cutout service with threads, a global rate limit and retry/backoff. Crops exactly 224 px around each source and applies `preprocess_v1` straight away, storing only uint8. A small raw float32 subset is kept. Every object has a status row. | `cutouts/tile_<t>.h5`, `raw_subset/tile_<t>.h5` |
| `preprocess verify` | Checks tile fingerprints, and that `preprocess_v1(raw)` reproduces every stored raw-subset image bit for bit. | `stats/preprocess.json` |
| `preprocess apply` | Runs the same function on any FITS files (fine-tuning or test images). | the given HDF5 file |
| `build_dataset_shards` | Appends completed tiles to fixed-size shards. Incremental, and truncates half-written appends. With `delete_tile_files`, replaces each sharded tile file with a small status parquet. | `dataset/shard_<n>.h5`, `dataset/index.parquet`, `cutouts/tile_<t>.status.parquet` |
| `crossmatch_labels` | Matches the Walmsley et al. (2025) Q1 morphology catalogue by `object_id` (position as fallback), and optionally DESI labels by position (1″). Flags labelled objects in the index. | `labels/*_labelled.parquet`, `label_*` columns in the index |

`dataset/index.parquet` links each image to its `shard`/`shard_row`, its tile, its preprocessing
statistics (`noise_sigma`, `vmax`, `asinh_a`, `blank_fraction`) and the catalogue columns. See
[Loading data for training](#loading-data-for-training) for the readers.

For the full run, submit [scripts/download_q1.slurm](scripts/download_q1.slurm) from the repository
root: `sbatch projects/euclid_byol/scripts/download_q1.slurm`. It runs every stage on the
`gpu_strw` partition (CPU only, 8 cores, 16 GB), with three download passes so that transient
failures are retried. It writes to `$HOME/project_data/q1_seg_area_100` and logs to
`projects/euclid_byol/slurm_output/`. If the job stops, submit it again: every stage resumes where
it left off.

## Disk space

**Check before downloading.** `estimate_storage` writes nothing to disk. It counts the objects the
configured selection would download, tile by tile (~10 min for all of Q1), and prints the space
needed:

```bash
python -m projects.euclid_byol.pipelines.estimate_storage --config projects/euclid_byol/configs/pipeline.yaml
```

For the current default selection (VIS-detected, not spurious, not stars,
`segmentation_area >= 100`), the archive counts 4,247,297 objects: ~0.17 TB of gzip images, ~6 GB
of metadata, ~4 GB of raw subset. Result with only the first three cuts (no size or star-probability
cuts), all 352 Q1 tiles, 2026-09:

```
Tiles: 352   detections: 29,953,430   selected: 20,720,208 (69.2%)
uint8 images in shards (gzip):  0.826 TB   [uncompressed 1.046 TB]
raw float32 subset (fraction 0.001):  4.15 GB   [0.0001: 0.40 GB, 0.01: 41.80 GB]
catalogue + index metadata:  11.6 GB
delete_tile_files = True, tile files gzip:  peak 0.846 TB, after the run 0.842 TB

Extra cut on top of the selection (uncompressed images):
  segmentation_area >=   25:   17,624,573 objects  0.890 TB
  segmentation_area >=   50:   10,312,763 objects  0.521 TB
  segmentation_area >=  100:    4,718,853 objects  0.238 TB
  segmentation_area >=  200:    2,261,057 objects  0.114 TB
```

**What takes space**, per object (measured on Q1 cutouts):

| Item | Size | Setting |
|---|---|---|
| uint8 image | 50.2 kB uncompressed, ~40 kB gzip | `cutouts.compression`, `dataset.compression` |
| raw float32 cutout (subset only) | 201 kB | `cutouts.raw_subset_fraction` |
| catalogue row (all detections) | ~90 B, stored twice (parts + master index) | `catalogue.columns` |
| dataset index row | ~150 B, stored twice (index parts + index) | |

**Space-saving defaults** in `configs/pipeline.yaml`:

- `compression: gzip` for tile files and shards. It is lossless, saves ~21%, and decompressing a
  40 kB image costs well under a millisecond at training time.
- `delete_tile_files: true`. Once a tile's images are in the shards, its tile file is replaced by
  `cutouts/tile_<t>.status.parquet` (per-object status, no images). Images are then never stored
  twice (with `false`, peak disk roughly doubles), and re-running `download_cutouts` skips those
  tiles.
- `raw_subset_fraction: 0.001`, about 21k raw float32 cutouts (~4 GB) used by
  `preprocess verify`. Set `0.0001` (~0.4 GB) to go smaller, or `0` to disable it; `verify` then
  only checks fingerprints.

**Running with limited disk:**

1. `estimate_storage`. If the peak is too large, add a cut to `catalogue.selection` (for example
   `segmentation_area >= 50`) and estimate again.
2. `query_catalogue` writes ~5 GB of catalogue metadata.
3. `download_cutouts`, in one process (`--part I --num-parts N` splits the tiles across processes).
4. `build_dataset_shards`. This can run at any time, as often as you like: it adds completed tiles
   and deletes their tile files. Because the tile files are compressed too, the order does not
   change the peak.
5. `preprocess verify`, then `crossmatch_labels`.

`status` reports the actual disk use per directory (`disk_gb`), so you can watch it against the
estimate during the run.

## Download time

Throughput of the SAS cutout service, measured on 2026-09-27 from node861, counting only
correct cutouts:

| requests in flight (`workers`) | correct cutouts/s | 4.25M (default selection) | 20.7M (no size cut) |
|---|---|---|---|
| 16 | 78 | ~15 h | ~3 days |
| **32 (default)** | 94 | **~13 h** | ~2.5 days |
| 64 | 99 | ~12 h | ~2.4 days |

The service's speed varies: earlier the same day, 16 requests in flight gave only ~17/s. At that
speed, the default selection would take ~3 days. Past ~32 in flight there is little gain, and the
archive is a shared service, so the default is 32 workers with a 150 requests/s safety cap
(`cutouts.max_requests_per_second`). The logs report the rate as the download runs.

**Misplaced cutouts.** Under concurrent load, the service answers 15–20% of requests with the
cutout of another, simultaneous request. The file is internally consistent (header and pixels
belong together) but is centred on the wrong position. The download stage checks every response
against its own WCS and re-requests it if the source is more than `max_centre_offset_pix` (2 px)
from the centre. Correct cutouts are always within 0.64 px. The logs report the count as
"misplaced responses re-requested". Tested on 1,120 objects: every object ended up with its own
cutout, and the 100-object sample matches a one-at-a-time re-fetch exactly.

## Loading data for training

Shards are gzip HDF5 with one chunk per image. Measured on node861:

| Access | Speed |
|---|---|
| NFS (`/zfsstore`), random single images | **~44 images/s**: never do this |
| NFS, blocks of 256 consecutive images | ~2,900 images/s (NFS link ~115 MB/s) |
| node-local `/scratchdata/<user>/<jobid>`, sequential | 2.6 GB/s |
| gzip decode, h5py (global lock) | ~5k images/s per process |
| gzip decode, raw chunks + zlib in threads (used by the readers) | ~23k images/s per process |

[dataset.py](dataset.py) has three readers. All work as PyTorch datasets without importing torch.

**Problem cutouts are skipped by default** (`exclude_flagged=True` on every reader): objects whose
`det_quality_flag` has bit 128 or 256 set, which marks detections near saturated stars, ghosts and
other artefacts. Checked against the Galaxy Zoo Euclid labels, this removes 60% of the objects
labelled "problem" and 1.3% of clean galaxies. On the Q1 dataset it drops 142,735 of 4,304,524
images, leaving 4,161,789 (3,797,775 of them unlabelled). Pass `exclude_flagged=False` to get every
downloaded image.

- **`EuclidBlockStream`** (iterable) streams straight from NFS. It shuffles blocks of 256
  consecutive images, splits them over DataLoader workers and distributed ranks, and shuffles
  again in a buffer. Each image is yielded once per epoch; call `set_epoch(e)` to reshuffle.
  About 2.9k images/s is well above what a T4 trains BYOL at.
- **`EuclidShardDataset`** (map-style, random access) is for data on local disk. Copy the shards
  to the job's scratch directory first; at ~115 MB/s this takes ~2 h for 0.83 TB:
  `cp <root>/dataset/{index.parquet,shard_*.h5} /scratchdata/$USER/$SLURM_JOB_ID/`.
- **`load_into_memory`** loads a subset into one uint8 array, e.g. the labelled set for
  fine-tuning (`only=["label_morphology"]`, ~380k images = 19 GB). node861 has 62 GB RAM, so the
  full dataset (~1 TB uncompressed) does not fit; the function refuses to allocate more than 80%
  of free RAM.

```python
from torch.utils.data import DataLoader
from projects.euclid_byol.dataset import EuclidBlockStream, load_into_memory

stream = EuclidBlockStream(root / "dataset", exclude=["label_morphology"], decode_threads=2)
loader = DataLoader(stream, batch_size=256, num_workers=8)
for epoch in range(n_epochs):
    stream.set_epoch(epoch)   # before creating the iterator; workers get a copy
    for images, object_ids in loader:
        ...

images, index = load_into_memory(root / "dataset", only=["label_morphology"])
```

## Decisions taken

- **Archive:** ESA EAS. Q1 cutouts and catalogue queries work anonymously. Set
  `archive.credentials_file` if a login is needed.
- **Tiles as the unit of work:** each MER tile has one VIS mosaic and one catalogue file, so the
  catalogue query, download, resumption and sharding all work per tile (~85k detections each).
- **Selection:** VIS-detected (`vis_det == 1`), not spurious, and resolved
  (`segmentation_area >= 100` px), which drops the noise-dominated blobs. Stars are removed three
  ways, because `point_like_flag` is only set for clean detections and misses saturated stars:
  `point_like_flag != 1`, `~(point_like_prob > 0.5)`, and `~(mumax_minus_mag < -2.6)` (the stellar
  locus: Gaia-confirmed stars sit at −3.3 to −2.7, large galaxies at −1.75 to +1.9). This keeps
  4.25M of the 30.0M detections. Counts for other size cuts are in `configs/pipeline.yaml`. To
  change the selection, run `query_catalogue --reselect`; no re-query is needed.
- **Stretch (v1):** per-image and noise-adaptive. `sigma` is the sigma-clipped MAD of the cutout.
  The asinh softening `a` is 3 σ, the lower bound is −2 σ, and the upper bound is the image
  maximum (min-max). No local background is subtracted, because the MER mosaics are already
  background-subtracted. Details are in the [preprocess_v1.py](preprocess_v1.py) docstring.
- **Storage:** gzip-compressed HDF5 shards of 100k images, chunked per image, plus a parquet
  index. This matches the other projects in this repository and allows random access from
  PyTorch.

## Changing the preprocessing

Never edit `preprocess_v1.py` once a dataset exists. Copy it to `preprocess_v2.py`, change
`VERSION`, register it in `config.PREPROCESSORS`, set `preprocess.version: v2`, and generate into
a new `paths.root`. The regression test's pinned hash is only re-pinned for a version that has
not produced data yet.
