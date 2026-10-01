"""Stage 3: guard the frozen preprocessing, and apply it to other images.

Preprocessing itself runs inside the download stage (so float32 cutouts are
never stored at scale). This stage has two commands:

``verify`` (part of the main pipeline; run after downloading and/or sharding)
    * every tile file and shard must carry the fingerprint of the configured
      preprocessing module (no v1/v2 mixing, no silently edited function);
    * for every object in the raw float32 subset, re-running the preprocessing
      on the raw cutout must reproduce the stored uint8 image **bit for bit**
      (read from the tile file, or from the shards once the tile file has been
      deleted). Any mismatch means the function, or numpy/astropy underneath
      it, has drifted since the images were made.

``apply`` (for fine-tuning, evaluation and future images)
    Runs the same preprocessing on FITS files (e.g. cutouts of labelled
    galaxies obtained elsewhere) and writes an HDF5 file with the uint8
    images, the preprocessing fingerprint and per-image statistics::

        python -m projects.euclid_byol.pipelines.preprocess apply \\
            --inputs cutouts/*.fits --output finetune.h5 [--coords coords.csv]

    ``--coords`` is a CSV with ``file,ra,dec`` columns; without it each image
    is cut out around its centre.

Usage::

    python -m projects.euclid_byol.pipelines.preprocess verify --config projects/euclid_byol/configs/sample.yaml
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
import pandas as pd

from ..config import get_paths, get_preprocessor, load_config, stage_logger, write_stats
from ..tilestore import STATUS_OK, FingerprintMismatch, TileStore, check_fingerprint

STAGE = "preprocess"


def _stored_image_lookup(paths, tile_index: int, index: Optional[pd.DataFrame], bad: set, handles: dict):
    """Function object_id -> stored uint8 image (or None), reading the tile file,
    or the shards once the tile file has been deleted."""
    tile_path = paths.tile_file(tile_index)
    if tile_path.exists():
        if tile_path.name in bad:
            return None
        store = handles.setdefault(tile_path.name, TileStore.open(tile_path))
        row_of = {int(o): i for i, (o, s) in enumerate(zip(store.object_ids, store.status)) if s == STATUS_OK}
        return lambda oid: store.h5["images"][row_of[oid]] if oid in row_of else None
    if index is None:
        return None
    sub = index[index["tile_index"] == tile_index]
    where = {int(o): (s, int(r)) for o, s, r in zip(sub["object_id"], sub["shard"], sub["shard_row"]) if s not in bad}

    def get(oid):
        if oid not in where:
            return None
        shard, row = where[oid]
        h5 = handles.setdefault(shard, h5py.File(paths.dataset_dir / shard, "r"))
        return h5["images"][row]

    return get


def verify(config: dict, logger=None) -> Dict[str, object]:
    """Check fingerprints of tile files and shards, and re-derive images from the raw subset."""
    logger = logger or stage_logger(config, STAGE)
    paths = get_paths(config)
    prep = get_preprocessor(config)
    tile_files = sorted(paths.cutouts_dir.glob("tile_*.h5"))
    shard_files = sorted(paths.dataset_dir.glob("shard_*.h5"))
    logger.info("Verifying %d tile files and %d shards against preprocessing %s (fingerprint %s...)",
                len(tile_files), len(shard_files), prep.VERSION, prep.FINGERPRINT[:12])

    bad_fingerprint: List[str] = []
    for path in tile_files + shard_files:
        with h5py.File(path, "r") as h5:
            try:
                check_fingerprint(h5.attrs, prep.VERSION, prep.FINGERPRINT, str(path))
            except FingerprintMismatch as err:
                bad_fingerprint.append(path.name)
                logger.error("%s", err)

    index = None
    if paths.dataset_index.exists():
        index = pd.read_parquet(paths.dataset_index, columns=["object_id", "tile_index", "shard", "shard_row"])
    bad = set(bad_fingerprint)
    handles: dict = {}
    n_checked = n_mismatch = n_missing = 0
    max_abs_diff = 0
    try:
        for raw_path in sorted(paths.raw_dir.glob("tile_*.h5")):
            tile_index = int(raw_path.stem.split("_")[1])
            lookup = _stored_image_lookup(paths, tile_index, index, bad, handles)
            with h5py.File(raw_path, "r") as raw_h5:
                raw_ids = raw_h5["object_id"][:]
                for j, oid in enumerate(raw_ids):
                    stored = lookup(int(oid)) if lookup is not None else None
                    if stored is None:
                        n_missing += 1
                        continue
                    expected = prep.preprocess(raw_h5["raw"][j])
                    n_checked += 1
                    if not np.array_equal(expected, stored):
                        n_mismatch += 1
                        diff = int(np.abs(expected.astype(int) - stored.astype(int)).max())
                        max_abs_diff = max(max_abs_diff, diff)
                        if n_mismatch <= 10:
                            logger.error("Mismatch for object %d of tile %d (max |diff| %d)", oid, tile_index, diff)
    finally:
        for handle in handles.values():
            handle.close()

    ok = not bad_fingerprint and n_mismatch == 0
    stats = {
        "preprocess_version": prep.VERSION,
        "preprocess_fingerprint": prep.FINGERPRINT,
        "tile_files": len(tile_files),
        "shards": len(shard_files),
        "files_wrong_fingerprint": bad_fingerprint,
        "raw_subset_checked": n_checked,
        "raw_subset_without_stored_image": n_missing,
        "mismatches": n_mismatch,
        "max_abs_diff": max_abs_diff,
        "passed": ok,
    }
    if ok:
        logger.info("Verified: all %d files carry the right fingerprint; %d/%d raw-subset images reproduce exactly",
                    len(tile_files) + len(shard_files), n_checked, n_checked)
        if n_checked == 0:
            logger.warning("No raw-subset images to check (cutouts.raw_subset_fraction = 0?); "
                           "only fingerprints were verified")
    else:
        logger.error("Verification FAILED: %d files with a wrong fingerprint, %d/%d images differ",
                     len(bad_fingerprint), n_mismatch, n_checked)
    write_stats(config, STAGE, stats)
    return stats


def apply(config: dict, inputs: List[str], output: str, coords: Optional[str] = None, logger=None) -> Dict[str, object]:
    """Preprocess FITS files into an HDF5 file of uint8 images."""
    logger = logger or stage_logger(config, STAGE)
    prep = get_preprocessor(config)
    positions = {}
    if coords:
        table = pd.read_csv(coords)
        positions = {str(Path(f)): (ra, dec) for f, ra, dec in zip(table["file"], table["ra"], table["dec"])}
    images, names, rows, failed = [], [], [], []
    for path in inputs:
        ra, dec = positions.get(str(Path(path)), (None, None))
        try:
            image, _, stats = prep.preprocess_fits(path, ra, dec)
        except Exception as err:
            failed.append(path)
            logger.error("%s: %s", path, err)
            continue
        images.append(image)
        names.append(str(path))
        rows.append(stats)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    size = prep.OUTPUT_SIZE
    with h5py.File(out, "w") as h5:
        h5.create_dataset("images", data=np.asarray(images, dtype=np.uint8).reshape(-1, size, size),
                          chunks=(1, size, size) if images else None)
        h5.create_dataset("source_file", data=np.asarray(names, dtype=object), dtype=h5py.string_dtype())
        for key in ("noise_sigma", "vmax", "asinh_a", "blank_fraction"):
            h5.create_dataset(key, data=np.asarray([r[key] for r in rows], dtype=np.float32))
        h5.create_dataset("degenerate", data=np.asarray([r["degenerate"] for r in rows], dtype=bool))
        h5.attrs.update(preprocess_version=prep.VERSION, preprocess_fingerprint=prep.FINGERPRINT, size=size)
    logger.info("Preprocessed %d images into %s (%d failed)", len(images), out, len(failed))
    return {"images": len(images), "failed": failed, "output": str(out)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("verify", "apply"):
        p = sub.add_parser(name)
        p.add_argument("--config", default=None)
        p.add_argument("--root", default=None, help="Override paths.root.")
    sub.choices["apply"].add_argument("--inputs", nargs="+", required=True, help="FITS files or glob patterns.")
    sub.choices["apply"].add_argument("--output", required=True, help="Output HDF5 file.")
    sub.choices["apply"].add_argument("--coords", default=None, help="CSV with file,ra,dec columns.")
    args = parser.parse_args(argv)
    config = load_config(args.config, root=args.root)
    if args.command == "verify":
        stats = verify(config)
        raise SystemExit(0 if stats["passed"] else 1)
    files = sorted({f for pattern in args.inputs for f in (glob.glob(pattern) or [pattern])})
    apply(config, files, args.output, args.coords)


if __name__ == "__main__":
    main()
