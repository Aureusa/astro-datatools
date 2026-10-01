"""Estimate the disk space of a full run before downloading anything.

Nothing is written to disk. The stage queries the catalogue tile by tile,
fetching only the columns the selection needs, and applies exactly the same
selection code as ``query_catalogue``. It counts the objects that would be
downloaded, then discards the rows. Tiles whose catalogue part already exists
under ``paths.root`` are read from disk instead of queried.

It reports:

* uint8 images, uncompressed and with ``gzip`` (the ratio is measured on Q1 cutouts);
* the raw float32 subset for several ``raw_subset_fraction`` values;
* catalogue and index metadata;
* peak and final disk use for the configured ``dataset.delete_tile_files`` and
  compression settings;
* the number of objects for extra ``segmentation_area`` cuts on top of the
  configured selection, to help fit a budget.

Usage (a full pass over the 352 Q1 tiles takes roughly 10-20 minutes)::

    python -m projects.euclid_byol.pipelines.estimate_storage --config projects/euclid_byol/configs/pipeline.yaml
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from astro_datatools.logger import setup_logging

from ..config import get_paths, get_preprocessor, load_config
from ..tilestore import raw_subset_mask
from .query_catalogue import _token, apply_selection, fetch_tiles, make_query_fn, normalise_dtypes, select_tiles

#: Measured on the Q1 sample (tile 102044822): bytes on disk per stored item.
HDF5_OVERHEAD_PER_IMAGE = 300          # chunk index, status and statistics arrays
RAW_BYTES_PER_IMAGE = 224 * 224 * 4 + 200
GZIP_RATIO = 0.79                      # compressed / uncompressed uint8 images (gzip level 4)
CATALOGUE_BYTES_PER_ROW = 90           # one parquet part; the master index stores it again
INDEX_BYTES_PER_ROW = 150              # dataset index; index_parts store it again
AREA_CUTS = (0, 25, 50, 100, 200, 500)
RAW_FRACTIONS = (0.0, 0.0001, 0.001, 0.01)


def selection_columns(config: dict) -> List[str]:
    """Catalogue columns needed to evaluate the selection (plus ids and sizes)."""
    exprs = " ".join(config["catalogue"]["selection"] or [])
    used = [c for c in config["catalogue"]["columns"] if re.search(rf"\b{re.escape(c)}\b", exprs)]
    return list(dict.fromkeys(["object_id", "segmentation_area", *used]))


def tile_counts(config: dict, query, tile: pd.Series) -> Dict[str, object]:
    """Selected object ids and sizes for one tile, without writing anything."""
    paths = get_paths(config)
    part = paths.catalogue_part(tile["tile_index"])
    columns = selection_columns(config)
    if part.exists():
        df = pd.read_parquet(part, columns=columns)
    else:
        table = _token(config["catalogue"]["table"], "table")
        df = query(f"SELECT {', '.join(_token(c, 'column') for c in columns)} FROM {table} "
                   f"WHERE basic_download_data_oid = {int(tile['catalogue_oid'])}")
        expected = tile.get("n_catalogue")
        if expected is not None and not pd.isna(expected) and len(df) != int(expected):
            raise RuntimeError(f"got {len(df)} rows, archive reports {int(expected)}")
        df = normalise_dtypes(df)
    n_detections = len(df)
    df = df[apply_selection(df, config["catalogue"]["selection"]).to_numpy()]
    limit = config["cutouts"].get("max_objects_per_tile")
    if limit and len(df) > limit:
        df = df.nlargest(int(limit), "segmentation_area")  # sizes of a random pick would differ slightly
    return {
        "tile_index": int(tile["tile_index"]),
        "detections": n_detections,
        "object_id": df["object_id"].to_numpy(np.int64),
        "segmentation_area": df["segmentation_area"].to_numpy(float),
    }


def summarize(config: dict, results: List[dict]) -> dict:
    prep = get_preprocessor(config)
    image_bytes = prep.OUTPUT_SIZE ** 2 + HDF5_OVERHEAD_PER_IMAGE
    ids = np.concatenate([r["object_id"] for r in results]) if results else np.zeros(0, np.int64)
    areas = np.concatenate([r["segmentation_area"] for r in results]) if results else np.zeros(0)
    per_tile = np.array([len(r["object_id"]) for r in results] or [0])
    n = len(ids)
    n_det = int(sum(r["detections"] for r in results))
    ratio = lambda compression: GZIP_RATIO if compression == "gzip" else 1.0  # noqa: E731
    shards = n * image_bytes * ratio(config["dataset"].get("compression"))
    tile_ratio = ratio(config["cutouts"].get("compression"))
    tile_files = n * image_bytes * tile_ratio
    fraction = float(config["cutouts"]["raw_subset_fraction"])
    raw = int(raw_subset_mask(ids, fraction).sum()) * RAW_BYTES_PER_IMAGE
    metadata = 2 * n_det * CATALOGUE_BYTES_PER_ROW + 2 * n * INDEX_BYTES_PER_ROW
    largest_tile = int(per_tile.max()) * image_bytes * tile_ratio
    delete = bool(config["dataset"].get("delete_tile_files"))
    if delete:
        # Each tile file is deleted once copied into the shards, so images exist
        # twice for at most one tile, even if everything is downloaded first.
        peak = max(tile_files, shards) + largest_tile + raw + metadata
        final = shards + raw + metadata
    else:
        peak = final = tile_files + shards + raw + metadata
    return {
        "tiles": len(results),
        "detections": n_det,
        "selected": n,
        "images_tb": shards / 1e12,
        "images_uncompressed_tb": n * image_bytes / 1e12,
        "images_gzip_tb": n * image_bytes * GZIP_RATIO / 1e12,
        "raw_subset_fraction": fraction,
        "raw_subset_gb": raw / 1e9,
        "raw_subset_gb_by_fraction": {
            str(f): int(raw_subset_mask(ids, f).sum()) * RAW_BYTES_PER_IMAGE / 1e9 for f in RAW_FRACTIONS
        },
        "metadata_gb": metadata / 1e9,
        "delete_tile_files": delete,
        "compression": config["dataset"].get("compression"),
        "tile_file_compression": config["cutouts"].get("compression"),
        "peak_tb": peak / 1e12,
        "final_tb": final / 1e12,
        "selected_by_min_segmentation_area": {
            str(a): {"objects": int((areas >= a).sum()), "images_tb": float((areas >= a).sum() * image_bytes / 1e12)}
            for a in AREA_CUTS
        },
    }


def format_report(s: dict, budget: Optional[float]) -> str:
    tb = lambda v: f"{v:.3f} TB"  # noqa: E731
    lines = [
        f"Tiles: {s['tiles']}   detections: {s['detections']:,}   selected: {s['selected']:,} "
        f"({100 * s['selected'] / max(s['detections'], 1):.1f}%)",
        "",
        f"uint8 images in shards ({s['compression'] or 'uncompressed'}):  {tb(s['images_tb'])}"
        f"   [uncompressed {tb(s['images_uncompressed_tb'])}, gzip ~{tb(s['images_gzip_tb'])}]",
        f"raw float32 subset (fraction {s['raw_subset_fraction']}):  {s['raw_subset_gb']:.2f} GB   "
        "[" + ", ".join(f"{f}: {g:.2f} GB" for f, g in s["raw_subset_gb_by_fraction"].items()) + "]",
        f"catalogue + index metadata:  {s['metadata_gb']:.1f} GB",
        f"delete_tile_files = {s['delete_tile_files']}, tile files {s['tile_file_compression'] or 'uncompressed'}:  "
        f"peak {tb(s['peak_tb'])}, after the run {tb(s['final_tb'])}",
    ]
    if budget:
        verdict = "fits" if s["peak_tb"] <= budget else "EXCEEDS"
        lines.append(f"storage_budget_tb = {budget}: peak {verdict} the budget")
    lines += ["", "Extra cut on top of the selection (uncompressed images):"]
    for area, v in s["selected_by_min_segmentation_area"].items():
        lines.append(f"  segmentation_area >= {area:>4}: {v['objects']:>12,} objects  {tb(v['images_tb'])}")
    return "\n".join(lines)


def run(config: dict, query=None, logger=None) -> dict:
    logger = logger or setup_logging(name="euclid_byol.estimate_storage")  # console only
    paths = get_paths(config)
    if paths.tiles.exists():
        all_tiles = pd.read_parquet(paths.tiles)
    else:
        query = query or make_query_fn(config)
        all_tiles = fetch_tiles(query, config["archive"]["release"], config["archive"]["band"])
    tiles = select_tiles(config, all_tiles)
    if query is None and not all(paths.catalogue_part(t).exists() for t in tiles["tile_index"]):
        query = make_query_fn(config)
    logger.info("Counting selected objects in %d tiles (nothing is written to disk)", len(tiles))

    results, failed = [], []
    with ThreadPoolExecutor(max_workers=max(1, int(config["catalogue"]["query_workers"]))) as pool:
        futures = {pool.submit(tile_counts, config, query, row): int(row["tile_index"]) for _, row in tiles.iterrows()}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                results.append(fut.result())
            except Exception as err:
                failed.append(futures[fut])
                logger.error("Tile %d failed: %s", futures[fut], err)
            if i % 25 == 0 or i == len(futures):
                logger.info("%d/%d tiles counted", i, len(futures))
    summary = summarize(config, results)
    summary["tiles_failed"] = failed
    print(format_report(summary, config["catalogue"].get("storage_budget_tb")))
    if failed:
        print(f"\nWARNING: {len(failed)} tiles failed and are missing from the estimate: {failed}")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--root", default=None, help="Override paths.root (only read, never written).")
    parser.add_argument("--json", default=None, help="Also write the numbers to this JSON file.")
    args = parser.parse_args(argv)
    summary = run(load_config(args.config, root=args.root))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
