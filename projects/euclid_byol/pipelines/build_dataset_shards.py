"""Stage 4: pack the preprocessed cutouts into fixed-size HDF5 shards.

Completed tile files are appended, in tile order, to ``dataset/shard_<n>.h5``
(``images`` uint8 (N, S, S) chunked per image, ``object_id`` int64 (N,)).
Only objects with status ``ok`` go in. For every tile, the index rows
(object -> shard file and row, preprocessing statistics, catalogue columns)
are written to ``dataset/index_parts/tile_<t>.parquet``, and all parts are
concatenated into ``dataset/index.parquet``.

Incremental and resumable: ``dataset/_state.json`` records which tiles are in
which shard rows and is updated after every tile, so the stage can run while
downloads continue and picks up newly completed tiles each time. Rows written
after the last state update (a crash mid-tile) are truncated on the next run.

The images keep the tile order; shuffle in the data loader.

Usage::

    python -m projects.euclid_byol.pipelines.build_dataset_shards --config projects/euclid_byol/configs/sample.yaml
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
from typing import Dict, List

import astropy
import h5py
import numpy as np
import pandas as pd

from ..config import (
    atomic_write_parquet,
    atomic_write_text,
    get_paths,
    get_preprocessor,
    load_config,
    stage_logger,
    write_stats,
)
from ..tilestore import STAT_FIELDS, STATUS_OK, TileStore, _compression_kwargs, check_fingerprint

STAGE = "build_dataset_shards"
READ_CHUNK = 2048


class ShardWriter:
    """Append images to numbered shard files of at most ``shard_size`` images."""

    def __init__(self, config: dict, state: dict):
        self.paths = get_paths(config)
        self.prep = get_preprocessor(config)
        self.shard_size = int(config["dataset"]["shard_size"])
        self.compression = config["dataset"].get("compression")
        self.state = state
        self._h5 = None
        self._number = None

    def _open(self, number: int) -> h5py.File:
        if self._number != number:
            self.close()
            path = self.paths.shard_file(number)
            size = self.prep.OUTPUT_SIZE
            h5 = h5py.File(path, "a")
            if "images" not in h5:
                h5.create_dataset("images", shape=(0, size, size), maxshape=(self.shard_size, size, size),
                                  dtype=np.uint8, chunks=(1, size, size), **_compression_kwargs(self.compression))
                h5.create_dataset("object_id", shape=(0,), maxshape=(self.shard_size,), dtype=np.int64)
                h5.attrs.update(
                    shard=number,
                    preprocess_version=self.prep.VERSION,
                    preprocess_fingerprint=self.prep.FINGERPRINT,
                    size=size,
                    created=_dt.datetime.now().isoformat(timespec="seconds"),
                    numpy_version=np.__version__,
                    astropy_version=astropy.__version__,
                )
            self._h5, self._number = h5, number
        return self._h5

    def append(self, object_ids: np.ndarray, images: np.ndarray) -> List[tuple]:
        """Append images; returns ``(shard_number, first_row, count)`` per shard written to."""
        placed = []
        start = 0
        shards = self.state["shards"]
        while start < len(object_ids):
            if not shards or shards[-1]["n"] >= self.shard_size:
                shards.append({"file": self.paths.shard_file(len(shards)).name, "n": 0})
            number, n = len(shards) - 1, shards[-1]["n"]
            k = min(len(object_ids) - start, self.shard_size - n)
            h5 = self._open(number)
            h5["images"].resize((n + k,) + h5["images"].shape[1:])
            h5["object_id"].resize((n + k,))
            h5["images"][n:n + k] = images[start:start + k]
            h5["object_id"][n:n + k] = object_ids[start:start + k]
            placed.append((number, n, k))
            shards[-1]["n"] = n + k
            start += k
        return placed

    def flush(self) -> None:
        if self._h5 is not None:
            self._h5.flush()

    def close(self) -> None:
        if self._h5 is not None and self._h5.id.valid:
            self._h5.close()
        self._h5, self._number = None, None


def load_state(config: dict) -> dict:
    paths = get_paths(config)
    prep = get_preprocessor(config)
    if paths.shard_state.exists():
        state = json.loads(paths.shard_state.read_text())
        check_fingerprint(state, prep.VERSION, prep.FINGERPRINT, f"Dataset in {paths.dataset_dir}")
        if state["shard_size"] != int(config["dataset"]["shard_size"]):
            raise ValueError(f"dataset.shard_size changed from {state['shard_size']}; use a new dataset directory.")
        return state
    return {
        "preprocess_version": prep.VERSION,
        "preprocess_fingerprint": prep.FINGERPRINT,
        "size": prep.OUTPUT_SIZE,
        "shard_size": int(config["dataset"]["shard_size"]),
        "shards": [],
        "tiles": {},
    }


def save_state(config: dict, state: dict) -> None:
    atomic_write_text(get_paths(config).shard_state, json.dumps(state, indent=1))


def recover(config: dict, state: dict, logger) -> None:
    """Make shard files agree with the state after an interrupted run."""
    paths = get_paths(config)
    known = {s["file"] for s in state["shards"]}
    for path in paths.dataset_dir.glob("shard_*.h5"):
        if path.name not in known:
            logger.warning("Removing %s (created after the last saved state)", path.name)
            path.unlink()
    for shard in state["shards"]:
        path = paths.dataset_dir / shard["file"]
        if not path.exists():
            raise FileNotFoundError(f"{path} is listed in {paths.shard_state.name} but missing.")
        with h5py.File(path, "r+") as h5:
            if h5["images"].shape[0] != shard["n"] or h5["object_id"].shape[0] != shard["n"]:
                logger.warning("Truncating %s from %d to %d rows (interrupted write)",
                               path.name, h5["images"].shape[0], shard["n"])
                h5["images"].resize((shard["n"],) + h5["images"].shape[1:])
                h5["object_id"].resize((shard["n"],))


def add_tile(config: dict, writer: ShardWriter, store: TileStore) -> pd.DataFrame:
    """Append a tile's ok images to the shards; returns its index rows."""
    paths = get_paths(config)
    ok = np.flatnonzero(store.status == STATUS_OK)
    object_ids = store.object_ids[ok]
    shard_col = np.empty(len(ok), dtype=object)
    row_col = np.empty(len(ok), dtype=np.int64)
    done = 0
    for lo in range(0, len(store), READ_CHUNK):
        hi = min(lo + READ_CHUNK, len(store))
        take = ok[(ok >= lo) & (ok < hi)]
        if len(take) == 0:
            continue
        images = store.h5["images"][lo:hi][take - lo]
        for number, first, k in writer.append(object_ids[done:done + len(take)], images):
            shard_col[done:done + k] = writer.state["shards"][number]["file"]
            row_col[done:done + k] = np.arange(first, first + k)
            done += k
    writer.flush()

    index = pd.DataFrame({"object_id": object_ids, "tile_index": store.tile_index,
                          "shard": shard_col.astype(str), "shard_row": row_col})
    for name in STAT_FIELDS:
        index[name] = store.h5[name][:][ok]
    catalogue = pd.read_parquet(paths.catalogue_part(store.tile_index))
    catalogue = catalogue.drop(columns=["tile_index", "selected"], errors="ignore")
    return index.merge(catalogue, on="object_id", how="left", validate="one_to_one")


def build_index(config: dict) -> int:
    """Concatenate the per-tile index parts into ``dataset/index.parquet`` (streamed)."""
    import pyarrow.parquet as pq

    paths = get_paths(config)
    parts = sorted(paths.index_parts.glob("tile_*.parquet"), key=lambda p: int(p.stem.split("_")[1]))
    tmp = paths.dataset_index.with_name(paths.dataset_index.name + ".tmp")
    writer, n = None, 0
    try:
        for part in parts:
            table = pq.read_table(part)
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema)
            writer.write_table(table.cast(writer.schema))
            n += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    if writer is not None:
        tmp.replace(paths.dataset_index)
        flags = paths.labels_dir / "label_flags.parquet"
        if flags.exists():
            from .crossmatch_labels import apply_label_flags

            apply_label_flags(paths.dataset_index, pd.read_parquet(flags))
    return n


def delete_tile_file(config: dict, path, logger) -> None:
    """Replace a sharded tile file by its (image-free) status table, then delete it.

    The status file keeps the download manifest (including failures), and tells
    ``download_cutouts`` that the tile is done so it is not downloaded again.
    """
    paths = get_paths(config)
    with TileStore.open(path) as store:
        status = store.status_table()
        tile_index = store.tile_index
    atomic_write_parquet(status, paths.tile_status_file(tile_index))
    path.unlink()
    logger.info("Deleted %s (status kept in %s)", path.name, paths.tile_status_file(tile_index).name)


def run(config: dict, logger=None) -> Dict[str, object]:
    logger = logger or stage_logger(config, STAGE)
    paths = get_paths(config)
    prep = get_preprocessor(config)
    paths.dataset_dir.mkdir(parents=True, exist_ok=True)
    paths.index_parts.mkdir(parents=True, exist_ok=True)
    state = load_state(config)
    recover(config, state, logger)

    tile_files = sorted(paths.cutouts_dir.glob("tile_*.h5"), key=lambda p: int(p.stem.split("_")[1]))
    added, skipped_incomplete = [], []
    delete = bool(config["dataset"].get("delete_tile_files"))
    writer = ShardWriter(config, state)
    try:
        for path in tile_files:
            tile_key = path.stem.split("_")[1]
            if tile_key in state["tiles"]:
                if delete:  # sharded before, but the run stopped before deleting it
                    delete_tile_file(config, path, logger)
                continue
            with TileStore.open(path) as store:
                store.check_fingerprint(prep.VERSION, prep.FINGERPRINT)
                if not store.complete:
                    skipped_incomplete.append(int(tile_key))
                    continue
                n_before = sum(s["n"] for s in state["shards"])
                index = add_tile(config, writer, store)
            atomic_write_parquet(index, paths.index_parts / f"tile_{tile_key}.parquet")
            state["tiles"][tile_key] = {"n": len(index), "first_global_row": n_before}
            save_state(config, state)
            added.append(int(tile_key))
            logger.info("Tile %s: %d images added (dataset now %d images in %d shards)",
                        tile_key, len(index), n_before + len(index), len(state["shards"]))
            if delete:
                delete_tile_file(config, path, logger)
    finally:
        writer.close()

    n_index = build_index(config)
    n_images = sum(s["n"] for s in state["shards"])
    if n_index != n_images:
        raise RuntimeError(f"Index has {n_index} rows but shards hold {n_images} images.")
    size_gb = sum((paths.dataset_dir / s["file"]).stat().st_size for s in state["shards"]) / 1e9
    if skipped_incomplete:
        logger.info("%d tiles not complete yet (skipped until downloaded): %s",
                    len(skipped_incomplete), skipped_incomplete[:20])
    logger.info("Dataset: %d images in %d shards (%.2f GB), %d tiles; index %s",
                n_images, len(state["shards"]), size_gb, len(state["tiles"]), paths.dataset_index)
    stats = {
        "tiles_added_this_run": added,
        "tiles_in_dataset": len(state["tiles"]),
        "tiles_incomplete": skipped_incomplete,
        "images": n_images,
        "shards": len(state["shards"]),
        "size_gb": round(size_gb, 3),
        "preprocess_version": prep.VERSION,
        "preprocess_fingerprint": prep.FINGERPRINT,
    }
    write_stats(config, STAGE, stats)
    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--root", default=None, help="Override paths.root.")
    args = parser.parse_args(argv)
    run(load_config(args.config, root=args.root))


if __name__ == "__main__":
    main()
