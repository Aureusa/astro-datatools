"""Counts at every stage, to catch silent drop-off between them.

Prints (and writes to ``stats/status.json``): disk use per directory; tiles queried, detections and
selected objects; tile files with download status counts; raw-subset size;
shards and images in the dataset; labelled objects.

Usage::

    python -m projects.euclid_byol.pipelines.status --config projects/euclid_byol/configs/pipeline.yaml
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import h5py
import pandas as pd

from ..config import get_paths, load_config, write_stats
from ..tilestore import tile_counts


def collect(config: dict) -> dict:
    paths = get_paths(config)
    out = {"root": str(paths.root)}

    parts = sorted(paths.catalogue_parts.glob("tile_*.parquet"))
    n_det = n_sel = 0
    for part in parts:
        sel = pd.read_parquet(part, columns=["selected"])["selected"]
        n_det += len(sel)
        n_sel += int(sel.sum())
    out["catalogue"] = {"tiles_queried": len(parts), "detections": n_det, "selected": n_sel}

    downloads: Counter = Counter()
    started = {int(p.name.split("_")[1].split(".")[0]) for p in paths.cutouts_dir.glob("tile_*")
               if p.name.endswith((".h5", ".status.parquet"))}
    for tile_index in sorted(started):
        counts = tile_counts(paths, tile_index)
        if counts is not None:
            downloads["tiles_complete"] += counts.pop("complete")
            downloads["tiles_started"] += 1
            downloads.update(counts)
    out["downloads"] = dict(downloads)

    raw = 0
    for path in paths.raw_dir.glob("tile_*.h5"):
        with h5py.File(path, "r") as h5:
            raw += h5["object_id"].shape[0]
    out["raw_subset"] = raw

    if paths.shard_state.exists():
        state = json.loads(paths.shard_state.read_text())
        out["dataset"] = {
            "tiles": len(state["tiles"]),
            "shards": len(state["shards"]),
            "images": sum(s["n"] for s in state["shards"]),
            "size_gb": round(sum((paths.dataset_dir / s["file"]).stat().st_size
                                 for s in state["shards"]) / 1e9, 3),
            "preprocess_version": state["preprocess_version"],
        }
    out["disk_gb"] = {
        d.name: round(sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e9, 3)
        for d in (paths.catalogue_dir, paths.cutouts_dir, paths.raw_dir, paths.dataset_dir, paths.labels_dir)
        if d.exists()
    }
    out["disk_gb"]["total"] = round(sum(out["disk_gb"].values()), 3)
    if paths.dataset_index.exists():
        import pyarrow.parquet as pq

        names = pq.read_schema(paths.dataset_index).names
        label_cols = [c for c in names if c.startswith("label_")]
        if label_cols:
            flags = pd.read_parquet(paths.dataset_index, columns=label_cols)
            out["labels"] = {c: int(flags[c].sum()) for c in label_cols}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--root", default=None, help="Override paths.root.")
    args = parser.parse_args(argv)
    config = load_config(args.config, root=args.root)
    status = collect(config)
    write_stats(config, "status", status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
