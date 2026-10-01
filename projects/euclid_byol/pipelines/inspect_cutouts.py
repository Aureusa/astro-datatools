"""Visual and numerical sanity check of downloaded cutouts (run before scaling up).

Writes to ``<root>/inspect/``:

* ``grid.png``: random preprocessed images (uint8, as the model sees them);
* ``largest.png``: the objects with the largest segmentation area, where
  structure such as spiral arms or bars should be visible;
* ``raw_vs_processed.png``: for raw-subset objects, the raw cutout (linear,
  zscale) next to the uint8 image and its pixel histogram.

and prints the fraction of pixels at 0 and 255, which shows whether faint
outskirts are clipped to black or bright cores saturated.

Usage::

    python -m projects.euclid_byol.pipelines.inspect_cutouts --config projects/euclid_byol/configs/sample.yaml
"""
from __future__ import annotations

import argparse

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from astropy.visualization import ZScaleInterval  # noqa: E402

from ..config import get_paths, load_config  # noqa: E402
from ..tilestore import STATUS_OK, TileStore  # noqa: E402


def load_ok_images(paths, limit_tiles: int = 5):
    """uint8 images, object ids and tile indices of ok rows from the first tiles."""
    images, ids, tiles = [], [], []
    for path in sorted(paths.cutouts_dir.glob("tile_*.h5"))[:limit_tiles]:
        with TileStore.open(path) as store:
            ok = np.flatnonzero(store.status == STATUS_OK)
            images.append(store.h5["images"][:][ok])
            ids.append(store.object_ids[ok])
            tiles.append(np.full(len(ok), store.tile_index))
    if not images and paths.dataset_index.exists():
        # Tile files deleted after sharding: read the first tiles from the shards.
        index = pd.read_parquet(paths.dataset_index, columns=["object_id", "tile_index", "shard", "shard_row"])
        index = index[index["tile_index"].isin(np.unique(index["tile_index"])[:limit_tiles])]
        for shard, rows in index.groupby("shard", sort=False):
            with h5py.File(paths.dataset_dir / shard, "r") as h5:
                order = np.argsort(rows["shard_row"].to_numpy())
                images.append(h5["images"][np.sort(rows["shard_row"].to_numpy())])
                ids.append(rows["object_id"].to_numpy()[order])
                tiles.append(rows["tile_index"].to_numpy()[order])
    if not images:
        return np.zeros((0, 1, 1), np.uint8), np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(images), np.concatenate(ids), np.concatenate(tiles)


def plot_grid(images, titles, path, ncols=8):
    n = len(images)
    nrows = max(1, int(np.ceil(n / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2.15 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, image, title in zip(axes.ravel(), images, titles):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255, origin="lower", interpolation="nearest")
        ax.set_title(title, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--root", default=None, help="Override paths.root.")
    parser.add_argument("-n", type=int, default=48, help="Images per grid.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    config = load_config(args.config, root=args.root)
    paths = get_paths(config)
    out_dir = paths.root / "inspect"
    out_dir.mkdir(parents=True, exist_ok=True)

    images, ids, tiles = load_ok_images(paths)
    if len(images) == 0:
        print("No downloaded images yet.")
        return
    catalogue = pd.concat([pd.read_parquet(paths.catalogue_part(t)) for t in np.unique(tiles)])
    catalogue = catalogue.set_index("object_id").reindex(ids)

    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(images), size=min(args.n, len(images)), replace=False)
    plot_grid(images[pick], [f"{ids[i]}\narea={catalogue['segmentation_area'].iloc[i]:.0f}" for i in pick],
              out_dir / "grid.png")
    largest = np.argsort(-catalogue["segmentation_area"].to_numpy())[: args.n]
    plot_grid(images[largest], [f"{ids[i]}\narea={catalogue['segmentation_area'].iloc[i]:.0f}" for i in largest],
              out_dir / "largest.png")

    at0 = (images == 0).mean(axis=(1, 2))
    at255 = (images == 255).mean(axis=(1, 2))
    print(f"{len(images)} ok images")
    print(f"fraction of pixels == 0   per image: median {np.median(at0):.4f}, 95th pct {np.percentile(at0, 95):.4f}")
    print(f"fraction of pixels == 255 per image: median {np.median(at255):.6f}, max {at255.max():.6f}")
    print(f"image median value: median {np.median(np.median(images, axis=(1, 2))):.0f} / 255")

    # Raw vs processed, for the largest objects that have a raw cutout.
    rows = []
    for path in sorted(paths.raw_dir.glob("tile_*.h5")):
        with h5py.File(path, "r") as h5:
            rows += [(int(o), path, j) for j, o in enumerate(h5["object_id"][:])]
    position = {int(o): i for i, o in enumerate(ids)}
    rows = [r for r in rows if r[0] in position]
    rows.sort(key=lambda r: -catalogue["segmentation_area"].iloc[position[r[0]]])
    rows = rows[:6]
    if rows:
        fig, axes = plt.subplots(len(rows), 3, figsize=(9, 3 * len(rows)), squeeze=False)
        for (oid, path, j), ax in zip(rows, axes):
            with h5py.File(path, "r") as h5:
                raw = h5["raw"][j]
            image = images[position[oid]]
            lo, hi = ZScaleInterval().get_limits(np.nan_to_num(raw))
            ax[0].imshow(raw, cmap="gray", vmin=lo, vmax=hi, origin="lower")
            ax[0].set_title(f"raw (linear zscale) {oid}", fontsize=8)
            ax[1].imshow(image, cmap="gray", vmin=0, vmax=255, origin="lower")
            ax[1].set_title("preprocessed uint8", fontsize=8)
            ax[2].hist(image.ravel(), bins=64, range=(0, 256), log=True)
            ax[2].set_title("uint8 histogram", fontsize=8)
            ax[0].axis("off")
            ax[1].axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / "raw_vs_processed.png", dpi=100)
        plt.close(fig)
    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
