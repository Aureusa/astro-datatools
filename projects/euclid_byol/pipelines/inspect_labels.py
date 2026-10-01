"""Figures of labelled and unlabelled galaxies from the finished dataset.

* ``labelled.png``: galaxies with Galaxy Zoo Euclid labels (Walmsley et al.
  2025), one row per morphology class, each captioned with its main vote
  fractions. Classes use confident votes so the examples are clear.
* ``unlabelled.png``: a random draw of galaxies without labels (the extra
  pretraining pool).

Run after ``crossmatch_labels``::

    python -m projects.euclid_byol.pipelines.inspect_labels --config projects/euclid_byol/configs/pipeline.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..config import get_paths, load_config  # noqa: E402
from ..dataset import EuclidShardDataset  # noqa: E402

F = "{}_fraction"
#: (row name, boolean selection on the vote fractions)
CLASSES = [
    ("smooth", lambda d: d[F.format("smooth-or-featured_smooth")] > 0.8),
    ("featured, face-on spiral", lambda d: (d[F.format("smooth-or-featured_featured-or-disk")] > 0.6)
     & (d[F.format("disk-edge-on_no")] > 0.7) & (d[F.format("has-spiral-arms_yes")] > 0.8)),
    ("barred", lambda d: (d[F.format("bar_strong")] + d[F.format("bar_weak")]) > 0.7),
    ("edge-on disc", lambda d: (d[F.format("smooth-or-featured_featured-or-disk")] > 0.6)
     & (d[F.format("disk-edge-on_yes")] > 0.8)),
    ("merger / disturbed", lambda d: (d[F.format("merging_merger")] + d[F.format("merging_major-disturbance")]) > 0.6),
    ("problem (star / artefact)", lambda d: d[F.format("smooth-or-featured_problem")] > 0.6),
]


def caption(row: pd.Series) -> str:
    """Short summary of a galaxy's votes, following the decision tree."""
    top = {
        "smooth": row[F.format("smooth-or-featured_smooth")],
        "featured": row[F.format("smooth-or-featured_featured-or-disk")],
        "problem": row[F.format("smooth-or-featured_problem")],
    }
    name = max(top, key=lambda k: np.nan_to_num(top[k]))
    parts = [f"{name} {top[name]:.2f}"]
    extra = []
    fr = lambda q: row.get(F.format(q), np.nan)  # noqa: E731
    if fr("disk-edge-on_yes") > 0.5:
        extra.append(f"edge-on {fr('disk-edge-on_yes'):.2f}")
    if fr("has-spiral-arms_yes") > 0.5:
        extra.append(f"spiral {fr('has-spiral-arms_yes'):.2f}")
    bar = fr("bar_strong") + fr("bar_weak")
    if bar > 0.5:
        extra.append(f"bar {bar:.2f}")
    merge = fr("merging_merger") + fr("merging_major-disturbance")
    if merge > 0.5:
        extra.append(f"merger {merge:.2f}")
    if name == "smooth":
        shapes = {"round": fr("how-rounded_round"), "in-between": fr("how-rounded_in-between"),
                  "cigar": fr("how-rounded_cigar-shaped")}
        shape = max(shapes, key=lambda k: np.nan_to_num(shapes[k]))
        extra.append(f"{shape} {shapes[shape]:.2f}")
    return "\n".join([parts[0], " · ".join(extra[:2])]) if extra else parts[0]


def fetch(ds: EuclidShardDataset, object_ids) -> dict:
    """uint8 images for the given object ids (random access; fine for a few hundred)."""
    pos = pd.Series(np.arange(len(ds)), index=ds.index["object_id"].to_numpy())
    return {int(o): ds[int(pos[o])][0] for o in object_ids}


def plot_rows(rows, path: Path, title: str, ncols: int) -> None:
    """rows: list of (row label, [(image, caption), ...])."""
    fig, axes = plt.subplots(len(rows), ncols, figsize=(2.1 * ncols, 2.45 * len(rows)), squeeze=False)
    for r, (name, items) in enumerate(rows):
        for c in range(ncols):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if c < len(items):
                image, text = items[c]
                ax.imshow(image, cmap="gray", vmin=0, vmax=255, origin="lower", interpolation="nearest")
                ax.set_title(text, fontsize=7, pad=2)
            if c == 0 and name:
                ax.set_ylabel(name, fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--root", default=None, help="Override paths.root.")
    parser.add_argument("--out-dir", default=None, help="Where to write the figures (default <root>/inspect).")
    parser.add_argument("--per-class", type=int, default=8)
    parser.add_argument("-n", type=int, default=48, help="Unlabelled galaxies to show.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    config = load_config(args.config, root=args.root)
    paths = get_paths(config)
    out_dir = Path(args.out_dir) if args.out_dir else paths.root / "inspect"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    labels = pd.read_parquet(paths.labels_dir / "morphology_labelled.parquet")
    labels = labels[labels["match_type"] != "none"].reset_index(drop=True)
    # Show everything that was downloaded, including the flagged "problem" objects.
    ds = EuclidShardDataset(paths.dataset_dir, columns=["segmentation_area", "label_morphology"], exclude_flagged=False)

    picked = []
    for name, select in CLASSES:
        pool = labels[select(labels).fillna(False).to_numpy()]
        take = pool.iloc[rng.choice(len(pool), size=min(args.per_class, len(pool)), replace=False)]
        picked.append((name, len(pool), take))
        print(f"{name:28s} {len(pool):>8,d} confident examples")
    images = fetch(ds, pd.concat([t["matched_object_id"] for _, _, t in picked]).astype("int64"))
    rows = [(f"{name}\n(n={n:,})", [(images[int(r.matched_object_id)], caption(r)) for _, r in take.iterrows()])
            for name, n, take in picked]
    plot_rows(rows, out_dir / "labelled.png",
              f"Labelled galaxies ({len(labels):,} with Galaxy Zoo Euclid vote fractions), examples per class",
              args.per_class)

    unlabelled = ds.index[~ds.index["label_morphology"].fillna(False).to_numpy(bool)]
    take = unlabelled.iloc[rng.choice(len(unlabelled), size=min(args.n, len(unlabelled)), replace=False)]
    images = fetch(ds, take["object_id"])
    items = [(images[int(o)], f"area {a:.0f} px") for o, a in zip(take["object_id"], take["segmentation_area"])]
    ncols = 8
    rows = [("", items[i:i + ncols]) for i in range(0, len(items), ncols)]
    plot_rows(rows, out_dir / "unlabelled.png",
              f"Unlabelled galaxies (random {len(items)} of {len(unlabelled):,})", ncols)
    print(f"Wrote {out_dir / 'labelled.png'} and {out_dir / 'unlabelled.png'}")


if __name__ == "__main__":
    main()
