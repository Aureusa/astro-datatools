"""Stage 5: flag objects that have morphology labels (for fine-tuning / evaluation).

* **Euclid Q1 visual morphology** (Walmsley et al. 2025, arXiv:2503.15310,
  Zenodo 15106473): matched on ``object_id`` (both come from the Q1 MER
  catalogue); label rows whose id is not in the dataset fall back to a
  positional match within ``labels.morphology.match_radius_arcsec``.
* **DESI / DECaLS** (optional, ``labels.desi.path``): positional match within
  ``labels.desi.match_radius_arcsec`` (1" by default) over the Q1-DESI overlap.

Outputs, under ``labels/``:

* ``morphology_labelled.parquet`` / ``desi_labelled.parquet``: the label rows
  with the matched ``object_id`` and its ``shard`` / ``shard_row``, ready to
  pull the labelled images for fine-tuning;
* ``label_flags.parquet``: ``object_id`` plus ``label_morphology`` /
  ``label_desi`` booleans. These columns are also written into
  ``dataset/index.parquet`` (and re-applied whenever the shard stage rebuilds
  the index), so the pretraining pool can include or exclude labelled objects
  explicitly.

It also reports labelled galaxies that are *not* in the dataset (outside the
selection, not downloaded yet, or not in the queried tiles).

Usage::

    python -m projects.euclid_byol.pipelines.crossmatch_labels --config projects/euclid_byol/configs/pipeline.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import requests

from ..config import atomic_write_parquet, get_paths, load_config, stage_logger, write_stats

STAGE = "crossmatch_labels"
INDEX_COLUMNS = ["object_id", "right_ascension", "declination", "tile_index", "shard", "shard_row"]


def download_file(url: str, dest: Path, logger) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    logger.info("Downloading %s -> %s", url, dest)
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    tmp.replace(dest)


def read_table(path: Path) -> pd.DataFrame:
    """Read a parquet, CSV or FITS table."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in (".csv", ".txt"):
        return pd.read_csv(path)
    from astropy.table import Table

    table = Table.read(path)
    names = [n for n in table.colnames if len(table[n].shape) <= 1]
    return table[names].to_pandas()


def normalize_id_column(df: pd.DataFrame, id_col: str, logger=None) -> pd.DataFrame:
    """Coerce a label table's id column to int64, matching the dataset's ``object_id``.

    Label catalogues are not always typed the way we'd expect: the Walmsley et al.
    Zenodo parquet stores ``object_id`` as a string, even though it is the same
    integer id as our ``catalogue.mer_catalogue`` (both derive from Q1 MER). A
    ``.map()``/``.isin()`` between a string column and our int64 ``object_id``
    matches nothing, with no error, so id-based matching would silently fall
    through to the (slower, less exact) positional match for every row. Rows
    whose id is missing or cannot be parsed as an integer are dropped (they
    cannot be id-matched anyway), with a warning.

    :param df: Label table.
    :param id_col: Name of its id column.
    :param logger: Logger for a warning about dropped rows.
    :return: Copy of ``df`` with ``id_col`` as ``int64``.
    """
    coerced = pd.to_numeric(df[id_col], errors="coerce")
    bad = coerced.isna()
    if bad.any():
        message = f"{int(bad.sum())} rows have a missing or non-integer '{id_col}' and are dropped."
        if logger:
            logger.warning(message)
        else:
            import warnings

            warnings.warn(message)
        df, coerced = df[~bad], coerced[~bad]
    df = df.copy()
    df[id_col] = coerced.astype("int64")
    return df


def positional_match(ra, dec, index: pd.DataFrame, radius_arcsec: float):
    """Nearest dataset object to each position; returns (index row or -1, separation arcsec)."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    if len(ra) == 0 or len(index) == 0:
        return np.full(len(ra), -1), np.full(len(ra), np.nan)
    targets = SkyCoord(np.asarray(ra, float), np.asarray(dec, float), unit="deg")
    catalogue = SkyCoord(index["right_ascension"].to_numpy(float), index["declination"].to_numpy(float), unit="deg")
    idx, sep, _ = targets.match_to_catalog_sky(catalogue)
    sep = sep.to_value(u.arcsec)
    return np.where(sep <= radius_arcsec, idx, -1), sep


def match_morphology(labels: pd.DataFrame, index: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Match label rows to the dataset index by id, then by position."""
    id_col = cfg["id_column"]
    row_of = pd.Series(np.arange(len(index)), index=index["object_id"].to_numpy())
    row_of = row_of[~row_of.index.duplicated()]
    rows = labels[id_col].map(row_of).fillna(-1).astype(int).to_numpy()
    match_type = np.where(rows >= 0, "id", "none").astype(object)
    sep = np.where(rows >= 0, 0.0, np.nan)

    radius = cfg.get("match_radius_arcsec")
    unmatched = np.flatnonzero(rows < 0)
    if radius and len(unmatched):
        pos_rows, pos_sep = positional_match(labels[cfg["ra_column"]].to_numpy()[unmatched],
                                             labels[cfg["dec_column"]].to_numpy()[unmatched], index, radius)
        hit = pos_rows >= 0
        rows[unmatched[hit]] = pos_rows[hit]
        match_type[unmatched[hit]] = "position"
        sep[unmatched[hit]] = pos_sep[hit]
    return _attach(labels, index, rows, match_type, sep)


def match_positions(labels: pd.DataFrame, index: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows, sep = positional_match(labels[cfg["ra_column"]].to_numpy(), labels[cfg["dec_column"]].to_numpy(),
                                 index, cfg["match_radius_arcsec"])
    match_type = np.where(rows >= 0, "position", "none").astype(object)
    return _attach(labels, index, rows, match_type, np.where(rows >= 0, sep, np.nan))


def _attach(labels, index, rows, match_type, sep) -> pd.DataFrame:
    """Label rows plus the matched dataset object's id and shard location (null if unmatched)."""
    out = labels.reset_index(drop=True).copy()
    hit = rows >= 0
    matched = index.iloc[rows[hit]]
    out["matched_object_id"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out.loc[hit, "matched_object_id"] = matched["object_id"].to_numpy()
    out["shard"] = pd.Series(None, index=out.index, dtype=object)
    out.loc[hit, "shard"] = matched["shard"].to_numpy()
    out["shard_row"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out.loc[hit, "shard_row"] = matched["shard_row"].to_numpy()
    out["match_type"] = match_type
    out["match_sep_arcsec"] = sep
    return out


def missing_breakdown(ids: np.ndarray, master_path: Path) -> Dict[str, int]:
    """Why labelled objects are absent from the dataset."""
    if not master_path.exists():
        return {"not_in_dataset": int(len(ids))}
    master = pd.read_parquet(master_path, columns=["object_id", "selected"])
    master = master[master["object_id"].isin(ids)]
    selected = int(master["selected"].sum())
    return {
        "not_in_queried_tiles": int(len(np.setdiff1d(ids, master["object_id"].to_numpy()))),
        "not_selected": int(len(master) - selected),
        "selected_not_in_dataset_yet": selected,
    }


def apply_label_flags(index_path: Path, flags: pd.DataFrame) -> None:
    """Rewrite ``index.parquet`` with the label flag columns (streamed)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    flag_cols = [c for c in flags.columns if c != "object_id"]
    labelled = {c: flags.loc[flags[c], "object_id"].to_numpy() for c in flag_cols}
    tmp = index_path.with_name(index_path.name + ".tmp")
    source = pq.ParquetFile(index_path)
    writer = None
    try:
        for batch in source.iter_batches(batch_size=1_000_000):
            table = pa.Table.from_batches([batch])
            ids = table.column("object_id").to_numpy()
            for col in flag_cols:
                values = pa.array(np.isin(ids, labelled[col]))
                if col in table.column_names:
                    table = table.set_column(table.column_names.index(col), col, values)
                else:
                    table = table.append_column(col, values)
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is not None:
        tmp.replace(index_path)


def run(config: dict, download: bool = True, logger=None) -> Dict[str, object]:
    logger = logger or stage_logger(config, STAGE)
    paths = get_paths(config)
    if not paths.dataset_index.exists():
        raise FileNotFoundError(f"{paths.dataset_index} not found; run build_dataset_shards first.")
    index = pd.read_parquet(paths.dataset_index, columns=INDEX_COLUMNS)
    logger.info("Dataset index: %d objects", len(index))
    paths.labels_dir.mkdir(parents=True, exist_ok=True)
    flags = pd.DataFrame({"object_id": index["object_id"].to_numpy()})
    stats: Dict[str, object] = {"dataset_objects": len(index)}

    morph = config["labels"].get("morphology")
    if morph:
        path = paths.labels_dir / morph["file"]
        if not path.exists() and morph.get("url") and download:
            download_file(morph["url"], path, logger)
        if path.exists():
            labels = normalize_id_column(read_table(path), morph["id_column"], logger)
            labels = labels.drop_duplicates(morph["id_column"])
            matched = match_morphology(labels, index, morph)
            atomic_write_parquet(matched, paths.labels_dir / "morphology_labelled.parquet")
            hit = matched["match_type"] != "none"
            flags["label_morphology"] = flags["object_id"].isin(matched.loc[hit, "matched_object_id"].astype("int64"))
            missing = missing_breakdown(labels.loc[~hit.to_numpy(), morph["id_column"]].to_numpy(), paths.master_index)
            stats["morphology"] = {
                "labels": len(labels),
                "matched_by_id": int((matched["match_type"] == "id").sum()),
                "matched_by_position": int((matched["match_type"] == "position").sum()),
                "unmatched": int((~hit).sum()),
                **missing,
            }
            logger.info("Morphology labels: %s", stats["morphology"])
        else:
            logger.warning("Morphology catalogue %s not found (download disabled?)", path)

    desi = config["labels"].get("desi")
    if desi and desi.get("path"):
        labels = read_table(Path(desi["path"]))
        matched = match_positions(labels, index, desi)
        atomic_write_parquet(matched, paths.labels_dir / "desi_labelled.parquet")
        hit = matched["match_type"] != "none"
        flags["label_desi"] = flags["object_id"].isin(matched.loc[hit, "matched_object_id"].astype("int64"))
        stats["desi"] = {"labels": len(labels), "matched": int(hit.sum()),
                         "median_sep_arcsec": float(np.nanmedian(matched.loc[hit, "match_sep_arcsec"])) if hit.any() else None}
        logger.info("DESI labels: %s", stats["desi"])

    if len(flags.columns) > 1:
        atomic_write_parquet(flags, paths.labels_dir / "label_flags.parquet")
        apply_label_flags(paths.dataset_index, flags)
        for col in flags.columns[1:]:
            logger.info("%s: %d of %d dataset objects", col, int(flags[col].sum()), len(flags))
            stats[f"n_{col}"] = int(flags[col].sum())
    write_stats(config, STAGE, stats)
    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--root", default=None, help="Override paths.root.")
    parser.add_argument("--no-download", action="store_true", help="Do not download the label catalogue.")
    args = parser.parse_args(argv)
    run(load_config(args.config, root=args.root), download=not args.no_download)


if __name__ == "__main__":
    main()
