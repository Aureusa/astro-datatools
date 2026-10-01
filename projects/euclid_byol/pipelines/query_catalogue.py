"""Stage 1: build the master index of Q1 detections from the MER catalogue.

1. ``catalogue/tiles.parquet``: one row per MER tile with the VIS mosaic path
   (for cutouts), the id of the tile's catalogue file
   (``basic_download_data_oid``) and its row count.
2. ``catalogue/parts/tile_<t>.parquet``: the catalogue rows of each tile, pulled
   with one ADQL query per tile (never the full 30M-row table in one request).
   A part is only written when its row count matches the archive's, so a
   re-run just fetches the missing tiles.
3. A ``selected`` column from the ``catalogue.selection`` expressions (which
   objects get downloaded), then ``catalogue/master_index.parquet`` with all
   parts concatenated.

Usage (from the repository root)::

    python -m projects.euclid_byol.pipelines.query_catalogue --config projects/euclid_byol/configs/pipeline.yaml
    python -m projects.euclid_byol.pipelines.query_catalogue --reselect   # re-apply the selection only
"""
from __future__ import annotations

import argparse
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional

import pandas as pd

from ..config import (
    atomic_write_parquet,
    get_paths,
    load_config,
    stage_logger,
    write_stats,
)

STAGE = "query_catalogue"
BYTES_PER_IMAGE = 224 * 224
#: Columns kept as 64-bit integers; everything else becomes float64 (NaN for nulls)
#: so that every tile part has the same schema.
INT_COLUMNS = {"object_id": "int64", "tile_index": "int64", "gaia_id": "Int64"}
_TOKEN = re.compile(r"^[A-Za-z0-9_.\-]+$")

QueryFn = Callable[[str], pd.DataFrame]


def _token(value: str, what: str) -> str:
    if not _TOKEN.match(str(value)):
        raise ValueError(f"Invalid {what} '{value}'.")
    return str(value)


def make_query_fn(config: dict) -> QueryFn:
    """ADQL runner using one astroquery client per thread, with retries."""
    from astroquery.esa.euclid import EuclidClass

    from astro_datatools.surveys import EuclidSurvey

    local = threading.local()

    def run(adql: str) -> pd.DataFrame:
        if not hasattr(local, "survey"):
            local.survey = EuclidSurvey(client=EuclidClass(environment="PDR"))
            local.survey.query_retries = 4
            local.survey.retry_wait = 10.0
        return local.survey._run_query(adql).to_pandas()

    return run


def normalise_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Give every catalogue part the same column types."""
    out = {}
    for col in df.columns:
        target = INT_COLUMNS.get(col)
        if target is not None:
            out[col] = df[col].astype(target)
        else:
            out[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    return pd.DataFrame(out)


def fetch_tiles(query: QueryFn, release: str, band: str) -> pd.DataFrame:
    """Tile table: VIS mosaic per tile joined with the tile's catalogue file."""
    release, band = _token(release, "release"), _token(band, "band")
    mosaics = query(
        "SELECT tile_index, file_name, file_path, ra, dec FROM sedm.mosaic_product "
        f"WHERE filter_name='{band}' AND release_name='{release}'"
    )
    mosaics = mosaics[mosaics["file_name"].astype(str).str.contains("BGSUB")]
    # If a tile was reprocessed, keep the latest file (names end in a timestamp).
    mosaics = mosaics.sort_values("file_name").drop_duplicates("tile_index", keep="last")

    catalogues = query(
        "SELECT basic_download_data_oid, tile_index_list FROM sedm.basic_download_data "
        f"WHERE product_type='dpdMerFinalCatalog' AND release_name='{release}'"
    )
    tile_lists = catalogues["tile_index_list"].astype(str).str.findall(r"\d+")
    if (tile_lists.str.len() != 1).any():
        raise ValueError("Some MER catalogue files cover more than one tile; cannot map objects to tiles.")
    catalogues = pd.DataFrame({
        "catalogue_oid": catalogues["basic_download_data_oid"].astype("int64"),
        "tile_index": tile_lists.str[0].astype("int64"),
    })
    counts = query(
        "SELECT basic_download_data_oid, COUNT(*) AS n FROM catalogue.mer_catalogue "
        "GROUP BY basic_download_data_oid"
    ).rename(columns={"basic_download_data_oid": "catalogue_oid", "n": "n_catalogue"})
    counts = counts.astype({"catalogue_oid": "int64", "n_catalogue": "int64"})

    tiles = mosaics.astype({"tile_index": "int64"}).merge(catalogues, on="tile_index", how="inner")
    tiles = tiles.merge(counts, on="catalogue_oid", how="left")
    tiles["file_name"] = tiles["file_name"].astype(str)
    tiles["file_path"] = tiles["file_path"].astype(str)
    return tiles.sort_values("tile_index").reset_index(drop=True)


def build_tile_query(config: dict, catalogue_oid: int) -> str:
    cat = config["catalogue"]
    columns = [_token(c, "column") for c in cat["columns"]]
    if "object_id" not in columns:
        columns.insert(0, "object_id")
    return (
        f"SELECT {', '.join(columns)} FROM {_token(cat['table'], 'table')} "
        f"WHERE basic_download_data_oid = {int(catalogue_oid)}"
    )


def apply_selection(df: pd.DataFrame, expressions: List[str]) -> pd.Series:
    """Boolean mask of rows passing every selection expression."""
    mask = pd.Series(True, index=df.index)
    for expr in expressions or []:
        result = df.eval(expr)
        mask &= pd.Series(result, index=df.index).fillna(False).astype(bool)
    return mask


def query_tile(query: QueryFn, config: dict, tile: pd.Series) -> pd.DataFrame:
    """Catalogue rows of one tile, checked against the expected count."""
    df = query(build_tile_query(config, tile["catalogue_oid"]))
    expected = tile.get("n_catalogue")
    if expected is not None and not pd.isna(expected) and len(df) != int(expected):
        raise RuntimeError(
            f"Tile {tile['tile_index']}: got {len(df)} rows, archive reports {int(expected)} "
            "(truncated result?)."
        )
    df["tile_index"] = int(tile["tile_index"])
    df = normalise_dtypes(df).sort_values("object_id").reset_index(drop=True)
    df["selected"] = apply_selection(df, config["catalogue"]["selection"])
    return df


def select_tiles(config: dict, tiles: pd.DataFrame) -> pd.DataFrame:
    wanted = config["catalogue"].get("tiles")
    if wanted:
        missing = set(int(t) for t in wanted) - set(tiles["tile_index"])
        if missing:
            raise ValueError(f"Tiles not in {config['archive']['release']}: {sorted(missing)}")
        tiles = tiles[tiles["tile_index"].isin([int(t) for t in wanted])]
    return tiles


def build_master_index(paths, tile_indices) -> int:
    """Concatenate the tile parts into ``master_index.parquet`` (streamed; low memory)."""
    import pyarrow.parquet as pq

    tmp = paths.master_index.with_name(paths.master_index.name + ".tmp")
    writer, n_rows = None, 0
    try:
        for t in sorted(tile_indices):
            part = paths.catalogue_part(t)
            if not part.exists():
                continue
            table = pq.read_table(part)
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema)
            writer.write_table(table.cast(writer.schema))
            n_rows += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    if writer is not None:
        tmp.replace(paths.master_index)
    return n_rows


def run(config: dict, query: Optional[QueryFn] = None, reselect: bool = False,
        refresh_tiles: bool = False, logger=None) -> Dict[str, object]:
    """Run the stage; returns the counts also written to ``stats/query_catalogue.json``."""
    logger = logger or stage_logger(config, STAGE)
    paths = get_paths(config)
    paths.catalogue_parts.mkdir(parents=True, exist_ok=True)
    query = query or make_query_fn(config)

    if refresh_tiles or not paths.tiles.exists():
        logger.info("Querying the tile table (%s, %s)", config["archive"]["release"], config["archive"]["band"])
        atomic_write_parquet(fetch_tiles(query, config["archive"]["release"], config["archive"]["band"]), paths.tiles)
    all_tiles = pd.read_parquet(paths.tiles)
    tiles = select_tiles(config, all_tiles)
    logger.info("%d tiles in the release, %d requested; %d catalogue rows expected",
                len(all_tiles), len(tiles), int(tiles["n_catalogue"].sum()))

    if reselect:
        for t in tiles["tile_index"]:
            part = paths.catalogue_part(t)
            if part.exists():
                df = pd.read_parquet(part)
                df["selected"] = apply_selection(df, config["catalogue"]["selection"])
                atomic_write_parquet(df, part)
        logger.info("Re-applied the selection to existing parts")

    todo = [row for _, row in tiles.iterrows() if not paths.catalogue_part(row["tile_index"]).exists()]
    logger.info("%d tiles already queried, %d to query", len(tiles) - len(todo), len(todo))
    failed = []
    with ThreadPoolExecutor(max_workers=max(1, int(config["catalogue"]["query_workers"]))) as pool:
        futures = {pool.submit(query_tile, query, config, row): row for row in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            tile_index = int(futures[fut]["tile_index"])
            try:
                df = fut.result()
            except Exception as err:  # keep going; the tile is retried on the next run
                failed.append(tile_index)
                logger.error("Tile %d failed: %s", tile_index, err)
                continue
            atomic_write_parquet(df, paths.catalogue_part(tile_index))
            logger.info("[%d/%d] tile %d: %d rows, %d selected",
                        i, len(todo), tile_index, len(df), int(df["selected"].sum()))

    n_total = n_selected = 0
    done = []
    for t in tiles["tile_index"]:
        part = paths.catalogue_part(t)
        if part.exists():
            sel = pd.read_parquet(part, columns=["selected"])["selected"]
            n_total += len(sel)
            n_selected += int(sel.sum())
            done.append(int(t))
    n_master = build_master_index(paths, done)

    est_tb = n_selected * BYTES_PER_IMAGE / 1e12
    budget = config["catalogue"].get("storage_budget_tb")
    logger.info("Queried %d/%d tiles: %d detections, %d selected (%.1f%%); dataset estimate %.3f TB",
                len(done), len(tiles), n_total, n_selected, 100 * n_selected / max(n_total, 1), est_tb)
    if config["cutouts"].get("max_objects_per_tile"):
        logger.info("max_objects_per_tile=%s limits the download to at most %d objects",
                    config["cutouts"]["max_objects_per_tile"],
                    len(done) * int(config["cutouts"]["max_objects_per_tile"]))
    elif budget and est_tb > budget:
        logger.warning("Selected objects need ~%.2f TB, above the %.2f TB budget; tighten catalogue.selection.",
                       est_tb, budget)
    if failed:
        logger.warning("%d tiles failed and will be retried on the next run: %s", len(failed), failed)

    stats = {
        "tiles_in_release": len(all_tiles),
        "tiles_requested": len(tiles),
        "tiles_queried": len(done),
        "tiles_failed": failed,
        "detections": n_total,
        "selected": n_selected,
        "master_index_rows": n_master,
        "estimated_dataset_tb": est_tb,
        "selection": config["catalogue"]["selection"],
    }
    write_stats(config, STAGE, stats)
    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Pipeline YAML (default configs/pipeline.yaml).")
    parser.add_argument("--root", default=None, help="Override paths.root.")
    parser.add_argument("--reselect", action="store_true", help="Re-apply catalogue.selection to existing parts.")
    parser.add_argument("--refresh-tiles", action="store_true", help="Re-query the tile table.")
    args = parser.parse_args(argv)
    run(load_config(args.config, root=args.root), reselect=args.reselect, refresh_tiles=args.refresh_tiles)


if __name__ == "__main__":
    main()
