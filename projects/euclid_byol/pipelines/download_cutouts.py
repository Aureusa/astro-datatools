"""Stage 2: download VIS cutouts and preprocess them straight to uint8.

For each tile, every selected object gets a row in ``cutouts/tile_<t>.h5``
(see :mod:`..tilestore`). Worker threads fetch the cutout from the SAS cutout
service (slightly larger than needed), cut exactly ``size_pix`` pixels around
the source and apply the frozen preprocessing; only the uint8 result is kept.
The raw float32 cutout is kept for a small deterministic subset
(``cutouts.raw_subset_fraction``) in ``raw_subset/tile_<t>.h5``, which the
preprocess stage uses to check that stored images still match the function.

Resumable: re-running fetches only rows that are pending or failed. A failing
object is retried on each run until ``cutouts.max_attempts`` runs have tried
it, then marked permanently failed. SIGTERM (e.g. a Slurm time limit) and
Ctrl-C flush the current tile before exiting.

Split the tiles across several processes or Slurm array tasks with
``--num-parts N --part I``.

Usage (from the repository root)::

    python -m projects.euclid_byol.pipelines.download_cutouts --config projects/euclid_byol/configs/sample.yaml
"""
from __future__ import annotations

import argparse
import signal
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import get_paths, get_preprocessor, load_config, stage_logger, write_stats
from ..cutouts import EASCutoutClient, PermanentCutoutError, TransientCutoutError
from ..tilestore import (
    STATUS_BLANK,
    STATUS_FAILED,
    STATUS_FAILED_PERMANENT,
    STATUS_OK,
    RawSubsetStore,
    TileStore,
    in_raw_subset,
    tile_counts,
)

STAGE = "download_cutouts"
PIXEL_SCALE_DEG = 0.1 / 3600.0


@dataclass
class ObjectResult:
    row: int
    status: int
    image: Optional[np.ndarray] = None
    raw: Optional[np.ndarray] = None
    stats: Dict[str, float] = field(default_factory=dict)
    error: str = ""
    swapped: int = 0  # responses that were the cutout of a different position


class MisplacedCutout(TransientCutoutError):
    """The archive returned a cutout centred on a different position."""


def centre_offset_pix(data: np.ndarray, wcs, ra: float, dec: float) -> float:
    """Distance in pixels between (ra, dec) and the centre of the image."""
    x, y = wcs.world_to_pixel_values(ra, dec)
    ny, nx = data.shape
    return float(np.hypot(float(x) - (nx - 1) / 2.0, float(y) - (ny - 1) / 2.0))


def objects_for_tile(config: dict, part: pd.DataFrame) -> pd.DataFrame:
    """Selected objects of one tile, in the order they are stored in the tile file."""
    df = part[part["selected"]]
    limit = config["cutouts"].get("max_objects_per_tile")
    if limit and len(df) > limit:
        if config["cutouts"].get("sample", "random") == "largest":
            df = df.nlargest(int(limit), "segmentation_area")
        else:
            df = df.sample(n=int(limit), random_state=int(config["cutouts"].get("sample_seed", 0)))
    return df.sort_values("object_id").reset_index(drop=True)


def request_radius_deg(config: dict) -> float:
    c = config["cutouts"]
    return (c["size_pix"] / 2 + c["request_margin_pix"]) * PIXEL_SCALE_DEG


def fetch_centred(client, tile: pd.Series, ra: float, dec: float, config: dict):
    """Fetch a cutout and check it is centred on (ra, dec); re-request it if not.

    Under concurrent load the SAS cutout service sometimes answers a request
    with the cutout of another, simultaneous request (seen for ~9% of requests
    at 8 in flight and ~25% at 64; the file is internally consistent, just for
    the wrong position). The returned file's own WCS gives it away.

    :return: ``(data, wcs, swapped)``, with the number of misplaced responses seen.
    :raises MisplacedCutout: If every attempt returned a misplaced cutout.
    """
    prep = get_preprocessor(config)
    c = config["cutouts"]
    swapped = 0
    for _ in range(1 + int(c.get("recentre_retries", 5))):
        content = client.fetch(
            file_path=f"{tile['file_path'].rstrip('/')}/{tile['file_name']}",
            collection=config["archive"]["band"],
            obs_id=int(tile["tile_index"]),
            ra=ra, dec=dec, radius_deg=request_radius_deg(config),
        )
        data, wcs = prep.read_image_hdu(content)
        offset = centre_offset_pix(data, wcs, ra, dec)
        if offset <= float(c.get("max_centre_offset_pix", 2.0)):
            return data, wcs, swapped
        swapped += 1
    raise MisplacedCutout(f"archive returned cutouts of another position ({swapped}x, last {offset:.0f} px off)")


def process_object(client, tile: pd.Series, row: int, ra: float, dec: float, config: dict,
                   keep_raw: bool) -> ObjectResult:
    """Fetch, cut out and preprocess one object (runs in a worker thread).

    Same steps as ``preprocess_fits`` (read, ``extract_cutout``,
    ``preprocess_with_stats``), plus the centring check of :func:`fetch_centred`.
    """
    prep = get_preprocessor(config)
    swapped = 0
    try:
        data, wcs, swapped = fetch_centred(client, tile, ra, dec, config)
    except PermanentCutoutError as err:
        return ObjectResult(row, STATUS_FAILED_PERMANENT, error=str(err))
    except MisplacedCutout as err:
        return ObjectResult(row, STATUS_FAILED, error=str(err), swapped=1 + int(config["cutouts"].get("recentre_retries", 5)))
    except TransientCutoutError as err:
        return ObjectResult(row, STATUS_FAILED, error=str(err))
    except Exception as err:  # unreadable FITS, wrong pixel scale, ...
        return ObjectResult(row, STATUS_FAILED, error=f"read: {type(err).__name__}: {err}")
    try:
        raw = prep.extract_cutout(data, wcs, ra, dec)
        image, stats = prep.preprocess_with_stats(raw)
    except Exception as err:
        return ObjectResult(row, STATUS_FAILED, error=f"preprocess: {type(err).__name__}: {err}", swapped=swapped)
    status = STATUS_BLANK if stats["degenerate"] else STATUS_OK
    return ObjectResult(row, status, image=image, raw=raw.astype(np.float32) if keep_raw else None,
                        stats=stats, swapped=swapped)


def open_tile_store(config: dict, tile_index: int, objects: pd.DataFrame, logger) -> TileStore:
    """Open the tile file, or create it; check it belongs to this selection and preprocessing."""
    paths = get_paths(config)
    prep = get_preprocessor(config)
    path = paths.tile_file(tile_index)
    if path.exists():
        try:
            store = TileStore.open(path, "r+")
        except OSError as err:
            broken = path.with_name(path.name + ".corrupt")
            path.replace(broken)
            logger.warning("Tile file %s is unreadable (%s); moved to %s and starting the tile again",
                           path, err, broken.name)
        else:
            try:
                store.check_fingerprint(prep.VERSION, prep.FINGERPRINT)
            except Exception:
                store.close()
                raise
            if not np.array_equal(store.object_ids, objects["object_id"].to_numpy()):
                store.close()
                raise RuntimeError(
                    f"{path} holds a different object list than the current selection "
                    "(catalogue.selection or max_objects_per_tile changed?). Use a new paths.root, "
                    "or delete the tile file to rebuild it."
                )
            return store
    return TileStore.create(
        path, tile_index, objects["object_id"].to_numpy(), objects["right_ascension"].to_numpy(),
        objects["declination"].to_numpy(), size=prep.OUTPUT_SIZE,
        preprocess_version=prep.VERSION, preprocess_fingerprint=prep.FINGERPRINT,
        compression=config["cutouts"].get("compression"),
    )


def download_tile(config: dict, client, tile: pd.Series, logger) -> Counter:
    """Download all outstanding objects of one tile; returns outcome counts for this run."""
    paths = get_paths(config)
    c = config["cutouts"]
    tile_index = int(tile["tile_index"])
    if not paths.tile_file(tile_index).exists() and paths.tile_status_file(tile_index).exists():
        logger.info("Tile %d: already in the dataset shards (tile file deleted); skipping", tile_index)
        return Counter()
    objects = objects_for_tile(config, pd.read_parquet(paths.catalogue_part(tile_index)))
    store = open_tile_store(config, tile_index, objects, logger)
    raw_store = None
    outcome: Counter = Counter()
    try:
        rows = store.rows_to_fetch()
        if len(rows) == 0:
            store.update_complete()
            logger.info("Tile %d: complete (%d objects)", tile_index, len(store))
            return outcome
        logger.info("Tile %d: fetching %d of %d objects", tile_index, len(rows), len(store))
        ids = objects["object_id"].to_numpy()
        ras = objects["right_ascension"].to_numpy()
        decs = objects["declination"].to_numpy()
        keep_raw = {int(r): in_raw_subset(ids[r], c["raw_subset_fraction"]) for r in rows}
        if any(keep_raw.values()):
            raw_store = RawSubsetStore(paths.raw_file(tile_index), store.h5.attrs["size"])

        start = time.monotonic()
        pending = iter(rows)
        window = max(1, int(c["workers"])) * 4
        with ThreadPoolExecutor(max_workers=max(1, int(c["workers"]))) as pool:
            running = set()

            def submit_next(n):
                for _ in range(n):
                    r = next(pending, None)
                    if r is None:
                        return
                    running.add(pool.submit(process_object, client, tile, int(r), float(ras[r]),
                                            float(decs[r]), config, keep_raw[int(r)]))

            submit_next(window)
            done_count = 0
            while running:
                finished, running = wait(running, return_when=FIRST_COMPLETED)
                for fut in finished:
                    res = fut.result()
                    status = res.status
                    if status == STATUS_FAILED and store.attempts(res.row) + 1 >= c["max_attempts"]:
                        status = STATUS_FAILED_PERMANENT
                    store.write(res.row, status, image=res.image, stats=res.stats, error=res.error)
                    if res.raw is not None and raw_store is not None:
                        raw_store.add(ids[res.row], res.raw)
                    outcome[status] += 1
                    outcome["swapped"] += res.swapped
                    done_count += 1
                    if done_count % int(c["flush_every"]) == 0:
                        store.flush()
                        if raw_store is not None:
                            raw_store.flush()
                        rate = done_count / (time.monotonic() - start)
                        logger.info("Tile %d: %d/%d (%.1f/s; ok %d, failed %d, misplaced responses re-requested %d)",
                                    tile_index, done_count, len(rows), rate, outcome[STATUS_OK],
                                    outcome[STATUS_FAILED] + outcome[STATUS_FAILED_PERMANENT], outcome["swapped"])
                submit_next(len(finished))
        complete = store.update_complete()
        elapsed = time.monotonic() - start
        logger.info(
            "Tile %d: %s in %.0fs (%.1f/s): ok %d, blank %d, failed %d (retry next run), failed permanently %d; "
            "%d misplaced responses re-requested",
            tile_index, "complete" if complete else "incomplete", elapsed, len(rows) / max(elapsed, 1e-9),
            outcome[STATUS_OK], outcome[STATUS_BLANK], outcome[STATUS_FAILED], outcome[STATUS_FAILED_PERMANENT],
            outcome["swapped"],
        )
        errors = Counter(e for e in store.h5["error"].asstr()[:] if e)
        for message, n in errors.most_common(3):
            logger.info("Tile %d: %d x '%s'", tile_index, n, message)
        return outcome
    finally:
        store.close()
        if raw_store is not None:
            raw_store.close()


def tiles_for_run(config: dict, part: int = 0, num_parts: int = 1, limit_tiles: Optional[int] = None) -> pd.DataFrame:
    paths = get_paths(config)
    tiles = pd.read_parquet(paths.tiles)
    wanted = config["catalogue"].get("tiles")
    if wanted:
        tiles = tiles[tiles["tile_index"].isin([int(t) for t in wanted])]
    tiles = tiles[[paths.catalogue_part(t).exists() for t in tiles["tile_index"]]]
    tiles = tiles.sort_values("tile_index").reset_index(drop=True)
    tiles = tiles.iloc[part::num_parts]
    if limit_tiles:
        tiles = tiles.iloc[:limit_tiles]
    return tiles


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt(f"signal {signum}")


def run(config: dict, client=None, part: int = 0, num_parts: int = 1, limit_tiles: Optional[int] = None,
        dry_run: bool = False, logger=None) -> Dict[str, object]:
    logger = logger or stage_logger(config, STAGE)
    paths = get_paths(config)
    if not paths.tiles.exists():
        raise FileNotFoundError(f"{paths.tiles} not found; run query_catalogue first.")
    tiles = tiles_for_run(config, part, num_parts, limit_tiles)
    prep = get_preprocessor(config)
    logger.info("Preprocessing %s (fingerprint %s...); %d tiles in this run (part %d/%d)",
                prep.VERSION, prep.FINGERPRINT[:12], len(tiles), part, num_parts)

    if dry_run:
        n = sum(len(objects_for_tile(config, pd.read_parquet(paths.catalogue_part(t)))) for t in tiles["tile_index"])
        logger.info("Dry run: %d objects to download (%.3f TB of uint8 images)", n, n * prep.OUTPUT_SIZE ** 2 / 1e12)
        return {"objects": n}

    client = client or EASCutoutClient.from_config(config)
    previous = signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    totals: Counter = Counter()
    start = time.monotonic()
    interrupted = False
    try:
        for i, (_, tile) in enumerate(tiles.iterrows(), 1):
            logger.info("[%d/%d] tile %d", i, len(tiles), int(tile["tile_index"]))
            totals.update(download_tile(config, client, tile, logger))
    except KeyboardInterrupt:
        interrupted = True
        logger.warning("Interrupted; progress so far is saved and the next run resumes from it")
    finally:
        signal.signal(signal.SIGTERM, previous)

    elapsed = time.monotonic() - start
    stats = {
        "part": part,
        "num_parts": num_parts,
        "tiles": len(tiles),
        "interrupted": interrupted,
        "elapsed_s": round(elapsed, 1),
        "this_run": {
            "ok": totals[STATUS_OK],
            "blank": totals[STATUS_BLANK],
            "failed_retry_next_run": totals[STATUS_FAILED],
            "failed_permanent": totals[STATUS_FAILED_PERMANENT],
            "misplaced_responses_rerequested": totals["swapped"],
        },
        "totals_on_disk": summarize_tiles(config, tiles["tile_index"]),
    }
    logger.info("This run: %s in %.0fs", stats["this_run"], elapsed)
    logger.info("On disk: %s", stats["totals_on_disk"])
    suffix = f"_part{part}of{num_parts}" if num_parts > 1 else ""
    write_stats(config, STAGE + suffix, stats)
    return stats


def summarize_tiles(config: dict, tile_indices) -> Dict[str, int]:
    """Status counts summed over the tiles (tile files, or status files of deleted ones)."""
    paths = get_paths(config)
    totals: Counter = Counter()
    for t in tile_indices:
        counts = tile_counts(paths, t)
        if counts is not None:
            totals["tiles_complete"] += counts.pop("complete")
            totals["tiles_started"] += 1
            totals.update(counts)
    return dict(totals)


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--root", default=None, help="Override paths.root.")
    parser.add_argument("--part", type=int, default=0, help="This process's share of the tiles (0-based).")
    parser.add_argument("--num-parts", type=int, default=1, help="Number of processes sharing the tiles.")
    parser.add_argument("--limit-tiles", type=int, default=None, help="Process at most N tiles.")
    parser.add_argument("--dry-run", action="store_true", help="Only count what would be downloaded.")
    args = parser.parse_args(argv)
    if not 0 <= args.part < args.num_parts:
        parser.error("--part must be in [0, --num-parts)")
    stats = run(load_config(args.config, root=args.root), part=args.part, num_parts=args.num_parts,
                limit_tiles=args.limit_tiles, dry_run=args.dry_run)
    if stats.get("interrupted"):
        raise SystemExit(3)  # progress is saved; run again to resume


if __name__ == "__main__":
    main()
