"""End-to-end test of the Euclid BYOL pipeline stages against a fake archive (no network)."""
import io
import re
import threading
from collections import Counter

import h5py
import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from projects.euclid_byol import preprocess_v1
from projects.euclid_byol.config import CONFIG_DIR, get_paths, load_config
from projects.euclid_byol.cutouts import PermanentCutoutError, TransientCutoutError
from projects.euclid_byol.dataset import EuclidShardDataset
from projects.euclid_byol.pipelines import (
    build_dataset_shards,
    crossmatch_labels,
    download_cutouts,
    estimate_storage,
    inspect_cutouts,
    preprocess,
    query_catalogue,
    status,
)
from projects.euclid_byol.tilestore import (
    STATUS_BLANK,
    STATUS_FAILED,
    STATUS_FAILED_PERMANENT,
    STATUS_PENDING,
    FingerprintMismatch,
    TileStore,
    in_raw_subset,
    raw_subset_mask,
)

TILES = {1: 11, 2: 12, 3: 13}  # tile_index -> catalogue oid
N_PER_TILE = 12


def catalogue_for(tile):
    k = np.arange(N_PER_TILE)
    return pd.DataFrame({
        "object_id": tile * 1000 + k,
        "right_ascension": 50.0 + tile * 0.2 + k * 0.002,
        "declination": -27.0 + k * 0.002,
        "vis_det": np.where(k == 0, 0, 1),
        "spurious_flag": np.where(k == 1, 1, 0),
        # Unset (NaN) except for one star.
        "point_like_flag": np.where(k == 2, 1.0, np.nan),
        "segmentation_area": 100.0 + 10 * k,
        "det_quality_flag": np.zeros(N_PER_TILE, dtype=np.int16),
    })


def fake_query(adql):
    if "sedm.mosaic_product" in adql:
        return pd.DataFrame({
            "tile_index": list(TILES),
            "file_name": [f"EUC_MER_BGSUB-MOSAIC-VIS_TILE{t}_00.fits" for t in TILES],
            "file_path": [f"/repo/{t}/VIS" for t in TILES],
            "ra": [50.0 + 0.2 * t for t in TILES],
            "dec": [-27.0] * len(TILES),
        })
    if "sedm.basic_download_data" in adql:
        return pd.DataFrame({"basic_download_data_oid": list(TILES.values()),
                             "tile_index_list": [f"({t})" for t in TILES]})
    if "GROUP BY" in adql:
        return pd.DataFrame({"basic_download_data_oid": list(TILES.values()), "n": [N_PER_TILE] * len(TILES)})
    oid = int(re.search(r"basic_download_data_oid = (\d+)", adql).group(1))
    tile = {v: k for k, v in TILES.items()}[oid]
    return catalogue_for(tile)


def fits_cutout(ra, dec, radius_deg, seed, blank=False):
    n = int(round(2 * radius_deg * 3600 / 0.1))
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra, dec]
    w.wcs.crpix = [n / 2 + 0.3, n / 2 + 0.6]
    w.wcs.cdelt = [-0.1 / 3600, 0.1 / 3600]
    rng = np.random.default_rng(seed)
    image = rng.normal(0.0, 1.0, (n, n))
    x0, y0 = w.world_to_pixel_values(ra, dec)
    yy, xx = np.mgrid[:n, :n]
    image += 80 * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * 5.0 ** 2))
    if blank:
        image[:] = 0.0
    buf = io.BytesIO()
    fits.PrimaryHDU(image.astype(np.float32), header=w.to_header()).writeto(buf)
    return buf.getvalue()


class FakeClient:
    """Serves synthetic cutouts; some objects fail transiently (once), permanently, or are blank."""

    def __init__(self, transient=(), permanent=(), blank=(), swap_first=(), swap_always=()):
        self.coords = {}
        for tile in TILES:
            cat = catalogue_for(tile)
            for oid, ra, dec in zip(cat["object_id"], cat["right_ascension"], cat["declination"]):
                self.coords[(round(ra, 8), round(dec, 8))] = int(oid)
        self.transient, self.permanent, self.blank = set(transient), set(permanent), set(blank)
        self.swap_first, self.swap_always = set(swap_first), set(swap_always)
        self.calls = Counter()
        self.lock = threading.Lock()

    def fetch(self, file_path, collection, obs_id, ra, dec, radius_deg):
        oid = self.coords[(round(ra, 8), round(dec, 8))]
        assert file_path == f"/repo/{obs_id}/VIS/EUC_MER_BGSUB-MOSAIC-VIS_TILE{obs_id}_00.fits"
        with self.lock:
            self.calls[oid] += 1
            first = self.calls[oid] == 1
        if oid in self.permanent:
            raise PermanentCutoutError("HTTP 404")
        if oid in self.transient and first:
            raise TransientCutoutError("HTTP 500 (after 6 attempts)")
        if oid in self.swap_always or (oid in self.swap_first and first):
            # The archive bug: the cutout of another position, 6 arcsec away.
            return fits_cutout(ra + 6 / 3600, dec, radius_deg, seed=oid + 1)
        return fits_cutout(ra, dec, radius_deg, seed=oid, blank=oid in self.blank)


@pytest.fixture
def config(tmp_path):
    cfg = load_config(CONFIG_DIR / "sample.yaml", root=str(tmp_path / "root"))
    cfg["catalogue"]["tiles"] = [1, 2]
    cfg["catalogue"]["columns"] = list(catalogue_for(1).columns)
    cfg["catalogue"]["selection"] = ["vis_det == 1", "spurious_flag == 0", "point_like_flag != 1"]
    cfg["cutouts"].update(workers=4, max_objects_per_tile=None, raw_subset_fraction=0.5,
                          max_attempts=3, flush_every=5)
    cfg["dataset"].update(shard_size=7, delete_tile_files=False)
    cfg["labels"]["morphology"]["url"] = None
    return cfg


def run_query(config):
    return query_catalogue.run(config, query=fake_query)


def test_query_catalogue_builds_tiles_parts_and_selection(config):
    stats = run_query(config)
    paths = get_paths(config)
    tiles = pd.read_parquet(paths.tiles)
    assert list(tiles["tile_index"]) == [1, 2, 3]
    assert list(tiles["catalogue_oid"]) == [11, 12, 13]
    assert stats["tiles_queried"] == 2 and stats["detections"] == 24
    # Objects 0 (NIR-only), 1 (spurious) and 2 (star) of each tile are dropped.
    assert stats["selected"] == 2 * (N_PER_TILE - 3)
    master = pd.read_parquet(paths.master_index)
    assert len(master) == 24 and master["selected"].sum() == 18
    assert master["object_id"].dtype == np.int64


def test_query_catalogue_resumes_and_reselects(config):
    run_query(config)
    calls = []

    def counting_query(adql):
        calls.append(adql)
        return fake_query(adql)

    query_catalogue.run(config, query=counting_query)
    assert calls == []  # everything was already on disk
    config["catalogue"]["selection"] = config["catalogue"]["selection"] + ["segmentation_area >= 200"]
    stats = query_catalogue.run(config, query=counting_query, reselect=True)
    assert stats["selected"] == 2 * 2  # k = 10, 11
    assert calls == []


def test_query_rejects_truncated_results(config):
    def short_query(adql):
        df = fake_query(adql)
        return df.iloc[:-1] if "WHERE basic_download_data_oid" in adql else df

    stats = query_catalogue.run(config, query=short_query)
    assert sorted(stats["tiles_failed"]) == [1, 2]
    assert not get_paths(config).catalogue_part(1).exists()


def test_full_pipeline(config):
    run_query(config)
    paths = get_paths(config)
    transient, permanent, blank = {1004, 2005}, {1006}, {2007}
    client = FakeClient(transient=transient, permanent=permanent, blank=blank)

    # --- download, first run -------------------------------------------------------
    stats = download_cutouts.run(config, client=client)
    assert stats["this_run"] == {"ok": 14, "blank": 1, "failed_retry_next_run": 2, "failed_permanent": 1,
                                 "misplaced_responses_rerequested": 0}
    with TileStore.open(paths.tile_file(1)) as store:
        assert len(store) == 9 and not store.complete
        status_by_id = dict(zip(store.object_ids, store.status))
        assert status_by_id[1004] == STATUS_FAILED and status_by_id[1006] == STATUS_FAILED_PERMANENT
        assert "404" in store.h5["error"].asstr()[list(store.object_ids).index(1006)]
    with TileStore.open(paths.tile_file(2)) as store:
        assert dict(zip(store.object_ids, store.status))[2007] == STATUS_BLANK

    # Incomplete tiles are not sharded yet.
    shard_stats = build_dataset_shards.run(config)
    assert shard_stats["images"] == 0 and sorted(shard_stats["tiles_incomplete"]) == [1, 2]

    # --- download, second run: only the transient failures are fetched again -------
    before = dict(client.calls)
    stats = download_cutouts.run(config, client=client)
    fetched = {oid for oid in client.calls if client.calls[oid] != before.get(oid, 0)}
    assert fetched == transient
    assert stats["this_run"]["ok"] == 2
    assert stats["totals_on_disk"]["tiles_complete"] == 2

    # --- resume after an interruption: pending rows only ---------------------------
    with TileStore.open(paths.tile_file(2), "r+") as store:
        rows = [0, 1]
        for r in rows:
            store.h5["status"][r] = STATUS_PENDING
        store.h5.attrs["complete"] = False
        reset_ids = set(store.object_ids[rows])
    before = dict(client.calls)
    download_cutouts.run(config, client=client)
    assert {oid for oid in client.calls if client.calls[oid] != before.get(oid, 0)} == reset_ids

    # --- stored images match the preprocessing of the raw subset -------------------
    verify = preprocess.verify(config)
    assert verify["passed"] and verify["raw_subset_checked"] > 0

    # --- shards --------------------------------------------------------------------
    shard_stats = build_dataset_shards.run(config)
    n_ok = 16  # 18 selected - 1 permanent failure - 1 blank
    assert shard_stats["images"] == n_ok and shard_stats["shards"] == 3  # 7 + 7 + 2
    index = pd.read_parquet(paths.dataset_index)
    assert len(index) == n_ok and index["object_id"].is_unique
    assert {"right_ascension", "segmentation_area", "noise_sigma", "shard", "shard_row"} <= set(index.columns)
    assert not ({1006, 2007} & set(index["object_id"]))
    for tile in (1, 2):
        with TileStore.open(paths.tile_file(tile)) as store:
            row_of = {int(o): i for i, o in enumerate(store.object_ids)}
            for _, rec in index[index["tile_index"] == tile].iterrows():
                with h5py.File(paths.dataset_dir / rec["shard"], "r") as shard:
                    assert shard["object_id"][rec["shard_row"]] == rec["object_id"]
                    assert np.array_equal(shard["images"][rec["shard_row"]],
                                          store.h5["images"][row_of[rec["object_id"]]])
                    assert shard.attrs["preprocess_fingerprint"] == preprocess_v1.FINGERPRINT

    # Re-running adds nothing; a half-written append is truncated on the next run.
    assert build_dataset_shards.run(config)["tiles_added_this_run"] == []
    last = paths.shard_file(2)
    with h5py.File(last, "r+") as h5:
        h5["images"].resize((5, 224, 224))
        h5["object_id"].resize((5,))
    assert build_dataset_shards.run(config)["images"] == n_ok
    with h5py.File(last, "r") as h5:
        assert h5["images"].shape[0] == 2

    # --- labels ----------------------------------------------------------------------
    cat1 = catalogue_for(1)
    labels = pd.DataFrame({
        "object_id": [1003, 1005, 999999, 888888],
        "right_ascension": [cat1.right_ascension[3], cat1.right_ascension[5],
                            cat1.right_ascension[7] + 0.1 / 3600, 10.0],
        "declination": [cat1.declination[3], cat1.declination[5], cat1.declination[7], 10.0],
        "smooth_fraction": [0.1, 0.2, 0.3, 0.4],
    })
    paths.labels_dir.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(paths.labels_dir / "morphology_catalogue.parquet")
    desi = pd.DataFrame({"ra": [cat1.right_ascension[8] + 0.5 / 3600, 0.0],
                         "dec": [cat1.declination[8], 0.0], "bar": [1, 0]})
    desi.to_csv(paths.labels_dir / "desi.csv", index=False)
    config["labels"]["desi"]["path"] = str(paths.labels_dir / "desi.csv")

    stats = crossmatch_labels.run(config, download=False)
    assert stats["morphology"]["matched_by_id"] == 2
    assert stats["morphology"]["matched_by_position"] == 1
    assert stats["morphology"]["unmatched"] == 1
    assert stats["desi"]["matched"] == 1
    matched = pd.read_parquet(paths.labels_dir / "morphology_labelled.parquet")
    row = matched[matched["object_id"] == 999999].iloc[0]
    assert row["matched_object_id"] == 1007 and row["match_type"] == "position"
    index = pd.read_parquet(paths.dataset_index)
    assert set(index.loc[index["label_morphology"], "object_id"]) == {1003, 1005, 1007}
    assert set(index.loc[index["label_desi"], "object_id"]) == {1008}

    # Rebuilding the index keeps the label flags.
    build_dataset_shards.build_index(config)
    assert pd.read_parquet(paths.dataset_index)["label_morphology"].sum() == 3

    # --- reading the dataset -----------------------------------------------------------
    ds = EuclidShardDataset(paths.dataset_dir)
    image, oid = ds[3]
    assert len(ds) == n_ok and image.shape == (224, 224) and image.dtype == np.uint8
    assert oid == index["object_id"].iloc[3]
    assert len(EuclidShardDataset(paths.dataset_dir, exclude=["label_morphology"])) == n_ok - 3
    assert len(EuclidShardDataset(paths.dataset_dir, only=["label_desi"])) == 1

    summary = status.collect(config)
    assert summary["downloads"]["ok"] == n_ok and summary["dataset"]["images"] == n_ok


def test_verify_detects_changed_images(config):
    run_query(config)
    config["cutouts"]["raw_subset_fraction"] = 1.0
    download_cutouts.run(config, client=FakeClient())
    paths = get_paths(config)
    with TileStore.open(paths.tile_file(1), "r+") as store:
        store.h5["images"][0] = store.h5["images"][0] // 2
    stats = preprocess.verify(config)
    assert not stats["passed"] and stats["mismatches"] == 1


def test_refuses_to_mix_preprocessing_fingerprints(config):
    run_query(config)
    download_cutouts.run(config, client=FakeClient(transient={1004}))
    paths = get_paths(config)
    with TileStore.open(paths.tile_file(1), "r+") as store:
        store.h5.attrs["preprocess_fingerprint"] = "0" * 64
    with pytest.raises(FingerprintMismatch):
        download_cutouts.run(config, client=FakeClient())
    assert preprocess.verify(config)["files_wrong_fingerprint"] == ["tile_1.h5"]


def test_refuses_changed_selection_for_existing_tile_file(config):
    run_query(config)
    download_cutouts.run(config, client=FakeClient())
    config["cutouts"]["max_objects_per_tile"] = 3
    with pytest.raises(RuntimeError, match="different object list"):
        download_cutouts.run(config, client=FakeClient())


def test_sample_limit_and_largest_strategy(config):
    run_query(config)
    config["cutouts"].update(max_objects_per_tile=4, sample="largest")
    part = pd.read_parquet(get_paths(config).catalogue_part(1))
    objects = download_cutouts.objects_for_tile(config, part)
    assert list(objects["object_id"]) == [1008, 1009, 1010, 1011]
    config["cutouts"]["sample"] = "random"
    a = download_cutouts.objects_for_tile(config, part)
    b = download_cutouts.objects_for_tile(config, part)
    assert len(a) == 4 and a["object_id"].is_monotonic_increasing and a.equals(b)


def test_preprocess_apply_on_fits_files(config, tmp_path):
    paths = []
    for i in range(3):
        path = tmp_path / f"img{i}.fits"
        path.write_bytes(fits_cutout(10.0 + i, 5.0, 12.0 / 3600, seed=i))
        paths.append(str(path))
    out = tmp_path / "applied.h5"
    stats = preprocess.apply(config, paths, str(out))
    assert stats["images"] == 3
    with h5py.File(out, "r") as h5:
        assert h5["images"].shape == (3, 224, 224)
        assert h5.attrs["preprocess_fingerprint"] == preprocess_v1.FINGERPRINT
        expected, _, _ = preprocess_v1.preprocess_fits(paths[1])
        assert np.array_equal(h5["images"][1], expected)


def test_delete_tile_files_keeps_status_and_is_not_redownloaded(config):
    run_query(config)
    config["cutouts"]["raw_subset_fraction"] = 1.0
    config["dataset"]["delete_tile_files"] = True
    paths = get_paths(config)
    client = FakeClient(permanent={1006})
    download_cutouts.run(config, client=client)
    stats = build_dataset_shards.run(config)
    assert stats["images"] == 17
    for tile in (1, 2):
        assert not paths.tile_file(tile).exists()
        assert paths.tile_status_file(tile).exists()
    status_1 = pd.read_parquet(paths.tile_status_file(1))
    assert len(status_1) == 9 and status_1.loc[status_1["object_id"] == 1006, "status"].item() == STATUS_FAILED_PERMANENT

    # A new download run must not fetch the sharded tiles again.
    before = sum(client.calls.values())
    stats = download_cutouts.run(config, client=client)
    assert sum(client.calls.values()) == before
    assert stats["totals_on_disk"]["ok"] == 17 and stats["totals_on_disk"]["tiles_complete"] == 2

    # Verification and inspection fall back to the shards.
    verify = preprocess.verify(config)
    assert verify["passed"] and verify["raw_subset_checked"] == 17 and verify["shards"] == 3
    images, ids, _ = inspect_cutouts.load_ok_images(paths)
    assert len(images) == 17 and 1006 not in set(ids)
    summary = status.collect(config)
    assert summary["downloads"]["ok"] == 17 and summary["downloads"]["tile_file_deleted"] == 2
    assert summary["disk_gb"]["total"] > 0


def test_leftover_tile_file_is_deleted_on_next_shard_run(config):
    run_query(config)
    paths = get_paths(config)
    download_cutouts.run(config, client=FakeClient())
    build_dataset_shards.run(config)  # delete_tile_files is False in the fixture
    config["dataset"]["delete_tile_files"] = True
    stats = build_dataset_shards.run(config)
    assert stats["tiles_added_this_run"] == [] and stats["images"] == 18
    assert not paths.tile_file(1).exists() and paths.tile_status_file(1).exists()


def test_raw_subset_mask_matches_scalar_version():
    ids = np.array([-514050782276575307, -518997019275521760, 0, 1, 2 ** 62, -1] + list(range(-5000, 5000)))
    for fraction in (0.0, 0.001, 0.1, 0.5, 1.0):
        assert list(raw_subset_mask(ids, fraction)) == [in_raw_subset(i, fraction) for i in ids]
    assert 0.08 < raw_subset_mask(np.arange(-50000, 50000), 0.1).mean() < 0.12


def test_estimate_storage_writes_nothing(config, capsys):
    root = get_paths(config).root
    summary = estimate_storage.run(config, query=fake_query)
    assert not root.exists()
    assert summary["tiles"] == 2 and summary["detections"] == 24 and summary["selected"] == 18
    image_bytes = 224 * 224 + estimate_storage.HDF5_OVERHEAD_PER_IMAGE
    assert summary["images_uncompressed_tb"] == pytest.approx(18 * image_bytes / 1e12)
    cuts = summary["selected_by_min_segmentation_area"]
    assert cuts["0"]["objects"] == 18 and cuts["200"]["objects"] == 4  # k = 10, 11 in both tiles
    assert summary["raw_subset_gb_by_fraction"]["0.0"] == 0
    assert "selected: 18" in capsys.readouterr().out

    # Same numbers from catalogue parts already on disk, without any query.
    run_query(config)

    def no_query(adql):
        raise AssertionError("should not query")

    assert estimate_storage.run(config, query=no_query)["selected"] == 18


def test_estimate_peak_disk_depends_on_deleting_tile_files(config):
    config["cutouts"]["compression"] = config["dataset"]["compression"] = None
    config["dataset"]["delete_tile_files"] = False
    keep = estimate_storage.run(config, query=fake_query)
    config["dataset"]["delete_tile_files"] = True
    delete = estimate_storage.run(config, query=fake_query)
    images = keep["images_tb"]
    assert keep["peak_tb"] - delete["peak_tb"] == pytest.approx(images - images / 18 * 9)  # all but one tile
    assert delete["final_tb"] < keep["final_tb"]


def test_misplaced_cutouts_are_rerequested(config):
    run_query(config)
    config["cutouts"]["recentre_retries"] = 2
    client = FakeClient(swap_first={1003, 2004}, swap_always={1005})
    stats = download_cutouts.run(config, client=client)
    assert stats["this_run"]["ok"] == 17 and stats["this_run"]["failed_retry_next_run"] == 1
    assert stats["this_run"]["misplaced_responses_rerequested"] == 2 + 3
    assert client.calls[1003] == 2 and client.calls[1005] == 3
    paths = get_paths(config)
    with TileStore.open(paths.tile_file(1)) as store:
        row = list(store.object_ids).index(1005)
        assert store.status[row] == STATUS_FAILED
        assert "another position" in store.h5["error"].asstr()[row]
        # The re-requested object got its own cutout, identical to a clean fetch.
        row = list(store.object_ids).index(1003)
        cat = catalogue_for(1).set_index("object_id").loc[1003]
        clean = fits_cutout(cat.right_ascension, cat.declination, download_cutouts.request_radius_deg(config), 1003)
        expected, _, _ = preprocess_v1.preprocess_fits(clean, cat.right_ascension, cat.declination)
        assert np.array_equal(store.h5["images"][row], expected)


def test_crossmatch_matches_by_id_when_label_ids_are_strings(config):
    # Regression test: the real Walmsley et al. Zenodo parquet stores object_id as a
    # string even though it is the same int64 id as our catalogue. Reproduce that
    # exactly (not pandas' int64 inference from a Python int list) so an id match
    # must survive a string/int64 dtype mismatch, not silently fall back to a
    # (slower, and for a near-duplicate position, potentially wrong) positional match.
    run_query(config)
    download_cutouts.run(config, client=FakeClient())
    build_dataset_shards.run(config)
    paths = get_paths(config)
    cat1 = catalogue_for(1)
    labels = pd.DataFrame({
        "object_id": np.array([1003, 1005, 999999], dtype=np.int64).astype(str),
        "right_ascension": [cat1.right_ascension[3], cat1.right_ascension[5], 10.0],
        "declination": [cat1.declination[3], cat1.declination[5], 10.0],
    })
    assert labels["object_id"].dtype == object  # i.e. Python str, as read from the real file
    paths.labels_dir.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(paths.labels_dir / "morphology_catalogue.parquet")

    stats = crossmatch_labels.run(config, download=False)
    assert stats["morphology"]["matched_by_id"] == 2
    assert stats["morphology"]["matched_by_position"] == 0
    assert stats["morphology"]["unmatched"] == 1
    assert stats["morphology"]["not_in_queried_tiles"] == 1  # 999999 genuinely isn't in this catalogue

    matched = pd.read_parquet(paths.labels_dir / "morphology_labelled.parquet")
    assert matched["object_id"].dtype == np.int64
    hit = matched[matched["object_id"].isin([1003, 1005])]
    assert (hit["match_type"] == "id").all() and (hit["match_sep_arcsec"] == 0.0).all()
    index = pd.read_parquet(paths.dataset_index)
    assert set(index.loc[index["label_morphology"], "object_id"]) == {1003, 1005}


def test_normalize_id_column_drops_unparseable_ids(recwarn):
    from projects.euclid_byol.pipelines.crossmatch_labels import normalize_id_column

    df = pd.DataFrame({"object_id": ["123", "-456", "not-a-number", None], "x": [1, 2, 3, 4]})
    out = normalize_id_column(df, "object_id")
    assert list(out["object_id"]) == [123, -456] and out["object_id"].dtype == np.int64
    assert list(out["x"]) == [1, 2]  # the rest of the row is dropped too, not just the id
